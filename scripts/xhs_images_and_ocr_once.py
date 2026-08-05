from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup
from rapidocr_onnxruntime import RapidOCR

SHORT_URL = "http://xhslink.cn/o/19VMletlEI5"
NOTE_ID = "6a6eaf1e00000000250052b1"
OUTPUT_DIR = Path("xhs-image-export")
IMAGE_DIR = OUTPUT_DIR / "images"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def balanced_json(value: str, start: int) -> str | None:
    opening = value[start]
    if opening not in "[{":
        return None
    stack = ["}" if opening == "{" else "]"]
    in_string = False
    escaped = False
    for index in range(start + 1, len(value)):
        char = value[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in "}]":
            if not stack or char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return value[start : index + 1]
    return None


def embedded_states(html: str) -> list[Any]:
    soup = BeautifulSoup(html, "lxml")
    states: list[Any] = []
    seen: set[str] = set()
    for script in soup.find_all("script"):
        raw = (script.string or script.get_text() or "").strip()
        if not raw:
            continue
        candidates: list[str] = []
        if script.get("type") in {"application/json", "application/ld+json"} or raw[:1] in "[{":
            candidates.append(raw)
        for marker in ("__INITIAL_STATE__", "__NEXT_DATA__", "__APOLLO_STATE__"):
            offset = 0
            while True:
                marker_pos = raw.find(marker, offset)
                if marker_pos < 0:
                    break
                starts = [
                    pos
                    for pos in (raw.find("{", marker_pos + len(marker)), raw.find("[", marker_pos + len(marker)))
                    if pos >= 0
                ]
                if starts:
                    candidate = balanced_json(raw, min(starts))
                    if candidate:
                        candidates.append(candidate)
                offset = marker_pos + len(marker)
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                states.append(json.loads(re.sub(r"\bundefined\b", "null", candidate)))
            except json.JSONDecodeError:
                continue
    return states


def looks_like_note(value: Any) -> bool:
    return isinstance(value, dict) and any(
        key in value for key in ("title", "desc", "description", "imageList", "image_list", "images", "video")
    )


def find_note(value: Any, note_id: str) -> dict[str, Any] | None:
    for node in walk(value):
        if not isinstance(node, dict):
            continue
        keyed = node.get(note_id)
        if isinstance(keyed, dict):
            candidate = keyed.get("note") if isinstance(keyed.get("note"), dict) else keyed
            if looks_like_note(candidate):
                return candidate
        identifier = next(
            (str(node[key]) for key in ("noteId", "note_id", "id") if node.get(key) is not None),
            None,
        )
        if identifier == note_id:
            candidate = node.get("note") if isinstance(node.get("note"), dict) else node
            if looks_like_note(candidate):
                return candidate
    return None


def media_url(value: Any) -> str | None:
    if isinstance(value, str):
        return value if value.startswith(("http://", "https://")) else None
    if not isinstance(value, dict):
        return None
    for key in ("urlDefault", "url_default", "url", "urlPre", "url_pre", "original", "masterUrl"):
        raw = value.get(key)
        if isinstance(raw, str) and raw.startswith(("http://", "https://")):
            return raw
    for key in ("infoList", "info_list", "urlList", "url_list"):
        raw = value.get(key)
        if isinstance(raw, list):
            for item in raw:
                url = media_url(item)
                if url:
                    return url
    return None


def note_image_urls(note: dict[str, Any]) -> list[str]:
    images = note.get("imageList") or note.get("image_list") or note.get("images") or []
    if isinstance(images, dict):
        images = list(images.values())
    urls: list[str] = []
    if isinstance(images, list):
        for image in images:
            url = media_url(image)
            if url and url not in urls:
                urls.append(url)
    return urls


def fallback_image_urls(payload: Any) -> list[str]:
    urls: list[str] = []
    for node in walk(payload):
        if not isinstance(node, dict):
            continue
        for key in ("imageList", "image_list", "images", "pics", "pic_list"):
            raw = node.get(key)
            values = list(raw.values()) if isinstance(raw, dict) else raw
            if isinstance(values, list):
                for item in values:
                    url = media_url(item)
                    if url and url not in urls:
                        urls.append(url)
        for key in ("cover", "cover_url", "image"):
            url = media_url(node.get(key))
            if url and "avatar" not in url.lower() and url not in urls:
                urls.append(url)
    return urls


def clean_ocr_text(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[ \t]+", " ", text)
    return text


def main() -> None:
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    IMAGE_DIR.mkdir(parents=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
        }
    )
    diagnostics: list[dict[str, Any]] = []
    image_urls: list[str] = []
    final_url = SHORT_URL

    try:
        response = session.get(SHORT_URL, allow_redirects=True, timeout=60)
        final_url = response.url
        diagnostics.append(
            {
                "source": "xiaohongshu_page",
                "status": response.status_code,
                "final_url": final_url,
                "bytes": len(response.content),
            }
        )
        states = embedded_states(response.text)
        note = find_note(states, NOTE_ID)
        if note:
            image_urls.extend(note_image_urls(note))
    except Exception as exc:
        diagnostics.append({"source": "xiaohongshu_page", "error": f"{type(exc).__name__}: {exc}"})

    if not image_urls:
        endpoints = [
            "https://api.mu-jie.cc/xhs",
            "https://api.lalkk.com/xhs",
            "https://api.bugpk.com/api/xhs",
            "https://ffapi.cn/int/v1/xiaohongshu",
        ]
        for endpoint in endpoints:
            try:
                response = session.get(endpoint, params={"url": SHORT_URL}, timeout=60)
                response.raise_for_status()
                payload = response.json()
                diagnostics.append({"source": endpoint, "status": response.status_code})
                note = find_note(payload, NOTE_ID)
                if note:
                    image_urls.extend(note_image_urls(note))
                if not image_urls:
                    image_urls.extend(fallback_image_urls(payload))
                if image_urls:
                    break
            except Exception as exc:
                diagnostics.append({"source": endpoint, "error": f"{type(exc).__name__}: {exc}"})

    image_urls = list(dict.fromkeys(image_urls))
    downloaded: list[Path] = []
    records: list[dict[str, Any]] = []
    for index, url in enumerate(image_urls, start=1):
        try:
            response = session.get(url, headers={"Referer": final_url}, timeout=90)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            suffix = {
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/webp": ".webp",
                "image/gif": ".gif",
                "image/avif": ".avif",
            }.get(content_type, ".jpg")
            path = IMAGE_DIR / f"{index:02d}{suffix}"
            path.write_bytes(response.content)
            downloaded.append(path)
            records.append({"order": index, "url": url, "path": path.name, "bytes": len(response.content)})
        except Exception as exc:
            records.append({"order": index, "url": url, "error": f"{type(exc).__name__}: {exc}"})

    ocr = RapidOCR()
    sections: list[str] = []
    for index, path in enumerate(downloaded, start=1):
        result, _ = ocr(str(path))
        lines: list[str] = []
        for item in result or []:
            text = clean_ocr_text(item[1] if len(item) > 1 else "")
            if text:
                lines.append(text)
        sections.append(f"## 图片 {index}\n\n" + ("\n".join(lines) if lines else "（未识别到文字）"))

    markdown = "# 图片文字\n\n" + "\n\n".join(sections) + "\n"
    (OUTPUT_DIR / "图片文字.md").write_text(markdown, encoding="utf-8")
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(
            {
                "source": SHORT_URL,
                "final_url": final_url,
                "image_urls": len(image_urls),
                "downloaded_images": len(downloaded),
                "records": records,
                "diagnostics": diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with zipfile.ZipFile(OUTPUT_DIR / "小红书图片.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in downloaded:
            archive.write(path, arcname=path.name)


if __name__ == "__main__":
    main()
