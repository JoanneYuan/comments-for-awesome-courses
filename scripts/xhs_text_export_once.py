from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path

import requests
from rapidocr_onnxruntime import RapidOCR

SOURCE_URL = "http://xhslink.cn/o/19VMletlEI5"
OUTPUT_DIR = Path("xhs-text-export")
IMAGE_DIR = OUTPUT_DIR / "_images"


def fetch_note() -> dict:
    endpoints = [
        "https://api.mu-jie.cc/xhs",
        "https://api.lalkk.com/xhs",
        "https://api.bugpk.com/api/xhs",
        "https://ffapi.cn/int/v1/xiaohongshu",
    ]
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    errors: list[str] = []
    for endpoint in endpoints:
        try:
            response = session.get(endpoint, params={"url": SOURCE_URL}, timeout=60)
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(data, dict):
                return data
            if isinstance(payload, dict) and any(key in payload for key in ("title", "desc", "images", "cover")):
                return payload
            errors.append(f"{endpoint}: {payload}")
        except Exception as exc:
            errors.append(f"{endpoint}: {type(exc).__name__}: {exc}")
    raise RuntimeError("\n".join(errors))


def image_urls(note: dict) -> list[str]:
    values = note.get("images") or []
    if isinstance(values, str):
        values = [values]
    urls = [value for value in values if isinstance(value, str) and value.startswith("http")]
    cover = note.get("cover")
    if not urls and isinstance(cover, str) and cover.startswith("http"):
        urls = [cover]
    return list(dict.fromkeys(urls))


def clean_text(value: object) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True)
    IMAGE_DIR.mkdir()

    note = fetch_note()
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": SOURCE_URL})
    paths: list[Path] = []
    for index, url in enumerate(image_urls(note), start=1):
        try:
            response = session.get(url, timeout=90)
            response.raise_for_status()
            path = IMAGE_DIR / f"{index:02d}.jpg"
            path.write_bytes(response.content)
            paths.append(path)
        except Exception:
            continue

    engine = RapidOCR()
    image_sections: list[str] = []
    seen: set[str] = set()
    for index, path in enumerate(paths, start=1):
        result, _ = engine(str(path))
        lines: list[str] = []
        for item in result or []:
            text = clean_text(item[1] if len(item) > 1 else "")
            if text and text not in seen:
                seen.add(text)
                lines.append(text)
        if lines:
            image_sections.append(f"## 图片 {index} 文字\n\n" + "\n".join(lines))

    title = clean_text(note.get("title")) or "小红书笔记正文"
    caption = clean_text(note.get("desc") or note.get("content") or note.get("title"))
    author = clean_text(note.get("author"))

    parts = [f"# {title}"]
    if author:
        parts.extend(["", f"作者：{author}"])
    if caption:
        parts.extend(["", "## 笔记正文", "", caption])
    if image_sections:
        parts.extend([""] + image_sections)
    markdown = "\n".join(parts).strip() + "\n"

    md_path = OUTPUT_DIR / "正文.md"
    md_path.write_text(markdown, encoding="utf-8")
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps({"source": SOURCE_URL, "images": len(paths), "ocr_sections": len(image_sections)}, ensure_ascii=False),
        encoding="utf-8",
    )
    shutil.rmtree(IMAGE_DIR, ignore_errors=True)

    with zipfile.ZipFile(OUTPUT_DIR / "小红书正文.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(md_path, arcname="正文.md")


if __name__ == "__main__":
    main()
