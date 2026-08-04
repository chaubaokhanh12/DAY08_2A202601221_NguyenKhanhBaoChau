"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf]"
    # Lưu ý: cần extra [pdf] để convert được file PDF. Chỉ "pip install markitdown"
    # (không có extra) sẽ báo MissingDependencyException khi convert PDF, dù JSON/DOCX
    # vẫn convert bình thường.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục

Cấu hình:
    - PDF/DOCX → dùng MarkItDown (extra [pdf] dùng pdfminer.six behind-the-scenes).
    - JSON (news) → extract content_markdown + metadata, viết header frontmatter
      bằng YAML-ish để downstream (chunking/indexing) dễ trích source.
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

VALID_LEGAL_EXTS = {".pdf", ".docx", ".doc"}
_MIN_CONTENT_CHARS = 200


def _slug(text: str) -> str:
    """Loại bỏ ký tự không hợp lệ cho tên file markdown."""
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in text).strip("-") or "untitled"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        print("  ⚠ Không tìm thấy data/landing/legal/")
        return []

    md = MarkItDown()
    saved = []

    for filepath in sorted(legal_dir.iterdir()):
        if not filepath.is_file() or filepath.suffix.lower() not in VALID_LEGAL_EXTS:
            continue
        if filepath.name.startswith("."):
            continue

        print(f"  Converting: {filepath.name}")
        try:
            result = md.convert(str(filepath))
            content = (result.text_content or "").strip()
        except Exception as e:
            print(f"    ✗ Lỗi convert: {type(e).__name__}: {e}")
            continue

        if len(content) < _MIN_CONTENT_CHARS:
            print(f"    ⚠ Nội dung quá ngắn ({len(content)} chars), có thể PDF scan/ảnh")
            # Vẫn lưu để downstream xử lý, nhưng cảnh báo.

        output_path = output_dir / f"{filepath.stem}.md"
        content = _utf8(content)
        output_path.write_text(content, encoding="utf-8")
        saved.append(output_path)
        print(f"    ✓ Saved: {output_path.name}  ({len(content)} chars)")
    return saved


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print("  ⚠ Không tìm thấy data/landing/news/")
        return []

    saved = []
    for filepath in sorted(news_dir.iterdir()):
        if not filepath.is_file() or filepath.suffix.lower() != ".json":
            continue
        if filepath.name.startswith("."):
            continue

        print(f"  Converting: {filepath.name}")
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"    ✗ Lỗi đọc JSON: {e}")
            continue

        title = str(data.get("title") or "Untitled").strip()
        url = str(data.get("url") or "").strip()
        date_crawled = str(data.get("date_crawled") or "").strip()
        description = str(data.get("description") or "").strip()
        body = str(data.get("content_markdown") or "").strip()

        # Header metadata (đơn giản, human-readable + dễ parse cho chunking).
        header = f"# {title}\n\n"
        if url:
            header += f"**Source:** {url}\n"
        if date_crawled:
            header += f"**Crawled:** {date_crawled}\n"
        if description:
            header += f"**Description:** {description}\n"
        header += "\n---\n\n"

        content = _utf8(header + body)
        if len(content) < _MIN_CONTENT_CHARS:
            print(f"    ⚠ Nội dung quá ngắn ({len(content)} chars)")

        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(content, encoding="utf-8")
        saved.append(output_path)
        print(f"    ✓ Saved: {output_path.name}  ({len(content)} chars)")
    return saved


def _utf8(text: str) -> str:
    """Đảm bảo output là chuỗi UTF-8 sạch (loại BOM, chuẩn down newline)."""
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    return text


def convert_all():
    """Convert toàn bộ files. Trả về (legal_count, news_count)."""
    print("=" * 60)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 60)

    print("\n--- Legal Documents ---")
    legal_saved = convert_legal_docs()

    print("\n--- News Articles ---")
    news_saved = convert_news_articles()

    print("\n" + "=" * 60)
    print(f"Tổng kết: {len(legal_saved)} legal + {len(news_saved)} news "
          f"= {len(legal_saved) + len(news_saved)} file markdown")
    print("Output tại:", OUTPUT_DIR)
    print("=" * 60)
    return len(legal_saved), len(news_saved)


if __name__ == "__main__":
    legal_n, news_n = convert_all()
    if legal_n + news_n == 0:
        print("\n⚠ Không convert được file nào — kiểm tra data/landing/.")
        raise SystemExit(1)
    print("\n✓ Task 3 hoàn thành.")
    raise SystemExit(0)