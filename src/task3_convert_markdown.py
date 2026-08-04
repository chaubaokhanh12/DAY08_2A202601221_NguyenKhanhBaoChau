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
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown.

    Dùng pdfplumber (đã có trong môi trường) thay vì MarkItDown vì trích text
    sạch hơn cho các PDF bảng biểu (học phí, danh sách nhà cung cấp chỗ ở).
    """
    import pdfplumber

    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in (".pdf", ".docx", ".doc"):
            continue
        print(f"Converting: {filepath.name}")
        pages_text = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                pages_text.append(page.extract_text() or "")
        body = "\n\n".join(pages_text)
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(body, encoding="utf-8")
        print(f"  ✓ Saved: {output_path} ({len(body)} chars)")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue
        print(f"Converting: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        output_path = output_dir / f"{filepath.stem}.md"

        # Metadata header để retriever + citation có thông tin nguồn
        header = f"# {data.get('title', 'Unknown')}\n\n"
        header += f"**Source:** {data.get('url', 'N/A')}\n"
        header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

        content = header + data.get("content_markdown", "")
        output_path.write_text(content, encoding="utf-8")
        print(f"  ✓ Saved: {output_path}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
