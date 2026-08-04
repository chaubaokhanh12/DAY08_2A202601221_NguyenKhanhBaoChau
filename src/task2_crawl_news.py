"""
Task 2 — Crawl bài viết/thông báo về dịch vụ đại học.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trang công khai của một trường đại học.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt (tuỳ chọn — script có fallback stdlib nên không bắt buộc):
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: thông báo tuyển sinh, sự kiện, dịch vụ thư viện, hỗ trợ sinh viên, học bổng.

Nguồn thực tế dùng trong script này (RMIT Vietnam — trang công khai, tiếng Anh):
    - Học bổng 2026 (record scholarships)
    - Học bổng 2025 (47.5 tỷ VND)
    - Dịch vụ thư viện (Newbie 101: Unlock Library Power)
    - Erasmus+ scholarship — exchange experience
    - VN-Intern Buddy — hỗ trợ thực tập
    - SEUP — hỗ trợ học tiếng Anh
    - Seminar hợp tác thư viện
"""

import asyncio
import json
import re
import urllib.request
import urllib.error
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

ARTICLE_URLS = [
    "https://www.rmit.edu.vn/news/all-news/2026/jan/"
    "rmit-vietnam-announces-record-2026-scholarships-worth-more-than-200-billion-vnd",
    "https://www.rmit.edu.vn/news/all-news/2025/oct/"
    "rmit-vietnam-awards-47-5-billion-vnd-in-2025-scholarships",
    "https://www.rmit.edu.vn/students/student-news-and-events/student-news/2026/"
    "newbie-101-unlock-library-power",
    "https://www.rmit.edu.vn/students/student-news-and-events/student-news/2026/"
    "first-year-erasmus-scholarship-barcelona-exchange-experience",
    "https://www.rmit.edu.vn/students/student-news-and-events/student-news/2025/"
    "vn-intern-buddy-internship-support",
    "https://www.rmit.edu.vn/students/student-news-and-events/student-news/2025/"
    "seup-students-supporting-english-learning-beyond-the-classroom",
    "https://www.rmit.edu.vn/news/all-news/2025/sep/"
    "rmit-vietnam-hosts-pioneering-seminar-to-strengthen-library-collaboration",
]


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Fallback crawler dùng stdlib (urllib + html.parser) — không cần browser.
# Dùng khi crawl4ai/playwright chưa cài. Trích title + nội dung text, convert
# cơ bản sang markdown (headings, paragraphs, links).
# ---------------------------------------------------------------------------

class _ArticleExtractor(HTMLParser):
    """HTMLParser đơn giản: trích <title>, thẻ metaOG, và nội dung text chính."""

    SKIP_TAGS = {"script", "style", "noscript", "svg", "header", "footer",
                 "nav", "aside", "form", "button"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self._in_h = 0
        self._in_p = 0
        self._in_link = False
        self._link_href = ""
        self.title = ""
        self.meta_description = ""
        self.blocks = []
        self._buf = []

    def _flush(self):
        if self._skip_depth > 0 or (self._in_h == 0 and self._in_p == 0):
            text = " ".join("".join(self._buf).split())
            if text:
                self.blocks.append(text)
            self._buf = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        a = dict(attrs)
        if tag == "title":
            self._in_title = True
        elif tag == "meta" and a.get("name", "").lower() == "description" \
                and not self.meta_description:
            self.meta_description = a.get("content", "")
        elif tag == "meta" and a.get("property", "").lower() == "og:description":
            self.meta_description = a.get("content", self.meta_description)
        elif re.match(r"^h[1-6]$", tag):
            if self._in_h == 0:
                self._flush()
            self._in_h = max(self._in_h, 1)
        elif tag == "p":
            if self._in_p == 0:
                self._flush()
            self._in_p = max(self._in_p, 1)
        elif tag == "a" and (self._in_h > 0 or self._in_p > 0):
            self._in_link = True
            self._link_href = a.get("href", "")

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(self._skip_depth - 1, 0)
            return
        if self._skip_depth > 0:
            return
        if tag == "title":
            self._in_title = False
        elif re.match(r"^h[1-6]$", tag):
            text = " ".join("".join(self._buf).split())
            if text:
                level = int(tag[1])
                self.blocks.append("#" * level + " " + text)
            self._buf = []
            self._in_h = 0
        elif tag == "p":
            text = " ".join("".join(self._buf).split())
            if text:
                self.blocks.append(text)
            self._buf = []
            self._in_p = 0
        elif tag == "a":
            self._in_link = False
            self._link_href = ""

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        if self._in_title and not self.title:
            t = data.strip()
            if t:
                self.title = t
        elif self._in_h > 0 or self._in_p > 0:
            text = data
            if self._in_link and self._link_href:
                text = f"[{text}]({self._link_href})"
            self._buf.append(text)


def _fetch_html(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    # RMIT trả về UTF-8; thử decode an toàn.
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def _html_to_markdown(html: str) -> tuple[str, str, str]:
    """Trả về (title, description, markdown_content)."""
    parser = _ArticleExtractor()
    parser.feed(html)
    title = parser.title.strip()
    desc = parser.meta_description.strip()

    # Lọc block rỗng/quá ngắt, giữ các block có ý nghĩa (≥ 3 chars).
    md_blocks = [b for b in parser.blocks if len(b) >= 3]
    # Giữ tối đa 120 block đầu (đủ nội dung, tránh nav/cookie banner dư).
    if len(md_blocks) > 120:
        md_blocks = md_blocks[:120]
    content = "\n\n".join(md_blocks)
    return title, desc, content


async def crawl_with_crawl4ai(url: str) -> dict | None:
    """Crawl bằng Crawl4AI nếu đã cài. Trả về None nếu không cài/lỗi."""
    try:
        from crawl4ai import AsyncWebCrawler
    except Exception as e:
        print(f"  • crawl4ai không khả dụng ({type(e).__name__}), fallback stdlib")
        return None

    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            md = getattr(result, "markdown", "") or ""
            meta = getattr(result, "metadata", {}) or {}
            return {
                "url": url,
                "title": meta.get("title", "Unknown"),
                "description": meta.get("description", ""),
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": md,
                "crawler": "crawl4ai",
            }
    except Exception as e:
        print(f"  • crawl4ai lỗi ({type(e).__name__}: {e}), fallback stdlib")
        return None


def crawl_with_stdlib(url: str) -> dict:
    """Crawl bằng urllib + html.parser (stdlib), không cần browser."""
    html = _fetch_html(url)
    title, desc, md = _html_to_markdown(html)
    if not title:
        title = url.rstrip("/").split("/")[-1].replace("-", " ").title()
    return {
        "url": url,
        "title": title,
        "description": desc,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": md,
        "crawler": "urllib+htmlparser",
    }


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "description": str,
            "date_crawled": str (ISO format),
            "content_markdown": str,
            "crawler": str
        }
    """
    result = await crawl_with_crawl4ai(url)
    if result is None:
        result = crawl_with_stdlib(url)
    return result


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS. Trả về danh sách filepath."""
    setup_directory()
    saved = []

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
        except urllib.error.HTTPError as e:
            print(f"  ✗ HTTP {e.code}: {e.reason}")
            continue
        except Exception as e:
            print(f"  ✗ Lỗi: {type(e).__name__}: {e}")
            continue

        if not article.get("content_markdown") or len(article["content_markdown"]) < 200:
            print(f"  ⚠ Nội dung quá ngắn ({len(article.get('content_markdown', ''))} chars)")
            continue

        # Tên file: article_NN.json
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        saved.append(filepath)
        print(f"  ✓ Saved: {filepath.name}  ({filepath.stat().st_size} bytes)")

    return saved


async def _main_async():
    print("=" * 70)
    print("Task 2 — Crawl bài viết/thông báo dịch vụ đại học")
    print(f"Nguồn: RMIT Vietnam (rmit.edu.vn) — {len(ARTICLE_URLS)} URL công khai")
    print("=" * 70)

    saved = await crawl_all()

    print("\n" + "=" * 70)
    print(f"Tổng kết: crawl thành công {len(saved)}/{len(ARTICLE_URLS)} bài")
    print("=" * 70)
    for f in saved:
        print(f"  • {f.name}  ({f.stat().st_size} bytes)")

    if len(saved) < 5:
        print("\n⚠ Cảnh báo: cần tối thiểu 5 file cho test Task 2.")
        return 1
    print("\n✓ Đã đủ ≥5 bài viết cho data/landing/news/ — Task 2 hoàn thành.")
    return 0


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang thông báo/sự kiện trên trang chính thức của trường đại học")
    else:
        asyncio.run(_main_async())