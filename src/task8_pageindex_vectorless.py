"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex fpdf2

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key → điền PAGEINDEX_API_KEY vào .env
    3. Upload documents:  python -m src.task8_pageindex_vectorless
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
    → Vì schema còn đang đổi, hàm parse bên dưới cố tình viết "phòng thủ": thử nhiều tên
      field, và có thể bật DEBUG_RESPONSE=1 để in nguyên response thật ra kiểm tra.

Vì sao cần vectorless làm fallback (dùng để trả lời khi demo):
    Dense + BM25 đều cắt tài liệu thành chunk rời rạc rồi so khớp từng chunk, nên câu hỏi
    cần đọc XUYÊN nhiều mục ("quy trình xin học bổng gồm mấy bước?") hay tài liệu dài có
    cấu trúc mục lục rõ sẽ bị vỡ ngữ cảnh. PageIndex giữ nguyên cây cấu trúc document và
    để LLM "duyệt" mục lục như con người tra sách → hợp với đúng loại câu hỏi mà hybrid
    search trả về điểm thấp.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

_ROOT = Path(__file__).parent.parent
PDF_DIR = _ROOT / "pageindex_pdfs"          # PDF tạm sinh từ markdown (đã .gitignore)
DOC_IDS_FILE = _ROOT / "pageindex_doc_ids.json"  # cache doc_id (đã .gitignore)

# Mỗi query phải gọi API + poll riêng cho từng document → giới hạn số document
# để fallback không kéo dài hàng chục giây giữa buổi demo.
MAX_DOCS_PER_QUERY = 3
POLL_INTERVAL_SEC = 2
POLL_TIMEOUT_SEC = 60
INDEX_TIMEOUT_SEC = 300  # PageIndex dựng cây cấu trúc + OCR, tài liệu dài mất vài phút
DEBUG_RESPONSE = os.getenv("PAGEINDEX_DEBUG", "") == "1"


# =============================================================================
# Helpers
# =============================================================================


def _get_client():
    """Khởi tạo PageIndexClient. Raise RuntimeError nếu thiếu key/SDK."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError(
            "Chưa có PAGEINDEX_API_KEY trong .env — đăng ký tại https://pageindex.ai/"
        )
    try:
        from pageindex.client import PageIndexClient
    except ImportError:  # SDK đổi vị trí export giữa các bản
        from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _pick_unicode_font() -> Path | None:
    """Tìm 1 font TTF có dấu tiếng Việt (fpdf2 core font chỉ hỗ trợ latin-1)."""
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    return next((p for p in candidates if p.exists()), None)


def _wrap_long_tokens(line: str, max_len: int = 80) -> str:
    """
    Chèn khoảng trắng vào các token quá dài (URL, bảng markdown dính liền).

    Một token không có chỗ ngắt và dài hơn bề ngang trang cũng làm fpdf2 ném
    "Not enough horizontal space to render a single character".
    """
    parts = []
    for token in line.split(" "):
        while len(token) > max_len:
            parts.append(token[:max_len])
            token = token[max_len:]
        parts.append(token)
    return " ".join(parts)


def markdown_to_pdf(md_file: Path, pdf_path: Path) -> Path:
    """
    Convert 1 file markdown sang PDF đơn giản.

    PageIndex nhận PDF chứ không nhận .md trực tiếp, nên phải qua bước này.
    Chỉ cần giữ đúng text + thứ tự heading, không cần đẹp.
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    font_path = _pick_unicode_font()
    if font_path:
        pdf.add_font("uni", "", str(font_path))
        pdf.set_font("uni", size=11)
        encode = lambda s: s  # noqa: E731 - font unicode, giữ nguyên tiếng Việt
    else:
        pdf.set_font("Helvetica", size=11)
        # Không có font unicode → ép về latin-1, mất dấu nhưng không crash.
        encode = lambda s: s.encode("latin-1", "replace").decode("latin-1")  # noqa: E731

    text = md_file.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        # new_x/new_y bắt buộc: mặc định fpdf2 để con trỏ ở LỀ PHẢI sau multi_cell,
        # nên dòng kế tiếp không còn bề ngang → FPDFException "Not enough horizontal space".
        pdf.multi_cell(
            0, 6, _wrap_long_tokens(encode(line)) or " ", new_x="LMARGIN", new_y="NEXT"
        )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(pdf_path))
    return pdf_path


def _load_doc_ids() -> dict[str, str]:
    """Đọc cache {tên file: doc_id} đã upload."""
    if DOC_IDS_FILE.exists():
        try:
            return json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_doc_ids(doc_ids: dict[str, str]) -> None:
    DOC_IDS_FILE.write_text(
        json.dumps(doc_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# =============================================================================
# Upload
# =============================================================================


def wait_until_ready(client, doc_ids: list[str], timeout: int = INDEX_TIMEOUT_SEC) -> None:
    """
    Chờ PageIndex xử lý xong document (dựng cây cấu trúc + OCR).

    `submit_document()` trả doc_id NGAY nhưng document chưa query được. Nếu gọi
    `submit_query()` luôn sẽ lỗi — nên phải poll `is_retrieval_ready()` trước.
    """
    pending = list(doc_ids)
    deadline = time.time() + timeout

    while pending and time.time() < deadline:
        still_pending = []
        for doc_id in pending:
            if client.is_retrieval_ready(doc_id):
                print(f"  ✓ Sẵn sàng retrieval: {doc_id}")
            else:
                still_pending.append(doc_id)
        pending = still_pending
        if pending:
            time.sleep(POLL_INTERVAL_SEC * 2)

    if pending:
        print(f"  ⚠ Còn {len(pending)} document chưa index xong sau {timeout}s: {pending}")


def upload_documents(force: bool = False, wait: bool = True) -> dict[str, str]:
    """
    Upload toàn bộ markdown documents lên PageIndex.

    Args:
        force: Upload lại kể cả file đã có doc_id trong cache.
        wait: Chờ document index xong trước khi trả về (nên bật, xem wait_until_ready).

    Returns:
        Dict {tên file markdown: doc_id}, đồng thời cache vào pageindex_doc_ids.json
        để lần chạy sau (và các thành viên khác trong nhóm) không phải upload lại.
    """
    client = _get_client()
    doc_ids = {} if force else _load_doc_ids()

    md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    if not md_files:
        print(f"  ⚠ Không có file .md nào trong {STANDARDIZED_DIR} — chạy Task 3 trước.")
        return doc_ids

    new_doc_ids = []
    for md_file in md_files:
        if md_file.name in doc_ids and not force:
            print(f"  · Bỏ qua (đã upload): {md_file.name}")
            continue

        pdf_path = PDF_DIR / f"{md_file.stem}.pdf"
        markdown_to_pdf(md_file, pdf_path)

        resp = client.submit_document(str(pdf_path))
        doc_id = resp.get("doc_id") or resp.get("id")
        if not doc_id:
            print(f"  ✗ Không lấy được doc_id cho {md_file.name}: {resp}")
            continue

        doc_ids[md_file.name] = doc_id
        new_doc_ids.append(doc_id)
        print(f"  ✓ Uploaded: {md_file.name} -> {doc_id}")

    _save_doc_ids(doc_ids)

    if wait and new_doc_ids:
        print(f"\nChờ PageIndex index {len(new_doc_ids)} document...")
        wait_until_ready(client, new_doc_ids)

    return doc_ids


# =============================================================================
# Retrieval
# =============================================================================


def _poll_retrieval(client, retrieval_id: str) -> dict:
    """Poll cho tới khi retrieval hoàn tất (hoặc hết POLL_TIMEOUT_SEC)."""
    deadline = time.time() + POLL_TIMEOUT_SEC
    retrieval: dict = {}

    while time.time() < deadline:
        retrieval = client.get_retrieval(retrieval_id) or {}
        status = str(retrieval.get("status", "")).lower()
        if status in ("completed", "success", "succeeded", "done"):
            return retrieval
        if status in ("failed", "error"):
            raise RuntimeError(f"PageIndex retrieval failed: {retrieval}")
        time.sleep(POLL_INTERVAL_SEC)

    raise TimeoutError(f"PageIndex retrieval quá {POLL_TIMEOUT_SEC}s chưa xong")


def _parse_nodes(retrieval: dict, doc_name: str) -> list[dict]:
    """
    Bóc content từ response.

    Schema đang dùng: retrieved_nodes[].relevant_contents = list[list[{section_title,
    relevant_content}]]. Viết phòng thủ vì API còn đang đổi (xem docstring đầu file).
    """
    if DEBUG_RESPONSE:
        print(json.dumps(retrieval, ensure_ascii=False, indent=2)[:3000])

    items: list[dict] = []
    for node in retrieval.get("retrieved_nodes", []) or []:
        node_title = node.get("title") or node.get("section_title")

        groups = node.get("relevant_contents") or []
        # Có bản trả thẳng list[dict] thay vì list[list[dict]] → chuẩn hoá về 2 tầng.
        if groups and isinstance(groups[0], dict):
            groups = [groups]

        for group in groups:
            for item in group or []:
                content = (item.get("relevant_content") or item.get("content") or "").strip()
                if content:
                    items.append(
                        {
                            "content": content,
                            "section": item.get("section_title") or node_title,
                            "node_score": node.get("relevance_score") or node.get("score"),
                        }
                    )

        # Fallback: node không có relevant_contents nhưng có text thô.
        if not groups:
            raw = (node.get("text") or node.get("content") or "").strip()
            if raw:
                items.append({"content": raw, "section": node_title, "node_score": None})

    return [{**it, "doc": doc_name} for it in items]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
        Trả [] (không raise) nếu chưa cấu hình key / chưa upload document, để Task 9
        chỉ việc rơi về kết quả hybrid thay vì chết cả pipeline.

    Về 'score': PageIndex KHÔNG trả điểm similarity. Điểm ở đây được gán theo THỨ HẠNG
    (1/(1+rank)) chỉ để giữ đúng format và sắp xếp — tuyệt đối đừng đem so với
    SCORE_THRESHOLD cosine của Task 9.
    """
    if not PAGEINDEX_API_KEY:
        print("  ⚠ Chưa có PAGEINDEX_API_KEY → bỏ qua PageIndex fallback")
        return []

    doc_ids = _load_doc_ids()
    if not doc_ids:
        print(
            "  ⚠ Chưa có document nào trên PageIndex → chạy "
            "`python -m src.task8_pageindex_vectorless` để upload trước"
        )
        return []

    try:
        client = _get_client()
    except Exception as e:
        print(f"  ⚠ Không khởi tạo được PageIndex client ({e})")
        return []

    collected: list[dict] = []
    for doc_name, doc_id in list(doc_ids.items())[:MAX_DOCS_PER_QUERY]:
        try:
            resp = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = resp.get("retrieval_id") or resp.get("id")
            if not retrieval_id:
                continue
            retrieval = _poll_retrieval(client, retrieval_id)
            collected.extend(_parse_nodes(retrieval, doc_name))
        except Exception as e:
            print(f"  ⚠ PageIndex lỗi trên {doc_name}: {e}")
            continue

    results = []
    for rank, item in enumerate(collected[:top_k]):
        results.append(
            {
                "content": item["content"],
                "score": float(item["node_score"])
                if item.get("node_score") is not None
                else 1.0 / (1 + rank),
                "metadata": {
                    "source": item.get("doc"),
                    "section": item.get("section"),
                    "type": "pageindex",
                },
                "source": "pageindex",
            }
        )
    return results


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("tuition fee payment methods", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
