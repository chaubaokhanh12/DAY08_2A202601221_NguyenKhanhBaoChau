"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.
    → Module này implement CẢ HAI: `lexical_search()` (BM25) và `tfidf_search()`
      (TF-IDF cosine) để lấy điểm bonus, xem phần so sánh ở cuối file.

Cài đặt:
    pip install rank-bm25 scikit-learn

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
    - ⚠ IDF của rank_bm25 bằng 0 khi từ xuất hiện ở đúng nửa số chunk → module này
      thay bằng IDF kiểu Lucene (luôn dương), xem chi tiết trong build_bm25_index().

BM25 khác TF-IDF ở đâu (dùng để trả lời câu hỏi bonus lúc demo):
    - TF-IDF: tf tăng TUYẾN TÍNH → từ lặp 100 lần được điểm gấp 10 lần từ lặp 10 lần.
    - BM25: tf bị bão hoà (term saturation) bởi k1 → điểm tiệm cận (k1+1)=2.5,
      lặp thêm gần như không tăng điểm nữa. Thực tế hợp lý hơn: một tài liệu nhắc
      "học phí" 50 lần không liên quan gấp 10 lần tài liệu nhắc 5 lần.
    - BM25 chuẩn hoá độ dài bằng b và avgdl; TF-IDF chỉ chuẩn hoá bằng L2 norm của vector.
    - TF-IDF so khớp bằng cosine trên không gian vector → điểm nằm trong [0,1];
      BM25 trả điểm không chặn trên → chỉ dùng để XẾP HẠNG, đừng so trực tiếp với
      threshold tuyệt đối (đây cũng là lý do Task 9 phải dùng điểm cosine của Task 5).

CORPUS lấy từ đâu:
    Ưu tiên đọc thẳng chunks đã index trong chroma_db/ (Task 4) để nội dung chunk của
    nhánh sparse TRÙNG KHỚP từng ký tự với nhánh dense — RRF ở Task 7/9 gộp theo khoá
    `content`, nếu hai nhánh chunk khác nhau thì không bao giờ có chunk nào được cả hai
    ranker cùng bình chọn, RRF mất hết tác dụng.
    Nếu chưa có chroma_db/ thì fallback đọc data/standardized/ và tự chunk lại bằng đúng
    tham số CHUNK_SIZE/CHUNK_OVERLAP của Task 4.
"""

import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

# Đồng bộ cấu hình với Task 4 (chỉ IMPORT hằng số, không sửa file của role khác).
# Bọc try/except để module vẫn chạy được khi gọi trực tiếp `python src/task6_...py`
# (lúc đó không có package context nên relative import sẽ lỗi).
try:
    from .task4_chunking_indexing import (
        CHROMA_DIR,
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        COLLECTION_NAME,
        STANDARDIZED_DIR,
    )
except ImportError:  # pragma: no cover - chỉ xảy ra khi chạy file như script rời
    _ROOT = Path(__file__).parent.parent
    STANDARDIZED_DIR = _ROOT / "data" / "standardized"
    CHROMA_DIR = _ROOT / "chroma_db"
    COLLECTION_NAME = "university_services_docs"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50


# Corpus + index được nạp LAZY (lần search đầu tiên) để việc `import` module này
# không tốn thời gian và không nổ lỗi khi nhóm chưa chạy xong Task 1-4.
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
_BM25 = None  # BM25Okapi index
_TFIDF = None  # (TfidfVectorizer, tfidf_matrix)


# =============================================================================
# Tokenization
# =============================================================================

# \w trong Python 3 đã match Unicode → giữ được chữ tiếng Việt có dấu (ữ, ệ, ơ...),
# đồng thời loại sạch ký tự markdown (#, *, |, -) và dấu câu.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """
    Tách token cho BM25.

    Dùng regex thay vì .split() vì .split() giữ nguyên dấu câu và ký tự markdown:
    "fee." và "fee" sẽ thành 2 token khác nhau → IDF bị loãng, recall giảm.

    Lưu ý tiếng Việt: đây là tách theo ÂM TIẾT, không phải theo TỪ ghép
    ("học phí" → ["học", "phí"]). Muốn chính xác hơn có thể dùng
    `underthesea.word_tokenize()`, nhưng âm tiết đã đủ tốt cho corpus lab này
    và không cần thêm dependency nặng.
    """
    return _TOKEN_RE.findall(text.lower())


# =============================================================================
# Corpus loading
# =============================================================================


def _load_corpus_from_chroma() -> list[dict]:
    """Đọc chunks đã index từ ChromaDB (Task 4). Trả [] nếu chưa có."""
    if not CHROMA_DIR.exists():
        return []
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(COLLECTION_NAME)
        data = collection.get(include=["documents", "metadatas"])
    except Exception as e:
        print(f"  ⚠ Không đọc được chroma_db/ ({e}) → fallback sang data/standardized/")
        return []

    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or [{}] * len(documents)
    return [
        {"content": doc, "metadata": dict(meta or {})}
        for doc, meta in zip(documents, metadatas)
        if doc and doc.strip()
    ]


def _split_text(text: str) -> list[str]:
    """Chunk text bằng đúng tham số của Task 4 để hai nhánh retrieval đồng bộ."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text)
    except ImportError:
        # Fallback tối giản: gom đoạn văn cho tới khi đầy chunk_size, giữ overlap.
        chunks, buf = [], ""
        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if len(buf) + len(para) + 2 <= CHUNK_SIZE:
                buf = f"{buf}\n\n{para}" if buf else para
            else:
                if buf:
                    chunks.append(buf)
                buf = (buf[-CHUNK_OVERLAP:] + "\n\n" + para) if buf else para
                while len(buf) > CHUNK_SIZE:
                    chunks.append(buf[:CHUNK_SIZE])
                    buf = buf[CHUNK_SIZE - CHUNK_OVERLAP :]
        if buf:
            chunks.append(buf)
        return chunks


def _load_corpus_from_markdown() -> list[dict]:
    """Đọc + chunk markdown từ data/standardized/ (fallback khi chưa có chroma_db/)."""
    if not STANDARDIZED_DIR.exists():
        return []

    corpus: list[dict] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        doc_type = "legal" if "legal" in str(md_file) else "news"
        for i, chunk_text in enumerate(_split_text(text)):
            if chunk_text.strip():
                corpus.append(
                    {
                        "content": chunk_text,
                        "metadata": {
                            "source": md_file.name,
                            "type": doc_type,
                            "chunk_index": i,
                        },
                    }
                )
    return corpus


def load_corpus(force_reload: bool = False) -> list[dict]:
    """
    Nạp CORPUS (ưu tiên chroma_db/, fallback data/standardized/).

    Args:
        force_reload: Nạp lại kể cả khi CORPUS đã có sẵn (dùng sau khi reindex).

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    global CORPUS, _BM25, _TFIDF

    if CORPUS and not force_reload:
        return CORPUS

    corpus = _load_corpus_from_chroma() or _load_corpus_from_markdown()
    if not corpus:
        print(
            "  ⚠ Corpus rỗng — chạy Task 1-3 (data/standardized/) "
            "hoặc Task 4 (chroma_db/) trước khi dùng lexical search."
        )

    CORPUS = corpus
    _BM25 = None  # invalidate index cũ
    _TFIDF = None
    return CORPUS


# =============================================================================
# BM25
# =============================================================================


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}

    Returns:
        BM25Okapi index, hoặc None nếu corpus rỗng.
    """
    if not corpus:
        return None

    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    # k1=1.5: mức bão hoà TF mặc định; b=0.75: chuẩn hoá độ dài document.
    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

    # --- VÁ IDF (bắt buộc với corpus nhỏ như bài lab này) ------------------
    # rank_bm25 dùng IDF Robertson gốc:  idf = ln((N - df + 0.5) / (df + 0.5))
    #   → df = N/2  ⇒ idf = 0        (từ đó ĐÓNG GÓP ĐÚNG 0 ĐIỂM)
    #   → df > N/2  ⇒ idf < 0        (rank_bm25 mới vá case này bằng epsilon)
    # Corpus của nhóm chỉ ~vài trăm chunk từ 8 tài liệu cùng một chủ đề, nên đúng
    # những từ khoá cần tìm ("student", "fee", "library", "RMIT") lại là từ xuất hiện
    # ở hơn nửa số chunk → idf = 0 → lexical_search() trả về RỖNG dù nội dung khớp.
    # Thay bằng IDF kiểu Lucene/Elasticsearch: ln(1 + (N - df + 0.5)/(df + 0.5)),
    # luôn > 0, thứ tự xếp hạng tương đối giữ nguyên.
    N = len(tokenized_corpus)
    doc_freq = Counter()
    for tokens in tokenized_corpus:
        doc_freq.update(set(tokens))
    for word, df in doc_freq.items():
        bm25.idf[word] = math.log(1 + (N - df + 0.5) / (df + 0.5))
    bm25.average_idf = sum(bm25.idf.values()) / len(bm25.idf) if bm25.idf else 0.0

    return bm25


def _get_bm25():
    """Lấy BM25 index, build lazy nếu chưa có."""
    global _BM25
    if _BM25 is None:
        _BM25 = build_bm25_index(load_corpus())
    return _BM25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score (không chặn trên, chỉ để xếp hạng)
            'metadata': dict
        }
        Sorted by score descending. Trả [] nếu corpus rỗng hoặc không khớp token nào.
    """
    corpus = load_corpus()
    bm25 = _get_bm25()
    if not corpus or bm25 is None:
        return []

    tokenized_query = tokenize(query)
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][: max(top_k, 0)]

    results = []
    for idx in top_indices:
        # Score = 0 nghĩa là không có token nào của query xuất hiện trong chunk
        # → loại luôn, giữ lại chỉ làm nhiễu cho RRF ở Task 7.
        if scores[idx] > 0:
            results.append(
                {
                    "content": corpus[idx]["content"],
                    "score": float(scores[idx]),
                    "metadata": corpus[idx]["metadata"],
                }
            )
    return results


# =============================================================================
# TF-IDF (bonus: phương pháp lexical khác BM25)
# =============================================================================


def build_tfidf_index(corpus: list[dict]):
    """
    Xây dựng TF-IDF index (sklearn) từ corpus.

    Returns:
        (vectorizer, matrix) hoặc None nếu corpus rỗng.
    """
    if not corpus:
        return None

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        lowercase=False,  # tokenize() đã lower rồi
        token_pattern=None,  # bắt buộc khi truyền tokenizer riêng
        sublinear_tf=True,  # dùng 1+log(tf) để giảm bớt ảnh hưởng của từ lặp nhiều
    )
    matrix = vectorizer.fit_transform([doc["content"] for doc in corpus])
    return vectorizer, matrix


def tfidf_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khoá bằng TF-IDF + cosine similarity.

    Khác `lexical_search()` (BM25) ở chỗ score nằm trong [0,1] nên dễ đặt ngưỡng,
    nhưng không có term saturation và chuẩn hoá độ dài kiểu BM25.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}, sorted desc.
    """
    global _TFIDF

    corpus = load_corpus()
    if not corpus:
        return []

    if _TFIDF is None:
        _TFIDF = build_tfidf_index(corpus)
    if _TFIDF is None:
        return []

    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer, matrix = _TFIDF
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix)[0]
    top_indices = np.argsort(scores)[::-1][: max(top_k, 0)]

    return [
        {
            "content": corpus[idx]["content"],
            "score": float(scores[idx]),
            "metadata": corpus[idx]["metadata"],
        }
        for idx in top_indices
        if scores[idx] > 0
    ]


if __name__ == "__main__":
    # Test + so sánh BM25 vs TF-IDF trên cùng 1 query (dùng cho phần demo bonus)
    query = "tuition fee payment methods"
    print(f"Corpus: {len(load_corpus())} chunks\n")

    print(f"[BM25]   {query}")
    for r in lexical_search(query, top_k=5):
        print(f"  [{r['score']:.3f}] ({r['metadata'].get('source')}) {r['content'][:90]}...")

    print(f"\n[TF-IDF] {query}")
    for r in tfidf_search(query, top_k=5):
        print(f"  [{r['score']:.3f}] ({r['metadata'].get('source')}) {r['content'][:90]}...")
