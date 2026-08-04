"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import re
from pathlib import Path

from rank_bm25 import BM25Okapi

# Corpus + BM25 index (lazy singleton, load 1 lần và cache trong process)
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
_bm25: BM25Okapi | None = None


def _tokenize(text: str) -> list[str]:
    """
    Tokenize đơn giản cho BM25 — hỗ trợ cả tiếng Việt lẫn tiếng Anh.

    Tách theo từ (gồm ký tự chữ + số), lowercase, bỏ token quá ngắn.
    Không dùng underthesea (rút gọn dependency) nhưng vẫn bắt được keyword
    quan trọng như 'tuition', 'scholarship', 'gpa', '₫', 'deposit'...
    """
    return [t for t in re.findall(r"[A-Za-zÀ-ỹ0-9]+", text.lower()) if len(t) > 1]


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    global CORPUS, _bm25
    CORPUS = corpus
    if not corpus:
        _bm25 = None
        return None
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    _bm25 = BM25Okapi(tokenized_corpus)
    return _bm25


def _ensure_index():
    """Lazy load BM25 index từ ChromaDB collection (đồng bộ với task4) nếu chưa có."""
    global CORPUS, _bm25
    if _bm25 is not None:
        return
    try:
        from .task4_chunking_indexing import get_collection
        col = get_collection()
        if col.count() == 0:
            return
        data = col.get(include=["documents", "metadatas"])
        corpus = [
            {"content": d, "metadata": m}
            for d, m in zip(data["documents"], data["metadatas"])
        ]
        build_bm25_index(corpus)
    except Exception as e:  # pragma: no cover
        print(f"  ⚠ BM25 index build failed: {e}")


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content', 'score', 'metadata'} sorted by BM25 score descending.
    """
    _ensure_index()
    if _bm25 is None or not CORPUS:
        return []
    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []
    scores = _bm25.get_scores(tokenized_query)
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"],
            })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("tuition fee payment methods", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
