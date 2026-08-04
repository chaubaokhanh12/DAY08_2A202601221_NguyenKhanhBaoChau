"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results

⚠️ BẪY THƯỜNG GẶP — đọc kỹ trước khi code:
    Nếu bạn dùng điểm RRF đã fuse (Task 7) để so với score_threshold, bạn sẽ gặp bug
    thật: RRF max score luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) BẤT KỂ nội dung có liên quan
    hay không. Nếu đặt threshold thấp (như 0.005) để "hợp" với thang điểm RRF, thực
    chất KHÔNG câu hỏi nào đủ thấp để trigger fallback nữa — kể cả query hoàn toàn vô
    nghĩa vẫn trả về kết quả "hybrid" (rác) thay vì fallback đúng như thiết kế.

    Cách sửa đúng: giữ điểm cosine similarity GỐC của semantic_search (trước khi qua
    RRF) làm căn cứ quyết định fallback, tách biệt khỏi điểm RRF dùng để sắp xếp kết
    quả cuối cùng. Calibrate threshold bằng cách tự đo: chạy vài câu hỏi chắc chắn
    liên quan và vài câu chắc chắn lạc đề/rác qua semantic_search, xem khoảng cách
    điểm số giữa hai nhóm rồi chọn ngưỡng nằm giữa.
"""

import os

from . import task5_semantic_search
from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf


# =============================================================================
# CONFIGURATION
# =============================================================================

# Calibrate bằng cách đo cosine của semantic_search cho câu liên quan vs lạc đề.
# Với paraphrase-multilingual-MiniLM-L12-v2 trên corpus University Services,
# câu liên quan thường > 0.55, câu lạc đề < 0.40 → chọn ngưỡng 0.40.
SCORE_THRESHOLD = 0.40   # Nếu best score (cosine gốc) < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "cross_encoder" | "mmr" | "rrf"

# Cho phép tắt HyDE khi benchmark (retrieval deterministic, ít LLM call).
# Mặc định True (giữ behaviour của chatbot); set RAG_USE_HYDE=false khi đánh giá.
task5_semantic_search.USE_HYDE = os.getenv("RAG_USE_HYDE", "true").lower() == "true"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
          ├→ Lexical Search  → sparse_results
          │
          ├→ Merge (RRF) → merged_results        (chỉ khi use_reranking=True)
          │
          └→ If dense_results[0]["score"] < threshold:
                └→ PageIndex Vectorless → fallback_results (nếu có)

    use_reranking=True  → hybrid (semantic + BM25 + RRF fusion)
    use_reranking=False → dense-only (chỉ semantic, không lexical/fusion)

    Returns:
        List of {'content', 'score', 'metadata', 'source'}
        ('source' = 'hybrid' | 'dense' | 'pageindex')
    """
    # Step 1: Semantic search (luôn chạy — dense_results giữ cosine gốc)
    dense_results = semantic_search(query, top_k=top_k * 2)
    best_score = dense_results[0]["score"] if dense_results else 0.0

    if use_reranking:
        # Step 2: Lexical search (BM25)
        sparse_results = lexical_search(query, top_k=top_k * 2)
        # Step 3: Merge bằng RRF
        if dense_results or sparse_results:
            merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
        else:
            merged = []
        for item in merged:
            item["source"] = "hybrid"
        final_results = merged[:top_k] if merged else dense_results[:top_k]
    else:
        # Dense-only: không fuse, không lexical
        final_results = dense_results[:top_k]
        for item in final_results:
            item["source"] = "dense"

    # Step 4: Fallback PageIndex nếu cosine gốc quá thấp (KHÔNG dùng điểm RRF)
    if best_score < score_threshold:
        try:
            from .task8_pageindex_vectorless import pageindex_search
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback
        except NotImplementedError:
            pass  # PageIndex chưa implement → giữ kết quả hybrid/dense

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "What is the tuition fee at RMIT Vietnam?",
        "How do I book a library study room?",
        "What scholarships are available for international students?",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
