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

Calibration thực tế (OpenAI text-embedding-3-small + cosine, ChromaDB):
    - Query liên quan  (học phí, học bổng, thư viện, chỗ ở, đăng ký): 0.44 – 0.64
      ,(max top-1 ≈ 0.64, min ≈ 0.44 với "course registration portal")
    - Query lạc đề    (pizza recipe, bicycle tyre, capital mongolia,    0.13 – 0.27
                      xyzabc123nonsense qwerty asdf)
    → Khoảng trống giữa hai nhóm: 0.27 – 0.44. Ngưỡng 0.32 nằm giữa khoảng trống
    này: query liên quan (>0.32) không bị fallback, query lạc đề (<0.32) trigger
    fallback PageIndex.
    (Đo lại nếu đổi corpus/embedding model — xem script calibration ở cuối file.)
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# Calibrate threshold này bằng cách tự đo điểm cosine của semantic_search
# cho câu hỏi liên quan vs câu hỏi lạc đề (xem ghi chú ở trên) — ĐỪNG copy nguyên
# giá trị mẫu, mỗi corpus/embedding model sẽ cho khoảng điểm khác nhau.
#
# Đo thực tế với OpenAI text-embedding-3-small (xem docstring đầu file):
#   related  ≈ 0.44 – 0.64,  off-topic ≈ 0.13 – 0.27 → chọn 0.32 giữa khoảng trống.
SCORE_THRESHOLD = 0.32   # Nếu best cosine gốc < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "cross_encoder" | "mmr" | "rrf"
# Lấy nhiều hơn top_k ở mỗi ranker để RRF có pool lớn hơn khi gộp, sau đó cắt.
CANDIDATE_MULTIPLIER = 2


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
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If dense_results[0]["score"] < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (KHÔNG phải điểm RRF)
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    pool = max(top_k * CANDIDATE_MULTIPLIER, top_k)

    # Step 1: Song song chạy semantic + lexical
    #   - dense_results["score"] = cosine similarity gốc (thang [0,1]) — dùng để
    #     quyết định fallback ở Step 4.
    #   - Bọc try/except để pipeline không chết khi 1 ranker lỗi (vd OpenAI rate
    #     limit, ChromaDB chưa index). Ranker lỗi → coi như không có kết quả.
    try:
        dense_results = semantic_search(query, top_k=pool)
    except Exception as e:
        print(f"  ⚠ semantic_search lỗi ({type(e).__name__}: {e}) → bỏ qua dense")
        dense_results = []

    try:
        sparse_results = lexical_search(query, top_k=pool)
    except Exception as e:
        print(f"  ⚠ lexical_search lỗi ({type(e).__name__}: {e}) → bỏ qua sparse")
        sparse_results = []

    # Step 2: Merge bằng RRF
    #   rerank_rrf gộp 2 ranked list, vứt thang điểm khác nhau (cosine vs BM25),
    #   chỉ giữ thứ hạng. `score` sau merge = điểm RRF (~0.016), điểm gốc được
    #   giữ trong `retrieval_score`.
    merged = rerank_rrf([dense_results, sparse_results], top_k=pool)
    for item in merged:
        item["source"] = "hybrid"

    # Step 3: Rerank thêm (tuỳ chọn)
    #   RERANK_METHOD mặc định "rrf" đã thực hiện ở Step 2 (merge RRF) — không rerank
    #   lại để tránh double-RRF. Chỉ rerank thêm khi chọn method khác (cross_encoder
    #   cần Jina key, mmr cần embedding từng chunk). Lỗi → lùi về merged.
    if use_reranking and merged and RERANK_METHOD != "rrf":
        try:
            final_results = rerank(
                query, merged, top_k=top_k, method=RERANK_METHOD
            )
            for item in final_results:
                item["source"] = "hybrid"
        except Exception as e:
            print(f"  ⚠ rerank({RERANK_METHOD}) lỗi ({e}) → giữ kết quả RRF merged")
            final_results = merged
    else:
        final_results = merged

    # Step 4: Check threshold DÙNG ĐIỂM COSINE GỐC (dense_results), KHÔNG PHẢI RRF
    #   dense_results[0]["score"] là cosine similarity gốc — thang [0,1] có ý nghĩa
    #   so với score_threshold. Điểm RRF (~0.016) KHÔNG dùng để so threshold.
    best_score = dense_results[0]["score"] if dense_results else 0.0
    if best_score < score_threshold:
        print(
            f"  ⚠ Semantic best score ({best_score:.3f}) < threshold "
            f"({score_threshold:.3f}) → fallback PageIndex"
        )
        try:
            fallback = pageindex_search(query, top_k=top_k)
        except Exception as e:
            print(f"  ⚠ pageindex_search lỗi ({type(e).__name__}: {e})")
            fallback = []
        if fallback:
            # fallback items đã có source="pageindex" từ task8
            return fallback

    return final_results[:top_k]


# =============================================================================
# Calibration helper — chạy `python -m src.task9_retrieval_pipeline` để xem
# điểm cosine của một batch query liên quan vs lạc đề, rồi cập nhật threshold.
# =============================================================================

_CALIB_RELATED = [
    "tuition fee payment",
    "scholarship eligibility requirements",
    "library study room booking",
    "accommodation services HCM",
    "course registration portal",
]
_CALIB_OFFTOPIC = [
    "xyzabc123nonsense qwerty asdf",
    "the best pizza recipe in italy",
    "how to fix a bicycle tyre puncture",
    "capital city of mongolia population",
]


def _calibrate():
    """In điểm cosine top-1 của các query liên quan vs lạc đề để xác định threshold."""
    print("=" * 60)
    print("Calibration — điểm cosine similarity (semantic_search top-1)")
    print("Lý tưởng: Related > threshold > Off-topic")
    print("=" * 60)
    print("\n=== RELATED ===")
    for q in _CALIB_RELATED:
        try:
            r = semantic_search(q, top_k=1)
            print(f"  {r[0]['score']:.4f}  {q}" if r else f"  EMPTY   {q}")
        except Exception as e:
            print(f"  ERROR   {q}  ({e})")
    print("\n=== OFF-TOPIC ===")
    for q in _CALIB_OFFTOPIC:
        try:
            r = semantic_search(q, top_k=1)
            print(f"  {r[0]['score']:.4f}  {q}" if r else f"  EMPTY   {q}")
        except Exception as e:
            print(f"  ERROR   {q}  ({e})")
    print(f"\nHiện tại SCORE_THRESHOLD = {SCORE_THRESHOLD}")
    print("Cập nhật SCORE_THRESHOLD nằm giữa nhóm related và off-topic.")


if __name__ == "__main__":
    import sys

    if "--calibrate" in sys.argv:
        _calibrate()
        sys.exit(0)

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
        if not results:
            print("  (no results)")