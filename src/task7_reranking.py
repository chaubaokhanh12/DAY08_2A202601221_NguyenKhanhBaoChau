"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — khuyến nghị vì không cần API key

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

→ Module này chọn RRF làm mặc định (không cần API key, không cần model, chạy offline),
  đồng thời implement đủ cả cross-encoder và MMR để so sánh khi demo.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.

    → Để Task 9 (và UI của nhóm) vẫn còn điểm số "thật" mà dùng, `rerank_rrf()` giữ lại
      điểm gốc trước khi fuse trong key `retrieval_score`, chỉ ghi đè `score` bằng điểm RRF.
"""

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_RERANK_MODEL = "jina-reranker-v2-base-multilingual"


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model (Jina Reranker v2 API).

    Khác bi-encoder (Task 5): bi-encoder embed query và document RIÊNG rồi so cosine,
    còn cross-encoder đưa CẢ CẶP (query, document) qua cùng một lần forward nên bắt
    được tương tác từ-với-từ giữa hai vế → chính xác hơn nhiều, nhưng đắt hơn nên chỉ
    dùng để chấm lại top-N candidates chứ không quét toàn corpus.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.

    Raises:
        RuntimeError: Nếu chưa có JINA_API_KEY hoặc API trả lỗi.
    """
    if not candidates:
        return []
    if not JINA_API_KEY:
        raise RuntimeError(
            "Chưa có JINA_API_KEY trong .env — lấy key miễn phí tại https://jina.ai/reranker/"
        )

    import requests

    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {JINA_API_KEY}"},
        json={
            "model": JINA_RERANK_MODEL,
            "query": query,
            "documents": [c["content"] for c in candidates],
            "top_n": top_k,
        },
        timeout=30,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Jina rerank API lỗi {response.status_code}: {response.text[:200]}")

    results = []
    for r in response.json().get("results", []):
        item = dict(candidates[r["index"]])
        item["retrieval_score"] = item.get("score")
        item["score"] = float(r["relevance_score"])  # cross-encoder score trong [0,1]
        results.append(item)
    return results[:top_k]


def _cosine_sim(a, b) -> float:
    """Cosine similarity giữa 2 vector (list[float] hoặc np.ndarray)."""
    import numpy as np

    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom else 0.0


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Dùng khi top-k bị trùng lặp nội dung (nhiều chunk cạnh nhau của cùng 1 file) —
    MMR phạt candidate quá giống thứ đã chọn, nhường chỗ cho khía cạnh khác của câu hỏi.

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR (key 'score' = điểm MMR).
    """
    if not candidates:
        return []

    missing = [i for i, c in enumerate(candidates) if not c.get("embedding")]
    if missing:
        raise ValueError(
            f"{len(missing)}/{len(candidates)} candidates thiếu key 'embedding' — "
            "MMR cần embedding của từng chunk."
        )

    selected: list[int] = []
    remaining = list(range(len(candidates)))
    mmr_scores: dict[int, float] = {}

    for _ in range(min(top_k, len(candidates))):
        best_idx, best_score = None, float("-inf")

        for idx in remaining:
            relevance = _cosine_sim(query_embedding, candidates[idx]["embedding"])

            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = _cosine_sim(
                    candidates[idx]["embedding"], candidates[sel_idx]["embedding"]
                )
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score, best_idx = mmr_score, idx

        selected.append(best_idx)
        mmr_scores[best_idx] = best_score
        remaining.remove(best_idx)

    results = []
    for idx in selected:
        item = dict(candidates[idx])
        item["retrieval_score"] = item.get("score")
        item["score"] = float(mmr_scores[idx])
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Vì sao dùng RRF để gộp dense + sparse: điểm cosine (Task 5, thang [0,1]) và điểm
    BM25 (Task 6, không chặn trên) KHÔNG cùng đơn vị, cộng thẳng vào nhau là vô nghĩa.
    RRF vứt bỏ giá trị điểm, chỉ dùng THỨ HẠNG nên hai thang đo khác nhau vẫn gộp được.
    Hằng số k=60 làm phẳng đường cong: chênh lệch giữa hạng 1 và hạng 2 không quá lớn,
    nhờ đó một chunk được CẢ HAI ranker bình chọn ở hạng trung bình vẫn thắng được
    chunk chỉ đứng nhất ở một ranker duy nhất.

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
        Mỗi item giữ điểm gốc trong 'retrieval_score', 'score' = điểm RRF.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list or [], 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            # Giữ bản ghi ĐẦU TIÊN gặp được: ranker đứng trước trong ranked_lists
            # (thường là semantic) sẽ quyết định metadata/điểm gốc hiển thị.
            content_map.setdefault(key, item)

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[: max(top_k, 0)]:
        item = dict(content_map[content])
        item["retrieval_score"] = item.get("score")
        item["score"] = float(score)
        results.append(item)
    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
    query_embedding: Optional[list[float]] = None,
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval.
            Với method="rrf" chấp nhận CẢ HAI dạng:
              - list[dict]        → coi như 1 ranked list (chỉ re-score theo thứ hạng)
              - list[list[dict]]  → nhiều ranked list, fuse đúng nghĩa RRF
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking
        query_embedding: Bắt buộc cho method="mmr"

    Returns:
        List of top_k reranked candidates.

    Ghi chú: cross_encoder/mmr nếu thiếu điều kiện (API key, embedding) sẽ IN CẢNH BÁO
    và tự lùi về RRF thay vì ném lỗi, để pipeline Task 9 của nhóm không chết giữa demo.
    """
    if not candidates:
        return []

    if method == "cross_encoder":
        try:
            return rerank_cross_encoder(query, candidates, top_k)
        except Exception as e:
            print(f"  ⚠ Cross-encoder rerank thất bại ({e}) → dùng RRF thay thế")
            method = "rrf"

    if method == "mmr":
        if query_embedding is None:
            print("  ⚠ MMR cần query_embedding → dùng RRF thay thế")
        else:
            try:
                return rerank_mmr(query_embedding, candidates, top_k)
            except Exception as e:
                print(f"  ⚠ MMR rerank thất bại ({e}) → dùng RRF thay thế")
        method = "rrf"

    if method == "rrf":
        # Cho phép truyền thẳng nhiều ranked list, hoặc 1 list phẳng đã merge sẵn.
        if isinstance(candidates[0], list):
            ranked_lists = candidates
        else:
            ranked_lists = [candidates]
        return rerank_rrf(ranked_lists, top_k=top_k)

    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Tuition fee payment schedule", "score": 0.8, "metadata": {}},
        {"content": "Scholarship eligibility requirements", "score": 0.6, "metadata": {}},
        {"content": "Library study room booking guide", "score": 0.5, "metadata": {}},
    ]
    results = rerank("tuition fee payment", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")

    # Demo RRF gộp 2 ranker: chunk được cả hai cùng bình chọn sẽ leo lên đầu
    dense = [
        {"content": "A - tuition fee 2025", "score": 0.71, "metadata": {}},
        {"content": "B - scholarship", "score": 0.65, "metadata": {}},
    ]
    sparse = [
        {"content": "C - fee payment portal", "score": 12.4, "metadata": {}},
        {"content": "A - tuition fee 2025", "score": 9.1, "metadata": {}},
    ]
    print("\n[RRF fusion dense + sparse]")
    for r in rerank_rrf([dense, sparse], top_k=3):
        print(f"[{r['score']:.4f}] (gốc={r['retrieval_score']}) {r['content']}")
