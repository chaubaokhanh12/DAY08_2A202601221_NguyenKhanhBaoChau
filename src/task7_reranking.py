"""
Task 7 — Reranking Module.

Sử dụng RRF (Reciprocal Rank Fusion) để gộp kết quả từ:
    1. Cross-encoder reranker: Qwen/Qwen3-Reranker-0.6B (local model)
    2. MMR (Maximal Marginal Relevance): tự implement

Pipeline:
    Candidates → Cross-encoder scoring → ranked_list_1
    Candidates → MMR (diversity)       → ranked_list_2
    RRF([ranked_list_1, ranked_list_2]) → Final reranked results

═══════════════════════════════════════════════════════════════════════════════
CƠ CHẾ HOẠT ĐỘNG
═══════════════════════════════════════════════════════════════════════════════

1. CROSS-ENCODER (Qwen/Qwen3-Reranker-0.6B):
   - Nhận cặp (query, document) → score relevance trực tiếp
   - Khác bi-encoder (embed riêng query và doc rồi so cosine): cross-encoder
     xử lý query + doc cùng lúc → attention giữa 2 phía → chính xác hơn
   - Nhược điểm: chậm hơn (phải chạy model cho mỗi cặp)
   - Qwen3-Reranker-0.6B: 0.6B params, multilingual, nhẹ, chạy được trên CPU

2. MMR (Maximal Marginal Relevance — Carbonell & Goldstein, 1998):
   - MMR(d) = λ × sim(query, d) - (1-λ) × max(sim(d, d_selected))
   - λ = 0.7: ưu tiên relevance (70%) > diversity (30%)
   - Mục đích: tránh trả về nhiều chunks gần giống nhau
   - Iterative selection: chọn document maximize MMR score từng bước

3. RRF (Reciprocal Rank Fusion — Cormack et al., 2009):
   - RRF(d) = Σ 1 / (k + rank_r(d))   với k=60
   - Gộp kết quả từ nhiều ranker mà KHÔNG cần normalize scores
   - Ưu điểm: robust, không bị ảnh hưởng bởi scale khác nhau giữa rankers

Cài đặt bổ sung:
    pip install transformers torch
    # transformers đã được kéo theo bởi sentence-transformers trong requirements.txt
"""

import numpy as np
from typing import Optional

from .task4_chunking_indexing import get_embedding_model


# =============================================================================
# CROSS-ENCODER — Qwen/Qwen3-Reranker-0.6B
# =============================================================================

# Singleton cache cho cross-encoder model (tránh load lại mỗi lần rerank)
_reranker_model = None
_reranker_tokenizer = None

# Model configuration
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_MAX_LENGTH = 1024   # Max tokens per (query, document) pair

# Task instruction cho Qwen3-Reranker (theo model card)
RERANKER_TASK = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def _get_reranker():
    """
    Lazy-load Qwen/Qwen3-Reranker-0.6B model và tokenizer.
    Singleton pattern — model chỉ load 1 lần, cache trong memory.

    Returns:
        Tuple of (model, tokenizer)
    """
    global _reranker_model, _reranker_tokenizer
    if _reranker_model is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        print(f"  ⏳ Loading reranker model: {RERANKER_MODEL_NAME}...")

        _reranker_tokenizer = AutoTokenizer.from_pretrained(
            RERANKER_MODEL_NAME,
            padding_side="left",
        )

        # Chọn dtype phù hợp: float16 cho GPU, float32 cho CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        _reranker_model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL_NAME,
            torch_dtype=dtype,
        ).to(device)
        _reranker_model.eval()

        print(f"  ✓ Reranker loaded on {device} ({dtype})")

    return _reranker_model, _reranker_tokenizer


def _format_reranker_input(query: str, document: str) -> str:
    """
    Format input cho Qwen3-Reranker theo đúng template từ model card.

    Format:
        <|im_start|>user
        <instruct>{task}
        <query>{query}
        <doc>{document}<|im_end|>
        <|im_start|>assistant
        <think>

        </think>

    Các token <instruct>, <query>, <doc> là special tokens trong vocabulary
    của Qwen3-Reranker, KHÔNG phải HTML tags.
    """
    prefix = "<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    body = f"<instruct>{RERANKER_TASK}\n<query>{query}\n<doc>{document}"
    return prefix + body + suffix


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng Qwen/Qwen3-Reranker-0.6B cross-encoder.

    Cross-encoder khác bi-encoder:
    - Bi-encoder: embed query và doc RIÊNG → so cosine (nhanh nhưng kém chính xác)
    - Cross-encoder: encode (query, doc) CÙNG LÚC → attention giữa 2 phía
      → score trực tiếp (chậm hơn nhưng chính xác hơn cho reranking)

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored by cross-encoder, sorted descending.
    """
    import torch

    if not candidates:
        return []

    model, tokenizer = _get_reranker()
    device = next(model.parameters()).device

    # Format tất cả (query, document) pairs theo Qwen3-Reranker template
    formatted_inputs = [
        _format_reranker_input(query, c["content"])
        for c in candidates
    ]

    # Tokenize batch
    inputs = tokenizer(
        formatted_inputs,
        padding=True,
        truncation=True,
        max_length=RERANKER_MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    # Inference — cross-encoder trả về logits (higher = more relevant)
    with torch.no_grad():
        logits = model(**inputs).logits.view(-1).float()
        scores = logits.cpu().tolist()

    # Gán cross-encoder score vào candidates
    scored = []
    for candidate, ce_score in zip(candidates, scores):
        scored.append({**candidate, "score": float(ce_score)})

    # Sort by cross-encoder score descending
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# =============================================================================
# MMR — Maximal Marginal Relevance
# =============================================================================

def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity giữa 2 vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    dot = np.dot(a_arr, b_arr)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm < 1e-8:
        return 0.0
    return float(dot / norm)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR(d) = λ × sim(query, d) - (1-λ) × max(sim(d, d_selected))

    λ = 0.7 → ưu tiên relevance (70%) > diversity (30%)
    Giảm λ để tăng diversity (tránh trùng lặp), tăng λ để ưu tiên relevance.

    Algorithm (greedy iterative):
    1. Chọn document có relevance cao nhất → thêm vào selected
    2. Lặp: chọn document maximize MMR score (cân bằng relevance & diversity)
    3. Dừng khi đủ top_k

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR, giữ nguyên score gốc.
    """
    if not candidates:
        return []

    if len(candidates) <= top_k:
        return candidates[:]

    # Tính relevance score cho tất cả candidates
    relevances = [
        _cosine_sim(query_embedding, c["embedding"])
        for c in candidates
    ]

    selected_indices = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_mmr = float("-inf")

        for idx in remaining:
            # Relevance to query
            relevance = relevances[idx]

            # Max similarity to already selected documents (diversity penalty)
            max_sim_to_selected = 0.0
            for sel_idx in selected_indices:
                sim = _cosine_sim(
                    candidates[idx]["embedding"],
                    candidates[sel_idx]["embedding"],
                )
                max_sim_to_selected = max(max_sim_to_selected, sim)

            # MMR score
            mmr_score = (
                lambda_param * relevance
                - (1 - lambda_param) * max_sim_to_selected
            )

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

    return [candidates[i] for i in selected_indices]


# =============================================================================
# RRF — Reciprocal Rank Fusion
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Ưu điểm so với weighted sum:
    - KHÔNG cần normalize scores giữa các rankers
    - Robust: hoạt động tốt dù scale/distribution of scores khác nhau
    - Simple: chỉ phụ thuộc vào thứ hạng (rank), không phụ thuộc giá trị score

    Lưu ý quan trọng (dùng ở Task 9):
    - Score sau RRF CHỈ phản ánh thứ hạng, KHÔNG phản ánh relevance thực
    - Top-1 luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) dù nội dung có liên quan hay không
    - ĐỪNG dùng RRF score cho fallback threshold → dùng cosine gốc từ Task 5

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    if not ranked_lists:
        return []

    rrf_scores: dict[str, float] = {}   # content → RRF score
    content_map: dict[str, dict] = {}   # content → full dict (lưu bản gốc)

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            # Giữ bản gốc (lấy version mới nhất nếu trùng)
            content_map[key] = item

    # Sort by RRF score descending
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, rrf_score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = rrf_score
        results.append(item)

    return results


# =============================================================================
# UNIFIED RERANK INTERFACE
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Khi method="rrf" (mặc định):
        Pipeline nâng cao kết hợp cả 3 phương pháp:
        1. Cross-encoder (Qwen3-Reranker-0.6B) → ranked_list_1 (relevance)
        2. MMR → ranked_list_2 (relevance + diversity)
        3. RRF gộp cả 2 → kết quả cuối cùng
        → Kết hợp ưu điểm: cross-encoder cho relevance chính xác,
          MMR cho diversity, RRF cho fusion robust.

    Khi method="cross_encoder":
        Chỉ dùng cross-encoder scoring.

    Khi method="mmr":
        Chỉ dùng MMR.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if not candidates:
        return []

    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)

    elif method == "mmr":
        # MMR cần embeddings — embed candidates nếu chưa có
        candidates_with_emb = _ensure_embeddings(query, candidates)
        model = get_embedding_model()
        query_embedding = model.encode(query, normalize_embeddings=True).tolist()
        return rerank_mmr(query_embedding, candidates_with_emb, top_k)

    elif method == "rrf":
        # === RRF PIPELINE: Cross-encoder + MMR → RRF fusion ===

        # Bước 1: Cross-encoder scoring → ranked_list_1
        # Qwen3-Reranker-0.6B score mỗi (query, candidate) pair
        ce_ranked = rerank_cross_encoder(query, candidates, top_k=len(candidates))

        # Bước 2: MMR → ranked_list_2
        # Embed candidates + query, chọn diverse + relevant set
        candidates_with_emb = _ensure_embeddings(query, candidates)
        model = get_embedding_model()
        query_embedding = model.encode(query, normalize_embeddings=True).tolist()
        mmr_ranked = rerank_mmr(
            query_embedding, candidates_with_emb, top_k=len(candidates), lambda_param=0.7
        )

        # Bước 3: RRF fusion — gộp 2 ranked lists
        # RRF không phụ thuộc scale of scores, chỉ dùng rank position
        final = rerank_rrf([ce_ranked, mmr_ranked], top_k=top_k)
        return final

    else:
        raise ValueError(f"Unknown rerank method: {method}")


def _ensure_embeddings(query: str, candidates: list[dict]) -> list[dict]:
    """
    Đảm bảo mỗi candidate có 'embedding' key.
    Nếu chưa có, embed bằng model từ Task 4 (BAAI/bge-m3).
    """
    # Kiểm tra xem candidates đã có embedding chưa
    needs_embedding = [c for c in candidates if "embedding" not in c]

    if not needs_embedding:
        return candidates  # Tất cả đã có embedding

    # Embed candidates chưa có embedding
    model = get_embedding_model()
    texts = [c["content"] for c in needs_embedding]
    embeddings = model.encode(texts, normalize_embeddings=True)

    # Tạo bản copy và gắn embedding
    result = []
    embed_idx = 0
    for c in candidates:
        if "embedding" not in c:
            new_c = {**c, "embedding": embeddings[embed_idx].tolist()}
            embed_idx += 1
            result.append(new_c)
        else:
            result.append(c)

    return result


# =============================================================================
# MAIN — Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Task 7: Reranking Test")
    print("  Method: RRF (Cross-encoder + MMR fusion)")
    print(f"  Cross-encoder: {RERANKER_MODEL_NAME}")
    print("=" * 60)

    # Dummy candidates (mô phỏng output từ retrieval)
    dummy_candidates = [
        {
            "content": "Học phí tại RMIT Vietnam năm 2024 là 150 triệu đồng/năm cho chương trình cử nhân",
            "score": 0.85,
            "metadata": {"source": "tuition-fees.md", "type": "legal"},
        },
        {
            "content": "Tuition fee payment schedule: students must pay before the start of each semester",
            "score": 0.78,
            "metadata": {"source": "tuition-fees.md", "type": "legal"},
        },
        {
            "content": "Scholarship eligibility: GPA >= 3.5, IELTS >= 6.5, financial need assessment",
            "score": 0.65,
            "metadata": {"source": "scholarships.md", "type": "legal"},
        },
        {
            "content": "Thư viện RMIT mở cửa từ 7h-22h, có phòng học nhóm đặt qua hệ thống online",
            "score": 0.55,
            "metadata": {"source": "library-services.md", "type": "news"},
        },
        {
            "content": "Python programming language was created by Guido van Rossum in 1991",
            "score": 0.30,
            "metadata": {"source": "random.md", "type": "news"},
        },
    ]

    print("\n--- Input candidates ---")
    for i, c in enumerate(dummy_candidates):
        print(f"  {i+1}. [{c['score']:.2f}] {c['content'][:70]}...")

    print("\n--- Reranking with RRF (Cross-encoder + MMR) ---")
    results = rerank("học phí RMIT bao nhiêu", dummy_candidates, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['score']:.4f}] {r['content'][:70]}...")
