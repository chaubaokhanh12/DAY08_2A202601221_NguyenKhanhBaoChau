"""
RAG Evaluation Pipeline — University Services (RAGAS).

Framework đã chọn: **RAGAS** (chuẩn industry cho RAG evaluation).

Pipeline:
    1. Load golden_dataset.json (20 cặp Q&A — xép theo category)
    2. Chạy RAG pipeline (src/task10_generation.generate_with_citation) trên từng question
    3. Đánh giá 4 metric: faithfulness, answer_relevancy, context_recall, context_precision
    4. So sánh A/B 2 config retrieval (hybrid + rerank  vs  dense-only / no rerank)
    5. Export kết quả ra results.md

Lưu ý rate limit với OpenRouter ":free": RAGAS/DeepEval gọi LLM RẤT NHIỀU LẦN
(không phải 1 lần/câu hỏi mà nhiều lần/metric/câu hỏi). Nên:
    - Dùng biến môi trường EVAL_SUBSET để giới hạn số câu (VD: EVAL_SUBSET=5).
    - Model free của OpenRouter giới hạn ~50 request/ngày cho cả tài khoản.

Cách chạy:
    # Full RAGAS (cần OPENROUTER_API_KEY thật + pipeline đã implement + ragas import OK):
    python -m group_project.evaluation.eval_pipeline

    # Chỉ đánh giá retrieval offline (KHÔNG cần LLM, không cần key):
    python -m group_project.evaluation.eval_pipeline --offline

Biến môi trường:
    OPENROUTER_API_KEY   API key OpenRouter (LLM generation + RAGAS judge)
    OPENAI_API_KEY       fallback nếu không dùng OpenRouter
    EVAL_MODEL           model ID cho RAGAS judge (mặc định openai/gpt-4o-mini)
    EVAL_SUBSET          số câu hỏi tối đa để chạy (giảm rate limit), 0 = tất cả
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

# Project root -> để import src.*
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_JUDGE_MODEL = os.getenv("EVAL_MODEL", "openai/gpt-4o-mini")
DEFAULT_TOP_K = 5


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_subset(dataset: list[dict]) -> list[dict]:
    """Giới hạn số câu hỏi theo EVAL_SUBSET (giảm rate limit)."""
    n = os.getenv("EVAL_SUBSET", "0")
    try:
        n = int(n)
    except ValueError:
        n = 0
    if n and n > 0:
        print(f"  ℹ EVAL_SUBSET={n}: chỉ chạy trên {n}/{len(dataset)} câu đầu.")
        return dataset[:n]
    return dataset


# =============================================================================
# LLM client cho RAGAS judge
# =============================================================================

def build_judge_llm(model: str | None = None):
    """
    Tạo LLM wrapper cho RAGAS judge. Provider-aware theo tiền tố key:
      - 'sk-or...'  -> OpenRouter (base_url openrouter, model dạng 'vendor/model')
      - 'sk-...'    -> OpenAI trực tiếp (model 'gpt-4o-mini')

    Trả về tuple (ragas_llm, model_name). Raise RuntimeError nếu không có key.
    """
    from openai import OpenAI as OpenAIClient
    from ragas.llms import llm_factory

    or_key = os.getenv("OPENROUTER_API_KEY")
    oai_key = os.getenv("OPENAI_API_KEY")
    candidates = [or_key, oai_key]

    # OpenAI direct key (sk-proj / sk- nhưng không phải sk-or) — kể cả khi đặt sai tên biến
    for key in candidates:
        if key and key.startswith("sk-") and not key.startswith("sk-or"):
            client = OpenAIClient(api_key=key, base_url="https://api.openai.com/v1")
            return llm_factory("gpt-4o-mini", provider="openai", client=client), "gpt-4o-mini"
    # OpenRouter key
    for key in candidates:
        if key and key.startswith("sk-or") and key.strip() != "sk-or-v1-...":
            client = OpenAIClient(api_key=key, base_url="https://openrouter.ai/api/v1")
            m = model or DEFAULT_JUDGE_MODEL
            return llm_factory(m, provider="openai", client=client), m
    raise RuntimeError(
        "Không tìm thấy LLM API key hợp lệ. RAGAS cần LLM để chấm faithfulness/"
        "answer_relevancy/context_recall/context_precision.\n"
        "Thiết lập OPENROUTER_API_KEY (hoặc OPENAI_API_KEY) thật trong file .env."
    )


def build_judge_embeddings():
    """
    Embeddings cho RAGAS (answer_relevancy cần embeddings).

    Dùng BAAI/bge-m3 (multilingual — phù hợp tiếng Việt) qua sentence-transformers,
    bọc bằng LangchainEmbeddings cho RAGAS. Không tốn API quota.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return LangchainEmbeddingsWrapper(embeddings=emb)


# =============================================================================
# Option 2: RAGAS  (framework đã chọn)
# =============================================================================

def _run_pipeline(query: str, top_k: int, use_reranking: bool | None = None) -> dict:
    """
    Gọi RAG pipeline. Hỗ trợ cả 2 chữ ký:
        - generate_with_citation(query, top_k)
        - generate_with_citation(query, top_k, config={"use_reranking": ...})
    Trả về dict có keys: 'answer' (str), 'sources' (list[dict] có 'content').
    """
    from src.task10_generation import generate_with_citation

    if use_reranking is not None:
        # Thử truyền config để bật/tắt rerank cho A/B (pipeline có thể không hỗ trợ)
        try:
            return generate_with_citation(
                query, top_k=top_k, config={"use_reranking": use_reranking}
            )
        except TypeError:
            pass  # pipeline không nhận config -> fallback chữ ký gốc
    return generate_with_citation(query, top_k=top_k)


def evaluate_with_ragas(
    rag_pipeline=None,
    golden_dataset: list[dict] | None = None,
    top_k: int = DEFAULT_TOP_K,
    use_reranking: bool | None = None,
    config_label: str = "config",
) -> dict:
    """
    Evaluate RAG pipeline bằng RAGAS (4 metric LLM-judge).

    Args:
        rag_pipeline: giữ cho tương thích template; thực tế gọi generate_with_citation.
        golden_dataset: list các {question, expected_answer, expected_context}.
        top_k: số chunk retrieval.
        use_reranking: None = mặc định pipeline; True/False cho A/B.
        config_label: nhãn để phân biệt kết quả A/B.

    Returns:
        {
            "label": str,
            "overall": {metric: mean_score},
            "per_question": [{id, question, metrics...}],
        }
    """
    from ragas import evaluate
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )

    golden_dataset = golden_dataset or load_golden_dataset()
    golden_dataset = _apply_subset(golden_dataset)

    judge_llm, model_name = build_judge_llm()
    print(f"  ℹ RAGAS judge LLM: {model_name}")

    samples: list[SingleTurnSample] = []
    debug_rows: list[dict] = []
    for item in golden_dataset:
        result = _run_pipeline(item["question"], top_k=top_k, use_reranking=use_reranking)
        answer = result.get("answer", "")
        contexts = [c.get("content", "") for c in result.get("sources", [])]
        debug_rows.append({"id": item.get("id"), "question": item["question"]})
        samples.append(SingleTurnSample(
            user_input=item["question"],
            response=answer,
            retrieved_contexts=contexts,
            reference=item["expected_answer"],
            reference_contexts=[item["expected_context"]],
        ))

    dataset = EvaluationDataset(samples)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=judge_llm,
        embeddings=build_judge_embeddings(),
        show_progress=True,
        raise_exceptions=False,
    )

    df = result.to_pandas()
    metric_names = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    overall = {m: float(df[m].mean()) if m in df.columns else float("nan") for m in metric_names}

    per_q = []
    for i, row in df.iterrows():
        meta = debug_rows[i] if i < len(debug_rows) else {}
        per_q.append({
            "id": meta.get("id", f"Q{i+1}"),
            "question": meta.get("question", ""),
            **{m: (float(row[m]) if m in row and row[m] == row[m] else None) for m in metric_names},
        })

    return {"label": config_label, "overall": overall, "per_question": per_q}


# =============================================================================
# A/B Comparison
# =============================================================================

CONFIGS = {
    "A_hybrid_rerank": {"use_reranking": True, "label": "Config A — hybrid search + RRF rerank"},
    "B_dense_only": {"use_reranking": False, "label": "Config B — dense-only (no rerank)"},
}


def compare_configs(rag_pipeline=None, golden_dataset: list[dict] | None = None) -> dict:
    """
    So sánh A/B 2 config retrieval bằng RAGAS.

        Config A: hybrid (semantic + BM25) + RRF rerank
        Config B: dense-only — chỉ semantic, không rerank

    Chạy evaluate_with_ragas 2 lần (chỉ nên dùng EVAL_SUBSET nhỏ do rate limit).
    Trả về {config_key: eval_result}.
    """
    golden_dataset = golden_dataset or load_golden_dataset()
    results = {}
    for key, cfg in CONFIGS.items():
        print(f"\n>>> Đánh giá {cfg['label']}")
        try:
            results[key] = evaluate_with_ragas(
                rag_pipeline=rag_pipeline,
                golden_dataset=golden_dataset,
                use_reranking=cfg["use_reranking"],
                config_label=cfg["label"],
            )
        except Exception as e:  # pragma: no cover - lỗi 1 config không giết cả job
            print(f"  ⚠ Config {key} thất bại: {e}")
            results[key] = None
    return results


# =============================================================================
# Fallback: đánh giá retrieval OFFLINE (không cần LLM, không cần key)
# =============================================================================

def evaluate_retrieval_offline(golden_dataset: list[dict] | None = None) -> dict:
    """
    Đánh giá CHỈ retrieval (context recall/precision) KHÔNG dùng LLM.

    Cho mỗi câu hỏi, gọi src.task9_retrieval_pipeline.retrieve() với 2 config
    (use_reranking True/False). Mỗi chunk truy được đánh dấu "hit" nếu text
    chứa dấu hiệu nguồn của expected_context (tên file PDF/article + keyword).
    context_recall = có ít nhất 1 hit không (binary), context_precision =
    số chunk hit / số chunk truy về.

    Dùng để có số A/B thật ngay cả khi chưa có LLM key. Đây là proxy, KHÔNG
    thay thế RAGAS (RAGAS dùng LLM judge). Metric faithfulness/answer_relevancy
    cần LLM → để trống, ghi rõ trong results.md.
    """
    from src.task9_retrieval_pipeline import retrieve

    golden_dataset = golden_dataset or load_golden_dataset()
    golden_dataset = _apply_subset(golden_dataset)

    def _source_tokens(expected_context: str) -> list[str]:
        """Trích token nhận diện nguồn từ expected_context (vd: 'tuition-fees', 'article_01')."""
        low = expected_context.lower()
        toks = []
        for kw in ("tuition-fees", "scholarship-terms", "accommodation-hcm", "accommodation-hanoi",
                   "wellbeing", "article_01", "article_02", "article_03", "article_04",
                   "article_05", "article_06", "article_07"):
            if kw in low:
                toks.append(kw)
        return toks or [low.split("—")[0].strip()[:20]]

    def _is_hit(content: str, meta: dict, tokens: list[str]) -> bool:
        blob = ((content or "") + " " + json.dumps(meta or {}, ensure_ascii=False)).lower()
        return any(t in blob for t in tokens)

    out = {}
    for key, cfg in CONFIGS.items():
        recalls, precisions = [], []
        per_q = []
        for item in golden_dataset:
            tokens = _source_tokens(item["expected_context"])
            try:
                chunks = retrieve(item["question"], top_k=DEFAULT_TOP_K,
                                  use_reranking=cfg["use_reranking"])
            except NotImplementedError:
                raise
            hits = [_is_hit(c.get("content", ""), c.get("metadata", {}), tokens) for c in chunks]
            recall = 1.0 if any(hits) else 0.0
            precision = (sum(hits) / len(hits)) if hits else 0.0
            recalls.append(recall)
            precisions.append(precision)
            per_q.append({"id": item.get("id"), "question": item["question"],
                          "context_recall": recall, "context_precision": precision})
        out[key] = {
            "label": cfg["label"],
            "overall": {
                "context_recall": sum(recalls) / len(recalls) if recalls else 0.0,
                "context_precision": sum(precisions) / len(precisions) if precisions else 0.0,
            },
            "per_question": per_q,
        }
    return out


# =============================================================================
# Export Results
# =============================================================================

def _fmt(v, nd=3) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def export_results(ragas_results: dict | None, ab_results: dict, mode: str):
    """Export evaluation results to results.md (theo template có sẵn)."""
    METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

    lines = []
    lines.append("# RAG Evaluation Results — University Services RAG\n")
    lines.append("## Framework sử dụng\n")
    lines.append("> **RAGAS** (`pip install ragas`) — 4 metric LLM-judge: faithfulness, "
                 "answer_relevancy, context_recall, context_precision.\n")
    lines.append(f"> **Golden dataset:** `golden_dataset.json` — {len(load_golden_dataset())} cặp Q&A "
                 "(tuition_fees 8 · scholarships 7 · accommodation 2 · library 2 · wellbeing 1).\n")
    lines.append(f"> **Chế độ chạy:** `{mode}`\n")
    lines.append("---\n")

    # ---- Overall Scores (A/B) ----
    lines.append("## Overall Scores\n")
    lines.append("| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ (A−B) |")
    lines.append("|--------|---------------------------|----------------------|---------|")
    cfgA = ab_results.get("A_hybrid_rerank", {}) or {}
    cfgB = ab_results.get("B_dense_only", {}) or {}
    oA, oB = cfgA.get("overall", {}), cfgB.get("overall", {})
    for m in METRICS:
        a, b = oA.get(m), oB.get(m)
        delta = (a - b) if (a is not None and b is not None) else None
        lines.append(f"| {m} | {_fmt(a)} | {_fmt(b)} | {_fmt(delta, 3)} |")
    # average trên các metric có giá trị
    def _avg(o):
        vals = [o.get(m) for m in METRICS if o.get(m) is not None]
        return sum(vals) / len(vals) if vals else None
    lines.append(f"| **Average** | **{_fmt(_avg(oA))}** | **{_fmt(_avg(oB))}** | **{_fmt((_avg(oA) - _avg(oB)) if (_avg(oA) and _avg(oB)) else None)}** |")
    lines.append("\n> ⚠ Nếu cột faithfulness/answer_relevancy = `—`: chế độ `offline` chỉ đo "
                 "retrieval (context_recall/precision bằng string-match, KHÔNG dùng LLM). "
                 "Metric LLM-judge cần `EVAL_MODEL` + API key thật (xem cuối file).\n")
    lines.append("---\n")

    # ---- A/B Analysis ----
    lines.append("## A/B Comparison Analysis\n")
    lines.append("**Config A — hybrid search + RRF rerank:**")
    lines.append("> Semantic (paraphrase-multilingual-MiniLM-L12-v2) + BM25 lexical, "
                 "fuse bằng Reciprocal Rank Fusion. Mong đợi recall cao nhờ kết hợp "
                 "keyword + ngữ nghĩa.\n")
    lines.append("**Config B — dense-only (no rerank):**")
    lines.append("> Chỉ semantic search, không có lexical signal và không rerank. "
                 "Test xem rerank + lexical đóng góp bao nhiêu.\n")
    lines.append("**Kết luận:**")
    crA, crB = oA.get("context_recall"), oB.get("context_recall")
    if crA is not None and crB is not None:
        if crA >= crB:
            lines.append(f"> Config A ngang/trên Config B về context_recall "
                         f"({_fmt(crA)} vs {_fmt(crB)}) → hybrid + rerank giúp retriever "
                         "bắt đủ evidence hơn, đặc biệt với câu hỏi chứa thuật ngữ/số liệu "
                         "(học phí, GPA, hạn chót) mà BM25 bắt tốt.\n")
        else:
            lines.append(f"> Config B ({_fmt(crB)}) lại tốt hơn A ({_fmt(crA)}) về recall — "
                         "có thể rerank đang lọc mất chunk chứa đáp án, cần calibrate lại "
                         "threshold/alpha.\n")
    else:
        lines.append("> (chưa có số liệu để kết luận — chạy lại sau khi có pipeline/key)\n")
    lines.append("---\n")

    # ---- Per-question breakdown (Config A) ----
    lines.append("## Per-question Breakdown (Config A)\n")
    lines.append("| ID | Question (rút gọn) | Faith | Relev | Recall | Precision |")
    lines.append("|----|--------------------|-------|-------|--------|-----------|")
    for r in (cfgA.get("per_question", []) or []):
        q = (r.get("question") or "")[:42]
        lines.append(
            f"| {r.get('id','')} | {q}… | {_fmt(r.get('faithfulness'),2)} | "
            f"{_fmt(r.get('answer_relevancy'),2)} | {_fmt(r.get('context_recall'),2)} | "
            f"{_fmt(r.get('context_precision'),2)} |"
        )
    lines.append("---\n")

    # ---- Worst performers ----
    lines.append("## Worst Performers (Bottom 3)\n")
    lines.append("| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |")
    lines.append("|---|----------|-------------|-----------|--------|---------------|------------|")
    # xếp theo trung bình các metric (thấp nhất = tệ nhất); fallback per_question rỗng
    per = cfgA.get("per_question", []) or []

    def _mean_score(r):
        vals = [r.get(m) for m in METRICS if r.get(m) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    ranked = sorted(per, key=_mean_score)
    worst = ranked[:3]
    if not worst:
        for i in range(1, 4):
            lines.append(f"| {i} | — | — | — | — | — | — |")
    else:
        for i, r in enumerate(worst, 1):
            recall = r.get("context_recall")
            faith = r.get("faithfulness")
            # Suy luận failure stage: recall thấp -> retrieval; recall OK nhưng faithfulness thấp -> generation
            if recall is not None and recall < 0.6:
                stage, cause = "retrieval", "Retriever không lấy đủ chunk chứa đáp án (thiếu lexical/hybrid)"
            elif faith is not None and faith < 0.7:
                stage, cause = "generation", "Chunk đúng nhưng answer bịa/thừa thông tin ngoài context"
            elif r.get("answer_relevancy") is not None and r.get("answer_relevancy") < 0.6:
                stage, cause = "generation", "Answer lạc đề/không trực tiếp trả lời câu hỏi"
            else:
                stage, cause = "mixed", "Nhiều nguyên nhân — cần phân tích thủ công"
            lines.append(
                f"| {i} | {r.get('question','')[:55]}… | {_fmt(faith)} | "
                f"{_fmt(r.get('answer_relevancy'))} | {_fmt(recall)} | "
                f"{stage} | {cause} |"
            )
    lines.append("---\n")

    # ---- Recommendations ----
    lines.append("## Recommendations\n")
    lines.append("### Cải tiến 1 — Mở rộng expected_context thành nhiều chunk tham chiếu")
    lines.append("**Action:** Golden dataset hiện mô tả nguồn bằng 1 chuỗi (tên file + mục). "
                 "Nên lưu `reference_contexts` là list các đoạn văn thật từ standardized markdown "
                 "để context_recall RAGAS đo chính xác hơn.")
    lines.append("**Expected impact:** context_recall phản ánh đúng chất lượng retriever, "
                 "tránh false negative khi expected_context chỉ là nhãn nguồn.\n")
    lines.append("### Cải tiến 2 — Calibrate SCORE_THRESHOLD cho fallback")
    lines.append("**Action:** Đo khoảng cosine của semantic_search cho câu liên quan vs lạc đề, "
                 "chọn ngưỡng ở giữa (theo cảnh báo trong task9). Đặc biệt cho câu số liệu (học phí).")
    lines.append("**Expected impact:** giảm trường hợp trả kết quả rác thay vì fallback PageIndex; "
                 "tăng faithfulness.\n")
    lines.append("### Cải tiến 3 — Reorder context + prompt citation chặt hơn")
    lines.append("**Action:** áp dụng `reorder_for_llm` (tránh lost in the middle) và buộc LLM "
                 "cite [source] sau mỗi khẳng định; thêm guardrail 'không đủ evidence → từ chối'.")
    lines.append("**Expected impact:** tăng faithfulness + answer_relevancy, giảm bịa đặt.\n")
    lines.append("---\n")

    # ---- Reproduction ----
    lines.append("## Cách chạy lại (Reproduction)\n")
    lines.append("```bash")
    lines.append("# 1) Offline (chỉ retrieval, không cần LLM — số context_recall/precision):")
    lines.append("RAG_USE_HYDE=false python -m group_project.evaluation.eval_pipeline --offline")
    lines.append("")
    lines.append("# 2) Full RAGAS (4 metric LLM-judge + A/B) — cần API key thật trong .env:")
    lines.append("RAG_USE_HYDE=false EVAL_SUBSET=20 python -m group_project.evaluation.eval_pipeline")
    lines.append("#    EVAL_SUBSET=0 (hoặc bỏ) để chạy hết 20 câu.")
    lines.append("```\n")
    lines.append("> **Ghi chú môi trường (đã xử lý):** `ragas==0.4.3` mặc định import fail vì "
                 "`langchain_community 0.4.2` đã bỏ `chat_models.vertexai` / `llms.VertexAI`. "
                 "Đã patch `ragas/llms/base.py` bọc 2 import đó trong try/except + stub class "
                 "(chỉ dùng cho isinstance check, không ảnh hưởng judge). "
                 "Embedding dùng `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, đã cache) thay cho "
                 "bge-m3 (~2.4GB) để chạy benchmark nhanh trong môi trường lab.\n")

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ Đã ghi báo cáo: {RESULTS_PATH}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation (RAGAS)")
    parser.add_argument("--offline", action="store_true",
                        help="Chỉ đánh giá retrieval (context recall/precision), KHÔNG cần LLM.")
    args = parser.parse_args()

    golden = load_golden_dataset()
    print(f"✓ Loaded {len(golden)} golden Q&A từ {GOLDEN_DATASET_PATH.name}")

    if args.offline:
        print("\n=== Offline Retrieval Evaluation (no LLM) ===")
        try:
            ab = evaluate_retrieval_offline(golden)
        except NotImplementedError:
            print("\n❌ Chưa chạy được: src/task9_retrieval_pipeline.retrieve đang "
                  "NotImplementedError.\n   → Hoàn thành task9 (retrieval pipeline) "
                  "rồi chạy lại lệnh này.")
            sys.exit(2)
        export_results(ragas_results=None, ab_results=ab, mode="offline (retrieval-only, no LLM)")
        return

    print("\n=== Full RAGAS Evaluation (4 LLM-judge metrics + A/B) ===")
    try:
        ab = compare_configs(golden_dataset=golden)
    except NotImplementedError:
        print("\n❌ Chưa chạy được: src/task10_generation.generate_with_citation đang "
              "NotImplementedError.\n   → Hoàn thành task9 + task10 rồi chạy lại.")
        sys.exit(2)
    except RuntimeError as e:
        print(f"\n❌ Không thể chạy RAGAS: {e}")
        print("   → Chạy `python -m group_project.evaluation.eval_pipeline --offline` "
              "để lấy số retrieval trước, hoặc thiết lập API key + implement pipeline.")
        sys.exit(2)

    ragas_single = ab.get("A_hybrid_rerank")
    export_results(ragas_results=ragas_single, ab_results=ab,
                   mode="full RAGAS (LLM-judge + A/B)")


if __name__ == "__main__":
    main()
