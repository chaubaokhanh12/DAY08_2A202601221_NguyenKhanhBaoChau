"""
RAG Evaluation Pipeline — RAGAS Framework.

Đánh giá chất lượng RAG pipeline sử dụng RAGAS (Retrieval Augmented Generation
Assessment) — chuẩn industry cho RAG evaluation.

Framework: RAGAS v0.1.x
Metrics:
    - Faithfulness: câu trả lời có bám đúng context không?
    - Answer Relevancy: câu trả lời có đúng câu hỏi không?
    - Context Recall: retriever có lấy đủ evidence không?
    - Context Precision: context lấy về có bao nhiêu % thực sự hữu ích?

So sánh A/B:
    - Config A: Hybrid search + Reranking (full pipeline)
    - Config B: Dense-only search (không reranking, không lexical)

Cài đặt:
    pip install ragas==0.1.21 datasets langchain-openai

Lưu ý rate limit:
    RAGAS gọi LLM NHIỀU LẦN cho mỗi câu hỏi (không phải 1 lần/câu hỏi mà
    nhiều lần/metric/câu hỏi). Model free OpenRouter giới hạn 50 req/ngày.
    Nếu bị rate limit, giảm xuống subset 5-10 câu hoặc nạp $10 credit.

Chạy:
    cd <project_root>
    python -m group_project.evaluation.eval_pipeline
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root → sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


# =============================================================================
# GOLDEN DATASET
# =============================================================================

def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# RAG PIPELINE WRAPPER
# =============================================================================

def run_rag_pipeline(question: str, use_reranking: bool = True) -> dict:
    """
    Chạy RAG pipeline cho 1 câu hỏi.

    Args:
        question: Câu hỏi
        use_reranking: Có dùng reranking hay không (cho A/B test)

    Returns:
        {'answer': str, 'sources': list[dict], 'retrieval_source': str}
    """
    from src.task9_retrieval_pipeline import retrieve
    from src.task10_generation import (
        reorder_for_llm,
        format_context,
        SYSTEM_PROMPT,
        LLM_MODEL,
        TEMPERATURE,
        TOP_P,
    )

    # Step 1: Retrieve
    chunks = retrieve(question, top_k=5, use_reranking=use_reranking)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    # Step 2: Reorder + Format context
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    # Step 3: Build prompt + call LLM
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {question}"

    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else "https://api.openai.com/v1"

    client = OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
    }


def run_dense_only_pipeline(question: str) -> dict:
    """
    RAG pipeline chỉ dùng dense search (Config B cho A/B test).
    Không dùng lexical search, không reranking.
    """
    from src.task5_semantic_search import semantic_search
    from src.task10_generation import (
        reorder_for_llm,
        format_context,
        SYSTEM_PROMPT,
        LLM_MODEL,
        TEMPERATURE,
        TOP_P,
    )

    # Dense search only
    chunks = semantic_search(question, top_k=5)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    # Reorder + Format
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {question}"

    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else "https://api.openai.com/v1"

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": "dense_only",
    }


# =============================================================================
# RAGAS EVALUATION
# =============================================================================

def evaluate_with_ragas(
    pipeline_fn,
    golden_dataset: list[dict],
    config_name: str = "default",
) -> dict:
    """
    Evaluate RAG pipeline sử dụng RAGAS v0.1.x.

    RAGAS metrics:
    - faithfulness: Câu trả lời có faithful (trung thành) với context?
      Đo bằng cách decompose answer thành claims → kiểm tra từng claim có
      supported bởi context không.
    - answer_relevancy: Câu trả lời có relevant với câu hỏi?
      Đo bằng cách generate questions từ answer → so cosine similarity với
      câu hỏi gốc.
    - context_recall: Retriever có lấy được đủ evidence?
      So sánh ground_truth với contexts → bao nhiêu % ground_truth được cover.
    - context_precision: Contexts lấy về có bao nhiêu % thực sự useful?
      Đo bằng cách kiểm tra mỗi context chunk có chứa relevant info không.

    Args:
        pipeline_fn: Callable(question) → {'answer': str, 'sources': list}
        golden_dataset: List of {'question', 'expected_answer', 'expected_context'}
        config_name: Tên config (để log)

    Returns:
        Dict with per-question scores and aggregate scores.
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from datasets import Dataset

    print(f"\n{'='*60}")
    print(f"RAGAS Evaluation — Config: {config_name}")
    print(f"{'='*60}")

    # Prepare evaluation data
    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for i, item in enumerate(golden_dataset, 1):
        print(f"  [{i}/{len(golden_dataset)}] Processing: {item['question'][:50]}...")

        try:
            result = pipeline_fn(item["question"])
            answer = result.get("answer", "")
            contexts = [c["content"] for c in result.get("sources", [])]
        except Exception as e:
            print(f"    ⚠ Pipeline error: {e}")
            answer = "Error generating answer"
            contexts = []

        eval_data["question"].append(item["question"])
        eval_data["answer"].append(answer)
        eval_data["contexts"].append(contexts if contexts else ["No context retrieved"])
        eval_data["ground_truth"].append(item["expected_answer"])

    # Create HuggingFace Dataset
    dataset = Dataset.from_dict(eval_data)

    # Run RAGAS evaluation
    print(f"\n  ⏳ Running RAGAS metrics (this may take a while)...")

    # Configure LLM for RAGAS — use OpenRouter or OpenAI
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if os.getenv("OPENROUTER_API_KEY"):
        llm = ChatOpenAI(
            model="openai/gpt-4o-mini",
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
        )
    else:
        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm,
    )

    # Convert to pandas for analysis
    df = result.to_pandas()

    # Aggregate scores
    aggregate = {
        "faithfulness": float(df["faithfulness"].mean()),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
        "context_recall": float(df["context_recall"].mean()),
        "context_precision": float(df["context_precision"].mean()),
    }
    aggregate["average"] = sum(aggregate.values()) / len(aggregate)

    print(f"\n  ✓ Evaluation complete!")
    print(f"  Aggregate scores:")
    for metric, score in aggregate.items():
        print(f"    {metric}: {score:.4f}")

    return {
        "config_name": config_name,
        "aggregate": aggregate,
        "per_question": df.to_dict("records"),
        "raw_dataframe": df,
    }


# =============================================================================
# A/B COMPARISON
# =============================================================================

def compare_configs(golden_dataset: list[dict]) -> dict:
    """
    So sánh A/B giữa 2 configs:
    - Config A: Hybrid search + Reranking (full pipeline)
    - Config B: Dense-only search (không reranking, không lexical)
    """
    print("\n" + "=" * 60)
    print("A/B COMPARISON")
    print("=" * 60)

    # Config A: Full pipeline (hybrid + rerank)
    print("\n>>> Config A: Hybrid + Reranking")
    config_a = evaluate_with_ragas(
        pipeline_fn=lambda q: run_rag_pipeline(q, use_reranking=True),
        golden_dataset=golden_dataset,
        config_name="Hybrid + Reranking",
    )

    # Config B: Dense-only (no lexical, no reranking)
    print("\n>>> Config B: Dense-only")
    config_b = evaluate_with_ragas(
        pipeline_fn=run_dense_only_pipeline,
        golden_dataset=golden_dataset,
        config_name="Dense-only",
    )

    return {"config_a": config_a, "config_b": config_b}


# =============================================================================
# EXPORT RESULTS
# =============================================================================

def export_results(comparison: dict):
    """Export evaluation results to results.md với phân tích chi tiết."""

    config_a = comparison["config_a"]
    config_b = comparison["config_b"]
    agg_a = config_a["aggregate"]
    agg_b = config_b["aggregate"]

    # Tính delta
    def delta(a_val, b_val):
        d = a_val - b_val
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.4f}"

    content = f"""# RAG Evaluation Results

> **Ngày chạy:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
> **Framework:** RAGAS v0.1.21
> **Golden Dataset:** {len(config_a['per_question'])} câu hỏi
> **LLM Judge:** OpenAI GPT-4o-mini (via OpenRouter)

---

## Framework sử dụng

**RAGAS** (Retrieval Augmented Generation Assessment) — chuẩn industry cho RAG evaluation.

RAGAS đánh giá RAG pipeline trên 4 trục chính:
- **Faithfulness**: Câu trả lời có trung thành với context? (decompose answer → verify claims)
- **Answer Relevancy**: Câu trả lời có đúng câu hỏi? (generate questions from answer → compare)
- **Context Recall**: Retriever có lấy đủ evidence? (ground_truth vs contexts)
- **Context Precision**: Context lấy về có bao nhiêu % hữu ích? (useful chunks / total chunks)

---

## Overall Scores

| Metric | Config A (Hybrid + Rerank) | Config B (Dense-only) | Δ (A-B) |
|--------|:--------------------------:|:---------------------:|:-------:|
| Faithfulness | {agg_a['faithfulness']:.4f} | {agg_b['faithfulness']:.4f} | {delta(agg_a['faithfulness'], agg_b['faithfulness'])} |
| Answer Relevancy | {agg_a['answer_relevancy']:.4f} | {agg_b['answer_relevancy']:.4f} | {delta(agg_a['answer_relevancy'], agg_b['answer_relevancy'])} |
| Context Recall | {agg_a['context_recall']:.4f} | {agg_b['context_recall']:.4f} | {delta(agg_a['context_recall'], agg_b['context_recall'])} |
| Context Precision | {agg_a['context_precision']:.4f} | {agg_b['context_precision']:.4f} | {delta(agg_a['context_precision'], agg_b['context_precision'])} |
| **Average** | **{agg_a['average']:.4f}** | **{agg_b['average']:.4f}** | **{delta(agg_a['average'], agg_b['average'])}** |

---

## A/B Comparison Analysis

**Config A: Hybrid Search + Reranking**
> Full pipeline: Semantic Search (BAAI/bge-m3) + Lexical Search (BM25) → RRF merge → Reranking → Generation.
> Ưu điểm: kết hợp dense + sparse retrieval, reranking cải thiện thứ tự kết quả.

**Config B: Dense-only Search**
> Chỉ dùng Semantic Search (BAAI/bge-m3) → Generation (không reranking, không lexical).
> Ưu điểm: đơn giản, nhanh hơn. Nhược điểm: bỏ lỡ keyword matches từ BM25.

"""

    # Determine winner
    if agg_a["average"] > agg_b["average"]:
        winner = "Config A (Hybrid + Reranking)"
        diff = agg_a["average"] - agg_b["average"]
        content += f"""**Kết luận:**
> **{winner}** cho kết quả tốt hơn với điểm trung bình cao hơn {diff:.4f} điểm.
> Hybrid search + reranking giúp retriever lấy được context đa dạng hơn (kết hợp semantic matching
> và keyword matching), và reranking cải thiện thứ tự kết quả — dẫn đến câu trả lời chính xác hơn.

"""
    elif agg_b["average"] > agg_a["average"]:
        winner = "Config B (Dense-only)"
        diff = agg_b["average"] - agg_a["average"]
        content += f"""**Kết luận:**
> **{winner}** cho kết quả tốt hơn với điểm trung bình cao hơn {diff:.4f} điểm.
> Dense-only search đơn giản nhưng hiệu quả cho corpus này, cho thấy BAAI/bge-m3
> đã capture semantic meaning đủ tốt mà không cần kết hợp thêm lexical search.

"""
    else:
        content += """**Kết luận:**
> Hai config cho kết quả tương đương nhau. Hybrid search thêm độ phức tạp nhưng
> không cải thiện đáng kể cho corpus size nhỏ này.

"""

    # Per-question scores table
    content += """---

## Per-Question Scores (Config A)

| # | Question | Faith. | Relev. | Recall | Prec. |
|---|----------|:------:|:------:|:------:|:-----:|
"""
    for i, row in enumerate(config_a["per_question"], 1):
        q = row.get("question", "")[:50]
        f_score = row.get("faithfulness", 0)
        r_score = row.get("answer_relevancy", 0)
        rc_score = row.get("context_recall", 0)
        cp_score = row.get("context_precision", 0)
        content += f"| {i} | {q}... | {f_score:.2f} | {r_score:.2f} | {rc_score:.2f} | {cp_score:.2f} |\n"

    # Worst performers analysis
    content += """
---

## Worst Performers (Bottom 3)

"""
    # Find worst 3 by average score
    scored_questions = []
    for i, row in enumerate(config_a["per_question"]):
        avg = (
            row.get("faithfulness", 0)
            + row.get("answer_relevancy", 0)
            + row.get("context_recall", 0)
            + row.get("context_precision", 0)
        ) / 4
        scored_questions.append((i, row, avg))

    worst = sorted(scored_questions, key=lambda x: x[2])[:3]

    content += "| # | Question | Avg Score | Weakest Metric | Root Cause Analysis |\n"
    content += "|---|----------|:---------:|:--------------:|---------------------|\n"

    for rank, (idx, row, avg) in enumerate(worst, 1):
        q = row.get("question", "")[:45]
        # Find weakest metric
        metrics = {
            "Faithfulness": row.get("faithfulness", 0),
            "Relevancy": row.get("answer_relevancy", 0),
            "Recall": row.get("context_recall", 0),
            "Precision": row.get("context_precision", 0),
        }
        weakest = min(metrics, key=metrics.get)
        weakest_val = metrics[weakest]

        if weakest == "Recall":
            cause = "Retriever không tìm được context phù hợp"
        elif weakest == "Faithfulness":
            cause = "LLM hallucinate ngoài context"
        elif weakest == "Relevancy":
            cause = "Câu trả lời lạc đề so với câu hỏi"
        else:
            cause = "Context chứa nhiều noise không liên quan"

        content += f"| {rank} | {q}... | {avg:.2f} | {weakest} ({weakest_val:.2f}) | {cause} |\n"

    # Recommendations
    content += f"""
---

## Recommendations

### Cải tiến 1: Mở rộng corpus
**Action:** Thêm nhiều documents hơn vào corpus, đặc biệt các chủ đề ít được cover (academic policies, student life).
**Expected impact:** Tăng Context Recall lên 10-15% nhờ nhiều evidence hơn cho mỗi câu hỏi.

### Cải tiến 2: Fine-tune chunking strategy
**Action:** Thử MarkdownHeaderTextSplitter cho documents có heading rõ (legal docs), giữ RecursiveCharacterTextSplitter cho news. Tăng chunk_size lên 800 để giữ ngữ cảnh tốt hơn.
**Expected impact:** Tăng Context Precision — mỗi chunk chứa thông tin hoàn chỉnh hơn, giảm noise.

### Cải tiến 3: Prompt engineering cho generation
**Action:** Thêm ví dụ few-shot vào system prompt, yêu cầu LLM trả lời cấu trúc hơn (bullet points), thêm instruction "Nếu không chắc chắn, hãy nói rõ mức độ tin cậy".
**Expected impact:** Tăng Faithfulness 5-10% — giảm hallucination bằng cách ép LLM bám sát context.

### Cải tiến 4: Query expansion
**Action:** Dùng HyDE hoặc query rewriting để mở rộng câu hỏi trước khi retrieval.
**Expected impact:** Tăng Context Recall — query mở rộng match được nhiều documents hơn.

---

## Appendix: Evaluation Setup

| Parameter | Value |
|-----------|-------|
| RAGAS version | 0.1.21 |
| LLM Judge | GPT-4o-mini (via OpenRouter) |
| Embedding | BAAI/bge-m3 (1024 dim) |
| Chunk size | 500 chars, overlap 50 |
| Vector store | ChromaDB (cosine similarity) |
| Retrieval top_k | 5 |
| Generation temperature | 0.3 |
| Golden dataset size | {len(config_a['per_question'])} questions |
"""

    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"\n✓ Results exported to: {RESULTS_PATH}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RAG Evaluation Pipeline (RAGAS)")
    print("=" * 60)

    # Load golden dataset
    golden_dataset = load_golden_dataset()
    print(f"✓ Loaded {len(golden_dataset)} test cases from golden_dataset.json")

    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ Cần API key để chạy evaluation!")
        print("   Set OPENROUTER_API_KEY hoặc OPENAI_API_KEY trong .env")
        sys.exit(1)

    # Run A/B comparison
    comparison = compare_configs(golden_dataset)

    # Export results
    export_results(comparison)

    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE!")
    print(f"   Xem kết quả: {RESULTS_PATH}")
    print("=" * 60)
