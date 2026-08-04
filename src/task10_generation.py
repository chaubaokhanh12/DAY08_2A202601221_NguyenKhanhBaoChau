"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"

Gợi ý LLM: OpenRouter có nhiều model gắn hậu tố ":free" không tính phí — xem
https://openrouter.ai/models?max_price=0 — phù hợp nếu chưa có credit trả phí.
Base URL: "https://openrouter.ai/api/v1", dùng chung interface với OpenAI SDK.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

# TODO: Chọn LLM model (OpenRouter model ID)
LLM_MODEL = "openai/gpt-4o-mini"  # hoặc model ":free" nếu chưa có credit


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về dịch vụ và chính sách đại học
(học phí, học bổng, ký túc xá, thư viện, đăng ký học phần).

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt
2. Mỗi khẳng định phải có trích dẫn ngay sau, ví dụ: [Tuition Fees, 2026]
3. Nếu context không đủ thông tin → trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có"
4. Trả lời bằng tiếng Việt, có cấu trúc rõ ràng theo đoạn văn
5. Không suy luận hay mở rộng ngoài những gì được nêu trong context"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)
    """
    if len(chunks) <= 2:
        return chunks
    front = chunks[::2]   # index 0, 2, 4 -> đặt ở đầu
    back = chunks[1::2]   # index 1, 3    -> đặt ở cuối (reversed)
    return front + back[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Source {i}")
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# LLM CLIENT RESOLVER (provider-aware)
# =============================================================================

def _resolve_llm():
    """
    Trả về (client, model) dựa trên API key có trong env.

    Tự nhận diện provider theo tiền tố key:
      - 'sk-or...' (không phải placeholder) -> OpenRouter
      - 'sk-...'  (sk-proj / sk-...)        -> OpenAI trực tiếp
    Giúp chạy được cả khi OpenAI key vô tình đặt dưới biến OPENROUTER_API_KEY.
    """
    from openai import OpenAI
    candidates = [os.getenv("OPENROUTER_API_KEY"), os.getenv("OPENAI_API_KEY")]
    # Ưu tiên OpenAI direct key
    for key in candidates:
        if key and key.startswith("sk-") and not key.startswith("sk-or"):
            return OpenAI(api_key=key, base_url="https://api.openai.com/v1"), "gpt-4o-mini"
    # OpenRouter key
    for key in candidates:
        if key and key.startswith("sk-or") and key.strip() != "sk-or-v1-...":
            return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1"), LLM_MODEL
    raise RuntimeError(
        "Không tìm thấy LLM API key hợp lệ. Set OPENROUTER_API_KEY hoặc OPENAI_API_KEY trong .env."
    )


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K, config: dict | None = None) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user
        top_k: Số chunks đưa vào context
        config: {"use_reranking": bool} để bật/tắt hybrid (dùng cho A/B eval)

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' | 'dense' | 'pageindex' | 'none'
        }
    """
    # Step 1: Retrieve (truyền use_reranking từ config nếu có — phục vụ A/B eval)
    use_reranking = True if config is None else bool(config.get("use_reranking", True))
    chunks = retrieve(query, top_k=top_k, use_reranking=use_reranking)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    # Step 2: Reorder
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    # Step 5: Call LLM
    client, model = _resolve_llm()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    answer = response.choices[0].message.content

    # Step 6: Return
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
    }


if __name__ == "__main__":
    test_queries = [
        "Học phí tại RMIT Vietnam là bao nhiêu?",
        "Làm sao để đặt phòng học nhóm ở thư viện?",
        "Sinh viên quốc tế có những học bổng nào?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
