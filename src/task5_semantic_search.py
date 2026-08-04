"""
Task 5 — Semantic Search Module + HyDE (Hypothetical Document Embeddings).

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.
Tích hợp HyDE để cải thiện chất lượng retrieval → +5 điểm bonus.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4

HyDE (Hypothetical Document Embeddings) — Gao et al. 2022:
    Thay vì embed trực tiếp câu hỏi ngắn (thường khác biệt về phong cách so với
    documents trong corpus), HyDE yêu cầu LLM tạo ra một "hypothetical document"
    — tức đoạn văn giả định trả lời câu hỏi đó. Sau đó embed đoạn văn giả định
    này để tìm kiếm. Vì hypothetical document có phong cách gần giống documents
    thật trong corpus hơn, cosine similarity sẽ chính xác hơn.

    Pipeline: Query → LLM generate hypothetical answer → Embed answer → Search

Cài đặt bổ sung:
    pip install openai  # Đã có trong requirements.txt (Task 10)
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task4_chunking_indexing import get_collection, get_embedding_model


# =============================================================================
# HyDE CONFIGURATION
# =============================================================================

# Mặc định bật HyDE. Set USE_HYDE=False nếu không có API key hoặc muốn search nhanh.
USE_HYDE = True

# LLM dùng cho HyDE — dùng model free trên OpenRouter để không tốn phí
HYDE_LLM_MODEL = "openai/gpt-4o-mini"

HYDE_PROMPT_TEMPLATE = """Bạn là chuyên gia về dịch vụ và chính sách đại học.
Hãy viết một đoạn văn ngắn (3-5 câu) trả lời câu hỏi sau, dựa trên kiến thức
chung về chính sách đại học (học phí, học bổng, ký túc xá, thư viện, đăng ký học phần).

Câu hỏi: {query}

Đoạn văn trả lời:"""


# =============================================================================
# HyDE IMPLEMENTATION
# =============================================================================

def generate_hypothetical_document(query: str) -> str:
    """
    Dùng LLM để tạo hypothetical document cho HyDE.

    HyDE (Gao et al., 2022) cải thiện retrieval bằng cách:
    1. Tạo đoạn văn giả định trả lời câu hỏi (không cần chính xác)
    2. Embed đoạn văn này thay vì embed query ngắn
    3. Đoạn văn giả định gần hơn về phong cách với documents thật → retrieval tốt hơn

    Args:
        query: Câu hỏi gốc của user

    Returns:
        Hypothetical document (đoạn văn giả định trả lời câu hỏi)
    """
    from openai import OpenAI

    # Thử OpenRouter trước (có model :free), fallback sang OpenAI
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Không có API key → trả về query gốc, semantic_search vẫn hoạt động bình thường
        return query

    # Xác định base_url dựa trên loại API key
    if os.getenv("OPENROUTER_API_KEY"):
        base_url = "https://openrouter.ai/api/v1"
    else:
        base_url = "https://api.openai.com/v1"

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=HYDE_LLM_MODEL,
            messages=[
                {"role": "user", "content": HYDE_PROMPT_TEMPLATE.format(query=query)}
            ],
            temperature=0.7,   # Cho phép sáng tạo vừa phải để tạo đoạn văn đa dạng
            max_tokens=200,    # Giới hạn độ dài — chỉ cần 3-5 câu
        )
        hypothetical_doc = response.choices[0].message.content.strip()
        return hypothetical_doc
    except Exception as e:
        # Nếu LLM call thất bại → graceful fallback về query gốc
        print(f"  ⚠ HyDE LLM call failed ({e}), falling back to original query")
        return query


# =============================================================================
# CORE SEARCH — dùng nội bộ, không phụ thuộc HyDE
# =============================================================================

def _vector_search(query_text: str, top_k: int = 10) -> list[dict]:
    """
    Core vector search: embed text → query ChromaDB → return results.

    Hàm nội bộ, nhận text bất kỳ (có thể là query gốc hoặc hypothetical document).

    Args:
        query_text: Text để embed và tìm kiếm (query hoặc hypothetical document)
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content', 'score', 'metadata'} sorted by score descending.
    """
    model = get_embedding_model()
    query_vector = model.encode(query_text, normalize_embeddings=True).tolist()

    collection = get_collection()

    # Kiểm tra collection có data không
    if collection.count() == 0:
        return []

    actual_top_k = min(top_k, collection.count())

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=actual_top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Convert cosine distance → similarity score
    # ChromaDB với hnsw:space="cosine" trả về distance = 1 - similarity
    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = max(0.0, 1.0 - dist)
        output.append({
            "content": doc,
            "score": round(score, 4),
            "metadata": meta,
        })

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


# =============================================================================
# PUBLIC API
# =============================================================================

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity (dense retrieval).

    Nếu USE_HYDE=True và có API key, sẽ dùng HyDE để cải thiện retrieval:
        Query → LLM tạo hypothetical document → Embed document → Search

    Nếu USE_HYDE=False hoặc không có API key, fallback về standard search:
        Query → Embed query → Search

    Args:
        query: Câu truy vấn (tiếng Việt hoặc tiếng Anh)
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score [0, 1]
            'metadata': dict     # source, type, chunk_index
        }
        Sorted by score descending.
    """
    if USE_HYDE:
        # HyDE: tạo hypothetical document rồi dùng nó để search
        hypothetical_doc = generate_hypothetical_document(query)
        return _vector_search(hypothetical_doc, top_k=top_k)
    else:
        # Standard: embed query trực tiếp
        return _vector_search(query, top_k=top_k)


if __name__ == "__main__":
    # Test
    print("=" * 60)
    print("Task 5: Semantic Search Test (with HyDE)")
    print(f"  HyDE enabled: {USE_HYDE}")
    print("=" * 60)

    test_queries = [
        "what is the tuition fee",
        "học phí tại RMIT",
        "scholarship eligibility",
        "thư viện đại học",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 50)
        results = semantic_search(q, top_k=5)
        if not results:
            print("  (Không có kết quả — hãy chạy Task 4 index trước)")
        for r in results:
            print(f"  [{r['score']:.3f}] {r['content'][:100]}...")
