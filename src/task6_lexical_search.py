"""
Task 6 — Lexical Search Module (Elasticsearch).

★ BONUS +5 điểm: Sử dụng Elasticsearch thay vì BM25 đơn giản (rank-bm25).

═══════════════════════════════════════════════════════════════════════════════
CƠ CHẾ HOẠT ĐỘNG CỦA ELASTICSEARCH (giải thích cho buổi demo)
═══════════════════════════════════════════════════════════════════════════════

Elasticsearch sử dụng BM25 (Best Match 25) làm scoring mặc định, nhưng khác
biệt lớn so với rank-bm25 (BM25Okapi) ở tầng infrastructure:

1. INVERTED INDEX (Chỉ mục nghịch đảo):
   - Elasticsearch xây dựng inverted index tại thời điểm indexing.
   - Mỗi term → danh sách document IDs chứa term đó.
   - Tra cứu O(1) thay vì scan toàn bộ corpus mỗi lần query.
   - rank-bm25 (Python) phải tính toán lại trên toàn bộ corpus mỗi lần query.

2. TEXT ANALYSIS PIPELINE:
   - Tokenizer: tách text thành tokens (standard, whitespace, ngram, ...)
   - Token filters: lowercase, stop words, stemming, synonyms
   - Character filters: HTML strip, pattern replace
   → rank-bm25 chỉ có `.split()` hoặc tokenizer đơn giản.

3. BM25 SCORING (giống nhau về công thức):
   score(q,d) = Σ IDF(qi) × (tf(qi,d) × (k1+1)) / (tf(qi,d) + k1 × (1-b+b × |d|/avgdl))
   - k1 = 1.2 (term saturation — ES mặc định, rank-bm25 dùng k1=1.5)
   - b = 0.75 (length normalization)
   - TF: term frequency trong document
   - IDF: inverse document frequency — log(1 + (N - n + 0.5) / (n + 0.5))
   - |d|/avgdl: document length / average document length

4. ƯU ĐIỂM SO VỚI rank-bm25:
   - Tốc độ: inverted index → O(1) lookup vs O(N) scan
   - Phân tích text: analyzers tốt hơn → match chính xác hơn
   - Scalable: hỗ trợ distributed search cho corpus lớn
   - Multi-field: có thể boost score theo trường (title, content, metadata)
   - Real-time: index mới và search gần như tức thì

Cài đặt:
    pip install elasticsearch>=8.0.0

Setup Elasticsearch (chọn 1):
    # Docker (đơn giản nhất):
    docker run -d --name elasticsearch -p 9200:9200 \
        -e "discovery.type=single-node" \
        -e "xpack.security.enabled=false" \
        elasticsearch:8.15.0

    # Hoặc Elasticsearch Cloud (free trial): https://cloud.elastic.co/
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ES_API_KEY = os.getenv("ELASTICSEARCH_API_KEY", "")
ES_INDEX = "university_services"

# Singleton cache
_es_client = None
_corpus_cache = None
_index_built = False


# =============================================================================
# ELASTICSEARCH CLIENT
# =============================================================================

def _get_es_client():
    """Lấy (hoặc khởi tạo) Elasticsearch client. Singleton pattern."""
    global _es_client
    if _es_client is None:
        from elasticsearch import Elasticsearch

        if ES_API_KEY:
            _es_client = Elasticsearch(ES_URL, api_key=ES_API_KEY)
        else:
            _es_client = Elasticsearch(ES_URL)

        # Kiểm tra kết nối
        if not _es_client.ping():
            raise ConnectionError(
                f"Không thể kết nối Elasticsearch tại {ES_URL}.\n"
                "Hãy chạy: docker run -d --name elasticsearch -p 9200:9200 "
                "-e 'discovery.type=single-node' "
                "-e 'xpack.security.enabled=false' "
                "elasticsearch:8.15.0"
            )
    return _es_client


def _load_corpus() -> list[dict]:
    """
    Load corpus từ ChromaDB (nhất quán với Task 4).
    Nếu ChromaDB rỗng, fallback load từ file và chunk lại.
    """
    global _corpus_cache
    if _corpus_cache is not None:
        return _corpus_cache

    from .task4_chunking_indexing import get_collection, load_documents, chunk_documents

    # Thử lấy từ ChromaDB trước (đã index ở Task 4)
    collection = get_collection()
    if collection.count() > 0:
        all_data = collection.get(include=["documents", "metadatas"])
        _corpus_cache = [
            {"content": doc, "metadata": meta}
            for doc, meta in zip(all_data["documents"], all_data["metadatas"])
        ]
    else:
        # Fallback: load từ file gốc và chunk lại
        docs = load_documents()
        _corpus_cache = chunk_documents(docs)

    return _corpus_cache


# =============================================================================
# INDEX MANAGEMENT
# =============================================================================

# Elasticsearch index mapping
# - content: text field với standard analyzer (tokenize + lowercase + BM25)
# - source: keyword (exact match, dùng cho filter/aggregation)
# - type: keyword (legal/news)
# - chunk_index: integer
ES_MAPPINGS = {
    "properties": {
        "content": {
            "type": "text",
            "analyzer": "standard",      # Standard analyzer: tokenize → lowercase
        },
        "source": {"type": "keyword"},
        "type": {"type": "keyword"},
        "chunk_index": {"type": "integer"},
    }
}

ES_SETTINGS = {
    "number_of_shards": 1,       # Single node → 1 shard
    "number_of_replicas": 0,     # Local dev → no replicas
    "index": {
        "similarity": {
            "default": {
                "type": "BM25",      # Explicit BM25 scoring
                "k1": 1.2,           # Term saturation
                "b": 0.75,           # Length normalization
            }
        }
    }
}


def build_es_index(force_rebuild: bool = False):
    """
    Xây dựng Elasticsearch index từ corpus.

    Sử dụng bulk API để index nhanh. Index dùng BM25 scoring mặc định.

    Args:
        force_rebuild: Nếu True, xóa index cũ và tạo lại
    """
    global _index_built
    from elasticsearch.helpers import bulk

    es = _get_es_client()
    corpus = _load_corpus()

    if not corpus:
        print("⚠ Corpus rỗng — hãy chạy Task 1-4 trước.")
        return

    # Xóa index cũ nếu force rebuild hoặc index chưa tồn tại
    if es.indices.exists(index=ES_INDEX):
        if force_rebuild:
            es.indices.delete(index=ES_INDEX)
            print(f"  ✓ Đã xóa index cũ: {ES_INDEX}")
        else:
            _index_built = True
            return  # Index đã có, không cần rebuild

    # Tạo index mới với mappings và settings
    es.indices.create(
        index=ES_INDEX,
        mappings=ES_MAPPINGS,
        settings=ES_SETTINGS,
    )
    print(f"  ✓ Tạo index: {ES_INDEX}")

    # Bulk index toàn bộ corpus
    actions = []
    for i, doc in enumerate(corpus):
        meta = doc.get("metadata", {})
        actions.append({
            "_index": ES_INDEX,
            "_id": str(i),
            "_source": {
                "content": doc["content"],
                "source": meta.get("source", ""),
                "type": meta.get("type", "unknown"),
                "chunk_index": meta.get("chunk_index", i),
            },
        })

    success_count, errors = bulk(es, actions, raise_on_error=False)
    print(f"  ✓ Indexed {success_count}/{len(actions)} documents")

    if errors:
        print(f"  ⚠ {len(errors)} lỗi khi indexing")

    # Refresh index để documents có thể search ngay
    es.indices.refresh(index=ES_INDEX)
    _index_built = True


def _ensure_index():
    """Đảm bảo Elasticsearch index đã được build. Lazy initialization."""
    global _index_built
    if not _index_built:
        es = _get_es_client()
        if es.indices.exists(index=ES_INDEX):
            _index_built = True
        else:
            build_es_index()


# =============================================================================
# LEXICAL SEARCH
# =============================================================================

def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng Elasticsearch BM25.

    Elasticsearch thực hiện:
    1. Tokenize query bằng standard analyzer (lowercase + split)
    2. Tra cứu inverted index để tìm documents chứa các terms
    3. Tính BM25 score cho mỗi document khớp
    4. Sắp xếp theo score giảm dần

    So với rank-bm25 (Python BM25Okapi):
    - Cùng công thức BM25 nhưng Elasticsearch nhanh hơn nhờ inverted index
    - Text analysis pipeline tốt hơn (standard analyzer vs simple split)
    - Hỗ trợ multi-field search và field boosting

    Args:
        query: Câu truy vấn (tiếng Việt hoặc tiếng Anh)
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # BM25 score (Elasticsearch)
            'metadata': dict     # source, type, chunk_index
        }
        Sorted by score descending (Elasticsearch trả sẵn sorted).
    """
    # Đảm bảo index đã build
    _ensure_index()

    es = _get_es_client()

    # Multi-match query: tìm kiếm trên field content
    # Operator "or": chỉ cần match 1 term → trả về kết quả
    # Fuzziness "auto": cho phép typo nhẹ (edit distance 1-2)
    search_body = {
        "query": {
            "match": {
                "content": {
                    "query": query,
                    "operator": "or",
                    "fuzziness": "auto",
                }
            }
        },
        "size": top_k,
    }

    response = es.search(index=ES_INDEX, body=search_body)

    # Parse kết quả
    output = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        output.append({
            "content": source["content"],
            "score": float(hit["_score"]),
            "metadata": {
                "source": source.get("source", ""),
                "type": source.get("type", "unknown"),
                "chunk_index": source.get("chunk_index", 0),
            },
        })

    # Elasticsearch đã sort by score descending, nhưng đảm bảo chắc chắn
    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Task 6: Lexical Search (Elasticsearch BM25)")
    print(f"  ES URL: {ES_URL}")
    print(f"  ES Index: {ES_INDEX}")
    print("=" * 60)

    # Build index
    print("\n--- Building Elasticsearch Index ---")
    build_es_index(force_rebuild=True)

    # Test queries
    test_queries = [
        "tuition fee payment methods",
        "học phí RMIT",
        "scholarship eligibility requirements",
        "library study room booking",
        "ký túc xá sinh viên",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 50)
        results = lexical_search(q, top_k=5)
        if not results:
            print("  (Không có kết quả)")
        for r in results:
            print(f"  [{r['score']:.3f}] {r['content'][:100]}...")
