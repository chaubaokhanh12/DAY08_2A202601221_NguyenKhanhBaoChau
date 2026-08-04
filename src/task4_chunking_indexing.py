"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho cả tiếng Việt lẫn tiếng Anh)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters openai python-dotenv chromadb
    # text-embedding-3-small là API model — không cần sentence-transformers.
    # Cần OPENAI_API_KEY trong file .env (đã có sẵn).

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # nạp OPENAI_API_KEY từ .env

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chunking strategy: RecursiveCharacterTextSplitter
# Vì sao chọn "recursive":
#   - An toàn và phổ biến nhất, hoạt động tốt với mọi loại văn bản
#   - Tự động tách theo thứ tự ưu tiên: paragraph → sentence → word
#   - Phù hợp cho corpus hỗn hợp (legal docs + news articles) vì không phụ thuộc
#     vào cấu trúc heading cụ thể như MarkdownHeaderTextSplitter
CHUNK_SIZE = 500        # Vì sao chọn 500? Đủ lớn để giữ ngữ cảnh hoàn chỉnh cho
                        # 1 đoạn thông tin (VD: 1 điều khoản học phí), nhưng đủ nhỏ
                        # để embedding model capture được semantic meaning chính xác.
                        # text-embedding-3-small hỗ trợ tối đa 8191 tokens, nhưng chunk
                        # nhỏ hơn cho retrieval chính xác hơn.
CHUNK_OVERLAP = 50      # Vì sao chọn 50? ~10% overlap giúp tránh mất ngữ cảnh tại
                        # ranh giới chunk, đặc biệt khi câu bị cắt giữa chừng.
                        # Không quá lớn để tránh trùng lặp quá nhiều trong index.
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Embedding model: OpenAI text-embedding-3-small
# Vì sao chọn text-embedding-3-small:
#   - API model (không cần tải weights về local): nhanh, nhẹ, không tốn RAM/GPU
#   - 1536 dimensions: chất lượng retrieval tốt, cân bằng với kích thước lưu trữ
#   - Hỗ trợ tốt multilingual (tiếng Việt + tiếng Anh) — đủ cho corpus University
#     Services có cả nội dung tiếng Việt và thuật ngữ tiếng Anh
#   - State-of-the-art trên MTEB benchmark cho embedding kích thước nhỏ giá rẻ
#   - Cần OPENAI_API_KEY trong .env (đã cấu hình sẵn)
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI API embedding
EMBEDDING_DIM = 1536
OPENAI_EMBED_BATCH = 64  # Số input/request — an toàn dưới giới hạn 2048 của API

# Vector store: ChromaDB
# Vì sao chọn ChromaDB:
#   - Đơn giản, local persistent, không cần Docker hay cloud service
#   - Hỗ trợ cosine similarity search built-in
#   - Tích hợp tốt với Python ecosystem
VECTOR_STORE = "chromadb"  # "chromadb" | "weaviate" | "faiss"
COLLECTION_NAME = "university_services_docs"


# =============================================================================
# SINGLETON CACHE — tránh load model/collection nhiều lần
# =============================================================================

_embedding_model = None
_chroma_collection = None


def get_embedding_model():
    """
    Lấy (hoặc khởi tạo) OpenAI embedding client.

    Trả về một wrapper có phương thức `.encode(texts, show_progress_bar=...,
    normalize_embeddings=...)` tương thích giao diện SentenceTransformer,
    để code downstream (Task 5 semantic search) không cần sửa khi đổi model.

    Singleton pattern để tránh khởi tạo client nhiều lần trong cùng process.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = _OpenAIEmbedder(
            model=EMBEDDING_MODEL,
            dim=EMBEDDING_DIM,
            batch_size=OPENAI_EMBED_BATCH,
        )
    return _embedding_model


class _OpenAIEmbedder:
    """Wrapper quanh OpenAI Embeddings API, tương thích `.encode()` của
    SentenceTransformer để dễ hoán đổi model mà không phải sửa code caller.

    `.encode(texts, normalize_embeddings=True)` trả về numpy.ndarray shape
    (n, dim) — cùng kiểu trả về của SentenceTransformer, dtype float32.
    """

    def __init__(self, model: str, dim: int, batch_size: int = 64):
        import os
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY chưa được cấu hình. Hãy tạo file .env với "
                "OPENAI_API_KEY=sk-... (xem .env.example)."
            )
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self._client = OpenAI(api_key=api_key)

    def encode(
        self,
        texts,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        **_unused,
    ):
        import numpy as np

        single = isinstance(texts, str)
        if single:
            texts = [texts]

        all_vecs = []
        total = len(texts)
        for start in range(0, total, self.batch_size):
            batch = texts[start:start + self.batch_size]
            resp = self._client.embeddings.create(model=self.model, input=batch)
            # API trả về list đã sắp xếp theo index, nhưng sort để an toàn.
            for item in sorted(resp.data, key=lambda d: d.index):
                all_vecs.append(item.embedding)
            if show_progress_bar:
                done = min(start + self.batch_size, total)
                print(f"    embedded {done}/{total}", end="\r", flush=True)
        if show_progress_bar:
            print()

        vecs = np.asarray(all_vecs, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # tránh chia 0
            vecs = vecs / norms
        return vecs


def get_collection():
    """
    Lấy (hoặc khởi tạo) ChromaDB collection.
    Singleton pattern để tránh tạo client nhiều lần.
    """
    global _chroma_collection
    if _chroma_collection is None:
        import chromadb

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _chroma_collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # Cosine similarity search
        )
    return _chroma_collection


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        # Xác định loại document dựa trên thư mục cha
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn (RecursiveCharacterTextSplitter).

    Sử dụng RecursiveCharacterTextSplitter vì:
    - Tách theo thứ tự ưu tiên: paragraph (\\n\\n) → line (\\n) → sentence (. ) → word ( )
    - An toàn cho mọi loại văn bản, không phụ thuộc cấu trúc heading
    - Giữ ngữ cảnh tốt nhờ chunk_overlap

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i}
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng OpenAI text-embedding-3-small.

    text-embedding-3-small tạo dense embeddings 1536 chiều, tối ưu cho
    semantic search multilingual. Embed qua API theo batch (64 input/request),
    L2-normalize để cosine similarity trong ChromaDB cho kết quả nhất quán.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    model = get_embedding_model()
    texts = [c["content"] for c in chunks]
    # Encode tất cả chunks cùng lúc để tối ưu batch processing
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào ChromaDB vector store.

    ChromaDB configuration:
    - Persistent storage tại chroma_db/
    - Collection sử dụng cosine similarity (hnsw:space = cosine)
    - Mỗi chunk có unique ID dạng: {source_filename}_chunk_{index}
    """
    collection = get_collection()

    # Tạo unique IDs cho mỗi chunk
    ids = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}" for c in chunks]

    # Upsert để tránh duplicate khi chạy lại
    collection.upsert(
        ids=ids,
        documents=[c["content"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    print(f"  ✓ Indexed {len(chunks)} chunks vào collection '{COLLECTION_NAME}'")
    print(f"  ✓ Total documents in collection: {collection.count()}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    if not docs:
        print("⚠ Không tìm thấy documents trong data/standardized/")
        print("  Hãy chạy Task 1-3 trước để tạo dữ liệu.")
        return

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
