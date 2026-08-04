# RAG Evaluation Results

> **Ngày chạy:** 2026-08-04 17:37
> **Framework:** RAGAS v0.1.21
> **Golden Dataset:** 20 câu hỏi
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
| Faithfulness | 0.2583 | 0.2400 | +0.0183 |
| Answer Relevancy | 0.1678 | 0.1614 | +0.0064 |
| Context Recall | 0.6333 | 0.7833 | -0.1500 |
| Context Precision | 0.9053 | 0.9287 | -0.0233 |
| **Average** | **0.4912** | **0.5284** | **-0.0372** |

---

## A/B Comparison Analysis

**Config A: Hybrid Search + Reranking**
> Full pipeline: Semantic Search (BAAI/bge-m3) + Lexical Search (BM25) → RRF merge → Reranking → Generation.
> Ưu điểm: kết hợp dense + sparse retrieval, reranking cải thiện thứ tự kết quả.

**Config B: Dense-only Search**
> Chỉ dùng Semantic Search (BAAI/bge-m3) → Generation (không reranking, không lexical).
> Ưu điểm: đơn giản, nhanh hơn. Nhược điểm: bỏ lỡ keyword matches từ BM25.

**Kết luận:**
> **Config B (Dense-only)** cho kết quả tốt hơn với điểm trung bình cao hơn 0.0372 điểm.
> Dense-only search đơn giản nhưng hiệu quả cho corpus này, cho thấy BAAI/bge-m3
> đã capture semantic meaning đủ tốt mà không cần kết hợp thêm lexical search.

---

## Per-Question Scores (Config A)

| # | Question | Faith. | Relev. | Recall | Prec. |
|---|----------|:------:|:------:|:------:|:-----:|
| 1 | Học phí hàng năm của chương trình Business tại RMI... | 0.00 | 0.00 | 0.00 | 1.00 |
| 2 | Học phí được thanh toán theo hình thức nào?... | 0.67 | 0.00 | 1.00 | 1.00 |
| 3 | Trường có cung cấp ký túc xá trong khuôn viên khôn... | 0.00 | 0.00 | 0.00 | 0.70 |
| 4 | Điều kiện để xin học bổng Academic Achievement Sch... | 0.00 | 0.00 | 0.50 | 0.89 |
| 5 | Sinh viên quốc tế có những loại học bổng nào tại R... | 0.00 | 0.00 | 0.00 | 0.59 |
| 6 | Làm sao để đặt phòng học nhóm ở thư viện RMIT?... | 1.00 | 0.69 | 1.00 | 0.95 |
| 7 | Thư viện RMIT mở cửa vào những giờ nào?... | 0.00 | 0.00 | 0.00 | 0.81 |
| 8 | Cách đăng ký học phần qua myRMIT như thế nào?... | 0.00 | 0.00 | 1.00 | 1.00 |
| 9 | Thời hạn đăng ký học phần cho học kỳ tới là khi nà... | 0.00 | 0.00 | 1.00 | 1.00 |
| 10 | RMIT Vietnam có những chương trình hỗ trợ sức khỏe... | 0.00 | 0.00 | 1.00 | 1.00 |
| 11 | Dịch vụ hỗ trợ chỗ ở cho sinh viên hoạt động như t... | 0.00 | 0.00 | 0.00 | 1.00 |
| 12 | Sinh viên có thể mượn bao nhiêu tài liệu cùng lúc ... | 0.00 | 0.00 | 1.00 | 1.00 |
| 13 | Có những sự kiện nào dành cho sinh viên mới nhập h... | 1.00 | 0.61 | 1.00 | 1.00 |
| 14 | Chính sách hoàn học phí khi rút môn trễ tại RMIT V... | 0.50 | 0.74 | 1.00 | 0.80 |
| 15 | RMIT Vietnam có hỗ trợ tài chính dạng trả góp học ... | 0.00 | 0.00 | 1.00 | 0.95 |
| 16 | Thư viện RMIT có cung cấp quyền truy cập cơ sở dữ ... | 1.00 | 0.62 | 1.00 | 0.89 |
| 17 | Quy trình xin chuyển ngành tại RMIT Vietnam gồm nh... | 0.00 | 0.00 | 1.00 | 1.00 |
| 18 | RMIT Vietnam có câu lạc bộ sinh viên nào đáng tham... | 0.00 | 0.00 | 0.00 | 0.53 |
| 19 | Làm thế nào để xin giấy xác nhận sinh viên tại RMI... | 1.00 | 0.70 | 0.50 | 1.00 |
| 20 | Chính sách nghỉ phép học tập (leave of absence) tạ... | 0.00 | 0.00 | 0.67 | 1.00 |

---

## Worst Performers (Bottom 3)

| # | Question | Avg Score | Weakest Metric | Root Cause Analysis |
|---|----------|:---------:|:--------------:|---------------------|
| 1 | RMIT Vietnam có câu lạc bộ sinh viên nào đáng... | 0.13 | Faithfulness (0.00) | LLM hallucinate ngoài context |
| 2 | Sinh viên quốc tế có những loại học bổng nào ... | 0.15 | Faithfulness (0.00) | LLM hallucinate ngoài context |
| 3 | Trường có cung cấp ký túc xá trong khuôn viên... | 0.17 | Faithfulness (0.00) | LLM hallucinate ngoài context |

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
| Golden dataset size | 20 questions |
