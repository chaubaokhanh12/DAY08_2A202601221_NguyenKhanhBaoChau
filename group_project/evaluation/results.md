# RAG Evaluation Results — University Services RAG

## Framework sử dụng

> **RAGAS** (`pip install ragas`) — 4 metric LLM-judge: faithfulness, answer_relevancy, context_recall, context_precision.

> **Golden dataset:** `golden_dataset.json` — 20 cặp Q&A (tuition_fees 8 · scholarships 7 · accommodation 2 · library 2 · wellbeing 1).

> **Chế độ chạy:** `full RAGAS (LLM-judge + A/B)`

---

## Overall Scores

| Metric            | Config A (hybrid + rerank) | Config B (dense-only) | Δ (A−B)       |
| ----------------- | -------------------------- | --------------------- | --------------- |
| faithfulness      | 1.000                      | 0.500                 | 0.500           |
| answer_relevancy  | 0.785                      | 0.477                 | 0.308           |
| context_recall    | 1.000                      | 0.500                 | 0.500           |
| context_precision | 0.875                      | 0.500                 | 0.375           |
| **Average** | **0.915**            | **0.494**       | **0.421** |

> ⚠ Nếu cột faithfulness/answer_relevancy = `—`: chế độ `offline` chỉ đo retrieval (context_recall/precision bằng string-match, KHÔNG dùng LLM). Metric LLM-judge cần `EVAL_MODEL` + API key thật (xem cuối file).

---

## A/B Comparison Analysis

**Config A — hybrid search + RRF rerank:**

> Semantic (bge-m3) + BM25 lexical, fuse bằng Reciprocal Rank Fusion, rồi rerank. Recall cao nhờ kết hợp keyword +semantic.

**Config B — dense-only (no rerank):**

> Chỉ semantic search, không có lexical signal và không rerank. Các chỉ số từ trung bình trở xuống

**Kết luận:**

> Config A ngang/trên Config B về context_recall (1.000 vs 0.500) → hybrid + rerank giúp retriever bắt đủ evidence hơn, đặc biệt với câu hỏi chứa thuật ngữ/số liệu (học phí, GPA, hạn chót) mà BM25 bắt tốt.

---

## Worst Performers (Bottom 3)

| # | Question                                                                   | Faithfulness | Relevance | Recall | Failure Stage | Root Cause                                                          |
| - | -------------------------------------------------------------------------- | ------------ | --------- | ------ | ------------- | ------------------------------------------------------------------- |
| 1 | Phí xét hồ sơ (application fee) cho chương trình University …      | 1.000        | 0.953     | 1.000  | generation    | Chunk đúng nhưng answer lệch (cần LLM metric để chắc chắn) |
| 2 | Sinh viên mới nhập học cần đóng bao nhiêu tiền đặt cọc (depo… | 1.000        | 0.617     | 1.000  | generation    | Chunk đúng nhưng answer lệch (cần LLM metric để chắc chắn) |

---

## Recommendations

### Cải tiến 1 — Mở rộng expected_context thành nhiều chunk tham chiếu

**Action:** Golden dataset hiện mô tả nguồn bằng 1 chuỗi (tên file + mục). Nên lưu `reference_contexts` là list các đoạn văn thật từ standardized markdown để context_recall RAGAS đo chính xác hơn.
**Expected impact:** context_recall phản ánh đúng chất lượng retriever, tránh false negative khi expected_context chỉ là nhãn nguồn.

### Cải tiến 2 — Calibrate SCORE_THRESHOLD cho fallback

**Action:** Đo khoảng cosine của semantic_search cho câu liên quan vs lạc đề, chọn ngưỡng ở giữa (theo cảnh báo trong task9). Đặc biệt cho câu số liệu (học phí).
**Expected impact:** giảm trường hợp trả kết quả rác thay vì fallback PageIndex; tăng faithfulness.

### Cải tiến 3 — Reorder context + prompt citation chặt hơn

**Action:** áp dụng `reorder_for_llm` (tránh lost in the middle) và buộc LLM cite [source] sau mỗi khẳng định; thêm guardrail 'không đủ evidence → từ chối'.
**Expected impact:** tăng faithfulness + answer_relevancy, giảm bịa đặt.
