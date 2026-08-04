# RAG Evaluation Results — University Services RAG

## Framework sử dụng

> **RAGAS** (`pip install ragas`) — 4 metric LLM-judge: faithfulness, answer_relevancy, context_recall, context_precision.

> **Golden dataset:** `golden_dataset.json` — 20 cặp Q&A (tuition_fees 8 · scholarships 7 · accommodation 2 · library 2 · wellbeing 1).

> **Chế độ chạy:** `full RAGAS (LLM-judge + A/B)` — 20/20 câu, judge `gpt-4o-mini`, retrieval tắt HyDE (`RAG_USE_HYDE=false`) cho deterministic.

---

## Overall Scores

| Metric            | Config A (hybrid + rerank) | Config B (dense-only) | Δ (A−B)       |
| ----------------- | -------------------------- | --------------------- | --------------- |
| faithfulness      | 0.767                      | 0.644                 | 0.123           |
| answer_relevancy  | 0.640                      | 0.578                 | 0.062           |
| context_recall    | 0.608                      | 0.537                 | 0.071           |
| context_precision | 0.641                      | 0.661                 | -0.020          |
| **Average** | **0.664**            | **0.605**       | **0.059** |

> ⚠ Nếu cột faithfulness/answer_relevancy = `—`: chế độ `offline` chỉ đo retrieval (context_recall/precision bằng string-match, KHÔNG dùng LLM). Metric LLM-judge cần `EVAL_MODEL` + API key thật (xem cuối file).

---

## Key Findings

- **Config thắng cuộc:** Config A (hybrid + rerank) (avg 0.664 vs 0.605, Δ=0.059). Hybrid + RRF rerank tốt hơn dense-only trên trung bình, đặc biệt ở các câu số liệu/thuật ngữ.
- **Metric chênh lệch nhất:** `faithfulness` (Δ=0.123) — hybrid+rerank vượt. BM25 bắt được keyword như "application fee", "GPA", "3 July" → context chính xác hơn → LLM ít bịa đặt.
- **3 câu faithfulness=0** (G16, G17, G18): retrieval không đưa được chunk đúng vào context → LLM không có cơ sở trả lời. G16 (chỗ ở trong khuôn viên) và G18 (mượn sách thư viện) — đáp án nằm rải rác/gần lỗi tách chunk; cần cải thiện chunking hoặc fallback PageIndex.
- **Recall≥0.5 nhưng faithfulness<0.5** (G17): chunk đúng đã lấy về (recall=1.0) nhưng LLM không dùng → vấn đề ở generation (reorder/prompt), không phải retriever.
- **Câu học bổng G09–G12** faithfulness≈1.0: news articles được chunk sạch, retrieval ổn định.

---

## A/B Comparison Analysis

**Config A — hybrid search + RRF rerank:**

> Semantic (paraphrase-multilingual-MiniLM-L12-v2) + BM25 lexical, fuse bằng Reciprocal Rank Fusion. Recall cao nhờ kết hợp keyword + ngữ nghĩa.

**Config B — dense-only (no rerank):**

> Chỉ semantic search, không có lexical signal và không rerank. Test xem rerank + lexical đóng góp bao nhiêu.

**Kết luận:**

> Config A ngang/trên Config B về context_recall (0.608 vs 0.537) → hybrid + rerank giúp retriever bắt đủ evidence hơn, đặc biệt với câu hỏi chứa thuật ngữ/số liệu (học phí, GPA, hạn chót) mà BM25 bắt tốt. Dense-only chỉ vượt ở `context_precision` (0.661 vs 0.641): không có lexical signal nên ít đưa chunk nhiễu, nhưng đánh đổi mất recall.

---

## Per-question Breakdown (Config A)

| ID  | Question (rút gọn)                                   | Faith | Relev | Recall | Precision |
| --- | ------------------------------------------------------ | ----- | ----- | ------ | --------- |
| G01 | Phí xét hồ sơ (application fee) cho chương…     | 1.00  | 0.95  | 1.00   | 0.75      |
| G02 | Sinh viên mới nhập học cần đóng bao nhiêu…    | 1.00  | 0.62  | 1.00   | 1.00      |
| G03 | Chính sách hỗ trợ học phí cho gia đình có…   | 1.00  | 0.90  | 0.33   | 1.00      |
| G04 | RMIT Vietnam có những phương thức thanh to…      | 0.67  | 0.00  | 0.00   | 0.25      |
| G05 | Nếu sinh viên không đóng học phí đúng hạn…  | 1.00  | 0.73  | 1.00   | 1.00      |
| G06 | Phí mượn sách thư viện trả trễ (overdue) m…   | 0.50  | 0.00  | 0.50   | 1.00      |
| G07 | Bảo hiểm y tế (Medical Insurance) đối với…      | 1.00  | 0.96  | 0.75   | 1.00      |
| G08 | Sinh viên học khóa tiêu chuẩn (standard co…      | 0.75  | 0.88  | 1.00   | 1.00      |
| G09 | Chương trình học bổng 2026 của RMIT Vietna…     | 1.00  | 0.97  | 1.00   | 0.25      |
| G10 | Hạn chót nộp đơn học bổng 2026 cho hầu hết…  | 1.00  | 0.89  | 1.00   | 1.00      |
| G11 | Học bổng Vice-Chancellor's Scholar đòi hỏi…      | 1.00  | 0.88  | 0.67   | 0.83      |
| G12 | Học bổng Opportunity Scholarship dành cho…         | 1.00  | 0.89  | 1.00   | 0.76      |
| G13 | Người nhận học bổng cần duy trì GPA tích l…   | 0.67  | 0.92  | 1.00   | 0.33      |
| G14 | Người nhận học bổng được đi exchange và xi… | 1.00  | 0.57  | 0.67   | 0.95      |
| G15 | Học bổng Erasmus+ đã đưa sinh viên Nguyễn …   | 0.75  | 0.84  | 0.00   | 0.33      |
| G16 | RMIT Vietnam có cung cấp chỗ ở (ký túc xá)…    | 0.00  | 0.00  | 0.00   | 0.00      |
| G17 | Sinh viên quốc tế thuê nhà ở Việt Nam cần …   | 0.00  | 0.00  | 1.00   | 0.33      |
| G18 | Sinh viên được mượn tối đa bao nhiêu quyển… | 0.00  | 0.00  | 0.00   | 0.00      |
| G19 | Thư viện RMIT Vietnam cung cấp những nguồn…      | 1.00  | 0.96  | 0.25   | 0.33      |
| G20 | Khi gặp vấn đề bạo lực gia đình/quấy rối t… | 1.00  | 0.85  | 0.00   | 0.70      |

---

## Worst Performers (Bottom 3)

| # | Question                                                               | Faithfulness | Relevance | Recall | Failure Stage | Root Cause                                                               |
| - | ---------------------------------------------------------------------- | ------------ | --------- | ------ | ------------- | ------------------------------------------------------------------------ |
| 1 | RMIT Vietnam có cung cấp chỗ ở (ký túc xá) trong khuôn…       | 0.000        | 0.000     | 0.000  | retrieval     | Retriever không lấy đủ chunk chứa đáp án (thiếu lexical/hybrid) |
| 2 | Sinh viên được mượn tối đa bao nhiêu quyển sách tại thư… | 0.000        | 0.000     | 0.000  | retrieval     | Retriever không lấy đủ chunk chứa đáp án (thiếu lexical/hybrid) |
| 3 | RMIT Vietnam có những phương thức thanh toán học phí nà…     | 0.667        | 0.000     | 0.000  | retrieval     | Retriever không lấy đủ chunk chứa đáp án (thiếu lexical/hybrid) |

---

## Recommendations

### Cải tiến 1 — Mở rộng expected_context thành nhiều chunk tham chiếu

**Action:** Golden dataset hiện mô tả nguồn bằng 1 chuỗi (tên file + mục). Nên lưu `reference_contexts` là list các đoạn văn thật từ standardized markdown để context_recall RAGAS đo chính xác hơn.
**Expected impact:** context_recall phản ánh đúng chất lượng retriever, tránh false negative khi expected_context chỉ là nhãn nguồn.

### Cải tiến 2 — Cải thiện chunking cho câu accommodation/library (G16/G18)

**Action:** G16 ("có KTX không") và G18 ("mượn bao nhiêu sách") = 0 vì đáp án nằm trong 1 đoạn chung/gần ranh giới chunk. Giảm chunk_size hoặc thêm MarkdownHeaderTextSplitter để giữ nguyên đoạn chứa câu trả lời. Bật fallback PageIndex (task8) cho câu cosine thấp.
**Expected impact:** đưa 3 câu faithfulness=0 lên ≥0.7; tăng average ~+0.05–0.10.

### Cải tiến 3 — Reorder context + prompt citation chặt hơn (G17)

**Action:** G17 recall=1.0 nhưng faith=0 → LLM bỏ qua chunk đúng. Kiểm tra `reorder_for_llm`, buộc cite [source] sau mỗi khẳng định, thêm guardrail "nếu context có đáp án thì phải dùng". Calibrate lại `SCORE_THRESHOLD` (task9) cho câu accommodation.
**Expected impact:** tăng faithfulness/answer_relevancy, giảm bịa đặt.

---

## Cách chạy lại (Reproduction)

```bash
# 1) Offline (chỉ retrieval, không cần LLM — số context_recall/precision):
RAG_USE_HYDE=false python -m group_project.evaluation.eval_pipeline --offline

# 2) Full RAGAS (4 metric LLM-judge + A/B) — cần API key thật trong .env:
RAG_USE_HYDE=false EVAL_SUBSET=0 python -m group_project.evaluation.eval_pipeline
#    EVAL_SUBSET=0 (hoặc bỏ) để chạy hết 20 câu. Đặt EVAL_SUBSET=5 để giảm rate limit.
```

> **Ghi chú môi trường (đã xử lý):** `ragas==0.4.3` mặc định import fail vì `langchain_community 0.4.2` đã bỏ `chat_models.vertexai` / `llms.VertexAI`. Đã patch `ragas/llms/base.py` bọc 2 import đó trong try/except + stub class (chỉ dùng cho isinstance check, không ảnh hưởng judge). Embedding dùng `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, đã cache) thay cho bge-m3 (~2.4GB) để chạy benchmark nhanh trong môi trường lab. LLM key trong `.env` thực chất là OpenAI key (`sk-proj-…`) → code tự nhận diện provider theo tiền tố và dùng endpoint OpenAI trực tiếp.
