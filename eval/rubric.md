# Hướng dẫn chấm tay (rubric v1)

Mỗi câu chấm 5 tiêu chí, mỗi tiêu chí **0 hoặc 1**. Chấm trong `reports/eval/labeling_sheet.csv`.
Không mở `key.csv` khi đang chấm (chấm mù để không thiên vị cấu hình nào).
Đọc `source_quote` và nếu cần, mở tài liệu gốc ở đúng phần (`section`).

| Tiêu chí | 1 điểm khi | 0 điểm khi |
|---|---|---|
| `factual` | Câu hỏi và đáp án được đánh dấu đúng là **đúng** theo tài liệu | Đáp án đánh dấu đúng thực ra sai, hoặc câu hỏi chứa thông tin sai |
| `answerable` | Người đã đọc kỹ phần tài liệu trả lời được, không cần kiến thức ngoài | Cần kiến thức không có trong tài liệu |
| `single_correct` | Chỉ một lựa chọn đúng | Có ≥ 2 lựa chọn có thể coi là đúng, hoặc không có lựa chọn nào đúng |
| `distractors` | Cả 3 đáp án nhiễu đều sai nhưng **hợp lý** với người chưa nắm bài | Có nhiễu vô lý, lộ liễu, sai ngữ pháp với câu hỏi, hoặc đáp án đúng dễ đoán vì dài/khác kiểu |
| `level_match` | Đúng mức yêu cầu (`level`) | Ví dụ ghi `apply` nhưng chỉ hỏi nhắc lại định nghĩa |

Mức độ: **remember** = nhắc lại định nghĩa/sự kiện · **understand** = giải thích, so sánh, vì sao ·
**apply** = dùng kiến thức cho tình huống mới không có nguyên văn trong tài liệu.

## Ví dụ ranh giới (tự bổ sung khi gặp ca khó)

- "Cổng của HTTPS?" với nhiễu `80, 22, 8080` → distractors = 1 (đều là cổng thật, dễ nhầm).
- Cùng câu với nhiễu `con mèo, 3,14, màu xanh` → distractors = 0.
- "Vì sao UDP phù hợp cho gọi thoại?" ghi level `remember` → level_match = 0 (đây là understand).
- Đáp án đúng "UDP không đảm bảo thứ tự", nhiễu "UDP không có cơ chế bảo đảm thứ tự gói" → single_correct = 0.

## Quy trình

1. Chấm thử 20 câu, ghi lại ca khó vào mục ví dụ ở trên, rồi mới chấm tiếp.
2. Chấm ≥ 150 câu, chia nhiều buổi (mỗi buổi ≤ 50 câu để không mệt).
3. Nhờ một bạn chấm độc lập 30 câu → tính kappa người–người (`python -m eval.kappa`).
   Nếu kappa người–người thấp, rubric chưa rõ: sửa rubric trước khi tin bất kỳ con số nào.
