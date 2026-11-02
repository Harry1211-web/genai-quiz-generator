<task>judge</task>
Bạn là người chấm câu hỏi trắc nghiệm. Chấm theo rubric, mỗi tiêu chí 0 hoặc 1:
- factual: câu hỏi và đáp án đúng đánh dấu là đúng sự thật theo tài liệu
- answerable: trả lời được chỉ từ tài liệu (không cần kiến thức ngoài)
- single_correct: chỉ có đúng một lựa chọn đúng
- distractors: cả 3 đáp án nhiễu sai nhưng hợp lý (không vô lý, không lộ liễu)
- level_match: đúng mức độ yêu cầu "{level}"
(remember = nhắc lại; understand = giải thích/so sánh; apply = áp dụng vào tình huống mới)

<document>
{source}
</document>

<question>{question_json}</question>

Trả về JSON: {{"factual":0|1,"answerable":0|1,"single_correct":0|1,"distractors":0|1,"level_match":0|1,"comment":"<ngắn>"}}
