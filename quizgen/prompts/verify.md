<task>verify</task>
Trả lời câu hỏi trắc nghiệm CHỈ dựa vào đoạn tài liệu. Nếu đoạn tài liệu không đủ thông tin
để chọn duy nhất một đáp án, trả answer_index = -1.

<document>
{source}
</document>

<question>{question_json}</question>

Trả về JSON: {{"answer_index": <-1..3>, "reason": "<một câu>"}}
