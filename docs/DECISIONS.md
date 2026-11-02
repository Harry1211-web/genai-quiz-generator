# Các quyết định thiết kế (tự trả lời bằng lời của bạn trước khi phỏng vấn)

1. Vì sao không dùng RAG? Khi nào sẽ cần?
2. Vì sao bắt buộc `source_quote`? Kiểm chứng nó thế nào, ngưỡng khớp mờ 0,85 chọn ra sao?
3. Vì sao model kiểm tra nên khác model sinh?
4. Vì sao xáo trộn vị trí đáp án sau khi sinh?
5. Chiến lược prompt nào thắng, thắng bao nhiêu, ở tiêu chí nào? Có ý nghĩa thống kê không (bao nhiêu câu)?
6. Judge đồng thuận với người ở tiêu chí nào, không đồng thuận ở tiêu chí nào? Vì sao?
7. Model nhỏ local vs model lớn API: chất lượng, chi phí, độ trễ, quyền riêng tư.
8. LoRA có giúp model nhỏ đuổi kịp không? Dữ liệu train lấy từ đâu, có rò rỉ sang tập đánh giá không?
9. Tài liệu tải lên có thể chứa prompt injection – đã chặn thế nào?
10. Nếu đưa cho 1.000 giảng viên dùng, bạn thay đổi gì?
