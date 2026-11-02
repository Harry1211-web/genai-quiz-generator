# GenAI Quiz Generator

Sinh câu hỏi trắc nghiệm (4 lựa chọn, đáp án, giải thích, **câu trích nguồn**) từ bài giảng PDF,
kiểm chứng tự động trước khi hiển thị, và **đánh giá chất lượng bằng số liệu**: 2 model × 3 chiến lược
prompt, ≥ 150 câu chấm tay theo rubric, đồng thuận LLM-as-judge với người (Cohen's kappa).

> ⚠️ Bản khởi đầu: các bảng kết quả dưới đây để trống – điền sau khi chạy thí nghiệm thật.

![demo](reports/demo.gif)

## Kết quả

| Model × Prompt | Đúng & có căn cứ | Nhiễu hợp lý | Đúng mức độ | Bị loại tự động | Lỗi định dạng | $/100 câu | Giây/bài |
|---|---|---|---|---|---|---|---|
| Lớn × zero-shot | | | | | | | |
| Lớn × few-shot | | | | | | | |
| Lớn × kế hoạch → sinh | | | | | | | |
| Nhỏ × zero-shot | | | | | | | |
| Nhỏ × few-shot | | | | | | | |
| Nhỏ × kế hoạch → sinh | | | | | | | |

Kappa judge–người theo tiêu chí: xem `reports/quality_report.md`.

## Kiến trúc

```
PDF → trích xuất (PyMuPDF, NFC) → chia phần theo tiêu đề (≤1.500 token, giữ số trang)
    → [v1 zero-shot | v2 few-shot | v3 lập kế hoạch ý chính → sinh]  (JSON + Pydantic, tự sửa khi sai)
    → xáo trộn lựa chọn
    → kiểm chứng: câu trích có thật? đáp án lộ? model thứ hai trả lời lại chỉ với đoạn nguồn → khớp?
    → chống trùng (embedding) → duyệt/sửa trên Streamlit → xuất JSON / CSV / GIFT (Moodle)
```

## Chạy nhanh

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                   # mặc định LLM_MODEL=fake: chạy thử không tốn tiền
pytest -q
python -m quizgen.pipeline data/samples/mang_may_tinh_tcp.md --n 2 --strategy v3
streamlit run app/streamlit_app.py
uvicorn api.main:app --reload                          # POST /generate
docker compose up --build                              # API :8000, giao diện :8501
```

Dùng model thật: sửa `.env` (`LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`). Mọi endpoint tương thích OpenAI
đều chạy được, kể cả Ollama local (`http://localhost:11434/v1`).

## Thí nghiệm

```bash
cp eval/configs.example.json eval/configs.json         # khai báo 2 model
python -m eval.run_eval data/samples --n 3 --max-sections 6
# chấm tay reports/eval/labeling_sheet.csv theo eval/rubric.md
python -m eval.report reports/eval
```

## Dữ liệu và giấy phép

Code: MIT. Tài liệu demo trong `data/samples/` do tác giả tự viết (CC BY 4.0) hoặc lấy từ nguồn có
giấy phép mở (ghi rõ nguồn từng file). Không đưa bài giảng có bản quyền lên repo.
