"""Giao diện: tải bài giảng → sinh → duyệt/sửa → xuất. Chạy: streamlit run app/streamlit_app.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st

from quizgen.export import to_csv, to_gift, to_json
from quizgen.extract import load_sections
from quizgen.llm import from_env
from quizgen.pipeline import run
from quizgen.schema import MCQ

MAX_MB = 10
st.set_page_config(page_title="GenAI Quiz Generator", layout="wide")
st.title("Tạo câu hỏi ôn tập từ bài giảng")
st.caption("Tài liệu chỉ được xử lý trong phiên làm việc này, không lưu lại.")

with st.sidebar:
    strategy = st.selectbox("Chiến lược prompt", ["v3", "v2", "v1"],
                            format_func=lambda s: {"v1": "v1 zero-shot", "v2": "v2 few-shot",
                                                   "v3": "v3 lập kế hoạch → sinh"}[s])
    n = st.slider("Số câu mỗi phần", 1, 5, 3)
    max_sections = st.slider("Số phần tối đa", 1, 20, 6)
    verify = st.checkbox("Kiểm chứng bằng model thứ hai", value=True)

up = st.file_uploader("Bài giảng (PDF, PPTX, TXT, MD)", type=["pdf", "pptx", "txt", "md"])
if up and up.size > MAX_MB * 1024 * 1024:
    st.error(f"File lớn hơn {MAX_MB} MB")
    st.stop()

if up and st.button("Sinh câu hỏi", type="primary"):
    with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as f:
        f.write(up.getvalue())
    sections = load_sections(f.name)
    Path(f.name).unlink(missing_ok=True)
    with st.spinner(f"Đang xử lý {min(len(sections), max_sections)} phần..."):
        qs, stats = run(sections, from_env("LLM"), from_env("VERIFY") if verify else None,
                        n, strategy, max_sections)
    st.session_state["qs"] = [q.to_dict() for q in qs]
    st.session_state["stats"] = stats

if "qs" in st.session_state:
    stats = st.session_state["stats"]
    c = st.columns(4)
    c[0].metric("Câu sinh ra", stats["total"])
    c[1].metric("Qua kiểm chứng", stats["accepted"])
    c[2].metric("Thời gian (s)", stats["seconds"])
    c[3].metric("Chi phí (USD)", stats["cost_usd"])
    show_rejected = st.toggle("Hiện cả câu bị loại", value=False)
    kept: list[MCQ] = []
    for i, d in enumerate(st.session_state["qs"]):
        if d["rejected"] and not show_rejected:
            continue
        warn = f"  ⚠ {d['rejected']}" if d["rejected"] else ""
        label = f"{i + 1}. [{d['level']}] {d['question'][:90]}{warn}"
        with st.expander(label):
            q = st.text_area("Câu hỏi", d["question"], key=f"q{i}")
            opts = [st.text_input(f"Lựa chọn {'ABCD'[j]}", o, key=f"o{i}{j}")
                    for j, o in enumerate(d["options"])]
            ans = st.radio("Đáp án đúng", range(4), index=d["answer_index"], key=f"a{i}",
                           format_func=lambda j: "ABCD"[j], horizontal=True)
            exp = st.text_area("Giải thích", d["explanation"], key=f"e{i}")
            st.info(f"Nguồn (trang {d['source_page']}): “{d['source_quote']}”")
            keep = st.checkbox("Giữ câu này", value=not d["rejected"], key=f"k{i}")
        if keep:
            try:
                kept.append(MCQ(question=q, options=opts, answer_index=ans, explanation=exp,
                                level=d["level"], source_quote=d["source_quote"],
                                source_page=d["source_page"]))
            except Exception as e:  # câu bị sửa sai định dạng
                st.warning(f"Câu {i + 1} không hợp lệ: {e}")
    st.subheader(f"Xuất {len(kept)} câu")
    c = st.columns(3)
    c[0].download_button("JSON", to_json(kept), "quiz.json")
    c[1].download_button("CSV", to_csv(kept), "quiz.csv")
    c[2].download_button("GIFT (Moodle)", to_gift(kept), "quiz.gift.txt")
