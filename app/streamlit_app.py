"""UI: upload a lecture -> generate -> review/edit -> export. Run: streamlit run app/streamlit_app.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st

from quizgen.export import to_csv, to_gift, to_json
from quizgen.extract import load_sections
from quizgen.llm import model_for_role
from quizgen.pipeline import run
from quizgen.schema import MCQ

MAX_MB = 10
st.set_page_config(page_title="GenAI Quiz Generator", layout="wide")
st.title("Generate review questions from a lecture")
st.caption("Uploaded files are processed in this session only and are not stored.")

with st.sidebar:
    strategy = st.selectbox("Prompting strategy", ["v3", "v2", "v1"],
                            format_func=lambda s: {"v1": "v1 zero-shot", "v2": "v2 few-shot",
                                                   "v3": "v3 plan -> generate"}[s])
    n = st.slider("Questions per section", 1, 5, 3)
    max_sections = st.slider("Max sections", 1, 20, 6)
    verify = st.checkbox("Verify with a second model", value=True)

up = st.file_uploader("Lecture (PDF, PPTX, TXT, MD)", type=["pdf", "pptx", "txt", "md"])
if up and up.size > MAX_MB * 1024 * 1024:
    st.error(f"File is larger than {MAX_MB} MB")
    st.stop()

if up and st.button("Generate questions", type="primary"):
    with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as f:
        f.write(up.getvalue())
    sections = load_sections(f.name)
    Path(f.name).unlink(missing_ok=True)
    with st.spinner(f"Processing {min(len(sections), max_sections)} sections..."):
        qs, stats = run(sections, model_for_role("generator"), model_for_role("verifier") if verify else None,
                        n, strategy, max_sections)
    st.session_state["qs"] = [q.to_dict() for q in qs]
    st.session_state["stats"] = stats

if "qs" in st.session_state:
    stats = st.session_state["stats"]
    c = st.columns(4)
    c[0].metric("Generated", stats["total"])
    c[1].metric("Passed verification", stats["accepted"])
    c[2].metric("Time (s)", stats["seconds"])
    c[3].metric("Cost (USD)", stats["cost_usd"])
    show_rejected = st.toggle("Show rejected questions", value=False)
    kept: list[MCQ] = []
    for i, d in enumerate(st.session_state["qs"]):
        if d["rejected"] and not show_rejected:
            continue
        warn = f"  ✖ {d['rejected']}" if d["rejected"] else ""
        warn += f"  ⚠ {d['warnings']}" if d.get("warnings") else ""
        label = f"{i + 1}. [{d['level']}] {d['question'][:90]}{warn}"
        with st.expander(label):
            q = st.text_area("Question", d["question"], key=f"q{i}")
            opts = [st.text_input(f"Option {'ABCD'[j]}", o, key=f"o{i}{j}")
                    for j, o in enumerate(d["options"])]
            ans = st.radio("Correct answer", range(4), index=d["answer_index"], key=f"a{i}",
                           format_func=lambda j: "ABCD"[j], horizontal=True)
            exp = st.text_area("Explanation", d["explanation"], key=f"e{i}")
            st.info(f"Source (page {d['source_page']}): “{d['source_quote']}”")
            keep = st.checkbox("Keep this question", value=not d["rejected"], key=f"k{i}")
        if keep:
            try:
                kept.append(MCQ(question=q, options=opts, answer_index=ans, explanation=exp,
                                level=d["level"], source_quote=d["source_quote"],
                                source_page=d["source_page"]))
            except Exception as e:  # the edited question is no longer valid
                st.warning(f"Question {i + 1} is invalid: {e}")
    st.subheader(f"Export {len(kept)} questions")
    c = st.columns(3)
    c[0].download_button("JSON", to_json(kept), "quiz.json")
    c[1].download_button("CSV", "\ufeff" + to_csv(kept), "quiz.csv")  # BOM for Excel
    c[2].download_button("GIFT (Moodle)", to_gift(kept), "quiz.gift.txt")
