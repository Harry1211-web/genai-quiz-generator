"""Known-defect test and second judge, end to end with the fake model (no network)."""
import subprocess
import sys
from pathlib import Path

import pytest

from eval.judge_check import (
    DEFECTS,
    load_doc_sections,
    load_questions,
    plant_cases,
    plant_table,
    run_items,
    to_question,
    write_summary,
)
from quizgen.llm import FakeLLM
from quizgen.schema import RUBRIC_KEYS

ROOT = Path(__file__).parent.parent
SAMPLES = ROOT / "data" / "samples"


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    env = {"PATH": "", "LLM_MODEL_1": "", "LLM_MODEL_2": "", "LLM_MODEL_3": ""}
    subprocess.run([sys.executable, "-m", "eval.run_eval", str(SAMPLES), "--configs", str(out / "none.json"),
                    "--n", "3", "--out", str(out)], cwd=ROOT, check=True, capture_output=True,
                   env={**__import__("os").environ, **env})
    return out


def test_planted_defects_are_what_they_claim(run_dir):
    q = load_questions(run_dir)
    cases = plant_cases(q, n=5, seed=1)
    by = {d: [c for c in cases if c["defect"] == d] for d in DEFECTS}
    assert by["control"] and by["wrong_answer"]
    for c in by["wrong_answer"]:
        assert to_question(c["row"], **c["change"]).mcq.answer_index != int(c["row"]["answer_index"])
    for c in by["duplicate_correct"]:
        m = to_question(c["row"], **c["change"]).mcq
        assert m.options.count(m.options[m.answer_index]) == 2
    for c in by["level_mismatch"]:
        assert c["row"]["level"] == "remember" and to_question(c["row"], **c["change"]).mcq.level == "apply"
    for c in by["foreign_question"]:
        assert to_question(c["row"], **c["change"]).mcq.question != c["row"]["question"]
    assert plant_cases(q, n=5, seed=1) == cases  # deterministic


def test_grading_is_saved_and_resumed(run_dir, tmp_path):
    q = load_questions(run_dir)
    sections = load_doc_sections(SAMPLES)
    cases = plant_cases(q, n=3, seed=0)
    judge = FakeLLM()
    recs = run_items(judge, cases, sections, tmp_path / "plant_fake.jsonl", log=lambda m: None)
    assert len(recs) == len(cases) and all(set(RUBRIC_KEYS) <= set(r) for r in recs if "error" not in r)
    calls = judge.usage.calls
    run_items(judge, cases, sections, tmp_path / "plant_fake.jsonl", log=lambda m: None)
    assert judge.usage.calls == calls  # nothing graded twice
    table = plant_table(recs)
    assert set(table["defect"]) <= set(DEFECTS)
    summary = write_summary(tmp_path, q)
    assert "Known defects" in summary.read_text(encoding="utf-8")


def test_summary_with_partial_results(run_dir, tmp_path):
    """A judge that has graded only some defects (no control group yet) must not break the summary."""
    import json
    q = load_questions(run_dir)
    rec = {"case": "wrong_answer:x", "defect": "wrong_answer", "id": "x", **{k: 1 for k in RUBRIC_KEYS}}
    (tmp_path / "plant_partial.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    (tmp_path / "plant_empty.jsonl").write_text("", encoding="utf-8")
    text = write_summary(tmp_path, q).read_text(encoding="utf-8")
    assert "wrong_answer" in text and "no graded question yet" in text
