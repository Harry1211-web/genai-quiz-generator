"""Check the LLM-as-judge without human ratings.

plant : known-defect test. Questions the judge rated perfect (5/5) are broken on purpose, in a way whose
        correct verdict is known, and graded again by the judge:
          wrong_answer       the answer key points to a distractor           -> factual must be 0
          duplicate_correct  a distractor is replaced by the correct option  -> single_correct must be 0
          absurd_distractor  a distractor is replaced by an unrelated phrase -> distractors must be 0
          foreign_question   a question from ANOTHER document is graded
                             against this section                            -> answerable must be 0
          level_mismatch     a "remember" question is labelled "apply"       -> level_match must be 0
          control            the question is NOT changed                     -> all criteria should stay 1
        The detection rate per defect says whether the judge notices that kind of error; the control
        group gives its false-alarm rate and its consistency with its own earlier grades.
cross : second judge. Every accepted question is graded again by ANOTHER model and compared with the
        first judge (agreement and Cohen's kappa per criterion).

  python -m eval.judge_check plant reports/eval --docs data/private/eval_set --judge 1 --n 30
  python -m eval.judge_check cross reports/eval --docs data/private/eval_set --judge 2
  (a local Ollama judge with thinking: set LLM_THINK_2=true in the shell; results go to *-think.jsonl)

Results are saved after every graded question in <run>/judge_check/ (a crash or Ctrl+C loses nothing,
running the same command again resumes) and summarised in <run>/judge_check/judge_check.md.
"""
from __future__ import annotations

import argparse
import ast
import random
import re
from pathlib import Path

import httpx
import pandas as pd

from eval.judge import judge_one
from eval.kappa import cohen_kappa, read_sheet
from quizgen.extract import load_sections
from quizgen.generate import GeneratedQuestion
from quizgen.llm import LLM, BudgetExceeded, LLMRequestError, model_for_role
from quizgen.pipeline import _append, _read_jsonl, log_stderr
from quizgen.schema import MCQ, RUBRIC_KEYS

DEFECTS = {  # defect -> criterion the judge must set to 0
    "wrong_answer": "factual",
    "duplicate_correct": "single_correct",
    "absurd_distractor": "distractors",
    "foreign_question": "answerable",
    "level_mismatch": "level_match",
    "control": None,
}
ABSURD = ["A recipe for chocolate cake", "The national flower of Japan", "A brand of running shoes",
          "The distance from the Earth to the Moon", "A painting by Vincent van Gogh",
          "The rules of a football match"]
SUFFIXES = {".pdf", ".pptx", ".txt", ".md"}


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name)


def load_questions(run: Path) -> pd.DataFrame:
    """Accepted questions of a finished run with the first judge's scores."""
    q = read_sheet(run / "questions.csv", dtype=str).fillna("")
    q = q[(q["rejected"] == "") & (q[f"judge_{RUBRIC_KEYS[0]}"] != "")].copy()
    for k in RUBRIC_KEYS:
        q[f"judge_{k}"] = q[f"judge_{k}"].astype(float).astype(int)
    q["answer_index"] = q["answer_index"].astype(int)
    q["options"] = q["options"].map(ast.literal_eval)
    return q.reset_index(drop=True)


def load_doc_sections(docs: Path) -> dict[tuple[str, str], object]:
    out = {}
    for p in sorted(docs.iterdir()):
        if p.suffix.lower() in SUFFIXES and not p.name.startswith(("~$", ".")):
            for s in load_sections(p):
                out[(p.name, s.title)] = s
    return out


def to_question(r: dict, **change) -> GeneratedQuestion:
    """Rebuild a question from a questions.csv row. model_construct: a planted defect (two identical
    options) would not pass the schema validation, and must still reach the judge."""
    fields = {"question": r["question"], "options": list(r["options"]),
              "answer_index": int(r["answer_index"]), "explanation": r["explanation"], "level": r["level"],
              "source_quote": r["source_quote"], "source_page": None}
    fields.update(change)
    return GeneratedQuestion(MCQ.model_construct(**fields), r["section"], r["strategy"], r["model"],
                             int(r["page_start"] or 0), int(r["page_end"] or 0))


def plant_cases(q: pd.DataFrame, n: int, seed: int = 0) -> list[dict]:
    """Build the known-defect cases (deterministic for a given seed)."""
    rng = random.Random(seed)
    perfect = q[q[[f"judge_{k}" for k in RUBRIC_KEYS]].sum(axis=1) == len(RUBRIC_KEYS)]
    rows = perfect.to_dict("records")
    cases = []
    for defect in DEFECTS:
        pool = [r for r in rows if r["level"] == "remember"] if defect == "level_mismatch" else rows
        for r in rng.sample(pool, min(n, len(pool))):
            opts, ans = list(r["options"]), int(r["answer_index"])
            wrong = [i for i in range(4) if i != ans]
            change: dict = {}
            graded_as = r
            if defect == "wrong_answer":
                change = {"answer_index": rng.choice(wrong)}
            elif defect == "duplicate_correct":
                opts[rng.choice(wrong)] = opts[ans]
                change = {"options": opts}
            elif defect == "absurd_distractor":
                opts[rng.choice(wrong)] = rng.choice(ABSURD)
                change = {"options": opts}
            elif defect == "foreign_question":
                others = [o for o in q.to_dict("records") if o["doc"] != r["doc"]]
                if not others:
                    continue
                f = rng.choice(others)  # a question of another document, graded against r's section
                change = {k: f[k] for k in ("question", "options", "answer_index", "explanation", "level",
                                             "source_quote")}
            elif defect == "level_mismatch":
                change = {"level": "apply"}
            cases.append({"case": f"{defect}:{r['id']}", "defect": defect, "id": r["id"],
                          "doc": graded_as["doc"], "section": graded_as["section"], "row": r,
                          "change": change})
    return cases


def grade(judge: LLM, q: GeneratedQuestion, section) -> dict:
    u0, c0 = judge.usage.prompt_tokens + judge.usage.completion_tokens, judge.cost_usd()
    js = judge_one(judge, q, section)
    if js:
        rec = {k: getattr(js, k) for k in RUBRIC_KEYS} | {"comment": js.comment}
    else:  # keep the model's last answer and the validation error, to see what went wrong
        rec = {"error": "format", **getattr(judge, "last_format_error", {})}
    rec["tokens"] = judge.usage.prompt_tokens + judge.usage.completion_tokens - u0
    rec["cost_usd"] = round(judge.cost_usd() - c0, 6)
    return rec


def run_items(judge: LLM, items: list[dict], sections: dict, path: Path, log,
              retry_errors: bool = False) -> list[dict]:
    """Grade every item not graded yet, saving after each one. Network errors are retried next run;
    answers that stayed invalid JSON are kept as format errors (retried with retry_errors=True)."""
    done = {r["case"]: r for r in _read_jsonl(path)}
    if retry_errors:
        done = {k: r for k, r in done.items() if "error" not in r}
    todo = [it for it in items if it["case"] not in done]
    errors = shown = 0
    for i, it in enumerate(todo, 1):
        section = sections.get((it["doc"], it["section"]))
        if section is None:
            log(f"[skip] {it['case']}: section not found in --docs (was the extraction changed?)")
            continue
        try:
            rec = grade(judge, to_question(it["row"], **it["change"]), section)
        except BudgetExceeded:
            raise
        except (httpx.HTTPError, LLMRequestError) as e:
            errors += 1
            log(f"[error] {it['case']}: {type(e).__name__} - retried on the next run")
            if errors >= 3:
                log("[stop] 3 errors in a row: fix the model (e.g. start Ollama) and run again to resume")
                break
            continue
        errors = 0
        rec = {"case": it["case"], "defect": it.get("defect", ""), "id": it["id"], **rec}
        _append(path, rec)
        done[it["case"]] = rec
        if "error" in rec and shown < 3:  # show the first few invalid answers to see what went wrong
            shown += 1
            log(f"    raw answer: {rec.get('raw', '')[:400]!r}\n    why: {rec.get('error', '')[:300]}")
        log(f"[{i}/{len(todo)}] {it['case']}: " + ("format error" if "error" in rec else
                                                    " ".join(f"{k[:4]}={rec[k]}" for k in RUBRIC_KEYS)))
    return list(done.values())


# ---------------------------------------------------------------- summaries
def plant_table(recs: list[dict]) -> pd.DataFrame:
    rows = []
    for defect, crit in DEFECTS.items():
        rs = [r for r in recs if r["defect"] == defect and "error" not in r]
        if not rs:
            continue
        row = {"defect": defect, "must be 0": crit or "(none)", "n": len(rs)}
        if crit:
            row["detected"] = round(sum(r[crit] == 0 for r in rs) / len(rs), 3)
            row["any criterion 0"] = round(sum(any(r[k] == 0 for k in RUBRIC_KEYS) for r in rs) / len(rs), 3)
        else:  # control: the question is unchanged and was rated 5/5 before
            same = sum(all(r[k] == 1 for k in RUBRIC_KEYS) for r in rs)
            row["still 5/5 (consistency)"] = round(same / len(rs), 3)
            for k in RUBRIC_KEYS:
                row[f"false alarm {k}"] = round(sum(r[k] == 0 for r in rs) / len(rs), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def cross_table(q: pd.DataFrame, recs: list[dict]) -> pd.DataFrame:
    second = {r["id"]: r for r in recs if "error" not in r}
    both = q[q["id"].isin(second)]
    rows = []
    for k in RUBRIC_KEYS:
        a = both[f"judge_{k}"].tolist()
        b = [int(second[i][k]) for i in both["id"]]
        if not a:
            continue
        agree = sum(x == y for x, y in zip(a, b, strict=True)) / len(a)
        rows.append({"criterion": k, "n": len(a), "agreement": round(agree, 3),
                     "kappa": round(cohen_kappa(a, b), 3) if len(set(a) | set(b)) > 1 else float("nan"),
                     "judge 1 mean": round(sum(a) / len(a), 3), "judge 2 mean": round(sum(b) / len(b), 3)})
    return pd.DataFrame(rows)


def write_summary(out: Path, q: pd.DataFrame) -> Path:
    parts = ["# Judge check (no human ratings)\n",
             "Known-defect test: questions rated 5/5 by the judge were broken on purpose; `detected` is the "
             "share where the judge set the expected criterion to 0. Control: unchanged questions graded "
             "again.\n"]
    for f in sorted(out.glob("plant_*.jsonl")):
        recs = _read_jsonl(f)
        cost = sum(r.get("cost_usd", 0) for r in recs)
        t = plant_table(recs)
        parts.append(f"\n## Known defects - judge `{f.stem.removeprefix('plant_')}` (${cost:.4f})\n")
        if t.empty:
            parts.append("(no graded question yet)\n")
            continue
        defects = t[t["must be 0"] != "(none)"].dropna(axis=1, how="all")
        control = t[t["must be 0"] == "(none)"].dropna(axis=1, how="all")
        if not defects.empty:
            parts += [defects.to_markdown(index=False), "\n"]
        if not control.empty:
            parts += ["\nControl (unchanged questions rated 5/5 before):\n",
                      control.drop(columns=["must be 0"], errors="ignore").to_markdown(index=False), "\n"]
    for f in sorted(out.glob("cross_*.jsonl")):
        recs = _read_jsonl(f)
        cost = sum(r.get("cost_usd", 0) for r in recs)
        parts += [f"\n## Second judge `{f.stem.removeprefix('cross_')}` vs first judge (${cost:.4f})\n",
                  "Kappa is unreliable when almost every score is 1 (kappa paradox): "
                  "read the agreement too.\n\n",
                  cross_table(q, recs).to_markdown(index=False), "\n"]
    path = out / "judge_check.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Check the LLM-as-judge without human ratings")
    ap.add_argument("mode", choices=["plant", "cross"])
    ap.add_argument("run", help="folder of a finished run, e.g. reports/eval")
    ap.add_argument("--docs", required=True, help="folder of the documents of that run")
    ap.add_argument("--judge", default=None, help="number of the judge model in .env (default: role default)")
    ap.add_argument("--n", type=int, default=30, help="plant: questions per defect")
    ap.add_argument("--limit", type=int, default=None,
                    help="cross: grade a random sample of this many questions (same sample for every judge)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retry-errors", action="store_true",
                    help="grade again the questions whose answer was invalid JSON last time")
    a = ap.parse_args()

    run, out = Path(a.run), Path(a.run) / "judge_check"
    out.mkdir(parents=True, exist_ok=True)
    q = load_questions(run)
    sections = load_doc_sections(Path(a.docs))
    judge = model_for_role("judge", a.judge)
    # results of the same model with thinking on and off are kept apart
    tag = _safe(judge.name) + ("-think" if getattr(judge, "think", None) else "")
    log_stderr(f"{len(q)} accepted questions, judge = {judge.name}")
    if a.mode == "plant":
        items = plant_cases(q, a.n, a.seed)
        path = out / f"plant_{tag}.jsonl"
    else:
        items = [{"case": r["id"], "id": r["id"], "doc": r["doc"], "section": r["section"], "row": r,
                  "change": {}} for r in q.to_dict("records")]
        if a.limit:
            items = random.Random(a.seed).sample(items, min(a.limit, len(items)))
        path = out / f"cross_{tag}.jsonl"
    run_items(judge, items, sections, path, log_stderr, retry_errors=a.retry_errors)
    u = judge.usage
    log_stderr(f"{judge.name} this run: {u.calls} calls, {u.prompt_tokens} in / {u.completion_tokens} "
               f"out tokens, ${judge.cost_usd():.4f}")
    summary = write_summary(out, q)
    print(summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
