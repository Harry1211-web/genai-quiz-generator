"""Main experiment: every model x every prompting strategy on the same documents.

Crash-safe: every (config, strategy, document) "cell" runs in its own folder under
reports/eval/runs/ and saves its work after EACH step (see quizgen/pipeline.py):
  generated.jsonl - after each section is generated
  verified.jsonl  - after each question is verified
  judged.jsonl    - after each question is judged
  cell_stats.json - token/cost/time totals of the cell, accumulated over runs
Run the same command again after a crash or Ctrl+C: finished work is skipped and nothing paid
is paid twice. Use --regenerate to get a NEW set of questions: previous cell folders are moved to
reports/eval/archive/<timestamp>/ (never deleted).

Outputs (rebuilt from the cell folders at the end of every run):
  reports/eval/questions.csv      - every generated question (including rejected ones), with a stable id
  reports/eval/labeling_sheet.csv - HUMAN rating sheet: shuffled, model/strategy HIDDEN (blind rating).
                                    Ratings you already typed are KEPT when the sheet is rebuilt.
  reports/eval/key.csv            - id -> model/strategy key (do not open while rating)
  reports/eval/judge_labels.csv   - LLM-as-judge scores
  reports/eval/comparison.md      - comparison table

Models are configured in eval/configs.json (see eval/configs.example.json). Each config may define its
own "verifier" so that no configuration verifies its own output; otherwise VERIFY_* from .env is used.
The judge is read from JUDGE_* in .env and is the same for every configuration.

Run: python -m eval.run_eval data/private/eval_set --n 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

from quizgen.extract import load_sections
from quizgen.llm import LLM, BudgetExceeded, FakeLLM, LLMRequestError, OpenAICompatLLM, from_env
from quizgen.pipeline import _append, _read_jsonl, log_stderr, run
from quizgen.schema import RUBRIC_KEYS

from .judge import judge_one

JUDGED = "judged.jsonl"
CELL_STATS = "cell_stats.json"
DOC_SUFFIXES = {".pdf", ".pptx", ".txt", ".md"}


def make_llm(cfg: dict) -> LLM:
    """Build a model from a config dict: name, model, base_url, api_key_env, price_in, price_out,
    and optionally json_mode, reasoning_effort, timeout."""
    if cfg["model"] == "fake":
        return FakeLLM(name=cfg.get("name", "fake"))
    return OpenAICompatLLM(model=cfg["model"], base_url=cfg.get("base_url", "https://api.openai.com/v1"),
                           api_key=os.getenv(cfg.get("api_key_env", ""), ""),
                           price_in=cfg.get("price_in", 0), price_out=cfg.get("price_out", 0),
                           budget_usd=float(os.getenv("LLM_BUDGET_USD", "2")),
                           json_mode=cfg.get("json_mode", True),
                           reasoning_effort=cfg.get("reasoning_effort"),
                           timeout=float(cfg.get("timeout", 120)))


def stable_id(config: str, strategy: str, doc: str, qid: str, question: str) -> str:
    """Short id that does not reveal the configuration (blind rating). It includes the question text,
    so a regenerated question never inherits the rating of the question it replaced, while an
    unchanged question keeps its id (and rating) across runs."""
    key = f"{config}|{strategy}|{doc}|{qid}|{question}"
    return "q" + hashlib.sha1(key.encode()).hexdigest()[:10]


def _usage(m: LLM | None) -> tuple:
    if m is None:
        return (0, 0, 0, 0.0, 0.0)
    u = m.usage
    return (u.calls, u.prompt_tokens, u.completion_tokens, m.cost_usd(), u.seconds)


def _delta(after: tuple, before: tuple) -> tuple:
    return tuple(a - b for a, b in zip(after, before, strict=True))


def judge_cell(judge: LLM, qs, sections, cell: Path, log) -> dict[str, dict]:
    """Judge accepted questions, saving after each one. Network errors are retried on the next run;
    a judge answer that is still invalid JSON after retries is recorded and not paid for again."""
    path = cell / JUDGED
    done = {r["qid"]: r for r in _read_jsonl(path)}
    by_title = {s.title: s for s in sections}
    todo = [q for q in qs if q.ok and q.qid not in done]
    for i, q in enumerate(todo, 1):
        try:
            js = judge_one(judge, q, by_title[q.section_title])
        except BudgetExceeded:
            raise
        except (httpx.HTTPError, LLMRequestError) as e:
            log(f"[judge] {q.qid}: {type(e).__name__} - will retry on the next run")
            continue
        rec = {"qid": q.qid, **({k: getattr(js, k) for k in RUBRIC_KEYS} if js else {"error": "format"})}
        _append(path, rec)
        done[q.qid] = rec
        log(f"[judge {i}/{len(todo)}] {q.qid}: {'error' if js is None else sum(rec[k] for k in RUBRIC_KEYS)}")
    return done


def update_cell_stats(cell: Path, gen: tuple, ver: tuple, jud: tuple, n_questions: int,
                      n_accepted: int) -> dict:
    """Add this run's usage to the cell's running totals (resumed runs keep counting)."""
    path = cell / CELL_STATS
    s = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for prefix, d in (("gen", gen), ("verify", ver), ("judge", jud)):
        for name, v in zip(("calls", "tokens_in", "tokens_out", "cost_usd", "seconds"), d, strict=True):
            s[f"{prefix}_{name}"] = round(s.get(f"{prefix}_{name}", 0) + v, 6)
    s.update(questions=n_questions, accepted=n_accepted)
    path.write_text(json.dumps(s, indent=2), encoding="utf-8")
    return s


def merge_labeling_sheet(path: Path, new: pd.DataFrame, log=print) -> pd.DataFrame:
    """Keep every rating already typed in the sheet; append rows for new questions only.
    Rows of questions that no longer exist (after --regenerate) are removed from the sheet; if they
    were already rated, they are saved to labeling_sheet.orphaned.csv so no rating work is lost."""
    if not path.exists():
        return new.sample(frac=1, random_state=42)
    old = pd.read_csv(path, dtype=str).fillna("")
    shutil.copyfile(path, path.with_suffix(".backup.csv"))
    gone = old[~old["id"].isin(new["id"])]
    rated_gone = gone[(gone[RUBRIC_KEYS] != "").any(axis=1)]
    if len(rated_gone):
        orphan_path = path.with_name("labeling_sheet.orphaned.csv")
        prev = pd.read_csv(orphan_path, dtype=str).fillna("") if orphan_path.exists() else None
        both = rated_gone if prev is None else pd.concat([prev, rated_gone]).drop_duplicates("id")
        both.to_csv(orphan_path, index=False)
        log(f"[sheet] {len(rated_gone)} rated questions no longer exist -> kept in {orphan_path.name}")
    kept = old[old["id"].isin(new["id"])]
    added = new[~new["id"].isin(old["id"])].sample(frac=1, random_state=42)
    if len(added) or len(gone):
        log(f"[sheet] {len(kept)} kept, {len(added)} new questions added, {len(gone)} removed")
    return pd.concat([kept, added.astype(str)], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("docs", help="folder of documents (pdf/pptx/txt/md)")
    ap.add_argument("--configs", default="eval/configs.json")
    ap.add_argument("--strategies", default="v1,v2,v3")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-sections", type=int, default=None,
                    help="limit sections per document (trial runs)")
    ap.add_argument("--out", default="reports/eval")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--regenerate", "--fresh", dest="regenerate", action="store_true",
                    help="generate a NEW set of questions; previous runs are moved to <out>/archive/ "
                         "(ratings already typed are kept)")
    a = ap.parse_args()
    log = log_stderr

    configs = json.loads(Path(a.configs).read_text(encoding="utf-8")) if Path(a.configs).exists() else \
        [{"name": "fake-large", "model": "fake"}, {"name": "fake-small", "model": "fake"}]
    files = sorted(p for p in Path(a.docs).iterdir() if p.suffix.lower() in DOC_SUFFIXES)
    docs = {f.name: load_sections(f)[: a.max_sections] if a.max_sections else load_sections(f) for f in files}
    judge = None if a.no_judge else from_env("JUDGE")
    out = Path(a.out)
    runs = out / "runs"
    if a.regenerate and runs.exists():
        dest = out / "archive" / time.strftime("%Y%m%d-%H%M%S")
        dest.mkdir(parents=True, exist_ok=True)
        shutil.move(str(runs), dest / "runs")
        log(f"[regenerate] previous runs moved to {dest}")

    rows, stats = [], []
    for cfg in configs:
        llm = make_llm(cfg)
        verifier = make_llm(cfg["verifier"]) if cfg.get("verifier") else from_env("VERIFY")
        if verifier.name == llm.name:
            log(f"[warn] {cfg['name']}: verifier is the same model as the generator ({llm.name})")
        for strat in a.strategies.split(","):
            for fname, secs in docs.items():
                cell = runs / cfg["name"] / strat / Path(fname).stem
                meta = {"model": llm.name, "verifier": verifier.name, "n": a.n,
                        "max_sections": a.max_sections}
                cell.mkdir(parents=True, exist_ok=True)
                meta_path = cell / "cell_meta.json"
                if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != meta:
                    sys.exit(f"{cell} was run with different settings. Use --regenerate or another --out.")
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                log(f"\n=== {cfg['name']} | {strat} | {fname} (verifier: {verifier.name})")

                b_gen, b_ver, b_jud = _usage(llm), _usage(verifier), _usage(judge)
                qs, _ = run(secs, llm, verifier, a.n, strat, use_st=False, log=log, out_dir=cell)
                judged = judge_cell(judge, qs, secs, cell, log) if judge else {}
                cs = update_cell_stats(cell, _delta(_usage(llm), b_gen), _delta(_usage(verifier), b_ver),
                                       _delta(_usage(judge), b_jud), len(qs), sum(q.ok for q in qs))
                stats.append({"config": cfg["name"], "strategy": strat, "doc": fname, **cs})
                for q in qs:
                    d = q.to_dict()
                    qid = stable_id(cfg["name"], strat, fname, q.qid, q.mcq.question)
                    d.update(id=qid, config=cfg["name"], doc=fname)
                    j = judged.get(q.qid)
                    if j and "error" not in j:
                        d.update({f"judge_{k}": j[k] for k in RUBRIC_KEYS})
                    rows.append(d)

    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df = df[["id"] + [c for c in df.columns if c != "id"]]
    df.to_csv(out / "questions.csv", index=False)

    # Human rating sheet: only verified questions, configuration hidden (blind rating)
    ok = df[df["rejected"].fillna("") == ""]
    sheet = ok[["id", "doc", "section", "question", "options", "answer_index", "explanation", "level",
                "source_quote"]].copy()
    sheet["options"] = sheet["options"].apply(
        lambda o: " | ".join(f"{'ABCD'[i]}. {x}" for i, x in enumerate(o)))
    sheet["answer"] = sheet.pop("answer_index").apply(lambda i: "ABCD"[i])
    for k in RUBRIC_KEYS:
        sheet[k] = ""
    merged = merge_labeling_sheet(out / "labeling_sheet.csv", sheet, log=log)
    merged.to_csv(out / "labeling_sheet.csv", index=False)
    ok[["id", "config", "strategy"]].to_csv(out / "key.csv", index=False)
    if "judge_factual" in df:
        jl = df.dropna(subset=["judge_factual"])[["id"] + [f"judge_{k}" for k in RUBRIC_KEYS]]
        jl.columns = ["id"] + RUBRIC_KEYS
        jl.astype({k: int for k in RUBRIC_KEYS}).to_csv(out / "judge_labels.csv", index=False)

    # Comparison table per (config, strategy), summed over documents
    st = pd.DataFrame(stats)
    num = [c for c in st.columns if c.startswith(("gen_", "verify_", "judge_"))] + ["questions", "accepted"]
    comp = st.groupby(["config", "strategy"])[num].sum().reset_index()
    comp["rejected_rate"] = (1 - comp["accepted"] / comp["questions"].clip(lower=1)).round(3)
    comp["gen_usd_per_100q"] = (100 * comp["gen_cost_usd"] / comp["questions"].clip(lower=1)).round(4)
    if "judge_factual" in df:
        jm = df.groupby(["config", "strategy"])[[f"judge_{k}" for k in RUBRIC_KEYS]].mean().round(3)
        comp = comp.merge(jm.reset_index(), on=["config", "strategy"], how="left")
    cols = ["config", "strategy", "questions", "accepted", "rejected_rate", "gen_cost_usd",
            "gen_usd_per_100q", "gen_seconds", "verify_seconds", "judge_cost_usd"]
    cols += [f"judge_{k}" for k in RUBRIC_KEYS]
    comp = comp[[c for c in dict.fromkeys(cols) if c in comp.columns]]
    (out / "comparison.md").write_text(comp.to_markdown(index=False), encoding="utf-8")
    print(comp.to_markdown(index=False))
    print(f"\nWrote {out}/ - rate labeling_sheet.csv by hand (0/1 in each rubric column).")


if __name__ == "__main__":
    main()
