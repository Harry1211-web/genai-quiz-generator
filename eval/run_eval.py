"""Main experiment: every model x every prompting strategy on the same documents.

Outputs:
  reports/eval/questions.csv      - every generated question (including rejected ones), with an id
  reports/eval/labeling_sheet.csv - HUMAN rating sheet: shuffled, model/strategy HIDDEN (blind rating)
  reports/eval/key.csv            - id -> model/strategy key (do not open while rating)
  reports/eval/judge_labels.csv   - LLM-as-judge scores
  reports/eval/comparison.md      - comparison table

Models are configured in eval/configs.json, for example:
[
  {"name": "large", "model": "gpt-4o-mini", "base_url": "https://api.openai.com/v1",
   "api_key_env": "OPENAI_API_KEY", "price_in": 0.15, "price_out": 0.6},
  {"name": "small", "model": "qwen2.5:7b-instruct", "base_url": "http://localhost:11434/v1"}
]
Run: python -m eval.run_eval data/samples --n 3 --max-sections 6
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from quizgen.extract import load_sections
from quizgen.llm import FakeLLM, OpenAICompatLLM, from_env
from quizgen.pipeline import run
from quizgen.schema import RUBRIC_KEYS

from .judge import judge_one


def make_llm(cfg: dict):
    if cfg["model"] == "fake":
        return FakeLLM(name=cfg.get("name", "fake"))
    return OpenAICompatLLM(model=cfg["model"], base_url=cfg.get("base_url", "https://api.openai.com/v1"),
                           api_key=os.getenv(cfg.get("api_key_env", ""), ""),
                           price_in=cfg.get("price_in", 0), price_out=cfg.get("price_out", 0),
                           budget_usd=float(os.getenv("LLM_BUDGET_USD", "2")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("docs", help="folder of documents (pdf/pptx/txt/md)")
    ap.add_argument("--configs", default="eval/configs.json")
    ap.add_argument("--strategies", default="v1,v2,v3")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-sections", type=int, default=6)
    ap.add_argument("--out", default="reports/eval")
    ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()

    configs = json.loads(Path(a.configs).read_text()) if Path(a.configs).exists() else \
        [{"name": "fake-large", "model": "fake"}, {"name": "fake-small", "model": "fake"}]
    files = sorted(p for p in Path(a.docs).iterdir() if p.suffix.lower() in {".pdf", ".pptx", ".txt", ".md"})
    docs = {f.name: load_sections(f)[: a.max_sections] for f in files}
    verifier = from_env("VERIFY")
    judge = from_env("JUDGE")

    rows, stats = [], []
    for cfg in configs:
        for strat in a.strategies.split(","):
            llm = make_llm(cfg)
            for fname, secs in docs.items():
                qs, _ = run(secs, llm, verifier, a.n, strat, use_st=False)
                by_title = {s.title: s for s in secs}
                for q in qs:
                    d = q.to_dict()
                    d.update(config=cfg["name"], doc=fname)
                    if not a.no_judge and q.ok:
                        js = judge_one(judge, q, by_title[q.section_title])
                        if js:
                            d.update({f"judge_{k}": getattr(js, k) for k in RUBRIC_KEYS})
                    rows.append(d)
            # usage is accumulated over all documents for each (model, strategy) pair
            stats.append({"config": cfg["name"], "strategy": strat, "calls": llm.usage.calls,
                          "format_errors": llm.usage.format_errors,
                          "format_failures": llm.usage.format_failures,
                          "cost_usd": round(llm.cost_usd(), 4), "seconds": round(llm.usage.seconds, 1)})

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.insert(0, "id", [f"q{i:04d}" for i in range(len(df))])
    df.to_csv(out / "questions.csv", index=False)

    # Human rating sheet: only verified questions, shuffled, configuration hidden (blind rating)
    ok = df[df["rejected"].fillna("") == ""].sample(frac=1, random_state=42)
    sheet = ok[["id", "doc", "section", "question", "options", "answer_index", "explanation", "level",
                "source_quote"]].copy()
    sheet["options"] = sheet["options"].apply(
        lambda o: " | ".join(f"{'ABCD'[i]}. {x}" for i, x in enumerate(o)))
    sheet["answer"] = sheet.pop("answer_index").apply(lambda i: "ABCD"[i])
    for k in RUBRIC_KEYS:
        sheet[k] = ""
    sheet.to_csv(out / "labeling_sheet.csv", index=False)
    ok[["id", "config", "strategy"]].to_csv(out / "key.csv", index=False)
    if not a.no_judge and "judge_factual" in df:
        jl = df.dropna(subset=["judge_factual"])[["id"] + [f"judge_{k}" for k in RUBRIC_KEYS]]
        jl.columns = ["id"] + RUBRIC_KEYS
        jl.to_csv(out / "judge_labels.csv", index=False)

    # Comparison table (automatic checks + judge; once human ratings exist use eval.report)
    g = df.groupby(["config", "strategy"])
    comp = pd.DataFrame({
        "questions": g.size(),
        "rejected_rate": g["rejected"].apply(lambda s: round((s.fillna("") != "").mean(), 3)),
    }).reset_index()
    if "judge_factual" in df:
        jm = df.groupby(["config", "strategy"])[[f"judge_{k}" for k in RUBRIC_KEYS]].mean().round(3)
        comp = comp.merge(jm.reset_index(), on=["config", "strategy"])
    comp = comp.merge(pd.DataFrame(stats), on=["config", "strategy"])
    (out / "comparison.md").write_text(comp.to_markdown(index=False), encoding="utf-8")
    print(comp.to_markdown(index=False))
    print(f"\nWrote {out}/ - rate labeling_sheet.csv by hand (0/1 in each rubric column).")


if __name__ == "__main__":
    main()
