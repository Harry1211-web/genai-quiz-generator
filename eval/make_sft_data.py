"""Build fine-tuning (SFT) data from HUMAN-APPROVED questions (all 5 rubric criteria = 1).

IMPORTANT - avoid leakage: only use questions from documents that are NOT in the evaluation set.
Run: python -m eval.make_sft_data reports/eval --exclude-docs transport_layer_tcp_udp.md
Output: data/sft/train.jsonl (chat-messages format)
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

from quizgen.extract import load_sections
from quizgen.generate import load_prompt
from quizgen.schema import RUBRIC_KEYS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("--docs", default="data/samples")
    ap.add_argument("--exclude-docs", nargs="*", default=[])
    ap.add_argument("--out", default="data/sft/train.jsonl")
    a = ap.parse_args()
    d = Path(a.eval_dir)
    lab = pd.read_csv(d / "labeling_sheet.csv").dropna(subset=RUBRIC_KEYS)
    q = pd.read_csv(d / "questions.csv")
    good = lab[lab[RUBRIC_KEYS].sum(axis=1) == 5][["id"]].merge(q, on="id")
    good = good[~good["doc"].isin(a.exclude_docs)]
    secs = {(f, s.title): s for f in good["doc"].unique()
            for s in load_sections(Path(a.docs) / f)}
    system = load_prompt("system.md")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for _, r in good.iterrows():
            sec = secs.get((r["doc"], r["section"]))
            if sec is None:
                continue
            target = {"questions": [{k: (ast.literal_eval(r[k]) if k == "options" else r[k]) for k in
                                     ["question", "options", "answer_index", "explanation", "level",
                                      "source_quote"]}]}
            target["questions"][0]["answer_index"] = int(target["questions"][0]["answer_index"])
            user = (f"Write 1 multiple-choice question at level {r['level']} from the document.\n"
                    f"<document>\n{sec.text}\n</document>")
            f.write(json.dumps({"messages": [
                {"role": "system", "content": system}, {"role": "user", "content": user},
                {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}]},
                ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} examples to {out}. LoRA needs at least ~300 examples to be meaningful.")


if __name__ == "__main__":
    main()
