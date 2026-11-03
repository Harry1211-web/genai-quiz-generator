"""After rating labeling_sheet.csv by hand: join with key.csv -> quality table per
model x strategy (HUMAN scores) + judge-vs-human kappa.

Run: python -m eval.report reports/eval
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from quizgen.schema import RUBRIC_KEYS

from .kappa import kappa_table


def main(folder: str) -> None:
    d = Path(folder)
    human = pd.read_csv(d / "labeling_sheet.csv").dropna(subset=RUBRIC_KEYS)
    key = pd.read_csv(d / "key.csv")
    m = human.merge(key, on="id")
    m["all_pass"] = (m[RUBRIC_KEYS].sum(axis=1) == len(RUBRIC_KEYS)).astype(int)
    q = m.groupby(["config", "strategy"])[RUBRIC_KEYS + ["all_pass"]].mean().round(3)
    q["n_rated"] = m.groupby(["config", "strategy"]).size()
    text = "## Quality (human ratings)\n\n" + q.reset_index().to_markdown(index=False)
    if (d / "judge_labels.csv").exists():
        kt = kappa_table(human[["id"] + RUBRIC_KEYS], pd.read_csv(d / "judge_labels.csv"))
        text += "\n\n## LLM-as-judge agreement with human ratings\n\n" + kt.to_markdown(index=False)
    (d / "quality_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "reports/eval")
