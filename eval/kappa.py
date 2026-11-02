"""Cohen's kappa giữa chấm tay và LLM-as-judge, theo từng tiêu chí rubric.

kappa = (p_o - p_e) / (1 - p_e)
  p_o: tỉ lệ hai bên chấm giống nhau
  p_e: tỉ lệ giống nhau do ngẫu nhiên (từ phân bố điểm của mỗi bên)
Đọc: < 0,4 kém · 0,4–0,6 vừa · 0,6–0,8 tốt · > 0,8 rất tốt

Chạy: python -m eval.kappa eval/human_labels.csv eval/judge_labels.csv
Hai file cùng cột id + 5 cột rubric.
"""
from __future__ import annotations

import sys

import pandas as pd

from quizgen.schema import RUBRIC_KEYS


def cohen_kappa(a: list[int], b: list[int]) -> float:
    assert len(a) == len(b) and a, "hai danh sách phải cùng độ dài, không rỗng"
    n = len(a)
    labels = sorted(set(a) | set(b))
    p_o = sum(x == y for x, y in zip(a, b, strict=True)) / n
    p_e = sum((a.count(c) / n) * (b.count(c) / n) for c in labels)
    if p_e == 1:  # cả hai chấm toàn một giá trị
        return 1.0 if p_o == 1 else 0.0
    return (p_o - p_e) / (1 - p_e)


def kappa_table(human: pd.DataFrame, judge: pd.DataFrame) -> pd.DataFrame:
    m = human.merge(judge, on="id", suffixes=("_h", "_j"))
    rows = []
    for k in RUBRIC_KEYS:
        a, b = m[f"{k}_h"].astype(int).tolist(), m[f"{k}_j"].astype(int).tolist()
        rows.append({"criterion": k, "n": len(a),
                     "agreement": round(sum(x == y for x, y in zip(a, b, strict=True)) / len(a), 3),
                     "kappa": round(cohen_kappa(a, b), 3),
                     "human_mean": round(sum(a) / len(a), 3), "judge_mean": round(sum(b) / len(b), 3)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(kappa_table(pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])).to_markdown(index=False))
