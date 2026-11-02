"""Tạo dữ liệu fine-tune (SFT) từ câu hỏi ĐÃ ĐƯỢC NGƯỜI DUYỆT (all 5 tiêu chí = 1).

QUAN TRỌNG – chống rò rỉ: chỉ lấy câu từ các tài liệu KHÔNG nằm trong tập đánh giá.
Chạy: python -m eval.make_sft_data reports/eval --exclude-docs mang_may_tinh_tcp.md
Đầu ra: data/sft/train.jsonl (định dạng chat messages)
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
            user = (f"Soạn 1 câu hỏi trắc nghiệm mức {r['level']} từ tài liệu.\n"
                    f"<document>\n{sec.text}\n</document>")
            f.write(json.dumps({"messages": [
                {"role": "system", "content": system}, {"role": "user", "content": user},
                {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}]},
                ensure_ascii=False) + "\n")
            n += 1
    print(f"Ghi {n} mẫu vào {out}. Cần ít nhất ~300 mẫu để LoRA có ý nghĩa.")


if __name__ == "__main__":
    main()
