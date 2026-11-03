"""Step 7 - Export to JSON, CSV and GIFT (Moodle's question import format)."""
from __future__ import annotations

import csv
import io
import json

from .schema import MCQ

GIFT_SPECIAL = "~=#{}:"


def gift_escape(s: str) -> str:
    s = s.replace("\\", "\\\\")
    for ch in GIFT_SPECIAL:
        s = s.replace(ch, "\\" + ch)
    return s.replace("\n", " ")


def to_gift(questions: list[MCQ], prefix: str = "q") -> str:
    """Example of one question:
    ::q1:: Default port of HTTPS? {=443 ~80 ~8080 ~22 ####Explanation}
    """
    blocks = []
    for i, q in enumerate(questions, 1):
        opts = " ".join(("=" if j == q.answer_index else "~") + gift_escape(o)
                        for j, o in enumerate(q.options))
        blocks.append(f"::{prefix}{i}:: {gift_escape(q.question)} {{\n  {opts}\n"
                      f"  ####{gift_escape(q.explanation)}\n}}")
    return "\n\n".join(blocks) + "\n"


def to_json(questions: list[MCQ]) -> str:
    return json.dumps([q.model_dump() for q in questions], ensure_ascii=False, indent=2)


def to_csv(questions: list[MCQ]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["question", "A", "B", "C", "D", "answer", "explanation", "level", "source_quote",
                "source_page"])
    for q in questions:
        w.writerow([q.question, *q.options, "ABCD"[q.answer_index], q.explanation, q.level,
                    q.source_quote, q.source_page])
    return buf.getvalue()
