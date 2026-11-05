"""Measure a local Ollama model as verifier, with thinking on (model default) vs off.

Usage:
  python scripts/ollama_speed.py reports/s1/v2/all_questions.jsonl data/private/Chapter4_PaaS.pdf --n 3
  python scripts/ollama_speed.py <questions.jsonl> <lecture> --levels understand,apply --per-level 5

--levels / --per-level pick that many questions of each level, spread over different sections.
Each question is verified twice (thinking on, then off). The table reports seconds, output tokens and
how often the verifier agreed with the generator's answer, overall and per level.
Nothing is written to disk (pipe through Tee-Object to keep the output).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow "python scripts/ollama_speed.py"

from quizgen.extract import load_sections  # noqa: E402
from quizgen.generate import GeneratedQuestion  # noqa: E402
from quizgen.llm import FakeLLM, OllamaLLM  # noqa: E402
from quizgen.verify import check_answer_back  # noqa: E402


def pick(qs: list[GeneratedQuestion], levels: list[str] | None, per_level: int,
         n: int) -> list[GeneratedQuestion]:
    if not levels:
        return qs[:n]
    out = []
    for level in levels:
        same = [q for q in qs if q.mcq.level == level]
        if not same:
            print(f"[warn] no '{level}' questions in this file", flush=True)
            continue
        step = max(1, len(same) // per_level)          # spread over different sections
        out += same[::step][:per_level]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", help="all_questions.jsonl or generated.jsonl from a previous run")
    ap.add_argument("lecture", help="the lecture file those questions were generated from")
    ap.add_argument("--model", default="qwen3.5:4b", help='"fake" for a dry run without Ollama')
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--n", type=int, default=3, help="first N questions (when --levels is not given)")
    ap.add_argument("--levels", default=None, help="e.g. understand,apply")
    ap.add_argument("--per-level", type=int, default=5)
    a = ap.parse_args()

    sections = {s.title: s for s in load_sections(a.lecture)}
    rows = [json.loads(x) for x in Path(a.questions).read_text(encoding="utf-8").splitlines() if x.strip()]
    qs = [GeneratedQuestion.from_dict(r) for r in rows if not r.get("empty")]
    qs = pick(qs, a.levels.split(",") if a.levels else None, a.per_level, a.n)
    print(f"{len(qs)} questions: " + ", ".join(f"{q.qid}({q.mcq.level})" for q in qs), flush=True)

    summary = {}
    for label, think in (("thinking ON (default)", None), ("thinking OFF", False)):
        llm = FakeLLM() if a.model == "fake" else OllamaLLM(model=a.model, base_url=a.base_url,
                                                           think=think, timeout=900)
        per_level: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        verdicts = {}
        for q in qs:
            reasons = check_answer_back(llm, q, sections[q.section_title])
            per_level[q.mcq.level][0] += not reasons
            per_level[q.mcq.level][1] += 1
            verdicts[q.qid] = q.verifier_answer
            print(f"[{label}] {q.qid} ({q.mcq.level}): verifier={q.verifier_answer} "
                  f"generator={q.mcq.answer_index} -> {reasons or 'ok'}", flush=True)
        u = llm.usage
        summary[label] = (u.seconds / max(u.calls, 1), u.completion_tokens / max(u.calls, 1),
                          dict(per_level), verdicts)

    print(f"\n{'mode':24s} {'s/question':>11s} {'out tokens/q':>13s}   agreed with generator")
    for label, (sec, tok, per_level, _) in summary.items():
        agreed = " | ".join(f"{lv} {ok}/{tot}" for lv, (ok, tot) in per_level.items())
        total = sum(ok for ok, _ in per_level.values())
        print(f"{label:24s} {sec:11.1f} {tok:13.0f}   {total}/{len(qs)}  ({agreed})")
    on, off = (summary[k][3] for k in summary)
    differ = [qid for qid in on if on[qid] != off[qid]]
    print(f"\nSame verifier answer in both modes: {len(qs) - len(differ)}/{len(qs)}"
          + (f" - different for: {', '.join(differ)}" if differ else ""))


if __name__ == "__main__":
    main()
