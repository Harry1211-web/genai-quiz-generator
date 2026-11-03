"""Glue for all steps: extract -> generate -> verify -> dedupe. Used by the CLI, the API and the UI."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .dedupe import mark_duplicates
from .export import to_csv, to_gift, to_json
from .extract import Section, load_sections
from .generate import STRATEGIES, GeneratedQuestion, generate_for_section
from .llm import LLM, from_env
from .verify import rejection_report, verify_all


def run(sections: list[Section], llm: LLM, verifier: LLM | None = None, n_per_section: int = 3,
        strategy: str = "v3", max_sections: int | None = None, use_st: bool = True,
        seed: int = 0) -> tuple[list[GeneratedQuestion], dict]:
    t0 = time.perf_counter()
    sections = sections[:max_sections] if max_sections else sections
    qs: list[GeneratedQuestion] = []
    for i, sec in enumerate(sections):
        qs += generate_for_section(llm, sec, n=n_per_section, strategy=strategy, seed=seed + i)
    by_title = {s.title: s for s in sections}
    verify_all(qs, by_title, verifier)
    mark_duplicates(qs, use_st=use_st)
    stats = rejection_report(qs)
    stats.update(model=llm.name, strategy=strategy, sections=len(sections),
                 seconds=round(time.perf_counter() - t0, 2), llm_calls=llm.usage.calls,
                 format_errors=llm.usage.format_errors, format_failures=llm.usage.format_failures,
                 cost_usd=round(llm.cost_usd() + (verifier.cost_usd() if verifier else 0), 4))
    return qs, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate review questions from a lecture")
    ap.add_argument("file", help="PDF / PPTX / TXT / MD")
    ap.add_argument("--strategy", choices=STRATEGIES, default="v3")
    ap.add_argument("--n", type=int, default=3, help="questions per section")
    ap.add_argument("--max-sections", type=int)
    ap.add_argument("--no-verify", action="store_true", help="skip the answer-back verification (cheaper)")
    ap.add_argument("--out", default="reports/output")
    a = ap.parse_args()

    llm = from_env("LLM")
    verifier = None if a.no_verify else from_env("VERIFY")
    sections = load_sections(a.file)
    print(f"{len(sections)} sections: " + ", ".join(s.title for s in sections[:8]))
    qs, stats = run(sections, llm, verifier, a.n, a.strategy, a.max_sections)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    good = [q.mcq for q in qs if q.ok]
    (out / "all_questions.jsonl").write_text(
        "\n".join(json.dumps(q.to_dict(), ensure_ascii=False) for q in qs), encoding="utf-8")
    (out / "quiz.json").write_text(to_json(good), encoding="utf-8")
    (out / "quiz.csv").write_text(to_csv(good), encoding="utf-8")
    (out / "quiz.gift.txt").write_text(to_gift(good), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
