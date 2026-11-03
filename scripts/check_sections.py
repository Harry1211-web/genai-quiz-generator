"""Inspect how a document is cleaned and split into sections.

Usage:
  python scripts/check_sections.py data/private/Chapter4_PaaS.pdf [more files...]

Prints a summary and writes reports/sections/<file>.md with:
  1. the lines removed as boilerplate (check that no real content was removed)
  2. a table of sections (pages, tokens, title) with warnings
  3. the full text of every section, to compare side by side with the original slides
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow "python scripts/check_sections.py"

from quizgen.extract import (  # noqa: E402
    MAX_TOKENS_PER_SECTION,
    MIN_TOKENS_PER_SECTION,
    load_pages,
    split_sections,
    strip_boilerplate,
)

OUT_DIR = Path("reports/sections")


def check(path: Path) -> list[str]:
    raw = load_pages(path)
    clean = strip_boilerplate(raw)
    sections = split_sections(raw)

    # 1. what boilerplate removal took out
    removed = Counter()
    for r, c in zip(raw, clean, strict=True):
        kept = {ln.strip() for ln in c.text.split("\n")}
        removed.update(ln.strip() for ln in r.text.split("\n") if ln.strip() and ln.strip() not in kept)

    # 2. coverage: pages that still have text after cleaning but belong to no section.
    # A slide with only a title line usually keeps its content in images/diagrams (not extractable).
    covered = {p for s in sections for p in range(s.page_start, s.page_end + 1)}
    def body_lines(p):
        return [ln for ln in p.text.split("\n") if ln.strip()]
    title_only = [p.number for p in clean if len(body_lines(p)) == 1]
    missing = [p.number for p in clean if len(body_lines(p)) > 1 and p.number not in covered]
    raw_chars = sum(len(p.text) for p in raw)
    kept_chars = sum(len(s.text) for s in sections)

    warnings = []
    if missing:
        warnings.append(f"pages with text but in no section (content lost!): {missing}")
    if title_only:
        warnings.append(f"title-only slides, content probably in images (not extracted): {title_only}")
    for i, s in enumerate(sections, 1):
        if s.n_tokens < MIN_TOKENS_PER_SECTION // 2:
            warnings.append(f"section {i} is very short ({s.n_tokens} tokens): {s.title[:60]}")
        if s.n_tokens > MAX_TOKENS_PER_SECTION * 1.3:
            warnings.append(f"section {i} is very long ({s.n_tokens} tokens): {s.title[:60]}")
    dup_titles = [t for t, n in Counter(s.title for s in sections).items() if n > 1]
    if dup_titles:
        warnings.append(f"repeated section titles (possible header detected as title): {dup_titles[:3]}")

    lines = [f"# {path.name}", "",
             f"- pages: {len(raw)} | sections: {len(sections)} | "
             f"text kept in sections: {kept_chars / max(raw_chars, 1):.0%} of extracted characters",
             f"- tokens per section: min {min((s.n_tokens for s in sections), default=0)}, "
             f"max {max((s.n_tokens for s in sections), default=0)}", ""]
    lines += ["## Warnings", ""] + ([f"- {w}" for w in warnings] or ["- none"]) + [""]
    lines += ["## Removed as boilerplate (line: number of pages)", ""]
    lines += [f"- `{ln[:80]}`: {n}" for ln, n in removed.most_common(15)] or ["- nothing"]
    lines += ["", "## Sections", "", "| # | Pages | Tokens | Title |", "|---|---|---|---|"]
    lines += [f"| {i} | {s.page_start}-{s.page_end} | {s.n_tokens} | {s.title[:90]} |"
              for i, s in enumerate(sections, 1)]
    lines += ["", "## Full text", ""]
    for i, s in enumerate(sections, 1):
        lines += [f"### {i}. {s.title} (pages {s.page_start}-{s.page_end}, {s.n_tokens} tokens)", "",
                  "```text", s.text, "```", ""]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{path.stem}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    summary = lines[:4] + ["Warnings:"] + ([f"  - {w}" for w in warnings] or ["  - none"])
    return summary + [f"Report: {out}", ""]


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print("\n".join(check(Path(arg))))
