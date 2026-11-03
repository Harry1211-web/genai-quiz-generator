"""Step 6 - Automatic verification before questions reach the user.

From cheapest to most expensive:
1. quote_not_found : source_quote is not in the document (the model invented its evidence)
2. answer_leak     : the correct option is unusually long compared with the distractors (easy to guess)
3. answer_mismatch : a verifier model, reading only the source excerpt, picks a different option
4. unanswerable    : the verifier model says the excerpt is not enough to answer
(duplicate is added by dedupe.py)
"""
from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher

from .extract import Section
from .generate import GeneratedQuestion, load_prompt
from .llm import LLM, generate_json
from .schema import VerifyAnswer


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def quote_in_source(quote: str, source: str, threshold: float = 0.85) -> bool:
    """Fuzzy match: models often change punctuation or whitespace slightly when quoting."""
    q, s = _norm(quote), _norm(source)
    if not q:
        return False
    if q in s:
        return True
    # slide a window of the same length over the source, keep the best similarity ratio
    n = len(q)
    step = max(1, n // 4)
    best = 0.0
    for i in range(0, max(1, len(s) - n + 1), step):
        best = max(best, SequenceMatcher(None, q, s[i : i + n]).ratio())
        if best >= threshold:
            return True
    return False


def find_context(quote: str, source: str, window: int = 600) -> str:
    """Return the text around the quote for the verifier (not the whole section)."""
    idx = _norm(source).find(_norm(quote)[:40])
    if idx == -1:
        return source[: window * 2]
    # the position in the normalized string approximates the original position - good enough
    return source[max(0, idx - window) : idx + window]


def check_static(q: GeneratedQuestion, section: Section) -> list[str]:
    reasons = []
    if not quote_in_source(q.mcq.source_quote, section.text):
        reasons.append("quote_not_found")
    lens = [len(o) for o in q.mcq.options]
    others = [x for i, x in enumerate(lens) if i != q.mcq.answer_index]
    if lens[q.mcq.answer_index] > 2.0 * max(others) and lens[q.mcq.answer_index] > 40:
        reasons.append("answer_leak")
    return reasons


def check_answer_back(llm: LLM, q: GeneratedQuestion, section: Section) -> list[str]:
    ctx = find_context(q.mcq.source_quote, section.text)
    payload = {"question": q.mcq.question, "options": q.mcq.options}
    prompt = load_prompt("verify.md").format(source=ctx,
                                             question_json=json.dumps(payload, ensure_ascii=False))
    res = generate_json(llm, [{"role": "user", "content": prompt}], VerifyAnswer, temperature=0.0)
    if res is None:
        return ["verify_format_error"]
    if res.answer_index == -1:
        return ["unanswerable"]
    if res.answer_index != q.mcq.answer_index:
        return ["answer_mismatch"]
    return []


def verify_all(questions: list[GeneratedQuestion], sections: dict[str, Section],
               verifier: LLM | None = None) -> list[GeneratedQuestion]:
    """The verifier should be a DIFFERENT model from the generator (or at least run at
    temperature 0) so a model does not grade its own work. Steps 3-4 are skipped when
    verifier=None (cheaper)."""
    for q in questions:
        sec = sections[q.section_title]
        q.rejected += check_static(q, sec)
        if verifier is not None and "quote_not_found" not in q.rejected:
            q.rejected += check_answer_back(verifier, q, sec)
    return questions


def rejection_report(questions: list[GeneratedQuestion]) -> dict[str, int]:
    rep: dict[str, int] = {"total": len(questions), "accepted": sum(q.ok for q in questions)}
    for q in questions:
        for r in q.rejected:
            rep[r] = rep.get(r, 0) + 1
    return rep
