"""Step 6 - Automatic verification before questions reach the user.

From cheapest to most expensive:
1. quote_not_found : source_quote is not in the document (the model invented its evidence)
2. answer_leak     : the correct option is unusually long compared with the distractors (easy to guess)
3. answer_mismatch : a verifier model, reading only the source excerpt, picks a different option
4. unanswerable    : the verifier model says the excerpt is not enough to answer
5. verify_error    : the verifier call failed (timeout, server down...) - retried on the next run
(duplicate is added by dedupe.py)
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from collections.abc import Callable
from difflib import SequenceMatcher

import httpx

from .extract import Section
from .generate import GeneratedQuestion, load_prompt
from .llm import LLM, BudgetExceeded, LLMRequestError, generate_json
from .schema import VerifyAnswer

MAX_CONSECUTIVE_ERRORS = 3  # stop verifying when the verifier keeps failing (e.g. Ollama is down)


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
    """Ask the verifier to answer from the excerpt only. Stores its answer and reason on q."""
    ctx = find_context(q.mcq.source_quote, section.text)
    payload = {"question": q.mcq.question, "options": q.mcq.options}
    prompt = load_prompt("verify.md").format(source=ctx,
                                             question_json=json.dumps(payload, ensure_ascii=False))
    res = generate_json(llm, [{"role": "user", "content": prompt}], VerifyAnswer, temperature=0.0)
    if res is None:
        return ["verify_format_error"]
    q.verifier_answer, q.verifier_reason = res.answer_index, res.reason
    if res.answer_index == -1:
        return ["unanswerable"]
    if res.answer_index != q.mcq.answer_index:
        return ["answer_mismatch"]
    return []


def verify_one(q: GeneratedQuestion, section: Section, verifier: LLM | None) -> None:
    """Run all checks on one question. A failed verifier call is recorded as 'verify_error'
    instead of crashing the whole run, so earlier (paid) work is never lost."""
    q.rejected = check_static(q, section)
    q.verifier_answer, q.verifier_reason = None, ""
    if verifier is None or "quote_not_found" in q.rejected:
        return
    try:
        q.rejected += check_answer_back(verifier, q, section)
    except BudgetExceeded:
        raise
    except (httpx.HTTPError, LLMRequestError) as e:
        q.rejected.append("verify_error")
        q.verifier_reason = f"{type(e).__name__}: {e}"[:300]


def verify_all(questions: list[GeneratedQuestion], sections: dict[str, Section],
               verifier: LLM | None = None, log: Callable[[str], None] = lambda msg: None,
               on_done: Callable[[GeneratedQuestion], None] | None = None,
               skip: set[str] | None = None) -> list[GeneratedQuestion]:
    """The verifier should be a DIFFERENT model from the generator (or at least run at
    temperature 0) so a model does not grade its own work. Steps 3-4 are skipped when
    verifier=None (cheaper).

    on_done: called after each question (used to save progress immediately).
    skip: qids already verified in a previous run (their stored result is kept)."""
    skip = skip or set()
    errors_in_a_row = 0
    for i, q in enumerate(questions, 1):
        if q.qid and q.qid in skip:
            continue
        ts = time.perf_counter()
        verify_one(q, sections[q.section_title], verifier)
        who = verifier.name if verifier is not None and "quote_not_found" not in q.rejected else "static"
        log(f"[verify {i}/{len(questions)}] {who}: {time.perf_counter() - ts:.1f}s -> "
            f"{', '.join(q.rejected) or 'ok'}")
        if on_done is not None:
            on_done(q)
        errors_in_a_row = errors_in_a_row + 1 if "verify_error" in q.rejected else 0
        if errors_in_a_row >= MAX_CONSECUTIVE_ERRORS:
            log(f"[verify] stopped: {MAX_CONSECUTIVE_ERRORS} verifier errors in a row "
                f"({q.verifier_reason}). Fix the verifier and run the same command again to resume.")
            break
    return questions


def rejection_report(questions: list[GeneratedQuestion]) -> dict[str, int]:
    rep: dict[str, int] = {"total": len(questions), "accepted": sum(q.ok for q in questions)}
    for q in questions:
        for r in q.rejected:
            rep[r] = rep.get(r, 0) + 1
    return rep
