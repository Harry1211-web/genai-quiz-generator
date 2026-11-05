"""Step 6 - Automatic verification before questions reach the user.

Rejections, from cheapest to most expensive:
1. quote_not_found : a sentence of source_quote is not in the document (the model invented its evidence).
                     Multi-sentence quotes are checked sentence by sentence, so a quote that combines two
                     bullets from different places in the slides is accepted when every sentence exists.
2. answer_mismatch : a verifier model, reading only the source excerpt, picks a different option
3. unanswerable    : the verifier says no option (or more than one) is supported by the excerpt
4. verify_error    : the verifier call failed (timeout, server down...) - retried on the next run
(duplicate is added by dedupe.py, not_verified by pipeline.py)

Warnings (shown to the reviewer, the question is NOT rejected):
- answer_leak      : the correct option is much longer than every distractor (may be easy to guess)
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


# sentence ends, line breaks, bullet symbols, and quoted items joined as "...", "..." (seen with Qwen)
QUOTE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\n+|\s*[✓•●▪➢❖■◆]\s*|[\"“”]\s*[,;]\s*[\"“”]")
QUOTE_CHARS = "\"'“”‘’ "


def quote_segments(quote: str, min_words: int = 3) -> list[str]:
    """Split a quote into sentences / bullet items; fragments shorter than min_words are ignored."""
    parts = (p.strip().strip(QUOTE_CHARS) for p in QUOTE_SPLIT_RE.split(quote))
    return [p for p in parts if len(p.split()) >= min_words]


def _match(q: str, s: str, threshold: float) -> bool:
    if not q:
        return False
    if q in s:
        return True
    # 1) coarse scan: slide a window of the same length over the source with a big step
    n = len(q)
    step = max(1, n // 4)
    last = max(1, len(s) - n + 1)
    coarse = []
    for i in range(0, last, step):
        r = SequenceMatcher(None, q, s[i : i + n]).ratio()
        if r >= threshold:
            return True
        coarse.append((r, i))
    # 2) fine scan around the best coarse positions: the big step can miss the exact alignment
    #    (e.g. the model fixed a typo of the slide, so the text is shifted by a few characters)
    for _, i in sorted(coarse, reverse=True)[:3]:
        for j in range(max(0, i - step), min(last, i + step + 1)):
            if SequenceMatcher(None, q, s[j : j + n]).ratio() >= threshold:
                return True
    return False


def quote_in_source(quote: str, source: str, threshold: float = 0.85) -> bool:
    """Fuzzy match: models often change punctuation or whitespace slightly when quoting.
    A quote made of several sentences passes if EVERY sentence is found (they may come from
    different places in the document); a single invented sentence makes it fail."""
    s = _norm(source)
    if _match(_norm(quote), s, threshold):
        return True
    segments = quote_segments(quote)
    if len(segments) < 2:
        return False
    return all(_match(_norm(seg), s, threshold) for seg in segments)


def _locate(segment: str, source: str) -> int:
    """Position of a segment in the ORIGINAL text, tolerant to line breaks and punctuation."""
    words = re.findall(r"\w+", unicodedata.normalize("NFC", segment))[:6]
    if not words:
        return -1
    m = re.search(r"\W+".join(map(re.escape, words)), source, re.IGNORECASE)
    return m.start() if m else -1


def find_context(quote: str, source: str, window: int = 600) -> str:
    """Return the text around the quote for the verifier/judge (not the whole section).
    For a multi-sentence quote, the windows around every sentence are included."""
    starts = sorted({p for p in (_locate(seg, source) for seg in (quote_segments(quote) or [quote]))
                     if p >= 0})
    if not starts:
        return source[: window * 2]
    spans: list[list[int]] = []
    for p in starts:
        lo, hi = max(0, p - window), min(len(source), p + window)
        if spans and lo <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], hi)
        else:
            spans.append([lo, hi])
    return "\n...\n".join(source[lo:hi] for lo, hi in spans)


def check_static(q: GeneratedQuestion, section: Section) -> list[str]:
    """Free checks that REJECT a question."""
    return [] if quote_in_source(q.mcq.source_quote, section.text) else ["quote_not_found"]


def check_warnings(q: GeneratedQuestion) -> list[str]:
    """Free checks that only WARN: the reviewer decides."""
    lens = [len(o) for o in q.mcq.options]
    others = [x for i, x in enumerate(lens) if i != q.mcq.answer_index]
    if lens[q.mcq.answer_index] > 2.0 * max(others) and lens[q.mcq.answer_index] > 40:
        return ["answer_leak"]
    return []


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
    q.warnings = check_warnings(q)
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
        for w in q.warnings:
            rep[f"warning_{w}"] = rep.get(f"warning_{w}", 0) + 1
    return rep
