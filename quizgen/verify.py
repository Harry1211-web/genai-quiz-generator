"""Bước 5 – Kiểm chứng tự động trước khi đưa câu hỏi cho người dùng.

Từ rẻ đến đắt:
1. quote_not_found : source_quote không có trong tài liệu (model bịa căn cứ)
2. answer_leak     : đáp án đúng dài bất thường so với nhiễu (dễ đoán)
3. answer_mismatch : model kiểm tra, chỉ đọc đoạn nguồn, chọn đáp án khác
4. unanswerable    : model kiểm tra nói không trả lời được từ đoạn nguồn
(duplicate do dedupe.py gắn)
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
    """Khớp mờ: model hay sửa nhẹ dấu câu/khoảng trắng khi trích."""
    q, s = _norm(quote), _norm(source)
    if not q:
        return False
    if q in s:
        return True
    # trượt cửa sổ cùng độ dài, lấy tỉ lệ giống cao nhất
    n = len(q)
    step = max(1, n // 4)
    best = 0.0
    for i in range(0, max(1, len(s) - n + 1), step):
        best = max(best, SequenceMatcher(None, q, s[i : i + n]).ratio())
        if best >= threshold:
            return True
    return False


def find_context(quote: str, source: str, window: int = 600) -> str:
    """Lấy đoạn quanh câu trích để model kiểm tra đọc (không đưa cả bài)."""
    idx = _norm(source).find(_norm(quote)[:40])
    if idx == -1:
        return source[: window * 2]
    # vị trí trên chuỗi chuẩn hóa xấp xỉ vị trí gốc – đủ dùng
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
    """verifier nên là model KHÁC model sinh (hoặc ít nhất temperature 0) để giảm
    'tự chấm bài mình'. Bỏ qua bước 3–4 nếu verifier=None (tiết kiệm)."""
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
