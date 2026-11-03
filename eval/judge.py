"""LLM-as-judge: grade questions with the same rubric as the human rater."""
from __future__ import annotations

import json

from quizgen.extract import Section
from quizgen.generate import GeneratedQuestion, load_prompt
from quizgen.llm import LLM, generate_json
from quizgen.schema import JudgeScore
from quizgen.verify import find_context


def judge_one(llm: LLM, q: GeneratedQuestion, section: Section) -> JudgeScore | None:
    m = q.mcq
    payload = {"question": m.question, "options": m.options, "answer_index": m.answer_index,
               "explanation": m.explanation}
    prompt = load_prompt("judge.md").format(level=m.level,
                                            source=find_context(m.source_quote, section.text, 900),
                                            question_json=json.dumps(payload, ensure_ascii=False))
    return generate_json(llm, [{"role": "user", "content": prompt}], JudgeScore, temperature=0.0)
