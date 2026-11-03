"""Step 5 - Generate questions with 3 prompting strategies.

v1 zero-shot      : 1 call, no examples
v2 few-shot       : 1 call, 3 good examples
v3 plan -> generate: call 1 extracts key points, call 2 writes one question per point
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .extract import Section
from .llm import LLM, generate_json
from .schema import MCQ, KeyPointList, MCQList

PROMPT_DIR = Path(__file__).parent / "prompts"
STRATEGIES = ("v1", "v2", "v3")


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def _examples() -> str:
    data = json.loads(load_prompt("fewshot_examples.json"))
    return json.dumps(data, ensure_ascii=False, indent=1)


def _schema(model) -> str:
    return json.dumps(model.model_json_schema(), ensure_ascii=False)


@dataclass
class GeneratedQuestion:
    mcq: MCQ
    section_title: str
    strategy: str
    model: str
    page_start: int
    page_end: int
    rejected: list[str] = field(default_factory=list)  # reasons the verification step rejected it
    qid: str = ""  # stable id "<section index>-<question index>", used to resume interrupted runs
    verifier_answer: int | None = None  # option chosen by the verifier (-1 = cannot answer)
    verifier_reason: str = ""  # the verifier's one-sentence reason (or the error message)

    @property
    def ok(self) -> bool:
        return not self.rejected

    def to_dict(self) -> dict:
        d = {"qid": self.qid, **self.mcq.model_dump()}
        d.update(section=self.section_title, strategy=self.strategy, model=self.model,
                 page_start=self.page_start, page_end=self.page_end,
                 rejected=";".join(self.rejected),
                 verifier_answer=self.verifier_answer, verifier_reason=self.verifier_reason)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> GeneratedQuestion:
        """Rebuild a question saved with to_dict() (used when resuming from a checkpoint)."""
        mcq = MCQ(**{k: d[k] for k in MCQ.model_fields if k in d})
        rejected = [r for r in (d.get("rejected") or "").split(";") if r]
        return cls(mcq, d["section"], d["strategy"], d["model"], d["page_start"], d["page_end"],
                   rejected, d.get("qid", ""), d.get("verifier_answer"), d.get("verifier_reason", ""))


def generate_for_section(llm: LLM, section: Section, n: int = 3, strategy: str = "v3",
                         levels: str = "remember, understand, apply",
                         temperature: float = 0.3, seed: int | None = None) -> list[GeneratedQuestion]:
    system = {"role": "system", "content": load_prompt("system.md")}
    fmt = {"n": n, "levels": levels, "title": section.title, "text": section.text,
           "schema": _schema(MCQList), "examples": _examples()}
    if strategy == "v1":
        user = load_prompt("v1_zeroshot.md").format(**fmt)
    elif strategy == "v2":
        user = load_prompt("v2_fewshot.md").format(**fmt)
    elif strategy == "v3":
        plan_prompt = load_prompt("v3_plan.md").format(k=n, title=section.title, text=section.text,
                                                       schema=_schema(KeyPointList))
        plan = generate_json(llm, [system, {"role": "user", "content": plan_prompt}], KeyPointList,
                             temperature=0.0)
        if plan is None:
            return []
        points = "\n".join(f"{i + 1}. [{p.suggested_level}] {p.point}  (evidence: \"{p.source_quote}\")"
                           for i, p in enumerate(plan.points[:n]))
        user = load_prompt("v3_generate.md").format(points=points, **fmt)
    else:
        raise ValueError(f"strategy must be one of {STRATEGIES}")

    result = generate_json(llm, [system, {"role": "user", "content": user}], MCQList,
                           temperature=temperature)
    if result is None:
        return []
    rng = random.Random(seed)
    out = []
    for q in result.questions[:n]:
        q = shuffle_options(q, rng)
        if q.source_page is None:
            q.source_page = section.page_start
        out.append(GeneratedQuestion(q, section.title, strategy, llm.name,
                                     section.page_start, section.page_end))
    return out


def shuffle_options(q: MCQ, rng: random.Random) -> MCQ:
    """Models tend to put the correct answer in position A -> shuffle so positions are balanced."""
    order = list(range(4))
    rng.shuffle(order)
    options = [q.options[i] for i in order]
    return q.model_copy(update={"options": options, "answer_index": order.index(q.answer_index)})
