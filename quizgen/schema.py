"""Step 3 - Schemas (Pydantic). The LLM must return JSON that matches these schemas.

Pydantic both validates the output and generates a JSON Schema that is inserted into the
prompt (or passed to the provider's structured-output / tool-use feature).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Level = Literal["remember", "understand", "apply"]


class KeyPoint(BaseModel):
    """A key point that can be tested (the 'plan' step of prompt v3)."""

    point: str = Field(min_length=5, description="The key point, one sentence")
    source_quote: str = Field(min_length=5,
                              description="One sentence or bullet copied EXACTLY from the document")
    suggested_level: Level


class KeyPointList(BaseModel):
    points: list[KeyPoint] = Field(min_length=1, max_length=15)


class MCQ(BaseModel):
    question: str = Field(min_length=10)
    options: list[str] = Field(min_length=4, max_length=4)
    answer_index: int = Field(ge=0, le=3, description="Position of the correct option: 0=A ... 3=D")
    explanation: str = Field(min_length=10,
                             description="Why the answer is right and why each other option is wrong")
    level: Level
    source_quote: str = Field(min_length=5,
                              description="Supporting sentence or bullet copied EXACTLY from the document")
    source_page: int | None = None

    @model_validator(mode="after")
    def distinct_options(self) -> MCQ:
        norm = {o.strip().lower().rstrip(".") for o in self.options}
        if len(norm) < 4:
            raise ValueError("options must be distinct")
        if any(not o.strip() for o in self.options):
            raise ValueError("empty option")
        # banned in both supported languages
        banned = {"all of the above", "none of the above", "tất cả các đáp án trên", "cả a và b"}
        if norm & banned:
            raise ValueError("do not use options like 'all of the above'")
        return self

    @property
    def answer(self) -> str:
        return self.options[self.answer_index]


class MCQList(BaseModel):
    questions: list[MCQ] = Field(min_length=1)


class VerifyAnswer(BaseModel):
    """The verifier model answers the question again using ONLY the source excerpt."""

    answer_index: int = Field(ge=-1, le=3, description="-1 if it cannot be answered from the excerpt")
    reason: str


class JudgeScore(BaseModel):
    """LLM-as-judge score on the 5-criterion rubric (0/1 per criterion)."""

    factual: int = Field(ge=0, le=1)
    answerable: int = Field(ge=0, le=1)
    single_correct: int = Field(ge=0, le=1)
    distractors: int = Field(ge=0, le=1)
    level_match: int = Field(ge=0, le=1)
    comment: str = ""


RUBRIC_KEYS = ["factual", "answerable", "single_correct", "distractors", "level_match"]
