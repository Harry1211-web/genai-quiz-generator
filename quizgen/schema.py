"""Bước 2 – Schema (Pydantic). LLM phải trả JSON đúng các schema này.

Pydantic vừa kiểm tra định dạng, vừa sinh JSON Schema để đưa vào prompt
(hoặc vào tính năng structured output / tool use của nhà cung cấp).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Level = Literal["remember", "understand", "apply"]


class KeyPoint(BaseModel):
    """Một ý chính có thể đặt câu hỏi (bước 'lập kế hoạch' của prompt v3)."""

    point: str = Field(min_length=5, description="Ý chính, một câu")
    source_quote: str = Field(min_length=5, description="Câu trích nguyên văn trong tài liệu")
    suggested_level: Level


class KeyPointList(BaseModel):
    points: list[KeyPoint] = Field(min_length=1, max_length=15)


class MCQ(BaseModel):
    question: str = Field(min_length=10)
    options: list[str] = Field(min_length=4, max_length=4)
    answer_index: int = Field(ge=0, le=3)
    explanation: str = Field(min_length=10)
    level: Level
    source_quote: str = Field(min_length=5, description="Câu trích NGUYÊN VĂN từ tài liệu làm căn cứ")
    source_page: int | None = None

    @model_validator(mode="after")
    def distinct_options(self) -> MCQ:
        norm = {o.strip().lower().rstrip(".") for o in self.options}
        if len(norm) < 4:
            raise ValueError("các lựa chọn bị trùng nhau")
        if any(not o.strip() for o in self.options):
            raise ValueError("có lựa chọn rỗng")
        bad = {"tất cả các đáp án trên", "all of the above", "none of the above", "cả a và b"}
        if norm & bad:
            raise ValueError("không dùng lựa chọn kiểu 'tất cả các đáp án trên'")
        return self

    @property
    def answer(self) -> str:
        return self.options[self.answer_index]


class MCQList(BaseModel):
    questions: list[MCQ] = Field(min_length=1)


class VerifyAnswer(BaseModel):
    """Model kiểm tra trả lời lại câu hỏi CHỈ dựa vào đoạn nguồn."""

    answer_index: int = Field(ge=-1, le=3, description="-1 nếu không trả lời được từ đoạn nguồn")
    reason: str


class JudgeScore(BaseModel):
    """Điểm LLM-as-judge theo rubric 5 tiêu chí (0/1 mỗi tiêu chí)."""

    factual: int = Field(ge=0, le=1)
    answerable: int = Field(ge=0, le=1)
    single_correct: int = Field(ge=0, le=1)
    distractors: int = Field(ge=0, le=1)
    level_match: int = Field(ge=0, le=1)
    comment: str = ""


RUBRIC_KEYS = ["factual", "answerable", "single_correct", "distractors", "level_match"]
