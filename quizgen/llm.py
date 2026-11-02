"""Bước 3 – Một interface chung cho mọi LLM.

- OpenAICompatLLM: gọi endpoint /chat/completions chuẩn OpenAI. Dùng được cho
  OpenAI, Ollama (http://localhost:11434/v1), Groq, OpenRouter, Gemini (endpoint
  tương thích OpenAI), vLLM... – chỉ cần đổi LLM_BASE_URL, LLM_MODEL, LLM_API_KEY.
- FakeLLM: LLM giả, trả JSON hợp lệ dựng từ chính tài liệu. Dùng cho test/CI và
  để chạy thử toàn bộ pipeline không tốn tiền.

Hàm quan trọng nhất: `generate_json(llm, messages, Schema)` – gọi model, parse
JSON, validate bằng Pydantic; lỗi thì GỬI LẠI THÔNG BÁO LỖI cho model sửa
(tối đa `max_retries` lần). Đây là cách đạt ~100% đúng định dạng.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    format_errors: int = 0  # số lần JSON sai phải thử lại
    format_failures: int = 0  # số lần hết lượt thử vẫn sai

    def cost_usd(self, price_in: float, price_out: float) -> float:
        """price_* tính theo USD / 1 triệu token."""
        return (self.prompt_tokens * price_in + self.completion_tokens * price_out) / 1e6


class LLM(Protocol):
    name: str
    usage: Usage

    def chat(self, messages: list[dict], temperature: float = 0.2, json_mode: bool = True) -> str: ...

    def cost_usd(self) -> float: ...


@dataclass
class OpenAICompatLLM:
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    price_in: float = 0.0  # USD / 1M token đầu vào
    price_out: float = 0.0
    budget_usd: float = 2.0
    timeout: float = 120.0
    json_mode: bool = True  # False nếu endpoint không hỗ trợ response_format (lỗi 400)
    usage: Usage = field(default_factory=Usage)

    @property
    def name(self) -> str:
        return self.model

    def cost_usd(self) -> float:
        return self.usage.cost_usd(self.price_in, self.price_out)

    def chat(self, messages: list[dict], temperature: float = 0.2, json_mode: bool = True) -> str:
        if self.cost_usd() >= self.budget_usd:
            raise BudgetExceeded(f"Đã dùng ${self.cost_usd():.3f} ≥ ngân sách ${self.budget_usd}")
        body: dict = {"model": self.model, "messages": messages, "temperature": temperature}
        if json_mode and self.json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        t0 = time.perf_counter()
        for attempt in range(4):  # retry lỗi mạng / 429 / 5xx với backoff
            try:
                r = httpx.post(f"{self.base_url.rstrip('/')}/chat/completions", json=body,
                               headers=headers, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                r.raise_for_status()
                break
            except (httpx.TransportError, httpx.HTTPStatusError):
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        data = r.json()
        self.usage.calls += 1
        self.usage.seconds += time.perf_counter() - t0
        u = data.get("usage") or {}
        self.usage.prompt_tokens += u.get("prompt_tokens", 0)
        self.usage.completion_tokens += u.get("completion_tokens", 0)
        return data["choices"][0]["message"]["content"]


def from_env(prefix: str = "LLM") -> LLM:
    """Đọc cấu hình từ biến môi trường, ví dụ LLM_MODEL, JUDGE_MODEL..."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    model = os.getenv(f"{prefix}_MODEL", "fake")
    if model == "fake":
        return FakeLLM()
    return OpenAICompatLLM(
        model=model,
        base_url=os.getenv(f"{prefix}_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv(f"{prefix}_API_KEY", ""),
        price_in=float(os.getenv(f"{prefix}_PRICE_IN", "0")),
        price_out=float(os.getenv(f"{prefix}_PRICE_OUT", "0")),
        budget_usd=float(os.getenv("LLM_BUDGET_USD", "2")),
        json_mode=os.getenv(f"{prefix}_JSON_MODE", "true").lower() == "true",
    )


# ---------------------------------------------------------------- JSON + retry
def extract_json(text: str) -> str:
    """Model hay bọc JSON trong ```json ... ``` hoặc thêm câu dẫn – cắt lấy phần JSON."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1)
    start = min([i for i in (text.find("{"), text.find("[")) if i != -1], default=-1)
    if start == -1:
        return text
    end = max(text.rfind("}"), text.rfind("]"))
    return text[start : end + 1]


def generate_json(llm: LLM, messages: list[dict], schema: type[T], temperature: float = 0.2,
                  max_retries: int = 2) -> T | None:
    msgs = list(messages)
    for attempt in range(max_retries + 1):
        raw = llm.chat(msgs, temperature=temperature)
        try:
            return schema.model_validate_json(extract_json(raw))
        except (ValidationError, ValueError) as e:
            llm.usage.format_errors += 1
            if attempt == max_retries:
                llm.usage.format_failures += 1
                return None
            err = str(e)[:1500]
            msgs = msgs + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"JSON của bạn không hợp lệ:\n{err}\n"
                                            "Hãy trả lại TOÀN BỘ JSON đã sửa, không giải thích."},
            ]
    return None


# ---------------------------------------------------------------- LLM giả
@dataclass
class FakeLLM:
    """Trả JSON hợp lệ dựng từ câu trong <document>. Không 'thông minh' – chỉ để
    chạy thử pipeline, viết test và CI. Kết quả đánh giá với FakeLLM vô nghĩa."""

    name: str = "fake"
    usage: Usage = field(default_factory=Usage)

    def cost_usd(self) -> float:
        return 0.0

    def chat(self, messages: list[dict], temperature: float = 0.2, json_mode: bool = True) -> str:
        self.usage.calls += 1
        prompt = "\n".join(m["content"] for m in messages)
        self.usage.prompt_tokens += len(prompt) // 4
        task = re.search(r"<task>(\w+)</task>", prompt)
        task = task.group(1) if task else "generate"
        # lấy khối <document ...>...</document> CUỐI CÙNG (system prompt cũng nhắc chữ <document>)
        docs = re.findall(r"<document[^>]*>\n(.*?)</document>", prompt, re.DOTALL)
        doc_text = docs[-1] if docs else prompt
        sents = _sentences(doc_text)
        if task == "plan":
            out = {"points": [{"point": s[:120], "source_quote": s, "suggested_level": "remember"}
                              for s in sents[:5]]}
        elif task == "verify":
            q = re.search(r"<question>(.*?)</question>", prompt, re.DOTALL).group(1)
            data = json.loads(q)
            src = doc_text.lower()
            idx = next((i for i, o in enumerate(data["options"]) if o.lower() in src), -1)
            out = {"answer_index": idx, "reason": "fake"}
        elif task == "judge":
            h = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
            out = {k: int((h >> i) % 5 != 0) for i, k in enumerate(
                ["factual", "answerable", "single_correct", "distractors", "level_match"])}
            out["comment"] = "fake judge"
        else:
            n = int((re.search(r"<n>(\d+)</n>", prompt) or re.search(r"(\d+)", "3")).group(1))
            out = {"questions": [_cloze(s, sents) for s in sents if len(s.split()) >= 6][:n]}
        text = json.dumps(out, ensure_ascii=False)
        self.usage.completion_tokens += len(text) // 4
        return text


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 25]


def _cloze(sent: str, all_sents: list[str]) -> dict:
    """Câu hỏi điền khuyết: che từ dài nhất, đáp án nhiễu = từ dài ở câu khác."""
    words = [w.strip(",.;:()\"'") for w in sent.split()]
    key = max(words, key=len)
    pool = [max((w.strip(",.;:()\"'") for w in s.split()), key=len) for s in all_sents if s != sent]
    distract = []
    for w in pool + ["giao thức", "bộ định tuyến", "tầng ứng dụng", "khung dữ liệu"]:
        if w.lower() != key.lower() and w.lower() not in {d.lower() for d in distract}:
            distract.append(w)
        if len(distract) == 3:
            break
    return {
        "question": "Điền vào chỗ trống: " + sent.replace(key, "_____", 1),
        "options": [key] + distract,
        "answer_index": 0,
        "explanation": f"Theo tài liệu: \"{sent}\"",
        "level": "remember",
        "source_quote": sent,
    }
