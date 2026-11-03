"""Step 4 - One interface for every LLM.

- OpenAICompatLLM: calls an OpenAI-compatible /chat/completions endpoint. Works with
  OpenAI, Ollama (http://localhost:11434/v1), Groq, OpenRouter, Gemini (OpenAI-compatible
  endpoint), vLLM... - only LLM_BASE_URL, LLM_MODEL and LLM_API_KEY change.
- FakeLLM: a fake LLM that returns valid JSON built from the document itself. Used for
  tests/CI and for running the whole pipeline at zero cost.

The key function is `generate_json(llm, messages, Schema)`: call the model, parse the
JSON, validate it with Pydantic; on failure SEND THE ERROR MESSAGE BACK so the model can
fix it (up to `max_retries` times). This is how we get ~100% valid output.
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
    format_errors: int = 0  # times the JSON was invalid and had to be retried
    format_failures: int = 0  # times it was still invalid after all retries

    def cost_usd(self, price_in: float, price_out: float) -> float:
        """price_* are in USD per 1M tokens."""
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
    price_in: float = 0.0  # USD per 1M input tokens
    price_out: float = 0.0
    budget_usd: float = 2.0
    timeout: float = 120.0
    json_mode: bool = True  # False if the endpoint rejects response_format (HTTP 400)
    send_temperature: bool = True  # reasoning models often accept only the default temperature
    reasoning_effort: str | None = None  # e.g. "low" / "medium" / "high" for models that support it
    usage: Usage = field(default_factory=Usage)

    @property
    def name(self) -> str:
        return self.model

    def cost_usd(self) -> float:
        return self.usage.cost_usd(self.price_in, self.price_out)

    def _body(self, messages: list[dict], temperature: float, json_mode: bool) -> dict:
        body: dict = {"model": self.model, "messages": messages}
        if self.send_temperature:
            body["temperature"] = temperature
        if json_mode and self.json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    def _post(self, body: dict) -> httpx.Response:
        """POST with retries for network errors, 429 and 5xx (exponential backoff).
        4xx errors other than 429 are returned immediately: retrying them never helps."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        for attempt in range(4):
            try:
                r = httpx.post(url, json=body, headers=headers, timeout=self.timeout)
            except httpx.TransportError:
                if attempt == 3:
                    raise
            else:
                if r.status_code not in (429, 500, 502, 503, 504) or attempt == 3:
                    return r
            time.sleep(2**attempt)
        raise RuntimeError("unreachable")

    def chat(self, messages: list[dict], temperature: float = 0.2, json_mode: bool = True) -> str:
        if self.cost_usd() >= self.budget_usd:
            raise BudgetExceeded(f"Spent ${self.cost_usd():.3f} >= budget ${self.budget_usd}")
        t0 = time.perf_counter()
        for _ in range(3):  # up to 2 automatic fixes of unsupported parameters
            r = self._post(self._body(messages, temperature, json_mode))
            if r.status_code < 400:
                break
            msg = _error_message(r)
            low = msg.lower()
            if r.status_code == 400 and self.send_temperature and "temperature" in low:
                _warn(f"{self.model} rejected 'temperature' -> retrying without it ({msg})")
                self.send_temperature = False
                continue
            if r.status_code == 400 and self.json_mode and json_mode and "response_format" in low:
                _warn(f"{self.model} rejected 'response_format' -> retrying without JSON mode ({msg})")
                self.json_mode = False
                continue
            raise LLMRequestError(f"{self.model} at {self.base_url}: HTTP {r.status_code} - {msg}")
        else:
            raise LLMRequestError(f"{self.model}: request still rejected after parameter fixes - {msg}")
        data = r.json()
        self.usage.calls += 1
        self.usage.seconds += time.perf_counter() - t0
        u = data.get("usage") or {}
        self.usage.prompt_tokens += u.get("prompt_tokens", 0)
        self.usage.completion_tokens += u.get("completion_tokens", 0)
        return data["choices"][0]["message"]["content"]


class LLMRequestError(RuntimeError):
    """The provider rejected the request; the message includes the provider's explanation."""


def _error_message(r: httpx.Response) -> str:
    try:
        err = r.json().get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)
        if err:
            return str(err)
    except ValueError:
        pass
    return r.text[:500]


def _warn(msg: str) -> None:
    import sys

    print(f"[llm] {msg}", file=sys.stderr)


def from_env(prefix: str = "LLM") -> LLM:
    """Read config from environment variables with a role prefix: LLM_*, VERIFY_*, JUDGE_*."""
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
        reasoning_effort=os.getenv(f"{prefix}_REASONING_EFFORT") or None,
        timeout=float(os.getenv(f"{prefix}_TIMEOUT", "120")),
    )


# ---------------------------------------------------------------- JSON + retry
def extract_json(text: str) -> str:
    """Models often wrap JSON in ```json ... ``` or add a preamble - cut out the JSON part."""
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
                {"role": "user", "content": f"Your JSON is invalid:\n{err}\n"
                                            "Return the FULL corrected JSON only, no explanation."},
            ]
    return None


# ---------------------------------------------------------------- fake LLM
@dataclass
class FakeLLM:
    """Returns valid JSON built from sentences in <document>. Not 'smart' - only for
    dry runs, tests and CI. Evaluation results obtained with FakeLLM are meaningless."""

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
        # take the LAST <document ...>...</document> block (the system prompt also mentions <document>)
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
    """Cloze question: blank out the longest word; distractors = longest words of other sentences."""
    words = [w.strip(",.;:()\"'") for w in sent.split()]
    key = max(words, key=len)
    pool = [max((w.strip(",.;:()\"'") for w in s.split()), key=len) for s in all_sents if s != sent]
    distract = []
    for w in pool + ["protocol", "router", "application layer", "data frame"]:
        if w.lower() != key.lower() and w.lower() not in {d.lower() for d in distract}:
            distract.append(w)
        if len(distract) == 3:
            break
    return {
        "question": "Fill in the blank: " + sent.replace(key, "_____", 1),
        "options": [key] + distract,
        "answer_index": 0,
        "explanation": f"According to the document: \"{sent}\"",
        "level": "remember",
        "source_quote": sent,
    }
