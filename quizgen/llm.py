"""Step 4 - One interface for every LLM.

- OpenAICompatLLM: calls an OpenAI-compatible /chat/completions endpoint. Works with
  OpenAI, Ollama (http://localhost:11434/v1), Groq, OpenRouter, Gemini (OpenAI-compatible
  endpoint), vLLM... - only LLM_BASE_URL, LLM_MODEL and LLM_API_KEY change.
- OllamaLLM: calls Ollama's native /api/chat, which can turn a model's "thinking" off explicitly
  (think=false). Thinking models such as Qwen3.5 otherwise write hundreds of hidden tokens before
  every short answer, which makes local verification take minutes per question.
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
    # Off by default: reasoning models such as gpt-6-luna only accept their default temperature.
    # Turn on (LLM_TEMPERATURE_<n>=true) for API models that support it.
    send_temperature: bool = False
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
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return post_with_retries(f"{self.base_url.rstrip('/')}/chat/completions", body, headers, self.timeout)

    def chat(self, messages: list[dict], temperature: float = 0.2, json_mode: bool = True) -> str:
        if self.cost_usd() >= self.budget_usd:
            raise BudgetExceeded(f"Spent ${self.cost_usd():.3f} >= budget ${self.budget_usd}")
        t0 = time.perf_counter()
        for _ in range(2):  # one automatic fix of an unsupported parameter
            r = self._post(self._body(messages, temperature, json_mode))
            if r.status_code < 400:
                break
            msg = _error_message(r)
            low = msg.lower()
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


def post_with_retries(url: str, body: dict, headers: dict, timeout: float) -> httpx.Response:
    """POST with retries for network errors, 429 and 5xx (exponential backoff).
    4xx errors other than 429 are returned immediately: retrying them never helps."""
    for attempt in range(4):
        try:
            r = httpx.post(url, json=body, headers=headers, timeout=timeout)
        except httpx.TransportError:
            if attempt == 3:
                raise
        else:
            if r.status_code not in (429, 500, 502, 503, 504) or attempt == 3:
                return r
        time.sleep(2**attempt)
    raise RuntimeError("unreachable")


@dataclass
class OllamaLLM:
    """Local model through Ollama's native API (http://localhost:11434/api/chat).

    think=False turns thinking off (supported by recent Ollama versions for thinking models);
    think=None sends nothing and keeps the model's default. Local calls cost $0.
    num_ctx: context window in tokens (prompt + answer + thinking). Ollama's default is small (a few
    thousand tokens): with thinking on, a long reasoning can fill it and the JSON answer is cut off."""

    model: str
    base_url: str = "http://localhost:11434"
    think: bool | None = False
    json_mode: bool = True
    timeout: float = 300.0
    num_ctx: int | None = None
    usage: Usage = field(default_factory=Usage)

    @property
    def name(self) -> str:
        return self.model

    def cost_usd(self) -> float:
        return 0.0

    def _url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):  # accept the OpenAI-style URL too
            base = base[:-3]
        return f"{base}/api/chat"

    def chat(self, messages: list[dict], temperature: float = 0.2, json_mode: bool = True) -> str:
        t0 = time.perf_counter()
        for _ in range(2):  # one automatic fix: drop "think" for models that do not support it
            body: dict = {"model": self.model, "messages": messages, "stream": False,
                          "options": {"temperature": temperature}}
            if self.num_ctx:
                body["options"]["num_ctx"] = self.num_ctx
            if self.think is not None:
                body["think"] = self.think
            if json_mode and self.json_mode:
                body["format"] = "json"
            r = post_with_retries(self._url(), body, {}, self.timeout)
            if r.status_code < 400:
                break
            msg = _error_message(r)
            if r.status_code == 400 and self.think is not None and "think" in msg.lower():
                _warn(f"{self.model} rejected 'think' -> retrying with the model's default ({msg})")
                self.think = None
                continue
            raise LLMRequestError(f"{self.model} at {self._url()}: HTTP {r.status_code} - {msg}")
        else:
            raise LLMRequestError(f"{self.model}: request still rejected - {msg}")
        data = r.json()
        self.usage.calls += 1
        self.usage.seconds += time.perf_counter() - t0
        self.usage.prompt_tokens += data.get("prompt_eval_count", 0)
        self.usage.completion_tokens += data.get("eval_count", 0)  # includes thinking tokens
        if data.get("done_reason") == "length":
            used = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            _warn(f"{self.model}: answer cut off at {used} tokens - the context window is full "
                  f"(set LLM_NUM_CTX_<n>=16384 in .env, or turn thinking off)")
        return data["message"]["content"]


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


# Which numbered model plays which role when nothing else is given. Change roles here, with the
# --generator / --verifier / --judge command-line options, or per configuration in eval/configs.json.
DEFAULT_ROLES = {"generator": 1, "verifier": 2, "judge": 1}

MODEL_KEYS = ("MODEL", "BASE_URL", "API_KEY", "PRICE_IN", "PRICE_OUT",            # what .env normally holds
              "PROVIDER", "THINK", "TIMEOUT", "JSON_MODE", "REASONING_EFFORT",   # optional extras
              "TEMPERATURE", "NUM_CTX")


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def model_settings(n: int | str) -> dict[str, str]:
    """Raw settings of model n from .env: LLM_MODEL_n, LLM_BASE_URL_n, LLM_API_KEY_n, LLM_PRICE_IN_n, ..."""
    _load_env()
    return {k: v for k in MODEL_KEYS if (v := os.getenv(f"LLM_{k}_{n}")) not in (None, "")}


def in_docker_url(url: str) -> str:
    """Inside the Docker image (QUIZGEN_IN_DOCKER=1), "localhost" is the container itself: a model
    served on the host machine (Ollama) is reached through host.docker.internal instead. The same
    .env file therefore works with and without Docker."""
    if os.getenv("QUIZGEN_IN_DOCKER") == "1":
        return re.sub(r"//(localhost|127\.0\.0\.1)(?=[:/]|$)", "//host.docker.internal", url)
    return url


def build_llm(s: dict, name: str = "fake") -> LLM:
    """Build a model from settings with the keys of MODEL_KEYS (lower or upper case).
    A base URL on port 11434 means a local Ollama server, called through its native API with
    thinking off unless THINK says otherwise."""
    s = {k.upper(): v for k, v in s.items()}
    model = s.get("MODEL", "fake")
    if model == "fake":
        return FakeLLM(name=name)
    base_url = in_docker_url(s.get("BASE_URL", ""))
    provider = str(s.get("PROVIDER") or ("ollama" if ":11434" in base_url else "openai")).lower()
    json_mode = str(s.get("JSON_MODE", "true")).lower() == "true"
    if provider == "ollama":
        return OllamaLLM(model=model, base_url=base_url or "http://localhost:11434",
                         think=parse_think(s.get("THINK", "false")), json_mode=json_mode,
                         timeout=float(s.get("TIMEOUT", 300)),
                         num_ctx=int(s["NUM_CTX"]) if s.get("NUM_CTX") else None)
    return OpenAICompatLLM(model=model, base_url=base_url or "https://api.openai.com/v1",
                           api_key=s.get("API_KEY", ""), price_in=float(s.get("PRICE_IN", 0)),
                           price_out=float(s.get("PRICE_OUT", 0)),
                           budget_usd=float(os.getenv("LLM_BUDGET_USD", "2")), json_mode=json_mode,
                           reasoning_effort=s.get("REASONING_EFFORT") or None,
                           send_temperature=str(s.get("TEMPERATURE", "false")).lower() == "true",
                           timeout=float(s.get("TIMEOUT", 120)))


def model_from_env(n: int | str) -> LLM:
    """Model number n declared in .env ("fake" = the fake LLM used by tests and dry runs).
    If LLM_MODEL_n is missing, a fake model is used and a warning is printed."""
    if str(n).lower() == "fake":
        return FakeLLM()
    s = model_settings(n)
    if "MODEL" not in s:
        _warn(f"LLM_MODEL_{n} is not set in .env -> using the fake model (no real LLM calls)")
        return FakeLLM(name=f"fake-{n}")
    return build_llm(s)


def model_for_role(role: str, n: int | str | None = None) -> LLM:
    """Model for a role (generator / verifier / judge): model n if given, else DEFAULT_ROLES[role]."""
    return model_from_env(n if n is not None else DEFAULT_ROLES[role])


def parse_think(value) -> bool | None:
    """'false' -> False, 'true' -> True, 'default' / '' -> None (send nothing)."""
    if value is None or isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    return None if v in ("", "default", "none") else v in ("1", "true", "yes", "on")


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
            err = str(e)[:1500]
            if attempt == max_retries:
                llm.usage.format_failures += 1
                try:  # kept so callers can show WHY the answer was rejected (e.g. eval.judge_check)
                    llm.last_format_error = {"raw": raw[:2000], "error": err[:500]}
                except AttributeError:
                    pass
                return None
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
