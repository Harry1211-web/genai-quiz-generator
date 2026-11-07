"""OllamaLLM: native API with thinking turned off."""
import httpx
import pytest

from quizgen.llm import OllamaLLM, model_from_env, parse_think


def fake_post(responses, calls):
    def post(url, json=None, headers=None, timeout=None):
        calls.append((url, json))
        status, payload = responses[len(calls) - 1]
        return httpx.Response(status, json=payload, request=httpx.Request("POST", url))
    return post


OK = {"message": {"role": "assistant", "content": '{"answer_index": 1, "reason": "x"}'},
      "prompt_eval_count": 120, "eval_count": 15}


def test_think_false_and_json_format_are_sent(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "post", fake_post([(200, OK)], calls))
    llm = OllamaLLM(model="qwen3.5:4b", base_url="http://localhost:11434/v1")   # /v1 is accepted
    out = llm.chat([{"role": "user", "content": "q"}], temperature=0.0)
    url, body = calls[0]
    assert url == "http://localhost:11434/api/chat"
    assert body["think"] is False and body["format"] == "json" and body["stream"] is False
    assert body["options"]["temperature"] == 0.0
    assert out.startswith("{") and llm.usage.prompt_tokens == 120 and llm.usage.completion_tokens == 15
    assert llm.cost_usd() == 0.0


def test_model_without_thinking_support_falls_back(monkeypatch):
    calls = []
    bad = {"error": "\"llama3\" does not support thinking"}
    monkeypatch.setattr(httpx, "post", fake_post([(400, bad), (200, OK)], calls))
    llm = OllamaLLM(model="llama3")
    llm.chat([{"role": "user", "content": "q"}])
    assert "think" in calls[0][1] and "think" not in calls[1][1] and llm.think is None


def test_port_11434_selects_ollama_with_thinking_off(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_2", "qwen3.5:4b")
    monkeypatch.setenv("LLM_BASE_URL_2", "http://localhost:11434")
    monkeypatch.delenv("LLM_PROVIDER_2", raising=False)
    monkeypatch.delenv("LLM_THINK_2", raising=False)
    llm = model_from_env(2)
    assert isinstance(llm, OllamaLLM) and llm.think is False      # no provider needed, thinking off


@pytest.mark.parametrize("value, expected",
                         [("false", False), ("true", True), ("default", None), (False, False)])
def test_parse_think(value, expected):
    assert parse_think(value) is expected


def test_num_ctx_is_sent_and_a_cut_off_answer_is_reported(monkeypatch, capsys):
    import httpx

    from quizgen import llm as L
    sent = {}

    def fake_post(url, body, headers, timeout):
        sent.update(body)
        return httpx.Response(200, json={"message": {"content": "{}"}, "prompt_eval_count": 900,
                                         "eval_count": 3196, "done_reason": "length"})
    monkeypatch.setattr(L, "post_with_retries", fake_post)
    m = L.build_llm({"MODEL": "qwen3.5:4b", "BASE_URL": "http://localhost:11434/v1", "NUM_CTX": "16384"})
    m.chat([{"role": "user", "content": "hi"}])
    assert sent["options"]["num_ctx"] == 16384
    assert "cut off at 4096 tokens" in capsys.readouterr().err
