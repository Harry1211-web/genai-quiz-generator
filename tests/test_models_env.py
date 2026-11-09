"""Numbered models in .env (LLM_MODEL_1, LLM_MODEL_2, ...); roles chosen in code / CLI / configs."""
from eval.run_eval import config_models
from quizgen.llm import DEFAULT_ROLES, FakeLLM, OllamaLLM, OpenAICompatLLM, model_for_role, model_from_env


def set_models(monkeypatch):
    for k, v in {"LLM_MODEL_1": "gpt-6-luna", "LLM_BASE_URL_1": "https://api.openai.com/v1",
                 "LLM_API_KEY_1": "sk-test", "LLM_PRICE_IN_1": "0.10", "LLM_PRICE_OUT_1": "0.50",
                 "LLM_MODEL_2": "qwen3.5:4b", "LLM_BASE_URL_2": "http://localhost:11434"}.items():
        monkeypatch.setenv(k, v)


def test_numbered_models(monkeypatch):
    set_models(monkeypatch)
    m1, m2 = model_from_env(1), model_from_env(2)
    assert isinstance(m1, OpenAICompatLLM) and m1.model == "gpt-6-luna" and m1.api_key == "sk-test"
    assert (m1.price_in, m1.price_out) == (0.10, 0.50)
    assert isinstance(m2, OllamaLLM) and m2.model == "qwen3.5:4b"


def test_default_roles_and_override(monkeypatch):
    set_models(monkeypatch)
    assert DEFAULT_ROLES == {"generator": 1, "verifier": 2, "judge": 1}
    assert model_for_role("generator").name == "gpt-6-luna"
    assert model_for_role("verifier").name == "qwen3.5:4b"
    assert model_for_role("generator", "2").name == "qwen3.5:4b"   # e.g. --generator 2


def test_experiment_config_by_number(monkeypatch):
    set_models(monkeypatch)
    gen, ver = config_models({"name": "qwen-4b", "generator": 2, "verifier": 1})
    assert gen.name == "qwen3.5:4b" and ver.name == "gpt-6-luna"


def test_missing_model_falls_back_to_fake(monkeypatch):
    monkeypatch.delenv("LLM_MODEL_7", raising=False)
    assert isinstance(model_from_env(7), FakeLLM)


def test_localhost_is_rewritten_only_inside_docker(monkeypatch):
    from quizgen.llm import in_docker_url
    monkeypatch.delenv("QUIZGEN_IN_DOCKER", raising=False)
    assert in_docker_url("http://localhost:11434/v1") == "http://localhost:11434/v1"
    monkeypatch.setenv("QUIZGEN_IN_DOCKER", "1")
    assert in_docker_url("http://localhost:11434/v1") == "http://host.docker.internal:11434/v1"
    assert in_docker_url("http://127.0.0.1:11434") == "http://host.docker.internal:11434"
    assert in_docker_url("https://api.openai.com/v1") == "https://api.openai.com/v1"
    assert in_docker_url("http://localhost.example.com/v1") == "http://localhost.example.com/v1"
