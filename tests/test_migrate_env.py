import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from migrate_env import migrate, parse  # noqa: E402

OLD = """LLM_MODEL=gpt-6-luna
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-secret
LLM_PRICE_IN=0.10
LLM_PRICE_OUT=0.50
VERIFY_MODEL=qwen3.5:4b
VERIFY_PROVIDER=ollama
VERIFY_BASE_URL=http://localhost:11434
VERIFY_THINK=false
JUDGE_MODEL=gpt-6-luna
JUDGE_BASE_URL=https://api.openai.com/v1
JUDGE_API_KEY=sk-secret
LLM_BUDGET_USD=2
"""


def test_migrate_roles_to_numbered_models():
    new, numbers = migrate(parse(OLD))
    assert numbers == {"LLM": 1, "VERIFY": 2, "JUDGE": 1}          # judge reuses model 1
    assert new["LLM_MODEL_1"] == "gpt-6-luna" and new["LLM_API_KEY_1"] == "sk-secret"
    assert new["LLM_MODEL_2"] == "qwen3.5:4b" and new["LLM_THINK_2"] == "false"
    assert new["LLM_BUDGET_USD"] == "2"
    assert not any(k.startswith(("VERIFY_", "JUDGE_")) or k == "LLM_MODEL" for k in new)
