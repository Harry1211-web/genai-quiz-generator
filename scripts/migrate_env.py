"""Convert an older role-based .env (LLM_* / VERIFY_* / JUDGE_*) to numbered models (LLM_<SETTING>_<n>).

  python scripts/migrate_env.py            # writes the new .env, keeps the old one as .env.old
  python scripts/migrate_env.py --dry-run  # only shows which variables would be created

Mapping: the generator (LLM_*) becomes model 1, the verifier (VERIFY_*) model 2, the judge (JUDGE_*)
reuses an existing number when it is the same model and base URL, otherwise it gets the next number.
Secret values are copied as they are and never printed.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

KEYS = ("MODEL", "BASE_URL", "API_KEY", "PRICE_IN", "PRICE_OUT", "PROVIDER", "THINK", "TIMEOUT",
        "JSON_MODE", "REASONING_EFFORT")
LEGACY = ("LLM", "VERIFY", "JUDGE")


def parse(text: str) -> dict[str, str]:
    env = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.split(" #")[0].strip()
    return env


def migrate(env: dict[str, str]) -> tuple[dict[str, str], dict[str, int]]:
    models: list[dict[str, str]] = []
    numbers: dict[str, int] = {}
    for prefix in LEGACY:
        s = {k: env[f"{prefix}_{k}"] for k in KEYS if env.get(f"{prefix}_{k}")}
        if not s.get("MODEL") or s["MODEL"] == "fake":
            continue
        for i, m in enumerate(models, 1):
            if m["MODEL"] == s["MODEL"] and m.get("BASE_URL") == s.get("BASE_URL"):
                for k, v in s.items():   # fill settings the first occurrence did not have (e.g. prices)
                    m.setdefault(k, v)
                numbers[prefix] = i
                break
        else:
            models.append(s)
            numbers[prefix] = len(models)
    new = {f"LLM_{k}_{i}": v for i, m in enumerate(models, 1) for k, v in m.items()}
    legacy_names = {f"{p}_{k}" for p in LEGACY for k in KEYS}
    new.update({k: v for k, v in env.items() if k not in legacy_names})   # e.g. LLM_BUDGET_USD
    return new, numbers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=".env")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    path = Path(a.env)
    env = parse(path.read_text(encoding="utf-8"))
    if any(k.startswith("LLM_MODEL_") for k in env):
        raise SystemExit(f"{path} already uses numbered models (LLM_MODEL_<n>); nothing to do.")
    new, numbers = migrate(env)
    print("Role -> model number:", {p: n for p, n in numbers.items()})
    print("Variables in the new file:", ", ".join(new))
    if a.dry_run:
        return
    shutil.copyfile(path, path.with_name(path.name + ".old"))
    lines = ["# Numbered models (see .env.example). Roles are chosen in code / CLI / eval/configs.json.", ""]
    for i in sorted({int(k.rsplit("_", 1)[1]) for k in new if k.startswith("LLM_MODEL_")}):
        lines += [f"# Model {i}"] + [f"{k}={v}" for k, v in new.items() if k.endswith(f"_{i}")] + [""]
    lines += [f"{k}={v}" for k, v in new.items() if not k.rsplit("_", 1)[-1].isdigit()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}; the previous file is {path.name}.old")


if __name__ == "__main__":
    main()
