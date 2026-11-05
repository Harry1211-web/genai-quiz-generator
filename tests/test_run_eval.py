"""Sprint 2 experiment runner: crash-safe, resumable, never overwrites human ratings."""
import json
import sys

import pandas as pd
import pytest

from eval import run_eval

CONFIGS = [
    {"name": "big", "model": "fake", "verifier": {"name": "small-verifier", "model": "fake"}},
    {"name": "small", "model": "fake", "verifier": {"name": "big-verifier", "model": "fake"}},
]


def run_main(monkeypatch, tmp_path, *extra):
    cfg = tmp_path / "configs.json"
    cfg.write_text(json.dumps(CONFIGS))
    monkeypatch.setenv("JUDGE_MODEL", "fake")
    argv = ["run_eval", "data/samples", "--n", "2", "--strategies", "v1", "--configs", str(cfg),
            "--out", str(tmp_path / "eval"), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    run_eval.main()
    return tmp_path / "eval"


def cell_stats(out):
    return {p.parent.as_posix(): json.loads(p.read_text()) for p in (out / "runs").rglob("cell_stats.json")}


def test_rerun_pays_nothing_twice(monkeypatch, tmp_path):
    out = run_main(monkeypatch, tmp_path)
    first = cell_stats(out)
    assert first and all(s["gen_calls"] > 0 for s in first.values())
    run_main(monkeypatch, tmp_path)
    second = cell_stats(out)
    for cell, s in second.items():   # totals unchanged: no new generator, verifier or judge calls
        assert s["gen_calls"] == first[cell]["gen_calls"]
        assert s["verify_calls"] == first[cell]["verify_calls"]
        assert s["judge_calls"] == first[cell]["judge_calls"]


def test_each_config_uses_its_own_verifier(monkeypatch, tmp_path):
    out = run_main(monkeypatch, tmp_path)
    metas = {p.parts[-4]: json.loads(p.read_text()) for p in (out / "runs").rglob("cell_meta.json")}
    assert metas["big"]["verifier"] == "small-verifier"
    assert metas["small"]["verifier"] == "big-verifier"


def test_human_ratings_are_kept(monkeypatch, tmp_path):
    out = run_main(monkeypatch, tmp_path)
    sheet = pd.read_csv(out / "labeling_sheet.csv", dtype=str).fillna("")
    assert len(sheet) > 0
    rated_id = sheet.loc[0, "id"]
    sheet.loc[0, ["factual", "answerable", "single_correct", "distractors", "level_match"]] = "1"
    sheet.to_csv(out / "labeling_sheet.csv", index=False)
    run_main(monkeypatch, tmp_path)
    again = pd.read_csv(out / "labeling_sheet.csv", dtype=str).fillna("")
    row = again[again["id"] == rated_id].iloc[0]
    assert row["factual"] == "1" and row["level_match"] == "1"
    assert len(again) == len(sheet)                     # no duplicated rows
    assert (out / "labeling_sheet.backup.csv").exists()


def test_ids_are_stable_and_blind():
    a = run_eval.stable_id("big", "v1", "doc.pdf", "0-1", "What is PaaS?")
    assert a == run_eval.stable_id("big", "v1", "doc.pdf", "0-1", "What is PaaS?")
    assert a != run_eval.stable_id("small", "v1", "doc.pdf", "0-1", "What is PaaS?")
    # a regenerated question at the same position gets a NEW id, so it cannot inherit an old rating
    assert a != run_eval.stable_id("big", "v1", "doc.pdf", "0-1", "Name one benefit of PaaS.")
    assert "big" not in a


def test_regenerated_questions_do_not_inherit_ratings(tmp_path):
    keys = ["factual", "answerable", "single_correct", "distractors", "level_match"]
    old = pd.DataFrame([{"id": "qold", "question": "old", **dict.fromkeys(keys, "1")},
                        {"id": "qkeep", "question": "same", **dict.fromkeys(keys, "0")},
                        {"id": "qunrated", "question": "x", **dict.fromkeys(keys, "")}])
    path = tmp_path / "labeling_sheet.csv"
    old.to_csv(path, index=False)
    new = pd.DataFrame([{"id": "qkeep", "question": "same", **dict.fromkeys(keys, "")},
                        {"id": "qnew", "question": "new", **dict.fromkeys(keys, "")}])
    merged = run_eval.merge_labeling_sheet(path, new, log=lambda m: None)
    assert set(merged["id"]) == {"qkeep", "qnew"}
    assert merged.set_index("id").loc["qkeep", "factual"] == "0"     # rating kept
    assert merged.set_index("id").loc["qnew", "factual"] == ""       # new question starts unrated
    orphans = pd.read_csv(tmp_path / "labeling_sheet.orphaned.csv", dtype=str)
    assert list(orphans["id"]) == ["qold"]                            # rated old question saved aside


def test_regenerate_flag_archives_runs(monkeypatch, tmp_path):
    out = run_main(monkeypatch, tmp_path)
    run_main(monkeypatch, tmp_path, "--regenerate")
    archived = list((out / "archive").glob("*/runs"))
    assert len(archived) == 1 and any(archived[0].rglob("generated.jsonl"))
    assert any((out / "runs").rglob("generated.jsonl"))


def test_cost_table_includes_verifier_cost_and_totals():
    def cell(cfg, strat, gen_usd, ver_usd, jud_usd):
        return {"config": cfg, "strategy": strat, "doc": "d", "questions": 10, "accepted": 9,
                "gen_tokens_in": 1000, "gen_tokens_out": 500, "gen_cost_usd": gen_usd, "gen_seconds": 60,
                "verify_tokens_in": 300, "verify_tokens_out": 20, "verify_cost_usd": ver_usd,
                "verify_seconds": 30, "judge_tokens_in": 400, "judge_tokens_out": 40,
                "judge_cost_usd": jud_usd, "judge_seconds": 12}
    st = pd.DataFrame([cell("luna", "v1", 0.01, 0.0, 0.002),       # local verifier: $0
                       cell("qwen-4b", "v1", 0.0, 0.003, 0.002)])  # paid verifier
    quality, cost, totals = run_eval.build_tables(st, pd.DataFrame({"rejected": []}))
    total = cost[cost["config"] == "TOTAL"].iloc[0]
    assert total["verify_usd"] == 0.003                             # verifier cost is reported
    assert total["total_usd"] == round(0.01 + 0.003 + 0.002 * 2, 4)
    assert total["total_tokens"] == 2 * (1500 + 320 + 440)
    assert totals["total_cost_usd"] == pytest.approx(0.017)
    assert totals["by_config"] == {"luna": pytest.approx(0.012), "qwen-4b": pytest.approx(0.005)}
    assert list(quality["rejected_rate"]) == [0.1, 0.1]
