"""Crash safety: paid work is saved immediately and never paid for twice."""
import json

import httpx
import pytest

from quizgen.extract import load_sections
from quizgen.llm import FakeLLM
from quizgen.pipeline import GENERATED, VERIFIED, _check_meta, run

SAMPLE = "data/samples/transport_layer_tcp_udp.md"


class TimeoutVerifier(FakeLLM):
    """A verifier that always times out (like a slow local model)."""

    def chat(self, messages, temperature=0.2, json_mode=True):
        raise httpx.ReadTimeout("timed out")


class CrashingGenerator(FakeLLM):
    """Works for the first `ok_calls` calls, then the 'network' dies."""

    def __init__(self, ok_calls):
        super().__init__()
        self.ok_calls = ok_calls

    def chat(self, messages, temperature=0.2, json_mode=True):
        if self.usage.calls >= self.ok_calls:
            raise httpx.ConnectError("network down")
        return super().chat(messages, temperature, json_mode)


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_verifier_timeout_keeps_generated_questions(tmp_path):
    secs = load_sections(SAMPLE)
    qs, stats = run(secs, FakeLLM(), TimeoutVerifier(), n_per_section=2, strategy="v1",
                    use_st=False, out_dir=tmp_path)
    # generation is on disk even though every verification failed
    assert len(lines(tmp_path / GENERATED)) == len(qs) > 0
    # the run did not crash; nothing unverified is exported as accepted
    assert stats["accepted"] == 0
    assert all(q.rejected for q in qs)
    assert stats.get("verify_error", 0) + stats.get("not_verified", 0) + stats.get("quote_not_found", 0) \
        >= len(qs) - stats.get("answer_leak", 0)


def test_resume_does_not_call_generator_again(tmp_path):
    secs = load_sections(SAMPLE)
    run(secs, FakeLLM(), TimeoutVerifier(), n_per_section=2, strategy="v1", use_st=False, out_dir=tmp_path)
    gen2, ver2 = FakeLLM(), FakeLLM()
    qs, stats = run(secs, gen2, ver2, n_per_section=2, strategy="v1", use_st=False, out_dir=tmp_path)
    assert gen2.usage.calls == 0          # nothing generated (paid) twice
    assert ver2.usage.calls > 0           # failed verifications are retried
    assert "verify_error" not in stats and "not_verified" not in stats
    saved = {r["qid"]: r for r in lines(tmp_path / VERIFIED)}
    assert all("verifier_answer" in r for r in saved.values())


def test_crash_during_generation_resumes_remaining_sections(tmp_path):
    secs = load_sections(SAMPLE)
    assert len(secs) >= 2
    with pytest.raises(httpx.ConnectError):
        run(secs, CrashingGenerator(ok_calls=1), None, n_per_section=2, strategy="v1",
            use_st=False, out_dir=tmp_path)
    assert {r["section_index"] for r in lines(tmp_path / GENERATED)} == {0}   # section 0 was saved
    gen2 = FakeLLM()
    qs, _ = run(secs, gen2, None, n_per_section=2, strategy="v1", use_st=False, out_dir=tmp_path)
    assert gen2.usage.calls == len(secs) - 1   # only the missing sections were generated
    assert {q.qid.split("-")[0] for q in qs} == {str(i) for i in range(len(secs))}


def test_verified_questions_are_not_verified_twice(tmp_path):
    secs = load_sections(SAMPLE)
    run(secs, FakeLLM(), FakeLLM(), n_per_section=2, strategy="v1", use_st=False, out_dir=tmp_path)
    ver2 = FakeLLM()
    run(secs, FakeLLM(), ver2, n_per_section=2, strategy="v1", use_st=False, out_dir=tmp_path)
    assert ver2.usage.calls == 0


def test_mixing_runs_in_one_folder_is_refused(tmp_path):
    _check_meta(tmp_path, {"file": "a.pdf", "strategy": "v1"}, regenerate=False)
    with pytest.raises(SystemExit, match="different run"):
        _check_meta(tmp_path, {"file": "a.pdf", "strategy": "v2"}, regenerate=False)
    _check_meta(tmp_path, {"file": "a.pdf", "strategy": "v2"}, regenerate=True)  # --regenerate allows it


def test_legacy_checkpoint_is_imported(tmp_path):
    """Questions saved by the older 'generated_unverified.jsonl' format are reused, not regenerated."""
    secs = load_sections(SAMPLE)
    first = tmp_path / "first"
    qs, _ = run(secs, FakeLLM(), None, n_per_section=2, strategy="v1", use_st=False, out_dir=first)
    legacy = tmp_path / "old"
    legacy.mkdir()
    new_keys = ("qid", "verifier_answer", "verifier_reason")
    old_rows = [{k: v for k, v in q.to_dict().items() if k not in new_keys} for q in qs]
    (legacy / "generated_unverified.jsonl").write_text("\n".join(json.dumps(r) for r in old_rows),
                                                       encoding="utf-8")
    gen2 = FakeLLM()
    qs2, _ = run(secs, gen2, None, n_per_section=2, strategy="v1", use_st=False, out_dir=legacy)
    assert gen2.usage.calls == 0 and len(qs2) == len(qs)


def test_regenerate_archives_old_questions_and_generates_new_ones(tmp_path):
    from quizgen.pipeline import archive_previous
    secs = load_sections(SAMPLE)
    run(secs, FakeLLM(), None, n_per_section=2, strategy="v1", use_st=False, out_dir=tmp_path)
    old = (tmp_path / GENERATED).read_text(encoding="utf-8")
    dest = archive_previous(tmp_path)
    assert dest is not None and (dest / GENERATED).read_text(encoding="utf-8") == old  # nothing deleted
    gen2 = FakeLLM()
    run(secs, gen2, None, n_per_section=2, strategy="v1", use_st=False, out_dir=tmp_path)
    assert gen2.usage.calls == len(secs)   # a new set was generated
