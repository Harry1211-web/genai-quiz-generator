"""Verification rules revised after reading the first real results (Sprint 1)."""
import json

from quizgen.extract import load_sections
from quizgen.generate import GeneratedQuestion, load_prompt, split_labels
from quizgen.llm import FakeLLM
from quizgen.pipeline import GENERATED, VERIFIED, archive_previous, run
from quizgen.schema import MCQ
from quizgen.verify import check_static, check_warnings, find_context, quote_in_source, verify_one

SOURCE = ("Business benefits of PaaS\n• Cost savings\n✓No need for large initial investments: Save on \n"
          "investment costs in hardware and software.\n✓Reduce operating costs: Reduce the burden.\n"
          + "Filler text about other topics. " * 40 +
          "\nDisadvantages\nHigh cost: Especially when deployed on a large scale.\nPractical examples")


def gq(**kw):
    d = {"question": "Which statement best describes the cost effects of PaaS?",
         "options": ["Short A", "Short B", "Short C", "Short D"], "answer_index": 0,
         "explanation": "Because the document says so.", "level": "understand",
         "source_quote": "No need for large initial investments"}
    d.update(kw)
    return GeneratedQuestion(MCQ(**d), "sec", "v3", "fake", 1, 1)


def test_quote_combining_two_bullets_is_accepted():
    quote = ("✓No need for large initial investments: Save on investment costs in hardware and software. "
             "High cost: Especially when deployed on a large scale.")
    assert quote_in_source(quote, SOURCE)


def test_quote_with_one_invented_sentence_is_rejected():
    quote = ("No need for large initial investments: Save on investment costs in hardware and software. "
             "PaaS always eliminates every operating cost.")
    assert not quote_in_source(quote, SOURCE)


def test_context_covers_every_quoted_sentence():
    quote = ("No need for large initial investments: Save on investment costs. "
             "High cost: Especially when deployed on a large scale.")
    ctx = find_context(quote, SOURCE, window=80)
    assert "large initial investments" in ctx and "High cost" in ctx


def test_answer_leak_is_a_warning_not_a_rejection():
    q = gq(options=["Limitations on use of third-party tools and services", "Supplier reliability",
                    "Vendor lock-in", "Performance issues"])
    assert check_warnings(q) == ["answer_leak"]
    from quizgen.extract import Section
    verify_one(q, Section("sec", SOURCE, 1, 1), None)
    assert q.ok and q.warnings == ["answer_leak"]


def test_old_results_with_answer_leak_are_no_longer_rejected():
    assert split_labels("answer_leak") == ([], ["answer_leak"])
    assert split_labels("answer_leak;answer_mismatch") == (["answer_mismatch"], ["answer_leak"])


def test_verify_prompt_allows_paraphrases():
    p = load_prompt("verify.md")
    assert "paraphrase" in p and "-1 only if NO option" in p


def test_static_check_still_rejects_invented_quote():
    from quizgen.extract import Section
    q = gq(source_quote="Something the slides never say at all")
    assert check_static(q, Section("s", SOURCE, 1, 1)) == ["quote_not_found"]


def test_reverify_reuses_generation(tmp_path):
    secs = load_sections("data/samples/transport_layer_tcp_udp.md")
    run(secs, FakeLLM(), FakeLLM(), n_per_section=2, strategy="v1", use_st=False, out_dir=tmp_path)
    archive_previous(tmp_path, (VERIFIED,))           # what --reverify does
    gen2, ver2 = FakeLLM(), FakeLLM()
    run(secs, gen2, ver2, n_per_section=2, strategy="v1", use_st=False, out_dir=tmp_path)
    assert gen2.usage.calls == 0 and ver2.usage.calls > 0
    rows = [json.loads(x) for x in (tmp_path / VERIFIED).read_text(encoding="utf-8").splitlines()]
    assert all(r["verify_prompt"] for r in rows)
    assert (tmp_path / GENERATED).exists()
