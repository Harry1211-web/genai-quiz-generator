import pytest
from pydantic import ValidationError

from eval.kappa import cohen_kappa
from quizgen.export import to_gift
from quizgen.extract import load_sections, normalize
from quizgen.generate import GeneratedQuestion, generate_for_section
from quizgen.llm import FakeLLM, extract_json, generate_json
from quizgen.schema import MCQ, MCQList
from quizgen.verify import check_static, quote_in_source

SAMPLE = "data/samples/transport_layer_tcp_udp.md"
SAMPLE_VI = "data/samples/transport_layer_tcp_udp_vi.md"


def mcq(**kw):
    d = {"question": "What is the default port of HTTPS?", "options": ["443", "80", "22", "8080"],
         "answer_index": 0, "explanation": "HTTPS uses port 443.", "level": "remember",
         "source_quote": "HTTPS uses port 443"}
    d.update(kw)
    return MCQ(**d)


@pytest.mark.parametrize("path", [SAMPLE, SAMPLE_VI])
def test_sections(path):
    secs = load_sections(path)
    assert len(secs) >= 2
    assert any("UDP" in s.title or "UDP" in s.text for s in secs)


def test_nfc_vietnamese():
    # "ế" written as e + combining circumflex + combining acute must become one code point
    assert normalize("Tiếng") == "Tiếng"


def test_schema_rejects_duplicate_options():
    with pytest.raises(ValidationError):
        mcq(options=["443", "443", "22", "80"])


def test_schema_rejects_bad_index():
    with pytest.raises(ValidationError):
        mcq(answer_index=4)


def test_schema_rejects_all_of_the_above():
    with pytest.raises(ValidationError):
        mcq(options=["443", "80", "22", "All of the above"])


def test_extract_json_from_fence():
    assert extract_json('Here you go:\n```json\n{"a": 1}\n```') == '{"a": 1}'


class FlakyLLM(FakeLLM):
    """Returns broken JSON first, valid JSON afterwards - exercises the retry loop."""

    def chat(self, messages, temperature=0.2, json_mode=True):
        self.usage.calls += 1
        if self.usage.calls == 1:
            return '{"questions": [ {"question": "missing fields"} ]}'
        return MCQList(questions=[mcq()]).model_dump_json()


def test_retry_on_format_error():
    llm = FlakyLLM()
    out = generate_json(llm, [{"role": "user", "content": "x"}], MCQList)
    assert out is not None and llm.usage.format_errors == 1 and llm.usage.calls == 2


@pytest.mark.parametrize("strategy", ["v1", "v2", "v3"])
def test_generate_fake(strategy):
    sec = load_sections(SAMPLE)[0]
    qs = generate_for_section(FakeLLM(), sec, n=2, strategy=strategy, seed=1)
    assert 1 <= len(qs) <= 2
    for q in qs:
        assert q.mcq.options[q.mcq.answer_index]


def test_quote_check():
    src = "HTTP uses port 80, HTTPS uses port 443 and SSH uses port 22."
    assert quote_in_source("HTTPS uses port 443", src)
    assert quote_in_source("HTTPS  uses port 443.", src)  # different whitespace/punctuation
    assert not quote_in_source("HTTPS encrypts traffic with the RSA algorithm", src)


def test_quote_check_vietnamese():
    src = "HTTP dùng cổng 80, HTTPS dùng cổng 443 và SSH dùng cổng 22."
    assert quote_in_source("HTTPS dùng cổng 443", src)


def test_fabricated_quote_rejected():
    sec = load_sections(SAMPLE)[0]
    q = GeneratedQuestion(mcq(source_quote="This sentence does not appear in the original document"),
                          sec.title, "v1", "fake", 1, 1)
    assert "quote_not_found" in check_static(q, sec)


def test_gift_escape():
    g = to_gift([mcq(question="Value {x} = 1: true?")])
    assert "\\{x\\} \\= 1\\:" in g and "=443" in g and "~80" in g


def test_kappa():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert abs(cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-9


def test_split_long_section():
    from quizgen.extract import Page, split_sections
    body = "\n\n".join(f"A fairly long sentence about computer networking, number {i}." for i in range(400))
    secs = split_sections([Page(1, "# Long\n" + body)], max_tokens=300)
    assert len(secs) > 3 and all(s.n_tokens <= 300 * 1.4 for s in secs)


def test_slide_deck_boilerplate_removed():
    """Real-world case: a Vietnamese slide deck with a repeated header and footer on every slide."""
    from quizgen.extract import Page, split_sections
    pages = [Page(i, f"TRUNG TÂM DỮ LIỆU\nTopic {i}\n" + f"Slide {i} content about servers and networks. " * 8
                  + f"\nĐiện toán đám mây - Hạ tầng\n{i}") for i in range(1, 11)]
    secs = split_sections(pages)
    text = "\n".join(s.text + s.title for s in secs)
    assert "TRUNG TÂM DỮ LIỆU" not in text and "Hạ tầng" not in text
    assert "Topic 1" in text and len(secs) < 10   # short slides are merged


def _fake_post(responses):
    """Replace httpx.post with a function returning the given responses in order."""
    import httpx
    calls = []

    def post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        status, payload = responses[len(calls) - 1]
        return httpx.Response(status, json=payload, request=httpx.Request("POST", url))
    return post, calls


def test_temperature_not_sent_to_api_models_by_default(monkeypatch):
    import httpx

    from quizgen.llm import OpenAICompatLLM, build_llm
    ok = {"choices": [{"message": {"content": "{}"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}
    post, calls = _fake_post([(200, ok), (200, ok)])
    monkeypatch.setattr(httpx, "post", post)
    OpenAICompatLLM(model="m", api_key="k").chat([{"role": "user", "content": "json"}], temperature=0.0)
    assert "temperature" not in calls[0] and len(calls) == 1      # one call, no retry
    llm = build_llm({"model": "m", "temperature": "true"})
    llm.chat([{"role": "user", "content": "json"}], temperature=0.3)
    assert calls[1]["temperature"] == 0.3                         # LLM_TEMPERATURE_<n>=true turns it on


def test_other_400_shows_provider_message(monkeypatch):
    import httpx

    from quizgen.llm import LLMRequestError, OpenAICompatLLM
    post, calls = _fake_post([(400, {"error": {"message": "The model 'x' does not exist"}})])
    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(LLMRequestError, match="does not exist"):
        OpenAICompatLLM(model="x", api_key="k").chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 1   # 4xx is not retried


# ---- extraction and prompt-version fixes (Sprint 2)
def test_front_matter_slides_are_detected():
    from quizgen.extract import Page, front_matter_kind
    assert front_matter_kind(Page(1, "WELCOME TO\nCLOUD COMPUTING")) == "cover"
    assert front_matter_kind(Page(2, "Storage Systems\nPART 07")) == "cover"
    assert front_matter_kind(Page(3, "INTRODUCTION\nTopics\n1. Disks\n2. Arrays\n3. Backup")) == "agenda"
    assert front_matter_kind(Page(3, "Mục tiêu\n• Hiểu cách lưu trữ dữ liệu\n• Mô tả RAID")) == "objectives"
    content = Page(9, "❖ Benefits of RAID\n1. Faster reads: data is striped over disks.\n"
                      "2. Fault tolerance: a mirror keeps a copy.\n3. Hot spare: replaces a failed disk.")
    assert front_matter_kind(content) is None


def test_prompt_and_sections_versions_change_with_their_inputs():
    from quizgen.extract import Section
    from quizgen.generate import gen_prompt_version, sections_version
    assert gen_prompt_version("v1") != gen_prompt_version("v2") != gen_prompt_version("v3")
    a = [Section("T", "text one", 1, 1)]
    assert sections_version(a) == sections_version([Section("T", "text one", 1, 1)])
    assert sections_version(a) != sections_version([Section("T", "text two", 1, 1)])


def test_pdf_words_without_space_characters_are_split(tmp_path):
    """A PDF that draws words with a visual gap but no space character (seen in a Vietnamese deck)."""
    import pymupdf

    from quizgen.extract import pdf_page_text
    doc = pymupdf.open()
    page = doc.new_page()
    x = 72
    for word in ["data", "center"]:  # the built-in test font has no Vietnamese glyphs
        page.insert_text((x, 100), word, fontsize=20, fontname="helv")
        x += pymupdf.get_text_length(word, fontname="helv", fontsize=20) + 8  # gap, no space char
    path = tmp_path / "gap.pdf"
    doc.save(path)
    with pymupdf.open(path) as d:
        assert "data center" in pdf_page_text(d[0])


def test_back_matter_slides_are_detected():
    from quizgen.extract import Page, back_matter_kind
    assert back_matter_kind(Page(40, "THANKS\nQ & A")) == "closing"
    assert back_matter_kind(Page(41, "End of file")) == "closing"
    assert back_matter_kind(Page(39, "11 Summary\nWe built a stack from an array")) == "summary"
    assert back_matter_kind(Page(39, "7 Summary (2/2)\nTry to add more methods")) == "summary"
    assert back_matter_kind(Page(42, "❖ Summary of Chapter 3\nTopics we covered")) == "summary"
    assert back_matter_kind(Page(43, "8 Practice Exercises\nExercise #1: reverse a list")) == "summary"
    assert back_matter_kind(Page(9, "Stack summary of operations\npush adds an item on top")) is None
    assert back_matter_kind(Page(9, "❖ Thanks to virtualization, one server runs many machines")) is None


def test_normalize_removes_odd_characters():
    from quizgen.extract import normalize
    out = normalize("Stack\x0bpush​ pop  item\x07end")
    assert out == "Stack\npush pop • item end"


def test_sheet_saved_by_excel_is_read(tmp_path):
    """Excel saves 'CSV UTF-8' with a BOM, and may use ';' as the separator (region settings)."""
    import pandas as pd

    from eval.kappa import read_sheet, write_sheet
    df = pd.DataFrame({"id": ["q1"], "question": ["Library’s size — big?"], "factual": [1]})
    write_sheet(df, tmp_path / "a.csv")
    assert (tmp_path / "a.csv").read_bytes()[:3] == b"\xef\xbb\xbf"
    assert read_sheet(tmp_path / "a.csv").equals(df)
    (tmp_path / "b.csv").write_bytes("﻿id;question;factual\nq1;Library’s size — big?;1\n".encode())
    assert read_sheet(tmp_path / "b.csv").equals(df)
