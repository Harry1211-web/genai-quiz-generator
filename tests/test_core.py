import pytest
from pydantic import ValidationError

from eval.kappa import cohen_kappa
from quizgen.export import to_gift
from quizgen.extract import load_sections, normalize
from quizgen.generate import GeneratedQuestion, generate_for_section
from quizgen.llm import FakeLLM, extract_json, generate_json
from quizgen.schema import MCQ, MCQList
from quizgen.verify import check_static, quote_in_source

SAMPLE = "data/samples/mang_may_tinh_tcp.md"

def mcq(**kw):
    d = dict(question="Cổng mặc định của HTTPS là gì?", options=["443", "80", "22", "8080"],
             answer_index=0, explanation="HTTPS dùng cổng 443.", level="remember",
             source_quote="HTTPS dùng cổng 443")
    d.update(kw)
    return MCQ(**d)

def test_sections():
    secs = load_sections(SAMPLE)
    assert len(secs) >= 2
    assert any("UDP" in s.title or "UDP" in s.text for s in secs)

def test_nfc():
    assert normalize("Tie\u0302\u0301ng") == "Tiếng"

def test_schema_rejects_duplicate_options():
    with pytest.raises(ValidationError):
        mcq(options=["443", "443", "22", "80"])

def test_schema_rejects_bad_index():
    with pytest.raises(ValidationError):
        mcq(answer_index=4)

def test_extract_json_from_fence():
    assert extract_json('Đây:\n```json\n{"a": 1}\n```') == '{"a": 1}'

class FlakyLLM(FakeLLM):
    """Lần đầu trả JSON hỏng, lần sau trả đúng – kiểm tra vòng retry."""
    def chat(self, messages, temperature=0.2, json_mode=True):
        self.usage.calls += 1
        if self.usage.calls == 1:
            return '{"questions": [ {"question": "thiếu trường"} ]}'
        return MCQList(questions=[mcq()]).model_dump_json()

def test_retry_on_format_error():
    llm = FlakyLLM()
    out = generate_json(llm, [{"role": "user", "content": "x"}], MCQList)
    assert out is not None and llm.usage.format_errors == 1 and llm.usage.calls == 2

@pytest.mark.parametrize("strategy", ["v1", "v2", "v3"])
def test_generate_fake(strategy):
    sec = load_sections(SAMPLE)[1]
    qs = generate_for_section(FakeLLM(), sec, n=2, strategy=strategy, seed=1)
    assert 1 <= len(qs) <= 2
    for q in qs:
        assert q.mcq.options[q.mcq.answer_index]

def test_quote_check():
    src = "HTTP dùng cổng 80, HTTPS dùng cổng 443 và SSH dùng cổng 22."
    assert quote_in_source("HTTPS dùng cổng 443", src)
    assert quote_in_source("HTTPS  dùng cổng 443.", src)  # khác khoảng trắng/dấu câu
    assert not quote_in_source("HTTPS mã hóa bằng thuật toán RSA", src)

def test_fabricated_quote_rejected():
    sec = load_sections(SAMPLE)[0]
    q = GeneratedQuestion(mcq(source_quote="Câu này không hề có trong tài liệu gốc"), sec.title,
                          "v1", "fake", 1, 1)
    assert "quote_not_found" in check_static(q, sec)

def test_gift_escape():
    g = to_gift([mcq(question="Giá trị {x} = 1: đúng?")])
    assert "\\{x\\} \\= 1\\:" in g and "=443" in g and "~80" in g

def test_kappa():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert abs(cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-9

def test_split_long_section():
    from quizgen.extract import Page, split_sections
    body = "\n\n".join(f"Đây là một câu khá dài về giao thức mạng số {i}." for i in range(400))
    text = "# Dài\n" + body
    secs = split_sections([Page(1, text)], max_tokens=300)
    assert len(secs) > 3 and all(s.n_tokens <= 300 * 1.4 for s in secs)


def test_slide_deck_boilerplate_removed():
    from quizgen.extract import Page, split_sections
    pages = [Page(i, f"TRUNG TÂM DỮ LIỆU\nChủ đề số {i}\n" + f"Nội dung slide {i} về máy chủ và mạng. " * 8
                  + f"\nĐiện toán đám mây - Hạ tầng\n{i}") for i in range(1, 11)]
    secs = split_sections(pages)
    text = "\n".join(s.text + s.title for s in secs)
    assert "TRUNG TÂM DỮ LIỆU" not in text and "Hạ tầng" not in text
    assert "Chủ đề số 1" in text and len(secs) < 10   # slide ngắn được gộp lại
