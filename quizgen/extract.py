"""Bước 1 – Trích xuất bài giảng (PDF / .txt / .md) và chia thành các phần (section).

Mỗi Section giữ: tiêu đề, nội dung, trang bắt đầu – kết thúc. Số trang dùng để
trích dẫn nguồn (source_page) và để kiểm chứng câu trích (verify.py).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

MAX_TOKENS_PER_SECTION = 1500


@dataclass
class Page:
    number: int  # bắt đầu từ 1
    text: str


@dataclass
class Section:
    title: str
    text: str
    page_start: int
    page_end: int
    lines: list[tuple[int, str]] = field(default_factory=list, repr=False)

    @property
    def n_tokens(self) -> int:
        return approx_tokens(self.text)


def normalize(text: str) -> str:
    """Chuẩn hóa Unicode NFC (tiếng Việt có 2 cách mã hóa dấu), gộp khoảng trắng."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"-\n(?=\w)", "", text)  # nối từ bị ngắt dòng bằng gạch nối
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def approx_tokens(text: str) -> int:
    """Ước lượng thô: ~1 token / 3,5 ký tự (tiếng Việt tốn token hơn tiếng Anh).
    Đủ để chia phần; muốn chính xác thì dùng tokenizer của model."""
    return int(len(text) / 3.5) + 1


def load_pages(path: str | Path) -> list[Page]:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        import pymupdf  # PyMuPDF

        with pymupdf.open(path) as doc:
            return [Page(i + 1, normalize(p.get_text("text"))) for i, p in enumerate(doc)]
    if path.suffix.lower() == ".pptx":
        return load_pptx(path)
    # .txt / .md: coi dấu "\f" (form feed) là ngắt trang, nếu không có thì 1 trang
    raw = path.read_text(encoding="utf-8")
    return [Page(i + 1, normalize(t)) for i, t in enumerate(raw.split("\f"))]


def load_pptx(path: Path) -> list[Page]:
    """Mỗi slide = một 'trang'. Tiêu đề slide đặt lên đầu; lấy cả chữ trong bảng và ghi chú
    của người trình bày (speaker notes) vì giảng viên hay giải thích ở đó."""
    from pptx import Presentation

    pages = []
    for i, slide in enumerate(Presentation(str(path)).slides, 1):
        title_shape = slide.shapes.title
        has_title = title_shape is not None and title_shape.has_text_frame
        parts = [title_shape.text_frame.text] if has_title else []
        title_id = title_shape.shape_id if title_shape is not None else None
        for sh in slide.shapes:
            if sh.shape_id == title_id:  # python-pptx tạo object mới mỗi lần → so bằng id, không dùng "is"
                continue
            if sh.has_text_frame:
                parts.append(sh.text_frame.text)
            elif getattr(sh, "has_table", False) and sh.has_table:
                parts += [" | ".join(c.text for c in row.cells) for row in sh.table.rows]
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append("Ghi chú: " + notes)
        pages.append(Page(i, normalize("\n".join(parts))))
    return pages


# Dòng được coi là tiêu đề: markdown "#", "Chương 3", "1.2 Tiêu đề", hoặc dòng ngắn VIẾT HOA
HEADING_RE = re.compile(
    r"^(#{1,3}\s+.+|(chương|chapter|bài|lesson|phần|part)\s+\d+.*|\d+(\.\d+)*\.?\s+[A-ZÀ-Ỹ].{2,80})$",
    re.IGNORECASE,
)


# Ký hiệu đầu dòng hay dùng làm tiêu đề slide (PowerPoint xuất PDF)
SLIDE_TITLE_MARKS = ("❖", "■", "◆")
# Dòng chỉ là số trang: "12", "Page 12", "Trang 12", "12 / 40"
PAGE_NO_RE = re.compile(r"^((page|trang)\s*)?\d+(\s*/\s*\d+)?$", re.IGNORECASE)
# Chữ mẫu của template slide, không phải nội dung
JUNK_RE = re.compile(r"(add (your )?(content|text|title) here|click to (add|edit)|add content this time)",
                     re.IGNORECASE)


def strip_boilerplate(pages: list[Page], min_share: float = 0.3) -> list[Page]:
    """Bỏ header/footer: dòng xuất hiện trên ≥ 30% số trang (tên môn, năm, tên chương lặp lại),
    dòng chỉ là số trang, và chữ mẫu của template."""
    if len(pages) < 4:
        return pages
    count: dict[str, int] = {}
    for p in pages:
        for ln in {x.strip() for x in p.text.split("\n") if x.strip()}:
            count[ln] = count.get(ln, 0) + 1
    repeated = {ln for ln, c in count.items() if c >= max(3, min_share * len(pages))}
    out = []
    for p in pages:
        keep = [ln for ln in p.text.split("\n")
                if ln.strip() not in repeated and not PAGE_NO_RE.match(ln.strip()) and not JUNK_RE.search(ln)]
        out.append(Page(p.number, "\n".join(keep).strip()))
    return out


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 90:
        return False
    if s.startswith(SLIDE_TITLE_MARKS) and len(s.split()) <= 12:
        return True
    if HEADING_RE.match(s):
        return True
    letters = [c for c in s if c.isalpha()]
    return len(letters) >= 4 and all(c.isupper() for c in letters) and len(s.split()) <= 10


def is_slide_deck(pages: list[Page]) -> bool:
    """Slide xuất PDF: nhiều trang, mỗi trang ít chữ (< ~900 ký tự)."""
    if len(pages) < 5:
        return False
    return sum(len(p.text) for p in pages) / len(pages) < 900


def _page_section(p: Page) -> Section:
    """Mỗi slide là một đơn vị; dòng đầu (sau khi bỏ header) làm tiêu đề nếu đủ ngắn."""
    lines = [ln for ln in p.text.split("\n") if ln.strip()]
    first = lines[0].strip() if lines else ""
    if first and len(first.split()) <= 12:
        title, body = first.lstrip("".join(SLIDE_TITLE_MARKS) + "#• ").strip(), lines[1:]
    else:
        title, body = f"Trang {p.number}", lines
    return Section(title, "\n".join(body).strip(), p.number, p.number, [(p.number, ln) for ln in body])


def split_sections(pages: list[Page], max_tokens: int = MAX_TOKENS_PER_SECTION,
                   min_tokens: int = 300) -> list[Section]:
    """Lọc header/footer → chia theo tiêu đề → cắt phần quá dài → gộp phần quá ngắn
    (slide thường chỉ 50–150 token/trang: hỏi từng slide thì ít ý, tốn nhiều lần gọi)."""
    pages = strip_boilerplate(pages)
    if is_slide_deck(pages):
        raw = [_page_section(p) for p in pages if p.text.strip()]
        out = [x for s in raw for x in _split_long(s, max_tokens)]
        out = _merge_short(out, min_tokens, max_tokens)
        return [s for s in out if s.n_tokens >= 40]
    lines: list[tuple[int, str]] = [(p.number, ln) for p in pages for ln in p.text.split("\n")]
    raw: list[Section] = []
    cur_title, cur_lines = "Mở đầu", []
    for page_no, ln in lines:
        if is_heading(ln):
            if any(t.strip() for _, t in cur_lines):
                raw.append(_make(cur_title, cur_lines))
            cur_title, cur_lines = ln.strip().lstrip("#").strip(), []
        else:
            cur_lines.append((page_no, ln))
    if any(t.strip() for _, t in cur_lines):
        raw.append(_make(cur_title, cur_lines))

    # bỏ phần gần như chỉ có tiêu đề (mục lục) TRƯỚC khi gộp, để không gộp mục lục thành một phần
    raw = [s for s in raw if s.n_tokens >= 25]
    out: list[Section] = []
    for s in raw:
        out.extend(_split_long(s, max_tokens))
    out = _merge_short(out, min_tokens, max_tokens)
    # bỏ phần quá ngắn (< 40 token) – không đủ nội dung để hỏi
    return [s for s in out if s.n_tokens >= 40]


def _merge_short(secs: list[Section], min_tokens: int, max_tokens: int) -> list[Section]:
    """Gộp các phần liền nhau cho tới khi đạt min_tokens (không vượt max_tokens).
    Tiêu đề con được giữ trong nội dung để model vẫn thấy cấu trúc."""
    merged: list[Section] = []
    for s in secs:
        prev = merged[-1] if merged else None
        if prev is not None and prev.n_tokens < min_tokens and prev.n_tokens + s.n_tokens <= max_tokens:
            same = s.title == prev.title or s.title in prev.title
            title = prev.title if same else f"{prev.title} · {s.title}"
            lines = prev.lines + [(s.page_start, ""), (s.page_start, s.title)] + s.lines
            merged[-1] = _make(title[:120], lines)
        else:
            merged.append(s)
    return merged


def _make(title: str, lines: list[tuple[int, str]]) -> Section:
    pages = [p for p, _ in lines] or [1]
    text = "\n".join(t for _, t in lines).strip()
    return Section(title, text, min(pages), max(pages), lines)


def _split_long(s: Section, max_tokens: int) -> list[Section]:
    if s.n_tokens <= max_tokens:
        return [s]
    parts, buf = [], []
    for item in s.lines:
        buf.append(item)
        size = approx_tokens("\n".join(t for _, t in buf))
        # cắt ở dòng trống khi đủ dài; không có dòng trống thì cắt cứng ở 130%
        if (size >= max_tokens and item[1].strip() == "") or size >= max_tokens * 1.3:
            parts.append(buf)
            buf = []
    if buf:
        parts.append(buf)
    return [_make(f"{s.title} ({i + 1}/{len(parts)})", p) for i, p in enumerate(parts)]


def load_sections(path: str | Path, max_tokens: int = MAX_TOKENS_PER_SECTION) -> list[Section]:
    return split_sections(load_pages(path), max_tokens)
