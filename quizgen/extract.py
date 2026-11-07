"""Step 2 - Extract lecture text (PDF / PPTX / TXT / MD) and split it into sections.

Each Section keeps its title, text and start/end page. Page numbers are used for
source citations (source_page) and to verify source quotes (verify.py).

Pipeline: load pages -> strip boilerplate (headers, footers, page numbers)
-> split into units (per heading, or per slide for slide decks)
-> split units that are too long -> merge consecutive units that are too short.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

MAX_TOKENS_PER_SECTION = 1500
MIN_TOKENS_PER_SECTION = 300  # ~enough content for 3 distinct questions


@dataclass
class Page:
    number: int  # 1-based position in the file (not the number printed on the slide)
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
    """Unicode NFC normalization (Vietnamese diacritics have two encodings), whitespace cleanup."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"[\x0b\x0c\u2028\u2029]", "\n", text)  # PowerPoint soft line break, page break
    text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", text)  # zero-width characters
    text = re.sub(r"[\ue000-\uf8ff]", "•", text)  # private-use glyphs of symbol fonts (Wingdings bullets)
    text = re.sub(r"[\x00-\x08\x0e-\x1f\x7f]", " ", text)  # other control characters
    text = re.sub(r"-\n(?=\w)", "", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def approx_tokens(text: str) -> int:
    """Rough estimate: ~1 token per 3.5 characters (Vietnamese uses more tokens than English).
    Good enough for chunking; costs are computed from the real token counts returned by the API."""
    return int(len(text) / 3.5) + 1


def pdf_page_text(page, gap: float = 0.2) -> str:
    """Text of a PDF page rebuilt from the position of every character.
    Some PDFs (seen in a Vietnamese deck) contain NO space character between words, only a visual gap,
    so get_text("text") returns "vềhạtầng" instead of "về hạ tầng". A space is inserted wherever the gap
    between two characters is larger than `gap` x font size (normal letter spacing is ~0.02 x size)."""
    lines = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            buf: list[str] = []
            prev_right = None
            for span in line["spans"]:
                for ch in span["chars"]:
                    c = ch["c"]
                    if (prev_right is not None and c != " " and buf and buf[-1] != " "
                            and ch["bbox"][0] - prev_right > gap * span["size"]):
                        buf.append(" ")
                    buf.append(c)
                    prev_right = ch["bbox"][2]
            lines.append("".join(buf))
    return "\n".join(lines)


def load_pages(path: str | Path) -> list[Page]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pymupdf  # PyMuPDF

        with pymupdf.open(path) as doc:
            return [Page(i + 1, normalize(pdf_page_text(p))) for i, p in enumerate(doc)]
    if suffix == ".pptx":
        return load_pptx(path)
    # .txt / .md: a form feed ("\f") marks a page break; otherwise the whole file is one page
    raw = path.read_text(encoding="utf-8")
    return [Page(i + 1, normalize(t)) for i, t in enumerate(raw.split("\f"))]


def load_pptx(path: Path) -> list[Page]:
    """One slide = one page. The slide title goes first; table text and speaker notes are
    included because lecturers often put explanations in the notes."""
    from pptx import Presentation

    pages = []
    for i, slide in enumerate(Presentation(str(path)).slides, 1):
        title_shape = slide.shapes.title
        has_title = title_shape is not None and title_shape.has_text_frame
        parts = [title_shape.text_frame.text] if has_title else []
        title_id = title_shape.shape_id if title_shape is not None else None
        for sh in slide.shapes:
            # python-pptx returns a new proxy object on each access -> compare ids, not identity
            if sh.shape_id == title_id:
                continue
            if sh.has_text_frame:
                parts.append(sh.text_frame.text)
            elif getattr(sh, "has_table", False) and sh.has_table:
                parts += [" | ".join(c.text for c in row.cells) for row in sh.table.rows]
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append("Notes: " + notes)
        pages.append(Page(i, normalize("\n".join(parts))))
    return pages


# Heading lines: markdown "#", "Chapter 3" / "Chương 3", "1.2 Title", or a short ALL-CAPS line.
# Vietnamese keywords (chương, bài, phần) are kept on purpose: the tool supports Vietnamese lectures.
HEADING_RE = re.compile(
    r"^(#{1,3}\s+.+|(chương|chapter|bài|lesson|phần|part)\s+\d+.*|\d+(\.\d+)*\.?\s+[A-ZÀ-Ỹ].{2,80})$",
    re.IGNORECASE,
)
# Bullet symbols commonly used as slide titles in PowerPoint exports
SLIDE_TITLE_MARKS = ("❖", "■", "◆")
# Lines that are only a page number: "12", "Page 12", "Trang 12", "12 / 40"
PAGE_NO_RE = re.compile(r"^((page|trang)\s*)?\d+(\s*/\s*\d+)?$", re.IGNORECASE)
# Placeholder text left over from slide templates
JUNK_RE = re.compile(r"(add (your )?(content|text|title) here|click to (add|edit)|add content this time)",
                     re.IGNORECASE)


def strip_boilerplate(pages: list[Page], min_share: float = 0.3) -> list[Page]:
    """Remove headers/footers. PDFs do not mark which line is a header, so we infer it:
    a line repeated on >= 30% of pages (course name, year, chapter title), a line that is
    only a page number, or template placeholder text."""
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


# Slides that hold no testable content: cover, agenda / table of contents, learning objectives,
# section dividers. Questions generated from them are about the course outline, not the subject
# (and an agenda may even contradict the slides, e.g. "Single-tasking" vs "Single-tenant").
COVER_RE = re.compile(r"\b(welcome to|part|chapter|chương|phần|lecture|bài)\s*\d*\b", re.IGNORECASE)
TOC_LINE_RE = re.compile(r"^(introduction|agenda|contents|table of contents|outline|nội dung|mục lục)$",
                         re.IGNORECASE)
OBJECTIVES_LINE_RE = re.compile(r"^(learning )?(objectives|outcomes|mục tiêu( bài học)?)$", re.IGNORECASE)
NUMBERED_RE = re.compile(r"^\d{1,2}[.)]\s*\S")
BULLETS = "❖■◆➢✓•●▪#-* "


def front_matter_kind(p: Page) -> str | None:
    """'cover', 'agenda', 'objectives', 'divider' or None for a slide with real content."""
    lines = [ln.strip().strip(BULLETS).strip() for ln in p.text.split("\n") if ln.strip().strip(BULLETS)]
    words = len(p.text.split())
    if not lines:
        return None
    short = words <= 15 and not any(len(ln.split()) > 8 for ln in lines)
    if short and (p.number <= 2 or COVER_RE.search(p.text)):
        return "cover"
    if words <= 80 and any(TOC_LINE_RE.match(ln) for ln in lines):
        return "agenda"
    if words <= 120 and any(OBJECTIVES_LINE_RE.match(ln) for ln in lines):
        return "objectives"
    if short and len(lines) <= 2 and any(NUMBERED_RE.match(ln) for ln in lines):
        return "divider"  # e.g. "Part title / 1. History of cloud computing" (not a numbered content slide)
    return None


# Slides / pages at the END of a lecture that hold no new content: closing slide, summary, practice
# exercises, references and links. Questions generated from them are about the lecture itself
# ("Which approach does the summary suggest?"), not about the subject.
CLOSING_RE = re.compile(r"^(thanks?( you)?( for watching)?|q\s*&\s*a|questions\??|end of (file|lecture)|"
                        r"the end|cảm ơn|hỏi đáp|notes)$", re.IGNORECASE)
BACK_TITLE_RE = re.compile(
    r"^(summary|summary of (chapter|part|lecture|module) \w+|conclusions?|recap|wrap[- ]up|"
    r"practice exercises|exercises|homework|references( (&|and) further reading)?|further reading|"
    r"bibliography|visuali[sz]ing data structures|tổng kết|tóm tắt|kết luận|bài tập|"
    r"tài liệu tham khảo)$", re.IGNORECASE)
SECTION_NO_RE = re.compile(r"^\d+(\.\d+)*\.?\s+|\s*\(\d+/\d+\)\s*$")


def back_matter_kind(p: Page) -> str | None:
    """'closing', 'summary' (incl. exercises, references, links) or None."""
    lines = [ln.strip().strip(BULLETS).strip() for ln in p.text.split("\n") if ln.strip().strip(BULLETS)]
    if not lines:
        return None
    words = len(p.text.split())
    if words <= 12 and any(CLOSING_RE.match(ln) for ln in lines):
        return "closing"
    if words <= 300 and BACK_TITLE_RE.match(SECTION_NO_RE.sub("", lines[0]).strip()):
        return "summary"
    return None


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
    """Slides exported to PDF/PPTX: many pages with little text each (< ~900 characters)."""
    if len(pages) < 5:
        return False
    return sum(len(p.text) for p in pages) / len(pages) < 900


def _page_section(p: Page) -> Section:
    """One slide = one unit; its first line (after boilerplate removal) is the title if short enough."""
    lines = [ln for ln in p.text.split("\n") if ln.strip()]
    first = lines[0].strip() if lines else ""
    if first and len(first.split()) <= 12:
        title, body = first.strip(BULLETS).strip(), lines[1:]
    else:
        title, body = f"Page {p.number}", lines
    return Section(title, "\n".join(body).strip(), p.number, p.number, [(p.number, ln) for ln in body])


def split_sections(pages: list[Page], max_tokens: int = MAX_TOKENS_PER_SECTION,
                   min_tokens: int = MIN_TOKENS_PER_SECTION) -> list[Section]:
    """Strip boilerplate -> split by heading (or by slide) -> split long units -> merge short ones.
    Slides usually hold only 50-150 tokens each: one section per slide gives too few ideas
    per question batch and many more LLM calls."""
    pages = strip_boilerplate(pages)
    if is_slide_deck(pages):
        pages = [p for p in pages if front_matter_kind(p) is None and back_matter_kind(p) is None]
        raw = [_page_section(p) for p in pages if p.text.strip()]
        out = [x for s in raw for x in _split_long(s, max_tokens)]
        out = _merge_short(out, min_tokens, max_tokens)
        return [s for s in out if s.n_tokens >= 40]

    pages = [p for p in pages if back_matter_kind(p) is None]  # references, notes, closing page
    lines: list[tuple[int, str]] = [(p.number, ln) for p in pages for ln in p.text.split("\n")]
    raw: list[Section] = []
    cur_title, cur_lines = "Introduction", []
    for page_no, ln in lines:
        if is_heading(ln):
            if any(t.strip() for _, t in cur_lines):
                raw.append(_make(cur_title, cur_lines))
            cur_title, cur_lines = ln.strip().lstrip("#").strip(), []
        else:
            cur_lines.append((page_no, ln))
    if any(t.strip() for _, t in cur_lines):
        raw.append(_make(cur_title, cur_lines))

    # drop heading-only units (tables of contents) BEFORE merging, so they are not merged into a section
    raw = [s for s in raw if s.n_tokens >= 25]
    out: list[Section] = []
    for s in raw:
        out.extend(_split_long(s, max_tokens))
    out = _merge_short(out, min_tokens, max_tokens)
    # drop sections that are still too short (< 40 tokens) to ask anything about
    return [s for s in out if s.n_tokens >= 40]


def _merge_short(secs: list[Section], min_tokens: int, max_tokens: int) -> list[Section]:
    """Merge consecutive sections until each reaches min_tokens (never above max_tokens).
    Sub-titles are kept inside the text so the model still sees the structure."""
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
        # cut at a blank line once long enough; without blank lines, hard cut at 130%
        if (size >= max_tokens and item[1].strip() == "") or size >= max_tokens * 1.3:
            parts.append(buf)
            buf = []
    if buf:
        parts.append(buf)
    return [_make(f"{s.title} ({i + 1}/{len(parts)})", p) for i, p in enumerate(parts)]


def load_sections(path: str | Path, max_tokens: int = MAX_TOKENS_PER_SECTION) -> list[Section]:
    return split_sections(load_pages(path), max_tokens)
