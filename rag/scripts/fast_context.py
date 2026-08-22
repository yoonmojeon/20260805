"""Slot-based compact context compression for Fast mode."""
from __future__ import annotations

import re
from dataclasses import dataclass

from rag_answer_lib import RetrievedChunk

TOC_LINE_RE = re.compile(
    r"^(contents|table of contents|목차|index)\b|^\d+(\.\d+)*\s+\.{3,}",
    re.I,
)
DATE_RE = re.compile(
    r"\b(20\d{2}|19\d{2})[-/.](0?[1-9]|1[0-2])[-/.](0?[1-9]|[12]\d|3[01])\b"
    r"|\b(1\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s*20\d{2})\b",
    re.I,
)
NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
FOCUS_TOKEN_STOP = {
    "dnv", "mepc", "msc", "imo", "rule", "rules", "guidance", "code",
    "document", "table", "according", "what", "which",
}


def question_focus_score(text: str, question: str) -> int:
    """Length of the strongest named Latin technical anchor in ``text``."""
    lower = str(text or "").lower()
    anchors = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+_.-]{2,}", question or "")
        if token.lower() not in FOCUS_TOKEN_STOP
    }
    return max((len(anchor) for anchor in anchors if anchor in lower), default=0)


def korean_question_focus_score(text: str, question: str) -> int:
    """Coverage score for Korean rule questions against Korean source text."""
    stop = {
        "문서", "문서에서", "따르면", "어떤", "무엇", "어떻게", "경우",
        "요건", "예외", "사항", "관련", "대해", "위해", "알려줘", "정리해줘",
    }
    suffixes = (
        "으로부터", "에서부터", "에게서", "까지", "부터", "에서", "에게", "한테",
        "으로", "하고", "처럼", "보다", "라도", "이며", "이고", "의", "을", "를",
        "이", "가", "은", "는", "에", "과", "와", "로", "도", "만",
    )
    terms: set[str] = set()
    for token in re.findall(r"[가-힣]{2,}", str(question or "")):
        candidate = token
        for suffix in suffixes:
            if candidate.endswith(suffix) and len(candidate) - len(suffix) >= 2:
                candidate = candidate[: -len(suffix)]
                break
        if candidate not in stop and len(candidate) >= 3:
            terms.add(candidate)
    source = str(text or "")
    matched = {term for term in terms if term in source}
    base = sum(min(len(term), 14) for term in matched) + 3 * len(matched)
    # A section heading or opening proposition is stronger than an incidental
    # mention hundreds of characters into a neighbouring clause.
    position_bonus = sum(
        max(0, 6 - min(source.find(term), 1200) // 200)
        for term in matched
    )
    return base + position_bonus


@dataclass
class FastEvidence:
    chunk: RetrievedChunk
    slot: str


def _norm_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _strip_boilerplate(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or TOC_LINE_RE.match(ln):
            continue
        if len(ln) < 4 and not NUMERIC_RE.search(ln):
            continue
        lines.append(ln)
    return _norm_space(" ".join(lines))


def _first_sentences(text: str, max_sentences: int = 3, max_chars: int = 420) -> str:
    text = _strip_boilerplate(text)
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?。])\s+|(?<=다\.)\s+", text)
    out: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        out.append(part)
        if len(out) >= max_sentences:
            break
    joined = " ".join(out)
    if len(joined) > max_chars:
        joined = joined[: max_chars - 1] + "…"
    return joined


def _question_focused_excerpt(text: str, question: str, *, max_chars: int = 620) -> str:
    """Keep the local source proposition around a named technical anchor."""
    raw = str(text or "")
    if not raw or not question:
        return ""
    # Literal-recovery callers pass the recovered multiword source phrase as
    # ``question``.  Match that whole phrase after removing PDF visual line
    # wraps.  Token-only matching can otherwise anchor on a filename word such
    # as ``experience`` and truncate the actual proposition several hundred
    # characters later.
    normalized_raw = _norm_space(raw)
    normalized_question = _norm_space(question)
    if (
        len(normalized_question) >= 10
        and " " in normalized_question
        and re.fullmatch(r"[A-Za-z0-9+_.%,'()\-/ ]+", normalized_question)
    ):
        phrase_position = normalized_raw.lower().find(normalized_question.lower())
        if phrase_position >= 0:
            literal_max_chars = max(max_chars, 900)
            start = max(0, phrase_position - 100)
            end = min(
                len(normalized_raw),
                phrase_position + len(normalized_question) + 780,
            )
            excerpt = normalized_raw[start:end]
            return excerpt[:literal_max_chars]
    anchors = [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+_.-]{2,}", question)
        if token.lower() not in FOCUS_TOKEN_STOP
    ]
    korean_stop = {
        "문서에서", "문서에", "따르면", "어떤", "무엇", "어떻게", "경우",
        "요건", "사항", "관련", "대한", "위해", "합니다", "됩니까",
    }
    korean_anchors = [
        token
        for token in re.findall(r"[가-힣]{3,}", question)
        if token not in korean_stop
    ]
    # KR rule text and its question are in the same language.  Select the
    # window with the greatest combined query-term coverage instead of merely
    # the first page heading.  This separates, for example, the requested
    # six-hour emergency loads from a later 18-hour special-system paragraph.
    korean_positions: list[tuple[int, str]] = []
    for anchor in sorted(dict.fromkeys(korean_anchors), key=lambda value: (-len(value), value)):
        start_at = 0
        while True:
            position = raw.find(anchor, start_at)
            if position < 0:
                break
            korean_positions.append((position, anchor))
            start_at = position + max(1, len(anchor))
    if korean_positions:
        def _window_score(item: tuple[int, str]) -> tuple[int, int, int]:
            position, anchor = item
            window = raw[max(0, position - 360) : min(len(raw), position + 1300)]
            present = {term for term in korean_anchors if term in window}
            return (sum(len(term) for term in present), len(present), len(anchor))

        position, anchor = max(korean_positions, key=_window_score)
        focused_max = max(max_chars, 1800)
        start = max(0, position - 320)
        end = min(len(raw), position + len(anchor) + focused_max - 320)
        return _norm_space(raw[start:end])[:focused_max]
    # Prefer longer, more specific anchors (``two-run`` before ``V``).
    anchors = sorted(dict.fromkeys(anchors), key=lambda value: (-len(value), value))
    lower = raw.lower()
    positions = [(lower.find(anchor), anchor) for anchor in anchors if lower.find(anchor) >= 0]
    if not positions:
        return ""
    position, anchor = min(positions, key=lambda item: (-len(item[1]), item[0]))
    start = max(0, position - 220)
    end = min(
        len(raw),
        position + len(anchor) + max(400, int(max_chars) - 220),
    )
    # Align to a nearby bullet/sentence boundary without losing the anchor.
    left = max(raw.rfind("\n", start, position), raw.rfind("—", start, position))
    if left >= start:
        start = left + 1
    # PDF extraction inserts a newline at every visual line wrap.  Treating the
    # first newline as a semantic boundary reduced an English paragraph to half
    # a line (for example it kept "dual fuel medium speed" but dropped the next
    # wrapped lines naming the FUMES study).  An em dash is a real parallel-item
    # boundary in the rule clauses this helper was designed to isolate.
    right_candidates = [
        value
        for value in (raw.find("—", position + len(anchor), end),)
        if value >= 0
    ]
    if right_candidates:
        end = min(right_candidates)
    excerpt = _norm_space(raw[start:end])
    return excerpt[:max_chars]


def _regulation_names(chunk: RetrievedChunk, text: str) -> str:
    names: list[str] = []
    fname = chunk.file_name or ""
    for pat in (
        r"DNV[-\s]CG[-\s]\d+",
        r"DNV[-\s]RU[-\s]\S+",
        r"MASS\s*Code",
        r"IGC\s*Code",
        r"MARPOL",
        r"MEPC\s*\d+",
        r"MSC\s*\d+",
        r"Notice\s*No\.?\s*\d+",
        r"Section\s*\d+",
    ):
        m = re.search(pat, fname + " " + text, re.I)
        if m:
            names.append(m.group(0))
    return ", ".join(dict.fromkeys(names))


def _extract_table_evidence_text(raw: str, chunk_type: str, slot: str) -> str:
    """Keep full table KV/markdown for LLM; prose chunks stay compressed."""
    if "[Table Row KV]" in raw:
        return raw[raw.index("[Table Row KV]") :].strip()
    if chunk_type == "table_markdown" or slot == "table_markdown":
        text = raw.strip()
        return text[:1400] + ("…" if len(text) > 1400 else "")
    if chunk_type in {"table_row", "table_summary"} or slot.startswith("table"):
        return _first_sentences(raw, max_sentences=6, max_chars=900)
    return _first_sentences(raw)


def compress_evidence(ev: FastEvidence, cite_id: int, *, question: str = "") -> str:
    c = ev.chunk
    raw = c.text or ""
    text = _extract_table_evidence_text(raw, str(c.chunk_type or ""), ev.slot)
    if not c.chunk_type or not str(c.chunk_type).startswith("table"):
        focused = _question_focused_excerpt(raw, question)
        if focused:
            text = focused
    page = c.page_number if c.page_number is not None else "?"
    source = c.file_name or c.doc_id
    clause = c.clause_number or ""
    table_ref = ""
    if c.chunk_type:
        table_ref = f" table={c.table_id or '?'}"
        if c.matched_columns:
            table_ref += f" cols={','.join(c.matched_columns[:3])}"
    regs = _regulation_names(c, text)
    dates = " ".join(m.group(0) for m in DATE_RE.finditer(text))[:80]
    nums = " ".join(NUMERIC_RE.findall(text)[:4])

    meta_bits = [f"[{cite_id}] slot={ev.slot}"]
    meta_bits.append(f"source={source}")
    meta_bits.append(f"p.{page}")
    if clause:
        meta_bits.append(f"clause={clause}")
    if regs:
        meta_bits.append(f"code={regs}")
    if dates:
        meta_bits.append(f"date={dates}")
    if table_ref.strip():
        meta_bits.append(table_ref.strip())
    if nums and ev.slot.startswith("table"):
        meta_bits.append(f"values={nums}")

    header = " | ".join(meta_bits)
    return f"{header}\n{text}"


def build_slot_compact_context(
    evidence: list[FastEvidence], *, question: str = ""
) -> str:
    blocks = [
        compress_evidence(ev, i, question=question)
        for i, ev in enumerate(evidence, start=1)
    ]
    return "\n\n".join(blocks)
