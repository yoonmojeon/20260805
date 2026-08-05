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


def compress_evidence(ev: FastEvidence, cite_id: int) -> str:
    c = ev.chunk
    raw = c.text or ""
    text = _extract_table_evidence_text(raw, str(c.chunk_type or ""), ev.slot)
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


def build_slot_compact_context(evidence: list[FastEvidence]) -> str:
    blocks = [compress_evidence(ev, i) for i, ev in enumerate(evidence, start=1)]
    return "\n\n".join(blocks)
