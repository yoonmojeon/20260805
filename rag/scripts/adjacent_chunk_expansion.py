"""Adjacent chunk expansion for Fast mode evidence slots.

Regulation text puts a requirement and the exception that limits it in
consecutive paragraphs, and the indexer cuts long paragraphs into ``_sNN``
pieces.  A selected chunk therefore often stops mid-clause, and the sentence
that decides the answer sits in the next chunk which no slot selected.

This module appends that neighbour from the document's own chunk order.  It
only fires on a continuation signal (split sibling, exception marker, unfinished
sentence) so the Fast context stays inside its 4k token budget.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from fast_context import FastEvidence

DEFAULT_MAX_NEIGHBORS = 2
NEIGHBOR_BODY_CHARS = 700
# One Korean sentence of substance runs well past this; anything shorter is a
# page number, a table label or a stray line.
MIN_NEIGHBOR_BODY = 20

EXCEPTION_HEAD_RE = re.compile(
    r"^\W{0,3}(?:다만|단(?:,|\s)|예외|이\s*경우|그러나|또한|이때|위\s*규정에|"
    r"however|except|unless|provided\s+that|notwithstanding|in\s+addition)",
    re.I,
)
SENTENCE_END_RE = re.compile(r"(?:[.!?;。:]|다|함|것|음|됨|요|호|목)\s*[)\]]?\s*$")
SPLIT_SUFFIX_RE = re.compile(r"_s(\d+)$")
ELEMENT_SUFFIX_RE = re.compile(r"__e\d+$")
METADATA_LINE_RE = re.compile(r"^\[(?:dnv|abs|kr|lr|msc|mepc|figure|table)\]", re.I)

_DOC_ORDER_CACHE: dict[str, tuple[tuple[dict[str, Any], ...], dict[str, int]]] = {}


def expansion_enabled() -> bool:
    return os.environ.get("MARITIME_ADJACENT_EXPANSION", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def max_neighbors() -> int:
    raw = os.environ.get("MARITIME_ADJACENT_EXPANSION_MAX", "").strip()
    try:
        return max(0, int(raw)) if raw else DEFAULT_MAX_NEIGHBORS
    except ValueError:
        return DEFAULT_MAX_NEIGHBORS


def clear_doc_order_cache() -> None:
    _DOC_ORDER_CACHE.clear()


def _body(text: str) -> str:
    lines = [ln for ln in (text or "").splitlines() if not METADATA_LINE_RE.match(ln.strip())]
    return "\n".join(lines).strip()


def _base_chunk_id(chunk_id: str) -> str:
    return ELEMENT_SUFFIX_RE.sub("", str(chunk_id or ""))


def _paragraph_stem(chunk_id: str) -> str:
    return SPLIT_SUFFIX_RE.sub("", _base_chunk_id(chunk_id))


def _doc_order(
    chunks_dir: Path, doc_id: str
) -> tuple[tuple[dict[str, Any], ...], dict[str, int]]:
    """Ordered chunk rows of one document, cached per (chunks_dir, doc_id)."""
    key = f"{chunks_dir}|{doc_id}"
    cached = _DOC_ORDER_CACHE.get(key)
    if cached is not None:
        return cached

    rows: list[dict[str, Any]] = []
    path = Path(chunks_dir) / doc_id / "chunks.jsonl"
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk_id = str(raw.get("chunk_id") or raw.get("element_id") or "")
                if not chunk_id:
                    continue
                rows.append(
                    {
                        "chunk_id": chunk_id,
                        "page_number": raw.get("page_number"),
                        "clause_number": str(raw.get("clause_number") or ""),
                        "element_type": str(raw.get("element_type") or "text"),
                        "text": str(raw.get("text") or ""),
                    }
                )
    index = {row["chunk_id"]: pos for pos, row in enumerate(rows)}
    out = (tuple(rows), index)
    _DOC_ORDER_CACHE[key] = out
    return out


def _position(index: dict[str, int], chunk_id: str) -> int | None:
    for candidate in (str(chunk_id or ""), _base_chunk_id(chunk_id), _paragraph_stem(chunk_id)):
        pos = index.get(candidate)
        if pos is not None:
            return pos
    return None


def _neighbor_row(
    chunks_dir: Path, chunk: Any, direction: int
) -> dict[str, Any] | None:
    doc_id = str(getattr(chunk, "doc_id", "") or "")
    if not doc_id:
        return None
    rows, index = _doc_order(chunks_dir, doc_id)
    if not rows:
        return None
    pos = _position(index, str(getattr(chunk, "chunk_id", "")))
    if pos is None:
        return None
    target = pos + direction
    if not 0 <= target < len(rows):
        return None
    row = rows[target]
    page = getattr(chunk, "page_number", None)
    neighbor_page = row.get("page_number")
    # Doc order crosses chapter boundaries too; a jump of more than one page
    # means the neighbour belongs to a different section.
    if page is not None and neighbor_page is not None:
        try:
            if abs(int(neighbor_page) - int(page)) > 1:
                return None
        except (TypeError, ValueError):
            pass
    return row


def _looks_unfinished(body: str) -> bool:
    """True for a paragraph cut mid-sentence, and for a bare clause heading."""
    return not SENTENCE_END_RE.search(body)


def _starts_with_exception(body: str) -> bool:
    return bool(EXCEPTION_HEAD_RE.match(body))


def expansion_reason(
    parent: Any,
    neighbor: dict[str, Any],
    direction: int,
) -> str | None:
    """Why (or whether) this neighbour is worth spending context on."""
    parent_body = _body(getattr(parent, "text", ""))
    neighbor_body = _body(str(neighbor.get("text") or ""))
    if len(neighbor_body) < MIN_NEIGHBOR_BODY:
        return None
    same_paragraph = _paragraph_stem(str(neighbor.get("chunk_id") or "")) == _paragraph_stem(
        str(getattr(parent, "chunk_id", ""))
    )
    if same_paragraph:
        return "split_sibling"
    if direction > 0:
        if _starts_with_exception(neighbor_body):
            return "exception_follows"
        if _looks_unfinished(parent_body):
            return "truncated_clause"
        return None
    # Backwards: only when this chunk is itself the qualifying sentence and the
    # requirement it limits was left behind.
    if _starts_with_exception(parent_body):
        return "exception_parent"
    return None


def _is_table_evidence(evidence: FastEvidence) -> bool:
    slot = str(getattr(evidence, "slot", "") or "")
    chunk_type = str(getattr(evidence.chunk, "chunk_type", "") or "")
    return slot.startswith("table") or chunk_type.startswith("table")


def _neighbor_chunk(parent: Any, row: dict[str, Any], pooled: Any | None) -> Any:
    if pooled is not None:
        return pooled
    text = str(row.get("text") or "")
    if len(text) > NEIGHBOR_BODY_CHARS:
        text = text[:NEIGHBOR_BODY_CHARS] + "…"
    return replace(
        parent,
        chunk_id=str(row.get("chunk_id") or ""),
        page_number=row.get("page_number"),
        clause_number=str(row.get("clause_number") or ""),
        element_type=str(row.get("element_type") or "text"),
        distance=float(getattr(parent, "distance", 0.0) or 0.0),
        text=text,
        content_preview=re.sub(r"\s+", " ", text).strip()[:180],
    )


def expand_chunks_with_neighbors(
    chunks: list[Any],
    *,
    pool: list[Any] | None = None,
    chunks_dir: Path | None = None,
    limit: int | None = None,
    slot: str = "evidence",
    require_cross_page: bool = False,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Same expansion for paths that carry plain chunk lists (rule lookup)."""
    evidence = [FastEvidence(chunk, slot) for chunk in chunks]
    expanded, trace = expand_evidence_with_neighbors(
        evidence,
        pool=pool,
        chunks_dir=chunks_dir,
        limit=limit,
        require_cross_page=require_cross_page,
    )
    return [ev.chunk for ev in expanded], trace


def expand_chunks_with_parent_context(
    chunks: list[Any],
    *,
    pool: list[Any] | None = None,
    chunks_dir: Path | None = None,
    limit: int = 6,
    seed_limit: int = 10,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Attach bounded same-section context to high-ranked Advanced evidence.

    Fast expansion intentionally fires only when a paragraph visibly continues.
    Advanced has a larger context budget and also needs the heading/qualification
    immediately before a requirement and the exception immediately after it.
    This helper therefore accepts neighbours on the same page or with the same
    clause number, while still refusing document jumps and page gaps larger than
    one.  Existing pooled chunks are reused so retrieval scores and metadata are
    preserved.
    """
    if not chunks or chunks_dir is None or limit <= 0:
        return list(chunks), []

    pooled_by_id = {
        str(getattr(candidate, "chunk_id", "")): candidate
        for candidate in (pool or [])
        if getattr(candidate, "chunk_id", "")
    }
    out = list(chunks)
    taken = {
        str(getattr(candidate, "chunk_id", ""))
        for candidate in out
        if getattr(candidate, "chunk_id", "")
    }
    trace: list[dict[str, Any]] = []
    for parent in list(chunks)[: max(1, seed_limit)]:
        if len(trace) >= limit:
            break
        if str(getattr(parent, "chunk_type", "") or "").startswith("table"):
            continue
        parent_page = getattr(parent, "page_number", None)
        parent_clause = str(getattr(parent, "clause_number", "") or "").strip()
        for direction in (-1, 1):
            if len(trace) >= limit:
                break
            row = _neighbor_row(Path(chunks_dir), parent, direction)
            if row is None:
                continue
            neighbor_id = str(row.get("chunk_id") or "")
            if not neighbor_id or neighbor_id in taken:
                continue
            neighbor_page = row.get("page_number")
            neighbor_clause = str(row.get("clause_number") or "").strip()
            same_page = (
                parent_page is not None
                and neighbor_page is not None
                and str(parent_page) == str(neighbor_page)
            )
            same_clause = bool(
                parent_clause and neighbor_clause and parent_clause == neighbor_clause
            )
            reason = expansion_reason(parent, row, direction)
            if reason is None and not same_page and not same_clause:
                continue
            body = _body(str(row.get("text") or ""))
            if len(body) < MIN_NEIGHBOR_BODY:
                continue
            chunk = _neighbor_chunk(parent, row, pooled_by_id.get(neighbor_id))
            out.append(chunk)
            taken.add(neighbor_id)
            trace.append(
                {
                    "parent_chunk_id": str(getattr(parent, "chunk_id", "")),
                    "neighbor_chunk_id": neighbor_id,
                    "direction": "next" if direction > 0 else "prev",
                    "reason": reason or ("same_clause" if same_clause else "same_page"),
                    "page": neighbor_page,
                    "clause": neighbor_clause,
                }
            )
    return out, trace


def expand_evidence_with_neighbors(
    evidence: list[FastEvidence],
    *,
    pool: list[Any] | None = None,
    chunks_dir: Path | None = None,
    limit: int | None = None,
    require_cross_page: bool = False,
) -> tuple[list[FastEvidence], list[dict[str, Any]]]:
    """Insert the neighbouring chunk of each selected chunk that reads as cut off.

    ``require_cross_page`` is for callers that already merge whole pages into the
    chunk text; for them only a clause continuing onto the next page is new.

    Returns the new evidence list and one trace entry per added neighbour.
    """
    if not evidence or chunks_dir is None or not expansion_enabled():
        return list(evidence), []

    budget = max_neighbors() if limit is None else max(0, limit)
    if budget <= 0:
        return list(evidence), []

    pooled_by_id = {
        str(getattr(c, "chunk_id", "")): c for c in (pool or []) if getattr(c, "chunk_id", "")
    }
    taken = {str(getattr(ev.chunk, "chunk_id", "")) for ev in evidence}
    taken_stems = {_paragraph_stem(cid) for cid in taken}
    out: list[FastEvidence] = []
    trace: list[dict[str, Any]] = []

    for ev in evidence:
        out.append(ev)
        if len(trace) >= budget or _is_table_evidence(ev):
            continue
        for direction in (1, -1):
            row = _neighbor_row(Path(chunks_dir), ev.chunk, direction)
            if row is None:
                continue
            neighbor_id = str(row.get("chunk_id") or "")
            if not neighbor_id or neighbor_id in taken:
                continue
            if require_cross_page and row.get("page_number") == getattr(
                ev.chunk, "page_number", None
            ):
                continue
            reason = expansion_reason(ev.chunk, row, direction)
            if reason is None:
                continue
            if reason != "split_sibling" and _paragraph_stem(neighbor_id) in taken_stems:
                continue
            chunk = _neighbor_chunk(ev.chunk, row, pooled_by_id.get(neighbor_id))
            slot = f"adjacent_{'next' if direction > 0 else 'prev'}:{reason}"
            out.append(FastEvidence(chunk, slot))
            taken.add(neighbor_id)
            taken_stems.add(_paragraph_stem(neighbor_id))
            trace.append(
                {
                    "parent_chunk_id": str(getattr(ev.chunk, "chunk_id", "")),
                    "parent_slot": ev.slot,
                    "neighbor_chunk_id": neighbor_id,
                    "direction": "next" if direction > 0 else "prev",
                    "reason": reason,
                }
            )
            break

    return out, trace
