"""Select one citation-stable evidence set for planning, generation and UI.

Evidence completion records the best chunk ids for every question requirement.
Historically those ids were used to report slot coverage, but answer generation
trimmed a different list and the Evidence Table was then built from yet another
list.  This module makes the ordered slot hits the single source of truth.

The implementation is intentionally document- and question-independent.  It
does not contain known answers, file names, page numbers or pilot-question
strings.
"""
from __future__ import annotations

from typing import Any, Iterable


def _identity(chunk: Any) -> str:
    return str(getattr(chunk, "chunk_id", "") or "") or (
        f"{getattr(chunk, 'doc_id', '')}:"
        f"{getattr(chunk, 'file_name', '')}:"
        f"{getattr(chunk, 'page_number', '')}"
    )


def _unique(chunks: Iterable[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()
    for chunk in chunks:
        identity = _identity(chunk)
        if not identity or identity in seen:
            continue
        seen.add(identity)
        output.append(chunk)
    return output


def select_planned_evidence(
    row: dict,
    retrieved: Iterable[Any],
    pool: Iterable[Any] | None = None,
    *,
    max_chunks: int = 12,
) -> tuple[list[Any], dict[str, Any]]:
    """Return a diverse, slot-first evidence set and an auditable selection log.

    Pass one guarantees one hit per planned slot.  Pass two adds each slot's
    remaining hits.  Only then are ordinary ranked candidates appended.  The
    returned order is also the citation numbering order used by the answer and
    Evidence Table.
    """
    retrieved_list = list(retrieved or [])
    pool_list = list(pool or [])
    candidates = _unique([*retrieved_list, *pool_list])
    by_id = {
        str(getattr(chunk, "chunk_id", "") or ""): chunk
        for chunk in candidates
        if str(getattr(chunk, "chunk_id", "") or "")
    }
    completion = row.get("_evidence_completion") or {}
    slot_hits = completion.get("slot_hits") or {}

    selected: list[Any] = []
    seen: set[str] = set()
    selected_by_slot: dict[str, list[str]] = {}

    def add(chunk: Any, slot_name: str | None = None) -> bool:
        identity = _identity(chunk)
        if not identity or identity in seen or len(selected) >= max_chunks:
            return False
        seen.add(identity)
        selected.append(chunk)
        if slot_name:
            selected_by_slot.setdefault(str(slot_name), []).append(identity)
        return True

    # Guarantee breadth before depth.
    for slot_name, chunk_ids in slot_hits.items():
        for chunk_id in list(chunk_ids or [])[:1]:
            chunk = by_id.get(str(chunk_id))
            if chunk is not None:
                add(chunk, str(slot_name))

    # Retain additional propositions for slots that explicitly requested them.
    for slot_name, chunk_ids in slot_hits.items():
        for chunk_id in list(chunk_ids or [])[1:]:
            chunk = by_id.get(str(chunk_id))
            if chunk is not None:
                add(chunk, str(slot_name))

    for chunk in candidates:
        if len(selected) >= max_chunks:
            break
        add(chunk)

    missing_slots = [
        str(slot_name)
        for slot_name, chunk_ids in slot_hits.items()
        if chunk_ids and not selected_by_slot.get(str(slot_name))
    ]
    return selected, {
        "strategy": "slot_first_single_source_of_truth",
        "max_chunks": max_chunks,
        "selected_chunk_ids": [_identity(chunk) for chunk in selected],
        "selected_by_slot": selected_by_slot,
        "missing_slots": missing_slots,
        "planned_slot_count": len(slot_hits),
    }
