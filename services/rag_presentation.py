"""Final user-facing normalization for document RAG answers."""
from __future__ import annotations

import re
from typing import Any


_CITATION_RE = re.compile(r"\[(\d+)\]")


def _citation_number(value: object) -> int | None:
    match = _CITATION_RE.search(str(value or ""))
    return int(match.group(1)) if match else None


def compact_citations(
    answer: str,
    evidence_table: list[dict[str, Any]] | None,
) -> tuple[str, list[dict[str, Any]], dict[int, int]]:
    """Renumber cited evidence by first use so UI footnotes never skip numbers.

    Retrieval ranks are internal identifiers.  Once an answer uses only a
    subset of those ranks, the displayed contract must be independent of the
    unused candidates: [2], [4], [7] becomes [1], [2], [3], and the evidence
    rows are reordered to the same first-use order.
    """
    ordered_old: list[int] = []
    for match in _CITATION_RE.finditer(answer or ""):
        value = int(match.group(1))
        if value not in ordered_old:
            ordered_old.append(value)

    mapping = {old: new for new, old in enumerate(ordered_old, start=1)}
    if not mapping:
        return answer, [], {}

    normalized_answer = _CITATION_RE.sub(
        lambda match: f"[{mapping[int(match.group(1))]}]",
        answer,
    )

    row_by_old: dict[int, dict[str, Any]] = {}
    for row in evidence_table or []:
        old = _citation_number(row.get("citation_id"))
        if old is not None and old not in row_by_old:
            row_by_old[old] = row

    normalized_rows: list[dict[str, Any]] = []
    for old in ordered_old:
        row = row_by_old.get(old)
        if row is None:
            continue
        normalized = dict(row)
        normalized["citation_id"] = f"[{mapping[old]}]"
        normalized_rows.append(normalized)

    return normalized_answer, normalized_rows, mapping
