"""Hard society (class) filtering for Rule/Guidance retrieval."""
from __future__ import annotations

from typing import Any


def filter_pool_for_society(
    pool: list[Any],
    society: str,
    *,
    hard: bool = False,
) -> tuple[list[Any], bool]:
    """
    Filter chunks to a single class society (DNV/LR/ABS/KR).

    hard=False (legacy): fall back to full pool if fewer than 3 matches.
    hard=True (rule guidance): never fall back; empty pool means insufficient LR/DNV evidence.
    """
    if not society:
        return pool, False
    soc = society.upper()
    matched = [
        c
        for c in pool
        if str(getattr(c, "source", "") or (getattr(c, "meta", {}) or {}).get("source", "")).upper() == soc
    ]
    if hard:
        return matched, bool(matched)
    return (matched if len(matched) >= 3 else pool), bool(matched)


def filter_pool_for_source_constraints(
    pool: list[Any],
    *,
    allowed_sources: list[str] | tuple[str, ...] | None = None,
    excluded_sources: list[str] | tuple[str, ...] | None = None,
) -> list[Any]:
    """Keep explicit source include/exclude constraints through generation."""
    allowed = {str(source).upper() for source in (allowed_sources or []) if source}
    excluded = {str(source).upper() for source in (excluded_sources or []) if source}
    out: list[Any] = []
    for chunk in pool:
        meta = getattr(chunk, "meta", {}) or {}
        source = str(getattr(chunk, "source", "") or meta.get("source", "")).upper()
        if source in excluded:
            continue
        if allowed and source not in allowed:
            continue
        out.append(chunk)
    return out


def society_hard_filter_enabled(row: dict | None) -> bool:
    if not row:
        return False
    if row.get("_hard_society_filter"):
        return True
    if row.get("_rule_guidance_lookup") and row.get("class_society_hint"):
        return True
    return str(row.get("category") or "") == "rule_lookup" and bool(row.get("class_society_hint"))
