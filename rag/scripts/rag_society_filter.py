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


def society_hard_filter_enabled(row: dict | None) -> bool:
    if not row:
        return False
    if row.get("_hard_society_filter"):
        return True
    if row.get("_rule_guidance_lookup") and row.get("class_society_hint"):
        return True
    return str(row.get("category") or "") == "rule_lookup" and bool(row.get("class_society_hint"))
