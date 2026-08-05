"""Filter thin retrieval chunks and backfill LLM context for richer answers."""
from __future__ import annotations

import re
from typing import Any

from rag_retrieval_metrics import TREND_CATEGORIES

try:
    from rule_lookup_context import is_crossref_table_chunk
except ImportError:
    def is_crossref_table_chunk(_chunk) -> bool:
        return False

MIN_LLM_CHARS = 140
RULE_LOOKUP_MIN_CHARS = 70
HEADER_ONLY_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\s+)?(outcome|summary|introduction|background|agenda|contents?)\s*\.?\s*$",
    re.I | re.M,
)
DELEGATION_LIST_RE = re.compile(
    r"\b(PARAGUAY|PHILIPPINES|BIMCO|ICS|IAPH|REPUBLIC OF KOREA)\b",
    re.I,
)
RULE_SNIPPET_RE = re.compile(
    r"doc_type=rule|\b(cg-|rp-|ru-|notice\s+no\.?)\b|\d+(?:\.\d+)+\s+\w",
    re.I,
)


def substantive_len(text: str) -> int:
    return len(re.sub(r"\s+", " ", (text or "").strip()))


def is_rule_lookup_snippet(text: str) -> bool:
    return bool(RULE_SNIPPET_RE.search(text or ""))


def min_chars_for_chunk(text: str, category: str) -> int:
    if category == "rule_lookup" and is_rule_lookup_snippet(text):
        return RULE_LOOKUP_MIN_CHARS
    return MIN_LLM_CHARS


def is_boilerplate_chunk(text: str) -> bool:
    """Participant lists, cover pages, and other low-information blocks."""
    t = (text or "").strip()
    if not t:
        return True
    if DELEGATION_LIST_RE.search(t):
        return True
    words = re.findall(r"[\w가-힣]+", t)
    if len(words) >= 12:
        upper = sum(1 for w in words if w.isupper() and len(w) > 2)
        if upper / len(words) > 0.45:
            return True
    return False


def is_thin_chunk(text: str, *, min_chars: int | None = None, category: str = "") -> bool:
    """Headers-only or near-empty chunks unsuitable for LLM summarization."""
    t = (text or "").strip()
    if not t:
        return True
    if is_boilerplate_chunk(t):
        return True
    threshold = min_chars if min_chars is not None else min_chars_for_chunk(t, category)
    if substantive_len(t) < threshold:
        return True
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if category != "rule_lookup" and len(lines) <= 2 and substantive_len(t) < 220:
        return True
    if HEADER_ONLY_RE.match(t) and substantive_len(t) < 320:
        return True
    return False


def early_page_bonus(chunk: Any, category: str, *, max_page: int = 8) -> float:
    """Boost executive-summary pages for trend/env meeting questions."""
    cat = str(category)
    if cat not in TREND_CATEGORIES and cat != "env_regulation":
        return 0.0
    p = getattr(chunk, "page_number", None)
    if p is None:
        return 0.0
    try:
        pn = int(p)
    except (TypeError, ValueError):
        return 0.0
    if pn <= max_page:
        return 0.018 * (max_page + 1 - pn)
    return 0.0


def _society_mismatch(chunk: Any, society: str) -> float:
    if not society:
        return 0.0
    src = str(getattr(chunk, "source", "") or "").upper()
    return 1.0 if src and src != society.upper() else 0.0


def _sort_key(chunk: Any, category: str, *, society: str = "") -> tuple[float, float, float, float]:
    text = getattr(chunk, "text", "")
    crossref = 1.0 if category == "rule_lookup" and is_crossref_table_chunk(chunk) else 0.0
    thin = 1.0 if is_thin_chunk(text, category=category) else 0.0
    dist = float(getattr(chunk, "distance", 1.0))
    return (_society_mismatch(chunk, society), crossref, thin, dist - early_page_bonus(chunk, category))


def llm_context_target_k(row: dict, top_k: int) -> int:
    category = str(row.get("category") or "")
    if not category:
        from retrieval_verification import effective_question_category

        category = effective_question_category(str(row.get("question", "")), row)
    bullet_min = int(row.get("answer_bullets_min", 5))
    if category == "rule_lookup":
        return min(max(top_k, 6), 10)
    if category == "trend_summary":
        return max(top_k, bullet_min, 10)
    if category in {"env_regulation", "autonomous"}:
        return max(top_k, bullet_min, 8)
    return max(top_k, bullet_min, 6)


def _filter_pool_for_society(pool: list[Any], society: str, *, hard: bool = False) -> list[Any]:
    from rag_society_filter import filter_pool_for_society

    filtered, _ = filter_pool_for_society(pool, society, hard=hard)
    return filtered


def refine_chunks_for_llm(
    selected: list[Any],
    pool: list[Any],
    *,
    row: dict,
    target_k: int | None = None,
) -> list[Any]:
    """
    Remove thin/header-only chunks; backfill from fetch pool; prefer early pages.
    """
    category = str(row.get("category", ""))
    society = str(row.get("class_society_hint") or "")
    k = target_k or llm_context_target_k(row, len(selected) or 8)
    hard_soc = bool(society and category == "rule_lookup")
    backfill_pool = _filter_pool_for_society(pool, society, hard=hard_soc) if category == "rule_lookup" else pool

    usable: list[Any] = []
    seen_ids: set[int] = set()

    def add(c: Any) -> None:
        cid = id(c)
        if cid in seen_ids:
            return
        seen_ids.add(cid)
        usable.append(c)

    def chunk_ok(c: Any) -> bool:
        if category == "rule_lookup" and is_crossref_table_chunk(c):
            return False
        return not is_thin_chunk(getattr(c, "text", ""), category=category)

    for c in selected:
        if chunk_ok(c):
            add(c)

    for c in sorted(backfill_pool, key=lambda x: _sort_key(x, category, society=society)):
        if len(usable) >= k:
            break
        if chunk_ok(c):
            add(c)

    min_fill = min(3, k)
    if len(usable) < min_fill:
        for c in sorted(backfill_pool, key=lambda x: float(getattr(x, "distance", 1.0))):
            if len(usable) >= k:
                break
            min_len = min_chars_for_chunk(getattr(c, "text", ""), category)
            if substantive_len(getattr(c, "text", "")) >= min_len:
                add(c)

    usable.sort(key=lambda x: _sort_key(x, category, society=society))
    return usable[:k]
