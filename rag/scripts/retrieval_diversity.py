"""Diversity reranking for retrieval results (post-fetch)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rag_retrieval_metrics import TREND_CATEGORIES

from retrieval_chunk_quality import early_page_bonus

try:
    from rule_lookup_context import is_crossref_table_chunk
except ImportError:
    def is_crossref_table_chunk(_chunk) -> bool:
        return False

TOKEN_RE = re.compile(r"[\w가-힣]+", re.UNICODE)


@dataclass
class DiversityConfig:
    max_chunks_per_doc: int = 3
    max_chunks_per_page: int = 1
    max_docs: int = 0
    mmr_lambda: float = 0.72
    doc_repeat_penalty: float = 0.06
    page_repeat_penalty: float = 0.04
    mmr_diversity_weight: float = 0.28
    enable_mmr: bool = True


def _page_key(chunk: Any) -> tuple[str, int]:
    return (str(chunk.doc_id), int(chunk.page_number or 0))


def _token_set(text: str) -> set[str]:
    return {t for t in TOKEN_RE.findall((text or "").lower()) if len(t) > 1}


def lexical_similarity(text_a: str, text_b: str) -> float:
    a, b = _token_set(text_a), _token_set(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dedupe_doc_page(chunks: list[Any]) -> list[Any]:
    """Keep the lowest-distance chunk per (doc_id, page)."""
    best: dict[tuple[str, int], Any] = {}
    for chunk in chunks:
        key = _page_key(chunk)
        if key not in best or float(chunk.distance) < float(best[key].distance):
            best[key] = chunk
    out = sorted(best.values(), key=lambda c: float(c.distance))
    return out


def _should_use_mmr(category: str, config: DiversityConfig, question_mode: str = "") -> bool:
    if question_mode == "broad":
        return config.enable_mmr
    return config.enable_mmr and str(category) in TREND_CATEGORIES


def diversity_rerank(
    chunks: list[Any],
    *,
    top_k: int,
    category: str = "",
    question_mode: str = "",
    config: DiversityConfig | None = None,
    class_society_hint: str = "",
) -> list[Any]:
    """
    Greedy selection combining dense distance, MMR (trend/summary), and doc/page caps.

    Same document may appear up to max_chunks_per_doc times if pages differ.
    """
    cfg = config or DiversityConfig()
    pool = dedupe_doc_page(chunks)
    if not pool:
        return []

    use_mmr = _should_use_mmr(category, cfg, question_mode)
    selected: list[Any] = []
    remaining = list(pool)
    doc_counts: dict[str, int] = {}
    page_counts: dict[tuple[str, int], int] = {}
    selected_docs: set[str] = set()

    while len(selected) < top_k and remaining:
        best_idx: int | None = None
        best_score = float("-inf")

        for i, cand in enumerate(remaining):
            pk = _page_key(cand)
            dk = str(cand.doc_id)
            if doc_counts.get(dk, 0) >= cfg.max_chunks_per_doc:
                continue
            if page_counts.get(pk, 0) >= cfg.max_chunks_per_page:
                continue
            if (
                cfg.max_docs > 0
                and dk not in selected_docs
                and len(selected_docs) >= cfg.max_docs
            ):
                continue

            relevance = -float(cand.distance) + early_page_bonus(cand, category)
            bm25_sc = getattr(cand, "bm25_score", None)
            if bm25_sc is not None and category == "rule_lookup":
                relevance += min(float(bm25_sc) * 0.06, 0.4)
            score = relevance
            score -= cfg.doc_repeat_penalty * doc_counts.get(dk, 0)
            score -= cfg.page_repeat_penalty * page_counts.get(pk, 0)

            society = str(class_society_hint or "").upper()
            if society and category == "rule_lookup":
                src = str(getattr(cand, "source", "") or "").upper()
                if src and src != society:
                    score -= 0.24
            if category == "rule_lookup" and is_crossref_table_chunk(cand):
                score -= 0.18

            if use_mmr and selected:
                max_sim = max(lexical_similarity(cand.text, s.text) for s in selected)
                score = (
                    cfg.mmr_lambda * relevance
                    - cfg.mmr_diversity_weight * max_sim
                    - cfg.doc_repeat_penalty * doc_counts.get(dk, 0)
                    - cfg.page_repeat_penalty * page_counts.get(pk, 0)
                )

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            break

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        doc_counts[str(chosen.doc_id)] = doc_counts.get(str(chosen.doc_id), 0) + 1
        page_counts[_page_key(chosen)] = page_counts.get(_page_key(chosen), 0) + 1
        selected_docs.add(str(chosen.doc_id))

    return selected


def unique_page_count(chunks: list[Any], k: int) -> int:
    return len({_page_key(c) for c in chunks[:k]})
