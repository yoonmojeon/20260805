"""Extended retrieval metrics for pilot RAG validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag_eval_lib import keyword_hits

TREND_CATEGORIES = frozenset({"trend_summary"})


def resolve_gold_pages(row: dict) -> list[int]:
    """Return gold page list; trend/summary may use gold_pages, others fall back to gold_page."""
    category = str(row.get("category", ""))
    pages = row.get("gold_pages")
    if pages and category in TREND_CATEGORIES:
        return [int(p) for p in pages]
    if pages and not row.get("gold_page"):
        return [int(p) for p in pages]
    gp = row.get("gold_page")
    if gp is not None:
        return [int(gp)]
    return []


def matched_keyword_list(text: str, keywords: list[str]) -> list[str]:
    lower = (text or "").lower()
    return [kw for kw in keywords if kw.lower() in lower]


def matched_topic_list(text: str, topics: list[str]) -> list[str]:
    lower = (text or "").lower()
    return [t for t in topics if t.lower() in lower]


def content_preview(text: str, limit: int = 500) -> str:
    flat = " ".join((text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3] + "..."


def duplicate_doc_ratio(chunks: list[Any], k: int) -> float:
    subset = chunks[:k]
    if not subset:
        return 0.0
    unique = len({c.doc_id for c in subset})
    return (len(subset) - unique) / len(subset)


def duplicate_page_ratio(chunks: list[Any], k: int) -> float:
    subset = chunks[:k]
    if not subset:
        return 0.0
    unique = len({(c.doc_id, c.page_number) for c in subset})
    return (len(subset) - unique) / len(subset)


def source_hit_at_k(chunks: list[Any], allowed_sources: set[str], k: int) -> bool:
    if not allowed_sources:
        return bool(chunks[:k])
    return any(c.source.upper() in allowed_sources for c in chunks[:k])


def source_hits_count(chunks: list[Any], allowed_sources: set[str], k: int) -> int:
    if not allowed_sources:
        return len(chunks[:k])
    return sum(1 for c in chunks[:k] if c.source.upper() in allowed_sources)


def resolve_gold_doc_candidates(row: dict) -> list[str]:
    cands = row.get("gold_doc_candidates")
    if cands:
        return [str(x) for x in cands if str(x).strip()]
    gold = str(row.get("gold_doc_id") or "").strip()
    return [gold] if gold else []


def gold_doc_candidates_hit_at_k(chunks: list[Any], candidates: list[str], k: int) -> bool:
    if not candidates:
        return False
    cand_set = set(candidates)
    return any(c.doc_id in cand_set for c in chunks[:k])


def gold_doc_candidates_rank(chunks: list[Any], candidates: list[str], k: int) -> int | None:
    if not candidates:
        return None
    cand_set = set(candidates)
    for rank, c in enumerate(chunks[:k], start=1):
        if c.doc_id in cand_set:
            return rank
    return None


def page_recall_at_k(
    chunks: list[Any],
    candidates: list[str],
    gold_pages: list[int],
    k: int,
) -> bool:
    """True if any gold page from any candidate doc appears in top-k."""
    if not candidates or not gold_pages:
        return False
    cand_set = set(candidates)
    page_set = set(gold_pages)
    return any(
        c.doc_id in cand_set
        and c.page_number is not None
        and int(c.page_number) in page_set
        for c in chunks[:k]
    )


def unique_doc_count_in_k(chunks: list[Any], k: int) -> int:
    return len({c.doc_id for c in chunks[:k] if getattr(c, "doc_id", "")})


def gold_doc_hit_at_k(chunks: list[Any], gold_doc: str, k: int) -> bool:
    if not gold_doc:
        return False
    return any(c.doc_id == gold_doc for c in chunks[:k])


def gold_doc_rank(chunks: list[Any], gold_doc: str, k: int) -> int | None:
    if not gold_doc:
        return None
    for rank, c in enumerate(chunks[:k], start=1):
        if c.doc_id == gold_doc:
            return rank
    return None


def gold_page_set_hit_at_k(
    chunks: list[Any], gold_doc: str, gold_pages: list[int], k: int
) -> bool:
    if not gold_doc or not gold_pages:
        return False
    page_set = set(gold_pages)
    return any(
        c.doc_id == gold_doc and c.page_number is not None and int(c.page_number) in page_set
        for c in chunks[:k]
    )


def keyword_coverage(chunks: list[Any], keywords: list[str], k: int) -> float:
    if not keywords:
        return 1.0
    found: set[str] = set()
    for c in chunks[:k]:
        for kw in matched_keyword_list(c.text, keywords):
            found.add(kw.lower())
    return len(found) / len(keywords)


def topic_hit_at_k(chunks: list[Any], topics: list[str], k: int) -> bool:
    if not topics:
        return True
    for c in chunks[:k]:
        if matched_topic_list(c.text, topics):
            return True
    return False


def boundary_error_rate(
    chunks: list[Any], allowed_sources: set[str], k: int
) -> float:
    subset = chunks[:k]
    if not subset or not allowed_sources:
        return 0.0
    errors = sum(1 for c in subset if c.source.upper() not in allowed_sources)
    return errors / len(subset)


def enrich_chunk_annotations(
    chunk: Any,
    *,
    keywords: list[str],
    topics: list[str],
    preview_limit: int = 500,
) -> Any:
    chunk.matched_keywords = matched_keyword_list(chunk.text, keywords)
    chunk.matched_topics = matched_topic_list(chunk.text, topics)
    chunk.content_preview = content_preview(chunk.text, preview_limit)
    return chunk


@dataclass
class RetrievalMetrics:
    question_id: str
    top_k: int
    eval_k: int = 5
    source_hit_at_5: bool = False
    gold_doc_hit_at_5: bool = False
    gold_page_set_hit_at_5: bool = False
    topic_hit_at_k: bool = False
    keyword_coverage: float = 0.0
    boundary_error_rate: float = 0.0
    duplicate_doc_ratio: float = 0.0
    duplicate_page_ratio: float = 0.0
    gold_doc_rank: int | None = None
    gold_pages: list[int] = field(default_factory=list)
    gold_doc_candidates: list[str] = field(default_factory=list)
    doc_recall_at_5: bool = False
    page_recall_at_5: bool = False
    eval_mode: str = "open"
    unique_doc_count: int = 0
    best_keyword_hits: int = 0
    keyword_total: int = 0
    source_hits_in_top_k: int = 0
    failure_reasons: list[str] = field(default_factory=list)
    ambiguous_reasons: list[str] = field(default_factory=list)
    is_failure: bool = False
    is_ambiguous: bool = False

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "top_k": self.top_k,
            "eval_k": self.eval_k,
            "source_hit_at_5": self.source_hit_at_5,
            "gold_doc_hit_at_5": self.gold_doc_hit_at_5,
            "gold_page_set_hit_at_5": self.gold_page_set_hit_at_5,
            "topic_hit_at_k": self.topic_hit_at_k,
            "keyword_coverage": round(self.keyword_coverage, 4),
            "boundary_error_rate": round(self.boundary_error_rate, 4),
            "duplicate_doc_ratio": round(self.duplicate_doc_ratio, 4),
            "duplicate_page_ratio": round(self.duplicate_page_ratio, 4),
            "gold_doc_rank": self.gold_doc_rank,
            "gold_pages": self.gold_pages,
            "gold_doc_candidates": self.gold_doc_candidates,
            "doc_recall_at_5": self.doc_recall_at_5,
            "page_recall_at_5": self.page_recall_at_5,
            "eval_mode": self.eval_mode,
            "unique_doc_count": self.unique_doc_count,
            "best_keyword_hits": self.best_keyword_hits,
            "keyword_total": self.keyword_total,
            "source_hits_in_top_k": self.source_hits_in_top_k,
            "failure_reasons": self.failure_reasons,
            "ambiguous_reasons": self.ambiguous_reasons,
            "is_failure": self.is_failure,
            "is_ambiguous": self.is_ambiguous,
        }


def chunk_to_report_dict(chunk: Any, rank: int) -> dict:
    return {
        "rank": rank,
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "source": chunk.source,
        "file_name": chunk.file_name,
        "page_number": chunk.page_number,
        "clause_number": chunk.clause_number,
        "distance": round(chunk.distance, 4),
        "matched_keywords": chunk.matched_keywords,
        "matched_topics": chunk.matched_topics,
        "content_preview": chunk.content_preview,
    }


def compute_retrieval_metrics(
    row: dict,
    retrieved: list[Any],
    *,
    top_k: int,
    eval_k: int = 5,
    preview_limit: int = 500,
    eval_mode: str = "open",
) -> RetrievalMetrics:
    keywords = list(row.get("expected_keywords") or [])
    topics = list(row.get("expected_topics") or [])
    sources = {s.upper() for s in row.get("retrieval_sources") or []}
    gold_doc = str(row.get("gold_doc_id") or "")
    gold_candidates = resolve_gold_doc_candidates(row)
    gold_pages = resolve_gold_pages(row)

    for chunk in retrieved:
        enrich_chunk_annotations(chunk, keywords=keywords, topics=topics, preview_limit=preview_limit)

    best_kw = 0
    for c in retrieved[:top_k]:
        kh, _ = keyword_hits(c.text, keywords)
        best_kw = max(best_kw, kh)

    doc_recall = gold_doc_candidates_hit_at_k(retrieved, gold_candidates, eval_k)
    page_rec = page_recall_at_k(retrieved, gold_candidates, gold_pages, eval_k)
    cand_rank = gold_doc_candidates_rank(retrieved, gold_candidates, top_k)

    m = RetrievalMetrics(
        question_id=str(row.get("question_id", "unknown")),
        top_k=top_k,
        eval_k=eval_k,
        source_hit_at_5=source_hit_at_k(retrieved, sources, eval_k),
        gold_doc_hit_at_5=gold_doc_hit_at_k(retrieved, gold_doc, eval_k),
        gold_page_set_hit_at_5=gold_page_set_hit_at_k(retrieved, gold_doc, gold_pages, eval_k),
        topic_hit_at_k=topic_hit_at_k(retrieved, topics, top_k),
        keyword_coverage=keyword_coverage(retrieved, keywords, top_k),
        boundary_error_rate=boundary_error_rate(retrieved, sources, top_k),
        duplicate_doc_ratio=duplicate_doc_ratio(retrieved, top_k),
        duplicate_page_ratio=duplicate_page_ratio(retrieved, top_k),
        gold_doc_rank=gold_doc_rank(retrieved, gold_doc, top_k),
        gold_pages=gold_pages,
        gold_doc_candidates=gold_candidates,
        doc_recall_at_5=doc_recall,
        page_recall_at_5=page_rec,
        eval_mode=eval_mode,
        unique_doc_count=unique_doc_count_in_k(retrieved, top_k),
        best_keyword_hits=best_kw,
        keyword_total=len(keywords),
        source_hits_in_top_k=source_hits_count(retrieved, sources, top_k),
    )
    if cand_rank is not None and m.gold_doc_rank is None:
        m.gold_doc_rank = cand_rank

    if not retrieved:
        m.failure_reasons.append("empty_retrieval")
    if eval_mode == "open" and gold_candidates and not doc_recall:
        m.failure_reasons.append("doc_recall_miss_at_5")
    if eval_mode == "constrained" and gold_doc and not m.gold_doc_hit_at_5:
        m.failure_reasons.append("gold_doc_miss_at_5")
    if sources and not m.source_hit_at_5:
        m.failure_reasons.append("source_miss_at_5")
    if m.boundary_error_rate >= 1.0:
        m.failure_reasons.append("all_chunks_out_of_boundary")

    m.is_failure = bool(m.failure_reasons)

    if gold_candidates and gold_pages and not page_rec:
        m.ambiguous_reasons.append("page_recall_miss_at_5")
    if m.gold_doc_hit_at_5 and gold_pages and not m.gold_page_set_hit_at_5:
        m.ambiguous_reasons.append("gold_page_set_miss_at_5")
    if m.duplicate_doc_ratio >= 0.5:
        m.ambiguous_reasons.append("high_duplicate_doc_ratio")
    if m.duplicate_page_ratio >= 0.5:
        m.ambiguous_reasons.append("high_duplicate_page_ratio")
    if topics and not m.topic_hit_at_k:
        m.ambiguous_reasons.append("topic_miss_at_k")
    if keywords and m.keyword_coverage < 0.5:
        m.ambiguous_reasons.append("low_keyword_coverage")
    if m.gold_doc_hit_at_5 and m.duplicate_doc_ratio >= 0.375:
        m.ambiguous_reasons.append("low_evidence_diversity")

    m.is_ambiguous = bool(m.ambiguous_reasons) and not m.is_failure
    return m
