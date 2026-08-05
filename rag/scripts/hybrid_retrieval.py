"""Dense + BM25 parallel search with RRF fusion for Rule/Guidance lookup."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bm25_index import (
    BM25Index,
    bm25_alt_fuel_terms_present,
    enrich_query_for_bm25,
    load_or_build_bm25,
    tokenize_for_bm25,
)
from retrieval_query_analysis import analyze_query, detect_class_society_hint
from retrieval_search import (
    _metadata_boosts,
    extract_clause_hints,
    query_with_hybrid_ranking,
)
from imo_doc_registry import priority_doc_ids_for_signals

RRF_K = 60


@dataclass
class FusedHit:
    chunk_id: str
    dense_score: float | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    metadata_boost: float = 0.0
    source_priority_score: float = 0.0
    final_score: float = 0.0
    distance: float = 1.0
    meta: dict = field(default_factory=dict)
    document: str = ""
    is_catalog_table: bool = False
    catalog_candidates: list[str] = field(default_factory=list)


def rrf_score(rank: int | None, k: int = RRF_K) -> float:
    if rank is None or rank < 1:
        return 0.0
    return 1.0 / (k + rank)


def is_catalog_table(meta: dict, document: str, caption: str = "") -> bool:
    blob = f"{caption} {document} {(meta or {}).get('caption', '')}".lower()
    if "document code" in blob and "title" in blob:
        return True
    codes = re.findall(
        r"\b(DNV|LR|ABS|KR)[- ]?(?:CG|RP|RU|CP|NV)[- ]?\d[\w.-]*",
        document or "",
        re.I,
    )
    return len(codes) >= 3


def extract_catalog_candidates(text: str) -> list[str]:
    pat = re.compile(
        r"\b((?:DNV|LR|ABS|KR)[- ]?(?:CG|RP|RU|CP|NV)[- ]?\d[\w.-]*)",
        re.I,
    )
    return list(dict.fromkeys(m.group(1).strip() for m in pat.finditer(text or "")))


def source_priority_score(meta: dict, society: str) -> float:
    if not society:
        return 0.0
    src = str(meta.get("source") or "").upper()
    fn = str(meta.get("file_name") or "").lower()
    soc = society.upper()
    if src == soc:
        return 0.25
    if soc.lower() in fn:
        return 0.18
    if src in {"ABS", "LR", "KR", "DNV"} and src != soc:
        return -0.20
    return 0.0


def fuse_dense_bm25(
    dense_ids: list[str],
    dense_dist: dict[str, float],
    dense_meta: dict[str, dict],
    dense_doc: dict[str, str],
    bm25_hits: list,
    *,
    query: str,
    signals,
    priority_doc_ids: set[str],
    society: str = "",
) -> list[FusedHit]:
    bm25_by_id = {h.chunk_id: h for h in bm25_hits}
    dense_rank_map = {cid: i + 1 for i, cid in enumerate(dense_ids)}
    bm25_rank_map = {h.chunk_id: h.rank for h in bm25_hits}
    all_ids = list(dict.fromkeys(dense_ids + [h.chunk_id for h in bm25_hits]))

    fused: list[FusedHit] = []
    clause_hints = extract_clause_hints(query)
    for cid in all_ids:
        meta = dense_meta.get(cid) or (bm25_by_id[cid].meta if cid in bm25_by_id else {})
        doc = dense_doc.get(cid) or (bm25_by_id[cid].document if cid in bm25_by_id else "")
        dr = dense_rank_map.get(cid)
        br = bm25_rank_map.get(cid)
        d_dist = dense_dist.get(cid)
        dense_sc = (1.0 - float(d_dist)) if d_dist is not None else None
        bm25_sc = float(bm25_by_id[cid].score) if cid in bm25_by_id else None
        rrf = rrf_score(dr) + rrf_score(br)
        m_boost, m_pen = _metadata_boosts(
            meta=meta,
            document=doc,
            signals=signals,
            priority_doc_ids=priority_doc_ids,
            query=query,
        )
        meta_boost = m_boost - m_pen
        src_pri = source_priority_score(meta, society)
        catalog = is_catalog_table(meta, doc, str(meta.get("caption") or ""))
        candidates = extract_catalog_candidates(doc) if catalog else []
        final = rrf + meta_boost * 0.05 + src_pri
        fused.append(
            FusedHit(
                chunk_id=cid,
                dense_score=round(dense_sc, 4) if dense_sc is not None else None,
                bm25_score=round(bm25_sc, 4) if bm25_sc is not None else None,
                dense_rank=dr,
                bm25_rank=br,
                rrf_score=round(rrf, 6),
                metadata_boost=round(meta_boost, 4),
                source_priority_score=round(src_pri, 4),
                final_score=round(final, 6),
                distance=1.0 - final,
                meta=meta,
                document=doc,
                is_catalog_table=catalog,
                catalog_candidates=candidates,
            )
        )
    fused.sort(key=lambda h: -h.final_score)
    return fused


def query_dense_pool(
    collection,
    query: str,
    query_vector: list[float],
    *,
    fetch_k: int,
    source: str | None,
    doc_id: str | None,
    timing=None,
    hard_source_filter: bool = False,
) -> tuple[list[str], dict[str, float], dict[str, dict], dict[str, str], list[str]]:
    """Return dense pool; warning_flags may include source_filter_fallback."""
    warnings: list[str] = []
    n_fetch = fetch_k

    def _run(src: str | None) -> dict:
        return query_with_hybrid_ranking(
            collection,
            query,
            query_vector,
            top_k=n_fetch,
            fetch_k=n_fetch,
            source=src,
            doc_id=doc_id,
            timing=timing,
        )

    raw = _run(source)
    ids = raw["ids"][0] if raw.get("ids") else []
    if source and not hard_source_filter and len(ids) < min(5, n_fetch // 3):
        warnings.append("source_filter_fallback")
        raw = _run(None)
        ids = raw["ids"][0] if raw.get("ids") else []
    elif source and hard_source_filter and len(ids) < min(5, n_fetch // 3):
        warnings.append("society_filter_strict_no_fallback")
    dist = {cid: float(d) for cid, d in zip(raw["ids"][0], raw["distances"][0])}
    meta = {cid: m for cid, m in zip(raw["ids"][0], raw["metadatas"][0])}
    doc = {cid: d for cid, d in zip(raw["ids"][0], raw["documents"][0])}
    return list(raw["ids"][0]), dist, meta, doc, warnings


def hybrid_rule_lookup_search(
    collection,
    bm25_index: BM25Index | None,
    query: str,
    query_vector: list[float],
    *,
    fetch_k: int = 56,
    top_k: int = 30,
    society: str | None = None,
    doc_id: str | None = None,
    timing=None,
    hard_source_filter: bool = True,
) -> tuple[list[FusedHit], dict[str, Any]]:
    signals = analyze_query(query)
    society = society or detect_class_society_hint(query)
    priority_ids = priority_doc_ids_for_signals(signals)
    priority_set = set(priority_ids)
    warning_flags: list[str] = []

    dense_ids, dense_dist, dense_meta, dense_doc, w = query_dense_pool(
        collection,
        query,
        query_vector,
        fetch_k=fetch_k,
        source=society,
        doc_id=doc_id,
        timing=timing,
        hard_source_filter=hard_source_filter,
    )
    warning_flags.extend(w)

    bm25_hits = []
    bm25_query = query
    bm25_tokens: list[str] = []
    bm25_term_checks: dict[str, bool] = {}
    if bm25_index is not None:
        bm25_query, bm25_tokens = enrich_query_for_bm25(query)
        bm25_term_checks = bm25_alt_fuel_terms_present(bm25_tokens)
        bm25_hits, w2 = bm25_index.search_with_fallback(
            bm25_query, top_k=fetch_k, source=society, hard_source_filter=hard_source_filter
        )
        warning_flags.extend(w2)
    else:
        warning_flags.append("bm25_no_hit")

    if bm25_index is not None and not bm25_hits:
        warning_flags.append("bm25_no_hit")

    fused = fuse_dense_bm25(
        dense_ids,
        dense_dist,
        dense_meta,
        dense_doc,
        bm25_hits,
        query=query,
        signals=signals,
        priority_doc_ids=priority_set,
        society=society or "",
    )[:top_k]

    bm25_top20 = [
        {
            "chunk_id": h.chunk_id,
            "rank": h.rank,
            "score": round(float(h.score), 4),
            "file_name": h.meta.get("file_name"),
            "page": h.meta.get("page_number"),
        }
        for h in bm25_hits[:20]
    ]
    fused_top10 = fused[:10]
    has_bm25_in_fused = any(h.bm25_score is not None for h in fused_top10)
    if bm25_hits and not has_bm25_in_fused:
        warning_flags.append("bm25_no_relevant_hit")

    log_payload = {
        "bm25_query": bm25_query,
        "bm25_query_tokens": bm25_tokens,
        "bm25_term_checks": bm25_term_checks,
        "dense_top20": [
            {
                "chunk_id": cid,
                "rank": i + 1,
                "distance": dense_dist.get(cid),
                "file_name": (dense_meta.get(cid) or {}).get("file_name"),
                "page": (dense_meta.get(cid) or {}).get("page_number"),
            }
            for i, cid in enumerate(dense_ids[:20])
        ],
        "dense_results": [
            {
                "chunk_id": cid,
                "rank": i + 1,
                "distance": dense_dist.get(cid),
                "file_name": (dense_meta.get(cid) or {}).get("file_name"),
                "page": (dense_meta.get(cid) or {}).get("page_number"),
            }
            for i, cid in enumerate(dense_ids[:fetch_k])
        ],
        "bm25_top20": bm25_top20,
        "bm25_results": bm25_top20,
        "fused_top10": [
            {
                "chunk_id": h.chunk_id,
                "rrf_score": h.rrf_score,
                "dense_score": h.dense_score,
                "bm25_score": h.bm25_score,
                "dense_rank": h.dense_rank,
                "bm25_rank": h.bm25_rank,
                "file_name": h.meta.get("file_name"),
                "page": h.meta.get("page_number"),
                "is_catalog_table": h.is_catalog_table,
            }
            for h in fused_top10
        ],
        "fused_results": [
            {
                "chunk_id": h.chunk_id,
                "rrf_score": h.rrf_score,
                "dense_score": h.dense_score,
                "bm25_score": h.bm25_score,
                "metadata_boost": h.metadata_boost,
                "source_priority_score": h.source_priority_score,
                "file_name": h.meta.get("file_name"),
                "is_catalog_table": h.is_catalog_table,
            }
            for h in fused
        ],
        "warning_flags": list(dict.fromkeys(warning_flags)),
    }
    return fused, log_payload


def get_bm25_index(
    collection,
    unified_id: str,
    index_dir: Path,
    *,
    fingerprint: str = "",
) -> BM25Index | None:
    return load_or_build_bm25(
        collection,
        unified_id=unified_id,
        index_dir=index_dir,
        fingerprint=fingerprint,
    )
