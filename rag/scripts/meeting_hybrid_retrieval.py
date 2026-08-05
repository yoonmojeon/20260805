"""Hybrid dense+BM25+RRF retrieval for meeting/regulation categories."""
from __future__ import annotations

from typing import Any

from hybrid_retrieval import fuse_dense_bm25, get_bm25_index, query_dense_pool, rrf_score
from imo_doc_registry import priority_doc_ids_for_signals
from meeting_category_profile import MeetingRetrievalProfile
from meeting_query_expansion import enrich_query_for_meeting
from meeting_topic_cluster import _topic_id_for_text, outcome_topic_id
from meeting_topic_scoring import intent_chunk_adjustment, is_excluded_chunk
from retrieval_query_analysis import analyze_query
from source_tier_lib import classify_source_tier, impact_boost, outcome_boost, tier_boost


def _apply_meeting_boosts(
    fused,
    *,
    profile: MeetingRetrievalProfile,
) -> list:
    out = []
    for hit in fused:
        meta = hit.meta or {}
        doc = hit.document or ""
        tier = classify_source_tier(type("_C", (), {"file_name": meta.get("file_name"), "text": doc, "caption": meta.get("caption")})())
        text = doc
        boost = 0.0
        if profile.use_source_tier:
            boost += tier_boost(tier)
        boost += outcome_boost(text)
        if profile.top_level_category == "env_regulation_response":
            boost += impact_boost(text)
        boost += intent_chunk_adjustment(text, internal_intent=profile.internal_intent)
        hit.final_score = round(hit.final_score + boost * profile.dense_weight, 6)
        hit.distance = 1.0 - hit.final_score
        hit.metadata_boost = round(hit.metadata_boost + boost, 4)
        setattr(hit, "_source_tier", tier)
        out.append(hit)
    out.sort(key=lambda h: -h.final_score)
    return out


def _single_hybrid_search(
    collection,
    bm25_index,
    query: str,
    query_vector: list[float],
    *,
    profile: MeetingRetrievalProfile,
    fetch_k: int,
    top_k: int,
    society: str | None,
    doc_id: str | None,
    timing=None,
) -> tuple[list, dict]:
    from hybrid_retrieval import FusedHit

    signals = analyze_query(query)
    priority_set = set(priority_doc_ids_for_signals(signals))
    warning_flags: list[str] = []

    dense_ids, dense_dist, dense_meta, dense_doc, w = query_dense_pool(
        collection,
        query,
        query_vector,
        fetch_k=fetch_k,
        source=society,
        doc_id=doc_id,
        timing=timing,
    )
    warning_flags.extend(w)

    bm25_hits = []
    bm25_query = query
    bm25_tokens: list[str] = []
    if bm25_index is not None and profile.use_bm25:
        bm25_query, bm25_tokens = enrich_query_for_meeting(
            query,
            top_level_category=profile.top_level_category,
            internal_intent=profile.internal_intent,
        )
        bm25_hits, w2 = bm25_index.search_with_fallback(
            bm25_query, top_k=fetch_k, source=society
        )
        warning_flags.extend(w2)
    elif profile.use_bm25:
        warning_flags.append("bm25_no_hit")

    if profile.use_bm25 and bm25_index is not None and not bm25_hits:
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
    )
    if profile.use_rrf:
        fused = _apply_meeting_boosts(fused, profile=profile)
    else:
        fused.sort(key=lambda h: -(h.dense_score or 0))

    # Apply category weight scaling on bm25/dense components
    if profile.bm25_weight != profile.dense_weight:
        for h in fused:
            extra = 0.0
            if h.bm25_score:
                extra += float(h.bm25_score) * 0.001 * (profile.bm25_weight - 1.0)
            if h.dense_score:
                extra += float(h.dense_score) * 0.05 * (profile.dense_weight - 1.0)
            h.final_score = round(h.final_score + extra, 6)
            h.distance = 1.0 - h.final_score
        fused.sort(key=lambda h: -h.final_score)

    fused = fused[:top_k]
    has_bm25_in_top = any(h.bm25_score is not None for h in fused[:10])
    if bm25_hits and not has_bm25_in_top:
        warning_flags.append("bm25_no_relevant_hit")

    log = {
        "bm25_query": bm25_query,
        "bm25_query_tokens": bm25_tokens,
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
        "bm25_top20": [
            {
                "chunk_id": h.chunk_id,
                "rank": h.rank,
                "score": round(float(h.score), 4),
                "file_name": h.meta.get("file_name"),
                "page": h.meta.get("page_number"),
            }
            for h in bm25_hits[:20]
        ],
        "fused_top10": [
            {
                "chunk_id": h.chunk_id,
                "rrf_score": h.rrf_score,
                "dense_score": h.dense_score,
                "bm25_score": h.bm25_score,
                "file_name": h.meta.get("file_name"),
                "page": h.meta.get("page_number"),
                "source_tier": getattr(h, "_source_tier", None),
            }
            for h in fused[:10]
        ],
        "warning_flags": list(dict.fromkeys(warning_flags)),
        "sub_query": query,
    }
    return fused, log


OUTCOME_TOPIC_BM25: dict[str, str] = {
    "ghg_safety": "MSC 111 alternative fuel ammonia hydrogen GHG safety ISE new technology",
    "lrit_vdes": "MSC 111 LRIT VDES long-range identification tracking system",
    "hormuz": "MSC 111 Strait of Hormuz maritime security navigation",
}


def _supplement_meeting_outcome(
    combined: list,
    bm25_index,
    *,
    society: str | None,
    profile: MeetingRetrievalProfile,
) -> list:
    if profile.internal_intent != "meeting_outcome" or bm25_index is None:
        return combined
    from hybrid_retrieval import FusedHit

    merged = {h.chunk_id: h for h in combined}
    present = {outcome_topic_id(h.document or "") for h in combined[:25]}

    for tid, query in OUTCOME_TOPIC_BM25.items():
        if tid in present:
            continue
        hits, _ = bm25_index.search_with_fallback(query, top_k=8, source=society)
        for bh in hits:
            if bh.chunk_id in merged:
                continue
            doc = bh.document or ""
            merged[bh.chunk_id] = FusedHit(
                chunk_id=bh.chunk_id,
                bm25_score=float(bh.score),
                bm25_rank=bh.rank,
                rrf_score=0.05,
                final_score=5.0 + float(bh.score) * 0.001,
                distance=0.01,
                meta=dict(bh.meta or {}),
                document=doc,
            )
            present.add(tid)
            break

    return sorted(merged.values(), key=lambda h: -h.final_score)


def meeting_hybrid_search(
    collection,
    bm25_index,
    question: str,
    query_vector: list[float],
    *,
    profile: MeetingRetrievalProfile,
    fetch_k: int = 56,
    top_k: int = 40,
    society: str | None = None,
    doc_id: str | None = None,
    timing=None,
    max_sub_queries: int = 1,
) -> tuple[list, dict[str, Any]]:
    # Multiple BM25 corpus scans (one per sub_query) cost ~10–15s each on the
    # full index. Collapse into one expanded query for interactive Accurate/Fast.
    if profile.sub_queries and max_sub_queries > 1:
        queries = [question, *profile.sub_queries[: max(0, max_sub_queries - 1)]]
    elif profile.sub_queries:
        queries = [" ".join([question, *profile.sub_queries[:3]]).strip()]
    else:
        queries = [question]

    merged: dict[str, Any] = {}
    logs: list[dict] = []
    all_warnings: list[str] = []

    for sq in queries:
        fused, log = _single_hybrid_search(
            collection,
            bm25_index,
            sq,
            query_vector,
            profile=profile,
            fetch_k=fetch_k,
            top_k=top_k,
            society=society,
            doc_id=doc_id,
            timing=timing,
        )
        logs.append(log)
        all_warnings.extend(log.get("warning_flags") or [])
        for h in fused:
            prev = merged.get(h.chunk_id)
            if prev is None or h.final_score > prev.final_score:
                merged[h.chunk_id] = h

    combined = sorted(merged.values(), key=lambda h: -h.final_score)
    if profile.internal_intent == "altfuel_ghg_safety":
        filtered = []
        for h in combined:
            doc = h.document or ""
            if is_excluded_chunk(
                type("_C", (), {"text": doc})(),
                profile=profile,
            ):
                continue
            filtered.append(h)
        if len(filtered) >= 5:
            combined = filtered
        else:
            all_warnings.append("weak_altfuel_evidence")
    combined = _supplement_meeting_outcome(
        combined, bm25_index, society=society, profile=profile
    )
    combined = combined[:top_k]
    primary = logs[0] if logs else {}
    payload = {
        "top_level_category": profile.top_level_category,
        "internal_intent": profile.internal_intent,
        "retrieval_profile": profile.to_log_dict(),
        "bm25_query": primary.get("bm25_query"),
        "bm25_query_tokens": primary.get("bm25_query_tokens"),
        "applied_query_expansion": primary.get("bm25_query"),
        "dense_top20": primary.get("dense_top20"),
        "bm25_top20": primary.get("bm25_top20"),
        "fused_top10": [
            {
                "chunk_id": h.chunk_id,
                "rrf_score": h.rrf_score,
                "dense_score": h.dense_score,
                "bm25_score": h.bm25_score,
                "file_name": h.meta.get("file_name"),
                "page": h.meta.get("page_number"),
                "source_tier": getattr(h, "_source_tier", None),
            }
            for h in combined[:10]
        ],
        "sub_query_logs": logs,
        "warning_flags": list(dict.fromkeys(all_warnings)),
    }
    return combined, payload
