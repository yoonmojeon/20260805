"""Clause-aware hybrid retrieval (dense + lexical + metadata boost)."""
from __future__ import annotations

import re
from typing import Any

from clause_parse import is_article_clause_number
from imo_doc_classify import classify_imo_filename, tier_for_query
from imo_doc_registry import priority_doc_ids_for_signals
from retrieval_query_analysis import (
    CLASS_RULE_SOURCES,
    QuerySignals,
    analyze_query,
    session_file_prefixes,
    topic_agenda_prefixes,
)

CLAUSE_IN_QUERY_RE = re.compile(
    r"(?:제?\s*)?(\d{3,4})\s*절|(\d{3,4})절|(?:^|\s)(\d{3,4})(?:\s*절|\s|$)",
    re.IGNORECASE,
)

TOKEN_RE = re.compile(r"[\w가-힣]+", re.UNICODE)

CLAUSE_EXACT_BOOST = 0.22
CLAUSE_IN_TEXT_BOOST = 0.08
LEXICAL_BOOST_SCALE = 0.18
REFERENCE_LINE_BOOST = 0.06
SESSION_FILE_BOOST = 0.14
TOPIC_PREFIX_BOOST = 0.20
RULE_FILENAME_BOOST = 0.24
PRIORITY_DOC_BOOST = 0.28
EXPANDED_TERM_BOOST = 0.06
SUBCOMM_PENALTY = 0.14
DOC_CODE_CROSSREF_RE = re.compile(r"document\s+code.*title", re.I)
DNV_RULE_CODE_RE = re.compile(r"^DNV-(?:CG|RP|RU)-[A-Z0-9-]+$", re.I)


def _priority_rule_file_names(signals: QuerySignals) -> list[str]:
    """Return exact class-rule filenames worth querying alongside dense hits.

    The global dense search can miss a short document code when the question is
    phrased by topic.  Exact metadata routing is cheap and keeps interactive
    retrieval fast without scanning the full BM25 corpus.
    """
    names: list[str] = []
    for hint in signals.rule_doc_hints:
        code = str(hint or "").strip()
        if DNV_RULE_CODE_RE.fullmatch(code):
            names.append(f"{code.upper()}.pdf")

    autonomous_hint = any(
        str(hint or "").lower() == "autonomous" for hint in signals.rule_doc_hints
    )
    if (
        signals.wants_rule_lookup
        and str(signals.class_society_hint or "").upper() == "DNV"
        and (autonomous_hint or "mass" in signals.topics)
    ):
        names.append("DNV-CG-0264.pdf")
    return list(dict.fromkeys(names))


def _resolve_priority_rule_doc_ids(collection, signals: QuerySignals) -> list[str]:
    """Resolve exact rule filenames to Chroma document ids."""
    names = _priority_rule_file_names(signals)
    if not names:
        return []
    where = {"file_name": names[0]} if len(names) == 1 else {"file_name": {"$in": names}}
    try:
        raw = collection.get(where=where, include=["metadatas"], limit=100)
    except Exception:
        return []
    doc_ids: list[str] = []
    for meta in raw.get("metadatas") or []:
        doc_id = str((meta or {}).get("doc_id") or "").strip()
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)
    return doc_ids


def extract_clause_hints(query: str) -> list[str]:
    hints: list[str] = []
    for groups in CLAUSE_IN_QUERY_RE.finditer(query):
        for g in groups.groups():
            if g and g.isdigit() and len(g) >= 3:
                hints.append(g)
    seen: set[str] = set()
    out: list[str] = []
    for h in hints:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def _meta_clause(meta: dict) -> str:
    return str(meta.get("clause_number") or meta.get("article_number") or "").strip()


def lexical_overlap(query: str, document: str, extra_terms: list[str] | None = None) -> float:
    q_tokens = {t for t in TOKEN_RE.findall(query.lower()) if len(t) > 1}
    if extra_terms:
        for term in extra_terms:
            q_tokens.update(t for t in TOKEN_RE.findall(term.lower()) if len(t) > 1)
    if not q_tokens:
        return 0.0
    d_tokens = {t for t in TOKEN_RE.findall(document.lower()) if len(t) > 1}
    return len(q_tokens & d_tokens) / len(q_tokens)


def _file_name(meta: dict) -> str:
    return str(meta.get("file_name") or meta.get("doc_id") or "").lower()


def _doc_type(meta: dict) -> str:
    dt = str(meta.get("doc_type") or "").strip()
    if dt:
        return dt
    return classify_imo_filename(str(meta.get("file_name") or ""))


def _metadata_boosts(
    *,
    meta: dict,
    document: str,
    signals: QuerySignals,
    priority_doc_ids: set[str],
    query: str = "",
) -> tuple[float, float]:
    boost = 0.0
    penalty = 0.0
    fname = _file_name(meta)
    doc_id = str(meta.get("doc_id") or "")
    doc_head = (document or "")[:800].lower()
    combined = f"{fname} {doc_head}"
    doc_type = _doc_type(meta)

    if doc_id and doc_id in priority_doc_ids:
        boost += PRIORITY_DOC_BOOST

    tier = tier_for_query(
        doc_type,
        wants_summary=signals.wants_summary,
        wants_outcome=signals.wants_outcome,
        wants_agenda=signals.wants_agenda,
    )
    if tier >= 0:
        boost += tier
    else:
        penalty += abs(tier)

    for prefix in session_file_prefixes(signals):
        if prefix.replace("/", "-") in fname.replace("/", "-") or prefix in fname:
            boost += SESSION_FILE_BOOST
            break

    for prefix in topic_agenda_prefixes(signals):
        if prefix in fname:
            boost += TOPIC_PREFIX_BOOST
            break

    if doc_type == "subcommittee_report" and (signals.wants_summary or signals.wants_outcome):
        penalty += SUBCOMM_PENALTY

    if signals.wants_agenda:
        if doc_type == "agenda":
            boost += 0.12

    if signals.wants_rule_lookup:
        ql = (query or "").lower()
        if "dnv-cg-0264" in fname or "cg-0264" in fname or "cg 0264" in fname:
            boost += RULE_FILENAME_BOOST + 0.12
        if "notice no.1" in fname or "notice no. 1" in fname:
            boost += RULE_FILENAME_BOOST
        if any(k in fname for k in ("autonomous", "remotely-operated", "remotely operated", "smart-vessel", "smart vessel")):
            boost += 0.14
        if any(k in combined for k in ("autonomous", "remotely operated", "smart vessel", "notation")):
            boost += 0.08
        if re.search(r"rp-c\d", fname) and "cg-0264" not in fname:
            if any(k in ql for k in ("smart", "autonomous", "자율", "vessel", "mass")):
                penalty += 0.10
        if DOC_CODE_CROSSREF_RE.search(doc_head):
            penalty += 0.16
        society = str(signals.class_society_hint or "").upper()
        chunk_source = str(meta.get("source") or "").upper()
        if society and chunk_source in CLASS_RULE_SOURCES:
            if chunk_source == society:
                boost += 0.20
            else:
                penalty += 0.22

    for hint in signals.rule_doc_hints:
        if hint.lower() in combined:
            boost += EXPANDED_TERM_BOOST

    for term in signals.expanded_terms:
        t = term.lower()
        if len(t) >= 4 and t in combined:
            boost += EXPANDED_TERM_BOOST

    if "mass" in signals.topics and "mass" in combined:
        boost += EXPANDED_TERM_BOOST
    if "ghg" in signals.topics and "ghg" in combined:
        boost += EXPANDED_TERM_BOOST
    if "alt_fuel" in signals.topics:
        if any(k in combined for k in ("low-flashpoint", "low flashpoint", "section 15", "alternative fuel", "igf")):
            boost += TOPIC_PREFIX_BOOST
    if "igc" in signals.topics and "igc" in combined:
        boost += EXPANDED_TERM_BOOST

    if signals.meeting_outcome_question:
        from meeting_outcome_retrieval import is_comparison_question, meeting_outcome_metadata_adjustment

        mo_boost, mo_penalty = meeting_outcome_metadata_adjustment(
            meta=meta,
            document=document,
            signals=signals,
            question=query,
            is_comparison=is_comparison_question(query),
        )
        boost += mo_boost
        penalty += mo_penalty

    return boost, penalty


def adjusted_distance(
    distance: float,
    *,
    query: str,
    document: str,
    meta: dict,
    clause_hints: list[str],
    signals: QuerySignals | None = None,
    priority_doc_ids: set[str] | None = None,
) -> float:
    score = float(distance)
    meta_clause = _meta_clause(meta)
    doc_head = (document or "")[:400]
    sig = signals or analyze_query(query)
    prio = priority_doc_ids or set()

    for hint in clause_hints:
        if meta_clause == hint:
            score -= CLAUSE_EXACT_BOOST
        elif hint in doc_head and is_article_clause_number(hint):
            score -= CLAUSE_IN_TEXT_BOOST
        if f"{hint}." in doc_head or f"{hint}절" in doc_head:
            score -= REFERENCE_LINE_BOOST

    meta_boost, meta_penalty = _metadata_boosts(
        meta=meta, document=document, signals=sig, priority_doc_ids=prio, query=query
    )
    score -= meta_boost
    score += meta_penalty
    score -= LEXICAL_BOOST_SCALE * lexical_overlap(query, document, sig.expanded_terms)
    return score


def _merge_where(*clauses: dict | None) -> dict | None:
    parts = [c for c in clauses if c]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}


def _meta_matches_where(meta: dict, where: dict | None) -> bool:
    if not where:
        return True
    meta = meta or {}
    if "$and" in where:
        return all(_meta_matches_where(meta, clause) for clause in where["$and"])
    if "$or" in where:
        return any(_meta_matches_where(meta, clause) for clause in where["$or"])
    for key, expected in where.items():
        actual = meta.get(key)
        if isinstance(expected, dict):
            for op, val in expected.items():
                if op == "$in":
                    if actual not in val:
                        return False
                elif op == "$eq":
                    if actual != val:
                        return False
                elif op == "$ne":
                    if actual == val:
                        return False
                else:
                    return False
        elif actual != expected:
            return False
    return True


def _filter_chroma_raw(raw: dict, where: dict | None) -> dict:
    if not where or not raw.get("ids") or not raw["ids"][0]:
        return raw
    out: dict[str, list[list]] = {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}
    for cid, dist, meta, doc in zip(
        raw["ids"][0],
        raw["distances"][0],
        raw["metadatas"][0],
        raw["documents"][0],
    ):
        if _meta_matches_where(meta or {}, where):
            out["ids"][0].append(cid)
            out["distances"][0].append(dist)
            out["metadatas"][0].append(meta)
            out["documents"][0].append(doc)
    return out


def safe_chroma_query(
    collection,
    *,
    query_embeddings: list[list[float]],
    n_results: int,
    where: dict | None = None,
) -> dict[str, Any]:
    """Chroma query with retries when the vector index returns stale/missing ids."""
    attempts: list[tuple[int, dict | None]] = [
        (n_results, where),
        (min(n_results, 40), where),
        (min(n_results, 40), None),
    ]
    last_exc: Exception | None = None
    for n, clause in attempts:
        try:
            kwargs: dict[str, Any] = {
                "query_embeddings": query_embeddings,
                "n_results": max(1, n),
            }
            if clause:
                kwargs["where"] = clause
            raw = collection.query(**kwargs)
            if clause is None and where is not None:
                return _filter_chroma_raw(raw, where)
            return raw
        except Exception as exc:
            last_exc = exc
            if "error finding id" not in str(exc).lower():
                raise
    if last_exc is not None:
        raise last_exc
    return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}


def query_with_hybrid_ranking(
    collection,
    query: str,
    query_vector: list[float],
    *,
    top_k: int = 5,
    fetch_k: int | None = None,
    source: str | None = None,
    doc_id: str | None = None,
    timing=None,
) -> dict[str, Any]:
    """Over-fetch vector hits, inject priority IMO docs, rerank with metadata boosts."""
    clause_hints = extract_clause_hints(query)
    signals = analyze_query(query)
    priority_ids = priority_doc_ids_for_signals(signals)
    if doc_id is None:
        for priority_id in _resolve_priority_rule_doc_ids(collection, signals):
            if priority_id not in priority_ids:
                priority_ids.append(priority_id)
    priority_set = set(priority_ids)
    n_fetch = fetch_k or max(top_k * 15, 80)
    base_where = _merge_where(
        {"source": source.upper()} if source else None,
        {"doc_id": doc_id} if doc_id else None,
    )

    merged_ids: list[str] = []
    merged_dist: dict[str, float] = {}
    merged_meta: dict[str, dict] = {}
    merged_doc: dict[str, str] = {}

    def absorb(raw: dict) -> None:
        if not raw.get("ids") or not raw["ids"][0]:
            return
        for cid, dist, meta, doc in zip(
            raw["ids"][0],
            raw["distances"][0],
            raw["metadatas"][0],
            raw["documents"][0],
        ):
            if cid in merged_dist:
                merged_dist[cid] = min(merged_dist[cid], float(dist))
            else:
                merged_ids.append(cid)
                merged_dist[cid] = float(dist)
                merged_meta[cid] = meta or {}
                merged_doc[cid] = doc or ""

    if timing is not None and hasattr(timing, "mark"):
        timing.mark("t_vector_search_start")

    raw_vector = safe_chroma_query(
        collection,
        query_embeddings=[query_vector],
        n_results=min(n_fetch, 150),
        where=base_where,
    )
    absorb(raw_vector)

    if timing is not None and hasattr(timing, "mark"):
        timing.mark("t_vector_search_end")
        timing.mark("t_metadata_filter_start")

    if priority_ids:
        batch = priority_ids[:20]
        try:
            routed = safe_chroma_query(
                collection,
                query_embeddings=[query_vector],
                n_results=min(20, n_fetch),
                where=_merge_where(base_where, {"doc_id": {"$in": batch}}),
            )
            absorb(routed)
        except Exception:
            for pid in batch[:8]:
                try:
                    one = safe_chroma_query(
                        collection,
                        query_embeddings=[query_vector],
                        n_results=8,
                        where=_merge_where(base_where, {"doc_id": pid}),
                    )
                    absorb(one)
                except Exception:
                    pass

    if clause_hints:
        for hint in clause_hints[:3]:
            try:
                clause_where = {
                    "$or": [
                        {"clause_number": hint},
                        {"article_number": hint},
                    ]
                }
                filtered = safe_chroma_query(
                    collection,
                    query_embeddings=[query_vector],
                    n_results=min(15, n_fetch),
                    where=_merge_where(base_where, clause_where),
                )
                absorb(filtered)
            except Exception:
                pass

    if timing is not None and hasattr(timing, "mark"):
        timing.mark("t_metadata_filter_end")
        timing.mark("t_rerank_start")

    ranked = sorted(
        merged_ids,
        key=lambda cid: adjusted_distance(
            merged_dist[cid],
            query=query,
            document=merged_doc[cid],
            meta=merged_meta[cid],
            clause_hints=clause_hints,
            signals=signals,
            priority_doc_ids=priority_set,
        ),
    )[:top_k]

    if timing is not None and hasattr(timing, "mark"):
        timing.mark("t_rerank_end")

    return {
        "ids": [ranked],
        "distances": [[merged_dist[cid] for cid in ranked]],
        "metadatas": [[merged_meta[cid] for cid in ranked]],
        "documents": [[merged_doc[cid] for cid in ranked]],
        "clause_hints": clause_hints,
        "query_signals": signals,
        "priority_doc_ids": priority_ids,
    }


def _enrich_query_standard(query: str) -> str:
    """Base query enrichment without meeting-outcome branch."""
    hints = extract_clause_hints(query)
    signals = analyze_query(query)
    parts: list[str] = []
    if hints:
        parts.extend(f"{h}절 {h}." for h in hints[:2])
    if signals.expanded_terms:
        parts.extend(signals.expanded_terms[:12])
    if signals.wants_summary or signals.wants_outcome:
        parts.extend(["outcome", "executive summary", "report", "resolution", "decision"])
    if not parts:
        return query
    return f"{' '.join(parts)} {query}".strip()


def enrich_query_for_embedding(query: str, model_name: str) -> str:
    """Prepend clause hints and cross-lingual/session terms for E5."""
    signals = analyze_query(query)
    if signals.meeting_outcome_question:
        from meeting_outcome_retrieval import enrich_meeting_outcome_query

        return enrich_meeting_outcome_query(query, model_name)
    from table_retrieval import enrich_table_query_for_embedding, is_table_question

    if is_table_question(query):
        return enrich_table_query_for_embedding(_enrich_query_standard(query))
    return _enrich_query_standard(query)
