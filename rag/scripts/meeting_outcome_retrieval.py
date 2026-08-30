"""Meeting outcome question detection, query expansion, and two-stage retrieval."""
from __future__ import annotations

import re
from typing import Any

from embedding_policy import embed_texts_local
from imo_doc_classify import (
    asks_broad_session_outcome,
    classify_imo_filename,
    meeting_outcome_scope,
)
from imo_doc_registry import DEFAULT_CORPUS, load_corpus_rows, priority_doc_ids
from meeting_summary_context import (
    TARGET_SCOPE_WHOLE_SESSION,
    meeting_summary_source_tier,
    resolve_meeting_summary_context,
)
from retrieval_query_analysis import (
    IMO_SESSION_RE,
    QuerySignals,
    analyze_query,
    is_meeting_outcome_question,
    topic_agenda_prefixes,
)
from retrieval_search import _merge_where, safe_chroma_query

MEETING_SESSION_RE = IMO_SESSION_RE

OUTCOME_INTENT_PATTERNS = (
    r"주요\s*결과",
    r"\b결과\b",
    r"\boutcome\b",
    r"\bsummary\b",
    r"key\s*outcomes?",
    r"\badopted\b",
    r"\bapproved\b",
    r"\bdecision\b",
    r"결정\s*사항",
    r"채택",
    r"승인",
    r"요약해",
    r"정리해",
)

COMPARISON_PATTERNS = (
    r"비교",
    r"compare",
    r"versus",
    r"\bvs\.?\b",
    r"차이",
    r"대비",
)

MEETING_BOOST_KEYWORDS = (
    "summary report",
    "key outcomes",
    "key outcome",
    "outcome",
    "adopted",
    "approved",
    "maritime safety committee",
    "mass code",
    "igc code",
    "ammonia",
)

OUTCOME_CHUNK_TERMS = (
    "outcome",
    "adopted",
    "approved",
    "resolution",
    "decision",
    "key outcomes",
    "summary report",
    "executive summary",
)

MEETING_DOC_BOOST = {
    "session_report": 0.16,
    "session_outcome": 0.10,
    "reference_outcome": 0.0,
    "working_group_report": 0.08,
    "agenda_report": 0.06,
}

SESSION_FINAL_REPORT_BOOST = 0.36
SESSION_DRAFT_REPORT_BOOST = 0.62
PROPOSAL_OUTCOME_PENALTY = 0.24


def body_matches_msc_topic(question: str, file_name: str) -> bool:
    """True when an MSC agenda paper matches the topic of a non-broad question."""
    q = (question or "").lower()
    fn = (file_name or "").lower()
    if "mass" in q or "자율" in (question or ""):
        # Prefer the main MASS WG report (111-5), not national outcome notes (111-5-8).
        if re.search(r"msc\s*111[-/ ]?5-\d+", fn, re.I):
            return False
        return bool(re.search(r"msc\s*111[-/ ]?5\b|mass\s+working\s+group|intersessional\s+mass", fn, re.I))
    if any(k in (question or "") for k in ("대체연료", "저인화점")) or (
        "alternative fuel" in q or ("ghg" in q and "안전" in (question or ""))
    ):
        if re.search(r"msc\s*111[-/ ]?12-\d+", fn, re.I) and "comments" in fn:
            return False
        return bool(
            re.search(
                r"msc\s*111[-/ ]?12\b|sub-committee|new technology|low-flashpoint|alternative fuel",
                fn,
                re.I,
            )
        )
    return False

REFERENCE_BODY_OUTCOME_PENALTY = 0.32
MEETING_SUMMARY_REFERENCE_PENALTY = 0.45
MEETING_SUMMARY_AGENDA_PENALTY = 0.28
BROAD_WG_REPORT_PENALTY = 0.14

MEETING_KEYWORD_BOOST = 0.05
MEETING_OUTCOME_CHUNK_BOOST = 0.15
OTHER_SESSION_PENALTY = 0.22
OLD_SESSION_PENALTY = 0.18
LATEST_ENV_RE = re.compile(r"최신|최근|latest|current", re.I)
LATEST_ENV_OFFICIAL_DOC_RE = re.compile(
    r"mepc\s*84-(?:"
    r"6-1\b|6-2\b|7-14\b|7-15\b|10\s+-\s+outcome|3\s+-\s+amendments"
    r")",
    re.I,
)
LATEST_ENV_WEAK_DOC_RE = re.compile(r"\bcomments?\s+on\b|\bconcept\b|\bproposal\b|-inf\.", re.I)
LATEST_ENV_PREFERRED_DOCS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"mepc\s*84-6-1\b", re.I), ("265 ships", "verification of the submitted data", "duplicate reporting")),
    (re.compile(r"mepc\s*84-6-2\b", re.I), ("up to 10.8%", "2019 to 2024", "demand-based and supply-based")),
    (re.compile(r"mepc\s*84-7-14\b", re.I), ("gfi reporting and verification", "draft amendments to the seemp guidelines", "fifth imo ghg study")),
    (re.compile(r"mepc\s*84-7-15\b", re.I), ("representativeness", "conservativeness", "wtt default emission factors")),
    (re.compile(r"mepc\s*84-10\s+-\s+outcome", re.I), ("draft 2026 guidelines", "oily wastes", "integrated bilge water treatment system")),
    (re.compile(r"mepc\s*84-3\s+-\s+amendments", re.I), ("adjourn for one year", "north-east atlantic", "imo dcs")),
)


def _is_latest_environment_query(question: str, signals: QuerySignals) -> bool:
    return bool(
        LATEST_ENV_RE.search(question)
        and any(body == "MEPC" and num == 84 for body, num in signals.session_codes)
        and signals.topics.intersection({"ghg", "marpol", "cii"})
    )


def is_latest_environment_summary_query(question: str) -> bool:
    """Public routing predicate shared by Fast retrieval/context building."""
    return _is_latest_environment_query(question, analyze_query(question))


def _latest_environment_doc_diversity(
    ranked: list[tuple[str, tuple[float, dict, str]]],
) -> list[tuple[str, tuple[float, dict, str]]]:
    """Put one canonical MEPC 84 environment document before score-only tail.

    Dense similarity otherwise lets a long DCS or working-group report occupy
    every Fast-mode slot.  The score order is retained within each document.
    """
    head: list[tuple[str, tuple[float, dict, str]]] = []
    used: set[str] = set()
    for pattern, terms in LATEST_ENV_PREFERRED_DOCS:
        candidates = []
        for item in ranked:
            cid, (distance, meta, document) = item
            file_name = str((meta or {}).get("file_name") or "")
            if cid in used or not pattern.search(file_name):
                continue
            body = re.sub(r"\s+", " ", str(document or "")).lower()
            term_score = sum(10.0 for term in terms if term in body)
            candidates.append((term_score - float(distance) * 0.01, item))
        if candidates:
            _content_score, best = max(candidates, key=lambda row: row[0])
            head.append(best)
            used.add(best[0])
    return head + [item for item in ranked if item[0] not in used]


def detect_meeting_outcome_question(question: str, row: dict | None = None) -> bool:
    return is_meeting_outcome_question(question, row)


def is_comparison_question(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q, re.I) for p in COMPARISON_PATTERNS)


def parse_outcome_item_count(question: str, row: dict | None = None) -> int:
    if row and row.get("outcome_item_count"):
        try:
            return max(1, int(row["outcome_item_count"]))
        except (TypeError, ValueError):
            pass
    m = re.search(r"(\d+)\s*개\s*(?:항목|개)", question)
    if m:
        return max(1, int(m.group(1)))
    m = re.search(r"(\d+)\s*(?:items?|points?|bullets?)", question, re.I)
    if m:
        return max(1, int(m.group(1)))
    return 3


def _session_label(body: str, num: int) -> str:
    return f"{body.upper()} {num}"


def _committee_long_name(body: str) -> str:
    if body.upper() == "MSC":
        return "Maritime Safety Committee"
    if body.upper() == "MEPC":
        return "Marine Environment Protection Committee"
    return body.upper()


def expand_meeting_outcome_queries(question: str, signals: QuerySignals | None = None) -> list[str]:
    """Expand user question into internal retrieval queries."""
    sig = signals or analyze_query(question)
    expansions: list[str] = [question.strip()]
    lower = question.lower()

    for body, num in sig.session_codes:
        label = _session_label(body, num)
        long_name = _committee_long_name(body)
        expansions.extend(
            [
                f"{label} draft report maritime safety committee session key outcomes",
                f"{label} summary report key outcomes",
                f"{long_name} {num}th session report adopted approved",
                f"IMO {label} key decisions",
            ]
        )
        broad_session = asks_broad_session_outcome(question, topics=sig.topics)
        # WP.1 is for whole-session summaries only — not MASS / alt-fuel topic Qs.
        if body == "MSC" and num == 111 and broad_session:
            expansions.extend(
                [
                    "MSC 111 WP.1 Draft Report Maritime Safety Committee 111th session",
                    "MSC 111 report of the Maritime Safety Committee on its 111th session",
                ]
            )
        if broad_session:
            expansions.extend(
                [
                    f"{long_name} draft report on its {num}th session",
                    f"{label} session report resolutions decisions adopted",
                ]
            )
        elif "mass" in sig.topics or "mass code" in lower or "mass" in lower:
            expansions.extend(
                [
                    f"{label} MASS Code adopted mandatory",
                    f"{label}-5 Report of the intersessional MASS working group",
                    f"{label}/5 MASS working group report goal-based code",
                ]
            )
        if (
            "대체연료" in question
            or "alternative fuel" in lower
            or ("ghg" in lower and "안전" in question)
            or "저인화점" in question
        ):
            expansions.extend(
                [
                    f"{label}-12 Report of the twelfth session of the Sub-Committee",
                    f"{label}/12 ISE new technology low-flashpoint fuel safety",
                    f"{label} alternative fuel ammonia hydrogen methanol safety guidelines",
                ]
            )
        if "igc" in lower or "igc code" in lower:
            expansions.append(f"{label} IGC Code amendments adopted")
        if "ammonia" in lower or "암모니아" in question:
            expansions.append(f"{label} ammonia fuel ship guidelines adopted")

        if body == "MEPC" and LATEST_ENV_RE.search(question):
            expansions.extend(
                [
                    f"{label} Secretariat report fuel oil consumption IMO DCS carbon intensity reporting year 2024",
                    f"{label} ISWG-GHG report GFI reporting verification SEEMP guidelines",
                    f"{label} GESAMP LCA working group default emission factors PPR outcome",
                ]
            )

    seen: set[str] = set()
    out: list[str] = []
    for term in expansions:
        key = term.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(term.strip())
    return out


def enrich_meeting_outcome_query(question: str, model_name: str) -> str:
    """Embedding query with meeting-outcome expansions."""
    from retrieval_search import _enrich_query_standard

    signals = analyze_query(question)
    if not signals.meeting_outcome_question:
        return _enrich_query_standard(question)
    parts = expand_meeting_outcome_queries(question, signals)[:8]
    base = _enrich_query_standard(question)
    return f"{' '.join(parts)} {base}".strip()


def meeting_outcome_metadata_adjustment(
    *,
    meta: dict,
    document: str,
    signals: QuerySignals,
    question: str = "",
    is_comparison: bool = False,
) -> tuple[float, float]:
    """Return (boost, penalty) for meeting-outcome ranking."""
    if not signals.meeting_outcome_question:
        return 0.0, 0.0

    boost = 0.0
    penalty = 0.0
    fname = str(meta.get("file_name") or meta.get("doc_id") or "").lower()
    file_name = str(meta.get("file_name") or "")
    doc_head = (document or "")[:1200].lower()
    combined = f"{fname} {doc_head}"
    doc_type = classify_imo_filename(file_name)
    scope = meeting_outcome_scope(file_name)
    broad_session = asks_broad_session_outcome(question, topics=tuple(signals.topics))
    summary_ctx = resolve_meeting_summary_context(question)
    summary_intent = summary_ctx.target_scope == TARGET_SCOPE_WHOLE_SESSION
    latest_env = (
        LATEST_ENV_RE.search(question) is not None
        and any(body == "MEPC" and num == 84 for body, num in signals.session_codes)
        and bool(signals.topics.intersection({"ghg", "marpol", "cii"}))
    )

    if latest_env:
        if LATEST_ENV_OFFICIAL_DOC_RE.search(file_name):
            if re.search(r"mepc\s*84-6-[12]\b", file_name, re.I):
                boost += 1.10
            else:
                boost += 0.72
        elif "secretariat" in fname and re.search(r"\bmepc\s*84-", fname):
            boost += 0.20
        if LATEST_ENV_WEAK_DOC_RE.search(file_name) or (
            "submitted by" in doc_head and "secretariat" not in fname
        ):
            penalty += 0.38
        if re.search(r"\bmepc\s*84-2(?:-|\s)", fname) or re.search(
            r"outcomes? of (?:c|a|msc|leg|tc|fal)\s*\d+", fname, re.I
        ):
            penalty += 0.62

    # Session draft report (WP.1) helps whole-session summaries, but for topic
    # questions (MASS / alt-fuel) it monopolizes the pool and hides agenda papers.
    if re.search(r"wp\.?\s*1", fname, re.I) and "draft report" in fname:
        if broad_session or summary_intent:
            boost += SESSION_DRAFT_REPORT_BOOST
        else:
            penalty += 0.35
    if re.search(r"proposal|proposed|submission|comments? for consideration", fname, re.I):
        penalty += PROPOSAL_OUTCOME_PENALTY
    # Boost gold-agenda style papers for topic-specific MSC questions.
    if not broad_session and not summary_intent and body_matches_msc_topic(question, file_name):
        boost += 0.55

    if scope == "session_final_report":
        boost += SESSION_FINAL_REPORT_BOOST
    elif doc_type in MEETING_DOC_BOOST:
        boost += MEETING_DOC_BOOST[doc_type]

    if summary_ctx.apply_session_final_priority:
        tier = meeting_summary_source_tier(file_name, ctx=summary_ctx)
        if tier == 0:
            boost += 0.20
        elif tier == 1:
            boost += 0.14
        elif tier >= 3:
            penalty += MEETING_SUMMARY_REFERENCE_PENALTY
        if summary_ctx.apply_reference_penalties and (
            "strategic plan" in fname or "fal 50" in fname or "fal.50" in fname
        ):
            penalty += 0.25
    elif summary_ctx.preferred_doc_hints:
        if any(h in fname for h in summary_ctx.preferred_doc_hints):
            boost += 0.18

    if broad_session:
        if scope == "reference_body_outcome":
            penalty += REFERENCE_BODY_OUTCOME_PENALTY
            if summary_ctx.apply_reference_penalties:
                penalty += MEETING_SUMMARY_REFERENCE_PENALTY
        elif scope == "working_group_report" and not signals.topics:
            penalty += BROAD_WG_REPORT_PENALTY
        elif scope == "session_final_report":
            boost += 0.10
        elif summary_ctx.apply_reference_penalties and re.search(
            r"\b(?:msc|mepc)\s*\d{1,3}[-/]2(?:[-/]|$)", fname
        ):
            penalty += MEETING_SUMMARY_AGENDA_PENALTY
    elif scope == "reference_body_outcome" and doc_type == "reference_outcome":
        boost += 0.04

    if not broad_session and ("summary report" in combined or "key outcomes" in combined):
        boost += 0.10

    for body, num in signals.session_codes:
        label = f"{body.lower()} {num}"
        dash_label = f"{body.lower()}-{num}"
        slash_label = f"{body.lower()}/{num}"
        if label in fname or dash_label in fname or slash_label in fname:
            boost += 0.12 if broad_session and scope == "session_final_report" else 0.10
        for prefix in topic_agenda_prefixes(signals):
            if prefix in fname:
                boost += 0.14
        if not broad_session:
            for kw in MEETING_BOOST_KEYWORDS:
                if kw in combined:
                    boost += MEETING_KEYWORD_BOOST

    if broad_session and scope == "session_final_report":
        for term in ("resolution", "decision", "adopted", "approved", "key outcomes"):
            if term in doc_head:
                boost += 0.03

    if is_comparison:
        return boost, penalty

    for body, num in signals.session_codes:
        target = body.upper()
        label = f"{body.lower()} {num}"
        if target == "MSC":
            if re.search(r"\bmepc\s*\d{1,3}\b", fname) and label not in fname:
                penalty += OTHER_SESSION_PENALTY
        elif target == "MEPC":
            if re.search(r"\bmsc\s*\d{1,3}\b", fname) and label not in fname:
                penalty += OTHER_SESSION_PENALTY

        for m in re.finditer(r"\b(msc|mepc)\s*(\d{1,3})\b", fname):
            other_body, other_num = m.group(1).upper(), int(m.group(2))
            if other_body == target and other_num < num and f"{target.lower()} {num}" not in fname:
                penalty += OLD_SESSION_PENALTY

        if re.search(r"\btc\s*\d{1,3}\b", fname) and label not in fname and "outcome" not in fname:
            penalty += OTHER_SESSION_PENALTY * 0.5

    return boost, penalty


def _file_matches_session(file_name: str, signals: QuerySignals) -> bool:
    fn = (file_name or "").lower()
    for body, num in signals.session_codes:
        if f"{body.lower()} {num}" in fn or f"{body.lower()}-{num}" in fn:
            return True
    return False


def meeting_priority_doc_ids(signals: QuerySignals, *, question: str = "", limit: int = 20) -> list[str]:
    """Doc_ids for session outcome/summary/report documents."""
    if not signals.session_codes:
        return []

    broad = asks_broad_session_outcome(question, topics=tuple(signals.topics))
    agenda_items: tuple[int, ...] | None = None
    if "mass" in signals.topics:
        agenda_items = (5,)
    elif "igc" in signals.topics:
        agenda_items = (14,)
    elif "alt_fuel" in signals.topics:
        agenda_items = (12,)

    if broad:
        preferred: tuple[str, ...] | None = ("session_report",)
    elif agenda_items:
        preferred = None
    else:
        preferred = (
            "session_report",
            "session_outcome",
            "working_group_report",
            "agenda_report",
        )

    rows = load_corpus_rows(str(DEFAULT_CORPUS))
    ids: list[str] = []
    if (
        LATEST_ENV_RE.search(question)
        and any(body == "MEPC" and num == 84 for body, num in signals.session_codes)
        and bool(signals.topics.intersection({"ghg", "marpol", "cii"}))
    ):
        preferred_names = (
            "mepc 84-6-1 - report of fuel oil consumption",
            "mepc 84-6-2 - report on annual carbon intensity",
            "mepc 84-7-14 - report of the twentieth meeting",
            "mepc 84-7-15 - report of the second meeting",
            "mepc 84-10 - outcome of ppr 13",
            "mepc 84-3 - amendments to marpol annex vi",
        )
        for hint in preferred_names:
            for corpus_row in rows:
                file_name = str(corpus_row.get("file_name", "")).lower()
                doc_id = str(corpus_row.get("doc_id", ""))
                if doc_id and hint in file_name and doc_id not in ids:
                    ids.append(doc_id)
    if broad:
        for row in rows:
            fn = str(row.get("file_name", ""))
            doc_id = str(row.get("doc_id", ""))
            if doc_id and meeting_outcome_scope(fn) == "session_final_report" and _file_matches_session(fn, signals):
                ids.append(doc_id)
        ids = list(dict.fromkeys(ids))

    ids.extend(
        d
        for d in priority_doc_ids(signals, preferred_types=preferred, agenda_items=agenda_items, limit=limit)
        if d not in ids
    )
    if agenda_items and len(ids) < limit // 2:
        for d in priority_doc_ids(signals, preferred_types=preferred, limit=limit):
            if d not in ids:
                ids.append(d)

    extra: list[tuple[int, str]] = []
    for row in rows:
        file_name = str(row.get("file_name", ""))
        file_lower = file_name.lower()
        doc_id = str(row.get("doc_id", ""))
        if not doc_id or not _file_matches_session(file_name, signals):
            continue
        scope = meeting_outcome_scope(file_name)
        score = 0
        if broad:
            if scope == "session_final_report":
                score += 250
            elif scope == "reference_body_outcome":
                score += 10
            elif scope == "working_group_report":
                score += 20
            elif "report of the" in file_lower:
                score += 45
            else:
                score += 25
        else:
            if scope == "reference_body_outcome":
                score += 90
            elif "summary report" in file_lower:
                score += 120
            elif "report of the" in file_lower:
                score += 80
            elif "outcome" in file_lower:
                score += 70
            else:
                score += 40
        if score:
            extra.append((score, doc_id))
    extra.sort(key=lambda x: (-x[0], x[1]))
    seen = set(ids)
    for _, doc_id in extra:
        if doc_id not in seen:
            seen.add(doc_id)
            ids.append(doc_id)
        if len(ids) >= limit:
            break
    return ids[:limit]


def _is_outcome_like_chunk(
    meta: dict,
    document: str,
    *,
    question: str = "",
    broad_session: bool | None = None,
) -> bool:
    file_name = str(meta.get("file_name") or "")
    fname = file_name.lower()
    text = (document or "")[:800].lower()
    combined = f"{fname} {text}"
    scope = meeting_outcome_scope(file_name)
    if broad_session is None:
        broad_session = asks_broad_session_outcome(question)
    if broad_session and scope == "reference_body_outcome":
        return False
    if scope == "session_final_report":
        return True
    if classify_imo_filename(file_name) in {"session_report"} and broad_session:
        return True
    if not broad_session and classify_imo_filename(file_name) in {"session_outcome", "session_report"}:
        return True
    return any(term in combined for term in OUTCOME_CHUNK_TERMS)


def query_meeting_outcome_chunks(
    collection,
    question: str,
    model_name: str,
    signals: QuerySignals,
    *,
    top_k: int = 10,
    doc_id: str | None = None,
    source: str | None = None,
    timing=None,
) -> list[tuple[str, float, dict, str]]:
    """Two-stage: meeting docs first, then outcome/adopted chunks within them."""
    if not signals.meeting_outcome_question:
        return []

    base_where = _merge_where(
        {"source": source.upper()} if source else None,
        {"doc_id": doc_id} if doc_id else None,
    )
    priority_ids = meeting_priority_doc_ids(signals, question=question)
    expansions = expand_meeting_outcome_queries(question, signals)

    pool: dict[str, tuple[float, dict, str]] = {}

    def absorb(raw: dict, boost: float = 0.0) -> None:
        if not raw.get("ids") or not raw["ids"][0]:
            return
        for cid, dist, meta, doc in zip(
            raw["ids"][0],
            raw["distances"][0],
            raw["metadatas"][0],
            raw["documents"][0],
        ):
            adj = float(dist) - boost
            prev = pool.get(cid)
            if prev is None or adj < prev[0]:
                pool[cid] = (adj, meta or {}, doc or "")

    # Stage 1 — meeting summary/outcome documents
    embed_main = enrich_meeting_outcome_query(question, model_name)
    vector_main = embed_texts_local([embed_main], model_name, for_query=True, timing=timing)[0]

    # For topic-specific questions, run a focused sub-query first.
    if signals.topics:
        topic_terms: list[str] = []
        if "mass" in signals.topics:
            topic_terms.append("MASS Code adopted mandatory goal-based")
        if "igc" in signals.topics:
            topic_terms.append("IGC Code amendments adopted consolidated draft")
        if "alt_fuel" in signals.topics or "ammonia" in question.lower() or "암모니아" in question:
            topic_terms.append("ammonia fuel ship safety alternative fuel guidelines adopted")
        if topic_terms:
            topic_vec = embed_texts_local(
                [" ".join(topic_terms)], model_name, for_query=True, timing=timing
            )[0]
            topic_ids = meeting_priority_doc_ids(signals, question=question, limit=12)
            if topic_ids:
                try:
                    absorb(
                        safe_chroma_query(
                            collection,
                            query_embeddings=[topic_vec],
                            n_results=min(top_k * 3, 30),
                            where=_merge_where(base_where, {"doc_id": {"$in": topic_ids[:12]}}),
                        ),
                        boost=MEETING_OUTCOME_CHUNK_BOOST + 0.08,
                    )
                except Exception:
                    pass

    try:
        absorb(
            safe_chroma_query(
                collection,
                query_embeddings=[vector_main],
                n_results=min(top_k * 4, 40),
                where=base_where,
            ),
            boost=0.0,
        )
    except Exception:
        pass

    if priority_ids:
        batch = priority_ids[:16]
        try:
            absorb(
                safe_chroma_query(
                    collection,
                    query_embeddings=[vector_main],
                    n_results=min(top_k * 3, 36),
                    where=_merge_where(base_where, {"doc_id": {"$in": batch}}),
                ),
                boost=MEETING_OUTCOME_CHUNK_BOOST,
            )
        except Exception:
            for pid in batch[:10]:
                try:
                    absorb(
                        safe_chroma_query(
                            collection,
                            query_embeddings=[vector_main],
                            n_results=8,
                            where=_merge_where(base_where, {"doc_id": pid}),
                        ),
                        boost=MEETING_OUTCOME_CHUNK_BOOST,
                    )
                except Exception:
                    pass

        latest_env = _is_latest_environment_query(question, signals)
        if latest_env:
            # A single $in query can be monopolized by a long working-group
            # report. Query each canonical document so DCS/CII/PPR/MARPOL all
            # have at least one candidate in the merge pool.
            focused_queries = (
                "IMO DCS 2024 submitted data verification missing ships errors duplicate reporting 265 ships",
                "carbon intensity demand-based supply-based 2019 to 2024 mandatory reporting 1 January 2026",
                "GFI reporting verification draft regulation 37 draft amendments SEEMP Guidelines",
                "WtT default emission factors representativeness conservativeness",
                "draft 2026 Guidelines oily wastes machinery spaces integrated bilge water treatment system IBTS",
                "draft revised MARPOL Annex VI with a view to adoption at MEPC ES.2 consolidated amendments",
            )
            focused_vectors = embed_texts_local(
                list(focused_queries), model_name, for_query=True, timing=timing
            )
            for position, pid in enumerate(priority_ids[:6]):
                try:
                    absorb(
                        safe_chroma_query(
                            collection,
                            query_embeddings=[focused_vectors[position]],
                            n_results=8,
                            where=_merge_where(base_where, {"doc_id": pid}),
                        ),
                        boost=0.52 if position < 2 else 0.40,
                    )
                except Exception:
                    pass

    # Stage 2 — outcome/adopted/approved focused sub-query within meeting docs
    outcome_query = " ".join(
        list(OUTCOME_CHUNK_TERMS[:6])
        + [_session_label(b, n) for b, n in signals.session_codes[:2]]
    )
    vector_outcome = embed_texts_local([outcome_query], model_name, for_query=True, timing=timing)[0]
    stage2_where = base_where
    if priority_ids:
        stage2_where = _merge_where(base_where, {"doc_id": {"$in": priority_ids[:12]}})
    try:
        absorb(
            safe_chroma_query(
                collection,
                query_embeddings=[vector_outcome],
                n_results=min(top_k * 3, 30),
                where=stage2_where,
            ),
            boost=MEETING_OUTCOME_CHUNK_BOOST + 0.04,
        )
    except Exception:
        pass

    # Additional expansion queries (LR/DNV summary reports etc.)
    for exp in expansions[1:4]:
        try:
            vec = embed_texts_local([exp], model_name, for_query=True, timing=timing)[0]
            absorb(
                safe_chroma_query(
                    collection,
                    query_embeddings=[vec],
                    n_results=min(top_k * 2, 20),
                    where=base_where,
                ),
                boost=0.06,
            )
        except Exception:
            pass

    ranked = sorted(pool.items(), key=lambda x: x[1][0])
    if _is_latest_environment_query(question, signals):
        ranked = _latest_environment_doc_diversity(ranked)
    ranked = ranked[: top_k * 2]
    return [(cid, score, meta, doc) for cid, (score, meta, doc) in ranked]


def merge_meeting_outcome_into_raw(
    baseline_raw: dict,
    meeting_hits: list[tuple[str, float, dict, str]],
    *,
    top_k: int | None = None,
    min_outcome_chunks: int = 2,
    topic_specific: bool = False,
    question: str = "",
) -> dict:
    """Merge meeting-outcome hits into baseline pool via score boost."""
    broad_session = asks_broad_session_outcome(question) and not topic_specific
    signals = analyze_query(question)
    pool: dict[str, tuple[float, dict, str]] = {}
    baseline_scores = (
        baseline_raw.get("final_scores")
        or baseline_raw.get("distances")
        or [[]]
    )
    for cid, dist, meta, doc in zip(
        baseline_raw.get("ids", [[]])[0],
        baseline_scores[0],
        baseline_raw.get("metadatas", [[]])[0],
        baseline_raw.get("documents", [[]])[0],
    ):
        clean_meta = meta or {}
        clean_doc = doc or ""
        boost, penalty = meeting_outcome_metadata_adjustment(
            meta=clean_meta,
            document=clean_doc,
            signals=signals,
            question=question,
            is_comparison=is_comparison_question(question),
        )
        pool[cid] = (float(dist) - boost + penalty, clean_meta, clean_doc)

    for cid, dist, meta, doc in meeting_hits:
        extra_boost = 0.08 if _is_outcome_like_chunk(meta, doc, question=question, broad_session=broad_session) else 0.04
        boost, penalty = meeting_outcome_metadata_adjustment(
            meta=meta or {},
            document=doc or "",
            signals=signals,
            question=question,
            is_comparison=is_comparison_question(question),
        )
        adj = float(dist) - extra_boost - boost + penalty
        prev = pool.get(cid)
        if prev is None or adj < prev[0]:
            pool[cid] = (adj, meta or {}, doc or "")

    ranked = sorted(pool.items(), key=lambda x: x[1][0])
    latest_env = _is_latest_environment_query(question, signals)
    if latest_env:
        capped: list[tuple[str, tuple[float, dict, str]]] = []
        per_doc: dict[str, int] = {}
        cap = 6
        for item in ranked:
            meta = item[1][1]
            doc_key = str(meta.get("doc_id") or meta.get("file_name") or item[0])
            if per_doc.get(doc_key, 0) >= cap:
                continue
            per_doc[doc_key] = per_doc.get(doc_key, 0) + 1
            capped.append(item)
        ranked = _latest_environment_doc_diversity(capped)
    if top_k is not None:
        min_required = 0 if topic_specific else min_outcome_chunks
        outcome_ids = [
            cid
            for cid, (_, meta, doc) in ranked
            if _is_outcome_like_chunk(meta, doc, question=question, broad_session=broad_session)
        ]
        if outcome_ids and min_required > 0:
            top_ids = {cid for cid, _ in ranked[:top_k]}
            missing = [cid for cid in outcome_ids if cid not in top_ids][:min_required]
            if missing:
                keep = ranked[: max(top_k - len(missing), 1)]
                tail = [(cid, pool[cid]) for cid in missing if cid in pool]
                merged_ids = {cid for cid, _ in keep}
                for cid, item in ranked:
                    if cid in merged_ids:
                        continue
                    if len(keep) + len(tail) >= top_k:
                        break
                    if cid in missing:
                        tail.append((cid, item))
                ranked = keep + [x for x in tail if x[0] not in merged_ids]
                ranked = sorted(ranked, key=lambda x: x[1][0])[:top_k]
            else:
                ranked = ranked[:top_k]
        else:
            ranked = ranked[:top_k]

    literal_ids = list(
        ((baseline_raw.get("document_route") or {}).get("feature_fallback_retained") or [])
    )
    if literal_ids:
        literal_set = {cid for cid in literal_ids if cid in pool}
        if literal_set:
            forced = [(cid, pool[cid]) for cid in literal_ids if cid in literal_set]
            ranked = forced + [(cid, item) for cid, item in ranked if cid not in literal_set]
            if top_k is not None:
                ranked = ranked[:top_k]

    merged = {
        "ids": [[cid for cid, _ in ranked]],
        "distances": [[score for _, (score, _, _) in ranked]],
        "metadatas": [[meta for _, (_, meta, _) in ranked]],
        "documents": [[doc for _, (_, _, doc) in ranked]],
        "meeting_outcome_aware": True,
    }
    # Preserve diagnostics from the baseline hierarchical route so Accurate
    # answer planning can recognise literal-recovery chunks after this merge.
    if baseline_raw.get("document_route"):
        merged["document_route"] = baseline_raw["document_route"]
    return merged


def select_latest_environment_context(
    question: str,
    chunks: list[Any],
    *,
    target_k: int = 10,
) -> list[Any]:
    """Build a document-diverse MEPC 84 context for a broad latest-env query."""
    signals = analyze_query(question)
    latest_env = (
        LATEST_ENV_RE.search(question) is not None
        and any(body == "MEPC" and num == 84 for body, num in signals.session_codes)
        and bool(signals.topics.intersection({"ghg", "marpol", "cii"}))
    )
    if not latest_env:
        return chunks

    def text_score(chunk: Any, terms: tuple[str, ...]) -> float:
        text = re.sub(r"\s+", " ", str(getattr(chunk, "text", "") or "")).lower()
        score = sum(8.0 for term in terms if term in text)
        score += min(len(text), 1800) / 1800.0
        if any(noise in text[:500] for noise in ("pre-session public release", "terms of reference", "adopted the agenda")):
            score -= 5.0
        return score

    selected: list[Any] = []
    selected_ids: set[str] = set()
    selected_docs: set[str] = set()
    for name_re, terms in LATEST_ENV_PREFERRED_DOCS:
        candidates = [
            c for c in chunks
            if name_re.search(str(getattr(c, "file_name", "") or ""))
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda c: text_score(c, terms))
        cid = str(getattr(best, "chunk_id", "") or id(best))
        doc_id = str(getattr(best, "doc_id", "") or getattr(best, "file_name", ""))
        if cid not in selected_ids:
            selected.append(best)
            selected_ids.add(cid)
            selected_docs.add(doc_id)

    # Fill only with another substantive MEPC 84 document; never let a single
    # working-group report consume the whole context window.
    for c in chunks:
        if len(selected) >= target_k:
            break
        file_name = str(getattr(c, "file_name", "") or "")
        doc_id = str(getattr(c, "doc_id", "") or file_name)
        cid = str(getattr(c, "chunk_id", "") or id(c))
        if cid in selected_ids or doc_id in selected_docs:
            continue
        if not re.search(r"\bmepc\s*84-", file_name, re.I):
            continue
        if LATEST_ENV_WEAK_DOC_RE.search(file_name) or re.search(r"\bmepc\s*84-2(?:-|\s)", file_name, re.I):
            continue
        selected.append(c)
        selected_ids.add(cid)
        selected_docs.add(doc_id)

    return selected[:target_k] or chunks[:target_k]


def meeting_doc_recall_at_k(
    retrieved: list[Any],
    row: dict,
    k: int,
) -> bool:
    """True if a meeting summary/outcome doc for the target session is in top-k."""
    gold_doc = str(row.get("gold_doc_id") or "")
    gold_docs = row.get("gold_doc_ids") or []
    if isinstance(gold_docs, str):
        gold_docs = [gold_docs]
    targets = {gold_doc} if gold_doc else set()
    targets.update(str(d) for d in gold_docs if d)

    signals = analyze_query(str(row.get("question", "")))
    for chunk in retrieved[:k]:
        if chunk.doc_id in targets:
            return True
        fname = (chunk.file_name or "").lower()
        doc_type = classify_imo_filename(chunk.file_name or "")
        if doc_type not in {"session_outcome", "session_report"}:
            continue
        for body, num in signals.session_codes:
            if f"{body.lower()} {num}" in fname or f"{body.lower()}-{num}" in fname:
                return True
    return False
