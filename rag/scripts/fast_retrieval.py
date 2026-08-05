"""Type-specific slot selection for Fast mode retrieval pool."""
from __future__ import annotations

import re
from typing import Callable

from fast_context import FastEvidence
from fast_question_classifier import FastQuestionType, classify_fast_question_type
from imo_doc_classify import (
    asks_broad_session_outcome,
    classify_imo_filename,
    meeting_outcome_scope,
    meeting_summary_source_tier,
)
from meeting_summary_context import (
    TARGET_SCOPE_WHOLE_SESSION,
    get_summary_context,
    is_penalty_summary_source,
    meeting_summary_source_tier,
    resolve_meeting_summary_context,
)
from rag_answer_lib import RetrievedChunk
from retrieval_query_analysis import analyze_query
from retrieval_search import extract_clause_hints

OUTCOME_TERMS = ("outcome", "summary report", "key outcomes", "adopted", "approved", "resolution")
SUMMARY_TERMS = ("summary", "overview", "executive", "highlight", "outcome", "report")
SCOPE_TERMS = ("scope", "application", "applicable", "적용", "정의", "definition")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower())


def _chunk_key(c: RetrievedChunk) -> str:
    return c.chunk_id or f"{c.doc_id}:{c.page_number}:{c.chunk_type}"


def _scope_rank(c: RetrievedChunk, question: str) -> int:
    if not asks_broad_session_outcome(question):
        return 1
    scope = meeting_outcome_scope(c.file_name or "")
    if scope == "session_final_report":
        return 0
    if scope == "reference_body_outcome":
        return 3
    if scope == "working_group_report":
        return 2
    return 1


def _is_outcome_chunk(c: RetrievedChunk, question: str = "") -> bool:
    fname = (c.file_name or "").lower()
    text = (c.text or "")[:600].lower()
    scope = meeting_outcome_scope(c.file_name or "")
    if asks_broad_session_outcome(question) and scope == "reference_body_outcome":
        return False
    if scope == "session_final_report":
        return True
    dt = classify_imo_filename(c.file_name or "")
    if dt in {"session_outcome", "session_report"} and scope != "reference_body_outcome":
        return True
    return any(t in fname or t in text for t in OUTCOME_TERMS)


def _infer_must_cover(question: str, row: dict) -> list[str]:
    terms: list[str] = list(row.get("must_cover") or [])
    terms.extend(row.get("expected_topics") or [])
    lower = question.lower()
    candidates = [
        ("MASS Code", ("mass code", "mass", "mandatory")),
        ("IGC Code", ("igc code", "igc")),
        ("ammonia", ("ammonia", "암모니아")),
        ("IP Code", ("ip code",)),
        ("GHG", ("ghg", "greenhouse")),
        ("CII", ("cii", "carbon intensity")),
    ]
    for label, keys in candidates:
        if any(k in lower for k in keys) and label not in terms:
            terms.append(label)
    sig = analyze_query(question)
    for body, num in sig.session_codes:
        label = f"{body} {num}"
        if label not in terms:
            terms.append(label)
    return list(dict.fromkeys(t for t in terms if t))


def _pick_first(
    pool: list[RetrievedChunk],
    predicate: Callable[[RetrievedChunk], bool],
    *,
    used: set[str],
    prefer: Callable[[RetrievedChunk], float] | None = None,
) -> RetrievedChunk | None:
    candidates = [c for c in pool if _chunk_key(c) not in used and predicate(c)]
    if not candidates:
        return None
    if prefer:
        candidates.sort(key=prefer)
    chosen = candidates[0]
    used.add(_chunk_key(chosen))
    return chosen


def _term_score(term: str, c: RetrievedChunk) -> float:
    t = _norm(term)
    blob = _norm(f"{c.file_name} {c.text}")
    if t in blob:
        return 0.0
    for tok in t.split():
        if len(tok) > 2 and tok in blob:
            return 0.1
    return 1.0


SUMMARY_TOPIC_SLOT_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mass", ("mass code", "maritime autonomous", "non-mandatory code")),
    ("ghg", ("ghg", "alternative fuel", "new technolog")),
    ("lrit", ("lrit", "long-range identification")),
    ("vdes", ("vdes", "vhf data exchange")),
    ("hormuz", ("strait of hormuz", "hormuz")),
    ("ammonia", ("ammonia fuel", "hydrogen fuel")),
)


def _chunk_has_terms(c: RetrievedChunk, terms: tuple[str, ...]) -> bool:
    blob = _norm(f"{c.file_name} {c.text}")
    return any(t in blob for t in terms)


def _eligible_summary_chunk(c: RetrievedChunk, question: str, row: dict | None = None) -> bool:
    ctx = get_summary_context(question, row)
    if is_penalty_summary_source(c.file_name or "", c.doc_id or "", ctx=ctx):
        return False
    if meeting_summary_source_tier(c.file_name or "", doc_id=c.doc_id or "", ctx=ctx) >= 3:
        return False
    if _summary_chunk_score(c, question, row) < 0.0:
        return False
    return True


def _summary_chunk_score(c: RetrievedChunk, question: str, row: dict | None = None) -> float:
    from meeting_summary_context import get_summary_context, score_summary_claim_text

    ctx = get_summary_context(question, row)
    source = c.file_name or c.doc_id or ""
    score = score_summary_claim_text(f"{c.file_name} {c.text}", source_name=source, doc_id=c.doc_id or "", ctx=ctx)
    tier = meeting_summary_source_tier(c.file_name or "", doc_id=c.doc_id or "", ctx=ctx)
    if tier == 0:
        score += 10.0
    elif tier == 1:
        score += 6.0
    elif tier >= 3:
        score -= 20.0
    scope = meeting_outcome_scope(c.file_name or "")
    if scope == "session_final_report":
        score += 4.0
    elif scope == "reference_body_outcome":
        score -= 15.0
    if is_penalty_summary_source(c.file_name or "", c.doc_id or "", ctx=ctx):
        score -= 25.0
    if ctx.apply_reference_penalties and asks_broad_session_outcome(question) and scope == "reference_body_outcome":
        score -= 20.0
    return score


def select_meeting_outcome_slots(
    pool: list[RetrievedChunk],
    question: str,
    row: dict,
    *,
    for_summary: bool | None = None,
) -> list[FastEvidence]:
    ctx = resolve_meeting_summary_context(question, row)
    summary_mode = for_summary if for_summary is not None else (
        ctx.target_scope == TARGET_SCOPE_WHOLE_SESSION
    )
    used: set[str] = set()
    out: list[FastEvidence] = []
    ranked_pool = sorted(
        pool,
        key=lambda c: (
            _scope_rank(c, question),
            -_summary_chunk_score(c, question, row) if summary_mode else 0.0,
            float(c.distance),
        ),
    )

    if not summary_mode and ctx.preferred_doc_hints:
        for hint in ctx.preferred_doc_hints[:4]:
            if len(out) >= 5:
                break
            chunk = _pick_first(
                ranked_pool,
                lambda c, h=hint: h in _norm(c.file_name) or h in _norm(c.doc_id or ""),
                used=used,
            )
            if chunk:
                out.append(FastEvidence(chunk, f"preferred:{hint}"))
        if out:
            return out[:5]

    if summary_mode:
        for _slot, terms in SUMMARY_TOPIC_SLOT_TERMS:
            if len(out) >= 5:
                break
            chunk = _pick_first(
                ranked_pool,
                lambda c, t=terms: _eligible_summary_chunk(c, question, row) and _chunk_has_terms(c, t),
                used=used,
                prefer=lambda c, q=question, r=row: -_summary_chunk_score(c, q, r),
            )
            if chunk:
                out.append(FastEvidence(chunk, f"meeting_summary:{_slot}"))
        for c in ranked_pool:
            if len(out) >= 5:
                break
            if _chunk_key(c) in used:
                continue
            if not _eligible_summary_chunk(c, question, row):
                continue
            out.append(FastEvidence(c, "meeting_summary"))
            used.add(_chunk_key(c))
        if out:
            return out[:5]

    summary = _pick_first(
        ranked_pool,
        lambda c: _is_outcome_chunk(c, question),
        used=used,
    )
    if summary:
        out.append(FastEvidence(summary, "meeting_summary"))

    for term in _infer_must_cover(question, row)[:3]:
        chunk = _pick_first(
            ranked_pool,
            lambda c, t=term: _term_score(t, c) < 0.5,
            used=used,
            prefer=lambda c, t=term: _term_score(t, c),
        )
        if chunk:
            out.append(FastEvidence(chunk, f"must_cover:{term}"))

    if len(out) < 4:
        extra = _pick_first(
            ranked_pool,
            lambda c: any(t in _norm(c.text) for t in ("adopted", "approved", "decision", "resolution")),
            used=used,
        )
        if extra:
            out.append(FastEvidence(extra, "outcome_decision"))

    return out[:5]


def select_table_slots(pool: list[RetrievedChunk], question: str) -> list[FastEvidence]:
    from table_retrieval import (
        INSPECTION_COLUMN_TERMS,
        ROW_ENTITY_TERMS,
        classify_table_query_mode,
        extract_page_hints,
        infer_age_column_hints,
    )

    used: set[str] = set()
    out: list[FastEvidence] = []
    page_hints = extract_page_hints(question)
    age_hints = infer_age_column_hints(question)

    def is_row(c: RetrievedChunk) -> bool:
        return c.chunk_type == "table_row" or "table_row" in (c.chunk_type or "")

    def is_md(c: RetrievedChunk) -> bool:
        return c.chunk_type == "table_markdown"

    def is_sum(c: RetrievedChunk) -> bool:
        return c.chunk_type == "table_summary"

    def table_score(c: RetrievedChunk) -> float:
        score = 0.0
        if page_hints and c.page_number in page_hints:
            score -= 3.0
        score -= len(c.matched_columns or []) * 1.5
        if c.chunk_type == "table_row":
            score -= 1.0
        blob = c.text or ""
        for term in ROW_ENTITY_TERMS:
            if term in question and term in blob:
                score -= 1.2
        if age_hints:
            for hint in age_hints:
                if hint in blob:
                    score -= 5.0
            if any(t in blob for t in INSPECTION_COLUMN_TERMS) and not any(h in blob for h in age_hints):
                score += 3.0
        elif "선령" in question and "선령" in blob:
            score -= 2.0
        return score

    mode = classify_table_query_mode(question)

    if mode == "table_summary":
        pick_order = (
            (is_sum, "table_summary"),
            (is_md, "table_markdown"),
            (is_row, "table_row_kv"),
        )
    elif mode in {"row_lookup", "column_comparison"}:
        pick_order = (
            (is_row, "table_row_kv"),
            (is_md, "table_markdown"),
            (is_sum, "table_summary"),
        )
    else:
        pick_order = (
            (is_row, "table_row_kv"),
            (is_md, "table_markdown"),
            (is_sum, "table_summary"),
        )

    for predicate, slot_name in pick_order:
        chunk = _pick_first(pool, predicate, used=used, prefer=table_score)
        if chunk:
            out.append(FastEvidence(chunk, slot_name))

    if not out:
        for c in pool[:3]:
            if _chunk_key(c) not in used:
                out.append(FastEvidence(c, "table_fallback"))
                used.add(_chunk_key(c))
    return out[:4]


def select_rule_slots(
    pool: list[RetrievedChunk],
    question: str,
    row: dict | None = None,
) -> list[FastEvidence]:
    from retrieval_query_analysis import detect_class_society_hint
    from rag_society_filter import filter_pool_for_society, society_hard_filter_enabled

    row = row or {}
    society = str(row.get("class_society_hint") or detect_class_society_hint(question))
    if society:
        filtered, had = filter_pool_for_society(
            pool, society, hard=society_hard_filter_enabled(row) or bool(row.get("_rule_guidance_lookup"))
        )
        if had:
            pool = filtered
        elif row.get("_hard_society_filter") or row.get("_rule_guidance_lookup"):
            return []
    used: set[str] = set()
    out: list[FastEvidence] = []
    hints = extract_clause_hints(question)

    def clause_match(c: RetrievedChunk) -> bool:
        if hints and c.clause_number in hints:
            return True
        return any(h in (c.text or "") for h in hints)

    exact = _pick_first(pool, clause_match if hints else lambda c: bool(c.clause_number), used=used)
    if exact:
        out.append(FastEvidence(exact, "exact_clause"))

    if exact and exact.page_number is not None:
        doc = exact.doc_id
        pg = exact.page_number
        for delta in (-1, 1):
            adj = _pick_first(
                pool,
                lambda c, d=delta: c.doc_id == doc and c.page_number == pg + d,
                used=used,
            )
            if adj:
                out.append(FastEvidence(adj, "adjacent_clause"))
                break

    scope = _pick_first(
        pool,
        lambda c: any(t in _norm(c.text) for t in SCOPE_TERMS),
        used=used,
    )
    if scope:
        out.append(FastEvidence(scope, "scope_definition"))

    if not out:
        for c in pool[:3]:
            if _chunk_key(c) not in used:
                out.append(FastEvidence(c, "rule_fallback"))
                used.add(_chunk_key(c))
    return out[:4]


def select_broad_summary_slots(pool: list[RetrievedChunk]) -> list[FastEvidence]:
    used: set[str] = set()
    seen_docs: set[str] = set()
    out: list[FastEvidence] = []

    def summary_score(c: RetrievedChunk) -> float:
        blob = _norm(f"{c.file_name} {c.text}")
        score = 0.0
        for t in SUMMARY_TERMS:
            if t in blob:
                score -= 1.0
        if c.doc_id in seen_docs:
            score += 2.0
        return score

    ranked = sorted(pool, key=summary_score)
    for c in ranked:
        if len(out) >= 5:
            break
        if _chunk_key(c) in used:
            continue
        if c.doc_id in seen_docs and len(seen_docs) >= 5:
            continue
        out.append(FastEvidence(c, f"doc_summary:{len(seen_docs)+1}"))
        used.add(_chunk_key(c))
        seen_docs.add(c.doc_id)
    return out[:5]


def select_figure_slots(pool: list[RetrievedChunk]) -> list[FastEvidence]:
    used: set[str] = set()
    out: list[FastEvidence] = []

    fig = _pick_first(
        pool,
        lambda c: c.element_type in {"figure", "picture"}
        or "[figure]" in (c.text or "").lower()
        or bool(c.caption),
        used=used,
    )
    if fig:
        out.append(FastEvidence(fig, "figure_caption"))

    for c in pool:
        if len(out) >= 3:
            break
        if _chunk_key(c) in used:
            continue
        if c.page_number == (fig.page_number if fig else None) and fig and c.doc_id == fig.doc_id:
            out.append(FastEvidence(c, "figure_context"))
            used.add(_chunk_key(c))
    if not out:
        for c in pool[:2]:
            out.append(FastEvidence(c, "figure_fallback"))
    return out[:3]


def select_general_slots(pool: list[RetrievedChunk], *, max_chunks: int = 3, max_docs: int = 2) -> list[FastEvidence]:
    used: set[str] = set()
    seen_docs: set[str] = set()
    out: list[FastEvidence] = []
    for c in pool:
        if len(out) >= max_chunks:
            break
        if c.doc_id not in seen_docs and len(seen_docs) >= max_docs:
            continue
        if _chunk_key(c) in used:
            continue
        out.append(FastEvidence(c, "general"))
        used.add(_chunk_key(c))
        seen_docs.add(c.doc_id)
    return out


def select_fast_evidence_slots(
    pool: list[RetrievedChunk],
    question: str,
    row: dict | None = None,
    *,
    fast_type: FastQuestionType | None = None,
) -> list[FastEvidence]:
    row = row or {}
    qtype = fast_type or classify_fast_question_type(question, row)
    if qtype == "meeting_summary":
        ctx = resolve_meeting_summary_context(question, row)
        if ctx.target_scope == TARGET_SCOPE_WHOLE_SESSION:
            return select_meeting_outcome_slots(pool, question, row, for_summary=True)
    if qtype == "meeting_outcome_question":
        return select_meeting_outcome_slots(pool, question, row, for_summary=False)
    if qtype == "table_question":
        return select_table_slots(pool, question)
    if qtype == "rule_question":
        return select_rule_slots(pool, question, row)
    if qtype == "broad_summary_question":
        return select_broad_summary_slots(pool)
    if qtype == "figure_or_diagram_question":
        return select_figure_slots(pool)
    return select_general_slots(pool)


def evidence_to_chunks(evidence: list[FastEvidence]) -> list[RetrievedChunk]:
    return [ev.chunk for ev in evidence]
