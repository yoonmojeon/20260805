"""Central query routing: category, society, answer_mode, retrieval profile."""
from __future__ import annotations

import re
from typing import Any

from meeting_category_profile import (
    TOP_LEVEL_RULE,
    TOP_LEVEL_LABELS_KO,
    build_meeting_retrieval_profile,
    resolve_top_level_category,
)
from question_classifier import CATEGORY_LABELS_KO, classify_question_category, category_label_ko
from retrieval_query_analysis import (
    analyze_query,
    detect_class_society_hint,
    detect_table_source_hint,
)
from retrieval_question_profile import build_retrieval_profile

RULE_GUIDANCE_TERMS = (
    r"\brule\b",
    r"\brules\b",
    r"\bregulation\b",
    r"\bguidance\b",
    r"\bguideline\b",
    r"class\s*rule",
    r"선급\s*규칙",
    r"규정",
    r"지침",
    r"요건",
    r"요구사항",
    r"조항",
    r"\bclause\b",
    r"\bpart\b",
    r"\bchapter\b",
    r"\bsection\b",
    r"rule/guidance",
    r"notice\s*no",
    r"cg-\d+",
)


def _question_has_rule_guidance_terms(question: str) -> bool:
    q = (question or "").lower()
    return any(re.search(p, q, re.I) for p in RULE_GUIDANCE_TERMS)


def is_rule_guidance_lookup(
    question: str,
    row: dict | None = None,
    *,
    category: str = "",
    top_level: str = "",
    internal_intent: str = "",
) -> bool:
    row = row or {}
    if row.get("_table_qa") or str(row.get("category") or "") == "table_qa":
        return False
    cat = category or str(row.get("category") or "")
    if not cat:
        cat = classify_question_category(question, row)
    top = top_level or resolve_top_level_category(cat)
    intent = internal_intent or str(row.get("internal_intent") or "")
    if not intent:
        mprof = build_meeting_retrieval_profile(question, {**row, "category": cat}, legacy_category=cat)
        intent = mprof.internal_intent
    if top == TOP_LEVEL_RULE:
        return True
    if intent == "rule_lookup" or cat == "rule_lookup":
        return True
    if TOP_LEVEL_LABELS_KO.get(top) == "Rule/Guidance 조회":
        return True
    return _question_has_rule_guidance_terms(question)


def resolve_pipeline_route(
    question: str,
    row: dict | None = None,
    *,
    latency_mode: str = "accurate",
) -> dict[str, Any]:
    row = dict(row or {})
    if row.get("_table_qa") or str(row.get("category") or "") == "table_qa":
        # Meeting acronyms (MEPC/MSC) must win over the KR corpus default;
        # otherwise table dense search is locked to class-rule PDFs.
        society = str(
            row.get("class_society_hint")
            or detect_table_source_hint(question, default_society="KR")
        )
        signals = analyze_query(question)
        return {
            "latency_mode": latency_mode,
            "question_category": "table_qa",
            "question_category_label": "표 질의응답",
            "top_level_category": "table_qa",
            "top_level_label": "표 질의응답",
            "internal_intent": "table_qa",
            "selected_retrieval_profile": "table_schema_two_stage",
            "selected_retrieval_label": "표 스키마→행·셀 조회",
            "selected_answer_mode": "table_qa",
            "detected_society": society,
            "detected_doc_type": "table",
            "hard_society_filter": False,
            "expanded_keywords": list(signals.expanded_terms or []),
            "rule_guidance_lookup": False,
            "meeting_retrieval_profile_id": "table_qa",
        }
    cat = str(row.get("category") or "").strip()
    if cat not in CATEGORY_LABELS_KO:
        cat = classify_question_category(question, row)
    work = {**row, "category": cat}
    rprof = build_retrieval_profile(question, work)
    mprof = build_meeting_retrieval_profile(question, work, legacy_category=cat)
    top = resolve_top_level_category(cat)
    society = str(row.get("class_society_hint") or detect_class_society_hint(question))
    signals = analyze_query(question)
    rule_guidance = is_rule_guidance_lookup(
        question,
        work,
        category=cat,
        top_level=top,
        internal_intent=mprof.internal_intent,
    )
    answer_mode = rprof.answer_mode
    retrieval_profile = rprof.profile_id
    if rule_guidance:
        answer_mode = "rule_guidance_lookup"
        retrieval_profile = "rule_guidance_lookup"
    return {
        "latency_mode": latency_mode,
        "question_category": cat,
        "question_category_label": category_label_ko(cat),
        "top_level_category": top,
        "top_level_label": TOP_LEVEL_LABELS_KO.get(top, top),
        "internal_intent": mprof.internal_intent,
        "selected_retrieval_profile": retrieval_profile,
        "selected_retrieval_label": rprof.label_ko,
        "selected_answer_mode": answer_mode,
        "detected_society": society,
        "detected_doc_type": "rule_guidance" if rule_guidance else cat,
        "hard_society_filter": bool(society and rule_guidance),
        "expanded_keywords": list(signals.expanded_terms or []),
        "rule_guidance_lookup": rule_guidance,
        "meeting_retrieval_profile_id": mprof.profile_id,
    }


def enrich_row_for_routing(row: dict, *, latency_mode: str = "accurate") -> dict:
    """Apply routing fields onto row before search/answer."""
    out = dict(row)
    route = resolve_pipeline_route(str(out.get("question") or ""), out, latency_mode=latency_mode)
    out["category"] = route["question_category"]
    out["_top_level_category"] = route["top_level_category"]
    out["_internal_intent"] = route["internal_intent"]
    out["_pipeline_route"] = route
    if route["selected_answer_mode"] == "table_qa":
        out["_table_qa"] = True
        out.pop("_rule_guidance_lookup", None)
        out.pop("_hard_society_filter", None)
    if route["detected_society"]:
        out["class_society_hint"] = route["detected_society"]
        # Replace (do not merge) so a prior KR default cannot block MEPC/MSC.
        out["retrieval_sources"] = [route["detected_society"]]
    if route["rule_guidance_lookup"]:
        out["_hard_society_filter"] = route["hard_society_filter"]
        out["_rule_guidance_lookup"] = True
    return out
