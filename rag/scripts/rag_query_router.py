"""Central query routing: category, society, answer_mode, retrieval profile."""
from __future__ import annotations

import re
from pathlib import Path
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
    detect_named_sources,
    detect_meeting_source_hint,
    detect_table_source_hint,
)
from retrieval_question_profile import build_retrieval_profile
from compound_regulatory import (
    is_compound_regulatory_class_question,
    requested_class_sources,
)

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

_EXPLICIT_RULE_DOCUMENT_RE = re.compile(
    r"\b(?:DNV|ABS|LR|KR)\s*[-–/]\s*(?:CG|CP|RP|RU|OS|SI|NV)\s*[-–/]?\s*[A-Z0-9.-]+",
    re.I,
)
_DOCUMENT_DISCOVERY_RE = re.compile(
    r"찾아|검색|문서\s*목록|관련.{0,20}(?:rule|rules|guidance|guide|규칙|지침)|"
    r"(?:CG|Rule|Guidance)\s*명칭",
    re.I,
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
    # A named rule document plus a clause/fact question must use the general
    # evidence answerer inside that document.  The document-card route is for
    # discovery and otherwise replaces exact PRA/TA answers with a title list.
    if _EXPLICIT_RULE_DOCUMENT_RE.search(question or "") and not _DOCUMENT_DISCOVERY_RE.search(
        question or ""
    ):
        return False
    # A question that asks to reconcile an IMO meeting outcome with class
    # requirements is not a society-only Rule lookup.  Several downstream
    # callers invoke this helper again after central routing, so keeping the
    # exception only in ``resolve_pipeline_route`` lets those callers silently
    # put the request back on the narrow Rule path.
    if row.get("_compound_regulatory_class") or is_compound_regulatory_class_question(
        question
    ):
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
    # Environmental/meeting questions often contain generic words such as
    # "요건" or "요구사항".  Those words describe the requested content; they
    # do not turn an IMO CII/GHG summary into a class-Rule lookup.
    if cat in {"trend_summary", "meeting_outcome", "env_regulation", "autonomous"}:
        return False
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
            "detected_sources": [society] if society else [],
            "excluded_sources": list(signals.excluded_sources),
            "constrained_sources": list(signals.constrained_sources),
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
    signals = analyze_query(question)
    detected_sources = list(signals.named_sources or detect_named_sources(question))
    if not detected_sources:
        row_sources = [
            str(source).upper()
            for source in (row.get("retrieval_sources") or [])
            if str(source).upper() not in set(signals.excluded_sources)
        ]
        detected_sources = list(dict.fromkeys(row_sources))
    if not detected_sources and cat == "env_regulation" and signals.topics.intersection(
        {"cii", "ghg", "marpol", "alt_fuel"}
    ):
        detected_sources = ["MEPC"]
    elif not detected_sources and cat == "autonomous" and "mass" in signals.topics:
        detected_sources = ["MSC"]
    compound_regulatory_class = is_compound_regulatory_class_question(question)
    if compound_regulatory_class:
        meeting = detect_meeting_source_hint(question)
        meeting_sources = [meeting] if meeting else [
            source for source in detected_sources if source in {"MSC", "MEPC", "IMO"}
        ]
        detected_sources = list(
            dict.fromkeys([*meeting_sources, *requested_class_sources(question)])
        )
    society = detected_sources[0] if len(detected_sources) == 1 else ""
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
    if compound_regulatory_class:
        # This is neither a meeting-only template nor a society-only lookup.
        # The answer needs independently grounded evidence from both lanes.
        rule_guidance = False
        answer_mode = "compound_regulatory_class"
        retrieval_profile = "compound_regulatory_class"
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
        "detected_sources": detected_sources,
        "excluded_sources": list(signals.excluded_sources),
        "constrained_sources": list(signals.constrained_sources),
        "detected_doc_type": (
            "meeting_and_class_rules"
            if compound_regulatory_class
            else ("rule_guidance" if rule_guidance else cat)
        ),
        "hard_society_filter": bool(society and rule_guidance),
        "expanded_keywords": list(signals.expanded_terms or []),
        "rule_guidance_lookup": rule_guidance,
        "compound_regulatory_class": compound_regulatory_class,
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
    out["_excluded_sources"] = list(route.get("excluded_sources") or [])
    out["_constrained_sources"] = list(route.get("constrained_sources") or [])
    if route.get("compound_regulatory_class"):
        out["_compound_regulatory_class"] = True
    else:
        out.pop("_compound_regulatory_class", None)
    if route["selected_answer_mode"] == "table_qa":
        out["_table_qa"] = True
        out.pop("_rule_guidance_lookup", None)
        out.pop("_hard_society_filter", None)
    if route["detected_sources"]:
        out["retrieval_sources"] = list(route["detected_sources"])
    elif route["excluded_sources"] and out.get("retrieval_sources"):
        excluded = {str(source).upper() for source in route["excluded_sources"]}
        out["retrieval_sources"] = [
            source
            for source in out["retrieval_sources"]
            if str(source).upper() not in excluded
        ]
    if route["detected_society"]:
        out["class_society_hint"] = route["detected_society"]
        # Replace (do not merge) so a prior KR default cannot block MEPC/MSC.
    elif len(route["detected_sources"]) > 1:
        out.pop("class_society_hint", None)
    if route["rule_guidance_lookup"]:
        out["_hard_society_filter"] = route["hard_society_filter"]
        out["_rule_guidance_lookup"] = True
    elif route.get("constrained_sources") and route.get("detected_society"):
        # ``ABS만`` is a user constraint even when the question is a factual
        # clause lookup rather than a Rule/Guidance discovery request.
        out["_hard_society_filter"] = True

    # Keep the four user-facing categories, but expose the orthogonal intent,
    # answer-shape, time and authority axes to retrieval and generation.
    from question_profile import build_question_profile

    profile = build_question_profile(str(out.get("question") or ""), out)
    out["_question_profile"] = profile.to_dict()

    # The annual business scope prioritises DNV/KR when the user asks broadly
    # for applicable Rules without naming a society.  Explicit ABS/LR/DNV/KR
    # requests and exclusions above remain hard constraints.
    if (
        profile.answer_style == "document_cards"
        and not route.get("detected_sources")
        and not route.get("excluded_sources")
        and not re.search(
            r"아니라|제외|빼고|말고|대신|not\s+the|except|exclude",
            str(out.get("question") or ""),
            re.I,
        )
    ):
        out["retrieval_sources"] = ["DNV", "KR"]
        out["_preferred_default_rule_sources"] = ["DNV", "KR"]
        out.pop("class_society_hint", None)

    profile_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
        / "index"
        / "unified_full_corpus_715_v1"
        / "document_profiles_v1.json"
    )
    if profile.time_scope in {"session_range", "latest_available"}:
        from corpus_coverage_guard import build_coverage_guard

        out["_coverage_guard"] = build_coverage_guard(
            profile, document_profile_path=profile_path
        )
    return out
