"""Stable, multi-axis question profile shared by retrieval and generation.

The UI exposes four business categories, but a retrieval decision also needs to
know whether the request is a bounded fact, a document guide, a timeline, or an
impact brief.  This module adds those orthogonal axes without replacing the
existing category classifier.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from question_requirements import analyze_requirements


SESSION_RANGE_RE = re.compile(
    r"\b(MSC|MEPC)\s*(\d{2,3})\s*(?:[-~～]|부터|에서)\s*(?:\1\s*)?(\d{2,3})\b",
    re.I,
)
LATEST_RE = re.compile(r"최신|최근|현재|latest|recent|current", re.I)
TIMELINE_RE = re.compile(r"일정|로드맵|회차별|흐름|발효|시행|timeline|road\s*map", re.I)
GUIDE_RE = re.compile(
    r"찾아(?:줘|주세요)|목록|어떤\s*(?:rule|guidance|규칙|지침)|"
    r"관련\s*(?:rule|guidance|규칙|지침)|rule\s*/?\s*guidance|"
    r"(?:문서(?:명)?|rule|guidance|requirements?).{0,35}(?:알려|정리|검색)|"
    r"(?:알려|정리|검색).{0,35}(?:문서(?:명)?|rule|guidance|requirements?)|"
    r"명칭(?:을|를)?\s*알려",
    re.I,
)
EXACT_FACT_RE = re.compile(
    r"얼마|몇\s*(?:개|건|시간|일|톤|배|mm|m|kV|V|%)|어느\s*(?:값|위치|조항|경우)|"
    r"어떤\s*(?:정격|조건|시험\s*절차)|정의|뜻|의미|생략할\s*수|"
    r"how\s+(?:many|much|long)|what\s+(?:rating|value|condition)",
    re.I,
)
FINAL_AUTHORITY_RE = re.compile(
    r"결정|결론|채택|승인|확정|발효|시행|최종|outcome|decision|adopt|approve|final",
    re.I,
)
PREMISE_VERIFICATION_RE = re.compile(
    r"(?:전제가\s*맞는지|전제(?:를|가)?\s*검증|사실인지\s*검증|"
    r"틀리면\s*(?:문서\s*)?근거로\s*바로잡)",
    re.I,
)
SPECIFIC_DOCUMENT_LOOKUP_RE = re.compile(
    r"(?:[A-Z]{2,}(?:-[A-Z0-9]+)+|Notice\s+No\.?\s*\d+|"
    r"Guide\s+for\s+.{2,80}|Requirements\s+for\s+.{2,80}|"
    r"Section\s*\d+|지정\s*문서).{0,120}?(?:에서|내에서)\s*"
    r".{1,100}?(?:찾아|확인).{0,60}?(?:근거|알려)",
    re.I,
)


@dataclass(frozen=True)
class QuestionProfile:
    primary_intent: str
    answer_style: str
    time_scope: str
    authority_need: str
    requested_facets: tuple[str, ...]
    requested_sources: tuple[str, ...]
    session_range: tuple[str, int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_intent": self.primary_intent,
            "answer_style": self.answer_style,
            "time_scope": self.time_scope,
            "authority_need": self.authority_need,
            "requested_facets": list(self.requested_facets),
            "requested_sources": list(self.requested_sources),
            "session_range": list(self.session_range) if self.session_range else None,
        }


def build_question_profile(question: str, row: dict | None = None) -> QuestionProfile:
    row = row or {}
    q = re.sub(r"\s+", " ", str(question or "")).strip()
    requirements = analyze_requirements(q, row)
    category = str(row.get("category") or "")
    top_level = str(row.get("_top_level_category") or "")
    range_match = SESSION_RANGE_RE.search(q)
    session_range = (
        (
            range_match.group(1).upper(),
            min(int(range_match.group(2)), int(range_match.group(3))),
            max(int(range_match.group(2)), int(range_match.group(3))),
        )
        if range_match
        else None
    )

    # This is the internal fifth intent requested by the feedback: exact
    # clause/value questions remain short even though they live under Rule QA.
    bounded_fact = bool(
        EXACT_FACT_RE.search(q)
        or {"value", "definition", "clause"}.intersection(requirements.facets)
        or SPECIFIC_DOCUMENT_LOOKUP_RE.search(q)
    ) and not requirements.broad_summary
    guide = bool(
        category == "rule_lookup"
        or top_level == "rule_guidance_lookup"
    ) and bool(GUIDE_RE.search(q)) and not bounded_fact

    if PREMISE_VERIFICATION_RE.search(q):
        intent, style = "premise_verification", "short_fact"
    elif bounded_fact:
        intent, style = "exact_fact", "short_fact"
    elif guide:
        intent, style = "rule_document_guide", "document_cards"
    elif category == "autonomous" or top_level == "autonomous_mass":
        intent = "autonomous_mass"
        style = "timeline_brief" if TIMELINE_RE.search(q) else "four_section_brief"
    elif category == "env_regulation" or top_level == "env_regulation_response":
        intent, style = "regulatory_impact", "four_section_brief"
    elif TIMELINE_RE.search(q) or session_range:
        intent, style = "timeline_summary", "timeline_brief"
    else:
        intent, style = "trend_summary", "summary_brief"

    if session_range:
        time_scope = "session_range"
    elif requirements.session_number:
        time_scope = "single_session"
    elif LATEST_RE.search(q):
        time_scope = "latest_available"
    else:
        time_scope = "unspecified"

    sources = tuple(
        dict.fromkeys(
            str(value).upper()
            for value in (
                [requirements.organization] if requirements.organization else []
            )
            + list(row.get("retrieval_sources") or [])
            if str(value).strip()
        )
    )
    return QuestionProfile(
        primary_intent=intent,
        answer_style=style,
        time_scope=time_scope,
        authority_need="final_outcome" if FINAL_AUTHORITY_RE.search(q) else "informational",
        requested_facets=tuple(requirements.facets),
        requested_sources=sources,
        session_range=session_range,
    )
