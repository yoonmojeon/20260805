"""Query expansion for meeting / regulation categories (dense + BM25)."""
from __future__ import annotations

from retrieval_query_analysis import analyze_query
from bm25_index import tokenize_for_bm25

ENV_REGULATION_TERMS = [
    "GHG",
    "greenhouse gas",
    "carbon intensity",
    "CII",
    "EEXI",
    "MARPOL Annex VI",
    "net-zero",
    "fuel intensity",
    "GFI",
    "lifecycle assessment",
    "LCA",
    "reporting",
    "verification",
    "compliance",
    "data collection",
    "DCS",
    "SEEMP",
    "onboard management",
    "operational measure",
    "energy efficiency",
    "fleet report",
    "emissions data",
]

ALT_FUEL_SAFETY_TERMS = [
    "alternative fuel",
    "low-flashpoint fuel",
    "ammonia",
    "hydrogen",
    "methanol",
    "LNG",
    "fuel cell",
    "battery",
    "GHG safety",
    "safety regulation",
    "interim guidelines",
    "risk assessment",
    "fire safety",
    "explosion risk",
    "training",
    "crew competence",
]

AUTONOMOUS_MASS_TERMS = [
    "MASS Code",
    "Maritime Autonomous Surface Ships",
    "autonomous surface ships",
    "non-mandatory",
    "mandatory",
    "mandatory code",
    "experience-building phase",
    "entry into force",
    "implementation",
    "timeline",
    "roadmap",
    "MSC 111",
    "adopted",
    "approved",
    "finalized",
    "safety code",
    "goal-based",
    "future work",
    "work plan",
]

TREND_TERMS = [
    "outcome",
    "adopted",
    "approved",
    "executive summary",
    "key outcomes",
    "meeting highlights",
    "final report",
    "resolution",
    "decision",
]


def enrich_query_for_meeting(
    question: str,
    *,
    top_level_category: str,
    internal_intent: str = "",
) -> tuple[str, list[str]]:
    signals = analyze_query(question)
    parts: list[str] = [question]
    if signals.expanded_terms:
        parts.extend(signals.expanded_terms[:14])
    ql = question.lower()
    intent = internal_intent or ""

    if intent == "meeting_outcome":
        parts.extend(TREND_TERMS)
        parts.extend(
            [
                "LRIT",
                "VDES",
                "GMDSS",
                "SOLAS",
                "alternative fuel",
                "GHG safety",
                "ammonia",
                "hydrogen",
                "Strait of Hormuz",
                "maritime safety",
            ]
        )
    elif intent == "altfuel_ghg_safety":
        parts.extend(ALT_FUEL_SAFETY_TERMS)
        parts.extend(["MSC 111", "ISE", "discussion", "conclusion", "approved", "agreed"])
    elif intent == "mass_code_timeline":
        parts.extend(AUTONOMOUS_MASS_TERMS)
    elif top_level_category == "latest_trend_summary":
        parts.extend(TREND_TERMS)
        if "mepc" in ql:
            parts.extend(["MEPC", "GHG", "emissions", "climate", "IMO Net-Zero Framework"])
        if "msc" in ql:
            parts.extend(["MSC", "maritime safety committee", "session report"])

    elif top_level_category == "env_regulation_response":
        parts.extend(ENV_REGULATION_TERMS)
        if any(k in ql or k in question for k in ("대체연료", "연료", "fuel", "ammonia", "hydrogen")):
            parts.extend(ALT_FUEL_SAFETY_TERMS)
        if "ghg" in ql or "안전" in question:
            parts.extend(["GHG safety", "alternative fuel safety"])

    elif top_level_category == "autonomous_mass":
        parts.extend(AUTONOMOUS_MASS_TERMS)

    for body, num in signals.session_codes:
        parts.extend([f"{body} {num}", f"{body}-{num}"])

    enriched = " ".join(dict.fromkeys(p.strip() for p in parts if p and p.strip()))
    return enriched, tokenize_for_bm25(enriched)
