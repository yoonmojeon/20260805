"""Source tier classification for IMO meeting / regulation documents."""
from __future__ import annotations

import re
from typing import Any

TIER0_PATTERNS = (
    r"meeting\s+highlights",
    r"official\s+summary",
    r"outcome\s+summary",
    r"executive\s+summary",
    r"key\s+outcomes?",
    r"closing\s+remarks",
    r"final\s+report",
    r"draft\s+report",
    r"session\s+report",
    r"summary\s+report",
)
TIER1_PATTERNS = (
    r"committee\s+report",
    r"sub.?committee",
    r"working\s+group\s+report",
    r"report\s+of\s+the",
    r"intersessional",
    r"iswg",
    r"wp\.?\s*1",
)
TIER2_PATTERNS = (
    r"working\s+paper",
    r"agenda\s+item",
    r"submission",
    r"information\s+paper",
    r"inf\.\s*\d",
    r"wp\.?\s*\d",
    r"stow",
)
TIER3_PATTERNS = (
    r"background",
    r"circular",
    r"outcome\s+of\s+(?:tc|c|a)\s*\d",
    r"strategic\s+plan",
    r"annotations",
)

OUTCOME_SIGNALS = (
    "adopted",
    "approved",
    "agreed",
    "endorsed",
    "decided",
    "finalized",
    "noted",
    "invited",
    "requested",
    "established",
    "entry into force",
    "implementation",
    "mandatory",
    "non-mandatory",
    "work plan",
    "timeline",
    "guidelines approved",
    "resolution",
    "conclusion",
)

IMPACT_SIGNALS = (
    "ships are required",
    "reporting requirement",
    "compliance",
    "verification",
    "safety assessment",
    "risk assessment",
    "operational impact",
    "fuel consumption",
    "emissions data",
    "training",
    "survey",
    "certification",
    "monitoring",
    "seemp",
    "data collection",
)


def _blob(chunk: Any) -> str:
    parts = [
        str(getattr(chunk, "file_name", "") or ""),
        str(getattr(chunk, "caption", "") or ""),
        str(getattr(chunk, "text", "") or "")[:800],
    ]
    return " ".join(parts).lower()


def classify_source_tier(chunk: Any) -> int:
    blob = _blob(chunk)
    for pat in TIER0_PATTERNS:
        if re.search(pat, blob, re.I):
            return 0
    for pat in TIER1_PATTERNS:
        if re.search(pat, blob, re.I):
            return 1
    for pat in TIER2_PATTERNS:
        if re.search(pat, blob, re.I):
            return 2
    for pat in TIER3_PATTERNS:
        if re.search(pat, blob, re.I):
            return 3
    return 1


def tier_boost(tier: int) -> float:
    return {0: 0.35, 1: 0.18, 2: -0.08, 3: -0.22}.get(tier, 0.0)


def count_outcome_signals(text: str) -> int:
    low = (text or "").lower()
    return sum(1 for s in OUTCOME_SIGNALS if s in low)


def count_impact_signals(text: str) -> int:
    low = (text or "").lower()
    return sum(1 for s in IMPACT_SIGNALS if s in low)


def outcome_boost(text: str) -> float:
    return min(0.4, count_outcome_signals(text) * 0.06)


def impact_boost(text: str) -> float:
    return min(0.35, count_impact_signals(text) * 0.07)


def tier_label(tier: int) -> str:
    return {0: "Tier0", 1: "Tier1", 2: "Tier2", 3: "Tier3"}.get(tier, "Tier1")
