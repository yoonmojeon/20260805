"""Category/intent-specific chunk scoring adjustments and filters."""
from __future__ import annotations

import re
from typing import Any

from meeting_category_profile import MeetingRetrievalProfile
from meeting_topic_cluster import _topic_id_for_text

MASS_RE = re.compile(r"mass\s*code|maritime autonomous|autonomous surface ship", re.I)
AROS_RE = re.compile(r"\baros\b|remotely operated", re.I)
ALT_FUEL_RE = re.compile(
    r"alternative fuel|low-flashpoint|ammonia|hydrogen|methanol|\blng\b|ghg safety|fuel safety|new technolog",
    re.I,
)
MEPC_GHG_RE = re.compile(r"mepc|net-zero|net zero|\bcii\b|seemp|gfi|marpol annex", re.I)
MSC111_RE = re.compile(r"msc\s*111|111th session", re.I)
TIMELINE_RE = re.compile(
    r"experience-building|mandatory code|entry into force|timeline|work plan|roadmap|ebp",
    re.I,
)
NON_MAND_RE = re.compile(r"non-mandatory|non mandatory|voluntary", re.I)
OUTCOME_RE = re.compile(r"\b(adopted|approved|agreed|endorsed|decided|finalized)\b", re.I)
GUIDELINE_RE = re.compile(r"interim guideline|safety guideline|guidelines approved", re.I)
LRIT_VDES_RE = re.compile(r"\blrit\b|\bvdes\b|gmdss|solas|long-range identification", re.I)
HORMUZ_RE = re.compile(r"hormuz|strait of", re.I)


def chunk_text(chunk: Any) -> str:
    return str(getattr(chunk, "text", "") or "")


def intent_chunk_adjustment(text: str, *, internal_intent: str) -> float:
    low = text.lower()
    adj = 0.0
    has_mass = bool(MASS_RE.search(low))
    has_alt = bool(ALT_FUEL_RE.search(low))
    has_lrit = bool(LRIT_VDES_RE.search(low))
    has_hormuz = bool(HORMUZ_RE.search(low))
    has_mepc = bool(MEPC_GHG_RE.search(low))
    has_msc = bool(MSC111_RE.search(low))
    has_timeline = bool(TIMELINE_RE.search(low))
    has_non_mand = bool(NON_MAND_RE.search(low))
    has_outcome = bool(OUTCOME_RE.search(low))
    has_guideline = bool(GUIDELINE_RE.search(low))

    if internal_intent == "meeting_outcome":
        if has_mass:
            adj += 0.6
        if has_alt:
            adj += 1.8
        if has_lrit:
            adj += 2.0
        if has_hormuz:
            adj += 1.8
        if has_outcome:
            adj += 0.6

    elif internal_intent == "altfuel_ghg_safety":
        if has_mass or AROS_RE.search(low):
            adj -= 4.0
        if has_alt:
            adj += 3.5
        if has_guideline:
            adj += 1.5
        if has_outcome:
            adj += 1.0
        if has_mepc and not has_alt:
            adj -= 1.5

    elif internal_intent == "mass_code_timeline":
        if has_mass:
            adj += 3.0
        if has_non_mand:
            adj += 1.5
        if has_timeline:
            adj += 2.0
        if has_alt and not has_mass:
            adj -= 2.5
        if has_mepc:
            adj -= 2.0
        if has_lrit and not has_mass:
            adj -= 1.5

    elif internal_intent in ("trend_summary", "env_regulation"):
        if has_mass or (has_msc and not has_mepc):
            adj -= 3.0
        if has_mepc:
            adj += 1.5
        if has_alt and internal_intent == "env_regulation":
            adj += 0.8

    return adj


def is_excluded_chunk(chunk: Any, *, profile: MeetingRetrievalProfile) -> bool:
    text = chunk_text(chunk)
    low = text.lower()
    intent = profile.internal_intent

    if intent == "altfuel_ghg_safety":
        if MASS_RE.search(low) or AROS_RE.search(low):
            return True
        if _topic_id_for_text(text) == "mass_code":
            return True

    if intent in ("trend_summary", "env_regulation") and profile.top_level_category != "autonomous_mass":
        src = str(getattr(chunk, "source", "") or "").upper()
        fn = str(getattr(chunk, "file_name", "") or "").lower()
        if MASS_RE.search(low) and "mepc" not in fn and src == "MSC":
            return True
        if intent == "env_regulation" and src == "MSC" and "mepc" not in fn:
            return True

    return False


def topic_caps_for_intent(internal_intent: str) -> dict[str, int]:
    if internal_intent == "meeting_outcome":
        return {"mass_code": 1}
    return {}


def exclude_topics_for_intent(internal_intent: str) -> set[str]:
    if internal_intent == "altfuel_ghg_safety":
        return {"mass_code"}
    if internal_intent == "mass_code_timeline":
        return {"ghg_framework", "cii_reporting"}
    return set()
