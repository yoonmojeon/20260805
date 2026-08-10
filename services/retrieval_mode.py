"""RAG retrieval mode: TEXT / TABLE / BOTH (rule-based, no extra LLM)."""
from __future__ import annotations

import re
from enum import Enum


class RetrievalMode(str, Enum):
    TEXT = "text"
    TABLE = "table"
    BOTH = "both"


# Explicit table / cell / interval cues
_TABLE_PATTERNS = [
    r"표\s*(에서|에|의|기준|내용|알려|정리|요약)?",
    r"표에\s*(나온|있는|관련)",
    r"검사\s*주기|검사주기|정기검사\s*표|선령별",
    r"선령\s*\d+|선령\s*(미만|초과|이상|이하)",
    r"평형수\s*탱크|밸러스트\s*탱크",
    r"개방검사|두께계측|의심지역",
    r"(?:intermediate|annual|special|docking|continuous|class)\s*survey",
    r"survey\s*interval|tank\s*inspection",
    r"열\s*\d+|row\s*\d+|cell",
]

# Prose / regulation-explanation cues (prefer text corpus)
_TEXT_PROSE_PATTERNS = [
    r"취지|목적|정의|의미|scope|요건\s*설명|requirement",
    r"회의\s*(결과|주요|결정|논의|요약)",
    r"주요\s*(결과|안건|내용|결정)",
    r"논의\s*(요약|내용|결과)",
    r"규정\s*(취지|목적|배경|설명)",
    r"근거\s*조항|조항\s*설명|circular|resolution",
    r"MEPC|MSC|MASS\s*Code|GHG\s*Strategy",
]

# When both sides appear, force BOTH
_BOTH_BRIDGE_PATTERNS = [
    r"취지.{0,24}(선령|검사\s*범위|주기|표)",
    r"(선령|검사\s*범위|주기|표).{0,24}취지",
    r"근거\s*조항.{0,24}(범위|주기|선령)",
    r"(범위|주기|선령).{0,24}근거",
    r"설명.{0,16}(표|선령별|검사\s*범위)",
    r"(표|선령별|검사\s*범위).{0,16}(설명|취지|조항)",
]


def _hit(patterns: list[str], q: str) -> bool:
    return any(re.search(p, q, flags=re.IGNORECASE) for p in patterns)


def classify_retrieval_mode(question: str) -> RetrievalMode:
    """Decide TEXT / TABLE / BOTH without an extra LLM call.

    Ambiguous questions that still carry a table cue default to BOTH so
    prose context is not dropped.
    """
    q = (question or "").strip()
    if not q:
        return RetrievalMode.TEXT

    table = _hit(_TABLE_PATTERNS, q)
    prose = _hit(_TEXT_PROSE_PATTERNS, q)
    bridge = _hit(_BOTH_BRIDGE_PATTERNS, q)

    if bridge or (table and prose):
        return RetrievalMode.BOTH
    if table:
        # Pure table lookup vs weak table word inside a meeting question
        strong_table = bool(
            re.search(
                r"표|선령|검사\s*주기|검사주기|평형수|밸러스트|개방검사|"
                r"survey\s*interval|intermediate\s*survey",
                q,
                flags=re.IGNORECASE,
            )
        )
        if strong_table:
            return RetrievalMode.TABLE
        return RetrievalMode.BOTH
    return RetrievalMode.TEXT


def mode_to_legacy_table_qa(mode: RetrievalMode) -> bool:
    """Backward-compatible flag used by older call sites."""
    return mode in {RetrievalMode.TABLE, RetrievalMode.BOTH}
