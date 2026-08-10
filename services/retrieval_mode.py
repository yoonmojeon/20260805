"""RAG retrieval mode: TEXT / TABLE / BOTH.

Primary signal: table_query_parser slots (cell/row/condition shape).
Fallback: light keyword cues. Ambiguous maritime questions → BOTH.
"""
from __future__ import annotations

import re
import sys
from enum import Enum
from pathlib import Path


class RetrievalMode(str, Enum):
    TEXT = "text"
    TABLE = "table"
    BOTH = "both"


# Minimal keyword backup (not the main decision path)
_TABLE_CUE_PATTERNS = [
    r"표\s*(에서|에|의|기준|내용|알려|정리|요약)?",
    r"표에\s*(나온|있는|관련)",
    r"검사\s*주기|검사주기|선령별|선령\s*\d+",
    r"평형수\s*탱크|밸러스트\s*탱크|개방검사|두께계측",
    r"(?:intermediate|annual|special|docking)\s*survey|survey\s*interval",
    r"열\s*\d+|row\s*\d+|cell",
]

_TEXT_PROSE_PATTERNS = [
    r"취지|목적|정의|의미|scope|요건\s*설명",
    r"회의\s*(결과|주요|결정|논의|요약)",
    r"주요\s*(결과|안건|내용|결정)",
    r"논의\s*(요약|내용|결과)",
    r"규정\s*(취지|목적|배경|설명)",
    r"근거\s*조항|조항\s*설명|circular|resolution",
    r"MEPC|MSC|MASS\s*Code|GHG\s*Strategy",
]

_BOTH_BRIDGE_PATTERNS = [
    r"취지.{0,24}(선령|검사\s*범위|주기|표)",
    r"(선령|검사\s*범위|주기|표).{0,24}취지",
    r"근거\s*조항.{0,24}(범위|주기|선령)",
    r"(범위|주기|선령).{0,24}근거",
    r"설명.{0,16}(표|선령별|검사\s*범위)",
    r"(표|선령별|검사\s*범위).{0,16}(설명|취지|조항)",
]

# Numeric/inequality cell-lookup shape (parser assist)
_NUMERIC_RANGE_RE = re.compile(
    r"(?:\d+(?:\.\d+)?\s*(?:m|mm|년|%|N\s*/?\s*mm)|"
    r"[Ll]\s*[<>≤≥=]|미만|이상|이하|초과)",
    re.IGNORECASE,
)
_VALUE_ASK_RE = re.compile(r"얼마|몇\s*(?:mm|m|년)?|값은|해당\s*값|요구되는|최소|최대")


def _hit(patterns: list[str], q: str) -> bool:
    return any(re.search(p, q, flags=re.IGNORECASE) for p in patterns)


def _ensure_parser_import() -> None:
    scripts = Path(__file__).resolve().parents[1] / "rag" / "scripts"
    s = str(scripts)
    if s not in sys.path:
        sys.path.insert(0, s)


def _parse_table_slots(question: str):
    _ensure_parser_import()
    from table_query_parser import parse_table_query

    return parse_table_query(question)


def table_shape_score(question: str) -> tuple[float, dict]:
    """Higher score ⇒ more like a table/cell lookup.

    Uses parser slots first; keywords only as a weak prior.
    """
    q = (question or "").strip()
    detail: dict = {"parser": None, "keyword_cue": False, "numeric_range": False}
    if not q:
        return 0.0, detail

    score = 0.0
    try:
        parsed = _parse_table_slots(q)
        detail["parser"] = {
            "query_type": parsed.query_type,
            "rows": list(parsed.row_entities)[:6],
            "cols": list(parsed.column_entities)[:6],
            "topics": list(parsed.table_topic_candidates)[:6],
            "units": list(parsed.unit_candidates)[:4],
            "conditions": list(parsed.condition_candidates)[:6],
        }
        qt = str(parsed.query_type or "")
        rows = list(parsed.row_entities or [])
        cols = list(parsed.column_entities or [])
        topics = list(parsed.table_topic_candidates or [])
        units = list(parsed.unit_candidates or [])
        conds = list(parsed.condition_candidates or [])
        # Generic attributes like "등급" alone (올해 CII 등급) are not table cues.
        weak_attrs = {"등급", "값", "항목", "내용", "방법", "결과", "상태"}
        strong_cols = [c for c in cols if c not in weak_attrs]
        # Parser marks bare "년" inside "올해" as a condition — ignore weak tokens.
        weak_conds = {"년", "조건", "구간", "이상", "이하", "초과", "미만"}
        strong_conds = [c for c in conds if c not in weak_conds]
        has_slots = bool(rows or strong_cols or units or strong_conds or topics)

        # Default parse often labels empty questions as cell_lookup — ignore that.
        if has_slots and qt in {
            "cell_lookup",
            "row_lookup",
            "condition_lookup",
            "column_lookup",
            "note_lookup",
            "table_lookup",
        }:
            score += 0.35 if qt != "table_lookup" else 0.2
        if rows:
            score += min(0.35, 0.15 * len(rows))
        if strong_cols:
            score += min(0.35, 0.15 * len(strong_cols))
        if units:
            score += 0.15
        if strong_conds:
            score += 0.12
        # Weak conds only count with a numeric/range ask.
        elif conds and _NUMERIC_RANGE_RE.search(q) and _VALUE_ASK_RE.search(q):
            score += 0.08
        if topics and any(
            t in {"치수", "dimension", "화학성분", "기계적성질", "정기검사", "inspection"}
            or "두께" in t
            or "용접" in t
            for t in topics
        ):
            score += 0.2
    except Exception as exc:
        detail["parser_error"] = str(exc)

    if _NUMERIC_RANGE_RE.search(q) and _VALUE_ASK_RE.search(q):
        score += 0.45
        detail["numeric_range"] = True
    if _hit(_TABLE_CUE_PATTERNS, q):
        score += 0.25
        detail["keyword_cue"] = True

    return min(score, 1.5), detail


def prose_shape_score(question: str) -> float:
    q = (question or "").strip()
    if not q:
        return 0.0
    score = 0.0
    if _hit(_TEXT_PROSE_PATTERNS, q):
        score += 0.7
    # Meeting/report framing without cell-ask leans prose
    if re.search(r"회의|동향|논의|요약해|정리해", q) and not _VALUE_ASK_RE.search(q):
        score += 0.25
    return min(score, 1.2)


def classify_retrieval_mode(question: str) -> RetrievalMode:
    """Decide TEXT / TABLE / BOTH.

    1) Parser/slot table shape → TABLE when clearly a cell/row lookup
    2) Clear prose (meeting/definition) without table shape → TEXT
    3) Bridge / both signals / weak-but-present table shape → BOTH
    4) Otherwise TEXT
    """
    q = (question or "").strip()
    if not q:
        return RetrievalMode.TEXT

    table_score, _detail = table_shape_score(q)
    prose_score = prose_shape_score(q)
    bridge = _hit(_BOTH_BRIDGE_PATTERNS, q)

    strong_table = table_score >= 0.55
    weak_table = 0.25 <= table_score < 0.55
    strong_prose = prose_score >= 0.55

    if bridge or (strong_table and strong_prose):
        return RetrievalMode.BOTH
    if strong_table and not strong_prose:
        return RetrievalMode.TABLE
    if strong_prose and not weak_table and not strong_table:
        return RetrievalMode.TEXT
    if weak_table:
        # Ambiguous: keep both corpora so a missed table cue still surfaces.
        return RetrievalMode.BOTH
    return RetrievalMode.TEXT


def mode_to_legacy_table_qa(mode: RetrievalMode) -> bool:
    """Backward-compatible flag used by older call sites."""
    return mode in {RetrievalMode.TABLE, RetrievalMode.BOTH}
