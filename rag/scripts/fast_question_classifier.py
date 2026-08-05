"""Fast-mode question type classification (additive layer)."""
from __future__ import annotations

import re
from typing import Literal

from meeting_outcome_retrieval import detect_meeting_outcome_question
from meeting_summary_context import (
    TARGET_SCOPE_WHOLE_SESSION,
    MEPC_ES_SESSION_RE,
    _has_whole_session_markers,
    resolve_meeting_summary_context,
)
from question_classifier import detect_broad_summary_mode, classify_question_category
from table_retrieval import is_table_question

FastQuestionType = Literal[
    "rule_question",
    "meeting_summary",
    "meeting_outcome_question",
    "table_question",
    "broad_summary_question",
    "figure_or_diagram_question",
    "general_question",
]

FAST_TYPE_LABELS_KO = {
    "rule_question": "규정/조항 질문",
    "meeting_summary": "회의 주요 결과 요약",
    "meeting_outcome_question": "회의 결과 질문",
    "table_question": "표/수치 질문",
    "broad_summary_question": "동향/요약 질문",
    "figure_or_diagram_question": "그림/도표 질문",
    "general_question": "일반 질문",
}

RULE_PATTERNS = (
    r"\brule\b",
    r"\bguidance\b",
    r"규정",
    r"조항",
    r"requirement",
    r"적용\s*대상",
    r"해야\s*하는가",
    r"해야\s*하나",
    r"의무",
    r"cg-\d+",
    r"notice\s*no",
    r"section\s*\d+",
    r"\d{3,4}\s*절",
)

FIGURE_PATTERNS = (
    r"그림",
    r"도표",
    r"diagram",
    r"schematic",
    r"illustration",
    r"figure\s*\d",
    r"그림\s*\d",
    r"도면",
)

TABLE_FAST_EXTRA = ("몇 개", "표시", "셀", "몇 년", "몇개")


def classify_fast_question_type(question: str, row: dict | None = None) -> FastQuestionType:
    """Classify question for Fast-mode retrieval/context strategy."""
    row = row or {}
    explicit = str(row.get("fast_question_type") or "").strip()
    if explicit in FAST_TYPE_LABELS_KO or explicit in {
        "rule_question",
        "meeting_summary",
        "meeting_outcome_question",
        "table_question",
        "broad_summary_question",
        "figure_or_diagram_question",
        "general_question",
    }:
        return explicit  # type: ignore[return-value]

    q = question.strip()
    lower = q.lower()

    ctx = resolve_meeting_summary_context(q, row)
    if ctx.target_scope == TARGET_SCOPE_WHOLE_SESSION:
        return "meeting_summary"
    if ctx.target_scope in {"other_body_outcome", "specific_document", "specific_topic"}:
        return "meeting_outcome_question"

    if MEPC_ES_SESSION_RE.search(q) and _has_whole_session_markers(q):
        return "meeting_summary"

    if is_table_question(q) or any(k in q for k in TABLE_FAST_EXTRA):
        return "table_question"

    if detect_meeting_outcome_question(q, row):
        return "meeting_outcome_question"

    if any(re.search(p, lower, re.I) for p in RULE_PATTERNS):
        return "rule_question"
    if str(row.get("category") or "") == "rule_lookup":
        return "rule_question"

    if any(re.search(p, q, re.I) for p in FIGURE_PATTERNS):
        return "figure_or_diagram_question"

    cat = classify_question_category(q, row)
    if detect_broad_summary_mode(q, row, cat):
        return "broad_summary_question"
    if cat in {"trend_summary", "env_regulation", "autonomous"} and any(
        k in lower for k in ("동향", "요약", "정리", "주요 내용", "최신", "summary", "highlight")
    ):
        return "broad_summary_question"

    return "general_question"


def fast_type_label_ko(fast_type: str) -> str:
    return FAST_TYPE_LABELS_KO.get(fast_type, fast_type)
