"""Meeting summary intent — broad session outcome questions (MSC 111 주요 결과 3개 등)."""
from __future__ import annotations

import re

from imo_doc_classify import asks_broad_session_outcome

MEETING_SUMMARY_MARKERS = (
    "주요 결과",
    "핵심 결과",
    "핵심 성과",
    "결과 요약",
    "결과를 요약",
    "결정된 주요 내용",
    "핵심 outcome",
    "outcome summary",
    "key outcomes",
    "key outcome",
    "main outcomes",
    "session outcomes",
    "회의 결과",
    "최종 결과",
)

# (term, weight) — higher weight = stronger signal for keyness scoring
SUMMARY_PRIORITY_WEIGHTED: tuple[tuple[str, float], ...] = (
    ("non-mandatory mass code", 6.0),
    ("mass code", 5.5),
    ("maritime autonomous", 4.0),
    ("ghg safety", 4.5),
    ("alternative fuel", 4.0),
    ("new technolog", 3.5),
    ("hydrogen fuel", 4.0),
    ("ammonia fuel", 4.0),
    ("ammonia", 3.0),
    ("hydrogen", 3.0),
    ("lrit", 4.0),
    ("vdes", 4.0),
    ("strait of hormuz", 4.0),
    ("hormuz", 3.5),
    ("adopted", 3.5),
    ("approved", 3.5),
    ("resolution", 3.0),
    ("amendments", 2.5),
    ("amendment", 2.5),
    ("guidelines", 2.5),
    ("guideline", 2.5),
    ("entry into force", 3.0),
    ("entered into force", 3.0),
    ("mandatory", 2.5),
    ("non-mandatory", 2.5),
    ("code", 2.0),
    ("finalized", 2.5),
    ("finalised", 2.5),
    ("work plan", 2.0),
    ("work programme", 2.0),
    ("채택", 3.0),
    ("승인", 3.0),
    ("결의안", 2.5),
    ("개정", 2.0),
    ("비강제", 2.5),
    ("지침", 2.0),
)

SUMMARY_PENALTY_WEIGHTED: tuple[tuple[str, float], ...] = (
    ("outcome of other imo bodies", 8.0),
    ("decisions of other imo bodies", 8.0),
    ("outcome of c 135", 7.0),
    ("outcome of a 34", 7.0),
    ("outcome of tc 75", 7.0),
    ("outcome of c 134", 7.0),
    ("outcome of mepc", 6.0),
    ("outcome of c ", 6.0),
    ("outcome of a ", 6.0),
    ("outcome of tc", 6.0),
    ("strategic plan", 6.0),
    ("fal 50", 6.0),
    ("fal.50", 6.0),
    ("consider the outcome", 5.0),
    ("consideration of the outcome", 5.0),
    ("already adopted at a 34", 5.0),
    ("identification number scheme", 4.0),
    ("imo identification number", 4.0),
    ("integrated imo identification", 4.0),
    ("observer status", 3.5),
    ("rules of procedure", 3.0),
    ("public release", 3.0),
    ("webcast", 3.0),
    ("noted only", 4.0),
    ("considered only", 4.0),
    ("no application", 4.0),
    ("no operational impact", 4.0),
    ("not confirmed in retrieved", 4.0),
    ("noted that", 3.0),
    ("invited the", 3.0),
    ("invited to", 3.0),
    ("no impact", 3.0),
    ("not applicable", 3.0),
    ("적용 대상 없음", 5.0),
    ("영향 없음", 5.0),
    ("시행 시점 없음", 5.0),
    ("발효 시점 없음", 5.0),
    ("확인 불가", 4.0),
    ("information on", 2.0),
)

EMPTY_OUTCOME_PHRASES = (
    "적용 대상 없음",
    "적용 대상: 없음",
    "시행/발효 시점 없음",
    "시행 시점 없음",
    "발효 시점 없음",
    "영향 없음",
    "선박 운항/설계/업무 영향: 없음",
    "선박 영향 없음",
    "검색 결과 내 확인 불가",
)

ANSWER_PENALTY_TOPICS = (
    "strategic plan",
    "fal 50",
    "outcome of c 135",
    "outcome of a 34",
    "c 135",
    "a 34",
    "outcome of tc 75",
    "decisions of other imo bodies",
    "outcome of other imo bodies",
    "identification number scheme",
)

PRIORITY_ANSWER_TOPICS = (
    ("mass code", ("mass code", "mass code", "자율운항", "maritime autonomous", "non-mandatory")),
    ("ghg", ("ghg", "alternative fuel", "new technolog", "대체연료")),
    ("lrit", ("lrit",)),
    ("vdes", ("vdes",)),
    ("hormuz", ("hormuz", "strait of hormuz", "호르무즈")),
    ("hydrogen/ammonia", ("hydrogen", "ammonia", "암모니아")),
)

REFERENCE_OUTCOME_DOC_PATTERNS = (
    "outcome_of_c_135",
    "outcome_of_a_34",
    "outcome_of_tc_75",
    "msc_111_2_1",
    "msc_111_2_outcome",
    "111-2-1",
    "111-2 -",
)

SUMMARY_TOPIC_PRIORITY: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("MASS Code", ("mass code", "non-mandatory code", "maritime autonomous", "non-mandatory and goal-based"), True),
    ("GHG / alt fuel", ("ghg", "alternative fuel", "new technolog", "low-flashpoint"), False),
    ("LRIT", ("lrit", "long-range identification"), False),
    ("VDES", ("vdes", "vhf data exchange"), False),
    ("hydrogen/ammonia", ("hydrogen fuel", "ammonia fuel", "ammonia as fuel"), False),
    ("Hormuz", ("strait of hormuz", "hormuz"), False),
)


def is_meeting_summary_intent(question: str, row: dict | None = None) -> bool:
    """True only for whole-session meeting summary (not C135/doc-specific questions)."""
    row = row or {}
    explicit = str(row.get("meeting_intent") or row.get("fast_question_type") or "").strip()
    if explicit == "meeting_summary":
        from meeting_summary_context import TARGET_SCOPE_WHOLE_SESSION, resolve_meeting_summary_context

        ctx = resolve_meeting_summary_context(question, row)
        return ctx.target_scope == TARGET_SCOPE_WHOLE_SESSION
    if explicit == "meeting_outcome_question":
        return False
    q = (question or "").strip()
    if not q:
        return False
    from meeting_summary_context import TARGET_SCOPE_WHOLE_SESSION, resolve_meeting_summary_context

    if not _has_summary_markers(q):
        if not (
            re.search(r"결과\s*\d+\s*개", q)
            or (re.search(r"\d+\s*개\s*(?:항목|개)", q) and asks_broad_session_outcome(q))
            or (
                re.search(r"\b(msc|mepc)\s*\d{1,3}\b", q.lower())
                and re.search(r"주요|핵심|outcome|결과|성과", q, re.I)
                and asks_broad_session_outcome(q)
            )
        ):
            return False
    ctx = resolve_meeting_summary_context(q, row)
    return ctx.target_scope == TARGET_SCOPE_WHOLE_SESSION


def _has_summary_markers(question: str) -> bool:
    lower = question.lower()
    return any(m.lower() in lower for m in MEETING_SUMMARY_MARKERS)


def is_penalty_summary_source(file_name: str, doc_id: str = "", **kwargs) -> bool:
    from meeting_summary_context import get_summary_context, is_penalty_summary_source as _pen

    ctx = kwargs.get("ctx")
    if ctx is None and kwargs.get("question"):
        ctx = get_summary_context(kwargs["question"], kwargs.get("row"))
    return _pen(file_name, doc_id, ctx=ctx)


def score_summary_claim_text(text: str, *, source_name: str = "", **kwargs) -> float:
    from meeting_summary_context import get_summary_context, score_summary_claim_text as _score

    ctx = kwargs.get("ctx")
    doc_id = kwargs.get("doc_id", "")
    if ctx is None and kwargs.get("question"):
        ctx = get_summary_context(kwargs["question"], kwargs.get("row"))
    return _score(text, source_name=source_name, doc_id=doc_id, ctx=ctx)


def validate_meeting_summary_answer(answer: str, *, row: dict | None = None, **kwargs) -> tuple[bool, list[str]]:
    from meeting_summary_context import get_summary_context, validate_meeting_summary_answer as _val

    ctx = kwargs.get("ctx")
    if ctx is None:
        ctx = get_summary_context(str((row or {}).get("question", "")), row)
    return _val(
        answer,
        row=row,
        evidence_sources=kwargs.get("evidence_sources"),
        ctx=ctx,
    )
