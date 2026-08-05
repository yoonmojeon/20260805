"""Scoped routing for meeting_summary — avoids overfitting to MSC 111 / V02."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from imo_doc_classify import asks_broad_session_outcome
from meeting_summary_intent import MEETING_SUMMARY_MARKERS, REFERENCE_OUTCOME_DOC_PATTERNS
from retrieval_query_analysis import analyze_query

TARGET_SCOPE_WHOLE_SESSION = "whole_session"
TARGET_SCOPE_SPECIFIC_DOCUMENT = "specific_document"
TARGET_SCOPE_OTHER_BODY_OUTCOME = "other_body_outcome"
TARGET_SCOPE_SPECIFIC_TOPIC = "specific_topic"
TARGET_SCOPE_GENERAL = "general"

SPECIFIC_DOC_REF_RE = re.compile(
    r"\b(?:msc|mepc)\s*(\d{1,3})[-/](\d{1,2})(?:[-/](\d+))?\b",
    re.I,
)
OTHER_BODY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"c\s*135(?!\d)", re.I), "C 135"),
    (re.compile(r"a\s*34(?!\d)", re.I), "A 34"),
    (re.compile(r"c\s*134(?!\d)", re.I), "C 134"),
    (re.compile(r"tc\s*75(?!\d)", re.I), "TC 75"),
    (re.compile(r"fal\s*50(?!\d)", re.I), "FAL 50"),
)
OTHER_BODY_FOCUS_LABELS = frozenset({"C 135", "A 34", "C 134", "TC 75", "FAL 50"})
MEPC_ES_SESSION_RE = re.compile(r"mepc\s*es\.?\s*2", re.I)
DOC_REQUEST_RE = re.compile(r"문서|document|내용을\s*요약|요약해", re.I)
FAL_TOPIC_RE = re.compile(r"fal\s*50", re.I)

MSC111_WHOLE_SESSION_REQUIRE_TOPICS: tuple[str, ...] = ("MASS Code",)

PENALTY_DOC_HINTS_WHOLE_SESSION = (
    "outcome of c 135",
    "outcome of a 34",
    "outcome of tc 75",
    "outcome of c 134",
    "outcome of mepc",
    "111-2-1",
    "111-2 -",
    "msc 111-2",
    "strategic plan",
    "fal 50",
    "decisions of other imo bodies",
)


@dataclass
class MeetingSummaryContext:
    detected_intent: str
    target_meeting: str | None
    target_body: str | None
    target_session: int | None
    target_scope: str
    topic_focus: list[str] = field(default_factory=list)
    apply_session_final_priority: bool = False
    apply_reference_penalties: bool = False
    require_topics: list[str] = field(default_factory=list)
    preferred_doc_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected_intent": self.detected_intent,
            "target_meeting": self.target_meeting,
            "target_body": self.target_body,
            "target_session": self.target_session,
            "target_scope": self.target_scope,
            "topic_focus": self.topic_focus,
            "apply_session_final_priority": self.apply_session_final_priority,
            "apply_reference_penalties": self.apply_reference_penalties,
            "require_topics": self.require_topics,
            "preferred_doc_hints": self.preferred_doc_hints,
        }


def _session_label(body: str, num: int) -> str:
    return f"{body.upper()} {num}"


def _has_whole_session_markers(question: str) -> bool:
    lower = question.lower()
    if any(m.lower() in lower for m in MEETING_SUMMARY_MARKERS):
        return True
    if re.search(r"결과를\s*\d+", question):
        return True
    if re.search(r"outcome\s+summary", lower):
        return True
    return False


def _is_other_body_focus(topic_focus: list[str]) -> bool:
    return any(t in OTHER_BODY_FOCUS_LABELS for t in topic_focus)


def _is_whole_session_summary_question(question: str) -> bool:
    if not _has_whole_session_markers(question):
        return False
    sig = analyze_query(question)
    if sig.session_codes:
        return asks_broad_session_outcome(question)
    if MEPC_ES_SESSION_RE.search(question):
        return True
    return asks_broad_session_outcome(question)


def _detect_topic_focus(question: str) -> list[str]:
    topics: list[str] = []
    for pat, label in OTHER_BODY_PATTERNS:
        if pat.search(question):
            topics.append(label)
    return topics


def _is_specific_document_request(question: str) -> bool:
    if not SPECIFIC_DOC_REF_RE.search(question):
        return False
    return bool(DOC_REQUEST_RE.search(question))


def resolve_meeting_summary_context(question: str, row: dict | None = None) -> MeetingSummaryContext:
    row = row or {}
    q = (question or "").strip()
    sig = analyze_query(q)
    body: str | None = None
    session_num: int | None = None
    if sig.session_codes:
        body, session_num = sig.session_codes[0]
    target_meeting = _session_label(body, session_num) if body and session_num else None

    topic_focus = _detect_topic_focus(q)
    explicit_intent = str(row.get("meeting_intent") or row.get("fast_question_type") or "").strip()
    is_summary = explicit_intent == "meeting_summary" or (
        explicit_intent != "meeting_outcome_question"
        and _has_whole_session_markers(q)
        and asks_broad_session_outcome(q)
    )

    if _is_specific_document_request(q):
        scope = TARGET_SCOPE_SPECIFIC_DOCUMENT
        hints = []
        m = SPECIFIC_DOC_REF_RE.search(q)
        if m:
            body_code = m.group(1).upper()
            sess = m.group(2)
            hints.append(f"{body_code} {sess}")
            hints.append(f"{body_code.lower()}-{sess}")
            if m.group(3):
                hints.append(f"{body_code.lower()}-{sess}-{m.group(3)}")
                hints.append(f"{body_code} {sess}/{m.group(3)}")
                hints.append("111-2-1")
        return MeetingSummaryContext(
            detected_intent="meeting_outcome_question" if not is_summary else "meeting_summary",
            target_meeting=target_meeting,
            target_body=body,
            target_session=session_num,
            target_scope=scope,
            topic_focus=topic_focus,
            apply_session_final_priority=False,
            apply_reference_penalties=False,
            require_topics=[],
            preferred_doc_hints=hints,
        )

    if _is_other_body_focus(topic_focus) and not _is_whole_session_summary_question(q):
        scope = TARGET_SCOPE_OTHER_BODY_OUTCOME
        if FAL_TOPIC_RE.search(q) and not _has_whole_session_markers(q):
            scope = TARGET_SCOPE_SPECIFIC_TOPIC
        preferred = []
        if "C 135" in topic_focus or "A 34" in topic_focus:
            preferred.extend(["outcome of c 135", "outcome of a 34", "111-2-1"])
        if "FAL 50" in topic_focus:
            preferred.append("fal 50")
        if "TC 75" in topic_focus:
            preferred.append("outcome of tc 75")
        if "MEPC ES.2" in topic_focus:
            preferred.append("mepc es.2")
        return MeetingSummaryContext(
            detected_intent="meeting_outcome_question" if not is_summary else "meeting_summary",
            target_meeting=target_meeting,
            target_body=body,
            target_session=session_num,
            target_scope=scope,
            topic_focus=topic_focus,
            apply_session_final_priority=False,
            apply_reference_penalties=False,
            require_topics=[],
            preferred_doc_hints=preferred,
        )

    if FAL_TOPIC_RE.search(q) and not _has_whole_session_markers(q):
        return MeetingSummaryContext(
            detected_intent="meeting_outcome_question" if not is_summary else "meeting_summary",
            target_meeting=target_meeting,
            target_body=body,
            target_session=session_num,
            target_scope=TARGET_SCOPE_SPECIFIC_TOPIC,
            topic_focus=["FAL 50"],
            apply_session_final_priority=False,
            apply_reference_penalties=False,
            require_topics=[],
            preferred_doc_hints=["fal 50"],
        )

    if _is_whole_session_summary_question(q) and MEPC_ES_SESSION_RE.search(q):
        return MeetingSummaryContext(
            detected_intent="meeting_summary",
            target_meeting="MEPC ES.2",
            target_body="MEPC",
            target_session=2,
            target_scope=TARGET_SCOPE_WHOLE_SESSION,
            topic_focus=topic_focus,
            apply_session_final_priority=True,
            apply_reference_penalties=True,
            require_topics=[],
            preferred_doc_hints=["mepc es.2", "mepc es 2"],
        )

    if is_summary and asks_broad_session_outcome(q) and sig.session_codes:
        require: list[str] = []
        if target_meeting == "MSC 111":
            require = list(MSC111_WHOLE_SESSION_REQUIRE_TOPICS)
        return MeetingSummaryContext(
            detected_intent="meeting_summary",
            target_meeting=target_meeting,
            target_body=body,
            target_session=session_num,
            target_scope=TARGET_SCOPE_WHOLE_SESSION,
            topic_focus=topic_focus,
            apply_session_final_priority=True,
            apply_reference_penalties=True,
            require_topics=require,
            preferred_doc_hints=[],
        )

    return MeetingSummaryContext(
        detected_intent="meeting_summary" if is_summary else "general",
        target_meeting=target_meeting,
        target_body=body,
        target_session=session_num,
        target_scope=TARGET_SCOPE_GENERAL,
        topic_focus=topic_focus,
        apply_session_final_priority=False,
        apply_reference_penalties=False,
        require_topics=[],
        preferred_doc_hints=[],
    )


def get_summary_context(question: str, row: dict | None = None) -> MeetingSummaryContext:
    cached = (row or {}).get("_meeting_summary_ctx")
    if isinstance(cached, MeetingSummaryContext):
        return cached
    if isinstance(cached, dict):
        return MeetingSummaryContext(**{k: cached[k] for k in MeetingSummaryContext.__dataclass_fields__ if k in cached})
    return resolve_meeting_summary_context(question, row)


def _blob(file_name: str, doc_id: str = "") -> str:
    return f"{(file_name or '').lower()} {(doc_id or '').lower()}"


def is_penalty_summary_source(
    file_name: str,
    doc_id: str = "",
    *,
    ctx: MeetingSummaryContext | None = None,
) -> bool:
    if ctx and not ctx.apply_reference_penalties:
        return False
    return meeting_summary_source_tier(file_name, doc_id=doc_id, ctx=ctx) >= 3


def meeting_summary_source_tier(
    file_name: str,
    *,
    doc_id: str = "",
    ctx: MeetingSummaryContext | None = None,
) -> int:
    blob = _blob(file_name, doc_id)

    if ctx:
        if ctx.target_scope == TARGET_SCOPE_OTHER_BODY_OUTCOME:
            if any(h in blob for h in ctx.preferred_doc_hints):
                return 1
            if "outcome of c 135" in blob or "outcome of a 34" in blob or "111-2-1" in blob:
                return 1
        if ctx.target_scope == TARGET_SCOPE_SPECIFIC_DOCUMENT:
            if any(
                h.replace(" ", "").lower() in blob.replace(" ", "") or h in blob
                for h in ctx.preferred_doc_hints
            ):
                return 1
            if "111-2-1" in blob or "111/2/1" in blob:
                return 1
        if ctx.target_scope == TARGET_SCOPE_SPECIFIC_TOPIC:
            if ctx.topic_focus and any(t.lower().replace(" ", "") in blob.replace(" ", "") for t in ctx.topic_focus):
                return 2
        if ctx.apply_reference_penalties:
            if any(h in blob for h in PENALTY_DOC_HINTS_WHOLE_SESSION):
                return 3
            if any(p in blob for p in REFERENCE_OUTCOME_DOC_PATTERNS):
                return 3

    return _base_meeting_summary_source_tier(file_name, doc_id=doc_id)


def _base_meeting_summary_source_tier(file_name: str, doc_id: str = "") -> int:
    from imo_doc_classify import meeting_summary_source_tier

    return meeting_summary_source_tier(file_name, doc_id=doc_id)


def score_summary_claim_text(
    text: str,
    *,
    source_name: str = "",
    doc_id: str = "",
    ctx: MeetingSummaryContext | None = None,
) -> float:
    from meeting_summary_intent import SUMMARY_PENALTY_WEIGHTED, SUMMARY_PRIORITY_WEIGHTED

    blob = f"{source_name} {text}".lower()
    score = 0.0
    for term, weight in SUMMARY_PRIORITY_WEIGHTED:
        if term in blob:
            score += weight
    if not ctx or ctx.apply_reference_penalties:
        for term, weight in SUMMARY_PENALTY_WEIGHTED:
            if term in blob:
                score -= weight
    if re.search(r"\b(noted|invited|recalled)\b", blob) and not re.search(
        r"\b(adopted|approved|finalized|finalised)\b", blob
    ):
        score -= 3.0
    if source_name or doc_id:
        tier = meeting_summary_source_tier(source_name, doc_id=doc_id, ctx=ctx)
        if tier == 0:
            score += 6.0
        elif tier == 1:
            score += 4.0
        elif tier >= 3:
            score -= 12.0
    return score


def validate_meeting_summary_answer(
    answer: str,
    *,
    row: dict | None = None,
    evidence_sources: list[str] | None = None,
    ctx: MeetingSummaryContext | None = None,
) -> tuple[bool, list[str]]:
    from meeting_summary_intent import EMPTY_OUTCOME_PHRASES

    ctx = ctx or get_summary_context(str((row or {}).get("question", "")), row)
    if not (answer or "").strip():
        return False, ["empty_answer"]
    reasons: list[str] = []
    lower = answer.lower()

    if any(p in answer for p in EMPTY_OUTCOME_PHRASES):
        reasons.append("empty_regulatory_fields")

    if ctx.apply_reference_penalties:
        for topic in (
            "strategic plan",
            "fal 50",
            "outcome of c 135",
            "outcome of a 34",
            "c 135",
            "a 34",
            "outcome of tc 75",
        ):
            if topic in lower:
                reasons.append(f"penalty_topic:{topic.replace(' ', '_')}")

    if ctx.require_topics:
        for req in ctx.require_topics:
            keys = {
                "MASS Code": ("mass code", "maritime autonomous", "자율운항", "non-mandatory"),
            }.get(req, (req.lower(),))
            if not any(k in lower for k in keys):
                reasons.append(f"missing_required_topic:{req.replace(' ', '_')}")

    if ctx.target_scope == TARGET_SCOPE_WHOLE_SESSION:
        has_substance = any(
            t in lower
            for t in ("adopted", "approved", "mandatory", "code", "guideline", "resolution", "채택", "승인", "결의")
        )
        if not has_substance:
            reasons.append("missing_adopted_approved_substance")
        numbered = len(re.findall(r"^\s*\d+\.", answer, re.MULTILINE))
        if numbered < 2:
            reasons.append("insufficient_numbered_items")

    if evidence_sources and ctx.apply_session_final_priority:
        primary_ok = any(
            meeting_summary_source_tier(s, ctx=ctx) <= 1 for s in evidence_sources if s
        )
        penalty_primary = any(
            is_penalty_summary_source(s, ctx=ctx)
            or any(p in (s or "").lower() for p in REFERENCE_OUTCOME_DOC_PATTERNS)
            for s in evidence_sources[:2]
        )
        if not primary_ok:
            reasons.append("no_primary_source_evidence")
        if penalty_primary:
            reasons.append("reference_outcome_used_as_primary")

    return (not reasons, reasons)


def build_claim_based_summary_answer(
    ctx: MeetingSummaryContext,
    selected_claims: list[dict[str, Any]],
    *,
    row: dict | None = None,
) -> str:
    from meeting_outcome_answer import session_title_from_question
    from meeting_outcome_retrieval import parse_outcome_item_count

    row = row or {}
    question = str(row.get("question", ""))
    n = parse_outcome_item_count(question, row)
    session = ctx.target_meeting or session_title_from_question(question, row)
    claims = [c for c in selected_claims if c.get("claim")][:n]
    if not claims:
        return ""

    lines = [
        f"{session}의 주요 결과는 다음 {min(len(claims), n)}가지로 요약할 수 있습니다.",
        "",
    ]
    for i, c in enumerate(claims, 1):
        claim_text = str(c.get("claim") or c.get("evidence") or "").strip()
        title = _title_from_claim(claim_text)
        cite = c.get("cite_id")
        page = c.get("page")
        cite_str = f"[{cite}]" if cite is not None else (f"p.{page}" if page is not None else "—")
        lines.extend(
            [
                f"{i}. {title}",
                f"- 주요 내용: {claim_text}",
                "- 실무 의미: (검색 근거 범위 내 — 상세 해석은 Accurate mode 권장)",
                f"- 근거: {cite_str}",
                "",
            ]
        )
    lines.append("상세 분석은 Accurate mode에서 수행 가능합니다.")
    return "\n".join(lines)


def _title_from_claim(claim_text: str) -> str:
    lower = claim_text.lower()
    if "mass code" in lower or "maritime autonomous" in lower:
        return "비강제 MASS Code 채택"
    if "lrit" in lower:
        return "LRIT 관련 결정"
    if "vdes" in lower:
        return "VDES 도입·운용 관련 결정"
    if "hormuz" in lower:
        return "호르무즈 해협 관련 결의"
    if "alternative fuel" in lower or "ghg" in lower or "ammonia" in lower or "hydrogen" in lower:
        return "GHG safety 및 대체연료·신기술 선박 안전 관련 결정"
    if "adopted" in lower or "approved" in lower:
        m = re.search(r"(?:adopted|approved)\s+(.{10,80})", claim_text, re.I)
        if m:
            return m.group(1).strip().rstrip(".")[:80]
    return "핵심 회의 결정"


def topic_priority_for_context(ctx: MeetingSummaryContext) -> tuple[tuple[str, tuple[str, ...], bool], ...]:
    from meeting_summary_intent import SUMMARY_TOPIC_PRIORITY

    if ctx.require_topics and "MASS Code" in ctx.require_topics:
        return SUMMARY_TOPIC_PRIORITY
    return tuple((label, kws, False) for label, kws, _ in SUMMARY_TOPIC_PRIORITY)
