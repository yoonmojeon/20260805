"""Top-level category, internal intent, and retrieval profile for meeting QA."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from meeting_outcome_retrieval import parse_outcome_item_count
from meeting_summary_intent import is_meeting_summary_intent
from retrieval_query_analysis import is_meeting_outcome_question

# User-facing top-level categories
TOP_LEVEL_TREND = "latest_trend_summary"
TOP_LEVEL_ENV = "env_regulation_response"
TOP_LEVEL_AUTO = "autonomous_mass"
TOP_LEVEL_RULE = "rule_guidance_lookup"

LEGACY_TO_TOP: dict[str, str] = {
    "trend_summary": TOP_LEVEL_TREND,
    "meeting_outcome": TOP_LEVEL_TREND,
    "env_regulation": TOP_LEVEL_ENV,
    "autonomous": TOP_LEVEL_AUTO,
    "rule_lookup": TOP_LEVEL_RULE,
}

TOP_LEVEL_LABELS_KO: dict[str, str] = {
    TOP_LEVEL_TREND: "최신 동향 요약",
    TOP_LEVEL_ENV: "환경규제 대응",
    TOP_LEVEL_AUTO: "자율운항(MASS)",
    TOP_LEVEL_RULE: "Rule/Guidance 조회",
}


@dataclass
class MeetingRetrievalProfile:
    top_level_category: str
    internal_intent: str
    profile_id: str
    use_dense: bool = True
    use_bm25: bool = True
    use_rrf: bool = True
    use_source_tier: bool = True
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    sub_queries: list[str] = field(default_factory=list)
    section_emphasis: dict[str, str] = field(default_factory=dict)
    requested_bullet_count: int | None = None
    answer_variant: str = "default"

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "top_level_category": self.top_level_category,
            "internal_intent": self.internal_intent,
            "retrieval_profile": self.profile_id,
            "use_dense": self.use_dense,
            "use_bm25": self.use_bm25,
            "use_rrf": self.use_rrf,
            "use_source_tier": self.use_source_tier,
            "dense_weight": self.dense_weight,
            "bm25_weight": self.bm25_weight,
            "sub_queries": self.sub_queries,
            "section_emphasis": self.section_emphasis,
            "requested_bullet_count": self.requested_bullet_count,
        }


def resolve_internal_intent(question: str, row: dict, *, legacy_category: str) -> str:
    explicit = str(row.get("internal_intent") or "").strip()
    if explicit:
        return explicit

    q = question or ""
    ql = q.lower()
    data_quality_query = bool(
        re.search(r"\b(?:DCS|GISIS)\b|(?:보고|제출)\s*데이터", q, re.I)
        and re.search(
            r"품질|검증|오류|누락|중복|비현실|잘못된|"
            r"quality|verification|error|missing|duplicate|unrealistic|incorrect",
            q,
            re.I,
        )
    )

    if legacy_category == "autonomous" or (
        "mass code" in ql and any(k in q for k in ("mandatory", "일정", "timeline", "experience-building"))
    ):
        return "mass_code_timeline"

    if legacy_category == "env_regulation":
        if data_quality_query:
            return "data_quality_verification"
        if any(
            k.lower() in ql
            for k in (
                "대체연료",
                "GHG 안전",
                "ghg safety",
                "alternative fuel",
                "저인화점",
                "암모니아",
                "ammonia",
                "수소",
                "hydrogen",
                "igf code",
                "igc code",
            )
        ) or ("ghg" in ql and "안전" in q):
            return "altfuel_ghg_safety"
        return "env_regulation"

    if is_meeting_summary_intent(question, row):
        return "meeting_outcome"
    if legacy_category == "meeting_outcome":
        return "meeting_outcome"
    if is_meeting_outcome_question(question, row):
        return "meeting_outcome"
    if legacy_category == "trend_summary":
        return "trend_summary"
    return legacy_category or "general"


def resolve_top_level_category(legacy_category: str) -> str:
    return LEGACY_TO_TOP.get(legacy_category, TOP_LEVEL_TREND)


def build_meeting_retrieval_profile(question: str, row: dict, *, legacy_category: str) -> MeetingRetrievalProfile:
    from question_requirements import analyze_requirements

    top = resolve_top_level_category(legacy_category)
    intent = resolve_internal_intent(question, row, legacy_category=legacy_category)
    q = question or ""
    ql = q.lower()
    requirements = analyze_requirements(q, row)
    session_match = re.search(r"\b(MSC|MEPC)\s*[-/]?\s*(\d{1,3})\b", q, re.I)
    session = (
        f"{session_match.group(1).upper()} {session_match.group(2)}"
        if session_match
        else ""
    )

    if top == TOP_LEVEL_TREND:
        n = None
        subs: list[str] = []
        if intent == "meeting_outcome" or row.get("outcome_item_count") or re.search(
            r"\d+\s*개\s*(?:항목|개)", q
        ):
            n = parse_outcome_item_count(q, row)
        if intent == "meeting_outcome":
            prefix = session or re.sub(r"\s+", " ", q).strip()
            subs = [
                f"{prefix} official session report adopted approved agreed decisions",
                f"{prefix} committee outcome decisions resolutions",
                f"{prefix} report action taken key outcomes",
            ]
        return MeetingRetrievalProfile(
            top_level_category=top,
            internal_intent=intent,
            profile_id="trend_dense_first",
            use_dense=True,
            use_bm25=True,
            use_rrf=True,
            use_source_tier=True,
            dense_weight=1.2,
            bm25_weight=0.8,
            requested_bullet_count=n,
            sub_queries=subs,
            section_emphasis={"1": "primary", "2": "secondary", "3": "secondary", "4": "optional"},
        )

    if top == TOP_LEVEL_ENV:
        subs: list[str] = []
        if intent == "data_quality_verification":
            prefix = session or "MEPC"
            subs = [
                f"{prefix} IMO DCS submitted data quality control verification identified errors",
                f"{prefix} duplicate reporting unrealistic hours ship particulars incorrect ship type",
                f"{prefix} missing ships excluded from analysis not been included",
            ]
        elif requirements.is_concrete:
            subs = requirements.search_queries()
        elif intent == "altfuel_ghg_safety":
            prefix = session or re.sub(r"\s+", " ", q).strip()
            subs = [
                f"{prefix} alternative fuel safety interim guidelines decision",
                f"{prefix} GHG safety regulatory framework working group",
                f"{prefix} low-flashpoint fuel approved agreed conclusion",
            ]
        elif re.search(r"운항|규제\s*보고|CII|reporting|탄소", q, re.I):
            prefix = session or "MEPC"
            subs = [
                f"{prefix} operational reporting requirements verification DCS GISIS",
                f"{prefix} carbon intensity CII AER transport work",
                f"{prefix} submitted data quality control duplicate reporting",
            ]
        elif "mepc" in ql and re.search(r"최신|최근|latest|current", q, re.I):
            prefix = session or "MEPC"
            subs = [
                f"{prefix} MARPOL Annex VI amendments status adoption",
                f"{prefix} GHG carbon intensity reporting verification",
                f"{prefix} fuel lifecycle emission factors LCA",
            ]
        return MeetingRetrievalProfile(
            top_level_category=top,
            internal_intent=intent,
            profile_id=(
                "env_data_quality"
                if intent == "data_quality_verification"
                else "env_balanced"
                if intent == "env_regulation"
                else "env_altfuel_safety"
            ),
            use_dense=True,
            use_bm25=True,
            use_rrf=True,
            use_source_tier=True,
            dense_weight=1.0,
            bm25_weight=1.1 if intent == "altfuel_ghg_safety" else 1.0,
            sub_queries=subs,
            section_emphasis={"1": "secondary", "2": "primary", "3": "primary", "4": "optional"},
        )

    if top == TOP_LEVEL_AUTO:
        prefix = session or "MSC MASS"
        subs = [
            f"{prefix} MASS Code adopted non-mandatory decision",
            f"{prefix} mandatory MASS Code adoption target timeline road map",
            f"{prefix} mandatory Code entry into force target",
            f"{prefix} experience-building phase schedule uncertainty",
        ]
        return MeetingRetrievalProfile(
            top_level_category=top,
            internal_intent="mass_code_timeline",
            profile_id="autonomous_bm25_first",
            use_dense=True,
            use_bm25=True,
            use_rrf=True,
            use_source_tier=True,
            dense_weight=0.9,
            bm25_weight=1.3,
            sub_queries=subs,
            section_emphasis={"1": "primary", "2": "secondary", "3": "primary", "4": "optional"},
        )

    return MeetingRetrievalProfile(
        top_level_category=top,
        internal_intent=intent,
        profile_id="default",
    )


MEETING_BODY_RE = re.compile(
    r"MSC|MEPC|IMO|ISWG|회의|위원회|committee|session|결의|resolution|의제|안건",
    re.I,
)


def has_meeting_cue(question: str) -> bool:
    """True when the question itself carries a trend/environment/MASS/meeting cue.

    ``classify_question_category`` returns ``trend_summary`` as its fallback, so
    a class-rule question with no meeting cue used to arrive here and be answered
    with "회의 결정·결과" bullets. Requiring a cue keeps the structured meeting
    template on questions that actually asked about a meeting.
    """
    q = str(question or "")
    if not q.strip():
        return False
    if MEETING_BODY_RE.search(q):
        return True
    from question_classifier import AUTONOMOUS_PATTERNS, ENV_PATTERNS, TREND_PATTERNS

    return any(
        re.search(pattern, q, re.I)
        for pattern in TREND_PATTERNS + ENV_PATTERNS + AUTONOMOUS_PATTERNS
    )


def _is_narrow_meeting_fact_question(question: str) -> bool:
    """Detect value/list/reason asks that happen to name a meeting document."""
    return bool(
        re.search(
            r"얼마|몇\s*(?:개|건|년|개월|일|%|퍼센트)|언제|마감일|기한|"
            r"어느\s*회의|어떤\s*(?:값|수치|항목|사유|이유|용도|방법|법률|메커니즘)|"
            r"무엇(?:입니까|인가|이었|으로)|어떻게\s*(?:계산|산정|활용|사용|평가)|"
            r"포함(?:되어|해야)|비율|농도|기본값|상한|하한|보존\s*기간",
            question,
            re.I,
        )
    )


def uses_structured_meeting_answer(row: dict, *, legacy_category: str) -> bool:
    """Use the deterministic renderer only for actual meeting-level asks.

    A meeting acronym merely constrains the source.  A narrow fact question in
    an MEPC/MSC paper still needs grounded generation from the retrieved
    passage; rendering the canned session summary discards that passage.
    """
    # table_qa is not in LEGACY_TO_TOP and must not fall through to TREND,
    # or Fast retrieval injects meeting evidence slots and drops table crops.
    if row.get("_table_qa") or str(row.get("category") or "") == "table_qa":
        return False
    if str(legacy_category or "") == "table_qa":
        return False
    # A meeting + class-rule integration question needs two retrieval lanes and
    # an LLM-generated checklist. The meeting-only renderer cannot represent
    # the class evidence even when the right class PDF is already indexed.
    from compound_regulatory import is_compound_regulatory_class_question

    if row.get("_compound_regulatory_class") or is_compound_regulatory_class_question(
        str(row.get("question") or "")
    ):
        return False
    question = str(row.get("question") or "")
    top = resolve_top_level_category(legacy_category)
    if top == TOP_LEVEL_TREND and not has_meeting_cue(question):
        return False

    # Whole-session summaries and explicit adoption/approval/decision asks are
    # precisely what the structured renderer was built for.
    if is_meeting_summary_intent(question, row):
        return True

    if re.search(r"\b(?:DCS|GISIS)\b", question, re.I) and re.search(
        r"품질|검증|오류|누락|중복|quality|verification|error|missing|duplicate",
        question,
        re.I,
    ):
        return True

    # A paper/item identifier below the session level is a document scope, not
    # a request for the canned whole-session briefing.  Keep structured mode
    # only when the user explicitly asks to summarise that document's results.
    if re.search(
        r"\b(?:MEPC|MSC)\s*\d{1,3}(?:\s*[/.-]\s*[A-Z0-9]+)+",
        question,
        re.I,
    ) and not re.search(
        r"요약|정리|주요\s*(?:결과|결정|내용)|핵심\s*(?:결과|결정|내용)",
        question,
        re.I,
    ):
        return False

    if is_meeting_outcome_question(question, row) and not _is_narrow_meeting_fact_question(
        question
    ):
        return True

    intent = resolve_internal_intent(question, row, legacy_category=legacy_category)
    if top == TOP_LEVEL_ENV:
        # Preserve the agreed environmental response formats, including DCS
        # data-quality and alternative-fuel safety briefings.  A technical
        # fact that only happens to mention CII/LNG/IGF is generated normally.
        if intent == "data_quality_verification":
            return True
        return bool(
            re.search(
                r"최신|최근|동향|주요\s*내용|운항.{0,12}(?:영향|대응)|"
                r"규제\s*보고|보고\s*준비|업무\s*영향|대응\s*사항",
                question,
                re.I,
            )
        )
    if top == TOP_LEVEL_AUTO:
        return bool(
            re.search(
                r"MASS\s*Code.{0,40}(?:일정|로드맵|mandatory|강제|비강제|채택|승인|결정|요약|정리)|"
                r"(?:일정|로드맵|mandatory|강제|비강제|채택|승인|결정|요약|정리).{0,40}MASS\s*Code",
                question,
                re.I,
            )
        )
    return False
