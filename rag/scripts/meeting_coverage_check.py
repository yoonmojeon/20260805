"""Coverage checks and warning flags for V01–V05 meeting answers."""
from __future__ import annotations

import re
from typing import Any

CITATION_RE = re.compile(r"\[(\d+)\]")


def _has_any(text: str, terms: list[str]) -> bool:
    low = (text or "").lower()
    return any(t.lower() in low for t in terms)


def check_v01_coverage(answer: str, chunks: list[Any]) -> tuple[dict[str, bool], list[str]]:
    checks = {
        "mepc_related": _has_any(answer, ["mepc", "84"]),
        "ghg_content": _has_any(answer, ["ghg", "배출", "emission", "net-zero", "gfi"]),
        "cii_or_framework": _has_any(answer, ["cii", "seemp", "marpol", "framework", "annex vi"]),
    }
    warnings: list[str] = []
    if not checks["ghg_content"]:
        warnings.append("no_regulation_keyword_hit")
    if not CITATION_RE.search(answer):
        warnings.append("citation_missing")
    return checks, warnings


def check_v02_coverage(answer: str, chunks: list[Any], *, expected_count: int = 3) -> tuple[dict[str, bool], list[str]]:
    from meeting_answer_dedup import detect_section1_topics

    s1 = answer.split("## 2)")[0] if "## 2)" in answer else answer
    s1_bullets = len(re.findall(r"^-\s+", s1, re.M))
    topics = detect_section1_topics(s1)
    unique_topics = len(set(topics))
    mass_count = sum(1 for t in topics if t == "mass_code")
    checks = {
        "exactly_n_items": s1_bullets == expected_count,
        "distinct_topics": len(topics) == expected_count and len(set(topics)) == expected_count,
        "mass_at_most_one": mass_count <= 1,
        "has_outcome_signal": _has_any(answer, ["채택", "승인", "adopted", "approved", "결정", "endorsed"]),
        "has_citation": bool(CITATION_RE.search(answer)),
        "has_mass_or_key_topic": _has_any(answer, ["mass", "ghg", "lrit", "vdes", "연료", "fuel", "solas", "gmdss"]),
    }
    warnings: list[str] = []
    if not checks["exactly_n_items"]:
        warnings.append("answer_count_mismatch")
    if not checks["distinct_topics"]:
        warnings.append("duplicate_topic")
    if not checks["mass_at_most_one"]:
        warnings.append("duplicate_topic")
    if not checks["has_outcome_signal"]:
        warnings.append("no_outcome_signal")
    return checks, warnings


def check_v03_coverage(answer: str, chunks: list[Any]) -> tuple[dict[str, bool], list[str]]:
    checks = {
        "ghg_or_cii": _has_any(answer, ["ghg", "cii", "carbon intensity", "탄소"]),
        "reporting": _has_any(answer, ["reporting", "보고", "verification", "compliance", "data collection"]),
        "operation_impact": _has_any(answer, ["운항", "operational", "선박", "fleet", "seemp"]),
        "followup": _has_any(answer, ["추가 확인", "미확정", "후속", "확정 여부"]),
    }
    warnings: list[str] = []
    if not checks["operation_impact"]:
        warnings.append("missing_operation_impact")
    if not checks["reporting"]:
        warnings.append("missing_reporting_impact")
    if not checks["followup"]:
        warnings.append("missing_followup_items")
    return checks, warnings


def check_v04_coverage(answer: str, chunks: list[Any]) -> tuple[dict[str, bool], list[str]]:
    s1 = answer.split("## 2)")[0] if "## 2)" in answer else answer
    fuels = ["ammonia", "hydrogen", "methanol", "lng", "암모니아", "수소", "메탄올"]
    checks = {
        "alternative_fuel": _has_any(answer, ["alternative fuel", "대체연료", "low-flashpoint", "저인화점"]),
        "ghg_safety": _has_any(answer, ["ghg", "safety", "안전"]),
        "fuel_keyword": _has_any(answer, fuels),
        "guideline_or_conclusion": _has_any(answer, ["guideline", "interim", "결론", "논의", "approved", "agreed", "채택", "승인"]),
        "no_mass_in_summary": not _has_any(s1, ["mass code", "자율운항", "maritime autonomous"]),
    }
    warnings: list[str] = []
    if not checks["alternative_fuel"]:
        warnings.append("no_regulation_keyword_hit")
    if not checks["ghg_safety"]:
        warnings.append("missing_safety_impact")
    if not checks["guideline_or_conclusion"]:
        warnings.append("weak_evidence")
    if not checks["no_mass_in_summary"]:
        warnings.append("wrong_topic_in_answer")
    return checks, warnings


def check_v05_coverage(answer: str, chunks: list[Any]) -> tuple[dict[str, bool], list[str]]:
    s1 = answer.split("## 2)")[0] if "## 2)" in answer else answer
    checks = {
        "mass_code": _has_any(answer, ["mass code", "mass", "자율"]),
        "msc_111": _has_any(answer, ["msc 111", "msc111", "111"]),
        "non_mandatory": _has_any(answer, ["non-mandatory", "비강제", "non mandatory", "voluntary"]),
        "mandatory_timeline": _has_any(answer, ["mandatory", "일정", "timeline", "entry into force", "experience-building", "추가 확인"]),
        "three_sub_items": all(
            k in s1 for k in ("결정", "non-mandatory", "mandatory")
        ) or _has_any(s1, ["핵심 결정", "non-mandatory", "experience-building"]),
        "has_citation": bool(CITATION_RE.search(answer)),
    }
    warnings: list[str] = []
    if not checks["mass_code"]:
        warnings.append("missing_mass_code")
    if not checks["non_mandatory"] and not checks["mandatory_timeline"]:
        warnings.append("missing_non_mandatory")
    if "mandatory" in answer.lower() and "추가 확인" not in answer and not _has_any(
        answer, ["experience-building", "work plan", "timeline", "일정"]
    ):
        warnings.append("schedule_not_supported_by_evidence")
    return checks, warnings


def run_coverage_check(
    question_id: str,
    answer: str,
    chunks: list[Any],
    *,
    row: dict | None = None,
) -> tuple[dict[str, bool], list[str]]:
    row = row or {}
    qid = question_id.upper()
    if qid == "V01":
        return check_v01_coverage(answer, chunks)
    if qid == "V02":
        n = int(row.get("outcome_item_count") or 3)
        return check_v02_coverage(answer, chunks, expected_count=n)
    if qid == "V03":
        return check_v03_coverage(answer, chunks)
    if qid == "V04":
        return check_v04_coverage(answer, chunks)
    if qid == "V05":
        return check_v05_coverage(answer, chunks)
    return {}, []
