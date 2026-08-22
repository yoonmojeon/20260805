"""The structured meeting answer must not be the fallback for every question.

``classify_question_category`` answers ``trend_summary`` when nothing matches, so
a KR class-rule question used to be rendered with "회의 결정·결과" bullets.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAG_SCRIPTS = ROOT / "rag" / "scripts"
for path in (ROOT, RAG_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from meeting_category_profile import (  # noqa: E402
    has_meeting_cue,
    uses_structured_meeting_answer,
)
from question_classifier import classify_question_category  # noqa: E402
from fast_question_classifier import classify_fast_question_type  # noqa: E402
from rag_inprocess import _fast_route_defers_or_skips_llm  # noqa: E402


def test_meeting_questions_keep_the_structured_answer():
    for question in (
        "환경규제 대응과 관련된 최신 MEPC 회의 주요 내용을 정리해줘.",
        "MSC 111의 주요 결과를 3개 항목으로 요약해줘.",
        "최신 MEPC 회의에서 IMO DCS 제출 데이터의 오류나 누락은 어떻게 다뤘어?",
        "MSC 111에서 MASS Code와 관련된 핵심 결정사항을 요약해줘.",
    ):
        assert has_meeting_cue(question), question
        assert uses_structured_meeting_answer(
            {"question": question}, legacy_category="trend_summary"
        ), question


def test_class_rule_question_without_a_meeting_cue_is_not_a_meeting_answer():
    question = "잔류응력을 측정한 경우 시험 결과 문서에 어떤 정보를 기록해야 하는가?"

    assert not has_meeting_cue(question)
    assert not uses_structured_meeting_answer(
        {"question": question}, legacy_category="trend_summary"
    )


def test_named_society_technical_fact_is_not_rule_catalogue_lookup():
    question = "DNV 규정에 따라 운용 경험이 부족한 제품에 어떤 환경 시험이 요구됩니까?"
    assert classify_question_category(question, {}) != "rule_lookup"


def test_explicit_rule_guidance_discovery_stays_rule_lookup():
    for question in (
        "DNV에서 자율운항 관련 Rule/Guidance를 찾아줘.",
        "LR에서 대체연료 관련 Rule/Guidance를 찾아줘.",
    ):
        assert classify_question_category(question, {}) == "rule_lookup"


def test_narrow_meeting_paper_fact_uses_grounded_generation():
    for question, category in (
        ("MEPC 78 회의자료에 따르면 LNG 탱크 압력 상한은 얼마인가?", "trend_summary"),
        ("MSC 111 회의자료에서 경보 기록의 보존 기간은 얼마인가?", "trend_summary"),
        ("MSC 111 MASS Record에 포함되는 필드는 무엇인가?", "autonomous"),
        ("MEPC 84/3 문서의 ECA 개정안은 어느 회의에서 승인되었습니까?", "env_regulation"),
        ("MEPC 84-7-23의 LNG Cslip 기본값은 어떤 값을 사용합니까?", "env_regulation"),
    ):
        assert has_meeting_cue(question)
        assert not uses_structured_meeting_answer(
            {"question": question}, legacy_category=category
        )


def test_exact_meeting_paper_fact_uses_general_fast_prompt():
    for question in (
        "MEPC 84/3 문서의 ECA 개정안은 어느 회의에서 승인되었습니까?",
        "MEPC 84-7-23 문서의 Cslip 개정은 어떤 연구 결과에 기반합니까?",
    ):
        assert classify_fast_question_type(question, {}) == "general_question"

    assert (
        classify_fast_question_type("MEPC 84/3 문서의 주요 결과를 요약해줘.", {})
        == "meeting_outcome_question"
    )


def test_session_scoped_concrete_fact_uses_general_fast_prompt():
    row = {"category": "env_regulation", "_top_level_category": "trend_summary"}
    assert (
        classify_fast_question_type(
            "MEPC 회의자료에 따르면 BWMS 주요 구성 요소가 수정되면 당국은 어떻게 조치합니까?",
            row,
        )
        == "general_question"
    )
    assert (
        classify_fast_question_type(
            "MSC 회의자료에 따르면 MASS EBP의 증거는 어떤 연구를 포함합니까?",
            row,
        )
        == "general_question"
    )


def test_exact_rule_document_fact_is_not_catalogue_lookup():
    for question in (
        "DNV-CG-0138 문서에서 racking analysis 시 고려할 하중은 무엇입니까?",
        "Notice No.1에서 anchor handling winch가 견뎌야 할 하중은 얼마입니까?",
    ):
        assert classify_question_category(question, {}) != "rule_lookup"


def test_explicit_environment_category_still_uses_the_structured_answer():
    # Category 2/3 come from a real pattern match, never from the fallback.
    assert uses_structured_meeting_answer(
        {"question": "우리 배 배출 보고 준비사항"}, legacy_category="env_regulation"
    )


def test_table_questions_never_use_the_meeting_answer():
    assert not uses_structured_meeting_answer(
        {"question": "안전사용하중은 몇 톤인가?", "_table_qa": True},
        legacy_category="trend_summary",
    )


def test_multi_domain_msc_summary_does_not_collapse_to_mass_only():
    question = "MSC 111에서 안전·항해·자율운항 관련 채택 또는 승인 결과를 구분해 요약해줘."
    assert classify_question_category(question, {}) == "trend_summary"


def test_named_alternative_fuel_meeting_questions_use_environment_category():
    for question in (
        "MSC 111에서 암모니아 연료 관련 IGF/IGC Code 상태를 알려줘.",
        "MSC 111의 수소연료 선박 안전지침 결정을 알려줘.",
    ):
        assert classify_question_category(question, {}) == "env_regulation"


def test_fast_deterministic_routes_do_not_pay_an_eager_llm_warmup():
    assert _fast_route_defers_or_skips_llm(
        {"question": "MSC 111의 주요 결과를 3개 항목으로 요약해줘."},
        "structured_meeting",
    )
    assert _fast_route_defers_or_skips_llm(
        {
            "question": "DNV에서 자율운항 또는 Smart Vessel 관련 Rule/Guidance를 찾아줘.",
            "category": "rule_lookup",
        },
        "rule_guidance_lookup",
    )
    assert not _fast_route_defers_or_skips_llm(
        {"question": "일반 문서에서 핵심 내용을 설명해줘.", "category": "general"},
        "fast_rag",
    )


if __name__ == "__main__":
    test_meeting_questions_keep_the_structured_answer()
    test_class_rule_question_without_a_meeting_cue_is_not_a_meeting_answer()
    test_explicit_environment_category_still_uses_the_structured_answer()
    test_table_questions_never_use_the_meeting_answer()
    print("ok")
