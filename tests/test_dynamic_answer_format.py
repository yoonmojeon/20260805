from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "rag" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dynamic_answer_format import (  # noqa: E402
    AnswerFormatDecision,
    choose_answer_format,
    ensure_explicit_premise_verdict,
    render_answer_format,
)


FOUR_SECTION = """## 1) 핵심 요약
- 정격은 1 kV 및 3 kV이다. [1]

## 2) 선박 운항/업무 영향
- 검색 근거에서 확인되지 않음

## 3) 추후 확인 필요사항
- 추가 확인 필요사항이 별도로 식별되지 않았습니다.

## 4) 관련 선급 Rule / Guidance
- DNV-CP-0399, 1.1 Objective. [1]
"""


def test_short_fact_removes_empty_template_sections_without_rewriting_claim():
    decision = AnswerFormatDecision("short_factual", "test")
    rendered = render_answer_format(FOUR_SECTION, decision)
    assert rendered == "## 답변\n\n- 정격은 1 kV 및 3 kV이다. [1]"


def test_meeting_timeline_preserves_four_section_contract():
    decision = choose_answer_format(
        "MSC 111에서 MASS Code 결정과 mandatory code 일정을 정리해줘."
    )
    assert decision.kind == "meeting_timeline"
    assert render_answer_format(FOUR_SECTION, decision) == FOUR_SECTION


def test_rule_discovery_uses_compact_rule_guide_format():
    decision = choose_answer_format("DNV에서 자율운항 관련 Rule/Guidance를 찾아줘.")
    assert decision.kind == "rule_guide"
    rendered = render_answer_format(FOUR_SECTION, decision)
    assert "## 관련 Rule / Guidance" in rendered
    assert "선박 운항/업무 영향" not in rendered
    assert "DNV-CP-0399" in rendered


def test_document_guide_keeps_grounded_usage_and_reference_bullets():
    decision = choose_answer_format(
        "DNV에서 자율운항 관련 Rule/Guidance를 찾아줘.",
        {"_question_profile": {"answer_style": "document_cards"}},
    )
    assert decision.kind == "document_guide"
    rendered = render_answer_format(FOUR_SECTION, decision)
    assert "### 문서 안내" in rendered
    assert "### 관련 참조 문서" in rendered
    assert "DNV-CP-0399" in rendered


def test_negative_cited_finding_gets_explicit_premise_verdict():
    answer = """## 1) 핵심 요약

- 문서는 해당 최종 규칙이 채택됐다는 내용을 포함하고 있지 않습니다. [1]
"""
    repaired = ensure_explicit_premise_verdict(
        "'최종 규칙이 채택됐다'라는 전제가 맞는지 검증해줘.", answer
    )
    assert "- 전제는 맞지 않습니다." in repaired
    assert repaired.endswith("[1]")


def test_premise_verdict_survives_when_contract_removed_uncited_verdict_line():
    answer = "## 답변\n\n- 지표는 2019년부터 2024년까지 비교합니다. [1]"
    pre_contract = (
        "- 전제는 맞지 않습니다.\n"
        "- 지표는 2019년부터 2024년까지 비교합니다. [1]"
    )
    repaired = ensure_explicit_premise_verdict(
        "'2022년만 비교한다'는 전제가 맞는지 검증해줘.",
        answer,
        pre_contract,
    )
    assert "- 전제는 맞지 않습니다." in repaired
    assert repaired.endswith("[1]")


def test_negative_hint_can_restore_verdict_without_explicit_hint_phrase():
    answer = "## 답변\n\n- AER는 운송 작업 대리치 기반 지표입니다. [1]"
    pre_contract = "- 과징금 지표로 사용된다는 내용은 확인되지 않았습니다. [1]"
    repaired = ensure_explicit_premise_verdict(
        "'AER는 과징금 지표다'라는 전제가 맞는지 검증해줘.",
        answer,
        pre_contract,
    )
    assert "- 전제는 맞지 않습니다." in repaired


def test_verified_absent_lookup_is_presented_as_short_fact():
    decision = choose_answer_format(
        "지정 문서에서 목록을 찾아 근거로 알려줘.",
        {"_specific_lookup_verification": True},
    )
    assert decision.kind == "short_factual"
