from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
RAG_SCRIPTS = ROOT / "rag" / "scripts"
if str(RAG_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RAG_SCRIPTS))

from advanced_mode import (  # noqa: E402
    AdvancedRerankConfig,
    _canonicalize_section_headings,
    _bulletize_section_prose,
    _class_document_status_instruction,
    _compact_simple_rule_lookup_answer,
    _drop_uncited_factual_bullets,
    _ensure_advanced_premise_verdict,
    _ensure_advanced_four_section_shell,
    _extract_json_object,
    _normalize_bullet_citations,
    _restore_required_sections,
    _valid_revised_answer,
    clear_advanced_cache,
    listwise_rerank,
    plan_retrieval_followups,
    retrieval_confidence,
    review_answer,
    should_plan_retrieval,
)


def _chunk(index: int, *, doc: str = "rules.pdf", text: str = "relevant clause"):
    return SimpleNamespace(
        chunk_id=f"c{index}",
        doc_id=doc,
        source="DNV",
        file_name=doc,
        page_number=index,
        clause_number=str(index),
        document_status="rule",
        text=f"{text} {index}",
        dense_score=1.0 / index,
        bm25_score=None,
        rrf_score=1.0 / (60 + index),
        reranker_score=None,
    )


def test_extract_json_object_accepts_fenced_response() -> None:
    assert _extract_json_object('```json\n{"ranked_ids":[2,1]}\n```') == {
        "ranked_ids": [2, 1]
    }


def test_class_document_status_instruction_separates_guides_from_imo_outcomes() -> None:
    class_instruction = _class_document_status_instruction(
        [
            {
                "document": "ABS Guide for Smart Functions for Marine Vessels.pdf",
                "clause": "1/3",
                "evidence": "This Guide applies to marine vessels and offshore units.",
            }
        ]
    )
    assert "IMO 회의의 Proposal·Report·Outcome가 아니다" in class_instruction
    assert "[미확정 규제]" in class_instruction

    meeting_instruction = _class_document_status_instruction(
        [
            {
                "document": "MSC 111-WP.1 - Draft Report.pdf",
                "clause": "12.4",
                "evidence": "The Committee approved rules in the draft guidelines.",
            }
        ]
    )
    assert meeting_instruction == ""


def test_citation_groups_are_atomic_and_deduplicated() -> None:
    answer = (
        "## 1) 핵심 요약\n"
        "- 수소 지침을 승인했습니다. [1, 11] [1]\n"
        "- 액화수소 결의를 채택했습니다. [12]"
    )
    normalized = _normalize_bullet_citations(answer)
    assert "[1][11]" in normalized
    assert "[1, 11]" not in normalized
    assert normalized.count("[1]") == 1
    assert "[12]" in normalized


def test_advanced_headings_are_canonicalized() -> None:
    answer = (
        "## 1) 핵심 요약\n- 요약 [1]\n"
        "## 2) 설계 검토 체크리스트 (선급 규정 기반)\n- 조치 [1]\n"
        "## 3) 미확정 규제 및 추후 확인 필요사항\n- 확인 [1]\n"
        "## 4) 관련 선급 Rule / Guidance\n- 문서 [1]"
    )
    normalized = _canonicalize_section_headings(answer)
    assert "## 2) 선박 운항/업무 영향" in normalized
    assert "## 3) 추후 확인 필요사항" in normalized
    assert "설계 검토 체크리스트 (선급 규정 기반)" not in normalized


def test_short_negative_answer_is_wrapped_in_four_section_contract() -> None:
    wrapped = _ensure_advanced_four_section_shell(
        "## 답변\n\n- 지정 문서에서 확정표를 확인할 수 없습니다."
    )
    assert wrapped.count("## ") == 4
    assert "## 1) 핵심 요약" in wrapped
    assert "확정표를 확인할 수 없습니다" in wrapped


def test_auditor_prose_is_bulletized_and_negative_premise_gets_verdict() -> None:
    answer = _bulletize_section_prose(
        "## 1) 핵심 요약\nAER와 cgDIST는 공급 기반 지표입니다. [1]\n"
        "## 2) 선박 운항/업무 영향\n- 별도 영향 없음"
    )
    assert "- AER와 cgDIST" in answer
    repaired = _ensure_advanced_premise_verdict(
        "'AER가 과징금 지표다'라는 전제가 맞는지 검증하고 틀리면 바로잡아줘.",
        answer,
        signal_text="과징금 계산 지표로 명시되어 있지 않습니다.",
    )
    assert "- 전제는 맞지 않습니다. AER와 cgDIST" in repaired


def test_simple_rule_lookup_is_compacted_to_three_cited_facts() -> None:
    answer = (
        "## 1) 핵심 요약\n"
        "- 첫 번째 직접 문서와 범위입니다. [1]\n"
        "- 두 번째 직접 문서와 범위입니다. [2]\n"
        "- 주변 참고 문서입니다. [3]\n\n"
        "## 2) 선박 운항/업무 영향\n"
        "- 질문하지 않은 상세 검사 절차입니다. [4]\n\n"
        "## 3) 추후 확인 필요사항\n"
        "- 질문하지 않은 서비스 평가입니다. [5]\n\n"
        "## 4) 관련 선급 Rule / Guidance\n"
        "- 직접 사용한 Guide, p.1입니다. [1]\n"
        "- 부수 문서입니다. [6]"
    )
    compact = _compact_simple_rule_lookup_answer(answer)
    assert "첫 번째 직접 문서" in compact
    assert "두 번째 직접 문서" in compact
    assert "주변 참고 문서" not in compact
    assert "상세 검사 절차" not in compact
    assert "서비스 평가" not in compact
    assert len(re.findall(r"(?m)^[-*].*\[\d+\]", compact)) == 3


def test_uncited_optional_auditor_claim_is_dropped_but_group_label_survives() -> None:
    revised = (
        "## 2) 선박 운항/업무 영향\n"
        "- **데이터 점검**:\n"
        " - AER와 cgDIST를 비교합니다. [2]\n"
        "- 공식 체크리스트 양식이라고 추정합니다."
    )
    cleaned = _drop_uncited_factual_bullets(revised)
    assert "**데이터 점검**:" in cleaned
    assert "AER와 cgDIST" in cleaned
    assert "공식 체크리스트" not in cleaned


def test_listwise_rerank_reorders_candidates_and_preserves_pool() -> None:
    chunks = [_chunk(index, doc=f"doc{index}.pdf") for index in range(1, 9)]
    payload = {"ranked_ids": [8, 7, 6, 5, 4, 3, 2, 1], "coverage": ["요건"]}
    with patch(
        "advanced_mode._ollama_json",
        return_value=(payload, {"ok": True, "elapsed_seconds": 0.1}),
    ):
        retrieved, pool, meta = listwise_rerank(
            "요건은?",
            chunks[:4],
            chunks,
            config=AdvancedRerankConfig(candidate_limit=8, output_k=6),
        )
    assert meta["used"] is True
    assert [chunk.chunk_id for chunk in retrieved] == ["c8", "c7", "c6", "c5", "c4", "c3"]
    assert {chunk.chunk_id for chunk in pool} == {f"c{i}" for i in range(1, 9)}


def test_invalid_short_ranking_falls_back_to_accurate_order() -> None:
    chunks = [_chunk(index) for index in range(1, 9)]
    with patch(
        "advanced_mode._ollama_json",
        return_value=({"ranked_ids": [8]}, {"ok": True}),
    ):
        retrieved, _pool, meta = listwise_rerank("질문", chunks[:4], chunks)
    assert meta["used"] is False
    assert [chunk.chunk_id for chunk in retrieved] == ["c1", "c2", "c3", "c4"]


def test_outcome_literal_evidence_cannot_be_discarded_by_listwise_model() -> None:
    discussion = _chunk(
        1,
        doc="MSC 111-WP.1 - Draft Report.pdf",
        text=(
            "interim guidelines for the safety of ships using hydrogen as fuel "
            "were discussed by several delegations"
        ),
    )
    hydrogen = _chunk(
        2,
        doc="MSC 111-WP.1 - Draft Report.pdf",
        text=(
            "The Committee approved the draft interim guidelines for the safety "
            "of ships using hydrogen as fuel"
        ),
    )
    liquid = _chunk(
        3,
        doc="MSC 111-WP.1 - Draft Report.pdf",
        text=(
            "The Committee adopted a resolution on the Revised Interim "
            "Recommendations for carriage of liquefied hydrogen in bulk"
        ),
    )
    workplan = _chunk(
        4,
        doc="MSC 111-12 - Report of the twelfth session.pdf",
        text="The Committee approved the draft work plan for alternative fuels",
    )
    fillers = [_chunk(index, doc=f"doc{index}.pdf") for index in range(5, 10)]
    chunks = [discussion, hydrogen, liquid, workplan, *fillers]
    # Deliberately omit two protected outcome candidates from the model output.
    payload = {"ranked_ids": [9, 8, 7, 6, 5, 1], "coverage": ["일부 결과"]}
    with patch(
        "advanced_mode._ollama_json",
        return_value=(payload, {"ok": True, "elapsed_seconds": 0.1}),
    ):
        retrieved, _pool, meta = listwise_rerank(
            "MSC 111 결과 중 연료 안전·위험평가 관련만 추려줘.",
            chunks[:4],
            chunks,
            config=AdvancedRerankConfig(candidate_limit=9, output_k=8),
        )
    ids = [chunk.chunk_id for chunk in retrieved]
    assert ids[:3] == ["c2", "c3", "c4"]
    assert meta["protected_outcome_chunk_ids"] == ["c2", "c3", "c4"]


def test_revised_answer_rejects_out_of_range_or_uncited_claims() -> None:
    ok, reason = _valid_revised_answer(
        "## 1) 핵심 요약\n- 기존 답입니다. [1]",
        "## 1) 핵심 요약\n- 새 사실입니다. [3]" + " 충분한 설명" * 8,
        {1, 2},
    )
    assert not ok and reason == "out_of_range_citation"
    ok, reason = _valid_revised_answer(
        "## 1) 핵심 요약\n- 기존 답입니다. [1]",
        "## 1) 핵심 요약\n- 새로운 수치 요건입니다." + " 충분한 설명" * 8,
        {1},
    )
    assert not ok and reason == "citations_removed"


def test_review_answer_accepts_only_valid_local_revision() -> None:
    original = "## 1) 핵심 요약\n- 기존 요건입니다. [1]"
    revised = (
        "## 1) 핵심 요약\n- 수정된 요건과 적용 조건을 근거 범위 안에서 설명합니다. [1]\n"
        "## 2) 선박 운항/업무 영향\n- 설계 검토 시 해당 조건을 확인해야 합니다. [1]\n"
        "## 3) 추후 확인 필요사항\n- 최신 개정판은 원문 확인이 필요합니다. [1]"
    )
    evidence = [
        {
            "citation_id": "[1]",
            "file_name": "rule.pdf",
            "page": 3,
            "chunk_preview": "수정된 요건과 설계 검토 조건",
        }
    ]
    with patch(
        "advanced_mode._ollama_json",
        return_value=(
            {"decision": "revise", "issues": ["누락"], "revised_answer": revised},
            {"ok": True, "elapsed_seconds": 0.1},
        ),
    ):
        answer, meta = review_answer("요건과 영향을 알려줘", original, evidence)
    assert answer == revised
    assert meta["revision_accepted"] is True


def test_complex_question_plan_returns_only_missing_followups() -> None:
    clear_advanced_cache()
    chunks = [_chunk(1, text="MASS non-mandatory Code adopted")]
    payload = {
        "facets": ["비강제 Code 결정", "mandatory 일정"],
        "covered": ["비강제 Code 결정"],
        "missing": ["mandatory 일정"],
        "followup_queries": ["MSC 111 mandatory MASS Code 2030 2032 2036"],
        "confidence": 0.55,
    }
    with patch(
        "advanced_mode._ollama_json",
        return_value=(payload, {"ok": True, "elapsed_seconds": 0.1}),
    ):
        plan, meta = plan_retrieval_followups(
            "MSC 111에서 MASS Code 결정과 향후 mandatory 일정을 정리해줘",
            chunks,
            chunks,
        )
    assert meta["used"] is True
    assert plan.missing == ("mandatory 일정",)
    assert plan.followup_queries == ("MSC 111 mandatory MASS Code 2030 2032 2036",)


def test_numeric_missing_ordinals_are_replaced_by_followup_labels() -> None:
    clear_advanced_cache()
    payload = {
        "facets": ["MSC 결과", "선급 체크리스트"],
        "covered": [],
        "missing": ["14", "16", "17"],
        "followup_queries": [
            "ammonia cargo as fuel safety requirements",
            "ammonia detection system safety measures",
        ],
        "confidence": 0.5,
    }
    with patch(
        "advanced_mode._ollama_json",
        return_value=(payload, {"ok": True, "elapsed_seconds": 0.1}),
    ):
        plan, _meta = plan_retrieval_followups(
            "암모니아 연료선의 개념승인을 위한 MSC 결과와 선급 설계 체크리스트, 미확정 규제를 작성해줘.",
            [_chunk(1)],
            [_chunk(1)],
        )
    assert plan.missing == plan.followup_queries
    assert all(not value.isdigit() for value in plan.missing)


def test_simple_fact_skips_planning_call() -> None:
    assert should_plan_retrieval("DNV-CP-0399의 적용 전압은?") is False
    with patch("advanced_mode._ollama_json") as call:
        plan, meta = plan_retrieval_followups("DNV-CP-0399의 적용 전압은?", [], [])
    call.assert_not_called()
    assert meta["reason"] == "simple_question"
    assert plan.confidence == 1.0


def test_confidence_gate_flags_unsatisfied_named_document() -> None:
    confidence = retrieval_confidence(
        "DNV-CP-0399의 적용 범위는?",
        [_chunk(1, doc="other-rule.pdf", text="generic unrelated requirement")],
    )
    assert confidence["named_document_satisfied"] is False
    assert confidence["level"] == "low"


def test_revision_with_fifth_section_is_rejected() -> None:
    original = (
        "## 1) 핵심 요약\n- 사실입니다. [1]\n"
        "## 2) 선박 운항/업무 영향\n- 영향입니다. [1]\n"
        "## 3) 추후 확인 필요사항\n- 확인입니다. [1]\n"
        "## 4) 관련 선급 Rule / Guidance\n- 규정입니다. [1]"
    )
    revised = original + "\n## 5) 일정\n- 2030년입니다. [1]"
    ok, reason = _valid_revised_answer(original, revised, {1})
    assert ok is False
    assert reason == "unexpected_sections"


def test_missing_fourth_section_is_restored_from_safe_original() -> None:
    original = (
        "## 1) 핵심 요약\n- 기존입니다. [1]\n"
        "## 2) 선박 운항/업무 영향\n- 없음\n"
        "## 3) 추후 확인 필요사항\n- 확인 필요\n"
        "## 4) 관련 선급 Rule / Guidance\n- 검색 근거에서 확인되지 않음"
    )
    revised = (
        "## 1) 핵심 요약\n- 2030년 채택, 2032년 발효 목표입니다. [1]\n"
        "## 2) 선박 운항/업무 영향\n- 없음\n"
        "## 3) 추후 확인 필요사항\n- 확인 필요"
    )
    restored = _restore_required_sections(original, revised)
    assert "2030년 채택" in restored
    assert "## 4) 관련 선급 Rule / Guidance" in restored


def test_app_exposes_advanced_and_forces_gemma() -> None:
    import app

    assert app._normalize_latency_mode("advanced") == "advanced"
    assert app._effective_llm_model("advanced", "llama3.1:8b") == "gemma4:12b"
    config = app.demo.get_config_file()
    radios = [
        component
        for component in config.get("components", [])
        if component.get("type") == "radio"
        and component.get("props", {}).get("label") == "문서 답변 모드"
    ]
    assert radios
    assert "advanced" in [value for _label, value in radios[0]["props"]["choices"]]
