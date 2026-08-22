from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
RAG_SCRIPTS = ROOT / "rag" / "scripts"
for path in (ROOT, RAG_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from retrieval_query_analysis import (
    analyze_query,
    detect_excluded_sources,
    detect_named_sources,
    is_meeting_outcome_question,
)
from evidence_planner import build_evidence_plan
from rule_guidance_accurate import (
    _build_definition_extractive_answer,
    _ensure_named_rule_facts,
    _is_definition_lookup,
    _normalize_rule_translation,
    exact_rule_fact_slots,
    is_exact_rule_fact_question,
)
from question_classifier import classify_question_category
from imo_doc_registry import exact_doc_ids_for_query
from retrieval_verification import detect_narrow_doc_id
from retrieval_search import resolve_explicit_query_doc_id
from rag_society_filter import filter_pool_for_source_constraints
from rag_query_router import enrich_row_for_routing, is_rule_guidance_lookup
from fast_question_classifier import classify_fast_question_type
from fast_retrieval import select_rule_slots
from question_requirements import analyze_requirements


def test_document_code_exclusion_and_only_source_survive_analysis():
    signals = analyze_query(
        "DNV-CG-0264는 제외하고 ABS autonomous Requirements만 답해줘."
    )
    assert signals.excluded_sources == ["DNV"]
    assert signals.constrained_sources == ["ABS"]
    assert signals.named_sources == ["ABS"]
    assert detect_narrow_doc_id(
        "DNV-CG-0264는 제외하고 ABS autonomous Requirements만 답해줘.", {}
    ) is None


def test_dnv_cp_code_resolves_to_exact_collection_document():
    class FakeCollection:
        def get(self, **kwargs):
            assert kwargs["where"] == {"file_name": "DNV-CP-0399.pdf"}
            return {"metadatas": [{"doc_id": "dnv_cp_0399"}]}

    question = "DNV-CP-0399의 형식승인 적용 범위는?"
    assert resolve_explicit_query_doc_id(
        FakeCollection(), question, analyze_query(question)
    ) == "dnv_cp_0399"


def test_final_source_constraints_remove_excluded_society():
    pool = [
        SimpleNamespace(source="ABS", meta={}),
        SimpleNamespace(source="DNV", meta={}),
    ]
    filtered = filter_pool_for_source_constraints(
        pool, allowed_sources=["ABS"], excluded_sources=["DNV"]
    )
    assert [chunk.source for chunk in filtered] == ["ABS"]
from retrieval_search import (
    _direct_priority_rule_doc_ids,
    _document_route_candidates,
    _identifier_matches_filename,
    _priority_rule_file_names,
    _query_exact_identifier_hits,
    enrich_query_for_embedding,
    extract_exact_identifiers,
    extract_sparse_feature_terms,
    extract_sparse_latin_terms,
    extract_translated_feature_terms,
    feature_fallback_relevance_score,
    query_with_hybrid_ranking,
    rank_scoped_sparse_rows,
)
from table_qa_answer import build_deterministic_table_answer, verify_row_column_intersection
from meeting_structured_answer import (
    _generic_committee_outcome_claim,
    _section1_meeting_outcome,
)
from rag_answer_lib import (
    PREMISE_VERIFICATION_RE,
    SPECIFIC_DOCUMENT_LOOKUP_RE,
    _generate_specific_lookup_answer,
)
from build_text_eval_v3 import integration_secondary_id


def test_exact_identifiers_cover_sparse_maritime_codes():
    found = extract_exact_identifiers(
        "MEPC 84/7/14의 tcorr 및 AC-SD와 DNV-CG-0264를 확인해줘"
    )
    normalized = {item.lower() for item in found}
    assert "mepc 84/7/14" in normalized
    assert "tcorr" in normalized
    assert "ac-sd" in normalized
    assert "dnv-cg-0264" in normalized


def test_resolution_identifier_does_not_hard_route_to_meeting_source():
    assert detect_named_sources(
        "IMO Resolution MSC.288(87)에 따른 대체 시스템 승인 요건은?"
    ) == []
    assert detect_named_sources(
        "resolution MEPC.355(78)의 적용 조건은?"
    ) == []
    assert detect_named_sources("MSC 111의 주요 결과는?") == ["MSC"]


def test_exact_document_code_resolves_filename_metadata():
    mepc = exact_doc_ids_for_query("MEPC 84-7-23 문서의 기본값은 무엇인가?")
    dnv = exact_doc_ids_for_query("DNV-CP-0069 문서의 시험편 위치는?")
    base_item = exact_doc_ids_for_query("MEPC 84/3 문서의 개정안은?")
    assert any("mepc_84_7_23" in doc_id for doc_id in mepc)
    assert any("dnv_cp_0069" in doc_id for doc_id in dnv)
    assert len(base_item) == 1
    assert "mepc_84_3_amendments" in base_item[0]
    assert detect_narrow_doc_id("MEPC 84-7-23 문서의 기본값은?", {}) == mepc[0]


def test_sparse_feature_term_extracts_korean_compound_not_question_intent():
    assert extract_sparse_feature_terms("방식조치의 요건과 예외를 알려줘") == ["방식조치"]


def test_korean_maritime_concepts_get_selective_source_language_aliases():
    assert extract_translated_feature_terms(
        "선상 측정 완료 후 최종 보고서는 언제 제출합니까?"
    ) == [
        "within two (2) weeks after the job is terminated",
        "in addition to the measured values, the original scantlings, the minimum thickness and the substantial corrosion limits",
    ]
    assert extract_translated_feature_terms(
        "선상 측정 완료 후 최종 보고서의 필수 정보는?", limit=3
    )[-1] == "The report shall include a copy of the certificate of approval of the firm"
    assert extract_translated_feature_terms(
        "윤활 방식에 따른 원주 속도 조건은?"
    ) == ["Circumferential velocity should be"]
    assert extract_translated_feature_terms(
        "CA 챔버 진입 전 가스 제거 완료 확인 절차는?"
    ) == ["procedures for checking completed gas freeing prior to entry"]
    assert extract_translated_feature_terms(
        "구형 쉘의 편평도와 국부 곡률 반경 조건은?"
    ) == ["local outside curvature radius of Ro,l = 1.3"]
    assert extract_translated_feature_terms(
        "해상 LAN TA가 다른 규칙 준수도 보증합니까?"
    ) == ["will not confirm compliance with requirements in other parts of the rules"]
    assert extract_translated_feature_terms(
        "탱크 지지대 hardwood 등급과 variants 정의는?"
    ) == ["Defined by density, lamination", "variants: different numbers of plies"]
    assert extract_translated_feature_terms(
        "BBNJ 협정 관련 2026-2027년 협의·조정 의무를 위한 내부 메커니즘은?"
    ) == ["articulate and operationalize a clear internal mechanism"]
    assert extract_translated_feature_terms(
        "MSC 회의자료에 따르면 IACS Rec.165 Rev.1이 적용되는 대상과 그 목적은?",
        limit=3,
    ) == [
        "addresses designers, shipyards, technical managers responsible for calculations",
        "It also applies to deviations from CSR and other requirements",
        "addressed to Classification Societies",
    ]


def test_named_multiword_latin_feature_precedes_korean_request_word():
    class RecordingCollection:
        def __init__(self):
            self.calls = []

        def get(self, **kwargs):
            self.calls.append(kwargs)
            return {"ids": [], "metadatas": [], "documents": []}

    question = (
        "supply-based carbon intensity가 몇 퍼센트 감소했습니까?"
    )
    assert extract_sparse_latin_terms(question)[0] == "supply-based carbon intensity"
    collection = RecordingCollection()
    _raw, _scores, _identifiers, feature_terms = _query_exact_identifier_hits(
        collection,
        question,
        where=None,
        candidate_documents=["unrelated dense candidate"],
    )
    assert feature_terms == ["supply-based carbon intensity"]
    assert collection.calls[0]["where_document"] == {
        "$contains": "supply-based carbon intensity"
    }


def test_minimum_risk_condition_expands_to_current_fallback_state_term():
    query = "DNV-CG-0264에서 minimum risk condition 원칙은?"
    enriched = enrich_query_for_embedding(query, "intfloat/multilingual-e5-base")
    assert "fallback state" in enriched
    plan = build_evidence_plan(query, {"_internal_intent": "rule_lookup"})
    direct = next(slot for slot in plan.slots if slot.name == "specific_clause")
    assert "fallback state" in direct.terms
    assert "acceptable risk" in direct.terms


def test_fallback_state_is_not_translated_as_rear_state():
    chunk = SimpleNamespace(text="The vessel should enter and maintain a fallback state.")
    normalized = _normalize_rule_translation("선박은 후방 상태를 유지해야 한다.", [chunk])
    assert "폴백 상태(대체 안전상태)" in normalized
    assert "후방 상태" not in normalized


def test_exact_rule_fact_profile_distinguishes_scalar_from_inventory_question():
    exact = (
        "DNV-CP-0399의 형식승인(TA)은 어떤 정격의 전력 케이블과 "
        "제어·계측 회로용 케이블에 적용되나요?"
    )
    inventory = (
        "샌드위치 코어 재료의 형식 승인(TA)을 위해 제조사가 제출해야 하는 "
        "필수 문서 목록에는 어떤 항목들이 포함되는가?"
    )

    assert is_exact_rule_fact_question(exact)
    assert exact_rule_fact_slots(exact) == 2
    assert not is_exact_rule_fact_question(inventory)
    assert exact_rule_fact_slots("며칠이며 장소는 어디인가요?") == 2
    assert is_exact_rule_fact_question("감소 폭은 어느 정도인가요?")
    assert not is_exact_rule_fact_question(
        "비상전원은 어떤 요건을 갖추며 어느 장치들에 급전해야 합니까?"
    )
    assert not is_exact_rule_fact_question("IACS Rec.165가 적용되는 대상과 목적은 무엇인가요?")
    assert exact_rule_fact_slots("어떤 경우에 다른 시험 절차를 적용합니까?") == 1
    assert exact_rule_fact_slots("주최한 국가나 단체는 어디인가요?") == 3
    assert is_exact_rule_fact_question("두 가지 주요 이슈는 무엇인가요?")
    assert is_exact_rule_fact_question(
        "주요 구성 요소가 포함된 수정 시 당국은 어떻게 조치해야 합니까?"
    )
    assert is_exact_rule_fact_question("어느 단계이며 어떤 목표를 포함합니까?")
    assert exact_rule_fact_slots("어느 단계이며 어떤 목표를 포함합니까?") == 2


def test_named_revision_term_is_merged_without_adding_a_rule_bullet():
    answer = (
        "## 1) 핵심 요약\n\n"
        "- 운항범위를 초과하면 fallback state에 진입·유지할 수 있어야 합니다. [2]\n\n"
        "## 2) 선박 운항/업무 영향\n\n> 없음\n\n"
        "## 3) 추후 확인 필요사항\n\n- 적용 조건을 확인합니다. [2]\n\n"
        "## 4) 관련 선급 Rule / Guidance\n\n- DNV-CG-0264, clause 8.2 [2]"
    )
    chunks = [
        SimpleNamespace(
            text=(
                "Description: Replaced minimum risk condition (MRC) with the term "
                "fallback state. The definition remains the same."
            )
        ),
        SimpleNamespace(text="Maintain a safe state when exceeding the operational envelope."),
    ]
    updated = _ensure_named_rule_facts(
        answer,
        "DNV-CG-0264의 minimum risk condition 원칙은?",
        chunks,
    )
    assert "minimum risk condition" in updated
    assert "fallback state" in updated
    assert "정의는 동일" in updated
    assert len([line for line in updated.splitlines() if line.startswith("- ")]) == 3


def test_named_smart_vessel_catalogue_row_is_preserved():
    answer = (
        "## 1) 핵심 요약\n\n- DNV-CG-0264는 autonomous/remote guidance입니다. [2]\n\n"
        "## 2) 선박 운항/업무 영향\n\n> 없음\n\n"
        "## 3) 추후 확인 필요사항\n\n> 없음\n\n"
        "## 4) 관련 선급 Rule / Guidance\n\n- DNV-CG-0264 [2]"
    )
    chunks = [
        SimpleNamespace(
            text="Document code: DNV-RU-SHIP Pt.6 Ch.5 Sec.24 | Title: Smart vessel"
        ),
        SimpleNamespace(text="DNV-CG-0264 autonomous and remotely operated ships AROS"),
    ]
    updated = _ensure_named_rule_facts(
        answer,
        "DNV Smart Vessel 문서와 autonomous/remote vessel guidance를 구분해줘",
        chunks,
    )
    assert "DNV-RU-SHIP Pt.6 Ch.5 Sec.24" in updated
    assert "Smart vessel" in updated
    assert "DNV-CG-0264" in updated


def test_broad_dnv_autonomous_smart_pointer_excludes_incidental_guidance():
    answer = (
        "## 1) 핵심 요약\n\n"
        "- DNV-CG-0264와 DNV-CG-0508이 직접 관련됩니다. [2][3]\n\n"
        "## 2) 선박 운항/업무 영향\n\n> 없음\n\n"
        "## 3) 추후 확인 필요사항\n\n> 없음\n\n"
        "## 4) 관련 선급 Rule / Guidance\n\n"
        "- **DNV-CG-0557.pdf**, p.9: 일반 DNV 검색 결과입니다. [1]"
    )
    chunks = [
        SimpleNamespace(
            file_name="DNV-CG-0557.pdf", page_number=9,
            text="An incidental DNV class guideline.",
        ),
        SimpleNamespace(
            file_name="DNV-CG-0264.pdf", page_number=9,
            text="Autonomous and remotely operated ships and AROS.",
        ),
        SimpleNamespace(
            file_name="DNV-CG-0508.pdf", page_number=6,
            text="Document code: DNV-CG-0508 | Title: Smart vessel",
        ),
    ]
    updated = _ensure_named_rule_facts(
        answer,
        "DNV에서 자율운항 또는 Smart Vessel 관련 Rule/Guidance를 찾아줘.",
        chunks,
    )
    section4 = updated.split("## 4) 관련 선급 Rule / Guidance", 1)[1]
    assert "DNV-CG-0264" in section4
    assert "DNV-CG-0508" in section4
    assert "DNV-CG-0557" not in section4


def test_dnv_smart_instrument_slot_accepts_the_direct_cg_0508_document():
    plan = build_evidence_plan(
        "DNV에서 자율운항 또는 Smart Vessel 관련 Rule/Guidance를 찾아줘.",
        {"_internal_intent": "rule_lookup"},
    )
    smart_slot = next(
        slot for slot in plan.slots if slot.name == "smart_vessel_instrument"
    )
    assert "DNV-CG-0508" in smart_slot.terms
    assert any("DNV-CG-0508" in group for group in smart_slot.required_groups)


def test_dnv_smart_lookup_routes_directly_to_both_guideline_files():
    question = "DNV에서 자율운항 또는 Smart Vessel 관련 Rule/Guidance를 찾아줘."
    signals = analyze_query(question)
    names = _priority_rule_file_names(signals, question)
    assert "DNV-CG-0264.pdf" in names
    assert "DNV-CG-0508.pdf" in names


def test_mass_adoption_paragraph_without_nonmandatory_prefix_is_still_an_outcome():
    chunk = SimpleNamespace(
        text=(
            "5.36 Subsequently, the Committee adopted resolution MSC.[...](111) on "
            "Adoption of the International Code of Safety for Maritime Autonomous "
            "Surface Ships (MASS Code)."
        )
    )
    claim = _generic_committee_outcome_claim(chunk)
    assert "MASS Code" in claim
    assert "채택" in claim


def test_compound_noun_does_not_turn_into_method_facet():
    requirements = analyze_requirements("방식조치의 요건과 예외를 알려줘")
    assert "requirement" in requirements.facets
    assert "scope" in requirements.facets
    assert "method" not in requirements.facets


def test_direct_feature_heading_with_requested_exception_is_preferred():
    direct = feature_fallback_relevance_score(
        "방식조치의 요건과 예외를 알려줘",
        "3. 방식조치\n모든 강재 표면에 방식조치를 하여야 한다. 다만 갑판은 참작할 수 있다.",
        "방식조치",
    )
    incidental = feature_fallback_relevance_score(
        "방식조치의 요건과 예외를 알려줘",
        "정기검사에서 방식조치의 상태를 확인한다.",
        "방식조치",
    )
    assert direct > incidental + 0.5


class _LiteralRecoveryCollection:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def get(self, **kwargs):
        self.calls.append(kwargs)
        if (kwargs.get("where_document") or {}).get("$contains") != "방식조치":
            return {"ids": [], "metadatas": [], "documents": []}
        return {
            "ids": ["kr-generic", "kr-target"],
            "metadatas": [
                {"doc_id": "kr-generic-rule", "file_name": "2편_2025.pdf", "source": "KR"},
                {"doc_id": "kr-rule", "file_name": "1편_2025.pdf", "source": "KR"},
            ],
            "documents": [
                "방식조치가 설치되어 있다.",
                "방식조치의 요건을 만족하여야 한다. 다만 다음은 제외한다.",
            ],
        }


def test_sparse_feature_fallback_runs_once_only_when_dense_pool_has_zero_hits():
    collection = _LiteralRecoveryCollection()
    raw, scores, identifiers, feature_terms = _query_exact_identifier_hits(
        collection,
        "방식조치의 요건과 예외를 알려줘",
        where=None,
        candidate_documents=["CNG 운반선 지침의 일반 요건"],
    )
    assert identifiers == []
    assert feature_terms == ["방식조치"]
    assert len(collection.calls) == 1
    assert raw["ids"][0] == ["kr-generic", "kr-target"]
    assert scores["kr-target"] > scores["kr-generic"] > 1.0

    collection = _LiteralRecoveryCollection()
    raw, _scores, _identifiers, feature_terms = _query_exact_identifier_hits(
        collection,
        "방식조치의 요건과 예외를 알려줘",
        where=None,
        candidate_documents=["이미 방식조치가 포함된 dense 후보"],
    )
    assert feature_terms == []
    assert collection.calls == []
    assert raw["ids"][0] == []


def test_long_document_lookup_is_sent_to_absence_verifier():
    question = (
        "DNV-CG-0264 — Autonomous and remotely operated vessels에서 "
        "개별 자율운항선 인증서 번호를 찾아 문서 근거와 함께 알려줘."
    )
    assert SPECIFIC_DOCUMENT_LOOKUP_RE.search(question)


def test_absent_specific_lookup_is_rejected_without_unrelated_padding():
    row = {
        "question": (
            "ABS Requirements 문서에서 IMO MASS Code 확정 발효일을 "
            "찾아 문서 근거와 함께 알려줘."
        )
    }
    chunk = SimpleNamespace(text="Foundational requirements for connectivity and software")
    answer = _generate_specific_lookup_answer(
        row,
        [chunk],
        model="unused",
        ollama_base="http://unused",
        num_ctx=1024,
    )
    assert answer is not None
    assert "확인할 수 없습니다" in answer
    assert "Foundational" not in answer


def test_rule_counterfactual_is_sent_to_premise_verifier():
    question = (
        "ABS Requirements를 기준으로 '기능 위험범주는 고장 결과와 무관하다'라는 "
        "전제가 맞는지 검증하고, 틀리면 문서 근거로 바로잡아줘."
    )
    assert PREMISE_VERIFICATION_RE.search(question)


def test_korean_abs_risk_question_adds_bilingual_specific_clause_terms():
    plan = build_evidence_plan(
        "ABS Autonomous and Remote Control Requirements에서 기능 위험범주는 어떻게 정해져?",
        {},
    )
    slot = next(item for item in plan.slots if item.name == "specific_clause")
    assert "risk category" in slot.terms
    assert "operations supervision" in slot.terms
    assert "consequences of failure" in slot.terms


def test_abs_smart_named_document_always_adds_core_evidence_facets():
    plan = build_evidence_plan(
        "ABS 스마트 기능 Guide와 유사 Guidance를 구분해 찾아줘.",
        {},
    )
    slots = {slot.name: slot for slot in plan.slots}
    assert {
        "abs_smart_application",
        "abs_smart_implementation",
        "abs_smart_objectives",
        "abs_smart_notation",
    }.issubset(slots)
    assert ("risk-informed", "risk informed") in slots[
        "abs_smart_objectives"
    ].required_groups


def test_abs_autonomous_named_document_always_adds_core_evidence_facets():
    plan = build_evidence_plan(
        "문서명과 핵심 요건만 bullet로 ABS 자율/원격제어 Requirements를 정리해줘.",
        {},
    )
    assert {
        "abs_risk_classification",
        "abs_risk_informed_verification",
        "abs_foundational_requirements",
        "abs_cumulative_risk_requirements",
    }.issubset({slot.name for slot in plan.slots})


def test_mepc_iswg_briefing_uses_four_named_evidence_facets():
    plan = build_evidence_plan(
        "MEPC 84 ISWG-GHG 논의를 중심으로 환경규제 대응 핵심만 정리해줘.",
        {},
    )
    slots = {slot.name for slot in plan.slots}
    assert {
        "sfcs_label",
        "gfi_compliance",
        "gfi_reporting",
        "lca_method",
    }.issubset(slots)


def test_mass_training_question_adds_three_step_evidence_slot():
    plan = build_evidence_plan(
        "MSC 111의 MASS Code 결정과 원격운항자 훈련 접근법, 경험축적, 2030년 일정을 설명해줘.",
        {"_internal_intent": "mass_code_timeline"},
    )
    assert "remote_operator_training" in {slot.name for slot in plan.slots}


def test_abs_compound_risk_question_keeps_both_risk_facets():
    plan = build_evidence_plan(
        "ABS Requirements에서 기능 위험범주 기준과 상위 위험 기능의 추가 검증을 설명해줘.",
        {},
    )
    slots = {slot.name for slot in plan.slots}
    assert "specific_clause" not in slots
    assert {"risk_classification_basis", "higher_risk_verification"}.issubset(slots)


def test_dnv_0264_compound_question_keeps_scope_pra_and_qualification_facets():
    plan = build_evidence_plan(
        "DNV-CG-0264의 적용범위, 위험성 평가 요구사항과 Concept Qualification 역할을 설명해줘.",
        {},
    )
    slots = {slot.name for slot in plan.slots}
    assert "specific_clause" not in slots
    assert {"scope", "concept_qualification_role", "preliminary_risk_assessment"}.issubset(slots)
    assert plan.document_identifiers == ("DNV/CG/0264",)


def test_abs_external_mass_fact_does_not_enter_msc_timeline_plan():
    plan = build_evidence_plan(
        "ABS Requirements에서 IMO mandatory MASS Code의 확정 발효일과 결의번호를 찾아줘.",
        {},
    )
    assert plan.session_org == "ABS"
    assert plan.intent == "rule_lookup"
    assert "mandatory_adoption_target" not in {slot.name for slot in plan.slots}


def test_mepc_measurement_briefing_covers_method_metric_value_and_scope():
    plan = build_evidence_plan(
        "MEPC 84/6/2 CII fleet report 기준으로 운항·보고 영향을 정리해줘.",
        {},
    )
    slots = {slot.name for slot in plan.slots}
    assert {
        "question_scope",
        "question_method",
        "question_metric",
        "question_comparison",
        "question_value",
        "question_impact",
    }.issubset(slots)


def test_integration_gold_follows_the_second_named_society():
    msc = {"source": "MSC", "secondary_scenario": "V07"}
    assert integration_secondary_id(
        "MSC 111 안전 결론과 DNV guidance를 묶어줘", msc
    ) == "V06"
    assert integration_secondary_id(
        "MSC 111 안전 결론과 LR Rule을 묶어줘", msc
    ) == "V07"


def test_integration_gold_uses_mass_scenario_for_cross_source_msc_question():
    abs_scenario = {"source": "ABS", "secondary_scenario": "V08"}
    assert integration_secondary_id(
        "ABS Requirements와 MSC MASS Code를 함께 설명해줘", abs_scenario
    ) == "V05"


def test_rule_symbol_and_korean_cii_are_not_generic_trend_fallbacks():
    assert (
        classify_question_category(
            "구조 규칙에서 쓰는 tcorr 기호는 어떤 두께를 뜻하지?", {}
        )
        == "rule_lookup"
    )


def test_rule_symbol_definition_creates_direct_clause_evidence_slot():
    question = (
        "\uad6c\uc870 \uaddc\uce59\uc5d0\uc11c \uc4f0\ub294 tcorr \uae30\ud638\ub294 "
        "\uc5b4\ub5a4 \ub450\uaed8\ub97c \ub73b\ud558\uc9c0?"
    )
    plan = build_evidence_plan(
        question,
        {"_internal_intent": "rule_lookup", "category": "rule_lookup"},
    )
    slots = {slot.name: slot for slot in plan.slots}
    assert "specific_clause" in slots
    assert "tcorr" in {term.lower() for term in slots["specific_clause"].terms}
    assert _is_definition_lookup(question)


def test_definition_answer_uses_explicit_symbol_definition_line():
    generic = SimpleNamespace(
        text="3.2.1.1 구조부재에 대한 국부 부식추가 tcorr는 다음 식에 따라 계산된다.",
        file_name="12편_2014.pdf",
        page_number=117,
        clause_number="3.2.1.1",
    )
    explicit = SimpleNamespace(
        text="tgrs\n: 총 판두께(mm)\ntcorr\n: 6장/3.2에 정의된 부식추가(mm)",
        file_name="12편_2014.pdf",
        page_number=211,
        clause_number="1.3.3.7",
    )
    answer, cited = _build_definition_extractive_answer(
        "tcorr 기호는 어떤 두께를 뜻하지?", [generic, explicit], "KR"
    )
    assert cited is explicit
    assert "6장/3.2에 정의된 부식추가(mm)" in answer
    assert "p.211" in answer


def test_definition_answer_extracts_korean_multiword_term():
    unrelated = SimpleNamespace(
        text="과도한 부식이 발견되면 검사 범위를 확대할 수 있다.",
        file_name="1편_2025.pdf",
        page_number=132,
        clause_number="301",
    )
    explicit = SimpleNamespace(
        text=(
            "14. 과도한 부식(substantial corrosion)이라 함은 두께계측에 따른 "
            "부식의 유형을 평가한 결과 부식의 정도가 쇠모한도 이내에 있으나 "
            "쇠모한도의 75%를 초과하여 부식된 상태를 말한다."
        ),
        file_name="1편_2025.pdf",
        page_number=28,
        clause_number="14",
    )
    answer, cited = _build_definition_extractive_answer(
        "과도한 부식의 정의는 무엇인가?", [unrelated, explicit], "KR"
    )
    assert cited is explicit
    assert "쇠모한도의 75%를 초과" in answer
    assert "p.28" in answer
    cii_question = "IMO 문서 기준으로 선박 탄소집약도 등급 관리 요구사항을 요약해줘."
    assert classify_question_category(cii_question, {}) == "env_regulation"
    assert not is_rule_guidance_lookup(
        cii_question, {"category": "env_regulation"}
    )
    assert (
        classify_fast_question_type(
            "구조 규칙에서 쓰는 tcorr 기호는 어떤 두께를 뜻하지?",
            {"category": "rule_lookup"},
        )
        == "rule_question"
    )
    assert (
        classify_fast_question_type("방식조치의 요건과 예외를 알려줘", {})
        == "rule_question"
    )


def test_document_code_matches_primary_filename_not_reference_title():
    assert _identifier_matches_filename(
        "MEPC 84/7/14",
        "MEPC 84-7-14 - Report of the working group.pdf",
    )
    assert not _identifier_matches_filename(
        "MEPC 84/7/14",
        "MEPC 84-7-49 - Comments on document MEPC84714.pdf",
    )


def test_document_router_aggregates_chunks_before_clause_search():
    signals = analyze_query("DNV의 자율운항 guidance 핵심은?")
    candidates = _document_route_candidates(
        query="DNV의 자율운항 guidance 핵심은?",
        ids=["noise", "target-a", "target-b"],
        distances={"noise": 0.18, "target-a": 0.31, "target-b": 0.33},
        metadatas={
            "noise": {
                "doc_id": "noise-doc",
                "file_name": "DNV-RP-C205.pdf",
                "source": "DNV",
            },
            "target-a": {
                "doc_id": "target-doc",
                "file_name": "DNV-CG-0264.pdf",
                "source": "DNV",
            },
            "target-b": {
                "doc_id": "target-doc",
                "file_name": "DNV-CG-0264.pdf",
                "source": "DNV",
            },
        },
        documents={
            "noise": "unrelated structural analysis",
            "target-a": "autonomous and remotely operated ships",
            "target-b": "guidance for autonomous ship functions",
        },
        signals=signals,
        clause_hints=[],
        priority_doc_ids={"target-doc"},
    )
    assert candidates[0]["doc_id"] == "target-doc"
    assert candidates[0]["hit_count"] == 2


def test_meeting_agenda_and_discussion_wording_are_outcome_queries():
    assert is_meeting_outcome_question(
        "MEPC 84가 다룰 예정이었던 주요 의제 항목을 정리해줘."
    )
    assert is_meeting_outcome_question(
        "MSC 111 문서에서 IGC Code 개정 논의가 어떻게 정리됐는지 알려줘."
    )


def test_explicit_source_exclusion_does_not_become_a_hard_filter():
    question = "MEPC 문서는 제외하고 MSC 111의 MASS 결정만 정리해줘."
    assert detect_excluded_sources(question) == ["MEPC"]
    assert detect_named_sources(question) == ["MSC"]
    routed = enrich_row_for_routing({"question": question, "category": "autonomous"})
    assert routed["retrieval_sources"] == ["MSC"]
    assert ("MEPC", 111) not in analyze_query(question).session_codes


def test_multi_source_question_preserves_all_named_sources():
    question = "MSC 111의 MASS 결정과 DNV-CG-0264 지침을 함께 비교해줘."
    routed = enrich_row_for_routing({"question": question, "category": "autonomous"})
    assert routed["retrieval_sources"] == ["MSC", "DNV"]
    assert not routed.get("class_society_hint")


def test_named_meeting_topic_without_number_uses_latest_indexed_session():
    signals = analyze_query("MSC MASS Code 국제 일정과 후속조치를 정리해줘.")
    assert ("MSC", 111) in signals.session_codes


def test_abs_smart_functions_title_routes_to_the_named_guide_not_neighbor_docs():
    question = "ABS Guide for Smart Functions만 근거로 적용 범위를 설명해줘."
    signals = analyze_query(question)
    assert signals.constrained_sources == ["ABS"]
    assert "ABS-Smart-Functions-Guide" in signals.rule_doc_hints
    assert _direct_priority_rule_doc_ids(question, signals) == [
        "abs_abs_rules_guideforsmartfunctionsformarinevesselsandoffshoreunits_v8_bbfd9d9e"
    ]


def test_korean_suffix_after_society_still_detects_abs_requirements():
    question = "ABS에서 autonomous 또는 remote control function 관련 Requirements를 찾아줘."
    signals = analyze_query(question)
    assert signals.named_sources == ["ABS"]
    assert "ABS-Autonomous-Remote-Requirements" in signals.rule_doc_hints
    assert _direct_priority_rule_doc_ids(question, signals) == [
        "abs_abs_rules_requirementsforautonomousandremotecontrolfunctions_v4_1d89b7bb"
    ]


def test_rule_slot_prefers_question_focused_requirement_over_arbitrary_clause():
    generic = SimpleNamespace(
        chunk_id="generic",
        doc_id="abs-req",
        file_name="RequirementsforAutonomousandRemoteControlFunctions-v4.pdf",
        source="ABS",
        text="7.4 Remote operator and remote control conditions.",
        clause_number="7.4",
        page_number=32,
        distance=0.05,
    )
    focused = SimpleNamespace(
        chunk_id="focused",
        doc_id="abs-req",
        file_name="RequirementsforAutonomousandRemoteControlFunctions-v4.pdf",
        source="ABS",
        text=(
            "Each function is assigned a risk category using operations supervision "
            "and consequences of failure."
        ),
        clause_number="2",
        page_number=23,
        distance=0.12,
    )
    selected = select_rule_slots(
        [generic, focused],
        "ABS에서 기능 위험범주는 어떻게 정해져?",
        {"class_society_hint": "ABS", "_rule_guidance_lookup": True},
    )
    assert selected[0].chunk is focused


def test_explicit_igc_outcome_does_not_pad_answer_with_other_msc_topics():
    noise = SimpleNamespace(
        chunk_id="noise",
        text=(
            "IGC Code agenda context. The Committee approved the interim guidelines "
            "for the safety of ships using hydrogen as fuel."
        ),
        file_name="MSC 111-WP.1.pdf",
        page_number=55,
    )
    igc = SimpleNamespace(
        chunk_id="igc",
        text=(
            "The Committee instructed the working group to finalize the draft "
            "amendments to the IGC Code for approval at this session."
        ),
        file_name="MSC 111-WP.1.pdf",
        page_number=61,
    )
    profile = SimpleNamespace(answer_variant="", top_level_category="maritime_safety")
    answer, _warnings, picked = _section1_meeting_outcome(
        [(2.0, noise), (1.9, igc)],
        n=3,
        citation_map={"noise": 1, "igc": 2},
        profile=profile,
        row={
            "question": "MSC 111 문서에서 IGC Code 개정 논의가 어떻게 정리됐는지 알려줘.",
            "_evidence_completion": {"slot_hits": {"major_outcomes": ["noise"]}},
        },
    )
    assert "IGC Code 초안 개정" in answer
    assert "수소" not in answer
    assert picked == [igc]


def test_document_scoped_sparse_prefers_exact_code_chunk():
    signals = analyze_query("AC-SD 허용기준은 어느 절을 따르나?")
    ranked = rank_scoped_sparse_rows(
        "AC-SD 허용기준은 어느 절을 따르나?",
        signals,
        ["generic", "exact"],
        [
            {"file_name": "13편_2025.pdf"},
            {"file_name": "14편_2025.pdf"},
        ],
        [
            "허용기준 일반 설명",
            "AC-SD 판과 국부 지지부재의 허용응력은 5장 1절에 따른다.",
        ],
    )
    assert ranked[0][1] == "exact"


class _FakeCollection:
    def __init__(self):
        self.rows = {
            "noise": (
                0.16,
                {"doc_id": "noise-doc", "file_name": "DNV-RP-C205.pdf", "source": "DNV"},
                "structural design recommendation",
            ),
            "target-intro": (
                0.33,
                {"doc_id": "target-doc", "file_name": "DNV-CG-0264.pdf", "source": "DNV"},
                "autonomous and remotely operated ships guidance",
            ),
            "target-clause": (
                0.24,
                {"doc_id": "target-doc", "file_name": "DNV-CG-0264.pdf", "source": "DNV"},
                "The autonomous ship guidance safety principle requires a minimum risk condition and fail-safe response.",
            ),
        }

    @staticmethod
    def _matches(meta, where):
        if not where:
            return True
        if "$and" in where:
            return all(_FakeCollection._matches(meta, part) for part in where["$and"])
        for key, expected in where.items():
            actual = meta.get(key)
            if isinstance(expected, dict) and "$in" in expected:
                if actual not in expected["$in"]:
                    return False
            elif actual != expected:
                return False
        return True

    def query(self, *, query_embeddings, n_results, where=None):
        rows = []
        for cid, (distance, meta, document) in self.rows.items():
            if self._matches(meta, where):
                rows.append((distance, cid, meta, document))
        # The global vector search intentionally misses the decisive clause.
        if not where:
            rows = [row for row in rows if row[1] != "target-clause"]
        rows.sort()
        rows = rows[:n_results]
        return {
            "ids": [[row[1] for row in rows]],
            "distances": [[row[0] for row in rows]],
            "metadatas": [[row[2] for row in rows]],
            "documents": [[row[3] for row in rows]],
        }

    def get(self, *, where=None, where_document=None, limit=100, include=None):
        rows = []
        for cid, (_distance, meta, document) in self.rows.items():
            if not self._matches(meta, where):
                continue
            if where_document and where_document.get("$contains", "").lower() not in document.lower():
                continue
            rows.append((cid, meta, document))
        rows = rows[:limit]
        return {
            "ids": [row[0] for row in rows],
            "metadatas": [row[1] for row in rows],
            "documents": [row[2] for row in rows],
        }


def test_hierarchical_query_searches_inside_selected_document(monkeypatch):
    monkeypatch.setenv("MARITIME_TEXT_HIERARCHICAL", "1")
    out = query_with_hybrid_ranking(
        _FakeCollection(),
        "DNV 자율운항 guidance의 minimum risk condition 안전 원칙은?",
        [0.1, 0.2],
        top_k=2,
        fetch_k=10,
    )
    assert out["ids"][0][0] == "target-clause"
    route = out["document_route"]
    assert route["enabled"] is True
    assert route["selected_doc_ids"][0] == "target-doc"
    assert route["scoped_sparse_used"] is True


def test_cell_verifier_keeps_row_and_column_in_selected_table():
    wrong = SimpleNamespace(
        chunk_id="wrong",
        chunk_type="table_row",
        table_id="wrong-table",
        text="행 기준: 재화중량 100000 초과 150000 이하\n셀: 안전사용하중=999",
        distance=0.01,
    )
    good = SimpleNamespace(
        chunk_id="good",
        chunk_type="table_row",
        table_id="good-table",
        text="행 기준: 재화중량 100000 초과 150000 이하\n셀: 안전사용하중=250",
        distance=0.20,
        file_name="guide.pdf",
        page_number=107,
    )
    question = "재화중량이 10만 톤 초과 15만 톤 이하인 선박의 안전사용하중은 몇 톤인가?"
    debug = {
        "selected_table_id": "good-table",
        "selected_table_candidates": [
            {"table_id": "good-table"},
            {"table_id": "wrong-table"},
        ],
        "parsed_query": {
            "query_type": "cell_lookup",
            "row_entities": ["재화중량이 10만 톤 초과 15만 톤 이하인 선박"],
            "column_entities": ["안전사용하중"],
            "attribute_candidates": ["안전사용하중"],
        },
    }
    row = {"question": question}
    verification = verify_row_column_intersection(row, [wrong, good], debug=debug)
    assert verification["passes"] is True
    assert verification["table_id"] == "good-table"
    answer = build_deterministic_table_answer(row, [wrong, good], debug=debug)
    assert "250" in (answer or "")
    assert "999" not in (answer or "")


def test_multilevel_table_header_maps_subject_to_physical_subcolumn():
    table_id = "kr-rule-table"
    header = SimpleNamespace(
        chunk_id="header",
        chunk_type="table_row",
        table_id=table_id,
        text=(
            "영역=REG01 | 열1=(빈 셀) | 열2=판 및 국부 지지부재(1) | "
            "열3=1차 지지부재(1) | 열4=선체거더 부재"
        ),
        file_name="14편_2025.pdf",
        page_number=19,
    )
    subheaders = SimpleNamespace(
        chunk_id="subheaders",
        chunk_type="table_row",
        table_id=table_id,
        text=(
            "영역=REG02 | 열1=허용기준 AC-S AC-SD AC-A AC-T | "
            "열2=항복 | 열3=좌굴 | 열4=항복 | 열5=좌굴 | "
            "열6=항복 허용응력 : 5장 1절 | 열7=좌굴 허용좌굴 사용계수 : 8장 1절"
        ),
        file_name="14편_2025.pdf",
        page_number=19,
    )
    values = SimpleNamespace(
        chunk_id="values",
        chunk_type="table_row",
        table_id=table_id,
        text=(
            "영역=REG02 | 열2=항복: 허용응력 : 6장 4절 6장 5절 | "
            "열3=좌굴: 강성 및 치수비의 조정 : 8장 2절 | "
            "열4=항복: 허용응력 : 6장 6절 | "
            "열5=좌굴: 강성 및 치수비의 조정 : 8장 1절 8장 2절"
        ),
        file_name="14편_2025.pdf",
        page_number=19,
    )
    question = (
        "14편_2025.pdf 19쪽 표에서 AC-S·AC-SD·AC-A·AC-T "
        "판과 국부 지지부재의 허용응력은 어느 절을 따르나?"
    )
    debug = {
        "selected_table_id": table_id,
        "selected_table_candidates": [{"table_id": table_id}],
        "parsed_query": {
            "query_type": "cell_lookup",
            "row_entities": ["AC-S AC-SD AC-A AC-T"],
            "column_entities": ["판과 국부 지지부재", "허용응력"],
            "attribute_candidates": ["허용응력"],
        },
    }
    row = {"question": question}
    answer = build_deterministic_table_answer(
        row, [header, subheaders, values], debug=debug
    )
    assert "6장 4절 6장 5절" in (answer or "")
    assert "5장 1절" not in (answer or "")
    assert row["_cell_verification"]["method"] == "multilevel_header"
    assert row["_answer_citation_chunks"] == [header, values]


def test_summary_with_multiple_labeled_rows_selects_matching_count_cell():
    summary = SimpleNamespace(
        chunk_id="casting-summary",
        chunk_type="table_summary",
        table_id="casting-table",
        text=(
            "열1=주강품 종류: 일반 주강품 | 열2=시험재의 수: 제품마다 1개\n"
            "열1=주강품 종류: 모양이 복잡한 주강품 또는 1개의 중량이 10톤을 넘는 주강품 | "
            "열2=시험재의 수: 제품마다 2개(1)"
        ),
        file_name="2편_2025.pdf",
        page_number=78,
    )
    question = "복잡한 형상이거나 단품 중량이 10톤을 넘는 주강품은 제품별 시험재가 몇 개인가?"
    debug = {
        "selected_table_id": "casting-table",
        "selected_table_candidates": [{"table_id": "casting-table"}],
        "parsed_query": {
            "query_type": "cell_lookup",
            "row_entities": ["복잡한 형상", "중량 10톤을 넘는 주강품"],
            "column_entities": ["시험재의 수"],
            "attribute_candidates": ["시험재의 수"],
        },
    }
    answer = build_deterministic_table_answer(
        {"question": question}, [summary], debug=debug
    )
    assert "제품마다 2개" in (answer or "")
    assert "제품마다 1개" not in (answer or "")


def test_definition_table_allows_location_question_to_use_definition_cell():
    row_chunk = SimpleNamespace(
        chunk_id="engine-room-bulkhead",
        chunk_type="table_row",
        table_id="definition-table",
        text=(
            "열1=용어: 기관실 격벽 | "
            "열2=정의: 기관실 전방 또는 후방의 횡격벽"
        ),
        file_name="13편_2025.pdf",
        page_number=48,
    )
    question = "표에 정의된 기관실 격벽의 위치는 어디인가?"
    debug = {
        "selected_table_id": "definition-table",
        "selected_table_candidates": [{"table_id": "definition-table"}],
        "parsed_query": {
            "query_type": "cell_lookup",
            "row_entities": ["기관실 격벽"],
            "column_entities": ["위치"],
            "attribute_candidates": ["위치"],
        },
    }
    answer = build_deterministic_table_answer(
        {"question": question}, [row_chunk], debug=debug
    )
    assert "기관실 전방 또는 후방의 횡격벽" in (answer or "")
