from __future__ import annotations

import unittest

from compound_regulatory import (
    build_class_search_query,
    build_compound_evidence_scaffold,
    is_compound_regulatory_class_question,
    repair_compound_answer,
    validate_compound_answer,
)
from evidence_planner import build_evidence_plan
from grounded_dynamic_answer import normalize_generated_markdown
from meeting_category_profile import uses_structured_meeting_answer
from rag_answer_lib import RetrievedChunk
from rag_query_router import (
    enrich_row_for_routing,
    is_rule_guidance_lookup,
    resolve_pipeline_route,
)
from retrieval_search import extract_sparse_feature_terms, extract_sparse_latin_terms


QUESTION = (
    "암모니아 연료선의 개념승인을 준비한다고 가정하고, MSC 111 논의와 "
    "보유 선급 규정을 근거로 설계 검토 체크리스트와 미확정 규제를 작성해줘."
)


def _chunk(chunk_id: str, source: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=chunk_id,
        source=source,
        file_name=f"{source}-rule.pdf",
        page_number=1,
        clause_number="",
        element_type="text",
        distance=0.0,
        text=text,
    )


class CompoundRegulatoryTest(unittest.TestCase):
    def test_route_keeps_meeting_and_class_lanes(self):
        route = resolve_pipeline_route(QUESTION, {}, latency_mode="accurate")

        self.assertTrue(route["compound_regulatory_class"])
        self.assertEqual("compound_regulatory_class", route["selected_answer_mode"])
        self.assertEqual(["MSC", "DNV", "KR", "ABS", "LR"], route["detected_sources"])
        self.assertFalse(route["rule_guidance_lookup"])

        row = enrich_row_for_routing({"question": QUESTION})
        self.assertFalse(
            uses_structured_meeting_answer(row, legacy_category=row["category"])
        )

    def test_explicit_rule_terms_do_not_demote_compound_questions(self):
        questions = [
            "수소 연료선 기본설계에 MSC 111과 DNV·KR 선급 규정을 함께 근거로 체크리스트를 작성해줘.",
            "MSC 111 MASS Code와 DNV-CG-0264 및 ABS 규정을 근거로 개념승인 사항을 작성해줘.",
            "MSC 111 논의와 DNV Fuel ready·Gas fuelled ammonia 규정으로 비교해줘.",
        ]
        for question in questions:
            with self.subTest(question=question):
                row = enrich_row_for_routing({"question": question})
                self.assertTrue(row.get("_compound_regulatory_class"))
                self.assertFalse(
                    is_rule_guidance_lookup(
                        question,
                        row,
                        category=str(row.get("category") or ""),
                    )
                )

    def test_class_query_removes_session_dominance(self):
        rewritten = build_class_search_query(QUESTION)
        self.assertNotIn("MSC 111", rewritten)
        self.assertIn("approval in principle", rewritten)
        self.assertIn("ammonia", rewritten.lower())
        self.assertEqual(["암모니아"], extract_sparse_feature_terms(rewritten))
        self.assertIn("ammonia", extract_sparse_latin_terms(rewritten))

    def test_evidence_plan_requires_both_lanes_and_design(self):
        plan = build_evidence_plan(QUESTION, {"_internal_intent": "altfuel_ghg_safety"})
        names = {slot.name for slot in plan.slots}
        self.assertEqual("regulatory_class_integration", plan.intent)
        self.assertIn("compound_meeting_decision", names)
        self.assertIn("compound_class_instrument", names)
        self.assertIn("compound_design_arrangement", names)
        self.assertIn("compound_safety_systems", names)
        self.assertIn("compound_regulatory_uncertainty", names)

    def test_autonomous_compound_plan_uses_autonomous_design_facets(self):
        question = (
            "MSC 111 MASS Code와 DNV-CG-0264 및 ABS 규정으로 "
            "자율운항선 개념승인 체크리스트를 작성해줘."
        )
        plan = build_evidence_plan(question, {})
        slots = {slot.name: slot for slot in plan.slots}
        self.assertIn("CONOPS", slots["compound_design_arrangement"].terms)
        self.assertIn("FDIR", slots["compound_safety_systems"].terms)
        self.assertNotIn("fuel tank", slots["compound_design_arrangement"].terms)

    def test_markdown_normalizer_recovers_number_dot_heading(self):
        draft = """## 1) 핵심 요약
- 요약 [1]
**2. 설계 및 기술 사양 (선급 Rule 기반)**
* 탱크 배치를 검토한다. [2]
## 3) 추후 확인 필요사항
- 확인한다. [1]
## 4) 관련 선급 Rule / Guidance
- DNV [2]"""
        normalized = normalize_generated_markdown(draft)
        self.assertIn("## 2) 선박 운항/업무 영향", normalized)
        self.assertNotIn("**2. 설계", normalized)

    def test_validator_rejects_meeting_only_answer_and_wrong_code(self):
        chunks = [
            _chunk(
                "m1",
                "MSC",
                "The Committee approved interim guidelines for ammonia cargo as fuel "
                "and discussed amendments to the IGC Code.",
            )
        ]
        answer = """## 1) 핵심 요약
- MSC 111에서 IGF Code 개정을 승인했다. [1]

## 2) 선박 운항/업무 영향
- 설계 검토가 필요하다. [1]

## 3) 추후 확인 필요사항
- 향후 개정이 필요하다. [1]

## 4) 관련 선급 Rule / Guidance
- 검색 근거에서 확인되지 않음"""

        warnings = validate_compound_answer(answer, chunks, question=QUESTION)
        self.assertIn("compound_class_evidence_missing", warnings)
        self.assertIn("compound_class_rule_section_missing", warnings)
        self.assertIn("compound_design_checklist_incomplete", warnings)
        self.assertIn("compound_instrument_citation_mismatch:IGF_Code", warnings)

    def test_validator_accepts_two_lane_checklist(self):
        chunks = [
            _chunk(
                "m1",
                "MSC",
                "The Committee approved interim guidelines for ammonia cargo as fuel. "
                "Future revisions will consider other gas carriers.",
            ),
            _chunk(
                "c1",
                "DNV",
                "Fuel ready (Ammonia) concept design documentation includes general "
                "arrangement, fuel tanks, bunkering, pipe routing, hazardous areas, "
                "ventilation, gas detection, emergency shutdown, risk assessment and "
                "fire protection. Approval in principle (AIP) documentation is required.",
            ),
        ]
        answer = """## 1) 핵심 요약
- MSC 111은 암모니아 화물의 연료 사용을 위한 임시지침을 승인했다. [1]
- DNV Fuel ready (Ammonia)는 개념설계 자료의 검토 틀을 제공한다. [2]

## 2) 선박 운항/업무 영향
- 원칙승인(AIP) 자료와 탱크·일반배치를 검토한다. [2]
- 벙커링과 연료배관 경로를 검토한다. [2]
- 위험·독성구역과 환기·가스검지를 검토한다. [2]
- 비상정지, 위험성 평가와 화재방호를 검토한다. [2]

## 3) 추후 확인 필요사항
- 다른 가스운반선 적용은 향후 개정 범위를 확인해야 한다. [1]

## 4) 관련 선급 Rule / Guidance
- DNV Fuel ready (Ammonia)의 개념설계 제출자료와 적용범위를 확인한다. [2]"""

        self.assertEqual([], validate_compound_answer(answer, chunks, question=QUESTION))

    def test_evidence_scaffold_is_complete_and_valid(self):
        chunks = [
            _chunk(
                "m-final",
                "MSC",
                "The Committee approved the Interim guidelines for the safety of ships "
                "using ammonia cargo as fuel under the IGC Code.",
            ),
            _chunk(
                "m-scope",
                "MSC",
                "The guidelines cover ammonia cargo as fuel. Gas carriers carrying "
                "ammonia solely for use as fuel should be addressed in future revisions.",
            ),
            _chunk(
                "c-design",
                "DNV",
                "Fuel ready (Ammonia) and Approval in principle cover fuel tank and "
                "containment arrangement, ventilation, gas detection, emergency shutdown, "
                "HAZID, QRA, gas dispersion and fire and explosion analysis.",
            ),
        ]
        answer = build_compound_evidence_scaffold(QUESTION, {}, chunks)

        self.assertIn("## 1) 핵심 요약", answer)
        self.assertIn("## 4) 관련 선급 Rule / Guidance", answer)
        self.assertEqual([], validate_compound_answer(answer, chunks, question=QUESTION))

    def test_post_repair_uses_final_decision_and_supported_class_citation(self):
        chunks = [
            _chunk(
                "m-final",
                "MSC",
                "6.37 The Committee approved MSC.1/Circ. on Interim guidelines for "
                "use of ammonia cargo as fuel.",
            ),
            _chunk(
                "m-scope",
                "MSC",
                "The amended IGC Code will allow toxic ammonia cargo as fuel. "
                "Application to gas carriers carrying ammonia solely for use as fuel "
                "should be addressed through future revisions.",
            ),
            _chunk(
                "c-safety",
                "DNV",
                "A gas dispersion analysis shall justify any reduced safety distance. "
                "Additional gas detection and water spray systems may be required.",
            ),
        ]
        draft = """## 1) 핵심 요약
- 위원회는 승인을 추진하기 위해 작업반에 회부했다. [2]
- 다른 가스운반선 적용은 향후 개정한다. [2]

## 2) 선박 운항/업무 영향
- 안전 거리 단축은 가스 확산 분석과 가스 감지, 수분무 설비가 필요하다. [1]

## 3) 추후 확인 필요사항
- 향후 적용범위를 확인해야 한다. [2]

## 4) 관련 선급 Rule / Guidance
- DNV 안전설비 규정을 검토한다. [3]"""
        repaired = repair_compound_answer(draft, chunks, question=QUESTION)
        self.assertRegex(repaired, r"최종 결정.*승인했습니다\. \[1\]")
        self.assertRegex(repaired, r"IGC Code")
        self.assertRegex(repaired, r"안전 거리.*\[3\]")

    def test_autonomous_post_repair_moves_timeline_and_adds_cq(self):
        question = (
            "MSC 111 MASS Code mandatory 일정과 DNV-CG-0264 및 ABS 규정을 근거로 "
            "자율운항선 개념승인 체크리스트와 규제 공백을 작성해줘."
        )
        chunks = [
            _chunk(
                "m1",
                "MSC",
                "The Committee adopted the MASS Code. The timeline envisaging "
                "adoption in 2030 and entry into force in 2032 appeared unrealistic.",
            ),
            _chunk(
                "d1",
                "DNV",
                "DNV-CG-0264 Concept Qualification and System Qualification "
                "describe the approval process and verification scope.",
            ),
            _chunk(
                "a1",
                "ABS",
                "CONOPS shall identify operating boundaries and remote operator roles.",
            ),
        ]
        draft = """## 1) 핵심 요약
- MASS Code를 채택했다. [1]
- 2030 채택·2032 발효 일정은 비현실적이라는 의견이 있었다. [1]

## 2) 선박 운항/업무 영향
- CONOPS의 운용범위와 원격운영자 역할을 확인한다. [3]

## 3) 추후 확인 필요사항
- 일반 설계를 검토할 필요가 있다. [2]

## 4) 관련 선급 Rule / Guidance
- ABS 요구사항을 확인한다. [3]"""
        repaired = repair_compound_answer(draft, chunks, question=question)
        section3 = repaired.split("## 3) 추후 확인 필요사항", 1)[1].split(
            "## 4) 관련 선급 Rule / Guidance", 1
        )[0]
        self.assertIn("2030", section3)
        self.assertIn("2032", section3)
        self.assertIn("개념승인(CQ)", repaired)
        self.assertIn("DNV-rule", repaired)


if __name__ == "__main__":
    unittest.main()
