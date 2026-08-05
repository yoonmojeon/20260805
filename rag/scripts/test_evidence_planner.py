from __future__ import annotations

import unittest

from evidence_planner import build_evidence_plan, complete_evidence_slots
from answer_contract import apply_answer_contract
from grounded_answer_policy import verify_high_risk_claims
from meeting_structured_answer import _generic_committee_outcome_claim
from question_requirements import analyze_requirements
from rag_answer_lib import RetrievedChunk


def chunk(text: str, *, source: str = "MSC", file_name: str = "session draft report.pdf"):
    return RetrievedChunk(
        chunk_id="c1",
        doc_id="d1",
        source=source,
        file_name=file_name,
        page_number=1,
        clause_number="",
        element_type="text",
        distance=0.0,
        text=text,
    )


class EvidencePlannerTest(unittest.TestCase):
    def test_broad_dnv_rule_lookup_does_not_force_specific_clause(self):
        requirements = analyze_requirements(
            "DNV에서 자율운항 또는 Smart Vessel 관련 Rule/Guidance를 찾아줘."
        )
        plan = build_evidence_plan(
            "DNV에서 자율운항 또는 Smart Vessel 관련 Rule/Guidance를 찾아줘.",
            {},
        )

        self.assertNotIn("specific_clause", {slot.name for slot in plan.slots})

    def test_broad_lr_rule_lookup_does_not_force_specific_clause(self):
        requirements = analyze_requirements(
            "LR에서 대체연료 관련 Rule/Guidance를 찾아줘."
        )
        plan = build_evidence_plan(
            "LR에서 대체연료 관련 Rule/Guidance를 찾아줘.",
            {},
        )

        self.assertNotIn("specific_clause", {slot.name for slot in plan.slots})

    def test_korean_particle_keeps_class_society_scope(self):
        plan = build_evidence_plan(
            "DNV에서 자율운항 관련 Rule/Guidance를 찾아줘.", {}
        )
        self.assertEqual(plan.session_org, "DNV")
        self.assertEqual(plan.intent, "rule_lookup")

    def test_alt_fuel_rule_query_is_not_misrouted_as_meeting(self):
        plan = build_evidence_plan(
            "LR에서 대체연료 관련 Rule/Guidance를 찾아줘.", {}
        )
        self.assertEqual(plan.session_org, "LR")
        self.assertEqual(plan.intent, "rule_lookup")
        names = {slot.name for slot in plan.slots}
        self.assertIn("requirements", names)
        self.assertIn("safety_controls", names)

    def test_clause_lookup_requires_technical_phrase_in_selected_evidence(self):
        plan = build_evidence_plan(
            "DNV autonomous vessel ROC status and situational awareness requirements", {}
        )
        direct = next(slot for slot in plan.slots if slot.name == "specific_clause")
        self.assertIn("ROC", direct.terms)
        self.assertIn("status and situational awareness", direct.terms)

    def test_document_local_clause_search_prefers_direct_technical_evidence(self):
        class Collection:
            def get(self, **_kwargs):
                return {
                    "ids": ["scope", "direct"],
                    "documents": [
                        "2 Objective This guideline provides general guidance for autonomous vessels.",
                        "6.4.1 Status and situational awareness. It should be possible to observe real-time operational status from ROC.",
                    ],
                    "metadatas": [
                        {"source": "DNV", "file_name": "DNV-CG-0264.pdf", "page_number": 9},
                        {"source": "DNV", "file_name": "DNV-CG-0264.pdf", "page_number": 88, "clause_number": "6.4.1"},
                    ],
                }

        pool = [
            RetrievedChunk(
                chunk_id="scope", doc_id="dnv-cg", source="DNV", file_name="DNV-CG-0264.pdf",
                page_number=9, clause_number="2", element_type="text", distance=0.0,
                text="2 Objective This guideline provides general guidance for autonomous vessels.",
            )
        ]
        row = {"question": "DNV autonomous vessel ROC status and situational awareness requirements"}
        _ordered, meta = complete_evidence_slots(Collection(), pool, row)
        self.assertIn("direct", meta["slot_hits"]["specific_clause"])

    def test_specific_clause_heading_precedes_continuation_page(self):
        class Collection:
            def get(self, **_kwargs):
                return {
                    "ids": ["continuation", "heading"],
                    "documents": [
                        (
                            "status and situational awareness is lost during a "
                            "connectivity outage. Alerts in ROC should be descriptive."
                        ),
                        (
                            "6 6.4.1 Status and situational awareness. It should be "
                            "possible to observe real-time operational status, "
                            "readiness and capacity from ROC."
                        ),
                    ],
                    "metadatas": [
                        {
                            "source": "DNV",
                            "file_name": "DNV-CG-0264.pdf",
                            "page_number": 89,
                        },
                        {
                            "source": "DNV",
                            "file_name": "DNV-CG-0264.pdf",
                            "page_number": 88,
                            "clause_number": "6",
                        },
                    ],
                }

        continuation = RetrievedChunk(
            chunk_id="continuation",
            doc_id="dnv-cg",
            source="DNV",
            file_name="DNV-CG-0264.pdf",
            page_number=89,
            clause_number="",
            element_type="text",
            distance=0.0,
            text=(
                "status and situational awareness is lost during a connectivity "
                "outage. Alerts in ROC should be descriptive."
            ),
        )
        row = {
            "question": (
                "DNV autonomous vessel ROC status and situational awareness "
                "requirements"
            )
        }
        _ordered, meta = complete_evidence_slots(
            Collection(), [continuation], row
        )
        self.assertEqual(meta["slot_hits"]["specific_clause"][0], "heading")

    def test_one_chunk_may_satisfy_requirement_and_specific_slots(self):
        class Collection:
            def get(self, **_kwargs):
                return {
                    "ids": ["direct", "continuation"],
                    "documents": [
                        (
                            "6.4.1 Status and situational awareness. ROC requirements "
                            "should include real-time operational status."
                        ),
                        (
                            "Status and situational awareness may be affected by a "
                            "connectivity outage."
                        ),
                    ],
                    "metadatas": [
                        {
                            "source": "DNV",
                            "file_name": "DNV-CG-0264.pdf",
                            "page_number": 88,
                            "clause_number": "6",
                        },
                        {
                            "source": "DNV",
                            "file_name": "DNV-CG-0264.pdf",
                            "page_number": 89,
                        },
                    ],
                }

        direct = RetrievedChunk(
            chunk_id="direct",
            doc_id="dnv-cg",
            source="DNV",
            file_name="DNV-CG-0264.pdf",
            page_number=88,
            clause_number="6",
            element_type="text",
            distance=0.0,
            text=(
                "6.4.1 Status and situational awareness. ROC requirements "
                "should include real-time operational status."
            ),
        )
        row = {
            "question": (
                "DNV autonomous vessel ROC status and situational awareness "
                "requirements"
            )
        }
        _ordered, meta = complete_evidence_slots(Collection(), [direct], row)
        self.assertIn("direct", meta["slot_hits"]["requirements"])
        self.assertEqual(meta["slot_hits"]["specific_clause"][0], "direct")

    def test_direct_clause_slot_is_not_document_identity_slot(self):
        plan = build_evidence_plan(
            "DNV autonomous vessel ROC status and situational awareness requirements", {}
        )
        slots = {slot.name: slot for slot in plan.slots}
        self.assertNotIn("status and situational awareness", slots["rule_identity"].terms)
        self.assertIn("status and situational awareness", slots["specific_clause"].terms)

    def test_session_number_before_korean_particle(self):
        plan = build_evidence_plan(
            "MSC 111의 주요 결과를 3개 항목으로 요약해줘.",
            {"internal_intent": "meeting_outcome"},
        )
        self.assertEqual(("MSC", "111"), (plan.session_org, plan.session_number))
        self.assertEqual(3, plan.requested_count)

    def test_mass_timeline_has_completeness_slots(self):
        plan = build_evidence_plan(
            "MSC 112에서 MASS Code 결정과 mandatory code 일정을 정리해줘.",
            {"internal_intent": "mass_code_timeline"},
        )
        self.assertEqual(
            {
                "current_decision",
                "mandatory_adoption_target",
                "entry_into_force_target",
                "schedule_uncertainty",
                "experience_building",
            },
            {slot.name for slot in plan.slots},
        )

    def test_dcs_quality_treatment_includes_followup_process(self):
        plan = build_evidence_plan(
            "MEPC 자료의 IMO DCS 제출 데이터 품질검증 시 오류 유형과 처리 절차는?",
            {"internal_intent": "data_quality_verification"},
        )
        treatment = next(slot for slot in plan.slots if slot.name == "treatment")
        self.assertIn("further examined", treatment.terms)
        self.assertIn(
            ("Administrations", "recognized organizations"),
            treatment.required_groups,
        )

    def test_generic_outcome_extracts_mass_without_file_fixture(self):
        claim = _generic_committee_outcome_claim(
            chunk(
                "The Committee adopted the non-mandatory goal-based MASS Code. "
                "The Committee agreed to invite other bodies to note the decision."
            )
        )
        self.assertIn("MASS Code 채택", claim)

    def test_unsupported_operator_duty_is_removed(self):
        evidence = chunk(
            "The Secretariat carried out a quality control and verification process.",
            source="MEPC",
            file_name="meeting submission.pdf",
        )
        answer, _rows, warnings = verify_high_risk_claims(
            "## 핵심 답변\n\n- 선사는 모든 데이터를 재검증해야 합니다. [1]",
            [evidence],
        )
        self.assertNotIn("재검증해야", answer)
        self.assertTrue(any("prescriptive_inference" in warning for warning in warnings))

    def test_answer_contract_always_keeps_four_sections(self):
        evidence = chunk("The Committee approved the interim safety guidelines.")
        result = apply_answer_contract(
            "## 1) 핵심 요약\n\n- 임시 안전지침을 승인했습니다. [1]",
            [evidence],
        )
        for number in ("1)", "2)", "3)", "4)"):
            self.assertIn(f"## {number}", result.answer)


if __name__ == "__main__":
    unittest.main()
