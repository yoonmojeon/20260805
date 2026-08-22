from __future__ import annotations

import unittest

from evidence_planner import build_evidence_plan, complete_evidence_slots
from evidence_selection import select_planned_evidence
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
    def test_literal_feature_hit_survives_answer_evidence_planning(self):
        direct = RetrievedChunk(
            chunk_id="direct",
            doc_id="rules",
            source="KR",
            file_name="rules.pdf",
            page_number=34,
            clause_number="3",
            element_type="text",
            distance=0.0,
            text=(
                "3. 방식조치 (1) 연료유 탱크를 제외한 모든 강재 표면은 방식조치를 "
                "하여야 한다. 다만 갑판 및 내저판은 규정을 참작할 수 있다."
            ),
        )
        generic = RetrievedChunk(
            chunk_id="generic",
            doc_id="other",
            source="KR",
            file_name="other.pdf",
            page_number=1,
            clause_number="",
            element_type="text",
            distance=0.0,
            text="일반 요구사항을 설명한다.",
        )
        row = {
            "question": "방식조치의 요건과 예외를 알려줘",
            "_text_document_route": {"feature_fallback_terms": ["방식조치"]},
            "_evidence_completion": {"slot_hits": {"requirements": ["generic"]}},
        }
        selected, meta = select_planned_evidence(
            row, [generic, direct], max_chunks=1
        )
        self.assertEqual(["direct"], [item.chunk_id for item in selected])
        self.assertEqual(["direct"], meta["selected_by_slot"]["literal_feature"])

    def test_compound_mass_keeps_final_roadmap_position_after_objection(self):
        adopted = chunk(
            "The Committee adopted the non-mandatory MASS Code.",
        )
        adopted.chunk_id = "adopted"
        objection = chunk(
            "Several delegations viewed adoption in 2030 and entry into force in 2032 as unrealistic."
        )
        objection.chunk_id = "objection"
        final_position = chunk(
            "Notwithstanding the above, the Committee noted that the Group nevertheless agreed "
            "to continue working towards the target year of 2030 for adoption; 2032 was ambitious."
        )
        final_position.chunk_id = "final"
        row = {
            "question": (
                "MSC 111 MASS Code 결정과 2030 채택, 2032 발효 일정 및 "
                "DNV·ABS 규정을 근거로 설계 체크리스트를 정리해줘."
            ),
            "_compound_regulatory_class": True,
            "_evidence_completion": {
                "slot_hits": {
                    "compound_meeting_decision": ["adopted"],
                    "compound_regulatory_uncertainty": ["objection"],
                }
            },
        }
        selected, meta = select_planned_evidence(
            row,
            [adopted, objection],
            [final_position],
            max_chunks=4,
        )
        self.assertIn("final", [item.chunk_id for item in selected])
        self.assertEqual(
            ["final"], meta["selected_by_slot"]["compound_final_position"]
        )

    def test_compound_planner_adds_final_roadmap_candidate_after_objection(self):
        class Collection:
            def get(self, **_kwargs):
                return {
                    "ids": ["adopted", "objection", "final"],
                    "documents": [
                        "The Committee adopted the non-mandatory MASS Code.",
                        (
                            "Several delegations viewed adoption in 2030 and entry "
                            "into force in 2032 as unrealistic."
                        ),
                        (
                            "Notwithstanding the above, the Committee noted that the "
                            "Group nevertheless agreed to continue working towards "
                            "the target year of 2030 for adoption; 2032 was ambitious."
                        ),
                    ],
                    "metadatas": [
                        {"source": "MSC", "file_name": "MSC 111-WP.1.pdf", "page_number": 48},
                        {"source": "MSC", "file_name": "MSC 111-WP.1.pdf", "page_number": 49},
                        {"source": "MSC", "file_name": "MSC 111-WP.1.pdf", "page_number": 49},
                    ],
                }

        adopted = chunk("The Committee adopted the non-mandatory MASS Code.")
        adopted.chunk_id = "adopted"
        objection = chunk(
            "Several delegations viewed adoption in 2030 and entry into force in 2032 as unrealistic."
        )
        objection.chunk_id = "objection"
        row = {
            "question": (
                "Prepare an approval in principle checklist based on the MSC 111 "
                "MASS Code and DNV-CG-0264 and ABS rules, including the mandatory "
                "2030 adoption and 2032 entry-into-force targets."
            ),
            "_compound_regulatory_class": True,
        }

        ordered, meta = complete_evidence_slots(Collection(), [adopted, objection], row)

        self.assertIn("final", [item.chunk_id for item in ordered])
        self.assertEqual(["final"], meta["slot_hits"]["compound_final_position"])

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

    def test_msc111_major_results_use_distinct_outcome_lanes(self):
        plan = build_evidence_plan(
            "MSC 111의 주요 결과를 3개 항목으로 요약해줘.",
            {"_internal_intent": "meeting_outcome"},
        )
        names = {slot.name for slot in plan.slots}
        self.assertEqual(
            {
                "msc_mass_outcome",
                "msc_vdes_outcome",
                "msc_hydrogen_fuel_outcome",
                "msc_liquid_hydrogen_bulk_outcome",
            },
            names,
        )

        korean_name = build_evidence_plan(
            "제111차 해사안전위원회 결과를 3개 항목으로 정리해줘.",
            {},
        )
        self.assertEqual("MSC", korean_name.session_org)
        self.assertEqual("111", korean_name.session_number)
        self.assertIn(
            "msc_vdes_outcome", {slot.name for slot in korean_name.slots}
        )

    def test_named_meeting_fact_uses_concept_specific_lane(self):
        mepc = build_evidence_plan(
            "MEPC 84/7/14에서 GFI 준수 방식과 초안 규칙 36의 관계를 설명해줘.",
            {},
        )
        self.assertIn("gfi_compliance_lookup", {slot.name for slot in mepc.slots})

        msc = build_evidence_plan(
            "MSC 111-WP.1에서 VDES 관련 결의와 성능기준 결과를 정리해줘.",
            {},
        )
        self.assertIn("msc_vdes_lookup", {slot.name for slot in msc.slots})

    def test_mass_summary_covers_training_interim_and_followup_lanes(self):
        plan = build_evidence_plan(
            "MSC 111에서 MASS Code 핵심 결정과 mandatory code 일정을 정리해줘.",
            {"_internal_intent": "mass_code_timeline"},
        )
        names = {slot.name for slot in plan.slots}
        self.assertIn("remote_operator_training", names)
        self.assertIn("interim_equivalent_arrangements", names)
        self.assertIn("mass_working_group_actions", names)

    def test_broad_society_lookups_have_coverage_lanes(self):
        cases = (
            (
                "DNV에서 자율운항 또는 Smart Vessel 관련 Rule/Guidance를 찾아줘.",
                "dnv_concept_qualification",
            ),
            (
                "LR에서 대체연료 관련 Rule/Guidance를 찾아줘.",
                "lr_crankcase_assessment",
            ),
            (
                "ABS에서 Smart Function 관련 Guide/Guidance를 찾아줘.",
                "abs_smart_notation",
            ),
            (
                "ABS에서 autonomous 또는 remote control function 관련 Requirements를 찾아줘.",
                "abs_risk_classification",
            ),
        )
        for question, expected_slot in cases:
            plan = build_evidence_plan(question, {})
            self.assertIn(expected_slot, {slot.name for slot in plan.slots})

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
        self.assertTrue(
            {
                "current_decision",
                "mandatory_adoption_target",
                "entry_into_force_target",
                "schedule_uncertainty",
                "experience_building",
            }.issubset({slot.name for slot in plan.slots}),
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
