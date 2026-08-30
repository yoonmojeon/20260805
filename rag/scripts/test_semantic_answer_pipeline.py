from __future__ import annotations

import unittest

from rag_answer_lib import RetrievedChunk
from semantic_answer_pipeline import (
    analyze_question,
    evidence_coverage,
    extract_evidence_units,
    refine_answer,
)
from answer_depth_guidance import apply_category_total_bullet_limit
from grounded_dynamic_answer import repair_deadline_fact_answer
from question_classifier import classify_question_category


def chunk(text: str, *, source: str = "MSC", page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"c{page}",
        doc_id="d",
        source=source,
        file_name="MSC 111-WP.1 - Draft Report.pdf",
        page_number=page,
        clause_number="6.30",
        element_type="text",
        distance=0.1,
        text=text,
    )


class SemanticAnswerPipelineTests(unittest.TestCase):
    def test_agreed_seven_question_category_boundary(self):
        cases = {
            "환경규제 대응과 관련된 최신 MEPC 회의 주요 내용을 정리해줘.": "trend_summary",
            "MSC 111의 주요 결과를 3개 항목으로 요약해줘.": "trend_summary",
            "최신 MEPC 회의에서 선박 운항 및 규제 보고에 직접 영향을 주는 사항을 정리해줘.": "env_regulation",
            "MSC 111에서 대체연료·GHG 안전규제와 관련된 논의 및 결론을 요약해줘.": "env_regulation",
            "MSC 111에서 MASS Code와 관련된 핵심 결정사항을 요약하고, 향후 mandatory code 일정까지 정리해줘.": "autonomous",
            "DNV에서 자율운항 또는 Smart Vessel 관련 Rule/Guidance를 찾아줘.": "rule_lookup",
            "LR에서 대체연료 관련 Rule/Guidance를 찾아줘.": "rule_lookup",
        }
        self.assertEqual(
            cases,
            {question: classify_question_category(question, {}) for question in cases},
        )

    def test_alternative_fuel_environment_plan_gets_five_to_seven_budget(self):
        plan = analyze_question(
            "MSC 111에서 대체연료·GHG 안전규제와 관련된 논의 및 결론을 요약해줘.",
            {"category": "env_regulation"},
        )
        self.assertEqual("operational_impact", plan.task)
        self.assertEqual((5, 7), (plan.bullet_min, plan.bullet_max))

    def test_rule_whole_answer_is_limited_to_three_bullets(self):
        draft = """## 1) 핵심 요약
- 첫 문서입니다. [1]
- 둘째 문서입니다. [2]
- 셋째 문서입니다. [3]
## 2) 선박 운항/업무 영향
- 영향입니다. [1]
## 3) 추후 확인 필요사항
- 추가 확인 필요사항이 별도로 식별되지 않았습니다.
## 4) 관련 선급 Rule / Guidance
- 핵심 지침입니다. [1]"""
        answer, meta = apply_category_total_bullet_limit(draft, "rule_lookup")
        bullets = [line for line in answer.splitlines() if line.startswith("- ")]
        self.assertEqual(3, len(bullets))
        self.assertEqual(3, meta["after"])
        self.assertIn("> 추가 확인 필요사항", answer)
        self.assertIn("핵심 지침", answer)

    def test_rule_identity_is_visible_in_section_four_when_builder_omits_it(self):
        draft = """## 1) 핵심 요약
- **LR Notice No.1 — Section 15**: 저인화점 연료 적용범위입니다. [1]
- **LR Notice No.1 — dual fuel**: 환기 요건입니다. [2]
## 2) 선박 운항/업무 영향
> 없음
## 3) 추후 확인 필요사항
- 추가 검토가 필요합니다. [1]
## 4) 관련 선급 Rule / Guidance
> 없음"""
        answer, meta = apply_category_total_bullet_limit(draft, "rule_lookup")
        section4 = answer.split("## 4) 관련 선급 Rule / Guidance", 1)[1]
        self.assertIn("LR Notice No.1", section4)
        self.assertEqual(3, meta["after"])

    def test_exact_rule_fact_is_compact_and_deduplicated(self):
        draft = """## 1) 핵심 요약
- 전력 케이블은 1 kV 및 3 kV급(power cables for rated voltages 1 kV and 3 kV)입니다. [1]
- 제어·계측 회로용 케이블은 150/250 V(300 V)입니다. [1]
- 해당 프로그램은 제어·계측 회로용 케이블 150/250 V(300 V)의 형식승인 기반입니다. [1]
## 2) 선박 운항/업무 영향
- 설계 검토에 반영합니다. [1]
## 3) 추후 확인 필요사항
- 인접 조항을 확인합니다. [1]
## 4) 관련 선급 Rule / Guidance
- **DNV-CP-0399**, p.5-6입니다. [1]"""
        row = {
            "_answer_profile": "exact_rule_fact",
            "_answer_fact_slots": 2,
        }
        answer, meta = apply_category_total_bullet_limit(
            draft, "rule_lookup", row
        )

        self.assertIn("1 kV 및 3 kV", answer)
        self.assertNotIn("power cables for rated voltages", answer)
        self.assertIn("150/250 V(300 V)", answer)
        self.assertNotIn("해당 프로그램", answer)
        self.assertNotIn("## 2)", answer)
        self.assertNotIn("## 3)", answer)
        self.assertEqual("exact_rule_fact", meta["answer_profile"])
        self.assertEqual(
            1,
            meta["duplicate_facts_removed"] + meta["recap_duplicates_removed"],
        )
        self.assertEqual(
            3,
            len([line for line in answer.splitlines() if line.startswith("- ")]),
        )

    def test_exact_rule_fact_selects_question_relevant_late_bullet(self):
        draft = """## 1) 핵심 요약
- 중앙 제어소에서 지락 고장을 표시해야 합니다. [1]
- 저임피던스 시스템에서는 회로를 차단해야 합니다. [1]
- 중앙 제어소와 현장 수동 제어 사이의 제어권 전환에는 수신 측 확인이 필요하지 않습니다. [3]
## 2) 선박 운항/업무 영향
> 없음
## 3) 추후 확인 필요사항
> 없음
## 4) 관련 선급 Rule / Guidance
- ABS Part 4, 13.11 [3]"""
        row = {
            "question": "제어권 전환 시 수신 측의 확인(acknowledgment)이 필요한가요?",
            "_answer_profile": "exact_rule_fact",
            "_answer_fact_slots": 1,
        }
        answer, _meta = apply_category_total_bullet_limit(draft, "trend_summary", row)

        self.assertIn("필요하지 않습니다", answer)
        self.assertNotIn("지락 고장", answer)

    def test_later_aggregate_recap_is_removed(self):
        draft = """## 1) 핵심 요약
- 해양 활동이 주요 외부 위험입니다. [1]
- 항해 중 끌린 닻이 손상 원인입니다. [1]
- 주요 외부 위험은 해양 활동이며 끌린 닻이 손상 원인입니다. [1]
## 2) 선박 운항/업무 영향
> 없음
## 3) 추후 확인 필요사항
> 없음
## 4) 관련 선급 Rule / Guidance
> 없음"""
        answer, meta = apply_category_total_bullet_limit(draft, "trend_summary", {})

        self.assertEqual(2, len([line for line in answer.splitlines() if line.startswith("- ")]))
        self.assertEqual(1, meta["recap_duplicates_removed"])

    def test_submission_deadline_uses_explicit_by_sentence(self):
        draft = """## 1) 핵심 요약
- 임시지침은 2024년 10월에 승인되었습니다. [1]
## 2) 선박 운항/업무 영향
> 없음
## 3) 추후 확인 필요사항
> 없음
## 4) 관련 선급 Rule / Guidance
> 없음"""
        chunks = [
            type(
                "Chunk",
                (),
                {
                    "text": (
                        "The decision invited Parties and observers to submit "
                        "comments, by 15 November 2025, on the provisional guidance."
                    )
                },
            )()
        ]
        answer, used = repair_deadline_fact_answer(
            draft,
            "당사자 및 참관인이 의견을 제출해야 하는 마감일은 언제인가요?",
            chunks,
        )

        self.assertTrue(used)
        self.assertIn("2025년 11월 15일까지", answer)
        self.assertNotIn("2024년 10월", answer)

    def test_question_plan_has_no_question_id_dependency(self):
        plan = analyze_question(
            "최신 MEPC 회의에서 규제 보고에 직접 영향을 주는 사항을 정리해줘.",
            {},
        )
        self.assertEqual("operational_impact", plan.task)
        self.assertTrue(plan.require_operational_impact)
        self.assertIn("verification", plan.required_evidence)

    def test_evidence_units_are_typed(self):
        plan = analyze_question("MSC 111 주요 결정을 정리해줘.", {})
        units = extract_evidence_units(
            [chunk("6.30 The Committee approved the interim guidelines for ships using hydrogen as fuel.")],
            plan,
        )
        self.assertEqual("decision", units[0].evidence_type)
        self.assertTrue(evidence_coverage(plan, units)["decision"])

    def test_generic_uncited_claim_is_removed(self):
        chunks = [
            chunk("6.30 The Committee approved the interim guidelines for ships using hydrogen as fuel.")
        ]
        draft = """## 1) 핵심 요약
- 일반적으로 대체연료 안전관리가 중요합니다.
- MSC 111에서 수소연료 선박 임시지침을 승인했습니다. [1]
## 2) 선박 운항/업무 영향
## 3) 추후 확인 필요사항
## 4) 관련 선급 Rule / Guidance"""
        result = refine_answer("MSC 111 주요 결정을 정리해줘.", {}, draft, chunks)
        self.assertNotIn("일반적으로", result.answer)
        self.assertIn("수소연료", result.answer)

    def test_paraphrases_share_the_same_task(self):
        variants = [
            "MEPC 규제보고가 선박 업무에 주는 직접 영향은?",
            "선사에서 준비할 MEPC 보고·검증 업무를 알려줘.",
            "MEPC 제출 및 검증 요구가 실무에 미치는 영향을 정리해줘.",
        ]
        self.assertEqual(
            {"operational_impact"},
            {analyze_question(question, {}).task for question in variants},
        )

    def test_rule_paraphrases_do_not_need_fixture_ids(self):
        variants = [
            "DNV 자율운항 관련 지침을 찾아줘.",
            "DNV Smart Vessel에 적용할 class guidance는?",
            "자율·원격운항 선박에 대한 DNV rule을 알려줘.",
        ]
        plans = [analyze_question(question, {}) for question in variants]
        self.assertTrue(all(plan.task == "rule_lookup" for plan in plans))
        self.assertTrue(all(plan.organization == "DNV" for plan in plans))


if __name__ == "__main__":
    unittest.main()
