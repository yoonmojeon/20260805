from __future__ import annotations

import unittest

from rag_answer_lib import RetrievedChunk
from semantic_answer_pipeline import (
    analyze_question,
    evidence_coverage,
    extract_evidence_units,
    refine_answer,
)


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
