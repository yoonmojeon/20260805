from __future__ import annotations

import unittest
from types import SimpleNamespace

from grounded_dynamic_answer import (
    build_structured_finding_bullets,
    build_prompts,
    enforce_question_relevance,
    preserve_source_qualifiers,
    repair_numeric_citations,
    validate_answer_requirements,
)
from question_requirements import analyze_requirements


class GroundedDynamicAnswerTest(unittest.TestCase):
    def test_compound_finding_and_method_require_both_parts(self):
        req = analyze_requirements(
            "IMO DCS 품질검사에서 어떤 오류를 식별했고 어떻게 처리했나?"
        )
        answer = """## 1) 핵심 요약
- 잠재 오류가 있는 선박을 확인했습니다. [1]
## 2) 선박 운항/업무 영향
- 검색 근거에서 확인되지 않음
## 3) 추후 확인 필요사항
- 검색 근거에서 확인되지 않음
## 4) 관련 선급 Rule / Guidance
- 검색 근거에서 확인되지 않음"""
        valid, warnings = validate_answer_requirements(
            answer, req, [SimpleNamespace(text="potential errors were identified")]
        )
        self.assertFalse(valid)
        self.assertNotIn("requested_finding_missing", warnings)
        self.assertIn("requested_method_missing", warnings)

    def test_finding_question_detects_incomplete_enumeration(self):
        req = analyze_requirements(
            "데이터 품질검사에서 어떤 오류를 식별했고 어떻게 처리했나?"
        )
        chunks = [
            SimpleNamespace(
                text=(
                    "duplicate reporting was removed; ships with unrealistic "
                    "parameters were excluded; ships categorized under an "
                    "incorrect ship type were further examined"
                )
            )
        ]
        answer = """## 1) 핵심 요약
- 중복 보고 오류를 식별해 제외했습니다. [1]
## 2) 선박 운항/업무 영향
- 검색 근거에서 확인되지 않음
## 3) 추후 확인 필요사항
- 검색 근거에서 확인되지 않음
## 4) 관련 선급 Rule / Guidance
- 검색 근거에서 확인되지 않음"""
        valid, warnings = validate_answer_requirements(answer, req, chunks)
        self.assertFalse(valid)
        self.assertIn("requested_finding_incomplete", warnings)

    def test_relevance_filter_preserves_distinct_findings(self):
        req = analyze_requirements(
            "데이터 품질검사에서 어떤 오류를 식별했고 어떻게 처리했나?"
        )
        draft = """## 1) 핵심 요약
- 중복 보고 154건을 제거했습니다. [1]
- 비현실적인 선박 제원을 보고한 8척을 제외했습니다. [1]
- 연간 총시간을 넘는 운항시간을 보고한 65척을 제외했습니다. [1]
## 2) 선박 운항/업무 영향
- 검색 근거에서 확인되지 않음
## 3) 추후 확인 필요사항
- 검색 근거에서 확인되지 않음
## 4) 관련 선급 Rule / Guidance
- 검색 근거에서 확인되지 않음"""
        filtered = enforce_question_relevance(draft, req)
        self.assertIn("중복 보고 154건", filtered)
        self.assertIn("비현실적인 선박 제원", filtered)
        self.assertIn("운항시간을 보고한 65척", filtered)

    def test_relevance_filter_removes_non_korean_cjk_leak(self):
        req = analyze_requirements("품질검사에서 어떤 오류를 식별했나?")
        draft = """## 1) 핵심 요약
- 데이터가 누락된船舶을 식별했습니다. [1]
- 중복 보고 오류를 식별했습니다. [1]
## 2) 선박 운항/업무 영향
- 검색 근거에서 확인되지 않음
## 3) 추후 확인 필요사항
- 검색 근거에서 확인되지 않음
## 4) 관련 선급 Rule / Guidance
- 검색 근거에서 확인되지 않음"""
        filtered = enforce_question_relevance(draft, req)
        self.assertNotIn("船舶", filtered)
        self.assertIn("중복 보고 오류", filtered)

    def test_structured_finding_extractor_keeps_type_count_and_action(self):
        chunks = [
            SimpleNamespace(
                text=(
                    "Sixty-five ships were excluded because hours under way "
                    "were more than the total number of hours in a year. "
                    "154 records were removed as they were duplicate reporting. "
                    "8 ships were excluded for reporting unrealistic ship parameters."
                )
            )
        ]
        # The production corpus uses digits for the detailed counts.
        chunks[0].text = chunks[0].text.replace("Sixty-five", "65")
        bullets = build_structured_finding_bullets(chunks)
        joined = "\n".join(bullets)
        self.assertIn("65척", joined)
        self.assertIn("154건", joined)
        self.assertIn("8척", joined)
        self.assertIn("제외", joined)

    def test_source_numeric_qualifier_is_preserved(self):
        chunks = [
            SimpleNamespace(
                text="Carbon intensity decreased by up to 10.8% in 2024 relative to 2019."
            )
        ]
        answer = "- 탄소집약도는 2019년 대비 2024년에 10.8% 감소했습니다. [1]"
        qualified = preserve_source_qualifiers(answer, chunks)
        self.assertIn("최대 10.8%", qualified)

    def test_numeric_claim_is_remapped_to_supporting_citation(self):
        chunks = [
            SimpleNamespace(text="A decrease of up to 10.8% in 2024 relative to 2019."),
            SimpleNamespace(text="Supply-based measurements include AER and cgDIST."),
        ]
        answer = "- 2024년 개선 폭은 최대 10.8%입니다. [2]"
        repaired = repair_numeric_citations(answer, chunks)
        self.assertIn("[1]", repaired)
        self.assertNotIn("[2]", repaired)

    def setUp(self) -> None:
        self.question = (
            "MEPC 84에서 2024년 탄소집약도 개선 수치는 무엇이며, "
            "어떤 지표를 사용했나?"
        )
        self.chunks = [
            SimpleNamespace(
                source="MEPC",
                file_name="report.pdf",
                page_number=5,
                clause_number="3",
                text=(
                    "The fleet average supply-based carbon intensity in AER "
                    "and cgDIST decreased by up to 10.8% in 2024 compared to "
                    "2019. Demand-based estimated EEOI was also used."
                ),
            )
        ]

    def test_prompt_explicitly_requires_value_and_metric(self) -> None:
        _, user, req = build_prompts(self.question, {}, self.chunks)
        self.assertIn("수치", user)
        self.assertIn("사용 지표", user)
        self.assertIn("10.8%", user)
        self.assertIn("EEOI", user)
        self.assertIn("value", req.facets)

    def test_validation_detects_missing_metric(self) -> None:
        req = analyze_requirements(self.question)
        valid, warnings = validate_answer_requirements(
            "## 1) 핵심 요약\n- 2019년 대비 10.8% 개선되었습니다. [1]",
            req,
            self.chunks,
        )
        self.assertFalse(valid)
        self.assertIn("requested_metric_missing", warnings)

    def test_validation_accepts_grounded_complete_answer(self) -> None:
        req = analyze_requirements(self.question)
        answer = """## 1) 핵심 요약
- 2024년 공급기반 AER·cgDIST는 2019년 대비 최대 10.8% 감소했습니다. [1]
- 사용 지표는 공급기반 AER·cgDIST와 수요기반 추정 EEOI입니다. [1]
## 2) 선박 운항/업무 영향
- 검색 근거에서 확인되지 않음
## 3) 추후 확인 필요사항
- 검색 근거에서 확인되지 않음
## 4) 관련 선급 Rule / Guidance
- 검색 근거에서 확인되지 않음"""
        valid, warnings = validate_answer_requirements(answer, req, self.chunks)
        self.assertTrue(valid, warnings)

    def test_relevance_filter_removes_generic_impact_and_duplicate_value(self) -> None:
        req = analyze_requirements(self.question)
        draft = """## 1) 핵심 요약
- 개선 수치는 10.8%입니다. [1]
- 2019년 대비 평균 최대 10.8% 감소했습니다. [1]
- 지표는 AER, cgDIST, EEOI입니다. [1]
## 2) 선박 운항/업무 영향
- 보고서를 검토하고 적절히 대응해야 합니다. [1]
## 3) 추후 확인 필요사항
- 보고서를 추가 검토해야 합니다. [1]
## 4) 관련 선급 Rule / Guidance
- 관련 선급 지침입니다. [1]"""
        filtered = enforce_question_relevance(draft, req)
        self.assertEqual(filtered.count("10.8%"), 1)
        self.assertNotIn("적절히 대응", filtered)
        self.assertNotIn("추가 검토해야", filtered)
        self.assertNotIn("관련 선급 지침", filtered)


if __name__ == "__main__":
    unittest.main()
