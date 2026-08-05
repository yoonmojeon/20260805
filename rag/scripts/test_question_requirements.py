from __future__ import annotations

import unittest

from question_requirements import analyze_requirements


class QuestionRequirementsTest(unittest.TestCase):
    def test_broad_autonomous_rule_lookup_is_document_request(self):
        requirements = analyze_requirements(
            "DNV에서 자율운항 또는 Smart Vessel 관련 Rule/Guidance를 찾아줘."
        )

        self.assertIn("document", requirements.facets)
        self.assertNotIn("clause", requirements.facets)
        self.assertNotIn("impact", requirements.facets)

    def test_compound_finding_and_method_question(self):
        req = analyze_requirements(
            "MEPC 84/6/1의 IMO DCS 데이터 품질검사에서 어떤 오류를 식별했고 "
            "어떻게 처리했나?"
        )
        self.assertIn("finding", req.facets)
        self.assertIn("method", req.facets)
        self.assertEqual(("MEPC 84/6/1",), req.document_identifiers)

    def test_extracts_full_meeting_document_identifier(self):
        req = analyze_requirements(
            "MEPC 84/6/1의 IMO DCS 데이터 품질검사에서 어떤 오류를 식별했나?"
        )
        self.assertEqual(req.organization, "MEPC")
        self.assertEqual(req.session_number, "84")
        self.assertEqual(req.document_identifiers, ("MEPC 84/6/1",))
        self.assertIn("error", req.topic_terms)

    def test_numeric_metric_question_is_not_broad_summary(self) -> None:
        req = analyze_requirements(
            "MEPC 84에서 2024년 탄소집약도 개선 수치는 무엇이며, 어떤 지표를 사용했나?"
        )
        self.assertEqual(req.organization, "MEPC")
        self.assertEqual(req.session_number, "84")
        self.assertIn("value", req.facets)
        self.assertIn("metric", req.facets)
        self.assertIn("carbon intensity", req.topic_terms)
        self.assertIn("EEOI", req.topic_terms)
        self.assertFalse(req.broad_summary)

    def test_latest_overview_remains_broad(self) -> None:
        req = analyze_requirements(
            "환경규제 대응과 관련된 최신 MEPC 회의 주요 내용을 정리해줘."
        )
        self.assertTrue(req.broad_summary)

    def test_society_clause_question_extracts_requirements(self) -> None:
        req = analyze_requirements(
            "DNV 자율운항 선박에서 ROC의 status and situational awareness 요구사항을 찾아줘."
        )
        self.assertEqual(req.organization, "DNV")
        self.assertIn("requirement", req.facets)
        self.assertIn("ROC", req.topic_terms)
        self.assertIn("situational awareness", req.topic_terms)

    def test_search_queries_follow_requested_facets(self) -> None:
        req = analyze_requirements(
            "MEPC 84에서 탄소집약도 개선 수치와 사용 지표를 알려줘."
        )
        queries = req.search_queries()
        joined = " ".join(queries)
        self.assertIn("percentage", joined)
        self.assertIn("AER", joined)
        self.assertNotIn("duplicate reporting", joined)


if __name__ == "__main__":
    unittest.main()
