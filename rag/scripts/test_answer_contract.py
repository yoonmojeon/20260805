from __future__ import annotations

import unittest
from types import SimpleNamespace

from answer_contract import apply_answer_contract
from rag_answer_lib import build_answer_verification


def _chunk(number: int, text: str = "원문 근거") -> SimpleNamespace:
    return SimpleNamespace(
        chunk_id=f"chunk-{number}",
        doc_id=f"doc-{number}",
        file_name=f"document-{number}.pdf",
        page_number=number * 10,
        text=text,
    )


class AnswerContractTests(unittest.TestCase):
    def test_accurate_verification_ignores_runtime_metadata(self) -> None:
        result = build_answer_verification(
            {"question": "테스트 질문", "category": "trend_summary"},
            [],
            "검색 근거가 부족합니다.",
            config_dict={
                "latency_mode": "accurate",
                "fast_meta": {"cached": True},
                "question_mode": "broad",
                "question_category": "trend_summary",
                "answer_mode": "standard_rag",
            },
        )

        self.assertEqual(result["verification_summary"]["answer_mode"], "일반 RAG")

    def test_every_sentence_in_a_multi_sentence_bullet_gets_citation(self) -> None:
        result = apply_answer_contract(
            "## 1) 핵심 요약\n\n- 첫 번째 사실입니다. 두 번째 사실입니다. [1]",
            [_chunk(1)],
        )

        self.assertIn("## 1) 핵심 요약", result.answer)
        self.assertIn("- 첫 번째 사실입니다. [1]", result.answer)
        self.assertIn("- 두 번째 사실입니다. [1]", result.answer)
        self.assertTrue(result.valid)

    def test_uncited_claim_and_citation_only_line_are_blocked(self) -> None:
        result = apply_answer_contract(
            "## 핵심 요약\n\n- 근거 없는 주장입니다.\n- [1]\n- 확인된 주장입니다. [1]",
            [_chunk(1)],
        )

        self.assertNotIn("근거 없는 주장", result.answer)
        self.assertNotIn("- [1]", result.answer)
        self.assertIn("확인된 주장입니다. [1]", result.answer)

    def test_evidence_table_contains_only_cited_chunks_in_first_use_order(self) -> None:
        chunks = [_chunk(1, "첫 근거"), _chunk(2, "둘째 근거"), _chunk(3, "셋째 근거")]
        result = apply_answer_contract(
            "## 핵심 답변\n\n- 셋째 근거입니다. [3]\n- 첫 근거입니다. [1]",
            chunks,
        )

        self.assertEqual([row["citation_id"] for row in result.evidence_table], ["[3]", "[1]"])
        self.assertEqual(result.evidence_table[0]["file_name"], "document-3.pdf")
        self.assertEqual(result.evidence_table[0]["page"], 30)
        self.assertEqual(result.evidence_table[0]["chunk_id"], "chunk-3")
        self.assertEqual(result.evidence_table[0]["chunk_preview"], "셋째 근거")

    def test_numbered_section_headings_are_preserved(self) -> None:
        result = apply_answer_contract(
            "## 핵심 답변\n\n- 핵심입니다. [1]\n\n"
            "## 실무 영향\n\n- 영향입니다. [1]\n\n"
            "## 확인 필요\n\n- 확인 항목입니다. [1]",
            [_chunk(1)],
        )
        self.assertIn("## 1) 핵심 요약", result.answer)
        self.assertIn("## 2) 선박 운항/업무 영향", result.answer)
        self.assertIn("## 3) 추후 확인 필요사항", result.answer)

    def test_out_of_range_citation_is_not_exposed(self) -> None:
        result = apply_answer_contract(
            "## 핵심 답변\n\n- 지원되는 사실입니다. [1][9]",
            [_chunk(1)],
        )

        self.assertIn("지원되는 사실입니다. [1]", result.answer)
        self.assertNotIn("[9]", result.answer)
        self.assertEqual(len(result.evidence_table), 1)

    def test_no_evidence_answer_is_transparent(self) -> None:
        result = apply_answer_contract("검색된 근거가 없습니다.", [])

        self.assertIn("직접 답할 근거를 찾지 못했습니다", result.answer)
        self.assertEqual(result.evidence_table, [])
        self.assertTrue(result.valid)

    def test_rule_guidance_bullet_is_not_misclassified_as_heading(self) -> None:
        result = apply_answer_contract(
            "## 4) 관련 선급 Rule / Guidance\n\n"
            "- **DNV-CG-0264** (Rule/Guidance): 적용 조항입니다. [1]",
            [_chunk(1)],
        )

        self.assertIn("**DNV-CG-0264** (Rule/Guidance)", result.answer)
        self.assertNotIn(
            "관련 선급 Rule / Guidance가 검색 근거에 없거나 해당하지 않습니다",
            result.answer,
        )


if __name__ == "__main__":
    unittest.main()
