from __future__ import annotations

import unittest

from meeting_structured_answer import _build_data_quality_sections
from rag_answer_lib import RetrievedChunk


def chunk(chunk_id: str, text: str, page: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="mepc-84-6-1",
        source="MEPC",
        file_name=(
            "MEPC 84-6-1 - Report of fuel oil consumption data submitted "
            "to the IMO Ship Fuel Oil Consumption Database.pdf"
        ),
        page_number=page,
        clause_number="",
        element_type="text",
        distance=0.0,
        text=text,
    )


class DataQualityStructuredAnswerTests(unittest.TestCase):
    def test_source_driven_answer_covers_process_errors_and_treatment(self):
        chunks = [
            chunk(
                "quality",
                (
                    "The Secretariat carried out quality control and verification "
                    "of data submitted through GISIS to identify missing ships "
                    "for which no data had been reported, "
                    "obvious errors and unrealistic characteristics. Errors were "
                    "further examined to determine the cause and provided to the "
                    "Administrations and ROs concerned."
                ),
                2,
            ),
            chunk(
                "hours",
                (
                    "Errors included hours under way exceeding the total number "
                    "of hours in the year. A total of 265 ships have not been "
                    "included in the data analysis."
                ),
                8,
            ),
        ]
        sections, used = _build_data_quality_sections(
            chunks, {"quality": 1, "hours": 2}
        )
        answer = "\n".join(sections.values())

        self.assertEqual(len(used), 2)
        self.assertIn("GISIS", answer)
        self.assertIn("비현실적", answer)
        self.assertIn("미보고 선박", answer)
        self.assertIn("hours under way", answer)
        self.assertIn("265", answer)
        self.assertIn("기국과 RO", answer)
        self.assertNotIn("Action requested", answer)
        self.assertIn("[1]", answer)
        self.assertIn("[2]", answer)

    def test_no_matching_evidence_returns_no_answer(self):
        sections, used = _build_data_quality_sections(
            [chunk("other", "Action requested of the Committee.", 3)],
            {"other": 1},
        )
        self.assertEqual(sections, {})
        self.assertEqual(used, [])

    def test_detailed_error_clause_wins_over_earlier_summary(self):
        summary = chunk(
            "summary",
            "The Secretariat carried out a quality control and verification "
            "process to identify missing ships and obvious errors.",
            3,
        )
        detail = chunk(
            "detail",
            "The automated process identified unrealistic characteristics that "
            "were not technically possible, duplicate reporting, multiple "
            "reporting entries and an incorrect ship type. The errors were "
            "further examined to determine the cause and provided to "
            "Administrations and ROs.",
            8,
        )
        sections, _used = _build_data_quality_sections(
            [summary, detail],
            {"summary": 1, "detail": 2},
        )
        answer = "\n".join(sections.values())
        self.assertIn("중복·다중 보고", answer)
        self.assertIn("선종 분류", answer)
        self.assertIn("기국과 RO", answer)
        self.assertIn("[2]", answer)


if __name__ == "__main__":
    unittest.main()
