from __future__ import annotations

import unittest

from direct_clause_grounding import (
    build_clause_proposition_block,
    ensure_direct_clause_source_details,
    extract_clause_reference,
    modality_violation,
    replace_rule_reference_section,
    select_specific_clause_chunks,
    validate_direct_clause_answer,
)
from rag_answer_lib import RetrievedChunk


def dnv_chunk(
    chunk_id: str = "direct",
    *,
    page: int = 88,
    clause: str = "6",
    text: str = "",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="dnv-cg-0264",
        source="DNV",
        file_name="DNV-CG-0264.pdf",
        page_number=page,
        clause_number=clause,
        element_type="text",
        distance=0.0,
        text=text
        or (
            "6 6.4.1 Status and situational awareness. It should be possible "
            "to observe real-time operational status, readiness and capacity "
            "of the vessel function or system from ROC. CCTV should be "
            "considered to present video of the vessel's exterior and interior "
            "to the ROC operator. Such surveillance may provide important "
            "contextual information to support decision-making."
        ),
    )


class DirectClauseGroundingTests(unittest.TestCase):
    def test_body_clause_overrides_section_level_metadata(self):
        clause, title = extract_clause_reference(
            dnv_chunk(
                text=(
                    "6.4.1 Status and situational awareness\n"
                    "It should be possible to observe real-time status."
                )
            )
        )
        self.assertEqual(clause, "6.4.1")
        self.assertEqual(title, "Status and situational awareness")

    def test_propositions_preserve_modal_labels(self):
        chunk = dnv_chunk(
            text=(
                "6.4.1 Status and situational awareness\n"
                "The operator should achieve situational awareness. "
                "CCTV should be considered for video. "
                "Surveillance may provide contextual information."
            )
        )
        block = build_clause_proposition_block(chunk)
        self.assertIn("modality=SHOULD", block)
        self.assertIn("modality=CONSIDER", block)
        self.assertIn("modality=PERMISSIVE", block)
        self.assertNotIn("6.4.1 Status", block)
        self.assertNotIn("proposition 1", block)
        self.assertIn("source=[1]", block)

    def test_invented_consequence_is_rejected(self):
        evidence = (
            "It should be possible to observe real-time operational status, "
            "readiness and capacity from ROC."
        )
        claim = (
            "\uad00\ucc30\ud560 \uc218 \uc5c6\uac8c \ub418\uba74 "
            "\uc6d0\uaca9\uc6b4\ud56d\uc774 \uc5b4\ub824\uc6cc\uc9c4\ub2e4."
        )
        self.assertEqual(
            modality_violation(claim, evidence),
            "unsupported_inferred_consequence",
        )

    def test_invented_safety_threat_is_rejected(self):
        evidence = "The status should be observable from ROC."
        claim = (
            "\uc0c1\ud0dc\ub97c \ud655\uc778\ud560 \uc218 \uc5c6\uc744 \ub54c "
            "\uc120\ubc15 \uc548\uc804\uc131\uc774 \uc704\ud611\ubc1b\uc744 \uc218 \uc788\ub2e4."
        )
        self.assertEqual(
            modality_violation(claim, evidence),
            "unsupported_speculative_impact",
        )

    def test_specific_context_excludes_scope_and_unrelated_chunks(self):
        scope = dnv_chunk(
            "scope",
            page=9,
            clause="2",
            text="2 Objective. This guideline provides general guidance.",
        )
        emergency = dnv_chunk(
            "emergency",
            page=86,
            clause="5.5.2.3",
            text="5.5.2.3 Emergency power requirements shall be applied.",
        )
        direct = dnv_chunk()
        row = {
            "_evidence_completion": {
                "slot_hits": {"specific_clause": ["direct"]}
            }
        }
        selected = select_specific_clause_chunks(
            row, [scope, direct, emergency], [scope, direct, emergency]
        )
        self.assertEqual([chunk.chunk_id for chunk in selected], ["direct"])

    def test_unheaded_continuation_is_excluded_when_heading_exists(self):
        headed = dnv_chunk(
            "headed",
            text=(
                "6.4.1 Status and situational awareness\n"
                "The status should be observable from ROC."
            ),
        )
        fragment = dnv_chunk(
            "fragment",
            page=89,
            text="status and situational awareness is lost during an outage.",
        )
        row = {
            "_evidence_completion": {
                "slot_hits": {"specific_clause": ["headed", "fragment"]}
            }
        }
        selected = select_specific_clause_chunks(
            row, [headed, fragment], [headed, fragment]
        )
        self.assertEqual([chunk.chunk_id for chunk in selected], ["headed"])

    def test_should_be_considered_cannot_become_mandatory(self):
        evidence = "CCTV should be considered to present video to the ROC operator."
        self.assertEqual(
            modality_violation("CCTV 영상 감시는 필수적입니다.", evidence),
            "modal_strengthened_to_mandatory",
        )

    def test_may_cannot_become_duty(self):
        evidence = "Such surveillance may provide contextual information."
        self.assertEqual(
            modality_violation("감시 영상은 상황정보를 제공해야 합니다.", evidence),
            "permission_strengthened_to_duty",
        )

    def test_consideration_wording_is_preserved(self):
        evidence = "CCTV should be considered to present video to the ROC operator."
        self.assertIsNone(
            modality_violation("ROC 영상 제공을 위해 CCTV를 고려해야 합니다.", evidence)
        )

    def test_invalid_modal_claim_is_removed(self):
        answer = """## 1) 핵심 요약
- CCTV 영상 감시는 필수적입니다. [1]
- ROC에서 실시간 상태를 관찰할 수 있어야 합니다. [1]

## 2) 선박 운항/업무 영향
- CCTV 적용을 고려해야 합니다. [1]

## 3) 추후 확인 필요사항

## 4) 관련 선급 Rule / Guidance
- 임시 문구 [1]"""
        checked, rows, warnings = validate_direct_clause_answer(
            answer, [dnv_chunk()]
        )
        self.assertNotIn("필수적", checked)
        self.assertIn("실시간 상태", checked)
        self.assertTrue(any(not row["supported"] for row in rows))
        self.assertIn(
            "direct_clause_claim_removed:modal_strengthened_to_mandatory",
            warnings,
        )

    def test_rule_reference_is_derived_from_chunk(self):
        answer = """## 1) 핵심 요약
- 근거 내용입니다. [1]

## 2) 선박 운항/업무 영향
- 영향입니다. [1]

## 3) 추후 확인 필요사항
- 확인사항입니다. [1]

## 4) 관련 선급 Rule / Guidance
- 잘못된 p.9 clause 6 표기입니다. [1]"""
        replaced = replace_rule_reference_section(answer, [dnv_chunk()])
        self.assertIn("p.88", replaced)
        self.assertIn("clause 6.4.1", replaced)
        self.assertNotIn("p.9 clause 6", replaced)

    def test_source_details_are_not_silently_dropped(self):
        chunk = dnv_chunk(
            text=(
                "6.4.1 Status and situational awareness. "
                "The existing requirements for unattended machinery space "
                "operations should be observed. Additional considerations "
                "should be given to human senses. Compensating measures may "
                "include infrared cameras, microphones or vibration sensors, "
                "and communication solutions enabling efficient ship-shore "
                "collaboration."
            )
        )
        answer = """## 1) 핵심 요약
- ROC에서 상태를 확인할 수 있어야 합니다. [1]

## 2) 선박 운항/업무 영향

## 3) 추후 확인 필요사항

## 4) 관련 선급 Rule / Guidance
- DNV-CG-0264, 6.4.1 [1]"""
        enriched = ensure_direct_clause_source_details(answer, [chunk])
        self.assertIn("적외선", enriched)
        self.assertIn("마이크", enriched)
        self.assertIn("진동 센서", enriched)
        self.assertIn("선박-육상", enriched)
        self.assertIn("무인 기관실", enriched)
        for line in enriched.splitlines():
            if line.strip().startswith("- "):
                self.assertRegex(line, r"\[1\]\s*$")


if __name__ == "__main__":
    unittest.main()
