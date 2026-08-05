from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path

from grounded_answer_policy import (
    classify_document_status,
    select_key_clause_chunks,
    verify_claim_citations,
)
from rag_answer_lib import RetrievedChunk
from rule_lookup_context import enrich_rule_lookup_chunks
from rule_lookup_structured_answer import build_rule_lookup_structured_answer
from meeting_category_profile import build_meeting_retrieval_profile
from meeting_structured_answer import build_meeting_structured_answer, _english_outcome_to_ko_clause
from retrieval_query_analysis import analyze_query


@dataclass
class Chunk:
    chunk_id: str
    file_name: str
    source: str
    text: str
    page_number: int = 1


class GroundedAnswerPolicyTests(unittest.TestCase):
    def test_latest_mepc_resolves_to_index_session(self) -> None:
        signals = analyze_query("환경규제 대응과 관련된 최신 MEPC 회의 주요 내용을 정리해줘.")
        self.assertIn(("MEPC", 84), signals.session_codes)
        self.assertTrue(signals.meeting_outcome_question)

    def test_citation_only_bullet_is_removed(self) -> None:
        chunk = Chunk("x", "MEPC 84-6-1 - Report.pdf", "MEPC", "Verification of submitted data. " * 8)
        verified, rows, warnings = verify_claim_citations("## 1) 핵심 요약\n\n- [1]", [chunk])
        self.assertNotIn("- [1]", verified)
        self.assertEqual(rows[0]["reason"], "empty_claim")
        self.assertIn("unsupported_claim_removed:empty_claim", warnings)

    def test_resolution_action_is_bound_to_its_object(self) -> None:
        sentence = (
            "MEPC 81 adopted amendments to appendix IX of MARPOL Annex VI "
            "(resolution MEPC.385(81) on information to be submitted to the IMO DCS)."
        )
        summary = _english_outcome_to_ko_clause(sentence)
        self.assertIn("MARPOL Annex VI 부록 IX", summary)
        self.assertIn("채택", summary)
        self.assertNotIn("CII·carbon intensity 보고", summary)

    def test_proposal_cannot_prove_adoption(self) -> None:
        chunk = Chunk(
            "p1",
            "MSC 111-5-10 - Proposal for the MASS EBP.pdf",
            "MSC",
            "The Committee is invited to consider the proposed work plan for the MASS Code. " * 3,
        )
        self.assertEqual(classify_document_status(chunk).code, "proposal")
        answer = "## 1) 핵심 요약\n\n- MASS Code 의무 전환 일정이 확정되었습니다. [1]"
        verified, rows, warnings = verify_claim_citations(answer, [chunk])
        self.assertNotIn("확정되었습니다", verified)
        self.assertFalse(rows[0]["supported"])
        self.assertIn("unsupported_claim_removed", warnings[0])

    def test_adopted_instrument_supports_precise_resolution(self) -> None:
        chunk = Chunk(
            "r1",
            "Resolution MEPC.385(81).pdf",
            "MEPC",
            "RESOLUTION MEPC.385(81) ADOPTS amendments to MARPOL Annex VI. " * 3,
        )
        self.assertEqual(classify_document_status(chunk).code, "adopted_instrument")
        answer = "## 1) 핵심 요약\n\n- MEPC.385(81)은 MARPOL Annex VI 개정을 채택했습니다. [1]"
        verified, rows, warnings = verify_claim_citations(answer, [chunk])
        self.assertIn("MEPC.385(81)", verified)
        self.assertTrue(rows[0]["supported"])
        self.assertEqual(warnings, [])

    def test_wrong_citation_document_code_is_removed(self) -> None:
        chunk = Chunk(
            "r1",
            "Resolution MEPC.385(81).pdf",
            "MEPC",
            "RESOLUTION MEPC.385(81) ADOPTS amendments to MARPOL Annex VI. " * 3,
        )
        answer = "## 1) 핵심 요약\n\n- MEPC.338(76)이 CII 지침을 채택했습니다. [1]"
        verified, rows, _warnings = verify_claim_citations(answer, [chunk])
        self.assertNotIn("MEPC.338(76)", verified)
        self.assertEqual(rows[0]["reason"].split(":", 1)[0], "document_code_not_in_evidence")

    def test_resolution_topic_must_be_in_same_sentence(self) -> None:
        chunk = Chunk(
            "r1",
            "Resolution MEPC.385(81).pdf",
            "MEPC",
            (
                "The report separately discusses CII trends. "
                "MEPC 81 adopted amendments to appendix IX of MARPOL Annex VI "
                "(resolution MEPC.385(81) on information submitted to the IMO DCS). "
            ) * 3,
        )
        answer = "## 1) 핵심 요약\n\n- MEPC.385(81)이 CII 보고를 채택했습니다. [1]"
        verified, rows, _warnings = verify_claim_citations(answer, [chunk])
        self.assertNotIn("CII 보고를 채택", verified)
        self.assertEqual(rows[0]["reason"].split(":", 1)[0], "resolution_topic_not_in_same_sentence")

    def test_descriptive_report_cannot_be_turned_into_company_instruction(self) -> None:
        chunk = Chunk(
            "r1",
            "MEPC 84-10 - Outcome of PPR 13 (Secretariat).pdf",
            "MEPC",
            "The draft 2026 Guidelines address oily wastes in machinery spaces and IBTS. " * 3,
        )
        answer = "## 2) 선박 운항/업무 영향\n\n- 기술부서는 기존 절차와 초안의 차이를 추적해야 합니다. [1]"
        verified, rows, _warnings = verify_claim_citations(answer, [chunk])
        self.assertNotIn("기술부서는", verified)
        self.assertEqual(rows[0]["reason"], "prescriptive_inference_not_in_evidence")

    def test_explicitly_postponed_adoption_status_is_allowed(self) -> None:
        chunk = Chunk(
            "m1",
            "MEPC 84-3 - Amendments to MARPOL Annex VI (Secretariat).pdf",
            "MEPC",
            (
                "Following the decision of MEPC/ES.2 to adjourn for one year the discussion on the adoption "
                "of the draft revised MARPOL Annex VI 2025, amendments on data reporting, the North-East "
                "Atlantic Emission Control Area and IMO DCS accessibility were circulated. "
            ) * 2,
        )
        answer = (
            "## 1) 핵심 요약\n\n- MARPOL Annex VI 2025 개정안 채택 논의가 MEPC/ES.2에서 "
            "1년 연기됐습니다. [1]"
        )
        verified, rows, warnings = verify_claim_citations(answer, [chunk])
        self.assertIn("1년 연기", verified)
        self.assertTrue(rows[0]["supported"])
        self.assertEqual(warnings, [])

    def test_key_clause_selection_rejects_repeated_header(self) -> None:
        header = Chunk(
            "h",
            "DNV-CG-0264.pdf",
            "DNV",
            "Class guideline — DNV-CG-0264. Edition December 2025. " * 4,
        )
        clause = Chunk(
            "c",
            "DNV-CG-0264.pdf",
            "DNV",
            "2 Objective. This class guideline provides requirements for autonomous and remotely operated vessels, including notation and verification requirements. " * 2,
        )
        selected = select_key_clause_chunks("DNV 자율운항 Guidance", [header, clause], limit=2)
        self.assertEqual([c.chunk_id for c in selected], ["c"])

    def test_dnv_header_hits_are_replaced_by_local_key_clauses(self) -> None:
        headers = [
            RetrievedChunk(
                chunk_id=f"h{i}",
                doc_id="dnv_dnv_class_2026_04_dnv_cg_0264_00e0cc67",
                source="DNV",
                file_name="DNV-CG-0264.pdf",
                page_number=page,
                clause_number="",
                element_type="text",
                distance=0.1,
                text="Class guideline — DNV-CG-0264. Edition December 2025 Autonomous and remotely operated vessels",
            )
            for i, page in enumerate((66, 68, 79), 1)
        ]
        row = {
            "category": "rule_lookup",
            "question": "DNV에서 자율운항 관련 Rule/Guidance를 찾아줘.",
        }
        enriched = enrich_rule_lookup_chunks(
            headers,
            headers,
            chunks_dir=Path("data/processed/chunks"),
            row=row,
        )
        self.assertTrue(any(c.page_number in {9, 27} for c in enriched))
        self.assertTrue(all("Edition December" not in c.text[:120] for c in enriched))
        answer, _warnings = build_rule_lookup_structured_answer(
            enriched,
            question=row["question"],
            pool=enriched,
        )
        self.assertIn("DNV-CG-0264", answer)
        self.assertRegex(answer, r"\[[1-3]\]")

    def test_meeting_answer_attributes_draft_and_blocks_proposal_as_decision(self) -> None:
        proposal = RetrievedChunk(
            "p1", "proposal", "MSC",
            "MSC 111-5-10 - Proposal for the MASS EBP.pdf", 1, "", "text", 0.1,
            "The proposal states that the mandatory MASS Code should be adopted in 2030. " * 3,
        )
        draft = RetrievedChunk(
            "d1", "draft", "MSC",
            "MSC 111-WP.1 - Draft Report Of The Maritime Safety Committee.pdf", 42, "5", "text", 0.2,
            "The Committee adopted the non-mandatory MASS Code and agreed to continue the experience-building phase. " * 3,
        )
        question = "MSC 111 MASS Code의 결정 상태와 mandatory 일정을 정리해줘."
        row = {"category": "autonomous", "question": question}
        profile = build_meeting_retrieval_profile(question, row, legacy_category="autonomous")
        answer, warnings, meta = build_meeting_structured_answer(
            [proposal, draft], question=question, row=row, profile=profile
        )
        self.assertIn("회의 결과 초안 기록상", answer)
        self.assertNotIn("2030", answer)
        self.assertTrue(meta["claim_verification_pass"])
        self.assertGreaterEqual(meta["unsupported_claims_blocked"], 0)

    def test_latest_mepc_environment_answer_uses_current_report_facts(self) -> None:
        carbon = RetrievedChunk(
            "c1", "carbon", "MEPC",
            "MEPC 84-6-2 - Report on annual carbon intensity and efficiency of the fleet (Secretariat).pdf",
            2, "", "text", 0.1,
            (
                "This document reports on both demand-based and supply-based carbon intensity developments "
                "for the period from 2019 to 2024. MEPC 81 adopted amendments to appendix IX of MARPOL Annex VI "
                "(resolution MEPC.385(81) on information to be submitted to the IMO DCS). The Secretariat enabled "
                "reporting on a voluntary basis from 1 January 2025 and on a mandatory basis from 1 January 2026. "
            ) * 2,
        )
        references = RetrievedChunk(
            "r2", "refs", "MEPC",
            "MEPC 84-12 - Strengthening the PSSA framework.pdf", 3, "", "text", 0.2,
            "Di Cintio et al. Avoiding Paper Parks. https://doi.org/10.3390/su15054464. " * 5,
        )
        dcs = RetrievedChunk(
            "d3", "dcs", "MEPC",
            "MEPC 84-6-1 - Report of fuel oil consumption data submitted to the IMO DCS (Secretariat).pdf",
            8, "", "text", 0.3,
            (
                "Verification of the submitted data identified duplicate reporting and unrealistic hours under way. "
                "The number of identified errors that could potentially have a large impact was reduced to 265 ships. "
                "These ships have not been included in the data analysis. Some ships had unrealistic parameters or incorrect ship type. "
            ) * 2,
        )
        marpol = RetrievedChunk(
            "m0", "marpol", "MEPC",
            "MEPC 84-3 - Amendments to MARPOL Annex VI (Secretariat).pdf",
            2, "", "text", 0.2,
            (
                "Following the decision of MEPC/ES.2 to adjourn for one year the discussion on the adoption "
                "of the draft revised MARPOL Annex VI 2025, amendments on data reporting, the North-East Atlantic "
                "Emission Control Area and IMO DCS accessibility were circulated. "
            ) * 2,
        )
        gfi = RetrievedChunk(
            "g4", "gfi", "MEPC",
            "MEPC 84-7-14 - Report of ISWG-GHG 20 (Secretariat).pdf",
            22, "", "text", 0.2,
            (
                "Regarding GFI reporting and verification, draft regulation 37 of MARPOL Annex VI contains clear requirements. "
                "There was broad support for using document ISWG-GHG 20/2/1 as a basis to develop draft amendments to the SEEMP Guidelines. "
            ) * 2,
        )
        lca = RetrievedChunk(
            "l5", "lca", "MEPC",
            "MEPC 84-7-15 - Report of the second meeting of GESAMP-LCA Working Group (Secretariat).pdf",
            4, "", "text", 0.2,
            (
                "The Committee is invited to endorse the Group's uniform understanding of representativeness and conservativeness "
                "for the assessment of WtT default emission factors. "
            ) * 2,
        )
        ibts = RetrievedChunk(
            "i6", "ibts", "MEPC",
            "MEPC 84-10 - Outcome of PPR 13 (Secretariat).pdf",
            4, "", "text", 0.2,
            (
                "Action requested of the Committee following PPR 13: approve, in principle, the draft 2026 Guidelines for systems "
                "for handling oily wastes in machinery spaces incorporating guidance for an integrated bilge water treatment system (IBTS), "
                "with a view to final approval by MEPC 85. "
            ) * 2,
        )
        question = "환경규제 대응과 관련된 최신 MEPC 회의 주요 내용을 정리해줘."
        row = {"category": "env_regulation", "question": question}
        profile = build_meeting_retrieval_profile(question, row, legacy_category="env_regulation")
        answer, _warnings, meta = build_meeting_structured_answer(
            [carbon, references, dcs, marpol, gfi, lca, ibts], question=question, row=row, profile=profile
        )
        self.assertIn("2019~2024년", answer)
        self.assertIn("GFI 보고·검증", answer)
        self.assertIn("IBTS", answer)
        self.assertIn("MEPC 84", answer)
        self.assertIn("최종 회의 결과보고서가 아니므로", answer)
        self.assertNotIn("265척", answer)
        self.assertIn("## 2) 선박 운항/업무 영향", answer)
        self.assertIn("규제 확정 단계", answer)
        self.assertNotIn("기술부서는", answer)
        self.assertNotIn("관리 대상입니다", answer)
        self.assertNotIn("CII·carbon intensity 보고 채택", answer)
        self.assertNotRegex(answer, r"(?m)^-\s*\[\d+\]\s*$")
        self.assertTrue(meta["claim_verification_pass"])


if __name__ == "__main__":
    unittest.main()
