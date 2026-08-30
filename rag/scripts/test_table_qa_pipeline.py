from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rag_inprocess import normalize_table_question_row
from rag_answer_lib import generate_answer
from rag_fast_mode import generate_fast_answer
from meeting_trend_ab import is_trend_summary_ab_eligible
from rag_query_router import is_rule_guidance_lookup, resolve_pipeline_route
from retrieval_question_profile import build_retrieval_profile
from table_qa_answer import build_deterministic_table_answer, select_table_evidence
from table_retrieval import evaluate_table_qa_retrieval
from table_query_parser import parse_table_query


class TableQAPipelineTest(unittest.TestCase):
    def test_accurate_exact_cell_runs_before_coarse_confidence_gate(self) -> None:
        chunk = SimpleNamespace(
            chunk_id="row-gate",
            chunk_type="table_row",
            table_id="table-gate",
            doc_id="doc-gate",
            text="table row: RPV 32 | applicable thickness=6 to 50",
            file_name="rules.pdf",
            page_number=28,
            caption="plate type",
            distance=0.1,
        )
        row = {
            "question": "What is the applicable thickness for RPV 32?",
            "_table_qa": True,
            "_table_retrieval_debug": {"passes_confidence_gate": False},
        }
        with patch(
            "table_qa_answer.build_deterministic_table_answer",
            return_value="Conclusion: 6 to 50 [1]",
        ) as deterministic:
            answer, provider, _model = generate_answer(
                row,
                [chunk],
                provider="ollama",
                model="unused",
                ollama_base="http://localhost:11434",
                answer_mode="table_qa",
                pool=[chunk],
            )
        deterministic.assert_called_once()
        self.assertEqual(provider, "table_deterministic")
        self.assertIn("6 to 50", answer)

    def test_table_question_never_routes_to_trend_ab(self) -> None:
        row = normalize_table_question_row(
            {"question": "용접 입력을 측정한다면 시험 문서 중 무엇을 함께 확인하나?"}
        )
        self.assertFalse(is_trend_summary_ab_eligible(row["question"], row))

    def test_natural_parser_extracts_subject_and_attribute(self) -> None:
        parsed = parse_table_query(
            "호퍼탱크 경사판과 연결된 이중선측 수평거더 웨브는 어떤 방법으로 평가하는가?"
        )
        self.assertIn("호퍼탱크 경사판과 연결된 이중선측 수평거더 웨브", parsed.row_entities)
        self.assertIn("평가 방법", parsed.column_entities)

    def test_natural_parser_splits_possessive_lookup(self) -> None:
        parsed = parse_table_query("주갑판 아래의 안덮개 설치비율은 얼마인가?")
        self.assertIn("주갑판 아래", parsed.row_entities)
        self.assertIn("안덮개 설치비율", parsed.column_entities)

    def test_single_letter_alias_does_not_match_inside_korean_word(self) -> None:
        parsed = parse_table_query("기관실 격벽은 어느 위치의 횡격벽을 의미하는가?")
        self.assertNotIn("C", parsed.column_entities)
        self.assertNotIn("P", parsed.column_entities)
        self.assertNotIn("S", parsed.column_entities)

    def test_age_alias_does_not_match_inside_agenda(self) -> None:
        parsed = parse_table_query("MEPC 84 임시 의제 시간표 표에서 주요 agenda 배정은?")
        self.assertNotIn("선령", parsed.column_entities)
        self.assertFalse(any("선령" in c for c in parsed.column_entities))

    def test_cms_does_not_expand_to_chemistry_or_centimeters(self) -> None:
        parsed = parse_table_query("Sea Water Service System의 CMS 통일명칭은 무엇인가?")
        self.assertIn("Sea Water Service System", parsed.row_entities)
        self.assertIn("CMS 통일명칭", parsed.column_entities)
        self.assertNotIn("C", parsed.keyword_terms)
        self.assertNotIn("S", parsed.keyword_terms)
        self.assertNotIn("CM", parsed.unit_candidates)
        self.assertNotIn("화학성분", parsed.table_topic_candidates)

    def test_casting_specimen_question_keeps_row_and_column(self) -> None:
        parsed = parse_table_query(
            "형상이 복잡하거나 한 개의 중량이 10톤을 넘는 주강품은 "
            "제품마다 시험재가 몇 개 필요한가?"
        )
        self.assertTrue(any("주강품" in row for row in parsed.row_entities))
        self.assertIn("시험재의 수", parsed.column_entities)
        self.assertIn("시험재료", parsed.table_topic_candidates)
        self.assertNotIn("용접", parsed.table_topic_candidates)

    def test_open_table_questions_add_bilingual_row_and_column_slots(self) -> None:
        cases = [
            (
                "기관실 격벽은 어느 위치의 횡격벽을 의미하는가?",
                "engine room bulkhead",
                "최전방 수밀 횡격벽",
            ),
            (
                "적층제조 최종 재료의 제조법 승인에는 지침의 어느 장을 적용하는가?",
                "AM 최종 재료",
                "이 지침에서 적용되는 장 또는 하위 번호",
            ),
            (
                "ESP·EXP 부호가 있는 Oil/Bulk/Ore Carrier의 설계에는 어느 장을 적용하는가?",
                "Oil/Bulk/Ore Carrier 'ESP'(EXP)",
                "Design",
            ),
            (
                "이중저 늑판은 어떤 구조평가 방법을 적용하는가?",
                "Double bottom floors",
                "구조평가 방법",
            ),
            (
                "선수격벽 뒤에 있는 체인로커의 시험압력수두는 어떻게 정하는가?",
                "체인로커(선수격벽 후방에 있는 경우)",
                "시험압력수두(m)",
            ),
        ]
        for question, expected_row, expected_column in cases:
            with self.subTest(question=question):
                parsed = parse_table_query(question)
                self.assertIn(expected_row, parsed.row_entities)
                self.assertIn(expected_column, parsed.column_entities)

    def test_kr_part1_rule_query_is_prioritized(self) -> None:
        from retrieval_query_analysis import analyze_query
        from retrieval_search import _direct_priority_rule_doc_ids, infer_query_narrow_doc_id

        q = "902절 탈급(선급등록 취소)의 적용 대상과 절차는?"
        self.assertIn("kr_1_2025", _direct_priority_rule_doc_ids(q, analyze_query(q)))
        self.assertEqual(infer_query_narrow_doc_id(q, analyze_query(q)), "kr_1_2025")

    def test_literal_table_row_signal_survives_rerank_payload(self) -> None:
        from table_schema_retrieval import TableScoreBreakdown

        item = TableScoreBreakdown(table_id="t", literal_row_match=True)
        self.assertTrue(item.to_dict()["literal_row_match"])

    def test_korean_sparse_terms_strip_particles(self) -> None:
        from retrieval_query_analysis import analyze_query
        from retrieval_search import _sparse_query_terms

        q = "603절 증서의 재교부는 누가 신청하고 조치해야 하는가?"
        terms = dict(_sparse_query_terms(q, analyze_query(q)))
        self.assertIn("증서", terms)
        self.assertIn("재교부", terms)
        self.assertIn("신청", terms)
        self.assertIn("조치", terms)

    def test_scoped_sparse_prefers_exact_phrase_and_clause_topic(self) -> None:
        from retrieval_query_analysis import analyze_query
        from retrieval_search import rank_scoped_sparse_rows

        ids = ["wrong", "phrase", "cms", "withdrawal"]
        metas = [
            {"page_number": 73},
            {"page_number": 9},
            {"page_number": 84, "clause_number": "902"},
            {"page_number": 22, "clause_number": "902"},
        ]
        docs = [
            "정기검사는 원칙적으로 입거하여 시행하는 경우에 적용한다.",
            "시험 및 검사는 특별한 경우 외에는 검사원의 입회하에 시행한다.",
            "clause: 902 조문 902절 902. 검사사항 CMS 검사절차와 적용대상 및 취소",
            "clause: 902 조문 902절 902. 탈급 선급위원회의 승인 후 탈급한다.",
        ]

        phrase_query = "시험 및 검사는 원칙적으로 어떻게 시행해야 하는가?"
        phrase_ranked = rank_scoped_sparse_rows(
            phrase_query,
            analyze_query(phrase_query),
            ids,
            metas,
            docs,
            top_k=4,
        )
        self.assertEqual(phrase_ranked[0][1], "phrase")

        clause_query = "902절 탈급(선급등록 취소)의 적용 대상과 절차는?"
        clause_ranked = rank_scoped_sparse_rows(
            clause_query,
            analyze_query(clause_query),
            ids,
            metas,
            docs,
            top_k=4,
        )
        self.assertEqual(clause_ranked[0][1], "withdrawal")

    def test_table_qa_mepc_question_does_not_default_to_kr(self) -> None:
        from rag_query_router import enrich_row_for_routing

        row = enrich_row_for_routing(
            {"question": "MEPC 84 임시 의제 시간표 표에서 주요 agenda 배정은?", "_table_qa": True},
            latency_mode="fast",
        )
        self.assertEqual(row.get("class_society_hint"), "MEPC")
        self.assertEqual(row.get("retrieval_sources"), ["MEPC"])

    def test_table_number_boost_prefers_exact_caption(self) -> None:
        from table_query_parser import parse_table_query
        from table_schema_retrieval import score_table_candidate

        parsed = parse_table_query("표 2.1.65 종류 및 화학성분 표의 주요 열 구성은?")
        gold = score_table_candidate(
            parsed,
            vector_distance=0.25,
            meta={"caption": "표 2.1.65 종류 및 화학성분", "chunk_type": "table_summary"},
            document="표: 표 2.1.65 종류 및 화학성분 문서: 2편_2025.pdf 열1=종류 열2=C",
        )
        near = score_table_candidate(
            parsed,
            vector_distance=0.20,
            meta={"caption": "표 2.1.55 화학성분", "chunk_type": "table_summary"},
            document="표: 표 2.1.55 화학성분 문서: 2편_2025.pdf 열1=종류 및 열2=재료기호",
        )
        self.assertGreater(gold.combined_score, near.combined_score)

    def test_inspection_age_penalizes_cargo_density_tables(self) -> None:
        from table_query_parser import parse_table_query
        from table_schema_retrieval import score_table_candidate

        parsed = parse_table_query(
            "선령 5~10년 구간 선박의 화물창은 정기검사에서 어떤 reporting 요건이 있나?"
        )
        gold = score_table_candidate(
            parsed,
            vector_distance=0.30,
            meta={"caption": "", "chunk_type": "table_row"},
            document="영역=REG01 | 열1=화물창 | 열2=5년< 선령≤10년: 모든 화물창에 대한 현상검사",
        )
        noise = score_table_candidate(
            parsed,
            vector_distance=0.18,
            meta={"caption": "", "chunk_type": "table_summary"},
            document="영역=REG01 | 선박의 종류 화물 질량/ 화물 밀도 균일 적재상태 (만재 화물창)",
        )
        structural = score_table_candidate(
            parsed,
            vector_distance=0.15,
            meta={"caption": "", "chunk_type": "table_summary"},
            document="영역=REG01 | 열1=구조 부재 구분: C4 화물 창구 모서리부의 강판 (산적화물선",
        )
        self.assertGreater(gold.combined_score, noise.combined_score)
        self.assertGreater(gold.combined_score, structural.combined_score)

    def test_table_route_precedes_general_rule_and_meeting_routes(self) -> None:
        row = normalize_table_question_row(
            {"question": "정기검사 표에서 화물탱크 reporting 요건은?"}
        )
        route = resolve_pipeline_route(row["question"], row)
        self.assertEqual(route["question_category"], "table_qa")
        self.assertEqual(route["selected_answer_mode"], "table_qa")
        self.assertFalse(route["rule_guidance_lookup"])
        self.assertFalse(is_rule_guidance_lookup(row["question"], row))

    def test_table_profile_uses_two_stage_without_diversity_eviction(self) -> None:
        row = normalize_table_question_row(
            {"question": "선령 15년 초과 평형수탱크 검사 범위는?"}
        )
        profile = build_retrieval_profile(row["question"], row)
        self.assertEqual(profile.answer_mode, "table_qa")
        self.assertEqual(profile.profile_id, "table_schema_two_stage")
        self.assertFalse(profile.use_diversity_rerank)

    def test_evidence_keeps_four_chunk_types_for_selected_table(self) -> None:
        def chunk(cid: str, ctype: str, tid: str, text: str):
            return SimpleNamespace(
                chunk_id=cid,
                chunk_type=ctype,
                table_id=tid,
                text=text,
                distance=0.2,
            )

        selected = [
            chunk("r", "table_row", "gold", "구역=화물탱크 제1차 정기검사=○"),
            chunk("m", "table_markdown", "gold", "| 화물탱크 | ○ |"),
            chunk("s", "table_schema", "gold", "columns: 구역, 제1차 정기검사"),
            chunk("u", "table_summary", "gold", "정기검사 표"),
            chunk("x", "table_row", "other", "무관한 행"),
        ]
        debug = {
            "selected_table_id": "gold",
            "selected_table_candidates": [{"table_id": "gold"}],
            "parsed_query": {
                "row_entities": ["화물탱크"],
                "column_entities": ["제1차 정기검사"],
            },
        }
        evidence = select_table_evidence(
            {"question": "화물탱크 제1차 정기검사"}, selected, selected, debug=debug
        )
        gold_types = {c.chunk_type for c in evidence if c.table_id == "gold"}
        self.assertEqual(
            gold_types,
            {"table_row", "table_markdown", "table_schema", "table_summary"},
        )

    def test_exact_cell_answer_bypasses_llm(self) -> None:
        chunk = SimpleNamespace(
            chunk_id="row-1",
            chunk_type="table_row",
            table_id="table-1",
            text=(
                "[표 행 및 셀 사실]\n행 기준: RPV 32\n"
                "셀: 재료기호=RPV 32 | 적용두께(mm)=6∼150"
            ),
            file_name="2편_2025.pdf",
            page_number=28,
            caption="표 2.1.13 강판의 종류",
        )
        debug = {
            "parsed_query": {
                "query_type": "cell_lookup",
                "row_entities": ["RPV 32"],
                "column_entities": ["적용두께(mm)"],
            }
        }
        row = {}
        answer = build_deterministic_table_answer(row, [chunk], debug=debug)
        self.assertIn("결론:", answer or "")
        self.assertIn("6∼150", answer or "")
        self.assertIn("[1]", answer or "")
        self.assertEqual(row["_answer_citation_chunks"], [chunk])
        self.assertTrue(row.get("_verified_structured_answer"))

    def test_natural_cell_answer_ignores_identity_cell(self) -> None:
        chunk = SimpleNamespace(
            chunk_id="row-2",
            chunk_type="table_row",
            table_id="table-2",
            text=(
                "[표 행 및 셀 사실]\n행 기준: 100,000 < DWT ≤ 150,000 / 250\n"
                "셀: 재화중량(DWT) (ton)=100,000 < DWT ≤ 150,000 | "
                "안전사용하중(SWL) (ton)=250"
            ),
            file_name="guide.pdf",
            page_number=107,
            caption="안전사용하중",
            distance=0.1,
        )
        debug = {
            "selected_table_id": "table-2",
            "selected_table_candidates": [{"table_id": "table-2"}],
            "parsed_query": {
                "query_type": "cell_lookup",
                "row_entities": ["재화중량이 10만 톤 초과 15만 톤 이하인 선박"],
                "column_entities": ["안전사용하중"],
                "subject_candidates": ["재화중량이 10만 톤 초과 15만 톤 이하인 선박"],
                "attribute_candidates": ["안전사용하중"],
            },
        }
        row = {"question": "재화중량이 10만 톤 초과 15만 톤 이하인 선박의 안전사용하중은 몇 톤인가?"}
        answer = build_deterministic_table_answer(
            row,
            [chunk],
            debug=debug,
        )
        self.assertIn("결론:", answer or "")
        self.assertIn("250", answer or "")
        self.assertIn("[1]", answer or "")
        self.assertEqual(row["_answer_citation_chunks"], [chunk])
        self.assertTrue(row.get("_verified_structured_answer"))

    def test_opaque_column_key_remaps_for_allowance(self) -> None:
        chunk = SimpleNamespace(
            chunk_id="flood-1",
            chunk_type="table_row",
            table_id="t006",
            text=(
                "표: t006 문서: 13편_2025.pdf, 25쪽 영역=REG07 | "
                "열1=침수상태 | 열2=사고에 | 열3=의한 침수로 내부 수밀구획구조에 | "
                "열4=(빈 셀) | 열5=AC-SD"
            ),
            file_name="13편_2025.pdf",
            page_number=25,
            distance=0.4,
        )
        distractor = SimpleNamespace(
            chunk_id="ballast-1",
            chunk_type="table_row",
            table_id="t006",
            text=(
                "영역=REG05 | 열1=적하, 양하 및 평형수 적재 | "
                "열3=평형수 조작 상태에서 대표적 최대 하중 | 열4=S | 열5=AC-S"
            ),
            file_name="13편_2025.pdf",
            page_number=25,
            distance=0.45,
        )
        question = (
            "13편_2025.pdf 25페이지 표에서 침수상태 / 사고에 의한 침수로 "
            "내부 수밀구획구조에 미치는 대표적인 최대 하중 행의 허용기준은?"
        )
        from table_query_parser import parse_table_query

        parsed = parse_table_query(question).to_dict()
        self.assertIn("허용기준", " ".join(parsed.get("column_entities") or []))
        row: dict = {"question": question}
        answer = build_deterministic_table_answer(
            row,
            [chunk, distractor],
            debug={
                "parsed_query": parsed,
                "selected_table_id": "t006",
                "selected_table_candidates": [{"table_id": "t006"}],
            },
        )
        self.assertIn("AC-SD", answer or "")
        self.assertIn("허용기준", answer or "")
        self.assertTrue(row.get("_verified_structured_answer"))

    def test_fast_table_answer_prefers_deterministic_cell(self) -> None:
        chunk = SimpleNamespace(
            chunk_id="row-fast",
            chunk_type="table_row",
            table_id="table-fast",
            doc_id="doc-fast",
            text=(
                "[표 행 및 셀 사실]\n행 기준: RPV 32\n"
                "셀: 재료기호=RPV 32, RPV 36 | 적용두께(mm)=6∼150"
            ),
            file_name="2편_2025.pdf",
            page_number=28,
            caption="표 2.1.13 강판의 종류",
            distance=0.1,
        )
        row = normalize_table_question_row(
            {"question": "2편_2025.pdf 28페이지 표에서 RPV 32~50 행의 적용두께(mm)는?"}
        )

        def fake_llm(*args, **kwargs):
            self.fail("LLM should be skipped for deterministic table cells")

        with patch("rag_fast_mode.call_ollama_chat_timed", side_effect=fake_llm), patch(
            "rag_fast_mode.ensure_fast_warm_checked", return_value={}
        ), patch("rag_fast_mode.mark_fast_llm_run"):
            answer, meta = generate_fast_answer(
                row,
                [chunk],
                pool=[chunk],
                model="unused",
                ollama_base="http://localhost:11434",
                auto_llm_warm=False,
                fast_meta={"fast_question_type": "table_qa"},
            )
        self.assertIn("6∼150", answer)
        self.assertEqual(meta.get("answer_source"), "table_deterministic")
        self.assertFalse((meta.get("answer_generation") or {}).get("llm_used"))
        self.assertTrue(row.get("_verified_structured_answer"))

    def test_weak_row_match_refuses_wrong_cell(self) -> None:
        """Do not assert a high-scoring unrelated cell (Pin) for 넘침/순차 S+D."""
        wrong = SimpleNamespace(
            chunk_id="pin-1",
            chunk_type="table_row",
            table_id="t-load",
            text=(
                "[표 행 및 셀 사실]\n행 기준: 하중점 Pin\n"
                "셀: 구분=하중점 | 설계하중 시나리오=Pin"
            ),
            file_name="13편_2025.pdf",
            page_number=40,
            distance=0.1,
        )
        distractor = SimpleNamespace(
            chunk_id="other-1",
            chunk_type="table_row",
            table_id="t-load",
            text=(
                "[표 행 및 셀 사실]\n행 기준: 일반 하중\n"
                "셀: 구분=일반 | 설계하중 시나리오=S"
            ),
            file_name="13편_2025.pdf",
            page_number=40,
            distance=0.2,
        )
        question = "넘침식 또는 순차식 평형수 교환에는 어떤 설계하중 시나리오를 적용하는가?"
        from table_query_parser import parse_table_query

        parsed = parse_table_query(question).to_dict()
        row = {"question": question}
        answer = build_deterministic_table_answer(
            row,
            [wrong, distractor],
            debug={
                "parsed_query": parsed,
                "selected_table_id": "t-load",
                "selected_table_candidates": [{"table_id": "t-load"}],
            },
        )
        self.assertIsNone(answer)
        self.assertNotIn("Pin", answer or "")

    def test_caption_ask_uses_schema_caption(self) -> None:
        schema = SimpleNamespace(
            chunk_id="cap-1",
            chunk_type="table_schema",
            table_id="t-page10",
            text="표: t-page10 문서: 2편_2025.pdf, 10쪽",
            file_name="2편_2025.pdf",
            page_number=10,
            caption="표 2.1.1 시험편의 모양",
            distance=0.2,
        )
        row = {"question": "2편_2025.pdf 10쪽 구조화 표 제목?"}
        answer = build_deterministic_table_answer(
            row,
            [schema],
            debug={
                "selected_table_id": "t-page10",
                "selected_table_candidates": [{"table_id": "t-page10"}],
                "parsed_query": {"query_type": "table_lookup"},
            },
        )
        self.assertIn("시험편", answer or "")
        self.assertTrue(row.get("_verified_structured_answer"))

    def test_comparison_requires_both_gold_cells(self) -> None:
        def chunk(row: str, answer: str):
            return SimpleNamespace(
                chunk_type="table_row",
                table_id="table-1",
                doc_id="doc-1",
                page_number=10,
                text=f"셀: 재료={row} | 두께={answer}",
            )

        row = {
            "gold_doc_id": "doc-1",
            "gold_page": 10,
            "gold_table_id": "table-1",
            "gold_cells": [
                {"row_key": "A", "column": "두께", "answer": "10"},
                {"row_key": "B", "column": "두께", "answer": "20"},
            ],
        }
        incomplete = evaluate_table_qa_retrieval([chunk("A", "10")], row)
        complete = evaluate_table_qa_retrieval([chunk("A", "10"), chunk("B", "20")], row)
        self.assertFalse(incomplete["cell_exact_match"])
        self.assertEqual(incomplete["matched_cell_count"], 1)
        self.assertTrue(complete["cell_exact_match"])
        self.assertEqual(complete["matched_cell_count"], 2)


if __name__ == "__main__":
    unittest.main()
