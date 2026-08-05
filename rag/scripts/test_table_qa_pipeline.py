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
from rag_fast_mode import generate_fast_answer
from meeting_trend_ab import is_trend_summary_ab_eligible
from rag_query_router import is_rule_guidance_lookup, resolve_pipeline_route
from retrieval_question_profile import build_retrieval_profile
from table_qa_answer import build_deterministic_table_answer, select_table_evidence
from table_retrieval import evaluate_table_qa_retrieval
from table_query_parser import parse_table_query


class TableQAPipelineTest(unittest.TestCase):
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
        self.assertIn("결론: 6∼150", answer or "")
        self.assertIn("[1]", answer or "")
        self.assertEqual(row["_answer_citation_chunks"], [chunk])

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
        self.assertIn("결론: 250", answer or "")
        self.assertIn("[1]", answer or "")
        self.assertEqual(row["_answer_citation_chunks"], [chunk])

    def test_fast_table_answer_bypasses_ollama(self) -> None:
        chunk = SimpleNamespace(
            chunk_id="row-fast",
            chunk_type="table_row",
            table_id="table-fast",
            text=(
                "[표 행 및 셀 사실]\n행 기준: 잔류응력 측정\n"
                "셀: 종류=잔류응력 측정 | 시험 결과에 기록되어야 하는 항목=시험재 ID 및 결과"
            ),
            file_name="Circular.pdf",
            page_number=469,
            caption="승인 시험 및 시험 결과 관련 문서",
            distance=0.1,
        )
        row = normalize_table_question_row(
            {"question": "잔류응력을 측정한 경우 시험 결과 문서에 어떤 정보를 기록해야 하는가?"}
        )
        with patch(
            "rag_fast_mode.call_ollama_chat_timed",
            side_effect=AssertionError("Ollama must not be called for an exact table cell"),
        ):
            answer, meta = generate_fast_answer(
                row,
                [chunk],
                pool=[chunk],
                model="unused",
                ollama_base="http://localhost:11434",
                auto_llm_warm=False,
                fast_meta={"fast_question_type": "table_qa"},
            )
        self.assertIn("시험재 ID 및 결과", answer)
        self.assertTrue(meta.get("llm_skipped"))
        self.assertEqual(meta.get("answer_source"), "structured_table_cell")

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
