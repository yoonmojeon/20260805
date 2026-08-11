from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
RAG_SCRIPTS = ROOT / "rag" / "scripts"
for path in (ROOT, RAG_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from retrieval_query_analysis import analyze_query, is_meeting_outcome_question
from evidence_planner import build_evidence_plan
from rule_guidance_accurate import _build_definition_extractive_answer, _is_definition_lookup
from question_classifier import classify_question_category
from rag_query_router import is_rule_guidance_lookup
from fast_question_classifier import classify_fast_question_type
from retrieval_search import (
    _document_route_candidates,
    _identifier_matches_filename,
    extract_exact_identifiers,
    query_with_hybrid_ranking,
    rank_scoped_sparse_rows,
)
from table_qa_answer import build_deterministic_table_answer, verify_row_column_intersection
from meeting_structured_answer import _section1_meeting_outcome


def test_exact_identifiers_cover_sparse_maritime_codes():
    found = extract_exact_identifiers(
        "MEPC 84/7/14의 tcorr 및 AC-SD와 DNV-CG-0264를 확인해줘"
    )
    normalized = {item.lower() for item in found}
    assert "mepc 84/7/14" in normalized
    assert "tcorr" in normalized
    assert "ac-sd" in normalized
    assert "dnv-cg-0264" in normalized


def test_rule_symbol_and_korean_cii_are_not_generic_trend_fallbacks():
    assert (
        classify_question_category(
            "구조 규칙에서 쓰는 tcorr 기호는 어떤 두께를 뜻하지?", {}
        )
        == "rule_lookup"
    )


def test_rule_symbol_definition_creates_direct_clause_evidence_slot():
    question = (
        "\uad6c\uc870 \uaddc\uce59\uc5d0\uc11c \uc4f0\ub294 tcorr \uae30\ud638\ub294 "
        "\uc5b4\ub5a4 \ub450\uaed8\ub97c \ub73b\ud558\uc9c0?"
    )
    plan = build_evidence_plan(
        question,
        {"_internal_intent": "rule_lookup", "category": "rule_lookup"},
    )
    slots = {slot.name: slot for slot in plan.slots}
    assert "specific_clause" in slots
    assert "tcorr" in {term.lower() for term in slots["specific_clause"].terms}
    assert _is_definition_lookup(question)


def test_definition_answer_uses_explicit_symbol_definition_line():
    generic = SimpleNamespace(
        text="3.2.1.1 구조부재에 대한 국부 부식추가 tcorr는 다음 식에 따라 계산된다.",
        file_name="12편_2014.pdf",
        page_number=117,
        clause_number="3.2.1.1",
    )
    explicit = SimpleNamespace(
        text="tgrs\n: 총 판두께(mm)\ntcorr\n: 6장/3.2에 정의된 부식추가(mm)",
        file_name="12편_2014.pdf",
        page_number=211,
        clause_number="1.3.3.7",
    )
    answer, cited = _build_definition_extractive_answer(
        "tcorr 기호는 어떤 두께를 뜻하지?", [generic, explicit], "KR"
    )
    assert cited is explicit
    assert "6장/3.2에 정의된 부식추가(mm)" in answer
    assert "p.211" in answer
    cii_question = "IMO 문서 기준으로 선박 탄소집약도 등급 관리 요구사항을 요약해줘."
    assert classify_question_category(cii_question, {}) == "env_regulation"
    assert not is_rule_guidance_lookup(
        cii_question, {"category": "env_regulation"}
    )
    assert (
        classify_fast_question_type(
            "구조 규칙에서 쓰는 tcorr 기호는 어떤 두께를 뜻하지?",
            {"category": "rule_lookup"},
        )
        == "rule_question"
    )


def test_document_code_matches_primary_filename_not_reference_title():
    assert _identifier_matches_filename(
        "MEPC 84/7/14",
        "MEPC 84-7-14 - Report of the working group.pdf",
    )
    assert not _identifier_matches_filename(
        "MEPC 84/7/14",
        "MEPC 84-7-49 - Comments on document MEPC84714.pdf",
    )


def test_document_router_aggregates_chunks_before_clause_search():
    signals = analyze_query("DNV의 자율운항 guidance 핵심은?")
    candidates = _document_route_candidates(
        query="DNV의 자율운항 guidance 핵심은?",
        ids=["noise", "target-a", "target-b"],
        distances={"noise": 0.18, "target-a": 0.31, "target-b": 0.33},
        metadatas={
            "noise": {
                "doc_id": "noise-doc",
                "file_name": "DNV-RP-C205.pdf",
                "source": "DNV",
            },
            "target-a": {
                "doc_id": "target-doc",
                "file_name": "DNV-CG-0264.pdf",
                "source": "DNV",
            },
            "target-b": {
                "doc_id": "target-doc",
                "file_name": "DNV-CG-0264.pdf",
                "source": "DNV",
            },
        },
        documents={
            "noise": "unrelated structural analysis",
            "target-a": "autonomous and remotely operated ships",
            "target-b": "guidance for autonomous ship functions",
        },
        signals=signals,
        clause_hints=[],
        priority_doc_ids={"target-doc"},
    )
    assert candidates[0]["doc_id"] == "target-doc"
    assert candidates[0]["hit_count"] == 2


def test_meeting_agenda_and_discussion_wording_are_outcome_queries():
    assert is_meeting_outcome_question(
        "MEPC 84가 다룰 예정이었던 주요 의제 항목을 정리해줘."
    )
    assert is_meeting_outcome_question(
        "MSC 111 문서에서 IGC Code 개정 논의가 어떻게 정리됐는지 알려줘."
    )


def test_explicit_igc_outcome_does_not_pad_answer_with_other_msc_topics():
    noise = SimpleNamespace(
        chunk_id="noise",
        text=(
            "IGC Code agenda context. The Committee approved the interim guidelines "
            "for the safety of ships using hydrogen as fuel."
        ),
        file_name="MSC 111-WP.1.pdf",
        page_number=55,
    )
    igc = SimpleNamespace(
        chunk_id="igc",
        text=(
            "The Committee instructed the working group to finalize the draft "
            "amendments to the IGC Code for approval at this session."
        ),
        file_name="MSC 111-WP.1.pdf",
        page_number=61,
    )
    profile = SimpleNamespace(answer_variant="", top_level_category="maritime_safety")
    answer, _warnings, picked = _section1_meeting_outcome(
        [(2.0, noise), (1.9, igc)],
        n=3,
        citation_map={"noise": 1, "igc": 2},
        profile=profile,
        row={
            "question": "MSC 111 문서에서 IGC Code 개정 논의가 어떻게 정리됐는지 알려줘.",
            "_evidence_completion": {"slot_hits": {"major_outcomes": ["noise"]}},
        },
    )
    assert "IGC Code 초안 개정" in answer
    assert "수소" not in answer
    assert picked == [igc]


def test_document_scoped_sparse_prefers_exact_code_chunk():
    signals = analyze_query("AC-SD 허용기준은 어느 절을 따르나?")
    ranked = rank_scoped_sparse_rows(
        "AC-SD 허용기준은 어느 절을 따르나?",
        signals,
        ["generic", "exact"],
        [
            {"file_name": "13편_2025.pdf"},
            {"file_name": "14편_2025.pdf"},
        ],
        [
            "허용기준 일반 설명",
            "AC-SD 판과 국부 지지부재의 허용응력은 5장 1절에 따른다.",
        ],
    )
    assert ranked[0][1] == "exact"


class _FakeCollection:
    def __init__(self):
        self.rows = {
            "noise": (
                0.16,
                {"doc_id": "noise-doc", "file_name": "DNV-RP-C205.pdf", "source": "DNV"},
                "structural design recommendation",
            ),
            "target-intro": (
                0.33,
                {"doc_id": "target-doc", "file_name": "DNV-CG-0264.pdf", "source": "DNV"},
                "autonomous and remotely operated ships guidance",
            ),
            "target-clause": (
                0.24,
                {"doc_id": "target-doc", "file_name": "DNV-CG-0264.pdf", "source": "DNV"},
                "The autonomous ship guidance safety principle requires a minimum risk condition and fail-safe response.",
            ),
        }

    @staticmethod
    def _matches(meta, where):
        if not where:
            return True
        if "$and" in where:
            return all(_FakeCollection._matches(meta, part) for part in where["$and"])
        for key, expected in where.items():
            actual = meta.get(key)
            if isinstance(expected, dict) and "$in" in expected:
                if actual not in expected["$in"]:
                    return False
            elif actual != expected:
                return False
        return True

    def query(self, *, query_embeddings, n_results, where=None):
        rows = []
        for cid, (distance, meta, document) in self.rows.items():
            if self._matches(meta, where):
                rows.append((distance, cid, meta, document))
        # The global vector search intentionally misses the decisive clause.
        if not where:
            rows = [row for row in rows if row[1] != "target-clause"]
        rows.sort()
        rows = rows[:n_results]
        return {
            "ids": [[row[1] for row in rows]],
            "distances": [[row[0] for row in rows]],
            "metadatas": [[row[2] for row in rows]],
            "documents": [[row[3] for row in rows]],
        }

    def get(self, *, where=None, where_document=None, limit=100, include=None):
        rows = []
        for cid, (_distance, meta, document) in self.rows.items():
            if not self._matches(meta, where):
                continue
            if where_document and where_document.get("$contains", "").lower() not in document.lower():
                continue
            rows.append((cid, meta, document))
        rows = rows[:limit]
        return {
            "ids": [row[0] for row in rows],
            "metadatas": [row[1] for row in rows],
            "documents": [row[2] for row in rows],
        }


def test_hierarchical_query_searches_inside_selected_document(monkeypatch):
    monkeypatch.setenv("MARITIME_TEXT_HIERARCHICAL", "1")
    out = query_with_hybrid_ranking(
        _FakeCollection(),
        "DNV 자율운항 guidance의 minimum risk condition 안전 원칙은?",
        [0.1, 0.2],
        top_k=2,
        fetch_k=10,
    )
    assert out["ids"][0][0] == "target-clause"
    route = out["document_route"]
    assert route["enabled"] is True
    assert route["selected_doc_ids"][0] == "target-doc"
    assert route["scoped_sparse_used"] is True


def test_cell_verifier_keeps_row_and_column_in_selected_table():
    wrong = SimpleNamespace(
        chunk_id="wrong",
        chunk_type="table_row",
        table_id="wrong-table",
        text="행 기준: 재화중량 100000 초과 150000 이하\n셀: 안전사용하중=999",
        distance=0.01,
    )
    good = SimpleNamespace(
        chunk_id="good",
        chunk_type="table_row",
        table_id="good-table",
        text="행 기준: 재화중량 100000 초과 150000 이하\n셀: 안전사용하중=250",
        distance=0.20,
        file_name="guide.pdf",
        page_number=107,
    )
    question = "재화중량이 10만 톤 초과 15만 톤 이하인 선박의 안전사용하중은 몇 톤인가?"
    debug = {
        "selected_table_id": "good-table",
        "selected_table_candidates": [
            {"table_id": "good-table"},
            {"table_id": "wrong-table"},
        ],
        "parsed_query": {
            "query_type": "cell_lookup",
            "row_entities": ["재화중량이 10만 톤 초과 15만 톤 이하인 선박"],
            "column_entities": ["안전사용하중"],
            "attribute_candidates": ["안전사용하중"],
        },
    }
    row = {"question": question}
    verification = verify_row_column_intersection(row, [wrong, good], debug=debug)
    assert verification["passes"] is True
    assert verification["table_id"] == "good-table"
    answer = build_deterministic_table_answer(row, [wrong, good], debug=debug)
    assert "250" in (answer or "")
    assert "999" not in (answer or "")


def test_multilevel_table_header_maps_subject_to_physical_subcolumn():
    table_id = "kr-rule-table"
    header = SimpleNamespace(
        chunk_id="header",
        chunk_type="table_row",
        table_id=table_id,
        text=(
            "영역=REG01 | 열1=(빈 셀) | 열2=판 및 국부 지지부재(1) | "
            "열3=1차 지지부재(1) | 열4=선체거더 부재"
        ),
        file_name="14편_2025.pdf",
        page_number=19,
    )
    subheaders = SimpleNamespace(
        chunk_id="subheaders",
        chunk_type="table_row",
        table_id=table_id,
        text=(
            "영역=REG02 | 열1=허용기준 AC-S AC-SD AC-A AC-T | "
            "열2=항복 | 열3=좌굴 | 열4=항복 | 열5=좌굴 | "
            "열6=항복 허용응력 : 5장 1절 | 열7=좌굴 허용좌굴 사용계수 : 8장 1절"
        ),
        file_name="14편_2025.pdf",
        page_number=19,
    )
    values = SimpleNamespace(
        chunk_id="values",
        chunk_type="table_row",
        table_id=table_id,
        text=(
            "영역=REG02 | 열2=항복: 허용응력 : 6장 4절 6장 5절 | "
            "열3=좌굴: 강성 및 치수비의 조정 : 8장 2절 | "
            "열4=항복: 허용응력 : 6장 6절 | "
            "열5=좌굴: 강성 및 치수비의 조정 : 8장 1절 8장 2절"
        ),
        file_name="14편_2025.pdf",
        page_number=19,
    )
    question = (
        "14편_2025.pdf 19쪽 표에서 AC-S·AC-SD·AC-A·AC-T "
        "판과 국부 지지부재의 허용응력은 어느 절을 따르나?"
    )
    debug = {
        "selected_table_id": table_id,
        "selected_table_candidates": [{"table_id": table_id}],
        "parsed_query": {
            "query_type": "cell_lookup",
            "row_entities": ["AC-S AC-SD AC-A AC-T"],
            "column_entities": ["판과 국부 지지부재", "허용응력"],
            "attribute_candidates": ["허용응력"],
        },
    }
    row = {"question": question}
    answer = build_deterministic_table_answer(
        row, [header, subheaders, values], debug=debug
    )
    assert "6장 4절 6장 5절" in (answer or "")
    assert "5장 1절" not in (answer or "")
    assert row["_cell_verification"]["method"] == "multilevel_header"
    assert row["_answer_citation_chunks"] == [header, values]


def test_summary_with_multiple_labeled_rows_selects_matching_count_cell():
    summary = SimpleNamespace(
        chunk_id="casting-summary",
        chunk_type="table_summary",
        table_id="casting-table",
        text=(
            "열1=주강품 종류: 일반 주강품 | 열2=시험재의 수: 제품마다 1개\n"
            "열1=주강품 종류: 모양이 복잡한 주강품 또는 1개의 중량이 10톤을 넘는 주강품 | "
            "열2=시험재의 수: 제품마다 2개(1)"
        ),
        file_name="2편_2025.pdf",
        page_number=78,
    )
    question = "복잡한 형상이거나 단품 중량이 10톤을 넘는 주강품은 제품별 시험재가 몇 개인가?"
    debug = {
        "selected_table_id": "casting-table",
        "selected_table_candidates": [{"table_id": "casting-table"}],
        "parsed_query": {
            "query_type": "cell_lookup",
            "row_entities": ["복잡한 형상", "중량 10톤을 넘는 주강품"],
            "column_entities": ["시험재의 수"],
            "attribute_candidates": ["시험재의 수"],
        },
    }
    answer = build_deterministic_table_answer(
        {"question": question}, [summary], debug=debug
    )
    assert "제품마다 2개" in (answer or "")
    assert "제품마다 1개" not in (answer or "")


def test_definition_table_allows_location_question_to_use_definition_cell():
    row_chunk = SimpleNamespace(
        chunk_id="engine-room-bulkhead",
        chunk_type="table_row",
        table_id="definition-table",
        text=(
            "열1=용어: 기관실 격벽 | "
            "열2=정의: 기관실 전방 또는 후방의 횡격벽"
        ),
        file_name="13편_2025.pdf",
        page_number=48,
    )
    question = "표에 정의된 기관실 격벽의 위치는 어디인가?"
    debug = {
        "selected_table_id": "definition-table",
        "selected_table_candidates": [{"table_id": "definition-table"}],
        "parsed_query": {
            "query_type": "cell_lookup",
            "row_entities": ["기관실 격벽"],
            "column_entities": ["위치"],
            "attribute_candidates": ["위치"],
        },
    }
    answer = build_deterministic_table_answer(
        {"question": question}, [row_chunk], debug=debug
    )
    assert "기관실 전방 또는 후방의 횡격벽" in (answer or "")
