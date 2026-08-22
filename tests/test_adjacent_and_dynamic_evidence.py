from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RAG_SCRIPTS = ROOT / "rag" / "scripts"
for path in (ROOT, RAG_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adjacent_chunk_expansion import (
    clear_doc_order_cache,
    expand_chunks_with_parent_context,
    expand_chunks_with_neighbors,
    expand_evidence_with_neighbors,
    expansion_reason,
)
from dynamic_evidence import detect_facets, knee_cut, plan_evidence_budget
from fast_context import FastEvidence
from fast_retrieval import evidence_budget, select_general_slots
from rag_answer_lib import RetrievedChunk


def make_chunk(chunk_id: str, text: str, *, doc_id="doc_a", page=10, distance=0.1, chunk_type=""):
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source="text",
        file_name=f"{doc_id}.pdf",
        page_number=page,
        clause_number="",
        element_type="text",
        distance=distance,
        text=text,
        chunk_type=chunk_type,
    )


@pytest.fixture
def scratch_dir():
    """Repo-local scratch dir; the shared temp root is not writable here."""
    base = ROOT / ".pytest_scratch"
    base.mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(dir=str(base)))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def write_doc(root: Path, doc_id: str, rows: list[dict]) -> Path:
    doc_dir = root / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    with (doc_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    clear_doc_order_cache()
    return root


# --- dynamic evidence budget -------------------------------------------------


def test_definition_only_question_gets_the_smallest_budget():
    budget = plan_evidence_budget(
        [0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
        "과도한 부식의 정의는 무엇인가?",
        base_count=3,
        base_max_docs=2,
    )
    assert budget.facets == ("definition",)
    assert budget.count == 2


def test_compound_requirement_question_gets_more_evidence_than_the_fixed_three():
    question = "검사강화제도 적용 대상과 예외, 검사 주기 및 조건을 알려줘"
    facets = detect_facets(question)
    assert {"scope", "exception", "interval", "condition"} <= set(facets)

    budget = plan_evidence_budget(
        [0.12, 0.121, 0.122, 0.123, 0.124, 0.125, 0.126],
        question,
        base_count=3,
        base_max_docs=2,
    )
    assert budget.count >= 5
    assert budget.basis == "facet_demand"
    # A four-facet ask may legitimately span one more document.
    assert budget.max_docs == 3


def test_score_knee_cuts_the_budget_when_later_chunks_fall_off():
    question = "ESP 적용 대상과 예외 조건을 알려줘"
    budget = plan_evidence_budget(
        [0.10, 0.101, 0.102, 0.402, 0.404, 0.405],
        question,
        base_count=3,
        base_max_docs=2,
    )
    assert budget.count == 3
    assert budget.basis == "score_knee"


def test_flat_pool_has_no_knee():
    assert knee_cut([0.15, 0.151, 0.152, 0.153, 0.154], floor=2, ceiling=6) is None


def test_budget_is_bypassed_when_the_feature_is_switched_off(monkeypatch):
    monkeypatch.setenv("MARITIME_DYNAMIC_EVIDENCE", "0")
    budget = evidence_budget(
        [make_chunk("c1", "text")],
        "적용 대상과 예외, 주기, 조건",
        base_count=3,
        base_max_docs=2,
    )
    assert (budget.count, budget.max_docs, budget.basis) == (3, 2, "fixed")


def test_general_slots_respect_the_planned_budget():
    pool = [
        make_chunk(f"c{i}", f"본문 {i}", doc_id=f"doc_{i % 3}", distance=0.1 + i / 100)
        for i in range(8)
    ]
    picked = select_general_slots(pool, max_chunks=5, max_docs=3)
    assert len(picked) == 5
    assert len({ev.chunk.doc_id for ev in picked}) == 3


# --- adjacent chunk expansion ------------------------------------------------


def test_exception_paragraph_is_pulled_in_after_the_requirement(scratch_dir):
    rows = [
        {"chunk_id": "doc_a_p0010_m001", "page_number": 10, "text": "선체검사는 5년마다 시행한다."},
        {
            "chunk_id": "doc_a_p0010_m002",
            "page_number": 10,
            "text": "다만, 선령 15년 미만의 선박은 이 요건을 적용하지 아니한다.",
        },
    ]
    chunks_dir = write_doc(scratch_dir, "doc_a", rows)
    parent = make_chunk("doc_a_p0010_m001", rows[0]["text"])
    evidence = [FastEvidence(parent, "general")]

    out, trace = expand_evidence_with_neighbors(
        evidence, pool=[parent], chunks_dir=chunks_dir
    )

    assert [ev.chunk.chunk_id for ev in out] == ["doc_a_p0010_m001", "doc_a_p0010_m002"]
    assert trace[0]["reason"] == "exception_follows"
    assert out[1].slot == "adjacent_next:exception_follows"


def test_split_paragraph_pieces_are_stitched_back_together(scratch_dir):
    rows = [
        {"chunk_id": "doc_a_p0117_m004_s01", "page_number": 117, "text": "국부 부식추가 tcorr 는"},
        {
            "chunk_id": "doc_a_p0117_m004_s02",
            "page_number": 117,
            "text": "부재의 양면에 대한 부식추가 두께의 합으로 정의한다.",
        },
    ]
    chunks_dir = write_doc(scratch_dir, "doc_a", rows)
    parent = make_chunk("doc_a_p0117_m004_s01", rows[0]["text"], page=117)

    out, trace = expand_evidence_with_neighbors(
        [FastEvidence(parent, "scope_definition")], pool=[], chunks_dir=chunks_dir
    )

    assert len(out) == 2
    assert trace[0]["reason"] == "split_sibling"


def test_complete_paragraph_is_left_alone(scratch_dir):
    rows = [
        {
            "chunk_id": "doc_a_p0010_m001",
            "page_number": 10,
            "text": "선급은 매년 정기검사를 시행하며 그 결과를 선박소유자에게 통보한다. "
            "검사 결과는 선급 기록에 보존한다.",
        },
        {
            "chunk_id": "doc_a_p0010_m002",
            "page_number": 10,
            "text": "제 3 절 재화중량에 따른 구조 기준은 별표에 따른다. 별표는 매년 개정된다.",
        },
    ]
    chunks_dir = write_doc(scratch_dir, "doc_a", rows)
    parent = make_chunk("doc_a_p0010_m001", rows[0]["text"])

    out, trace = expand_evidence_with_neighbors(
        [FastEvidence(parent, "general")], pool=[], chunks_dir=chunks_dir
    )

    assert len(out) == 1
    assert trace == []


def test_advanced_parent_context_adds_same_clause_qualification(scratch_dir):
    rows = [
        {
            "chunk_id": "doc_a_p0010_m001",
            "page_number": 10,
            "clause_number": "2.4",
            "text": "2.4 적용범위는 전자 장치가 설치된 제품으로 한정한다.",
        },
        {
            "chunk_id": "doc_a_p0010_m002",
            "page_number": 10,
            "clause_number": "2.4",
            "text": "방사성 방출 시험은 요구될 수 있다.",
        },
    ]
    chunks_dir = write_doc(scratch_dir, "doc_a", rows)
    selected = make_chunk(
        "doc_a_p0010_m002", rows[1]["text"], page=10
    )
    selected.clause_number = "2.4"
    expanded, trace = expand_chunks_with_parent_context(
        [selected], pool=[], chunks_dir=chunks_dir, limit=2
    )
    assert [chunk.chunk_id for chunk in expanded] == [
        "doc_a_p0010_m002",
        "doc_a_p0010_m001",
    ]
    assert trace[0]["reason"] == "same_clause"


def test_table_rows_are_never_expanded(scratch_dir):
    rows = [
        {"chunk_id": "doc_a_p0010_m001", "page_number": 10, "text": "재화중량(DWT) (ton): 100,000"},
        {"chunk_id": "doc_a_p0010_m002", "page_number": 10, "text": "다만, 예외 행이 이어진다."},
    ]
    chunks_dir = write_doc(scratch_dir, "doc_a", rows)
    parent = make_chunk(
        "doc_a_p0010_m001", rows[0]["text"], chunk_type="table_row"
    )

    out, _ = expand_evidence_with_neighbors(
        [FastEvidence(parent, "table_row_kv")], pool=[], chunks_dir=chunks_dir
    )
    assert len(out) == 1


def test_expansion_stops_at_the_neighbor_budget(scratch_dir):
    rows = []
    for page in (10, 20, 30):
        rows.append(
            {"chunk_id": f"doc_a_p00{page}_m001", "page_number": page, "text": "요건은 다음과"}
        )
        rows.append(
            {
                "chunk_id": f"doc_a_p00{page}_m002",
                "page_number": page,
                "text": "같은 조건을 충족하여야 한다. 세부 기준은 별표에 따른다.",
            }
        )
    chunks_dir = write_doc(scratch_dir, "doc_a", rows)
    evidence = [
        FastEvidence(make_chunk(f"doc_a_p00{page}_m001", "요건은 다음과", page=page), "general")
        for page in (10, 20, 30)
    ]

    out, trace = expand_evidence_with_neighbors(evidence, pool=[], chunks_dir=chunks_dir)

    assert len(trace) == 2
    assert len(out) == 5


def test_neighbor_from_another_section_is_not_borrowed(scratch_dir):
    rows = [
        {"chunk_id": "doc_a_p0010_m001", "page_number": 10, "text": "요건은 다음과"},
        {"chunk_id": "doc_a_p0044_m001", "page_number": 44, "text": "다만, 다른 장의 예외 규정이다."},
    ]
    chunks_dir = write_doc(scratch_dir, "doc_a", rows)
    parent = make_chunk("doc_a_p0010_m001", rows[0]["text"])

    out, trace = expand_evidence_with_neighbors(
        [FastEvidence(parent, "general")], pool=[], chunks_dir=chunks_dir
    )
    assert len(out) == 1
    assert trace == []


def test_expansion_is_bypassed_when_the_feature_is_switched_off(scratch_dir, monkeypatch):
    monkeypatch.setenv("MARITIME_ADJACENT_EXPANSION", "0")
    rows = [
        {"chunk_id": "doc_a_p0010_m001", "page_number": 10, "text": "선체검사는 5년마다 시행한다."},
        {"chunk_id": "doc_a_p0010_m002", "page_number": 10, "text": "다만, 예외가 있다."},
    ]
    chunks_dir = write_doc(scratch_dir, "doc_a", rows)
    parent = make_chunk("doc_a_p0010_m001", rows[0]["text"])

    out, trace = expand_evidence_with_neighbors(
        [FastEvidence(parent, "general")], pool=[], chunks_dir=chunks_dir
    )
    assert (len(out), trace) == (1, [])


def test_backward_expansion_recovers_the_requirement_behind_an_exception(scratch_dir):
    rows = [
        {
            "chunk_id": "doc_a_p0010_m001",
            "page_number": 10,
            "text": "선체검사는 5년마다 시행하며 그 결과를 선급에 보고하여야 한다.",
        },
        {
            "chunk_id": "doc_a_p0010_m002",
            "page_number": 10,
            "text": "다만, 선령 15년 미만의 선박은 이 요건을 적용하지 아니한다.",
        },
    ]
    chunks_dir = write_doc(scratch_dir, "doc_a", rows)
    parent = make_chunk("doc_a_p0010_m002", rows[1]["text"])

    out, trace = expand_evidence_with_neighbors(
        [FastEvidence(parent, "general")], pool=[], chunks_dir=chunks_dir
    )

    assert [ev.chunk.chunk_id for ev in out] == ["doc_a_p0010_m002", "doc_a_p0010_m001"]
    assert trace[0]["reason"] == "exception_parent"
    assert trace[0]["direction"] == "prev"


def test_cross_page_mode_skips_a_neighbor_from_the_same_page(scratch_dir):
    """Rule lookup already merges whole pages, so only a page break is new."""
    rows = [
        {"chunk_id": "doc_a_p0010_m001", "page_number": 10, "text": "검사 요건은 다음과"},
        {
            "chunk_id": "doc_a_p0010_m002",
            "page_number": 10,
            "text": "같은 조건을 충족하여야 하며 세부 기준은 별표에 따른다.",
        },
        {"chunk_id": "doc_a_p0011_m001", "page_number": 11, "text": "이어지는 조항은 다음과"},
        {
            "chunk_id": "doc_a_p0012_m001",
            "page_number": 12,
            "text": "같은 예외를 인정하며 검사원의 확인을 받아야 한다.",
        },
    ]
    chunks_dir = write_doc(scratch_dir, "doc_a", rows)
    same_page = make_chunk("doc_a_p0010_m001", rows[0]["text"], page=10)
    across_page = make_chunk("doc_a_p0011_m001", rows[2]["text"], page=11)

    out, trace = expand_chunks_with_neighbors(
        [same_page, across_page], chunks_dir=chunks_dir, require_cross_page=True
    )

    assert [t["neighbor_chunk_id"] for t in trace] == ["doc_a_p0012_m001"]
    assert len(out) == 3


def test_expansion_reason_ignores_a_tiny_neighbor():
    parent = make_chunk("doc_a_p0010_m001", "요건은 다음과")
    assert expansion_reason(parent, {"chunk_id": "doc_a_p0010_m002", "text": "표 1"}, 1) is None
