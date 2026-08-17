"""User-facing RAG citation presentation tests."""
from __future__ import annotations

from services.rag_presentation import compact_citations


def test_citations_are_compacted_in_first_use_order_with_matching_rows():
    answer = (
        "## 1) 핵심 요약\n\n"
        "- 결정사항입니다. [2]\n"
        "- 훈련 접근입니다. [4]\n"
        "- 일정입니다. [7][4]"
    )
    rows = [
        {"citation_id": "[2]", "file_name": "decision.pdf", "page": 1},
        {"citation_id": "[4]", "file_name": "training.pdf", "page": 2},
        {"citation_id": "[7]", "file_name": "timeline.pdf", "page": 3},
    ]

    normalized, evidence, mapping = compact_citations(answer, rows)

    assert "결정사항입니다. [1]" in normalized
    assert "훈련 접근입니다. [2]" in normalized
    assert "일정입니다. [3][2]" in normalized
    assert [row["citation_id"] for row in evidence] == ["[1]", "[2]", "[3]"]
    assert [row["file_name"] for row in evidence] == [
        "decision.pdf",
        "training.pdf",
        "timeline.pdf",
    ]
    assert mapping == {2: 1, 4: 2, 7: 3}


def test_first_cited_source_becomes_one_even_when_evidence_rows_are_unsorted():
    answer = "- 두 번째 근거를 먼저 사용합니다. [5]\n- 다음 근거입니다. [1]"
    rows = [
        {"citation_id": "[1]", "file_name": "later.pdf"},
        {"citation_id": "[5]", "file_name": "first.pdf"},
    ]

    normalized, evidence, _mapping = compact_citations(answer, rows)

    assert normalized.endswith("다음 근거입니다. [2]")
    assert [row["file_name"] for row in evidence] == ["first.pdf", "later.pdf"]
    assert [row["citation_id"] for row in evidence] == ["[1]", "[2]"]
