from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAG_SCRIPTS = ROOT / "rag" / "scripts"
for path in (ROOT, RAG_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from retrieval_query_analysis import analyze_query  # noqa: E402
from retrieval_search import extract_translated_feature_terms  # noqa: E402
from rag_inprocess import _advanced_exact_literal_hits  # noqa: E402


def test_generic_dnv_fact_does_not_inject_autonomous_documents():
    signals = analyze_query(
        "DNV 규정에 따라 2회 용접 시험편은 어느 위치에서 절단해야 합니까?"
    )
    assert "DNV-CG-0264" not in signals.expanded_terms
    assert "Smart Vessel" not in signals.expanded_terms


def test_dnv_autonomous_lookup_keeps_autonomous_expansion():
    signals = analyze_query(
        "DNV에서 자율운항 또는 Smart Vessel 관련 Rule/Guidance를 찾아줘."
    )
    assert "DNV-CG-0264" in signals.expanded_terms
    assert "Smart Vessel" in signals.expanded_terms


def test_parenthesized_meeting_source_before_korean_exclusion_is_excluded():
    signals = analyze_query(
        "환경규제(MEPC) 안건은 제외하고 MASS Code 결정·일정만 정리해줘."
    )
    assert "MEPC" in signals.excluded_sources
    assert "MEPC" not in signals.named_sources
    assert "MEPC" not in signals.constrained_sources


def test_msc_fuel_safety_query_has_four_literal_recovery_anchors():
    terms = extract_translated_feature_terms(
        "MSC 111 결과 중 연료 안전·위험평가 관련만 추려줘.", limit=4
    )
    assert terms == [
        "interim guidelines for the safety of ships using hydrogen as fuel",
        "Revised Interim Recommendations for carriage of liquefied hydrogen in bulk",
        "approve the draft work plan",
        "wind-assisted propulsion",
    ]


def test_advanced_literal_lookup_reads_exact_chroma_paragraphs() -> None:
    class FakeCollection:
        def get(self, **kwargs):
            assert kwargs["where_document"] == {"$contains": "approve the draft work plan"}
            return {
                "ids": ["msc-result-p3"],
                "documents": ["The Committee was invited to approve the draft work plan."],
                "metadatas": [
                    {
                        "doc_id": "msc-111-12",
                        "source": "MSC",
                        "file_name": "MSC 111-12 - Report of the twelfth session.pdf",
                        "page_number": 3,
                        "document_status": "report",
                    }
                ],
            }

    hits = _advanced_exact_literal_hits(
        FakeCollection(), "approve the draft work plan"
    )
    assert [hit.chunk_id for hit in hits] == ["msc-result-p3"]
    assert hits[0].page_number == 3
    assert hits[0].metadata_boost == 2.0
