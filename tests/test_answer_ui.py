"""Smoke tests for Gradio Evidence / related-table HTML helpers."""
from __future__ import annotations

from pathlib import Path

from services.answer_ui import render_evidence_table_html, render_related_tables_html


def test_evidence_table_maps_citation_to_location() -> None:
    html = render_evidence_table_html(
        [
            {
                "citation_id": "[2]",
                "file_name": "MEPC 84-6-2.pdf",
                "page": 5,
                "chunk_preview": "AER decreased by 10.8%",
            }
        ]
    )
    assert "Evidence Table" in html
    assert "[2]" in html
    assert "MEPC 84-6-2.pdf" in html
    assert "5" in html
    assert "10.8%" in html


def test_related_tables_prefer_crop_image() -> None:
    crop = Path("data/processed/precise_tables/2025/p0051_t002/crop.png")
    if not crop.is_file():
        return  # local corpus not present
    html = render_related_tables_html(
        [
            {
                "table_id": "kr_1_2025_p0051_t002",
                "file_name": "1편_2025.pdf",
                "page": 51,
                "crop_path": str(crop.resolve()),
                "markdown": "| should | not | dominate |\n| --- | --- | --- |",
            }
        ]
    )
    assert "원본 crop" in html
    assert "data:image" in html
    assert "1편_2025.pdf" in html
    assert "should | not | dominate" not in html
