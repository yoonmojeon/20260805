#!/usr/bin/env python3
"""Build a review queue for the expanded KR Part 7 table pilot.

This is a diagnostic classifier, not a ground-truth evaluator.  It compares the
TATR grid with the legacy detector and PDF vector-line candidates, then attaches
HancomEQN restoration coverage so risky tables can be reviewed first.
"""

from __future__ import annotations

import csv
import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAP_PATH = ROOT / "data/manifests/kr7_expanded_snap_results.json"
FORMULA_PATH = ROOT / "data/manifests/kr7_expanded_formula_results.json"
OUT_JSON = ROOT / "data/manifests/kr7_expanded_audit_queue.json"
OUT_CSV = ROOT / "data/manifests/kr7_expanded_audit_queue.csv"


MANUAL_FINDINGS = {
    "kr_kr_rules_7_2025_2f2d6373_p0014_t009": {
        "manual_structure_finding": "body_grid_6x12_plus_footnote; tatr_includes_footnote_as_row",
        "manual_action": "separate_footnote_then_keep_6x12_body_grid",
    },
    "kr_kr_rules_7_2025_2f2d6373_p0019_t012": {
        "manual_structure_finding": "tatr_oversegments_multiline_coefficient_groups",
        "manual_action": "prefer_full_width_pdf_row_boundaries; use_tatr_for_spans_only",
    },
    "kr_kr_rules_7_2025_2f2d6373_p0200_t073": {
        "manual_structure_finding": "tatr_8x2_matches_visual_logical_grid",
        "manual_action": "accept_after_formula_review",
    },
    "kr_kr_rules_7_2025_2f2d6373_p0207_t075": {
        "manual_structure_finding": "compound_layout_with_multiple_nested_subtables",
        "manual_action": "segment_into_regions_before_table_structure_recognition",
    },
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def classify(row: dict) -> str:
    vector_rows = max(0, row["pdf_horizontal_candidates"] - 1)
    vector_columns = max(0, row["pdf_vertical_candidates"] - 1)
    tatr_vector = (row["tatr_rows"], row["tatr_columns"]) == (
        vector_rows,
        vector_columns,
    )
    legacy_vector = (row["legacy_rows"], row["legacy_columns"]) == (
        vector_rows,
        vector_columns,
    )
    tatr_legacy = row["row_match_legacy"] and row["column_match_legacy"]
    if tatr_vector and legacy_vector:
        return "three_way_agreement"
    if tatr_vector:
        return "tatr_vector_agreement"
    if legacy_vector:
        return "legacy_vector_agreement"
    if tatr_legacy:
        return "detectors_agree_vector_differs"
    return "all_disagree_manual_review"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snap", type=Path, default=SNAP_PATH)
    parser.add_argument("--formula", type=Path, default=FORMULA_PATH)
    parser.add_argument("--output", type=Path, default=OUT_JSON)
    args = parser.parse_args()
    snaps = load_json(args.snap)["tables"]
    formulas = {
        row["table_id"]: row for row in load_json(args.formula)["tables"]
    }
    rows = []
    for snap in snaps:
        row = dict(snap)
        row["pdf_vector_rows"] = max(0, row["pdf_horizontal_candidates"] - 1)
        row["pdf_vector_columns"] = max(0, row["pdf_vertical_candidates"] - 1)
        row["structure_signal"] = classify(row)
        formula = formulas.get(row["table_id"], {})
        for key in (
            "hancomeqn_cells",
            "formula_dominant_cells",
            "needs_review_cells",
            "fully_mapped_cells",
            "unknown_glyphs",
        ):
            row[key] = formula.get(key, [] if key == "unknown_glyphs" else 0)
        total = row["hancomeqn_cells"]
        row["formula_full_mapping_rate"] = (
            round(row["fully_mapped_cells"] / total, 4) if total else None
        )
        row.update(MANUAL_FINDINGS.get(row["table_id"], {}))
        row["priority_score"] = (
            3 * (row["structure_signal"] == "all_disagree_manual_review")
            + 2 * (row["structure_signal"] == "detectors_agree_vector_differs")
            + 2 * bool(row["needs_review_cells"])
            + int(row["table_id"] in MANUAL_FINDINGS)
        )
        rows.append(row)

    rows.sort(key=lambda r: (-r["priority_score"], r["page"], r["table_id"]))
    signals = Counter(row["structure_signal"] for row in rows)
    output = {
        "warning": (
            "Agreement with PDF vector candidates is a heuristic, not ground truth. "
            "Compound layouts and footnotes still require region-level review."
        ),
        "summary": {
            "tables": len(rows),
            "structure_signals": dict(signals),
            "tables_with_unrestored_formula_cells": sum(
                bool(row["needs_review_cells"]) for row in rows
            ),
            "manual_findings_recorded": len(MANUAL_FINDINGS),
        },
        "tables": rows,
    }
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fields = [
        "priority_score",
        "table_id",
        "page",
        "caption",
        "structure_signal",
        "tatr_rows",
        "tatr_columns",
        "legacy_rows",
        "legacy_columns",
        "pdf_vector_rows",
        "pdf_vector_columns",
        "hancomeqn_cells",
        "fully_mapped_cells",
        "needs_review_cells",
        "formula_full_mapping_rate",
        "manual_structure_finding",
        "manual_action",
    ]
    out_csv = args.output.with_suffix(".csv")
    with out_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(args.output)
    print(out_csv)


if __name__ == "__main__":
    main()
