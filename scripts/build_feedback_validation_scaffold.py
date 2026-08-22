"""Build a human-reviewable feedback validation scaffold from existing rows."""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/eval/pilot_validation_text_v3.jsonl"
OUT_JSONL = ROOT / "data/eval/feedback_validation_scaffold_40.jsonl"
OUT_CSV = ROOT / "data/eval/feedback_validation_scaffold_40.csv"
SEED = 260821


def main() -> None:
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Prefer evidence-precision, scope, boundary and original demo questions;
    # all content still comes verbatim from the existing 405-row set.
    priority = {
        "seed": 0,
        "evidence_precision": 1,
        "boundary": 2,
        "scope": 3,
        "counterfactual_robustness": 4,
        "format": 5,
        "paraphrase": 6,
        "noise_robustness": 7,
        "negative_rejection": 8,
        "integration": 9,
    }
    rng = random.Random(SEED)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("category") or "unknown")].append(row)
    selected: list[dict[str, Any]] = []
    # Eight from each of the five existing product categories.
    for category in sorted(grouped):
        items = list(grouped[category])
        rng.shuffle(items)
        items.sort(key=lambda row: priority.get(str(row.get("test_type") or ""), 99))
        selected.extend(items[:8])

    scaffold: list[dict[str, Any]] = []
    for row in selected:
        scaffold.append(
            {
                "question_id": row.get("question_id"),
                "category": row.get("category"),
                "test_type": row.get("test_type"),
                "question": row.get("question"),
                "expected_behavior": row.get("expected_behavior"),
                "expected_document_ids": row.get("acceptable_doc_ids") or row.get("gold_doc_ids") or [],
                "expected_evidence_chunk_ids": row.get("gold_chunk_ids") or [],
                "expected_pages": row.get("gold_pages") or [],
                "expected_keypoints": [
                    point.get("text") for point in row.get("gold_answer_points") or []
                ],
                "source_constraints": row.get("source_constraints") or {},
                "human_review_status": "pending",
                "human_notes": "",
                "source_dataset": str(SOURCE.relative_to(ROOT)),
                "source_row_unchanged": True,
            }
        )
    OUT_JSONL.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scaffold),
        encoding="utf-8",
    )
    fields = [
        "question_id", "category", "test_type", "question", "expected_behavior",
        "expected_document_ids", "expected_evidence_chunk_ids", "expected_pages",
        "expected_keypoints", "source_constraints", "human_review_status",
        "human_notes", "source_dataset", "source_row_unchanged",
    ]
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in scaffold:
            writer.writerow(
                {
                    key: (
                        json.dumps(row[key], ensure_ascii=False)
                        if isinstance(row[key], (list, dict))
                        else row[key]
                    )
                    for key in fields
                }
            )
    print(json.dumps({"rows": len(scaffold), "jsonl": str(OUT_JSONL), "csv": str(OUT_CSV)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
