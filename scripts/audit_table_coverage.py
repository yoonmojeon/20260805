#!/usr/bin/env python
"""Audit table coverage across the 715-PDF corpus vs precise table Chroma index.

Classification uses local artifacts (not 'missing from Chroma ⇒ no table'):
  - full_corpus_715.csv
  - table index manifest (indexed / missing_chunks_doc_ids)
  - precise manifest missing_table_documents
  - data/processed/tables/<doc>/tables.jsonl + summaries
  - precise audit + quarantine lists
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import (
    DATA_DIR,
    DEFAULT_TABLE_COLLECTION,
    PROCESSED_DIR,
    RAG_INDEX_DIR,
)


def _load_corpus() -> dict[str, dict[str, str]]:
    path = DATA_DIR / "manifests" / "full_corpus_715.csv"
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return {row["doc_id"]: row for row in csv.DictReader(fh)}


def _read_tables_jsonl(doc_id: str) -> list[dict[str, Any]]:
    path = PROCESSED_DIR / "tables" / doc_id / "tables.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _classify_missing(
    doc_id: str,
    *,
    corpus_row: dict[str, str],
    missing_detected: set[str],
    audit_rows: list[dict[str, Any]],
    quarantine_ids: set[str],
) -> tuple[str, dict[str, Any]]:
    info: dict[str, Any] = {
        "doc_id": doc_id,
        "file_name": corpus_row.get("file_name"),
        "source": corpus_row.get("source"),
        "category": corpus_row.get("category"),
    }
    detected = _read_tables_jsonl(doc_id)
    real = [r for r in detected if not r.get("is_pseudo_table")]
    pseudo = [r for r in detected if r.get("is_pseudo_table")]
    info["detected_total"] = len(detected)
    info["detected_real"] = len(real)
    info["detected_pseudo"] = len(pseudo)
    if pseudo:
        info["pseudo_reasons"] = dict(
            Counter(str(r.get("pseudo_reason") or "pseudo") for r in pseudo)
        )

    info["audit_tables"] = len(audit_rows)
    if audit_rows:
        indexable = [r for r in audit_rows if r.get("indexable")]
        quarantined = [r for r in audit_rows if str(r.get("table_id")) in quarantine_ids]
        info["indexable_tables"] = len(indexable)
        info["quarantined_tables"] = len(quarantined)
        reasons: Counter[str] = Counter()
        for r in audit_rows:
            for reason in r.get("reasons") or []:
                reasons[str(reason)] += 1
        info["audit_reason_counts"] = dict(reasons)

    # A) Precise pipeline listed doc as having no tables, and no real early detections
    if doc_id in missing_detected and not real and not audit_rows:
        info["reason"] = "precise_manifest_missing_table_documents; tables.jsonl has no real tables"
        return "no_table_detected", info

    # C) Only TOC/pseudo tables, or all precise tables quarantined / non-indexable
    if real == [] and pseudo:
        info["reason"] = "early_table_extract_skipped_pseudo_only"
        return "filtered_empty", info
    if audit_rows and info.get("indexable_tables", 0) == 0:
        info["reason"] = "precise_audit_all_non_indexable"
        return "filtered_empty", info
    if audit_rows and info.get("indexable_tables", 0) > 0:
        if info.get("quarantined_tables", 0) >= info.get("indexable_tables", 0):
            info["reason"] = "precise_tables_quarantined"
            return "filtered_empty", info

    # B) Had real detections or audit rows but not in final index
    if real or audit_rows:
        info["reason"] = "detected_or_audited_but_absent_from_table_chroma"
        return "extraction_failed", info

    info["reason"] = "insufficient_artifacts"
    return "unknown", info


def run_audit() -> dict[str, Any]:
    corpus = _load_corpus()
    expected = len(corpus)

    idx_path = RAG_INDEX_DIR / f"unified_{DEFAULT_TABLE_COLLECTION}" / "index_manifest.json"
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    indexed_ids = list(idx.get("doc_ids") or [])
    missing_ids = list(idx.get("missing_chunks_doc_ids") or [])

    pm = json.loads(
        (PROCESSED_DIR / "logs" / "full_corpus_715_precise_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    missing_detected = set(pm.get("missing_table_documents") or [])

    audit = json.loads(
        (PROCESSED_DIR / "logs" / "full_corpus_715_precise_audit.json").read_text(
            encoding="utf-8"
        )
    )
    audit_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit.get("tables") or []:
        audit_by_doc[str(row.get("doc_id"))].append(row)

    quarantine = json.loads(
        (PROCESSED_DIR / "logs" / "precise_table_quarantine_ids.json").read_text(
            encoding="utf-8"
        )
    )
    quarantine_ids = set(quarantine.get("table_ids") or [])

    categories: dict[str, list[dict[str, Any]]] = {
        "indexed": [],
        "no_table_detected": [],
        "extraction_failed": [],
        "filtered_empty": [],
        "unknown": [],
    }

    for doc_id in indexed_ids:
        row = corpus.get(doc_id) or {}
        categories["indexed"].append(
            {
                "doc_id": doc_id,
                "file_name": row.get("file_name"),
                "source": row.get("source"),
            }
        )

    for doc_id in missing_ids:
        cat, info = _classify_missing(
            doc_id,
            corpus_row=corpus.get(doc_id) or {},
            missing_detected=missing_detected,
            audit_rows=audit_by_doc.get(doc_id) or [],
            quarantine_ids=quarantine_ids,
        )
        categories[cat].append(info)

    summary = {
        "expected_documents": expected,
        "table_collection": DEFAULT_TABLE_COLLECTION,
        "indexed": len(categories["indexed"]),
        "without_table_chunks": len(missing_ids),
        "no_table_detected": len(categories["no_table_detected"]),
        "extraction_failed": len(categories["extraction_failed"]),
        "filtered_empty": len(categories["filtered_empty"]),
        "unknown": len(categories["unknown"]),
        "coverage_pct": round(100.0 * len(categories["indexed"]) / expected, 2)
        if expected
        else 0.0,
        "notes": [
            "no_table_detected: precise manifest missing_table_documents and no real tables.jsonl rows",
            "filtered_empty: TOC/pseudo-only detections or quarantined/non-indexable precise tables",
            "extraction_failed: real detection/audit existed but doc absent from table Chroma",
        ],
    }

    return {
        "summary": summary,
        "indexed": categories["indexed"],
        "no_table_detected": categories["no_table_detected"],
        "extraction_failed": categories["extraction_failed"],
        "filtered_empty": categories["filtered_empty"],
        "unknown": categories["unknown"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "eval" / "table_coverage_audit.json",
    )
    args = parser.parse_args()
    report = run_audit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    s = report["summary"]
    print("TOTAL PDFs                ", s["expected_documents"])
    print("TABLE INDEXED             ", s["indexed"])
    print("NO TABLE DETECTED         ", s["no_table_detected"])
    print("TABLE EXTRACTION FAILED   ", s["extraction_failed"])
    print("FILTERED / EMPTY          ", s["filtered_empty"])
    print("UNKNOWN                   ", s["unknown"])
    print("coverage                  ", f"{s['coverage_pct']}%")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
