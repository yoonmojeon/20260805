#!/usr/bin/env python
"""Inspect live MaritimeOpsRAG text/table Chroma indexes and coverage."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import DEFAULT_RAG_COLLECTION, DEFAULT_TABLE_COLLECTION
from services.rag_index_diag import diagnose_rag_indexes, format_rag_index_banner
from services.rag_service import dual_retrieval_enabled


def _print_text_coverage() -> None:
    out = ROOT / "data" / "eval" / "text_corpus_coverage.json"
    if not out.exists():
        sys.path.insert(0, str(ROOT / "scripts"))
        import audit_text_coverage

        audit_text_coverage.main()
    report = json.loads(out.read_text(encoding="utf-8"))
    print("=== TEXT COVERAGE DETAIL ===")
    print()
    print("collection:")
    print(DEFAULT_RAG_COLLECTION)
    print()
    print("indexed_documents:")
    print(report.get("text_indexed_documents"))
    print()
    print("expected_documents:")
    print(report.get("corpus_csv_rows"))
    print()
    missing = report.get("missing_from_text_index") or []
    print("missing_documents:")
    print(len(missing))
    print()
    print("missing:")
    for item in missing:
        print(f"- {item.get('file_name')} ({item.get('doc_id')})")
        print(f"  reason: {item.get('reason')}")
        print(f"  pdf_bytes: {item.get('pdf_bytes')} page_text_len: {item.get('page_text_len')}")
    print()


def _print_table_coverage(*, run_audit: bool) -> None:
    out = ROOT / "data" / "eval" / "table_coverage_audit.json"
    if run_audit or not out.exists():
        sys.path.insert(0, str(ROOT / "scripts"))
        import audit_table_coverage

        audit_table_coverage.main()
    report = json.loads(out.read_text(encoding="utf-8"))
    s = report["summary"]
    print("=== TABLE INDEX ===")
    print()
    print("collection:")
    print(DEFAULT_TABLE_COLLECTION)
    print()
    print("indexed_documents:")
    print(s.get("indexed"))
    print()
    print("expected_documents:")
    print(s.get("expected_documents"))
    print()
    print("without_table_chunks:")
    print(s.get("without_table_chunks"))
    print()
    print("coverage:")
    print(f"{s.get('coverage_pct')}%")
    print()
    print("classification:")
    print(f"no_table_detected: {s.get('no_table_detected')}")
    print(f"extraction_failed: {s.get('extraction_failed')}")
    print(f"filtered_empty: {s.get('filtered_empty')}")
    print(f"unknown: {s.get('unknown')}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--json", action="store_true", help="Print diagnose JSON")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also run text/table coverage audits (writes data/eval/*.json)",
    )
    args = parser.parse_args()

    report = diagnose_rag_indexes(sample_size=args.sample_size)
    print(format_rag_index_banner(report))
    print()

    # Always show lightweight coverage headers from manifests
    text = report.get("text") or {}
    table = report.get("table") or {}
    print("=== TEXT INDEX ===")
    print()
    print("collection:")
    print(report.get("text_collection"))
    print()
    print("chunks:")
    print(text.get("chunks"))
    print()
    print("indexed_documents:")
    print(text.get("documents"))
    print()
    print("expected_documents:")
    print(715)
    miss = (text.get("manifest") or {}).get("missing_docs")
    print()
    print("missing_documents:")
    print(miss if miss is not None else "?")
    print()

    print("=== TABLE INDEX ===")
    print()
    print("collection:")
    print(report.get("table_collection"))
    print()
    print("chunks:")
    print(table.get("chunks"))
    print()
    print("indexed_documents:")
    print(table.get("documents"))
    print()
    print("expected_documents:")
    print(715)
    print()
    without = None
    if table.get("documents") is not None:
        without = 715 - int(table.get("documents") or 0)
        # prefer manifest missing_docs if present
        if table.get("missing_docs") is not None:
            without = table.get("missing_docs")
    print("without_table_chunks:")
    print(without)
    if table.get("documents"):
        print()
        print("coverage:")
        print(f"{round(100.0 * int(table['documents']) / 715, 2)}%")
    print()

    print("=== RAG CONFIG ===")
    print()
    print("dual_retrieval_enabled:")
    print(str(dual_retrieval_enabled()).lower())
    print()
    print("MARITIME_RAG_DUAL env:")
    print(repr(os.environ.get("MARITIME_RAG_DUAL", "<unset → default 1>")))
    print()
    print("default_retrieval_modes:")
    print("TEXT / TABLE / BOTH")
    print()

    if args.full:
        _print_text_coverage()
        _print_table_coverage(run_audit=True)

    for kind in ("text", "table"):
        sample = (report.get(kind) or {}).get("sample_chunk")
        if not sample:
            continue
        print(f"Sample {kind} chunk:")
        for key in (
            "chunk_type",
            "doc_id",
            "file_name",
            "page_number",
            "table_id",
            "caption",
            "crop_path",
            "element_id",
        ):
            if sample.get(key) not in (None, ""):
                print(f"  {key}: {sample.get(key)}")
        print("  text:")
        for line in str(sample.get("text") or "").splitlines()[:12]:
            print(f"    {line}")
        print()

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
