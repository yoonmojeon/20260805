"""Build structured table chunks and a table-only index for a document corpus.

This is the reproducible corpus-scale entry point. It uses the generic PDF
coordinate pipeline. The KR7 TATR+HancomEQN reference pipeline remains in
scripts/59..68 because PUA maps and audit decisions are document-specific.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def doc_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "doc_id" not in rows[0]:
        raise ValueError(f"doc-list must contain doc_id: {path}")
    return [str(row["doc_id"]) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-list", type=Path, default=Path("data/manifests/full_corpus_715.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/full_corpus_715.csv"))
    parser.add_argument("--collection-id", default="full_corpus_715_tables_v1")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-v1-extraction", action="store_true")
    parser.add_argument("--skip-v2-reconstruction", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument("--embedding-preset", default="e5-base")
    args = parser.parse_args()

    python = str(args.python)
    ids = doc_ids(args.doc_list)
    print(f"Selected documents: {len(ids)}", flush=True)

    if not args.skip_preprocess:
        command = [
            python, "scripts/run_rag_batch.py", "--manifest", str(args.manifest),
            "--doc-list", str(args.doc_list), "--steps", "pdf,layout,merge,crop,chunks",
        ]
        if args.resume_completed:
            command.append("--resume-completed")
        run(command)

    if not args.skip_v1_extraction:
        for index, doc_id in enumerate(ids, 1):
            output = ROOT / "data/processed/tables" / doc_id / "tables.jsonl"
            if args.resume_completed and output.exists():
                print(f"[{index}/{len(ids)}] reuse {doc_id}", flush=True)
                continue
            print(f"[{index}/{len(ids)}] extract {doc_id}", flush=True)
            run([
                python, "scripts/07b_extract_table_chunks.py", "--doc-id", doc_id,
                "--manifest", str(args.manifest),
            ])

    if not args.skip_v2_reconstruction:
        run([
            python, "scripts/44_build_kr_tables_v2.py",
            "--doc-list", str(args.doc_list), "--manifest", str(args.manifest),
            "--report", "data/processed/logs/full_corpus_715_tables_v2_quality.json",
        ])

    if not args.skip_index:
        run([
            python, "scripts/10_build_unified_index.py",
            "--doc-list", str(args.doc_list), "--manifest", str(args.manifest),
            "--collection-id", args.collection_id,
            "--chunks-dir", "data/processed/chunks",
            "--table-chunks-dir", "data/processed/chunks_v2",
            "--embedding-preset", args.embedding_preset,
            "--include-types", "table", "--structured-tables", "only",
            "--max-embedding-tokens", "420", "--embedding-overlap-tokens", "60",
        ])

    print(f"Completed table collection: {args.collection_id}", flush=True)


if __name__ == "__main__":
    main()
