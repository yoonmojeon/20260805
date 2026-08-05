"""Freeze the real 715-PDF corpus and derive the unfinished preprocessing list."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the complete 715-PDF corpus manifests.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/pdf_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/full_corpus_715.csv"),
    )
    parser.add_argument(
        "--remaining-output",
        type=Path,
        default=Path("data/manifests/full_corpus_715_remaining.csv"),
    )
    parser.add_argument(
        "--reverse-output",
        type=Path,
        default=Path("data/manifests/full_corpus_715_remaining_reverse.csv"),
        help="Write the unfinished rows in reverse order for a second worker",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("data/processed/chunks"),
    )
    parser.add_argument("--expected-count", type=int, default=715)
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    df = pd.read_csv(args.manifest)
    required = {"doc_id", "file_path"}
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise ValueError(f"Manifest missing columns: {missing_columns}")
    if df["doc_id"].duplicated().any():
        duplicates = df.loc[df["doc_id"].duplicated(), "doc_id"].astype(str).tolist()
        raise ValueError(f"Duplicate doc_id values: {duplicates}")

    exists_mask = df["file_path"].map(lambda raw: Path(str(raw)).is_file())
    corpus = df.loc[exists_mask].copy()
    missing_files = df.loc[~exists_mask, ["doc_id", "file_path"]].to_dict(orient="records")

    if len(corpus) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} existing PDFs, found {len(corpus)}. "
            f"Missing file rows: {len(missing_files)}"
        )
    if corpus["file_path"].duplicated().any():
        raise ValueError("The complete corpus contains duplicate file_path values")

    chunks_exist = corpus["doc_id"].astype(str).map(
        lambda doc_id: (args.chunks_dir / doc_id / "chunks.jsonl").exists()
    )
    remaining = corpus.loc[~chunks_exist].copy()
    empty_chunks = [
        doc_id
        for doc_id in corpus.loc[chunks_exist, "doc_id"].astype(str)
        if (args.chunks_dir / doc_id / "chunks.jsonl").stat().st_size == 0
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(args.output, index=False, encoding="utf-8")
    remaining.to_csv(args.remaining_output, index=False, encoding="utf-8")
    remaining.iloc[::-1].to_csv(args.reverse_output, index=False, encoding="utf-8")

    source_counts = {
        str(key): int(value)
        for key, value in corpus.groupby("source").size().sort_index().items()
    }
    summary = {
        "manifest_rows": int(len(df)),
        "existing_pdf_count": int(len(corpus)),
        "missing_file_rows": missing_files,
        "preprocessed_count": int(chunks_exist.sum()),
        "remaining_count": int(len(remaining)),
        "remaining_source_counts": {
            str(key): int(value)
            for key, value in remaining.groupby("source").size().sort_index().items()
        },
        "empty_chunk_doc_ids": empty_chunks,
        "source_counts": source_counts,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {args.remaining_output}")
    print(f"Wrote {args.reverse_output}")


if __name__ == "__main__":
    main()
