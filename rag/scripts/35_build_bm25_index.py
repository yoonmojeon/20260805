"""
Build BM25 sparse index for a unified Chroma collection.

  python scripts/35_build_bm25_index.py --unified full_corpus
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from bm25_index import (
    CHROMA_GET_BATCH_SIZE,
    TABLE_TOKENIZER_VERSION,
    build_bm25_from_collection,
    table_bm25_index_dir,
)
from rag_answer_lib import load_unified_collection
from rag_resource_cache import unified_index_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description="Build persisted BM25 index for unified corpus.")
    parser.add_argument("--unified", type=str, default="full_corpus_715_v1")
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/index"))
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--batch-size", type=int, default=CHROMA_GET_BATCH_SIZE)
    parser.add_argument("--table", action="store_true", help="Build the table-specific char-ngram BM25 index")
    parser.add_argument(
        "--chunk-types",
        type=str,
        default="",
        help="Comma-separated chunk_type filter. "
        "For --table, default is table_schema,table_summary (slim lexical index).",
    )
    args = parser.parse_args()

    collection, _, _ = load_unified_collection(args.unified, args.index_dir)
    fp = unified_index_fingerprint(args.unified, args.index_dir)
    if args.rebuild:
        from bm25_index import bm25_index_dir
        import shutil

        out = table_bm25_index_dir(args.index_dir, args.unified) if args.table else bm25_index_dir(args.index_dir, args.unified)
        if out.exists():
            shutil.rmtree(out)

    chunk_types = None
    raw_types = (args.chunk_types or "").strip()
    if raw_types:
        chunk_types = {t.strip() for t in raw_types.split(",") if t.strip()}
    elif args.table:
        # Prefer schema-only lexical index: summaries are often split and bloat
        # the pickle (~700MB+). Dense stage-1 still uses summary/row vectors.
        chunk_types = {"table_schema"}

    try:
        total = collection.count()
    except Exception:
        total = None
    if total is not None:
        print(f"Chroma collection: {total} chunks (batch_size={args.batch_size})")
    else:
        print(f"Fetching chunks in batches of {args.batch_size}…")
    if chunk_types:
        print(f"chunk_type filter: {sorted(chunk_types)}")

    inst = build_bm25_from_collection(
        collection,
        unified_id=args.unified,
        index_dir=args.index_dir,
        fingerprint=fp,
        batch_size=args.batch_size,
        tokenizer_mode=TABLE_TOKENIZER_VERSION if args.table else "generic",
        out_dir=table_bm25_index_dir(args.index_dir, args.unified) if args.table else None,
        chunk_types=chunk_types,
    )
    index_name = "table_bm25" if args.table else "bm25"
    print(f"BM25 index built: {len(inst.chunk_ids)} chunks → {args.index_dir / f'unified_{args.unified}' / index_name}")


if __name__ == "__main__":
    main()
