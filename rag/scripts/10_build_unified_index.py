"""
Build a single Chroma collection across multiple processed documents.

  python scripts/10_build_unified_index.py --doc-list data/manifests/pilot_100_docs.csv
  python scripts/10_build_unified_index.py --doc-id kr_1_2025 --collection-id pilot_100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from embedding_policy import (
    DEFAULT_EMBEDDING_PRESET,
    ALLOWED_EMBEDDING_PRESETS,
    REQUIRED_LANGUAGES,
    embed_texts_local,
    get_embedding_tokenizer,
    resolve_embedding_config,
)
from embedding_chunk_policy import (
    CHUNK_POLICY_VERSION,
    DEFAULT_EMBEDDING_OVERLAP_TOKENS,
    DEFAULT_MAX_EMBEDDING_TOKENS,
    EMBEDDING_POLICY_VERSION,
    prepare_chunks_for_embedding,
)
from index_build_lib import (
    DEFAULT_INDEX_TYPES,
    MIN_INDEX_TEXT_CHARS,
    build_chroma_index,
    chunk_metadata,
    embedding_text_and_mode,
    filter_chunks_for_index,
    folder_from_path,
    load_chunk_ids_from_suspicious_csv,
    load_chunks,
)


def load_doc_list(path: Path) -> list[dict]:
    import pandas as pd

    df = pd.read_csv(path)
    if "doc_id" not in df.columns:
        raise ValueError(f"doc-list must have doc_id column: {path}")
    return df.to_dict(orient="records")


def load_manifest_map(manifest_path: Path) -> dict[str, dict]:
    import pandas as pd

    if not manifest_path.exists():
        return {}
    df = pd.read_csv(manifest_path)
    return {str(row["doc_id"]): row.to_dict() for _, row in df.iterrows()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified multi-document Chroma index.")
    parser.add_argument("--doc-list", type=Path, default=None, help="CSV with doc_id column")
    parser.add_argument("--doc-id", action="append", dest="doc_ids", help="Repeatable doc_id")
    parser.add_argument("--collection-id", type=str, default="pilot_100")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/pdf_manifest.csv"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument(
        "--table-chunks-dir",
        type=Path,
        default=None,
        help="Optional separate root containing <doc_id>/table_chunks.jsonl",
    )
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/index"))
    parser.add_argument("--embedding-preset", type=str, default=DEFAULT_EMBEDDING_PRESET,
                        choices=sorted(ALLOWED_EMBEDDING_PRESETS))
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--include-types", type=str, default="text,table,picture")
    parser.add_argument("--skip-suspicious", action="store_true", default=True)
    parser.add_argument("--no-skip-suspicious", action="store_false", dest="skip_suspicious")
    parser.add_argument("--min-chars", type=int, default=MIN_INDEX_TEXT_CHARS)
    parser.add_argument(
        "--structured-tables",
        choices=("include", "exclude", "only"),
        default="include",
        help="Include, exclude, or exclusively index structured table chunks",
    )
    parser.add_argument("--max-embedding-tokens", type=int, default=DEFAULT_MAX_EMBEDDING_TOKENS)
    parser.add_argument("--embedding-overlap-tokens", type=int, default=DEFAULT_EMBEDDING_OVERLAP_TOKENS)
    parser.add_argument("--analyze-only", action="store_true", help="Measure/split chunks without embedding or writing Chroma")
    args = parser.parse_args()

    if args.doc_list:
        rows = load_doc_list(args.doc_list)
        doc_ids = [str(r["doc_id"]) for r in rows]
    elif args.doc_ids:
        doc_ids = args.doc_ids
        rows = [{"doc_id": d} for d in doc_ids]
    else:
        parser.error("Provide --doc-list or --doc-id")

    manifest_map = load_manifest_map(args.manifest)
    include_types = frozenset(t.strip().lower() for t in args.include_types.split(",") if t.strip())
    embed_config = resolve_embedding_config(args.embedding_preset, args.model)
    model_name = str(embed_config["model"])
    tokenizer = get_embedding_tokenizer(model_name)

    all_index_chunks: list[dict] = []
    all_texts: list[str] = []
    all_metas: list[dict] = []
    per_doc_stats: list[dict] = []
    missing_chunks: list[str] = []
    total_split_input = 0
    total_oversized = 0
    total_generated = 0
    max_tokens_before = 0
    max_tokens_after = 0

    for row in rows:
        doc_id = str(row["doc_id"])
        meta_row = manifest_map.get(doc_id, row)
        source = str(meta_row.get("source", row.get("source", "UNKNOWN")))
        file_name = str(meta_row.get("file_name", row.get("file_name", "")))
        file_path = str(meta_row.get("file_path", row.get("file_path", "")))
        folder = str(row.get("folder", "")) or folder_from_path(file_path)

        chunks_path = args.chunks_dir / doc_id / "chunks.jsonl"
        if not chunks_path.exists():
            missing_chunks.append(doc_id)
            continue

        skip_ids: set[str] = set()
        if args.skip_suspicious:
            suspicious_csv = Path("data/processed/logs") / f"{doc_id}_suspicious_chunks.csv"
            skip_ids = load_chunk_ids_from_suspicious_csv(suspicious_csv)

        all_chunks = load_chunks(chunks_path)
        table_chunks_root = args.table_chunks_dir or args.chunks_dir
        table_chunks_path = table_chunks_root / doc_id / "table_chunks.jsonl"
        table_chunks: list[dict] = []
        has_structured_tables = table_chunks_path.exists()
        if has_structured_tables:
            table_chunks = load_chunks(table_chunks_path)
            # Skip legacy plain-text table elements when structured table chunks exist.
            all_chunks = [
                c
                for c in all_chunks
                if not (
                    str(c.get("element_type", "")).lower() == "table"
                    and not c.get("chunk_type")
                )
            ]
            all_chunks.extend(table_chunks)
        index_chunks = filter_chunks_for_index(
            all_chunks,
            include_types,
            skip_ids,
            args.min_chars,
            structured_table_mode=args.structured_tables,
        )
        if not index_chunks:
            missing_chunks.append(doc_id)
            continue

        render_embedding = lambda chunk: embedding_text_and_mode(
            chunk, source=source, file_name=file_name, folder=folder
        )
        prepared_chunks, prepared_texts, prepared_modes, split_stats = prepare_chunks_for_embedding(
            index_chunks,
            tokenizer=tokenizer,
            model_name=model_name,
            render_embedding=render_embedding,
            max_tokens=args.max_embedding_tokens,
            overlap_tokens=args.embedding_overlap_tokens,
        )
        for chunk, text, mode in zip(prepared_chunks, prepared_texts, prepared_modes):
            all_index_chunks.append(chunk)
            all_texts.append(text)
            all_metas.append(
                chunk_metadata(
                    chunk,
                    source=source,
                    file_name=file_name,
                    folder=folder,
                    embedding_mode=mode,
                )
            )

        total_split_input += split_stats.input_chunks
        total_oversized += split_stats.oversized_chunks
        total_generated += split_stats.generated_segments
        max_tokens_before = max(max_tokens_before, split_stats.max_tokens_before)
        max_tokens_after = max(max_tokens_after, split_stats.max_tokens_after)

        table_n = len(table_chunks) if has_structured_tables else 0
        per_doc_stats.append(
            {
                "doc_id": doc_id,
                "source": source,
                "folder": folder,
                "indexed": len(index_chunks),
                "indexed_after_split": len(prepared_chunks),
                "oversized_chunks": split_stats.oversized_chunks,
                "generated_segments": split_stats.generated_segments,
                "total": len(all_chunks),
                "table_chunks": table_n,
            }
        )
        suffix = f", +{table_n} table chunks" if table_n else ""
        print(
            f"  {doc_id}: {len(index_chunks)} -> {len(prepared_chunks)}/{len(all_chunks)} "
            f"chunks ({source}, oversized={split_stats.oversized_chunks}){suffix}"
        )

    if not all_index_chunks:
        raise ValueError(
            "No chunks indexed. Process documents first (run_rag_batch --doc-list ... --steps chunks)."
            f" Missing chunks for: {missing_chunks[:10]}"
        )

    if missing_chunks:
        print(f"[WARN] Skipped {len(missing_chunks)} doc(s) without chunks: {missing_chunks[:8]}...")

    collection_name = f"maritime_{args.collection_id}_chunks"
    index_root = args.index_dir / f"unified_{args.collection_id}"
    chroma_dir = index_root / "chroma"

    split_summary = {
        "input_chunks": total_split_input,
        "output_chunks": len(all_index_chunks),
        "oversized_chunks": total_oversized,
        "generated_segments": total_generated,
        "max_tokens_before": max_tokens_before,
        "max_tokens_after": max_tokens_after,
    }

    if args.analyze_only:
        analysis_path = Path("data/processed/logs") / f"{args.collection_id}_embedding_policy_analysis.json"
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        analysis = {
            "collection_id": args.collection_id,
            "embedding_model": model_name,
            "embedding_model_revision": embed_config.get("revision"),
            "embedding_policy_version": EMBEDDING_POLICY_VERSION,
            "chunk_policy_version": CHUNK_POLICY_VERSION,
            "max_embedding_tokens": args.max_embedding_tokens,
            "embedding_overlap_tokens": args.embedding_overlap_tokens,
            "structured_table_mode": args.structured_tables,
            "documents": len(per_doc_stats),
            "missing_chunks_doc_ids": missing_chunks,
            "split_summary": split_summary,
            "per_doc": per_doc_stats,
        }
        analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nAnalysis only: {analysis_path}")
        print("Split summary:", split_summary)
        return

    print(f"\nEmbedding {len(all_texts)} chunks with {model_name}...")
    embeddings = embed_texts_local(all_texts, model_name, for_query=False)

    build_chroma_index(
        all_index_chunks,
        all_texts,
        all_metas,
        embeddings,
        chroma_dir,
        collection_name,
    )

    mode_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for m in all_metas:
        mode_counts[m["embedding_mode"]] = mode_counts.get(m["embedding_mode"], 0) + 1
        source_counts[m["source"]] = source_counts.get(m["source"], 0) + 1

    manifest_out = {
        "collection_id": args.collection_id,
        "collection_name": collection_name,
        "embedding_preset": args.embedding_preset,
        "embedding_model": model_name,
        "embedding_model_revision": embed_config.get("revision"),
        "embedding_provider": embed_config["provider"],
        "embedding_policy_version": EMBEDDING_POLICY_VERSION,
        "chunk_policy_version": CHUNK_POLICY_VERSION,
        "max_embedding_tokens": args.max_embedding_tokens,
        "embedding_overlap_tokens": args.embedding_overlap_tokens,
        "structured_table_mode": args.structured_tables,
        "table_chunks_dir": args.table_chunks_dir.resolve().as_posix() if args.table_chunks_dir else None,
        "embedding_split_stats": split_summary,
        "languages": list(embed_config.get("languages", REQUIRED_LANGUAGES)),
        "chroma_path": chroma_dir.resolve().as_posix(),
        "doc_list": args.doc_list.resolve().as_posix() if args.doc_list else None,
        "doc_ids": [s["doc_id"] for s in per_doc_stats],
        "indexed_chunks": len(all_index_chunks),
        "indexed_by_source": source_counts,
        "indexed_by_embedding_mode": mode_counts,
        "per_doc": per_doc_stats,
        "missing_chunks_doc_ids": missing_chunks,
    }
    index_root.mkdir(parents=True, exist_ok=True)
    manifest_path = index_root / "index_manifest.json"
    manifest_path.write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nUnified index: {chroma_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Total indexed: {len(all_index_chunks)} chunks from {len(per_doc_stats)} documents")
    print("By source:", source_counts)
    print("By embedding_mode:", mode_counts)


if __name__ == "__main__":
    main()
