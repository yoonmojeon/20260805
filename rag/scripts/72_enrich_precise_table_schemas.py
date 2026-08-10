#!/usr/bin/env python3
"""Add table_schema chunks to the precise table corpus and upsert into Chroma.

Does NOT re-run TATR/OCR. Derives schema catalog text from existing
table_summary / table_row chunks, embeds only the new schema chunks, then
optionally rebuilds a slim table BM25 (schema+summary).

  python rag/scripts/72_enrich_precise_table_schemas.py
  python rag/scripts/72_enrich_precise_table_schemas.py --skip-index
  python rag/scripts/72_enrich_precise_table_schemas.py --skip-bm25
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parents[1]
sys.path[:0] = [str(_SCRIPT_DIR), str(_ROOT)]

from project_paths import DEFAULT_TABLE_COLLECTION, RAG_INDEX_DIR, RAG_TABLE_CHUNKS_DIR
from table_schema_lib import build_table_schema_text, parse_schema_from_document


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _schema_chunk_from_group(table_id: str, group: list[dict]) -> dict | None:
    summary = next((c for c in group if c.get("chunk_type") == "table_summary"), None)
    rows = [c for c in group if c.get("chunk_type") == "table_row"]
    base = summary or (rows[0] if rows else None)
    if base is None:
        return None
    raw = str((summary or {}).get("text") or "")
    if not raw:
        raw = "\n".join(str(r.get("text") or "") for r in rows[:48])
    meta = {
        "table_id": table_id,
        "caption": str(base.get("caption") or ""),
        "source_file": str(base.get("file_name") or ""),
        "file_name": str(base.get("file_name") or ""),
        "page": base.get("page") or base.get("page_number") or 0,
        "doc_id": str(base.get("doc_id") or ""),
    }
    schema = parse_schema_from_document(raw, meta)
    schema["doc_id"] = meta["doc_id"]
    schema["source_file"] = meta["file_name"]
    schema["page"] = int(meta["page"] or 0)
    schema["table_id"] = table_id
    schema["row_count"] = max(len(rows), int(schema.get("row_count") or 0))
    if not schema.get("_raw_snippet"):
        schema["_raw_snippet"] = raw[:900]
    text = build_table_schema_text(schema)
    return {
        "doc_id": meta["doc_id"],
        "source": str(base.get("source") or "KR"),
        "file_name": meta["file_name"],
        "page": schema["page"],
        "page_number": schema["page"],
        "element_type": "table",
        "table_id": table_id,
        "caption": str(schema.get("caption") or base.get("caption") or ""),
        "section_title": str(schema.get("section_title") or ""),
        "column_names": list(schema.get("column_names") or []),
        "parser_version": str(base.get("parser_version") or ""),
        "quality_status": str(base.get("quality_status") or "pass"),
        "quality_score": float(base.get("quality_score") or 1.0),
        "crop_path": str(base.get("crop_path") or ""),
        "chunk_id": f"{table_id}__schema",
        "element_id": f"{table_id}__schema",
        "chunk_type": "table_schema",
        "text": text,
    }


def enrich_jsonl(chunks_root: Path) -> tuple[int, int, list[dict]]:
    """Rewrite jsonl files with one table_schema per table_id. Returns (docs, schemas, schema_chunks)."""
    schema_chunks: list[dict] = []
    docs = 0
    for path in sorted(chunks_root.glob("*/table_chunks.jsonl")):
        docs += 1
        rows = _read_jsonl(path)
        by_table: dict[str, list[dict]] = defaultdict(list)
        kept: list[dict] = []
        for row in rows:
            if str(row.get("chunk_type") or "") == "table_schema":
                continue
            kept.append(row)
            tid = str(row.get("table_id") or "")
            if tid:
                by_table[tid].append(row)
        new_schemas: list[dict] = []
        for tid, group in by_table.items():
            chunk = _schema_chunk_from_group(tid, group)
            if chunk:
                new_schemas.append(chunk)
        _write_jsonl(path, kept + new_schemas)
        schema_chunks.extend(new_schemas)
        if docs % 50 == 0:
            print(f"  jsonl {docs} docs, schemas so far {len(schema_chunks)}", flush=True)
    return docs, len(schema_chunks), schema_chunks


def upsert_schemas_to_chroma(
    schema_chunks: list[dict],
    *,
    collection_id: str,
    index_dir: Path,
    batch_size: int = 64,
) -> int:
    from embedding_policy import embed_texts_local
    from index_build_lib import chunk_metadata, embedding_text_for_table_chunk
    from rag_resource_cache import clear_process_caches, load_unified_collection

    clear_process_caches()
    collection, embed_model, manifest = load_unified_collection(collection_id, index_dir)

    # Drop any previous schema ids to keep re-runs idempotent.
    existing = collection.get(where={"chunk_type": "table_schema"}, include=[])
    old_ids = list(existing.get("ids") or [])
    if old_ids:
        for start in range(0, len(old_ids), 500):
            collection.delete(ids=old_ids[start : start + 500])
        print(f"deleted existing table_schema chunks: {len(old_ids)}", flush=True)

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    embed_texts: list[str] = []
    for chunk in schema_chunks:
        cid = str(chunk["chunk_id"])
        text = str(chunk.get("text") or "").strip()
        if len(text) < 10:
            continue
        source = str(chunk.get("source") or "")
        file_name = str(chunk.get("file_name") or "")
        emb_text, mode = embedding_text_for_table_chunk(
            chunk,
            source=source,
            file_name=file_name,
        )
        ids.append(cid)
        docs.append(text)
        metas.append(
            chunk_metadata(
                chunk,
                source=source,
                file_name=file_name,
                folder="",
                embedding_mode=mode,
            )
        )
        embed_texts.append(emb_text)

    print(f"embedding {len(ids)} schema chunks with {embed_model}…", flush=True)
    t0 = time.time()
    vectors = embed_texts_local(embed_texts, embed_model, for_query=False)
    print(f"embed done in {time.time() - t0:.1f}s", flush=True)

    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=docs[start:end],
            embeddings=vectors[start:end],
            metadatas=metas[start:end],
        )
        if end % 1024 == 0 or end >= len(ids):
            print(f"  upserted {end}/{len(ids)}", flush=True)

    # Refresh manifest indexed_chunks so BM25 fingerprint moves.
    root = index_dir / f"unified_{collection_id}"
    man_path = root / "index_manifest.json"
    if man_path.exists():
        payload = json.loads(man_path.read_text(encoding="utf-8"))
        try:
            payload["indexed_chunks"] = int(collection.count())
        except Exception:
            payload["indexed_chunks"] = int(payload.get("indexed_chunks") or 0) + len(ids)
        payload["schema_enrichment"] = {
            "schema_chunks_added": len(ids),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        man_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    clear_process_caches()
    return len(ids)


def rebuild_slim_table_bm25(collection_id: str, index_dir: Path) -> int:
    from bm25_index import (
        TABLE_TOKENIZER_VERSION,
        build_bm25_from_collection,
        clear_bm25_caches,
        table_bm25_index_dir,
    )
    from rag_resource_cache import load_unified_collection, unified_index_fingerprint

    clear_bm25_caches()
    out = table_bm25_index_dir(index_dir, collection_id)
    if out.exists():
        shutil.rmtree(out)
    collection, _, _ = load_unified_collection(collection_id, index_dir)
    fp = unified_index_fingerprint(collection_id, index_dir)
    print("building slim table BM25 (table_schema only)…", flush=True)
    t0 = time.time()
    inst = build_bm25_from_collection(
        collection,
        unified_id=collection_id,
        index_dir=index_dir,
        fingerprint=fp,
        tokenizer_mode=TABLE_TOKENIZER_VERSION,
        out_dir=out,
        chunk_types={"table_schema"},
    )
    size_mb = sum(p.stat().st_size for p in out.glob("*")) / (1024 * 1024)
    print(
        f"BM25 done: {len(inst.chunk_ids)} chunks, {size_mb:.1f} MB, {time.time() - t0:.1f}s",
        flush=True,
    )
    return len(inst.chunk_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-root", type=Path, default=RAG_TABLE_CHUNKS_DIR)
    parser.add_argument("--collection-id", default=DEFAULT_TABLE_COLLECTION)
    parser.add_argument("--index-dir", type=Path, default=RAG_INDEX_DIR)
    parser.add_argument("--skip-jsonl", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-bm25", action="store_true")
    args = parser.parse_args()

    t_all = time.time()
    schema_chunks: list[dict] = []
    if not args.skip_jsonl:
        print(f"enriching jsonl under {args.chunks_root} …", flush=True)
        docs, n_schema, schema_chunks = enrich_jsonl(args.chunks_root)
        print(f"jsonl done: docs={docs} schemas={n_schema}", flush=True)
    else:
        # Load schemas from disk for upsert-only runs.
        for path in sorted(args.chunks_root.glob("*/table_chunks.jsonl")):
            for row in _read_jsonl(path):
                if row.get("chunk_type") == "table_schema":
                    schema_chunks.append(row)
        print(f"loaded {len(schema_chunks)} schema chunks from disk", flush=True)

    if not args.skip_index:
        n = upsert_schemas_to_chroma(
            schema_chunks,
            collection_id=args.collection_id,
            index_dir=args.index_dir,
        )
        print(f"chroma upserted schemas: {n}", flush=True)

    if not args.skip_bm25:
        rebuild_slim_table_bm25(args.collection_id, args.index_dir)

    print(f"ALL DONE in {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
