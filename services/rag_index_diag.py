"""Read live Chroma stats for text/table indexes (no hardcoded counts)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from project_paths import (
    DEFAULT_RAG_COLLECTION,
    DEFAULT_TABLE_COLLECTION,
    RAG_DIR,
    RAG_INDEX_DIR,
    RAG_SCRIPTS_DIR,
)


def unified_root(collection_id: str) -> Path:
    return RAG_INDEX_DIR / f"unified_{collection_id}"


def index_ready(collection_id: str) -> bool:
    root = unified_root(collection_id)
    return (root / "chroma").exists() or (root / "index_manifest.json").exists()


def _load_collection(collection_id: str):
    import os
    import sys

    if str(RAG_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(RAG_SCRIPTS_DIR))
    prev = os.getcwd()
    try:
        os.chdir(RAG_DIR)
        from rag_resource_cache import load_unified_collection  # type: ignore

        return load_unified_collection(collection_id, RAG_INDEX_DIR)
    finally:
        os.chdir(prev)


def _manifest_stats(collection_id: str) -> dict[str, Any]:
    path = unified_root(collection_id) / "index_manifest.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, Any] = {"manifest_path": str(path)}
    if not isinstance(payload, dict):
        return out
    if "indexed_chunks" in payload:
        out["indexed_chunks"] = payload["indexed_chunks"]
    doc_ids = payload.get("doc_ids")
    if isinstance(doc_ids, list):
        out["documents"] = len(doc_ids)
    per_doc = payload.get("per_doc")
    if isinstance(per_doc, list):
        out["per_doc_entries"] = len(per_doc)
    missing = payload.get("missing_chunks_doc_ids")
    if isinstance(missing, list):
        out["missing_docs"] = len(missing)
    out["collection_id"] = payload.get("collection_id")
    out["structured_table_mode"] = payload.get("structured_table_mode")
    return out


def _sample_aggregate(collection, *, sample_size: int = 12000) -> dict[str, Any]:
    """Approximate unique docs/tables from a metadata sample + full count()."""
    total = int(collection.count())
    sample_size = min(sample_size, total) if total else 0
    docs: set[str] = set()
    tables: set[str] = set()
    types: Counter[str] = Counter()
    pages: set[tuple[str, int]] = set()
    sample_chunk: dict[str, Any] | None = None

    if sample_size <= 0:
        return {
            "chunks": total,
            "documents_sampled_unique": 0,
            "tables_sampled_unique": 0,
            "chunk_types_sampled": {},
            "sample_chunk": None,
        }

    got = collection.get(limit=sample_size, include=["documents", "metadatas"])
    metas = got.get("metadatas") or []
    docs_text = got.get("documents") or []
    for i, meta in enumerate(metas):
        meta = meta or {}
        did = str(meta.get("doc_id") or meta.get("file_name") or "")
        if did:
            docs.add(did)
        tid = str(meta.get("table_id") or "")
        if tid:
            tables.add(tid)
        ctype = str(meta.get("chunk_type") or meta.get("element_type") or "unknown")
        types[ctype] += 1
        page = meta.get("page_number")
        try:
            page_i = int(page) if page is not None and str(page).strip() != "" else None
        except Exception:
            page_i = None
        if did and page_i is not None:
            pages.add((did, page_i))
        if sample_chunk is None and ctype in {"table_row", "unknown", "text"}:
            sample_chunk = {
                "chunk_type": ctype,
                "table_id": tid,
                "doc_id": did,
                "file_name": meta.get("file_name"),
                "page_number": meta.get("page_number"),
                "caption": meta.get("caption"),
                "crop_path": meta.get("crop_path"),
                "element_id": meta.get("element_id"),
                "text": (docs_text[i] or "")[:500],
            }

    return {
        "chunks": total,
        "sample_size": sample_size,
        "documents_sampled_unique": len(docs),
        "tables_sampled_unique": len(tables),
        "doc_page_pairs_sampled": len(pages),
        "chunk_types_sampled": dict(types),
        "sample_chunk": sample_chunk,
        "note": (
            "documents/tables unique counts are from a metadata sample, "
            "not a full scan; chunks is collection.count()."
        ),
    }


def diagnose_rag_indexes(*, sample_size: int = 12000) -> dict[str, Any]:
    """Return live stats for text + table collections."""
    report: dict[str, Any] = {
        "text_collection": DEFAULT_RAG_COLLECTION,
        "table_collection": DEFAULT_TABLE_COLLECTION,
        "index_dir": str(RAG_INDEX_DIR),
        "text": {"ready": index_ready(DEFAULT_RAG_COLLECTION)},
        "table": {"ready": index_ready(DEFAULT_TABLE_COLLECTION)},
    }

    if report["text"]["ready"]:
        collection, _, _ = _load_collection(DEFAULT_RAG_COLLECTION)
        report["text"].update(_sample_aggregate(collection, sample_size=sample_size))
        report["text"]["manifest"] = _manifest_stats(DEFAULT_RAG_COLLECTION)
        man = report["text"]["manifest"]
        if man.get("documents") is not None:
            report["text"]["documents"] = man["documents"]
        if man.get("indexed_chunks") is not None:
            report["text"]["chunks_manifest"] = man["indexed_chunks"]
    if report["table"]["ready"]:
        collection, _, _ = _load_collection(DEFAULT_TABLE_COLLECTION)
        report["table"].update(_sample_aggregate(collection, sample_size=sample_size))
        report["table"]["manifest"] = _manifest_stats(DEFAULT_TABLE_COLLECTION)
        man = report["table"]["manifest"]
        if man.get("documents") is not None:
            report["table"]["documents"] = man["documents"]
        if man.get("indexed_chunks") is not None:
            report["table"]["chunks_manifest"] = man["indexed_chunks"]
        if man.get("missing_docs") is not None:
            report["table"]["missing_docs"] = man["missing_docs"]

    return report


def format_rag_index_banner(report: dict[str, Any] | None = None) -> str:
    report = report or diagnose_rag_indexes(sample_size=2000)
    text = report.get("text") or {}
    table = report.get("table") or {}

    def _fmt_block(title: str, collection: str, block: dict[str, Any]) -> list[str]:
        lines = [f"{title} collection: {collection}"]
        if not block.get("ready"):
            lines.append(f"{title} chunks: (index missing)")
            return lines
        lines.append(f"{title} chunks: {block.get('chunks', 0):,}")
        if block.get("documents") is not None:
            lines.append(f"{title} documents: {block.get('documents'):,}")
        elif block.get("documents_sampled_unique") is not None:
            lines.append(
                f"{title} documents (sample unique): {block.get('documents_sampled_unique'):,}"
            )
        if block.get("tables_sampled_unique"):
            lines.append(
                f"{title} tables (sample unique / {block.get('sample_size', '?')} chunks): "
                f"{block.get('tables_sampled_unique'):,}"
            )
        if block.get("missing_docs") is not None:
            lines.append(f"{title} docs without table chunks: {block.get('missing_docs'):,}")
        types = block.get("chunk_types_sampled") or {}
        if types:
            top = ", ".join(f"{k}={v}" for k, v in list(types.items())[:6])
            lines.append(f"{title} chunk types (sample): {top}")
        return lines

    out = ["[RAG INDEX]"]
    out.extend(_fmt_block("Text", report["text_collection"], text))
    out.append("")
    out.extend(_fmt_block("Table", report["table_collection"], table))
    return "\n".join(out)


def fetch_table_rows_by_id(collection, table_id: str) -> tuple[list[str], list[dict[str, Any]]]:
    got = collection.get(
        where={"$and": [{"table_id": table_id}, {"chunk_type": "table_row"}]},
        include=["documents", "metadatas"],
    )
    docs = list(got.get("documents") or [])
    metas = list(got.get("metadatas") or [])
    if docs:
        return docs, metas
    got = collection.get(where={"table_id": table_id}, include=["documents", "metadatas"])
    docs, metas = [], []
    for d, m in zip(got.get("documents") or [], got.get("metadatas") or []):
        ctype = str((m or {}).get("chunk_type") or "")
        if ctype in {"table_summary"}:
            continue
        if ctype in {"table_row", "table_row_aux"} or ":ROW" in str((m or {}).get("element_id") or ""):
            docs.append(d)
            metas.append(m or {})
    return docs, metas
