"""Process-level resource cache for embedding model, ChromaDB, manifests (Streamlit + benchmark)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from embedding_policy import (
    DEFAULT_EMBEDDING_PRESET,
    embedding_model_is_cached,
    resolve_embedding_config,
    warm_embed_model,
)

_UNIFIED_COLLECTION_CACHE: dict[str, tuple[str, Any, str, dict]] = {}


def _resolve_chroma_path(root: Path, manifest: dict) -> Path:
    """Prefer the Chroma directory beside the manifest after a project move/copy."""
    local_path = root / "chroma"
    if local_path.exists():
        return local_path.resolve()
    configured = Path(str(manifest.get("chroma_path") or local_path))
    return configured.expanduser().resolve()


def unified_index_fingerprint(unified_id: str, index_dir: Path) -> str:
    root = index_dir / f"unified_{unified_id}"
    manifest_path = root / "index_manifest.json"
    if not manifest_path.exists():
        return "missing"
    parts = [str(manifest_path.stat().st_mtime_ns)]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chroma_path = _resolve_chroma_path(root, manifest)
        chroma_db = chroma_path / "chroma.sqlite3"
        if chroma_db.exists():
            parts.append(str(chroma_db.stat().st_mtime_ns))
        parts.append(str(manifest.get("indexed_chunks", "")))
        parts.append(str(len(manifest.get("doc_ids") or [])))
    except Exception:
        pass
    return ":".join(parts)


def unified_cache_key(unified_id: str, index_dir: Path) -> str:
    return f"{unified_id}:{index_dir.resolve().as_posix()}"


def is_vector_db_cached(unified_id: str, index_dir: Path) -> bool:
    return unified_cache_key(unified_id, index_dir) in _UNIFIED_COLLECTION_CACHE


def is_manifest_cached(unified_id: str, index_dir: Path) -> bool:
    return is_vector_db_cached(unified_id, index_dir)


def is_metadata_cached(unified_id: str, index_dir: Path) -> bool:
    return is_vector_db_cached(unified_id, index_dir)


def cache_status_snapshot(unified_id: str, index_dir: Path, embed_model: str) -> dict[str, bool]:
    return {
        "embedding_model_loaded_from_cache": embedding_model_is_cached(embed_model),
        "vector_db_loaded_from_cache": is_vector_db_cached(unified_id, index_dir),
        "metadata_loaded_from_cache": is_metadata_cached(unified_id, index_dir),
        "manifest_loaded_from_cache": is_manifest_cached(unified_id, index_dir),
    }


def apply_cache_flags_to_timing(timing, unified_id: str, index_dir: Path, embed_model: str) -> None:
    if timing is None or not hasattr(timing, "set_cache"):
        return
    for key, value in cache_status_snapshot(unified_id, index_dir, embed_model).items():
        timing.set_cache(key, value)


def load_unified_collection(
    unified_id: str,
    index_dir: Path,
    timing=None,
) -> tuple[Any, str, dict]:
    """Load Chroma collection + manifest; reuse within process."""
    key = unified_cache_key(unified_id, index_dir)
    fingerprint = unified_index_fingerprint(unified_id, index_dir)
    cached = _UNIFIED_COLLECTION_CACHE.get(key)

    if timing is not None:
        apply_cache_flags_to_timing(timing, unified_id, index_dir, "")

    if cached and cached[0] == fingerprint:
        _, collection, embed_model, manifest = cached
        if timing is not None:
            apply_cache_flags_to_timing(timing, unified_id, index_dir, embed_model)
        return collection, embed_model, manifest

    if cached:
        _UNIFIED_COLLECTION_CACHE.pop(key, None)

    from imo_doc_registry import clear_corpus_rows_cache

    clear_corpus_rows_cache()

    import chromadb
    from chromadb.config import Settings

    root = index_dir / f"unified_{unified_id}"
    manifest = json.loads((root / "index_manifest.json").read_text(encoding="utf-8"))
    chroma_path = _resolve_chroma_path(root, manifest)
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(manifest["collection_name"])
    preset = manifest.get("embedding_preset", DEFAULT_EMBEDDING_PRESET)
    embed_config = resolve_embedding_config(preset, str(manifest.get("embedding_model", "")))
    embed_model = str(embed_config["model"])
    _UNIFIED_COLLECTION_CACHE[key] = (fingerprint, collection, embed_model, manifest)
    if timing is not None:
        apply_cache_flags_to_timing(timing, unified_id, index_dir, embed_model)
    return collection, embed_model, manifest


def warm_all_resources(
    unified_id: str = "full_corpus_715_v1",
    index_dir: Path | None = None,
) -> tuple[Any, str, dict]:
    """Eager-load vector DB + embedding model (benchmark warm-start)."""
    index_dir = index_dir or Path("data/processed/index")
    collection, embed_model, manifest = load_unified_collection(unified_id, index_dir)
    warm_embed_model(embed_model)
    return collection, embed_model, manifest


def prime_interactive_retrieval(collection, embed_model: str) -> dict[str, float | bool]:
    """Run one tiny encode/query so the first UI click is not a cold vector search."""
    import time

    from embedding_policy import embed_texts_local

    started = time.perf_counter()
    try:
        vector = embed_texts_local(
            ["maritime safety regulation guidance"],
            embed_model,
            for_query=True,
        )[0]
        encoded = time.perf_counter()
        collection.query(query_embeddings=[vector], n_results=3)
        queried = time.perf_counter()
        return {
            "retrieval_primed": True,
            "prime_embedding_time": round(encoded - started, 4),
            "prime_vector_time": round(queried - encoded, 4),
        }
    except Exception:
        return {
            "retrieval_primed": False,
            "prime_embedding_time": 0.0,
            "prime_vector_time": 0.0,
        }


def clear_process_caches() -> None:
    """Drop in-process caches (cold-start benchmark)."""
    from bm25_index import clear_bm25_caches
    from embedding_policy import clear_encoder_cache

    _UNIFIED_COLLECTION_CACHE.clear()
    clear_bm25_caches()
    clear_encoder_cache()
    load_json_file.cache_clear()
    load_questions_file.cache_clear()
    load_reference_file.cache_clear()


@lru_cache(maxsize=64)
def load_json_file(path_str: str) -> Any:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def load_questions_file(path_str: str) -> tuple[dict, ...]:
    from rag_eval_lib import load_questions

    return tuple(load_questions(Path(path_str)))


@lru_cache(maxsize=8)
def load_reference_file(path_str: str) -> dict[str, dict]:
    from rag_answer_lib import load_reference_answers

    return load_reference_answers(Path(path_str))
