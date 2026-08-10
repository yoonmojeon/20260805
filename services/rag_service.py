"""Bridge to MaritimeRAG (Chroma / document QA)."""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from project_paths import (
    DEFAULT_RAG_COLLECTION,
    DEFAULT_TABLE_COLLECTION,
    RAG_CHUNKS_DIR,
    RAG_DIR,
    RAG_INDEX_DIR,
    RAG_SCRIPTS_DIR,
    RAG_TABLE_CHUNKS_DIR,
)
from services.rag_index_diag import (
    diagnose_rag_indexes,
    format_rag_index_banner,
    index_ready as _index_ready,
)
from services.retrieval_mode import RetrievalMode, classify_retrieval_mode
from services.table_render import (
    extract_table_ids_from_chunks,
    format_related_tables_section,
)

# Process-level handles so Gradio does not cold-load Chroma/e5 on every question.
_WARM: dict[str, Any] = {
    "collection": None,
    "embed_model": None,
    "manifest": None,
    "unified_id": None,
}
_TABLE_WARM: dict[str, Any] = {
    "collection": None,
    "embed_model": None,
    "manifest": None,
    "unified_id": None,
}


def _ensure_rag_path() -> None:
    if str(RAG_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(RAG_SCRIPTS_DIR))


def _unified_root(collection_id: str) -> Path:
    return RAG_INDEX_DIR / f"unified_{collection_id}"


def rag_index_ready(collection_id: str = DEFAULT_RAG_COLLECTION) -> bool:
    return _index_ready(collection_id)


def rag_status_message() -> str:
    if not rag_index_ready(DEFAULT_RAG_COLLECTION):
        return (
            "문서 RAG 인덱스가 아직 없습니다. PDF는 data/raw_pdfs 에 연결되어 있습니다.\n"
            "인덱스를 만들려면 MaritimeRAG 전처리·통합 인덱스 구축을 실행하세요 "
            f"(목표 collection: {DEFAULT_RAG_COLLECTION})."
        )
    table = (
        f"표QA({DEFAULT_TABLE_COLLECTION}): OK"
        if rag_index_ready(DEFAULT_TABLE_COLLECTION)
        else f"표QA({DEFAULT_TABLE_COLLECTION}): 없음 → 본문 인덱스만 사용"
    )
    return f"RAG 인덱스 준비됨: {DEFAULT_RAG_COLLECTION}  |  {table}"


def rag_index_diagnostics(*, sample_size: int = 2000) -> dict[str, Any]:
    """Startup/ops helper: live collection.count() + sample metadata aggregates."""
    return diagnose_rag_indexes(sample_size=sample_size)


def rag_index_banner(*, sample_size: int = 2000) -> str:
    return format_rag_index_banner(diagnose_rag_indexes(sample_size=sample_size))


def warmup_rag_resources(unified_id: str = DEFAULT_RAG_COLLECTION) -> dict[str, Any]:
    """Eager-load Chroma + e5 + optional Ollama warm (Streamlit parity)."""
    if not rag_index_ready(unified_id):
        return {"ok": False, "reason": "index_missing"}

    _ensure_rag_path()
    prev = os.getcwd()
    try:
        os.chdir(RAG_DIR)
        from rag_resource_cache import (  # type: ignore
            load_unified_collection,
            prime_interactive_retrieval,
            warm_all_resources,
        )

        collection, embed_model, manifest = warm_all_resources(
            unified_id=unified_id,
            index_dir=RAG_INDEX_DIR,
        )
        collection, embed_model, manifest = load_unified_collection(
            unified_id, RAG_INDEX_DIR
        )
        prime = prime_interactive_retrieval(collection, embed_model)
        bm25_warm = False
        if unified_id == DEFAULT_TABLE_COLLECTION:
            try:
                from bm25_index import load_or_build_table_bm25  # type: ignore
                from rag_resource_cache import unified_index_fingerprint  # type: ignore

                fp = unified_index_fingerprint(unified_id, RAG_INDEX_DIR)
                bm25_warm = (
                    load_or_build_table_bm25(
                        collection,
                        unified_id=unified_id,
                        index_dir=RAG_INDEX_DIR,
                        fingerprint=fp,
                        allow_disk_load=True,
                    )
                    is not None
                )
            except Exception:
                bm25_warm = False
        try:
            from ollama_warmup import ensure_fast_warm  # type: ignore
            from rag_answer_lib import DEFAULT_OLLAMA_BASE, DEFAULT_OLLAMA_MODEL  # type: ignore

            ensure_fast_warm(DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_BASE)
            ollama_warm = True
        except Exception:
            ollama_warm = False

        bucket = _TABLE_WARM if unified_id == DEFAULT_TABLE_COLLECTION else _WARM
        bucket["collection"] = collection
        bucket["embed_model"] = embed_model
        bucket["manifest"] = manifest
        bucket["unified_id"] = unified_id
        return {
            "ok": True,
            "unified_id": unified_id,
            "prime": prime,
            "ollama_warm": ollama_warm,
            "bm25_warm": bm25_warm,
        }
    finally:
        os.chdir(prev)


def _classify_category(question: str) -> str:
    try:
        _ensure_rag_path()
        from question_classifier import classify_question_category  # type: ignore

        return str(classify_question_category(question, {}) or "")
    except Exception:
        return ""


def _build_rag_row(question: str, *, table_qa: bool) -> dict[str, Any]:
    category = "table_qa" if table_qa else _classify_category(question)
    row: dict[str, Any] = {
        "question_id": "ui_q",
        "question": question,
        "category": category,
    }
    if table_qa:
        row["_table_qa"] = True
        try:
            _ensure_rag_path()
            from rag_inprocess import normalize_table_question_row  # type: ignore

            row = normalize_table_question_row(row)
        except Exception:
            pass
    return row


def _resolve_mode(
    question: str,
    *,
    retrieval_mode: RetrievalMode | str | None,
    table_qa: bool | None,
) -> RetrievalMode:
    if retrieval_mode is not None:
        if isinstance(retrieval_mode, RetrievalMode):
            return retrieval_mode
        return RetrievalMode(str(retrieval_mode).strip().lower())
    if table_qa is True:
        return RetrievalMode.TABLE
    if table_qa is False:
        # Explicit legacy false → text only
        return RetrievalMode.TEXT
    return classify_retrieval_mode(question)


def _index_for(mode: RetrievalMode) -> tuple[str, Any]:
    _ensure_rag_path()
    from project_paths import DEFAULT_TABLE_COLLECTION as table_id  # local alias

    if mode in {RetrievalMode.TABLE, RetrievalMode.BOTH} and rag_index_ready(table_id):
        # Keep in sync with project_paths (single source of truth)
        return table_id, RAG_TABLE_CHUNKS_DIR
    return DEFAULT_RAG_COLLECTION, RAG_CHUNKS_DIR


def _warm_handles(unified: str, *, table_side: bool):
    bucket = _TABLE_WARM if table_side else _WARM
    if bucket.get("unified_id") == unified and bucket.get("collection") is not None:
        return bucket["collection"], bucket["embed_model"], bucket["manifest"]
    return None, None, None


def _extract_answer(out: dict[str, Any]) -> str:
    answer = (
        out.get("answer")
        or out.get("final_answer")
        or (out.get("answer_out") or {}).get("answer")
        or ""
    )
    if not answer and isinstance(out.get("answer_out"), dict):
        answer = out["answer_out"].get("text") or out["answer_out"].get("content") or ""
    if not answer:
        search = out.get("search_out") or {}
        answer = search.get("answer") or ""
    return str(answer or "").strip()


def _tag_chunks(chunks: list[Any], *, source_kind: str) -> list[Any]:
    for chunk in chunks:
        try:
            setattr(chunk, "evidence_source", source_kind)
        except Exception:
            pass
        meta = getattr(chunk, "metadata", None)
        if isinstance(meta, dict):
            meta.setdefault("evidence_source", source_kind)
    return chunks


def _dedupe_chunks(chunks: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for chunk in chunks:
        cid = str(getattr(chunk, "chunk_id", "") or id(chunk))
        key = f"{getattr(chunk, 'evidence_source', '')}:{cid}"
        if key in seen:
            continue
        seen.add(key)
        out.append(chunk)
    return out


def _page_key(chunk: Any) -> str:
    doc = str(getattr(chunk, "doc_id", "") or getattr(chunk, "file_name", "") or "")
    page = str(getattr(chunk, "page_number", "") or getattr(chunk, "page", "") or "")
    return f"{doc}::{page}"


def fuse_evidence(
    *,
    text_hits: list[Any],
    table_hits: list[Any],
    text_top_k: int = 6,
    table_top_k: int = 8,
    prefer: str = "balanced",
) -> list[Any]:
    """Merge text/table hits with separate caps and light page dedupe."""
    text_hits = _tag_chunks(list(text_hits or []), source_kind="text")[:text_top_k]
    table_hits = _tag_chunks(list(table_hits or []), source_kind="table")[:table_top_k]

    if prefer == "table_primary":
        ordered = table_hits + text_hits
    elif prefer == "text_primary":
        ordered = text_hits + table_hits
    else:
        # Interleave a little so BOTH keeps both sides visible early
        ordered = []
        for i in range(max(len(table_hits), len(text_hits))):
            if i < len(table_hits):
                ordered.append(table_hits[i])
            if i < len(text_hits):
                ordered.append(text_hits[i])

    # Drop near-duplicate same doc/page text chunks (keep first)
    seen_pages: set[str] = set()
    fused: list[Any] = []
    for chunk in ordered:
        src = getattr(chunk, "evidence_source", "")
        pk = _page_key(chunk)
        if src == "text" and pk in seen_pages and pk != "::":
            continue
        if src == "text" and pk != "::":
            seen_pages.add(pk)
        fused.append(chunk)
    return _dedupe_chunks(fused)


def _chunk_meta_dict(chunk: Any) -> dict[str, Any]:
    if isinstance(chunk, dict):
        return dict(chunk)
    meta = getattr(chunk, "metadata", None)
    out: dict[str, Any] = dict(meta) if isinstance(meta, dict) else {}
    for key in (
        "table_id",
        "file_name",
        "doc_id",
        "page_number",
        "caption",
        "crop_path",
        "chunk_type",
        "element_id",
        "chunk_id",
        "text",
    ):
        val = getattr(chunk, key, None)
        if val not in (None, ""):
            out[key] = val
    if "text" not in out or not out.get("text"):
        out["text"] = str(getattr(chunk, "text", "") or "")
    return out


def _resolve_crop_path(meta: dict[str, Any], table_id: str) -> str:
    crop = str(meta.get("crop_path") or "")
    if crop and Path(crop).is_file():
        return crop
    # Precise corpus layout: .../precise_tables/{year}/p####_t###/crop.png
    import re

    match = re.search(r"(20\d{2}).*?(p\d+_t\d+)", table_id or "", re.I)
    if not match:
        return ""
    root = Path(__file__).resolve().parents[1] / "data" / "processed" / "precise_tables"
    candidate = root / match.group(1) / match.group(2) / "crop.png"
    return str(candidate) if candidate.is_file() else ""


def _related_tables_from_hits(
    question: str, table_chunks: list[Any]
) -> tuple[str, list[str], list[dict[str, Any]]]:
    """Collect unique table crops from retrieved hits (MaritimeRAG-style).

    Prefer ``crop_path`` images over Markdown rebuild. No second Chroma scan.
    """
    del question  # highlight reserved for optional future use
    empty: tuple[str, list[str], list[dict[str, Any]]] = ("", [], [])
    if not table_chunks:
        return empty

    table_ids = extract_table_ids_from_chunks(table_chunks, limit=2)
    if not table_ids:
        return empty

    rendered: list[dict[str, Any]] = []
    images: list[str] = []
    for tid in table_ids:
        meta0: dict[str, Any] = {}
        crop = ""
        for chunk in table_chunks:
            meta = _chunk_meta_dict(chunk)
            chunk_tid = str(meta.get("table_id") or "")
            if not chunk_tid:
                # Recover from chunk_id / text when metadata was stripped.
                recovered = extract_table_ids_from_chunks([chunk], limit=1)
                chunk_tid = recovered[0] if recovered else ""
            if chunk_tid and chunk_tid != tid:
                continue
            if not meta0:
                meta0 = meta
            candidate = _resolve_crop_path(meta, tid)
            if candidate:
                crop = candidate
                meta0 = meta
                break
        if not crop:
            crop = _resolve_crop_path(meta0, tid)
        if not crop:
            continue
        if crop not in images:
            images.append(crop)
        rendered.append(
            {
                "table_id": tid,
                "file_name": meta0.get("file_name"),
                "doc_id": meta0.get("doc_id"),
                "page": meta0.get("page_number"),
                "caption": meta0.get("caption"),
                "markdown": "",
                "crop_path": crop,
                "highlight_rows": [],
                "source": "retrieved_crop",
            }
        )
    section = format_related_tables_section(rendered) if rendered else ""
    return section, images, rendered


def _extract_evidence_table(out: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Pull Evidence Table rows from run_full / run_answer payloads."""
    if not isinstance(out, dict):
        return []
    for key in ("evidence_table",):
        rows = out.get(key)
        if isinstance(rows, list) and rows:
            return list(rows)
    answer_out = out.get("answer_out")
    if isinstance(answer_out, dict):
        rows = answer_out.get("evidence_table")
        if isinstance(rows, list) and rows:
            return list(rows)
    return []


def _run_single_rag(
    question: str,
    *,
    latency_mode: str,
    mode: RetrievalMode,
) -> dict[str, Any]:
    want_table = mode == RetrievalMode.TABLE
    use_table_index = want_table and rag_index_ready(DEFAULT_TABLE_COLLECTION)
    target = DEFAULT_TABLE_COLLECTION if use_table_index else DEFAULT_RAG_COLLECTION
    if not rag_index_ready(target):
        return {
            "answer": (
                f"요청한 인덱스(`{target}`)가 없습니다. {rag_status_message()}"
            ),
            "files": [],
            "evidence_table": [],
            "related_tables": [],
            "source": "rag",
            "citations": [],
            "ready": False,
            "meta": {"missing_index": target, "retrieval_mode": mode.value},
        }

    _ensure_rag_path()
    prev = os.getcwd()
    try:
        os.chdir(RAG_DIR)
        from rag_inprocess import run_full_inprocess  # type: ignore

        row = _build_rag_row(question, table_qa=want_table)
        unified, chunks_dir = _index_for(mode if want_table else RetrievalMode.TEXT)
        collection, embed_model, manifest = _warm_handles(
            unified, table_side=use_table_index
        )

        out = run_full_inprocess(
            row,
            unified_id=unified,
            index_dir=RAG_INDEX_DIR,
            chunks_dir=chunks_dir,
            latency_mode=latency_mode,
            skip_llm=False,
            collection=collection,
            embed_model=embed_model,
            manifest=manifest,
            auto_llm_warm=True,
        )
        answer = _extract_answer(out) or "검색은 완료되었으나 답변 텍스트를 찾지 못했습니다."
        files: list[str] = []
        related_tables: list[dict[str, Any]] = []
        evidence_table = _extract_evidence_table(out)
        if use_table_index:
            search = out.get("search_out") or {}
            retrieved = list(search.get("retrieved") or [])
            pool = list(search.get("retrieval_pool") or [])
            # Prefer the fuller pool so crop_path survives Fast slot trimming.
            crop_hits = pool or retrieved
            _related_md, images, related_tables = _related_tables_from_hits(
                question, crop_hits
            )
            files.extend(images)

        bucket = _TABLE_WARM if use_table_index else _WARM
        if bucket.get("collection") is None:
            try:
                from rag_resource_cache import load_unified_collection  # type: ignore

                c, e, m = load_unified_collection(unified, RAG_INDEX_DIR)
                bucket["collection"] = c
                bucket["embed_model"] = e
                bucket["manifest"] = m
                bucket["unified_id"] = unified
            except Exception:
                pass

        return {
            "answer": answer,
            "files": files,
            "evidence_table": evidence_table,
            "related_tables": related_tables,
            "source": "rag",
            "ready": True,
            "meta": {
                "unified_id": unified,
                "latency_mode": latency_mode,
                "category": row.get("category"),
                "retrieval_mode": mode.value,
                "table_qa_requested": want_table,
                "table_index_used": use_table_index,
                "related_tables_appended": bool(related_tables),
                "evidence_rows": len(evidence_table),
                "answer_mode": out.get("answer_mode")
                or (out.get("search_out") or {}).get("answer_mode"),
                "timing_metrics": (out.get("timing_metrics") or {}),
            },
            "raw": {k: out[k] for k in out if k not in {"raw", "timing_log"}},
        }
    except Exception as exc:
        return {
            "answer": f"[문서 RAG 오류] {exc}",
            "files": [],
            "evidence_table": [],
            "related_tables": [],
            "source": "rag",
            "ready": False,
            "error": str(exc),
            "meta": {"retrieval_mode": mode.value},
        }
    finally:
        os.chdir(prev)


def _run_search_only(
    question: str,
    *,
    latency_mode: str,
    table_side: bool,
) -> dict[str, Any]:
    _ensure_rag_path()
    prev = os.getcwd()
    try:
        os.chdir(RAG_DIR)
        from rag_inprocess import run_full_inprocess  # type: ignore

        row = _build_rag_row(question, table_qa=table_side)
        mode = RetrievalMode.TABLE if table_side else RetrievalMode.TEXT
        unified, chunks_dir = _index_for(mode)
        collection, embed_model, manifest = _warm_handles(
            unified, table_side=table_side
        )
        out = run_full_inprocess(
            row,
            unified_id=unified,
            index_dir=RAG_INDEX_DIR,
            chunks_dir=chunks_dir,
            latency_mode=latency_mode,
            skip_llm=True,
            collection=collection,
            embed_model=embed_model,
            manifest=manifest,
            auto_llm_warm=False,
        )
        search = out.get("search_out") or {}
        return {
            "row": row,
            "unified_id": unified,
            "search_out": search,
            "retrieved": list(search.get("retrieved") or []),
            "timing_metrics": out.get("timing_metrics") or {},
            "evidence_source": "table" if table_side else "text",
        }
    finally:
        os.chdir(prev)


def _run_both_fused(
    question: str,
    *,
    latency_mode: str,
    prefer: str = "balanced",
) -> dict[str, Any]:
    """Search text + table in parallel, fuse evidence, answer once."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_table = pool.submit(
            _run_search_only, question, latency_mode=latency_mode, table_side=True
        )
        fut_text = pool.submit(
            _run_search_only, question, latency_mode=latency_mode, table_side=False
        )
        table_hit = fut_table.result()
        text_hit = fut_text.result()

    table_chunks = list(table_hit.get("retrieved") or [])
    text_chunks = list(text_hit.get("retrieved") or [])
    merged_chunks = fuse_evidence(
        text_hits=text_chunks,
        table_hits=table_chunks,
        text_top_k=6,
        table_top_k=8,
        prefer=prefer,
    )

    if not merged_chunks:
        return {
            "answer": "표·문서 인덱스에서 관련 근거를 찾지 못했습니다.",
            "files": [],
            "evidence_table": [],
            "related_tables": [],
            "source": "rag",
            "ready": True,
            "meta": {
                "dual_retrieval": True,
                "retrieval_mode": RetrievalMode.BOTH.value,
                "dual_mode": "search2_llm1",
                "n_chunks": 0,
            },
        }

    _ensure_rag_path()
    prev = os.getcwd()
    try:
        os.chdir(RAG_DIR)
        from rag_inprocess import run_answer_inprocess  # type: ignore

        use_table_answer = prefer in {"balanced", "table_primary"} and bool(table_chunks)
        row = (
            (table_hit.get("row") if use_table_answer else text_hit.get("row"))
            or _build_rag_row(question, table_qa=use_table_answer)
        )
        base_search = table_hit.get("search_out") if use_table_answer else text_hit.get("search_out")
        base_search = base_search or {}
        answer_out = run_answer_inprocess(
            row=row,
            chunks=merged_chunks,
            pool=merged_chunks,
            config_dict=base_search.get("retrieval_config"),
            metrics=base_search.get("retrieval_metrics"),
            doc_groups=base_search.get("doc_groups"),
            answer_mode=base_search.get("answer_mode")
            or ("table_qa" if use_table_answer else "rag"),
            question_category=("table_qa" if use_table_answer else row.get("category")),
            latency_mode=latency_mode,
            auto_llm_warm=True,
        )
        answer = _extract_answer(answer_out) or "검색은 완료되었으나 답변 텍스트를 찾지 못했습니다."
        _related_md, images, related_tables = _related_tables_from_hits(
            question, table_chunks
        )
        evidence_table = _extract_evidence_table(answer_out)
        return {
            "answer": answer,
            "files": images,
            "evidence_table": evidence_table,
            "related_tables": related_tables,
            "source": "rag",
            "ready": True,
            "meta": {
                "dual_retrieval": True,
                "dual_mode": "search2_llm1",
                "dual_retrieval_enabled": True,
                "retrieval_mode": RetrievalMode.BOTH.value,
                "fuse_policy": prefer,
                "n_table_chunks": len(table_chunks),
                "n_text_chunks": len(text_chunks),
                "n_merged_chunks": len(merged_chunks),
                "text_collection": DEFAULT_RAG_COLLECTION,
                "table_collection": DEFAULT_TABLE_COLLECTION,
                "latency_mode": latency_mode,
                "table_index_used": True,
                "related_tables_appended": bool(related_tables),
                "evidence_rows": len(evidence_table),
                "debug": {
                    "mode": "BOTH",
                    "text_hits": len(text_chunks),
                    "table_hits": len(table_chunks),
                    "fused_hits": len(merged_chunks),
                },
                "timing_metrics": {
                    "table_search": table_hit.get("timing_metrics") or {},
                    "text_search": text_hit.get("timing_metrics") or {},
                    "answer": (answer_out.get("timing_metrics") or {}),
                },
            },
            "raw": {"answer_out": answer_out, "table_search": table_hit.get("search_out")},
        }
    except Exception as exc:
        return {
            "answer": f"[문서 RAG 오류] {exc}",
            "files": [],
            "evidence_table": [],
            "related_tables": [],
            "source": "rag",
            "ready": False,
            "error": str(exc),
            "meta": {"retrieval_mode": RetrievalMode.BOTH.value},
        }
    finally:
        os.chdir(prev)


def dual_retrieval_enabled(dual: bool | None = None) -> bool:
    """Return whether BOTH (search2) fusion is allowed.

    Default ON when env unset. Explicit MARITIME_RAG_DUAL=0 disables fusion.
    """
    if dual is True:
        return True
    if dual is False:
        return False
    env = os.environ.get("MARITIME_RAG_DUAL", "1").strip().lower()
    return env not in {"0", "false", "off", "no"}


def run_rag_query(
    question: str,
    *,
    latency_mode: str = "fast",
    table_qa: bool | None = None,
    dual: bool | None = None,
    retrieval_mode: RetrievalMode | str | None = None,
) -> dict[str, Any]:
    """Run MaritimeRAG in-process against the shared data/ tree.

    retrieval_mode: text | table | both
    Legacy: table_qa=True → TABLE (or BOTH when dual env forces).
    BOTH fusion default: ON (MARITIME_RAG_DUAL defaults to "1").
    """
    mode = _resolve_mode(question, retrieval_mode=retrieval_mode, table_qa=table_qa)
    both_ready = rag_index_ready(DEFAULT_RAG_COLLECTION) and rag_index_ready(
        DEFAULT_TABLE_COLLECTION
    )
    dual_on = dual_retrieval_enabled(dual)

    if mode == RetrievalMode.BOTH and both_ready and dual_on:
        return _run_both_fused(question, latency_mode=latency_mode, prefer="balanced")
    if mode == RetrievalMode.TABLE and both_ready and dual is True:
        # Explicit dual=True even for TABLE → still fuse, table-primary
        return _run_both_fused(question, latency_mode=latency_mode, prefer="table_primary")
    if mode == RetrievalMode.BOTH and not both_ready:
        fallback = (
            RetrievalMode.TABLE
            if rag_index_ready(DEFAULT_TABLE_COLLECTION)
            else RetrievalMode.TEXT
        )
        out = _run_single_rag(question, latency_mode=latency_mode, mode=fallback)
        meta = dict(out.get("meta") or {})
        meta["dual_retrieval_enabled"] = False
        meta["dual_fallback_reason"] = "missing_index"
        out["meta"] = meta
        return out
    if mode == RetrievalMode.BOTH and both_ready and not dual_on:
        # Env explicitly disabled dual: fall back to table-primary single index.
        out = _run_single_rag(question, latency_mode=latency_mode, mode=RetrievalMode.TABLE)
        meta = dict(out.get("meta") or {})
        meta["retrieval_mode"] = RetrievalMode.BOTH.value
        meta["dual_retrieval_enabled"] = False
        meta["dual_fallback_reason"] = "MARITIME_RAG_DUAL=0"
        out["meta"] = meta
        return out

    out = _run_single_rag(question, latency_mode=latency_mode, mode=mode)
    meta = dict(out.get("meta") or {})
    meta.setdefault("dual_retrieval_enabled", dual_on)
    out["meta"] = meta
    return out
