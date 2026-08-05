"""Bridge to MaritimeRAG (Chroma / document QA)."""
from __future__ import annotations

import os
import sys
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

# Process-level handles so Gradio does not cold-load Chroma/e5 on every question.
_WARM: dict[str, Any] = {
    "collection": None,
    "embed_model": None,
    "manifest": None,
    "unified_id": None,
}


def _ensure_rag_path() -> None:
    if str(RAG_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(RAG_SCRIPTS_DIR))


def _unified_root(collection_id: str) -> Path:
    # MaritimeRAG stores unified indexes under index/unified_<collection_id>/
    return RAG_INDEX_DIR / f"unified_{collection_id}"


def rag_index_ready(collection_id: str = DEFAULT_RAG_COLLECTION) -> bool:
    root = _unified_root(collection_id)
    return (root / "chroma").exists() or (root / "index_manifest.json").exists()


def rag_status_message() -> str:
    if rag_index_ready():
        return f"RAG 인덱스 준비됨: {DEFAULT_RAG_COLLECTION}"
    return (
        "문서 RAG 인덱스가 아직 없습니다. PDF는 data/raw_pdfs 에 연결되어 있습니다.\n"
        "인덱스를 만들려면 MaritimeRAG 전처리·통합 인덱스 구축을 실행하세요 "
        f"(목표 collection: {DEFAULT_RAG_COLLECTION})."
    )


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
        # Ensure absolute index path is used for subsequent loads
        collection, embed_model, manifest = load_unified_collection(
            unified_id, RAG_INDEX_DIR
        )
        prime = prime_interactive_retrieval(collection, embed_model)
        try:
            from ollama_warmup import ensure_fast_warm  # type: ignore
            from rag_answer_lib import DEFAULT_OLLAMA_BASE, DEFAULT_OLLAMA_MODEL  # type: ignore

            ensure_fast_warm(DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_BASE)
            ollama_warm = True
        except Exception:
            ollama_warm = False

        _WARM["collection"] = collection
        _WARM["embed_model"] = embed_model
        _WARM["manifest"] = manifest
        _WARM["unified_id"] = unified_id
        return {
            "ok": True,
            "unified_id": unified_id,
            "prime": prime,
            "ollama_warm": ollama_warm,
        }
    finally:
        os.chdir(prev)


def _classify_category(question: str) -> str:
    """Match MaritimeRAG UI behavior: set category before retrieval."""
    try:
        _ensure_rag_path()
        from question_classifier import classify_question_category  # type: ignore

        return str(classify_question_category(question, {}) or "")
    except Exception:
        return ""


def run_rag_query(
    question: str,
    *,
    latency_mode: str = "fast",
    table_qa: bool = False,
) -> dict[str, Any]:
    """Run MaritimeRAG in-process against the shared data/ tree."""
    target = DEFAULT_TABLE_COLLECTION if table_qa else DEFAULT_RAG_COLLECTION
    if not rag_index_ready(target if table_qa else DEFAULT_RAG_COLLECTION):
        return {
            "answer": rag_status_message(),
            "files": [],
            "source": "rag",
            "citations": [],
            "ready": False,
        }

    _ensure_rag_path()
    prev = os.getcwd()
    try:
        os.chdir(RAG_DIR)
        from rag_inprocess import (  # type: ignore
            TABLE_QA_UNIFIED,
            normalize_table_question_row,
            run_full_inprocess,
        )

        category = "" if table_qa else _classify_category(question)
        row: dict[str, Any] = {
            "question_id": "ui_q",
            "question": question,
            "category": category,
        }
        unified = DEFAULT_RAG_COLLECTION
        chunks_dir = RAG_CHUNKS_DIR
        if table_qa:
            row = normalize_table_question_row(row)
            unified = TABLE_QA_UNIFIED or DEFAULT_TABLE_COLLECTION
            chunks_dir = RAG_TABLE_CHUNKS_DIR

        # Reuse warm handles when available (same collection).
        collection = None
        embed_model = None
        manifest = None
        if (
            not table_qa
            and _WARM.get("unified_id") == unified
            and _WARM.get("collection") is not None
        ):
            collection = _WARM["collection"]
            embed_model = _WARM["embed_model"]
            manifest = _WARM["manifest"]

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
            answer = search.get("answer") or "검색은 완료되었으나 답변 텍스트를 찾지 못했습니다."

        # Keep warm after first successful load
        if not table_qa and _WARM.get("collection") is None:
            try:
                from rag_resource_cache import load_unified_collection  # type: ignore

                c, e, m = load_unified_collection(unified, RAG_INDEX_DIR)
                _WARM["collection"] = c
                _WARM["embed_model"] = e
                _WARM["manifest"] = m
                _WARM["unified_id"] = unified
            except Exception:
                pass

        return {
            "answer": answer,
            "files": [],
            "source": "rag",
            "ready": True,
            "meta": {
                "unified_id": unified,
                "latency_mode": latency_mode,
                "category": category,
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
            "source": "rag",
            "ready": False,
            "error": str(exc),
        }
    finally:
        os.chdir(prev)
