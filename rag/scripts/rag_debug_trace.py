"""Structured RAG debug trace for terminal + Streamlit."""
from __future__ import annotations

import json
import os
import sys
from typing import Any

from rag_query_router import resolve_pipeline_route


def _preview(text: str, n: int) -> str:
    t = (text or "").replace("\n", " ").strip()
    return t[:n] + ("…" if len(t) > n else "")


def _chunk_row(c: Any, rank: int, score: float | None = None) -> dict[str, Any]:
    return {
        "rank": rank,
        "score": round(score, 4) if score is not None else getattr(c, "distance", None),
        "document_title": getattr(c, "file_name", "") or getattr(c, "doc_id", ""),
        "society": getattr(c, "source", ""),
        "doc_type": getattr(c, "element_type", "") or getattr(c, "chunk_type", ""),
        "page": getattr(c, "page_number", None),
        "clause_no": getattr(c, "clause_number", "") or "",
        "chunk_type": getattr(c, "chunk_type", ""),
        "first_150_chars": _preview(getattr(c, "text", ""), 150),
    }


def build_latency_breakdown_ms(timing_metrics: dict[str, Any] | None, route_ms: float = 0.0) -> dict[str, Any]:
    m = timing_metrics or {}
    embed = float(m.get("query_embedding_time") or 0) * 1000
    vector = float(m.get("vector_search_time") or 0) * 1000
    meta_f = float(m.get("metadata_filter_time") or 0) * 1000
    rerank = float(m.get("rerank_time") or 0) * 1000
    ctx = float(m.get("context_build_time") or 0) * 1000
    llm_gen = float(m.get("llm_generation_time") or m.get("llm_ttft") or 0) * 1000
    pre_llm = float(m.get("pre_llm_latency") or 0) * 1000
    acc_llm_vis = m.get("accurate_first_visible_llm_latency")
    if acc_llm_vis is not None:
        total = float(acc_llm_vis) * 1000
    else:
        total = float(m.get("end_to_end_ttft") or m.get("end_to_end_total") or 0) * 1000
    if not total and m.get("answer_first_visible_latency"):
        total = float(m["answer_first_visible_latency"]) * 1000
    if not total and pre_llm and llm_gen:
        total = pre_llm + llm_gen
    post = max(0.0, total - (route_ms + embed + vector + meta_f + rerank + ctx + llm_gen))
    pass_3s = total <= 3000.0 if total else None
    return {
        "routing_ms": round(route_ms, 1),
        "query_expansion_ms": round(embed * 0.05, 1),
        "vector_search_ms": round(vector + embed, 1),
        "keyword_search_ms": round(meta_f, 1),
        "rerank_ms": round(rerank, 1),
        "context_build_ms": round(ctx, 1),
        "llm_generate_ms": round(llm_gen, 1),
        "postprocess_ms": round(post, 1),
        "total_ms": round(total, 1),
        "pass_3s": pass_3s,
    }


def build_rag_debug_trace(
    *,
    run_id: str,
    row: dict,
    route: dict[str, Any] | None,
    pool_before: list[Any] | None,
    pool_after: list[Any] | None,
    final_chunks: list[Any] | None,
    retrieval_params: dict[str, Any] | None,
    llm_params: dict[str, Any] | None,
    timing_metrics: dict[str, Any] | None,
    route_ms: float = 0.0,
    where_filter: dict[str, Any] | str | None = None,
    answer_generation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    question = str(row.get("question") or "")
    route = route or resolve_pipeline_route(question, row)
    pool_before = pool_before or []
    pool_after = pool_after or pool_before
    final_chunks = final_chunks or []
    rp = retrieval_params or {}
    lp = llm_params or {}

    trace = {
        "ROUTING": {
            "run_id": run_id,
            "user_question": question,
            "latency_mode": route.get("latency_mode") or row.get("_latency_mode"),
            "top_level": route.get("top_level_label"),
            "internal_intent": route.get("internal_intent"),
            "selected_retrieval_profile": route.get("selected_retrieval_profile"),
            "selected_answer_mode": route.get("selected_answer_mode"),
        },
        "QUERY_CONSTRAINTS": {
            "detected_society": route.get("detected_society"),
            "detected_doc_type": route.get("detected_doc_type"),
            "hard_filter": route.get("hard_society_filter"),
            "expanded_keywords": route.get("expanded_keywords"),
        },
        "RETRIEVAL_PARAMS": {
            "corpus": rp.get("corpus", "full_corpus"),
            "top_k": rp.get("top_k"),
            "fetch_k": rp.get("fetch_k"),
            "rerank_on/off": rp.get("rerank"),
            "max_docs": rp.get("max_docs"),
            "max_chunks_per_doc": rp.get("max_chunks_per_doc"),
            "actual_where_filter_sent_to_db": where_filter or rp.get("where_filter"),
        },
        "CANDIDATES_BEFORE_FILTER": [
            _chunk_row(c, i + 1, getattr(c, "distance", None)) for i, c in enumerate(pool_before[:15])
        ],
        "CANDIDATES_AFTER_FILTER": [
            _chunk_row(c, i + 1, getattr(c, "distance", None)) for i, c in enumerate(pool_after[:15])
        ],
        "FINAL_CONTEXT_TO_LLM": {
            "number_of_chunks": len(final_chunks),
            "total_context_chars": sum(len(getattr(c, "text", "") or "") for c in final_chunks),
            "chunks": [
                {
                    "document_title": getattr(c, "file_name", ""),
                    "society": getattr(c, "source", ""),
                    "page": getattr(c, "page_number"),
                    "clause_no": getattr(c, "clause_number", ""),
                    "first_200_chars": _preview(getattr(c, "text", ""), 200),
                }
                for c in final_chunks
            ],
        },
        "LLM_PARAMS": lp,
        "ANSWER_GENERATION": answer_generation or row.get("_answer_generation") or {},
        "RULE_GUIDANCE_TTFT": {
            "pre_llm_latency": (timing_metrics or {}).get("pre_llm_latency"),
            "llm_ttft": (timing_metrics or {}).get("llm_ttft"),
            "first_token_latency": (answer_generation or row.get("_answer_generation") or {}).get(
                "rule_guidance_first_token_latency"
            ),
            "pass_3s": (answer_generation or row.get("_answer_generation") or {}).get(
                "rule_guidance_first_token_3s_pass"
            ),
        },
        "LATENCY_BREAKDOWN": build_latency_breakdown_ms(timing_metrics, route_ms=route_ms),
    }
    return trace


def format_debug_trace_text(trace: dict[str, Any]) -> str:
    lines: list[str] = []
    for section, payload in trace.items():
        lines.append(f"[{section}]")
        if isinstance(payload, dict):
            for k, v in payload.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    lines.append(f"  {k}:")
                    for item in v[:10]:
                        lines.append(f"    - {json.dumps(item, ensure_ascii=False)}")
                else:
                    lines.append(f"  {k}: {v}")
        else:
            lines.append(f"  {payload}")
        lines.append("")
    return "\n".join(lines)


def log_debug_trace(trace: dict[str, Any], *, run_id: str = "") -> None:
    if os.getenv("RAG_DEBUG_TRACE_STDERR", "1").strip().lower() in {"0", "false", "no", "off"}:
        return
    text = format_debug_trace_text(trace)
    banner = f"=== RAG DEBUG TRACE run_id={run_id or trace.get('ROUTING', {}).get('run_id', '')} ==="
    sys.stderr.write(banner + "\n" + text + "\n")
    sys.stderr.flush()
