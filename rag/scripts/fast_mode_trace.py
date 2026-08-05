"""Fast mode answer trace logging (compare with Accurate)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_TRACE_LOG = Path("data/processed/logs/fast_mode_answer_trace.jsonl")


def append_fast_answer_trace(record: dict[str, Any], path: Path = DEFAULT_TRACE_LOG) -> None:
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_trace_record(
    *,
    row: dict,
    chunks: list,
    pipeline: Any,
    answer: str,
    citation_mapping: list[dict],
    timing_metrics: dict | None = None,
    prompt_meta: dict | None = None,
    latency_mode: str = "fast",
) -> dict[str, Any]:
    pages = sorted({c.page_number for c in chunks if c.page_number is not None})
    titles = list(dict.fromkeys(c.file_name for c in chunks if getattr(c, "file_name", None)))
    tm = timing_metrics or {}
    msc = pipeline.summary_ctx.to_dict() if getattr(pipeline, "summary_ctx", None) else {}
    return {
        "question_id": row.get("question_id"),
        "question": row.get("question"),
        "latency_mode": latency_mode,
        "retrieved_document_titles": titles,
        "retrieved_page_numbers": pages,
        "detected_intent": msc.get("detected_intent") or (pipeline.fast_type if pipeline else None),
        "target_meeting": msc.get("target_meeting"),
        "target_scope": msc.get("target_scope"),
        "selected_primary_doc": getattr(pipeline, "selected_primary_doc", None),
        "rejected_docs_with_reason": getattr(pipeline, "rejected_docs", []),
        "detected_document_scope": pipeline.scope.to_dict() if pipeline else {},
        "extracted_claims": pipeline.all_claims if pipeline else [],
        "filtered_claims": pipeline.filtered_claims if pipeline else [],
        "selected_claims": pipeline.selected_claims if pipeline else [],
        "validation_result": getattr(pipeline, "validation_result", {}),
        "fallback_used": getattr(pipeline, "fallback_used", False),
        "final_answer": answer,
        "citation_mapping": citation_mapping,
        "terminology_violations": __import__("fast_imo_terms", fromlist=["detect_terminology_violations"]).detect_terminology_violations(answer),
        "first_visible_latency": tm.get("e2e_ttft") or tm.get("llm_ttft"),
        "ttft": tm.get("llm_ttft"),
        "total_generation_time": tm.get("llm_generation_time") or tm.get("e2e_total"),
        "e2e_total": tm.get("e2e_total"),
        "prompt_meta": {
            k: prompt_meta.get(k)
            for k in (
                "input_token_estimate",
                "final_prompt_chars",
                "selected_chunk_count",
                "fast_question_type",
                "fast_confidence",
            )
            if prompt_meta
        },
    }
