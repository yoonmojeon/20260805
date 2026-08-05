"""Persist Rule/Guidance hybrid retrieval execution logs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LOG_DIR = Path("data/processed/logs/rule_lookup_hybrid")


def _parse_citations(answer: str) -> list[int]:
    import re

    return sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer or "")})


def save_rule_lookup_run_log(
    *,
    question: str,
    row: dict,
    category: str,
    dense_results: list[dict],
    bm25_results: list[dict],
    fused_results: list[dict],
    retrieved: list[Any],
    answer: str,
    warning_flags: list[str],
    log_dir: Path | None = None,
    tag: str = "",
) -> Path:
    log_dir = log_dir or DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    qid = str(row.get("question_id") or "adhoc")
    suffix = f"_{tag}" if tag else ""
    path = log_dir / f"{ts}_{qid}{suffix}.json"

    used = _parse_citations(answer)
    used_citations = []
    for cid in used:
        if 1 <= cid <= len(retrieved):
            c = retrieved[cid - 1]
            used_citations.append(
                {
                    "citation_id": f"[{cid}]",
                    "file_name": getattr(c, "file_name", ""),
                    "page": getattr(c, "page_number", None),
                    "chunk_id": getattr(c, "chunk_id", ""),
                }
            )

    payload = {
        "timestamp": ts,
        "question_id": qid,
        "question": question,
        "top_level_category": category,
        "internal_intent": row.get("internal_intent") or f"{category}_lookup",
        "class_society_hint": row.get("class_society_hint"),
        "dense_top20": (row.get("_hybrid_retrieval_log") or {}).get("dense_top20")
        or (row.get("_hybrid_retrieval_log") or {}).get("dense_results", [])[:20],
        "bm25_query": (row.get("_hybrid_retrieval_log") or {}).get("bm25_query"),
        "bm25_query_tokens": (row.get("_hybrid_retrieval_log") or {}).get("bm25_query_tokens"),
        "bm25_term_checks": (row.get("_hybrid_retrieval_log") or {}).get("bm25_term_checks"),
        "bm25_top20": (row.get("_hybrid_retrieval_log") or {}).get("bm25_top20")
        or (row.get("_hybrid_retrieval_log") or {}).get("bm25_results", [])[:20],
        "dense_results": dense_results,
        "bm25_results": bm25_results,
        "fused_top10": (row.get("_hybrid_retrieval_log") or {}).get("fused_top10"),
        "fused_results": fused_results,
        "used_citations": used_citations,
        "answer": answer,
        "warning_flags": list(dict.fromkeys(warning_flags)),
        "retrieval_variant": row.get("retrieval_variant") or "hybrid_rrf",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
