"""Inspect table row/cell selection for curated evaluation questions."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUESTIONS = ROOT / "data" / "eval" / "table_questions_22docs_practical_v1_curated.jsonl"


def _read_rows() -> dict[str, dict]:
    return {
        row["qid"]: row
        for row in (
            json.loads(line)
            for line in QUESTIONS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("qids", nargs="+")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("RAG_DEBUG_TRACE_STDERR", "0")

    from services.rag_service import _run_search_only

    rag_scripts = ROOT / "rag" / "scripts"
    if str(rag_scripts) not in sys.path:
        sys.path.insert(0, str(rag_scripts))
    from table_qa_answer import (  # type: ignore
        build_deterministic_table_answer,
        select_table_evidence,
        top_table_cell_hints,
        verify_row_column_intersection,
    )

    rows = _read_rows()
    output: list[dict] = []
    for qid in args.qids:
        gold = rows[qid]
        search = _run_search_only(
            str(gold["question"]), latency_mode="fast", table_side=True
        )
        row = search["row"]
        search_out = search["search_out"]
        retrieved = list(search_out.get("retrieved") or [])
        pool = list(search_out.get("retrieval_pool") or retrieved)
        debug = (
            row.get("_table_retrieval_debug")
            or search_out.get("table_retrieval_debug")
            or search_out.get("retrieval_debug")
            or search_out.get("debug")
            or {}
        )
        evidence = select_table_evidence(row, retrieved, pool, debug=debug)
        verification = verify_row_column_intersection(row, evidence, debug=debug)
        answer = build_deterministic_table_answer(row, evidence, debug=debug)

        def chunk_view(chunk: object) -> dict:
            return {
                "chunk_id": str(getattr(chunk, "chunk_id", "") or ""),
                "table_id": str(getattr(chunk, "table_id", "") or ""),
                "file_name": str(getattr(chunk, "file_name", "") or ""),
                "page": getattr(chunk, "page_number", None),
                "text": str(getattr(chunk, "text", "") or "")[:500],
            }

        ranked = []
        for item in list(verification.get("ranked") or [])[:8]:
            ranked.append(
                {
                    "score": round(float(item[0]), 3),
                    "chunk_id": str(getattr(item[1], "chunk_id", "") or ""),
                    "key": item[2],
                    "value": item[3],
                }
            )
        record = {
                "qid": qid,
                "question": gold["question"],
                "gold_answer": gold["gold_answer"],
                "parsed_query": debug.get("parsed_query"),
                "selected_table_id": debug.get("selected_table_id"),
                "retrieved": [chunk_view(chunk) for chunk in retrieved[:5]],
                "evidence": [chunk_view(chunk) for chunk in evidence[:5]],
                "verification": {
                    key: value
                    for key, value in verification.items()
                    if key not in {"selected", "ranked", "support_chunks"}
                },
                "ranked_cells": ranked,
                "hints": top_table_cell_hints(row, evidence, debug=debug),
                "deterministic_answer": answer,
            }
        if args.compact:
            record.pop("retrieved", None)
            record.pop("evidence", None)
            record.pop("hints", None)
        output.append(record)
    print(json.dumps(output, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
