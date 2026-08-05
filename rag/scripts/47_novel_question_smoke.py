"""Run arbitrary questions through the real local RAG pipeline.

This is a diagnostic CLI.  It contains no expected answers or document/page
fixtures, so it is also useful for checking that a change generalises beyond
the seven UI examples.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rag_answer_lib import load_unified_collection
from rag_inprocess import run_full_inprocess


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--unified-id", default="full_corpus_715_v1")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("RAG_DEBUG_TRACE_STDERR", "0")

    index_dir = Path("data/processed/index")
    chunks_dir = Path("data/processed/chunks")
    collection, embed_model, manifest = load_unified_collection(
        args.unified_id, index_dir
    )
    row = {
        "question_id": "novel_smoke",
        "question": args.question,
    }
    if args.source:
        row["retrieval_sources"] = [value.upper() for value in args.source]

    output = run_full_inprocess(
        row,
        collection=collection,
        embed_model=embed_model,
        manifest=manifest,
        index_dir=index_dir,
        chunks_dir=chunks_dir,
        latency_mode="accurate",
        skip_ollama_probe=True,
    )
    answer_out = output.get("answer_out") or {}
    result = {
        "question": args.question,
        "answer": output.get("answer") or "",
        "provider": answer_out.get("provider"),
        "prompt_meta": answer_out.get("prompt_meta"),
        "verification_summary": answer_out.get("verification_summary"),
        "evidence": [
            {
                "citation_id": row.get("citation_id"),
                "file_name": row.get("file_name"),
                "page": row.get("page"),
                "chunk_id": row.get("chunk_id"),
                "evidence": (
                    row.get("chunk_text")
                    or row.get("text")
            or row.get("chunk_preview")
            or row.get("preview")
                    or ""
                )[:900],
            }
            for row in answer_out.get("evidence_table") or []
        ],
        "timing_metrics": output.get("timing_metrics"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
