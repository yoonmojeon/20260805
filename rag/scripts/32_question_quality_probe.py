"""Run one unseen question through the same in-process pipeline as the UI.

This is a diagnostic harness, not an answer fixture.  It accepts arbitrary
questions and records the generated answer and its final Evidence Table.
"""
from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

from rag_answer_lib import load_unified_collection
from rag_inprocess import run_full_inprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--sources", nargs="*", default=[])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/logs/unseen_question_probe.json"),
    )
    parser.add_argument("--with-llm", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("RAG_DEBUG_TRACE_STDERR", "0")

    index_dir = Path("data/processed/index")
    chunks_dir = Path("data/processed/chunks")
    collection, embed_model, manifest = load_unified_collection(
        "full_corpus_715_v1", index_dir
    )
    output = run_full_inprocess(
        {
            "question_id": "unseen_probe",
            "question": args.question,
            "retrieval_sources": args.sources,
        },
        collection=collection,
        embed_model=embed_model,
        manifest=manifest,
        index_dir=index_dir,
        chunks_dir=chunks_dir,
        latency_mode="accurate",
        skip_llm=not args.with_llm,
        skip_ollama_probe=not args.with_llm,
    )
    search = output.get("search_out") or {}
    answer_out = output.get("answer_out") or {}
    answer = output.get("answer") or answer_out.get("answer") or ""
    record = {
        "question": args.question,
        "sources": args.sources,
        "answer": answer,
        "answer_mode": output.get("answer_mode"),
        "verification_summary": answer_out.get("verification_summary"),
        "evidence_table": answer_out.get("evidence_table") or [],
        "retrieved": [
            {
                "file_name": getattr(chunk, "file_name", ""),
                "page": getattr(chunk, "page_number", None),
                "chunk_id": getattr(chunk, "chunk_id", ""),
                "text": getattr(chunk, "text", "")[:800],
            }
            for chunk in search.get("retrieved") or []
        ],
        "timing_metrics": output.get("timing_metrics"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
