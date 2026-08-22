"""Compact diagnostics for a compound meeting + class-rule search."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--latency-mode", choices=("fast", "accurate"), default="accurate")
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("RAG_DEBUG_TRACE_STDERR", "0")

    from services.rag_service import _run_search_only

    result = _run_search_only(
        args.question,
        latency_mode=args.latency_mode,
        table_side=False,
    )
    row = result["row"]
    search = result["search_out"]

    def view(chunk: object) -> dict:
        return {
            "chunk_id": str(getattr(chunk, "chunk_id", "") or ""),
            "source": str(getattr(chunk, "source", "") or ""),
            "file_name": str(getattr(chunk, "file_name", "") or ""),
            "page": getattr(chunk, "page_number", None),
            "text": str(getattr(chunk, "text", "") or "")[:360],
        }

    output = {
        "route": row.get("_text_document_route"),
        "evidence_completion": row.get("_evidence_completion"),
        "retrieved": [view(chunk) for chunk in list(search.get("retrieved") or [])],
        "pool_size": len(list(search.get("retrieval_pool") or [])),
        "exact_pool": [
            view(chunk)
            for chunk in list(search.get("retrieval_pool") or [])
            if any(
                phrase in str(getattr(chunk, "text", "") or "").lower()
                for phrase in (
                    "gas fuelled ammonia",
                    "fuel ready(ammonia",
                    "ammonia ready",
                )
            )
        ][:20],
        "timing": result.get("timing_metrics"),
    }
    print(json.dumps(output, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
