"""Show what text retrieval actually returns for a question.

    python scripts/probe_text_retrieval.py "과도한 부식의 정의는 무엇인가?" --needle "쇠모한도의 75"

Prints every retrieved chunk with document, page and whether it carries the
needle, which separates "retrieval never found it" from "the answer builder had
it and ignored it".
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def norm(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--needle", default="")
    parser.add_argument("--latency-mode", default="fast")
    parser.add_argument("--tables", action="store_true")
    args = parser.parse_args()

    import services.rag_service as rs

    out = rs._run_search_only(
        args.question, latency_mode=args.latency_mode, table_side=args.tables
    )
    search_out = out.get("search_out") or {}
    retrieved = list(out.get("retrieved") or search_out.get("retrieved") or [])
    pool = list(search_out.get("retrieval_pool") or out.get("pool") or [])
    needle = norm(args.needle)
    print(f"질문: {args.question}")
    print(f"회수 {len(retrieved)}개 (pool {len(pool)}개), latency={args.latency_mode}\n")
    completion = search_out.get("evidence_completion") or {}
    if completion:
        print(
            "증거 슬롯: "
            f"hits={completion.get('slot_hits') or {}} "
            f"missing={completion.get('missing_slots') or []}\n"
        )
    for scope, chunks in (("retrieved", retrieved), ("pool", pool)):
        if not chunks:
            continue
        print(f"--- {scope}")
        for idx, chunk in enumerate(chunks, start=1):
            text = str(getattr(chunk, "text", "") or "")
            mark = "HIT " if needle and needle in norm(text) else "    "
            print(
                f"{mark}[{idx:2d}] {getattr(chunk, 'file_name', '')} "
                f"p{getattr(chunk, 'page_number', '')} "
                f"{getattr(chunk, 'chunk_id', '')} :: "
                f"{re.sub(r'[[]s+', ' ', text)[:110]}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
