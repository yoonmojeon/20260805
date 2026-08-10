#!/usr/bin/env python3
"""Category smoke regression — fail by type, not by one-off question text.

Usage:
  .\\.venv\\Scripts\\python.exe scripts/run_smoke_categories.py
  .\\.venv\\Scripts\\python.exe scripts/run_smoke_categories.py --route-only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "rag" / "scripts"), str(ROOT / "ops")]


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _needle_hits(answer: str, needles: list[str]) -> tuple[list[str], list[str]]:
    hits, miss = [], []
    for n in needles or []:
        if re.search(n, answer or "", flags=re.IGNORECASE):
            hits.append(n)
        else:
            miss.append(n)
    return hits, miss


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data" / "eval" / "smoke_categories.jsonl",
    )
    ap.add_argument("--route-only", action="store_true")
    ap.add_argument("--latency-mode", default="fast")
    args = ap.parse_args()

    from router.intent_router import route_question
    from services.retrieval_mode import classify_retrieval_mode

    rows = _load_rows(args.questions)
    results = []

    if not args.route_only:
        from project_paths import DEFAULT_RAG_COLLECTION, DEFAULT_TABLE_COLLECTION
        from services.orchestrator import handle_question
        from services.rag_service import warmup_rag_resources

        print("warming…")
        warmup_rag_resources(DEFAULT_RAG_COLLECTION)
        warmup_rag_resources(DEFAULT_TABLE_COLLECTION)

    for row in rows:
        q = row["question"]
        intent = route_question(q, use_llm_fallback=False)
        rmode = classify_retrieval_mode(q).value
        route_ok = intent.route == row.get("expect_route")
        ret_ok = True
        exp_ret = row.get("expect_retrieval")
        if exp_ret:
            ret_ok = rmode == exp_ret or (
                exp_ret == "table" and rmode in {"table", "both"}
            )

        ans = ""
        dt = 0.0
        ans_ok = True
        miss: list[str] = []
        if not args.route_only:
            t0 = time.perf_counter()
            out = handle_question(q, use_llm_router=False)
            dt = time.perf_counter() - t0
            ans = str((out or {}).get("answer") or "")
            _hits, miss = _needle_hits(ans, row.get("needles") or [])
            ans_ok = len(miss) == 0

        ok = route_ok and ret_ok and ans_ok
        verdict = "PASS" if ok else "FAIL"
        if ok is False and route_ok and ret_ok and miss and len(miss) < len(row.get("needles") or []):
            verdict = "WEAK"
        print(
            f"[{verdict}] {row.get('category')} route={intent.route}"
            f"{'' if route_ok else '!'}/{row.get('expect_route')}"
            f" mode={rmode} {dt:.1f}s | {q[:40]}"
        )
        if miss:
            print(f"    miss={miss} ans={ans[:160].replace(chr(10), ' ')}")
        results.append(
            {
                "id": row.get("id"),
                "category": row.get("category"),
                "verdict": verdict,
                "route": intent.route,
                "retrieval_mode": rmode,
                "dt": round(dt, 2),
                "miss": miss,
            }
        )

    counts = Counter(r["verdict"] for r in results)
    print("SUMMARY", dict(counts))
    out_path = ROOT / "data" / "processed" / "logs" / "smoke_categories_last.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out_path)
    return 0 if counts.get("FAIL", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
