#!/usr/bin/env python3
"""Run the example-10 gold set against one Ollama model and score needles.

Usage:
  .\\.venv\\Scripts\\python.exe scripts/run_model_compare_10.py --model llama3.1:8b
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data" / "eval" / "example_10_gold.jsonl",
    )
    ap.add_argument("--latency-mode", default="fast")
    ap.add_argument("--warmup", action="store_true", default=True)
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    model = args.model.strip()
    os.environ["MODEL_NAME"] = model
    os.environ["MARITIME_OLLAMA_MODEL"] = model
    os.environ.setdefault("PYTHONUTF8", "1")

    sys.path[:0] = [str(ROOT), str(ROOT / "rag" / "scripts"), str(ROOT / "ops")]

    # Import after env so DEFAULT_OLLAMA_MODEL picks it up.
    from rag_answer_lib import DEFAULT_OLLAMA_MODEL  # noqa: E402
    from project_paths import DEFAULT_RAG_COLLECTION, DEFAULT_TABLE_COLLECTION  # noqa: E402
    from services.orchestrator import handle_question  # noqa: E402
    from services.rag_service import warmup_rag_resources  # noqa: E402

    print(f"MODEL env={model} DEFAULT_OLLAMA_MODEL={DEFAULT_OLLAMA_MODEL}", flush=True)
    if DEFAULT_OLLAMA_MODEL != model:
        print(
            "WARN: DEFAULT_OLLAMA_MODEL mismatch — already imported elsewhere?",
            flush=True,
        )

    if not args.no_warmup:
        print("warming retrieval…", flush=True)
        warmup_rag_resources(DEFAULT_RAG_COLLECTION)
        warmup_rag_resources(DEFAULT_TABLE_COLLECTION)
        # One tiny LLM warm so first Q isn't cold-start only.
        try:
            import urllib.request

            body = json.dumps(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                    "options": {"num_predict": 8},
                    "keep_alive": "24h",
                }
            ).encode()
            req = urllib.request.Request(
                "http://127.0.0.1:11434/api/chat",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=180) as resp:
                resp.read()
            print(f"llm warm {time.perf_counter() - t0:.1f}s", flush=True)
        except Exception as exc:
            print(f"llm warm failed: {exc}", flush=True)

    rows = _load_rows(args.questions)
    results = []
    for row in rows:
        q = row["question"]
        t0 = time.perf_counter()
        try:
            out = handle_question(q, use_llm_router=False)
            ans = str((out or {}).get("answer") or "")
            route_obj = (out or {}).get("route")
            if isinstance(route_obj, dict):
                route = str(route_obj.get("route") or "")
            else:
                route = str(
                    (out or {}).get("source")
                    or (out or {}).get("last_route")
                    or route_obj
                    or ""
                )
            err = ""
        except Exception as exc:
            ans, route, err = "", "", f"{type(exc).__name__}: {exc}"
        dt = time.perf_counter() - t0
        hits, miss = _needle_hits(ans, row.get("needles") or [])
        expect_route = row.get("route")
        route_ok = (not expect_route) or (route == expect_route)
        if err:
            verdict = "FAIL"
        elif not route_ok:
            verdict = "FAIL"
            miss = list(miss) + [f"route!={expect_route}"]
        elif not miss:
            verdict = "PASS"
        elif hits:
            verdict = "WEAK"
        else:
            verdict = "FAIL"
        print(
            f"[{verdict}] {row.get('id')} {dt:.1f}s route={route} miss={miss}",
            flush=True,
        )
        if miss or err:
            preview = (ans or err)[:220].replace("\n", " ")
            print(f"    gold={row.get('gold')} | ans={preview}", flush=True)
        results.append(
            {
                "id": row.get("id"),
                "verdict": verdict,
                "dt": round(dt, 2),
                "route": route,
                "miss": miss,
                "hits": hits,
                "error": err,
                "answer_preview": (ans or "")[:500],
                "gold": row.get("gold"),
                "question": q,
            }
        )

    counts = Counter(r["verdict"] for r in results)
    avg_dt = sum(r["dt"] for r in results) / max(1, len(results))
    summary = {
        "model": model,
        "counts": dict(counts),
        "avg_dt": round(avg_dt, 2),
        "pass": counts.get("PASS", 0),
        "n": len(results),
    }
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
    out_dir = ROOT / "data" / "processed" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", model)
    out_path = out_dir / f"model_compare_10_{safe}.json"
    out_path.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("wrote", out_path, flush=True)
    return 0 if counts.get("FAIL", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
