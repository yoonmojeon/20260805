#!/usr/bin/env python3
"""50-item open-table heavy quality eval across one or all Ollama models.

Usage:
  .\\.venv\\Scripts\\python.exe scripts/run_quality_50.py
  .\\.venv\\Scripts\\python.exe scripts/run_quality_50.py --model llama3.1:8b
  .\\.venv\\Scripts\\python.exe scripts/run_quality_50.py --all-models
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _needles(answer: str, needles: list[str]) -> tuple[list[str], list[str]]:
    hits, miss = [], []
    for n in needles or []:
        if re.search(n, answer or "", flags=re.I):
            hits.append(n)
        else:
            miss.append(n)
    return hits, miss


_THIN_RE = re.compile(
    r"확인하지 못했습니다|인용으로 검증되지 않은|표 근거에서 질문에 해당하는 셀을 확정하지|"
    r"추가 확인 필요|검색된 .{0,40}연결되는 .{0,20}확인하지|"
    r"^## 1\) 핵심 요약\s*$",
    re.I,
)
_REFUSE_EMPTY_RE = re.compile(r"^\s*$|오류\]|항차 데이터 없음")


def _quality(answer: str, hits: list[str], miss: list[str], qtype: str) -> str:
    text = answer or ""
    if _REFUSE_EMPTY_RE.search(text) or len(text.strip()) < 25:
        return "BAD"
    if not hits and miss:
        return "BAD"
    if _THIN_RE.search(text) and len(hits) < max(1, len(hits) + len(miss) - 1):
        return "THIN"
    if miss and hits:
        if len(hits) >= len(miss) and len(text) >= 80:
            return "THIN"
        return "THIN"
    if qtype.startswith("ops") and len(text) < 60:
        return "THIN"
    if (qtype.startswith("table") or qtype == "table_open") and (
        "결론:" in text or re.search(r"\d", text)
    ):
        return "GOOD"
    if len(text) >= 60:
        return "GOOD"
    return "THIN"


def _run_one(model: str, rows: list[dict], out_path: Path) -> dict:
    os.environ["MODEL_NAME"] = model
    os.environ["MARITIME_OLLAMA_MODEL"] = model
    sys.path[:0] = [str(ROOT), str(ROOT / "rag" / "scripts"), str(ROOT / "ops")]

    from project_paths import DEFAULT_RAG_COLLECTION, DEFAULT_TABLE_COLLECTION
    from services.orchestrator import handle_question
    from services.rag_service import warmup_rag_resources

    print(f"\n===== model={model} =====", flush=True)
    print("warming…", flush=True)
    warmup_rag_resources(DEFAULT_RAG_COLLECTION)
    warmup_rag_resources(DEFAULT_TABLE_COLLECTION)

    results = []
    by_type: dict[str, list] = defaultdict(list)

    for row in rows:
        q = row["question"]
        t0 = time.perf_counter()
        try:
            out = handle_question(q, use_llm_router=False, llm_model=model)
            ans = str((out or {}).get("answer") or "")
            route_obj = (out or {}).get("route")
            if isinstance(route_obj, dict):
                route = str(route_obj.get("route") or "")
            else:
                route = str((out or {}).get("source") or (out or {}).get("last_route") or "")
            meta = (out or {}).get("meta") or {}
            retrieval_mode = meta.get("retrieval_mode") or meta.get("debug", {}).get("mode")
            dual = meta.get("dual_retrieval") or meta.get("dual_retrieval_enabled")
            fuse = meta.get("fuse_policy")
            err = ""
        except Exception as exc:
            ans, route, err = "", "", f"{type(exc).__name__}: {exc}"
            retrieval_mode, dual, fuse = None, None, None
        dt = time.perf_counter() - t0
        hits, miss = _needles(ans, row.get("needles") or [])
        expect = row.get("route")
        if expect == "hybrid":
            route_ok = route in {"hybrid", "ops", "rag"}
        elif expect == "chat" and row.get("type") == "oos":
            route_ok = route in {"chat", "oos"} or bool(hits)
        else:
            route_ok = (not expect) or route == expect
        if err:
            needle_v = "FAIL"
        elif not miss:
            needle_v = "PASS"
        elif hits:
            needle_v = "WEAK"
        else:
            needle_v = "FAIL"
        if not route_ok and needle_v == "PASS":
            needle_v = "WEAK"
            miss = list(miss) + [f"route={route}!={expect}"]
        elif not route_ok and needle_v != "FAIL":
            miss = list(miss) + [f"route={route}!={expect}"]

        qtype = str(row.get("type") or "other")
        quality = "BAD" if err else _quality(ans, hits, miss, qtype)
        if needle_v == "FAIL" and quality == "GOOD":
            quality = "THIN"

        print(
            f"[{needle_v}/{quality}] {row['id']} type={qtype} {dt:.1f}s "
            f"route={route} mode={retrieval_mode} fuse={fuse} dual={dual} miss={miss}",
            flush=True,
        )
        if needle_v != "PASS" or quality != "GOOD":
            print(
                f"    gold={row.get('gold')} | ans={(ans or err)[:180].replace(chr(10), ' ')}",
                flush=True,
            )

        item = {
            "id": row["id"],
            "type": qtype,
            "question": q,
            "gold": row.get("gold"),
            "expect_route": expect,
            "route": route,
            "route_ok": route_ok,
            "retrieval_mode": retrieval_mode,
            "fuse_policy": fuse,
            "dual_retrieval": dual,
            "needle": needle_v,
            "quality": quality,
            "dt": round(dt, 2),
            "miss": miss,
            "hits": hits,
            "answer_preview": (ans or "")[:600],
            "error": err,
            "llm_model": model,
        }
        results.append(item)
        by_type[qtype].append(item)

    overall_q = Counter(i["quality"] for i in results)
    overall_n = Counter(i["needle"] for i in results)
    open_items = [
        i
        for i in results
        if i["type"] in {"table_open", "table_cell", "table_caption", "table_reporting"}
        or str(i["id"]).startswith("open_")
        or str(i["id"]).startswith("table_")
    ]
    open_n = Counter(i["needle"] for i in open_items)
    summary = {
        "model": model,
        "n": len(results),
        "needle": dict(overall_n),
        "quality": dict(overall_q),
        "avg_dt": round(sum(i["dt"] for i in results) / max(1, len(results)), 2),
        "open_table_n": len(open_items),
        "open_table_needle": dict(open_n),
        "pass_rate": round(overall_n.get("PASS", 0) / max(1, len(results)), 3),
        "open_pass_rate": round(open_n.get("PASS", 0) / max(1, len(open_items)), 3),
    }
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("wrote", out_path, flush=True)
    return summary


def main() -> int:
    from services.llm_models import AVAILABLE_LLM_MODELS, DEFAULT_LLM_MODEL

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data" / "eval" / "quality_50_open_mix.jsonl",
    )
    ap.add_argument("--model", default=DEFAULT_LLM_MODEL)
    ap.add_argument("--all-models", action="store_true")
    args = ap.parse_args()

    if not args.questions.exists():
        from scripts.build_quality_50 import main as build_main

        build_main()

    rows = _load(args.questions)
    models = list(AVAILABLE_LLM_MODELS) if args.all_models else [args.model.strip()]
    summaries = []
    for model in models:
        safe = model.replace(":", "_").replace("/", "_")
        out = ROOT / "data" / "processed" / "logs" / f"quality_50_{safe}.json"
        summaries.append(_run_one(model, rows, out))

    board = ROOT / "data" / "processed" / "logs" / "quality_50_models.json"
    board.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nMODEL BOARD", json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)
    print("wrote", board, flush=True)
    worst_fail = max((s.get("needle", {}).get("FAIL", 0) for s in summaries), default=0)
    return 0 if worst_fail <= 12 else 1


if __name__ == "__main__":
    raise SystemExit(main())
