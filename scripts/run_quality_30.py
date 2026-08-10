#!/usr/bin/env python3
"""30-type quality eval: route + needles + answer-quality rubric.

Usage:
  .\\.venv\\Scripts\\python.exe scripts/run_quality_30.py
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
        # partial gold coverage
        if len(hits) >= len(miss) and len(text) >= 80:
            return "THIN"
        return "THIN"
    # all needles hit
    if qtype.startswith("ops") and len(text) < 60:
        return "THIN"
    if qtype.startswith("table") and ("결론:" in text or re.search(r"\d", text)):
        return "GOOD"
    if len(text) >= 60:
        return "GOOD"
    return "THIN"


def _can_do(quality_counts: Counter, needle_pass: int, n: int) -> str:
    if n == 0:
        return "N/A"
    good = quality_counts.get("GOOD", 0)
    thin = quality_counts.get("THIN", 0)
    bad = quality_counts.get("BAD", 0)
    if bad == 0 and needle_pass >= n * 0.8 and good >= n * 0.5:
        return "YES"
    if bad <= max(1, n // 4) and (good + thin) >= n * 0.6:
        return "PARTIAL"
    return "NO"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data" / "eval" / "quality_30_types.jsonl",
    )
    ap.add_argument("--model", default=os.environ.get("MARITIME_OLLAMA_MODEL", "llama3.1:8b"))
    args = ap.parse_args()

    model = args.model.strip()
    os.environ["MODEL_NAME"] = model
    os.environ["MARITIME_OLLAMA_MODEL"] = model
    sys.path[:0] = [str(ROOT), str(ROOT / "rag" / "scripts"), str(ROOT / "ops")]

    from project_paths import DEFAULT_RAG_COLLECTION, DEFAULT_TABLE_COLLECTION
    from services.orchestrator import handle_question
    from services.rag_service import warmup_rag_resources

    print(f"model={model}", flush=True)
    print("warming…", flush=True)
    warmup_rag_resources(DEFAULT_RAG_COLLECTION)
    warmup_rag_resources(DEFAULT_TABLE_COLLECTION)

    rows = _load(args.questions)
    results = []
    by_type: dict[str, list] = defaultdict(list)

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
                route = str((out or {}).get("source") or (out or {}).get("last_route") or "")
            err = ""
        except Exception as exc:
            ans, route, err = "", "", f"{type(exc).__name__}: {exc}"
        dt = time.perf_counter() - t0
        hits, miss = _needles(ans, row.get("needles") or [])
        expect = row.get("route")
        # hybrid may surface as ops/rag depending on decomposition — accept hybrid/ops/rag if expect hybrid
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
            f"route={route} miss={miss}",
            flush=True,
        )
        if needle_v != "PASS" or quality != "GOOD":
            print(f"    gold={row.get('gold')} | ans={(ans or err)[:180].replace(chr(10),' ')}", flush=True)

        item = {
            "id": row["id"],
            "type": qtype,
            "question": q,
            "gold": row.get("gold"),
            "expect_route": expect,
            "route": route,
            "route_ok": route_ok,
            "needle": needle_v,
            "quality": quality,
            "dt": round(dt, 2),
            "miss": miss,
            "hits": hits,
            "answer_preview": (ans or "")[:600],
            "error": err,
        }
        results.append(item)
        by_type[qtype].append(item)

    # type capability board
    type_board = []
    for t, items in sorted(by_type.items()):
        qc = Counter(i["quality"] for i in items)
        nc = Counter(i["needle"] for i in items)
        n = len(items)
        needle_pass = nc.get("PASS", 0)
        can = _can_do(qc, needle_pass, n)
        avg_dt = sum(i["dt"] for i in items) / n
        type_board.append(
            {
                "type": t,
                "n": n,
                "can_do": can,
                "needle": dict(nc),
                "quality": dict(qc),
                "avg_dt": round(avg_dt, 2),
            }
        )
        print(
            f"TYPE {t:16} n={n} can={can:7} needle={dict(nc)} quality={dict(qc)} avg={avg_dt:.1f}s",
            flush=True,
        )

    overall_q = Counter(i["quality"] for i in results)
    overall_n = Counter(i["needle"] for i in results)
    summary = {
        "model": model,
        "n": len(results),
        "needle": dict(overall_n),
        "quality": dict(overall_q),
        "avg_dt": round(sum(i["dt"] for i in results) / max(1, len(results)), 2),
        "can_do_yes": sum(1 for t in type_board if t["can_do"] == "YES"),
        "can_do_partial": sum(1 for t in type_board if t["can_do"] == "PARTIAL"),
        "can_do_no": sum(1 for t in type_board if t["can_do"] == "NO"),
        "types": type_board,
    }
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
    out = ROOT / "data" / "processed" / "logs" / "quality_30_last.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("wrote", out, flush=True)
    return 0 if overall_n.get("FAIL", 0) <= 5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
