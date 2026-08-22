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
    r"확인하지 못했습니다|근거를 찾지 못했습니다|표 근거에서 질문에 해당하는 셀을 확정하지|"
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
    if qtype.startswith("ops") and len(text) < 40:
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
    # Windows redirected stdout otherwise inherits a legacy locale such as
    # CP949 and can crash mid-evaluation when a model emits uncommon Unicode.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data" / "eval" / "quality_30_types.jsonl",
    )
    ap.add_argument("--model", default=os.environ.get("MARITIME_OLLAMA_MODEL", "gemma4:12b"))
    ap.add_argument("--rules-only", action="store_true")
    ap.add_argument("--output", type=Path)
    ap.add_argument(
        "--id-prefix",
        action="append",
        default=[],
        help="Run only question ids that start with one of these prefixes (repeatable).",
    )
    args = ap.parse_args()
    # Some retrieval modules temporarily change the process working directory.
    # Resolve CLI paths before importing/running them so evaluation artifacts do
    # not leak into ``rag/data`` when a relative path was supplied.
    launch_cwd = Path.cwd()
    if not args.questions.is_absolute():
        args.questions = (launch_cwd / args.questions).resolve()
    if args.output is not None and not args.output.is_absolute():
        args.output = (launch_cwd / args.output).resolve()

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
    if args.id_prefix:
        prefixes = tuple(str(value) for value in args.id_prefix if str(value))
        rows = [row for row in rows if str(row.get("id") or "").startswith(prefixes)]
    results = []
    by_type: dict[str, list] = defaultdict(list)

    for row in rows:
        q = row["question"]
        t0 = time.perf_counter()
        try:
            out = handle_question(
                q,
                use_llm_router=not args.rules_only,
                llm_model=model,
            )
            ans = str((out or {}).get("answer") or "")
            route_obj = (out or {}).get("route")
            if isinstance(route_obj, dict):
                route = str(route_obj.get("route") or "")
            else:
                route = str((out or {}).get("source") or (out or {}).get("last_route") or "")
                route_obj = {}
            response_meta = dict((out or {}).get("meta") or {})
            err = ""
        except Exception as exc:
            ans, route, err, route_obj, response_meta = "", "", f"{type(exc).__name__}: {exc}", {}, {}
        dt = time.perf_counter() - t0
        hits, miss = _needles(ans, row.get("needles") or [])
        expect = row.get("route")
        if expect == "chat" and row.get("type") == "oos":
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
            "answer": ans,
            "answer_preview": (ans or "")[:600],
            "error": err,
            "router_latency_ms": float((route_obj or {}).get("router_latency_ms") or 0.0),
            "router_fallback_used": bool((route_obj or {}).get("fallback_used")),
            "router_error_kind": (route_obj or {}).get("llm_error_kind"),
            "router_done_reason": (route_obj or {}).get("llm_done_reason"),
            "router_eval_count": (route_obj or {}).get("llm_eval_count"),
            "answer_empty": bool(response_meta.get("answer_empty")) or not bool(ans.strip()),
            "answer_error": response_meta.get("answer_error"),
            "answer_done_reason": response_meta.get("answer_done_reason")
            or response_meta.get("hybrid_synthesis_done_reason"),
            "answer_model": response_meta.get("answer_model") or model,
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
    qa_pass = sum(
        1
        for item in results
        if item["route_ok"] and item["needle"] == "PASS" and item["quality"] != "BAD"
    )
    route_mismatch = sum(1 for item in results if not item["route_ok"])
    invalid_empty = sum(1 for item in results if item["answer_empty"])
    failures = sum(
        1
        for item in results
        if item["error"]
        or item["answer_error"]
        or item["answer_empty"]
        or str(item["answer_preview"]).startswith("[")
    )
    router_calls = sum(
        1
        for item in results
        if item["router_latency_ms"] > 0 or item["router_error_kind"] is not None
    )
    router_fallbacks = sum(1 for item in results if item["router_fallback_used"])
    summary = {
        "model": model,
        "routing_mode": "rules_only" if args.rules_only else "llm_primary",
        "n": len(results),
        "needle": dict(overall_n),
        "quality": dict(overall_q),
        "avg_dt": round(sum(i["dt"] for i in results) / max(1, len(results)), 2),
        "qa_pass": qa_pass,
        "qa_pass_rate": qa_pass / max(1, len(results)),
        "route_mismatch_count": route_mismatch,
        "invalid_empty_answer_count": invalid_empty,
        "failure_count": failures,
        "router_calls": router_calls,
        "router_fallback_count": router_fallbacks,
        "router_fallback_rate": router_fallbacks / max(1, router_calls),
        "mean_router_latency_ms": round(
            sum(i["router_latency_ms"] for i in results) / max(1, router_calls), 2
        ),
        "can_do_yes": sum(1 for t in type_board if t["can_do"] == "YES"),
        "can_do_partial": sum(1 for t in type_board if t["can_do"] == "PARTIAL"),
        "can_do_no": sum(1 for t in type_board if t["can_do"] == "NO"),
        "types": type_board,
    }
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    out = args.output or (
        ROOT
        / "data"
        / "processed"
        / "logs"
        / f"quality_30_{safe_model}_{summary['routing_mode']}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("wrote", out, flush=True)
    return 0 if overall_n.get("FAIL", 0) <= 5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
