"""Run mixed ~100 question eval across chat / ops / text / table paths."""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import DEFAULT_RAG_COLLECTION, DEFAULT_TABLE_COLLECTION
from services.orchestrator import handle_question
from services.ops_service import ops_db_ready
from services.rag_service import rag_index_ready, warmup_rag_resources

SUITE = ROOT / "data/eval/suite_100_mixed.json"
OUT = ROOT / "data/processed/logs/suite_100_mixed_results.json"

WEAK_RAG = (
    "인덱스가 아직",
    "모두 없습니다",
    "찾을 수 없",
    "자료가 부족",
    "제공된 문서에서 확인되지",
    "검색 근거에서 질문에 직접 답할 내용을 확인하지 못했습니다",
    "관련 근거를 찾지 못했습니다",
    "답변 텍스트를 찾지 못했습니다",
    "Traceback",
    "[문서 RAG 오류]",
)
WEAK_OPS = (
    "운항 SQLite DB가 아직",
    "데이터 없음",
    "계산할 수 없습니다",
    "지원되지 않는 연도",
    "Traceback",
    "[LLM 오류]",
)


def grade(kind: str, answer: str, route: str | None, expected_route: str) -> dict:
    ans = (answer or "").strip()
    route_ok = (route or "") == expected_route
    empty = not ans
    if kind == "chat":
        weak = empty or len(ans) < 20
    elif kind == "ops":
        weak = empty or any(m in ans for m in WEAK_OPS) or len(ans) < 30
    else:
        weak = empty or any(m in ans for m in WEAK_RAG) or len(ans) < 40
    if empty:
        g = "empty"
    elif not route_ok:
        g = "route_mismatch"
    elif weak:
        g = "weak"
    else:
        g = "good"
    return {"grade": g, "route_ok": route_ok, "chars": len(ans)}


def main() -> None:
    # Faster suite: table path uses table index only (no dual LLM). Production dual remains default.
    os.environ.setdefault("MARITIME_RAG_DUAL", "0")
    os.environ.setdefault("PYTHONUTF8", "1")

    cases = json.loads(SUITE.read_text(encoding="utf-8"))
    print("n_cases", len(cases))
    print("ops_db", ops_db_ready())
    print("doc_idx", rag_index_ready(DEFAULT_RAG_COLLECTION))
    print("tbl_idx", rag_index_ready(DEFAULT_TABLE_COLLECTION))
    print("dual_env", os.environ.get("MARITIME_RAG_DUAL"))

    try:
        print("warmup", warmup_rag_resources(DEFAULT_RAG_COLLECTION))
        print("warmup", warmup_rag_resources(DEFAULT_TABLE_COLLECTION))
    except Exception as exc:
        print("warmup_err", exc)

    rows: list[dict] = []
    t_all = time.time()
    for i, case in enumerate(cases, 1):
        q = case["q"]
        kind = case["kind"]
        expected = case["expected_route"]
        t0 = time.time()
        try:
            result = handle_question(
                q,
                use_llm_router=False,
                rag_latency_mode="fast",
            )
            ans = str(result.get("answer") or "")
            route = (result.get("route") or {}).get("route")
            g = grade(kind, ans, route, expected)
            row = {
                "id": case["id"],
                "kind": kind,
                "expected_route": expected,
                "route": route,
                "q": q,
                "sec": round(time.time() - t0, 2),
                "preview": ans[:280].replace("\n", " "),
                **g,
            }
        except Exception as exc:
            row = {
                "id": case["id"],
                "kind": kind,
                "expected_route": expected,
                "route": None,
                "q": q,
                "sec": round(time.time() - t0, 2),
                "grade": "error",
                "route_ok": False,
                "chars": 0,
                "preview": "",
                "error": str(exc),
            }
        rows.append(row)
        print(
            f"[{i:03d}/{len(cases)}] {row['grade']:14} {kind:5} -> {row.get('route')} "
            f"{row['sec']:6.1f}s | {q[:42]}",
            flush=True,
        )
        if i % 10 == 0:
            _write(rows, partial=True)

    _write(rows, partial=False, elapsed=round(time.time() - t_all, 1))
    print("done", OUT)


def _write(rows: list[dict], *, partial: bool, elapsed: float | None = None) -> None:
    by_kind_grade: Counter[tuple[str, str]] = Counter((r["kind"], r["grade"]) for r in rows)
    by_kind: Counter[str] = Counter(r["kind"] for r in rows)
    summary = {
        "n": len(rows),
        "partial": partial,
        "elapsed_sec": elapsed,
        "good_rate": round(sum(1 for r in rows if r["grade"] == "good") / max(len(rows), 1), 3),
        "route_ok_rate": round(sum(1 for r in rows if r.get("route_ok")) / max(len(rows), 1), 3),
        "by_kind": dict(by_kind),
        "by_kind_grade": {f"{k}_{g}": v for (k, g), v in sorted(by_kind_grade.items())},
        "avg_sec": round(sum(r["sec"] for r in rows) / max(len(rows), 1), 2),
        "weak_or_worse": [
            {"id": r["id"], "kind": r["kind"], "grade": r["grade"], "q": r["q"], "route": r.get("route")}
            for r in rows
            if r["grade"] != "good"
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not partial:
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
