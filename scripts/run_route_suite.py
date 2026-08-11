"""JSON 질문 세트로 의도 라우터 정확도와 지연시간을 측정한다."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router.intent_router import route_question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="gemma4:12b")
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="LLM 1차 라우터를 건너뛰고 규칙 fallback만 검증한다.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    questions_path = args.questions.resolve()
    output_path = args.output.resolve()
    cases = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("질문 파일의 최상위 값은 JSON 배열이어야 합니다.")

    results: list[dict] = []
    latencies: list[float] = []
    for index, case in enumerate(cases, start=1):
        question = str(case.get("q") or case.get("question") or "").strip()
        expected = str(case.get("expected_route") or "").strip()
        started = time.perf_counter()
        decision = route_question(
            question,
            use_llm_fallback=not args.rules_only,
            active_model=args.model,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        latencies.append(elapsed_ms)
        ok = decision.route == expected
        results.append(
            {
                **case,
                "actual_route": decision.route,
                "ok": ok,
                "method": decision.method,
                "confidence": decision.confidence,
                "fallback_used": decision.fallback_used,
                "latency_ms": elapsed_ms,
                "router_latency_ms": decision.router_latency_ms,
                "reason": decision.reason,
            }
        )
        print(
            f"[{index:03d}/{len(cases):03d}] "
            f"{'PASS' if ok else 'FAIL'} expected={expected} "
            f"actual={decision.route} method={decision.method} {elapsed_ms:.0f}ms",
            flush=True,
        )

    passed = sum(1 for row in results if row["ok"])
    fallbacks = sum(1 for row in results if row["fallback_used"])
    try:
        questions_label = questions_path.relative_to(ROOT).as_posix()
    except ValueError:
        questions_label = str(questions_path)
    summary = {
        "questions": questions_label,
        "model": args.model,
        "rules_only": args.rules_only,
        "count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "accuracy": round(passed / len(results), 4) if results else 0.0,
        "fallback_count": fallbacks,
        "mean_latency_ms": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        "median_latency_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
    }
    payload = {"summary": summary, "results": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
