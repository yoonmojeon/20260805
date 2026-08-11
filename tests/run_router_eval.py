"""Compare deterministic and active-model semantic router performance."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router.intent_router import RouteDecision, route_question
from tests.router_golden_cases import CASES_BY_ROUTE
from tests.router_heldout_cases import (
    HELDOUT_SINGLE,
    MULTITURN_SCENARIOS,
    SEMANTIC_HELDOUT_SINGLE,
)

MODELS = ("llama3.1:8b", "gemma4:12b", "mistral-nemo:12b")
LABELS = ("chat", "ops", "rag", "hybrid")


def _post_json(path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    base = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def available_models() -> set[str]:
    base = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    with urllib.request.urlopen(f"{base}/api/tags", timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {str(row.get("name") or row.get("model") or "") for row in payload.get("models", [])}


def warm_model(model: str) -> None:
    _post_json(
        "/api/generate",
        {"model": model, "prompt": "", "stream": False, "keep_alive": "24h"},
        timeout=300,
    )


def _empty_metrics() -> dict[str, Any]:
    return {
        "total": 0,
        "correct": 0,
        "hard_guard_count": 0,
        "llm_router_calls": 0,
        "llm_fallback_count": 0,
        "error_kinds": Counter(),
        "latencies_ms": [],
        "methods": Counter(),
        "confusion": defaultdict(Counter),
        "misses": [],
    }


def _record(metrics: dict[str, Any], expected: str, question: str, decision: RouteDecision) -> None:
    metrics["total"] += 1
    metrics["correct"] += int(decision.route == expected)
    metrics["confusion"][expected][str(decision.route)] += 1
    metrics["methods"][decision.method] += 1
    if decision.method in {"hard_guard", "manual"}:
        metrics["hard_guard_count"] += 1
    if decision.llm_router_success is not None:
        metrics["llm_router_calls"] += 1
        metrics["latencies_ms"].append(float(decision.router_latency_ms or 0.0))
    if decision.fallback_used:
        metrics["llm_fallback_count"] += 1
    if decision.llm_error_kind:
        metrics["error_kinds"][decision.llm_error_kind] += 1
    if decision.route != expected:
        metrics["misses"].append(
            {
                "expected": expected,
                "predicted": decision.route,
                "question": question,
                "method": decision.method,
                "reason": decision.reason,
            }
        )


def _finish(metrics: dict[str, Any]) -> dict[str, Any]:
    latencies = sorted(metrics.pop("latencies_ms"))
    error_counts = dict(metrics["error_kinds"])
    total = int(metrics["total"])
    calls = int(metrics["llm_router_calls"])
    fallback = int(metrics["llm_fallback_count"])
    p95 = latencies[max(0, math.ceil(len(latencies) * 0.95) - 1)] if latencies else 0.0
    return {
        **metrics,
        "accuracy": metrics["correct"] / max(1, total),
        "fallback_rate": fallback / max(1, calls),
        "mean_router_latency_ms": sum(latencies) / max(1, len(latencies)),
        "p95_router_latency_ms": p95,
        "invalid_json_count": int(error_counts.get("invalid_json", 0)),
        "empty_response_count": int(error_counts.get("empty_response", 0)),
        "invalid_response_count": sum(
            int(error_counts.get(kind, 0))
            for kind in (
                "invalid_json",
                "missing_field",
                "invalid_boolean",
                "invalid_confidence",
                "invalid_reason",
                "invalid_query",
                "invalid_payload",
            )
        ),
        "error_kinds": error_counts,
        "methods": dict(metrics["methods"]),
        "confusion": {
            expected: {predicted: int(metrics["confusion"][expected][predicted]) for predicted in LABELS}
            for expected in LABELS
        },
    }


def _route(
    question: str,
    *,
    model: str,
    llm_primary: bool,
    dialogue_state: dict[str, Any] | None = None,
) -> RouteDecision:
    return route_question(
        question,
        use_llm_fallback=llm_primary,
        dialogue_state=dialogue_state,
        active_model=model,
    )


def evaluate_router(model: str, *, llm_primary: bool) -> dict[str, Any]:
    overall = _empty_metrics()
    sections: dict[str, dict[str, Any]] = {}

    groups = {
        "golden": [
            (expected, question)
            for expected, questions in CASES_BY_ROUTE.items()
            for question in questions
        ],
        "held_out": [*HELDOUT_SINGLE, *SEMANTIC_HELDOUT_SINGLE],
    }
    for name, rows in groups.items():
        section = _empty_metrics()
        for expected, question in rows:
            decision = _route(question, model=model, llm_primary=llm_primary)
            _record(section, expected, question, decision)
            _record(overall, expected, question, decision)
        sections[name] = _finish(section)

    section = _empty_metrics()
    for scenario in MULTITURN_SCENARIOS:
        state = None
        for question, expected in scenario["turns"]:
            decision = _route(
                question,
                model=model,
                llm_primary=llm_primary,
                dialogue_state=state,
            )
            tagged = f"{scenario['id']}: {question}"
            _record(section, expected, tagged, decision)
            _record(overall, expected, tagged, decision)
            state = decision.dialogue_state
    sections["multi_turn"] = _finish(section)
    sections["overall"] = _finish(overall)
    return sections


def _print_confusion(confusion: dict[str, dict[str, int]]) -> None:
    print("expected \\ predicted")
    print(f"{'':10}" + "".join(f"{label:>9}" for label in LABELS))
    for expected in LABELS:
        print(f"{expected:10}" + "".join(f"{confusion[expected][label]:9d}" for label in LABELS))


def print_result(label: str, result: dict[str, Any]) -> None:
    print(label)
    for name in ("golden", "held_out", "multi_turn", "overall"):
        row = result[name]
        print(
            f"{name:12} {row['correct']:3d}/{row['total']:<3d} "
            f"= {row['accuracy']:.1%}"
        )
    overall = result["overall"]
    printable = {
        "invalid_response_count": 0,
        "empty_response_count": 0,
        **overall,
    }
    print(
        "hard_guard={hard_guard_count} llm_calls={llm_router_calls} "
        "fallback={llm_fallback_count} ({fallback_rate:.1%}) "
        "invalid={invalid_response_count} empty={empty_response_count} "
        "mean={mean_router_latency_ms:.1f}ms p95={p95_router_latency_ms:.1f}ms".format(
            **printable
        )
    )
    print(f"errors={overall['error_kinds']}")
    _print_confusion(overall["confusion"])
    if overall["misses"]:
        print(f"misses ({len(overall['misses'])}):")
        for miss in overall["misses"][:30]:
            print(
                f"  exp={miss['expected']:6} got={miss['predicted']:6} "
                f"[{miss['method']}] {miss['question']}"
            )


def run_comparison(models: list[str], *, warm: bool = True) -> dict[str, Any]:
    installed = available_models()
    results: dict[str, Any] = {
        "rules_only": evaluate_router(models[0], llm_primary=False),
        "llm_primary": {},
        "installed_models": sorted(installed),
    }
    for model in models:
        if model not in installed:
            results["llm_primary"][model] = {"available": False}
            continue
        if warm:
            warm_model(model)
        result = evaluate_router(model, llm_primary=True)
        result["available"] = True
        results["llm_primary"][model] = result
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--no-warm", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--router-timeout", type=float, default=30.0)
    args = parser.parse_args()
    os.environ["ROUTER_TIMEOUT_SECONDS"] = str(args.router_timeout)

    print("Router Evaluation")
    results = run_comparison(args.models, warm=not args.no_warm)
    print()
    print_result("rules_only", results["rules_only"])
    for model in args.models:
        print()
        result = results["llm_primary"].get(model) or {}
        if not result.get("available"):
            print(f"{model}: unavailable")
            continue
        print_result(model, result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
