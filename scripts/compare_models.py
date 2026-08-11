#!/usr/bin/env python3
"""Fair router + end-to-end comparison for the three selectable Ollama models."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.llm_models import AVAILABLE_LLM_MODELS, DEFAULT_LLM_MODEL
from tests.run_router_eval import print_result, run_comparison


def _safe_model(model: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model)


def _run_quality(model: str, *, questions: Path | None = None) -> dict[str, Any]:
    output = ROOT / "data" / "eval" / f"quality_30_{_safe_model(model)}.json"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_quality_30.py"),
        "--model",
        model,
        "--output",
        str(output),
    ]
    if questions is not None:
        command.extend(["--questions", str(questions)])
    env = os.environ.copy()
    env["MODEL_NAME"] = model
    env["MARITIME_OLLAMA_MODEL"] = model
    env["MARITIME_OPS_DETERMINISTIC_SHORTCUTS"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        stripped = line.lstrip()
        if not stripped.startswith(
            (
                "[PASS",
                "[FAIL",
                "[WEAK",
                "TYPE ",
                "SUMMARY ",
                "wrote ",
                "model=",
                "warming",
                "Warning:",
                "gold=",
            )
        ):
            continue
        rendered = f"[{model}] {line}"
        console_encoding = sys.stdout.encoding or "utf-8"
        rendered = rendered.encode(console_encoding, errors="replace").decode(
            console_encoding, errors="replace"
        )
        print(rendered, end="", flush=True)
    return_code = process.wait()
    if not output.exists():
        return {
            "summary": {
                "model": model,
                "failure_count": 1,
                "qa_pass": 0,
                "qa_pass_rate": 0.0,
                "runner_return_code": return_code,
            },
            "results": [],
        }
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload.setdefault("summary", {})["runner_return_code"] = return_code
    return payload


def _ranking_key(model: str, router: dict[str, Any], quality: dict[str, Any]) -> tuple:
    summary = quality.get("summary") or {}
    overall = router.get("overall") or {}
    held = router.get("held_out") or {}
    multi = router.get("multi_turn") or {}
    n = max(1, int(summary.get("n") or 0))
    failure_rate = float(summary.get("failure_count") or 0) / n
    return (
        float(summary.get("qa_pass_rate") or 0.0),
        float(held.get("accuracy") or 0.0),
        float(overall.get("accuracy") or 0.0),
        -failure_rate,
        float(multi.get("accuracy") or 0.0),
        -float(summary.get("avg_dt") or 0.0),
    )


def _select_winner(
    models: list[str],
    router_results: dict[str, Any],
    quality_results: dict[str, Any],
) -> tuple[str | None, str, bool]:
    eligible = [
        model
        for model in models
        if (router_results.get(model) or {}).get("available")
        and (quality_results.get(model) or {}).get("summary")
    ]
    if not eligible:
        return None, "평가 가능한 모델이 없습니다.", False
    ranked = sorted(
        eligible,
        key=lambda model: _ranking_key(
            model, router_results[model], quality_results[model]
        ),
        reverse=True,
    )
    winner = ranked[0]
    if len(ranked) == 1:
        return winner, "유일하게 전체 평가가 완료된 모델입니다.", True

    top = quality_results[winner]["summary"]
    second = quality_results[ranked[1]]["summary"]
    qa_gap = int(top.get("qa_pass") or 0) - int(second.get("qa_pass") or 0)
    top_router = router_results[winner]["overall"]
    second_router = router_results[ranked[1]]["overall"]
    failure_gap = int(second.get("failure_count") or 0) - int(top.get("failure_count") or 0)
    held_gap = float(router_results[winner]["held_out"]["accuracy"]) - float(
        router_results[ranked[1]]["held_out"]["accuracy"]
    )
    clear = qa_gap >= 2 or (
        qa_gap == 0
        and (
            failure_gap >= 2
            or held_gap >= 0.05
            or float(top_router.get("accuracy") or 0.0)
            - float(second_router.get("accuracy") or 0.0)
            >= 0.05
        )
    )
    reason = (
        f"end-to-end QA {top.get('qa_pass')}/{top.get('n')}, "
        f"held-out {router_results[winner]['held_out']['accuracy']:.1%}, "
        f"overall router {top_router.get('accuracy', 0):.1%}, "
        f"failures {top.get('failure_count', 0)}, mean E2E {top.get('avg_dt', 0)}s."
    )
    if not clear:
        reason += " 상위 모델 간 차이가 작아 기존 default를 자동 변경할 정도로 명확하지 않습니다."
    return winner, reason, clear


def _summary_row(model: str, router: dict[str, Any], quality: dict[str, Any]) -> dict[str, Any]:
    summary = quality.get("summary") or {}
    overall = router.get("overall") or {}
    return {
        "model": model,
        "router_accuracy": overall.get("accuracy"),
        "router_heldout_accuracy": (router.get("held_out") or {}).get("accuracy"),
        "multi_turn_accuracy": (router.get("multi_turn") or {}).get("accuracy"),
        "end_to_end_qa_pass": summary.get("qa_pass"),
        "end_to_end_qa_total": summary.get("n"),
        "end_to_end_qa_pass_rate": summary.get("qa_pass_rate"),
        "route_mismatch_count": summary.get("route_mismatch_count"),
        "invalid_empty_answer_count": summary.get("invalid_empty_answer_count"),
        "failure_count": summary.get("failure_count"),
        "router_fallback_rate": overall.get("fallback_rate"),
        "router_invalid_response_count": overall.get("invalid_response_count", 0),
        "router_empty_response_count": overall.get("empty_response_count", 0),
        "mean_router_latency_ms": overall.get("mean_router_latency_ms"),
        "p95_router_latency_ms": overall.get("p95_router_latency_ms"),
        "mean_end_to_end_latency_seconds": summary.get("avg_dt"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(AVAILABLE_LLM_MODELS))
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--skip-slow", action="store_true", help="router-only comparison")
    parser.add_argument(
        "--router-results",
        type=Path,
        help="reuse a completed tests/run_router_eval.py JSON instead of rerunning it",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "eval" / "model_comparison.json",
    )
    parser.add_argument("--router-timeout", type=float, default=30.0)
    args = parser.parse_args()
    os.environ["ROUTER_TIMEOUT_SECONDS"] = str(args.router_timeout)

    print("Experiment A: rules_only vs llm_primary", flush=True)
    if args.router_results:
        router_block = json.loads(args.router_results.read_text(encoding="utf-8"))
        missing = [
            model
            for model in args.models
            if model not in (router_block.get("llm_primary") or {})
        ]
        if missing:
            parser.error(f"router results do not contain models: {', '.join(missing)}")
    else:
        router_block = run_comparison(args.models, warm=True)
    print_result("rules_only", router_block["rules_only"])
    for model in args.models:
        result = router_block["llm_primary"].get(model) or {}
        if result.get("available"):
            print_result(model, result)
        else:
            print(f"{model}: unavailable")

    quality: dict[str, Any] = {}
    if not args.skip_slow:
        print("\nExperiment B: end-to-end model comparison", flush=True)
        for model in args.models:
            if (router_block["llm_primary"].get(model) or {}).get("available"):
                quality[model] = _run_quality(model, questions=args.questions)

    winner, reason, clear = _select_winner(
        args.models, router_block["llm_primary"], quality
    )
    rows = [
        _summary_row(model, router_block["llm_primary"][model], quality[model])
        for model in args.models
        if model in quality and (router_block["llm_primary"].get(model) or {}).get("available")
    ]
    payload = {
        "selection_rule": "end_to_end_quality_first",
        "evaluated_models": args.models,
        "existing_default": DEFAULT_LLM_MODEL,
        "models": {row["model"]: row for row in rows},
        "winner": winner,
        "clear_winner": clear,
        "recommended_default": winner if clear else DEFAULT_LLM_MODEL,
        "selection_reason": reason,
        "experiment_a": {
            "rules_only": router_block["rules_only"],
            "llm_primary": router_block["llm_primary"],
        },
        "experiment_b": quality,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nModel comparison")
    print(
        f"{'Model':20} {'Route':>8} {'Heldout':>8} {'Multi':>8} "
        f"{'QA':>8} {'Fail':>6} {'R-fb':>8} {'R-ms':>9} {'E2E-s':>8}"
    )
    for row in rows:
        qa = f"{row['end_to_end_qa_pass']}/{row['end_to_end_qa_total']}"
        print(
            f"{row['model']:20} {row['router_accuracy']:8.1%} "
            f"{row['router_heldout_accuracy']:8.1%} {row['multi_turn_accuracy']:8.1%} "
            f"{qa:>8} {int(row['failure_count'] or 0):6d} "
            f"{float(row['router_fallback_rate'] or 0):8.1%} "
            f"{float(row['mean_router_latency_ms'] or 0):9.1f} "
            f"{float(row['mean_end_to_end_latency_seconds'] or 0):8.1f}"
        )
    print(f"\nRecommended default model: {winner or 'none'}")
    print(f"Default to configure: {payload['recommended_default']}")
    print(f"Reason: {reason}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
