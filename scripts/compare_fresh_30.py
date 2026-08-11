#!/usr/bin/env python3
"""Compare rules-only and three same-model LLM-primary pipelines on a fresh 30-set."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ("llama3.1:8b", "gemma4:12b", "mistral-nemo:12b")


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _emit(label: str, line: str) -> None:
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
        return
    rendered = f"[{label}] {line}"
    encoding = sys.stdout.encoding or "utf-8"
    rendered = rendered.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(rendered, end="", flush=True)


def _run(
    *,
    label: str,
    model: str,
    questions: Path,
    rules_only: bool,
    output_dir: Path,
) -> dict[str, Any]:
    output = output_dir / f"fresh_30_{_safe(label)}.json"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_quality_30.py"),
        "--questions",
        str(questions),
        "--model",
        model,
        "--output",
        str(output),
    ]
    if rules_only:
        command.append("--rules-only")
    env = os.environ.copy()
    env.update(
        {
            "MODEL_NAME": model,
            "MARITIME_OLLAMA_MODEL": model,
            "MARITIME_OPS_DETERMINISTIC_SHORTCUTS": "0",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
    )
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
        _emit(label, line)
    return_code = process.wait()
    if not output.exists():
        raise RuntimeError(f"{label} did not produce {output} (exit={return_code})")
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["summary"]["runner_return_code"] = return_code
    payload["summary"]["pipeline"] = label
    payload["summary"]["answer_model"] = model
    return payload


def _summary_row(label: str, payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    results = payload["results"]
    return {
        "pipeline": label,
        "routing_mode": summary["routing_mode"],
        "answer_model": summary["answer_model"],
        "total": summary["n"],
        "route_correct": sum(int(row["route_ok"]) for row in results),
        "route_accuracy": sum(int(row["route_ok"]) for row in results) / max(1, len(results)),
        "qa_pass": summary["qa_pass"],
        "qa_pass_rate": summary["qa_pass_rate"],
        "needle": summary["needle"],
        "quality": summary["quality"],
        "route_mismatch_count": summary["route_mismatch_count"],
        "invalid_empty_answer_count": summary["invalid_empty_answer_count"],
        "failure_count": summary["failure_count"],
        "router_fallback_count": summary["router_fallback_count"],
        "router_fallback_rate": summary["router_fallback_rate"],
        "mean_router_latency_ms": summary["mean_router_latency_ms"],
        "mean_end_to_end_latency_seconds": summary["avg_dt"],
        "types": summary["types"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data" / "eval" / "quality_30_fresh_all_types.jsonl",
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "eval" / "quality_30_fresh_comparison.json",
    )
    args = parser.parse_args()
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    pipelines: dict[str, dict[str, Any]] = {}
    for model in args.models:
        baseline_label = f"rules_only+{model}"
        pipelines[baseline_label] = _run(
            label=baseline_label,
            model=model,
            questions=args.questions,
            rules_only=True,
            output_dir=output_dir,
        )
    for model in args.models:
        label = f"llm_primary+{model}"
        pipelines[label] = _run(
            label=label,
            model=model,
            questions=args.questions,
            rules_only=False,
            output_dir=output_dir,
        )

    rows = [_summary_row(label, payload) for label, payload in pipelines.items()]
    comparison = {
        "dataset": str(args.questions),
        "baseline_definition": (
            "For each answer model, compare the rules-only router with the same-model "
            "LLM-primary router and answer generation pipeline."
        ),
        "pipelines": {row["pipeline"]: row for row in rows},
        "result_files": {
            label: str(output_dir / f"fresh_30_{_safe(label)}.json") for label in pipelines
        },
    }
    args.output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nFresh 30 comparison")
    print(f"{'Pipeline':34} {'Route':>8} {'QA':>8} {'Bad':>6} {'Mismatch':>9} {'R-ms':>9} {'E2E-s':>8}")
    for row in rows:
        print(
            f"{row['pipeline']:34} {row['route_accuracy']:8.1%} "
            f"{row['qa_pass']:>2}/{row['total']:<2} "
            f"{int((row['quality'] or {}).get('BAD', 0)):6d} "
            f"{row['route_mismatch_count']:9d} "
            f"{row['mean_router_latency_ms']:9.1f} "
            f"{row['mean_end_to_end_latency_seconds']:8.1f}"
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
