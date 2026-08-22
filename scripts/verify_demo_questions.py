#!/usr/bin/env python3
"""Re-run the demo handout questions against the current code and score them.

The handout is built from evaluation runs recorded on 2026-08-11 (표) and
2026-08-15 (텍스트). Retrieval and answering code changed on 08-13/08-14, so the
표 side in particular needs a fresh check before anyone demos it.

Scoring reuses the original graders so the numbers stay comparable:
- 표: scripts/run_quality_50.py (needle regex + route + quality label)
- 텍스트: rag/scripts/eval_text_answers_v3.py (completeness + behavior + quality)

Usage:
  .\\.venv\\Scripts\\python.exe scripts/verify_demo_questions.py
  .\\.venv\\Scripts\\python.exe scripts/verify_demo_questions.py --skip-text
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed" / "logs" / "demo_question_verify"
TABLE_SUBSET = OUT_DIR / "table10_questions.jsonl"
TEXT_SUBSET = OUT_DIR / "text10_questions.jsonl"
TABLE_RESULT = OUT_DIR / f"table10_result_{date.today():%Y%m%d}.json"
TABLE_VERDICTS = OUT_DIR / "table_verdicts.json"
TEXT_RESULT_DIR = OUT_DIR / f"text10_{date.today():%Y%m%d}"
NEEDLE_SOURCES = [
    ROOT / "data" / "eval" / "quality_50_open_mix.jsonl",
    ROOT / "data" / "eval" / "balanced_quality_100.jsonl",
]
TEXT_EVAL_SET = ROOT / "data" / "eval" / "pilot_validation_text_v3.jsonl"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def norm(text: str) -> str:
    return " ".join(str(text or "").split())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_subsets(exporter, count: int, table_candidates: int) -> tuple[list[dict], list[dict]]:
    """Materialise the exact handout questions in each grader's input schema.

    table_candidates can exceed the handout size so that replacements for any
    regressed question are verified in the same run.
    """
    needle_rows: dict[str, dict] = {}
    for path in NEEDLE_SOURCES:
        for row in read_jsonl(path):
            needle_rows.setdefault(norm(row["question"]), row)

    table_rows: list[dict] = []
    for item in exporter.pick_table_questions(table_candidates):
        key = norm(item["result"]["question"])
        source = needle_rows.get(key)
        if source is None:
            print(f"WARN no needle contract for: {key[:60]}", flush=True)
            continue
        table_rows.append(source)

    wanted_ids = {item["record"]["question_id"] for item in exporter.pick_text_questions(count)}
    text_rows = [row for row in read_jsonl(TEXT_EVAL_SET) if row["question_id"] in wanted_ids]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, rows in ((TABLE_SUBSET, table_rows), (TEXT_SUBSET, text_rows)):
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
    return table_rows, text_rows


def run_table(rows: list[dict], model: str) -> dict:
    runner = load_module(ROOT / "scripts" / "run_quality_50.py", "run_quality_50")
    return runner._run_one(model, rows, TABLE_RESULT)


def run_text(model: str) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "rag" / "scripts" / "eval_text_answers_v3.py"),
        "--questions",
        str(TEXT_SUBSET),
        "--out-dir",
        str(TEXT_RESULT_DIR),
        "--llm-model",
        model,
    ]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return json.loads((TEXT_RESULT_DIR / "summary.json").read_text(encoding="utf-8"))


def write_table_verdicts() -> None:
    """Record pass/fail per question so the exporter can drop regressed ones."""
    data = json.loads(TABLE_RESULT.read_text(encoding="utf-8"))
    passed, failed, timings = [], [], {}
    for item in data["results"]:
        question = norm(item["question"])
        target = passed if item["needle"] == "PASS" and item["quality"] == "GOOD" else failed
        target.append(question)
        timings[question] = item["dt"]
    TABLE_VERDICTS.write_text(
        json.dumps(
            {
                "verified_on": date.today().isoformat(),
                "passed": passed,
                "failed": failed,
                "seconds": timings,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {TABLE_VERDICTS.name}: pass {len(passed)}, fail {len(failed)}", flush=True)


def report_table() -> None:
    data = json.loads(TABLE_RESULT.read_text(encoding="utf-8"))
    print(f"\n=== 표 {len(data['results'])}문항 재검증 ===", flush=True)
    for item in data["results"]:
        mark = "PASS" if item["needle"] == "PASS" and item["quality"] == "GOOD" else "FAIL"
        print(f"[{mark}] {item['needle']}/{item['quality']} {item['dt']}s route={item['route']}", flush=True)
        print(f"      {norm(item['question'])[:80]}", flush=True)
        if mark == "FAIL":
            print(f"      gold={item['gold']} miss={item['miss']}", flush=True)
    passed = sum(1 for i in data["results"] if i["needle"] == "PASS" and i["quality"] == "GOOD")
    print(f"표 통과 {passed}/{len(data['results'])}", flush=True)


def report_text() -> None:
    records = read_jsonl(TEXT_RESULT_DIR / "records.jsonl")
    print("\n=== 문서 10문항 재검증 ===", flush=True)
    for rec in records:
        ok = (
            rec.get("behavior_pass")
            and not rec.get("forbidden_violations")
            and float(rec.get("completeness") or 0) == 1.0
        )
        print(
            f"[{'PASS' if ok else 'FAIL'}] {rec['question_id']} "
            f"completeness={rec.get('completeness')} quality={rec.get('quality_score')} "
            f"behavior={rec.get('behavior_pass')} {round(float(rec.get('e2e_seconds') or 0), 1)}s",
            flush=True,
        )
        print(f"      {norm(rec['question'])[:80]}", flush=True)
    passed = sum(
        1
        for rec in records
        if rec.get("behavior_pass")
        and not rec.get("forbidden_violations")
        and float(rec.get("completeness") or 0) == 1.0
    )
    print(f"문서 통과 {passed}/{len(records)}", flush=True)


def main() -> int:
    # Windows PowerShell may expose a legacy cp949 console.  Evaluation
    # questions intentionally contain Unicode punctuation, so keep the
    # reporting phase from failing after results have already been written.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument(
        "--table-candidates",
        type=int,
        default=0,
        help="검증할 표 후보 수 (0이면 --count와 동일)",
    )
    parser.add_argument("--model", default="gemma4:12b")
    parser.add_argument("--skip-table", action="store_true")
    parser.add_argument("--skip-text", action="store_true")
    args = parser.parse_args()

    exporter = load_module(ROOT / "scripts" / "export_question_samples_docx.py", "export_samples")
    table_rows, text_rows = build_subsets(exporter, args.count, args.table_candidates or args.count)
    print(f"subsets: 표 {len(table_rows)}문항, 문서 {len(text_rows)}문항", flush=True)

    if not args.skip_table:
        run_table(table_rows, args.model)
        write_table_verdicts()
        report_table()
    if not args.skip_text:
        run_text(args.model)
        report_text()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
