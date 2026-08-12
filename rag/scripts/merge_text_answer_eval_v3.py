"""Merge targeted v3 re-evaluations into one reproducible 405-row snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_text_answers_v3 import aggregate, load_jsonl, report_markdown, row_score


ROOT = Path(__file__).resolve().parents[2]


def load_result_records(path: Path) -> list[dict]:
    """Load resumable JSONL, ignoring a final line cut by process termination."""
    records: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"warning: skipped malformed JSONL line {path}:{line_number}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=ROOT / "data/eval/pilot_validation_text_v3.jsonl")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--replace-dir", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    questions = load_jsonl(args.questions)
    question_by_id = {str(row["question_id"]): row for row in questions}
    records = {
        str(row["question_id"]): row
        for row in load_result_records(args.base_dir / "records.jsonl")
    }
    provenance = {question_id: str(args.base_dir) for question_id in records}
    for directory in args.replace_dir:
        for row in load_result_records(directory / "records.jsonl"):
            question_id = str(row["question_id"])
            records[question_id] = row
            provenance[question_id] = str(directory)

    ordered: list[dict] = []
    for question in questions:
        question_id = str(question["question_id"])
        if question_id not in records:
            raise SystemExit(f"missing result: {question_id}")
        record = dict(records[question_id])
        if not record.get("error"):
            record.update(row_score(question_by_id[question_id], str(record.get("answer") or "")))
        record["result_source"] = provenance[question_id]
        ordered.append(record)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "records.jsonl").open("w", encoding="utf-8") as stream:
        for row in ordered:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = aggregate(ordered)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "report.md").write_text(report_markdown(summary), encoding="utf-8")
    (args.out_dir / "provenance.json").write_text(
        json.dumps(
            {
                "base_dir": str(args.base_dir),
                "replacement_dirs": [str(path) for path in args.replace_dir],
                "replacement_count": sum(
                    1 for value in provenance.values() if value != str(args.base_dir)
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
