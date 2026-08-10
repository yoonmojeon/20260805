#!/usr/bin/env python3
"""Evaluate open retrieval and optional Ollama answers for KR7 table questions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from kr7_expanded_pilot import answer_kr7_tables, rank_kr7_tables


def compact(text: str) -> str:
    return "".join(str(text).lower().split()).replace("×", "x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data/eval/kr7_expanded_table_qa_28.json",
    )
    parser.add_argument("--answers", action="store_true")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--ollama-base", default="http://127.0.0.1:11434")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--qid", action="append", help="Optional repeatable case filter")
    args = parser.parse_args()

    cases = json.loads(args.questions.read_text(encoding="utf-8"))
    if args.qid:
        wanted = set(args.qid)
        cases = [case for case in cases if case["qid"] in wanted]
    results = []
    for index, case in enumerate(cases, 1):
        hits = rank_kr7_tables(case["question"], top_k=args.top_k)
        ranked_tables = []
        for hit in hits:
            table_id = hit["metadata"].get("table_id")
            if table_id and table_id not in ranked_tables:
                ranked_tables.append(table_id)
        gold = set(case["gold_table_ids"])
        row = {
            "qid": case["qid"],
            "question": case["question"],
            "gold_table_ids": case["gold_table_ids"],
            "ranked_table_ids": ranked_tables,
            "table_hit_at_1": bool(ranked_tables and ranked_tables[0] in gold),
            "table_hit_at_3": any(table_id in gold for table_id in ranked_tables[:3]),
            "table_hit_at_5": any(table_id in gold for table_id in ranked_tables[:5]),
            "top_chunk_id": hits[0]["chunk_id"] if hits else None,
            "top_distance": hits[0]["distance"] if hits else None,
        }
        if args.answers:
            answer_result = answer_kr7_tables(
                case["question"],
                top_k=args.top_k,
                model=args.model,
                ollama_base=args.ollama_base,
            )
            answer = answer_result["answer"]
            normalized = compact(answer)
            expected = case.get("expected_any", [])
            matched = [token for token in expected if compact(token) in normalized]
            required = (
                len(expected)
                if case.get("required_all")
                else max(1, math.ceil(len(expected) / 2))
            )
            row.update(
                {
                    "answer": answer,
                    "expected_tokens": expected,
                    "matched_tokens": matched,
                    "answer_token_coverage": round(len(matched) / len(expected), 4) if expected else 1.0,
                    "answer_pass": len(matched) >= required,
                    "evidence": answer_result["evidence"],
                }
            )
        results.append(row)
        print(
            f"[{index:02d}/{len(cases)}] {case['qid']} "
            f"R1={row['table_hit_at_1']} R3={row['table_hit_at_3']}"
            + (f" answer={row.get('answer_pass')}" if args.answers else ""),
            flush=True,
        )

    count = len(results)
    summary = {
        "questions": count,
        "table_recall_at_1": round(sum(row["table_hit_at_1"] for row in results) / count, 4),
        "table_recall_at_3": round(sum(row["table_hit_at_3"] for row in results) / count, 4),
        "table_recall_at_5": round(sum(row["table_hit_at_5"] for row in results) / count, 4),
        "answers_generated": args.answers,
    }
    if args.answers:
        summary["answer_token_pass_rate"] = round(
            sum(row["answer_pass"] for row in results) / count, 4
        )
        summary["joint_r3_answer_pass_rate"] = round(
            sum(row["table_hit_at_3"] and row["answer_pass"] for row in results) / count,
            4,
        )
    stem = (
        "kr7_expanded_qa_selected"
        if args.qid
        else args.questions.stem
    )
    output = ROOT / "data/processed/logs" / (
        f"{stem}_with_answers.json" if args.answers else f"{stem}_retrieval.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": summary, "cases": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({**summary, "output": str(output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
