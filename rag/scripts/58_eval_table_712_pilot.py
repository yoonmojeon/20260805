"""Run UTF-8 retrieval/answer regression cases for the table 7.1.2 UI pilot."""
from __future__ import annotations

import json
from pathlib import Path

from table_712_pilot import answer_table_712

ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "data/eval/table_712_dense_eval.json"


def main() -> None:
    rows = []
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    for case in cases:
        question, gold_cell, expected = case["question"], case["gold_cell"], case["expected_text"]
        result = answer_table_712(question)
        evidence = result.get("evidence") or [{}]
        selected = evidence[0].get("cell_id")
        answer = result.get("answer", "")
        rows.append({"question": question, "gold_cell": gold_cell, "selected_cell": selected,
                     "top1_distance": result["dense_hits"][0]["distance"], "top1_margin": result["top1_margin"],
                     "expected_text": expected, "answer": answer,
                     "cell_pass": selected == gold_cell, "answer_pass": expected in answer,
                     "passed": selected == gold_cell and expected in answer})
    report = {"method": "dense_top1_no_cell_override", "eval_path": str(EVAL_PATH.resolve()),
              "passed": sum(r["passed"] for r in rows), "total": len(rows), "cases": rows}
    output = ROOT / "data/processed/index/unified_kr_table_7_1_2_restored_pilot/query_validation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "passed": report["passed"], "total": report["total"],
                      "cell_accuracy": sum(r["cell_pass"] for r in rows) / len(rows),
                      "answer_accuracy": sum(r["answer_pass"] for r in rows) / len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
