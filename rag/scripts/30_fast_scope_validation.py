"""Validate Fast mode document scope correction and IMO terminology."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from embedding_policy import DEFAULT_EMBEDDING_PRESET
from fast_imo_terms import detect_terminology_violations
from rag_answer_lib import DEFAULT_OLLAMA_MODEL, load_unified_collection
from rag_eval_lib import load_questions
from rag_inprocess import DEFAULT_INDEX_DIR, run_full_inprocess
from rag_resource_cache import warm_all_resources


def _check_patterns(text: str, patterns: list[str], *, forbid: bool) -> tuple[bool, list[str]]:
    hits = [p for p in patterns if p.lower() in text.lower()]
    if forbid:
        return len(hits) == 0, hits
    return len(hits) > 0, hits


def evaluate_answer(row: dict, answer: str) -> dict:
    checks: dict[str, bool] = {}
    details: dict[str, list] = {}

    if row.get("expected_scope_correction"):
        ok = row["expected_scope_correction"].lower() in answer.lower()
        checks["scope_correction"] = ok
        details["scope_correction"] = [row["expected_scope_correction"]] if ok else []

    ok, hits = _check_patterns(answer, row.get("forbid_patterns") or [], forbid=True)
    checks["no_forbidden_terms"] = ok
    details["forbidden_hits"] = hits

    ok, hits = _check_patterns(answer, row.get("require_patterns") or [], forbid=False)
    checks["required_patterns"] = ok
    details["required_hits"] = hits

    any_pats = row.get("require_any_patterns") or []
    if any_pats:
        checks["require_any"] = any(p.lower() in answer.lower() for p in any_pats)
        details["require_any_hits"] = [p for p in any_pats if p.lower() in answer.lower()]

    numbered = len(re.findall(r"^\s*\d+\.", answer, re.M))
    target = int(row.get("outcome_item_count") or 3)
    checks["item_count"] = numbered == target

    violations = detect_terminology_violations(answer)
    checks["no_terminology_violations"] = len(violations) == 0
    details["terminology_violations"] = violations

    checks["pass"] = all(checks.values())
    return {"checks": checks, "details": details}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast scope/terminology validation.")
    parser.add_argument("--questions", type=Path, default=Path("data/eval/fast_scope_tests.jsonl"))
    parser.add_argument("--collection-id", type=str, default="full_corpus_715_v1")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--llm-model", type=str, default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-base", type=str, default="http://localhost:11434")
    parser.add_argument("--compare-accurate", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("data/processed/logs/fast_scope_validation.json"))
    args = parser.parse_args()

    questions = load_questions(args.questions)
    warm_all_resources(args.collection_id, args.index_dir, args.chunks_dir)

    results: list[dict] = []
    for row in questions:
        entry: dict = {"question_id": row.get("question_id"), "question": row.get("question")}

        t0 = time.time()
        fast_out = run_full_inprocess(
            row,
            unified_id=args.collection_id,
            index_dir=args.index_dir,
            chunks_dir=args.chunks_dir,
            llm_model=args.llm_model,
            ollama_base=args.ollama_base,
            latency_mode="fast",
            start_type="warm",
        )
        fast_answer = fast_out.get("answer") or ""
        fast_tm = fast_out.get("timing_metrics") or {}
        entry["fast"] = {
            "answer": fast_answer,
            "evaluation": evaluate_answer(row, fast_answer),
            "first_visible_latency": fast_tm.get("e2e_ttft") or fast_tm.get("llm_ttft"),
            "ttft": fast_tm.get("llm_ttft"),
            "total_generation_time": fast_tm.get("llm_generation_time"),
            "e2e_total": fast_tm.get("e2e_total"),
            "trace_log": "data/processed/logs/fast_mode_answer_trace.jsonl",
            "prompt_meta": (fast_out.get("answer_out") or {}).get("prompt_meta", {}),
        }

        if args.compare_accurate:
            acc_out = run_full_inprocess(
                row,
                unified_id=args.collection_id,
                index_dir=args.index_dir,
                chunks_dir=args.chunks_dir,
                llm_model=args.llm_model,
                ollama_base=args.ollama_base,
                latency_mode="accurate",
                start_type="warm",
                top_k=10,
                fetch_k=120,
                use_rerank=True,
            )
            acc_answer = acc_out.get("answer") or ""
            acc_tm = acc_out.get("timing_metrics") or {}
            entry["accurate"] = {
                "answer_preview": acc_answer[:500],
                "evaluation": evaluate_answer(row, acc_answer),
                "e2e_total": acc_tm.get("e2e_total"),
                "llm_ttft": acc_tm.get("llm_ttft"),
            }

        results.append(entry)

    passed = sum(1 for r in results if r.get("fast", {}).get("evaluation", {}).get("checks", {}).get("pass"))
    summary = {
        "question_count": len(questions),
        "fast_pass_count": passed,
        "fast_pass_rate": round(passed / max(len(questions), 1), 3),
    }
    payload = {"summary": summary, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
