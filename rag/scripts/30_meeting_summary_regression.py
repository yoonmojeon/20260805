"""Regression tests for scoped meeting_summary routing (no LLM)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from fast_claim_extraction import extract_claim_candidates, select_claims_for_question
from fast_question_classifier import classify_fast_question_type
from meeting_summary_context import (
    TARGET_SCOPE_OTHER_BODY_OUTCOME,
    TARGET_SCOPE_SPECIFIC_DOCUMENT,
    TARGET_SCOPE_WHOLE_SESSION,
    get_summary_context,
    meeting_summary_source_tier,
    resolve_meeting_summary_context,
)
from rag_answer_lib import load_unified_collection
from rag_fast_mode import run_fast_retrieval_only

TESTS = [
    {
        "test_id": "A_V02",
        "question": "MSC 111의 주요 결과를 3개 항목으로 요약해줘.",
        "expect_intent": "meeting_summary",
        "expect_scope": TARGET_SCOPE_WHOLE_SESSION,
        "expect_meeting": "MSC 111",
        "primary_doc_contains": "wp.1",
        "forbid_primary_contains": ["111-2-1", "outcome of c 135"],
        "require_claim_topic": "mass",
        "forbid_claim_topics": ["strategic plan", "fal 50"],
    },
    {
        "test_id": "B_C135_A34",
        "question": "C 135와 A 34의 결과를 3개로 요약해줘.",
        "retrieval_sources": ["MSC"],
        "expect_intent": "meeting_outcome_question",
        "expect_scope": TARGET_SCOPE_OTHER_BODY_OUTCOME,
        "primary_doc_contains": "outcome of c 135",
        "must_not_primary_contains": ["wp.1"],
        "require_claim_any": ["c 135", "a 34"],
    },
    {
        "test_id": "C_DOC_111_2_1",
        "question": "MSC 111/2/1 문서 내용을 요약해줘.",
        "expect_intent": "meeting_outcome_question",
        "expect_scope": TARGET_SCOPE_SPECIFIC_DOCUMENT,
        "primary_doc_contains": "111-2",
        "must_not_block": True,
    },
    {
        "test_id": "D_MEPC_ES2",
        "question": "MEPC ES.2 주요 결과를 3개로 요약해줘.",
        "retrieval_sources": ["MEPC"],
        "expect_fast_type": "meeting_summary",
        "expect_scope": TARGET_SCOPE_WHOLE_SESSION,
        "expect_meeting": "MEPC ES.2",
        "must_not_require_mass": True,
        "must_not_primary_contains": ["wp.1", "msc 111"],
    },
]


def _norm(s: str) -> str:
    return (s or "").lower()


def _primary_doc(chunks) -> str:
    if not chunks:
        return ""
    return _norm(chunks[0].file_name or chunks[0].doc_id or "")


def run_test(case: dict, collection, embed, chunks_dir: Path) -> dict:
    sources = case.get("retrieval_sources") or ["MSC"]
    row = {"question": case["question"], "retrieval_sources": sources}
    ctx = resolve_meeting_summary_context(case["question"], row)
    fast_type = classify_fast_question_type(case["question"], row)
    result = run_fast_retrieval_only(row, collection, embed, chunks_dir=chunks_dir)
    chunks = result["retrieved"]
    primary = _primary_doc(chunks)
    claims = extract_claim_candidates(chunks, question=case["question"], row=row)
    selected = select_claims_for_question(claims, case["question"], target_n=3, row=row)
    claim_blob = _norm(" ".join(c.get("claim", "") for c in selected))

    failures: list[str] = []

    if case.get("expect_scope") and ctx.target_scope != case["expect_scope"]:
        failures.append(f"scope={ctx.target_scope} expected={case['expect_scope']}")
    if case.get("expect_meeting") and ctx.target_meeting != case["expect_meeting"]:
        failures.append(f"meeting={ctx.target_meeting} expected={case['expect_meeting']}")
    if case.get("expect_intent") and fast_type != case["expect_intent"]:
        failures.append(f"fast_type={fast_type} expected={case['expect_intent']}")
    if case.get("expect_fast_type") and fast_type != case["expect_fast_type"]:
        failures.append(f"fast_type={fast_type} expected={case['expect_fast_type']}")
    if case.get("primary_doc_contains") and case["primary_doc_contains"] not in primary:
        failures.append(f"primary missing {case['primary_doc_contains']}: {primary[:80]}")
    for bad in case.get("forbid_primary_contains") or []:
        if bad in primary:
            failures.append(f"forbidden primary contains {bad}")
    for bad in case.get("must_not_primary_contains") or []:
        if bad in primary:
            failures.append(f"must_not primary contains {bad}")
    if case.get("require_claim_topic") and case["require_claim_topic"] not in claim_blob:
        failures.append(f"claim missing topic {case['require_claim_topic']}")
    if case.get("require_claim_any") and not any(t in claim_blob for t in case["require_claim_any"]):
        failures.append(f"claim missing any of {case['require_claim_any']}")
    for bad in case.get("forbid_claim_topics") or []:
        if bad in claim_blob:
            failures.append(f"forbidden claim topic {bad}")
    if case.get("must_not_require_mass") and "MASS Code" in ctx.require_topics:
        failures.append("MASS Code should not be required")
    if case.get("must_not_block"):
        tier = meeting_summary_source_tier(chunks[0].file_name or "", doc_id=chunks[0].doc_id or "", ctx=ctx)
        if tier >= 3:
            failures.append(f"doc incorrectly penalized tier={tier}")

    return {
        "test_id": case["test_id"],
        "passed": not failures,
        "failures": failures,
        "target_scope": ctx.target_scope,
        "target_meeting": ctx.target_meeting,
        "fast_type": fast_type,
        "primary_doc": chunks[0].file_name if chunks else None,
        "selected_claims": [c.get("claim", "")[:100] for c in selected],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Meeting summary regression (retrieval only).")
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/index"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/logs/meeting_summary_regression.json"))
    args = parser.parse_args()

    collection, embed, _ = load_unified_collection("full_corpus_715_v1", args.index_dir)
    results = [run_test(t, collection, embed, args.chunks_dir) for t in TESTS]
    passed = sum(1 for r in results if r["passed"])
    summary = {"passed": passed, "total": len(results), "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if passed < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
