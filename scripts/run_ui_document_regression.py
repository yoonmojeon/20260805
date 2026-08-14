#!/usr/bin/env python3
"""Run the ten user-facing document QA regressions through the real UI service path."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "rag" / "scripts"), str(ROOT / "ops")]


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _group_hits(text: str, groups: list[list[str]]) -> list[bool]:
    lowered = (text or "").casefold()
    return [any(alias.casefold() in lowered for alias in group) for group in groups]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data" / "eval" / "ui_document_regression_10.jsonl",
    )
    parser.add_argument("--model", default="gemma4:12b")
    parser.add_argument("--latency-mode", default="fast")
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--full-answer", action="store_true")
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Print retrieval/evidence-plan diagnostics without generating answers.",
    )
    args = parser.parse_args()

    from services.orchestrator import handle_question
    from services.retrieval_mode import classify_retrieval_mode

    selected = set(args.ids or [])
    rows = [row for row in _load(args.questions) if not selected or row["id"] in selected]
    results: list[dict] = []
    for row in rows:
        question = row["question"]
        if args.search_only:
            from rag.scripts.rag_inprocess import run_search_inprocess

            search_row = {"question": question, "question_id": row["id"]}
            search = run_search_inprocess(
                search_row,
                latency_mode=args.latency_mode,
                use_rerank=False,
            )
            fast_meta = ((search.get("retrieval_config") or {}).get("fast_meta") or {})
            completion = fast_meta.get("evidence_completion") or {}
            print(
                "SEARCH",
                json.dumps(
                    {
                        "id": row["id"],
                        "answer_mode": search.get("answer_mode"),
                        "evidence_completion": completion,
                        "retrieved": [
                            {
                                "chunk_id": getattr(chunk, "chunk_id", ""),
                                "file_name": getattr(chunk, "file_name", ""),
                                "page": getattr(chunk, "page_number", None),
                            }
                            for chunk in (search.get("retrieved") or [])
                        ],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if search.get("answer_mode") == "structured_meeting":
                from meeting_category_profile import build_meeting_retrieval_profile
                from meeting_structured_answer import build_meeting_structured_answer

                search_row["_evidence_completion"] = completion
                legacy = str(search.get("question_category") or "env_regulation")
                profile = build_meeting_retrieval_profile(
                    question, search_row, legacy_category=legacy
                )
                raw_answer, raw_warnings, raw_meta = build_meeting_structured_answer(
                    list(search.get("retrieved") or []),
                    question=question,
                    row=search_row,
                    profile=profile,
                )
                print(
                    "RAW_STRUCTURED",
                    json.dumps(
                        {
                            "answer": raw_answer,
                            "warnings": raw_warnings,
                            "claim_verification": raw_meta.get("claim_verification") or [],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            continue
        route_mode = classify_retrieval_mode(question).value
        started = time.perf_counter()
        out = handle_question(
            question,
            force_route="rag",
            use_llm_router=False,
            rag_latency_mode=args.latency_mode,
            llm_model=args.model,
        )
        latency = time.perf_counter() - started
        answer = str(out.get("answer") or "")
        required = _group_hits(answer, row.get("required_groups") or [])
        forbidden = _group_hits(answer, row.get("forbidden_groups") or [])
        mode_ok = route_mode == row["expect_retrieval"] or (
            row["expect_retrieval"] == "table" and route_mode in {"table", "both"}
        )
        evidence = list(out.get("evidence_table") or [])
        evidence_text = " ".join(
            str(item.get(key) or "")
            for item in evidence
            for key in ("file_name", "document", "page", "page_number", "chunk_id")
        ).casefold()
        doc_hits = [doc.casefold() in evidence_text for doc in row.get("gold_documents") or []]
        ok = mode_ok and all(required) and not any(forbidden) and all(doc_hits)
        result = {
            "id": row["id"],
            "ok": ok,
            "retrieval_mode": route_mode,
            "required": required,
            "forbidden": forbidden,
            "gold_document_hits": doc_hits,
            "evidence_rows": len(evidence),
            "latency_s": round(latency, 2),
            "answer_mode": (out.get("meta") or {}).get("answer_mode"),
        }
        results.append(result)
        print("RESULT", json.dumps(result, ensure_ascii=False), flush=True)
        preview = answer if args.full_answer else answer[:1000]
        print("ANSWER", preview.replace("\n", "\\n"), flush=True)
        print(
            "EVIDENCE",
            json.dumps(
                [
                    {
                        key: item.get(key)
                        for key in ("file_name", "document", "page", "page_number", "chunk_id")
                        if item.get(key) not in (None, "")
                    }
                    for item in evidence[:8]
                ],
                ensure_ascii=False,
            ),
            flush=True,
        )

    if args.search_only:
        return 0
    passed = sum(result["ok"] for result in results)
    mean_latency = round(sum(result["latency_s"] for result in results) / max(1, len(results)), 2)
    print(
        "SUMMARY",
        json.dumps(
            {"passed": passed, "total": len(results), "mean_latency_s": mean_latency},
            ensure_ascii=False,
        ),
    )
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
