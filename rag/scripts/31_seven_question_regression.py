"""Run the seven UI example questions through the real in-process RAG pipeline.

This is a regression harness, not an answer fixture: questions are fixed, answers are not.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rag_answer_lib import load_unified_collection
from rag_inprocess import run_full_inprocess
from seven_question_cases import QUESTIONS as CANONICAL_QUESTIONS


LEGACY_QUESTIONS = [
    ("q1", "환경규제 대응과 관련된 최신 MEPC 회의 주요 내용을 정리해줘.", ["MEPC"]),
    ("q2", "MSC 111의 주요 결과를 3개 항목으로 요약해줘.", ["MSC"]),
    ("q3", "최신 MEPC 회의에서 선박 운항 및 규제 보고에 직접 영향을 주는 사항을 정리해줘.", ["MEPC"]),
    ("q4", "MSC 111에서 대체연료·GHG 안전규제와 관련된 논의 및 결론을 요약해줘.", ["MSC"]),
    (
        "q5",
        "MSC 111에서 MASS Code와 관련된 핵심 결정사항을 요약하고, 향후 mandatory code 일정까지 정리해줘.",
        ["MSC"],
    ),
    ("q6", "DNV에서 자율운항 또는 Smart Vessel 관련 Rule/Guidance를 찾아줘.", ["DNV"]),
    ("q7", "LR에서 대체연료 관련 Rule/Guidance를 찾아줘.", ["LR"]),
]

# Keep old log parsing stable, but run the real Korean UI questions.
QUESTIONS = CANONICAL_QUESTIONS


def _evidence_row(row: dict) -> dict:
    return {
        "citation_id": row.get("citation_id"),
        "file_name": row.get("file_name"),
        "page": row.get("page"),
        "chunk_id": row.get("chunk_id"),
        "evidence": (
            row.get("chunk_text")
            or row.get("text")
            or row.get("content_preview")
            or row.get("preview")
            or ""
        )[:500],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=[q[0] for q in QUESTIONS])
    parser.add_argument("--out", type=Path, default=Path("data/processed/logs/seven_question_regression.json"))
    parser.add_argument("--with-llm", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("RAG_DEBUG_TRACE_STDERR", "0")

    index_dir = Path("data/processed/index")
    chunks_dir = Path("data/processed/chunks")
    collection, embed_model, manifest = load_unified_collection("full_corpus_715_v1", index_dir)
    results = []
    for question_id, question, sources in QUESTIONS:
        if args.only and question_id != args.only:
            continue
        output = run_full_inprocess(
            {"question_id": question_id, "question": question, "retrieval_sources": sources},
            collection=collection,
            embed_model=embed_model,
            manifest=manifest,
            index_dir=index_dir,
            chunks_dir=chunks_dir,
            latency_mode="accurate",
            skip_llm=not args.with_llm,
            skip_ollama_probe=not args.with_llm,
        )
        search = output.get("search_out") or {}
        answer_out = output.get("answer_out") or {}
        answer = output.get("answer") or answer_out.get("answer") or search.get("answer") or ""
        log = output.get("timing_log") or {}
        fast_meta = (
            search.get("fast_meta")
            or (search.get("retrieval_config") or {}).get("fast_meta")
            or (log.get("retrieval_config") or {}).get("fast_meta")
            or {}
        )
        results.append(
            {
                "question_id": question_id,
                "question": question,
                "answer_mode": output.get("answer_mode"),
                "answer": answer,
                "prompt_meta": answer_out.get("prompt_meta"),
                "verification_summary": answer_out.get("verification_summary"),
                "evidence_completion": fast_meta.get("evidence_completion"),
                "evidence_table": [
                    _evidence_row(row) for row in answer_out.get("evidence_table") or []
                ],
                "retrieved": [
                    {
                        "file_name": getattr(chunk, "file_name", ""),
                        "page": getattr(chunk, "page_number", None),
                        "chunk_id": getattr(chunk, "chunk_id", ""),
                        "text": getattr(chunk, "text", "")[:500],
                    }
                    for chunk in search.get("retrieved") or []
                ],
                "timing_metrics": output.get("timing_metrics"),
            }
        )
        print(
            json.dumps(
                {
                    "question_id": question_id,
                    "answer": answer,
                    "evidence_completion": fast_meta.get("evidence_completion"),
                    "retrieved": [
                        [getattr(c, "file_name", ""), getattr(c, "page_number", None)]
                        for c in search.get("retrieved") or []
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
