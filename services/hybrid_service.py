"""Hybrid path — run ops and rag, then merge with source labels."""
from __future__ import annotations

from typing import Any

from services.ops_service import run_ops_query
from services.rag_service import run_rag_query


def merge_hybrid_answers(ops_answer: str, rag_answer: str) -> str:
    ops_text = (ops_answer or "").strip() or "(운항 데이터 답을 만들지 못했습니다.)"
    rag_text = (rag_answer or "").strip() or "(문서 검색 답을 만들지 못했습니다.)"
    return (
        "운항 숫자와 규정/회의 문서를 나눠 정리했습니다.\n\n"
        "## 운항 데이터 (ops)\n\n"
        f"{ops_text}\n\n"
        "---\n\n"
        "## 규정·회의 문서 (rag)\n\n"
        f"{rag_text}\n"
    )


def run_hybrid_query(
    question: str,
    history: list | None = None,
    *,
    rag_latency_mode: str = "fast",
    table_qa: bool = False,
) -> dict[str, Any]:
    ops_result = run_ops_query(question, history)
    rag_result = run_rag_query(
        question, latency_mode=rag_latency_mode, table_qa=table_qa
    )
    answer = merge_hybrid_answers(
        str(ops_result.get("answer") or ""),
        str(rag_result.get("answer") or ""),
    )
    hist = list(history or [])
    hist.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )
    files = list(ops_result.get("files") or []) + list(rag_result.get("files") or [])
    return {
        "answer": answer,
        "history": hist,
        "files": files,
        "map_html": ops_result.get("map_html") or "",
        "source": "hybrid",
        "meta": {
            "ops": ops_result.get("meta") or {"source": ops_result.get("source")},
            "rag": rag_result.get("meta") or {"source": rag_result.get("source")},
        },
    }
