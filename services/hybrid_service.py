"""Hybrid path — run ops and rag on split queries, then merge with source labels."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from services.ops_service import run_ops_query
from services.rag_service import run_rag_query
from services.retrieval_mode import RetrievalMode


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


def _synthesize_hybrid_answer(
    question: str,
    ops_answer: str,
    rag_answer: str,
    *,
    model: str,
) -> tuple[str, dict[str, Any]]:
    """Use the active model once more to combine the two grounded answers."""
    fallback = merge_hybrid_answers(ops_answer, rag_answer)
    base = (
        os.environ.get("MARITIME_OLLAMA_BASE")
        or os.environ.get("OLLAMA_HOST")
        or "http://127.0.0.1:11434"
    ).rstrip("/")
    body: dict[str, Any] = {
        "model": model,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.1, "num_predict": 1200},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the grounded answer synthesizer for MaritimeOpsRAG. "
                    "Answer in Korean using only the provided OPS and DOCUMENT results. "
                    "Preserve useful numbers and citations. Clearly separate current vessel facts "
                    "from external requirements. Never invent missing data."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"질문: {question}\n\n"
                    f"[OPS RESULT]\n{ops_answer}\n\n"
                    f"[DOCUMENT RESULT]\n{rag_answer}\n\n"
                    "두 근거를 결합해 질문에 직접 답하세요."
                ),
            },
        ],
    }
    started = time.perf_counter()

    def send(payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{base}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        try:
            payload = send(body)
        except urllib.error.HTTPError as exc:
            if exc.code not in {400, 422}:
                raise
            compatible = dict(body)
            compatible.pop("think", None)
            payload = send(compatible)
        content = str((payload.get("message") or {}).get("content") or "").strip()
        meta = {
            "hybrid_synthesis_model": model,
            "hybrid_synthesis_success": bool(content),
            "hybrid_synthesis_latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "hybrid_synthesis_done_reason": payload.get("done_reason"),
            "hybrid_synthesis_eval_count": payload.get("eval_count"),
            "hybrid_synthesis_empty": not bool(content),
        }
        return (content or fallback), meta
    except Exception as exc:
        return fallback, {
            "hybrid_synthesis_model": model,
            "hybrid_synthesis_success": False,
            "hybrid_synthesis_latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "hybrid_synthesis_error": f"{type(exc).__name__}: {exc}",
            "hybrid_synthesis_empty": True,
        }


def run_hybrid_query(
    question: str,
    history: list | None = None,
    *,
    rag_latency_mode: str = "fast",
    table_qa: bool | None = None,
    retrieval_mode: RetrievalMode | str | None = None,
    ops_query: str | None = None,
    rag_query: str | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    from services.llm_models import normalize_llm_model

    model = normalize_llm_model(llm_model)
    ops_q = (ops_query or question).strip()
    rag_q = (rag_query or question).strip()
    ops_result = run_ops_query(ops_q, history, llm_model=model)
    rag_result = run_rag_query(
        rag_q,
        latency_mode=rag_latency_mode,
        table_qa=table_qa,
        retrieval_mode=retrieval_mode,
        llm_model=model,
    )
    answer, synthesis_meta = _synthesize_hybrid_answer(
        question,
        str(ops_result.get("answer") or ""),
        str(rag_result.get("answer") or ""),
        model=model,
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
        "evidence_table": list(rag_result.get("evidence_table") or []),
        "related_tables": list(rag_result.get("related_tables") or []),
        "source": "hybrid",
        "meta": {
            "ops": ops_result.get("meta") or {"source": ops_result.get("source")},
            "rag": rag_result.get("meta") or {"source": rag_result.get("source")},
            "ops_query": ops_q,
            "rag_query": rag_q,
            "retrieval_mode": (rag_result.get("meta") or {}).get("retrieval_mode"),
            **synthesis_meta,
        },
    }
