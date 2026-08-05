"""Top-level question handler: route → ops or rag backend."""
from __future__ import annotations

from typing import Any, Literal

from router.intent_router import RouteDecision, route_question
from services.ops_service import run_ops_query
from services.rag_service import run_rag_query

ForceRoute = Literal["auto", "ops", "rag"]


def handle_question(
    question: str,
    history: list | None = None,
    *,
    force_route: ForceRoute = "auto",
    use_llm_router: bool = True,
    rag_latency_mode: str = "fast",
) -> dict[str, Any]:
    q = (question or "").strip()
    if not q:
        return {
            "answer": "질문을 입력하세요.",
            "route": None,
            "history": history or [],
            "files": [],
            "map_html": "",
        }

    forced = None if force_route == "auto" else force_route
    decision: RouteDecision = route_question(
        q, use_llm_fallback=use_llm_router, force_route=forced
    )

    if decision.route == "ops":
        result = run_ops_query(q, history)
    else:
        # simple table-qa hint
        table_qa = any(k in q for k in ("표", "선령", "정기검사", "평형수탱크"))
        result = run_rag_query(q, latency_mode=rag_latency_mode, table_qa=table_qa)
        result.setdefault("history", (history or []) + [
            {"role": "user", "content": q},
            {"role": "assistant", "content": result.get("answer", "")},
        ])
        result.setdefault("map_html", "")

    return {
        "answer": result.get("answer", ""),
        "route": decision.to_dict(),
        "history": result.get("history") or history or [],
        "files": result.get("files") or [],
        "map_html": result.get("map_html") or "",
        "source": result.get("source"),
        "meta": result.get("meta"),
    }
