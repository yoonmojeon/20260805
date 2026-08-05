"""Top-level question handler: route → chat / ops / rag / hybrid."""
from __future__ import annotations

from typing import Any, Literal

from router.intent_router import RouteDecision, route_question
from services.chat_service import run_chat_query
from services.hybrid_service import run_hybrid_query
from services.ops_service import run_ops_query
from services.rag_service import run_rag_query

ForceRoute = Literal["auto", "ops", "rag", "chat", "hybrid"]
TABLE_HINTS = ("표", "선령", "정기검사", "평형수탱크", "밸러스트", "검사 주기", "검사주기")


def _table_qa_hint(question: str) -> bool:
    return any(k in question for k in TABLE_HINTS)


def _next_last_route(decision: RouteDecision, previous: str | None) -> str | None:
    if decision.route in {"ops", "rag", "hybrid"}:
        return decision.route
    return previous


def handle_question(
    question: str,
    history: list | None = None,
    *,
    force_route: ForceRoute = "auto",
    use_llm_router: bool = True,
    rag_latency_mode: str = "fast",
    last_route: str | None = None,
) -> dict[str, Any]:
    q = (question or "").strip()
    if not q:
        return {
            "answer": "질문을 입력하세요.",
            "route": None,
            "history": history or [],
            "files": [],
            "map_html": "",
            "last_route": last_route,
        }

    forced = None if force_route == "auto" else force_route
    decision: RouteDecision = route_question(
        q,
        use_llm_fallback=use_llm_router,
        force_route=forced,
        last_route=last_route,
    )

    if decision.route == "chat":
        result = run_chat_query(q, history, chat_mode=decision.chat_mode)
    elif decision.route == "ops":
        result = run_ops_query(q, history)
    elif decision.route == "hybrid":
        result = run_hybrid_query(
            q,
            history,
            rag_latency_mode=rag_latency_mode,
            table_qa=_table_qa_hint(q),
        )
    else:
        result = run_rag_query(
            q, latency_mode=rag_latency_mode, table_qa=_table_qa_hint(q)
        )
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
        "last_route": _next_last_route(decision, last_route),
    }
