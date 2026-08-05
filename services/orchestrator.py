"""Top-level question handler: route → chat / ops / rag / hybrid."""
from __future__ import annotations

from typing import Any, Literal

from router.dialogue import DialogueState, parse_dialogue_state
from router.intent_router import RouteDecision, route_question
from services.chat_service import run_chat_query
from services.hybrid_service import run_hybrid_query
from services.ops_service import run_ops_query
from services.rag_service import run_rag_query

ForceRoute = Literal["auto", "ops", "rag", "chat", "hybrid"]
TABLE_HINTS = ("표", "선령", "정기검사", "평형수탱크", "밸러스트", "검사 주기", "검사주기")


def _table_qa_hint(question: str) -> bool:
    return any(k in question for k in TABLE_HINTS)


def handle_question(
    question: str,
    history: list | None = None,
    *,
    force_route: ForceRoute = "auto",
    use_llm_router: bool = True,
    rag_latency_mode: str = "fast",
    last_route: str | None = None,
    dialogue_state: DialogueState | dict | None = None,
) -> dict[str, Any]:
    q = (question or "").strip()
    state = parse_dialogue_state(dialogue_state, last_route)
    if not q:
        return {
            "answer": "질문을 입력하세요.",
            "route": None,
            "history": history or [],
            "files": [],
            "map_html": "",
            "last_route": state.last_route,
            "dialogue_state": state.to_dict(),
        }

    forced = None if force_route == "auto" else force_route
    decision: RouteDecision = route_question(
        q,
        use_llm_fallback=use_llm_router,
        force_route=forced,
        last_route=state.last_route,
        dialogue_state=state,
    )
    next_state = decision.dialogue_state or state.to_dict()
    effective_q = (decision.expanded_question or q).strip()

    if decision.route == "chat":
        result = run_chat_query(q, history, chat_mode=decision.chat_mode)
    elif decision.route == "ops":
        result = run_ops_query(effective_q, history)
    elif decision.route == "hybrid":
        result = run_hybrid_query(
            q,
            history,
            rag_latency_mode=rag_latency_mode,
            table_qa=_table_qa_hint(effective_q),
            ops_query=decision.ops_query,
            rag_query=decision.rag_query,
        )
    else:
        result = run_rag_query(
            effective_q, latency_mode=rag_latency_mode, table_qa=_table_qa_hint(effective_q)
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
        "last_route": next_state.get("last_route"),
        "dialogue_state": next_state,
    }
