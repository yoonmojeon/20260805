"""Top-level question handler: route → chat / ops / rag / hybrid."""
from __future__ import annotations

from typing import Any, Literal

from router.dialogue import DialogueState, parse_dialogue_state
from router.intent_router import RouteDecision, route_question
from services.chat_service import run_chat_query
from services.hybrid_service import run_hybrid_query
from services.ops_service import run_ops_query
from services.rag_service import run_rag_query
from services.retrieval_mode import classify_retrieval_mode

ForceRoute = Literal["auto", "ops", "rag", "chat", "hybrid"]


def handle_question(
    question: str,
    history: list | None = None,
    *,
    force_route: ForceRoute = "auto",
    use_llm_router: bool = True,
    rag_latency_mode: str = "fast",
    last_route: str | None = None,
    dialogue_state: DialogueState | dict | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    from services.llm_models import normalize_llm_model

    q = (question or "").strip()
    model = normalize_llm_model(llm_model)
    state = parse_dialogue_state(dialogue_state, last_route)
    if not q:
        return {
            "answer": "질문을 입력하세요.",
            "route": None,
            "history": history or [],
            "files": [],
            "map_html": "",
            "evidence_table": [],
            "related_tables": [],
            "last_route": state.last_route,
            "dialogue_state": state.to_dict(),
            "llm_model": model,
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
        result = run_ops_query(effective_q, history, llm_model=model)
    elif decision.route == "hybrid":
        rag_mode = classify_retrieval_mode(decision.rag_query or effective_q)
        result = run_hybrid_query(
            q,
            history,
            rag_latency_mode=rag_latency_mode,
            retrieval_mode=rag_mode,
            ops_query=decision.ops_query,
            rag_query=decision.rag_query,
            llm_model=model,
        )
    else:
        rag_mode = classify_retrieval_mode(effective_q)
        result = run_rag_query(
            effective_q,
            latency_mode=rag_latency_mode,
            retrieval_mode=rag_mode,
            llm_model=model,
        )
        result.setdefault(
            "history",
            (history or [])
            + [
                {"role": "user", "content": q},
                {"role": "assistant", "content": result.get("answer", "")},
            ],
        )
        result.setdefault("map_html", "")

    meta = dict(result.get("meta") or {})
    if decision.route in {"rag", "hybrid"} and "retrieval_mode" not in meta:
        meta["retrieval_mode"] = classify_retrieval_mode(
            (decision.rag_query if decision.route == "hybrid" else None) or effective_q
        ).value

    return {
        "answer": result.get("answer", ""),
        "route": decision.to_dict(),
        "history": result.get("history") or history or [],
        "files": result.get("files") or [],
        "map_html": result.get("map_html") or "",
        "evidence_table": list(result.get("evidence_table") or []),
        "related_tables": list(result.get("related_tables") or []),
        "source": result.get("source"),
        "meta": meta or result.get("meta"),
        "last_route": next_state.get("last_route"),
        "dialogue_state": next_state,
        "llm_model": model,
    }
