"""
질문 의도 라우터 — chat / ops / rag / hybrid.

슬롯(화행·소스·주제)을 본 뒤 경로를 고른다.
문장 전체를 외우지 않고, 모르는 입력은 RAG로 보내지 않는다.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Literal

from prompts.router_prompt import ROUTER_SYSTEM_PROMPT, build_router_user_prompt
from router.cues import (
    CAPABILITY_PATTERN,
    GREET_PATTERN,
    IDENTITY_PATTERN,
    META_PATTERN,
    OOS_PATTERN,
    SHIP_STRONG_PATTERN,
    SWITCH_HYBRID_PATTERN,
    SWITCH_OPS_PATTERN,
    SWITCH_RAG_PATTERN,
    THANKS_PATTERN,
    extract_entities,
    extract_topics,
    has_compare_frame,
    has_dual_mark,
    is_followup_text,
    looks_like_technical_ops,
    looks_like_technical_rag,
    score_question,
)
from router.dialogue import DialogueState, next_dialogue_state, parse_dialogue_state
from router.prototypes import prototype_vote
from router.rewrite import expand_question, split_hybrid_queries

RouteKind = Literal["ops", "rag", "chat", "hybrid", "ambiguous"]
FinalRoute = Literal["ops", "rag", "chat", "hybrid"]
ChatMode = Literal["identity", "greeting", "meta", "clarify", "dual", "oos", "thanks"]
PersistentRoute = Literal["ops", "rag", "hybrid"]


@dataclass
class RouteDecision:
    route: RouteKind
    confidence: float
    ops_score: float
    rag_score: float
    reason: str
    method: str
    chat_mode: ChatMode | None = None
    ops_query: str | None = None
    rag_query: str | None = None
    expanded_question: str | None = None
    slots: dict[str, Any] | None = None
    dialogue_state: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _chat_decision(
    reason: str,
    *,
    chat_mode: ChatMode,
    confidence: float = 0.9,
    ops_score: float = 0.0,
    rag_score: float = 0.0,
    slots: dict[str, Any] | None = None,
    expanded_question: str | None = None,
) -> RouteDecision:
    return RouteDecision(
        route="chat",
        confidence=confidence,
        ops_score=ops_score,
        rag_score=rag_score,
        reason=reason,
        method="rules",
        chat_mode=chat_mode,
        slots=slots,
        expanded_question=expanded_question,
    )


def _attach_state(
    decision: RouteDecision,
    *,
    question: str,
    previous: DialogueState,
    topics: list[str],
    entities: list[str],
) -> RouteDecision:
    state = next_dialogue_state(
        previous=previous,
        route=str(decision.route),
        question=question,
        topics=topics,
        entities=entities,
        ops_query=decision.ops_query,
        rag_query=decision.rag_query,
    )
    decision.dialogue_state = state.to_dict()
    return decision


def _fill_hybrid_queries(decision: RouteDecision, question: str, state: DialogueState) -> None:
    if decision.route != "hybrid":
        return
    ops_q, rag_q = split_hybrid_queries(question, state)
    decision.ops_query = decision.ops_query or ops_q
    decision.rag_query = decision.rag_query or rag_q


def _speech_act(question: str) -> RouteDecision | None:
    q = (question or "").strip()
    if META_PATTERN.search(q):
        return _chat_decision("시스템·라우터 안내 질문이라 chat 경로입니다.", chat_mode="meta")
    if GREET_PATTERN.search(q):
        return _chat_decision("인사는 chat 경로입니다.", chat_mode="greeting")
    if THANKS_PATTERN.search(q):
        return _chat_decision("감사 인사는 chat 경로입니다.", chat_mode="thanks")
    # 능력·범위 질문은 운항/문서 단어가 섞여도 chat (예: 운항이랑 문서 둘 다 가능해?).
    if CAPABILITY_PATTERN.search(q):
        return _chat_decision("기능·범위 안내는 chat 경로입니다.", chat_mode="identity")
    if IDENTITY_PATTERN.search(q):
        ops, rag = score_question(q)
        if ops == 0 and rag == 0:
            return _chat_decision("자기소개·기능 안내는 chat 경로입니다.", chat_mode="identity")
    return None


def _multiturn_override(
    question: str,
    expanded: str,
    state: DialogueState,
    ops: float,
    rag: float,
) -> RouteDecision | None:
    last_route = state.last_route
    if last_route not in {"ops", "rag", "hybrid"}:
        return None
    q = (question or "").strip()

    if SWITCH_HYBRID_PATTERN.search(q) or (has_compare_frame(q) and ops > 0 and rag > 0):
        return RouteDecision(
            route="hybrid",
            confidence=0.86,
            ops_score=ops,
            rag_score=rag,
            reason="이전 맥락에서 운항과 문서를 같이 요청했습니다.",
            method="multiturn",
            expanded_question=expanded,
        )

    switch_ops = bool(SWITCH_OPS_PATTERN.search(q))
    switch_rag = bool(SWITCH_RAG_PATTERN.search(q))
    if switch_ops and not switch_rag and rag - ops < 1.0:
        return RouteDecision(
            route="ops",
            confidence=0.85,
            ops_score=max(ops, 1.0),
            rag_score=rag,
            reason="후속 질문이 운항 쪽으로 전환되었습니다.",
            method="multiturn",
            expanded_question=expanded,
        )
    if switch_rag and not switch_ops and ops - rag < 1.0:
        return RouteDecision(
            route="rag",
            confidence=0.85,
            ops_score=ops,
            rag_score=max(rag, 1.0),
            reason="후속 질문이 문서 쪽으로 전환되었습니다.",
            method="multiturn",
            expanded_question=expanded,
        )

    if last_route == "rag" and SHIP_STRONG_PATTERN.search(q) and not switch_rag:
        if ops - rag >= 1.0 and not is_followup_text(q) and not has_compare_frame(q):
            return None
        return RouteDecision(
            route="hybrid",
            confidence=0.82,
            ops_score=max(ops, 1.0),
            rag_score=max(rag, 1.0),
            reason="문서 맥락에 우리 선박을 대보고 있어 hybrid로 봅니다.",
            method="multiturn",
            expanded_question=expanded,
        )
    if last_route == "ops" and has_compare_frame(q):
        return RouteDecision(
            route="hybrid",
            confidence=0.82,
            ops_score=max(ops, 1.0),
            rag_score=max(rag, 1.0),
            reason="운항 맥락에 규정 기준을 대보고 있어 hybrid로 봅니다.",
            method="multiturn",
            expanded_question=expanded,
        )

    if not is_followup_text(q):
        return None
    if ops - rag >= 1.0 and last_route != "ops":
        return None
    if rag - ops >= 1.0 and last_route != "rag":
        return None
    return RouteDecision(
        route=last_route,  # type: ignore[arg-type]
        confidence=0.75,
        ops_score=ops,
        rag_score=rag,
        reason=f"짧은 후속 질문이라 이전 경로({last_route})를 유지합니다.",
        method="multiturn",
        expanded_question=expanded,
    )


def _rules_route(question: str, margin: float = 1.0) -> RouteDecision:
    ops, rag = score_question(question)
    if ops > 0 and rag > 0 and (has_dual_mark(question) or has_compare_frame(question)):
        return RouteDecision(
            route="hybrid",
            confidence=0.84 if has_dual_mark(question) else 0.78,
            ops_score=ops,
            rag_score=rag,
            reason="운항 단서와 문서 단서를 같이 물어 hybrid로 보냅니다.",
            method="rules",
            expanded_question=question,
        )
    if ops == 0 and rag == 0:
        if OOS_PATTERN.search(question or ""):
            return _chat_decision(
                "운항/문서 범위 밖 질문이라 chat에서 안내합니다.",
                chat_mode="oos",
                confidence=0.8,
                expanded_question=question,
            )
        return RouteDecision(
            route="ambiguous",
            confidence=0.0,
            ops_score=ops,
            rag_score=rag,
            reason="운항/문서 단서가 없어 분류가 불명확합니다.",
            method="rules",
            chat_mode="clarify",
            expanded_question=question,
        )
    if ops - rag >= margin:
        return RouteDecision(
            route="ops",
            confidence=min(1.0, (ops - rag) / max(ops, 1.0)),
            ops_score=ops,
            rag_score=rag,
            reason="운항·선박 데이터 단서가 더 강합니다.",
            method="rules",
            expanded_question=question,
        )
    if rag - ops >= margin:
        return RouteDecision(
            route="rag",
            confidence=min(1.0, (rag - ops) / max(rag, 1.0)),
            ops_score=ops,
            rag_score=rag,
            reason="규정·회의·선급 문서 단서가 더 강합니다.",
            method="rules",
            expanded_question=question,
        )

    proto_route, proto_margin, proto_scores = prototype_vote(question)
    if proto_route in {"ops", "rag"} and proto_margin >= 0.06:
        return RouteDecision(
            route=proto_route,  # type: ignore[arg-type]
            confidence=min(0.8, 0.55 + proto_margin),
            ops_score=ops,
            rag_score=rag,
            reason=(
                f"운항/문서 점수가 비슷해 프로토타입 투표로 '{proto_route}'를 골랐습니다 "
                f"(proto={proto_scores})."
            ),
            method="rules+proto",
            expanded_question=question,
        )
    if proto_route == "hybrid" and proto_margin >= 0.08 and ops > 0 and rag > 0:
        return RouteDecision(
            route="hybrid",
            confidence=0.7,
            ops_score=ops,
            rag_score=rag,
            reason="근접 점수에서 프로토타입이 hybrid를 지지합니다.",
            method="rules+proto",
            expanded_question=question,
        )

    return RouteDecision(
        route="ambiguous",
        confidence=0.35,
        ops_score=ops,
        rag_score=rag,
        reason=f"운항/문서 단서가 비슷해 되묻습니다(ops={ops:.1f}, rag={rag:.1f}).",
        method="rules",
        chat_mode="clarify",
        expanded_question=question,
    )


def _llm_classify(
    question: str,
    *,
    state: DialogueState,
    ops: float,
    rag: float,
    expanded: str,
) -> dict[str, Any] | None:
    base = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("MODEL_NAME", "llama3.1:8b")
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_router_user_prompt(
                        question,
                        last_route=state.last_route,
                        last_topic=state.last_topic,
                        last_question=state.last_question,
                        ops_score=ops,
                        rag_score=rag,
                        expanded_question=expanded,
                    ),
                },
            ],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        content = (payload.get("message") or {}).get("content") or ""
        data = json.loads(content)
        route = str(data.get("route", "")).strip().lower()
        if route not in {"ops", "rag", "chat", "hybrid"}:
            return None
        return data if isinstance(data, dict) else None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError):
        return None


def _shape_override_route(question: str) -> FinalRoute | None:
    """When cue scores miss, still route technical shapes away from chat clarify."""
    if looks_like_technical_rag(question):
        return "rag"
    if looks_like_technical_ops(question):
        return "ops"
    return None


def _resolve_ambiguous(decision: RouteDecision, question: str = "") -> RouteDecision:
    shaped = _shape_override_route(question or decision.expanded_question or "")
    if shaped in {"ops", "rag"}:
        decision.route = shaped
        decision.chat_mode = None
        decision.method = f"{decision.method}+shape"
        decision.confidence = max(decision.confidence, 0.68)
        decision.reason += f" 기술형 발화라 chat 되묻기 대신 {shaped}로 보냅니다."
        if shaped == "rag":
            decision.rag_score = max(decision.rag_score, 1.0)
        else:
            decision.ops_score = max(decision.ops_score, 1.0)
        return decision

    if decision.ops_score == 0 and decision.rag_score == 0:
        decision.route = "chat"
        decision.chat_mode = "clarify"
        decision.reason += " 추측하지 않고 되묻습니다. 문서 RAG로 보내지 않습니다."
        decision.confidence = 0.55
        return decision

    decision.route = "chat"
    decision.chat_mode = "clarify"
    decision.reason += " 한쪽만 추측하지 않고 되묻습니다."
    decision.confidence = 0.55
    return decision


def route_question(
    question: str,
    *,
    use_llm_fallback: bool = True,
    force_route: RouteKind | None = None,
    last_route: str | None = None,
    dialogue_state: DialogueState | dict | None = None,
) -> RouteDecision:
    state = parse_dialogue_state(dialogue_state, last_route)
    q = (question or "").strip()
    topics = extract_topics(q)
    entities = extract_entities(q)
    slots = {
        "topics": topics,
        "entities": entities,
        "followup": is_followup_text(q),
        "dual": has_dual_mark(q),
        "compare": has_compare_frame(q),
        "last_route": state.last_route,
        "last_topic": state.last_topic,
    }

    if force_route in {"ops", "rag", "chat", "hybrid"}:
        decision = RouteDecision(
            route=force_route,
            confidence=1.0,
            ops_score=0.0,
            rag_score=0.0,
            reason="사용자가 경로를 직접 지정했습니다.",
            method="manual",
            chat_mode="clarify" if force_route == "chat" else None,
            expanded_question=q,
            slots=slots,
        )
        if force_route == "hybrid":
            _fill_hybrid_queries(decision, q, state)
        return _attach_state(decision, question=q, previous=state, topics=topics, entities=entities)

    speech = _speech_act(q)
    if speech:
        speech.slots = slots
        speech.expanded_question = q
        return _attach_state(speech, question=q, previous=state, topics=topics, entities=entities)

    expanded = expand_question(q, state)
    ops, rag = score_question(expanded if expanded != q else q)
    slots["ops_score"] = ops
    slots["rag_score"] = rag
    slots["expanded_question"] = expanded

    if not (
        META_PATTERN.search(q) or IDENTITY_PATTERN.search(q) or GREET_PATTERN.search(q)
    ):
        multiturn = _multiturn_override(q, expanded, state, ops, rag)
        if multiturn:
            multiturn.slots = slots
            _fill_hybrid_queries(multiturn, expanded, state)
            return _attach_state(
                multiturn, question=q, previous=state, topics=topics, entities=entities
            )

    decision = _rules_route(expanded)
    decision.slots = slots
    decision.expanded_question = expanded
    if decision.route != "ambiguous":
        _fill_hybrid_queries(decision, expanded, state)
        return _attach_state(decision, question=q, previous=state, topics=topics, entities=entities)

    proto_route, proto_margin, _ = prototype_vote(q)
    if decision.ops_score == 0 and decision.rag_score == 0 and proto_route in {"ops", "rag", "hybrid"}:
        if proto_margin >= 0.12:
            decision.route = proto_route  # type: ignore[assignment]
            decision.method = "proto"
            decision.confidence = 0.62
            decision.chat_mode = None
            decision.reason = f"규칙 단서는 없지만 프로토타입이 '{proto_route}'를 강하게 지지합니다."
            _fill_hybrid_queries(decision, expanded, state)
            return _attach_state(
                decision, question=q, previous=state, topics=topics, entities=entities
            )

    if use_llm_fallback:
        llm = _llm_classify(
            q, state=state, ops=decision.ops_score, rag=decision.rag_score, expanded=expanded
        )
        if llm:
            llm_route = str(llm.get("route", "")).strip().lower()
            if llm_route in {"ops", "rag", "hybrid"}:
                decision = RouteDecision(
                    route=llm_route,  # type: ignore[arg-type]
                    confidence=0.7,
                    ops_score=decision.ops_score,
                    rag_score=decision.rag_score,
                    reason=f"저신뢰 구간이라 LLM이 '{llm_route}'로 분류했습니다.",
                    method="rules+llm",
                    expanded_question=str(llm.get("expanded_question") or expanded),
                    ops_query=llm.get("ops_query"),
                    rag_query=llm.get("rag_query"),
                    slots=slots,
                )
                _fill_hybrid_queries(decision, expanded, state)
                return _attach_state(
                    decision, question=q, previous=state, topics=topics, entities=entities
                )
            if llm_route == "chat":
                shaped = _shape_override_route(expanded)
                if shaped in {"ops", "rag"}:
                    decision = RouteDecision(
                        route=shaped,
                        confidence=0.72,
                        ops_score=decision.ops_score if shaped == "ops" else max(decision.ops_score, 0.0),
                        rag_score=max(decision.rag_score, 1.0) if shaped == "rag" else decision.rag_score,
                        reason=(
                            f"LLM은 chat이었지만 기술형 발화라 {shaped}로 교정했습니다."
                        ),
                        method="rules+llm+shape",
                        expanded_question=expanded,
                        slots=slots,
                    )
                    if shaped == "ops":
                        decision.ops_score = max(decision.ops_score, 1.0)
                    _fill_hybrid_queries(decision, expanded, state)
                    return _attach_state(
                        decision, question=q, previous=state, topics=topics, entities=entities
                    )
                decision = RouteDecision(
                    route="chat",
                    confidence=0.65,
                    ops_score=decision.ops_score,
                    rag_score=decision.rag_score,
                    reason="저신뢰 구간이라 LLM이 chat으로 분류했습니다.",
                    method="rules+llm",
                    chat_mode="clarify",
                    expanded_question=expanded,
                    slots=slots,
                )
                return _attach_state(
                    decision, question=q, previous=state, topics=topics, entities=entities
                )

    fallback = _resolve_ambiguous(decision, expanded)
    fallback.slots = slots
    fallback.expanded_question = expanded
    if fallback.route == "chat" and use_llm_fallback:
        fallback.reason += " LLM 분류 불가라 되묻기로 처리했습니다."
    return _attach_state(fallback, question=q, previous=state, topics=topics, entities=entities)
