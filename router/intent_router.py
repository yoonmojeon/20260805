"""
질문 의도 라우터 — hard guard → LLM source needs → deterministic fallback.

선택 모델이 질문 전체와 대화 문맥에서 OPS/문서 필요 여부를 판단한다.
기존 단서·prototype·technical shape은 모델 실패/저신뢰 fallback에만 쓴다.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Literal

from prompts.router_prompt import (
    ROUTER_OUTPUT_SCHEMA,
    ROUTER_SYSTEM_PROMPT,
    build_router_user_prompt,
)
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
    need_ops: bool | None = None
    need_documents: bool | None = None
    model: str | None = None
    router_model: str | None = None
    llm_router_success: bool | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    router_latency_ms: float = 0.0
    llm_error_kind: str | None = None
    llm_done_reason: str | None = None
    llm_eval_count: int | None = None
    chat_mode: ChatMode | None = None
    ops_query: str | None = None
    rag_query: str | None = None
    expanded_question: str | None = None
    slots: dict[str, Any] | None = None
    dialogue_state: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        source_needs = {
            "chat": (False, False),
            "ops": (True, False),
            "rag": (False, True),
            "hybrid": (True, True),
        }
        expected = source_needs.get(self.route)
        if expected:
            if self.need_ops is None:
                self.need_ops = expected[0]
            if self.need_documents is None:
                self.need_documents = expected[1]

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
    method: str = "hard_guard",
) -> RouteDecision:
    return RouteDecision(
        route="chat",
        confidence=confidence,
        ops_score=ops_score,
        rag_score=rag_score,
        reason=reason,
        method=method,
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


def _usable_source_query(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    query = " ".join(value.strip().split())[:300]
    if len(query) < 4 or not any(ch.isalnum() for ch in query):
        return None
    return query


def _fill_hybrid_queries(decision: RouteDecision, question: str, state: DialogueState) -> None:
    if decision.route != "hybrid":
        return
    ops_q, rag_q = split_hybrid_queries(question, state)
    decision.ops_query = _usable_source_query(decision.ops_query) or ops_q
    decision.rag_query = _usable_source_query(decision.rag_query) or rag_q


def _speech_act(question: str) -> RouteDecision | None:
    q = (question or "").strip()
    if not q:
        return _chat_decision("빈 질문은 chat에서 입력을 요청합니다.", chat_mode="clarify")
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
                method="rules",
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


def _route_from_needs(need_ops: bool, need_documents: bool) -> FinalRoute:
    if need_ops and need_documents:
        return "hybrid"
    if need_ops:
        return "ops"
    if need_documents:
        return "rag"
    return "chat"


def _active_model(model: str | None) -> str:
    return (
        model
        or os.getenv("MARITIME_OLLAMA_MODEL")
        or os.getenv("MODEL_NAME")
        or "gemma4:12b"
    ).strip() or "gemma4:12b"


def _bounded_env_float(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return min(high, max(low, value))


def _normalise_llm_payload(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    required = {
        "need_ops",
        "need_documents",
        "confidence",
        "reason",
        "ops_query",
        "rag_query",
    }
    if not required.issubset(data):
        return None

    need_ops = data["need_ops"]
    need_documents = data["need_documents"]
    if not isinstance(need_ops, bool) or not isinstance(need_documents, bool):
        return None

    confidence = data.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        return None

    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return None
    if not isinstance(data.get("ops_query"), str) or not isinstance(data.get("rag_query"), str):
        return None
    return {
        "need_ops": need_ops,
        "need_documents": need_documents,
        "confidence": confidence,
        "reason": " ".join(reason.strip().split())[:180],
        "ops_query": _usable_source_query(data.get("ops_query")),
        "rag_query": _usable_source_query(data.get("rag_query")),
    }


def _llm_payload_error_kind(data: Any) -> str | None:
    if not isinstance(data, dict):
        return "invalid_json"
    required = {
        "need_ops",
        "need_documents",
        "confidence",
        "reason",
        "ops_query",
        "rag_query",
    }
    if not required.issubset(data):
        return "missing_field"
    if not isinstance(data.get("need_ops"), bool) or not isinstance(
        data.get("need_documents"), bool
    ):
        return "invalid_boolean"
    confidence = data.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        return "invalid_confidence"
    if not isinstance(data.get("reason"), str) or not str(data.get("reason") or "").strip():
        return "invalid_reason"
    if not isinstance(data.get("ops_query"), str) or not isinstance(
        data.get("rag_query"), str
    ):
        return "invalid_query"
    return None


def _llm_classify(
    question: str,
    *,
    state: DialogueState,
    ops: float,
    rag: float,
    expanded: str,
    model: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    base = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    request_body: dict[str, Any] = {
        "model": model,
        "stream": False,
        "think": False,
        "format": ROUTER_OUTPUT_SCHEMA,
        "options": {"temperature": 0, "num_predict": 180},
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
    timeout = _bounded_env_float("ROUTER_TIMEOUT_SECONDS", 8.0, 1.0, 30.0)

    def failure(kind: str, **extra: Any) -> dict[str, Any]:
        return {
            "_success": False,
            "error_kind": kind,
            "router_latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "router_model": model,
            **extra,
        }

    def send(body: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/api/chat",
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        try:
            payload = send(request_body)
        except urllib.error.HTTPError as exc:
            if exc.code not in {400, 422}:
                raise
            # Older Ollama builds may reject think=false or a JSON schema.
            compatible = dict(request_body)
            compatible.pop("think", None)
            compatible["format"] = "json"
            payload = send(compatible)
        message = payload.get("message") or {}
        content = message.get("content") or ""
        done_reason = payload.get("done_reason")
        eval_count = payload.get("eval_count")
        if not isinstance(content, (str, dict)) or (
            isinstance(content, str) and not content.strip()
        ):
            return failure(
                "empty_response",
                done_reason=done_reason,
                eval_count=eval_count,
            )
        try:
            data = content if isinstance(content, dict) else json.loads(content)
        except json.JSONDecodeError:
            return failure(
                "invalid_json",
                done_reason=done_reason,
                eval_count=eval_count,
            )
        error_kind = _llm_payload_error_kind(data)
        if error_kind:
            return failure(
                error_kind,
                done_reason=done_reason,
                eval_count=eval_count,
            )
        normalised = _normalise_llm_payload(data)
        if normalised is None:
            return failure("invalid_payload", done_reason=done_reason, eval_count=eval_count)
        normalised.update(
            {
                "_success": True,
                "router_model": model,
                "router_latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "done_reason": done_reason,
                "eval_count": eval_count,
            }
        )
        return normalised
    except TimeoutError:
        return failure("timeout")
    except urllib.error.HTTPError as exc:
        return failure("http_error", http_status=exc.code)
    except urllib.error.URLError as exc:
        kind = "timeout" if isinstance(getattr(exc, "reason", None), TimeoutError) else "unavailable"
        return failure(kind)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return failure("generation_failure")


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
        decision.__post_init__()
        return decision

    if decision.ops_score == 0 and decision.rag_score == 0:
        decision.route = "chat"
        decision.chat_mode = "clarify"
        decision.reason += " 추측하지 않고 되묻습니다. 문서 RAG로 보내지 않습니다."
        decision.confidence = 0.55
        decision.__post_init__()
        return decision

    decision.route = "chat"
    decision.chat_mode = "clarify"
    decision.reason += " 한쪽만 추측하지 않고 되묻습니다."
    decision.confidence = 0.55
    decision.__post_init__()
    return decision


def _decision_from_llm(
    data: dict[str, Any],
    *,
    ops_score: float,
    rag_score: float,
    expanded: str,
    slots: dict[str, Any],
    active_model: str,
) -> RouteDecision:
    need_ops = bool(data["need_ops"])
    need_documents = bool(data["need_documents"])
    route = _route_from_needs(need_ops, need_documents)
    reason = str(data.get("reason") or "의미 기반으로 필요한 데이터 소스를 판정했습니다.")
    return RouteDecision(
        route=route,
        confidence=float(data["confidence"]),
        ops_score=ops_score,
        rag_score=rag_score,
        reason=reason,
        method="llm",
        need_ops=need_ops,
        need_documents=need_documents,
        model=active_model,
        router_model=active_model,
        llm_router_success=True,
        fallback_used=False,
        router_latency_ms=float(data.get("router_latency_ms") or 0.0),
        llm_done_reason=str(data.get("done_reason") or "") or None,
        llm_eval_count=(
            int(data["eval_count"])
            if isinstance(data.get("eval_count"), int)
            else None
        ),
        chat_mode="clarify" if route == "chat" else None,
        ops_query=data.get("ops_query"),
        rag_query=data.get("rag_query"),
        expanded_question=expanded,
        slots=slots,
    )


def _guard_overbroad_hybrid(
    decision: RouteDecision,
    *,
    question: str,
    ops_score: float,
    rag_score: float,
) -> RouteDecision:
    """Collapse an unsupported hybrid decision back to the clear OPS source.

    CII vocabulary is shared by live vessel analytics and IMO documents.  A
    zero-temperature local model can still occasionally request both sources
    for a plain attained/required ship scorecard.  If no document cue, compare
    frame, or explicit dual marker exists, the extra RAG call is unsupported and
    needlessly slow.
    """
    if (
        decision.route == "hybrid"
        and ops_score >= 2.5
        and rag_score == 0.0
        and not has_dual_mark(question)
        and not has_compare_frame(question)
    ):
        decision.route = "ops"
        decision.need_ops = True
        decision.need_documents = False
        decision.rag_query = None
        decision.method = "llm_guarded"
        decision.reason = (
            "LLM이 두 소스를 요청했지만 문서 단서가 없는 명확한 선박 운항 질의라 "
            "불필요한 문서 검색을 생략합니다."
        )
    return decision


def _guard_technical_chat_miss(
    decision: RouteDecision,
    *,
    question: str,
    ops_score: float,
    rag_score: float,
) -> RouteDecision:
    """Do not let an LLM chat miss discard a strong document-shaped query."""
    if (
        decision.route == "chat"
        and ops_score == 0.0
        and (rag_score >= 2.0 or looks_like_technical_rag(question))
        and not OOS_PATTERN.search(question or "")
    ):
        decision.route = "rag"
        decision.need_ops = False
        decision.need_documents = True
        decision.chat_mode = None
        decision.method = "llm_guarded"
        decision.confidence = max(decision.confidence, 0.75)
        decision.reason = (
            "LLM이 chat으로 분류했지만 조항·선급 문서 단서가 명확하여 "
            "문서 검색으로 보정합니다."
        )
    return decision


def _deterministic_fallback(
    question: str,
    *,
    expanded: str,
    state: DialogueState,
    ops: float,
    rag: float,
    slots: dict[str, Any],
) -> RouteDecision:
    multiturn = _multiturn_override(question, expanded, state, ops, rag)
    if multiturn is not None:
        multiturn.slots = slots
        _fill_hybrid_queries(multiturn, expanded, state)
        return multiturn

    decision = _rules_route(expanded)
    decision.slots = slots
    decision.expanded_question = expanded
    if decision.route == "ambiguous":
        proto_route, proto_margin, _ = prototype_vote(question)
        if (
            decision.ops_score == 0
            and decision.rag_score == 0
            and proto_route in {"ops", "rag", "hybrid"}
            and proto_margin >= 0.12
        ):
            decision.route = proto_route  # type: ignore[assignment]
            decision.method = "proto"
            decision.confidence = 0.62
            decision.chat_mode = None
            decision.reason = f"규칙 단서는 없지만 프로토타입이 '{proto_route}'를 강하게 지지합니다."
            decision.__post_init__()
        else:
            decision = _resolve_ambiguous(decision, expanded)

    _fill_hybrid_queries(decision, expanded, state)
    return decision


def route_question(
    question: str,
    *,
    use_llm_fallback: bool = True,
    force_route: RouteKind | None = None,
    last_route: str | None = None,
    dialogue_state: DialogueState | dict | None = None,
    active_model: str | None = None,
) -> RouteDecision:
    state = parse_dialogue_state(dialogue_state, last_route)
    model = _active_model(active_model)
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
            model=model,
            router_model=model,
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
        speech.model = model
        speech.router_model = model
        return _attach_state(speech, question=q, previous=state, topics=topics, entities=entities)

    expanded = expand_question(q, state)
    ops, rag = score_question(expanded if expanded != q else q)
    slots["ops_score"] = ops
    slots["rag_score"] = rag
    slots["expanded_question"] = expanded

    fallback_reason: str | None = None
    llm_meta: dict[str, Any] = {}
    if use_llm_fallback:
        llm = _llm_classify(
            q,
            state=state,
            ops=ops,
            rag=rag,
            expanded=expanded,
            model=model,
        )
        # Unit-test doubles may return only the public source-need payload.
        if "_success" not in llm:
            normalised = _normalise_llm_payload(llm)
            llm = (
                {**normalised, "_success": True, "router_latency_ms": 0.0}
                if normalised is not None
                else {"_success": False, "error_kind": "invalid_payload", "router_latency_ms": 0.0}
            )
        llm_meta = llm
        if llm.get("_success"):
            threshold = _bounded_env_float(
                "ROUTER_CONFIDENCE_THRESHOLD", 0.65, 0.0, 1.0
            )
            if float(llm["confidence"]) >= threshold:
                decision = _decision_from_llm(
                    llm,
                    ops_score=ops,
                    rag_score=rag,
                    expanded=expanded,
                    slots=slots,
                    active_model=model,
                )
                decision = _guard_overbroad_hybrid(
                    decision,
                    question=expanded,
                    ops_score=ops,
                    rag_score=rag,
                )
                decision = _guard_technical_chat_miss(
                    decision,
                    question=expanded,
                    ops_score=ops,
                    rag_score=rag,
                )
                _fill_hybrid_queries(decision, expanded, state)
                return _attach_state(
                    decision,
                    question=q,
                    previous=state,
                    topics=topics,
                    entities=entities,
                )
            fallback_reason = (
                f"LLM confidence {float(llm['confidence']):.2f}가 "
                f"기준 {threshold:.2f}보다 낮습니다."
            )
            llm_meta = {**llm, "error_kind": "low_confidence"}
        else:
            kind = str(llm.get("error_kind") or "generation_failure")
            fallback_reason = f"LLM router 실패({kind})로 유효한 분류가 없습니다."

    decision = _deterministic_fallback(
        q,
        expanded=expanded,
        state=state,
        ops=ops,
        rag=rag,
        slots=slots,
    )
    if use_llm_fallback:
        fallback_method = decision.method
        if decision.slots is not None:
            decision.slots["fallback_method"] = fallback_method
        decision.fallback_reason = fallback_reason
        decision.method = "llm_fallback_rules"
        decision.reason = f"{fallback_reason} deterministic fallback: {decision.reason}"
        decision.model = model
        decision.router_model = model
        decision.llm_router_success = False
        decision.fallback_used = True
        decision.router_latency_ms = float(llm_meta.get("router_latency_ms") or 0.0)
        decision.llm_error_kind = str(llm_meta.get("error_kind") or "") or None
        decision.llm_done_reason = str(llm_meta.get("done_reason") or "") or None
        decision.llm_eval_count = (
            int(llm_meta["eval_count"])
            if isinstance(llm_meta.get("eval_count"), int)
            else None
        )
    else:
        decision.model = model
        decision.router_model = model
        decision.llm_router_success = None
        decision.fallback_used = False
    return _attach_state(
        decision,
        question=q,
        previous=state,
        topics=topics,
        entities=entities,
    )
