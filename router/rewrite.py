"""Follow-up expansion and hybrid query split."""
from __future__ import annotations

import re

from router.cues import extract_entities, extract_topics, is_followup_text, score_question
from router.dialogue import DialogueState

_SPLIT_RE = re.compile(r"(?:이랑|랑|하고|,)")
_TRAIL_RE = re.compile(r"(알려줘|설명해줘|정리해줘|봐줘|보여줘|필요해)$")


def expand_question(question: str, state: DialogueState | None) -> str:
    q = (question or "").strip()
    if not state or not state.last_question:
        return q
    if not is_followup_text(q):
        return q
    topic = state.last_topic or ""
    entities = " ".join(state.last_entities[:4])
    bits = " ".join(x for x in (topic, entities) if x).strip()
    prefix = state.last_question
    if bits:
        return f"{prefix} (이어서, 주제 {bits}). 후속: {q}"
    return f"{prefix} (이어서). 후속: {q}"


def split_hybrid_queries(question: str, state: DialogueState | None = None) -> tuple[str, str]:
    q = (question or "").strip()
    core = re.sub(r"같이|둘\s*다|동시에|양쪽", " ", q)
    parts = [p.strip() for p in _SPLIT_RE.split(core) if p.strip()]
    parts = [_TRAIL_RE.sub("", p).strip() for p in parts]
    parts = [p for p in parts if len(p) >= 2]

    ops_parts: list[str] = []
    rag_parts: list[str] = []
    if len(parts) >= 2:
        for part in parts:
            ops, rag = score_question(part)
            if ops >= rag:
                ops_parts.append(part)
            else:
                rag_parts.append(part)

    ops_q = " ".join(ops_parts).strip() or _ops_focus(q, state)
    rag_q = " ".join(rag_parts).strip() or _rag_focus(q, state)
    if state:
        if state.last_topic and state.last_topic in {"cii", "voyage", "fuel", "position", "report", "seemp"}:
            if "CII" not in ops_q.upper() and state.last_topic == "cii":
                ops_q = f"{ops_q} CII".strip()
        if state.last_topic in {"mepc", "msc", "class", "table"} and state.last_topic not in rag_q.lower():
            rag_q = f"{state.last_topic} {rag_q}".strip()
    return (
        f"우리 선박 운항 데이터만 사용해 답하세요: {ops_q}",
        f"규정·회의·선급 문서만 사용해 답하세요: {rag_q}",
    )


def _ops_focus(question: str, state: DialogueState | None) -> str:
    topics = extract_topics(question)
    ops_topics = [t for t in topics if t in {"cii", "voyage", "fuel", "position", "report", "seemp"}]
    entities = extract_entities(question)
    if state:
        entities = list(dict.fromkeys([*entities, *state.last_entities]))[:4]
        if not ops_topics and state.last_topic in {"cii", "voyage", "fuel", "position", "report", "seemp"}:
            ops_topics = [state.last_topic]
    label = " ".join(ops_topics or ["운항 수치"])
    ship = "올해 우리 선박" if re.search(r"올해", question or "") else "우리 선박"
    extra = " ".join(entities)
    return f"{ship} {label} {extra}".strip()


def _rag_focus(question: str, state: DialogueState | None) -> str:
    topics = extract_topics(question)
    rag_topics = [t for t in topics if t in {"mepc", "msc", "class", "table", "seemp"}]
    if state and not rag_topics and state.last_topic in {"mepc", "msc", "class", "table", "seemp"}:
        rag_topics = [state.last_topic]
    label = " ".join(rag_topics or ["관련 규정·회의"])
    return f"{label} 요지: {question}".strip()
