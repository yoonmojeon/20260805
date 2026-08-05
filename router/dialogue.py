"""Multi-turn dialogue state carried across Gradio turns."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DialogueState:
    last_route: str | None = None
    last_topic: str | None = None
    last_entities: list[str] = field(default_factory=list)
    last_question: str | None = None
    last_ops_query: str | None = None
    last_rag_query: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_dialogue_state(
    dialogue_state: DialogueState | dict | None = None,
    last_route: str | None = None,
) -> DialogueState:
    if isinstance(dialogue_state, DialogueState):
        state = dialogue_state
    elif isinstance(dialogue_state, dict) and dialogue_state:
        state = DialogueState(
            last_route=dialogue_state.get("last_route"),
            last_topic=dialogue_state.get("last_topic"),
            last_entities=list(dialogue_state.get("last_entities") or []),
            last_question=dialogue_state.get("last_question"),
            last_ops_query=dialogue_state.get("last_ops_query"),
            last_rag_query=dialogue_state.get("last_rag_query"),
        )
    else:
        state = DialogueState(last_route=last_route)
    if last_route and not state.last_route:
        state.last_route = last_route
    return state


def next_dialogue_state(
    *,
    previous: DialogueState,
    route: str,
    question: str,
    topics: list[str],
    entities: list[str],
    ops_query: str | None,
    rag_query: str | None,
) -> DialogueState:
    keep_route = previous.last_route
    next_route = route if route in {"ops", "rag", "hybrid"} else keep_route
    next_topics = topics or ([previous.last_topic] if previous.last_topic else [])
    merged_entities = list(dict.fromkeys([*entities, *previous.last_entities]))[:8]
    return DialogueState(
        last_route=next_route,
        last_topic=next_topics[0] if next_topics else previous.last_topic,
        last_entities=merged_entities,
        last_question=question if route in {"ops", "rag", "hybrid"} else previous.last_question,
        last_ops_query=ops_query or previous.last_ops_query,
        last_rag_query=rag_query or previous.last_rag_query,
    )
