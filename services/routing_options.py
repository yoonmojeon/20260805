"""Routing-mode choices shared by the Gradio UI and tests."""
from __future__ import annotations

LLM_PRIMARY_MODE = "llm_primary"
RULES_ONLY_MODE = "rules_only"
DEFAULT_ROUTING_MODE = LLM_PRIMARY_MODE

ROUTING_MODE_CHOICES: tuple[tuple[str, str], ...] = (
    ("LLM-primary (기본·권장)", LLM_PRIMARY_MODE),
    ("Rules-only (비교용)", RULES_ONLY_MODE),
)


def use_llm_primary_router(mode: str | None) -> bool:
    """Unknown or empty UI values safely keep the recommended LLM-primary mode."""
    return (mode or DEFAULT_ROUTING_MODE).strip() != RULES_ONLY_MODE
