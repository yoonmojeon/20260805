"""Selectable Ollama chat models for the Gradio UI / orchestrator."""
from __future__ import annotations

# Keep in sync with models used in local quality-30 runs.
AVAILABLE_LLM_MODELS: tuple[str, ...] = (
    "gemma4:12b",
    "llama3.1:8b",
)

LLM_MODEL_CHOICES: tuple[tuple[str, str], ...] = (
    ("Gemma 4 12B (기본·권장)", "gemma4:12b"),
    ("Llama 3.1 8B (빠른 응답)", "llama3.1:8b"),
)

DEFAULT_LLM_MODEL = "gemma4:12b"


def normalize_llm_model(model: str | None) -> str:
    name = (model or "").strip()
    if name in AVAILABLE_LLM_MODELS:
        return name
    return DEFAULT_LLM_MODEL
