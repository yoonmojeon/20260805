"""Selectable Ollama chat models for the Gradio UI / orchestrator."""
from __future__ import annotations

import os

# Keep in sync with models used in local quality-30 runs.
AVAILABLE_LLM_MODELS: tuple[str, ...] = (
    "llama3.1:8b",
    "gemma4:12b",
    "mistral-nemo:12b",
)

DEFAULT_LLM_MODEL = os.environ.get("MARITIME_OLLAMA_MODEL") or os.environ.get(
    "MODEL_NAME", "llama3.1:8b"
)
if DEFAULT_LLM_MODEL not in AVAILABLE_LLM_MODELS:
    DEFAULT_LLM_MODEL = "llama3.1:8b"


def normalize_llm_model(model: str | None) -> str:
    name = (model or "").strip()
    if name in AVAILABLE_LLM_MODELS:
        return name
    return DEFAULT_LLM_MODEL
