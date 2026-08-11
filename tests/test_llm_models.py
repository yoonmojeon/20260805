from __future__ import annotations

import inspect

import pytest

from services.llm_models import (
    AVAILABLE_LLM_MODELS,
    DEFAULT_LLM_MODEL,
    LLM_MODEL_CHOICES,
    normalize_llm_model,
)
from services.orchestrator import handle_question
from services.routing_options import (
    DEFAULT_ROUTING_MODE,
    LLM_PRIMARY_MODE,
    ROUTING_MODE_CHOICES,
    RULES_ONLY_MODE,
    use_llm_primary_router,
)


def test_ui_offers_three_models_with_gemma_as_default() -> None:
    assert DEFAULT_LLM_MODEL == "gemma4:12b"
    assert AVAILABLE_LLM_MODELS == (
        "gemma4:12b",
        "llama3.1:8b",
        "mistral-nemo:12b",
    )
    assert tuple(value for _label, value in LLM_MODEL_CHOICES) == AVAILABLE_LLM_MODELS


def test_unknown_or_empty_model_falls_back_to_gemma() -> None:
    assert normalize_llm_model(None) == "gemma4:12b"
    assert normalize_llm_model("") == "gemma4:12b"
    assert normalize_llm_model("not-installed") == "gemma4:12b"


def test_orchestrator_uses_llm_primary_by_default() -> None:
    parameter = inspect.signature(handle_question).parameters["use_llm_router"]
    assert parameter.default is True


def test_ui_defaults_to_llm_primary_but_keeps_rules_only() -> None:
    assert DEFAULT_ROUTING_MODE == LLM_PRIMARY_MODE
    assert tuple(value for _label, value in ROUTING_MODE_CHOICES) == (
        LLM_PRIMARY_MODE,
        RULES_ONLY_MODE,
    )
    assert use_llm_primary_router(LLM_PRIMARY_MODE) is True
    assert use_llm_primary_router(RULES_ONLY_MODE) is False
    assert use_llm_primary_router(None) is True


@pytest.mark.parametrize("model", AVAILABLE_LLM_MODELS)
def test_selected_model_reaches_router_and_answer_metadata(model: str) -> None:
    result = handle_question(
        "서비스 소개를 해줘",
        force_route="chat",
        llm_model=model,
    )
    assert result["llm_model"] == model
    assert result["route"]["model"] == model
    assert result["meta"]["active_model"] == model
    assert result["meta"]["answer_model"] == model
