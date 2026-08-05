"""Route-specific prompts. Chat / ops / rag keep separate system text."""

from prompts.chat import (
    CHAT_SYSTEM_PROMPT,
    render_chat_answer,
)
from prompts.ops import build_ops_system_prompt
from prompts.rag import RAG_ROUTE_IDENTITY, apply_rag_identity
from prompts.router_prompt import ROUTER_SYSTEM_PROMPT, build_router_user_prompt

__all__ = [
    "CHAT_SYSTEM_PROMPT",
    "RAG_ROUTE_IDENTITY",
    "ROUTER_SYSTEM_PROMPT",
    "apply_rag_identity",
    "build_ops_system_prompt",
    "build_router_user_prompt",
    "render_chat_answer",
]
