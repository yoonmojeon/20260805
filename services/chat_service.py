"""안내(chat) 경로 — 고정 템플릿. LLM/DB/RAG를 호출하지 않는다."""
from __future__ import annotations

from typing import Any

from prompts.chat import render_chat_answer


def run_chat_query(
    question: str,
    history: list | None = None,
    *,
    chat_mode: str | None = None,
) -> dict[str, Any]:
    answer = render_chat_answer(question, chat_mode=chat_mode)
    hist = list(history or [])
    hist.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )
    return {
        "answer": answer,
        "history": hist,
        "files": [],
        "map_html": "",
        "source": "chat",
        "meta": {"chat_mode": chat_mode or "clarify"},
    }
