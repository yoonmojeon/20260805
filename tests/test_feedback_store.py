from __future__ import annotations

import json

from services import feedback_store


def test_feedback_is_written_to_local_jsonl(tmp_path, monkeypatch) -> None:
    target = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback_store, "FEEDBACK_PATH", target)
    result = feedback_store.save_feedback(
        question="테스트 질문",
        answer_html="<div>근거 답변</div>",
        rating="helpful",
        mode="advanced",
        model="gemma4:12b",
        workspace="document",
    )
    assert result["ok"] is True
    row = json.loads(target.read_text(encoding="utf-8"))
    assert row["question"] == "테스트 질문"
    assert row["rating"] == "helpful"
    assert row["mode"] == "advanced"
    assert len(row["answer_sha256"]) == 64


def test_feedback_requires_a_question(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(feedback_store, "FEEDBACK_PATH", tmp_path / "feedback.jsonl")
    result = feedback_store.save_feedback(
        question="",
        answer_html="",
        rating="needs_improvement",
        mode="accurate",
        model="gemma4:12b",
        workspace="document",
    )
    assert result == {"ok": False, "reason": "missing_question"}
