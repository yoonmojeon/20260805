"""Local-only UI feedback persistence for RAG quality review."""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_PATH = ROOT / "data" / "processed" / "feedback" / "ui_feedback.jsonl"
_LOCK = threading.Lock()


def save_feedback(
    *,
    question: str,
    answer_html: str,
    rating: str,
    mode: str,
    model: str,
    workspace: str,
) -> dict:
    question = str(question or "").strip()
    if not question:
        return {"ok": False, "reason": "missing_question"}
    normalized_rating = "helpful" if rating == "helpful" else "needs_improvement"
    answer = str(answer_html or "")[:24000]
    record = {
        "schema_version": "maritime-ui-feedback-v1",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "rating": normalized_rating,
        "question": question[:2000],
        "answer_html": answer,
        "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "mode": str(mode or "accurate"),
        "model": str(model or ""),
        "workspace": str(workspace or "document"),
    }
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with FEEDBACK_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True, "path": str(FEEDBACK_PATH), "rating": normalized_rating}
