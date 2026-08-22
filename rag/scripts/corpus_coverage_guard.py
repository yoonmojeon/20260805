"""Detect timeline requests that exceed the meeting sessions in this corpus."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from question_profile import QuestionProfile


@lru_cache(maxsize=4)
def _available_sessions(profile_path: str) -> dict[str, list[int]]:
    path = Path(profile_path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = payload.get("documents") if isinstance(payload, dict) else payload
    out: dict[str, set[int]] = {}
    for item in documents or []:
        org = str(item.get("session_org") or "").upper()
        number = item.get("session_number")
        if org in {"MSC", "MEPC"} and number not in (None, ""):
            out.setdefault(org, set()).add(int(number))
    return {key: sorted(values) for key, values in out.items()}


def build_coverage_guard(
    profile: QuestionProfile,
    *,
    document_profile_path: Path,
) -> dict[str, Any]:
    available = _available_sessions(str(document_profile_path.resolve()))
    requested: list[int] = []
    org = ""
    if profile.session_range:
        org, start, end = profile.session_range
        requested = list(range(start, end + 1))
    elif profile.time_scope == "latest_available":
        # "Latest" is answerable as latest *indexed* material.  It is not a
        # claim that the corpus is externally complete.
        return {
            "active": True,
            "complete": False,
            "scope": "latest_indexed",
            "notice": "최신성은 현재 색인된 문서 범위 기준입니다.",
        }
    else:
        return {"active": False, "complete": True, "missing_sessions": []}
    present = available.get(org, [])
    missing = [number for number in requested if number not in set(present)]
    return {
        "active": True,
        "complete": not missing,
        "scope": "session_range",
        "organization": org,
        "requested_sessions": requested,
        "available_sessions": present,
        "missing_sessions": missing,
        "notice": (
            "요청 범위 중 색인에 없는 회차: "
            + ", ".join(f"{org} {number}" for number in missing)
            if missing
            else "요청한 회차 범위가 색인에 포함되어 있습니다."
        ),
    }


def apply_coverage_notice(answer: str, guard: dict[str, Any] | None) -> str:
    """Add a non-factual corpus boundary note without changing answer claims."""
    guard = guard or {}
    if not guard.get("active") or guard.get("complete"):
        return answer
    notice = str(guard.get("notice") or "").strip()
    if not notice or notice in answer:
        return answer
    return f"> **검색 범위 주의**: {notice}\n\n{answer}"
