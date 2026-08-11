"""Bridge to ship-data Maritime Ops Agent (SQLite KPI / CII / reports)."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from project_paths import OPS_DIR, OPS_DB_PATH, REPORTS_DIR

if str(OPS_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_DIR))


def ops_db_ready() -> bool:
    return OPS_DB_PATH.exists() and OPS_DB_PATH.stat().st_size > 0


def _try_deterministic_ops(question: str) -> dict[str, Any] | None:
    """Slot-based shortcuts so clear voyage/CII asks do not depend on LLM tool args."""
    q = (question or "").strip()
    if not q:
        return None

    from agent.briefing import build_answer_from_tools
    from agent.tools import (
        calculate_cii_rating,
        get_current_voyage_status,
        get_voyage_analysis,
    )

    # CII grade / attained-required
    if re.search(r"\bCII\b|탄소집약", q, re.I) and re.search(
        r"등급|attained|required|잠정|YTD|올해|연간", q, re.I
    ):
        year_m = re.search(r"(20\d{2})", q)
        year = int(year_m.group(1)) if year_m else None
        result = calculate_cii_rating(year=year)
        formatted = build_answer_from_tools(
            [("calculate_cii_rating", {"year": year}, result)]
        )
        if formatted:
            answer, _show = formatted
            return {"answer": answer, "tool": "calculate_cii_rating"}

    # Current status summary
    if re.search(r"현재\s*(운항\s*)?상태|운항\s*상태\s*요약|지금\s*운항", q, re.I):
        result = get_current_voyage_status()
        formatted = build_answer_from_tools(
            [("get_current_voyage_status", {}, result)]
        )
        if formatted:
            answer, _show = formatted
            return {"answer": answer, "tool": "get_current_voyage_status"}

    # Voyage analysis: 이전/현재/올해 + optional Laden/Ballast
    if re.search(
        r"이전\s*항차|직전\s*항차|현재\s*항차|이번\s*항차|올해\s*(항차|누계)|\bYTD\b",
        q,
        re.I,
    ) or (
        re.search(r"\b(Laden|Ballast)\b", q, re.I)
        and re.search(r"항차|연료|배출|실적", q)
    ):
        period = "current"
        if re.search(r"이전|직전|previous", q, re.I):
            period = "previous"
        elif re.search(r"올해|YTD|누계", q, re.I):
            period = "ytd"
        cond_m = re.search(r"\b(Laden|Ballast)\b", q, re.I)
        fake_id = ""
        if period == "previous" and cond_m:
            fake_id = f"이전 항차({cond_m.group(1)})"
        elif cond_m:
            fake_id = cond_m.group(1)
        result = get_voyage_analysis(voyage_id=fake_id, period=period)
        if isinstance(result, dict) and result.get("error"):
            return None
        formatted = build_answer_from_tools(
            [("get_voyage_analysis", {"period": period, "voyage_id": fake_id}, result)]
        )
        if formatted:
            answer, _show = formatted
            return {"answer": answer, "tool": "get_voyage_analysis"}

    return None


def run_ops_query(
    question: str,
    history: list | None = None,
    *,
    llm_model: str | None = None,
) -> dict[str, Any]:
    """
    Returns:
      answer, history, files, show_map, map_html
    """
    if not ops_db_ready():
        return {
            "answer": (
                "운항 SQLite DB가 아직 없습니다. 다음을 실행하세요:\n"
                "  python ops/scripts/load_hodata.py"
            ),
            "history": history or [],
            "files": [],
            "show_map": False,
            "map_html": "",
            "source": "ops",
        }

    from agent.maritime_agent import run_agent_sync
    from agent.tools import render_voyage_map
    from services.llm_models import normalize_llm_model

    model = normalize_llm_model(llm_model)

    deterministic = _try_deterministic_ops(question)
    shortcut_on = os.environ.get("MARITIME_OPS_DETERMINISTIC_SHORTCUTS", "0").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }
    if shortcut_on and deterministic and deterministic.get("answer"):
        answer = str(deterministic["answer"])
        new_history = list(history or []) + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        return {
            "answer": answer,
            "history": new_history,
            "files": [],
            "show_map": False,
            "map_html": "",
            "source": "ops",
            "reports_dir": str(REPORTS_DIR),
            "deterministic_tool": deterministic.get("tool"),
            "llm_model": model,
        }

    answer, new_history, files, show_map = run_agent_sync(
        question, list(history or []), model=model
    )
    answer_fallback_used = False
    if deterministic and deterministic.get("answer") and (
        not str(answer or "").strip() or str(answer).startswith("[LLM 오류]")
    ):
        answer = str(deterministic["answer"])
        new_history = list(history or []) + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        answer_fallback_used = True
    map_html = ""
    if show_map:
        try:
            map_html = render_voyage_map()
        except Exception:
            map_html = ""
    existing = [f for f in (files or []) if Path(f).exists()]
    return {
        "answer": answer,
        "history": new_history,
        "files": existing,
        "show_map": show_map,
        "map_html": map_html,
        "source": "ops",
        "reports_dir": str(REPORTS_DIR),
        "llm_model": model,
        "answer_fallback_used": answer_fallback_used,
    }
