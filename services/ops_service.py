"""Bridge to ship-data Maritime Ops Agent (SQLite KPI / CII / reports)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from project_paths import OPS_DIR, OPS_DB_PATH, REPORTS_DIR

if str(OPS_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_DIR))


def ops_db_ready() -> bool:
    return OPS_DB_PATH.exists() and OPS_DB_PATH.stat().st_size > 0


def run_ops_query(question: str, history: list | None = None) -> dict[str, Any]:
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

    answer, new_history, files, show_map = run_agent_sync(question, list(history or []))
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
    }
