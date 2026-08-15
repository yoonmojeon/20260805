"""Operations map rendering and deterministic current-status behavior."""
from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

from agent.tools import render_voyage_map  # noqa: E402
from services.ops_service import run_ops_query  # noqa: E402


def test_voyage_map_is_gradio_embeddable_without_notebook_warning():
    rendered = render_voyage_map()

    assert rendered.startswith("<div")
    assert "유효 좌표 구간만 표시합니다" in rendered
    assert '<iframe title="현재 항차 이동 경로 지도"' in rendered
    assert "data:text/html;base64," in rendered
    assert "Make this Notebook Trusted" not in rendered

    payload = re.search(r"base64,([^\"]+)", rendered)
    assert payload is not None
    document = base64.b64decode(payload.group(1)).decode("utf-8")
    assert "leaflet" in document.lower()


def test_current_status_shortcut_includes_map():
    result = run_ops_query("현재 운항상태는?", llm_model="gemma4:12b")

    assert result["deterministic_tool"] == "get_current_voyage_status"
    assert result["show_map"] is True
    assert 'title="현재 항차 이동 경로 지도"' in result["map_html"]
