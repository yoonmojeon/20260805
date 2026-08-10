"""Ops voyage_id / period / condition normalization."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

from agent.tools import get_voyage_analysis, resolve_voyage_query  # noqa: E402


def test_resolve_strips_laden_as_voyage_id():
    vid, period, cond = resolve_voyage_query("Laden", "current")
    assert vid == ""
    assert cond == "Laden"
    assert period in {"current", "previous"}


def test_resolve_previous_laden_phrase():
    vid, period, cond = resolve_voyage_query("이전 항차(Laden)", "current")
    assert vid == ""
    assert period == "previous"
    assert cond == "Laden"


def test_resolve_keeps_real_voyage_id():
    vid, period, cond = resolve_voyage_query("H2521_V20_Laden", "current")
    assert vid.startswith("H2521_V20")
    assert period == "current"


def test_previous_laden_analysis_not_empty():
    out = get_voyage_analysis(voyage_id="이전 항차(Laden)", period="current")
    assert "error" not in out, out
    assert "V20" in str(out.get("voyage_id") or "") or "Laden" in str(
        out.get("loading_status") or out.get("voyage_id") or ""
    )
    assert float(out.get("distance_nm") or 0) > 0


def test_previous_period_analysis():
    out = get_voyage_analysis(period="previous")
    assert "error" not in out, out
    assert out.get("voyage_id")
