from pathlib import Path


def test_latest_noon_uses_actual_db_window_and_missing_values():
    from ops.agent.data_store import get_store

    noon = get_store().latest_noon()

    assert noon["report_datetime"] == "2026-06-28 04:00:00"
    assert noon["period_start"] == "2026-06-28 00:00:00"
    assert noon["period_end"] == "2026-06-28 04:00:00"
    assert noon["record_count"] == 5
    assert noon["partial_day"] is True
    assert noon["lon"] is None
    assert noon["cog_deg"] is None
    assert noon["wind_speed_kts"] is None
    assert noon["co2_per_nm_kg"] > 0


def test_noon_report_shortcut_is_complete_and_attaches_file():
    from services.ops_service import _try_deterministic_ops

    result = _try_deterministic_ops("noon report 생성")

    assert result is not None
    assert result["tool"] == "generate_noon_report"
    assert result["files"] and Path(result["files"][0]).exists()
    answer = result["answer"]
    assert "2026-06-28 04:00 UTC" in answer
    assert "부분일 누계, 5건" in answer
    assert "경도 미제공" in answer
    assert "연초~DB 최신일 잠정(YTD) CII" in answer
    assert answer.endswith("아래에서 Word 파일을 다운로드해 검토하세요.")
