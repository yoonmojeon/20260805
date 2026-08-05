"""
운항 브리핑 — 툴 결과를 기대 답변 수준의 줄글로 변환
LLM이 KPI 숫자를 누락하지 않도록 Python에서 직접 서술형 답변 생성
"""
from __future__ import annotations

from config import CURRENT_DATE


def _fmt(val, decimals=2, suffix=""):
    try:
        v = float(val or 0)
        return f"{v:.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return str(val) if val not in (None, "") else "미제공"


def _pos_text(lat: float, lon: float) -> str:
    lat_s = f"{lat:.3f}°N" if lat else "미제공"
    if lon and lon != 0:
        lon_s = f"{lon:.3f}°E"
    else:
        lon_s = "미제공 (원본 경도 미기록 구간)"
    return f"위도 {lat_s}, 경도 {lon_s}"


_CII_SCOPE_LABEL = {
    "annual":         "공식 연간 CII",
    "current_voyage": "현재 항차 잠정(Indicative) CII",
    "voyage":         "항차 참고(Indicative Voyage) CII",
}


def _cii_text(cii: dict) -> str:
    """
    agent/cii.py compute_cii() 결과(dict)를 한국어 줄글로 변환.
    숫자 계산은 전혀 하지 않고 이미 계산된 값만 포맷한다.
    """
    if not cii:
        return "CII 등급은 산출할 수 없습니다."

    if cii.get("status") != "success":
        reason = cii.get("reason") or "데이터가 부족합니다."
        return f"CII를 계산할 수 없습니다. ({reason})"

    scope_label = _CII_SCOPE_LABEL.get(cii.get("scope"), "CII")
    rating = cii.get("rating")
    ratio_pct = (cii.get("rating_ratio") or 0) * 100

    parts = [
        f"[{scope_label}] 기준 연도 {cii.get('year', '')}, "
        f"선종/DWT {cii.get('ship_type', '')} / {_fmt(cii.get('dwt'), 0)} DWT."
    ]
    parts.append(
        f"총 CO₂ {_fmt(cii.get('co2_mt'), 2, ' MT')}, "
        f"운항 거리 {_fmt(cii.get('distance_nm'), 1, ' nm')}."
    )
    parts.append(
        f"Attained CII {_fmt(cii.get('attained_cii'), 3, ' gCO₂/(DWT·nm)')}, "
        f"Reference CII {_fmt(cii.get('reference_cii'), 3, ' gCO₂/(DWT·nm)')}, "
        f"Required CII {_fmt(cii.get('required_cii'), 3, ' gCO₂/(DWT·nm)')} "
        f"(감축계수 Z={_fmt(cii.get('reduction_factor_percent'), 3, '%')})."
    )
    parts.append(
        f"Required CII 대비 {_fmt(ratio_pct, 1, '%')}이며, CII 등급은 {rating}입니다."
    )
    if cii.get("warning"):
        parts.append(f"※ {cii['warning']}")
    return " ".join(parts)


def _distance_text(d: dict) -> str:
    dist = d.get("distance_nm", 0)
    method = d.get("distance_method", "")
    note = d.get("distance_note", "")
    days = float(d.get("days_at_sea", 0) or 0)
    avg = float(d.get("avg_sog_kts", 0) or 0)

    if method == "sog_integration":
        base = f"항주 거리 약 {_fmt(dist, 1, ' nm')} (SOG 적분 추정"
        if note:
            base += f", {note}"
        base += ")"
        implied = days * 24 * avg if days and avg else 0
        if implied > 0 and abs(dist - implied) / max(dist, 1) < 0.2:
            base += f". 항해 {_fmt(days, 1)}일·평균 {_fmt(avg, 1)}kn과 일치합니다"
        return base + "."
    if dist:
        return f"항주 거리 {_fmt(dist, 1, ' nm')} ({note or '좌표 기반'})."
    return "항주 거리: 위치 결측으로 산출 불가"


def format_current_status(d: dict) -> str:
    if d.get("error"):
        return f"운항 데이터를 조회할 수 없습니다. ({d['error']})"

    pos = d.get("position", {})
    lat = float(pos.get("latitude", 0) or 0)
    lon = float(pos.get("longitude", 0) or 0)
    cii = d.get("cii_ytd", {})

    p1 = (
        f"【시점 기준】 현재 = 항차 {d.get('current_voyage_id', '')} "
        f"(Voyage No 기준, {d.get('voyage_start', '')} ~ {d.get('voyage_end', '')}) 구간, "
        f"최신 센서 시각 {d.get('last_reading_time', CURRENT_DATE)} 기준입니다. "
        f"선박 {d.get('ship_name', '')}(IMO {d.get('ship_imo', '')})은 "
        f"{d.get('departure_port', '미제공')}에서 {d.get('arrival_port', '미제공')} 방향으로 "
        f"운항 중이며, 항해 일수는 {_fmt(d.get('days_at_sea'), 1)}일입니다. "
        f"{_distance_text(d)}"
    )

    p2 = (
        f"【위치·운항】 {_pos_text(lat, lon)}. "
        f"Loading 상태는 {d.get('loading_status', '미제공')}입니다. "
        f"현재 선속(SOG)은 {_fmt(d.get('sog_kts'), 1, ' 노트')}, "
        f"평균 선속은 {_fmt(d.get('avg_sog_kts'), 1, ' 노트')}입니다. "
        f"M/E RPM은 {d.get('me_rpm_note', '미측정')}이며, "
        f"주기관 출력은 {_fmt(d.get('me_power_kw'), 0, ' kW')}입니다."
    )

    p3 = (
        f"【연료·배출 (항차 누계)】 "
        f"FOC(Oil) {_fmt(d.get('voyage_foc_oil_mt'), 2, ' MT')}, "
        f"FGC(Gas) {_fmt(d.get('voyage_fgc_gas_mt'), 2, ' MT')}, "
        f"CO₂ {_fmt(d.get('voyage_co2_mt'), 2, ' MT')}, "
        f"CH₄ {_fmt(d.get('voyage_ch4_mt'), 4, ' MT')}, "
        f"CO₂e {_fmt(d.get('voyage_co2e_mt'), 2, ' MT')}입니다. "
        f"순간 소비율은 Oil {_fmt(d.get('foc_oil_rate_mt_h'), 4, ' MT/h')}, "
        f"Gas {_fmt(d.get('fgc_gas_rate_mt_h'), 4, ' MT/h')}, "
        f"CO₂ {_fmt(d.get('co2_rate_mt_h'), 4, ' MT/h')}입니다."
    )

    p4 = f"【CII】 {_cii_text(cii)} (연초~현재 YTD 기준, 현재 항차와 기간이 다릅니다.)"

    p5 = (
        "【데이터 한계】 원본에 항만명·RPM·기상 정보가 없습니다. "
        "경도(lon=0) 구간은 위치·지도에 제한이 있습니다."
    )

    return "\n\n".join([p1, p2, p3, p4, p5])


def format_voyage_analysis(d: dict, period: str = "current") -> str:
    if d.get("error"):
        return f"항차 분석 데이터를 조회할 수 없습니다. ({d['error']})"

    labels = {
        "current":  "현재 항차 (항차 시작일 ~ 현재)",
        "previous": "이전 항차 (직전 완료 항차)",
        "ytd":      f"올해 누계 ({CURRENT_DATE[:4]}-01-01 ~ {CURRENT_DATE})",
    }
    label = labels.get(period, d.get("period", "지정 항차"))

    if period == "ytd" or "summary" in d:
        s = d.get("summary", d)
        cii = d.get("cii", {})
        p1 = (
            f"【시점 기준】 {label}입니다. "
            f"총 항차 수 {s.get('voyages_count', 0)}회, "
            f"총 항해 거리 {_fmt(s.get('total_distance_nm'), 1, ' nm')}, "
            f"총 항해 일수 {_fmt(s.get('total_days_at_sea'), 1, '일')}입니다."
        )
        p2 = (
            f"【연료·배출】 FOC(Oil) {_fmt(s.get('total_foc_oil_mt'), 2, ' MT')}, "
            f"FGC(Gas) {_fmt(s.get('total_fgc_gas_mt'), 2, ' MT')}, "
            f"CO₂ {_fmt(s.get('total_co2_mt'), 2, ' MT')}, "
            f"CH₄ {_fmt(s.get('total_ch4_mt'), 4, ' MT')}, "
            f"CO₂e {_fmt(s.get('total_co2e_mt'), 2, ' MT')}입니다."
        )
        p3 = f"【CII】 {_cii_text(cii)}"
        return "\n\n".join([p1, p2, p3])

    p1 = (
        f"【시점 기준】 {label} — 항차 {d.get('voyage_id', '')} "
        f"({d.get('voyage_start', '')} ~ {d.get('voyage_end', '')}). "
        f"{d.get('departure_port', '미제공')} → {d.get('arrival_port', '미제공')}, "
        f"항해 {_fmt(d.get('days_at_sea'), 1, '일')}. "
        f"{_distance_text({**d, 'avg_sog_kts': d.get('avg_sog_kts')})}"
    )
    p2 = (
        f"【운항 KPI】 평균 선속 {_fmt(d.get('avg_sog_kts'), 1, ' 노트')}. "
        f"Loading 상태는 미제공, M/E RPM은 미측정입니다. "
        f"FOC {_fmt(d.get('foc_oil_mt'), 2, ' MT')}, "
        f"FGC {_fmt(d.get('fgc_gas_mt'), 2, ' MT')}, "
        f"일평균 연료 {_fmt(d.get('foc_per_day_mt'), 2, ' MT/일')}입니다."
    )
    p3 = (
        f"【배출】 CO₂ {_fmt(d.get('co2_mt'), 2, ' MT')}, "
        f"CH₄ {_fmt(d.get('ch4_mt'), 4, ' MT')}, "
        f"CO₂e {_fmt(d.get('co2e_mt'), 2, ' MT')}, "
        f"탄소강도 {_fmt(d.get('cii_kg_per_nm'), 1, ' kg CO₂/nm')}입니다."
    )
    p4 = f"【CII】 {_cii_text(d.get('cii', {}))}"
    return "\n\n".join([p1, p2, p3, p4])


def format_cii(d: dict) -> str:
    if d.get("error"):
        return f"CII를 계산할 수 없습니다. ({d['error']})"
    return f"【CII 등급】 {_cii_text(d)}"


def format_emissions(d: dict) -> str:
    if d.get("error"):
        return f"배출량을 조회할 수 없습니다. ({d['error']})"
    return (
        f"【배출량】 항차 {d.get('voyage_id', '')} 기준, "
        f"FOC {_fmt(d.get('foc_oil_mt'), 2, ' MT')}, "
        f"FGC {_fmt(d.get('fgc_gas_mt'), 2, ' MT')}, "
        f"CO₂ {_fmt(d.get('co2_mt'), 2, ' MT')}, "
        f"CH₄ {_fmt(d.get('ch4_mt'), 4, ' MT')}, "
        f"CO₂e {_fmt(d.get('co2e_mt'), 2, ' MT')}입니다. "
        f"{d.get('note', '')}"
    )


def _report_cii_line(d: dict) -> str:
    cii = d.get("cii")
    if not cii:
        return ""
    if cii.get("status") != "success":
        return f" CII: 계산 불가 ({cii.get('reason', '데이터 부족')})."
    return (
        f" {_CII_SCOPE_LABEL.get(cii.get('scope'), 'CII')} 등급 {cii.get('rating')} "
        f"(Attained {_fmt(cii.get('attained_cii'), 3)} / "
        f"Required {_fmt(cii.get('required_cii'), 3)} gCO₂/(DWT·nm))."
    )


def format_report_result(tool_name: str, d: dict) -> str:
    if d.get("error"):
        return f"보고서 생성에 실패했습니다. ({d['error']})"

    if tool_name == "generate_noon_report":
        return (
            f"Noon Report를 생성했습니다. 보고일 {d.get('report_date', '')}, "
            f"위치 {d.get('position', '')}, "
            f"FOC {_fmt(d.get('foc_oil_mt'), 3, ' MT')}, "
            f"FGC {_fmt(d.get('fgc_gas_mt'), 3, ' MT')}, "
            f"CO₂ {_fmt(d.get('co2_mt'), 2, ' MT')}."
            f"{_report_cii_line(d)} "
            f"아래에서 Word 파일을 다운로드하세요. "
            f"(Position, Heading, Sailed Distance, Displacement, M/E RPM, "
            f"Ship speed, Wind/Wave, M/E FOC·FGC, 잠정 CII 포함)"
        )
    if tool_name == "generate_mrv_voyage_report":
        return (
            f"MRV Voyage Report를 생성했습니다. "
            f"항차 {d.get('voyage_id', '')}, 구간 {d.get('route', '')}, "
            f"CO₂ {_fmt(d.get('co2_mt'), 2, ' MT')}."
            f"{_report_cii_line(d)} "
            f"아래에서 Word 파일을 다운로드하세요. "
            f"(Voyage ID, 항만, 거리, 항해시간, 연료별 소비, CO₂, 화물/운송일, "
            f"항차 참고 CII 포함)"
        )
    if tool_name == "generate_mrv_annual_report":
        return (
            f"MRV Annual Report({d.get('year', '')}년)를 생성했습니다. "
            f"총 {d.get('voyages', 0)}개 항차, "
            f"CO₂ {_fmt(d.get('total_co2_mt'), 2, ' MT')}."
            f"{_report_cii_line(d)} "
            f"아래에서 Word 파일을 다운로드하세요. "
            f"(보고기간, 연료·배출·거리·운송일 누계, 공식 연간 CII 등급 포함)"
        )
    return d.get("status", "완료")


def build_answer_from_tools(tool_results: list[tuple[str, dict, dict]]) -> tuple[str, bool] | None:
    """브리핑/리포트 툴 결과가 있으면 줄글 답변 생성. 없으면 None."""
    if not tool_results:
        return None

    parts = []
    show_map = False
    for tool_name, args, result in tool_results:
        if result.get("error"):
            parts.append(f"[오류] {result['error']}")
            continue

        if tool_name == "get_current_voyage_status":
            parts.append(format_current_status(result))
            show_map = True
        elif tool_name == "get_voyage_analysis":
            period = args.get("period", "current")
            parts.append(format_voyage_analysis(result, period))
            if period == "current":
                show_map = True
        elif tool_name == "calculate_cii_rating":
            parts.append(format_cii(result))
        elif tool_name == "calculate_emissions":
            parts.append(format_emissions(result))
        elif tool_name.startswith("generate_"):
            parts.append(format_report_result(tool_name, result))

    if not parts:
        return None
    return "\n\n".join(parts), show_map
