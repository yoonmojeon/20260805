"""
Maritime Ops Agent - Tool 정의
LLM이 호출하는 7개 도구 + 지도 렌더링
"""
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import VESSEL, CII_PARAMS, CURRENT_DATE, CURRENT_VOYAGE_ID, REPORTS_DIR

from agent.data_store import get_store
from agent import cii as cii_calc
from agent.reports import (
    generate_noon_report_docx,
    generate_mrv_voyage_docx,
    generate_mrv_annual_docx,
)

# Real voyage ids look like H2521_V21_Ballast — never treat Laden/이전 as an id.
_VOYAGE_ID_RE = re.compile(r"^[A-Za-z0-9]+_V\d+", re.IGNORECASE)
_CONDITION_RE = re.compile(r"\b(Laden|Ballast)\b", re.IGNORECASE)
_PREV_RE = re.compile(r"이전|직전|previous", re.IGNORECASE)
_CUR_RE = re.compile(r"현재|이번|current", re.IGNORECASE)
_YTD_RE = re.compile(r"올해|연간|YTD|누계", re.IGNORECASE)


def resolve_voyage_query(
    voyage_id: str = "",
    period: str = "current",
) -> tuple[str, str, str]:
    """Normalize LLM tool args into (voyage_id, period, condition).

    Prevents false ``항차 데이터 없음`` when the model stuffs Laden/이전 into voyage_id.
    """
    raw_id = str(voyage_id or "").strip()
    period = str(period or "current").strip().lower() or "current"
    condition = ""

    blob = raw_id
    cond_m = _CONDITION_RE.search(blob)
    if cond_m:
        token = cond_m.group(1)
        condition = "Laden" if token.lower() == "laden" else "Ballast"

    if raw_id and _VOYAGE_ID_RE.match(raw_id):
        return raw_id, period, condition

    # Non-id tokens: derive period/condition, clear voyage_id.
    if _PREV_RE.search(blob) or period in {"prev", "previous"}:
        period = "previous"
    elif _YTD_RE.search(blob) or period == "ytd":
        period = "ytd"
    elif _CUR_RE.search(blob):
        period = "current"
    elif condition:
        # Bare Laden/Ballast (or "Laden 항차") → previous matching leg, not current.
        period = "previous"

    return "", period, condition


def _pick_voyage(voyage_id: str = "", period: str = "current") -> dict:
    store = get_store()
    vid, period, condition = resolve_voyage_query(voyage_id, period)
    if vid:
        hit = store.get_voyage(vid)
        if hit:
            return hit
        # Soft-fail: malformed id with condition/period still recoverable.
    if period == "ytd":
        return {}
    return store.find_voyage(period=period, condition=condition)


# ── CII 계산 헬퍼 ─────────────────────────────────────────────────────────────
# 실제 공식/등급 판정은 agent/cii.py 에 단일화되어 있다. 이 모듈은
# DataStore에서 CO2·거리·DWT를 모아 agent.cii.compute_cii()에 전달하는 역할만 한다.

def _get_vessel_dwt() -> float:
    """DWT는 DB(vessel.dwt)를 우선 사용하고, 없거나 0/NULL이면 config.VESSEL로 폴백한다."""
    try:
        vessel = get_store().get_vessel()
        dwt = float(vessel.get("dwt", 0) or 0)
        if dwt > 0:
            return dwt
    except Exception:
        pass
    return float(VESSEL.get("dwt", 0) or 0)


def _current_data_year() -> int:
    """데이터 기준 '현재' 연도 (시스템 시계가 아닌 CURRENT_DATE 기준)."""
    return int(str(CURRENT_DATE)[:4])


def _partial_year_note(year: int) -> str | None:
    """요청 연도가 아직 종료되지 않은 데이터 기준 연도라면 잠정치임을 알린다."""
    if year == _current_data_year():
        return (
            f"{year}-01-01 ~ {CURRENT_DATE} 데이터 기준이며, 해당 연도가 아직 "
            f"종료되지 않아 공식 연간 CII 확정치가 아닌 잠정(YTD)치입니다."
        )
    return None


def _annual_cii_result(year: int, scope: str = "annual") -> dict:
    """연간(또는 YTD 잠정) CII — ytd_summary 집계를 사용."""
    store = get_store()
    dwt   = _get_vessel_dwt()
    year  = cii_calc._coerce_year(year)

    # 지원하지 않는 연도는 데이터 유무와 무관하게 먼저 명확히 걸러낸다
    # (임의 보간/외삽 금지 — compute_cii가 이유를 결정론적으로 반환).
    if year is None or cii_calc.reduction_factor_percent(year) is None:
        result = cii_calc.compute_cii(scope=scope, year=year, dwt=dwt,
                                       co2_mt=None, distance_nm=None)
        return result.to_dict()

    ytd    = store.ytd_summary(year)
    co2_mt = ytd.get("total_co2_mt") if ytd else None
    dist   = ytd.get("total_distance_nm") if ytd else None

    if not ytd:
        result = cii_calc.compute_cii(
            scope=scope, year=year, dwt=dwt, co2_mt=None, distance_nm=None,
        )
        result.reason = f"{year}년 운항 데이터가 없습니다."
        return result.to_dict()

    result = cii_calc.compute_cii(
        scope=scope, year=year, dwt=dwt, co2_mt=co2_mt, distance_nm=dist,
    )
    if result.status == "success":
        note = _partial_year_note(year)
        if note:
            result.warning = note
    return result.to_dict()


def _voyage_cii_result(voyage_stats: dict, end_time: str, scope: str = "voyage") -> dict:
    """항차(참고) CII — voyage_stats의 CO2/거리 사용, 연도는 항차 종료일 기준."""
    dwt = _get_vessel_dwt()
    year = int(str(end_time)[:4]) if end_time else None
    co2_mt = voyage_stats.get("co2_mt") if voyage_stats else None
    dist   = voyage_stats.get("distance_nm") if voyage_stats else None
    result = cii_calc.compute_cii(
        scope=scope, year=year, dwt=dwt, co2_mt=co2_mt, distance_nm=dist,
    )
    return result.to_dict()


# ── Tool 1: 현재 운항 상태 ──────────────────────────────────────────────────────
def get_current_voyage_status() -> dict:
    """현재 항차 KPI 전체 (위경도·SOG·FOC·FGC·CO2·CH4·CO2e·CII)"""
    store  = get_store()
    voyage = store.current_voyage()
    vessel = store.get_vessel()

    if not voyage:
        return {"error": "운항 데이터 없음"}

    vid   = voyage.get("voyage_id", CURRENT_VOYAGE_ID)
    stats = store.voyage_stats(vid)

    year = int(str(voyage.get("end_time", CURRENT_DATE))[:4])

    return {
        "ship_name":         vessel.get("name", VESSEL["name"]),
        "ship_imo":          vessel.get("imo",  VESSEL["imo"]),
        "current_voyage_id": vid,
        "reference_period":  f"{str(voyage.get('start_time', ''))[:16]} ~ {str(voyage.get('end_time', ''))[:16]} (현재 항차)",
        "departure_port":    voyage.get("departure_port", "Unknown"),
        "arrival_port":      voyage.get("arrival_port",   "Unknown"),
        "voyage_start":      str(voyage.get("start_time", ""))[:16],
        "voyage_end":        str(voyage.get("end_time",   ""))[:16],
        "days_at_sea":       stats.get("days_at_sea", 0),
        "distance_nm":       stats.get("distance_nm", 0),
        "distance_method":   stats.get("distance_method", ""),
        "distance_note":     stats.get("distance_note", ""),
        "coord_distance_nm": stats.get("coord_distance_nm"),
        "position": {
            "latitude":  stats.get("last_lat", 0),
            "longitude": stats.get("last_lon", 0),
            "note": "경도(lon) 원본 미제공 구간 있음 — 위도·SOG는 실측",
        },
        "sog_kts":            stats.get("last_sog", 0) or stats.get("avg_sog", 0),
        "avg_sog_kts":        stats.get("avg_sog", 0),
        "me_power_kw":        stats.get("last_me_power", 0),
        "me_rpm":             None,
        "me_rpm_note":        "미측정 (원본 데이터에 RPM 컬럼 없음)",
        "loading_status":     voyage.get("condition") or "미제공 (원본 데이터에 Loading 컬럼 없음)",
        # 최신 순간 연료/배출 (kg/h → MT/h)
        "foc_oil_rate_mt_h":  round(float(stats.get("last_oil_rate", 0)) / 1000, 4),
        "fgc_gas_rate_mt_h":  round(float(stats.get("last_gas_rate", 0)) / 1000, 4),
        "co2_rate_mt_h":      round(float(stats.get("last_co2_rate", 0)) / 1000, 4),
        # 항차 누계 (질문 '현재 운항 상태'의 핵심)
        "voyage_foc_oil_mt":  stats.get("foc_oil_mt", 0),
        "voyage_fgc_gas_mt":  stats.get("fgc_gas_mt", 0),
        "voyage_co2_mt":      stats.get("co2_mt",     0),
        "voyage_ch4_mt":      stats.get("ch4_mt",     0),
        "voyage_co2e_mt":     stats.get("co2e_mt",    0),
        "voyage_cii_kg_per_nm": stats.get("cii_value", 0),
        "cii_ytd":            _annual_cii_result(year, scope="current_voyage"),
        "last_reading_time":  stats.get("last_ts", ""),
        "data_source":        "ho_data (sensor)",
        "units": {
            "sog": "knots", "foc_rate": "MT/h", "fgc_rate": "MT/h",
            "co2_rate": "MT/h", "voyage_fuel": "MT", "voyage_emission": "MT",
        },
    }


# ── Tool 2: 항차 분석 ───────────────────────────────────────────────────────────
def get_voyage_analysis(voyage_id: str = "", period: str = "current") -> dict:
    """특정 항차 또는 현재/이전/올해 기간의 운항 KPI 집계"""
    store = get_store()
    vid, period, _condition = resolve_voyage_query(voyage_id, period)

    if period == "ytd" and not vid:
        year = int(CURRENT_DATE[:4])
        ytd  = store.ytd_summary(year)
        return {
            "period":   f"{year} YTD (1/1 ~ 현재)",
            "summary":  ytd,
            "cii":      _annual_cii_result(year, scope="annual"),
        }

    voyage = _pick_voyage(voyage_id=voyage_id, period=period)
    if not voyage:
        return {"error": "항차 데이터 없음"}

    vid   = voyage.get("voyage_id", "")
    stats = store.voyage_stats(vid)
    days  = max(stats.get("days_at_sea", 1), 1)
    dist  = max(stats.get("distance_nm", 1), 1)

    return {
        "voyage_id":       vid,
        "departure_port":  voyage.get("departure_port", "Unknown"),
        "arrival_port":    voyage.get("arrival_port",   "Unknown"),
        "voyage_start":    str(voyage.get("start_time", ""))[:16],
        "voyage_end":      str(voyage.get("end_time",   ""))[:16],
        "days_at_sea":     days,
        "distance_nm":     stats.get("distance_nm", 0),
        "avg_sog_kts":     stats.get("avg_sog",     0),
        "foc_oil_mt":      stats.get("foc_oil_mt",  0),
        "fgc_gas_mt":      stats.get("fgc_gas_mt",  0),
        "foc_per_day_mt":  round((stats.get("foc_oil_mt", 0) + stats.get("fgc_gas_mt", 0)) / days, 2),
        "co2_mt":          stats.get("co2_mt",  0),
        "ch4_mt":          stats.get("ch4_mt",  0),
        "co2e_mt":         stats.get("co2e_mt", 0),
        "cii_kg_per_nm":   stats.get("cii_value", 0),
        "co2_per_nm":      round(stats.get("co2_mt", 0) / dist, 4),
        "distance_method": stats.get("distance_method", ""),
        "distance_note":   stats.get("distance_note", ""),
        "loading_status":  voyage.get("condition") or "미제공",
        "me_rpm_note":     "미측정 (원본 데이터에 RPM 컬럼 없음)",
        "data_source":     "ho_data (sensor)",
        # 해당 항차만의 참고(Indicative) CII·A~E 등급 — 공식 연간 CII 아님
        "cii":             _voyage_cii_result(stats, voyage.get("end_time", ""), scope="voyage"),
    }


# ── Tool 3: CII 등급 계산 ───────────────────────────────────────────────────────
def calculate_cii_rating(year: int | None = None) -> dict:
    """
    연간(또는 진행 중인 연도는 YTD 잠정) IMO CII 등급 계산.
    실제 공식/등급 판정은 agent/cii.py 로 단일화되어 있다 (여기서는 재계산하지 않음).
    year 생략/"올해" 호출 시 데이터 기준 연도(CURRENT_DATE)를 쓴다.
    """
    year = cii_calc._coerce_year(year)
    if year is None:
        year = _current_data_year()

    store = get_store()
    ytd   = store.ytd_summary(year)

    result = _annual_cii_result(year, scope="annual")

    # 하위 호환: 기존에 제공하던 배출량 요약 필드(co2e, 항차 수, 센서 kg/nm)도 유지
    if ytd:
        result["total_co2e_mt"]        = ytd.get("total_co2e_mt")
        result["voyages_included"]     = ytd.get("voyages_count")
        dist = ytd.get("total_distance_nm") or 0
        co2  = ytd.get("total_co2_mt") or 0
        result["sensor_cii_kg_per_nm"] = round(co2 * 1000 / dist, 4) if dist > 0 else None
        result["calculation_basis"]    = "IMO MEPC.354(78) / ClassNK CII 안내자료"

    return result


# ── Tool 4: 배출량 계산 ─────────────────────────────────────────────────────────
def calculate_emissions(voyage_id: str = "", period: str = "current") -> dict:
    """항차 또는 기간별 배출량 (센서 실측 CO2/CH4/CO2e)"""
    store = get_store()
    vid, period, _condition = resolve_voyage_query(voyage_id, period)

    if period == "ytd" and not vid:
        year = int(CURRENT_DATE[:4])
        ytd  = store.ytd_summary(year)
        return {
            "period":         f"{year} YTD",
            "co2_mt":         ytd.get("total_co2_mt",  0),
            "ch4_mt":         ytd.get("total_ch4_mt",  0),
            "co2e_mt":        ytd.get("total_co2e_mt", 0),
            "foc_oil_mt":     ytd.get("total_foc_oil_mt", 0),
            "fgc_gas_mt":     ytd.get("total_fgc_gas_mt", 0),
            "data_source":    "sensor (ME+GE+AB+GCU)",
        }

    voyage = _pick_voyage(voyage_id=voyage_id, period=period)
    if not voyage:
        return {"error": "데이터 없음"}

    vid   = voyage.get("voyage_id", "")
    stats = store.voyage_stats(vid)

    return {
        "voyage_id":   vid,
        "co2_mt":      stats.get("co2_mt",     0),
        "ch4_mt":      stats.get("ch4_mt",     0),
        "co2e_mt":     stats.get("co2e_mt",    0),
        "foc_oil_mt":  stats.get("foc_oil_mt", 0),
        "fgc_gas_mt":  stats.get("fgc_gas_mt", 0),
        "data_source": "sensor (ME+GE+AB+GCU 합산)",
        "note":        "실측 센서 기반 — ME, GE, 보조보일러, GCU 포함한 선박 전체 배출량",
    }


# ── Tool 5: Noon Report 생성 ────────────────────────────────────────────────────
def generate_noon_report(report_date: str = "") -> dict:
    """최신 또는 지정 날짜의 Noon Report Word 파일 생성"""
    store  = get_store()
    noon   = store.get_noon_by_date(report_date) if report_date else store.latest_noon()
    voyage = store.current_voyage()
    vessel = store.get_vessel()

    if not noon:
        return {"error": "Noon Report 데이터 없음"}

    report_year = int(str(noon.get("report_datetime", CURRENT_DATE))[:4])
    cii = _annual_cii_result(report_year, scope="current_voyage")

    path = generate_noon_report_docx(noon, voyage, vessel, cii)
    return {
        "status":      "생성 완료",
        "file_path":   str(path),
        "report_date": str(noon.get("report_datetime", ""))[:10],
        "position":    f"{noon.get('lat', 0):.4f}N, {noon.get('lon', 0):.4f}E",
        "foc_oil_mt":  noon.get("foc_oil_mt", 0),
        "fgc_gas_mt":  noon.get("fgc_gas_mt", 0),
        "co2_mt":      noon.get("co2_mt",     0),
        "data_source": "sensor",
        "cii":         cii,
    }


# ── Tool 6: MRV Voyage Report 생성 ─────────────────────────────────────────────
def generate_mrv_voyage_report(voyage_id: str = "") -> dict:
    """항차 MRV Report Word 파일 생성"""
    store  = get_store()
    voyage = store.get_voyage(voyage_id) if voyage_id else store.previous_voyage()
    vessel = store.get_vessel()

    if not voyage:
        return {"error": "항차 데이터 없음"}

    vid   = voyage.get("voyage_id", "")
    stats = store.voyage_stats(vid)
    cii   = _voyage_cii_result(stats, voyage.get("end_time", ""), scope="voyage")

    # reports.py 가 사용하는 키 형태로 병합
    voyage_dict = {**voyage, **stats,
                   "departure_date": str(voyage.get("start_time", ""))[:10],
                   "arrival_date":   str(voyage.get("end_time",   ""))[:10]}

    path = generate_mrv_voyage_docx(voyage_dict, vessel, cii)
    return {
        "status":      "생성 완료",
        "file_path":   str(path),
        "voyage_id":   vid,
        "route":       f"{voyage.get('departure_port')} → {voyage.get('arrival_port')}",
        "co2_mt":      stats.get("co2_mt", 0),
        "data_source": "ho_data (sensor)",
        "cii":         cii,
    }


# ── Tool 7: MRV Annual Report 생성 ─────────────────────────────────────────────
def generate_mrv_annual_report(year: int | None = None) -> dict:
    """연간 MRV Report Word 파일 생성"""
    year = cii_calc._coerce_year(year)
    if year is None:
        year = _current_data_year()

    store  = get_store()
    annual = store.annual_summary(year)
    vessel = store.get_vessel()

    if not annual:
        return {"error": f"{year}년 데이터 없음"}

    voyages_df = store.annual_voyages(year)
    cii = _annual_cii_result(year, scope="annual")

    path = generate_mrv_annual_docx(annual, voyages_df, vessel, year, cii)
    return {
        "status":       "생성 완료",
        "file_path":    str(path),
        "year":         year,
        "voyages":      annual.get("voyages_count", 0),
        "total_co2_mt": annual.get("total_co2_mt", 0),
        "data_source":  "ho_data (sensor)",
        "cii":          cii,
    }


# ── 항차 경로 지도 렌더링 ───────────────────────────────────────────────────────
def render_voyage_map(voyage_id: str = "") -> str:
    try:
        import folium
        store = get_store()
        track = store.voyage_track(voyage_id) if voyage_id else store.current_track()

        if track.empty:
            return "<p>경로 데이터 없음</p>"

        center_lat = float(track["lat"].mean())
        center_lon = float(track["lon"].mean())
        m = folium.Map(location=[center_lat, center_lon], zoom_start=4)

        coords = list(zip(track["lat"].astype(float), track["lon"].astype(float)))
        folium.PolyLine(coords, color="#1E88E5", weight=2.5, opacity=0.8).add_to(m)

        if coords:
            folium.Marker(coords[0],  popup="출발", icon=folium.Icon(color="green")).add_to(m)
            folium.Marker(coords[-1], popup="최신위치", icon=folium.Icon(color="red")).add_to(m)

        return m._repr_html_()
    except Exception as e:
        return f"<p>지도 오류: {e}</p>"


# ── Tool 스펙 (LLM 함수 스키마) ────────────────────────────────────────────────
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_current_voyage_status",
            "description": "현재 운항 상태 조회. 위경도·SOG·FOC·FGC·CO2·CH4·CO2e·CII 등 전체 KPI 반환.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_voyage_analysis",
            "description": "특정 항차 또는 기간(current/previous/ytd) 운항 데이터 집계 분석.",
            "parameters": {
                "type": "object",
                "properties": {
                    "voyage_id": {"type": "string", "description": "항차 ID (예: H2521_V043). 비워두면 period 사용."},
                    "period":    {"type": "string", "enum": ["current", "previous", "ytd"],
                                  "description": "current=현재항차, previous=이전항차, ytd=올해누계"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_cii_rating",
            "description": (
                "지정 연도의 IMO CII(탄소집약도) Attained/Required 값과 A~E 등급 계산 "
                "(Bulk Carrier, DWT 기준). 올해/연도 미지정이면 데이터 기준 연도"
                f"({CURRENT_DATE[:4]})를 사용. 진행 중인 연도는 YTD 잠정치."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer",
                        "description": (
                            f"계산 연도. 올해={CURRENT_DATE[:4]}. "
                            "생략 시 데이터 기준 연도 사용. 문자열로 와도 정수로 해석됨."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_emissions",
            "description": "항차 또는 기간의 배출량 조회 (CO2·CH4·CO2e). 센서 실측값 기반.",
            "parameters": {
                "type": "object",
                "properties": {
                    "voyage_id": {"type": "string", "description": "항차 ID"},
                    "period":    {"type": "string", "enum": ["current", "previous", "ytd"]},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_noon_report",
            "description": "Noon Report Word 파일 생성. 최신 또는 지정 날짜 기준.",
            "parameters": {
                "type": "object",
                "properties": {
                    "report_date": {"type": "string", "description": "날짜 (YYYY-MM-DD). 비우면 최신."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_mrv_voyage_report",
            "description": "MRV Voyage Report Word 파일 생성. 지정 항차 또는 최근 완료 항차.",
            "parameters": {
                "type": "object",
                "properties": {
                    "voyage_id": {"type": "string", "description": "항차 ID"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_mrv_annual_report",
            "description": "MRV Annual Report Word 파일 생성. 지정 연도의 전체 항차 집계.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer", "description": "연도 (예: 2024, 2025)"},
                },
                "required": [],
            },
        },
    },
]

TOOL_MAP = {
    "get_current_voyage_status":  get_current_voyage_status,
    "get_voyage_analysis":        get_voyage_analysis,
    "calculate_cii_rating":       calculate_cii_rating,
    "calculate_emissions":        calculate_emissions,
    "generate_noon_report":       generate_noon_report,
    "generate_mrv_voyage_report": generate_mrv_voyage_report,
    "generate_mrv_annual_report": generate_mrv_annual_report,
}

# maritime_agent.py 호환 alias
TOOL_SCHEMAS = TOOLS_SPEC


def dispatch_tool(fn_name: str, fn_args: dict) -> str:
    import json
    fn = TOOL_MAP.get(fn_name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {fn_name}"})
    try:
        args = dict(fn_args or {})
        # Coerce period only. Keep original voyage_id string so tools can
        # re-parse Laden/이전 phrases (clearing id here dropped condition).
        if fn_name in {
            "get_voyage_analysis",
            "calculate_emissions",
            "generate_mrv_voyage_report",
        }:
            _vid, period, _cond = resolve_voyage_query(
                str(args.get("voyage_id") or ""),
                str(args.get("period") or "current"),
            )
            args["period"] = period
            if _vid:
                args["voyage_id"] = _vid
        result = fn(**args)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})
