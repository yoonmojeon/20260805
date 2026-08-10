"""
IMO CII (Carbon Intensity Indicator) 계산 — 단일 진실 공급원(Single Source of Truth)

이 모듈은 Reference CII, Required CII, Attained CII, A~E 등급 판정을
결정론적 순수 함수로 제공한다. LLM은 이 모듈을 호출하지 않으며,
tools.py를 통해서만 간접적으로 사용된다. briefing.py / reports.py는
이 모듈이 반환한 값을 표시만 하고 재계산하지 않는다.

참고: ClassNK "Carbon Intensity Indicator (CII)" 안내자료, IMO MEPC.354(78).

공식 (Bulk Carrier, DWT < 279,000, Capacity = DWT):
    reference_cii = a * DWT ** (-c)                       [gCO2/(DWT·nm)]
    required_cii  = reference_cii * ((100 - Z) / 100)      [gCO2/(DWT·nm)]
    attained_cii  = total_co2_mt * 1e6 / (dwt * distance_nm)  [gCO2/(DWT·nm)]

주의: Reference 공식의 지수는 음의 c 이다 (a * DWT^(-c)),
      a * DWT^(+c) 로 계산하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from config import CII_PARAMS, VESSEL

# 공식 등급이 유효한 연간(annual) 산정 범위. 이 범위를 벗어나는 연도는
# 임의 보간/외삽/최근접 연도 대체 없이 명시적으로 계산 불가 처리한다.
SUPPORTED_YEARS = tuple(sorted(CII_PARAMS["reduction"].keys()))


@dataclass
class CIIResult:
    status: str                       # "success" | "unavailable"
    scope: str                        # "annual" | "voyage" | "current_voyage" 등
    year: Optional[int] = None
    ship_type: Optional[str] = None
    dwt: Optional[float] = None
    capacity: Optional[float] = None
    co2_mt: Optional[float] = None
    distance_nm: Optional[float] = None
    attained_cii: Optional[float] = None
    reference_cii: Optional[float] = None
    required_cii: Optional[float] = None
    reduction_factor_percent: Optional[float] = None
    rating: Optional[str] = None
    rating_ratio: Optional[float] = None
    d1: Optional[float] = None
    d2: Optional[float] = None
    d3: Optional[float] = None
    d4: Optional[float] = None
    boundary_a_b: Optional[float] = None
    boundary_b_c: Optional[float] = None
    boundary_c_d: Optional[float] = None
    boundary_d_e: Optional[float] = None
    unit: str = "gCO2/(DWT·nm)"
    warning: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _unavailable(scope: str, reason: str, **extra) -> CIIResult:
    return CIIResult(status="unavailable", scope=scope, rating=None, reason=reason, **extra)


def reference_cii(dwt: float) -> float:
    """Reference CII = a * DWT^(-c). 지수는 음수(-c)임에 유의."""
    a = CII_PARAMS["a"]
    c = CII_PARAMS["c"]
    return a * (dwt ** (-c))


def _coerce_year(year) -> Optional[int]:
    """Tool/LLM args often arrive as str/float; normalize to int year."""
    if year is None or year == "":
        return None
    try:
        return int(float(year))
    except (TypeError, ValueError):
        return None


def reduction_factor_percent(year: int) -> Optional[float]:
    """연도별 감축계수 Z(%). 지원하지 않는 연도는 None."""
    year_i = _coerce_year(year)
    if year_i is None:
        return None
    return CII_PARAMS["reduction"].get(year_i)


def required_cii(ref_cii: float, z_percent: float) -> float:
    """Required CII = reference_cii * ((100 - Z) / 100)."""
    return ref_cii * ((100.0 - z_percent) / 100.0)


def attained_cii(co2_mt: float, dwt: float, distance_nm: float) -> float:
    """Attained CII = total_co2_mt * 1e6 / (dwt * distance_nm)."""
    return (co2_mt * 1_000_000.0) / (dwt * distance_nm)


def rating_boundaries(req_cii: float) -> dict:
    d = CII_PARAMS["d_factors"]
    return {
        "d1": d["d1"], "d2": d["d2"], "d3": d["d3"], "d4": d["d4"],
        "boundary_a_b": req_cii * d["d1"],
        "boundary_b_c": req_cii * d["d2"],
        "boundary_c_d": req_cii * d["d3"],
        "boundary_d_e": req_cii * d["d4"],
    }


def rate(attained: float, req_cii: float) -> str:
    """A~E 등급 판정. 경계값과 정확히 같으면 더 좋은(낮은 문자) 등급에 포함."""
    d = CII_PARAMS["d_factors"]
    if attained <= req_cii * d["d1"]:
        return "A"
    if attained <= req_cii * d["d2"]:
        return "B"
    if attained <= req_cii * d["d3"]:
        return "C"
    if attained <= req_cii * d["d4"]:
        return "D"
    return "E"


def compute_cii(
    *,
    scope: str,
    year: Optional[int],
    dwt: float,
    co2_mt: Optional[float],
    distance_nm: Optional[float],
    ship_type: str = None,
) -> CIIResult:
    """
    CII 계산 단일 진입점. tools.py는 오직 이 함수만 호출한다.

    scope: "annual"(연간 공식 등급), "voyage"(항차 참고 CII),
           "current_voyage"(현재 항차 잠정 CII) 등 호출측에서 의미 부여.
    year:  해당 scope에 대응하는 연도. 호출측이 명시적으로 결정해서 전달해야 하며
           이 함수는 "현재 시스템 연도"를 임의로 사용하지 않는다.
    """
    ship_type = ship_type or VESSEL.get("type", "Bulk Carrier")
    year = _coerce_year(year)

    if year is None:
        return _unavailable(scope, "연도가 지정되지 않았습니다.", ship_type=ship_type, dwt=dwt)

    if dwt is None or dwt <= 0:
        return _unavailable(scope, "DWT가 설정되지 않았습니다 (dwt <= 0).",
                             year=year, ship_type=ship_type, dwt=dwt)

    z = reduction_factor_percent(year)
    if z is None:
        return _unavailable(
            scope,
            f"{year}년은 지원되지 않는 연도입니다 (지원 연도: "
            f"{SUPPORTED_YEARS[0]}~{SUPPORTED_YEARS[-1]}). 임의 보간/외삽하지 않습니다.",
            year=year, ship_type=ship_type, dwt=dwt,
        )

    if co2_mt is None:
        return _unavailable(scope, "CO2 값이 없습니다 (NULL).", year=year, ship_type=ship_type, dwt=dwt)

    if distance_nm is None:
        return _unavailable(scope, "운항 거리 데이터가 없습니다.", year=year, ship_type=ship_type, dwt=dwt)

    if distance_nm < 0:
        return _unavailable(scope, "운항 거리가 음수입니다 (distance_nm < 0).",
                             year=year, ship_type=ship_type, dwt=dwt)

    if distance_nm == 0:
        return _unavailable(scope, "운항 거리가 0nm이라 CII를 계산할 수 없습니다 (distance_nm == 0).",
                             year=year, ship_type=ship_type, dwt=dwt)

    if co2_mt < 0:
        return _unavailable(scope, "CO2 값이 음수입니다.", year=year, ship_type=ship_type, dwt=dwt)

    ref = reference_cii(dwt)
    req = required_cii(ref, z)
    att = attained_cii(co2_mt, dwt, distance_nm)
    boundaries = rating_boundaries(req)
    rating = rate(att, req)
    ratio = att / req if req else None

    warning = None
    if scope != "annual":
        warning = (
            "부분 기간(항차/현재 항차) 데이터 기반 참고값이며, "
            "IMO 공식 연간 CII 등급이 아닙니다."
        )

    return CIIResult(
        status="success",
        scope=scope,
        year=year,
        ship_type=ship_type,
        dwt=dwt,
        capacity=dwt,
        co2_mt=co2_mt,
        distance_nm=distance_nm,
        attained_cii=att,
        reference_cii=ref,
        required_cii=req,
        reduction_factor_percent=z,
        rating=rating,
        rating_ratio=ratio,
        **boundaries,
        unit="gCO2/(DWT·nm)",
        warning=warning,
    )
