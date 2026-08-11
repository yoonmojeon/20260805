import os
import sys
from pathlib import Path

# Unified project root: MaritimeOpsRAG/ (ops/ is one level below)
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DATA_DIR, REPORTS_DIR as _SHARED_REPORTS_DIR

REPORTS_DIR = _SHARED_REPORTS_DIR
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Ollama (로컬 LLM) — quality-30 winner gemma4:12b (override with MODEL_NAME)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = "ollama"
MODEL_NAME = os.getenv("MODEL_NAME", "gemma4:12b")

# 선박 기본 정보 (ho_data 기반 실제 선박: h2521)
# CII 산정 대상: DWT 279,000 미만 Bulk Carrier, DWT=81,200 (CII Capacity = DWT)
VESSEL = {
    "name": "H2521",
    "imo": "H2521",
    "mmsi": "Unknown",
    "type": "Bulk Carrier",
    "flag": "Unknown",
    "gt": 0,
    "dwt": 81200,
    "loa_m": 0.0,
    "beam_m": 0.0,
    "draft_design_m": 0.0,
    "me_model": "Unknown",
    "me_mcr_kw": 10_703,   # ho_data 실측 최대값 기반
    "me_mcr_rpm": 0,
    "service_speed_kts": 18.0,
    "service_rpm": 0,
    "foc_service_mt_day": 0.0,
    "displacement_laden_mt": 0,
    "displacement_ballast_mt": 0,
    "cargo_capacity_teu": 0,
}

# CII 계산 파라미터 (IMO MEPC.354(78) / ClassNK 안내자료 기준)
# 대상: DWT 279,000 미만 Bulk Carrier
#   reference_cii = a * DWT ** (-c)   ← 지수는 음의 c (a * DWT^(+c) 아님)
#   required_cii  = reference_cii * ((100 - Z) / 100)
# config.py와 DB(vessel.dwt) 중 어느 값을 쓸지: DB에 유효한(>0) DWT가 있으면
# DB 값을 우선 사용하고, DB 값이 없거나 0/NULL이면 이 VESSEL["dwt"]로 폴백한다
# (agent/tools.py _get_vessel_dwt() 참고). CII 계산에는 절대 0이 전달되지 않도록
# scripts/load_hodata.py의 sync_vessel_dwt()가 DB 적재 시 이 값을 vessel 테이블에 동기화한다.
CII_PARAMS = {
    "a": 4745,
    "c": 0.622,
    "d_factors": {"d1": 0.86, "d2": 0.94, "d3": 1.06, "d4": 1.18},  # Bulk Carrier dd vector
    # 연도별 감축계수 Z(%). 지원하지 않는 연도는 임의 보간/외삽하지 않고 계산 불가 처리한다.
    "reduction": {
        2023: 5.000,
        2024: 7.000,
        2025: 9.000,
        2026: 11.000,
        2027: 13.625,
        2028: 16.250,
        2029: 18.875,
        2030: 21.500,
    },
}

# 연료별 CO2 배출계수 (t CO2 / t fuel)
EMISSION_FACTORS = {
    "HFO":  3.114,
    "VLSFO": 3.114,
    "MGO":  3.206,
    "LSMGO": 3.206,
    "LNG":  2.750,
    "CH4_slip_factor": 0.036,   # LNG slip (GWP 21)
}

# 현재 기준 날짜 및 항차 (ho_data / sensor_log 마지막 데이터 기준)
CURRENT_DATE = "2026-06-28"
CURRENT_VOYAGE_ID = "H2521_V21_Ballast"
