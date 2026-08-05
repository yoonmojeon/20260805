"""
maritime.db → Excel보내기 (엑셀처럼 보기용)
DB 구조는 그대로 두고, 테이블별 시트로 .xlsx 생성
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

_OPS_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _OPS_DIR.parent
sys.path.insert(0, str(_OPS_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from project_paths import DATA_DIR, OPS_DB_PATH

DB_PATH = OPS_DB_PATH
OUT_PATH = DATA_DIR / "maritime_db_export.xlsx"

TABLES = ["vessel", "voyages", "sensor_log"]


def export_excel(out_path: Path = OUT_PATH) -> Path:
    conn = sqlite3.connect(DB_PATH)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name in TABLES:
            df = pd.read_sql_query(f"SELECT * FROM {name}", conn)
            # sensor_log는 너무 크면 최근 5000행 + 전체는 별도 안내
            if name == "sensor_log" and len(df) > 5000:
                df_recent = df.tail(5000)
                df_recent.to_excel(writer, sheet_name="sensor_log(최근5000)", index=False)
                # 요약 시트
                summary = pd.DataFrame([{
                    "전체행수": len(df),
                    "기간시작": df["ds_timeindex"].iloc[0],
                    "기간끝": df["ds_timeindex"].iloc[-1],
                    "비고": "전체는 DB Browser 또는 SQL로 조회",
                }])
                summary.to_excel(writer, sheet_name="sensor_log_요약", index=False)
            else:
                df.to_excel(writer, sheet_name=name[:31], index=False)
    conn.close()
    return out_path


if __name__ == "__main__":
    p = export_excel()
    print(f"생성: {p}")
    print("Excel에서 열어서 테이블별 시트로 확인하세요.")
