"""
ho_data Excel → SQLite sensor_log 적재
- 원시 1시간 단위 데이터 저장
- ship_info Schedule 기준 항차 구간 → voyages 테이블
실행: python scripts/load_hodata.py
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

_OPS_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _OPS_DIR.parent
sys.path.insert(0, str(_OPS_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.db_schema import DB_PATH, get_conn
from config import VESSEL
from project_paths import HO_DATA_DIR
VESSEL_IMO  = "H2521"
VESSEL_NAME = "H2521"

# 원본 컬럼 → DB 컬럼 매핑
COL_MAP = {
    "ds_timeindex": "ds_timeindex",
    "lat":          "lat",
    "lon":          "lon",
    "sog":          "sog",
    "me1_power":    "me1_power",
    "me2_power":    "me2_power",
    "me_oil_flow":  "me_oil_flow",
    "me_gas_flow":  "me_gas_flow",
    "me_co2":       "me_co2",
    "me_ch4":       "me_ch4",
    "me_co2e":      "me_co2e",
    "ge_oil_flow":  "ge_oil_flow",
    "ge_gas_flow":  "ge_gas_flow",
    "ge_co2":       "ge_co2",
    "ge_ch4":       "ge_ch4",
    "ge_co2e":      "ge_co2e",
    "ab_oil_flow":  "ab_oil_flow",
    "ab_gas_flow":  "ab_gas_flow",
    "gcu_oil_flow": "gcu_oil_flow",
    "gcu_gas_flow": "gcu_gas_flow",
    "oil_flow":     "oil_flow",
    "gas_flow":     "gas_flow",
    "co2":          "co2",
    "ch4":          "ch4",
    "co2e":         "co2e",
    "cii":          "cii",
}

DB_COLS = list(COL_MAP.values())  # sensor_log 컬럼 순서


def load_raw(xlsx_path: Path = None) -> pd.DataFrame:
    if xlsx_path is None:
        files = [f for f in HO_DATA_DIR.glob("*.xlsx")
                 if "ship_info" not in f.name.lower()]
        xlsx_path = max(files, key=lambda x: x.stat().st_size)
    print(f"읽는 중: {xlsx_path.name}")
    df = pd.read_excel(xlsx_path)
    df["ds_timeindex"] = pd.to_datetime(df["ds_timeindex"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.sort_values("ds_timeindex").reset_index(drop=True)
    df = df.fillna(0)
    print(f"  → {len(df):,}행  {df['ds_timeindex'].iloc[0]} ~ {df['ds_timeindex'].iloc[-1]}")
    return df


def _find_ship_info() -> Path | None:
    files = [f for f in HO_DATA_DIR.glob("*.xlsx")
             if "ship_info" in f.name.lower() and not f.name.startswith("~")]
    return files[0] if files else None


def load_voyage_schedule(ship_info_path: Path = None) -> pd.DataFrame:
    """동국대_ship_info_r1.xlsx Schedule 시트 — 항차번호·Ballast/Laden·기간"""
    path = ship_info_path or _find_ship_info()
    if not path or not path.exists():
        print("  ⚠ ship_info Schedule 없음 — SOG 기반 항차 감지로 폴백")
        return pd.DataFrame()

    df = pd.read_excel(path, sheet_name="Schedule")
    df["Start Time"] = pd.to_datetime(df["Start Time"])
    df["End Time"]   = pd.to_datetime(df["End Time"])
    print(f"  → Schedule {len(df)}구간 ({path.name})")
    return df


def build_voyages_from_schedule(schedule: pd.DataFrame, imo: str,
                                sensor_end: str = None) -> pd.DataFrame:
    """Schedule 날짜 기준 항차 ID: H2521_V21_Ballast 형식"""
    if schedule.empty:
        return pd.DataFrame()

    sensor_end_ts = pd.to_datetime(sensor_end) if sensor_end else None
    voyages = []

    for _, row in schedule.iterrows():
        vno  = int(row["Voyage No"])
        cond = str(row["Condition"]).strip()
        start = row["Start Time"]
        end   = row["End Time"]

        if pd.isna(start):
            continue
        if pd.isna(end):
            if sensor_end_ts is not None and sensor_end_ts >= start:
                end = sensor_end_ts
            else:
                continue

        voyages.append({
            "voyage_id":      f"{imo}_V{vno:02d}_{cond}",
            "imo":            imo,
            "voyage_num":     vno,
            "condition":      cond,
            "start_time":     start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time":       end.strftime("%Y-%m-%d %H:%M:%S"),
            "departure_port": "Unknown",
            "arrival_port":   "Unknown",
        })

    result = pd.DataFrame(voyages)
    print(f"  → Schedule 기반 항차 {len(result)}개 (Voyage No 1~{schedule['Voyage No'].max()})")
    return result


def _covered_times(schedule_voyages: pd.DataFrame) -> list[tuple]:
    if schedule_voyages.empty:
        return []
    return [
        (pd.to_datetime(r["start_time"]), pd.to_datetime(r["end_time"]))
        for _, r in schedule_voyages.iterrows()
    ]


def detect_voyages_sog(df: pd.DataFrame, imo: str = VESSEL_IMO,
                       skip_ranges: list[tuple] = None) -> pd.DataFrame:
    """Schedule 미커버 구간만 SOG 기반 보조 감지"""
    sailing = (df["sog"] > 0.5) & (df["lat"] != 0)
    if skip_ranges:
        ts = pd.to_datetime(df["ds_timeindex"])
        covered = pd.Series(False, index=df.index)
        for s, e in skip_ranges:
            covered |= (ts >= s) & (ts <= e)
        sailing = sailing & ~covered

    seg_start = sailing & ~sailing.shift(1, fill_value=False)
    voyage_num = seg_start.cumsum().where(sailing, 0)

    voyages = []
    ext = 1
    for _, grp in df[sailing].groupby(voyage_num[sailing]):
        voyages.append({
            "voyage_id":      f"{imo}_EXT_{ext:03d}",
            "imo":            imo,
            "voyage_num":     9000 + ext,
            "condition":      "",
            "start_time":     grp["ds_timeindex"].iloc[0],
            "end_time":       grp["ds_timeindex"].iloc[-1],
            "departure_port": "Unknown",
            "arrival_port":   "Unknown",
        })
        ext += 1
    return pd.DataFrame(voyages) if voyages else pd.DataFrame()


def detect_voyages(df: pd.DataFrame, imo: str = VESSEL_IMO) -> pd.DataFrame:
    """ship_info Schedule 우선, 미정의 구간만 SOG 폴백"""
    sensor_end = df["ds_timeindex"].iloc[-1] if len(df) else None
    schedule   = load_voyage_schedule()
    primary    = build_voyages_from_schedule(schedule, imo, sensor_end=sensor_end)

    if primary.empty:
        result = detect_voyages_sog(df, imo)
        print(f"  → SOG 폴백 항차 {len(result)}개")
        return result

    covered = _covered_times(primary)
    fallback = detect_voyages_sog(df, imo, skip_ranges=covered)
    if not fallback.empty:
        print(f"  → Schedule 미커버 SOG 폴백 {len(fallback)}개")

    result = pd.concat([primary, fallback], ignore_index=True)
    result = result.sort_values("start_time").reset_index(drop=True)
    print(f"  → 총 항차 {len(result)}개")
    return result


def insert_sensor_log(conn: "sqlite3.Connection", df: pd.DataFrame,
                      latest_ts: str = None) -> int:
    available = {k: v for k, v in COL_MAP.items() if k in df.columns}
    sub = df[list(available.keys())].rename(columns=available).copy()

    if latest_ts:
        sub = sub[sub["ds_timeindex"] > latest_ts]
    if sub.empty:
        return 0

    # 없는 컬럼은 0으로 채움
    for col in DB_COLS:
        if col not in sub.columns:
            sub[col] = 0.0

    sub = sub[DB_COLS]
    placeholders = ",".join(["?"] * len(DB_COLS))
    sql = f"INSERT OR IGNORE INTO sensor_log ({','.join(DB_COLS)}) VALUES ({placeholders})"
    conn.executemany(sql, sub.itertuples(index=False, name=None))
    conn.commit()
    print(f"  → sensor_log +{len(sub)}행 삽입")
    return len(sub)


def insert_voyages(conn: "sqlite3.Connection", voyages_df: pd.DataFrame) -> int:
    if voyages_df.empty:
        return 0
    for _, row in voyages_df.iterrows():
        conn.execute("""
            INSERT OR REPLACE INTO voyages
            (voyage_id, imo, voyage_num, condition, start_time, end_time,
             departure_port, arrival_port)
            VALUES (?,?,?,?,?,?,?,?)
        """, (row["voyage_id"], row["imo"], row["voyage_num"],
              row.get("condition", ""),
              row["start_time"], row["end_time"],
              row["departure_port"], row["arrival_port"]))
    conn.commit()
    print(f"  → voyages {len(voyages_df)}건")
    return len(voyages_df)


def sync_vessel_dwt(conn: "sqlite3.Connection", dwt: float = None, ship_type: str = None) -> None:
    """
    vessel.dwt를 config.VESSEL의 값으로 동기화 (idempotent).

    ho_data Excel 원본에는 DWT/GT 컬럼이 없으므로, CII 계산에 필요한 DWT는
    항상 config.py의 설정값을 신뢰 소스(source of truth)로 사용한다.
    이 함수는 몇 번을 실행해도 결과가 같으며(idempotent), DWT를 절대
    0/NULL로 되돌리지 않는다 — dwt가 유효(>0)할 때만 반영한다.
    """
    dwt = VESSEL["dwt"] if dwt is None else dwt
    ship_type = VESSEL["type"] if ship_type is None else ship_type

    if not dwt or dwt <= 0:
        print(f"  ⚠ config.VESSEL['dwt']가 유효하지 않아({dwt}) DWT 동기화를 건너뜁니다.")
        return

    conn.execute(
        "INSERT OR IGNORE INTO vessel (imo, name, type) VALUES (?,?,?)",
        (VESSEL_IMO, VESSEL_NAME, ship_type),
    )
    conn.execute(
        "UPDATE vessel SET dwt = ?, type = ?, updated_at = datetime('now') WHERE imo = ?",
        (dwt, ship_type, VESSEL_IMO),
    )
    conn.commit()
    print(f"  → vessel.dwt = {dwt:,.0f} ({ship_type}) 동기화 완료 (imo={VESSEL_IMO})")


def rebuild(xlsx_path: Path = None, conn=None):
    """전체 재구축"""
    own = conn is None
    if own:
        conn = get_conn()

    print("기존 데이터 삭제...")
    conn.execute("DELETE FROM sensor_log")
    conn.execute("DELETE FROM voyages")
    conn.execute("DELETE FROM vessel")
    conn.execute(
        "INSERT OR IGNORE INTO vessel (imo, name, type) VALUES (?,?,?)",
        (VESSEL_IMO, VESSEL_NAME, VESSEL["type"])
    )
    conn.commit()
    # ho_data Excel에는 DWT/GT가 없으므로 config.py 값을 신뢰 소스로 동기화.
    sync_vessel_dwt(conn)

    df      = load_raw(xlsx_path)
    voyages = detect_voyages(df)
    n_s = insert_sensor_log(conn, df)
    n_v = insert_voyages(conn, voyages)

    _update_config(conn)
    if own:
        conn.close()

    print(f"\n=== 완료 ===")
    print(f"  sensor_log: {n_s:,}행")
    print(f"  voyages:    {n_v}건")


def _update_config(conn):
    try:
        import config
        max_ts = conn.execute(
            "SELECT MAX(ds_timeindex) FROM sensor_log"
        ).fetchone()[0]
        if not max_ts:
            return
        row = conn.execute("""
            SELECT voyage_id FROM voyages
            WHERE ? BETWEEN start_time AND end_time
            ORDER BY voyage_num DESC, start_time DESC
            LIMIT 1
        """, (max_ts,)).fetchone()
        if row and row[0]:
            config.CURRENT_DATE      = str(max_ts)[:10]
            config.CURRENT_VOYAGE_ID = row[0]
    except Exception:
        pass


if __name__ == "__main__":
    if "--sync-dwt" in sys.argv:
        # 기존 DB를 재구축하지 않고 vessel.dwt만 config.py 기준으로 동기화.
        print("=== vessel.dwt 동기화 (config.py 기준) ===\n")
        _conn = get_conn()
        sync_vessel_dwt(_conn)
        _conn.close()
    else:
        print("=== ho_data → SQLite 원시 데이터 적재 ===\n")
        rebuild()
