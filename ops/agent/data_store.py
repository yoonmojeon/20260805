"""
SQLite 기반 데이터 스토어 — 원시 센서 데이터 실시간 집계
선원 에이전트 예상 질문:
  - "지금 속력 얼마야?"      → latest_sensor()
  - "이번 항차 FOC 얼마야?"  → voyage_stats()
  - "YTD CII 등급은?"        → ytd_summary()
"""
import math
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from .db_schema import DB_PATH, get_conn

_store_instance = None


def get_store() -> "DataStore":
    global _store_instance
    if _store_instance is None:
        _store_instance = DataStore()
    return _store_instance


def reset_store():
    global _store_instance
    if _store_instance:
        try:
            _store_instance.conn.close()
        except Exception:
            pass
    _store_instance = None


class DataStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.conn = get_conn(db_path)

    def q(self, sql: str, params=()) -> pd.DataFrame:
        try:
            return pd.read_sql_query(sql, self.conn, params=params)
        except Exception:
            return pd.DataFrame()

    def qone(self, sql: str, params=()) -> Optional[dict]:
        row = self.conn.execute(sql, params).fetchone()
        if row is None:
            return None
        keys = [d[0] for d in self.conn.execute(sql, params).description
                ] if hasattr(self.conn.execute(sql, params), 'description') else []
        # 재조회 없이 sqlite3.Row 활용
        row2 = self.conn.execute(sql, params).fetchone()
        if row2 is None:
            return None
        return dict(row2)

    # ── 선박 정보 ──────────────────────────────────────────────────────────────
    def get_vessel(self) -> dict:
        row = self.conn.execute("SELECT * FROM vessel LIMIT 1").fetchone()
        return dict(row) if row else {}

    # ── 항차 조회 ──────────────────────────────────────────────────────────────
    def current_voyage(self) -> dict:
        row = self.conn.execute("""
            SELECT * FROM voyages ORDER BY end_time DESC LIMIT 1
        """).fetchone()
        return dict(row) if row else {}

    def previous_voyage(self) -> dict:
        row = self.conn.execute("""
            SELECT * FROM voyages ORDER BY end_time DESC LIMIT 1 OFFSET 1
        """).fetchone()
        return dict(row) if row else {}

    def get_voyage(self, voyage_id: str) -> dict:
        row = self.conn.execute(
            "SELECT * FROM voyages WHERE voyage_id = ?", (voyage_id,)
        ).fetchone()
        return dict(row) if row else {}

    def get_all_voyages(self) -> pd.DataFrame:
        return self.q("SELECT * FROM voyages ORDER BY start_time")

    # ── 현재 최신 센서 레코드 ──────────────────────────────────────────────────
    def latest_sensor(self, voyage_id: str = None) -> dict:
        """현재 항차의 가장 최신 레코드 (위경도, SOG, 출력, 순간 연료/배출)"""
        if voyage_id is None:
            voy = self.current_voyage()
            voyage_id = voy.get("voyage_id", "") if voy else ""

        if not voyage_id:
            return {}

        voy = self.get_voyage(voyage_id)
        row = self.conn.execute("""
            SELECT * FROM sensor_log
            WHERE ds_timeindex BETWEEN ? AND ?
              AND (lat != 0 OR sog > 0)
            ORDER BY ds_timeindex DESC LIMIT 1
        """, (voy.get("start_time", ""), voy.get("end_time", ""))).fetchone()
        if row:
            return dict(row)
        # fallback: 항차 구간 내 가장 최신 행
        row = self.conn.execute("""
            SELECT * FROM sensor_log
            WHERE ds_timeindex BETWEEN ? AND ?
            ORDER BY ds_timeindex DESC LIMIT 1
        """, (voy.get("start_time", ""), voy.get("end_time", ""))).fetchone()
        return dict(row) if row else {}

    # ── 항차 KPI 집계 ───────────────────────────────────────────────────────────
    def voyage_stats(self, voyage_id: str) -> dict:
        """원시 sensor_log를 집계 → 항차 전체 KPI"""
        voy = self.get_voyage(voyage_id)
        if not voy:
            return {}
        start = voy["start_time"]
        end   = voy["end_time"]

        # 집계
        row = self.conn.execute("""
            SELECT
                COUNT(*)                        AS hours,
                SUM(oil_flow) / 1000.0          AS foc_oil_mt,
                SUM(gas_flow) / 1000.0          AS fgc_gas_mt,
                SUM(co2)      / 1000.0          AS co2_mt,
                SUM(ch4)      / 1000.0          AS ch4_mt,
                SUM(co2e)     / 1000.0          AS co2e_mt,
                AVG(CASE WHEN sog > 0.5 THEN sog END) AS avg_sog
            FROM sensor_log
            WHERE ds_timeindex BETWEEN ? AND ?
        """, (start, end)).fetchone()
        agg = dict(row) if row else {}

        # 최신 레코드 (현재 위치)
        latest = self.latest_sensor(voyage_id)

        dist_info = _compute_distance_range(self.conn, start, end)
        dist_nm   = dist_info["distance_nm"]

        co2_mt  = float(agg.get("co2_mt",  0) or 0)
        cii_val = round(co2_mt * 1000 / dist_nm, 4) if dist_nm > 0 else float(latest.get("cii", 0) or 0)
        hours   = float(agg.get("hours",   0) or 0)

        return {
            "voyage_id":      voyage_id,
            "departure_port": voy.get("departure_port", "Unknown"),
            "arrival_port":   voy.get("arrival_port",   "Unknown"),
            "condition":      voy.get("condition", ""),
            "start_time":     start[:16],
            "end_time":       end[:16],
            "hours":          int(hours),
            "days_at_sea":    round(hours / 24, 1),
            "distance_nm":    dist_nm,
            "distance_method": dist_info.get("distance_method", ""),
            "distance_note":  dist_info.get("distance_note", ""),
            "coord_distance_nm": dist_info.get("coord_distance_nm"),
            "avg_sog":        round(float(agg.get("avg_sog", 0) or 0), 1),
            "foc_oil_mt":     round(float(agg.get("foc_oil_mt", 0) or 0), 2),
            "fgc_gas_mt":     round(float(agg.get("fgc_gas_mt", 0) or 0), 2),
            "co2_mt":         round(co2_mt, 2),
            "ch4_mt":         round(float(agg.get("ch4_mt",  0) or 0), 4),
            "co2e_mt":        round(float(agg.get("co2e_mt", 0) or 0), 2),
            "cii_value":      cii_val,
            # 최신 순간값
            "last_lat":       round(float(latest.get("lat", 0)), 4),
            "last_lon":       round(float(latest.get("lon", 0)), 4),
            "last_sog":       round(float(latest.get("sog", 0)), 1),
            "last_me_power":  round(float(latest.get("me1_power", 0) or 0) +
                                    float(latest.get("me2_power", 0) or 0), 0),
            "last_oil_rate":  round(float(latest.get("oil_flow", 0)), 2),
            "last_gas_rate":  round(float(latest.get("gas_flow", 0)), 2),
            "last_co2_rate":  round(float(latest.get("co2", 0)), 2),
            "last_ts":        str(latest.get("ds_timeindex", ""))[:16],
        }

    # ── 일별 집계 (Noon Report용) ───────────────────────────────────────────────
    def noon_on_date(self, date_str: str) -> dict:
        df = self.q("""
            SELECT s.*, v.voyage_id
            FROM sensor_log s
            LEFT JOIN voyages v
              ON s.ds_timeindex BETWEEN v.start_time AND v.end_time
            WHERE substr(s.ds_timeindex, 1, 10) = ?
            ORDER BY s.ds_timeindex
        """, (date_str,))

        if df.empty:
            return {}

        vids = df["voyage_id"].dropna() if "voyage_id" in df.columns else pd.Series(dtype=str)
        voyage_id = str(vids.iloc[-1]) if len(vids) > 0 else ""

        last  = df.iloc[-1]
        first = df.iloc[0]
        dlat  = float(last["lat"]) - float(first["lat"])
        dlon  = float(last["lon"]) - float(first["lon"])
        cog   = round(math.degrees(math.atan2(dlon, dlat)) % 360, 1) if (dlat or dlon) else 0.0

        sailing  = df[(df["sog"] > 0.5) & (df["lat"] != 0)]
        dist_info = _compute_distance_range(
            self.conn,
            f"{date_str} 00:00:00",
            f"{date_str} 23:59:59",
        )
        dist_nm  = dist_info["distance_nm"]
        if dist_nm <= 0:
            dist_nm = _haversine_total(sailing[["lat", "lon"]])
        me_power = float(sailing[["me1_power", "me2_power"]].sum(axis=1).mean()) if len(sailing) > 0 else 0

        foc_oil = float(df["oil_flow"].sum()) / 1000.0
        fgc_gas = float(df["gas_flow"].sum()) / 1000.0
        co2_mt  = float(df["co2"].sum())      / 1000.0
        ch4_mt  = float(df["ch4"].sum())      / 1000.0
        co2e_mt = float(df["co2e"].sum())     / 1000.0
        cii_val = round(co2_mt * 1000 / dist_nm, 4) if dist_nm > 0 else float(df["cii"].mean())

        return {
            "report_datetime": f"{date_str} 12:00:00",
            "voyage_id":       voyage_id,
            "lat":             round(float(last["lat"]), 4),
            "lon":             round(float(last["lon"]), 4),
            "cog_deg":         cog,
            "sog_kts":         round(float(last["sog"]), 1),
            "sailed_nm":       round(dist_nm, 1),
            "me_rpm":          None,
            "me_rpm_note":     "미측정 (원본 데이터에 RPM 컬럼 없음)",
            "displacement_mt": None,
            "me_power_kw":     round(me_power, 0),
            "foc_oil_mt":      round(foc_oil, 3),
            "fgc_gas_mt":      round(fgc_gas, 3),
            "co2_mt":          round(co2_mt, 2),
            "ch4_mt":          round(ch4_mt, 4),
            "co2e_mt":         round(co2e_mt, 2),
            "cii_value":       round(cii_val, 4),
            "wind_speed_kts":  0.0,
            "wind_dir_deg":    0.0,
            "wave_height_m":   0.0,
            "weather_source":  "N/A",
            "fuel_source":     "sensor",
        }

    def latest_noon(self) -> dict:
        row = self.conn.execute(
            "SELECT substr(MAX(ds_timeindex),1,10) AS d FROM sensor_log WHERE lat != 0"
        ).fetchone()
        return self.noon_on_date(row["d"]) if row and row["d"] else {}

    def get_noon_by_date(self, date_str: str) -> dict:
        return self.noon_on_date(date_str)

    # ── 항차 경로 (지도용) ──────────────────────────────────────────────────────
    def voyage_track(self, voyage_id: str) -> pd.DataFrame:
        voy = self.get_voyage(voyage_id)
        if not voy:
            return pd.DataFrame()
        return self.q("""
            SELECT ds_timeindex, lat, lon, sog, co2, oil_flow, gas_flow
            FROM sensor_log
            WHERE ds_timeindex BETWEEN ? AND ?
              AND lat != 0 AND lon != 0
            ORDER BY ds_timeindex
        """, (voy["start_time"], voy["end_time"]))

    def current_track(self) -> pd.DataFrame:
        voy = self.current_voyage()
        return self.voyage_track(voy.get("voyage_id", "")) if voy else pd.DataFrame()

    # ── 연간/YTD 집계 ───────────────────────────────────────────────────────────
    def ytd_summary(self, year: int) -> dict:
        row = self.conn.execute("""
            SELECT
                COUNT(DISTINCT v.voyage_id)     AS voyages_count,
                SUM(s.oil_flow) / 1000.0        AS total_foc_oil_mt,
                SUM(s.gas_flow) / 1000.0        AS total_fgc_gas_mt,
                SUM(s.co2)      / 1000.0        AS total_co2_mt,
                SUM(s.ch4)      / 1000.0        AS total_ch4_mt,
                SUM(s.co2e)     / 1000.0        AS total_co2e_mt,
                COUNT(*)                        AS total_hours
            FROM sensor_log s
            JOIN voyages v ON s.ds_timeindex BETWEEN v.start_time AND v.end_time
            WHERE substr(s.ds_timeindex, 1, 4) = ?
              AND s.sog > 0.5
        """, (str(year),)).fetchone()

        if not row or not row["voyages_count"]:
            return {}

        agg    = dict(row)
        dist_info = _compute_distance_ytd(self.conn, year)
        dist_nm   = dist_info["distance_nm"]
        hours    = float(agg.get("total_hours", 0) or 0)
        foc_oil  = float(agg.get("total_foc_oil_mt", 0) or 0)
        fgc_gas  = float(agg.get("total_fgc_gas_mt", 0) or 0)

        return {
            "year":               year,
            "voyages_count":      int(agg.get("voyages_count", 0)),
            "total_distance_nm":  round(dist_nm, 1),
            "total_days_at_sea":  round(hours / 24, 1),
            "total_foc_oil_mt":   round(foc_oil, 2),
            "total_fgc_gas_mt":   round(fgc_gas, 2),
            "total_foc_mt":       round(foc_oil + fgc_gas, 2),
            "total_co2_mt":       round(float(agg.get("total_co2_mt",  0) or 0), 2),
            "total_ch4_mt":       round(float(agg.get("total_ch4_mt",  0) or 0), 4),
            "total_co2e_mt":      round(float(agg.get("total_co2e_mt", 0) or 0), 2),
            "total_cargo_mt":     0.0,
        }

    def annual_summary(self, year: int) -> dict:
        return self.ytd_summary(year)

    def annual_voyages(self, year: int) -> pd.DataFrame:
        voyages = self.q("""
            SELECT * FROM voyages
            WHERE substr(start_time,1,4) = ? OR substr(end_time,1,4) = ?
            ORDER BY start_time
        """, (str(year), str(year)))
        if voyages.empty:
            return voyages

        rows = []
        for _, vrow in voyages.iterrows():
            s = self.voyage_stats(vrow["voyage_id"])
            rows.append({
                "voyage_id":      vrow["voyage_id"],
                "departure_port": vrow["departure_port"],
                "arrival_port":   vrow["arrival_port"],
                "departure_date": str(vrow["start_time"])[:10],
                "arrival_date":   str(vrow["end_time"])[:10],
                "distance_nm":    s.get("distance_nm", 0),
                "foc_oil_mt":     s.get("foc_oil_mt",  0),
                "fgc_gas_mt":     s.get("fgc_gas_mt",  0),
                "co2_mt":         s.get("co2_mt",       0),
                "co2e_mt":        s.get("co2e_mt",      0),
            })
        return pd.DataFrame(rows)

    # ── 모니터 통계 ─────────────────────────────────────────────────────────────
    def get_realtime_stats(self) -> dict:
        row = self.conn.execute("""
            SELECT COUNT(*) AS total_rows,
                   MIN(ds_timeindex) AS first_ts,
                   MAX(ds_timeindex) AS last_ts,
                   COUNT(DISTINCT substr(ds_timeindex,1,10)) AS total_days
            FROM sensor_log
        """).fetchone()
        v_row = self.conn.execute("SELECT COUNT(*) AS cnt FROM voyages").fetchone()
        return {
            "total_rows":    int(row["total_rows"])    if row else 0,
            "first_ts":      str(row["first_ts"])[:16] if row else "",
            "last_ts":       str(row["last_ts"])[:16]  if row else "",
            "total_days":    int(row["total_days"])    if row else 0,
            "total_voyages": int(v_row["cnt"])         if v_row else 0,
        }

    def recent_sensor_df(self, n: int = 20) -> pd.DataFrame:
        return self.q("""
            SELECT s.ds_timeindex, v.voyage_id,
                   s.lat, s.lon, s.sog,
                   s.oil_flow, s.gas_flow, s.co2
            FROM sensor_log s
            LEFT JOIN voyages v
              ON s.ds_timeindex BETWEEN v.start_time AND v.end_time
            ORDER BY s.ds_timeindex DESC LIMIT ?
        """, (n,))


def _haversine_total(pos_df: pd.DataFrame) -> float:
    if pos_df is None or len(pos_df) < 2:
        return 0.0
    lats = pos_df["lat"].values
    lons = pos_df["lon"].values
    total = 0.0
    for i in range(1, len(lats)):
        dlat = math.radians(float(lats[i]) - float(lats[i-1]))
        dlon = math.radians(float(lons[i]) - float(lons[i-1]))
        a = (math.sin(dlat/2)**2 +
             math.cos(math.radians(float(lats[i-1]))) *
             math.cos(math.radians(float(lats[i]))) *
             math.sin(dlon/2)**2)
        total += 2 * math.asin(math.sqrt(max(a, 0))) * 3440.065
    return total


def _compute_distance_range(conn, start: str, end: str) -> dict:
    """
    항주 거리 산정.
    lon=0 결측이 많으면 Haversine이 과소 추정 → SOG 적분(1h×kt≈nm) 우선.
    """
    row = conn.execute("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN lat != 0 AND lon != 0 THEN 1 ELSE 0 END) AS valid_coord,
            SUM(CASE WHEN sog > 0.5 THEN sog ELSE 0 END) AS sog_nm
        FROM sensor_log
        WHERE ds_timeindex BETWEEN ? AND ?
    """, (start, end)).fetchone()

    total     = int(row["total"] or 0)
    valid     = int(row["valid_coord"] or 0)
    sog_nm    = float(row["sog_nm"] or 0)
    valid_pct = (valid / total * 100) if total > 0 else 0

    pos_df = pd.read_sql_query("""
        SELECT lat, lon FROM sensor_log
        WHERE ds_timeindex BETWEEN ? AND ?
          AND lat != 0 AND lon != 0
        ORDER BY ds_timeindex
    """, conn, params=(start, end))
    coord_nm = _haversine_total(pos_df)

    # 좌표 유효율 낮거나, 좌표 거리가 SOG 추정의 70% 미만 → SOG 적분
    use_sog = valid_pct < 50 or coord_nm <= 0 or (sog_nm > 0 and coord_nm < sog_nm * 0.7)

    if use_sog and sog_nm > 0:
        return {
            "distance_nm":       round(sog_nm, 1),
            "distance_method":   "sog_integration",
            "distance_reliable": True,
            "distance_note": (
                f"경도 lon=0 결측 {100-valid_pct:.0f}%, "
                f"좌표 기반 {coord_nm:.0f}nm는 과소 추정되어 미사용"
                if valid_pct < 50 else "1시간×kt≈nm"
            ),
            "coord_distance_nm": round(coord_nm, 1) if coord_nm > 0 else None,
        }

    return {
        "distance_nm":       round(coord_nm, 1),
        "distance_method":   "coordinates",
        "distance_reliable": True,
        "distance_note":       "위·경도 Haversine 합산",
        "coord_distance_nm": round(coord_nm, 1),
    }


def _compute_distance_ytd(conn, year: int) -> dict:
    """연간(YTD) 거리 — voyage 구간 내 센서만 대상"""
    row = conn.execute("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN s.lat != 0 AND s.lon != 0 THEN 1 ELSE 0 END) AS valid_coord,
            SUM(CASE WHEN s.sog > 0.5 THEN s.sog ELSE 0 END) AS sog_nm
        FROM sensor_log s
        JOIN voyages v ON s.ds_timeindex BETWEEN v.start_time AND v.end_time
        WHERE substr(s.ds_timeindex, 1, 4) = ?
    """, (str(year),)).fetchone()

    total     = int(row["total"] or 0)
    valid     = int(row["valid_coord"] or 0)
    sog_nm    = float(row["sog_nm"] or 0)
    valid_pct = (valid / total * 100) if total > 0 else 0

    pos_df = pd.read_sql_query("""
        SELECT s.lat, s.lon FROM sensor_log s
        JOIN voyages v ON s.ds_timeindex BETWEEN v.start_time AND v.end_time
        WHERE substr(s.ds_timeindex, 1, 4) = ?
          AND s.lat != 0 AND s.lon != 0
        ORDER BY s.ds_timeindex
    """, conn, params=(str(year),))
    coord_nm = _haversine_total(pos_df)

    use_sog = valid_pct < 50 or coord_nm <= 0 or (sog_nm > 0 and coord_nm < sog_nm * 0.7)

    if use_sog and sog_nm > 0:
        return {
            "distance_nm":     round(sog_nm, 1),
            "distance_method": "sog_integration",
            "distance_note":   f"SOG 적분 (lon=0 결측 {100-valid_pct:.0f}%)",
        }
    return {
        "distance_nm":     round(coord_nm, 1),
        "distance_method": "coordinates",
        "distance_note":   "Haversine",
    }
