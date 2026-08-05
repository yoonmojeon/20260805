"""
SQLite 스키마 — 원시 시간별 센서 데이터 저장
집계 없이 raw 1시간 단위 데이터 그대로 보존.
LLM 툴이 필요할 때 SQL로 집계.
"""
import sqlite3
import sys
from pathlib import Path

_OPS_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _OPS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from project_paths import OPS_DB_PATH

DB_PATH = OPS_DB_PATH

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS vessel (
    imo         TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    type        TEXT DEFAULT 'Container Ship',
    gt          REAL DEFAULT 0,
    dwt         REAL DEFAULT 0,
    flag        TEXT DEFAULT 'Unknown',
    built_year  INTEGER DEFAULT 0,
    updated_at  TEXT DEFAULT (datetime('now'))
);

-- 원시 시간별 센서 로그 (1시간 단위, 집계 없음)
CREATE TABLE IF NOT EXISTS sensor_log (
    ds_timeindex    TEXT NOT NULL PRIMARY KEY,
    lat             REAL DEFAULT 0,
    lon             REAL DEFAULT 0,
    sog             REAL DEFAULT 0,
    me1_power       REAL DEFAULT 0,
    me2_power       REAL DEFAULT 0,
    me_oil_flow     REAL DEFAULT 0,   -- kg/h
    me_gas_flow     REAL DEFAULT 0,   -- kg/h
    me_co2          REAL DEFAULT 0,   -- kg/h
    me_ch4          REAL DEFAULT 0,
    me_co2e         REAL DEFAULT 0,
    ge_oil_flow     REAL DEFAULT 0,
    ge_gas_flow     REAL DEFAULT 0,
    ge_co2          REAL DEFAULT 0,
    ge_ch4          REAL DEFAULT 0,
    ge_co2e         REAL DEFAULT 0,
    ab_oil_flow     REAL DEFAULT 0,
    ab_gas_flow     REAL DEFAULT 0,
    gcu_oil_flow    REAL DEFAULT 0,
    gcu_gas_flow    REAL DEFAULT 0,
    oil_flow        REAL DEFAULT 0,   -- 선박 전체 유류 kg/h
    gas_flow        REAL DEFAULT 0,   -- 선박 전체 가스 kg/h
    co2             REAL DEFAULT 0,   -- 선박 전체 CO2 kg/h
    ch4             REAL DEFAULT 0,
    co2e            REAL DEFAULT 0,
    cii             REAL DEFAULT 0    -- kg CO2/nm
);

-- 항차 구간 (ship_info Schedule 기준, 미정의 구간은 SOG 폴백)
CREATE TABLE IF NOT EXISTS voyages (
    voyage_id       TEXT PRIMARY KEY,
    imo             TEXT NOT NULL,
    voyage_num      INTEGER,
    condition       TEXT DEFAULT '',
    start_time      TEXT NOT NULL,
    end_time        TEXT NOT NULL,
    departure_port  TEXT DEFAULT 'Unknown',
    arrival_port    TEXT DEFAULT 'Unknown',
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_sensor_time ON sensor_log(ds_timeindex);
CREATE INDEX IF NOT EXISTS idx_sensor_sog  ON sensor_log(sog);
CREATE INDEX IF NOT EXISTS idx_voyage_time ON voyages(start_time, end_time);
"""


def get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    # 기존 DB 마이그레이션
    cols = {r[1] for r in conn.execute("PRAGMA table_info(voyages)").fetchall()}
    if "condition" not in cols:
        conn.execute("ALTER TABLE voyages ADD COLUMN condition TEXT DEFAULT ''")
        conn.commit()
    conn.execute("PRAGMA cache_size=-32000")  # 32MB 캐시
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def get_db_path() -> Path:
    return DB_PATH
