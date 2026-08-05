"""Feature flags for table-aware RAG (schema retrieval, debug, confidence gate)."""
from __future__ import annotations

import os

# Stage-1 routing: table_summary + table_schema hybrid (default on for _table_qa rows).
USE_TABLE_SCHEMA_RETRIEVAL = os.environ.get("MARITIME_TABLE_SCHEMA_RETRIEVAL", "1").strip() not in (
    "0",
    "false",
    "False",
)

TABLE_SCHEMA_ROUTE_K = int(os.environ.get("MARITIME_TABLE_SCHEMA_ROUTE_K", "12"))
TABLE_SCHEMA_STAGE2_ROW_K = int(os.environ.get("MARITIME_TABLE_SCHEMA_ROW_K", "8"))
CONFIDENCE_GATE_THRESHOLD = float(os.environ.get("MARITIME_TABLE_CONFIDENCE_THRESHOLD", "0.38"))
SHOW_TABLE_RETRIEVAL_DEBUG = os.environ.get("MARITIME_TABLE_RETRIEVAL_DEBUG", "1").strip() not in (
    "0",
    "false",
)

# Schema scoring weights (override via env MARITIME_TABLE_SCORE_VECTOR=0.35 etc.)
def _score_weight(name: str, default: float) -> float:
    return float(os.environ.get(f"MARITIME_TABLE_SCORE_{name.upper()}", str(default)))


SCORING_WEIGHTS = {
    "vector": _score_weight("vector", 0.32),
    "caption_match": _score_weight("caption", 0.06),
    "table_topic_match": _score_weight("topic", 0.20),
    "column_match": _score_weight("column", 0.22),
    "row_entity_match": _score_weight("row", 0.16),
    "unit_match": _score_weight("unit", 0.04),
    "keyword_match": _score_weight("keyword", 0.06),
}
SCORING_BOOST_ROW_COL = _score_weight("boost_row_col", 0.18)
SCORING_BOOST_ROW_COL_TOPIC = _score_weight("boost_row_col_topic", 0.22)
SCORING_PENALTY_MISSING_COLUMN = _score_weight("penalty_missing_column", 0.22)
SCORING_PENALTY_ROW_ONLY = _score_weight("penalty_row_only", 0.14)
SCORING_PENALTY_TOPIC_MISMATCH = _score_weight("penalty_topic_mismatch", 0.18)
SCORING_CAPTION_AUX_MIN = _score_weight("caption_aux_min", 0.15)


def use_table_schema_retrieval() -> bool:
    return os.environ.get("MARITIME_TABLE_SCHEMA_RETRIEVAL", "1").strip() not in (
        "0",
        "false",
        "False",
    )
