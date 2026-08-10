"""Dense retrieval plus exact structured answering for restored table 7.1.2."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from embedding_policy import embed_texts_local

ROOT = Path(__file__).resolve().parents[1]
COLLECTION_ID = "kr_table_7_1_2_restored_pilot"
INDEX_ROOT = ROOT / "data/processed/index" / f"unified_{COLLECTION_ID}"
PILOT_QUESTIONS = (
    "표 7.1.2에서 선수미부의 선저종늑골 단면계수는?",
    "선수미부에서 만곡부를 포함하는 선측종늑골의 단면계수와 최소 단면계수는?",
    "선박 중앙부 및 선수단에서 0.15L와 선수격벽 사이의 선저종늑골 단면계수는?",
    "선박 중앙부 및 선수단에서 0.15L와 선수격벽 사이의 선측종늑골 최소 단면계수는?",
)


@lru_cache(maxsize=1)
def _resources():
    import chromadb
    from chromadb.config import Settings
    manifest = json.loads((INDEX_ROOT / "index_manifest.json").read_text(encoding="utf-8"))
    records = json.loads((INDEX_ROOT / "pilot_records.json").read_text(encoding="utf-8"))
    client = chromadb.PersistentClient(path=str(INDEX_ROOT / "chroma"), settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection(manifest["collection_name"])
    return collection, manifest, {record["record_id"]: record for record in records}


def _select_value(record: dict, minimum_only: bool) -> str:
    lines = [line.strip() for line in str(record["value_display"]).splitlines() if line.strip()]
    if minimum_only: return next((line for line in lines if "min" in line or "ₘᵢₙ" in line), lines[-1])
    return " / ".join(lines)


def answer_table_712(question: str, top_k: int = 4) -> dict[str, Any]:
    collection, manifest, records = _resources()
    vector = embed_texts_local([question], manifest["embedding_model"], for_query=True)[0]
    raw = collection.query(query_embeddings=[vector], n_results=min(top_k, len(records)), include=["distances", "metadatas"])
    hits = []
    for rank, (record_id, distance, metadata) in enumerate(zip(raw["ids"][0], raw["distances"][0], raw["metadatas"][0]), 1):
        hits.append({"rank": rank, "record_id": record_id, "distance": round(float(distance), 6),
                     "cell_id": metadata["cell_id"], "row_header": metadata["row_header"],
                     "column_header": metadata["column_header"]})
    # Deliberately use the embedding rank without table-specific row/column
    # correction.  Ground truth belongs only in the separate evaluation file.
    selected = [records[hits[0]["record_id"]]]
    compact = "".join(question.split())
    asks_both = any(key in compact for key in ("단면계수와최소", "단면계수및최소", "계수와최소", "계수및최소"))
    minimum_only = ("최소" in compact or "Zmin" in compact) and not asks_both
    evidence, sentences = [], []
    for index, record in enumerate(selected, 1):
        value = _select_value(record, minimum_only); column = " > ".join(record["column_header_path"])
        sentences.append(f"{record['row_header']}의 {column} 값은 {value}입니다. [{index}]")
        evidence.append({"citation_id": index, "cell_id": record["cell_id"], "row_header": record["row_header"],
                         "column_header": column, "value": value, "page": 11,
                         "table_id": record["table_id"], "file_name": "7편_2025.pdf"})
    crop_path = next(iter(collection.get(limit=1, include=["metadatas"])["metadatas"]), {}).get("crop_path", "")
    margin = round(hits[1]["distance"] - hits[0]["distance"], 6) if len(hits) > 1 else None
    return {"status": "retrieved", "answer": "표 7.1.2에 따르면 " + " ".join(sentences), "question": question,
            "selection_method": "dense_top1_no_cell_override", "dense_hits": hits, "top1_margin": margin,
            "low_confidence": margin is not None and margin < 0.01, "evidence": evidence, "crop_path": crop_path,
            "answer_mode": "restored_dense_top1", "llm_used": False}
