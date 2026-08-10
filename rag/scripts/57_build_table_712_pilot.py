"""Build a four-cell embedding index for the restored KR table 7.1.2 pilot."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from embedding_policy import embed_texts_local, resolve_embedding_config
from index_build_lib import build_chroma_index

COLLECTION_ID = "kr_table_7_1_2_restored_pilot"
COLLECTION_NAME = f"maritime_{COLLECTION_ID}_chunks"
DEFAULT_SOURCE = ROOT / ("data/processed/vlm_table_pilot/kr_kr_rules_7_2025_2f2d6373/"
    "kr_kr_rules_7_2025_2f2d6373_p0011_t004/tatr_v1_1_all/snapped_structure_restored.json")


def build_document(record: dict) -> str:
    column = " > ".join(record["column_header_path"])
    aliases = " ".join(["표 7.1.2", "선저 및 선측 종늑골의 단면계수", "단면계수", "최소 단면계수",
                         record["row_header"], column, record["value_display"], record["value_normalized"]])
    return ("[table_row] source=KR file=7편_2025.pdf\n"
            "caption: 표 7.1.2 선저 및 선측 종늑골의 단면계수\n"
            f"행 위치: {record['row_header']}\n열: {column}\n"
            f"수식 표시: {record['value_display']}\n수식 정규화: {record['value_normalized']}\n검색어: {aliases}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--index-dir", type=Path, default=ROOT / "data/processed/index")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    records = source.get("rag_records") or []
    if len(records) != 4: raise ValueError(f"Expected four restored data-cell records, got {len(records)}")
    if any(item.get("needs_review") for item in records): raise ValueError("Refusing to index a formula marked needs_review")
    config = resolve_embedding_config("e5-base", args.model)
    model = str(config["model"])
    documents = [build_document(record) for record in records]
    embeddings = embed_texts_local(documents, model, for_query=False)
    chunks, metadatas = [], []
    crop = args.source.parents[1] / "crop.png"
    for record in records:
        chunks.append({"chunk_id": record["record_id"]})
        metadatas.append({
            "doc_id": "kr_kr_rules_7_2025_2f2d6373", "source": "KR", "file_name": "7편_2025.pdf",
            "page_number": 11, "element_type": "table", "element_id": "kr_kr_rules_7_2025_2f2d6373_p0011_e003",
            "chunk_type": "table_row", "table_id": source["table_id"],
            "caption": "표 7.1.2 선저 및 선측 종늑골의 단면계수", "row_index": int(record["cell_id"][1:3]),
            "row_header": record["row_header"], "column_header": " > ".join(record["column_header_path"]),
            "cell_id": record["cell_id"], "value_display": record["value_display"],
            "value_normalized": record["value_normalized"], "crop_path": str(crop.resolve()),
            "quality_status": "pass", "parser_version": "tatr+pdf-vector+hancomeqn-v1"})
    index_root = args.index_dir / f"unified_{COLLECTION_ID}"
    build_chroma_index(chunks, documents, metadatas, embeddings, index_root / "chroma", COLLECTION_NAME)
    manifest = {"collection_id": COLLECTION_ID, "collection_name": COLLECTION_NAME, "embedding_preset": "e5-base",
                "embedding_model": model, "embedding_model_revision": config.get("revision"),
                "embedding_provider": "sentence-transformers", "indexed_chunks": len(records),
                "source_structure": str(args.source.resolve()), "chroma_path": str((index_root / "chroma").resolve())}
    (index_root / "index_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (index_root / "pilot_records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"index": str(index_root.resolve()), "model": model, "records": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
