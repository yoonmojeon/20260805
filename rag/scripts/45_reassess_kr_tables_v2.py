"""Reassess existing v2 table grids and regenerate safe embedding chunks."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from table_v2_lib import assess_quality, build_v2_chunks


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-list", type=Path, default=ROOT / "data/manifests/kr_table_top22.csv")
    parser.add_argument("--tables-root", type=Path, default=ROOT / "data/processed/tables_v2")
    parser.add_argument("--chunks-root", type=Path, default=ROOT / "data/processed/chunks_v2")
    parser.add_argument("--report", type=Path, default=ROOT / "data/processed/logs/kr_tables_v2_quality.json")
    args = parser.parse_args()

    with args.doc_list.open(encoding="utf-8-sig", newline="") as f:
        docs = list(csv.DictReader(f))
    summaries: list[dict] = []
    totals: Counter[str] = Counter()
    for doc in docs:
        doc_id = str(doc["doc_id"])
        source = str(doc.get("source") or "KR")
        file_name = str(doc.get("file_name") or "")
        table_path = args.tables_root / doc_id / "tables.jsonl"
        tables = load_jsonl(table_path)
        chunks: list[dict] = []
        status: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        for table in tables:
            if table.get("quality_reasons") == ["coordinate_grid_not_found"]:
                status["reject"] += 1
                reasons["coordinate_grid_not_found"] += 1
                continue
            columns = list(table.get("column_names") or [])
            rows = list((table.get("table_json") or {}).get("rows") or [])
            strategy = str(table.get("extraction_method") or "").rsplit(":", 1)[-1]
            score, state, why, metrics = assess_quality(
                columns=columns,
                rows=rows,
                caption=str(table.get("caption") or ""),
                strategy=strategy,
            )
            table["quality_score"] = score
            table["quality_status"] = state
            table["quality_reasons"] = why
            table["quality_metrics"] = metrics
            status[state] += 1
            reasons.update(reason.split(":", 1)[0] for reason in why)
            chunks.extend(build_v2_chunks(table, source=source, file_name=file_name))
        write_jsonl(table_path, tables)
        write_jsonl(args.chunks_root / doc_id / "table_chunks.jsonl", chunks)
        totals.update(status)
        summaries.append({
            "doc_id": doc_id,
            "file_name": file_name,
            "output_tables": len(tables),
            "indexed_tables": status["pass"],
            "output_chunks": len(chunks),
            "quality_status": dict(status),
            "quality_reasons": dict(reasons),
        })
        print(f"{doc_id}: {dict(status)} chunks={len(chunks)}", flush=True)
    payload = {
        "corpus": "kr_tables_v2",
        "documents": len(summaries),
        "totals": dict(totals),
        "indexed_tables": sum(x["indexed_tables"] for x in summaries),
        "output_chunks": sum(x["output_chunks"] for x in summaries),
        "per_doc": summaries,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["totals"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
