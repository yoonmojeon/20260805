"""Build KR table-QA corpus: 07b extraction + unified_kr_tables index (top-22 by layout table count)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CHUNKS_DIR = ROOT / "data/processed/chunks"
RAG_CORPUS = ROOT / "data/manifests/rag_corpus_457.csv"
OUT_MANIFEST = ROOT / "data/manifests/kr_table_top22.csv"
MIN_LAYOUT_TABLES = 50
COLLECTION_ID = "kr_tables_v1"


def doc_ids_top_kr(*, min_tables: int = MIN_LAYOUT_TABLES) -> list[str]:
    kr = pd.read_csv(RAG_CORPUS)
    kr = kr[kr["source"] == "KR"].copy()
    rows: list[tuple[int, str]] = []
    for _, r in kr.iterrows():
        did = str(r["doc_id"])
        chunks_path = CHUNKS_DIR / did / "chunks.jsonl"
        if not chunks_path.exists():
            continue
        n_table = 0
        with chunks_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if str(rec.get("element_type", "")).lower() == "table":
                    n_table += 1
        if n_table >= min_tables:
            rows.append((n_table, did))
    rows.sort(reverse=True)
    return [did for _, did in rows]


def write_manifest(doc_ids: list[str]) -> Path:
    df = pd.read_csv(RAG_CORPUS)
    out = df[df["doc_id"].isin(doc_ids)].copy()
    missing = [d for d in doc_ids if d not in set(out["doc_id"])]
    if missing:
        raise ValueError(f"doc_id(s) missing from rag_corpus_457.csv: {missing}")
    order = {d: i for i, d in enumerate(doc_ids)}
    out["_order"] = out["doc_id"].map(order)
    out = out.sort_values("_order").drop(columns=["_order"])
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_MANIFEST, index=False)
    return OUT_MANIFEST


def run_07b(doc_id: str, *, force: bool) -> None:
    table_chunks = CHUNKS_DIR / doc_id / "table_chunks.jsonl"
    if table_chunks.exists() and not force:
        print(f"[skip 07b] {doc_id} (table_chunks.jsonl exists)")
        return
    subprocess.run(
        [sys.executable, str(SCRIPTS / "07b_extract_table_chunks.py"), "--doc-id", doc_id],
        cwd=str(ROOT),
        check=True,
    )


def build_index(manifest_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "10_build_unified_index.py"),
            "--collection-id",
            COLLECTION_ID,
            "--doc-list",
            str(manifest_path),
            "--manifest",
            str(RAG_CORPUS),
            "--include-types",
            "table",
            "--structured-tables",
            "only",
            "--max-embedding-tokens",
            "420",
            "--embedding-overlap-tokens",
            "60",
        ],
        cwd=str(ROOT),
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="KR top-22 structured table extraction + index.")
    parser.add_argument("--min-tables", type=int, default=MIN_LAYOUT_TABLES)
    parser.add_argument("--skip-07b", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--force-07b", action="store_true", help="Re-run 07b even if table_chunks.jsonl exists")
    parser.add_argument("--doc-id", action="append", dest="doc_ids", help="Override doc_id list")
    args = parser.parse_args()

    doc_ids = args.doc_ids or doc_ids_top_kr(min_tables=args.min_tables)
    if not doc_ids:
        raise SystemExit("No KR documents matched the table threshold.")

    manifest_path = write_manifest(doc_ids)
    print(f"Manifest: {manifest_path} ({len(doc_ids)} docs)")

    if not args.skip_07b:
        for i, doc_id in enumerate(doc_ids, 1):
            print(f"\n[07b {i}/{len(doc_ids)}] {doc_id}")
            run_07b(doc_id, force=args.force_07b)

    if not args.skip_index:
        print(f"\n[index] unified_{COLLECTION_ID}")
        build_index(manifest_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
