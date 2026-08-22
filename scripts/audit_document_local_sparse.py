"""Audit the bounded per-PDF sparse fallback against broad PDF gold pages."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAG_SCRIPTS = ROOT / "rag" / "scripts"
for path in (ROOT, RAG_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rag_fast_mode import _document_local_query_hits  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data/eval/broad_pdf_150_final.jsonl",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = []
    for row in rows:
        doc_id = str(row.get("gold_doc_id") or "")
        hits = _document_local_query_hits(
            ROOT / "data/processed/chunks",
            doc_id,
            str(row.get("question") or ""),
            existing=[],
            limit=max(args.top_k, 10),
            preview_chars=4000,
        )
        top = hits[: args.top_k]
        gold_pages = {int(page) for page in row.get("gold_pages") or []}
        gold_chunk_ids = {str(cid) for cid in row.get("gold_chunk_ids") or []}
        records.append(
            {
                "question_id": row.get("question_id"),
                "question": row.get("question"),
                "source": row.get("gold_source"),
                "gold_doc_id": doc_id,
                "gold_pages": sorted(gold_pages),
                "gold_chunk_ids": sorted(gold_chunk_ids),
                "top_chunk_ids": [hit.chunk_id for hit in top],
                "top_pages": [hit.page_number for hit in top],
                "page_hit": any(hit.page_number in gold_pages for hit in top),
                "chunk_hit": any(hit.chunk_id in gold_chunk_ids for hit in top),
            }
        )

    valid = [record for record in records if record["gold_doc_id"]]
    summary = {
        "n": len(valid),
        "top_k": args.top_k,
        "page_hit_rate": sum(record["page_hit"] for record in valid) / len(valid),
        "chunk_hit_rate": sum(record["chunk_hit"] for record in valid) / len(valid),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
