"""Find which chunks contain a phrase, to tell a missing answer from a missed answer.

    python scripts/grep_corpus.py "과도한 부식" --limit 8
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import RAG_CHUNKS_DIR, RAG_TABLE_CHUNKS_DIR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phrase")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--tables", action="store_true", help="표 청크에서 찾는다")
    parser.add_argument("--context", type=int, default=180)
    args = parser.parse_args()

    root = RAG_TABLE_CHUNKS_DIR if args.tables else RAG_CHUNKS_DIR
    name = "table_chunks.jsonl" if args.tables else "chunks.jsonl"
    needle = args.phrase
    hits = 0
    docs: set[str] = set()
    for path in sorted(root.glob(f"*/{name}")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if needle not in line:
                continue
            rec = json.loads(line)
            text = str(rec.get("text") or "")
            if needle not in text:
                continue
            hits += 1
            docs.add(str(rec.get("doc_id") or ""))
            if hits <= args.limit:
                idx = text.find(needle)
                window = re.sub(
                    r"\s+", " ", text[max(0, idx - args.context) : idx + args.context]
                )
                print(
                    f"[{hits}] {rec.get('file_name') or rec.get('doc_id')} p{rec.get('page_number')} "
                    f"{rec.get('chunk_id')}\n    …{window}…\n"
                )
    print(f"총 {hits}개 청크 · {len(docs)}개 문서")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
