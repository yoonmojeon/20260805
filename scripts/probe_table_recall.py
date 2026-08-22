"""Ask whether the failing table lookups are a recall problem or a ranking problem.

Runs the raw dense query against the table collection with a widening pool and
reports the first position where the gold value appears. If a wider pool finds
the value, the fix is retrieval breadth; if even 400 candidates miss it, the
embedding simply does not represent the row and the fix has to be lexical.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import DEFAULT_TABLE_COLLECTION, RAG_INDEX_DIR, RAG_SCRIPTS_DIR  # noqa: E402

TABLE_SET = ROOT / "data" / "eval" / "table_questions_22docs_practical_v1_curated.jsonl"
QIDS = ["TC22_005", "TC22_011", "TC22_014", "TC22_017", "TC22_019", "TC22_021"]
POOLS = (30, 100, 400)


def norm(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def main() -> int:
    if str(RAG_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(RAG_SCRIPTS_DIR))
    from rag_resource_cache import load_unified_collection  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

    collection, embed_model, _ = load_unified_collection(DEFAULT_TABLE_COLLECTION, RAG_INDEX_DIR)
    model = SentenceTransformer(embed_model)

    rows = {json.loads(l)["qid"]: json.loads(l) for l in TABLE_SET.read_text(encoding="utf-8").splitlines() if l.strip()}
    for qid in QIDS:
        row = rows[qid]
        question = row["question"]
        gold = row["gold_answer"]
        doc_id = row["gold_doc_id"]
        vector = model.encode([question], normalize_embeddings=True)[0].tolist()
        line = [f"{qid} | gold={gold!r} | doc={row['gold_file_name']} p{row['gold_page']}"]
        for pool in POOLS:
            res = collection.query(query_embeddings=[vector], n_results=pool)
            docs = res["documents"][0]
            metas = res["metadatas"][0]
            value_rank = next(
                (i for i, d in enumerate(docs, start=1) if norm(gold) in norm(d)), None
            )
            doc_rank = next(
                (
                    i
                    for i, m in enumerate(metas, start=1)
                    if doc_id in str((m or {}).get("doc_id") or (m or {}).get("chunk_id") or "")
                ),
                None,
            )
            line.append(f"pool {pool}: 값 rank {value_rank}, 정답문서 rank {doc_rank}")
        print(" | ".join(line), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
