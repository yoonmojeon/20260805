"""Retrieval diagnosis: compare vector rank vs rerank vs gold doc for eval questions."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from embedding_policy import embed_texts_local
from imo_doc_classify import classify_imo_filename
from imo_doc_registry import priority_doc_ids_for_signals
from rag_answer_lib import load_unified_collection, load_questions, retrieve_for_question
from retrieval_query_analysis import analyze_query
from retrieval_search import enrich_query_for_embedding, query_with_hybrid_ranking


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose retrieval for eval questions.")
    parser.add_argument("--unified", default="full_corpus_715_v1")
    parser.add_argument("--questions", type=Path, default=Path("data/eval/pilot_validation_questions.jsonl"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/index"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/logs/retrieval_diagnosis.json"))
    args = parser.parse_args()

    collection, model, _ = load_unified_collection(args.unified, args.index_dir)
    questions = load_questions(args.questions)
    rows: list[dict] = []

    for q in questions:
        question = str(q["question"])
        signals = analyze_query(question)
        embed_q = enrich_query_for_embedding(question, model)
        vector = embed_texts_local([embed_q], model, for_query=True)[0]
        sources = q.get("retrieval_sources") or []
        source_filter = sources[0] if len(sources) == 1 else None

        raw = query_with_hybrid_ranking(
            collection, question, vector, top_k=args.top_k, source=source_filter
        )
        retrieved = retrieve_for_question(
            collection, model, q, top_k=args.top_k, chunks_dir=args.chunks_dir
        )
        gold_doc = str(q.get("gold_doc_id") or "")
        gold_page = q.get("gold_page")

        top_meta = (raw["metadatas"][0][0] if raw["metadatas"][0] else {}) or {}
        top_ret = retrieved[0] if retrieved else None

        rows.append(
            {
                "question_id": q["question_id"],
                "question": question,
                "gold_doc_id": gold_doc,
                "gold_page": gold_page,
                "signals": {
                    "sessions": signals.session_codes,
                    "topics": sorted(signals.topics),
                    "wants_summary": signals.wants_summary,
                    "wants_outcome": signals.wants_outcome,
                },
                "priority_doc_ids": raw.get("priority_doc_ids", [])[:8],
                "top1_file": top_meta.get("file_name"),
                "top1_doc_type": classify_imo_filename(str(top_meta.get("file_name") or "")),
                "top1_page": top_meta.get("page_number"),
                "gold_in_topk": any(c.doc_id == gold_doc for c in retrieved),
                "gold_rank": next(
                    (i for i, c in enumerate(retrieved, start=1) if c.doc_id == gold_doc),
                    None,
                ),
                "gold_page_in_topk": any(
                    c.doc_id == gold_doc and c.page_number == int(gold_page)
                    for c in retrieved
                )
                if gold_doc and gold_page is not None
                else None,
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "unified_id": args.unified,
        "embedding_model": model,
        "questions": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    for r in rows:
        status = "OK" if r["gold_in_topk"] else "MISS"
        print(f"[{status}] {r['question_id']} rank={r['gold_rank']} top={r['top1_file']}")


if __name__ == "__main__":
    main()
