"""Meeting outcome QA benchmark: meeting_doc_recall, must_cover, citation, depth."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from embedding_policy import DEFAULT_EMBEDDING_PRESET, resolve_embedding_config
from meeting_outcome_answer import answer_depth_score
from meeting_outcome_retrieval import meeting_doc_recall_at_k
from rag_answer_lib import RetrievedChunk, load_unified_collection, retrieve_for_question
from rag_eval_lib import load_chunk_text_map, load_questions
from retrieval_query_analysis import is_meeting_outcome_question
from retrieval_search import enrich_query_for_embedding, query_with_hybrid_ranking
from embedding_policy import embed_texts_local


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _must_cover_hit(retrieved: list[RetrievedChunk], row: dict, k: int) -> bool:
    must = row.get("must_cover") or []
    if not must:
        return True
    corpus = " ".join(_norm(c.text) for c in retrieved[:k])
    hits = sum(1 for term in must if _norm(term) in corpus)
    return hits >= max(1, len(must) // 2 + len(must) % 2)


def _citation_match(retrieved: list[RetrievedChunk], row: dict, k: int) -> bool:
    gold_doc = str(row.get("gold_doc_id") or "")
    gold_docs = row.get("gold_doc_ids") or []
    if isinstance(gold_docs, str):
        gold_docs = [gold_docs]
    targets = {gold_doc} if gold_doc else set()
    targets.update(str(d) for d in gold_docs if d)
    if not targets:
        return False
    gold_page = row.get("gold_page")
    gold_pages = row.get("gold_pages") or []
    doc_hit = False
    page_hit = False
    for c in retrieved[:k]:
        if c.doc_id not in targets:
            continue
        doc_hit = True
        if gold_page is not None and c.page_number == int(gold_page):
            page_hit = True
        if gold_pages and c.page_number in gold_pages:
            page_hit = True
    if not gold_page and not gold_pages:
        return doc_hit
    return page_hit or doc_hit


def retrieve_baseline_only(
    collection,
    model_name: str,
    row: dict,
    *,
    top_k: int,
    chunks_dir: Path,
) -> list[RetrievedChunk]:
    question = str(row["question"])
    embed_query = enrich_query_for_embedding(question, model_name)
    vector = embed_texts_local([embed_query], model_name, for_query=True)[0]
    raw = query_with_hybrid_ranking(
        collection,
        question,
        vector,
        top_k=top_k,
        fetch_k=max(top_k * 10, 80),
    )
    chunk_text_cache: dict[str, dict[str, str]] = {}
    out: list[RetrievedChunk] = []
    for chunk_id, distance, meta, doc in zip(
        raw["ids"][0],
        raw["distances"][0],
        raw["metadatas"][0],
        raw["documents"][0],
    ):
        meta = meta or {}
        doc_id = str(meta.get("doc_id", ""))
        if doc_id not in chunk_text_cache:
            cp = chunks_dir / doc_id / "chunks.jsonl"
            chunk_text_cache[doc_id] = load_chunk_text_map(cp) if cp.exists() else {}
        full_text = chunk_text_cache[doc_id].get(chunk_id) or doc or ""
        out.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                source=str(meta.get("source", "")),
                file_name=str(meta.get("file_name", "")),
                page_number=meta.get("page_number"),
                clause_number=str(meta.get("clause_number") or meta.get("article_number") or ""),
                element_type=str(meta.get("element_type", "")),
                distance=float(distance),
                text=full_text,
            )
        )
    return out


def evaluate_row(retrieved: list[RetrievedChunk], row: dict, k: int) -> dict:
    n_items = int(row.get("outcome_item_count") or 3)
    return {
        "meeting_outcome_detected": int(is_meeting_outcome_question(str(row.get("question", "")), row)),
        "meeting_doc_recall@k": int(meeting_doc_recall_at_k(retrieved, row, k)),
        "must_cover_hit": int(_must_cover_hit(retrieved, row, k)),
        "citation_match": int(_citation_match(retrieved, row, k)),
        "answer_depth_score": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Meeting outcome retrieval benchmark.")
    parser.add_argument("--questions", type=Path, default=Path("data/eval/meeting_questions.jsonl"))
    parser.add_argument("--collection-id", type=str, default="full_corpus_715_v1")
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/index"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--embedding-preset", type=str, default=DEFAULT_EMBEDDING_PRESET)
    parser.add_argument("--compare-baseline", action="store_true", default=True)
    parser.add_argument("--out", type=Path, default=Path("data/processed/logs/meeting_qa_benchmark.json"))
    args = parser.parse_args()

    questions = load_questions(args.questions)
    collection, model_name, _manifest = load_unified_collection(args.collection_id, args.index_dir)
    embed_config = resolve_embedding_config(args.embedding_preset, None)
    if not model_name:
        model_name = str(embed_config["model"])

    results: list[dict] = []
    agg = {
        "meeting_doc_recall@k": 0,
        "must_cover_hit": 0,
        "citation_match": 0,
    }
    baseline_agg = dict(agg)

    for row in questions:
        retrieved = retrieve_for_question(
            collection,
            model_name,
            row,
            top_k=args.top_k,
            fetch_k=max(args.top_k * 10, 80),
            chunks_dir=args.chunks_dir,
            gold_doc_filter=False,
        )
        metrics = evaluate_row(retrieved, row, args.top_k)
        entry = {
            "question_id": row.get("question_id"),
            "question": row.get("question"),
            **metrics,
            "top_docs": list(dict.fromkeys(c.file_name for c in retrieved[: args.top_k]))[:5],
        }

        if args.compare_baseline:
            baseline = retrieve_baseline_only(
                collection, model_name, row, top_k=args.top_k, chunks_dir=args.chunks_dir
            )
            bmetrics = evaluate_row(baseline, row, args.top_k)
            entry["baseline"] = bmetrics
            entry["baseline_top_docs"] = list(dict.fromkeys(c.file_name for c in baseline[: args.top_k]))[:5]
            for key in baseline_agg:
                baseline_agg[key] += bmetrics[key]

        for key in agg:
            agg[key] += metrics[key]
        results.append(entry)

    n = max(len(questions), 1)
    summary = {
        "collection_id": args.collection_id,
        "top_k": args.top_k,
        "question_count": len(questions),
        "meeting_outcome_layer": {k: round(v / n, 3) for k, v in agg.items()},
    }
    if args.compare_baseline:
        summary["baseline"] = {k: round(v / n, 3) for k, v in baseline_agg.items()}

    payload = {"summary": summary, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
