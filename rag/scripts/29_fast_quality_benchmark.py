"""Fast mode quality benchmark: latency + retrieval/answer quality metrics."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from embedding_policy import DEFAULT_EMBEDDING_PRESET, resolve_embedding_config
from fast_confidence import assess_fast_confidence
from fast_question_classifier import classify_fast_question_type
from fast_retrieval import select_fast_evidence_slots
from meeting_outcome_retrieval import meeting_doc_recall_at_k
from rag_answer_lib import RetrievedChunk, load_unified_collection
from rag_eval_lib import load_questions
from rag_fast_mode import (
    FAST_RETRIEVAL,
    build_fast_context_and_chunks,
    trim_fast_chunks,
)
from rag_inprocess import DEFAULT_OLLAMA_MODEL, run_full_inprocess
from rag_resource_cache import warm_all_resources

TOKEN_RE = re.compile(r"[\w가-힣]+", re.UNICODE)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _must_cover_hit(retrieved: list[RetrievedChunk], row: dict) -> bool:
    must = row.get("must_cover") or row.get("expected_topics") or []
    if not must:
        return True
    corpus = _norm(" ".join(c.text or "" for c in retrieved))
    hits = sum(1 for t in must if _norm(str(t)) in corpus)
    return hits >= max(1, (len(must) + 1) // 2)


def _citation_match(retrieved: list[RetrievedChunk], row: dict) -> bool:
    gold_doc = str(row.get("gold_doc_id") or "")
    if not gold_doc:
        return False
    gold_page = row.get("gold_page")
    gold_pages = row.get("gold_pages") or []
    for c in retrieved:
        if c.doc_id != gold_doc:
            continue
        if gold_page is not None and c.page_number == int(gold_page):
            return True
        if gold_pages and c.page_number in gold_pages:
            return True
        if c.table_id and c.table_id == str(row.get("gold_table_id") or ""):
            return True
        if gold_page is None and not gold_pages:
            return True
    return False


def _answer_depth_simple(answer: str, fast_type: str) -> float:
    if not answer:
        return 0.0
    score = 0.0
    if re.search(r"\[\d+\]", answer):
        score += 0.25
    bullets = len(re.findall(r"^[\s]*[-*•]", answer, re.M)) + len(re.findall(r"^\s*\d+\.", answer, re.M))
    score += min(0.35, bullets * 0.08)
    if fast_type == "meeting_outcome_question" and ("결정" in answer or "영향" in answer):
        score += 0.2
    if fast_type == "table_question" and ("정답" in answer or "근거" in answer):
        score += 0.2
    if fast_type == "rule_question" and ("결론" in answer or "조항" in answer):
        score += 0.2
    if len(answer) >= 80:
        score += 0.15
    return min(1.0, score)


def _hallucination_flag_simple(answer: str, retrieved: list[RetrievedChunk], row: dict) -> int:
    """1 if answer cites [N] beyond retrieved count or obvious placeholder."""
    if not answer:
        return 0
    cites = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
    if cites and max(cites) > len(retrieved):
        return 1
    if re.search(r"\[근거\]|\[99\]|placeholder", answer, re.I):
        return 1
    gold = str(row.get("gold_answer") or "")
    if gold and gold not in answer and row.get("fast_question_type") == "table_question":
        return 0
    return 0


def evaluate_retrieval(row: dict, retrieved: list[RetrievedChunk], *, fast_type: str) -> dict:
    metrics = {
        "must_cover_hit": int(_must_cover_hit(retrieved, row)),
        "citation_match": int(_citation_match(retrieved, row)),
    }
    if fast_type == "meeting_outcome_question":
        metrics["meeting_doc_recall@k"] = int(meeting_doc_recall_at_k(retrieved, row, len(retrieved)))
    if fast_type == "table_question":
        metrics["has_table_evidence"] = int(
            any(c.chunk_type in {"table_row", "table_markdown", "table_summary"} for c in retrieved)
        )
    return metrics


def run_retrieval_benchmark(row: dict, collection, model_name: str, chunks_dir: Path) -> dict:
    fast_type = classify_fast_question_type(str(row["question"]), row)

    pool_fetch = FAST_RETRIEVAL.get("pool_fetch_k", FAST_RETRIEVAL["fetch_k"])
    from rag_answer_lib import retrieve_for_question

    pool = retrieve_for_question(
        collection,
        model_name,
        row,
        top_k=pool_fetch,
        fetch_k=pool_fetch,
        chunks_dir=chunks_dir,
        preview_chars=FAST_RETRIEVAL["preview_chars"],
        gold_doc_filter=False,
    )
    typed_chunks, _, fast_meta = build_fast_context_and_chunks(pool, row, use_typed_slots=True)
    legacy_chunks = trim_fast_chunks(pool, max_chunks=3, max_docs=2, max_per_doc=1)

    evidence = select_fast_evidence_slots(pool, str(row["question"]), row, fast_type=fast_type)
    conf = assess_fast_confidence(str(row["question"]), row, evidence, fast_type=fast_type)

    return {
        "fast_type": fast_type,
        "fast_meta": {
            k: fast_meta.get(k)
            for k in (
                "fast_question_type",
                "fast_question_type_label",
                "fast_evidence_slots",
                "fast_confidence",
                "fast_low_confidence",
            )
        },
        "confidence": conf.score,
        "low_confidence": conf.low_confidence,
        "typed_chunk_count": len(typed_chunks),
        "legacy_chunk_count": len(legacy_chunks),
        "typed_metrics": evaluate_retrieval(row, typed_chunks, fast_type=fast_type),
        "legacy_metrics": evaluate_retrieval(row, legacy_chunks, fast_type=fast_type),
        "top_docs_typed": list(dict.fromkeys(c.file_name for c in typed_chunks))[:5],
        "top_docs_legacy": list(dict.fromkeys(c.file_name for c in legacy_chunks))[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast mode quality + latency benchmark.")
    parser.add_argument("--questions", type=Path, default=Path("data/eval/fast_questions.jsonl"))
    parser.add_argument("--collection-id", type=str, default="full_corpus_715_v1")
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/index"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--embedding-preset", type=str, default=DEFAULT_EMBEDDING_PRESET)
    parser.add_argument("--with-llm", action="store_true", help="Run full fast inprocess for latency/answer metrics")
    parser.add_argument("--llm-model", type=str, default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-base", type=str, default="http://localhost:11434")
    parser.add_argument("--out", type=Path, default=Path("data/processed/logs/fast_quality_benchmark.json"))
    args = parser.parse_args()

    questions = load_questions(args.questions)
    results: list[dict] = []
    agg_typed = {"must_cover_hit": 0, "citation_match": 0}
    agg_legacy = {"must_cover_hit": 0, "citation_match": 0}

    for row in questions:
        coll_id = str(row.get("collection_id") or args.collection_id)
        collection, model_name, _ = load_unified_collection(coll_id, args.index_dir)
        embed_config = resolve_embedding_config(args.embedding_preset, None)
        if not model_name:
            model_name = str(embed_config["model"])

        entry = {
            "question_id": row.get("question_id"),
            "question": row.get("question"),
            **run_retrieval_benchmark(row, collection, model_name, args.chunks_dir),
        }

        if args.with_llm:
            warm_all_resources(coll_id, args.index_dir, args.chunks_dir)
            t0 = time.time()
            out = run_full_inprocess(
                row,
                unified_id=coll_id,
                index_dir=args.index_dir,
                chunks_dir=args.chunks_dir,
                llm_model=args.llm_model,
                ollama_base=args.ollama_base,
                latency_mode="fast",
                start_type="warm",
            )
            tm = (out.get("timing_metrics") or {})
            answer = out.get("answer") or ""
            ft = entry["fast_type"]
            entry["llm"] = {
                "first_visible_latency": tm.get("e2e_ttft") or tm.get("llm_ttft"),
                "e2e_total": tm.get("e2e_total"),
                "answer_depth_simple": round(_answer_depth_simple(answer, ft), 3),
                "hallucination_flag_simple": _hallucination_flag_simple(
                    answer, out.get("search_out", {}).get("retrieved") or [], row
                ),
                "answer_chars": len(answer),
                "answer_preview": answer[:300],
            }

        for k in agg_typed:
            agg_typed[k] += entry["typed_metrics"].get(k, 0)
            agg_legacy[k] += entry["legacy_metrics"].get(k, 0)
        results.append(entry)

    n = max(len(questions), 1)
    summary = {
        "question_count": len(questions),
        "typed_layer": {k: round(v / n, 3) for k, v in agg_typed.items()},
        "legacy_trim": {k: round(v / n, 3) for k, v in agg_legacy.items()},
    }
    if args.with_llm:
        latencies = [r["llm"]["first_visible_latency"] for r in results if r.get("llm", {}).get("first_visible_latency")]
        if latencies:
            summary["avg_first_visible_latency"] = round(sum(latencies) / len(latencies), 3)
            summary["fast_3s_pass_rate"] = round(sum(1 for x in latencies if x <= 3.0) / len(latencies), 3)

    payload = {"summary": summary, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
