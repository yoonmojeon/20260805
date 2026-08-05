"""End-to-end retrieval/answer benchmark for the balanced 22-document table QA set."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from rag_answer_lib import DEFAULT_OLLAMA_BASE, DEFAULT_OLLAMA_MODEL, check_ollama_model
from rag_eval_lib import load_questions
from rag_inprocess import normalize_table_question_row, run_full_inprocess
from rag_resource_cache import load_unified_collection
from table_retrieval import evaluate_table_qa_retrieval


DEFAULT_QUESTIONS = ROOT / "data/eval/table_questions_22docs_v2.jsonl"
DEFAULT_RETRIEVAL_OUT = ROOT / "data/processed/logs/table_eval_22docs_retrieval_v2.json"
DEFAULT_ANSWER_OUT = ROOT / "data/processed/logs/table_eval_22docs_answers_v2.json"


def normalize(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣○△%+\-]+", "", str(value or "")).lower()


def answer_contains_gold(answer: str, gold: str) -> bool:
    if not gold:
        return False
    if gold == "-":
        return bool(re.search(r"(?:별도\s*요건\s*(?:없|없음)|해당\s*(?:없|없음)|=\s*-|\b-\b)", answer))
    return normalize(gold) in normalize(answer)


def answer_cites_gold(answer: str, row: dict) -> bool:
    page = str(row.get("gold_page") or "")
    file_name = str(row.get("gold_file_name") or "")
    stem = Path(file_name).stem
    file_hit = bool(stem and normalize(stem) in normalize(answer))
    page_hit = bool(page and re.search(rf"(?:p\.?\s*|페이지\s*){re.escape(page)}\b", answer, re.I))
    return file_hit and page_hit


def aggregate(results: list[dict]) -> dict:
    metric_keys = (
        "table_recall@k",
        "row_recall@k",
        "cell_exact_match",
        "citation_match",
        "answer_contains_gold",
        "answer_cites_gold",
    )

    def summarize(items: list[dict]) -> dict:
        out = {"n": len(items)}
        for key in metric_keys:
            present = [item[key] for item in items if key in item and item[key] is not None]
            if present:
                out[key] = round(sum(bool(v) for v in present) / len(present), 3)
        return out

    by_doc: dict[str, list[dict]] = defaultdict(list)
    by_type: dict[str, list[dict]] = defaultdict(list)
    by_scope: dict[str, list[dict]] = defaultdict(list)
    for item in results:
        by_doc[item["gold_doc_id"]].append(item)
        by_type[item["question_type"]].append(item)
        by_scope[item["eval_scope"]].append(item)
    return {
        "overall": summarize(results),
        "by_doc": {key: summarize(value) for key, value in sorted(by_doc.items())},
        "by_question_type": {key: summarize(value) for key, value in sorted(by_type.items())},
        "by_scope": {key: summarize(value) for key, value in sorted(by_scope.items())},
    }


def write_payload(path: Path, *, args, results: list[dict]) -> None:
    payload = {
        "collection_id": args.collection_id,
        "questions": str(args.questions),
        "top_k": args.top_k,
        "with_llm": args.with_llm,
        "model": args.model if args.with_llm else None,
        "eval_constrained": False,
        "summary": aggregate(results),
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--collection-id", default="kr_tables_v2")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "data/processed/index")
    parser.add_argument("--chunks-dir", type=Path, default=ROOT / "data/processed/chunks_v2")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-base", default=DEFAULT_OLLAMA_BASE)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--one-per-doc", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out_path = args.out or (DEFAULT_ANSWER_OUT if args.with_llm else DEFAULT_RETRIEVAL_OUT)

    if args.with_llm:
        ok, message = check_ollama_model(args.ollama_base, args.model, timeout=10)
        if not ok:
            raise SystemExit(message)

    questions = load_questions(args.questions)
    if args.one_per_doc:
        seen_docs: set[str] = set()
        one_each: list[dict] = []
        for row in questions:
            doc_id = str(row.get("gold_doc_id") or "")
            if doc_id and doc_id not in seen_docs:
                seen_docs.add(doc_id)
                one_each.append(row)
        questions = one_each
    end = len(questions) if not args.limit else min(len(questions), args.start + args.limit)
    selected_questions = questions[args.start:end]
    existing: list[dict] = []
    done: set[str] = set()
    if args.resume and out_path.exists():
        old = json.loads(out_path.read_text(encoding="utf-8"))
        existing = list(old.get("results") or [])
        done = {str(item.get("qid") or "") for item in existing}

    collection, embed_model, manifest = load_unified_collection(args.collection_id, args.index_dir)
    results = existing
    run_index = len(existing) + 1
    for raw in selected_questions:
        row = normalize_table_question_row(raw)
        qid = str(row.get("qid") or row.get("question_id") or "")
        if qid in done:
            continue
        out = run_full_inprocess(
            row,
            unified_id=args.collection_id,
            index_dir=args.index_dir,
            chunks_dir=args.chunks_dir,
            collection=collection,
            embed_model=embed_model,
            manifest=manifest,
            eval_constrained=False,
            llm_model=args.model,
            ollama_base=args.ollama_base,
            temperature=0.05,
            latency_mode="accurate",
            top_k=args.top_k,
            fetch_k=30,
            max_doc=8,
            max_docs=3,
            use_rerank=False,
            run_index=run_index,
            start_type="cold" if run_index == 1 else "warm",
            skip_llm=not args.with_llm,
            auto_llm_warm=False,
            skip_ollama_probe=run_index > 1,
        )
        search = out["search_out"] if not args.with_llm else out["search_out"]
        pool = search.get("retrieval_pool") or search.get("retrieved") or []
        retrieval = evaluate_table_qa_retrieval(pool, row, k=args.top_k)
        answer = str(out.get("answer") or "") if args.with_llm else ""
        item = {
            "qid": qid,
            "question": row.get("question"),
            "question_type": row.get("question_type"),
            "eval_scope": row.get("eval_scope"),
            "gold_doc_id": row.get("gold_doc_id"),
            "gold_file_name": row.get("gold_file_name"),
            "gold_page": row.get("gold_page"),
            "gold_table_id": row.get("gold_table_id"),
            "gold_row_key": row.get("gold_row_key"),
            "gold_column": row.get("gold_column"),
            "gold_answer": row.get("gold_answer"),
            "gold_cells": row.get("gold_cells") or [],
            **retrieval,
            "answer": answer,
            "answer_contains_gold": answer_contains_gold(answer, str(row.get("gold_answer") or "")) if args.with_llm else None,
            "answer_cites_gold": answer_cites_gold(answer, row) if args.with_llm else None,
            "answer_mode": out.get("answer_mode"),
            "top_hits": [
                {
                    "doc_id": chunk.doc_id,
                    "file_name": chunk.file_name,
                    "page": chunk.page_number,
                    "table_id": chunk.table_id,
                    "chunk_type": chunk.chunk_type,
                    "text": chunk.text[:350],
                }
                for chunk in pool[: args.top_k]
            ],
        }
        results.append(item)
        write_payload(out_path, args=args, results=results)
        print(
            f"[{len(results)}/{len(questions)}] {qid} type={item['question_type']} "
            f"table={item['table_recall@k']} cell={item['cell_exact_match']} "
            f"answer={item['answer_contains_gold']}",
            flush=True,
        )
        run_index += 1
    write_payload(out_path, args=args, results=results)
    print(json.dumps(aggregate(results), ensure_ascii=False, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
