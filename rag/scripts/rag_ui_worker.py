"""CLI worker for Streamlit UI — runs RAG in a separate process (avoids Streamlit+torch crash)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from rag_answer_lib import (
    DEFAULT_OLLAMA_BASE,
    DEFAULT_OLLAMA_MODEL,
    RetrievedChunk,
    build_answer_verification,
    check_ollama_model,
    generate_answer,
    load_unified_collection,
    reference_for_question,
    run_retrieval_only,
)
from retrieval_timing import (
    TimingTrace,
    populate_timing_meta,
)
from retrieval_verification import append_retrieval_trace_log, serialize_chunk_list

TRACE_LOG = _ROOT / "data/processed/logs/pilot_validation/retrieval_trace_ui.jsonl"


def _init_timing(args: argparse.Namespace) -> TimingTrace:
    timing = TimingTrace()
    if getattr(args, "user_submit_ts", None):
        timing.set_user_submit(float(args.user_submit_ts))
    return timing


def _finalize_timing_log(timing: TimingTrace) -> dict:
    return timing.to_log_row()


def cmd_search(args: argparse.Namespace) -> int:
    row = json.loads(args.row_json)
    timing = _init_timing(args)
    timing.mark("t_retrieval_start")
    collection, embed_model, _ = load_unified_collection(
        args.unified, Path(args.index_dir), timing=timing
    )
    result = run_retrieval_only(
        row,
        collection,
        embed_model,
        chunks_dir=Path(args.chunks_dir),
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        use_diversity_rerank=not args.no_rerank,
        max_chunks_per_doc=args.max_doc,
        max_docs=args.max_docs,
        eval_constrained_mode=args.eval_constrained,
        gold_doc_filter=False if not args.eval_constrained else None,
        timing=timing,
    )
    mode = result.get("answer_mode", "standard_rag")
    populate_timing_meta(
        timing,
        row=row,
        mode=mode,
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        retrieved=result["retrieved"],
        pool=result.get("retrieval_pool") or [],
        action="search",
    )
    timing.set_cache("llm_server_ready", False)
    log_row = _finalize_timing_log(timing)
    summary = result.get("verification_summary") or {}
    summary["timing_metrics"] = log_row["timing_metrics"]
    summary["timing_summary_lines"] = timing.summary_lines()
    out = {
        "retrieved": serialize_chunk_list(result["retrieved"]),
        "retrieval_pool": serialize_chunk_list(result.get("retrieval_pool") or []),
        "retrieval_metrics": result["retrieval_metrics"],
        "retrieval_config": result["retrieval_config"],
        "answer_mode": mode,
        "question_category": result.get("question_category"),
        "question_category_label": result.get("question_category_label"),
        "broad_summary_mode": result.get("broad_summary_mode", False),
        "doc_groups": result.get("doc_groups", []),
        "pipeline_warnings": result.get("pipeline_warnings", []),
        "evidence_table": result["evidence_table"],
        "must_cover_coverage": result["must_cover_coverage"],
        "verification_summary": summary,
        "timing_metrics": log_row["timing_metrics"],
        "timing_log": log_row,
    }
    return _json_out(out)


def _reference_for_row(row: dict) -> dict | None:
    return reference_for_question(row)


def _chunk_from_dict(d: dict) -> RetrievedChunk:
    return RetrievedChunk(**d)


def _json_out(payload: dict) -> int:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    return 0


def _resolve_answer_inputs(args: argparse.Namespace) -> dict:
    """Load answer inputs from --payload-file (preferred) or legacy CLI JSON args."""
    if args.payload_file:
        raw = json.loads(Path(args.payload_file).read_text(encoding="utf-8"))
        return {
            "row": raw["row"],
            "chunks": [_chunk_from_dict(d) for d in raw["chunks"]],
            "pool": [_chunk_from_dict(d) for d in raw.get("pool") or raw["chunks"]],
            "config_dict": raw.get("config"),
            "metrics": raw.get("metrics"),
            "doc_groups": raw.get("doc_groups"),
            "answer_mode": raw.get("answer_mode", "standard_rag"),
            "question_category": raw.get("question_category") or None,
            "llm_model": raw.get("llm_model", DEFAULT_OLLAMA_MODEL),
            "ollama_base": raw.get("ollama_base", DEFAULT_OLLAMA_BASE),
            "temperature": float(raw.get("temperature", 0.15)),
            "save_trace": bool(raw.get("save_trace", False)),
            "multi_doc_strategy": raw.get("multi_doc_strategy", "single_pass"),
            "max_llm_docs": int(raw.get("max_llm_docs", 6)),
            "top_k": int(raw.get("top_k", 10)),
            "fetch_k": int(raw.get("fetch_k", 120)),
        }
    if not args.row_json or not args.chunks_json:
        raise ValueError("answer requires --payload-file or --row-json and --chunks-json")
    chunks = [_chunk_from_dict(d) for d in json.loads(args.chunks_json)]
    return {
        "row": json.loads(args.row_json),
        "chunks": chunks,
        "pool": [_chunk_from_dict(d) for d in json.loads(args.pool_json)] if args.pool_json else chunks,
        "config_dict": json.loads(args.config_json) if args.config_json else None,
        "metrics": json.loads(args.metrics_json) if args.metrics_json else None,
        "doc_groups": json.loads(args.doc_groups_json) if args.doc_groups_json else None,
        "answer_mode": args.answer_mode or "standard_rag",
        "question_category": args.question_category or None,
        "llm_model": args.llm_model,
        "ollama_base": args.ollama_base,
        "temperature": args.temperature,
        "save_trace": args.save_trace,
        "multi_doc_strategy": getattr(args, "multi_doc_strategy", "batched"),
        "max_llm_docs": getattr(args, "max_llm_docs", 6),
    }


def cmd_answer(args: argparse.Namespace) -> int:
    inp = _resolve_answer_inputs(args)
    row = inp["row"]
    chunks = inp["chunks"]
    pool = inp["pool"]
    timing = _init_timing(args)
    llm_ok, _ = check_ollama_model(inp["ollama_base"], inp["llm_model"])
    timing.set_cache("llm_server_ready", llm_ok)
    answer, provider, model = generate_answer(
        row,
        chunks,
        provider="ollama",
        model=inp["llm_model"],
        ollama_base=inp["ollama_base"],
        temperature=inp["temperature"],
        allow_extractive_fallback=True,
        reference=_reference_for_row(row),
        answer_mode=inp["answer_mode"],
        pool=pool,
        category=inp["question_category"],
        doc_groups=inp["doc_groups"],
        multi_doc_strategy=inp.get("multi_doc_strategy", "single_pass"),
        max_llm_docs=inp.get("max_llm_docs", 6),
        timing=timing,
    )
    verification = build_answer_verification(
        row,
        chunks,
        answer,
        config_dict=inp["config_dict"],
        pool=pool,
        metrics=inp["metrics"],
    )
    if inp["save_trace"]:
        entry = verification.get("trace", {})
        entry["llm_provider"] = provider
        entry["llm_model"] = model
        entry["answer_mode"] = inp["answer_mode"]
        append_retrieval_trace_log(TRACE_LOG, entry)
    populate_timing_meta(
        timing,
        row=row,
        mode=inp["answer_mode"],
        top_k=inp.get("top_k", 10),
        fetch_k=inp.get("fetch_k", 120),
        retrieved=chunks,
        pool=pool,
        answer=answer,
        action="answer",
    )
    log_row = _finalize_timing_log(timing)
    summary = verification.get("verification_summary") or {}
    summary["timing_metrics"] = log_row["timing_metrics"]
    summary["timing_summary_lines"] = timing.summary_lines()
    return _json_out(
        {
            "answer": answer,
            "provider": provider,
            "model": model,
            "evidence_table": verification["evidence_table"],
            "answer_citation_mapping": verification["answer_citation_mapping"],
            "must_cover_coverage": verification["must_cover_coverage"],
            "verification_summary": summary,
            "timing_metrics": log_row["timing_metrics"],
            "timing_log": log_row,
        }
    )


def cmd_full(args: argparse.Namespace) -> int:
    """Search + LLM answer in one worker process (for timing benchmark)."""
    row = json.loads(args.row_json)
    timing = _init_timing(args)
    timing.mark("t_retrieval_start")
    collection, embed_model, _ = load_unified_collection(
        args.unified, Path(args.index_dir), timing=timing
    )
    result = run_retrieval_only(
        row,
        collection,
        embed_model,
        chunks_dir=Path(args.chunks_dir),
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        use_diversity_rerank=not args.no_rerank,
        max_chunks_per_doc=args.max_doc,
        max_docs=args.max_docs,
        eval_constrained_mode=args.eval_constrained,
        gold_doc_filter=False if not args.eval_constrained else None,
        timing=timing,
    )
    llm_ok, _ = check_ollama_model(args.ollama_base, args.llm_model)
    timing.set_cache("llm_server_ready", llm_ok)
    answer, provider, model = generate_answer(
        row,
        result["retrieved"],
        provider="ollama",
        model=args.llm_model,
        ollama_base=args.ollama_base,
        temperature=args.temperature,
        allow_extractive_fallback=True,
        reference=_reference_for_row(row),
        answer_mode=result.get("answer_mode", "standard_rag"),
        pool=result.get("retrieval_pool") or [],
        category=result.get("question_category"),
        doc_groups=result.get("doc_groups"),
        multi_doc_strategy=args.multi_doc_strategy,
        max_llm_docs=args.max_llm_docs,
        timing=timing,
    )
    mode = result.get("answer_mode", "standard_rag")
    populate_timing_meta(
        timing,
        row=row,
        mode=mode,
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        retrieved=result["retrieved"],
        pool=result.get("retrieval_pool") or [],
        answer=answer,
        action="full_rag",
    )
    log_row = _finalize_timing_log(timing)
    metrics = log_row["timing_metrics"]
    return _json_out(
        {
            "question_id": row.get("question_id"),
            "answer_mode": mode,
            "answer_chars": len(answer),
            "timing_metrics": metrics,
            "timing_log": log_row,
            "timing_summary_lines": timing.summary_lines(),
        }
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="RAG worker for Streamlit subprocess calls.")
    parser.add_argument("--unified", default="full_corpus_715_v1")
    parser.add_argument("--index-dir", default=str(_ROOT / "data/processed/index"))
    parser.add_argument("--chunks-dir", default=str(_ROOT / "data/processed/chunks"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_search = sub.add_parser("search")
    p_search.add_argument("--row-json", required=True)
    p_search.add_argument("--top-k", type=int, default=10)
    p_search.add_argument("--fetch-k", type=int, default=120)
    p_search.add_argument("--max-doc", type=int, default=3)
    p_search.add_argument("--max-docs", type=int, default=10)
    p_search.add_argument("--no-rerank", action="store_true")
    p_search.add_argument("--eval-constrained", action="store_true")
    p_search.add_argument(
        "--user-submit-ts",
        type=float,
        default=None,
        help="perf_counter() at UI button click (for end-to-end timing)",
    )
    p_search.set_defaults(func=cmd_search)

    p_ans = sub.add_parser("answer")
    p_ans.add_argument(
        "--payload-file",
        default="",
        help="JSON file with row/chunks/pool/config (avoids Windows cmdline length limit)",
    )
    p_ans.add_argument("--row-json", default="")
    p_ans.add_argument("--chunks-json", default="")
    p_ans.add_argument("--pool-json", default="")
    p_ans.add_argument("--config-json", default="")
    p_ans.add_argument("--metrics-json", default="")
    p_ans.add_argument("--doc-groups-json", default="")
    p_ans.add_argument("--answer-mode", default="standard_rag")
    p_ans.add_argument("--question-category", default="")
    p_ans.add_argument("--llm-model", default=DEFAULT_OLLAMA_MODEL)
    p_ans.add_argument("--ollama-base", default=DEFAULT_OLLAMA_BASE)
    p_ans.add_argument("--temperature", type=float, default=0.15)
    p_ans.add_argument("--save-trace", action="store_true")
    p_ans.add_argument("--user-submit-ts", type=float, default=None)
    p_ans.set_defaults(func=cmd_answer)

    p_full = sub.add_parser("full")
    p_full.add_argument("--row-json", required=True)
    p_full.add_argument("--top-k", type=int, default=10)
    p_full.add_argument("--fetch-k", type=int, default=120)
    p_full.add_argument("--max-doc", type=int, default=3)
    p_full.add_argument("--max-docs", type=int, default=10)
    p_full.add_argument("--no-rerank", action="store_true")
    p_full.add_argument("--eval-constrained", action="store_true")
    p_full.add_argument("--llm-model", default=DEFAULT_OLLAMA_MODEL)
    p_full.add_argument("--ollama-base", default=DEFAULT_OLLAMA_BASE)
    p_full.add_argument("--temperature", type=float, default=0.15)
    p_full.add_argument("--multi-doc-strategy", default="single_pass")
    p_full.add_argument("--max-llm-docs", type=int, default=6)
    p_full.add_argument("--user-submit-ts", type=float, default=None)
    p_full.set_defaults(func=cmd_full)

    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        sys.stderr.write(f"WORKER_ERROR: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
