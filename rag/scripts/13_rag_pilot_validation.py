"""
Pilot validation: retrieval + optional diversity rerank + 1.2-format answers.

  python scripts/13_rag_pilot_validation.py --skip-llm
  python scripts/13_rag_pilot_validation.py --skip-llm --diversity-rerank
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from rag_answer_lib import (
    RetrievedChunk,
    ValidationResult,
    format_result_markdown,
    generate_answer,
    load_questions,
    load_unified_collection,
    run_retrieval_only,
    score_retrieval,
)
from rag_retrieval_metrics import chunk_to_report_dict, compute_retrieval_metrics
from retrieval_diversity import DiversityConfig, diversity_rerank, unique_page_count


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate_from_metrics(metrics_list: list[dict], eval_k: int) -> dict:
    n = len(metrics_list)
    if not n:
        return {}
    return {
        "eval_k": eval_k,
        "source_hit_at_5": sum(1 for m in metrics_list if m.get("source_hit_at_5")) / n,
        "gold_doc_hit_at_5": sum(1 for m in metrics_list if m.get("gold_doc_hit_at_5")) / n,
        "gold_page_set_hit_at_5": sum(1 for m in metrics_list if m.get("gold_page_set_hit_at_5")) / n,
        "topic_hit_at_k": sum(1 for m in metrics_list if m.get("topic_hit_at_k")) / n,
        "mean_keyword_coverage": _mean([float(m.get("keyword_coverage", 0)) for m in metrics_list]),
        "mean_boundary_error_rate": _mean([float(m.get("boundary_error_rate", 0)) for m in metrics_list]),
        "mean_duplicate_doc_ratio": _mean([float(m.get("duplicate_doc_ratio", 0)) for m in metrics_list]),
        "mean_duplicate_page_ratio": _mean([float(m.get("duplicate_page_ratio", 0)) for m in metrics_list]),
    }


def _metrics_snapshot(m: dict) -> dict:
    keys = (
        "source_hit_at_5",
        "gold_doc_hit_at_5",
        "gold_page_set_hit_at_5",
        "topic_hit_at_k",
        "keyword_coverage",
        "duplicate_doc_ratio",
        "duplicate_page_ratio",
        "gold_doc_rank",
    )
    return {k: m.get(k) for k in keys}


def _build_comparison(
    baseline_results: list[ValidationResult],
    reranked_results: list[ValidationResult],
    *,
    eval_k: int,
) -> dict:
    b_agg = _aggregate_from_metrics([r.retrieval_metrics for r in baseline_results], eval_k)
    r_agg = _aggregate_from_metrics([r.retrieval_metrics for r in reranked_results], eval_k)
    delta = {}
    for key in (
        "source_hit_at_5",
        "gold_doc_hit_at_5",
        "gold_page_set_hit_at_5",
        "topic_hit_at_k",
        "mean_keyword_coverage",
        "mean_duplicate_doc_ratio",
        "mean_duplicate_page_ratio",
    ):
        delta[key] = r_agg.get(key, 0) - b_agg.get(key, 0)

    per_q = []
    for b, r in zip(baseline_results, reranked_results):
        per_q.append(
            {
                "question_id": b.question_id,
                "category": b.category,
                "baseline": _metrics_snapshot(b.retrieval_metrics),
                "reranked": _metrics_snapshot(r.retrieval_metrics),
                "baseline_pages_top_k": [c.page_number for c in b.retrieved],
                "reranked_pages_top_k": [c.page_number for c in r.retrieved],
                "baseline_unique_pages": unique_page_count(b.retrieved, len(b.retrieved)),
                "reranked_unique_pages": unique_page_count(r.retrieved, len(r.retrieved)),
            }
        )

    return {
        "baseline": b_agg,
        "reranked": r_agg,
        "delta_reranked_minus_baseline": delta,
        "per_question": per_q,
    }


def _clone_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return copy.deepcopy(chunks)


def _run_pass(
    *,
    variant: str,
    questions: list[dict],
    collection,
    embed_model: str,
    chunks_dir: Path,
    top_k: int,
    eval_k: int,
    fetch_k: int,
    diversity_cfg: DiversityConfig | None,
    output_dir: Path,
    skip_llm: bool,
    llm_provider: str,
    llm_model: str,
    ollama_base: str,
    suffix: str,
    eval_constrained_mode: bool = False,
    max_docs: int = 4,
) -> tuple[list[ValidationResult], list[dict]]:
    results: list[ValidationResult] = []
    failure_rows: list[dict] = []
    eval_label = "constrained" if eval_constrained_mode else "open"

    for row in questions:
        print(
            f"\n[{variant}/{eval_label}] {row['question_id']}: {row['question']}",
            flush=True,
        )
        use_rerank = diversity_cfg is not None
        pack = run_retrieval_only(
            row,
            collection,
            embed_model,
            chunks_dir=chunks_dir,
            top_k=top_k,
            fetch_k=fetch_k if use_rerank else top_k,
            use_diversity_rerank=use_rerank,
            max_chunks_per_doc=diversity_cfg.max_chunks_per_doc if diversity_cfg else None,
            max_chunks_per_page=diversity_cfg.max_chunks_per_page if diversity_cfg else 1,
            max_docs=max_docs,
            eval_k=eval_k,
            eval_constrained_mode=eval_constrained_mode,
            gold_doc_filter=True if eval_constrained_mode else False,
        )
        retrieved = pack["retrieved"]
        metrics = pack["retrieval_metrics"]

        kw_hits = metrics.get("best_keyword_hits", 0)
        kw_total = metrics.get("keyword_total", 0)
        src_hits = metrics.get("source_hits_in_top_k", 0)
        hit = metrics.get("doc_recall_at_5") if not eval_constrained_mode else metrics.get(
            "gold_doc_hit_at_5"
        )
        gold_doc_hit = metrics.get("gold_doc_hit_at_5", False)
        gold_page_hit = metrics.get("page_recall_at_5", False) or metrics.get(
            "gold_page_set_hit_at_5", False
        )
        gold_rank = metrics.get("gold_doc_rank")

        vr = ValidationResult(
            question_id=str(row["question_id"]),
            category=str(row["category"]),
            question=str(row["question"]),
            retrieval_sources=list(row.get("retrieval_sources") or []),
            retrieved=retrieved,
            retrieval_keyword_hits=kw_hits,
            retrieval_keyword_total=kw_total,
            retrieval_source_hits=src_hits,
            retrieval_hit_at_k=bool(hit),
            gold_doc_hit_at_k=gold_doc_hit,
            gold_page_hit_at_k=gold_page_hit,
            gold_doc_rank=gold_rank,
            retrieval_metrics=metrics,
            retrieval_variant=f"{variant}_{eval_label}",
        )
        print(
            f"  eval={eval_label} doc_recall={metrics.get('doc_recall_at_5')} "
            f"page_recall={metrics.get('page_recall_at_5')} "
            f"gold_doc={metrics.get('gold_doc_hit_at_5')} "
            f"unique_docs={metrics.get('unique_doc_count')} "
            f"kw={metrics.get('keyword_coverage', 0):.0%}",
            flush=True,
        )
        pages = [c.page_number for c in retrieved[: min(5, top_k)]]
        print(f"  top pages: {pages}", flush=True)

        if metrics.get("is_failure") or metrics.get("is_ambiguous"):
            failure_rows.append(
                {
                    "variant": variant,
                    "question_id": row["question_id"],
                    "category": row["category"],
                    "question": row["question"],
                    "status": "failure" if metrics.get("is_failure") else "ambiguous",
                    "failure_reasons": metrics.get("failure_reasons", []),
                    "ambiguous_reasons": metrics.get("ambiguous_reasons", []),
                    "metrics": metrics,
                    "top_chunks": [
                        chunk_to_report_dict(c, i)
                        for i, c in enumerate(retrieved[:top_k], start=1)
                    ],
                }
            )

        if not skip_llm and variant == "baseline":
            try:
                answer, provider, model = generate_answer(
                    row,
                    retrieved,
                    provider=llm_provider,
                    model=llm_model,
                    ollama_base=ollama_base,
                )
                vr.answer = answer
                vr.llm_provider = provider
                vr.llm_model = model
            except Exception as exc:
                vr.error = str(exc)

        md_path = output_dir / f"{row['question_id']}{suffix}.md"
        md_path.write_text(format_result_markdown(vr, variant=variant), encoding="utf-8")
        results.append(vr)

    return results, failure_rows


def _write_summary(
    results: list[ValidationResult],
    *,
    output_dir: Path,
    filename: str,
    unified_id: str,
    embed_model: str,
    manifest: dict,
    top_k: int,
    eval_k: int,
    diversity_config: dict | None,
) -> dict:
    agg = _aggregate_from_metrics([r.retrieval_metrics for r in results], eval_k)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "unified_id": unified_id,
        "embedding_model": embed_model,
        "index_chunks": manifest.get("total_indexed"),
        "num_questions": len(results),
        "top_k": top_k,
        "eval_k": eval_k,
        "variant": results[0].retrieval_variant if results else "unknown",
        "diversity_config": diversity_config,
        "aggregate_metrics": agg,
        "results": [
            {
                "question_id": r.question_id,
                "category": r.category,
                "retrieval_metrics": r.retrieval_metrics,
                "unique_pages_in_top_k": unique_page_count(r.retrieved, top_k),
                "pages_in_top_k": [c.page_number for c in r.retrieved],
                "top_doc": r.retrieved[0].file_name if r.retrieved else None,
            }
            for r in results
        ],
    }
    path = output_dir / filename
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Pilot RAG validation (retrieval + diversity rerank).")
    parser.add_argument("--unified", type=str, default="full_corpus_715_v1")
    parser.add_argument("--questions", type=Path, default=Path("data/eval/pilot_validation_questions.jsonl"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--eval-k", type=int, default=5)
    parser.add_argument("--fetch-k", type=int, default=0, help="Over-fetch pool for diversity (0=top_k*5)")
    parser.add_argument("--max-chunks-per-doc", type=int, default=3)
    parser.add_argument("--max-chunks-per-page", type=int, default=1)
    parser.add_argument(
        "--eval-mode",
        choices=("open", "constrained", "both"),
        default="open",
        help="open=full corpus doc_recall; constrained=gold_doc filter page recall",
    )
    parser.add_argument("--max-docs", type=int, default=4)
    parser.add_argument("--no-diversity-rerank", action="store_true", help="Skip reranked pass")
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/index"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/logs/pilot_validation"))
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--llm-provider", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--ollama-base", type=str, default="http://localhost:11434")
    args = parser.parse_args()

    run_diversity = not args.no_diversity_rerank
    fetch_k = args.fetch_k or max(args.top_k * 5, 40)
    diversity_cfg = DiversityConfig(
        max_chunks_per_doc=args.max_chunks_per_doc,
        max_chunks_per_page=args.max_chunks_per_page,
    )
    diversity_cfg_dict = {
        "max_chunks_per_doc": diversity_cfg.max_chunks_per_doc,
        "max_chunks_per_page": diversity_cfg.max_chunks_per_page,
        "mmr_lambda": diversity_cfg.mmr_lambda,
        "enable_mmr": diversity_cfg.enable_mmr,
        "fetch_k": fetch_k,
    }

    if not args.questions.exists():
        raise FileNotFoundError(args.questions)

    default_models = {"ollama": "llama3.1:8b", "openai": "gpt-4o-mini"}
    llm_model = args.llm_model or default_models[args.llm_provider]

    collection, embed_model, manifest = load_unified_collection(args.unified, args.index_dir)
    questions = load_questions(args.questions)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    eval_modes = (
        [False, True] if args.eval_mode == "both" else [args.eval_mode == "constrained"]
    )

    all_baseline: list[ValidationResult] = []
    for constrained in eval_modes:
        label = "constrained" if constrained else "open"
        print(f"\n=== BASELINE retrieval ({label}) ===", flush=True)
        baseline_results, baseline_failures = _run_pass(
            variant="baseline",
            questions=questions,
            collection=collection,
            embed_model=embed_model,
            chunks_dir=args.chunks_dir,
            top_k=args.top_k,
            eval_k=args.eval_k,
            fetch_k=args.top_k,
            diversity_cfg=None,
            output_dir=args.output_dir,
            skip_llm=args.skip_llm,
            llm_provider=args.llm_provider,
            llm_model=llm_model,
            ollama_base=args.ollama_base,
            suffix=f"_{label}" if args.eval_mode == "both" else "",
            eval_constrained_mode=constrained,
            max_docs=args.max_docs,
        )
        all_baseline.extend(baseline_results)
        fname = (
            f"pilot_validation_summary_{label}.json"
            if args.eval_mode == "both"
            else "pilot_validation_summary.json"
        )
        _write_summary(
            baseline_results,
            output_dir=args.output_dir,
            filename=fname,
            unified_id=args.unified,
            embed_model=embed_model,
            manifest=manifest,
            top_k=args.top_k,
            eval_k=args.eval_k,
            diversity_config={"eval_mode": label},
        )
        combined = args.output_dir / (
            f"pilot_validation_answers_{label}.md"
            if args.eval_mode == "both"
            else "pilot_validation_answers.md"
        )
        combined.write_text(
            "\n\n---\n\n".join(
                format_result_markdown(r, variant=r.retrieval_variant) for r in baseline_results
            ),
            encoding="utf-8",
        )
        fail_path = args.output_dir / (
            f"rag_validation_failures_{label}.jsonl"
            if args.eval_mode == "both"
            else "rag_validation_failures.jsonl"
        )
        with fail_path.open("w", encoding="utf-8") as f:
            for row in baseline_failures:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    baseline_results = all_baseline

    reranked_results: list[ValidationResult] = []
    if run_diversity:
        print("\n=== DIVERSITY RERANKED retrieval ===", flush=True)
        reranked_results, reranked_failures = _run_pass(
            variant="reranked",
            questions=questions,
            collection=collection,
            embed_model=embed_model,
            chunks_dir=args.chunks_dir,
            top_k=args.top_k,
            eval_k=args.eval_k,
            fetch_k=fetch_k,
            diversity_cfg=diversity_cfg,
            output_dir=args.output_dir,
            skip_llm=True,
            llm_provider=args.llm_provider,
            llm_model=llm_model,
            ollama_base=args.ollama_base,
            suffix="_reranked",
            eval_constrained_mode=False,
            max_docs=args.max_docs,
        )

        _write_summary(
            reranked_results,
            output_dir=args.output_dir,
            filename="pilot_validation_summary_reranked.json",
            unified_id=args.unified,
            embed_model=embed_model,
            manifest=manifest,
            top_k=args.top_k,
            eval_k=args.eval_k,
            diversity_config=diversity_cfg_dict,
        )
        combined_r = args.output_dir / "pilot_validation_answers_reranked.md"
        combined_r.write_text(
            "\n\n---\n\n".join(format_result_markdown(r, variant="reranked") for r in reranked_results),
            encoding="utf-8",
        )
        with (args.output_dir / "rag_validation_failures_reranked.jsonl").open("w", encoding="utf-8") as f:
            for row in reranked_failures:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        comparison = _build_comparison(baseline_results, reranked_results, eval_k=args.eval_k)
        comparison["generated_at"] = datetime.now(timezone.utc).isoformat()
        comparison["diversity_config"] = diversity_cfg_dict
        comparison_path = args.output_dir / "pilot_validation_diversity_comparison.json"
        comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

        report_lines = [
            "Pilot RAG Diversity Comparison",
            f"Generated: {comparison['generated_at']}",
            f"top_k={args.top_k} fetch_k={fetch_k} max/doc={diversity_cfg.max_chunks_per_doc} max/page={diversity_cfg.max_chunks_per_page}",
            "",
            f"{'Metric':<28} {'Baseline':>10} {'Reranked':>10} {'Delta':>10}",
            "-" * 60,
        ]
        for key in (
            "source_hit_at_5",
            "gold_doc_hit_at_5",
            "gold_page_set_hit_at_5",
            "topic_hit_at_k",
            "mean_keyword_coverage",
            "mean_duplicate_doc_ratio",
            "mean_duplicate_page_ratio",
        ):
            b = comparison["baseline"].get(key, 0)
            r = comparison["reranked"].get(key, 0)
            d = comparison["delta_reranked_minus_baseline"].get(key, 0)
            report_lines.append(f"{key:<28} {b:>10.1%} {r:>10.1%} {d:>+10.1%}")
        report_lines.append("")
        for row in comparison["per_question"]:
            report_lines.append(
                f"{row['question_id']}: pages baseline={row['baseline_pages_top_k']} "
                f"reranked={row['reranked_pages_top_k']} "
                f"(unique {row['baseline_unique_pages']} -> {row['reranked_unique_pages']})"
            )
        (args.output_dir / "pilot_validation_diversity_comparison.txt").write_text(
            "\n".join(report_lines) + "\n",
            encoding="utf-8",
        )

        print("\n=== COMPARISON (reranked - baseline) ===", flush=True)
        for key, d in comparison["delta_reranked_minus_baseline"].items():
            print(f"  {key}: {d:+.1%}", flush=True)
        print(f"Comparison: {comparison_path}", flush=True)

    b_agg = _aggregate_from_metrics([r.retrieval_metrics for r in baseline_results], args.eval_k)
    print("\nBaseline aggregate:", flush=True)
    for key in (
        "source_hit_at_5",
        "gold_doc_hit_at_5",
        "gold_page_set_hit_at_5",
        "topic_hit_at_k",
        "mean_keyword_coverage",
        "mean_duplicate_doc_ratio",
        "mean_duplicate_page_ratio",
    ):
        print(f"  {key}: {b_agg.get(key, 0):.1%}", flush=True)


if __name__ == "__main__":
    main()
