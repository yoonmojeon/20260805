"""Run a balanced augmented-question sample through the document UI path.

The document tab ultimately calls ``services.orchestrator.handle_question``
with a forced RAG route, LLM routing enabled, the selected latency mode, and
the selected answer model.  This audit uses those same arguments and records
both the generated answer and the gold-evidence checks.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAG_SCRIPTS = ROOT / "rag" / "scripts"
for value in (ROOT, RAG_SCRIPTS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from eval_text_answers_v3 import row_score  # noqa: E402
from services.orchestrator import handle_question  # noqa: E402
from services.rag_service import warmup_rag_resources  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def balanced_sample(
    rows: list[dict[str, Any]], *, sample_size: int, seed: int
) -> list[dict[str, Any]]:
    """Balance five UI categories and ten augmentation types.

    For the canonical 405-row suite and a 40-row sample this produces eight
    questions per category and four per augmentation type.  Source selection
    inside mixed categories prefers the source used least often so MEPC/MSC
    and ABS/DNV/LR remain represented.
    """

    rng = random.Random(seed)
    categories = sorted({str(row.get("category") or "") for row in rows})
    if not categories or sample_size % len(categories):
        raise ValueError("sample_size must divide evenly across categories")
    per_category = sample_size // len(categories)
    test_types = sorted({str(row.get("test_type") or "") for row in rows})
    global_type_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []

    for category_index, category in enumerate(categories):
        category_rows = [row for row in rows if row.get("category") == category]
        available_types = sorted({str(row.get("test_type") or "") for row in category_rows})
        tie_order = list(available_types)
        rng.shuffle(tie_order)
        tie_rank = {value: index for index, value in enumerate(tie_order)}
        chosen_types = sorted(
            available_types,
            key=lambda value: (global_type_counts[value], tie_rank[value]),
        )[:per_category]
        source_counts: Counter[str] = Counter()
        for test_type in chosen_types:
            candidates = [
                row for row in category_rows if row.get("test_type") == test_type
            ]
            rng.shuffle(candidates)
            candidates.sort(
                key=lambda row: source_counts[str(row.get("gold_source") or "")]
            )
            chosen = candidates[0]
            selected.append(chosen)
            global_type_counts[test_type] += 1
            source_counts[str(chosen.get("gold_source") or "")] += 1

    if len(selected) != sample_size:
        raise RuntimeError(f"Expected {sample_size} selected rows, got {len(selected)}")
    rng.shuffle(selected)
    return selected


def evidence_score(
    row: dict[str, Any], evidence: list[dict[str, Any]]
) -> tuple[bool, bool]:
    targets = {
        str(value)
        for value in (
            list(row.get("acceptable_doc_ids") or [])
            + list(row.get("gold_doc_ids") or [])
            + [row.get("gold_doc_id")]
        )
        if str(value or "").strip()
    }
    gold_pages = {
        int(value) for value in row.get("gold_pages") or [] if value is not None
    }
    source_hit = False
    page_hit = False
    for item in evidence:
        chunk_id = str(item.get("chunk_id") or "")
        hit = any(chunk_id.startswith(target) for target in targets)
        if not hit:
            continue
        source_hit = True
        try:
            page_hit = page_hit or not gold_pages or int(item.get("page")) in gold_pages
        except (TypeError, ValueError):
            pass
    return source_hit, page_hit


def run_one(
    row: dict[str, Any], *, model: str, latency_mode: str
) -> dict[str, Any]:
    started = time.perf_counter()
    result = handle_question(
        str(row.get("question") or ""),
        history=[],
        force_route="rag",
        use_llm_router=True,
        rag_latency_mode=latency_mode,
        dialogue_state={},
        llm_model=model,
    )
    elapsed = time.perf_counter() - started
    answer = str(result.get("answer") or "")
    meta = dict(result.get("meta") or {})
    route = dict(result.get("route") or {})
    evidence = list(result.get("evidence_table") or [])
    automatic = row_score(row, answer)
    source_hit, page_hit = evidence_score(row, evidence)
    route_ok = str(route.get("route") or "") == "rag"
    mode_ok = str(meta.get("latency_mode") or "") == latency_mode
    requested_model_ok = str(meta.get("llm_model") or "") == model
    answer_model = str(meta.get("answer_model") or "")
    answer_model_ok = not answer_model or answer_model == model
    answerable = bool(row.get("answerability"))
    if answerable:
        evidence_ok = source_hit and page_hit and automatic["citation_count"] > 0
    else:
        evidence_ok = bool(automatic["behavior_pass"])
    quality_pass = (
        route_ok
        and mode_ok
        and requested_model_ok
        and answer_model_ok
        and bool(automatic["behavior_pass"])
        and float(automatic["quality_score"]) >= 0.8
        and evidence_ok
    )
    strict_pass = quality_pass and (
        not answerable or float(automatic["completeness"]) == 1.0
    )
    return {
        "question_id": row.get("question_id"),
        "scenario_id": row.get("scenario_id"),
        "category": row.get("category"),
        "test_type": row.get("test_type"),
        "gold_source": row.get("gold_source"),
        "answerability": answerable,
        "question": row.get("question"),
        "gold_answer": row.get("gold_answer"),
        "answer": answer,
        "evidence_table": evidence,
        "route": route.get("route"),
        "retrieval_mode": meta.get("retrieval_mode"),
        "answer_mode": meta.get("answer_mode"),
        "latency_mode": meta.get("latency_mode"),
        "llm_model": meta.get("llm_model"),
        "answer_model": meta.get("answer_model"),
        "elapsed_seconds": round(elapsed, 3),
        "gold_source_hit": source_hit,
        "gold_page_hit": page_hit,
        "quality_pass": quality_pass,
        "strict_pass": strict_pass,
        **automatic,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in records if row["answerability"]]
    return {
        "n": len(records),
        "quality_pass": sum(bool(row["quality_pass"]) for row in records),
        "quality_pass_rate": round(
            sum(bool(row["quality_pass"]) for row in records) / len(records), 4
        ),
        "strict_pass": sum(bool(row["strict_pass"]) for row in records),
        "strict_pass_rate": round(
            sum(bool(row["strict_pass"]) for row in records) / len(records), 4
        ),
        "mean_quality_score": round(
            statistics.fmean(float(row["quality_score"]) for row in records), 4
        ),
        "mean_completeness": round(
            statistics.fmean(float(row["completeness"]) for row in answerable), 4
        ),
        "behavior_pass_rate": round(
            statistics.fmean(float(row["behavior_pass"]) for row in records), 4
        ),
        "citation_rate": round(
            statistics.fmean(float(row["citation_count"] > 0) for row in records), 4
        ),
        "gold_source_hit_rate": round(
            statistics.fmean(float(row["gold_source_hit"]) for row in answerable), 4
        ),
        "gold_page_hit_rate": round(
            statistics.fmean(float(row["gold_page_hit"]) for row in answerable), 4
        ),
        "mean_seconds": round(
            statistics.fmean(float(row["elapsed_seconds"]) for row in records), 3
        ),
        "median_seconds": round(
            statistics.median(float(row["elapsed_seconds"]) for row in records), 3
        ),
        "route_ok": sum(row["route"] == "rag" for row in records),
        "mode_ok": sum(row["latency_mode"] == "accurate" for row in records),
        "model_ok": sum(row["llm_model"] == "gemma4:12b" for row in records),
        "by_category": {
            key: {
                "n": len(items),
                "quality_pass": sum(bool(row["quality_pass"]) for row in items),
                "mean_quality": round(
                    statistics.fmean(float(row["quality_score"]) for row in items), 4
                ),
            }
            for key, items in sorted(_group(records, "category").items())
        },
        "by_test_type": {
            key: {
                "n": len(items),
                "quality_pass": sum(bool(row["quality_pass"]) for row in items),
                "mean_quality": round(
                    statistics.fmean(float(row["quality_score"]) for row in items), 4
                ),
            }
            for key, items in sorted(_group(records, "test_type").items())
        },
        "failed_question_ids": [
            row["question_id"] for row in records if not row["quality_pass"]
        ],
    }


def _group(
    rows: list[dict[str, Any]], field: str
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field) or "")].append(row)
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data/eval/pilot_validation_text_v3.jsonl",
    )
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--model", default="gemma4:12b")
    parser.add_argument("--latency-mode", choices=("fast", "accurate"), default="accurate")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "data/processed/logs/augmented_405/sample40_ui_accurate_gemma_20260819",
    )
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.questions)
    selected = balanced_sample(rows, sample_size=args.sample_size, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "selected_questions.jsonl", selected)
    if not args.skip_warmup:
        warmup_rag_resources()

    records: list[dict[str, Any]] = []
    partial = args.out_dir / "records.partial.jsonl"
    write_jsonl(partial, [])
    for index, row in enumerate(selected, 1):
        record = run_one(row, model=args.model, latency_mode=args.latency_mode)
        records.append(record)
        with partial.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[{index}/{len(selected)}] {record['question_id']} "
            f"{'PASS' if record['quality_pass'] else 'FAIL'} "
            f"quality={record['quality_score']:.2f} "
            f"t={record['elapsed_seconds']:.1f}s",
            flush=True,
        )

    write_jsonl(args.out_dir / "records.jsonl", records)
    summary = aggregate(records)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
