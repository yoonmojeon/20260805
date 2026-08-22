"""Evaluate v3 retrieval at candidate, final-document, and evidence-chunk levels."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rag_answer_lib import load_unified_collection
from rag_inprocess import DEFAULT_UNIFIED, run_search_inprocess

DEFAULT_QUESTIONS = ROOT / "data/eval/pilot_validation_text_v3.jsonl"
DEFAULT_OUT = ROOT / "data/processed/logs/text_rag_eval_v3/retrieval_current"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def chunk_id(item: Any) -> str:
    return str(item.get("chunk_id") if isinstance(item, dict) else getattr(item, "chunk_id", ""))


def doc_id(item: Any) -> str:
    return str(item.get("doc_id") if isinstance(item, dict) else getattr(item, "doc_id", ""))


def chunk_text(item: Any) -> str:
    return str(item.get("text") if isinstance(item, dict) else getattr(item, "text", ""))


def normalized(text: str) -> str:
    import re

    return re.sub(r"[^0-9a-z\uac00-\ud7a3%]+", "", (text or "").lower())


def point_recall(row: dict[str, Any], retrieved_ids: set[str]) -> float:
    points = row.get("gold_answer_points") or []
    if not points:
        return 0.0
    hits = 0
    for point in points:
        evidence_ids = set(point.get("evidence_chunk_ids") or [])
        if evidence_ids.intersection(retrieved_ids):
            hits += 1
    return hits / len(points)


def semantic_point_recall(row: dict[str, Any], items: list[Any]) -> float:
    """Credit equivalent positive chunks, while retaining exact anchors.

    MIRAGE-style exact chunk ids remain the strict metric.  This companion
    metric prevents a later session report paragraph containing the same fact
    from being scored as a miss merely because the catalog anchored an earlier
    working-paper paragraph.
    """
    points = row.get("gold_answer_points") or []
    if not points:
        return 0.0
    hits = 0
    for point in points:
        evidence_ids = set(point.get("evidence_chunk_ids") or [])
        allowed_docs = {
            value.rsplit("_p", 1)[0]
            for value in evidence_ids
            if "_p" in value
        }
        allowed_docs.update(str(value) for value in (row.get("acceptable_doc_ids") or []))
        aliases = [
            normalized(value)
            for value in (point.get("aliases") or [])
            if len(normalized(value)) >= 3
        ]
        matched = False
        for item in items:
            cid = chunk_id(item)
            if cid in evidence_ids:
                matched = True
                break
            if allowed_docs and doc_id(item) not in allowed_docs:
                continue
            body = normalized(chunk_text(item))
            alias_hits = sum(1 for alias in aliases if alias in body)
            threshold = 1 if len(aliases) <= 1 else 2
            if alias_hits >= threshold:
                matched = True
                break
        hits += int(matched)
    return hits / len(points)


def evaluate_one(row: dict[str, Any], output: dict[str, Any], elapsed: float) -> dict[str, Any]:
    pool = output.get("retrieval_pool") or []
    search_final = output.get("retrieved") or []
    # Measure the exact slot-first evidence set passed to answer generation,
    # not merely the intermediate search result.  The production answer path
    # performs this same selection from retrieval_config.fast_meta.
    from evidence_selection import select_planned_evidence

    eval_row = dict(row)
    completion = output.get("evidence_completion") or (
        (output.get("retrieval_config") or {})
        .get("fast_meta", {})
        .get("evidence_completion")
    )
    if completion:
        eval_row["_evidence_completion"] = completion
    final, selection_meta = select_planned_evidence(
        eval_row, search_final, pool, max_chunks=12
    )
    pool_chunk_ids = [chunk_id(item) for item in pool]
    final_chunk_ids = [chunk_id(item) for item in final]
    pool_doc_ids = [doc_id(item) for item in pool]
    final_doc_ids = [doc_id(item) for item in final]
    gold_chunks = set(row.get("gold_chunk_ids") or [])
    gold_docs = set(row.get("gold_doc_ids") or [])
    acceptable_docs = set(row.get("acceptable_doc_ids") or gold_docs)
    hard_negatives = set(row.get("hard_negative_chunk_ids") or [])
    answerable = bool(row.get("answerability"))

    def recall(gold: set[str], found: list[str]) -> float | None:
        if not gold:
            return None
        return len(gold.intersection(found)) / len(gold)

    first_gold_rank = next(
        (index for index, value in enumerate(pool_doc_ids, 1) if value in gold_docs),
        None,
    )
    first_hard_negative_rank = next(
        (index for index, value in enumerate(pool_chunk_ids, 1) if value in hard_negatives),
        None,
    )
    timing = output.get("timing_metrics") or {}
    return {
        "question_id": row["question_id"],
        "scenario_id": row["scenario_id"],
        "test_type": row["test_type"],
        "question": row["question"],
        "answerability": answerable,
        "answer_mode": output.get("answer_mode"),
        "candidate_doc_any": bool(acceptable_docs.intersection(pool_doc_ids)) if answerable else None,
        "candidate_doc_all": gold_docs.issubset(set(pool_doc_ids)) if answerable else None,
        "final_doc_any": bool(acceptable_docs.intersection(final_doc_ids)) if answerable else None,
        "final_doc_all": gold_docs.issubset(set(final_doc_ids)) if answerable else None,
        "candidate_chunk_recall": recall(gold_chunks, pool_chunk_ids) if answerable else None,
        "final_chunk_recall": recall(gold_chunks, final_chunk_ids) if answerable else None,
        "candidate_point_recall": point_recall(row, set(pool_chunk_ids)) if answerable else None,
        "final_point_recall": point_recall(row, set(final_chunk_ids)) if answerable else None,
        "candidate_semantic_point_recall": semantic_point_recall(row, pool) if answerable else None,
        "final_semantic_point_recall": semantic_point_recall(row, final) if answerable else None,
        "first_gold_doc_rank": first_gold_rank,
        "candidate_hard_negative_hits": len(hard_negatives.intersection(pool_chunk_ids)),
        "final_hard_negative_hits": len(hard_negatives.intersection(final_chunk_ids)),
        "first_hard_negative_rank": first_hard_negative_rank,
        "pool_doc_ids": list(dict.fromkeys(pool_doc_ids)),
        "final_doc_ids": list(dict.fromkeys(final_doc_ids)),
        "pool_chunk_ids": pool_chunk_ids,
        "final_chunk_ids": final_chunk_ids,
        "final_selection_meta": selection_meta,
        "retrieval_seconds": float(timing.get("retrieval_time") or elapsed),
        "wall_seconds": elapsed,
        "error": None,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if not record.get("error")]
    answerable = [record for record in valid if record["answerability"]]

    def mean(field: str, rows: list[dict[str, Any]] = answerable) -> float | None:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        return round(statistics.fmean(values), 4) if values else None

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        answer_rows = [row for row in rows if row["answerability"]]
        times = [float(row["retrieval_seconds"]) for row in rows]
        return {
            "n": len(rows),
            "answerable_n": len(answer_rows),
            "candidate_doc_any_rate": mean("candidate_doc_any", answer_rows),
            "candidate_doc_all_rate": mean("candidate_doc_all", answer_rows),
            "final_doc_any_rate": mean("final_doc_any", answer_rows),
            "final_doc_all_rate": mean("final_doc_all", answer_rows),
            "candidate_chunk_recall": mean("candidate_chunk_recall", answer_rows),
            "final_chunk_recall": mean("final_chunk_recall", answer_rows),
            "candidate_point_recall": mean("candidate_point_recall", answer_rows),
            "final_point_recall": mean("final_point_recall", answer_rows),
            "candidate_semantic_point_recall": mean("candidate_semantic_point_recall", answer_rows),
            "final_semantic_point_recall": mean("final_semantic_point_recall", answer_rows),
            "mean_retrieval_seconds": round(statistics.fmean(times), 4) if times else None,
            "median_retrieval_seconds": round(statistics.median(times), 4) if times else None,
        }

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in valid:
        by_type[record["test_type"]].append(record)
        by_scenario[record["scenario_id"]].append(record)
    return {
        "total": len(records),
        "completed": len(valid),
        "errors": len(records) - len(valid),
        "overall": summarize(valid),
        "by_test_type": {key: summarize(rows) for key, rows in sorted(by_type.items())},
        "by_scenario": {key: summarize(rows) for key, rows in sorted(by_scenario.items())},
    }


def report_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    lines = [
        "# TEXT RAG v3 검색 평가",
        "",
        f"- 완료: **{summary['completed']}/{summary['total']}** (오류 {summary['errors']})",
        f"- 후보군 gold 문서 포함(any): **{overall['candidate_doc_any_rate']:.1%}**",
        f"- 최종 컨텍스트 gold 문서 포함(any): **{overall['final_doc_any_rate']:.1%}**",
        f"- 후보군 근거 포인트 Recall: **{overall['candidate_point_recall']:.1%}**",
        f"- 최종 컨텍스트 근거 포인트 Recall: **{overall['final_point_recall']:.1%}**",
        f"- 후보군 의미동등 포인트 Recall: **{overall['candidate_semantic_point_recall']:.1%}**",
        f"- 최종 컨텍스트 의미동등 포인트 Recall: **{overall['final_semantic_point_recall']:.1%}**",
        f"- 평균/중앙 검색시간: **{overall['mean_retrieval_seconds']:.2f}s / {overall['median_retrieval_seconds']:.2f}s**",
        "",
        "## 유형별",
        "",
        "| 유형 | n | 후보 문서 any | 최종 문서 any | 후보 point recall | 최종 point recall | 평균초 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in summary["by_test_type"].items():
        def pct(value: float | None) -> str:
            return "-" if value is None else f"{value:.1%}"
        lines.append(
            f"| {key} | {item['n']} | {pct(item['candidate_doc_any_rate'])} | "
            f"{pct(item['final_doc_any_rate'])} | {pct(item['candidate_point_recall'])} | "
            f"{pct(item['final_point_recall'])} | {item['mean_retrieval_seconds']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 해석 원칙",
            "",
            "- `candidate` 실패는 검색/라우팅 문제입니다.",
            "- candidate에는 있으나 `final`에서 빠지면 다양성·압축·rerank 문제입니다.",
            "- 문서는 맞지만 point recall이 낮으면 문서 내부 청크 검색 문제입니다.",
            "- negative rejection 45개는 정답 청크가 없는 설계이므로 검색 Recall에서 제외합니다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--test-type", action="append", default=[])
    parser.add_argument(
        "--latency-mode", choices=("accurate", "fast", "advanced"), default="accurate"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("RAG_DEBUG_TRACE_STDERR", "0")
    rows = load_jsonl(args.questions)
    if args.scenario:
        wanted = set(args.scenario)
        rows = [row for row in rows if row.get("scenario_id") in wanted]
    if args.test_type:
        wanted_types = set(args.test_type)
        rows = [row for row in rows if row.get("test_type") in wanted_types]
    if args.limit is not None:
        rows = rows[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.out_dir / "records.jsonl"
    existing = load_jsonl(records_path) if args.resume and records_path.exists() else []
    done_ids = {str(record.get("question_id")) for record in existing}
    todo = [row for row in rows if str(row["question_id"]) not in done_ids]
    collection, embed_model, manifest = load_unified_collection(
        DEFAULT_UNIFIED, ROOT / "data/processed/index"
    )
    records: list[dict[str, Any]] = list(existing)
    open_mode = "a" if existing else "w"
    with records_path.open(open_mode, encoding="utf-8") as stream:
        for local_index, row in enumerate(todo, 1):
            index = len(existing) + local_index
            started = time.perf_counter()
            try:
                search_row = dict(row)
                pipeline_latency_mode = args.latency_mode
                if args.latency_mode == "advanced":
                    # The UI maps Advanced onto the established Accurate
                    # retrieval route and adds the local planning/rerank loop.
                    search_row["_advanced_mode"] = True
                    search_row["_advanced_llm_model"] = "gemma4:12b"
                    pipeline_latency_mode = "accurate"
                output = run_search_inprocess(
                    search_row,
                    collection=collection,
                    embed_model=embed_model,
                    manifest=manifest,
                    index_dir=ROOT / "data/processed/index",
                    chunks_dir=ROOT / "data/processed/chunks",
                    latency_mode=pipeline_latency_mode,
                    start_type="warm",
                    run_index=index,
                )
                record = evaluate_one(row, output, time.perf_counter() - started)
            except Exception as exc:
                record = {
                    "question_id": row["question_id"],
                    "scenario_id": row["scenario_id"],
                    "test_type": row["test_type"],
                    "question": row["question"],
                    "answerability": row["answerability"],
                    "error": repr(exc),
                }
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            if index == 1 or index % 20 == 0 or index == len(rows):
                print(f"[{index}/{len(rows)}] {row['question_id']} error={bool(record.get('error'))}", flush=True)

    summary = aggregate(records)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "report.md").write_text(report_markdown(summary), encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
