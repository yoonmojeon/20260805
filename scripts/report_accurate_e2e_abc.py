"""Publish the six official metrics from the fixed 150 A/B/C E2E runs."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data/eval/accurate_eval_150.jsonl"
RUN_ROOT = ROOT / "data/processed/logs/accurate_abc"
OUT_JSON = ROOT / "reports/accurate_abc_official_20260821.json"
OUT_MD = ROOT / "reports/accurate_abc_official_20260821.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def targets(row: dict[str, Any]) -> set[str]:
    return {
        str(value)
        for value in (
            list(row.get("gold_doc_candidates") or [])
            + list(row.get("acceptable_doc_ids") or [])
            + list(row.get("gold_doc_ids") or [])
            + [row.get("gold_doc_id")]
        )
        if str(value or "").strip()
    }


def document_hit(row: dict[str, Any], record: dict[str, Any]) -> bool:
    expected = targets(row)
    return any(
        chunk_id == doc_id or chunk_id.startswith(doc_id + "_")
        for chunk_id in record.get("retrieved_chunk_ids") or []
        for doc_id in expected
    )


def evidence_recall(row: dict[str, Any], record: dict[str, Any]) -> float:
    retrieved = set(record.get("retrieved_chunk_ids") or [])
    points = list(row.get("gold_answer_points") or [])
    if not points:
        return 0.0
    hits = 0
    for point in points:
        evidence = set(point.get("evidence_chunk_ids") or [])
        if evidence.intersection(retrieved):
            hits += 1
    return hits / len(points)


def percentile(values: list[float], value: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * value)))
    return ordered[index]


def compute(
    questions: dict[str, dict[str, Any]], records: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    valid = [record for record in records.values() if not record.get("error")]
    answerable = [record for record in valid if questions[record["question_id"]].get("answerability")]
    page = [record for record in answerable if record.get("citation_page_accuracy") is not None]
    e2e = [float(record.get("e2e_seconds") or 0.0) for record in valid]
    return {
        "Document Hit@10": statistics.fmean(
            float(document_hit(questions[record["question_id"]], record))
            for record in answerable
        ),
        "Evidence Recall@10": statistics.fmean(
            evidence_recall(questions[record["question_id"]], record)
            for record in answerable
        ),
        "Citation Page Accuracy": (
            statistics.fmean(float(record["citation_page_accuracy"]) for record in page)
            if page else 0.0
        ),
        "Answer Pass Rate": statistics.fmean(float(record.get("behavior_pass", False)) for record in valid),
        "Groundedness": statistics.fmean(float(record.get("groundedness") or 0.0) for record in valid),
        "Avg E2E Latency": statistics.fmean(e2e),
        "debug": {
            "completed": len(valid),
            "errors": len(records) - len(valid),
            "p95_e2e_seconds": percentile(e2e, 0.95),
            "mean_completeness": statistics.fmean(float(record.get("completeness") or 0.0) for record in answerable),
        },
    }


def write_regressions(
    left: str,
    right: str,
    questions: dict[str, dict[str, Any]],
    records: dict[str, dict[str, dict[str, Any]]],
) -> Path:
    path = ROOT / f"reports/regressions_{right}_vs_{left}.csv"
    fields = [
        "question_id", "question", "expected_document", "expected_evidence",
        f"{left}_doc_hit", f"{right}_doc_hit", f"{left}_pass", f"{right}_pass",
        f"{left}_quality", f"{right}_quality", "estimated_cause",
    ]
    rows: list[dict[str, Any]] = []
    for qid, old in records[left].items():
        row = questions[qid]
        new = records[right][qid]
        old_doc = document_hit(row, old) if row.get("answerability") else None
        new_doc = document_hit(row, new) if row.get("answerability") else None
        quality_drop = float(old.get("quality_score") or 0) - float(new.get("quality_score") or 0)
        regressed = (
            (old_doc is True and new_doc is False)
            or (bool(old.get("behavior_pass")) and not bool(new.get("behavior_pass")))
            or quality_drop >= 0.1
        )
        if not regressed:
            continue
        if old_doc is True and new_doc is False:
            cause = "retrieval_document_ranking"
        elif quality_drop >= 0.1 and new_doc:
            cause = "evidence_selection_or_generation"
        else:
            cause = "generation_or_citation"
        rows.append(
            {
                "question_id": qid,
                "question": row.get("question"),
                "expected_document": " | ".join(sorted(targets(row))),
                "expected_evidence": " | ".join(row.get("gold_chunk_ids") or []),
                f"{left}_doc_hit": old_doc,
                f"{right}_doc_hit": new_doc,
                f"{left}_pass": old.get("behavior_pass"),
                f"{right}_pass": new.get("behavior_pass"),
                f"{left}_quality": old.get("quality_score"),
                f"{right}_quality": new.get("quality_score"),
                "estimated_cause": cause,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    question_rows = read_jsonl(QUESTIONS)
    questions = {row["question_id"]: row for row in question_rows}
    records = {
        variant: {
            row["question_id"]: row
            for row in read_jsonl(RUN_ROOT / f"fixed150_{variant}_answers_dynamic" / "records.jsonl")
        }
        for variant in "ABC"
    }
    metrics = {variant: compute(questions, records[variant]) for variant in "ABC"}
    regression_paths = [
        write_regressions("A", "B", questions, records),
        write_regressions("A", "C", questions, records),
    ]
    result = {
        "schema_version": "maritime-accurate-official-six-v1",
        "question_set": str(QUESTIONS.relative_to(ROOT)),
        "sample_size": len(questions),
        "sample_seed": 260821,
        "generation": {"model": "gemma4:12b", "temperature": 0.1, "top_k_context": 10},
        "metrics": metrics,
        "decision": {
            "selected": "A Legacy Accurate",
            "A": "GO/default",
            "B": "NO-GO",
            "C": "NO-GO",
            "reason": "B/C did not improve final answer quality and added latency; document hit also regressed.",
        },
        "regression_files": [str(path.relative_to(ROOT)) for path in regression_paths],
        "groundedness_note": (
            "Automated citation-contract proxy: factual bullet citation coverage, "
            "answer-contract validity and forbidden-claim cleanliness; expert review still required."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    official = [
        "Document Hit@10", "Evidence Recall@10", "Citation Page Accuracy",
        "Answer Pass Rate", "Groundedness", "Avg E2E Latency",
    ]
    lines = [
        "# Accurate A/B/C 공식 6개 지표",
        "",
        "고정 150문항, 동일 Gemma 4 12B·temperature 0.1·Top-10 생성 컨텍스트 기준입니다.",
        "",
        "| Metric | A Legacy | B Hybrid RRF | C Hybrid RRF + Reranker |",
        "|---|---:|---:|---:|",
    ]
    for key in official:
        values = []
        for variant in "ABC":
            value = metrics[variant][key]
            values.append(f"{value:.2f}s" if key == "Avg E2E Latency" else f"{value:.1%}")
        lines.append(f"| {key} | {values[0]} | {values[1]} | {values[2]} |")
    lines.extend(
        [
            "",
            "## 판정",
            "",
            "- A: **GO / 기본 유지**",
            "- B: **NO-GO** — 근거 recall debug는 상승했지만 최종 문서 hit·완전성은 하락하고 E2E가 증가했습니다.",
            "- C: **NO-GO** — B 대비 공식 품질 이득 없이 재랭커 비용만 추가됐습니다.",
            "",
            "> Groundedness는 citation contract 기반 자동 proxy이며 전문가 entailment 판정이 아닙니다.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(OUT_MD)


if __name__ == "__main__":
    main()
