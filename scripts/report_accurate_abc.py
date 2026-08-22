"""Compare fixed Accurate retrieval A/B/C runs and publish a regression report."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data/eval/accurate_eval_150.jsonl"
RUN_ROOT = ROOT / "data/processed/logs/accurate_abc"
REPORT_JSON = ROOT / "reports/accurate_retrieval_abc_20260821.json"
REPORT_MD = ROOT / "reports/accurate_retrieval_abc_20260821.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def main() -> None:
    questions = {row["question_id"]: row for row in read_jsonl(QUESTIONS)}
    records = {
        variant: {row["question_id"]: row for row in read_jsonl(RUN_ROOT / f"fixed150_{variant}" / "records.jsonl")}
        for variant in "ABC"
    }
    summaries = {
        variant: json.loads((RUN_ROOT / f"fixed150_{variant}" / "summary.json").read_text(encoding="utf-8"))["overall"]
        for variant in "ABC"
    }
    by_category: dict[str, dict[str, dict[str, float | int | None]]] = defaultdict(dict)
    for variant in "ABC":
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for qid, record in records[variant].items():
            grouped[str(questions[qid].get("category") or "unknown")].append(record)
        for category, items in grouped.items():
            answerable = [item for item in items if item.get("answerability")]
            by_category[category][variant] = {
                "n": len(items),
                "document_hit_at_10": (
                    statistics.fmean(float(bool(item.get("final_doc_any"))) for item in answerable)
                    if answerable else None
                ),
                "evidence_recall_at_10": (
                    statistics.fmean(float(item.get("final_point_recall") or 0.0) for item in answerable)
                    if answerable else None
                ),
            }

    def transitions(left: str, right: str) -> tuple[list[str], list[str]]:
        gains: list[str] = []
        regressions: list[str] = []
        for qid, old in records[left].items():
            if not old.get("answerability"):
                continue
            new = records[right][qid]
            if not old.get("final_doc_any") and new.get("final_doc_any"):
                gains.append(qid)
            elif old.get("final_doc_any") and not new.get("final_doc_any"):
                regressions.append(qid)
        return gains, regressions

    b_gains, b_regressions = transitions("A", "B")
    c_gains, c_regressions = transitions("A", "C")
    result = {
        "schema_version": "maritime-accurate-retrieval-abc-v1",
        "question_set": str(QUESTIONS.relative_to(ROOT)),
        "question_count": len(questions),
        "seed": 260821,
        "variants": {
            "A": "legacy Accurate dense/hierarchical/exact path",
            "B": "A + FTS5 sparse top60 + dense top60 + RRF(k=60) top50",
            "C": "B + local multilingual semantic reranker top14",
        },
        "overall": {
            variant: {
                "document_hit_at_10": summaries[variant]["final_doc_any_rate"],
                "evidence_recall_at_10": summaries[variant]["final_point_recall"],
                "semantic_evidence_recall_at_10": summaries[variant]["final_semantic_point_recall"],
                "avg_retrieval_seconds": summaries[variant]["mean_retrieval_seconds"],
            }
            for variant in "ABC"
        },
        "by_category": by_category,
        "document_hit_transitions": {
            "A_to_B_gains": b_gains,
            "A_to_B_regressions": b_regressions,
            "A_to_C_gains": c_gains,
            "A_to_C_regressions": c_regressions,
        },
        "decision": {
            "A": "GO/default",
            "B": "MIXED/feature flag only",
            "C": "NO-GO",
            "selected_default": "A",
            "reason": (
                "B improved evidence recall but slightly regressed document hit; "
                "C matched B quality with higher latency. Legacy remains default."
            ),
        },
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Accurate Retrieval A/B/C 평가",
        "",
        "- 평가셋: 기존 405문항에서 seed 260821로 고정 추출한 150문항(질문 수정·생성 없음)",
        "- 공식 retrieval 범위: 실제 생성 컨텍스트 Top-10",
        "",
        "| 변형 | Document Hit@10 | Evidence Recall@10 | 의미동등 Recall@10 | 평균 검색 | 판정 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    decisions = {"A": "GO/default", "B": "MIXED", "C": "NO-GO"}
    for variant in "ABC":
        item = result["overall"][variant]
        lines.append(
            f"| {variant} | {pct(item['document_hit_at_10'])} | "
            f"{pct(item['evidence_recall_at_10'])} | "
            f"{pct(item['semantic_evidence_recall_at_10'])} | "
            f"{item['avg_retrieval_seconds']:.2f}s | {decisions[variant]} |"
        )
    lines.extend(["", "## 카테고리별", "", "| 카테고리 | 변형 | n | 문서 Hit@10 | 근거 Recall@10 |", "|---|---|---:|---:|---:|"])
    for category in sorted(by_category):
        for variant in "ABC":
            item = by_category[category][variant]
            lines.append(
                f"| {category} | {variant} | {item['n']} | "
                f"{pct(item['document_hit_at_10'])} | {pct(item['evidence_recall_at_10'])} |"
            )
    lines.extend(
        [
            "",
            "## 회귀 판정",
            "",
            f"- A→B 신규 문서 성공: {len(b_gains)}개 — {', '.join(b_gains) or '없음'}",
            f"- A→B 문서 회귀: {len(b_regressions)}개 — {', '.join(b_regressions) or '없음'}",
            f"- A→C 신규 문서 성공: {len(c_gains)}개 — {', '.join(c_gains) or '없음'}",
            f"- A→C 문서 회귀: {len(c_regressions)}개 — {', '.join(c_regressions) or '없음'}",
            "",
            "## 결론",
            "",
            "B는 희소 용어·정확 식별자 근거 Recall을 높였지만 문서 Hit 회귀가 있어 플래그 뒤에 유지합니다. "
            "C는 B 대비 품질 이득 없이 지연만 늘어 사용하지 않습니다. 기본 UI는 기존 A로 유지합니다.",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))
    print(REPORT_MD)


if __name__ == "__main__":
    main()
