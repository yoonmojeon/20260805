#!/usr/bin/env python3
"""Grounded second-pass judge for the 150-PDF generated answers."""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "rag" / "scripts"
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("RAG_DEBUG_TRACE_STDERR", "0")

from rag_answer_lib import DEFAULT_OLLAMA_BASE, call_ollama_chat_timed  # noqa: E402
from services.llm_models import DEFAULT_LLM_MODEL  # noqa: E402


DEFAULT_QUESTIONS = ROOT / "data" / "eval" / "broad_pdf_150_final.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_json(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.I | re.S)
    if match:
        raw = match.group(1)
    else:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    return json.loads(raw)


def judge_prompt(question: dict[str, Any], record: dict[str, Any]) -> str:
    contexts = []
    for item in question.get("gold_evidence") or []:
        context = str(item.get("context") or "")
        if context and context not in contexts:
            contexts.append(context)
    source = "\n\n".join(contexts)[:6200]
    gold = [point.get("text") for point in question.get("gold_answer_points") or []]
    evidence_files = record.get("evidence_file_names") or [
        item.get("file_name") for item in record.get("evidence_table") or []
    ]
    return f"""다음 RAG 답변을 골드 원문에 대조해 엄격히 채점하라.

질문: {question.get('question')}
골드 사실: {json.dumps(gold, ensure_ascii=False)}
골드 문서: {question.get('gold_file_name')}
답변이 실제 인용한 문서: {json.dumps(evidence_files, ensure_ascii=False)}

[골드 원문]
{source}
[/골드 원문]

[실제 답변]
{record.get('answer')}
[/실제 답변]

채점 규칙:
- correctness 4: 핵심 결론·수치·조건·예외가 모두 맞음, 3: 사소한 누락만 있고 핵심은 맞음, 2: 절반 정도만 맞음, 1: 거의 답하지 못함, 0: 틀림.
- completeness 4: 골드 핵심을 모두 다룸, 3: 대부분, 2: 일부, 1: 매우 적음, 0: 없음.
- relevance 2: 질문에 직접 답함, 1: 우회적·장황하지만 일부 답함, 0: 무관함.
- 원문과 모순되거나 근거 없는 구체적 수치·의무를 만들면 contradiction/unsupported_specific_claim을 true로 한다.
- 단순히 인용 번호가 있다는 이유로 점수를 주지 않는다.
- 골드 문서와 다른 문서를 인용했더라도 실제 답이 골드 사실과 일치하면 내용 점수는 줄 수 있다. 문서 귀속 문제는 reason에 명시한다.
- JSON만 출력한다.

{{
  "correctness": 0,
  "completeness": 0,
  "relevance": 0,
  "contradiction": false,
  "unsupported_specific_claim": false,
  "reason": "한두 문장 근거"
}}"""


def valid_score(payload: dict[str, Any]) -> bool:
    try:
        return (
            0 <= int(payload["correctness"]) <= 4
            and 0 <= int(payload["completeness"]) <= 4
            and 0 <= int(payload["relevance"]) <= 2
            and isinstance(payload["contradiction"], bool)
            and isinstance(payload["unsupported_specific_claim"], bool)
        )
    except Exception:
        return False


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("judge_error")]

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"n": 0}
        return {
            "n": len(items),
            "pass_rate": round(statistics.fmean(float(row["judge_pass"]) for row in items), 4),
            "mean_correctness": round(statistics.fmean(float(row["judge_correctness"]) for row in items), 3),
            "mean_completeness": round(statistics.fmean(float(row["judge_completeness"]) for row in items), 3),
            "direct_rate": round(statistics.fmean(float(row["judge_relevance"] == 2) for row in items), 4),
            "contradiction_rate": round(statistics.fmean(float(row["judge_contradiction"]) for row in items), 4),
            "unsupported_rate": round(statistics.fmean(float(row["judge_unsupported_specific_claim"]) for row in items), 4),
            "gold_final_doc_hit_rate": round(statistics.fmean(float(row.get("gold_final_doc_hit", False)) for row in items), 4),
            "gold_evidence_source_hit_rate": round(statistics.fmean(float(row.get("gold_evidence_source_hit", False)) for row in items), 4),
        }

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        by_source[str(row.get("gold_source") or "unknown")].append(row)
        by_mode[str(row.get("answer_mode") or "unknown")].append(row)
    return {
        "total": len(rows),
        "judged": len(valid),
        "errors": len(rows) - len(valid),
        "overall": summarize(valid),
        "by_source": {key: summarize(value) for key, value in sorted(by_source.items())},
        "by_answer_mode": {key: summarize(value) for key, value in sorted(by_mode.items())},
        "failure_reasons": dict(Counter(row.get("judge_reason") for row in valid if not row["judge_pass"])),
    }


def report_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    lines = [
        "# 150-PDF 실제 답변 근거 판정",
        "",
        f"- 판정 완료: **{summary['judged']}/{summary['total']}**",
        f"- 답변 통과율: **{overall.get('pass_rate', 0):.1%}**",
        f"- 평균 정확성: **{overall.get('mean_correctness', 0):.2f}/4**",
        f"- 평균 완전성: **{overall.get('mean_completeness', 0):.2f}/4**",
        f"- 질문 직접 응답률: **{overall.get('direct_rate', 0):.1%}**",
        f"- 모순률: **{overall.get('contradiction_rate', 0):.1%}**",
        f"- 근거 없는 구체 주장률: **{overall.get('unsupported_rate', 0):.1%}**",
        "",
        "## 기관별",
        "",
        "| 기관 | n | 통과 | 정확성/4 | 완전성/4 | 직접응답 | 정답PDF 최종 | 정답PDF 인용 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source, item in summary["by_source"].items():
        lines.append(
            f"| {source} | {item['n']} | {item['pass_rate']:.1%} | {item['mean_correctness']:.2f} | "
            f"{item['mean_completeness']:.2f} | {item['direct_rate']:.1%} | "
            f"{item['gold_final_doc_hit_rate']:.1%} | {item['gold_evidence_source_hit_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "> 로컬 모델을 이용한 출처 기반 내부 판정이며 실제 해사 전문가의 공식 서명 검수는 아닙니다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    questions = {row["question_id"]: row for row in load_jsonl(args.questions)}
    source_records = load_jsonl(args.records)
    if args.limit is not None:
        source_records = source_records[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / "judged_records.jsonl"
    existing = load_jsonl(output_path) if args.resume and output_path.exists() else []
    done = {row["question_id"] for row in existing}
    output = list(existing)
    with output_path.open("a" if existing else "w", encoding="utf-8") as stream:
        for local_index, record in enumerate(source_records, 1):
            question_id = str(record.get("question_id") or "")
            if question_id in done:
                continue
            question = questions[question_id]
            payload: dict[str, Any] | None = None
            error = ""
            for attempt in range(2):
                try:
                    raw = call_ollama_chat_timed(
                        args.model,
                        "당신은 해사 RAG 답변의 엄격한 출처 기반 평가자다. 골드 원문과 실제 답변만 비교하고 JSON만 출력한다.",
                        judge_prompt(question, record) + ("\n출력 스키마를 지켜 다시 채점하라." if attempt else ""),
                        DEFAULT_OLLAMA_BASE,
                        temperature=0.0,
                        num_predict=260,
                        num_ctx=8192,
                    )
                    candidate = extract_json(raw)
                    if valid_score(candidate):
                        payload = candidate
                        break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            joined = dict(record)
            joined["gold_source"] = question.get("gold_source")
            joined["gold_file_name"] = question.get("gold_file_name")
            joined["judge_error"] = error if payload is None else ""
            if payload is not None:
                joined.update(
                    {
                        "judge_correctness": int(payload["correctness"]),
                        "judge_completeness": int(payload["completeness"]),
                        "judge_relevance": int(payload["relevance"]),
                        "judge_contradiction": bool(payload["contradiction"]),
                        "judge_unsupported_specific_claim": bool(payload["unsupported_specific_claim"]),
                        "judge_reason": str(payload.get("reason") or ""),
                    }
                )
                joined["judge_pass"] = bool(
                    joined["judge_correctness"] >= 3
                    and joined["judge_completeness"] >= 3
                    and joined["judge_relevance"] >= 1
                    and not joined["judge_contradiction"]
                    and not joined["judge_unsupported_specific_claim"]
                )
            else:
                joined["judge_pass"] = False
            output.append(joined)
            stream.write(json.dumps(joined, ensure_ascii=False) + "\n")
            stream.flush()
            if local_index == 1 or local_index % 10 == 0 or local_index == len(source_records):
                print(
                    f"[{local_index}/{len(source_records)}] {question_id} pass={joined['judge_pass']} "
                    f"correct={joined.get('judge_correctness')} error={bool(joined['judge_error'])}",
                    flush=True,
                )

    summary = aggregate(output)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "report.md").write_text(report_markdown(summary), encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
