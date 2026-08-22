#!/usr/bin/env python3
"""Second-pass review and repair for the broad PDF evaluation set."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
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


DEFAULT_INPUT = ROOT / "data" / "eval" / "broad_pdf_150.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "eval" / "broad_pdf_150_reviewed.jsonl"
DEFAULT_REPORT = ROOT / "data" / "processed" / "logs" / "broad_pdf_150_review.json"
AMBIGUOUS_RE = re.compile(
    r"^(?:이|해당|그)\s*(?:프로그램|문서|서비스|규정|지침)|"
    r"이 프로그램|해당 서비스 문서|이 클래스 프로그램",
    re.I,
)


def _norm(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣.%]+", "", (text or "").lower())


def _extract_json(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.I | re.S)
    if match:
        raw = match.group(1)
    else:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    return json.loads(raw)


def _facts_payload(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"text": point.get("text"), "aliases": point.get("aliases") or []}
        for point in row.get("gold_answer_points") or []
    ]


def _valid_revision(payload: dict[str, Any], context: str) -> bool:
    question = str(payload.get("question") or "").strip()
    facts = payload.get("gold_facts")
    if not (18 <= len(question) <= 220) or not re.search(r"[가-힣]", question):
        return False
    if AMBIGUOUS_RE.search(question) or re.search(r"위 문서|제공된 근거|본문에 따르면", question):
        return False
    if not isinstance(facts, list) or not (1 <= len(facts) <= 3):
        return False
    context_norm = _norm(context)
    for fact in facts:
        if len(str((fact or {}).get("text") or "").strip()) < 12:
            return False
        aliases = [str(value).strip() for value in (fact or {}).get("aliases") or []]
        grounded = [alias for alias in aliases if len(_norm(alias)) >= 2 and _norm(alias) in context_norm]
        if not grounded:
            return False
        fact["aliases"] = list(dict.fromkeys(grounded))[:4]
    return True


def _review_prompt(row: dict[str, Any], context: str) -> str:
    force = bool(AMBIGUOUS_RE.search(str(row.get("question") or "")))
    return f"""다음 RAG 평가 문항과 골드 사실을 원문 근거에 대조해 엄격히 검수하라.

기관: {row.get('gold_source')}
문서: {row.get('gold_file_name')}
질문: {row.get('question')}
골드 사실: {json.dumps(_facts_payload(row), ensure_ascii=False)}
자동 모호성 경고: {force}

[원문 근거]
{context}
[/원문 근거]

통과 조건:
- 질문이 문맥 없이도 독립적으로 이해된다.
- 질문 하나의 초점이 명확하고 원문만으로 답할 수 있다.
- 골드 사실의 상태·수치·예외가 원문과 일치한다.
- aliases는 원문에 실제 등장하는 짧은 고유 용어·코드·수치다.
- 질문에 정답 결론이나 수치를 미리 노출하지 않는다.

문제가 있으면 question과 gold_facts를 같은 원문 안에서 직접 수정하라. JSON만 출력한다.
{{
  "pass": true 또는 false,
  "issues": ["문제 설명"],
  "question": "통과 가능한 최종 한국어 질문",
  "gold_facts": [{{"text": "원문 기반 사실", "aliases": ["원문 앵커"]}}]
}}"""


def _apply_revision(row: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    updated["question"] = str(payload["question"]).strip()
    chunk_ids = list(row.get("gold_chunk_ids") or [])
    facts = []
    for index, fact in enumerate(payload["gold_facts"], 1):
        facts.append(
            {
                "point_id": f"{row['question_id']}-P{index}",
                "text": str(fact["text"]).strip(),
                "aliases": list(fact["aliases"]),
                "evidence_chunk_ids": chunk_ids,
            }
        )
    updated["gold_answer_points"] = facts
    updated["must_cover"] = [fact["text"] for fact in facts]
    updated["gold_answer"] = "\n".join(f"- {fact['text']}" for fact in facts)
    return updated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]
    output: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        context = str(((row.get("gold_evidence") or [{}])[0]).get("context") or "")
        payload: dict[str, Any] | None = None
        error = ""
        try:
            raw = call_ollama_chat_timed(
                args.model,
                "당신은 해사 문서 평가셋의 독립 검수자다. 원문 충실성과 질문 독립성을 엄격히 판단하고 JSON만 출력한다.",
                _review_prompt(row, context),
                DEFAULT_OLLAMA_BASE,
                temperature=0.0,
                num_predict=500,
                num_ctx=6144,
            )
            payload = _extract_json(raw)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        valid = bool(payload and _valid_revision(payload, context))
        changed = False
        reviewed = row
        if valid and payload:
            proposed_question = str(payload.get("question") or "").strip()
            proposed_facts = payload.get("gold_facts") or []
            changed = proposed_question != row.get("question") or proposed_facts != _facts_payload(row)
            reviewed = _apply_revision(row, payload)
        issues = list((payload or {}).get("issues") or [])
        unresolved = not valid
        reviewed["review_meta"] = {
            "model": args.model,
            "review_pass": bool((payload or {}).get("pass")) and valid,
            "valid_contract": valid,
            "changed": changed,
            "issues": issues,
            "error": error,
            "unresolved": unresolved,
        }
        output.append(reviewed)
        audit_rows.append(
            {
                "question_id": row["question_id"],
                "pass": reviewed["review_meta"]["review_pass"],
                "changed": changed,
                "unresolved": unresolved,
                "issues": issues,
                "before": row["question"],
                "after": reviewed["question"],
                "error": error,
            }
        )
        print(
            f"[{index:03d}/{len(rows):03d}] {row['question_id']} "
            f"pass={reviewed['review_meta']['review_pass']} changed={changed} unresolved={unresolved}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n",
        encoding="utf-8",
    )
    summary = {
        "rows": len(output),
        "distinct_docs": len({row.get("gold_doc_id") for row in output}),
        "review_pass": sum(bool(row["review_meta"]["review_pass"]) for row in output),
        "changed": sum(bool(row["review_meta"]["changed"]) for row in output),
        "unresolved": sum(bool(row["review_meta"]["unresolved"]) for row in output),
        "by_source": dict(Counter(str(row.get("gold_source")) for row in output)),
        "records": audit_rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, ensure_ascii=False, indent=2))
    return 0 if summary["unresolved"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
