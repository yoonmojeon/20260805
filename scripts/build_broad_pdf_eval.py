#!/usr/bin/env python3
"""Build a grounded, stratified 150-PDF RAG evaluation set.

Each case is based on a different indexed PDF.  The local LLM writes a natural
Korean question and a small evidence contract from one information-dense chunk.
The gold document is *not* passed to retrieval as a filter; it is only used by
the evaluator after retrieval has completed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
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

from rag_answer_lib import (  # noqa: E402
    DEFAULT_OLLAMA_BASE,
    call_ollama_chat_timed,
)
from rag_resource_cache import load_unified_collection  # noqa: E402
from services.llm_models import DEFAULT_LLM_MODEL  # noqa: E402


DEFAULT_OUTPUT = ROOT / "data" / "eval" / "broad_pdf_150.jsonl"
DEFAULT_AUDIT = ROOT / "data" / "processed" / "logs" / "broad_pdf_150_build.json"
MANIFEST = ROOT / "data" / "manifests" / "full_corpus_715.csv"
INDEX_DIR = ROOT / "data" / "processed" / "index"
COLLECTION = "full_corpus_715_v1"

# Slightly over-sample the small societies so they are visible in the audit.
SOURCE_QUOTAS = {
    "DNV": 60,
    "MEPC": 40,
    "MSC": 22,
    "KR": 18,
    "ABS": 9,
    "LR": 1,
}

BOILERPLATE_RE = re.compile(
    r"table of contents|contents\s*$|copyright|all rights reserved|"
    r"blank page|intentionally left blank|amendments and corrections|"
    r"bibliography|foreword\s*$|개정사항|목\s*차",
    re.I,
)
RULE_TERMS_RE = re.compile(
    r"\b(?:shall|must|should|required|requirement|is to|are to|may not|"
    r"approved|adopted|agreed|decided|prohibited|permitted|unless|except)\b|"
    r"하여야|해야|요건|예외|금지|허용|승인|채택|합의",
    re.I,
)
DISTINCTIVE_RE = re.compile(
    r"\b[A-Z]{2,}(?:[-./][A-Z0-9]+)+\b|\b\d+(?:\.\d+){1,3}\b|"
    r"\b(?:MARPOL|SOLAS|SEEMP|CII|EEXI|GHG|IGF|IGC|MASS|AROS|LEL)\b",
    re.I,
)


def _norm(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣.%]+", "", (text or "").lower())


def _stable_number(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def _strip_index_prefix(text: str) -> str:
    lines = (text or "").splitlines()
    if lines and re.match(r"^\[[^]]+\]\s+file=", lines[0], re.I):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _chunk_score(text: str, meta: dict[str, Any]) -> float:
    clean = _strip_index_prefix(text)
    length = len(clean)
    if length < 220 or length > 4200:
        return -10_000.0
    if BOILERPLATE_RE.search(clean[:700]):
        return -1_000.0
    alpha_tokens = re.findall(r"[A-Za-z가-힣]{3,}", clean)
    if len(alpha_tokens) < 35:
        return -1_000.0
    score = min(length, 1800) / 180.0
    score += 5.0 if RULE_TERMS_RE.search(clean) else 0.0
    score += min(4, len(DISTINCTIVE_RE.findall(clean))) * 0.8
    score += 2.0 if str(meta.get("clause_number") or meta.get("article_number") or "") else 0.0
    score += 1.0 if int(meta.get("page_number") or 0) >= 3 else 0.0
    # Contents pages often contain many short dotted entries.
    score -= min(5.0, clean.count("...") * 0.8)
    score -= 3.0 if clean.count("\n") > 45 else 0.0
    return score


def _document_context(collection, doc_id: str) -> dict[str, Any] | None:
    raw = collection.get(
        where={"doc_id": doc_id},
        include=["documents", "metadatas"],
    )
    rows: list[tuple[str, str, dict[str, Any], float]] = []
    for chunk_id, text, meta in zip(
        raw.get("ids") or [], raw.get("documents") or [], raw.get("metadatas") or []
    ):
        score = _chunk_score(str(text or ""), dict(meta or {}))
        if score > 0:
            rows.append((str(chunk_id), str(text or ""), dict(meta or {}), score))
    if not rows:
        return None
    rows.sort(key=lambda item: (-item[3], _stable_number(item[0])))
    best_id, best_text, best_meta, best_score = rows[0]
    page = int(best_meta.get("page_number") or 0)
    companions = [
        item
        for item in rows[1:]
        if abs(int(item[2].get("page_number") or 0) - page) <= 1
        and item[0] != best_id
    ]
    selected = [(best_id, best_text, best_meta, best_score)]
    if companions and len(_strip_index_prefix(best_text)) < 1300:
        selected.append(companions[0])
    context_parts = [_strip_index_prefix(item[1]) for item in selected]
    context = "\n\n".join(context_parts)[:5200]
    return {
        "context": context,
        "chunk_ids": [item[0] for item in selected],
        "pages": sorted(
            {int(item[2].get("page_number") or 0) for item in selected if item[2].get("page_number") is not None}
        ),
        "score": round(best_score, 3),
        "file_name": str(best_meta.get("file_name") or ""),
    }


def _extract_json(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.I | re.S)
    if fenced:
        raw = fenced.group(1)
    else:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    return json.loads(raw)


def _style(source: str, doc_id: str) -> str:
    value = _stable_number(doc_id) % 10
    if source in {"MEPC", "MSC"}:
        return "document_scoped" if value >= 7 else "meeting_scoped"
    if value < 5:
        return "topic_open"
    if value < 8:
        return "society_scoped"
    return "document_scoped"


def _prompt(*, source: str, file_name: str, style: str, page: int | None, context: str) -> str:
    style_instruction = {
        "topic_open": (
            "기관명이나 파일명을 질문에 쓰지 말고, 근거의 특징적인 기술 주제와 요건만으로 묻는다. "
            "단, 정답 수치나 결론을 질문에 미리 넣지 않는다."
        ),
        "society_scoped": (
            f"질문에 {source}를 자연스럽게 언급하되 전체 파일명은 쓰지 않는다. "
            "기술 주제·적용범위·요건 또는 예외를 묻는다."
        ),
        "document_scoped": (
            "사용자가 특정 문서나 문서 코드를 알고 찾는 상황으로 질문한다. 파일명에서 식별 가능한 "
            "짧은 문서 코드·회의 문서번호만 사용할 수 있지만 정답 문장 자체는 노출하지 않는다."
        ),
        "meeting_scoped": (
            f"질문에 {source} 회의자료라는 범위는 드러내되, 문서의 결론·수치·상태를 질문에 미리 답하지 않는다."
        ),
    }[style]
    return f"""다음 해사 문서 근거 하나로 RAG 품질평가 문항을 작성하라.

기관: {source}
파일명: {file_name}
페이지: {page if page is not None else '미상'}
질문 스타일: {style}
스타일 지시: {style_instruction}

[근거]
{context}
[/근거]

규칙:
1. 실제 사용자가 한국어로 물을 법한 단일 질문을 만든다.
2. 질문은 이 근거만으로 답할 수 있어야 하며, 여러 문서 비교나 최신 전체 동향을 요구하지 않는다.
3. 요구사항, 적용범위, 예외, 결정상태, 일정, 정의, 설계·운항 영향 중 근거가 가장 명확한 유형을 택한다.
4. 제안(proposed), 합의(agreed), 승인(approved), 채택(adopted)을 서로 바꾸지 않는다.
5. gold_facts는 근거가 직접 뒷받침하는 핵심 사실 1~3개만 쓴다.
6. 각 aliases에는 답변에서 확인하기 쉬우며 근거 원문에도 실제 등장하는 짧은 문자열 2~4개를 넣는다. 숫자·규정코드·고유 기술용어를 우선한다.
7. '위 문서', '제공된 근거', '본문에 따르면' 같은 평가용 표현을 질문에 쓰지 않는다.
8. JSON 외에는 출력하지 않는다.

출력 형식:
{{
  "question": "한국어 질문",
  "question_type": "requirement_exception|scope_definition|decision_status|timeline|design_operation|document_lookup 중 하나",
  "gold_facts": [
    {{"text": "근거 기반 정답 사실", "aliases": ["원문 앵커1", "원문 앵커2"]}}
  ],
  "forbidden_claims": ["근거와 반대되는 대표적 과도한 단정이 있을 때만 기입"]
}}"""


def _validate_generated(payload: dict[str, Any], context: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    question = str(payload.get("question") or "").strip()
    if not (18 <= len(question) <= 220) or not re.search(r"[가-힣]", question):
        errors.append("bad_question")
    if re.search(r"위 문서|제공된 근거|본문에 따르면", question):
        errors.append("evaluation_leak")
    facts = payload.get("gold_facts")
    if not isinstance(facts, list) or not (1 <= len(facts) <= 3):
        errors.append("bad_fact_count")
        return False, errors
    context_norm = _norm(context)
    for index, fact in enumerate(facts):
        if not isinstance(fact, dict) or len(str(fact.get("text") or "").strip()) < 12:
            errors.append(f"bad_fact_{index}")
            continue
        aliases = [str(value).strip() for value in fact.get("aliases") or [] if str(value).strip()]
        grounded = [alias for alias in aliases if len(_norm(alias)) >= 2 and _norm(alias) in context_norm]
        if not grounded:
            errors.append(f"ungrounded_alias_{index}")
        fact["aliases"] = list(dict.fromkeys(grounded))[:4]
    return not errors, errors


def _load_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _allocate_for_limit(limit: int | None) -> dict[str, int]:
    if limit is None or limit >= sum(SOURCE_QUOTAS.values()):
        return dict(SOURCE_QUOTAS)
    raw = {source: limit * quota / 150 for source, quota in SOURCE_QUOTAS.items()}
    allocated = {source: int(value) for source, value in raw.items()}
    for source in ("ABS", "LR"):
        if limit >= 10 and allocated[source] == 0:
            allocated[source] = 1
    while sum(allocated.values()) > limit:
        source = max(allocated, key=lambda key: allocated[key] - raw[key])
        allocated[source] -= 1
    while sum(allocated.values()) < limit:
        source = max(allocated, key=lambda key: raw[key] - allocated[key])
        allocated[source] += 1
    return allocated


def _existing_rows(path: Path, resume: bool) -> list[dict[str, Any]]:
    if not resume or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--seed", type=int, default=715150)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    quotas = _allocate_for_limit(args.limit)
    manifest = _load_manifest()
    index_manifest = json.loads(
        (INDEX_DIR / f"unified_{COLLECTION}" / "index_manifest.json").read_text(encoding="utf-8")
    )
    indexed = set(index_manifest.get("doc_ids") or [])
    by_source: dict[str, list[dict[str, str]]] = {source: [] for source in quotas}
    for row in manifest:
        source = str(row.get("source") or "").upper()
        if source in by_source and str(row.get("doc_id") or "") in indexed:
            by_source[source].append(row)
    rng = random.Random(args.seed)
    for source, rows in by_source.items():
        rng.shuffle(rows)
        rows.sort(key=lambda row: _stable_number(f"{args.seed}:{row['doc_id']}"))

    existing = _existing_rows(args.output, args.resume)
    done_docs = {str(row.get("gold_doc_id") or "") for row in existing}
    counts = Counter(str(row.get("gold_source") or "") for row in existing)
    records = list(existing)
    collection, _, _ = load_unified_collection(COLLECTION, INDEX_DIR)
    build_failures: list[dict[str, Any]] = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if existing else "w"
    with args.output.open(mode, encoding="utf-8") as stream:
        for source, quota in quotas.items():
            needed = max(0, quota - counts[source])
            if needed == 0:
                continue
            for doc in by_source[source]:
                if counts[source] >= quota:
                    break
                doc_id = str(doc["doc_id"])
                if doc_id in done_docs:
                    continue
                context_info = _document_context(collection, doc_id)
                if not context_info:
                    build_failures.append({"doc_id": doc_id, "reason": "no_dense_chunk"})
                    continue
                style = _style(source, doc_id)
                user_prompt = _prompt(
                    source=source,
                    file_name=str(doc.get("file_name") or context_info["file_name"]),
                    style=style,
                    page=(context_info["pages"] or [None])[0],
                    context=context_info["context"],
                )
                generated: dict[str, Any] | None = None
                last_errors: list[str] = []
                for attempt in range(2):
                    try:
                        raw = call_ollama_chat_timed(
                            args.model,
                            "당신은 해사 규정 RAG 평가셋 작성자다. 원문에 없는 사실을 만들지 말고 JSON만 출력한다.",
                            user_prompt + ("\n이전 출력이 유효하지 않았다. JSON과 원문 앵커를 다시 점검하라." if attempt else ""),
                            DEFAULT_OLLAMA_BASE,
                            temperature=0.05,
                            num_predict=520,
                            num_ctx=6144,
                        )
                        candidate = _extract_json(raw)
                        valid, last_errors = _validate_generated(candidate, context_info["context"])
                        if valid:
                            generated = candidate
                            break
                    except Exception as exc:
                        last_errors = [f"{type(exc).__name__}: {exc}"]
                if generated is None:
                    build_failures.append(
                        {"doc_id": doc_id, "file_name": doc.get("file_name"), "reason": last_errors}
                    )
                    continue

                seq = counts[source] + 1
                facts = []
                for fact_index, fact in enumerate(generated["gold_facts"], 1):
                    facts.append(
                        {
                            "point_id": f"BPDF-{source}-{seq:03d}-P{fact_index}",
                            "text": str(fact["text"]).strip(),
                            "aliases": list(fact["aliases"]),
                            "evidence_chunk_ids": list(context_info["chunk_ids"]),
                        }
                    )
                question_type = str(generated.get("question_type") or "evidence_precision")
                record = {
                    "schema_version": "broad-pdf-eval-v1",
                    "question_id": f"BPDF-{source}-{seq:03d}",
                    "scenario_id": doc_id,
                    "parent_id": doc_id,
                    "category": "",
                    "evaluation_context": str(doc.get("file_name") or context_info["file_name"]),
                    "test_type": question_type,
                    "augment_type": style,
                    "retrieval_difficulty": "hard" if style == "topic_open" else "medium",
                    "question": str(generated["question"]).strip(),
                    "answerability": True,
                    "expected_behavior": "answer_from_evidence",
                    "gold_answer": "\n".join(f"- {fact['text']}" for fact in facts),
                    "gold_answer_points": facts,
                    "must_cover": [fact["text"] for fact in facts],
                    "gold_evidence": [
                        {
                            "doc_id": doc_id,
                            "file_name": str(doc.get("file_name") or context_info["file_name"]),
                            "page": page,
                            "chunk_ids": list(context_info["chunk_ids"]),
                            "context": context_info["context"],
                        }
                        for page in (context_info["pages"] or [None])
                    ],
                    "acceptable_doc_ids": [doc_id],
                    "gold_doc_candidates": [doc_id],
                    "gold_source": source,
                    "gold_file_name": str(doc.get("file_name") or context_info["file_name"]),
                    "gold_doc_id": doc_id,
                    "gold_doc_ids": [doc_id],
                    "gold_chunk_ids": list(context_info["chunk_ids"]),
                    "gold_pages": list(context_info["pages"]),
                    "gold_page": (context_info["pages"] or [None])[0],
                    "forbidden_claims": list(generated.get("forbidden_claims") or []),
                    "forbid_claims": list(generated.get("forbidden_claims") or []),
                    # Deliberately empty: evaluation retrieval must not use the gold source/doc as a filter.
                    "source_constraints": {"required": [], "excluded": [], "only": []},
                    "retrieval_target_required": True,
                    "format_contract": {"required_sections": [], "citation_required": True},
                    "required_sections": [],
                    "quality_gate": {
                        "answerable_from_corpus": True,
                        "single_interpretation": True,
                        "evidence_contract_complete": True,
                        "passed": True,
                    },
                    "build_meta": {
                        "style": style,
                        "chunk_score": context_info["score"],
                        "model": args.model,
                    },
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                records.append(record)
                done_docs.add(doc_id)
                counts[source] += 1
                print(
                    f"[{len(records):03d}/{sum(quotas.values()):03d}] {record['question_id']} "
                    f"style={style} page={record['gold_page']} {record['question']}",
                    flush=True,
                )
            if counts[source] < quota:
                raise RuntimeError(f"insufficient valid documents for {source}: {counts[source]}/{quota}")

    audit = {
        "target": sum(quotas.values()),
        "rows": len(records),
        "quotas": quotas,
        "source_counts": dict(counts),
        "distinct_docs": len({row["gold_doc_id"] for row in records}),
        "question_types": dict(Counter(row["test_type"] for row in records)),
        "styles": dict(Counter(row["augment_type"] for row in records)),
        "build_failures": build_failures,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit | {"build_failures": len(build_failures)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
