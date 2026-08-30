"""Run Gemma through the v3 RAG set and score evidence-contract behaviors."""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rag_answer_lib import DEFAULT_OLLAMA_BASE, RetrievedChunk, load_unified_collection
from rag_inprocess import DEFAULT_UNIFIED, run_answer_inprocess, run_search_inprocess

DEFAULT_QUESTIONS = ROOT / "data/eval/pilot_validation_text_v3.jsonl"
DEFAULT_OUT = ROOT / "data/processed/logs/text_rag_eval_v3/gemma4_12b_full405"
REJECTION_MARKERS = (
    "확인할 수 없",
    "확인되지 않",
    "근거가 없",
    "제공되지 않",
    "포함되어 있지 않",
    "추가 자료",
    "문서만으로는",
    "단정할 수 없",
    "찾을 수 없",
)
CORRECTION_MARKERS = (
    "전제는 맞지 않습니다",
    "전제가 맞지 않습니다",
    "전제는 틀립니다",
    "전제가 틀립니다",
    "전제는 틀렸",
    "전제가 틀렸",
    "아닙니다",
    "아니며",
    "잘못",
    "틀린",
    "정확하지 않",
    "확정되지 않",
    "발효되지 않",
    "바로잡",
)
NEGATION_MARKERS = REJECTION_MARKERS + CORRECTION_MARKERS + (
    "않습니다",
    "없습니다",
    "아니라",
    "아닌",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣.%]+", "", (text or "").lower())


def alias_hit(answer: str, aliases: list[str]) -> bool:
    normalized = norm(answer)
    return any(norm(alias) and norm(alias) in normalized for alias in aliases)


def forbidden_violations(answer: str, claims: list[str]) -> list[str]:
    violations: list[str] = []
    lowered = answer.lower()
    for claim in claims:
        needle = claim.lower().strip()
        if not needle:
            continue
        start = lowered.find(needle)
        if start < 0:
            continue
        window = lowered[max(0, start - 36) : start + len(needle) + 36]
        if not any(marker in window for marker in NEGATION_MARKERS):
            violations.append(claim)
    return violations


def negative_irrelevance(answer: str, target: str) -> list[str]:
    """Flag factual bullets unrelated to an intentionally absent target."""
    target_terms = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[가-힣]{3,}", target or "")
        if token.lower() not in {"문서", "근거", "정보", "목록", "요청", "승인"}
    }
    output: list[str] = []
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not re.match(r"^[-*]\s+", line):
            continue
        lowered = line.lower()
        if any(marker in line for marker in REJECTION_MARKERS + ("추정하지",)):
            continue
        if target_terms and any(term in lowered for term in target_terms):
            continue
        output.append(line)
    return output


def row_score(row: dict[str, Any], answer: str) -> dict[str, Any]:
    points = row.get("gold_answer_points") or []
    point_hits = [
        point["point_id"]
        for point in points
        if alias_hit(answer, list(point.get("aliases") or []) + [str(point.get("text") or "")])
    ]
    completeness = len(point_hits) / len(points) if points else 0.0
    citations = len(re.findall(r"\[(\d+)\]", answer))
    violations = forbidden_violations(answer, row.get("forbidden_claims") or [])
    rejection = any(marker in answer for marker in REJECTION_MARKERS)
    correction = any(marker in answer for marker in CORRECTION_MARKERS) or bool(
        re.search(r"전제.{0,40}(?:틀렸|틀립|맞지\s*않|옳지\s*않)", answer)
    )
    target = str(row.get("unanswerable_target") or "")
    target_asserted = bool(target and norm(target) in norm(answer) and not rejection)
    irrelevant_bullets = negative_irrelevance(answer, target) if target else []
    required_sections = list((row.get("format_contract") or {}).get("required_sections") or [])
    section_rate = (
        sum(1 for section in required_sections if section in answer) / len(required_sections)
        if required_sections
        else 1.0
    )

    if row["test_type"] == "negative_rejection":
        behavior_pass = rejection and not target_asserted
        quality = (
            0.6 * float(behavior_pass)
            + 0.2 * float(not violations)
            + 0.2 * float(not irrelevant_bullets)
        )
    elif row["test_type"] == "counterfactual_robustness":
        behavior_pass = correction and not violations
        quality = 0.45 * completeness + 0.35 * float(behavior_pass) + 0.2 * float(citations > 0)
    else:
        behavior_pass = not violations
        quality = (
            0.55 * completeness
            + 0.2 * float(citations > 0)
            + 0.15 * float(not violations)
            + 0.1 * section_rate
        )
    return {
        "completeness": round(completeness, 4),
        "point_hits": point_hits,
        "citation_count": citations,
        "forbidden_violations": violations,
        "rejection_detected": rejection,
        "correction_detected": correction,
        "target_asserted": target_asserted,
        "irrelevant_bullets": irrelevant_bullets,
        "irrelevance_clean": not irrelevant_bullets,
        "format_section_rate": round(section_rate, 4),
        "behavior_pass": behavior_pass,
        "quality_score": round(quality, 4),
    }


def citation_page_accuracy(row: dict[str, Any], evidence: list[dict]) -> float | None:
    """Return the fraction of cited evidence rows on accepted gold pages."""
    pages = {int(value) for value in row.get("gold_pages") or [] if value is not None}
    doc_ids = {
        str(value)
        for value in (
            list(row.get("acceptable_doc_ids") or [])
            + list(row.get("gold_doc_ids") or [])
            + [row.get("gold_doc_id")]
        )
        if str(value or "").strip()
    }
    scored: list[bool] = []
    for item in evidence:
        page = item.get("page")
        chunk_id = str(item.get("chunk_id") or "")
        if page is None:
            continue
        correct_doc = not doc_ids or any(
            chunk_id == doc_id or chunk_id.startswith(doc_id + "_")
            for doc_id in doc_ids
        )
        scored.append(bool(correct_doc and (not pages or int(page) in pages)))
    if not scored:
        return None
    return round(statistics.fmean(float(value) for value in scored), 4)


def groundedness_proxy(answer: str, *, contract_valid: bool, forbidden_clean: bool) -> float:
    """Repeatable citation-contract proxy, not an expert entailment label."""
    factual = [
        line.strip()
        for line in answer.splitlines()
        if line.strip().startswith(("-", "*"))
        and "검색 근거에서 확인되지 않" not in line
        and "별도로 식별되지 않" not in line
    ]
    if not factual:
        return float(bool(contract_valid and forbidden_clean))
    cited = statistics.fmean(
        float(bool(re.search(r"\[\d+\]", line))) for line in factual
    )
    return round(cited * float(contract_valid) * float(forbidden_clean), 4)


def injected_chunks(collection, ids: list[str]) -> list[RetrievedChunk]:
    if not ids:
        return []
    raw = collection.get(ids=ids, include=["metadatas", "documents"])
    by_id = {
        chunk_id: (meta or {}, document or "")
        for chunk_id, meta, document in zip(raw["ids"], raw["metadatas"], raw["documents"])
    }
    out: list[RetrievedChunk] = []
    for chunk_id_value in ids:
        if chunk_id_value not in by_id:
            continue
        meta, document = by_id[chunk_id_value]
        out.append(
            RetrievedChunk(
                chunk_id=chunk_id_value,
                doc_id=str(meta.get("doc_id") or ""),
                source=str(meta.get("source") or ""),
                file_name=str(meta.get("file_name") or ""),
                page_number=meta.get("page_number"),
                clause_number=str(meta.get("clause_number") or meta.get("article_number") or ""),
                element_type=str(meta.get("element_type") or ""),
                distance=0.0,
                text=document,
            )
        )
    return out


def retrieve_ids(items: list[Any]) -> list[str]:
    return [str(getattr(item, "chunk_id", "") or "") for item in items]


def _gold_retrieval_score(row: dict[str, Any], search: dict[str, Any], evidence: list[dict]) -> dict[str, Any]:
    """Compute gold-document/page visibility without enabling a gold filter."""
    targets = {
        str(value)
        for value in (
            list(row.get("gold_doc_candidates") or [])
            + list(row.get("acceptable_doc_ids") or [])
            + list(row.get("gold_doc_ids") or [])
            + [row.get("gold_doc_id")]
        )
        if str(value or "").strip()
    }
    retrieved = list(search.get("retrieved") or [])
    pool = list(search.get("retrieval_pool") or [])

    def first_rank(chunks: list[Any]) -> int | None:
        return next(
            (
                index
                for index, chunk in enumerate(chunks, 1)
                if str(getattr(chunk, "doc_id", "") or "") in targets
            ),
            None,
        )

    final_rank = first_rank(retrieved)
    pool_rank = first_rank(pool)
    gold_pages = {int(value) for value in row.get("gold_pages") or [] if value is not None}
    page_hit = any(
        str(getattr(chunk, "doc_id", "") or "") in targets
        and getattr(chunk, "page_number", None) is not None
        and int(getattr(chunk, "page_number")) in gold_pages
        for chunk in retrieved
    ) if gold_pages else None
    gold_file = str(row.get("gold_file_name") or "").strip().lower()
    evidence_files = [str(item.get("file_name") or "") for item in evidence]
    evidence_source_hit = any(name.strip().lower() == gold_file for name in evidence_files) if gold_file else None
    return {
        "gold_final_doc_hit": final_rank is not None,
        "gold_final_doc_rank": final_rank,
        "gold_pool_doc_hit": pool_rank is not None,
        "gold_pool_doc_rank": pool_rank,
        "gold_page_hit": page_hit,
        "gold_evidence_source_hit": evidence_source_hit,
        "retrieved_doc_ids": [str(getattr(chunk, "doc_id", "") or "") for chunk in retrieved],
        "retrieved_file_names": [str(getattr(chunk, "file_name", "") or "") for chunk in retrieved],
        "evidence_file_names": evidence_files,
    }


def run_one(
    row: dict[str, Any],
    *,
    collection,
    embed_model: str,
    manifest: dict,
    llm_model: str,
    latency_mode: str,
    index: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    search_row = dict(row)
    pipeline_latency_mode = latency_mode
    if latency_mode == "advanced":
        search_row["_advanced_mode"] = True
        search_row["_advanced_llm_model"] = llm_model
        pipeline_latency_mode = "accurate"
    search = run_search_inprocess(
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
    chunks = list(search["retrieved"])
    pool = list(search["retrieval_pool"])
    injected: list[RetrievedChunk] = []
    if row["test_type"] == "noise_robustness":
        injected = injected_chunks(collection, list(row.get("hard_negative_chunk_ids") or []))
        chunks = [*injected, *chunks]
        pool = [*injected, *pool]
    answer_started = time.perf_counter()
    answer_row = dict(row)
    if latency_mode == "advanced":
        answer_row["_advanced_mode"] = True
        answer_row["_advanced_llm_model"] = llm_model
        answer_row["_advanced_retrieval_meta"] = search.get("advanced_retrieval") or {}
        answer_row["_advanced_confidence"] = search.get("advanced_confidence") or {}
    answer_row["_text_document_route"] = search.get("text_document_route") or {}
    answer_row["_evidence_completion"] = search.get("evidence_completion") or {}
    answer_out = run_answer_inprocess(
        row=answer_row,
        chunks=chunks,
        pool=pool,
        config_dict=search.get("retrieval_config"),
        metrics=search.get("retrieval_metrics"),
        doc_groups=search.get("doc_groups"),
        answer_mode=search.get("answer_mode", "standard_rag"),
        question_category=search.get("question_category"),
        llm_model=llm_model,
        ollama_base=DEFAULT_OLLAMA_BASE,
        temperature=0.1,
        top_k=10,
        fetch_k=120,
        start_type="warm",
        run_index=index,
        latency_mode=pipeline_latency_mode,
        auto_llm_warm=True,
        skip_ollama_probe=True,
    )
    answer = str(answer_out.get("answer") or "")
    if latency_mode == "advanced":
        from services.rag_service import (
            _advanced_review_answer,
            _advanced_review_evidence,
            _cited_rows,
        )

        review_evidence = _advanced_review_evidence(
            list(answer_out.get("evidence_table") or []),
            list(search.get("retrieved") or []),
        )
        reviewed_answer, advanced_review = _advanced_review_answer(
            str(row.get("question") or ""),
            answer,
            review_evidence,
            model=llm_model,
            confidence=search.get("advanced_confidence") or {},
        )
        if reviewed_answer != answer:
            answer = reviewed_answer
            answer_out["evidence_table"] = _cited_rows(answer, review_evidence)
        else:
            answer = reviewed_answer
    else:
        advanced_review = {}
    scored = row_score(row, answer)
    evidence_table = answer_out.get("evidence_table") or []
    verification_summary = answer_out.get("verification_summary") or {}
    answer_generation = verification_summary.get("answer_generation") or {}
    contract_valid = bool(verification_summary.get("answer_contract_valid"))
    retrieval_scored = _gold_retrieval_score(row, search, evidence_table)
    elapsed = time.perf_counter() - started
    return {
        "question_id": row["question_id"],
        "scenario_id": row["scenario_id"],
        "test_type": row["test_type"],
        "category": row.get("category"),
        "question": row["question"],
        "answerability": row["answerability"],
        "expected_behavior": row["expected_behavior"],
        "answer": answer,
        "advanced_answer_review": advanced_review,
        "answer_chars": len(answer),
        "provider": answer_out.get("provider"),
        "model": answer_out.get("model"),
        "answer_mode": search.get("answer_mode"),
        "advanced_retrieval": search.get("advanced_retrieval") or {},
        "advanced_rerank": search.get("advanced_rerank") or {},
        "advanced_confidence": search.get("advanced_confidence") or {},
        "answer_scope_status": (answer_out.get("verification_summary") or {}).get(
            "answer_scope_status"
        ),
        "answer_format": verification_summary.get("answer_format") or {},
        "answer_contract_valid": contract_valid,
        "answer_contract_warnings": (
            answer_out.get("verification_summary") or {}
        ).get("answer_contract_warnings") or [],
        "feature_fallback_terms": (
            (search.get("retrieval_config") or {}).get("fast_meta") or {}
        ).get("feature_fallback_terms") or [],
        "retrieved_chunk_ids": retrieve_ids(search["retrieved"]),
        "injected_hard_negative_chunk_ids": retrieve_ids(injected),
        "generation_context_chunk_ids": retrieve_ids(chunks),
        "llm_context_chunk_ids": answer_generation.get("llm_context_chunk_ids") or [],
        "priority_local_used": bool(answer_generation.get("priority_local_used")),
        "query_focused_used": bool(answer_generation.get("query_focused_used")),
        "answer_generation": answer_generation,
        "evidence_table": evidence_table,
        "citation_page_accuracy": citation_page_accuracy(row, evidence_table),
        "retrieval_seconds": round(answer_started - started, 4),
        "generation_seconds": round(time.perf_counter() - answer_started, 4),
        "e2e_seconds": round(elapsed, 4),
        "fast_under_10s": elapsed <= 10.0 if latency_mode == "fast" else None,
        **retrieval_scored,
        **scored,
        "groundedness": groundedness_proxy(
            answer,
            contract_valid=contract_valid,
            forbidden_clean=not bool(scored.get("forbidden_violations")),
        ),
        "error": None,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if not record.get("error")]

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"n": 0}
        completeness_items = [
            x for x in items if x.get("test_type") != "negative_rejection"
        ]
        answerable_items = [x for x in items if bool(x.get("answerability"))]
        page_items = [x for x in answerable_items if x.get("gold_page_hit") is not None]
        evidence_source_items = [
            x for x in answerable_items if x.get("gold_evidence_source_hit") is not None
        ]
        fast_items = [x for x in items if x.get("fast_under_10s") is not None]
        citation_page_items = [
            x for x in answerable_items if x.get("citation_page_accuracy") is not None
        ]
        return {
            "n": len(items),
            "mean_completeness": round(
                statistics.fmean(float(x["completeness"]) for x in completeness_items), 4
            ) if completeness_items else 0.0,
            "behavior_pass_rate": round(statistics.fmean(float(x["behavior_pass"]) for x in items), 4),
            "answer_pass_rate": round(statistics.fmean(float(x["behavior_pass"]) for x in items), 4),
            "citation_page_accuracy": round(
                statistics.fmean(float(x["citation_page_accuracy"]) for x in citation_page_items), 4
            ) if citation_page_items else None,
            "groundedness": round(
                statistics.fmean(float(x.get("groundedness") or 0.0) for x in items), 4
            ),
            "citation_rate": round(statistics.fmean(float(x["citation_count"] > 0) for x in items), 4),
            "answerable_citation_rate": round(
                statistics.fmean(float(x["citation_count"] > 0) for x in answerable_items), 4
            ) if answerable_items else None,
            "gold_final_doc_hit_rate": round(
                statistics.fmean(float(x.get("gold_final_doc_hit", False)) for x in answerable_items), 4
            ) if answerable_items else None,
            "gold_pool_doc_hit_rate": round(
                statistics.fmean(float(x.get("gold_pool_doc_hit", False)) for x in answerable_items), 4
            ) if answerable_items else None,
            "gold_page_hit_rate": round(
                statistics.fmean(float(x.get("gold_page_hit", False)) for x in page_items), 4
            ) if page_items else None,
            "gold_evidence_source_hit_rate": round(
                statistics.fmean(float(x.get("gold_evidence_source_hit", False)) for x in evidence_source_items), 4
            ) if evidence_source_items else None,
            "fast_under_10s_rate": round(
                statistics.fmean(float(x.get("fast_under_10s", False)) for x in fast_items), 4
            ) if fast_items else None,
            "forbidden_clean_rate": round(statistics.fmean(float(not x["forbidden_violations"]) for x in items), 4),
            "irrelevance_clean_rate": round(
                statistics.fmean(float(x.get("irrelevance_clean", True)) for x in items), 4
            ),
            "mean_quality_score": round(statistics.fmean(float(x["quality_score"]) for x in items), 4),
            "mean_retrieval_seconds": round(statistics.fmean(float(x["retrieval_seconds"]) for x in items), 4),
            "mean_generation_seconds": round(statistics.fmean(float(x["generation_seconds"]) for x in items), 4),
            "mean_e2e_seconds": round(statistics.fmean(float(x["e2e_seconds"]) for x in items), 4),
            "median_e2e_seconds": round(statistics.median(float(x["e2e_seconds"]) for x in items), 4),
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
        "provider_counts": dict(
            Counter(str(row.get("provider") or "unknown") for row in valid)
        ),
        "mean_answer_chars": (
            round(statistics.fmean(float(row.get("answer_chars") or 0) for row in valid), 1)
            if valid
            else 0.0
        ),
        "by_test_type": {key: summarize(value) for key, value in sorted(by_type.items())},
        "by_scenario": {key: summarize(value) for key, value in sorted(by_scenario.items())},
    }


def report_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    lines = [
        "# TEXT RAG v3 Gemma 답변 평가",
        "",
        f"- 완료: **{summary['completed']}/{summary['total']}** (오류 {summary['errors']})",
        f"- keypoint 완전성: **{overall['mean_completeness']:.1%}**",
        f"- 행동 통과율: **{overall['behavior_pass_rate']:.1%}**",
        f"- 인용 페이지 정확도: **{(overall.get('citation_page_accuracy') or 0):.1%}**",
        f"- Groundedness(자동 계약 proxy): **{overall.get('groundedness', 0):.1%}**",
        f"- 인용 포함률: **{overall['citation_rate']:.1%}**",
        f"- 답변 가능 문항 인용 포함률: **{(overall.get('answerable_citation_rate') or 0):.1%}**",
        f"- 최종 후보 정답 PDF 포함률: **{(overall.get('gold_final_doc_hit_rate') or 0):.1%}**",
        f"- 검색 풀 정답 PDF 포함률: **{(overall.get('gold_pool_doc_hit_rate') or 0):.1%}**",
        f"- 최종 근거 정답 PDF 인용률: **{(overall.get('gold_evidence_source_hit_rate') or 0):.1%}**",
        *(
            [f"- Fast 10초 이내 비율: **{overall.get('fast_under_10s_rate', 0):.1%}**"]
            if overall.get("fast_under_10s_rate") is not None
            else []
        ),
        f"- 금지 주장 미발생률: **{overall['forbidden_clean_rate']:.1%}**",
        f"- 불필요 사실 미발생률: **{overall['irrelevance_clean_rate']:.1%}**",
        f"- 평균 품질점수: **{overall['mean_quality_score']:.1%}**",
        f"- 평균/중앙 E2E: **{overall['mean_e2e_seconds']:.2f}s / {overall['median_e2e_seconds']:.2f}s**",
        "",
        "## 유형별",
        "",
        "| 유형 | n | 완전성 | 행동 통과 | 인용 | 금지주장 없음 | 불필요사실 없음 | 품질 | E2E초 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in summary["by_test_type"].items():
        lines.append(
            f"| {key} | {item['n']} | {item['mean_completeness']:.1%} | "
            f"{item['behavior_pass_rate']:.1%} | {item['citation_rate']:.1%} | "
            f"{item['forbidden_clean_rate']:.1%} | {item['irrelevance_clean_rate']:.1%} | "
            f"{item['mean_quality_score']:.1%} | "
            f"{item['mean_e2e_seconds']:.2f} |"
        )
    lines.extend(
        [
            "",
            "> 이 점수는 v3의 keypoint/금지주장/거절/형식 계약을 이용한 재현 가능한 자동 점검입니다. "
            "ARES의 PPI 보정에 해당하는 최종 수치는 별도 인간 라벨 표본이 있어야 산출할 수 있습니다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--llm-model", default="gemma4:12b")
    parser.add_argument(
        "--latency-mode", choices=("accurate", "fast", "advanced"), default="accurate"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--test-type", action="append", default=[])
    parser.add_argument(
        "--question-id",
        action="append",
        default=[],
        help="Evaluate only the selected question id(s); may be repeated.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("RAG_DEBUG_TRACE_STDERR", "0")
    rows = load_jsonl(args.questions)
    if args.scenario:
        wanted = set(args.scenario)
        rows = [row for row in rows if row["scenario_id"] in wanted]
    if args.test_type:
        wanted_types = set(args.test_type)
        rows = [row for row in rows if row["test_type"] in wanted_types]
    if args.question_id:
        wanted_ids = set(args.question_id)
        rows = [row for row in rows if row["question_id"] in wanted_ids]
    if args.limit is not None:
        rows = rows[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.out_dir / "records.jsonl"
    existing = load_jsonl(records_path) if args.resume and records_path.exists() else []
    done_ids = {str(record.get("question_id")) for record in existing}
    todo = [row for row in rows if row["question_id"] not in done_ids]
    records = list(existing)
    collection, embed_model, manifest = load_unified_collection(
        DEFAULT_UNIFIED, ROOT / "data/processed/index"
    )
    with records_path.open("a" if existing else "w", encoding="utf-8") as stream:
        for local_index, row in enumerate(todo, 1):
            index = len(existing) + local_index
            try:
                record = run_one(
                    row,
                    collection=collection,
                    embed_model=embed_model,
                    manifest=manifest,
                llm_model=args.llm_model,
                latency_mode=args.latency_mode,
                index=index,
                )
            except Exception as exc:
                record = {
                    "question_id": row["question_id"],
                    "scenario_id": row["scenario_id"],
                    "test_type": row["test_type"],
                    "question": row["question"],
                    "error": repr(exc),
                }
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            if index == 1 or index % 10 == 0 or index == len(rows):
                print(
                    f"[{index}/{len(rows)}] {row['question_id']} "
                    f"quality={record.get('quality_score')} error={bool(record.get('error'))}",
                    flush=True,
                )

    summary = aggregate(records)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "report.md").write_text(report_markdown(summary), encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
