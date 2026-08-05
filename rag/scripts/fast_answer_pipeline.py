"""Fast answer pipeline — evidence-first with scoped meeting_summary routing."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from fast_claim_extraction import (
    extract_claim_candidates,
    filter_claims_for_answer,
    select_claims_for_question,
)
from fast_document_scope import DocumentScopeResult, classify_document_scope
from fast_imo_terms import IMO_GLOSSARY_PROMPT, detect_terminology_violations, normalize_imo_terms
from fast_question_classifier import classify_fast_question_type
from meeting_outcome_answer import session_title_from_question
from meeting_summary_context import (
    MeetingSummaryContext,
    build_claim_based_summary_answer,
    get_summary_context,
    validate_meeting_summary_answer,
)
from rag_answer_lib import RetrievedChunk

MARITIME_FAST_SYSTEM = """너는 IMO/해사 도메인 문서를 기반으로 답변하는 MaritimeRAG Assistant다.

**필수 절차 (한 번의 응답 안에서 수행):**
1) 검색 근거의 제목·문서번호·agenda item·summary/action requested를 확인해 문서 성격과 적용 범위를 판단한다.
2) 사용자 질문 표현이 문서 실제 범위보다 넓으면 **답변 첫 문장**에서 범위를 정정한다.
3) 근거에서 핵심 claim을 추출하고, evidence가 충분한 claim만 사용한다.
4) 최종 답변은 검색 근거에만 기반하며 각 항목 끝에 근거 page 또는 [N]을 표시한다.

**금지:**
- 근거가 회의 자체 최종 결과가 아닌데 "○○ 회의의 주요 결과는…"라고 단정하지 말 것
- 문서에 없는 내용 추측 금지
- resolution을 "결정"으로 번역 금지

{glossary}
"""


@dataclass
class FastAnswerPipelineContext:
    fast_type: str
    scope: DocumentScopeResult
    summary_ctx: MeetingSummaryContext | None = None
    all_claims: list[dict[str, Any]] = field(default_factory=list)
    filtered_claims: list[dict[str, Any]] = field(default_factory=list)
    selected_claims: list[dict[str, Any]] = field(default_factory=list)
    use_evidence_first: bool = False
    item_count: int = 3
    validation_result: dict[str, Any] = field(default_factory=dict)
    fallback_used: bool = False
    selected_primary_doc: str | None = None
    rejected_docs: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "fast_type": self.fast_type,
            "use_evidence_first": self.use_evidence_first,
            "item_count": self.item_count,
            "document_scope": self.scope.to_dict(),
            "all_claims": self.all_claims,
            "filtered_claims": self.filtered_claims,
            "selected_claims": self.selected_claims,
            "validation_result": self.validation_result,
            "fallback_used": self.fallback_used,
            "selected_primary_doc": self.selected_primary_doc,
            "rejected_docs": self.rejected_docs,
        }
        if self.summary_ctx:
            out["meeting_summary_context"] = self.summary_ctx.to_dict()
        return out


def _uses_evidence_first(fast_type: str) -> bool:
    return fast_type in {"meeting_summary", "meeting_outcome_question", "broad_summary_question"}


def _primary_doc_from_chunks(chunks: list[RetrievedChunk], ctx: MeetingSummaryContext | None) -> str | None:
    from meeting_summary_context import meeting_summary_source_tier

    if not chunks:
        return None
    ranked = sorted(
        chunks,
        key=lambda c: meeting_summary_source_tier(
            c.file_name or "", doc_id=c.doc_id or "", ctx=ctx
        ),
    )
    return ranked[0].file_name or ranked[0].doc_id


def _collect_rejected_docs(pool: list[RetrievedChunk], selected: list[RetrievedChunk], ctx: MeetingSummaryContext | None) -> list[dict[str, str]]:
    from meeting_summary_context import is_penalty_summary_source

    if not ctx or not ctx.apply_reference_penalties:
        return []
    selected_ids = {c.chunk_id for c in selected}
    rejected: list[dict[str, str]] = []
    for c in pool:
        if c.chunk_id in selected_ids:
            continue
        if is_penalty_summary_source(c.file_name or "", c.doc_id or "", ctx=ctx):
            rejected.append(
                {
                    "file_name": c.file_name or "",
                    "doc_id": c.doc_id or "",
                    "reason": "reference_outcome_penalty",
                }
            )
    return rejected[:8]


def prepare_fast_answer_pipeline(
    row: dict,
    chunks: list[RetrievedChunk],
    *,
    fast_meta: dict[str, Any] | None = None,
    pool: list[RetrievedChunk] | None = None,
) -> FastAnswerPipelineContext:
    question = str(row.get("question", ""))
    summary_ctx = get_summary_context(question, row)
    fast_type = (fast_meta or {}).get("fast_question_type") or classify_fast_question_type(question, row)
    scope = classify_document_scope(chunks, question)
    all_claims = extract_claim_candidates(chunks, question=question, row=row)
    filtered = filter_claims_for_answer(all_claims, min_confidence="medium")
    if not filtered:
        filtered = filter_claims_for_answer(all_claims, min_confidence="low")
    item_count = int(row.get("outcome_item_count") or 3)
    m = __import__("re").search(r"(\d+)\s*개", question)
    if m:
        item_count = int(m.group(1))
    selected = select_claims_for_question(
        filtered or all_claims, question, target_n=item_count, row=row
    )

    return FastAnswerPipelineContext(
        fast_type=fast_type,
        scope=scope,
        summary_ctx=summary_ctx,
        all_claims=all_claims,
        filtered_claims=filtered,
        selected_claims=selected,
        use_evidence_first=_uses_evidence_first(fast_type),
        item_count=item_count,
        selected_primary_doc=_primary_doc_from_chunks(chunks, summary_ctx),
        rejected_docs=_collect_rejected_docs(pool or chunks, chunks, summary_ctx),
    )


def build_evidence_first_prompts(
    row: dict,
    chunks: list[RetrievedChunk],
    compact_context: str,
    pipeline: FastAnswerPipelineContext,
    *,
    low_confidence: bool = False,
) -> tuple[str, str]:
    question = str(row.get("question", ""))
    scope = pipeline.scope
    n = pipeline.item_count
    ctx = pipeline.summary_ctx

    system = MARITIME_FAST_SYSTEM.format(glossary=IMO_GLOSSARY_PROMPT).strip()

    scope_block = ""
    if scope.scope_correction_sentence:
        scope_block = (
            f"\n[사전 판별 — 문서 범위]\n"
            f"- 유형: {scope.scope_label_ko} ({scope.scope_type})\n"
            f"- 제목: {', '.join(scope.primary_titles[:2])}\n"
            f"- 참조 body: {', '.join(scope.referenced_bodies) or '—'}\n"
            f"- 범위 정정 문장(답변 첫 문장에 반영): {scope.scope_correction_sentence}\n"
        )
    elif scope.scope_type != "unknown":
        scope_block = (
            f"\n[사전 판별 — 문서 범위]\n"
            f"- 유형: {scope.scope_label_ko}\n"
            f"- 제목: {', '.join(scope.primary_titles[:2])}\n"
        )

    if ctx:
        scope_block += (
            f"\n[meeting_summary routing]\n"
            f"- target_meeting: {ctx.target_meeting}\n"
            f"- target_scope: {ctx.target_scope}\n"
            f"- session_final_priority: {ctx.apply_session_final_priority}\n"
            f"- require_topics: {', '.join(ctx.require_topics) or '—'}\n"
        )

    claims_json = json.dumps(pipeline.selected_claims, ensure_ascii=False, indent=2)
    claims_block = (
        f"\n[사전 추출 claim 후보 — high/medium만, 근거 부족 시 사용 금지]\n{claims_json}\n"
    )

    if pipeline.fast_type == "meeting_summary" and ctx and ctx.apply_session_final_priority:
        mass_rule = (
            "**필수:** 1번은 비강제 MASS Code 채택(근거 있을 때)."
            if ctx.require_topics
            else "**우선:** adopted/approved 핵심 결정."
        )
        format_block = f"""
[출력 형식 — meeting_summary whole_session]
제목: {session_title_from_question(question, row)} 주요 결과 {n}개

1. [핵심 결과명]
- 주요 내용:
- 실무 의미:
- 근거: [N] 또는 p.N

(2~{n}번 동일)

{mass_rule}
**근거 우선순위:** corpus에 IMO official highlights가 있으면 최우선, 없으면 Draft Report(WP.1) 또는 Session Report(예: MSC 111/22).
**금지:** 적용 대상/발효/영향 없음, 확인 불가, Strategic Plan, FAL 50, A 34/C 135(전체 회의 요약 시).
**사전 추출 claim 후보를 우선 사용.**
마지막 줄: "상세 분석은 Accurate mode에서 수행 가능합니다."
"""
    elif pipeline.fast_type == "meeting_summary":
        format_block = f"""
[출력 형식 — meeting_summary]
제목: {session_title_from_question(question, row)} 주요 결과 {n}개
각 항목: 핵심 결과명 / 주요 내용 / 실무 의미(있을 때) / 근거 [N]
규정 분석 형식(적용 대상/발효/영향 없음) 금지.
"""
    elif pipeline.fast_type == "meeting_outcome_question":
        format_block = f"""
[출력 형식 — 반드시 준수]
1) **첫 문장**: 문서 범위 정정 (근거 문서가 회의 자체 결과가 아니면 명시)
2) **정확히 {n}개 번호 항목** (1. 2. 3. …)
   - 각 항목 1~3문장, 근거 문서 내용 반영
   - resolution→결의안, adopted→채택, approved→승인
   - 각 항목 끝: `근거: [N]` 또는 `근거: p.N`
3) 근거 없는 항목은 쓰지 말 것
4) 마지막 줄: "상세 분석은 Accurate mode에서 수행 가능합니다."
"""
    else:
        format_block = f"""
[출력 형식]
1) 첫 문장: 문서 범위·성격 (필요 시 정정)
2) 핵심 {n}~5 bullet, 각 bullet 끝 근거 [N] 또는 p.N
3) 마지막 줄: "상세 분석은 Accurate mode에서 수행 가능합니다."
"""

    conf = ""
    if low_confidence:
        conf = (
            "\n[참고] 현재 Fast 검색 결과 기준으로 확인 가능한 범위는 다음과 같으며, "
            "상세 검증은 Accurate mode에서 추가 확인이 필요합니다.\n"
        )

    user = (
        f"질문: {question}\n"
        f"{scope_block}"
        f"{claims_block}"
        f"\n검색 근거 (context):\n{compact_context}\n"
        f"{format_block}"
        f"{conf}"
        "\n위 절차에 따라 **최종 답변만** 한국어로 작성하세요 (내부 분석 과정은 출력하지 마세요)."
    )
    return system, user


def build_citation_mapping(
    chunks: list[RetrievedChunk],
    answer: str,
) -> list[dict[str, Any]]:
    import re

    mapping: list[dict[str, Any]] = []
    for i, c in enumerate(chunks, start=1):
        mapping.append(
            {
                "cite_id": i,
                "chunk_id": c.chunk_id,
                "file_name": c.file_name,
                "page": c.page_number,
                "referenced_in_answer": bool(
                    re.search(rf"\[{i}\]|p\.{c.page_number}\b", answer or "", re.I)
                ),
            }
        )
    return mapping


def postprocess_fast_answer(
    answer: str,
    pipeline: FastAnswerPipelineContext,
    *,
    row: dict | None = None,
) -> str:
    out = normalize_imo_terms(answer or "")
    violations = detect_terminology_violations(out)
    if violations and pipeline.scope.scope_mismatch and pipeline.scope.scope_correction_sentence:
        if pipeline.scope.scope_correction_sentence not in out:
            out = pipeline.scope.scope_correction_sentence + "\n\n" + out

    ctx = pipeline.summary_ctx
    if pipeline.fast_type == "meeting_summary" and ctx:
        sources = list(
            dict.fromkeys(
                str(c.get("source") or "") for c in pipeline.selected_claims if c.get("source")
            )
        )
        passed, reasons = validate_meeting_summary_answer(
            out, row=row, evidence_sources=sources, ctx=ctx
        )
        pipeline.validation_result = {"passed": passed, "reasons": reasons}
        if not passed and ctx.apply_session_final_priority and pipeline.selected_claims:
            fallback = build_claim_based_summary_answer(ctx, pipeline.selected_claims, row=row or {})
            if fallback:
                pipeline.fallback_used = True
                out = fallback
                passed, reasons = validate_meeting_summary_answer(
                    out, row=row, evidence_sources=sources, ctx=ctx
                )
                pipeline.validation_result = {
                    "passed": passed,
                    "reasons": reasons,
                    "fallback": True,
                }
    return out
