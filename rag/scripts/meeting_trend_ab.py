"""A/B answer variants for latest_trend_summary (V01/V02-style) questions — LLM + evidence."""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from meeting_category_profile import (
    TOP_LEVEL_TREND,
    MeetingRetrievalProfile,
    build_meeting_retrieval_profile,
    resolve_top_level_category,
)
from meeting_outcome_answer import (
    build_meeting_outcome_system_prompt,
    build_meeting_outcome_user_prompt,
    build_meeting_summary_system_prompt,
    build_meeting_summary_user_prompt,
)
from meeting_outcome_retrieval import parse_outcome_item_count
from meeting_structured_answer import (
    OUTCOME_ACTION_RE,
    REFERENCE_OUTCOME_RE,
    RESOLUTION_REF_RE,
    _is_penalty_reference_chunk,
    _rank_official_dense,
    _strip_meta,
    score_chunk,
)
from meeting_summary_context import build_claim_based_summary_answer, get_summary_context
from meeting_summary_intent import is_meeting_summary_intent
from meeting_topic_cluster import MSC_OUTCOME_TOPIC_PRIORITY, outcome_topic_id, pick_diverse_topic_chunks
from question_classifier import classify_question_category
from retrieval_query_analysis import is_meeting_outcome_question
from retrieval_verification import meeting_routing_fields_from_row
from source_tier_lib import classify_source_tier, count_outcome_signals

try:
    from rag_answer_lib import DEFAULT_OLLAMA_BASE, DEFAULT_OLLAMA_MODEL
except ImportError:
    DEFAULT_OLLAMA_BASE = "http://localhost:11434"
    DEFAULT_OLLAMA_MODEL = "llama3.1:8b"

VARIANT_A = "official_dense"
VARIANT_B = "topic_diverse"

VARIANT_META: dict[str, dict[str, str]] = {
    VARIANT_A: {
        "label_ko": "답변 A",
        "subtitle_ko": "공식 보고·결의안 중심",
        "description_ko": "공식 session report·highlights에서 **결의안 번호·채택/승인 결정**을 구체적으로 요약합니다.",
    },
    VARIANT_B: {
        "label_ko": "답변 B",
        "subtitle_ko": "주제 분산·실무 영향 중심",
        "description_ko": "GHG·LRIT·SOLAS·CII·MASS 등 **서로 다른 topic**별 핵심 결정 + 실무 의미를 요약합니다.",
    },
}

VARIANT_A_INSTRUCTION = """

---
**[답변 A — 공식 보고·결의안 중심]**
- context의 adopted/approved 문장을 **한국어 2~3문장**으로 직접 요약할 것.
- 각 항목에 **결의안 번호·채택/승인 대상**을 context에서 그대로 반영.
- **절대 금지:** "추가 확인 필요", "본문 참조", "확인 불가", "영향 없음", "적용 대상 없음", "명확히 확인되지 않음".
- 같은 topic이라도 **서로 다른 결의**면 별도 항목으로 작성.
"""

VARIANT_B_INSTRUCTION = """

---
**[답변 B — 주제 분산·실무 영향 중심]**
- **서로 다른 topic**(GHG·대체연료·LRIT·SOLAS·CII·MASS 등)별 핵심 결정 1개씩.
- 각 항목: **주요 내용 2문장 + 선박 운항/설계/규제 관점 실무 의미 1문장** (근거 있을 때).
- MASS Code 외 topic을 **최소 2개** 포함(근거 있을 때).
- **절대 금지:** A와 동일한 결의 나열, "추가 확인 필요" 등 회피 문구.
"""

TOPIC_DIVERSE_PRIORITY: tuple[str, ...] = (
    "ghg_safety",
    "lrit_vdes",
    "maritime_safety",
    "hormuz",
    "ghg_framework",
    "cii_reporting",
    "mass_code",
    "general_outcome",
)

FORBIDDEN_ANSWER_PHRASES = (
    "추가 확인 필요",
    "본문 참조",
    "확인 불가",
    "명확히 확인되지",
    "영향 없음",
    "적용 대상 없음",
    "검색 결과 내에서",
)


def resolve_legacy_category(question: str, row: dict) -> str:
    explicit = str(row.get("_eval_category") or row.get("category") or "").strip()
    if explicit:
        return explicit
    return classify_question_category(question, row)


def resolve_ab_legacy_category(question: str, row: dict) -> tuple[dict, str]:
    """Align legacy category / internal_intent with UI 「최신 동향 요약」 tab questions."""
    work = dict(row)
    legacy = resolve_legacy_category(question, work)
    q = question or ""
    ql = q.lower()

    if is_meeting_summary_intent(q, work) or is_meeting_outcome_question(q, work):
        work["internal_intent"] = "meeting_outcome"
        return work, "meeting_outcome"

    if re.search(r"\bmepc\b", ql) and re.search(r"최신|주요|동향|정리|요약", q, re.I):
        work["internal_intent"] = "trend_summary"
        work["category"] = "trend_summary"
        return work, "trend_summary"

    if legacy == "meeting_outcome":
        work["internal_intent"] = "meeting_outcome"
    return work, legacy


def is_trend_summary_ab_eligible(question: str, row: dict | None = None) -> bool:
    """True for latest_trend_summary tab questions (MEPC 동향, MSC 111 주요 결과 3개 등)."""
    row = row or {}
    # Table QA has its own retrieval and answer pipeline. Some table questions
    # contain generic words such as "결과" or "요약" and can otherwise be
    # mistaken for a meeting/trend request by the broad legacy classifiers.
    # The A/B UI stores an empty answer until a variant is selected, making the
    # table tab appear to have failed in both Fast and Accurate modes.
    explicit_category = str(row.get("category") or row.get("_eval_category") or "").strip().lower()
    if (
        row.get("_table_qa")
        or row.get("question_type")
        or row.get("gold_table_id")
        or explicit_category == "table_qa"
    ):
        return False
    legacy = resolve_legacy_category(question, row)
    if resolve_top_level_category(legacy) == TOP_LEVEL_TREND:
        return True
    if is_meeting_summary_intent(question, row):
        return True
    if is_meeting_outcome_question(question, row):
        return True
    q = question or ""
    ql = q.lower()
    if re.search(r"\b(mepc|msc)\s*\d{0,3}\b", ql) and re.search(
        r"최신|주요|동향|정리|요약|회의", q, re.I
    ):
        if re.search(r"\bmepc\b", ql) and re.search(r"최신|주요|동향", q):
            return True
        if re.search(r"\bmsc\b", ql) and re.search(r"주요|핵심|결과|outcome", q, re.I):
            return True
    return False


def apply_variant_profile(base: MeetingRetrievalProfile, variant_id: str) -> MeetingRetrievalProfile:
    if variant_id == VARIANT_A:
        return replace(
            base,
            answer_variant=VARIANT_A,
            profile_id=f"{base.profile_id}_official",
            dense_weight=max(base.dense_weight, 1.2),
            bm25_weight=min(base.bm25_weight, 0.9),
        )
    if variant_id == VARIANT_B:
        return replace(
            base,
            answer_variant=VARIANT_B,
            profile_id=f"{base.profile_id}_diverse",
            dense_weight=min(base.dense_weight, 1.0),
            bm25_weight=max(base.bm25_weight, 1.1),
        )
    return base


def topic_priority_for_variant(profile: MeetingRetrievalProfile) -> tuple[str, ...] | None:
    if profile.answer_variant == VARIANT_B:
        return TOPIC_DIVERSE_PRIORITY
    if profile.internal_intent == "meeting_outcome":
        return MSC_OUTCOME_TOPIC_PRIORITY
    return None


def _pick_context_official_dense(scored: list[tuple[float, Any]], *, max_chunks: int = 14) -> list[Any]:
    official = [(s, c) for s, c in scored if classify_source_tier(c) <= 1 and not _is_penalty_reference_chunk(c)]
    pool = sorted(official if len(official) >= 4 else scored, key=_rank_official_dense, reverse=True)

    picked: list[Any] = []
    seen_res: set[str] = set()
    seen_cid: set[str] = set()
    for _s, chunk in pool:
        if len(picked) >= max_chunks:
            break
        cid = str(getattr(chunk, "chunk_id", "") or "")
        if cid in seen_cid:
            continue
        text = _strip_meta(getattr(chunk, "text", ""))
        if not OUTCOME_ACTION_RE.search(text) and count_outcome_signals(text) < 1:
            continue
        m = RESOLUTION_REF_RE.search(text)
        if m:
            res_key = m.group(1).strip().lower()
            if res_key in seen_res:
                continue
            seen_res.add(res_key)
        seen_cid.add(cid)
        picked.append(chunk)

    if len(picked) < max_chunks:
        for _s, chunk in pool:
            if len(picked) >= max_chunks:
                break
            cid = str(getattr(chunk, "chunk_id", "") or "")
            if cid in seen_cid or _is_penalty_reference_chunk(chunk):
                continue
            seen_cid.add(cid)
            picked.append(chunk)
    return picked


def _pick_context_topic_diverse(scored: list[tuple[float, Any]], *, max_chunks: int = 14) -> list[Any]:
    filtered = [(s, c) for s, c in scored if not _is_penalty_reference_chunk(c)]
    pool = filtered if filtered else scored
    topic_caps = {tid: 1 for tid in TOPIC_DIVERSE_PRIORITY}
    topic_caps["ghg_framework"] = 1
    topic_caps["ghg_safety"] = 1

    picked = pick_diverse_topic_chunks(
        pool,
        max_chunks,
        topic_priority=TOPIC_DIVERSE_PRIORITY,
        topic_caps=topic_caps,
        topic_fn=outcome_topic_id,
    )
    if len(picked) >= max_chunks:
        return picked[:max_chunks]

    seen = {str(getattr(c, "chunk_id", "") or "") for c in picked}
    for _s, chunk in pool:
        if len(picked) >= max_chunks:
            break
        cid = str(getattr(chunk, "chunk_id", "") or "")
        if cid in seen:
            continue
        seen.add(cid)
        picked.append(chunk)
    return picked


def _build_ab_prompts(row: dict, context: str, variant_id: str) -> tuple[str, str]:
    question = str(row.get("question") or "")
    instr = VARIANT_A_INSTRUCTION if variant_id == VARIANT_A else VARIANT_B_INSTRUCTION

    if is_meeting_summary_intent(question, row):
        return build_meeting_summary_system_prompt(row) + instr, build_meeting_summary_user_prompt(row, context)
    if is_meeting_outcome_question(question, row):
        return build_meeting_outcome_system_prompt(row) + instr, build_meeting_outcome_user_prompt(row, context)

    from rag_answer_lib import build_system_prompt, build_user_prompt

    row = dict(row)
    row.setdefault("category", "trend_summary")
    return build_system_prompt(row) + instr, build_user_prompt(row, context)


def _answer_has_forbidden_placeholders(answer: str) -> bool:
    low = answer.lower()
    hits = sum(1 for p in FORBIDDEN_ANSWER_PHRASES if p in answer or p.lower() in low)
    return hits >= 2 or ("추가 확인" in answer and "본문" in answer)


def _claim_fallback(chunks: list[Any], *, row: dict, variant_id: str) -> str:
    from fast_claim_extraction import extract_claim_candidates, select_claims_for_question

    question = str(row.get("question") or "")
    n = parse_outcome_item_count(question, row) or 3
    claims = extract_claim_candidates(chunks, question=question, row=row)
    if variant_id == VARIANT_B:
        selected = select_claims_for_question(claims, question, target_n=n, row=row)
    else:
        selected = sorted(claims, key=lambda c: -float(c.get("summary_score", 0)))[:n]
    ctx = get_summary_context(question, row)
    out = build_claim_based_summary_answer(ctx, selected, row=row)
    return out or ""


def _generate_variant_answer(
    chunks: list[Any],
    *,
    question: str,
    row: dict,
    profile: MeetingRetrievalProfile,
    variant_id: str,
    llm_model: str,
    ollama_base: str,
    temperature: float = 0.15,
    timing=None,
) -> tuple[str, list[str], dict[str, Any]]:
    from rag_answer_lib import build_context_block, call_ollama_chat_timed

    warnings: list[str] = []
    scored = [(score_chunk(c, profile=profile), c) for c in chunks]
    scored.sort(key=lambda x: -x[0])

    if variant_id == VARIANT_A:
        ctx_chunks = _pick_context_official_dense(scored)
    else:
        ctx_chunks = _pick_context_topic_diverse(scored)

    if not ctx_chunks:
        ctx_chunks = [c for _, c in scored[:14]]
        warnings.append("weak_context_selection")

    context = build_context_block(ctx_chunks)
    system, user = _build_ab_prompts(row, context, variant_id)

    answer = ""
    try:
        answer = call_ollama_chat_timed(
            llm_model,
            system,
            user,
            ollama_base,
            temperature=temperature,
            num_ctx=16384,
            timing=timing,
        )
    except Exception as exc:
        warnings.append(f"llm_error:{type(exc).__name__}")

    if not (answer or "").strip() or _answer_has_forbidden_placeholders(answer):
        fallback = _claim_fallback(ctx_chunks, row=row, variant_id=variant_id)
        if fallback.strip():
            answer = fallback
            warnings.append("claim_fallback_used")
        elif not answer.strip():
            warnings.append("empty_answer")

    meta = {
        "answer_variant": variant_id,
        "context_chunk_count": len(ctx_chunks),
        "context_chunk_ids": [str(getattr(c, "chunk_id", "")) for c in ctx_chunks[:8]],
        "answer_mode": "trend_ab_llm",
    }
    return answer.strip(), warnings, meta


def generate_trend_ab_answers(
    chunks: list[Any],
    *,
    question: str,
    row: dict,
    legacy_category: str | None = None,
    llm_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base: str = DEFAULT_OLLAMA_BASE,
    temperature: float = 0.15,
    timing=None,
) -> dict[str, dict[str, Any]]:
    legacy = legacy_category or resolve_legacy_category(question, row)
    work_row, legacy = resolve_ab_legacy_category(question, dict(row))
    base = build_meeting_retrieval_profile(question, work_row, legacy_category=legacy)
    out: dict[str, dict[str, Any]] = {}

    for key, variant_id in (("a", VARIANT_A), ("b", VARIANT_B)):
        variant_row = dict(work_row)
        profile = apply_variant_profile(base, variant_id)
        answer, warnings, meta = _generate_variant_answer(
            list(chunks)[:40],
            question=question,
            row=variant_row,
            profile=profile,
            variant_id=variant_id,
            llm_model=llm_model,
            ollama_base=ollama_base,
            temperature=temperature,
            timing=timing,
        )
        variant_row["_meeting_answer_meta"] = meta
        variant_row["_top_level_category"] = profile.top_level_category
        variant_row["_internal_intent"] = profile.internal_intent
        variant_row["_meeting_retrieval_profile"] = profile.to_log_dict()
        variant_row["warning_flags"] = warnings
        summary = {
            "answer_mode": "trend_ab_llm",
            "answer_variant": variant_id,
            **VARIANT_META[variant_id],
            **meeting_routing_fields_from_row(variant_row, answer=answer),
        }
        out[key] = {
            "variant_id": variant_id,
            "answer": answer,
            "warnings": warnings,
            "meta": meta,
            "summary": summary,
            "profile": profile.to_log_dict(),
        }
    return out
