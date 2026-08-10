"""Fast RAG mode — minimal retrieval context + short prompt for low TTFT."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from rag_answer_lib import (
    RetrievedChunk,
    call_ollama_chat_timed,
    model_prefers_think_off,
    retrieve_for_question,
)
from ollama_warmup import ensure_fast_warm, ensure_fast_warm_checked, mark_fast_llm_run
from retrieval_timing import TimingTrace, estimate_tokens

from fast_confidence import assess_fast_confidence
from fast_answer_pipeline import (
    build_citation_mapping,
    build_evidence_first_prompts,
    postprocess_fast_answer,
    prepare_fast_answer_pipeline,
)
from fast_mode_trace import append_fast_answer_trace, build_trace_record
from fast_context import build_slot_compact_context
from fast_prompts import build_fast_system_prompt, build_fast_user_prompt
from fast_question_classifier import classify_fast_question_type, fast_type_label_ko
from fast_retrieval import evidence_to_chunks, select_fast_evidence_slots

FAST_RETRIEVAL = {
    "top_k": 3,
    "fetch_k": 10,
    "pool_fetch_k": 18,
    "max_docs": 2,
    "max_chunks_per_doc": 1,
    "use_rerank": False,
    "preview_chars": 600,
    "use_typed_slots": True,
    # The 271k-chunk global BM25 takes ~14s even after loading. Interactive UI
    # modes use dense retrieval plus typed evidence slots; BM25 remains an
    # offline evaluation/build artifact until a sub-second sparse backend lands.
    "use_hybrid_bm25": False,
}

TABLE_FAST_RETRIEVAL = {
    "top_k": 10,
    "fetch_k": 30,
    "pool_fetch_k": 30,
    "max_docs": 3,
    "max_chunks_per_doc": 5,
    "use_rerank": False,
    "preview_chars": 4000,
    "use_typed_slots": True,
}

RULE_GUIDANCE_FAST_RETRIEVAL = {
    "top_k": 8,
    "fetch_k": 40,
    "pool_fetch_k": 56,
    "max_docs": 3,
    "max_chunks_per_doc": 2,
    "use_rerank": False,
    "preview_chars": 800,
    "use_typed_slots": True,
    "use_hybrid_bm25": False,
    "hard_society_filter": True,
}

MEETING_FAST_RETRIEVAL = {
    "top_k": 8,
    "fetch_k": 40,
    "pool_fetch_k": 48,
    "max_docs": 5,
    "max_chunks_per_doc": 2,
    "use_rerank": False,
    "preview_chars": 900,
    "use_typed_slots": True,
    # Full-corpus BM25 is ~10–15s/query and stalls interactive Accurate.
    # Meeting quality relies on dense expansion + WP.1/topic boosts instead.
    "use_hybrid_bm25": False,
}

ACCURATE_MEETING_RETRIEVAL = {
    **MEETING_FAST_RETRIEVAL,
    "top_k": 10,
    "fetch_k": 48,
    "pool_fetch_k": 56,
    "max_docs": 6,
    "max_chunks_per_doc": 2,
}

FAST_LLM = {
    "num_ctx": 4096,
    "max_new_tokens": 512,
    "max_new_tokens_meeting": 600,
    "temperature": 0.1,
}

# Legacy prompts (fallback when use_typed_slots=False)
FAST_SYSTEM_PROMPT = (
    "너는 해사 규정 문서 기반 RAG assistant다. "
    "제공된 근거 context 안에서만 답변해라. "
    "근거가 부족하면 부족하다고 말해라. 답변은 간결하게 작성해라."
)


def trim_fast_chunks(
    pool: list[RetrievedChunk],
    *,
    max_chunks: int = 3,
    max_docs: int = 2,
    max_per_doc: int = 1,
) -> list[RetrievedChunk]:
    out: list[RetrievedChunk] = []
    doc_counts: dict[str, int] = {}
    seen_docs: set[str] = set()
    for chunk in pool:
        doc_id = chunk.doc_id
        if doc_id not in seen_docs and len(seen_docs) >= max_docs:
            continue
        n = doc_counts.get(doc_id, 0)
        if n >= max_per_doc:
            continue
        out.append(chunk)
        doc_counts[doc_id] = n + 1
        seen_docs.add(doc_id)
        if len(out) >= max_chunks:
            break
    return out


def build_compact_context(chunks: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        page = c.page_number if c.page_number is not None else "?"
        name = c.file_name or c.doc_id
        text = (c.text or "").strip().replace("\n", " ")
        if len(text) > 500:
            text = text[:500] + "…"
        lines.append(f"[{i}] {name} p.{page}: {text}")
    return "\n".join(lines)


def build_fast_context_and_chunks(
    pool: list[RetrievedChunk],
    row: dict,
    *,
    use_typed_slots: bool = True,
) -> tuple[list[RetrievedChunk], str, dict[str, Any]]:
    """Slot-based (typed) or legacy trim for Fast context."""
    question = str(row.get("question", ""))
    fast_type = classify_fast_question_type(question, row)
    meta: dict[str, Any] = {
        "fast_question_type": fast_type,
        "fast_question_type_label": fast_type_label_ko(fast_type),
    }

    if use_typed_slots and pool:
        evidence = select_fast_evidence_slots(pool, question, row, fast_type=fast_type)
        chunks = evidence_to_chunks(evidence)
        compact = build_slot_compact_context(evidence) if evidence else ""
        conf = assess_fast_confidence(question, row, evidence, fast_type=fast_type)
        meta["fast_evidence_slots"] = [ev.slot for ev in evidence]
        meta["fast_confidence"] = conf.score
        meta["fast_low_confidence"] = conf.low_confidence
        meta["fast_confidence_reasons"] = conf.reasons
        return chunks, compact, meta

    chunks = trim_fast_chunks(
        pool,
        max_chunks=FAST_RETRIEVAL["top_k"],
        max_docs=FAST_RETRIEVAL["max_docs"],
        max_per_doc=FAST_RETRIEVAL["max_chunks_per_doc"],
    )
    compact = build_compact_context(chunks)
    meta["fast_evidence_slots"] = ["legacy_trim"] * len(chunks)
    meta["fast_confidence"] = None
    meta["fast_low_confidence"] = False
    return chunks, compact, meta


def build_fast_prompts(
    row: dict,
    chunks: list[RetrievedChunk],
    *,
    compact_context: str | None = None,
    fast_meta: dict[str, Any] | None = None,
) -> tuple[str, str, str]:
    fast_meta = fast_meta or {}
    fast_type = fast_meta.get("fast_question_type") or classify_fast_question_type(
        str(row.get("question", "")), row
    )
    compact = compact_context or build_compact_context(chunks)
    slots = fast_meta.get("fast_evidence_slots") or []
    use_typed = bool(slots) and slots != ["legacy_trim"]

    if use_typed or fast_meta.get("fast_question_type"):
        system = build_fast_system_prompt(fast_type)
        user = build_fast_user_prompt(
            row,
            compact,
            fast_type=fast_type,
            low_confidence=bool(fast_meta.get("fast_low_confidence")),
        )
    else:
        system = FAST_SYSTEM_PROMPT
        user = (
            f"질문: {row.get('question', '')}\n\n"
            f"근거:\n{compact}\n\n"
            "위 근거를 바탕으로 핵심만 3~5개 bullet로 답변해줘. "
            "각 bullet은 1~2문장. 가능하면 문서명 또는 페이지를 짧게 표시해줘. "
            "근거가 부족하면 '상세 확인 필요'를 bullet로 표시해줘. "
            "마지막 줄에 '상세 분석은 Accurate mode에서 수행 가능합니다.'를 붙여줘."
        )
    return system, user, compact


def record_llm_prompt_meta(
    timing: TimingTrace | None,
    *,
    latency_mode: str,
    system: str,
    user: str,
    compact_context: str,
    chunks: list[RetrievedChunk],
    model_name: str,
    max_new_tokens: int,
    num_ctx: int,
    temperature: float,
    fast_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    doc_ids = sorted({c.doc_id for c in chunks if c.doc_id})
    meta = {
        "latency_mode": latency_mode,
        "mode": "fast_rag" if latency_mode == "fast" else latency_mode,
        "selected_doc_count": len(doc_ids),
        "selected_chunk_count": len(chunks),
        "total_context_chars": len(compact_context),
        "system_prompt_chars": len(system),
        "user_prompt_chars": len(user),
        "final_prompt_chars": len(system) + len(user),
        "input_token_estimate": estimate_tokens(system + user),
        "max_new_tokens": max_new_tokens,
        "model_name": model_name,
        "num_ctx": num_ctx,
        "temperature": temperature,
        "streaming_enabled": True,
    }
    if fast_meta:
        meta.update(
            {
                "fast_question_type": fast_meta.get("fast_question_type"),
                "fast_question_type_label": fast_meta.get("fast_question_type_label"),
                "fast_evidence_slots": fast_meta.get("fast_evidence_slots"),
                "fast_confidence": fast_meta.get("fast_confidence"),
                "fast_low_confidence": fast_meta.get("fast_low_confidence"),
            }
        )
        pipe = fast_meta.get("fast_pipeline") or {}
        scope = pipe.get("document_scope") or {}
        if scope:
            meta["document_scope_type"] = scope.get("scope_type")
            meta["scope_mismatch"] = scope.get("scope_mismatch")
    if timing is not None:
        timing.meta.update(meta)
    return meta


def run_fast_retrieval_only(
    row: dict,
    collection,
    embed_model: str,
    *,
    chunks_dir: Path,
    timing=None,
    retrieval_cfg: dict[str, Any] | None = None,
    eval_constrained: bool = False,
) -> dict[str, Any]:
    from question_classifier import category_label_ko, classify_question_category
    from rag_query_router import enrich_row_for_routing, is_rule_guidance_lookup

    row = enrich_row_for_routing(row, latency_mode="fast")
    pool_before_filter: list[RetrievedChunk] = []

    if timing is not None and hasattr(timing, "mark") and "t_retrieval_start" not in timing.monotonic:
        timing.mark("t_retrieval_start")

    table_qa = bool(row.get("_table_qa") or str(row.get("category") or "") == "table_qa")
    rule_guidance = (not table_qa) and is_rule_guidance_lookup(
        str(row.get("question") or ""),
        row,
        category=str(row.get("category") or ""),
    )
    from meeting_category_profile import uses_structured_meeting_answer

    meeting_q = (not table_qa) and uses_structured_meeting_answer(
        row, legacy_category=str(row.get("category") or row.get("_eval_category") or "")
    )
    if retrieval_cfg is not None:
        cfg = retrieval_cfg
    elif rule_guidance:
        cfg = RULE_GUIDANCE_FAST_RETRIEVAL
    elif meeting_q:
        cfg = MEETING_FAST_RETRIEVAL
    else:
        cfg = FAST_RETRIEVAL
    if "use_hybrid_bm25" in cfg:
        row["_use_hybrid_bm25"] = bool(cfg["use_hybrid_bm25"])
    pool_fetch = cfg.get("pool_fetch_k", cfg["fetch_k"])
    from meeting_outcome_retrieval import (
        is_latest_environment_summary_query,
        select_latest_environment_context,
    )

    latest_environment_summary = is_latest_environment_summary_query(
        str(row.get("question") or "")
    )
    # The decisive status sentence in MEPC 84/3 and several action-request
    # clauses occur after the first 600 characters.  Loading a longer local
    # preview is cheap and prevents Fast mode from summarizing only the header.
    preview_chars = (
        max(int(cfg.get("preview_chars", 600)), 2200)
        if latest_environment_summary
        else int(cfg.get("preview_chars", 600))
    )
    gold_filter = bool(row.get("gold_doc_id")) if eval_constrained else False
    pool = retrieve_for_question(
        collection,
        embed_model,
        row,
        top_k=pool_fetch,
        fetch_k=pool_fetch,
        chunks_dir=chunks_dir,
        preview_chars=preview_chars,
        gold_doc_filter=gold_filter,
        timing=timing,
    )
    if meeting_q or rule_guidance:
        from evidence_planner import complete_evidence_slots

        pool, evidence_completion = complete_evidence_slots(collection, pool, row)
        row["_evidence_completion"] = evidence_completion
    pool_before_filter = list(pool)

    if not (row.get("_evidence_completion") or {}).get("plan", {}).get("slots"):
        pool = select_latest_environment_context(
            str(row.get("question") or ""),
            pool,
            target_k=max(int(cfg.get("top_k", 3)) * 3, 9),
        )

    if rule_guidance:
        from rule_lookup_answer import filter_pool_for_rule_lookup
        from rag_society_filter import filter_pool_for_society, society_hard_filter_enabled

        pool = filter_pool_for_rule_lookup(pool)
        society = str(row.get("class_society_hint") or "")
        if society:
            pool, _ = filter_pool_for_society(
                pool, society, hard=society_hard_filter_enabled(row)
            )

    if timing is not None and hasattr(timing, "mark"):
        timing.mark("t_context_build_start")

    if meeting_q:
        # Structured meeting answers need evidence from several agenda topics.
        # The generic three-slot Fast selector was collapsing broad MEPC/MSC
        # questions to one or two documents, which produced hollow summaries.
        retrieved = trim_fast_chunks(
            pool,
            max_chunks=int(cfg.get("top_k", 8)),
            max_docs=int(cfg.get("max_docs", 5)),
            # A draft session report can legitimately support several distinct
            # outcomes.  Capping it at two chunks removed the third requested
            # MSC result and the MASS timeline evidence.
            max_per_doc=max(int(cfg.get("max_chunks_per_doc", 2)), 5),
        )
        fast_meta = {
            "fast_question_type": classify_fast_question_type(
                str(row.get("question", "")), row
            ),
            "fast_evidence_slots": ["meeting_diverse"] * len(retrieved),
            "fast_confidence": None,
            "fast_low_confidence": not bool(retrieved),
            "fast_confidence_reasons": [] if retrieved else ["meeting_evidence_empty"],
            "evidence_completion": row.get("_evidence_completion") or {},
        }
    else:
        retrieved, _, fast_meta = build_fast_context_and_chunks(
            pool,
            row,
            use_typed_slots=cfg.get("use_typed_slots", True),
        )
        # Preserve the semantic slot plan through the generic context builder.
        # Accurate Rule/Guidance reuses this retrieval path; without this field
        # answer generation could not reconstruct the scope/requirement hits.
        fast_meta["evidence_completion"] = row.get("_evidence_completion") or {}
    if rule_guidance:
        from rule_lookup_context import enrich_rule_lookup_chunks

        retrieved = enrich_rule_lookup_chunks(
            retrieved,
            pool,
            chunks_dir=chunks_dir,
            row=row,
        )

    if timing is not None and hasattr(timing, "mark"):
        timing.mark("t_context_build_end")
        timing.mark("t_retrieval_end")

    category = str(row.get("category") or classify_question_category(str(row.get("question", "")), row))
    table_qa = bool(table_qa or row.get("_table_qa") or category == "table_qa")
    answer_mode = (
        "table_qa"
        if table_qa
        else (
            "rule_guidance_lookup"
            if rule_guidance
            else ("structured_meeting" if meeting_q else "fast_rag")
        )
    )
    doc_ids = sorted({c.doc_id for c in retrieved})
    summary = {
        "answer_mode": answer_mode,
        "retrieval_mode": "fast",
        "latency_mode": "fast",
        "question_category": category,
        "question_category_label": category_label_ko(category),
        "fast_question_type": fast_meta.get("fast_question_type"),
        "fast_question_type_label": fast_meta.get("fast_question_type_label"),
        "fast_confidence": fast_meta.get("fast_confidence"),
        "fast_low_confidence": fast_meta.get("fast_low_confidence"),
        "unique_doc_count": len(doc_ids),
        "final_doc_count": len(doc_ids),
        "final_chunk_count": len(retrieved),
        "pool_unique_doc_count": len({c.doc_id for c in pool}),
        "fast_evidence_slots": fast_meta.get("fast_evidence_slots"),
        "evidence_completion": fast_meta.get("evidence_completion") or {},
    }
    from retrieval_verification import meeting_routing_fields_from_row

    summary.update(meeting_routing_fields_from_row(row))
    return {
        "question_id": row.get("question_id"),
        "category": category,
        "question": row.get("question"),
        "retrieved": retrieved,
        "retrieval_pool": pool,
        "retrieval_metrics": {
            "unique_doc_count": len(doc_ids),
            "fast_mode": True,
            **{k: fast_meta.get(k) for k in ("fast_question_type", "fast_confidence", "fast_low_confidence")},
        },
        "retrieval_config": {"latency_mode": "fast", **cfg, "fast_meta": fast_meta},
        "answer_mode": answer_mode,
        "question_category": category,
        "question_category_label": category_label_ko(category),
        "broad_summary_mode": False,
        "doc_groups": [],
        "pipeline_warnings": fast_meta.get("fast_confidence_reasons") or [],
        "evidence_table": [],
        "must_cover_coverage": [],
        "verification_summary": summary,
        "fast_meta": fast_meta,
        "table_retrieval_debug": row.get("_table_retrieval_debug"),
        "pool_before_society_filter": pool_before_filter,
    }


def generate_fast_answer(
    row: dict,
    chunks: list[RetrievedChunk],
    *,
    model: str,
    ollama_base: str,
    timing=None,
    on_token: Callable[[str], None] | None = None,
    temperature: float | None = None,
    auto_llm_warm: bool = True,
    allow_rewarm: bool | None = None,
    fast_meta: dict[str, Any] | None = None,
    pool: list[RetrievedChunk] | None = None,
) -> tuple[str, dict[str, Any]]:
    meta = dict(fast_meta or {})
    if not chunks and pool:
        chunks, compact_pre, meta = build_fast_context_and_chunks(
            pool, row, use_typed_slots=FAST_RETRIEVAL.get("use_typed_slots", True)
        )
    elif not meta.get("fast_question_type"):
        _, compact_pre, meta = build_fast_context_and_chunks(
            pool or chunks,
            row,
            use_typed_slots=bool(pool) and FAST_RETRIEVAL.get("use_typed_slots", True),
        )
    else:
        compact_pre = None

    if not chunks:
        if str(row.get("category") or "") == "rule_lookup" or row.get("_rule_guidance_lookup"):
            society = str(row.get("class_society_hint") or "해당 선급")
            ag = {
                "answer_source": "fallback_no_evidence",
                "llm_used": False,
                "llm_call_function": None,
                "llm_prompt_chars": 0,
                "llm_context_chunks": 0,
                "llm_output_chars": 0,
                "llm_grounded_check_pass": False,
                "fallback_reason": "society_evidence_insufficient",
            }
            row["_answer_generation"] = ag
            return (
                f"## 1) 핵심 요약\n\n- {society} 근거 부족. {society} Rule/Guidance 검색 결과에서 질문과 직접 연결되는 근거를 찾지 못했습니다. "
                f"다른 선급(KR/DNV/ABS 등) 문서로 대체하지 않았습니다.\n\n"
                "## 2) 선박 운항/업무 영향\n\n- 해당 선급 규정 원문 확인이 필요합니다.\n\n"
                "## 3) 후속 확인 필요\n\n- {society} Notice/Rule 문서명·Section을 지정해 재검색하세요.\n\n"
                "## 4) 관련 선급 Rule / Guidance\n\n- 본 응답은 {society} 전용 검색(하드 필터) 결과입니다.",
                {"answer_mode": "rule_guidance_lookup", "society_evidence_insufficient": True, "answer_generation": ag},
            )
        return "검색된 근거가 없습니다. 상세 확인 필요.", {}

    rule_guidance = (
        str(row.get("category") or "") == "rule_lookup"
        or row.get("_rule_guidance_lookup")
    )
    if rule_guidance:
        # A direct clause hit is a precision question, not a document-discovery
        # question.  The previous Fast path rendered a document-level template
        # even after retrieval had found the exact clause, which produced a
        # generic answer unrelated to the user's technical term.  Use the
        # compact grounded generator in this case; broad Rule discovery still
        # keeps the zero-LLM Fast template below.
        completion = row.get("_evidence_completion") or {}
        direct_clause_found = bool(
            (completion.get("slot_hits") or {}).get("specific_clause")
        )
        if direct_clause_found:
            from rule_guidance_accurate import generate_rule_guidance_accurate_answer

            answer, _provider, _model, generation = generate_rule_guidance_accurate_answer(
                row,
                chunks,
                pool=pool,
                model=model,
                ollama_base=ollama_base,
                timing=timing,
                on_token=None,
                temperature=0.0,
            )
            meta = dict(fast_meta or {})
            meta.update(
                {
                    "answer_mode": "rule_guidance_lookup",
                    "structured_rule_lookup": False,
                    "llm_skipped": not generation.get("llm_used"),
                    "answer_source": generation.get("answer_source"),
                    "answer_generation": generation,
                    "direct_clause_grounded": True,
                }
            )
            row["_answer_generation"] = generation
            if timing is not None and hasattr(timing, "mark_wall"):
                timing.mark_wall("t_answer_complete")
            return answer, meta

        from rule_lookup_structured_answer import build_rule_lookup_structured_answer

        warnings = list(row.get("warning_flags") or [])
        answer, ans_warnings = build_rule_lookup_structured_answer(
            chunks,
            question=str(row.get("question") or ""),
            pool=pool,
            warning_flags=warnings,
        )
        row["warning_flags"] = list(dict.fromkeys(warnings + ans_warnings))
        meta = dict(fast_meta or {})
        meta["answer_mode"] = "rule_guidance_lookup"
        meta["structured_rule_lookup"] = True
        meta["llm_skipped"] = True
        meta["answer_source"] = "structured_template"
        meta["answer_generation"] = {
            "answer_source": "structured_template",
            "llm_used": False,
            "llm_call_function": None,
            "llm_prompt_chars": 0,
            "llm_context_chunks": len(chunks),
            "llm_output_chars": len(answer or ""),
            "llm_grounded_check_pass": True,
            "fallback_reason": None,
        }
        row["_answer_generation"] = meta["answer_generation"]
        if timing is not None and hasattr(timing, "mark_wall"):
            timing.mark_wall("t_answer_complete")
        return answer, meta

    if row.get("_table_qa") or str(row.get("category") or "") == "table_qa":
        from table_qa_answer import (
            build_table_answer_prompts,
            select_table_evidence,
            top_table_cell_hints,
        )

        debug = row.get("_table_retrieval_debug") or {}
        full_table_pool = list(chunks) + list(pool or chunks)
        evidence = select_table_evidence(
            row, chunks, pool or chunks, debug=debug, max_chunks=12
        )
        if not evidence:
            evidence = list(chunks) or list(pool or [])
        hints = top_table_cell_hints(row, full_table_pool, debug=debug)
        # Citation order must match prompt numbering for Evidence Table.
        row["_answer_citation_chunks"] = list(evidence)
        row.pop("_verified_structured_answer", None)
        system, user = build_table_answer_prompts(
            row, evidence, debug=debug, cell_hints=hints
        )
        llm_cfg = FAST_LLM
        temp = min(temperature if temperature is not None else llm_cfg["temperature"], 0.15)
        warm_meta: dict[str, Any] = {}
        if auto_llm_warm:
            warm_meta = ensure_fast_warm_checked(
                model,
                ollama_base,
                timing=timing,
                allow_rewarm=True if allow_rewarm is None else allow_rewarm,
            )
        prompt_meta = record_llm_prompt_meta(
            timing,
            latency_mode="fast",
            system=system,
            user=user,
            compact_context=user,
            chunks=evidence,
            model_name=model,
            max_new_tokens=480,
            num_ctx=llm_cfg["num_ctx"],
            temperature=temp,
            fast_meta={**meta, "answer_mode": "table_qa"},
        )
        prompt_meta["answer_mode"] = "table_qa"
        prompt_meta["answer_source"] = "table_llm"
        prompt_meta["answer_generation"] = {
            "answer_source": "table_llm",
            "llm_used": True,
            "llm_call_function": "call_ollama_chat_timed",
            "llm_prompt_chars": len(system) + len(user),
            "llm_context_chunks": len(evidence),
            "llm_output_chars": 0,
            "llm_grounded_check_pass": None,
            "fallback_reason": None,
            "cell_hints": [f"{k}={v}" for k, v in hints[:5]],
        }
        prompt_meta["warmup"] = warm_meta
        # Table fast path used num_predict=480. Gemma4 default thinking can
        # consume the whole budget (done_reason=length, content="") — disable
        # think and keep a safer token ceiling for gemma*.
        table_num_predict = 480
        table_think: bool | None = None
        if model_prefers_think_off(model):
            table_think = False
            table_num_predict = max(table_num_predict, 1200)
        answer = call_ollama_chat_timed(
            model,
            system,
            user,
            ollama_base,
            temperature=temp,
            num_predict=table_num_predict,
            num_ctx=llm_cfg["num_ctx"],
            think=table_think,
            timing=timing,
            on_token=on_token,
        )
        prompt_meta["answer_generation"]["llm_output_chars"] = len(answer or "")
        prompt_meta["answer_generation"]["num_predict"] = table_num_predict
        prompt_meta["answer_generation"]["think"] = table_think
        row["_answer_generation"] = prompt_meta["answer_generation"]
        mark_fast_llm_run(model, llm_cfg["num_ctx"])
        return answer, prompt_meta

    from meeting_category_profile import build_meeting_retrieval_profile, uses_structured_meeting_answer

    legacy_cat = str(row.get("_eval_category") or row.get("category") or "")
    if uses_structured_meeting_answer(row, legacy_category=legacy_cat):
        from meeting_structured_answer import build_meeting_structured_answer

        mprofile = build_meeting_retrieval_profile(
            str(row.get("question") or ""), row, legacy_category=legacy_cat
        )
        # Broad meeting questions need more than the three latency-optimized
        # display chunks to produce a useful agenda summary.  Reuse the pool
        # that was already fetched (no extra search/LLM call), deduplicate it,
        # and preserve this exact order for [N] and the UI Evidence Table.
        ctx: list[RetrievedChunk] = []
        seen_context_ids: set[str] = set()
        for candidate in [*list(chunks), *list(pool or [])]:
            identity = str(candidate.chunk_id or "") or (
                f"{candidate.doc_id}:{candidate.page_number}:{len(candidate.text or '')}"
            )
            if identity in seen_context_ids:
                continue
            seen_context_ids.add(identity)
            ctx.append(candidate)
            if len(ctx) >= 10:
                break
        row["_answer_citation_chunks"] = list(ctx)
        answer, ans_warnings, ans_meta = build_meeting_structured_answer(
            ctx,
            question=str(row.get("question") or ""),
            row=row,
            profile=mprofile,
            warning_flags=list(row.get("warning_flags") or []),
        )
        row["warning_flags"] = list(dict.fromkeys((row.get("warning_flags") or []) + ans_warnings))
        row["_meeting_answer_meta"] = ans_meta
        row["_top_level_category"] = mprofile.top_level_category
        row["_internal_intent"] = mprofile.internal_intent
        meta["structured_meeting"] = True
        meta["meeting_answer_meta"] = ans_meta
        from rag_answer_lib import _structured_meeting_answer_is_hollow

        if not _structured_meeting_answer_is_hollow(answer):
            return answer, meta
        row.setdefault("warning_flags", []).append("structured_meeting_sparse_kept")
        meta["llm_fallback"] = False
        return answer, meta

    llm_cfg = FAST_LLM
    temp = temperature if temperature is not None else llm_cfg["temperature"]
    warm_meta: dict[str, Any] = {}
    if auto_llm_warm:
        rewarm_ok = True if allow_rewarm is None else allow_rewarm
        warm_meta = ensure_fast_warm_checked(
            model,
            ollama_base,
            timing=timing,
            allow_rewarm=rewarm_ok,
        )
        if timing is not None and hasattr(timing, "meta"):
            timing.meta.setdefault("rewarm_triggered", warm_meta.get("rewarm_triggered"))
            timing.meta.setdefault("rewarm_reason", warm_meta.get("rewarm_reason"))

    system, user, compact = build_fast_prompts(
        row, chunks, compact_context=compact_pre, fast_meta=meta
    )

    pipeline = prepare_fast_answer_pipeline(row, chunks, fast_meta=meta, pool=pool)
    meta["fast_pipeline"] = pipeline.to_dict()

    if pipeline.use_evidence_first:
        system, user = build_evidence_first_prompts(
            row,
            chunks,
            compact,
            pipeline,
            low_confidence=bool(meta.get("fast_low_confidence")),
        )

    if timing is not None and hasattr(timing, "mark_wall"):
        timing.mark_wall("t_prompt_build_end")

    max_tokens = llm_cfg["max_new_tokens"]
    if pipeline.fast_type in {"meeting_summary", "meeting_outcome_question"}:
        max_tokens = llm_cfg.get("max_new_tokens_meeting", max_tokens)

    prompt_meta = record_llm_prompt_meta(
        timing,
        latency_mode="fast",
        system=system,
        user=user,
        compact_context=compact,
        chunks=chunks,
        model_name=model,
        max_new_tokens=max_tokens,
        num_ctx=llm_cfg["num_ctx"],
        temperature=temp,
        fast_meta=meta,
    )
    prompt_meta["warmup"] = warm_meta
    prompt_meta["fast_pipeline"] = pipeline.to_dict()
    answer = call_ollama_chat_timed(
        model,
        system,
        user,
        ollama_base,
        temperature=temp,
        num_predict=max_tokens,
        num_ctx=llm_cfg["num_ctx"],
        timing=timing,
        on_token=on_token,
    )
    answer = postprocess_fast_answer(answer, pipeline, row=row)
    citation_mapping = build_citation_mapping(chunks, answer)
    prompt_meta["citation_mapping"] = citation_mapping

    timing_metrics = {}
    if timing is not None and hasattr(timing, "meta"):
        timing_metrics = dict(timing.meta.get("timing_metrics") or {})
    trace = build_trace_record(
        row=row,
        chunks=chunks,
        pipeline=pipeline,
        answer=answer,
        citation_mapping=citation_mapping,
        timing_metrics=timing_metrics,
        prompt_meta=prompt_meta,
    )
    append_fast_answer_trace(trace)
    prompt_meta["fast_answer_trace"] = trace

    mark_fast_llm_run(model, llm_cfg["num_ctx"])
    return answer, prompt_meta


def fast_summary_lines(extra: dict[str, Any]) -> list[str]:
    lines = ["**Fast mode LLM input**"]
    labels = [
        ("Doc scope", "document_scope_type"),
        ("Scope mismatch", "scope_mismatch"),
        ("Fast type", "fast_question_type_label"),
        ("Confidence", "fast_confidence"),
        ("Low confidence", "fast_low_confidence"),
        ("Slots", "fast_evidence_slots"),
        ("Latency mode", "latency_mode"),
        ("Chunks", "selected_chunk_count"),
        ("Docs", "selected_doc_count"),
        ("Context chars", "total_context_chars"),
        ("System prompt chars", "system_prompt_chars"),
        ("User prompt chars", "user_prompt_chars"),
        ("Final prompt chars", "final_prompt_chars"),
        ("Input token est.", "input_token_estimate"),
        ("max_new_tokens", "max_new_tokens"),
        ("num_ctx", "num_ctx"),
        ("model", "model_name"),
        ("temperature", "temperature"),
        ("streaming", "streaming_enabled"),
    ]
    for label, key in labels:
        val = extra.get(key)
        lines.append(f"- {label}: {val if val is not None else '—'}")
    return lines
