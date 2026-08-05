"""
MaritimeRAG — Streamlit UI (in-process RAG with cached resources).

  streamlit run scripts/15_rag_ui.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import pandas as pd
import streamlit as st

from rag_answer_lib import (
    DEFAULT_OLLAMA_BASE,
    DEFAULT_OLLAMA_MODEL,
    check_ollama_model,
)
from rag_inprocess import (
    DEFAULT_CHUNKS_DIR,
    DEFAULT_INDEX_DIR,
    DEFAULT_UNIFIED,
    TABLE_QA_UNIFIED,
    TABLE_QA_CHUNKS_DIR,
    chunks_from_session,
    chunks_to_session,
    normalize_table_question_row,
    run_answer_inprocess,
    run_full_inprocess,
    run_search_inprocess,
)
from accurate_streaming import (
    ACCURATE_MODE_BANNER,
    mark_accurate_initial_ack,
)
from ollama_warmup import (
    bootstrap_all_resources,
    ensure_fast_warm,
    ensure_fast_warm_checked,
    get_fast_warm_ui_state,
    get_resource_ready_status,
    get_warm_preflight_snapshot,
    get_warm_status_display,
    warmup_fast_chat,
)
from retrieval_timing import (
    DEFAULT_TIMING_LOG,
    TimingTrace,
    append_timing_log,
    finalize_ui_timing,
    populate_timing_meta,
)
from rag_resource_cache import unified_index_fingerprint
from table_retrieval import evaluate_table_qa_retrieval
from meeting_trend_ab import (
    VARIANT_META,
    generate_trend_ab_answers,
    is_trend_summary_ab_eligible,
)

LOG_DIR = _ROOT / "data/processed/logs/pilot_validation"
TABLE_QA_LOG_DIR = _ROOT / "data/processed/logs"
SAVED_ANSWERS = LOG_DIR / "pilot_validation_answers_llm.md"
UI_SAVED_ANSWERS = LOG_DIR / "pilot_validation_answers_ui.md"
TABLE_UI_SAVED_ANSWERS = TABLE_QA_LOG_DIR / "table_qa_answers_ui.md"
TRACE_LOG = LOG_DIR / "retrieval_trace_ui.jsonl"
TREND_AB_FEEDBACK_LOG = LOG_DIR / "trend_ab_feedback.jsonl"

CATEGORY_LABELS = {
    "trend_summary": "최신 동향 요약",
    "env_regulation": "환경규제 대응",
    "autonomous": "자율운항",
    "rule_lookup": "단순 Rule 질문",
}

GENERAL_QA_EXAMPLE_GROUPS: dict[str, tuple[str, ...]] = {
    "1) 최신 동향 요약": (
        "환경규제 대응과 관련된 최신 MEPC 회의 주요 내용을 정리해줘.",
        "MSC 111의 주요 결과를 3개 항목으로 요약해줘.",
    ),
    "2) 환경규제 대응 관련": (
        "최신 MEPC 회의에서 선박 운항 및 규제 보고에 직접 영향을 주는 사항을 정리해줘.",
        "MSC 111에서 대체연료·GHG 안전규제와 관련된 논의 및 결론을 요약해줘.",
    ),
    "3) 자율운항 관련 질문": (
        "MSC 111에서 MASS Code와 관련된 핵심 결정사항을 요약하고, 향후 mandatory code 일정까지 정리해줘.",
    ),
    "4) 단순 Rule 질문": (
        "DNV에서 자율운항 또는 Smart Vessel 관련 Rule/Guidance를 찾아줘.",
        "LR에서 대체연료 관련 Rule/Guidance를 찾아줘.",
    ),
}

TABLE_QA_EXAMPLE_QUESTIONS = (
    "선령 5~10년 선박의 평형수탱크는 정기검사에서 어떤 방식으로 선정·검사하나?",
    "이중저탱크, 디프탱크, 평형수탱크, 피크탱크의 주위벽은 정기검사 각 차수에서 어떤 reporting 요건이 있나?",
    "선령 15년을 초과한 선박의 평형수탱크 검사 범위는 어떻게 되나?",
)


@st.cache_resource(show_spinner="Vector index + ChromaDB 로드 중…")
def cached_rag_resources(unified_id: str, index_dir: str, index_fingerprint: str):
    from rag_resource_cache import load_unified_collection

    return load_unified_collection(unified_id, Path(index_dir))


def _index_fingerprint(unified_id: str = DEFAULT_UNIFIED) -> str:
    return unified_index_fingerprint(unified_id, DEFAULT_INDEX_DIR)


def _next_run_context() -> tuple[str, int]:
    st.session_state.setdefault("rag_run_counter", 0)
    st.session_state.setdefault("rag_resources_warmed", False)
    st.session_state["rag_run_counter"] += 1
    start_type = "cold" if not st.session_state["rag_resources_warmed"] else "warm"
    return start_type, st.session_state["rag_run_counter"]


def _mark_resources_warmed() -> None:
    st.session_state["rag_resources_warmed"] = True


@st.cache_resource(show_spinner="앱 시작 warm-up (embedding + vector DB + LLM Fast)…")
def cached_app_bootstrap(model: str, ollama_base: str, unified_id: str, index_dir: str, index_fingerprint: str) -> dict:
    result = bootstrap_all_resources(
        model,
        ollama_base,
        unified_id,
        Path(index_dir),
        force_llm_warm=True,
    )
    # Import lazy Fast answer modules while the startup spinner is visible.
    # Otherwise their first import is charged to the user's first answer click.
    from meeting_structured_answer import build_meeting_structured_answer  # noqa: F401
    from rule_lookup_structured_answer import build_rule_lookup_structured_answer  # noqa: F401

    return result


def _judge_3s(e2e_ttft: float | None) -> str:
    if e2e_ttft is None:
        return "—"
    return "PASS" if e2e_ttft <= 3.0 else "FAIL"


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")


def _set_text_area_value(key: str, value: str) -> None:
    """Streamlit on_click callback: must run before the keyed widget is drawn."""
    st.session_state[key] = value


def _show_thinking(ph) -> None:
    ph.markdown(
        '<p style="color:#6b7280;font-size:0.95rem;margin:0.25rem 0 0.75rem 0;">생각 중…</p>',
        unsafe_allow_html=True,
    )


def _clear_thinking(ph, cleared: dict) -> None:
    if not cleared.get("done"):
        ph.empty()
        cleared["done"] = True


def _render_general_qa_example_picker(text_area_key: str = "general_custom_text") -> None:
    """Category tabs + per-tab question dropdown for pilot-style examples."""
    st.caption("카테고리 탭 → 드롭다운에서 예시 선택 → **입력란에 적용**")
    tab_labels = list(GENERAL_QA_EXAMPLE_GROUPS.keys())
    tabs = st.tabs(tab_labels)
    for tab, (group_label, questions) in zip(tabs, GENERAL_QA_EXAMPLE_GROUPS.items()):
        with tab:
            slug = re.sub(r"[^\w]+", "_", group_label).strip("_")
            picked = st.selectbox(
                "예시 질문",
                options=questions,
                key=f"general_example_select_{slug}",
                label_visibility="collapsed",
            )
            st.button(
                "입력란에 적용",
                key=f"general_example_apply_{slug}",
                use_container_width=True,
                on_click=_set_text_area_value,
                args=(text_area_key, picked),
            )


def _fmt_s(val: float | None) -> str:
    if val is None:
        return "—"
    if abs(val) < 0.01:
        return f"{val * 1000:.1f}ms"
    return f"{val:.2f}s"


def _pass_3s_label(mode: str, metrics: dict) -> str:
    if mode == "fast":
        if metrics.get("fast_e2e_ttft_3s_pass") is True:
            return "PASS"
        if metrics.get("fast_e2e_ttft_3s_pass") is False:
            return "FAIL"
        lat = metrics.get("answer_first_visible_latency") or metrics.get("end_to_end_ttft")
        return _judge_3s(lat)
    if metrics.get("accurate_initial_response_3s_pass") is True:
        return "PASS"
    if metrics.get("accurate_initial_response_3s_pass") is False:
        return "FAIL"
    return _judge_initial_3s(metrics.get("initial_ack_latency"))


def _store_last_run(
    *,
    run_id: str,
    qid: str,
    mode: str,
    model_name: str,
    timing_log: dict,
    timing_metrics: dict,
) -> None:
    st.session_state["last_run"] = {
        "run_id": run_id,
        "qid": qid,
        "mode": mode,
        "model_name": model_name,
        "timestamp": timing_log.get("timestamp") or _utc_iso(),
        "timing_log": timing_log,
        "timing_metrics": timing_metrics,
        "pass_3s": _pass_3s_label(mode, timing_metrics),
        "rag_debug_trace": (timing_log.get("meta") or {}).get("rag_debug_trace")
        or timing_log.get("rag_debug_trace"),
        "warm_preflight_before_click": timing_log.get("warm_preflight_before_click")
        or (timing_log.get("meta") or {}).get("warm_preflight_before_click"),
    }


def _render_rag_debug_panel() -> None:
    run = st.session_state.get("last_run")
    if not run:
        return
    trace = run.get("rag_debug_trace")
    if not trace:
        tl = run.get("timing_log") or {}
        trace = (tl.get("meta") or {}).get("rag_debug_trace")
    if not trace:
        return
    from rag_debug_trace import format_debug_trace_text

    st.markdown("#### RAG 디버그 trace")
    st.code(format_debug_trace_text(trace), language="text")
    with st.expander("RAG 디버그 JSON"):
        st.json(trace)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _apply_search_session(qid: str, out: dict, latency_mode: str) -> None:
    st.session_state["pilot_retrieved"] = chunks_to_session(out["retrieved"])
    st.session_state["pilot_pool"] = chunks_to_session(out.get("retrieval_pool") or out["retrieved"])
    st.session_state["pilot_metrics"] = out["retrieval_metrics"]
    st.session_state["pilot_config"] = out.get("retrieval_config")
    st.session_state["pilot_answer_mode"] = out.get("answer_mode", "standard_rag")
    st.session_state["pilot_question_category"] = out.get("question_category")
    st.session_state["pilot_doc_groups"] = out.get("doc_groups", [])
    st.session_state["pilot_evidence"] = out.get("evidence_table")
    st.session_state["pilot_must_cover"] = out.get("must_cover_coverage")
    st.session_state["pilot_summary"] = out.get("verification_summary")
    st.session_state["pilot_retrieved_qid"] = qid
    st.session_state["pilot_latency_mode"] = latency_mode


def _apply_table_session(qid: str, out: dict, latency_mode: str, eval_metrics: dict | None = None) -> None:
    st.session_state["table_retrieved"] = chunks_to_session(out["retrieved"])
    st.session_state["table_pool"] = chunks_to_session(out.get("retrieval_pool") or out["retrieved"])
    st.session_state["table_metrics"] = out.get("retrieval_metrics")
    st.session_state["table_eval_metrics"] = eval_metrics or {}
    st.session_state["table_evidence"] = out.get("evidence_table")
    st.session_state["table_summary"] = out.get("verification_summary")
    st.session_state["table_retrieval_debug"] = out.get("table_retrieval_debug")
    st.session_state["table_retrieved_qid"] = qid
    st.session_state["table_latency_mode"] = latency_mode


def _render_last_run_sidebar(*, latency_mode: str, llm_model: str) -> None:
    run = st.session_state.get("last_run")
    if not run:
        return
    tm = run.get("timing_metrics") or {}
    st.caption("**Last run (single-click)**")
    st.markdown(
        f"- run_id: `{run.get('run_id', '—')}`\n"
        f"- qid: `{run.get('qid', '—')}` · mode: `{run.get('mode', '—')}`\n"
        f"- model: `{run.get('model_name', llm_model)}`"
    )
    if run.get("mode") == "fast":
        vis = tm.get("answer_first_visible_latency") or tm.get("end_to_end_ttft")
        st.markdown(
            f"- first visible latency: `{_fmt_s(vis)}`\n"
            f"- 3s: `{run.get('pass_3s', '—')}`"
        )
    else:
        st.markdown(
            f"- initial ack: `{_fmt_s(tm.get('initial_ack_latency'))}`\n"
            f"- initial 3s: `{run.get('pass_3s', '—')}`"
        )


def _render_timing_breakdown() -> None:
    run = st.session_state.get("last_run")
    if not run:
        return
    tm = run.get("timing_metrics") or {}
    meta = (run.get("timing_log") or {})
    st.markdown("#### Latency breakdown")
    st.markdown(
        f"- **run_id:** `{run.get('run_id', '—')}` · **qid:** `{run.get('qid', '—')}` · "
        f"**mode:** `{run.get('mode', '—')}`"
    )
    rows = [
        ("Retrieval total", tm.get("retrieval_total_time")),
        ("Context build", tm.get("context_build_time")),
        ("Pre-LLM setup", tm.get("pre_llm_latency")),
        ("LLM TTFT", tm.get("llm_ttft")),
        ("First visible latency", tm.get("answer_first_visible_latency")),
        ("End-to-end TTFT", tm.get("end_to_end_ttft")),
        ("End-to-end total", tm.get("end_to_end_total")),
        ("LLM generation", tm.get("llm_generation_time")),
    ]
    if run.get("mode") == "accurate":
        rows.extend([
            ("Initial ack latency", tm.get("initial_ack_latency")),
            ("Full report total", tm.get("accurate_full_report_total_time")),
        ])
    for label, val in rows:
        st.markdown(f"- {label}: `{_fmt_s(val)}`")
    if run.get("mode") == "fast":
        st.markdown("**Pre-LLM breakdown**")
        pre_rows = [
            ("ensure_fast_warm_check_time", tm.get("ensure_fast_warm_check_time")),
            ("rewarm_time", tm.get("rewarm_time")),
            ("ollama_probe_time", tm.get("ollama_probe_time")),
            ("model_match_check_time", tm.get("model_match_check_time")),
            ("prompt_build_time", tm.get("prompt_build_time")),
            ("stream_init_time", tm.get("stream_init_time")),
            ("session_state_update_time", tm.get("session_state_update_time")),
        ]
        for label, val in pre_rows:
            st.markdown(f"- {label}: `{_fmt_s(val)}`")
        rewarm = meta.get("rewarm_triggered")
        if rewarm is not None:
            st.markdown(
                f"- rewarm_triggered: `{rewarm}` · rewarm_reason: `{meta.get('rewarm_reason') or '—'}`"
            )
        preflight = run.get("warm_preflight_before_click") or {}
        if preflight:
            st.markdown("**Warm state before click**")
            st.markdown(
                f"- warm_state_valid_before_click: `{preflight.get('warm_state_valid_before_click')}`\n"
                f"- rewarm_needed_before_click: `{preflight.get('rewarm_needed_before_click')}`\n"
                f"- warmed_model: `{preflight.get('warmed_model') or '—'}`\n"
                f"- warmed_num_ctx: `{preflight.get('warmed_num_ctx') or '—'}`\n"
                f"- time_since_last_warmup: `{preflight.get('time_since_last_warmup')}`s\n"
                f"- keep_alive_valid: `{preflight.get('keep_alive_valid')}`"
            )
        v06 = _fast_v06_checks(tm)
        st.markdown(
            f"- **V06 Fast checks:** retrieval≤0.5s `{v06['retrieval']}` · "
            f"pre-LLM≤0.1s `{v06['pre_llm']}` · LLM TTFT≤1.0s `{v06['llm_ttft']}` · "
            f"first visible≤3.0s `{v06['first_visible']}`"
        )
    st.markdown(f"- **3s criteria:** `{run.get('pass_3s', '—')}`")


def _fast_v06_checks(tm: dict) -> dict[str, str]:
    def _chk(val: float | None, limit: float) -> str:
        if val is None:
            return "—"
        return "PASS" if val <= limit else "FAIL"

    vis = tm.get("answer_first_visible_latency") or tm.get("end_to_end_ttft")
    return {
        "retrieval": _chk(tm.get("retrieval_total_time"), 0.5),
        "pre_llm": _chk(tm.get("pre_llm_latency"), 0.1),
        "llm_ttft": _chk(tm.get("llm_ttft"), 1.0),
        "first_visible": _chk(vis, 3.0),
    }


def _render_warm_sidebar(
    *,
    latency_mode: str,
    llm_model: str,
    bootstrap: dict | None,
) -> None:
    ready = (bootstrap or {}).get("resource_ready") or {}
    warm = get_warm_status_display()
    ui_state = get_fast_warm_ui_state(llm_model)
    st.caption("**Resource / LLM warm status**")
    st.markdown(
        f"- embedding_ready: `{ready.get('embedding_ready', False)}`\n"
        f"- vector_db_ready: `{ready.get('vector_db_ready', False)}`\n"
        f"- metadata_ready: `{ready.get('metadata_ready', False)}`\n"
        f"- llm_fast_warm_ready: `{warm.get('llm_fast_warm_ready')}`\n"
        f"- warmed_model_name: `{warm.get('warmed_model_name') or '—'}`\n"
        f"- warmed_num_ctx: `{warm.get('warmed_num_ctx') or '—'}`\n"
        f"- warmed_api_type: `{warm.get('warmed_api_type') or '—'}`\n"
        f"- warmup_ttft: `{warm.get('warmup_ttft') or '—'}`s\n"
        f"- warmup_total_time: `{warm.get('warmup_total_time') or '—'}`s\n"
        f"- last_warmup_time: `{warm.get('last_warmup_time') or '—'}`"
    )
    st.caption("**Runtime**")
    st.markdown(
        f"- latency mode: `{latency_mode}`\n"
        f"- Fast LLM warm: `{ui_state}`\n"
        f"- last warm-up TTFT: `{warm.get('warmup_ttft') or '—'}`s"
    )
    if latency_mode == "fast":
        live = get_warm_preflight_snapshot(llm_model)
        st.caption("**Warm preflight (before click)**")
        st.markdown(
            f"- warm_state_valid_before_click: `{live.get('warm_state_valid_before_click')}`\n"
            f"- rewarm_needed_before_click: `{live.get('rewarm_needed_before_click')}`\n"
            f"- warmed_model: `{live.get('warmed_model') or '—'}`\n"
            f"- warmed_num_ctx: `{live.get('warmed_num_ctx') or '—'}`\n"
            f"- warmed_api_type: `{live.get('warmed_api_type') or '—'}`\n"
            f"- keep_alive_valid: `{live.get('keep_alive_valid')}`\n"
            f"- accurate_invalidated: `{live.get('accurate_invalidated')}`\n"
            f"- time_since_last_warmup: `{live.get('time_since_last_warmup')}`s"
        )
    _render_last_run_sidebar(latency_mode=latency_mode, llm_model=llm_model)
    preflight = st.session_state.get("warm_preflight_before_click")
    if latency_mode == "fast" and preflight:
        st.caption("**Before last click (warm preflight)**")
        st.markdown(
            f"- warm_state_valid_before_click: `{preflight.get('warm_state_valid_before_click')}`\n"
            f"- rewarm_needed_before_click: `{preflight.get('rewarm_needed_before_click')}`\n"
            f"- warmed_model: `{preflight.get('warmed_model') or '—'}`\n"
            f"- warmed_num_ctx: `{preflight.get('warmed_num_ctx') or '—'}`\n"
            f"- time_since_last_warmup: `{preflight.get('time_since_last_warmup')}`s"
        )
    if ui_state == "expired":
        st.warning("keep_alive 만료 — Fast 답변 전 자동 re-warm 됩니다.")
    elif ui_state == "invalid" and warm.get("last_latency_mode") == "accurate":
        st.warning("Accurate 실행 후 Fast warm 무효 — 답변 생성 시 자동 re-warm 됩니다.")
    elif warm.get("warmed_model_name") and warm.get("warmed_model_name") != llm_model:
        st.warning(f"모델 변경됨 ({warm.get('warmed_model_name')} → {llm_model}) — re-warm 예정")


def _keyword_badge(metrics: dict) -> str:
    kw = metrics.get("keyword_coverage", 0)
    color = "green" if kw >= 0.85 else ("orange" if kw >= 0.7 else "red")
    return f":{color}[keyword_coverage {kw:.0%}]"


def _render_meeting_routing_panel(summary: dict) -> None:
    """Meeting category 1–3 routing metadata (always show when present)."""
    top = summary.get("top_level_category")
    if not top and not summary.get("internal_intent"):
        return
    st.markdown("#### 라우팅 / Coverage (Meeting)")
    lines = [
        f"- **top_level_category:** `{summary.get('top_level_category', '—')}`",
        f"- **internal_intent:** `{summary.get('internal_intent', '—')}`",
        f"- **retrieval_profile:** `{summary.get('retrieval_profile', '—')}`",
        f"- **dense / BM25 / RRF:** "
        f"{summary.get('use_dense', '—')} / {summary.get('use_bm25', '—')} / {summary.get('use_rrf', '—')}",
        f"- **source_tier:** {summary.get('use_source_tier', '—')}",
    ]
    cov = summary.get("coverage_check")
    if cov:
        cov_str = ", ".join(f"{k}={'✓' if v else '✗'}" for k, v in cov.items())
        lines.append(f"- **coverage_check:** {cov_str}")
        lines.append(f"- **coverage_pass:** {summary.get('coverage_pass', '—')}")
    flags = summary.get("warning_flags") or []
    if flags:
        lines.append(f"- **warning_flags:** `{', '.join(flags)}`")
    else:
        lines.append("- **warning_flags:** (없음)")
    variant = summary.get("answer_variant")
    if variant and variant != "default":
        lines.append(f"- **answer_variant:** `{variant}`")
    st.markdown("\n".join(lines))


def _log_trend_ab_feedback(*, qid: str, question: str, choice: str, variant: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": _utc_iso(),
        "question_id": qid,
        "question": question,
        "selected": choice,
        "variant_id": variant.get("variant_id"),
        "answer_preview": (variant.get("answer") or "")[:500],
    }
    with TREND_AB_FEEDBACK_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _apply_trend_ab_selection(*, qid: str, question: str, choice: str) -> None:
    ab = st.session_state.get("general_trend_ab") or {}
    if ab.get("qid") != qid or choice not in ("a", "b"):
        return
    variant = (ab.get("variants") or {}).get(choice) or {}
    st.session_state["general_trend_ab"]["selected"] = choice
    st.session_state["pilot_answer"] = variant.get("answer", "")
    st.session_state["pilot_summary"] = variant.get("summary", {})
    st.session_state["pilot_answer_qid"] = qid
    st.session_state["pilot_answer_mode"] = "trend_ab_llm"
    _log_trend_ab_feedback(qid=qid, question=question, choice=choice, variant=variant)


def _render_trend_ab_picker(*, qid: str, question: str, llm_model: str) -> None:
    ab = st.session_state.get("general_trend_ab") or {}
    if ab.get("qid") != qid:
        return
    variants = ab.get("variants") or {}
    if not variants:
        return

    selected = ab.get("selected")
    meta = st.session_state.get("pilot_run_meta") or {}

    if selected:
        choice = variants.get(selected) or {}
        vid = choice.get("variant_id", "")
        vmeta = VARIANT_META.get(vid, {})
        st.caption(
            f"**A/B 선택 완료** · model={meta.get('model', llm_model)} · "
            f"docs={meta.get('max_docs')} · chunks={meta.get('num_chunks')}"
        )
        st.subheader("선택한 답변")
        st.markdown(f"**{vmeta.get('label_ko', selected.upper())}** · {vmeta.get('subtitle_ko', '')}")
        st.markdown(st.session_state.get("pilot_answer") or choice.get("answer", ""))
        summary = st.session_state.get("pilot_summary") or choice.get("summary") or {}
        if summary.get("top_level_category") or summary.get("internal_intent"):
            with st.expander("라우팅 / Coverage (Meeting)", expanded=True):
                _render_meeting_routing_panel(summary)
        if st.button("다른 답변으로 다시 선택", key=f"trend_ab_repick_{qid}"):
            st.session_state["general_trend_ab"]["selected"] = None
            st.session_state["pilot_answer"] = None
        return

    st.caption(
        f"**A/B 비교 모드** · model={meta.get('model', llm_model)} · "
        f"docs={meta.get('max_docs')} · chunks={meta.get('num_chunks')}"
    )
    st.subheader("두 가지 요약 비교")
    st.markdown("**어느 답변이 마음에 드시나요?** 아래에서 하나를 선택하면 해당 답변만 표시됩니다.")

    col_a, col_b = st.columns(2)
    for key, col in (("a", col_a), ("b", col_b)):
        v = variants.get(key) or {}
        vid = v.get("variant_id", "")
        vmeta = VARIANT_META.get(vid, {})
        with col:
            st.markdown(f"### {vmeta.get('label_ko', key.upper())}")
            st.caption(vmeta.get("subtitle_ko", ""))
            st.info(vmeta.get("description_ko", ""))
            with st.container(border=True):
                st.markdown(v.get("answer", ""))
            for w in v.get("warnings") or []:
                st.warning(w)
            cov = (v.get("meta") or {}).get("coverage_check")
            if cov:
                cov_ok = all(cov.values())
                st.caption(f"coverage: {'PASS' if cov_ok else 'CHECK'}")
            if st.button(
                f"{vmeta.get('label_ko', key.upper())} 선택",
                key=f"trend_ab_pick_{qid}_{key}",
                use_container_width=True,
                type="primary",
            ):
                _apply_trend_ab_selection(qid=qid, question=question, choice=key)


def _run_trend_ab_pipeline(
    *,
    row: dict,
    qid: str,
    latency_mode: str,
    llm_model: str,
    ollama_base: str,
    top_k: int,
    fetch_k: int,
    max_doc: int,
    max_docs: int,
    use_rerank: bool,
    eval_constrained: bool,
    collection,
    embed_model,
    manifest,
    unified_id: str = DEFAULT_UNIFIED,
    apply_session_fn=None,
    result_prefix: str = "pilot",
    temperature: float = 0.15,
) -> dict:
    apply_session = apply_session_fn or _apply_search_session
    preflight = get_warm_preflight_snapshot(llm_model)
    st.session_state["warm_preflight_before_click"] = preflight
    if latency_mode == "fast":
        if preflight.get("rewarm_needed_before_click"):
            ensure_fast_warm(llm_model, ollama_base)
        else:
            ensure_fast_warm_checked(llm_model, ollama_base, allow_rewarm=False)
    user_click_ts = time.time()
    run_id = _new_run_id()
    start_type, run_index = _next_run_context()
    timing = TimingTrace()
    timing.run_id = run_id
    timing.set_user_click(user_click_ts)
    timing.set_cache("llm_server_ready", True)
    timing.meta.update(
        {
            "latency_mode": latency_mode,
            "model_name": llm_model,
            "question_id": qid,
            "ui_flow": "trend_ab_compare",
            "warm_preflight_before_click": preflight,
        }
    )
    st.session_state["pilot_streaming_done"] = True
    thinking_ph = st.empty()
    thinking_cleared: dict = {"done": False}
    _show_thinking(thinking_ph)

    search_out = run_search_inprocess(
        row,
        top_k=top_k,
        fetch_k=fetch_k,
        max_doc=max_doc,
        max_docs=max_docs,
        use_rerank=use_rerank,
        eval_constrained=eval_constrained,
        user_submit_ts=user_click_ts,
        unified_id=unified_id,
        index_dir=DEFAULT_INDEX_DIR,
        chunks_dir=DEFAULT_CHUNKS_DIR,
        collection=collection,
        embed_model=embed_model,
        manifest=manifest,
        start_type=start_type,
        run_index=run_index,
        timing=timing,
        latency_mode=latency_mode,
    )
    _mark_resources_warmed()
    apply_session(qid, search_out, latency_mode)

    retrieved = search_out["retrieved"]
    pool = search_out.get("retrieval_pool") or retrieved
    metrics = search_out.get("retrieval_metrics") or {}
    question = str(row.get("question") or "")

    with st.spinner("답변 A/B 생성 중 (LLM + 검색 근거)…"):
        ab_variants = generate_trend_ab_answers(
            list(retrieved),
            question=question,
            row=row,
            legacy_category=search_out.get("question_category"),
            llm_model=llm_model,
            ollama_base=ollama_base,
            temperature=temperature,
            timing=timing,
        )
    st.session_state["general_trend_ab"] = {
        "qid": qid,
        "variants": ab_variants,
        "selected": None,
    }
    st.session_state[f"{result_prefix}_answer"] = None
    st.session_state[f"{result_prefix}_answer_qid"] = qid
    st.session_state[f"{result_prefix}_answer_mode"] = "trend_ab_llm"
    st.session_state[f"{result_prefix}_summary"] = search_out.get("verification_summary")

    _clear_thinking(thinking_ph, thinking_cleared)

    timing.meta["action"] = "search_and_ab_answers"
    populate_timing_meta(
        timing,
        row=row,
        mode="trend_ab_llm",
        top_k=top_k,
        fetch_k=fetch_k,
        retrieved=retrieved,
        pool=pool,
        answer="",
        action="search_and_ab_answers",
    )
    log_row = timing.to_log_row()
    final_timing = _finalize_and_store_timing({"timing_log": log_row}, user_click_ts)
    tm = (final_timing or log_row).get("timing_metrics") or {}
    _store_last_run(
        run_id=run_id,
        qid=qid,
        mode=latency_mode,
        model_name=llm_model,
        timing_log=final_timing or log_row,
        timing_metrics=tm,
    )
    if st.session_state.get("last_run"):
        st.session_state["last_run"]["warm_preflight_before_click"] = preflight

    st.session_state[f"{result_prefix}_run_meta"] = {
        "source": "ui_trend_ab",
        "run_id": run_id,
        "latency_mode": latency_mode,
        "model": llm_model,
        "top_k": top_k,
        "fetch_k": fetch_k,
        "rerank": use_rerank,
        "max_doc": max_doc,
        "max_docs": max_docs,
        "eval_constrained": eval_constrained,
        "num_chunks": len(retrieved),
        "ab_mode": True,
    }
    st.success(
        f"완료 — {latency_mode} · run_id={run_id} · "
        f"두 가지 LLM 요약(A/B) 생성 · 3s={st.session_state['last_run'].get('pass_3s', '—')}"
    )
    return {
        "answer": "",
        "verification_summary": search_out.get("verification_summary"),
        "retrieval_metrics": metrics,
        "timing_log": final_timing or log_row,
        "ab_variants": ab_variants,
    }


def _render_verification_summary(summary: dict) -> None:
    st.markdown("#### 검증 요약")
    _render_meeting_routing_panel(summary)
    st.markdown("---")
    lines = [
        f"- **질문 카테고리:** {summary.get('question_category_label') or summary.get('question_category', '—')}",
        f"- **답변 모드:** {summary.get('answer_mode', '—')}",
        f"- **검색 모드:** {summary.get('retrieval_mode', '—')}",
        f"- **검색 문서 수:** {summary.get('unique_doc_count', 0)}개 "
        f"(pool {summary.get('pool_unique_doc_count', '—')}개)",
        f"- **최종 사용 문서 수:** {summary.get('final_doc_count', summary.get('unique_doc_count', 0))}개",
        f"- **최종 사용 청크 수:** {summary.get('final_chunk_count', 0)}개",
        f"- **답변 citation 수:** {summary.get('citations_used_count', 0)}개",
        f"- **단일 문서 편중:** {'Yes' if summary.get('single_doc_skew') else 'No'}",
        f"- **must-cover (chunks):** {summary.get('must_cover_in_chunks', '—')}",
        f"- **must-cover (answer):** {summary.get('must_cover_in_answer', '—')}",
        f"- **선급 Rule/Guidance:** {summary.get('class_rule_note', '—')}",
    ]
    for w in summary.get("pipeline_warnings") or summary.get("warnings") or []:
        lines.append(f"- ⚠️ {w}")
    for line in summary.get("timing_summary_lines") or []:
        if line.startswith("**"):
            lines.append("")
        lines.append(line if line.startswith("-") or line.startswith("**") else f"- {line}")
    st.markdown("\n".join(lines))


def _finalize_and_store_timing(out: dict, user_click_ts: float | None) -> dict | None:
    from retrieval_timing import TimingTrace

    final = finalize_ui_timing(out.get("timing_log"), user_click_ts)
    if final:
        append_timing_log(final, DEFAULT_TIMING_LOG)
        if out.get("verification_summary") is not None:
            t = TimingTrace()
            t.wall_clock = final.get("timestamps_wall", {})
            t.monotonic = final.get("timestamps_monotonic", {})
            t.cache_flags = final.get("cache_flags", {})
            t.warnings = final.get("warnings", [])
            t.debug_notes = final.get("debug_notes", [])
            out["verification_summary"]["timing_metrics"] = final["timing_metrics"]
            out["verification_summary"]["timing_summary_lines"] = t.summary_lines()
    return final


def _judge_initial_3s(latency: float | None) -> str:
    if latency is None:
        return "—"
    return "PASS" if latency <= 3.0 else "FAIL"


def _render_on_target(container, render_fn) -> None:
    """Render on st module, placeholder, or block container without `with st`."""
    if container is st:
        render_fn(st)
        return
    if hasattr(container, "__enter__") and hasattr(container, "__exit__"):
        with container:
            render_fn(st)
        return
    if hasattr(container, "markdown"):
        render_fn(container)
        return
    render_fn(st)


def _render_evidence_table_into(container, rows: list[dict], title: str = "Evidence Table") -> None:
    if not rows:
        return

    def _draw(target) -> None:
        target.markdown(f"> **{title}**")
        df = pd.DataFrame(rows)
        answer_columns = [
            "citation_id",
            "file_name",
            "page",
            "chunk_id",
            "chunk_preview",
        ]
        debug_columns = [
            "rank",
            "citation_id",
            "chunk_type",
            "table_id",
            "page",
            "caption",
            "row_index",
            "matched_columns",
            "source",
            "file_name",
            "score",
            "dense_score",
            "bm25_score",
            "rrf_score",
            "metadata_boost",
            "source_priority_score",
            "is_catalog_table",
            "used_in_answer",
            "chunk_preview",
        ]
        col_order = debug_columns if title.startswith("검색") else answer_columns
        show = [c for c in col_order if c in df.columns]
        visible = df[show].rename(
            columns={
                "citation_id": "각주",
                "file_name": "문서",
                "page": "페이지",
                "chunk_id": "청크 ID",
                "chunk_preview": "인용 근거 청크",
            }
        )
        target.dataframe(visible, use_container_width=True, hide_index=True)

    _render_on_target(container, _draw)


def _render_evidence_table(rows: list[dict], title: str = "Evidence Table") -> None:
    _render_evidence_table_into(st, rows, title=title)


def _render_must_cover_into(container, rows: list[dict]) -> None:
    if not rows:
        return

    def _draw(target) -> None:
        target.markdown("#### Must-cover coverage")
        df = pd.DataFrame(rows)
        target.dataframe(df, use_container_width=True, hide_index=True)

    _render_on_target(container, _draw)


def _render_must_cover(rows: list[dict]) -> None:
    _render_must_cover_into(st, rows)


def _fast_path_requires_llm(row: dict, *, unified_id: str) -> bool:
    """Return False when Fast can answer deterministically without Ollama."""
    if unified_id != DEFAULT_UNIFIED:
        return True

    from meeting_category_profile import uses_structured_meeting_answer
    from rag_query_router import resolve_pipeline_route

    question = str(row.get("question") or "")
    route = resolve_pipeline_route(question, row)
    if route.get("rule_guidance_lookup"):
        return False
    work = {**row, "category": route.get("question_category") or row.get("category") or ""}
    return not uses_structured_meeting_answer(
        work,
        legacy_category=str(route.get("question_category") or ""),
    )


def _run_single_click_pipeline(
    *,
    row: dict,
    qid: str,
    latency_mode: str,
    llm_model: str,
    ollama_base: str,
    top_k: int,
    fetch_k: int,
    max_doc: int,
    max_docs: int,
    use_rerank: bool,
    eval_constrained: bool,
    temperature: float,
    multi_doc_strategy: str,
    max_llm_docs: int,
    collection,
    embed_model,
    manifest,
    unified_id: str = DEFAULT_UNIFIED,
    apply_session_fn=None,
    result_prefix: str = "pilot",
    fast_answer_ph=None,
    defer_session_apply: bool = False,
) -> dict:
    active_chunks_dir = TABLE_QA_CHUNKS_DIR if unified_id == TABLE_QA_UNIFIED else DEFAULT_CHUNKS_DIR
    apply_session = apply_session_fn or _apply_search_session
    defer_session = defer_session_apply or (
        apply_session_fn is not None and apply_session_fn is not _apply_search_session
    )
    st.session_state["general_trend_ab"] = None
    preflight = get_warm_preflight_snapshot(llm_model)
    st.session_state["warm_preflight_before_click"] = preflight
    if latency_mode == "fast" and _fast_path_requires_llm(row, unified_id=unified_id):
        if preflight.get("rewarm_needed_before_click"):
            ensure_fast_warm(llm_model, ollama_base)
        else:
            ensure_fast_warm_checked(llm_model, ollama_base, allow_rewarm=False)
    user_click_ts = time.time()
    run_id = _new_run_id()
    start_type, run_index = _next_run_context()
    timing = TimingTrace()
    timing.run_id = run_id
    timing.set_user_click(user_click_ts)
    timing.set_cache("llm_server_ready", True)
    timing.meta.update(
        {
            "latency_mode": latency_mode,
            "model_name": llm_model,
            "question_id": qid,
            "ui_flow": "single_click",
            "warm_preflight_before_click": preflight,
        }
    )
    st.session_state["pilot_streaming_done"] = False
    thinking_ph = st.empty()
    thinking_cleared: dict = {"done": False}
    _show_thinking(thinking_ph)

    if latency_mode == "accurate":
        report_ph = st.empty()
        evidence_slot = st.container()
        coverage_slot = st.container()
        mark_accurate_initial_ack(timing)

    if latency_mode == "fast":
        answer_ph = fast_answer_ph if fast_answer_ph is not None else st.empty()
        streamed: list[str] = []

        def _on_fast_token(tok: str) -> None:
            _clear_thinking(thinking_ph, thinking_cleared)
            streamed.append(tok)
            answer_ph.markdown("".join(streamed))

        full_out = run_full_inprocess(
            row,
            top_k=top_k,
            fetch_k=fetch_k,
            max_doc=max_doc,
            max_docs=max_docs,
            use_rerank=use_rerank,
            eval_constrained=eval_constrained,
            llm_model=llm_model,
            ollama_base=ollama_base,
            temperature=temperature,
            multi_doc_strategy=multi_doc_strategy,
            max_llm_docs=max_llm_docs,
            user_submit_ts=user_click_ts,
            unified_id=unified_id,
            index_dir=DEFAULT_INDEX_DIR,
            chunks_dir=active_chunks_dir,
            collection=collection,
            embed_model=embed_model,
            manifest=manifest,
            start_type=start_type,
            run_index=run_index,
            latency_mode="fast",
            on_token=_on_fast_token,
            auto_llm_warm=False,
            skip_ollama_probe=True,
            timing=timing,
        )
        search_out = full_out.get("search_out") or {}
        out = dict(full_out.get("answer_out") or {})
        out["answer"] = full_out.get("answer") or out.get("answer", "")
        if not out.get("evidence_table") and full_out.get("answer_out", {}).get("evidence_table"):
            out["evidence_table"] = full_out["answer_out"]["evidence_table"]
        _mark_resources_warmed()
        if not defer_session:
            apply_session(qid, search_out, latency_mode)
        mode = search_out.get("answer_mode", "standard_rag")
        retrieved = search_out["retrieved"]
        pool = search_out.get("retrieval_pool") or retrieved
        metrics = search_out.get("retrieval_metrics") or {}
        _clear_thinking(thinking_ph, thinking_cleared)
    else:
        search_out = run_search_inprocess(
            row,
            top_k=top_k,
            fetch_k=fetch_k,
            max_doc=max_doc,
            max_docs=max_docs,
            use_rerank=use_rerank,
            eval_constrained=eval_constrained,
            user_submit_ts=user_click_ts,
            unified_id=unified_id,
            index_dir=DEFAULT_INDEX_DIR,
            chunks_dir=active_chunks_dir,
            collection=collection,
            embed_model=embed_model,
            manifest=manifest,
            start_type=start_type,
            run_index=run_index,
            timing=timing,
            latency_mode=latency_mode,
        )
        _mark_resources_warmed()
        if not defer_session:
            apply_session(qid, search_out, latency_mode)
        mode = search_out.get("answer_mode", "standard_rag")
        retrieved = search_out["retrieved"]
        pool = (
            retrieved
            if mode == "multi_doc_summary" and latency_mode == "accurate"
            else search_out.get("retrieval_pool") or retrieved
        )
        metrics = search_out.get("retrieval_metrics") or {}
        streamed_acc: list[str] = []

        def _on_accurate_token(tok: str) -> None:
            _clear_thinking(thinking_ph, thinking_cleared)
            streamed_acc.append(tok)
            report_ph.markdown("".join(streamed_acc))

        out = run_answer_inprocess(
            row=row,
            chunks=retrieved,
            pool=pool,
            config_dict=search_out.get("retrieval_config") or {},
            metrics=metrics,
            doc_groups=search_out.get("doc_groups") or [],
            answer_mode=mode,
            question_category=search_out.get("question_category") or "",
            llm_model=llm_model,
            ollama_base=ollama_base,
            temperature=temperature,
            save_trace=True,
            multi_doc_strategy=multi_doc_strategy,
            max_llm_docs=max_llm_docs,
            top_k=top_k,
            fetch_k=fetch_k,
            user_submit_ts=user_click_ts,
            start_type=start_type,
            run_index=run_index,
            latency_mode=latency_mode,
            on_token=_on_accurate_token,
            timing=timing,
            mark_initial_ack=False,
        )
        _clear_thinking(thinking_ph, thinking_cleared)
        if out.get("answer"):
            report_ph.markdown(out["answer"])
        _render_evidence_table_into(
            evidence_slot.empty(),
            out.get("evidence_table") or [],
            title="Evidence Table",
        )
        _render_must_cover_into(coverage_slot.empty(), out.get("must_cover_coverage") or [])
        st.session_state["pilot_streaming_done"] = True

    timing.meta["action"] = "search_and_answer"
    populate_timing_meta(
        timing,
        row=row,
        mode="fast_rag" if latency_mode == "fast" else mode,
        top_k=top_k,
        fetch_k=fetch_k,
        retrieved=retrieved,
        pool=pool,
        answer=out["answer"],
        action="search_and_answer",
    )
    log_row = timing.to_log_row()
    out["timing_log"] = log_row
    final_timing = _finalize_and_store_timing(out, user_click_ts)
    tm = (final_timing or log_row).get("timing_metrics") or {}
    _store_last_run(
        run_id=run_id,
        qid=qid,
        mode=latency_mode,
        model_name=llm_model,
        timing_log=final_timing or log_row,
        timing_metrics=tm,
    )
    if st.session_state.get("last_run"):
        st.session_state["last_run"]["warm_preflight_before_click"] = preflight
    answer_text = str(out.get("answer") or "")
    st.session_state[f"{result_prefix}_answer"] = answer_text
    st.session_state[f"{result_prefix}_answer_qid"] = qid
    prev_evidence = st.session_state.get(f"{result_prefix}_evidence") or []
    new_evidence = out.get("evidence_table") or []
    st.session_state[f"{result_prefix}_evidence"] = new_evidence or prev_evidence
    st.session_state[f"{result_prefix}_citation_map"] = out.get("answer_citation_mapping")
    st.session_state[f"{result_prefix}_must_cover"] = out.get("must_cover_coverage")
    st.session_state[f"{result_prefix}_summary"] = out.get("verification_summary")
    st.session_state[f"{result_prefix}_run_meta"] = {
        "source": "ui_single_click",
        "run_id": run_id,
        "latency_mode": latency_mode,
        "model": llm_model,
        "top_k": top_k,
        "fetch_k": fetch_k,
        "rerank": use_rerank,
        "max_doc": max_doc,
        "max_docs": max_docs,
        "eval_constrained": eval_constrained,
        "temperature": temperature,
        "num_chunks": len(retrieved),
        "multi_doc_strategy": multi_doc_strategy,
        "max_llm_docs": max_llm_docs,
        "prompt_meta": out.get("prompt_meta"),
    }
    if defer_session:
        final_session_out = {
            **search_out,
            **out,
            "retrieved": search_out.get("retrieved") or retrieved,
            "retrieval_pool": search_out.get("retrieval_pool") or pool,
            "answer": answer_text,
        }
        apply_session(qid, final_session_out, latency_mode)
    if latency_mode == "accurate":
        # The placeholders above are only for progress/streaming.  The normal
        # result panel below renders the persisted answer and evidence once.
        report_ph.empty()
        evidence_slot.empty()
        coverage_slot.empty()
    st.success("답변 생성 완료")
    if not answer_text.strip():
        st.warning(
            "LLM 답변이 비어 있습니다. 사이드바 Ollama 연결·모델명을 확인하고 "
            "「리소스 미리 로드」를 다시 실행해 보세요."
        )
    return out


def _render_table_retrieval_debug(debug: dict | None) -> None:
    if not debug:
        return
    st.markdown("#### 표 검색 디버그 (schema 2-stage)")
    st.json(debug.get("parsed_query") or {})
    conf = debug.get("retrieval_confidence")
    gate = debug.get("passes_confidence_gate")
    c1, c2, c3 = st.columns(3)
    c1.metric("confidence", f"{conf:.3f}" if conf is not None else "—")
    c2.metric("confidence gate", "PASS" if gate else "FAIL")
    c3.metric("selected table", (debug.get("selected_table_id") or "—")[:40])
    if debug.get("matched_row") or debug.get("matched_column"):
        st.caption(
            f"matched row: `{debug.get('matched_row') or '—'}` · "
            f"matched column: `{debug.get('matched_column') or '—'}`"
        )
    candidates = debug.get("selected_table_candidates") or []
    if candidates:
        rows = []
        for i, c in enumerate(candidates, 1):
            rows.append(
                {
                    "rank": i,
                    "table_id": c.get("table_id", ""),
                    "combined": c.get("combined_score"),
                    "topic": c.get("table_topic_match"),
                    "column": c.get("column_match"),
                    "row": c.get("row_entity_match"),
                    "vector": c.get("vector_distance"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_table_chunk_hits(pool: list[dict], *, limit: int = 10) -> None:
    if not pool:
        return
    rows: list[dict] = []
    for i, ch in enumerate(pool[:limit], start=1):
        rows.append(
            {
                "rank": i,
                "chunk_type": ch.get("chunk_type") or ch.get("element_type") or "",
                "table_id": ch.get("table_id") or "",
                "page": ch.get("page_number"),
                "row_index": ch.get("row_index"),
                "chunk_id": ch.get("chunk_id") or "",
                "preview": (ch.get("text") or "")[:160],
            }
        )
    st.markdown("#### 검색된 표 청크 (top)")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _qid_from_question(question: str, *, prefix: str = "ui") -> str:
    q = question.strip()
    if not q:
        return ""
    digest = hashlib.sha256(q.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _row_from_user_question(question: str) -> dict:
    from rag_query_router import enrich_row_for_routing

    q = question.strip()
    base = {"question_id": _qid_from_question(q), "question": q}
    return enrich_row_for_routing(base, latency_mode="accurate")


def _preview_query_routing(question: str, row: dict) -> dict:
    """Classify any free-text question into 4 UI categories + downstream routing."""
    from meeting_trend_ab import is_trend_summary_ab_eligible
    from meeting_category_profile import uses_structured_meeting_answer
    from rag_query_router import resolve_pipeline_route

    route = resolve_pipeline_route(question, row)
    work = {**row, "category": route["question_category"]}
    return {
        "question_category": route["question_category"],
        "question_category_label": route["question_category_label"],
        "top_level_category": route["top_level_category"],
        "top_level_label": route["top_level_label"],
        "internal_intent": route["internal_intent"],
        "retrieval_profile_id": route["selected_retrieval_profile"],
        "answer_mode": route["selected_answer_mode"],
        "retrieval_label": route["selected_retrieval_label"],
        "detected_society": route.get("detected_society"),
        "hard_society_filter": route.get("hard_society_filter"),
        "structured_meeting": uses_structured_meeting_answer(work, legacy_category=route["question_category"]),
        "trend_ab_eligible": is_trend_summary_ab_eligible(question, work),
    }


def _render_query_routing_preview(question: str, row: dict) -> None:
    routing = _preview_query_routing(question, row)
    st.markdown("#### 질문 라우팅 (자동 분류)")
    st.caption(
        "드롭다운 7개 예시와 무관하게, **입력한 질문**을 키워드·패턴으로 "
        "4유형(동향/환경/자율운항/Rule) 중 하나로 분류해 검색·답변 경로를 선택합니다."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("카테고리 (4유형)", routing["question_category_label"])
    c2.metric("검색 프로필", routing["retrieval_label"])
    c3.metric("답변 모드", routing["answer_mode"])
    extras: list[str] = [
        f"- **top_level:** {routing['top_level_label']} (`{routing['top_level_category']}`)",
        f"- **internal_intent:** `{routing['internal_intent']}`",
        f"- **meeting retrieval:** `{routing['retrieval_profile_id']}`",
    ]
    if routing.get("detected_society"):
        extras.append(f"- **detected_society:** `{routing['detected_society']}` · hard_filter: `{routing.get('hard_society_filter')}`")
    if routing["structured_meeting"]:
        extras.append("- **structured meeting:** evidence 4섹션 (LLM 없음)")
    if routing["trend_ab_eligible"]:
        extras.append("- **A/B 비교 UI:** 적용 (최신 동향 요약형)")
    st.markdown("\n".join(extras))


def _render_table_qa_tab(
    *,
    latency_mode: str,
    llm_model: str,
    ollama_base: str,
    top_k: int,
    fetch_k: int,
    max_doc: int,
    max_docs: int,
    use_rerank: bool,
    temperature: float,
    multi_doc_strategy: str,
    max_llm_docs: int,
    table_eval_constrained: bool,
    table_top_k_eval: int,
) -> None:
    st.markdown(
        f"**코퍼스:** `{TABLE_QA_UNIFIED}` (KR 표 상위 22건 · 구조화 표 청크)  \n"
        "KR Rule·지침 표·수치 규정에 대해 **실무형 질문**을 입력하면, 표 검색 후 reporting·검사 요건을 문장으로 답합니다."
    )

    question_text = st.text_area(
        "질문",
        key="table_custom_text",
        height=120,
        placeholder=(
            "예: 선령 5~10년 선박의 평형수탱크는 정기검사에서 어떤 방식으로 선정·검사하나?\n"
            "예: 빌지저장탱크는 정기검사 1~4차에서 검사 보고 대상인가?"
        ),
    )
    with st.expander("질문 예시 (클릭하면 입력란에 채워집니다)", expanded=False):
        for i, example in enumerate(TABLE_QA_EXAMPLE_QUESTIONS):
            st.button(
                example,
                key=f"table_example_{i}",
                use_container_width=True,
                on_click=_set_text_area_value,
                args=("table_custom_text", example),
            )

    row: dict | None = None
    qid = ""
    if question_text.strip():
        row = normalize_table_question_row(
            {
                "qid": _qid_from_question(question_text, prefix="table"),
                "question": question_text.strip(),
            }
        )
        qid = str(row.get("question_id") or row.get("qid") or "")
    else:
        st.info("질문을 입력한 뒤 아래 버튼으로 검색·답변을 실행하세요.")
        return

    run_btn = st.button(
        "표 QA — 검색 및 답변 생성",
        type="primary",
        use_container_width=True,
        key="table_run",
    )
    answer_ph = st.empty()

    def _table_apply(qid_: str, out: dict, mode: str) -> None:
        out = dict(out)
        pool_objs = out.get("retrieval_pool") or out.get("retrieved") or []
        if not out.get("evidence_table") and pool_objs:
            from retrieval_verification import build_evidence_table

            out["evidence_table"] = build_evidence_table(pool_objs)
        eval_metrics = (
            evaluate_table_qa_retrieval(pool_objs, row, k=table_top_k_eval)
            if row.get("gold_table_id") or row.get("gold_page")
            else {}
        )
        _apply_table_session(qid_, out, mode, eval_metrics)

    if run_btn:
        collection, embed_model, manifest = cached_rag_resources(
            TABLE_QA_UNIFIED, str(DEFAULT_INDEX_DIR), _index_fingerprint(TABLE_QA_UNIFIED)
        )
        try:
            with st.spinner("표 QA 검색·LLM 답변 생성 중 (수십 초 소요)…"):
                _run_single_click_pipeline(
                    row=row,
                    qid=qid,
                    latency_mode=latency_mode,
                    llm_model=llm_model,
                    ollama_base=ollama_base,
                    top_k=top_k,
                    fetch_k=fetch_k,
                    max_doc=max_doc,
                    max_docs=max_docs,
                    use_rerank=use_rerank,
                    eval_constrained=table_eval_constrained,
                    temperature=temperature,
                    multi_doc_strategy=multi_doc_strategy,
                    max_llm_docs=max_llm_docs,
                    collection=collection,
                    embed_model=embed_model,
                    manifest=manifest,
                    unified_id=TABLE_QA_UNIFIED,
                    apply_session_fn=_table_apply,
                    result_prefix="table",
                    fast_answer_ph=answer_ph,
                    defer_session_apply=True,
                )
        except Exception as exc:
            st.error(f"실행 실패: {exc}")
            import traceback

            st.code(traceback.format_exc())

    show_search = st.checkbox("검색 결과 보기", value=True, key="table_show_search")
    if (
        show_search
        and st.session_state.get("table_retrieved_qid") == qid
        and st.session_state.get("table_pool")
    ):
        eval_m = st.session_state.get("table_eval_metrics") or {}
        if eval_m:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"table_recall@{table_top_k_eval}", "YES" if eval_m.get("table_recall@k") else "NO")
            c2.metric(f"row_recall@{table_top_k_eval}", "YES" if eval_m.get("row_recall@k") else "NO")
            c3.metric("cell_exact_match", "YES" if eval_m.get("cell_exact_match") else "NO")
            c4.metric("citation_match", "YES" if eval_m.get("citation_match") else "NO")
        _render_table_retrieval_debug(st.session_state.get("table_retrieval_debug"))
        _render_table_chunk_hits(st.session_state.get("table_pool") or [], limit=table_top_k_eval)

    if st.session_state.get("table_answer_qid") == qid:
        meta = st.session_state.get("table_run_meta") or {}
        answer_text = str(st.session_state.get("table_answer") or "")
        st.caption(
            f"**표 QA** · corpus={TABLE_QA_UNIFIED} · eval_constrained={table_eval_constrained} · "
            f"mode={meta.get('latency_mode', latency_mode)}"
        )
        st.subheader("답변")
        if answer_text.strip():
            st.markdown(answer_text)
        else:
            st.warning(
                "답변이 비어 있습니다. Ollama가 실행 중인지, 사이드바 모델명이 맞는지 확인하세요. "
                "검색 결과는 아래에 표시됩니다."
            )

        evidence = st.session_state.get("table_evidence") or []
        if evidence:
            _render_evidence_table(evidence, title="Evidence Table")
        elif st.session_state.get("table_pool"):
            st.caption("Fast 모드: 슬롯 청크는 위 검색 결과를 참고하세요.")

        if st.button("이 답변을 표 QA UI 로그에 저장", key=f"table_save_{qid}"):
            TABLE_QA_LOG_DIR.mkdir(parents=True, exist_ok=True)
            eval_m = st.session_state.get("table_eval_metrics") or {}
            block = "\n".join(
                [
                    f"## {qid} (Table UI {datetime.now(timezone.utc).isoformat()})",
                    f"- question: {row.get('question', '')}",
                    f"- eval: {json.dumps(eval_m, ensure_ascii=False)}",
                    f"- settings: {json.dumps(meta, ensure_ascii=False)}",
                    "",
                    st.session_state["table_answer"],
                    "",
                    "---",
                    "",
                ]
            )
            mode = "a" if TABLE_UI_SAVED_ANSWERS.exists() else "w"
            header = "# Table QA UI Answers\n\n" if mode == "w" else ""
            with TABLE_UI_SAVED_ANSWERS.open(mode, encoding="utf-8") as f:
                f.write(header + block)
            st.success(f"저장: `{TABLE_UI_SAVED_ANSWERS}`")


def main() -> None:
    st.set_page_config(page_title="MaritimeRAG Pilot", page_icon="⚓", layout="wide")
    st.title("MaritimeRAG — 검색·답변 검증")
    st.caption("Open retrieval + multi-doc diversity · 근거 추적 가능")

    st.session_state.setdefault("pilot_retrieved", None)
    st.session_state.setdefault("pilot_retrieved_qid", None)
    st.session_state.setdefault("pilot_metrics", None)
    st.session_state.setdefault("pilot_config", None)
    st.session_state.setdefault("pilot_evidence", None)
    st.session_state.setdefault("pilot_must_cover", None)
    st.session_state.setdefault("pilot_summary", None)
    st.session_state.setdefault("pilot_answer", None)
    st.session_state.setdefault("pilot_answer_qid", None)
    st.session_state.setdefault("pilot_run_meta", None)
    st.session_state.setdefault("pilot_pool", None)
    st.session_state.setdefault("pilot_doc_groups", None)
    st.session_state.setdefault("pilot_answer_mode", None)
    st.session_state.setdefault("pilot_question_category", None)
    st.session_state.setdefault("pilot_latency_mode", "fast")
    st.session_state.setdefault("pilot_streaming_done", False)
    st.session_state.setdefault("general_trend_ab", None)
    st.session_state.setdefault("last_run", None)
    st.session_state.setdefault("debug_search_only_log", None)
    st.session_state.setdefault("table_retrieved", None)
    st.session_state.setdefault("table_pool", None)
    st.session_state.setdefault("table_retrieved_qid", None)
    st.session_state.setdefault("table_metrics", None)
    st.session_state.setdefault("table_eval_metrics", None)
    st.session_state.setdefault("table_evidence", None)
    st.session_state.setdefault("table_answer", None)
    st.session_state.setdefault("table_answer_qid", None)
    st.session_state.setdefault("table_run_meta", None)
    st.session_state.setdefault("table_latency_mode", "fast")

    # Auto bootstrap on app start (cached per model/base/index)
    index_fp = _index_fingerprint(DEFAULT_UNIFIED)
    bootstrap = cached_app_bootstrap(
        DEFAULT_OLLAMA_MODEL,
        DEFAULT_OLLAMA_BASE,
        DEFAULT_UNIFIED,
        str(DEFAULT_INDEX_DIR),
        index_fp,
    )
    st.session_state["app_bootstrap"] = bootstrap
    _mark_resources_warmed()
    st.session_state["llm_server_ready"] = True

    with st.sidebar:
        st.header("설정")
        latency_mode = st.radio(
            "Latency mode",
            options=["fast", "accurate"],
            index=0,
            format_func=lambda x: {
                "fast": "Fast — TTFT 3초 목표, 짧은 context/답변",
                "accurate": "Accurate — 3초 이내 첫 답변 + 넓은 근거 검색",
            }[x],
            help="Fast: top_k=3, fetch_k=10, streaming. Accurate: 기존 파일럿 설정.",
        )
        st.selectbox("Corpus", [DEFAULT_UNIFIED], index=0)
        llm_model = st.text_input("Ollama 모델", value=DEFAULT_OLLAMA_MODEL)
        ollama_base = st.text_input("Ollama URL", value=DEFAULT_OLLAMA_BASE)
        if latency_mode == "accurate":
            st.caption(ACCURATE_MODE_BANNER)
            st.caption(
                "질문 유형에 따라 fetch_k·max_docs가 **자동 조정**됩니다 "
                "(Rule 조회=좁게, MEPC/MSC 요약=넓게)."
            )
            top_k = st.slider("top_k", 3, 12, 10)
            fetch_k = st.slider("fetch_k (broad pool)", 40, 200, 120)
            use_rerank = st.checkbox("Diversity rerank (standard RAG)", value=True)
            max_doc = st.slider("max_chunks_per_doc", 1, 8, 3, disabled=not use_rerank)
            max_docs = st.slider("max_docs (broad)", 3, 15, 10)
            multi_doc_strategy = st.selectbox(
                "Multi-doc 속도",
                options=["single_pass", "batched", "full"],
                index=0,
                format_func=lambda x: {
                    "single_pass": "빠름 (~3–6분, LLM 1회)",
                    "batched": "균형 (~6–10분, 배치 mini-summary)",
                    "full": "상세 (~15분+, 문서별 mini-summary)",
                }[x],
            )
            max_llm_docs = st.slider("LLM 사용 문서 수", 3, 12, 6)
        else:
            st.caption(
                "Fast retrieval: top_k=3 · fetch_k=10 · max_docs=2 · "
                "max_chunks_per_doc=1 · rerank off"
            )
            top_k, fetch_k, max_doc, max_docs = 3, 10, 1, 2
            use_rerank = False
            multi_doc_strategy = "single_pass"
            max_llm_docs = 2
        eval_constrained = st.checkbox(
            "Eval constrained (gold doc 필터)",
            value=False,
            help="체크 시 gold_doc_id 내부만 검색 — eval용. 기본은 open retrieval.",
        )
        st.divider()
        st.markdown(f"**표 QA (`{TABLE_QA_UNIFIED}`)**")
        table_eval_constrained = st.checkbox(
            "표 QA: gold doc 필터 (eval)",
            value=False,
            help="벤치 eval용. 체크 시 질문 row의 gold_doc_id로 검색 범위를 제한합니다.",
        )
        table_top_k_eval = st.slider("표 QA retrieval 평가 top-k", 3, 15, 10)
        temperature = st.slider("Temperature", 0.0, 0.5, 0.15, 0.05)

        ok, msg = check_ollama_model(ollama_base, llm_model)
        st.success(f"Ollama · {llm_model}") if ok else st.warning(msg)

        st.divider()
        if st.button("리소스 미리 로드 (warm)", help="자동 warm-up 실패 시 수동 재실행"):
            with st.spinner("ChromaDB + manifest + LLM Fast warm…"):
                cached_rag_resources(DEFAULT_UNIFIED, str(DEFAULT_INDEX_DIR), _index_fingerprint(DEFAULT_UNIFIED))
                cached_rag_resources(TABLE_QA_UNIFIED, str(DEFAULT_INDEX_DIR), _index_fingerprint(TABLE_QA_UNIFIED))
                b = bootstrap_all_resources(
                    llm_model, ollama_base, DEFAULT_UNIFIED, DEFAULT_INDEX_DIR, force_llm_warm=True
                )
                _mark_resources_warmed()
                st.session_state["app_bootstrap"] = b
            st.success("Vector index + LLM Fast warm 완료")

        _render_warm_sidebar(
            latency_mode=latency_mode,
            llm_model=llm_model,
            bootstrap=st.session_state.get("app_bootstrap"),
        )

        if llm_model != DEFAULT_OLLAMA_MODEL:
            cached_app_bootstrap(llm_model, ollama_base, DEFAULT_UNIFIED, str(DEFAULT_INDEX_DIR), index_fp)

        st.divider()
        st.markdown(
            "**사용 순서**\n"
            f"1. **일반 QA** — IMO {DEFAULT_UNIFIED}에 질문 입력\n"
            "2. **표 QA** — KR 표 상위 22건 코퍼스에 질문 입력\n"
            "3. Latency breakdown에서 run_id 확인\n\n"
            "로그: `retrieval_timing_trace.jsonl`"
        )

    tab_general, tab_table, tab_saved = st.tabs(["일반 QA", "표 QA (KR 22건)", "저장된 답변"])

    with tab_general:
        st.markdown(
            f"**코퍼스:** `{DEFAULT_UNIFIED}` (IMO 회의·선급 Rule 등)  \n"
            "질문을 입력하면 검색 후 Fast/Accurate 모드로 답변합니다."
        )
        general_question = st.text_area(
            "질문",
            key="general_custom_text",
            height=120,
            placeholder=(
                "예: MSC 111의 주요 결과를 3개 항목으로 요약해줘.\n"
                "예: 최신 MEPC 회의에서 선박 운항·규제 보고에 영향을 주는 사항을 정리해줘."
            ),
        )
        with st.expander("질문 예시", expanded=False):
            _render_general_qa_example_picker("general_custom_text")

        if not general_question.strip():
            st.info("질문을 입력한 뒤 아래 버튼으로 검색·답변을 실행하세요.")
        else:
            row = _row_from_user_question(general_question)
            qid = str(row["question_id"])

            run_btn = st.button("검색 및 답변 생성", type="primary", use_container_width=True)

            if run_btn:
                collection, embed_model, manifest = cached_rag_resources(
                    DEFAULT_UNIFIED, str(DEFAULT_INDEX_DIR), _index_fingerprint(DEFAULT_UNIFIED)
                )
                try:
                    _run_single_click_pipeline(
                        row=row,
                        qid=qid,
                        latency_mode=latency_mode,
                        llm_model=llm_model,
                        ollama_base=ollama_base,
                        top_k=top_k,
                        fetch_k=fetch_k,
                        max_doc=max_doc,
                        max_docs=max_docs,
                        use_rerank=use_rerank,
                        eval_constrained=eval_constrained,
                        temperature=temperature,
                        multi_doc_strategy=multi_doc_strategy,
                        max_llm_docs=max_llm_docs,
                        collection=collection,
                        embed_model=embed_model,
                        manifest=manifest,
                    )
                except Exception as exc:
                    st.error(f"실행 실패: {exc}")

            with st.sidebar.expander("고급 디버그 옵션"):
                debug_search_btn = st.button("검색만 실행 (E2E 판정 제외)", key="debug_search_only")
                show_search = st.checkbox("검색 결과 보기", value=False, key="debug_show_search")
                show_debug_timing = st.checkbox("retrieval timing 보기 (debug)", value=False, key="debug_timing")

                if debug_search_btn:
                    dbg_ts = time.time()
                    start_type, run_index = _next_run_context()
                    collection, embed_model, manifest = cached_rag_resources(
                        DEFAULT_UNIFIED, str(DEFAULT_INDEX_DIR), _index_fingerprint()
                    )
                    dbg_timing = TimingTrace()
                    dbg_timing.set_user_click(dbg_ts)
                    dbg_timing.meta["ui_flow"] = "debug_search_only"
                    dbg_timing.meta["latency_mode"] = latency_mode
                    try:
                        out = run_search_inprocess(
                            row,
                            top_k=top_k,
                            fetch_k=fetch_k,
                            max_doc=max_doc,
                            max_docs=max_docs,
                            use_rerank=use_rerank,
                            eval_constrained=eval_constrained,
                            user_submit_ts=dbg_ts,
                            unified_id=DEFAULT_UNIFIED,
                            index_dir=DEFAULT_INDEX_DIR,
                            chunks_dir=DEFAULT_CHUNKS_DIR,
                            collection=collection,
                            embed_model=embed_model,
                            manifest=manifest,
                            start_type=start_type,
                            run_index=run_index,
                            timing=dbg_timing,
                            latency_mode=latency_mode,
                        )
                        _mark_resources_warmed()
                        _apply_search_session(qid, out, latency_mode)
                        st.session_state["pilot_answer"] = None
                        st.session_state["pilot_answer_qid"] = None
                        st.session_state["general_trend_ab"] = None
                        st.session_state["pilot_streaming_done"] = False
                        st.session_state["debug_search_only_log"] = dbg_timing.to_log_row()
                        st.info("검색만 완료 — E2E TTFT 판정에 사용되지 않습니다.")
                    except Exception as exc:
                        st.error(f"검색 실패: {exc}")

                if show_debug_timing and st.session_state.get("debug_search_only_log"):
                    st.json(st.session_state["debug_search_only_log"].get("timing_metrics", {}))

            if (
                st.session_state.get("pilot_retrieved_qid") == qid
                and st.session_state.get("pilot_metrics")
                and st.session_state.get("debug_show_search")
            ):
                m = st.session_state["pilot_metrics"]
                is_accurate = st.session_state.get("pilot_latency_mode") == "accurate"
                if is_accurate:
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("doc_recall@5", "YES" if m.get("doc_recall_at_5") else "NO")
                    c2.metric("page_recall@5", "YES" if m.get("page_recall_at_5") else "NO")
                    c3.metric("unique_docs", m.get("unique_doc_count", 0))
                    c4.metric("topic@k", "YES" if m.get("topic_hit_at_k") else "NO")
                    c5.markdown(_keyword_badge(m))
                else:
                    st.caption(
                        f"Fast 검색 — {m.get('unique_doc_count', len(st.session_state.get('pilot_retrieved') or []))} docs · "
                        f"{len(st.session_state.get('pilot_retrieved') or [])} chunks"
                    )

                if st.session_state.get("pilot_summary"):
                    _render_verification_summary(st.session_state["pilot_summary"])

                if is_accurate and not st.session_state.get("pilot_streaming_done"):
                    _render_evidence_table(
                        st.session_state.get("pilot_evidence") or [],
                        title="검색 Evidence Table (citation_id = LLM context 번호)",
                    )
                    _render_must_cover(st.session_state.get("pilot_must_cover") or [])

            ab_state = st.session_state.get("general_trend_ab") or {}
            if (
                ab_state.get("qid") == qid
                and ab_state.get("variants")
                and st.session_state.get("pilot_answer_qid") == qid
            ):
                _render_trend_ab_picker(qid=qid, question=general_question.strip(), llm_model=llm_model)
                if ab_state.get("selected") and st.button("이 답변을 UI 로그 파일에 저장", key=f"save_ab_{qid}"):
                    LOG_DIR.mkdir(parents=True, exist_ok=True)
                    meta = st.session_state.get("pilot_run_meta") or {}
                    block = "\n".join(
                        [
                            f"## {qid} (UI A/B {datetime.now(timezone.utc).isoformat()})",
                            f"- question: {row.get('question', '')}",
                            f"- selected: {ab_state.get('selected')}",
                            f"- settings: {json.dumps(meta, ensure_ascii=False)}",
                            "",
                            st.session_state.get("pilot_answer") or "",
                            "",
                            "---",
                            "",
                        ]
                    )
                    mode = "a" if UI_SAVED_ANSWERS.exists() else "w"
                    header = "# Pilot UI Answers\n\n" if mode == "w" else ""
                    with UI_SAVED_ANSWERS.open(mode, encoding="utf-8") as f:
                        f.write(header + block)
                    st.success(f"저장: `{UI_SAVED_ANSWERS}` · trace: `{TRACE_LOG}`")
            elif st.session_state.get("pilot_answer_qid") == qid and st.session_state.get(
                "pilot_answer"
            ):
                meta = st.session_state.get("pilot_run_meta") or {}
                st.subheader("답변")
                st.markdown(st.session_state["pilot_answer"])
                _render_evidence_table(
                    st.session_state.get("pilot_evidence") or [],
                    title="Evidence Table",
                )

                if st.button("이 답변을 UI 로그 파일에 저장", key=f"save_{qid}"):
                    LOG_DIR.mkdir(parents=True, exist_ok=True)
                    block = "\n".join(
                        [
                            f"## {qid} (UI {datetime.now(timezone.utc).isoformat()})",
                            f"- question: {row.get('question', '')}",
                            f"- settings: {json.dumps(meta, ensure_ascii=False)}",
                            "",
                            st.session_state["pilot_answer"],
                            "",
                            "---",
                            "",
                        ]
                    )
                    mode = "a" if UI_SAVED_ANSWERS.exists() else "w"
                    header = "# Pilot UI Answers\n\n" if mode == "w" else ""
                    with UI_SAVED_ANSWERS.open(mode, encoding="utf-8") as f:
                        f.write(header + block)
                    st.success(f"저장: `{UI_SAVED_ANSWERS}` · trace: `{TRACE_LOG}`")

    with tab_table:
        _render_table_qa_tab(
            latency_mode=latency_mode,
            llm_model=llm_model,
            ollama_base=ollama_base,
            top_k=top_k,
            fetch_k=fetch_k,
            max_doc=max_doc,
            max_docs=max_docs,
            use_rerank=use_rerank,
            temperature=temperature,
            multi_doc_strategy=multi_doc_strategy,
            max_llm_docs=max_llm_docs,
            table_eval_constrained=table_eval_constrained,
            table_top_k_eval=table_top_k_eval,
        )

    with tab_saved:
        st.warning(
            "**배치 스냅샷** — `python scripts/16_run_pilot_llm.py`로 생성된 고정 파일. "
            "「파일럿 7문항」탭은 open retrieval + 근거 추적이 적용된 실시간 실행입니다."
        )
        st.code("python scripts/16_run_pilot_llm.py", language="powershell")
        if SAVED_ANSWERS.exists():
            text = SAVED_ANSWERS.read_text(encoding="utf-8")
            st.info(f"파일 메타:\n```\n{chr(10).join(text.splitlines()[:4])}\n```")
            st.markdown(text)
        else:
            st.info(f"아직 없음: `{SAVED_ANSWERS}`")
        if TRACE_LOG.exists():
            st.divider()
            st.markdown("### 최근 retrieval trace (JSONL)")
            lines = TRACE_LOG.read_text(encoding="utf-8").strip().splitlines()
            for line in lines[-3:]:
                try:
                    entry = json.loads(line)
                    st.json(
                        {
                            "timestamp": entry.get("timestamp"),
                            "question_id": entry.get("question_id"),
                            "mode": entry.get("mode"),
                            "unique_doc_count": entry.get("unique_doc_count"),
                            "warnings": entry.get("warnings"),
                        }
                    )
                except json.JSONDecodeError:
                    pass
        if UI_SAVED_ANSWERS.exists():
            st.divider()
            st.markdown("### UI에서 저장한 답변")
            st.markdown(UI_SAVED_ANSWERS.read_text(encoding="utf-8"))
        if TABLE_UI_SAVED_ANSWERS.exists():
            st.divider()
            st.markdown("### 표 QA UI에서 저장한 답변")
            st.markdown(TABLE_UI_SAVED_ANSWERS.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
