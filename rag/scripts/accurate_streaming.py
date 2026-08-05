"""Accurate mode staged UI text and timing helpers (does not alter RAG/LLM prompts)."""
from __future__ import annotations

from typing import Any

STAGE_LABELS = {
    "initial_ack": "분석 시작",
    "retrieval_status": "문서 검색",
    "draft_summary": "초안 요약",
    "full_report": "보고서 생성",
    "evidence_table": "근거표 생성",
    "coverage_check": "검증",
    "done": "완료",
}

ACCURATE_MODE_BANNER = (
    "상세 분석 모드입니다. 3초 이내 첫 답변 표시를 목표로 넓은 dense 후보군을 재정렬합니다."
)


def build_initial_ack(row: dict | None = None) -> str:
    q = (row or {}).get("question", "")
    lead = "관련 문서를 검색하고 상세 분석을 시작합니다."
    follow = "먼저 핵심 쟁점을 요약한 뒤, 근거 문서별 세부 내용을 정리하겠습니다."
    if q:
        return f"{lead}\n\n{follow}\n\n**질문:** {q}"
    return f"{lead}\n\n{follow}"


def build_retrieval_status(
    *,
    doc_count: int,
    chunk_count: int,
    pool_doc_count: int | None = None,
    profile_label: str = "",
) -> str:
    pool = pool_doc_count if pool_doc_count is not None else doc_count
    profile = f" · **{profile_label}**" if profile_label else ""
    return (
        f"검색 완료 — 문서 **{doc_count}**개 · 청크 **{chunk_count}**개 "
        f"(pool 문서 {pool}개){profile}"
    )


def build_draft_summary(row: dict | None = None, *, category_label: str = "") -> str:
    cat = category_label or (row or {}).get("category", "")
    topic = f" ({cat})" if cat else ""
    return (
        f"**초안 안내{topic}** — 핵심 쟁점을 먼저 정리한 뒤, "
        "근거 문서별 세부 내용과 citation을 포함한 보고서를 생성합니다."
    )


def stage_caption(stage: str) -> str:
    label = STAGE_LABELS.get(stage, stage)
    return f"현재 단계: {label}"


def mark_accurate_initial_ack(timing) -> None:
    if timing is None:
        return
    if hasattr(timing, "mark_wall"):
        timing.mark_wall("t_initial_ack_rendered")
        timing.mark_wall("t_first_visible_response")


def mark_accurate_llm_token_received(timing) -> None:
    if timing is None or not hasattr(timing, "mark_wall"):
        return
    w = timing.wall_clock
    if "t_accurate_first_token_received" not in w and "t_first_token" in w:
        timing.mark_wall("t_accurate_first_token_received", at=w["t_first_token"])


def mark_accurate_llm_token_rendered(timing) -> None:
    if timing is None or not hasattr(timing, "mark_wall"):
        return
    if "t_accurate_first_token_rendered" not in timing.wall_clock:
        timing.mark_wall("t_accurate_first_token_rendered")


def wrap_accurate_on_token(on_token, timing):
    """Invoke UI/benchmark token callback and record first-render wall time."""
    rendered = {"done": False}

    def _wrapped(tok: str) -> None:
        mark_accurate_llm_token_received(timing)
        if not rendered["done"]:
            mark_accurate_llm_token_rendered(timing)
            if timing is not None and hasattr(timing, "mark_first_token_rendered"):
                timing.mark_first_token_rendered()
            rendered["done"] = True
        if on_token is not None:
            on_token(tok)

    return _wrapped


def compute_accurate_metrics(wall: dict[str, float]) -> dict[str, Any]:
    def _d(a: str, b: str) -> float | None:
        if wall.get(a) is None or wall.get(b) is None:
            return None
        return round(max(0.0, wall[b] - wall[a]), 4)

    click = wall.get("t_user_click") or wall.get("t_user_submit")
    initial_ack = _d("t_user_click", "t_initial_ack_rendered") or _d("t_user_submit", "t_initial_ack_rendered")
    first_visible = _d("t_user_submit", "t_first_visible_response") or initial_ack
    accurate_llm_ttft = _d("t_accurate_llm_request_start", "t_accurate_first_token_received")
    if accurate_llm_ttft is None:
        accurate_llm_ttft = _d("t_llm_request_start", "t_first_token")
    first_llm_visible = _d("t_user_submit", "t_accurate_first_token_rendered")
    if first_llm_visible is None:
        first_llm_visible = _d("t_user_submit", "t_first_token")
    total = _d("t_user_submit", "t_all_done")
    evidence_t = _d("t_full_report_complete", "t_evidence_table_complete")
    coverage_t = _d("t_evidence_table_complete", "t_coverage_check_complete")
    initial_pass = initial_ack is not None and initial_ack <= 3.0
    first_llm_pass = first_llm_visible is not None and first_llm_visible <= 3.0
    return {
        "initial_ack_latency": initial_ack,
        "first_visible_response_latency": first_visible,
        "retrieval_time": _d("t_retrieval_start", "t_retrieval_end"),
        "accurate_llm_ttft": accurate_llm_ttft,
        "accurate_first_visible_llm_latency": first_llm_visible,
        "accurate_full_report_total_time": _d("t_user_submit", "t_full_report_complete"),
        "accurate_total_time": total,
        "evidence_table_time": evidence_t,
        "coverage_check_time": coverage_t,
        "accurate_initial_response_3s_pass": initial_pass,
        "accurate_first_llm_visible_3s_pass": first_llm_pass,
        "fast_e2e_ttft_3s_pass": None,
    }


def merge_pass_criteria(timing_metrics: dict, latency_mode: str) -> dict[str, Any]:
    """Attach mode-specific 3s pass flags without changing fast metrics."""
    out = dict(timing_metrics)
    wall = timing_metrics.get("_wall_for_accurate") or {}
    if latency_mode == "accurate":
        acc = compute_accurate_metrics(wall) if wall else {}
        out.update({k: v for k, v in acc.items() if k != "fast_e2e_ttft_3s_pass"})
        out["accurate_initial_response_3s_pass"] = acc.get("accurate_initial_response_3s_pass", False)
    elif latency_mode == "fast":
        e2e = out.get("end_to_end_ttft")
        out["fast_e2e_ttft_3s_pass"] = e2e is not None and e2e <= 3.0
        out["accurate_initial_response_3s_pass"] = None
    return out
