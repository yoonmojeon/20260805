"""Latency breakdown tracing for MaritimeRAG pilot (retrieval + LLM + UI)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_TIMING_LOG = Path("data/processed/logs/retrieval_timing_trace.jsonl")

TIMESTAMP_KEYS = (
    "t_user_click",
    "t_user_submit",
    "t_initial_ack_rendered",
    "t_first_visible_response",
    "t_retrieval_start",
    "t_query_embedding_start",
    "t_query_embedding_end",
    "t_vector_search_start",
    "t_vector_search_end",
    "t_metadata_filter_start",
    "t_metadata_filter_end",
    "t_rerank_start",
    "t_rerank_end",
    "t_retrieval_end",
    "t_pre_llm_start",
    "t_ensure_fast_warm_check_end",
    "t_ollama_probe_end",
    "t_prompt_build_end",
    "t_context_build_start",
    "t_context_build_end",
    "t_llm_request_start",
    "t_accurate_llm_request_start",
    "t_first_token",
    "t_first_token_received",
    "t_first_token_rendered",
    "t_accurate_first_token_received",
    "t_accurate_first_token_rendered",
    "t_llm_response_end",
    "t_answer_complete",
    "t_full_report_complete",
    "t_evidence_table_complete",
    "t_coverage_check_complete",
    "t_all_done",
    "t_ui_render_end",
)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _delta(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return round(end - start, 4)


@dataclass
class TimingTrace:
    """Collect monotonic + wall timestamps for latency breakdown."""

    run_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
    monotonic: dict[str, float] = field(default_factory=dict)
    wall_clock: dict[str, float] = field(default_factory=dict)
    wall_iso: dict[str, str] = field(default_factory=dict)
    cache_flags: dict[str, bool] = field(default_factory=dict)
    debug_notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    ollama_meta: dict[str, Any] = field(default_factory=dict)

    def set_ollama_meta(self, meta: dict[str, Any]) -> None:
        self.ollama_meta = dict(meta or {})

    def enrich_llm_metrics(self, metrics: dict[str, float | None]) -> dict[str, Any]:
        out: dict[str, Any] = dict(metrics)
        om = self.ollama_meta or {}
        load_ms = om.get("load_duration_ms")
        prompt_eval_ms = om.get("prompt_eval_duration_ms")
        llm_ttft = metrics.get("llm_ttft")
        reload_s = round(load_ms / 1000, 4) if load_ms else None
        if reload_s is not None and llm_ttft is not None:
            pure = max(0.0, llm_ttft - reload_s)
        elif prompt_eval_ms and llm_ttft is not None:
            pure = max(0.0, llm_ttft - prompt_eval_ms / 1000)
        else:
            pure = llm_ttft
        out["llm_reload_or_warmup_time"] = reload_s
        out["pure_inference_ttft"] = round(pure, 4) if pure is not None else None
        out["ollama_load_duration_ms"] = load_ms
        out["ollama_prompt_eval_duration_ms"] = prompt_eval_ms
        out["context_changed"] = self.meta.get("context_changed")
        out["warmup_matched_request"] = self.meta.get("warmup_matched_request")
        out["rewarm_triggered"] = self.meta.get("rewarm_triggered")
        out["rewarm_reason"] = self.meta.get("rewarm_reason")
        return out

    def mark(self, key: str, *, at: float | None = None) -> None:
        if key not in TIMESTAMP_KEYS:
            raise ValueError(f"Unknown timing key: {key}")
        self.monotonic[key] = at if at is not None else time.perf_counter()
        self.wall_clock[key] = time.time()
        self.wall_iso[key] = _utc_iso()
        if key == "t_llm_request_start" and self.meta.get("latency_mode") == "accurate":
            if "t_accurate_llm_request_start" not in self.wall_clock:
                self.wall_clock["t_accurate_llm_request_start"] = self.wall_clock[key]
                self.wall_iso["t_accurate_llm_request_start"] = self.wall_iso[key]
        if key == "t_first_token" and self.meta.get("latency_mode") == "accurate":
            if "t_accurate_first_token_received" not in self.wall_clock:
                self.wall_clock["t_accurate_first_token_received"] = self.wall_clock[key]
                self.wall_iso["t_accurate_first_token_received"] = self.wall_iso[key]
        if key == "t_first_token":
            if "t_first_token_received" not in self.wall_clock:
                self.wall_clock["t_first_token_received"] = self.wall_clock[key]
                self.wall_iso["t_first_token_received"] = self.wall_iso[key]

    def set_user_click(self, ts: float | None = None) -> None:
        """Single-button UI anchor: click time = E2E start."""
        click_ts = ts if ts is not None else time.time()
        self.wall_clock["t_user_click"] = click_ts
        self.wall_iso["t_user_click"] = _utc_iso()
        self.set_user_submit(click_ts)

    def set_user_submit(self, ts: float | None = None) -> None:
        """Wall-clock seconds (time.time()) from UI — comparable across subprocess."""
        self.wall_clock["t_user_submit"] = ts if ts is not None else time.time()
        self.wall_iso["t_user_submit"] = _utc_iso()

    def mark_wall(self, key: str, *, at: float | None = None) -> None:
        if key not in TIMESTAMP_KEYS:
            raise ValueError(f"Unknown timing key: {key}")
        ts = at if at is not None else time.time()
        self.wall_clock[key] = ts
        self.wall_iso[key] = _utc_iso()

    def mark_first_token_rendered(self) -> None:
        if "t_first_token_rendered" not in self.wall_clock:
            self.mark_wall("t_first_token_rendered")

    def set_cache(self, key: str, value: bool) -> None:
        self.cache_flags[key] = bool(value)

    def add_debug(self, msg: str) -> None:
        if msg and msg not in self.debug_notes:
            self.debug_notes.append(msg)

    def add_warning(self, msg: str) -> None:
        if msg and msg not in self.warnings:
            self.warnings.append(msg)

    def mark_ui_render_end(self) -> None:
        self.wall_clock["t_ui_render_end"] = time.time()
        self.wall_iso["t_ui_render_end"] = _utc_iso()

    def compute_metrics(self) -> dict[str, float | None]:
        m = self.monotonic
        w = self.wall_clock
        click = w.get("t_user_click") or w.get("t_user_submit")
        first_recv = w.get("t_first_token_received") or w.get("t_first_token")
        first_rendered = w.get("t_first_token_rendered")
        answer_done = (
            w.get("t_answer_complete")
            or w.get("t_all_done")
            or w.get("t_llm_response_end")
            or w.get("t_ui_render_end")
        )
        answer_first_visible = _delta(click, first_rendered)
        e2e_ttft = _delta(click, first_recv)
        if answer_first_visible is not None and e2e_ttft is not None:
            pass_3s_metric = answer_first_visible
        else:
            pass_3s_metric = e2e_ttft
        return {
            "query_embedding_time": _delta(m.get("t_query_embedding_start"), m.get("t_query_embedding_end")),
            "vector_search_time": _delta(m.get("t_vector_search_start"), m.get("t_vector_search_end")),
            "metadata_filter_time": _delta(m.get("t_metadata_filter_start"), m.get("t_metadata_filter_end")),
            "rerank_time": _delta(m.get("t_rerank_start"), m.get("t_rerank_end")),
            "retrieval_total_time": _delta(m.get("t_retrieval_start"), m.get("t_retrieval_end")),
            "context_build_time": _delta(m.get("t_context_build_start"), m.get("t_context_build_end")),
            "llm_ttft": _delta(m.get("t_llm_request_start"), m.get("t_first_token")),
            "llm_generation_time": _delta(m.get("t_llm_request_start"), m.get("t_llm_response_end")),
            "answer_first_visible_latency": answer_first_visible,
            "pre_llm_latency": _delta(
                w.get("t_retrieval_end") or w.get("t_pre_llm_start"),
                w.get("t_llm_request_start"),
            ),
            "ensure_fast_warm_check_time": self.meta.get("ensure_fast_warm_check_time"),
            "rewarm_time": self.meta.get("rewarm_time"),
            "ollama_probe_time": _delta(w.get("t_ensure_fast_warm_check_end"), w.get("t_ollama_probe_end")),
            "model_match_check_time": self.meta.get("model_match_check_time"),
            "prompt_build_time": _delta(w.get("t_ollama_probe_end"), w.get("t_prompt_build_end")),
            "stream_init_time": _delta(w.get("t_prompt_build_end"), w.get("t_llm_request_start")),
            "session_state_update_time": self.meta.get("session_state_update_time"),
            "end_to_end_ttft": e2e_ttft,
            "end_to_end_total": _delta(click, answer_done),
            "ui_after_worker_time": _delta(
                w.get("t_llm_response_end") or w.get("t_retrieval_end"),
                w.get("t_ui_render_end"),
            ),
            "worker_wall_time": _delta(w.get("t_retrieval_start"), w.get("t_llm_response_end") or w.get("t_retrieval_end")),
            "_pass_3s_latency": pass_3s_metric,
        }

    def analyze_slow_retrieval(self, metrics: dict[str, float | None]) -> None:
        total = metrics.get("retrieval_total_time")
        if total is None or total < 10.0:
            return
        if self.cache_flags.get("embedding_model_loaded_from_cache") and total < 15.0:
            return
        self.add_warning(f"retrieval_total_time {total:.2f}s >= 10s — see debug_notes")
        flags = self.cache_flags
        if not flags.get("embedding_model_loaded_from_cache", True):
            self.add_debug("embedding model cold-loaded in this worker process (subprocess = every UI click)")
        if not flags.get("vector_db_loaded_from_cache", True):
            self.add_debug("ChromaDB collection opened fresh (new worker process)")
        if not flags.get("manifest_loaded_from_cache", True):
            self.add_debug("index manifest read from disk (new worker process)")
        emb = metrics.get("query_embedding_time") or 0
        if emb >= 5.0:
            self.add_debug(f"query_embedding_time {emb:.2f}s — likely E5 model load + encode")
        vec = metrics.get("vector_search_time") or 0
        if vec >= 3.0:
            self.add_debug(f"vector_search_time {vec:.2f}s — Chroma query over full_corpus (~134k vectors)")
        rer = metrics.get("rerank_time") or 0
        if rer >= 2.0:
            self.add_debug(f"rerank_time {rer:.2f}s — hybrid/diversity rerank over fetch pool (not full corpus scan)")
        ctx = metrics.get("context_build_time") or 0
        if ctx >= 2.0:
            self.add_debug(f"context_build_time {ctx:.2f}s — chunks.jsonl disk reads / doc grouping")
        self.add_debug("Streamlit spawns new subprocess per search/answer — full pipeline repeats each click")

    def to_log_row(self) -> dict[str, Any]:
        metrics = self.enrich_llm_metrics(self.compute_metrics())
        self.analyze_slow_retrieval(metrics)
        latency_mode = self.meta.get("latency_mode")
        if latency_mode == "accurate":
            from accurate_streaming import compute_accurate_metrics

            acc = compute_accurate_metrics(self.wall_clock)
            metrics.update(acc)
            e2e = metrics.get("end_to_end_ttft")
            metrics["fast_e2e_ttft_3s_pass"] = None
        elif latency_mode == "fast":
            pass_lat = metrics.get("answer_first_visible_latency") or metrics.get("end_to_end_ttft")
            metrics["fast_e2e_ttft_3s_pass"] = pass_lat is not None and pass_lat <= 3.0
            metrics["accurate_initial_response_3s_pass"] = None
        cache_status = dict(self.cache_flags)
        row = {
            "timestamp": _utc_iso(),
            "run_id": self.run_id,
            "start_type": self.meta.get("start_type", "unknown"),
            "run_index": self.meta.get("run_index", 0),
            "cache_status": cache_status,
            "timestamps_iso": dict(self.wall_iso),
            "timestamps_monotonic": {k: round(v, 6) for k, v in self.monotonic.items()},
            "timestamps_wall": {k: round(v, 6) for k, v in self.wall_clock.items()},
            **{k: v for k, v in self.meta.items() if k not in ("start_type", "run_index")},
            "timing_metrics": metrics,
            "cache_flags": cache_status,
            "debug_notes": list(self.debug_notes),
            "warnings": list(self.warnings),
        }
        return row

    def summary_lines(self) -> list[str]:
        m = self.compute_metrics()
        labels = [
            ("Query embedding", "query_embedding_time"),
            ("Vector search", "vector_search_time"),
            ("Metadata filter", "metadata_filter_time"),
            ("Rerank", "rerank_time"),
            ("Retrieval total", "retrieval_total_time"),
            ("Context build", "context_build_time"),
            ("LLM TTFT", "llm_ttft"),
            ("LLM reload/warmup", "llm_reload_or_warmup_time"),
            ("Pure inference TTFT", "pure_inference_ttft"),
            ("LLM total generation", "llm_generation_time"),
            ("First visible latency", "answer_first_visible_latency"),
            ("End-to-end TTFT", "end_to_end_ttft"),
            ("End-to-end total", "end_to_end_total"),
        ]
        lines = ["**Latency breakdown**"]
        for label, key in labels:
            val = m.get(key)
            if val is None:
                lines.append(f"- {label}: —")
            else:
                lines.append(f"- {label}: {val:.2f}s")
        if self.cache_flags:
            lines.append("")
            lines.append("**Cache status**")
            for k, v in self.cache_flags.items():
                lines.append(f"- {k}: {v}")
        start_type = self.meta.get("start_type")
        if start_type:
            lines.append(f"- start_type: {start_type} (run #{self.meta.get('run_index', 0)})")
        if self.warnings:
            lines.append("")
            lines.append("**Timing warnings**")
            for w in self.warnings:
                lines.append(f"- ⚠ {w}")
        if self.meta.get("latency_mode") == "accurate":
            from accurate_streaming import compute_accurate_metrics

            acc = compute_accurate_metrics(self.wall_clock)
            acc_lines = [
                ("Initial ack latency", "initial_ack_latency"),
                ("Accurate LLM TTFT", "accurate_llm_ttft"),
                ("First LLM visible latency", "accurate_first_visible_llm_latency"),
                ("Full report total", "accurate_full_report_total_time"),
                ("Accurate total time", "accurate_total_time"),
                ("Evidence table time", "evidence_table_time"),
                ("Coverage check time", "coverage_check_time"),
            ]
            lines.append("")
            lines.append("**Accurate streaming**")
            for label, key in acc_lines:
                val = acc.get(key)
                if val is not None:
                    lines.append(f"- {label}: {val:.2f}s")
            init_pass = acc.get("accurate_initial_response_3s_pass")
            if init_pass is not None:
                lines.append(
                    f"- initial 3s: {'PASS' if init_pass else 'FAIL'}"
                )
        if self.debug_notes:
            lines.append("")
            lines.append("**Debug notes**")
            for n in self.debug_notes[:8]:
                lines.append(f"- {n}")
        return lines


def append_timing_log(row: dict[str, Any], path: Path = DEFAULT_TIMING_LOG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 3)


def _chunk_field(chunk: Any, key: str, default: Any = "") -> Any:
    if isinstance(chunk, dict):
        return chunk.get(key, default)
    return getattr(chunk, key, default)


def set_run_context(
    timing: TimingTrace,
    *,
    start_type: str,
    run_index: int = 1,
) -> None:
    timing.meta["start_type"] = start_type
    timing.meta["run_index"] = run_index
    if start_type == "warm":
        timing.add_debug(f"warm start run_index={run_index} — resources expected cached in-process")
    elif start_type == "cold":
        timing.add_debug("cold start — process caches cleared or first load")


def finalize_ui_timing(timing_log: dict | None, user_click_ts: float | None = None) -> dict | None:
    """Attach UI render end; preserve single-click anchor when already set."""
    if not timing_log:
        return None
    trace = TimingTrace()
    trace.run_id = timing_log.get("run_id", trace.run_id)
    trace.monotonic = dict(timing_log.get("timestamps_monotonic") or {})
    trace.wall_clock = dict(timing_log.get("timestamps_wall") or {})
    trace.wall_iso = dict(timing_log.get("timestamps_iso") or {})
    trace.cache_flags = dict(timing_log.get("cache_flags") or {})
    trace.meta = {k: v for k, v in timing_log.items() if k not in {
        "timestamps_monotonic", "timestamps_wall", "timestamps_iso", "timing_metrics",
        "cache_flags", "cache_status", "debug_notes", "warnings", "timestamp", "run_id",
        "start_type", "run_index",
    }}
    if timing_log.get("start_type"):
        trace.meta["start_type"] = timing_log["start_type"]
    if timing_log.get("run_index") is not None:
        trace.meta["run_index"] = timing_log["run_index"]
    if user_click_ts is not None and "t_user_click" not in trace.wall_clock:
        trace.set_user_click(user_click_ts)
    elif user_click_ts is not None and trace.wall_clock.get("t_user_click") != user_click_ts:
        trace.set_user_click(user_click_ts)
    trace.mark_ui_render_end()
    if "t_answer_complete" not in trace.wall_clock:
        trace.mark_wall("t_answer_complete", at=trace.wall_clock.get("t_ui_render_end"))
    trace.warnings = list(timing_log.get("warnings") or [])
    trace.debug_notes = list(timing_log.get("debug_notes") or [])
    return trace.to_log_row()


def merge_timing_logs(search_log: dict | None, answer_log: dict | None) -> dict | None:
    """Combine search + answer worker logs for UI two-step flow."""
    if not answer_log:
        return search_log
    if not search_log:
        return answer_log
    merged = dict(answer_log)
    for bucket in ("timestamps_monotonic", "timestamps_wall", "timestamps_iso"):
        base = dict(search_log.get(bucket) or {})
        base.update(answer_log.get(bucket) or {})
        merged[bucket] = base
    merged["cache_flags"] = {**(search_log.get("cache_flags") or {}), **(answer_log.get("cache_flags") or {})}
    merged["debug_notes"] = list(dict.fromkeys(
        (search_log.get("debug_notes") or []) + (answer_log.get("debug_notes") or [])
    ))
    merged["warnings"] = list(dict.fromkeys(
        (search_log.get("warnings") or []) + (answer_log.get("warnings") or [])
    ))
    merged["action"] = "search_and_answer"
    return merged


def populate_timing_meta(
    timing: TimingTrace,
    *,
    row: dict,
    mode: str,
    top_k: int,
    fetch_k: int,
    retrieved: list | None = None,
    pool: list | None = None,
    answer: str = "",
    source_filter: str | None = None,
    action: str = "search",
) -> None:
    chunks = retrieved or []
    pool_list = pool or chunks
    doc_ids = sorted({str(_chunk_field(c, "doc_id")) for c in chunks if _chunk_field(c, "doc_id")})
    input_text = " ".join(str(_chunk_field(c, "text")) for c in chunks[:20])
    token_est = timing.meta.get("input_token_estimate") or estimate_tokens(input_text)
    timing.meta.update(
        {
            "action": action,
            "query": str(row.get("question", "")),
            "question_id": str(row.get("question_id", "")),
            "mode": mode,
            "source_filter": source_filter or ",".join(row.get("retrieval_sources") or []),
            "top_k": top_k,
            "fetch_k": fetch_k,
            "selected_doc_count": len(doc_ids),
            "selected_chunk_count": len(chunks),
            "input_token_estimate": token_est,
            "output_token_estimate": estimate_tokens(answer) if answer else None,
            "retrieved_doc_ids": doc_ids,
            "used_citations": [],
            "pool_chunk_count": len(pool_list),
        }
    )
