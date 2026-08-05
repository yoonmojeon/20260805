"""Fast-mode Ollama warm-up and context-slot tracking (match /api/chat num_ctx=4096)."""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_answer_lib import _ollama_chat_payload

ACCURATE_NUM_CTX = 4096
FAST_NUM_CTX = 4096
FAST_TEMPERATURE = 0.1
KEEP_ALIVE = "24h"
KEEP_ALIVE_SECONDS = 24 * 60 * 60
FAST_WARMUP_NUM_PREDICT = 64
WARMUP_TRACE_LOG = Path("data/processed/logs/warmup_state_trace.jsonl")

FAST_WARMUP_SYSTEM = (
    "너는 해사 규정 문서 기반 RAG assistant다. "
    "제공된 근거 context 안에서만 답변해라. "
    "근거가 부족하면 부족하다고 말해라. 답변은 간결하게 작성해라."
)

FAST_WARMUP_USER = (
    "질문: warm-up probe\n\n"
    "근거:\n[1] sample.pdf p.1: placeholder context for GPU warm-up.\n\n"
    "위 근거를 바탕으로 한 줄만 답변해줘."
)

_state: dict[str, Any] = {
    "model": "",
    "num_ctx": 0,
    "api_type": "",
    "warmed_at": 0.0,
    "keep_alive_seconds": KEEP_ALIVE_SECONDS,
    "warmup_ttft": None,
    "warmup_total_time": None,
    "fast_warm_valid": False,
    "last_llm_num_ctx": None,
    "last_latency_mode": None,
    "last_rewarm_reason": None,
    "last_rewarm_triggered": False,
    "accurate_invalidated": False,
}


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_keep_alive_seconds(value: str | int | float) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    if text.endswith("m"):
        return int(float(text[:-1]) * 60)
    if text.endswith("h"):
        return int(float(text[:-1]) * 60 * 60)
    if text.endswith("s"):
        return int(float(text[:-1]))
    if text.isdigit():
        return int(text)
    return KEEP_ALIVE_SECONDS


def is_keep_alive_expired(now: float | None = None) -> bool:
    warmed_at = _state.get("warmed_at") or 0
    if not warmed_at or not _state.get("fast_warm_valid"):
        return True
    ttl = int(_state.get("keep_alive_seconds") or KEEP_ALIVE_SECONDS)
    now = now if now is not None else time.time()
    return (now - warmed_at) >= ttl


def append_warmup_trace(event: str, payload: dict[str, Any]) -> None:
    row = {"timestamp": _utc_iso(), "event": event, **payload}
    WARMUP_TRACE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with WARMUP_TRACE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_llm_warm_state() -> dict[str, Any]:
    return dict(_state)


def get_fast_warm_ui_state(model: str | None = None) -> str:
    if not _state.get("fast_warm_valid"):
        return "invalid"
    if is_keep_alive_expired():
        return "expired"
    if model and _state.get("model") != model:
        return "invalid"
    return "ready"


def get_warm_status_display() -> dict[str, Any]:
    s = get_llm_warm_state()
    warmed_at = s.get("warmed_at") or 0
    return {
        "llm_fast_warm_ready": bool(s.get("fast_warm_valid")) and not is_keep_alive_expired(),
        "llm_fast_warm": bool(s.get("fast_warm_valid")),
        "warmed_model_name": s.get("model") or "",
        "warmed_num_ctx": s.get("num_ctx") or 0,
        "warmed_api_type": s.get("api_type") or "",
        "warmup_ttft": s.get("warmup_ttft"),
        "warmup_total_time": s.get("warmup_total_time"),
        "last_warmup_time": datetime.fromtimestamp(warmed_at, tz=timezone.utc).isoformat()
        if warmed_at
        else "",
        "keep_alive_expired": is_keep_alive_expired(),
        "fast_warm_ui_state": get_fast_warm_ui_state(s.get("model")),
        "last_llm_num_ctx": s.get("last_llm_num_ctx"),
        "last_latency_mode": s.get("last_latency_mode"),
        "last_rewarm_reason": s.get("last_rewarm_reason"),
    }


def get_resource_ready_status(
    *,
    unified_id: str,
    index_dir: Path,
    embed_model: str,
    model: str,
) -> dict[str, Any]:
    from rag_resource_cache import (
        cache_status_snapshot,
        is_vector_db_cached,
    )

    snap = cache_status_snapshot(unified_id, index_dir, embed_model)
    warm = get_warm_status_display()
    return {
        "embedding_ready": snap.get("embedding_model_loaded_from_cache", False),
        "vector_db_ready": snap.get("vector_db_loaded_from_cache", False),
        "metadata_ready": snap.get("metadata_loaded_from_cache", False),
        "manifest_ready": snap.get("manifest_loaded_from_cache", False),
        "llm_fast_warm_ready": warm.get("llm_fast_warm_ready"),
        **warm,
        "model_matches": warm.get("warmed_model_name") == model,
    }


def diagnose_rewarm_reason(model: str, num_ctx: int | None = None) -> str | None:
    ctx = num_ctx if num_ctx is not None else FAST_NUM_CTX
    if not _state.get("fast_warm_valid"):
        return "no_warm_state"
    if is_keep_alive_expired():
        return "keep_alive_expired"
    if _state.get("model") and _state.get("model") != model:
        return "model_changed"
    if _state.get("api_type") != "chat":
        return "api_type_mismatch"
    if _state.get("num_ctx") != ctx:
        return "num_ctx_changed"
    last_ctx = _state.get("last_llm_num_ctx")
    if last_ctx is not None and last_ctx != ctx:
        return "num_ctx_changed"
    if _state.get("last_latency_mode") == "accurate":
        return "accurate_run_after_fast"
    if _state.get("accurate_invalidated"):
        return "accurate_invalidated"
    return None


def check_fast_warm_match(model: str, num_ctx: int | None = None) -> dict[str, Any]:
    ctx = num_ctx if num_ctx is not None else FAST_NUM_CTX
    reason = diagnose_rewarm_reason(model, ctx)
    matched = reason is None
    last_ctx = _state.get("last_llm_num_ctx")
    context_changed = last_ctx is not None and last_ctx != ctx
    needs_rewarm = not matched
    return {
        "warmup_matched": matched,
        "context_changed": context_changed,
        "needs_rewarm": needs_rewarm,
        "rewarm_reason": reason,
        "keep_alive_expired": is_keep_alive_expired(),
        "expected_num_ctx": ctx,
        "warmed_model_name": _state.get("model"),
        "warmed_num_ctx": _state.get("num_ctx"),
    }


def _stream_chat_probe(
    payload_bytes: bytes,
    base_url: str,
    *,
    timeout: int = 300,
) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=payload_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    t_first_content: float | None = None
    t_done: float | None = None
    ollama_meta: dict[str, Any] = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            if not line.strip():
                continue
            data = json.loads(line.decode("utf-8"))
            chunk = data.get("message", {}).get("content", "")
            if chunk and t_first_content is None:
                t_first_content = time.perf_counter()
            if data.get("done"):
                t_done = time.perf_counter()
                ollama_meta = {
                    "load_duration_ms": round((data.get("load_duration") or 0) / 1e6, 1),
                    "prompt_eval_duration_ms": round(
                        (data.get("prompt_eval_duration") or 0) / 1e6, 1
                    ),
                    "eval_duration_ms": round((data.get("eval_duration") or 0) / 1e6, 1),
                    "eval_count": data.get("eval_count"),
                }
                break
    if t_done is None:
        t_done = time.perf_counter()
    ttft = (t_first_content or t_done) - t0
    return {
        "warmup_ttft": round(ttft, 4),
        "warmup_total_time": round(t_done - t0, 4),
        "ollama_meta": ollama_meta,
    }


def warmup_fast_chat(
    model: str,
    base_url: str = "http://127.0.0.1:11434",
    *,
    num_ctx: int | None = None,
    num_predict: int = FAST_WARMUP_NUM_PREDICT,
    keep_alive: str = KEEP_ALIVE,
    force: bool = False,
    rewarm_reason_override: str | None = None,
) -> dict[str, Any]:
    """Warm Ollama with the same /api/chat payload shape as Fast mode."""
    ctx = num_ctx if num_ctx is not None else FAST_NUM_CTX
    ka_secs = parse_keep_alive_seconds(keep_alive)
    check = check_fast_warm_match(model, ctx)
    if not force and check["warmup_matched"] and not check["needs_rewarm"]:
        out = {"skipped": True, "rewarm_triggered": False, **check, **get_warm_status_display()}
        append_warmup_trace("warmup_skipped", {"model_name": model, **out})
        return out

    rewarm_reason = rewarm_reason_override or check.get("rewarm_reason") or "forced_warmup"
    payload = _ollama_chat_payload(
        model,
        FAST_WARMUP_SYSTEM,
        FAST_WARMUP_USER,
        stream=True,
        temperature=FAST_TEMPERATURE,
        num_predict=num_predict,
        num_ctx=ctx,
        keep_alive=keep_alive,
    )
    metrics = _stream_chat_probe(payload, base_url)
    now = time.time()
    _state.update(
        {
            "model": model,
            "num_ctx": ctx,
            "api_type": "chat",
            "warmed_at": now,
            "keep_alive_seconds": ka_secs,
            "warmup_ttft": metrics["warmup_ttft"],
            "warmup_total_time": metrics["warmup_total_time"],
            "fast_warm_valid": True,
            "last_llm_num_ctx": ctx,
            "last_latency_mode": "fast_warmup",
            "last_rewarm_triggered": True,
            "last_rewarm_reason": rewarm_reason,
            "accurate_invalidated": False,
        }
    )
    result = {
        "skipped": False,
        "rewarm_triggered": True,
        "rewarm_reason": rewarm_reason,
        **check,
        **metrics,
        **get_warm_status_display(),
        "payload_summary": {
            "api_type": "chat",
            "stream": True,
            "num_ctx": ctx,
            "num_predict": num_predict,
            "temperature": FAST_TEMPERATURE,
            "keep_alive": keep_alive,
        },
    }
    append_warmup_trace("warmup_completed", {"model_name": model, **result})
    return result


def get_warm_preflight_snapshot(model: str) -> dict[str, Any]:
    """Snapshot warm state for UI display before button click."""
    check = check_fast_warm_match(model)
    warmed_at = float(_state.get("warmed_at") or 0)
    now = time.time()
    return {
        "warm_state_valid_before_click": bool(check["warmup_matched"]),
        "rewarm_needed_before_click": bool(check["needs_rewarm"]),
        "rewarm_reason_before_click": check.get("rewarm_reason"),
        "warmed_model": _state.get("model") or "",
        "warmed_num_ctx": _state.get("num_ctx") or 0,
        "warmed_api_type": _state.get("api_type") or "",
        "keep_alive_valid": not is_keep_alive_expired(now),
        "accurate_invalidated": bool(_state.get("accurate_invalidated")),
        "time_since_last_warmup": round(now - warmed_at, 3) if warmed_at else None,
        "fast_warm_ui_state": get_fast_warm_ui_state(model),
    }


def ensure_fast_warm_checked(
    model: str,
    base_url: str = "http://127.0.0.1:11434",
    *,
    timing=None,
    force: bool = False,
    allow_rewarm: bool = True,
    num_ctx: int | None = None,
) -> dict[str, Any]:
    """Check (and optionally re-warm) Fast LLM; record sub-step timings on ``timing``."""
    import time as _time

    t0 = _time.perf_counter()
    check = check_fast_warm_match(model, num_ctx)
    t_check = _time.perf_counter()
    check_ms = round((t_check - t0) * 1000, 2)
    if timing is not None and hasattr(timing, "meta"):
        timing.meta["model_match_check_time"] = round(t_check - t0, 4)
        timing.meta["ensure_fast_warm_check_time"] = round(t_check - t0, 4)

    matched = check["warmup_matched"] and not check["needs_rewarm"]
    if not force and matched:
        out = {
            "skipped": True,
            "rewarm_triggered": False,
            "rewarm_reason": None,
            "ensure_fast_warm_check_time": round(t_check - t0, 4),
            **check,
            **get_warm_status_display(),
        }
        if timing is not None:
            if hasattr(timing, "mark_wall"):
                timing.mark_wall("t_ensure_fast_warm_check_end")
            if hasattr(timing, "meta"):
                timing.meta["warmup_matched_request"] = True
                timing.meta["rewarm_triggered"] = False
                timing.meta["rewarm_reason"] = None
                timing.meta["rewarm_time"] = 0.0
        return out

    if not allow_rewarm:
        out = {
            "skipped": True,
            "rewarm_triggered": False,
            "rewarm_reason": check.get("rewarm_reason"),
            "rewarm_blocked_in_hot_path": True,
            "ensure_fast_warm_check_time": round(t_check - t0, 4),
            **check,
            **get_warm_status_display(),
        }
        if timing is not None and hasattr(timing, "meta"):
            timing.meta["warmup_matched_request"] = False
            timing.meta["rewarm_triggered"] = False
            timing.meta["rewarm_reason"] = check.get("rewarm_reason")
            timing.meta["rewarm_would_be_needed"] = True
            timing.meta["rewarm_time"] = 0.0
            if hasattr(timing, "mark_wall"):
                timing.mark_wall("t_ensure_fast_warm_check_end")
        append_warmup_trace(
            "ensure_check_only_blocked",
            {"model_name": model, "check_ms": check_ms, **out},
        )
        return out

    t_rewarm = _time.perf_counter()
    result = warmup_fast_chat(model, base_url, force=True, num_ctx=num_ctx)
    rewarm_s = round(_time.perf_counter() - t_rewarm, 4)
    result["ensure_fast_warm_check_time"] = round(t_check - t0, 4)
    result["rewarm_time"] = rewarm_s
    if timing is not None and hasattr(timing, "add_debug"):
        timing.add_debug(
            f"LLM fast re-warm reason={result.get('rewarm_reason')} "
            f"ttft={result.get('warmup_ttft')}s"
        )
    if timing is not None and hasattr(timing, "meta"):
        timing.meta["warmup_matched_request"] = False
        timing.meta["context_changed"] = check.get("context_changed", False)
        timing.meta["rewarm_triggered"] = True
        timing.meta["rewarm_reason"] = result.get("rewarm_reason")
        timing.meta["rewarm_time"] = rewarm_s
        if hasattr(timing, "mark_wall"):
            timing.mark_wall("t_ensure_fast_warm_check_end")
    append_warmup_trace(
        "ensure_rewarm",
        {"model_name": model, "timing_action": "pre_fast_answer", **result},
    )
    return result


def ensure_fast_warm(
    model: str,
    base_url: str = "http://127.0.0.1:11434",
    *,
    timing=None,
    force: bool = False,
) -> dict[str, Any]:
    return ensure_fast_warm_checked(
        model, base_url, timing=timing, force=force, allow_rewarm=True
    )


def bootstrap_all_resources(
    model: str,
    ollama_base: str,
    unified_id: str = "full_corpus_715_v1",
    index_dir: Path | None = None,
    *,
    force_llm_warm: bool = True,
) -> dict[str, Any]:
    from rag_resource_cache import prime_interactive_retrieval, warm_all_resources

    index_dir = index_dir or Path("data/processed/index")
    collection, embed_model, manifest = warm_all_resources(unified_id, index_dir)
    retrieval_prime = prime_interactive_retrieval(collection, embed_model)
    llm_result = warmup_fast_chat(model, ollama_base, force=force_llm_warm)
    ready = get_resource_ready_status(
        unified_id=unified_id,
        index_dir=index_dir,
        embed_model=embed_model,
        model=model,
    )
    payload = {
        "bootstrap": True,
        "model_name": model,
        "resource_ready": ready,
        "llm_warm": llm_result,
        "retrieval_prime": retrieval_prime,
        "collection": collection,
        "embed_model": embed_model,
        "manifest": manifest,
    }
    trace_payload = {
        "bootstrap": True,
        "model_name": model,
        "resource_ready": ready,
        "llm_warm": {k: v for k, v in llm_result.items() if k != "ollama_meta"},
        "retrieval_prime": retrieval_prime,
    }
    append_warmup_trace("bootstrap", trace_payload)
    return payload


def mark_accurate_llm_run(model: str, ollama_base: str = "http://127.0.0.1:11434") -> None:
    if ACCURATE_NUM_CTX == FAST_NUM_CTX:
        _state["last_llm_num_ctx"] = ACCURATE_NUM_CTX
        _state["last_latency_mode"] = "accurate"
        _state["model"] = model
        _state["fast_warm_valid"] = True
        _state["accurate_invalidated"] = False
        append_warmup_trace(
            "accurate_shared_context",
            {"model_name": model, "last_llm_num_ctx": ACCURATE_NUM_CTX},
        )
        return
    _state["last_llm_num_ctx"] = ACCURATE_NUM_CTX
    _state["last_latency_mode"] = "accurate"
    _state["model"] = model
    _state["fast_warm_valid"] = False
    _state["accurate_invalidated"] = True
    append_warmup_trace(
        "accurate_invalidate",
        {"model_name": model, "last_llm_num_ctx": ACCURATE_NUM_CTX},
    )
    # Re-warm Fast context slot immediately so next Fast request starts warm.
    warmup_fast_chat(
        model, ollama_base, force=True, rewarm_reason_override="accurate_run_after_fast"
    )


def mark_fast_llm_run(model: str, num_ctx: int | None = None) -> None:
    ctx = num_ctx if num_ctx is not None else FAST_NUM_CTX
    _state["last_llm_num_ctx"] = ctx
    _state["last_latency_mode"] = "fast"
    _state["model"] = model
    _state["accurate_invalidated"] = False


def legacy_warmup_generate(model: str, base_url: str = "http://127.0.0.1:11434") -> dict[str, Any]:
    """Old /api/generate num_ctx=512 warm-up (benchmark A/B baseline)."""
    payload = json.dumps(
        {
            "model": model,
            "prompt": "hello",
            "stream": False,
            "options": {"num_predict": 8, "num_ctx": 512},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    _state["fast_warm_valid"] = False
    _state["last_llm_num_ctx"] = 512
    _state["last_latency_mode"] = "legacy_warmup"
    _state["api_type"] = "generate"
    return {
        "warmup_mode": "legacy",
        "warmup_elapsed_s": round(elapsed, 3),
        "api_type": "generate",
        "num_ctx": 512,
        "warmup_response": {
            "load_duration_ns": data.get("load_duration"),
            "total_duration_ns": data.get("total_duration"),
        },
    }


def invalidate_fast_warm_for_test(reason: str = "test_invalidate") -> None:
    _state["fast_warm_valid"] = False
    _state["warmed_at"] = 0
    append_warmup_trace("invalidate", {"reason": reason})
