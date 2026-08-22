"""Optional offline cross-encoder scoring for Advanced retrieval.

The production runtime never downloads a model.  A model must already exist
under ``models/`` (or an explicitly configured local path).  The default model
is the Apache-2.0, non-Chinese ``cross-encoder/ms-marco-MiniLM-L4-v2`` snapshot.
Its score is supplied to the multilingual Gemma listwise ranker as an
additional signal; it is not allowed to discard protected Dense/BM25 hits.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "models" / "cross-encoder-ms-marco-MiniLM-L4-v2"
_MODEL = None
_MODEL_PATH = ""
_LOCK = threading.Lock()


def configured_model_path() -> Path:
    raw = os.environ.get("MARITIME_ADVANCED_CROSS_ENCODER", "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_MODEL_DIR


def is_available() -> bool:
    path = configured_model_path()
    return path.is_dir() and (path / "config.json").exists()


def _load_model():
    global _MODEL, _MODEL_PATH
    path = configured_model_path().resolve()
    key = str(path)
    with _LOCK:
        if _MODEL is not None and _MODEL_PATH == key:
            return _MODEL
        if not is_available():
            raise FileNotFoundError(f"offline cross-encoder not installed: {path}")
        from sentence_transformers import CrossEncoder

        _MODEL = CrossEncoder(
            key,
            device="cpu",
            local_files_only=True,
            trust_remote_code=False,
        )
        _MODEL_PATH = key
        return _MODEL


def score_candidates(
    query: str,
    candidates: list[Any],
    *,
    preview_chars: int = 1300,
) -> tuple[list[float], dict[str, Any]]:
    if not candidates:
        return [], {"used": False, "reason": "no_candidates"}
    if not is_available():
        return [], {
            "used": False,
            "reason": "model_not_installed",
            "model_path": str(configured_model_path()),
        }
    started = time.perf_counter()
    try:
        model = _load_model()
        passages = [
            str(getattr(candidate, "text", "") or "")[:preview_chars]
            for candidate in candidates
        ]
        values = model.predict(
            [(query, passage) for passage in passages],
            batch_size=16,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = [float(value) for value in values]
        return scores, {
            "used": True,
            "backend": "sentence_transformers_cross_encoder",
            "model_path": str(configured_model_path()),
            "candidate_count": len(candidates),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "score_min": round(min(scores), 6) if scores else None,
            "score_max": round(max(scores), 6) if scores else None,
        }
    except Exception as exc:
        return [], {
            "used": False,
            "reason": "cross_encoder_error",
            "model_path": str(configured_model_path()),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }
