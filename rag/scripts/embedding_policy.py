from __future__ import annotations

import re

# 중국 기관·모델 계열 차단 (절대 사용 금지)
CHINESE_MODEL_BLOCKLIST_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"baai",
        r"\bbge[-_]",
        r"qwen",
        r"tongyi",
        r"dashscope",
        r"alibaba",
        r"damo",
        r"zhipu",
        r"chatglm",
        r"\bglm[-_]",
        r"baichuan",
        r"internlm",
        r"internvl",
        r"text2vec",
        r"\bm3e[-_]",
        r"piccolo",
        r"yuque",
        r"modelscope",
        r"wenxin",
        r"ernie",
        r"paddle",
        r"telechat",
        r"minimax",
        r"moonshot",
        r"deepseek",
    )
)

# 영어 위주 단일어 모델 차단 (한·영 규정 문서에 부적합)
ENGLISH_ONLY_MODEL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"all-minilm-l6",
        r"all-mpnet-base",
        r"e5-small(?!.*multilingual)",
        r"nomic-embed",
    )
)

LOCAL_PROVIDER = "sentence-transformers"
REQUIRED_LANGUAGES = ("ko", "en")

# 로컬 HuggingFace 모델만 허용 (한국어·영어 다국어 지원)
ALLOWED_EMBEDDING_PRESETS: dict[str, dict[str, str | list[str]]] = {
    "e5-base": {
        "provider": LOCAL_PROVIDER,
        "model": "intfloat/multilingual-e5-base",
        "revision": "d128750597153bb5987e10b1c3493a34e5a4502a",
        "deployment": "local",
        "languages": ["ko", "en"],
        "note": "Microsoft E5 multilingual — 로컬 실행, 한·영 권장 기본",
    },
    "e5-large": {
        "provider": LOCAL_PROVIDER,
        "model": "intfloat/multilingual-e5-large",
        "deployment": "local",
        "languages": ["ko", "en"],
        "note": "Microsoft E5 multilingual large — 로컬, 품질↑·속도↓",
    },
    "snowflake-arctic": {
        "provider": LOCAL_PROVIDER,
        "model": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "deployment": "local",
        "languages": ["ko", "en"],
        "note": "Snowflake Arctic Embed — 로컬, 다국어",
    },
    "paraphrase-multilingual": {
        "provider": LOCAL_PROVIDER,
        "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "deployment": "local",
        "languages": ["ko", "en"],
        "note": "로컬 다국어 MiniLM — 가볍지만 e5보다 성능 낮음",
    },
}

DEFAULT_EMBEDDING_PRESET = "e5-base"


def is_chinese_model_blocked(model_name: str) -> bool:
    return any(pattern.search(model_name) for pattern in CHINESE_MODEL_BLOCKLIST_PATTERNS)


def is_english_only_model_blocked(model_name: str) -> bool:
    return any(pattern.search(model_name) for pattern in ENGLISH_ONLY_MODEL_PATTERNS)


def validate_embedding_model(model_name: str) -> None:
    if is_chinese_model_blocked(model_name):
        raise ValueError(
            f"Chinese-origin embedding model is not allowed: {model_name}\n"
            f"Allowed local presets: {', '.join(sorted(ALLOWED_EMBEDDING_PRESETS))}"
        )
    if is_english_only_model_blocked(model_name):
        raise ValueError(
            f"English-only embedding model is not allowed for KR/EN regulations: {model_name}\n"
            f"Use a multilingual local preset (e.g. e5-base, e5-large)."
        )


def assert_local_provider(provider: str) -> None:
    if provider != LOCAL_PROVIDER:
        raise ValueError(
            f"Only local embeddings are allowed (provider={LOCAL_PROVIDER}). "
            f"Cloud/API embedding providers are disabled."
        )


def resolve_embedding_config(preset: str, model_override: str | None = None) -> dict[str, str | list[str]]:
    if preset not in ALLOWED_EMBEDDING_PRESETS:
        allowed = ", ".join(sorted(ALLOWED_EMBEDDING_PRESETS))
        raise ValueError(f"Unknown preset '{preset}'. Allowed presets: {allowed}")

    config: dict[str, str | list[str]] = dict(ALLOWED_EMBEDDING_PRESETS[preset])
    if model_override:
        validate_embedding_model(model_override)
        config["model"] = model_override
    else:
        validate_embedding_model(str(config["model"]))

    assert_local_provider(str(config["provider"]))
    config["preset"] = preset
    return config


def e5_query_prefix(model_name: str) -> str:
    lower = model_name.lower()
    if "e5" in lower and "multilingual" in lower:
        return "query: "
    return ""


def e5_passage_prefix(model_name: str) -> str:
    lower = model_name.lower()
    if "e5" in lower and "multilingual" in lower:
        return "passage: "
    return ""


def embed_texts_local(
    texts: list[str],
    model_name: str,
    *,
    for_query: bool = False,
    timing=None,
) -> list[list[float]]:
    """Run sentence-transformers locally (no API calls)."""
    if timing is not None:
        if hasattr(timing, "set_cache"):
            timing.set_cache("embedding_model_loaded_from_cache", embedding_model_is_cached(model_name))
        if hasattr(timing, "mark"):
            timing.mark("t_query_embedding_start")
    prefix_fn = e5_query_prefix if for_query else e5_passage_prefix
    prefix = prefix_fn(model_name)
    prefixed = [f"{prefix}{text}" if prefix else text for text in texts]

    encoder = get_sentence_transformer(model_name)
    vectors = encoder.encode(
        prefixed,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 8,
    )
    if timing is not None and hasattr(timing, "mark"):
        timing.mark("t_query_embedding_end")
    return vectors.tolist()


_ENCODER_CACHE: dict[str, object] = {}


def embedding_model_is_cached(model_name: str) -> bool:
    return model_name in _ENCODER_CACHE


def clear_encoder_cache() -> None:
    _ENCODER_CACHE.clear()


def get_sentence_transformer(model_name: str):
    """Load embedding model once per process (reuse across queries)."""
    if model_name not in _ENCODER_CACHE:
        import os

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import SentenceTransformer

        revision = next(
            (
                str(config.get("revision"))
                for config in ALLOWED_EMBEDDING_PRESETS.values()
                if str(config.get("model")) == model_name and config.get("revision")
            ),
            None,
        )
        kwargs = {"revision": revision} if revision else {}
        _ENCODER_CACHE[model_name] = SentenceTransformer(model_name, **kwargs)
    return _ENCODER_CACHE[model_name]


def get_embedding_tokenizer(model_name: str):
    """Return the exact tokenizer used by the cached sentence-transformer."""
    encoder = get_sentence_transformer(model_name)
    tokenizer = getattr(encoder, "tokenizer", None)
    if tokenizer is None:
        first_module = encoder._first_module()  # sentence-transformers compatibility
        tokenizer = getattr(first_module, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(f"Embedding tokenizer is unavailable: {model_name}")
    return tokenizer


def warm_embed_model(model_name: str) -> None:
    """Eager-load embedding weights (UI startup / pre-warm)."""
    get_sentence_transformer(model_name)
