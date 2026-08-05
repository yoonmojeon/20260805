"""Token-aware chunk preparation for embedding-policy-v1.

Existing layout/clause chunks are preserved when they fit the embedding model.
Only oversized chunks are split, preferring paragraph/sentence boundaries and
retaining a token overlap between adjacent segments.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from typing import Callable, Iterable

from embedding_policy import e5_passage_prefix


EMBEDDING_POLICY_VERSION = "embedding-policy-v1"
CHUNK_POLICY_VERSION = "chunk-v1-420-60"
DEFAULT_MAX_EMBEDDING_TOKENS = 420
DEFAULT_EMBEDDING_OVERLAP_TOKENS = 60

_BOUNDARY_RE = re.compile(r"(?:\r?\n\s*)+|(?<=[.!?;:])\s+|(?<=다\.)\s+")


@dataclass(frozen=True)
class SplitStats:
    input_chunks: int
    output_chunks: int
    oversized_chunks: int
    generated_segments: int
    max_tokens_before: int
    max_tokens_after: int

    def as_dict(self) -> dict[str, int]:
        return {
            "input_chunks": self.input_chunks,
            "output_chunks": self.output_chunks,
            "oversized_chunks": self.oversized_chunks,
            "generated_segments": self.generated_segments,
            "max_tokens_before": self.max_tokens_before,
            "max_tokens_after": self.max_tokens_after,
        }


def embedding_token_count(tokenizer, text: str, model_name: str) -> int:
    passage = f"{e5_passage_prefix(model_name)}{text}"
    return len(tokenizer.encode(passage, add_special_tokens=True, truncation=False))


def _token_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    ids = list(encoded["input_ids"])
    offsets = [tuple(x) for x in encoded["offset_mapping"]]
    return ids, offsets


def _boundary_token_end(
    text: str,
    offsets: list[tuple[int, int]],
    token_start: int,
    ideal_end: int,
) -> int:
    """Choose the last useful text boundary in the final 35% of a window."""
    if ideal_end >= len(offsets):
        return ideal_end
    search_start_token = token_start + max(1, int((ideal_end - token_start) * 0.65))
    char_start = offsets[min(search_start_token, ideal_end - 1)][0]
    char_end = offsets[ideal_end - 1][1]
    matches = list(_BOUNDARY_RE.finditer(text, char_start, char_end))
    if not matches:
        return ideal_end
    boundary_char = matches[-1].end()
    token_ends = [end for _, end in offsets]
    candidate = bisect.bisect_right(token_ends, boundary_char)
    return min(ideal_end, max(token_start + 1, candidate))


def _split_body(
    text: str,
    tokenizer,
    *,
    body_token_budget: int,
    overlap_tokens: int,
) -> list[str]:
    ids, offsets = _token_offsets(tokenizer, text)
    if len(ids) <= body_token_budget:
        return [text]
    if body_token_budget < 32:
        raise ValueError(f"Embedding header leaves too little body capacity: {body_token_budget}")

    overlap = min(overlap_tokens, max(0, body_token_budget // 2))
    segments: list[str] = []
    start = 0
    while start < len(ids):
        ideal_end = min(len(ids), start + body_token_budget)
        end = _boundary_token_end(text, offsets, start, ideal_end)
        if end <= start:
            end = ideal_end
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        segment = text[char_start:char_end].strip()
        if segment:
            segments.append(segment)
        if end >= len(ids):
            break
        next_start = max(start + 1, end - overlap)
        start = next_start
    return segments


def split_chunk_for_embedding(
    chunk: dict,
    *,
    tokenizer,
    model_name: str,
    render_embedding: Callable[[dict], tuple[str, str]],
    max_tokens: int = DEFAULT_MAX_EMBEDDING_TOKENS,
    overlap_tokens: int = DEFAULT_EMBEDDING_OVERLAP_TOKENS,
) -> tuple[list[dict], list[str], list[str], int]:
    """Return prepared chunks, rendered documents, modes, and original token count."""
    rendered, mode = render_embedding(chunk)
    original_count = embedding_token_count(tokenizer, rendered, model_name)
    if original_count <= max_tokens:
        out = dict(chunk)
        out["embedding_token_count"] = original_count
        out["chunk_policy_version"] = CHUNK_POLICY_VERSION
        return [out], [rendered], [mode], original_count

    raw_text = str(chunk.get("text", "")).strip()
    if not raw_text:
        raise ValueError(f"Oversized chunk has no splittable text: {chunk.get('chunk_id')}")

    empty = dict(chunk)
    empty["text"] = ""
    empty_rendered, _ = render_embedding(empty)
    header_tokens = embedding_token_count(tokenizer, empty_rendered, model_name)
    body_budget = max_tokens - header_tokens
    bodies = _split_body(
        raw_text,
        tokenizer,
        body_token_budget=body_budget,
        overlap_tokens=overlap_tokens,
    )

    accepted_bodies: list[str] = []
    pending_bodies = list(bodies)
    while pending_bodies:
        body = pending_bodies.pop(0)
        probe = dict(chunk)
        probe["text"] = body
        probe_document, _ = render_embedding(probe)
        probe_count = embedding_token_count(tokenizer, probe_document, model_name)
        if probe_count <= max_tokens:
            accepted_bodies.append(body)
            continue

        body_ids = tokenizer.encode(body, add_special_tokens=False, truncation=False)
        reduced_budget = max(32, len(body_ids) - (probe_count - max_tokens) - 8)
        if reduced_budget >= len(body_ids):
            reduced_budget = max(32, len(body_ids) - 16)
        smaller = _split_body(
            body,
            tokenizer,
            body_token_budget=reduced_budget,
            overlap_tokens=overlap_tokens,
        )
        if len(smaller) <= 1:
            raise ValueError(
                f"Unable to fit oversized chunk within policy ({probe_count}>{max_tokens}): "
                f"{chunk.get('chunk_id')}"
            )
        pending_bodies = smaller + pending_bodies

    prepared: list[dict] = []
    documents: list[str] = []
    modes: list[str] = []
    base_id = str(chunk.get("chunk_id", ""))
    for index, body in enumerate(accepted_bodies, start=1):
        part = dict(chunk)
        part["text"] = body
        part["text_char_count"] = len(body)
        part["chunk_id"] = f"{base_id}__e{index:03d}"
        part["split_from"] = base_id
        part["embedding_split_index"] = index
        part["embedding_split_count"] = len(accepted_bodies)
        part["chunk_policy_version"] = CHUNK_POLICY_VERSION
        document, part_mode = render_embedding(part)
        token_count = embedding_token_count(tokenizer, document, model_name)
        if token_count > max_tokens:
            raise ValueError(
                f"Split chunk still exceeds policy ({token_count}>{max_tokens}): {part['chunk_id']}"
            )
        part["embedding_token_count"] = token_count
        prepared.append(part)
        documents.append(document)
        modes.append(part_mode)
    return prepared, documents, modes, original_count


def prepare_chunks_for_embedding(
    chunks: Iterable[dict],
    *,
    tokenizer,
    model_name: str,
    render_embedding: Callable[[dict], tuple[str, str]],
    max_tokens: int = DEFAULT_MAX_EMBEDDING_TOKENS,
    overlap_tokens: int = DEFAULT_EMBEDDING_OVERLAP_TOKENS,
) -> tuple[list[dict], list[str], list[str], SplitStats]:
    prepared: list[dict] = []
    documents: list[str] = []
    modes: list[str] = []
    input_count = 0
    oversized = 0
    max_before = 0
    max_after = 0

    for chunk in chunks:
        input_count += 1
        parts, texts, part_modes, before = split_chunk_for_embedding(
            chunk,
            tokenizer=tokenizer,
            model_name=model_name,
            render_embedding=render_embedding,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        if len(parts) > 1:
            oversized += 1
        max_before = max(max_before, before)
        max_after = max(max_after, *(int(p["embedding_token_count"]) for p in parts))
        prepared.extend(parts)
        documents.extend(texts)
        modes.extend(part_modes)

    stats = SplitStats(
        input_chunks=input_count,
        output_chunks=len(prepared),
        oversized_chunks=oversized,
        generated_segments=len(prepared) - input_count,
        max_tokens_before=max_before,
        max_tokens_after=max_after,
    )
    return prepared, documents, modes, stats
