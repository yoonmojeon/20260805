"""Unit tests for token-aware embedding chunk preparation."""
from __future__ import annotations

from embedding_chunk_policy import prepare_chunks_for_embedding
from index_build_lib import filter_chunks_for_index


class CharacterTokenizer:
    """Small deterministic tokenizer used without loading model weights."""

    def encode(self, text, *, add_special_tokens=True, truncation=False):
        ids = list(range(len(text)))
        return ([-1] + ids + [-2]) if add_special_tokens else ids

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    ):
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


def test_oversized_chunk_is_split_and_bounded():
    tokenizer = CharacterTokenizer()
    chunk = {
        "chunk_id": "doc_p1_e1",
        "doc_id": "doc",
        "element_type": "text",
        "text": ("A" * 300) + ".\n" + ("B" * 300) + ".\n" + ("C" * 300),
    }

    def render(item):
        return f"source=KR\n{item['text']}", "text_native"

    chunks, documents, modes, stats = prepare_chunks_for_embedding(
        [chunk],
        tokenizer=tokenizer,
        model_name="intfloat/multilingual-e5-base",
        render_embedding=render,
        max_tokens=420,
        overlap_tokens=60,
    )
    assert stats.oversized_chunks == 1
    assert len(chunks) > 1
    assert len(chunks) == len(documents) == len(modes)
    assert max(c["embedding_token_count"] for c in chunks) <= 420
    assert all(c["split_from"] == "doc_p1_e1" for c in chunks)


def test_structured_table_modes():
    text = {"chunk_id": "t", "element_type": "text", "text": "normal text"}
    table = {
        "chunk_id": "r",
        "element_type": "table",
        "chunk_type": "table_row",
        "text": "row data",
    }
    include_types = frozenset({"text", "table", "picture"})
    included = filter_chunks_for_index([text, table], include_types, set(), 1, "include")
    excluded = filter_chunks_for_index([text, table], include_types, set(), 1, "exclude")
    only = filter_chunks_for_index([text, table], include_types, set(), 1, "only")
    assert [c["chunk_id"] for c in included] == ["t", "r"]
    assert [c["chunk_id"] for c in excluded] == ["t"]
    assert [c["chunk_id"] for c in only] == ["r"]


if __name__ == "__main__":
    test_oversized_chunk_is_split_and_bounded()
    test_structured_table_modes()
    print("ok")
