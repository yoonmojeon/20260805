from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAG_SCRIPTS = ROOT / "rag" / "scripts"
for path in (ROOT, RAG_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from accurate_hybrid_v2 import (
    SparseFTSIndex,
    build_sparse_index,
    fuse_dense_sparse,
    protect_dense_candidates,
)


class FakeCollection:
    def __init__(self):
        self.rows = [
            (
                "dnv_answer",
                {
                    "doc_id": "dnv_doc",
                    "file_name": "DNV-CG-0264.pdf",
                    "source": "DNV",
                    "page_number": 12,
                    "clause_number": "5.2.3.1",
                },
                "5.2.3.1 Unless otherwise agreed, Remote Operation Centre requirements apply.",
            ),
            (
                "dnv_intro",
                {
                    "doc_id": "dnv_doc",
                    "file_name": "DNV-CG-0264.pdf",
                    "source": "DNV",
                    "page_number": 1,
                    "clause_number": "",
                },
                "Introduction and general objective.",
            ),
            (
                "kr_answer",
                {
                    "doc_id": "kr_doc",
                    "file_name": "1편_2025.pdf",
                    "source": "KR",
                    "page_number": 44,
                    "clause_number": "401",
                },
                "방식조치를 적용하는 경우의 요건과 예외를 규정한다.",
            ),
        ]

    def get(self, *, include, limit, offset):
        rows = self.rows[offset : offset + limit]
        return {
            "ids": [row[0] for row in rows],
            "metadatas": [row[1] for row in rows],
            "documents": [row[2] for row in rows],
        }


def test_fts_recovers_document_code_clause_and_korean_compound(tmp_path):
    path = tmp_path / "sparse.sqlite3"
    result = build_sparse_index(
        FakeCollection(),
        out_path=path,
        fingerprint="fp",
        expected_count=3,
        progress_every=0,
    )
    assert result["chunk_count"] == 3
    index = SparseFTSIndex(path, expected_fingerprint="fp")
    try:
        hits, _ = index.search(
            "DNV-CG-0264의 5.2.3.1 예외 조건",
            top_k=10,
            source="DNV",
            doc_id="dnv_doc",
        )
        assert hits and hits[0].chunk_id == "dnv_answer"
        assert all(hit.meta["source"] == "DNV" for hit in hits)

        korean, _ = index.search("방식조치의 요건과 예외", top_k=10, source="KR")
        assert korean and korean[0].chunk_id == "kr_answer"
    finally:
        index.close()


def test_rrf_keeps_dense_only_and_sparse_only_candidates():
    dense = {
        "ids": [["dense_only", "shared"]],
        "distances": [[0.2, 0.3]],
        "metadatas": [[
            {"doc_id": "d1", "source": "DNV"},
            {"doc_id": "d2", "source": "DNV"},
        ]],
        "documents": [["dense", "shared"]],
    }
    from accurate_hybrid_v2 import SparseHit

    sparse = [
        SparseHit("shared", 1, 4.0, -4.0, {"doc_id": "d2", "source": "DNV"}, "shared"),
        SparseHit("sparse_only", 2, 3.0, -3.0, {"doc_id": "d3", "source": "DNV"}, "sparse"),
    ]
    fused = fuse_dense_sparse(dense, sparse, rrf_k=60, top_k=10)
    assert fused[0].chunk_id == "shared"
    assert {hit.chunk_id for hit in fused} == {"dense_only", "shared", "sparse_only"}
    shared = fused[0]
    assert shared.dense_rank == 2
    assert shared.bm25_rank == 1


def test_protected_hybrid_never_drops_dense_prefix():
    dense = {
        "ids": [["d1", "d2", "d3"]],
        "distances": [[0.1, 0.2, 0.3]],
        "metadatas": [[
            {"doc_id": "one", "source": "DNV"},
            {"doc_id": "two", "source": "DNV"},
            {"doc_id": "three", "source": "DNV"},
        ]],
        "documents": [["one", "two", "three"]],
    }
    from accurate_hybrid_v2 import SparseHit

    fused = fuse_dense_sparse(
        dense,
        [SparseHit("s1", 1, 9.0, -9.0, {"doc_id": "s", "source": "KR"}, "sparse")],
        rrf_k=60,
        top_k=10,
    )
    protected = protect_dense_candidates(dense, fused, protected_k=2, top_k=10)
    assert [hit.chunk_id for hit in protected[:2]] == ["d1", "d2"]
    assert {hit.chunk_id for hit in protected} == {"d1", "d2", "d3", "s1"}
