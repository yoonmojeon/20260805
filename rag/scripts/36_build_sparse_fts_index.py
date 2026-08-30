"""Build the Accurate FTS5 sidecar from the existing unified Chroma corpus."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from accurate_hybrid_v2 import build_sparse_index, sparse_index_path
from rag_resource_cache import load_unified_collection, unified_index_fingerprint
from project_paths import DEFAULT_RAG_COLLECTION


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified", default=DEFAULT_RAG_COLLECTION)
    parser.add_argument("--index-dir", type=Path, default=ROOT / "data/processed/index")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    collection, _model, manifest = load_unified_collection(args.unified, args.index_dir)
    fingerprint = unified_index_fingerprint(args.unified, args.index_dir)
    out_path = args.out or sparse_index_path(args.index_dir, args.unified)
    result = build_sparse_index(
        collection,
        out_path=out_path,
        fingerprint=fingerprint,
        expected_count=int(manifest.get("indexed_chunks") or collection.count()),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
