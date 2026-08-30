"""Build document_profiles_v1.json from the current sparse FTS sidecar."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "rag" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from accurate_hybrid_v2 import SPARSE_DB_NAME
from document_profile_catalog import PROFILE_FILE_NAME, build_document_profiles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-id", default="full_corpus_715_v1")
    args = parser.parse_args()
    folder = ROOT / "data" / "processed" / "index" / f"unified_{args.unified_id}"
    payload = build_document_profiles(folder / SPARSE_DB_NAME, folder / PROFILE_FILE_NAME)
    print(json.dumps({"document_count": payload["document_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
