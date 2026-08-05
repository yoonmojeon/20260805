"""Rebuild the complete policy-v1 text collection from the real 715 PDFs."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/full_corpus_715.csv"


def main() -> None:
    if not MANIFEST.exists():
        subprocess.run(
            [sys.executable, "scripts/prepare_full_corpus_715.py"],
            cwd=ROOT,
            check=True,
        )

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    subprocess.run(
        [
            sys.executable,
            "scripts/10_build_unified_index.py",
            "--doc-list",
            str(MANIFEST),
            "--manifest",
            str(MANIFEST),
            "--collection-id",
            "full_corpus_715_v1",
            "--embedding-preset",
            "e5-base",
            "--include-types",
            "text,picture",
            "--structured-tables",
            "exclude",
            "--max-embedding-tokens",
            "420",
            "--embedding-overlap-tokens",
            "60",
        ],
        cwd=ROOT,
        check=True,
        env=env,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/35_build_bm25_index.py",
            "--unified",
            "full_corpus_715_v1",
            "--rebuild",
        ],
        cwd=ROOT,
        check=True,
        env=env,
    )


if __name__ == "__main__":
    main()
