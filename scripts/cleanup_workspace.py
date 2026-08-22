"""Remove one-off experiment output that the app, tests and eval scripts never read.

Everything listed here was checked against the source tree: the live Chroma
index, chunk store, precise-table crops, question sets and the manifests that
the index build reads back are deliberately absent.

Note for anyone extending the list: ``rag/data/processed/index`` and
``rag/data/processed/chunks`` are junctions onto ``data/processed``. They look
like an 8 GB duplicate and are not one; deleting through them destroys the live
index.

Dry run by default::

    python scripts/cleanup_workspace.py            # 무엇을 지울지 보여주기만
    python scripts/cleanup_workspace.py --apply    # 실제 삭제
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "data" / "processed" / "logs"
EVAL = ROOT / "data" / "eval"

# Scratch directories that pytest and the UI recreate on demand.
SCRATCH_DIRS = [
    ROOT / ".test-tmp",
    ROOT / ".test-tmp-publish",
    ROOT / ".pytest_cache",
    ROOT / ".pytest-tmp",
    ROOT / ".pytest_scratch",
    # outputs/ppt_ui_captures holds the presentation screenshots, so only the
    # pytest scratch and stray server logs under outputs/ are removed.
    ROOT / "outputs" / "pytest_map_final_20260815",
    ROOT / "outputs" / "pytest_map_validation_20260815",
    ROOT / "data" / "pytest_verify_20260814",
    ROOT / "data" / "pytest_verify_20260814_ui_final2",
    ROOT / "data" / "processed" / "pytest_text_audit",
]

# Evaluation runs that later revisions replaced.
STALE_LOG_DIRS = [
    LOGS / "text_rag_eval_v2",
    LOGS / "augmented_answer_eval",
    LOGS / "gemma_empty_diag",
    LOGS / "ui_server",
]

# One-off dumps from model comparisons and pipeline debugging.
STALE_LOG_FILES = [
    LOGS / "llm_contract_diag_table10.json",
    LOGS / "llm_contract_diag_table10.md",
    LOGS / "llm_compare_table10.jsonl",
    LOGS / "llm_compare_table10_summary.txt",
    LOGS / "table_llm_eval10.txt",
    LOGS / "table_search_verify10.json",
    LOGS / "table_search_verify10.md",
    LOGS / "debug_spa_raw.json",
    LOGS / "manual10_eval.json",
    LOGS / "suite_100_mixed_results.json",
    LOGS / "quality_30_last.json",
    LOGS / "quality_30_rerun_stdout.txt",
    LOGS / "quality_50_models.json",
    LOGS / "smoke_categories_last.json",
    LOGS / "pipeline_complete.txt",
    *[LOGS / f"quality_30_{m}.json" for m in ("gemma4_12b", "llama3.1_8b", "mistral-nemo_12b")],
    *[LOGS / f"quality_50_{m}.json" for m in ("gemma4_12b", "llama3.1_8b", "mistral-nemo_12b")],
    *sorted(LOGS.glob("model_compare_10_*.json")),
]

# Question drafts superseded by the curated sets, and result dumps that no
# script reads back. The gemma dumps used by the demo docx export stay.
STALE_EVAL_FILES = [
    EVAL / "table_questions_22docs_practical_v1_draft.jsonl",
    EVAL / "table_questions_22docs_v1.jsonl",
    EVAL / "table_questions_22docs_v2.jsonl",
    EVAL / "kr_1_2025_questions_pilot_30p.jsonl",
    EVAL / "example_10_gold.jsonl",
    EVAL / "fast_questions.jsonl",
    EVAL / "fast_scope_tests.jsonl",
    EVAL / "table_schema_regression.jsonl",
    EVAL / "hierarchical_retrieval_20_gemma4_12b.json",
    EVAL / "hierarchical_retrieval_20_gemma4_12b_after.json",
    EVAL / "router_comparison.json",
]

GLOB_GROUPS = [
    (LOGS, "*_merge_comparison.txt"),  # 715 preprocessing diffs from the corpus build
    (LOGS, "*.log"),  # preprocess/index build stdout
    (ROOT / "outputs", "ui_server_map_*.log"),
    (ROOT / "outputs" / "ppt_ui_captures", "ui_restart_*.log"),
]


def pycache_dirs() -> list[Path]:
    """__pycache__ under our own packages only, never inside .venv."""
    skip = {".venv", ".git", "node_modules"}
    out: list[Path] = []
    for path in ROOT.rglob("__pycache__"):
        parts = set(path.relative_to(ROOT).parts)
        if parts & skip:
            continue
        # rag/data/* are junctions onto data/*; never walk through them.
        if "data" in path.relative_to(ROOT).parts[:2] and path.parts[-2] == "processed":
            continue
        out.append(path)
    return out


def size_of(path: Path) -> int:
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return path.stat().st_size if path.exists() else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="실제로 삭제한다")
    args = parser.parse_args()

    targets: list[Path] = []
    targets += [p for p in SCRATCH_DIRS if p.exists()]
    targets += [p for p in STALE_LOG_DIRS if p.exists()]
    targets += [p for p in STALE_LOG_FILES if p.exists()]
    targets += [p for p in STALE_EVAL_FILES if p.exists()]
    for base, pattern in GLOB_GROUPS:
        targets += sorted(base.glob(pattern))
    targets += pycache_dirs()

    total = 0
    files = 0
    for path in targets:
        size = size_of(path)
        total += size
        files += 1 if path.is_file() else sum(1 for _ in path.rglob("*"))
        if size > 1_000_000 or path.is_dir():
            print(f"  {size/1e6:8.1f} MB  {path.relative_to(ROOT)}")
        if args.apply:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    verb = "삭제함" if args.apply else "삭제 예정(드라이런)"
    print(f"\n{verb}: {len(targets)}개 항목 · 파일 {files}개 · {total/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
