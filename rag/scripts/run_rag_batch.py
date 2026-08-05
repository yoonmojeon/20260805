"""
Run the MaritimeRAG preprocessing + indexing pipeline for one or more manifest documents.

Typical full run (KR 1편 전편):
  python scripts/00_build_manifest.py
  python scripts/run_rag_batch.py --doc-id kr_1_2025 --run-eval

Steps: pdf, layout, merge, crop, chunks, quality, index, eval
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ALL_STEPS = ("pdf", "layout", "merge", "crop", "chunks", "quality", "index", "eval")
SKIP_MARKER = "index_skipped.json"


def requested_steps_complete(
    project_root: Path,
    doc_id: str,
    steps: tuple[str, ...],
    run_eval: bool,
) -> bool:
    """Return True when the last requested durable output already exists."""
    if run_eval or "eval" in steps:
        return False
    if "index" in steps:
        index_dir = project_root / "data/processed/index" / doc_id
        return (index_dir / "chroma").exists() or (index_dir / "index_manifest.json").exists()
    if "quality" in steps:
        chunks_path = project_root / "data/processed/chunks" / doc_id / "chunks.jsonl"
        report_path = project_root / "data/processed/logs" / f"{doc_id}_chunk_quality_report.txt"
        return chunks_path.exists() and report_path.exists()
    if "chunks" in steps:
        return (project_root / "data/processed/chunks" / doc_id / "chunks.jsonl").exists()
    if "crop" in steps:
        return (
            project_root
            / "data/processed/crops_merged"
            / doc_id
            / "crops_manifest.jsonl"
        ).exists()
    if "merge" in steps:
        merged_dir = project_root / "data/processed/layout_json_merged" / doc_id
        return merged_dir.exists() and any(merged_dir.glob("page_*.json"))
    if "layout" in steps:
        layout_dir = project_root / "data/processed/layout_json" / doc_id
        return layout_dir.exists() and any(layout_dir.glob("page_*.json"))
    if "pdf" in steps:
        pages_dir = project_root / "data/processed/pages" / doc_id
        return pages_dir.exists() and any(pages_dir.glob("page_*.png"))
    return False


def mark_index_skipped(project_root: Path, doc_id: str, reason: str) -> None:
    skip_dir = project_root / "data/processed/index" / doc_id
    skip_dir.mkdir(parents=True, exist_ok=True)
    (skip_dir / SKIP_MARKER).write_text(
        json.dumps({"reason": reason}, ensure_ascii=False),
        encoding="utf-8",
    )


def run_step(cmd: list[str], cwd: Path) -> None:
    print(f"\n[RUN] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_manifest_rows(
    manifest_path: Path,
    doc_ids: list[str] | None,
    source: str | None,
) -> list[dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    if doc_ids:
        df = df[df["doc_id"].isin(doc_ids)]
    if source:
        df = df[df["source"].str.upper() == source.upper()]

    if df.empty:
        raise ValueError("No manifest rows matched the filter.")

    dup = df[df["doc_id"].duplicated()]
    if not dup.empty:
        raise ValueError(
            f"Manifest contains duplicate doc_id values: {dup['doc_id'].tolist()}"
        )

    return df.to_dict(orient="records")


def count_page_images(pages_dir: Path) -> int:
    if not pages_dir.exists():
        return 0
    return len(list(pages_dir.glob("page_*.png")))


def process_document(
    row: dict,
    *,
    project_root: Path,
    python: str,
    steps: tuple[str, ...],
    dpi: int,
    overwrite_images: bool,
    layout_conf: float,
    layout_model: Path,
    run_eval: bool,
    eval_questions: Path,
    top_k: int,
) -> None:
    doc_id = str(row["doc_id"])
    pdf_path = Path(str(row["file_path"]))
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found for {doc_id}: {pdf_path}")

    pages_dir = project_root / "data/processed/pages" / doc_id
    layout_dir = project_root / "data/processed/layout_json" / doc_id
    merged_dir = project_root / "data/processed/layout_json_merged" / doc_id
    model_path = layout_model if layout_model.is_absolute() else project_root / layout_model

    print(f"\n{'=' * 60}\nDocument: {doc_id}\nPDF: {pdf_path}\n{'=' * 60}")

    if "pdf" in steps:
        cmd = [
            python,
            "scripts/01_pdf_to_images.py",
            "--pdf",
            str(pdf_path),
            "--out-dir",
            "data/processed/pages",
            "--doc-id",
            doc_id,
            "--dpi",
            str(dpi),
        ]
        if overwrite_images:
            cmd.append("--overwrite")
        run_step(cmd, project_root)

    page_count = count_page_images(pages_dir)
    print(f"Page images: {page_count} under {pages_dir}")
    if page_count == 0 and any(s in steps for s in ("layout", "merge", "crop", "chunks")):
        raise RuntimeError(f"No page images for {doc_id}. Run pdf step first.")

    if "layout" in steps:
        if not model_path.exists():
            raise FileNotFoundError(f"Layout model not found: {model_path}")
        run_step(
            [
                python,
                "scripts/02_layout_detect.py",
                "--image-dir",
                str(pages_dir),
                "--doc-id",
                doc_id,
                "--model",
                str(model_path),
                "--out-dir",
                "data/processed/layout_json",
                "--conf",
                str(layout_conf),
            ],
            project_root,
        )

    if "merge" in steps:
        run_step(
            [python, "scripts/06_merge_text_blocks.py", "--doc-id", doc_id],
            project_root,
        )

    if "crop" in steps:
        run_step(
            [
                python,
                "scripts/03_crop_elements.py",
                "--doc-id",
                doc_id,
                "--merged",
            ],
            project_root,
        )

    if "chunks" in steps:
        if not merged_dir.exists():
            raise FileNotFoundError(f"Merged layout missing: {merged_dir}")
        run_step(
            [python, "scripts/07_extract_chunks.py", "--doc-id", doc_id],
            project_root,
        )

    if "quality" in steps:
        run_step(
            [python, "scripts/08_analyze_chunks_quality.py", "--doc-id", doc_id],
            project_root,
        )

    if "index" in steps:
        run_step(
            [python, "scripts/09_build_index.py", "--doc-id", doc_id],
            project_root,
        )

    if "eval" in steps or run_eval:
        questions = eval_questions
        if not questions.is_absolute():
            questions = project_root / questions
        if not questions.exists():
            print(f"[WARN] Eval questions not found, skipping: {questions}")
        else:
            run_step(
                [
                    python,
                    "scripts/11_eval_retrieval.py",
                    "--doc-id",
                    doc_id,
                    "--top-k",
                    str(top_k),
                    "--questions",
                    str(questions),
                ],
                project_root,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run RAG pipeline for manifest documents.")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/pdf_manifest.csv"))
    parser.add_argument(
        "--doc-list",
        type=Path,
        default=None,
        help="CSV with doc_id column (e.g. data/manifests/pilot_100_docs.csv)",
    )
    parser.add_argument("--doc-id", action="append", dest="doc_ids", help="Repeatable doc_id filter")
    parser.add_argument("--source", type=str, default=None, help="Filter by source (KR, DNV, ABS)")
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help=f"Comma-separated steps or 'all'. Choices: {','.join(ALL_STEPS)}",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PDF render DPI (200 recommended for full books)")
    parser.add_argument("--overwrite-images", action="store_true", help="Re-render all page PNGs")
    parser.add_argument("--layout-conf", type=float, default=0.25)
    parser.add_argument(
        "--layout-model",
        type=Path,
        default=Path("models/layout/yolov10m_doclaynet.pt"),
    )
    parser.add_argument("--run-eval", action="store_true", help="Run retrieval eval after index")
    parser.add_argument(
        "--eval-questions",
        type=Path,
        default=Path("data/eval/kr_1_2025_questions.jsonl"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--skip-on-error",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Continue batch on per-document failure (default: on when --doc-list is set)",
    )
    parser.add_argument(
        "--resume-completed",
        action="store_true",
        help="Skip documents whose last requested durable output already exists",
    )
    args = parser.parse_args()
    if args.skip_on_error is None:
        args.skip_on_error = args.doc_list is not None

    project_root = Path.cwd()
    if args.steps.strip().lower() == "all":
        steps = ALL_STEPS
    else:
        steps = tuple(s.strip().lower() for s in args.steps.split(",") if s.strip())
        unknown = [s for s in steps if s not in ALL_STEPS]
        if unknown:
            raise ValueError(f"Unknown steps: {unknown}. Valid: {ALL_STEPS}")

    if args.doc_list:
        if not args.doc_list.exists():
            raise FileNotFoundError(f"Doc list not found: {args.doc_list}")
        list_df = pd.read_csv(args.doc_list)
        list_ids = list_df["doc_id"].astype(str).tolist()
        full_df = pd.read_csv(args.manifest)
        rows = full_df[full_df["doc_id"].isin(list_ids)].to_dict(orient="records")
        if not rows:
            raise ValueError(f"No manifest rows for doc_ids in {args.doc_list}")
        order = {d: i for i, d in enumerate(list_ids)}
        rows.sort(key=lambda r: order.get(str(r["doc_id"]), 9999))
    else:
        rows = load_manifest_rows(args.manifest, args.doc_ids, args.source)

    print(f"Processing {len(rows)} document(s)")
    if len(rows) <= 10:
        print([r["doc_id"] for r in rows])

    failed: list[str] = []
    resumed = 0
    for row in rows:
        doc_id = str(row["doc_id"])
        if args.resume_completed and requested_steps_complete(
            project_root, doc_id, steps, args.run_eval
        ):
            print(f"[RESUME] {doc_id}: requested outputs already exist", flush=True)
            resumed += 1
            continue
        try:
            process_document(
                row,
                project_root=project_root,
                python=args.python,
                steps=steps,
                dpi=args.dpi,
                overwrite_images=args.overwrite_images,
                layout_conf=args.layout_conf,
                layout_model=args.layout_model,
                run_eval=args.run_eval,
                eval_questions=args.eval_questions,
                top_k=args.top_k,
            )
        except Exception as exc:
            if not args.skip_on_error:
                raise
            reason = str(exc)
            print(f"\n[SKIP] {doc_id}: {reason}\n", flush=True)
            mark_index_skipped(project_root, doc_id, reason)
            failed.append(doc_id)

    if failed:
        fail_log = project_root / "data/processed/logs/batch_skipped_docs.jsonl"
        fail_log.parent.mkdir(parents=True, exist_ok=True)
        with fail_log.open("a", encoding="utf-8") as f:
            for doc_id in failed:
                f.write(json.dumps({"doc_id": doc_id}, ensure_ascii=False) + "\n")
        print(f"\nSkipped {len(failed)} document(s). See {fail_log}")

    if resumed:
        print(f"\nResume skipped {resumed} completed document(s).")

    print("\nBatch pipeline completed.")


if __name__ == "__main__":
    main()
