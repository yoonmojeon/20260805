"""Reconstruct and quality-gate the 22-document KR table corpus (v2)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pdf_io import resolve_pdf_path
from table_v2_lib import (
    CAPTION_RE,
    PARSER_VERSION,
    assess_quality,
    build_v2_chunks,
    extract_coordinate_grid,
    rows_to_markdown,
)


DEFAULT_DOC_LIST = ROOT / "data/manifests/kr_table_top22.csv"
DEFAULT_MANIFEST = ROOT / "data/manifests/pdf_manifest.csv"
DEFAULT_V1_TABLES = ROOT / "data/processed/tables"
DEFAULT_LAYOUT = ROOT / "data/processed/layout_json_merged"
DEFAULT_TABLES_OUT = ROOT / "data/processed/tables_v2"
DEFAULT_CHUNKS_OUT = ROOT / "data/processed/chunks_v2"
DEFAULT_REPORT = ROOT / "data/processed/logs/kr_tables_v2_quality.json"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def page_layout_size(layout_root: Path, doc_id: str, page: int) -> tuple[float, float]:
    path = layout_root / doc_id / f"page_{page:04d}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data.get("width") or 1), float(data.get("height") or 1)


def pdf_rect(page: fitz.Page, bbox: list[float], layout_size: tuple[float, float]) -> fitz.Rect:
    width, height = layout_size
    sx = page.rect.width / max(width, 1)
    sy = page.rect.height / max(height, 1)
    x0, y0, x1, y1 = [float(v) for v in bbox]
    pad = 1.5
    return fitz.Rect(
        max(0, x0 * sx - pad),
        max(0, y0 * sy - pad),
        min(page.rect.x1, x1 * sx + pad),
        min(page.rect.y1, y1 * sy + pad),
    )


def explicit_v1_caption(table: dict) -> str:
    caption = str(table.get("caption") or "").strip()
    return caption if CAPTION_RE.match(caption) else ""


def process_doc(
    *,
    row: dict[str, str],
    manifest: Path,
    v1_tables_root: Path,
    layout_root: Path,
    tables_out: Path,
    chunks_out: Path,
) -> dict:
    doc_id = str(row["doc_id"])
    source = str(row.get("source") or "KR")
    file_name = str(row.get("file_name") or "")
    pdf_path = resolve_pdf_path(doc_id, manifest, None, ROOT)
    if pdf_path is None:
        raise FileNotFoundError(f"PDF not found for {doc_id}")
    if not file_name:
        file_name = pdf_path.name

    v1_path = v1_tables_root / doc_id / "tables.jsonl"
    source_tables = load_jsonl(v1_path)
    out_tables: list[dict] = []
    out_chunks: list[dict] = []
    counts: Counter[str] = Counter()
    strategies: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()

    doc = fitz.open(pdf_path)
    try:
        size_cache: dict[int, tuple[float, float]] = {}
        for old in source_tables:
            if old.get("is_pseudo_table"):
                counts["v1_pseudo_skipped"] += 1
                continue
            page_no = int(old.get("page") or 0)
            bbox = list(old.get("bbox") or [])
            if page_no < 1 or page_no > len(doc) or len(bbox) != 4:
                counts["invalid_location"] += 1
                continue
            if page_no not in size_cache:
                size_cache[page_no] = page_layout_size(layout_root, doc_id, page_no)
            page = doc[page_no - 1]
            rect = pdf_rect(page, bbox, size_cache[page_no])
            grid = extract_coordinate_grid(page, rect)
            if grid is None:
                record = {
                    "doc_id": doc_id,
                    "source_file": file_name,
                    "page": page_no,
                    "table_id": str(old.get("table_id") or ""),
                    "element_id": str(old.get("element_id") or ""),
                    "bbox": bbox,
                    "caption": explicit_v1_caption(old),
                    "section_title": str(old.get("section_title") or ""),
                    "parser_version": PARSER_VERSION,
                    "quality_score": 0.0,
                    "quality_status": "reject",
                    "quality_reasons": ["coordinate_grid_not_found"],
                    "quality_metrics": {},
                    "table_json": {"columns": [], "rows": []},
                    "column_names": [],
                    "row_count": 0,
                    "markdown_table": "",
                }
                out_tables.append(record)
                counts["reject"] += 1
                reason_counts["coordinate_grid_not_found"] += 1
                continue

            caption = grid.caption or explicit_v1_caption(old)
            score, status, reasons, metrics = assess_quality(
                columns=grid.columns,
                rows=grid.rows,
                caption=caption,
                strategy=grid.strategy,
            )
            strategies[grid.strategy] += 1
            counts[status] += 1
            reason_counts.update(reason.split(":", 1)[0] for reason in reasons)
            record = {
                "doc_id": doc_id,
                "source_file": file_name,
                "page": page_no,
                "table_id": str(old.get("table_id") or ""),
                "element_id": str(old.get("element_id") or ""),
                "bbox": bbox,
                "caption": caption,
                "section_title": str(old.get("section_title") or ""),
                "parser_version": PARSER_VERSION,
                "extraction_method": f"pymupdf_find_tables:{grid.strategy}",
                "header_depth": grid.header_depth,
                "quality_score": score,
                "quality_status": status,
                "quality_reasons": reasons,
                "quality_metrics": metrics,
                "table_json": {"columns": grid.columns, "rows": grid.rows},
                "column_names": grid.columns,
                "row_count": len(grid.rows),
                "markdown_table": rows_to_markdown(grid.columns, grid.rows),
                "raw_grid": grid.raw_grid,
            }
            out_tables.append(record)
            out_chunks.extend(build_v2_chunks(record, source=source, file_name=file_name))
    finally:
        doc.close()

    write_jsonl(tables_out / doc_id / "tables.jsonl", out_tables)
    write_jsonl(chunks_out / doc_id / "table_chunks.jsonl", out_chunks)
    summary = {
        "doc_id": doc_id,
        "file_name": file_name,
        "source_tables": len(source_tables),
        "output_tables": len(out_tables),
        "indexed_tables": sum(1 for t in out_tables if t.get("quality_status") == "pass"),
        "output_chunks": len(out_chunks),
        "quality_status": dict(counts),
        "strategies": dict(strategies),
        "quality_reasons": dict(reason_counts),
    }
    print(
        f"{doc_id}: pass={counts['pass']} review={counts['review']} reject={counts['reject']} "
        f"chunks={len(out_chunks)} strategies={dict(strategies)}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-list", type=Path, default=DEFAULT_DOC_LIST)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--v1-tables-root", type=Path, default=DEFAULT_V1_TABLES)
    parser.add_argument("--layout-root", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--tables-out", type=Path, default=DEFAULT_TABLES_OUT)
    parser.add_argument("--chunks-out", type=Path, default=DEFAULT_CHUNKS_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--doc-id", action="append", dest="doc_ids")
    args = parser.parse_args()

    rows = load_rows(args.doc_list)
    if args.doc_ids:
        wanted = set(args.doc_ids)
        rows = [row for row in rows if str(row.get("doc_id")) in wanted]
    if not rows:
        raise SystemExit("No documents selected")

    summaries = [
        process_doc(
            row=row,
            manifest=args.manifest,
            v1_tables_root=args.v1_tables_root,
            layout_root=args.layout_root,
            tables_out=args.tables_out,
            chunks_out=args.chunks_out,
        )
        for row in rows
    ]
    totals: Counter[str] = Counter()
    for summary in summaries:
        totals.update(summary["quality_status"])
    report = {
        "corpus": "kr_tables_v2",
        "parser_version": PARSER_VERSION,
        "documents": len(summaries),
        "totals": dict(totals),
        "indexed_tables": sum(s["indexed_tables"] for s in summaries),
        "output_chunks": sum(s["output_chunks"] for s in summaries),
        "per_doc": summaries,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["totals"], ensure_ascii=False), flush=True)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
