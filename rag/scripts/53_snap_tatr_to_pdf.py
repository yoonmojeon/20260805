"""Snap TATR geometry to native PDF vector lines and build a grounded cell grid."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import fitz

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pdf_io import resolve_pdf_path

PUA_RE = re.compile(r"[\ue000-\uf8ff]+")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


TABLE_ID_RE = re.compile(r"^(?P<doc_id>.+)_p\d{4}_t\d+$")


def doc_id_from_table_id(table_id: str) -> str:
    """Parse doc_id from table_id without breaking on '_part' / other '_p' prefixes."""
    match = TABLE_ID_RE.match(table_id)
    if not match:
        raise ValueError(f"Unrecognized table_id: {table_id}")
    return match.group("doc_id")


def find_table(tables_root: Path, table_id: str) -> dict[str, Any]:
    doc_id = doc_id_from_table_id(table_id)
    return next(item for item in load_jsonl(tables_root / doc_id / "tables.jsonl") if item["table_id"] == table_id)


def cluster_positions(values: Iterable[float], tolerance: float = 2.0) -> list[float]:
    clusters: list[list[float]] = []
    for value in sorted(float(v) for v in values):
        if not clusters or value - sum(clusters[-1]) / len(clusters[-1]) > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [round(sum(cluster) / len(cluster), 3) for cluster in clusters]


def object_boundaries(objects: list[dict[str, Any]], label: str, axis: str) -> list[float]:
    selected = [item for item in objects if item["label"] == label]
    if axis == "x":
        selected.sort(key=lambda item: (item["bbox_crop"][0] + item["bbox_crop"][2]) / 2)
        lo, hi = 0, 2
    else:
        selected.sort(key=lambda item: (item["bbox_crop"][1] + item["bbox_crop"][3]) / 2)
        lo, hi = 1, 3
    if not selected:
        return []
    boundaries = [float(selected[0]["bbox_crop"][lo])]
    for left, right in zip(selected, selected[1:]):
        boundaries.append((float(left["bbox_crop"][hi]) + float(right["bbox_crop"][lo])) / 2.0)
    boundaries.append(float(selected[-1]["bbox_crop"][hi]))
    return boundaries


def snap_boundaries(proposals: list[float], candidates: list[float], tolerance: float) -> list[float]:
    snapped: list[float] = []
    for proposal in proposals:
        nearest = min(candidates, key=lambda value: abs(value - proposal), default=proposal)
        snapped.append(nearest if abs(nearest - proposal) <= tolerance else round(proposal, 3))
    return cluster_positions(snapped, tolerance=1.0)


def layout_size(layout_root: Path, doc_id: str, page_number: int) -> tuple[float, float]:
    value = json.loads((layout_root / doc_id / f"page_{page_number:04d}.json").read_text(encoding="utf-8"))
    return float(value["width"]), float(value["height"])


def render_clip(page: fitz.Page, bbox: list[float], size: tuple[float, float]) -> fitz.Rect:
    sx, sy = page.rect.width / size[0], page.rect.height / size[1]
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return fitz.Rect(max(0, x0 * sx - 3), max(0, y0 * sy - 3), min(page.rect.width, x1 * sx + 3), min(page.rect.height, y1 * sy + 3))


def pdf_point_to_crop(point: fitz.Point, clip: fitz.Rect, crop_size: tuple[int, int]) -> tuple[float, float]:
    return (
        (point.x - clip.x0) * crop_size[0] / clip.width,
        (point.y - clip.y0) * crop_size[1] / clip.height,
    )


def vector_boundaries(page: fitz.Page, clip: fitz.Rect, crop_size: tuple[int, int]) -> tuple[list[float], list[float]]:
    horizontal: list[float] = []
    vertical: list[float] = []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] != "l":
                continue
            start, end = item[1], item[2]
            if abs(start.y - end.y) < 0.2:
                overlap = max(0.0, min(max(start.x, end.x), clip.x1) - max(min(start.x, end.x), clip.x0))
                if overlap >= clip.width * 0.25 and clip.y0 - 1 <= start.y <= clip.y1 + 1:
                    horizontal.append(pdf_point_to_crop(start, clip, crop_size)[1])
            elif abs(start.x - end.x) < 0.2:
                overlap = max(0.0, min(max(start.y, end.y), clip.y1) - max(min(start.y, end.y), clip.y0))
                if overlap >= clip.height * 0.25 and clip.x0 - 1 <= start.x <= clip.x1 + 1:
                    vertical.append(pdf_point_to_crop(start, clip, crop_size)[0])
    return cluster_positions(horizontal), cluster_positions(vertical)


def interval_coverage(box: list[float], boundaries: list[float], axis: str, minimum: float = 0.45) -> list[int]:
    lo_index, hi_index = (0, 2) if axis == "x" else (1, 3)
    selected: list[int] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        overlap = max(0.0, min(float(box[hi_index]), end) - max(float(box[lo_index]), start))
        if overlap / max(end - start, 1e-9) >= minimum:
            selected.append(index)
    return selected


def build_spans(detections: list[dict[str, Any]], rows: list[float], columns: list[float]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for item in detections:
        if item["label"] != "table spanning cell":
            continue
        row_ids = interval_coverage(item["bbox_crop"], rows, "y")
        col_ids = interval_coverage(item["bbox_crop"], columns, "x")
        if not row_ids or not col_ids:
            continue
        spans.append({
            "row": min(row_ids), "column": min(col_ids),
            "rowspan": max(row_ids) - min(row_ids) + 1,
            "colspan": max(col_ids) - min(col_ids) + 1,
            "score": item["score"],
        })
    return sorted(spans, key=lambda value: (value["row"], value["column"]))


def extract_words(page: fitz.Page, clip: fitz.Rect, crop_size: tuple[int, int]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for index, word in enumerate(page.get_text("words", clip=clip, sort=True), 1):
        p0 = pdf_point_to_crop(fitz.Point(word[0], word[1]), clip, crop_size)
        p1 = pdf_point_to_crop(fitz.Point(word[2], word[3]), clip, crop_size)
        words.append({
            "id": f"W{index:03d}", "text": str(word[4]),
            "bbox": [round(p0[0], 2), round(p0[1], 2), round(p1[0], 2), round(p1[1], 2)],
            "order": [int(word[5]), int(word[6]), int(word[7])],
        })
    return words


def point_in_box(point: tuple[float, float], box: list[float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def build_cells(rows: list[float], columns: list[float], spans: list[dict[str, Any]], words: list[dict[str, Any]], header_rows: set[int]) -> list[dict[str, Any]]:
    span_by_anchor = {(item["row"], item["column"]): item for item in spans}
    covered: set[tuple[int, int]] = set()
    for span in spans:
        for row in range(span["row"], span["row"] + span["rowspan"]):
            for col in range(span["column"], span["column"] + span["colspan"]):
                if (row, col) != (span["row"], span["column"]):
                    covered.add((row, col))
    cells: list[dict[str, Any]] = []
    for row in range(len(rows) - 1):
        for col in range(len(columns) - 1):
            if (row, col) in covered:
                continue
            span = span_by_anchor.get((row, col), {"rowspan": 1, "colspan": 1})
            row_end, col_end = row + span["rowspan"], col + span["colspan"]
            box = [columns[col], rows[row], columns[col_end], rows[row_end]]
            assigned = []
            for word in words:
                center = ((word["bbox"][0] + word["bbox"][2]) / 2, (word["bbox"][1] + word["bbox"][3]) / 2)
                if point_in_box(center, box):
                    assigned.append(word)
            assigned.sort(key=lambda value: value["order"])
            raw = " ".join(word["text"] for word in assigned).strip()
            cells.append({
                "cell_id": f"R{row:02d}C{col:02d}", "row": row, "column": col,
                "rowspan": span["rowspan"], "colspan": span["colspan"],
                "bbox_crop": [round(value, 3) for value in box],
                "type": "header" if row in header_rows else "data",
                "text_raw": raw,
                "text_safe": PUA_RE.sub("<PDF_MATH_GLYPHS>", raw),
                "word_ids": [word["id"] for word in assigned],
            })
    return cells


def add_header_paths(cells: list[dict[str, Any]], header_rows: set[int], column_count: int) -> None:
    for col in range(column_count):
        path = [
            cell["text_safe"] for cell in cells
            if cell["type"] == "header" and cell["column"] <= col < cell["column"] + cell["colspan"] and cell["text_safe"]
        ]
        for cell in cells:
            if cell["type"] == "data" and cell["column"] == col:
                cell["column_header_path"] = path
    for cell in cells:
        if cell["type"] != "data" or cell["column"] == 0:
            continue
        row_header = next((candidate["text_safe"] for candidate in cells if candidate["row"] == cell["row"] and candidate["column"] == 0), "")
        cell["row_header_path"] = [row_header] if row_header else []


def header_row_indices(detections: list[dict[str, Any]], rows: list[float]) -> set[int]:
    headers = [item for item in detections if item["label"] == "table column header"]
    result: set[int] = set()
    for header in headers:
        result.update(interval_coverage(header["bbox_crop"], rows, "y", minimum=0.4))
    return result


def draw_overlay(crop: Path, output: Path, rows: list[float], columns: list[float], cells: list[dict[str, Any]]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(crop).convert("RGB")
    # Empty grids (failed snap) still get a debug image; never crash the pipeline.
    if len(rows) >= 2 and len(columns) >= 2:
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for x in columns:
            draw.line((x, rows[0], x, rows[-1]), fill="#0066ff", width=2)
        for y in rows:
            draw.line((columns[0], y, columns[-1], y), fill="#ff2020", width=2)
        for cell in cells:
            box = cell["bbox_crop"]
            draw.rectangle(box, outline="#00a060" if cell["type"] == "header" else "#ff8c00", width=2)
            draw.text((box[0] + 3, box[1] + 3), cell["cell_id"], fill="#111111", font=font, stroke_width=2, stroke_fill="white")
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-id", required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/pdf_manifest.csv")
    parser.add_argument("--tables-root", type=Path, default=ROOT / "data/processed/tables_v2")
    parser.add_argument("--layout-root", type=Path, default=ROOT / "data/processed/layout_json_merged")
    parser.add_argument("--pilot-root", type=Path, default=ROOT / "data/processed/vlm_table_pilot")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Optional short table work directory (contains crop.png). Overrides pilot-root/doc_id/table_id.",
    )
    args = parser.parse_args()

    table = find_table(args.tables_root, args.table_id)
    doc_id, page_number = table["doc_id"], int(table["page"])
    table_dir = args.work_dir if args.work_dir is not None else (args.pilot_root / doc_id / args.table_id)
    crop = table_dir / "crop.png"
    tatr_path = table_dir / "tatr_v1_1_all" / "structure.json"
    tatr = json.loads(tatr_path.read_text(encoding="utf-8"))
    pdf = resolve_pdf_path(doc_id, args.manifest, None, ROOT)
    if pdf is None:
        raise FileNotFoundError(doc_id)

    doc = fitz.open(pdf)
    try:
        page = doc[page_number - 1]
        clip = render_clip(page, table["bbox"], layout_size(args.layout_root, doc_id, page_number))
        crop_size = tuple(int(value) for value in tatr["crop_size"])
        horizontal, vertical = vector_boundaries(page, clip, crop_size)
        row_proposals = object_boundaries(tatr["detections"], "table row", "y")
        col_proposals = object_boundaries(tatr["detections"], "table column", "x")
        rows = snap_boundaries(row_proposals, horizontal, tolerance=max(25.0, crop_size[1] * 0.08))
        columns = snap_boundaries(col_proposals, vertical, tolerance=max(25.0, crop_size[0] * 0.08))
        spans = build_spans(tatr["detections"], rows, columns)
        words = extract_words(page, clip, crop_size)
        header_rows = header_row_indices(tatr["detections"], rows)
        cells = build_cells(rows, columns, spans, words, header_rows)
        add_header_paths(cells, header_rows, len(columns) - 1)
    finally:
        doc.close()

    grid = table.get("raw_grid") or []
    result = {
        "table_id": args.table_id, "page": page_number,
        "method": "tatr-v1.1-all+pdf-vector-snap+pdf-text-bbox",
        "row_boundaries_crop": rows, "column_boundaries_crop": columns,
        "row_count": len(rows) - 1, "column_count": len(columns) - 1,
        "header_rows": sorted(header_rows), "spanning_cells": spans,
        "vector_candidates": {"horizontal": horizontal, "vertical": vertical},
        "coordinate_parser_comparison": {
            "raw_grid_rows": len(grid), "raw_grid_columns": max((len(row) for row in grid), default=0),
            "row_count_match": len(grid) == len(rows) - 1,
            "column_count_match": max((len(row) for row in grid), default=0) == len(columns) - 1,
        },
        "cells": cells,
    }
    output_dir = table_dir / "tatr_v1_1_all"
    (output_dir / "snapped_structure.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        draw_overlay(crop, output_dir / "snapped_overlay.png", rows, columns, cells)
    except Exception as exc:  # noqa: BLE001 - overlay is debug-only; structure already saved
        print(f"overlay skipped: {exc}", flush=True)
    print(json.dumps({key: result[key] for key in ("row_count", "column_count", "header_rows", "spanning_cells", "coordinate_parser_comparison")}, ensure_ascii=False, indent=2))
    print(f"saved: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
