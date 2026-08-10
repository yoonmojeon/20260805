#!/usr/bin/env python3
"""Split KR Part 7 compound table crops into locally coherent ruled regions.

The splitter uses native PDF vector segments.  A region boundary is introduced
when the local vertical-rule topology changes or a ruled grid transitions to a
prose/footnote band.  This prevents a single TATR grid from being stretched over
nested subtables, variable definitions, and notes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import importlib

snap = importlib.import_module("53_snap_tatr_to_pdf")
from hancomeqn_restore import PUA_RE
from pdf_io import resolve_pdf_path

@dataclass
class Segment:
    orientation: str
    fixed: float
    start: float
    end: float

    @property
    def length(self) -> float:
        return self.end - self.start


def cluster(values: Iterable[float], tolerance: float = 4.0) -> list[float]:
    groups: list[list[float]] = []
    for value in sorted(values):
        if not groups or value - sum(groups[-1]) / len(groups[-1]) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [sum(group) / len(group) for group in groups]


def vector_segments(
    page: fitz.Page, clip: fitz.Rect, crop_size: tuple[int, int]
) -> tuple[list[Segment], list[Segment]]:
    horizontal: list[Segment] = []
    vertical: list[Segment] = []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] != "l":
                continue
            a, b = item[1], item[2]
            if abs(a.y - b.y) < 0.25:
                start, end = sorted((a.x, b.x))
                start, end = max(start, clip.x0), min(end, clip.x1)
                if end <= start or not clip.y0 - 1 <= a.y <= clip.y1 + 1:
                    continue
                p0 = snap.pdf_point_to_crop(fitz.Point(start, a.y), clip, crop_size)
                p1 = snap.pdf_point_to_crop(fitz.Point(end, a.y), clip, crop_size)
                horizontal.append(Segment("h", p0[1], p0[0], p1[0]))
            elif abs(a.x - b.x) < 0.25:
                start, end = sorted((a.y, b.y))
                start, end = max(start, clip.y0), min(end, clip.y1)
                if end <= start or not clip.x0 - 1 <= a.x <= clip.x1 + 1:
                    continue
                p0 = snap.pdf_point_to_crop(fitz.Point(a.x, start), clip, crop_size)
                p1 = snap.pdf_point_to_crop(fitz.Point(a.x, end), clip, crop_size)
                vertical.append(Segment("v", p0[0], p0[1], p1[1]))
    return horizontal, vertical


def local_verticals(vertical: list[Segment], y0: float, y1: float) -> list[float]:
    height = max(y1 - y0, 1.0)
    values = []
    for line in vertical:
        overlap = max(0.0, min(line.end, y1) - max(line.start, y0))
        if overlap / height >= 0.62:
            values.append(line.fixed)
    return cluster(values)


def signature_similarity(left: list[float], right: list[float], tolerance: float = 14.0) -> float:
    if not left or not right:
        return 0.0
    hits = sum(any(abs(value - candidate) <= tolerance for candidate in right) for value in left)
    reverse = sum(any(abs(value - candidate) <= tolerance for candidate in left) for value in right)
    return min(hits / len(left), reverse / len(right))


def words_in_band(words: list[dict], y0: float, y1: float) -> list[dict]:
    return [
        word
        for word in words
        if y0 <= (word["bbox"][1] + word["bbox"][3]) / 2 <= y1
    ]


def line_count(words: list[dict]) -> int:
    if not words:
        return 0
    return len(cluster(((word["bbox"][1] + word["bbox"][3]) / 2 for word in words), 8.0))


def band_kind(verticals: list[float], words: list[dict]) -> str:
    interior = max(0, len(verticals) - 2)
    lines = line_count(words)
    if interior:
        return "grid"
    if lines >= 3:
        return "prose_or_definitions"
    return "spanning_or_single_cell"


def should_merge(left: dict, right: dict) -> bool:
    if left["kind"] == right["kind"] == "prose_or_definitions":
        return True
    similarity = signature_similarity(left["verticals"], right["verticals"])
    if left["kind"] == right["kind"] == "grid":
        return similarity >= 0.72
    # A shallow spanning header normally belongs to the grid immediately below.
    if left["kind"] == "spanning_or_single_cell" and right["kind"] == "grid":
        return left["line_count"] <= 2 and similarity >= 0.45
    return False


def nested_panels(region: dict, horizontal: list[Segment], width: float) -> list[dict]:
    """Find side-by-side independent subtables inside one tall ruled region."""
    y0, y1 = region["y0"], region["y1"]
    height = y1 - y0
    if region["kind"] != "grid" or height < width * 0.28:
        return []
    verticals = [value for value in region["verticals"] if width * 0.08 < value < width * 0.92]
    if len(verticals) != 1:
        return []
    split = verticals[0]

    def local_grid(x0: float, x1: float, region_id: str) -> dict | None:
        panel_width = x1 - x0
        candidates = []
        for line in horizontal:
            if not y0 + 5 < line.fixed < y1 - 5:
                continue
            overlap = max(0.0, min(line.end, x1) - max(line.start, x0))
            if overlap >= panel_width * 0.46 and line.length < width * 0.72:
                candidates.append(line)
        groups: list[list[Segment]] = []
        for line in sorted(candidates, key=lambda value: (value.start, value.end, value.fixed)):
            target = next(
                (
                    group
                    for group in groups
                    if abs(line.start - sum(item.start for item in group) / len(group)) <= 32
                    and abs(line.end - sum(item.end for item in group) / len(group)) <= 32
                ),
                None,
            )
            if target is None:
                groups.append([line])
            else:
                target.append(line)
        if not groups:
            return None
        best = max(groups, key=lambda group: len(cluster(item.fixed for item in group)))
        y_values = cluster(item.fixed for item in best)
        if len(y_values) < 3:
            return None
        gx0 = max(x0, sum(item.start for item in best) / len(best) - 5)
        gx1 = min(x1, sum(item.end for item in best) / len(best) + 5)
        return {
            "region_id": region_id,
            "kind": "nested_grid",
            "bbox_crop": [round(gx0, 3), round(min(y_values) - 5, 3), round(gx1, 3), round(max(y_values) + 5, 3)],
            "panel_bbox_crop": [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)],
            "requires_local_structure": True,
            "local_horizontal_rule_count": len(y_values),
        }

    left = local_grid(0.0, split, f"{region['region_id']}_P01_GRID")
    right = local_grid(split, width, f"{region['region_id']}_P02_GRID")
    if left is None or right is None:
        return []
    return [left, right]


def draw_overlay(crop_path: Path, output: Path, regions: list[dict]) -> None:
    image = Image.open(crop_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = ["#e53935", "#1e88e5", "#43a047", "#8e24aa", "#fb8c00"]
    for index, region in enumerate(regions):
        box = region["bbox_crop"]
        color = colors[index % len(colors)]
        draw.rectangle(box, outline=color, width=5)
        draw.text(
            (box[0] + 6, box[1] + 6),
            f"{region['region_id']} {region['kind']}",
            fill=color,
            font=font,
            stroke_width=2,
            stroke_fill="white",
        )
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/manifests/kr7_expanded_table_pilot.json",
    )
    parser.add_argument(
        "--pilot-root",
        type=Path,
        default=ROOT / "data/processed/kr7_expanded_table_pilot",
    )
    parser.add_argument("--pdf-manifest", type=Path, default=ROOT / "data/manifests/pdf_manifest.csv")
    parser.add_argument("--tables-root", type=Path, default=ROOT / "data/processed/tables_v2")
    parser.add_argument("--layout-root", type=Path, default=ROOT / "data/processed/layout_json_merged")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    documents: dict[str, fitz.Document] = {}
    try:
        for item in manifest["tables"]:
            table_id = item["table_id"]
            doc_id = str(item.get("doc_id") or snap.doc_id_from_table_id(table_id))
            table = snap.find_table(args.tables_root, table_id)
            if doc_id not in documents:
                # Windows handle limit: keep only the current PDF open.
                for opened in documents.values():
                    opened.close()
                documents.clear()
                pdf = resolve_pdf_path(doc_id, args.pdf_manifest, None, ROOT)
                if pdf is None:
                    raise FileNotFoundError(doc_id)
                documents[doc_id] = fitz.open(pdf)
            doc = documents[doc_id]
            page_number = int(table["page"])
            page = doc[page_number - 1]
            if item.get("work_dir"):
                table_dir = Path(item["work_dir"])
            elif item.get("crop_path"):
                table_dir = Path(item["crop_path"]).parent
            else:
                table_dir = args.pilot_root / doc_id / table_id
            structure_path = table_dir / "tatr_v1_1_all/structure.json"
            structure = json.loads(structure_path.read_text(encoding="utf-8"))
            crop_size = tuple(int(value) for value in structure["crop_size"])
            clip = snap.render_clip(
                page,
                table["bbox"],
                snap.layout_size(args.layout_root, doc_id, page_number),
            )
            horizontal, vertical = vector_segments(page, clip, crop_size)
            width, height = crop_size
            separators = cluster(
                [0.0, float(height)]
                + [line.fixed for line in horizontal if line.length >= width * 0.58]
            )
            separators = [max(0.0, min(float(height), value)) for value in separators]
            separators = sorted(set(round(value, 3) for value in separators))
            # Remove tiny raster/line-width slivers.
            clean = [separators[0]]
            for value in separators[1:]:
                if value - clean[-1] >= max(12.0, height * 0.012):
                    clean.append(value)
                else:
                    clean[-1] = value
            if clean[-1] < height - 1:
                clean.append(float(height))

            words = snap.extract_words(page, clip, crop_size)
            bands = []
            for y0, y1 in zip(clean, clean[1:]):
                band_words = words_in_band(words, y0, y1)
                verticals = local_verticals(vertical, y0, y1)
                bands.append(
                    {
                        "y0": y0,
                        "y1": y1,
                        "verticals": verticals,
                        "word_ids": [word["id"] for word in band_words],
                        "line_count": line_count(band_words),
                        "kind": band_kind(verticals, band_words),
                    }
                )

            merged: list[dict] = []
            for band in bands:
                if merged and should_merge(merged[-1], band):
                    previous = merged[-1]
                    previous["y1"] = band["y1"]
                    previous["word_ids"].extend(band["word_ids"])
                    previous["line_count"] += band["line_count"]
                    if len(band["verticals"]) > len(previous["verticals"]):
                        previous["verticals"] = band["verticals"]
                    if band["kind"] == "grid":
                        previous["kind"] = "grid"
                else:
                    merged.append(dict(band))

            by_id = {word["id"]: word for word in words}
            regions = []
            nonempty = [region for region in merged if region["word_ids"]]
            for index, region in enumerate(nonempty, 1):
                region_words = [by_id[word_id] for word_id in region["word_ids"] if word_id in by_id]
                region_words.sort(key=lambda word: word["order"])
                raw = " ".join(word["text"] for word in region_words).strip()
                region_record = {
                        "region_id": f"REG{index:02d}",
                        "kind": region["kind"],
                        "bbox_crop": [0.0, round(region["y0"], 3), float(width), round(region["y1"], 3)],
                        "local_vertical_boundaries": [round(value, 3) for value in region["verticals"]],
                        "text_raw": raw,
                        "text_safe": PUA_RE.sub("<PDF_MATH_GLYPH>", raw),
                        "word_ids": region["word_ids"],
                        "requires_local_structure": region["kind"] == "grid",
                    }
                region["region_id"] = region_record["region_id"]
                region_record["children"] = nested_panels(region, horizontal, float(width))
                regions.append(region_record)

            result = {
                "table_id": table_id,
                "page": page_number,
                "method": "pdf-vector-local-topology-region-split-v1",
                "crop_size": list(crop_size),
                "region_count": len(regions),
                "compound": len(regions) > 1,
                "regions": regions,
            }
            output_dir = table_dir / "region_split_v1"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "regions.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            crop_path = table_dir / "crop.png"
            draw_overlay(crop_path, output_dir / "regions_overlay.png", regions)
            crop_image = Image.open(crop_path).convert("RGB")
            for region in regions:
                x0, y0, x1, y1 = region["bbox_crop"]
                crop_image.crop((int(x0), int(y0), int(x1), int(y1))).save(
                    output_dir / f"{region['region_id']}.png"
                )
                for child in region.get("children", []):
                    cx0, cy0, cx1, cy1 = child["bbox_crop"]
                    crop_image.crop((int(cx0), int(cy0), int(cx1), int(cy1))).save(
                        output_dir / f"{child['region_id']}.png"
                    )
            rows.append(
                {
                    "table_id": table_id,
                    "doc_id": doc_id,
                    "page": page_number,
                    "region_count": len(regions),
                    "compound": len(regions) > 1,
                    "region_kinds": [region["kind"] for region in regions],
                    "nested_panels": sum(len(region.get("children", [])) for region in regions),
                    "output": str((output_dir / "regions.json").resolve()),
                }
            )
            print(f"{table_id}: regions={len(regions)} {[r['kind'] for r in regions]}")
    finally:
        for document in documents.values():
            document.close()

    output = args.output or args.manifest.with_name(f"{args.manifest.stem}_region_results.json")
    output.write_text(
        json.dumps(
            {
                "summary": {
                    "tables": len(rows),
                    "compound_tables": sum(row["compound"] for row in rows),
                    "regions": sum(row["region_count"] for row in rows),
                },
                "tables": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output.resolve()), "tables": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
