#!/usr/bin/env python3
"""Render contextual samples for the most frequent unknown HancomEQN glyphs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hancomeqn_restore import PUA_RE, iter_page_glyphs, load_mapping
from pdf_io import resolve_pdf_path

DOC_ID = "kr_kr_rules_7_2025_2f2d6373"


def load_tables() -> dict[str, dict]:
    path = ROOT / "data/processed/tables_v2" / DOC_ID / "tables.jsonl"
    return {
        row["table_id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=ROOT / "data/config/hancomeqn_maps/7_2025_bdc15136d686.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/manifests/kr7_expanded_table_pilot.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/hancomeqn_inventory" / DOC_ID / "expanded_unknown_atlas.png",
    )
    args = parser.parse_args()

    mapping = load_mapping(args.mapping)
    known = set(mapping.get("glyphs", {}))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    table_lookup = load_tables()
    pdf = resolve_pdf_path(DOC_ID, ROOT / "data/manifests/pdf_manifest.csv", None, ROOT)
    if pdf is None:
        raise FileNotFoundError(DOC_ID)

    counts: Counter[str] = Counter()
    samples: dict[str, list[dict]] = defaultdict(list)
    doc = fitz.open(pdf)
    try:
        for record in manifest["tables"]:
            table = table_lookup[record["table_id"]]
            page_number = int(table["page"])
            page = doc[page_number - 1]
            layout = json.loads(
                (
                    ROOT
                    / "data/processed/layout_json_merged"
                    / DOC_ID
                    / f"page_{page_number:04d}.json"
                ).read_text(encoding="utf-8")
            )
            sx = page.rect.width / float(layout["width"])
            sy = page.rect.height / float(layout["height"])
            x0, y0, x1, y1 = [float(value) for value in table["bbox"]]
            clip = fitz.Rect(x0 * sx, y0 * sy, x1 * sx, y1 * sy)
            glyphs = list(iter_page_glyphs(page, clip))
            for index, glyph in enumerate(glyphs):
                if "HancomEQN" not in glyph.font or not PUA_RE.fullmatch(glyph.char):
                    continue
                code = f"U+{ord(glyph.char):04X}"
                if code in known:
                    continue
                counts[code] += 1
                if len(samples[code]) >= args.samples:
                    continue
                left = max(clip.x0, glyph.bbox[0] - 55)
                right = min(clip.x1, glyph.bbox[2] + 55)
                top = max(clip.y0, glyph.bbox[1] - 18)
                bottom = min(clip.y1, glyph.bbox[3] + 18)
                samples[code].append(
                    {
                        "page": page_number,
                        "rect": [left, top, right, bottom],
                        "glyph_bbox": list(glyph.bbox),
                        "table_id": record["table_id"],
                    }
                )

        selected = [code for code, _ in counts.most_common(args.top)]
        cell_w, cell_h, label_h = 430, 125, 30
        width = cell_w * args.samples
        height = max(1, len(selected)) * (cell_h + label_h)
        atlas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(atlas)
        font = ImageFont.load_default()
        metadata = []
        for row_index, code in enumerate(selected):
            y0 = row_index * (cell_h + label_h)
            label = f"{code}  occurrences={counts[code]}"
            draw.text((8, y0 + 8), label, fill="black", font=font)
            row_meta = {"code": code, "occurrences": counts[code], "samples": []}
            for sample_index, sample in enumerate(samples[code]):
                page = doc[sample["page"] - 1]
                rect = fitz.Rect(sample["rect"])
                pix = page.get_pixmap(matrix=fitz.Matrix(4, 4), clip=rect, alpha=False)
                crop = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                scale = min((cell_w - 12) / crop.width, (cell_h - 12) / crop.height, 1.0)
                if scale < 1.0:
                    crop = crop.resize(
                        (max(1, int(crop.width * scale)), max(1, int(crop.height * scale))),
                        Image.Resampling.LANCZOS,
                    )
                x = sample_index * cell_w + 6
                y = y0 + label_h + 6
                atlas.paste(crop, (x, y))
                gx0, gy0, gx1, gy1 = sample["glyph_bbox"]
                factor_x = crop.width / max(rect.width * 4, 1)
                factor_y = crop.height / max(rect.height * 4, 1)
                mark = [
                    x + (gx0 - rect.x0) * 4 * factor_x,
                    y + (gy0 - rect.y0) * 4 * factor_y,
                    x + (gx1 - rect.x0) * 4 * factor_x,
                    y + (gy1 - rect.y0) * 4 * factor_y,
                ]
                draw.rectangle(mark, outline="#e00000", width=3)
                draw.text((x + 2, y + cell_h - 20), f"p.{sample['page']}", fill="#003399", font=font)
                row_meta["samples"].append(sample)
            draw.line((0, y0 + cell_h + label_h - 1, width, y0 + cell_h + label_h - 1), fill="#bbbbbb")
            metadata.append(row_meta)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atlas.save(args.output)
        metadata_path = args.output.with_suffix(".json")
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(args.output.resolve()), "metadata": str(metadata_path.resolve()), "codes": len(selected)}, ensure_ascii=False, indent=2))
    finally:
        doc.close()


if __name__ == "__main__":
    main()
