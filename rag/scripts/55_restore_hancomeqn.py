"""Restore HancomEQN formulas assigned to an existing snapped TATR cell grid."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hancomeqn_restore import is_hancomeqn, iter_page_glyphs, load_mapping, point_in_bbox, restore_formula, restore_inline
from pdf_io import resolve_pdf_path
snap = __import__("53_snap_tatr_to_pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-id", required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/pdf_manifest.csv")
    parser.add_argument("--pilot-root", type=Path, default=ROOT / "data/processed/vlm_table_pilot")
    parser.add_argument("--tables-root", type=Path, default=ROOT / "data/processed/tables_v2")
    parser.add_argument("--layout-root", type=Path, default=ROOT / "data/processed/layout_json_merged")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Optional short table work directory. Overrides pilot-root/doc_id/table_id.",
    )
    args = parser.parse_args()
    doc_id = snap.doc_id_from_table_id(args.table_id)
    table_dir = args.work_dir if args.work_dir is not None else (args.pilot_root / doc_id / args.table_id)
    structure_path = table_dir / "tatr_v1_1_all/snapped_structure.json"
    structure = json.loads(structure_path.read_text(encoding="utf-8"))
    table = snap.find_table(args.tables_root, args.table_id)
    pdf = resolve_pdf_path(doc_id, args.manifest, None, ROOT)
    if pdf is None: raise FileNotFoundError(doc_id)
    mapping = load_mapping(args.mapping)
    doc = fitz.open(pdf)
    try:
        page = doc[int(table["page"]) - 1]
        clip = snap.render_clip(page, table["bbox"], snap.layout_size(args.layout_root, doc_id, int(table["page"])))
        crop_size = tuple(int(v) for v in json.loads((table_dir / "tatr_v1_1_all/structure.json").read_text(encoding="utf-8"))["crop_size"])
        candidates = []
        for glyph in iter_page_glyphs(page, clip):
            p0 = snap.pdf_point_to_crop(fitz.Point(glyph.bbox[0], glyph.bbox[1]), clip, crop_size)
            p1 = snap.pdf_point_to_crop(fitz.Point(glyph.bbox[2], glyph.bbox[3]), clip, crop_size)
            candidates.append((glyph, [p0[0], p0[1], p1[0], p1[1]]))
        results = []
        for cell in structure["cells"]:
            all_glyphs = [g for g, crop_bbox in candidates if point_in_bbox(((crop_bbox[0]+crop_bbox[2])/2, (crop_bbox[1]+crop_bbox[3])/2), cell["bbox_crop"])]
            glyphs = [g for g in all_glyphs if is_hancomeqn(g.font)]
            if not glyphs: continue
            formula = restore_formula(glyphs, mapping)
            normal_count = sum(1 for g in all_glyphs if not is_hancomeqn(g.font) and g.char.strip())
            eqn_count = sum(1 for g in glyphs if g.char.strip())
            formula_dominant = eqn_count >= max(3, normal_count * 2)
            inline_text, inline_unknown = restore_inline(cell.get("text_raw", ""), mapping)
            results.append({"cell_id": cell["cell_id"], "bbox_crop": cell["bbox_crop"], "raw_pua": "".join(g.char for g in glyphs),
                            "formula_dominant": formula_dominant, "normal_character_count": normal_count,
                            "equation_character_count": eqn_count, "inline_text_restored": inline_text,
                            "inline_unknown_glyphs": inline_unknown, "formula": formula})
        output = {"table_id": args.table_id, "method": "pdf-HancomEQN-map+font-origin-layout",
                  "mapping": str(args.mapping.resolve()), "cells": results}
        out_path = table_dir / "tatr_v1_1_all/restored_formulas.json"
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(out_path.resolve()), "formula_cells": len(results),
                          "needs_review": sum(x["formula"]["needs_review"] for x in results)}, ensure_ascii=False, indent=2))
    finally: doc.close()


if __name__ == "__main__": main()
