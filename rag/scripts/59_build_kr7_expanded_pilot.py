"""Build the incrementally expanded KR Part 7 table manifest and PUA audit."""
from __future__ import annotations

import csv
import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import fitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hancomeqn_restore import PUA_RE, load_mapping
from pdf_io import resolve_pdf_path

DOC_ID = "kr_kr_rules_7_2025_2f2d6373"
MAPPING = ROOT / "data/config/hancomeqn_maps/7_2025_bdc15136d686.json"
V1_TABLE_IDS = [
    # Early rule tables: formula, multi-header, wide and existing reject cases.
    f"{DOC_ID}_p0009_t003", f"{DOC_ID}_p0011_t004", f"{DOC_ID}_p0013_t007",
    f"{DOC_ID}_p0014_t009", f"{DOC_ID}_p0019_t012", f"{DOC_ID}_p0031_t017",
    f"{DOC_ID}_p0048_t020", f"{DOC_ID}_p0069_t032", f"{DOC_ID}_p0072_t036",
    f"{DOC_ID}_p0080_t043", f"{DOC_ID}_p0111_t046", f"{DOC_ID}_p0124_t050",
    f"{DOC_ID}_p0144_t052", f"{DOC_ID}_p0152_t054", f"{DOC_ID}_p0153_t056",
    f"{DOC_ID}_p0178_t066", f"{DOC_ID}_p0194_t071", f"{DOC_ID}_p0200_t073",
    f"{DOC_ID}_p0207_t075", f"{DOC_ID}_p0211_t077", f"{DOC_ID}_p0218_t084",
    f"{DOC_ID}_p0221_t088", f"{DOC_ID}_p0255_t111", f"{DOC_ID}_p0333_t122",
    f"{DOC_ID}_p0334_t123", f"{DOC_ID}_p0340_t129", f"{DOC_ID}_p0366_t136",
    f"{DOC_ID}_p0390_t154",
]

# Second validation batch: deliberately includes old-parser rejects, wide/tall
# coefficient tables, multi-header tables, and long loading-condition tables.
V2_ADDITIONAL_TABLE_IDS = [
    f"{DOC_ID}_p0047_t018", f"{DOC_ID}_p0048_t019", f"{DOC_ID}_p0062_t031",
    f"{DOC_ID}_p0071_t035", f"{DOC_ID}_p0073_t038", f"{DOC_ID}_p0077_t042",
    f"{DOC_ID}_p0150_t053", f"{DOC_ID}_p0162_t060", f"{DOC_ID}_p0196_t072",
    f"{DOC_ID}_p0208_t076", f"{DOC_ID}_p0216_t082", f"{DOC_ID}_p0218_t085",
    f"{DOC_ID}_p0218_t086", f"{DOC_ID}_p0219_t087", f"{DOC_ID}_p0222_t089",
    f"{DOC_ID}_p0223_t090", f"{DOC_ID}_p0227_t091", f"{DOC_ID}_p0227_t093",
    f"{DOC_ID}_p0229_t094", f"{DOC_ID}_p0231_t098", f"{DOC_ID}_p0231_t100",
    f"{DOC_ID}_p0255_t110", f"{DOC_ID}_p0256_t113", f"{DOC_ID}_p0264_t116",
    f"{DOC_ID}_p0269_t117", f"{DOC_ID}_p0324_t120", f"{DOC_ID}_p0367_t137",
    f"{DOC_ID}_p0370_t140",
]

TABLE_IDS = V1_TABLE_IDS + V2_ADDITIONAL_TABLE_IDS


def load_tables() -> dict[str, dict[str, Any]]:
    path = ROOT / "data/processed/tables_v2" / DOC_ID / "tables.jsonl"
    return {row["table_id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)}


def page_clip(page: fitz.Page, table: dict[str, Any]) -> tuple[fitz.Rect, tuple[float, float]]:
    layout_path = ROOT / "data/processed/layout_json_merged" / DOC_ID / f"page_{int(table['page']):04d}.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    sx, sy = page.rect.width / float(layout["width"]), page.rect.height / float(layout["height"])
    x0, y0, x1, y1 = (float(v) for v in table["bbox"])
    return fitz.Rect(x0 * sx, y0 * sy, x1 * sx, y1 * sy), (float(layout["width"]), float(layout["height"]))


def pua_chars(page: fitz.Page, clip: fitz.Rect) -> list[str]:
    return [char["c"] for block in page.get_text("rawdict", clip=clip).get("blocks", [])
            for line in block.get("lines", []) for span in line.get("spans", [])
            for char in span.get("chars", []) if PUA_RE.fullmatch(char["c"])]


def category(table: dict[str, Any], rows: int, cols: int, pua_count: int) -> list[str]:
    tags = []
    if pua_count >= 100: tags.append("formula_heavy")
    elif pua_count: tags.append("formula")
    else: tags.append("no_pua")
    if cols >= 8: tags.append("wide")
    if rows >= 12: tags.append("tall")
    if int(table.get("header_depth") or 0) >= 2: tags.append("multi_header")
    if table.get("quality_status") != "pass": tags.append("legacy_reject")
    if "계속" in str(table.get("caption") or ""): tags.append("continuation")
    return tags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", choices=("curated", "all"), default="curated")
    parser.add_argument("--output", type=Path, default=ROOT / "data/manifests/kr7_expanded_table_pilot.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/processed/kr7_expanded_table_pilot" / DOC_ID)
    args = parser.parse_args()
    tables = load_tables()
    table_ids = TABLE_IDS if args.selection == "curated" else list(tables)
    missing = [table_id for table_id in table_ids if table_id not in tables]
    if missing: raise KeyError(f"Missing curated tables: {missing}")
    mapping = load_mapping(MAPPING); known = set(mapping.get("glyphs", {}))
    pdf = resolve_pdf_path(DOC_ID, ROOT / "data/manifests/pdf_manifest.csv", None, ROOT)
    if pdf is None: raise FileNotFoundError(DOC_ID)
    render = importlib.import_module("49_vlm_table_pilot")
    output_root = args.output_root
    records = []
    doc = fitz.open(pdf)
    try:
        for table_id in table_ids:
            table = tables[table_id]; page = doc[int(table["page"]) - 1]
            clip, layout_size = page_clip(page, table); chars = pua_chars(page, clip)
            codes = [f"U+{ord(char):04X}" for char in chars]
            mapped_count = sum(code in known for code in codes)
            grid = table.get("raw_grid") or []; rows = len(grid); cols = max((len(row) for row in grid), default=0)
            crop = output_root / table_id / "crop.png"
            render.render_crop(pdf, list(table["bbox"]), layout_size, int(table["page"]), crop)
            records.append({
                "table_id": table_id, "page": int(table["page"]), "caption": table.get("caption") or "",
                "row_count_legacy": rows, "column_count_legacy": cols,
                "header_depth_legacy": table.get("header_depth"), "quality_status_legacy": table.get("quality_status"),
                "categories": category(table, rows, cols, len(chars)), "crop_path": str(crop.resolve()),
                "pua_occurrences": len(chars), "pua_unique": len(set(codes)),
                "mapped_occurrences": mapped_count,
                "mapped_occurrence_coverage": round(mapped_count / len(codes), 4) if codes else 1.0,
                "unknown_pua": sorted(set(codes) - known),
            })
    finally: doc.close()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"doc_id": DOC_ID, "selection_method": ("all_detected_tables_v3" if args.selection == "all" else "incremental_complexity_stratified_v2"),
                "base_table_count": len(V1_TABLE_IDS),
                "added_table_count": len(V2_ADDITIONAL_TABLE_IDS),
                "mapping": str(MAPPING.resolve()), "table_count": len(records), "tables": records}
    json_path = args.output
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = json_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["table_id", "page", "caption", "row_count_legacy", "column_count_legacy", "header_depth_legacy",
                  "quality_status_legacy", "categories", "pua_occurrences", "pua_unique", "mapped_occurrences",
                  "mapped_occurrence_coverage", "unknown_pua", "crop_path"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for record in records:
            writer.writerow({**record, "categories": ",".join(record["categories"]),
                             "unknown_pua": ",".join(record["unknown_pua"])})
    total = sum(r["pua_occurrences"] for r in records); mapped = sum(r["mapped_occurrences"] for r in records)
    print(json.dumps({"tables": len(records), "json": str(json_path.resolve()), "csv": str(csv_path.resolve()),
                      "pua_occurrences": total, "mapped_occurrences": mapped,
                      "mapped_coverage": round(mapped / total, 4) if total else 1.0,
                      "tables_with_unknown": sum(bool(r["unknown_pua"]) for r in records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
