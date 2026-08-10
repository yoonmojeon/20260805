"""Inventory HancomEQN PUA glyphs and render a visual contact sheet."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hancomeqn_restore import PUA_RE, hancomeqn_fonts, iter_page_glyphs, load_mapping
from pdf_io import resolve_pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/hancomeqn_inventory")
    parser.add_argument("--max-pages", type=int)
    args = parser.parse_args()
    pdf = resolve_pdf_path(args.doc_id, ROOT / "data/manifests/pdf_manifest.csv", args.pdf, ROOT)
    if pdf is None: raise FileNotFoundError(args.doc_id)
    mapping = load_mapping(args.mapping) if args.mapping else {"glyphs": {}}
    counts: Counter[str] = Counter()
    samples: dict[str, dict] = {}
    fonts: dict[str, dict] = {}
    doc = fitz.open(pdf)
    try:
        page_limit = min(len(doc), args.max_pages or len(doc))
        for page_index in range(page_limit):
            page = doc[page_index]
            for font in hancomeqn_fonts(doc, page): fonts[font["fingerprint"]] = font
            for glyph in iter_page_glyphs(page):
                if "HancomEQN" not in glyph.font or not PUA_RE.fullmatch(glyph.char): continue
                code = f"U+{ord(glyph.char):04X}"
                counts[code] += 1
                samples.setdefault(code, {"page": page_index + 1, "bbox": list(glyph.bbox)})
        output = args.output_dir / args.doc_id
        output.mkdir(parents=True, exist_ok=True)
        rows = [{"code": code, "count": count, "mapped": code in mapping.get("glyphs", {}),
                 "mapping": mapping.get("glyphs", {}).get(code), "sample": samples[code]}
                for code, count in counts.most_common()]
        report = {"doc_id": args.doc_id, "pdf": str(pdf), "pages_scanned": page_limit,
                  "fonts": list(fonts.values()), "unique_pua": len(rows), "occurrences": sum(counts.values()), "glyphs": rows}
        (output / "inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        width, row_h = 900, 76
        image = Image.new("RGB", (width, max(row_h, row_h * len(rows))), "white")
        draw, font = ImageDraw.Draw(image), ImageFont.load_default()
        for i, item in enumerate(rows):
            y = i * row_h
            sample = item["sample"]; page = doc[sample["page"] - 1]
            rect = fitz.Rect(sample["bbox"]); rect.x0 -= 6; rect.y0 -= 6; rect.x1 += 6; rect.y1 += 6
            pix = page.get_pixmap(matrix=fitz.Matrix(4, 4), clip=rect, alpha=False)
            crop = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            crop.thumbnail((180, row_h - 8)); image.paste(crop, (8, y + 4))
            label = f'{item["code"]}  count={item["count"]}  mapped={item["mapping"]!r}  p.{sample["page"]}'
            draw.text((205, y + 26), label, fill="black", font=font)
            draw.line((0, y + row_h - 1, width, y + row_h - 1), fill="#dddddd")
        image.save(output / "glyph_contact_sheet.png")
        print(json.dumps({"output": str(output.resolve()), "unique_pua": len(rows), "occurrences": sum(counts.values()),
                          "mapped_unique": sum(r["mapped"] for r in rows)}, ensure_ascii=False, indent=2))
    finally:
        doc.close()


if __name__ == "__main__": main()
