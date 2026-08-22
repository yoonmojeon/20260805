#!/usr/bin/env python3
"""Build lossless contact sheets for visual inspection of rendered DOCX pages."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def page_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def build(folder: Path, *, cols: int = 4, rows: int = 2) -> None:
    pages = sorted(folder.glob("page-*.png"), key=page_number)
    if not pages:
        raise SystemExit(f"No page PNGs found in {folder}")
    out_dir = folder / "contact_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    per_sheet = cols * rows
    for offset in range(0, len(pages), per_sheet):
        batch = pages[offset : offset + per_sheet]
        with Image.open(batch[0]) as sample:
            page_w, page_h = sample.size
        label_h = 26
        canvas = Image.new("RGB", (page_w * cols, (page_h + label_h) * rows), "#B9C2CC")
        draw = ImageDraw.Draw(canvas)
        for index, path in enumerate(batch):
            row, col = divmod(index, cols)
            x = col * page_w
            y = row * (page_h + label_h)
            with Image.open(path) as page:
                canvas.paste(page.convert("RGB"), (x, y + label_h))
            label = f"{folder.name} · page {page_number(path)}"
            draw.rectangle((x, y, x + page_w, y + label_h), fill="#253B53")
            draw.text((x + 8, y + 7), label, fill="white", font=font)
        first = page_number(batch[0])
        last = page_number(batch[-1])
        canvas.save(out_dir / f"pages-{first:03d}-{last:03d}.png", optimize=True)
    print(f"{folder}: {len(pages)} pages, {len(list(out_dir.glob('*.png')))} sheets")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        build(Path(arg))
