from __future__ import annotations

import argparse
from pathlib import Path

import fitz
from PIL import Image
from tqdm import tqdm


def png_is_valid(path: Path, min_bytes: int = 100) -> bool:
    """False for missing, empty, or unreadable PNG (e.g. interrupted write)."""
    if not path.exists():
        return False
    if path.stat().st_size < min_bytes:
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def validate_page_range(start_page: int, end_page: int | None, total_pages: int) -> tuple[int, int]:
    if start_page < 1:
        raise ValueError("--start-page must be >= 1")
    if start_page > total_pages:
        raise ValueError(f"--start-page ({start_page}) exceeds total pages ({total_pages})")

    if end_page is None:
        end_page = total_pages
    if end_page < start_page:
        raise ValueError("--end-page must be >= --start-page")
    if end_page > total_pages:
        end_page = total_pages
    return start_page, end_page


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert one PDF to per-page PNG images using PyMuPDF.")
    parser.add_argument("--pdf", type=Path, required=True, help="Input PDF path")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output root directory")
    parser.add_argument("--doc-id", type=str, required=True, help="Document identifier")
    parser.add_argument("--dpi", type=int, default=200, help="Rendering DPI (default: 200)")
    parser.add_argument("--start-page", type=int, default=1, help="1-based start page (inclusive)")
    parser.add_argument("--end-page", type=int, default=None, help="1-based end page (inclusive)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing page images")
    args = parser.parse_args()

    if args.dpi <= 0:
        raise ValueError("--dpi must be > 0")
    if not args.pdf.exists():
        raise FileNotFoundError(f"PDF not found: {args.pdf}")
    if args.pdf.suffix.lower() != ".pdf":
        raise ValueError("--pdf must point to a .pdf file")

    doc_out_dir = args.out_dir / args.doc_id
    doc_out_dir.mkdir(parents=True, exist_ok=True)

    with fitz.open(args.pdf) as doc:
        total_pages = doc.page_count
        start_page, end_page = validate_page_range(args.start_page, args.end_page, total_pages)
        page_numbers = range(start_page, end_page + 1)
        scale = args.dpi / 72.0
        matrix = fitz.Matrix(scale, scale)

        for page_num in tqdm(page_numbers, total=(end_page - start_page + 1), desc="PDF -> PNG"):
            out_path = doc_out_dir / f"page_{page_num:04d}.png"
            if not args.overwrite and png_is_valid(out_path):
                continue

            page = doc.load_page(page_num - 1)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(out_path)

    print(f"Page images saved under: {doc_out_dir}")


if __name__ == "__main__":
    main()
