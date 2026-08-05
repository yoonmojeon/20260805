from __future__ import annotations

from pathlib import Path

import fitz
import pandas as pd


class PdfTextReader:
    def __init__(self, pdf_path: Path | None) -> None:
        self.pdf_path = pdf_path
        self._doc: fitz.Document | None = None
        if pdf_path is not None and pdf_path.exists():
            self._doc = fitz.open(pdf_path)

    @property
    def available(self) -> bool:
        return self._doc is not None

    def close(self) -> None:
        if self._doc is not None:
            self._doc.close()
            self._doc = None

    def extract_bbox_text(
        self,
        page_number: int,
        bbox: list[float],
        layout_page_width: int,
        layout_page_height: int,
    ) -> str:
        if self._doc is None:
            return ""
        page_index = page_number - 1
        if page_index < 0 or page_index >= len(self._doc):
            return ""

        page = self._doc[page_index]
        sx = page.rect.width / max(layout_page_width, 1)
        sy = page.rect.height / max(layout_page_height, 1)
        x1, y1, x2, y2 = bbox
        rect = fitz.Rect(x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        return page.get_text("text", clip=rect).strip()


def resolve_pdf_path(
    doc_id: str,
    manifest_path: Path,
    pdf_path_arg: Path | None,
    project_root: Path,
) -> Path | None:
    if pdf_path_arg is not None:
        if pdf_path_arg.exists():
            return pdf_path_arg.resolve()
        raise FileNotFoundError(f"PDF not found: {pdf_path_arg}")

    if not manifest_path.exists():
        return None

    df = pd.read_csv(manifest_path)
    matched = df[df["doc_id"] == doc_id]
    if matched.empty:
        return None
    if len(matched) > 1:
        raise ValueError(
            f"Manifest has {len(matched)} rows for doc_id={doc_id!r}. "
            "Rebuild manifest with scripts/00_build_manifest.py"
        )

    row = matched.iloc[0]
    candidates = [Path(str(row["file_path"]))]
    file_name = str(row.get("file_name", ""))
    if file_name:
        for found in (project_root / "data" / "raw_pdfs").rglob(file_name):
            candidates.append(found)

    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.as_posix()
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()
    return None
