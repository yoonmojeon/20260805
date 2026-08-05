from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from clause_parse import (
    article_number_from_text,
    clause_number_from_text,
    is_article_clause_number,
    split_text_by_subclauses,
)
from pdf_io import PdfTextReader, resolve_pdf_path


DEFAULT_CHUNK_TYPES = frozenset(
    {
        "text",
        "table",
        "picture",
        "figure",
        "caption",
        "title",
        "section-header",
        "list-item",
    }
)
SKIP_BY_DEFAULT = frozenset({"page-header", "page-footer"})
FIGURE_TYPES = frozenset({"picture", "figure"})
CAPTION_MAX_GAP_RATIO = 0.12

def parse_chunk_types(raw: str) -> frozenset[str]:
    return frozenset(part.strip().lower() for part in raw.split(",") if part.strip())


def normalize_whitespace(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def vertical_gap_below(upper: list[float], lower: list[float]) -> float:
    return max(0.0, lower[1] - upper[3])


def find_caption_for_figure(
    figure_bbox: list[float],
    captions: list[dict],
    page_height: int,
) -> dict | None:
    max_gap = page_height * CAPTION_MAX_GAP_RATIO
    best: dict | None = None
    best_gap = float("inf")

    for caption in captions:
        cap_bbox = [float(v) for v in caption["bbox"]]
        gap = vertical_gap_below(figure_bbox, cap_bbox)
        if gap > max_gap:
            continue
        x_overlap = max(0.0, min(figure_bbox[2], cap_bbox[2]) - max(figure_bbox[0], cap_bbox[0]))
        if x_overlap <= 0:
            continue
        if gap < best_gap:
            best_gap = gap
            best = caption
    return best


def load_crop_lookup(crops_manifest_path: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if not crops_manifest_path.exists():
        return lookup
    with crops_manifest_path.open(encoding="utf-8") as manifest_f:
        for line in manifest_f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            element_id = str(record.get("element_id", ""))
            crop_path = str(record.get("crop_path", ""))
            if element_id and crop_path:
                lookup[element_id] = crop_path
    return lookup


def build_chunk_record(
    *,
    doc_id: str,
    page_number: int,
    page_name: str,
    page_width: int,
    page_height: int,
    source_page_image: str,
    source_pdf: str,
    element: dict,
    text: str,
    crop_path: str,
    linked_caption_id: str | None = None,
) -> dict:
    element_id = str(element["element_id"])
    element_type = str(element.get("type", "unknown")).lower()
    bbox = [int(round(v)) for v in element.get("bbox", [0, 0, 0, 0])]

    chunk: dict = {
        "chunk_id": element_id,
        "doc_id": doc_id,
        "page_number": page_number,
        "page_name": page_name,
        "element_id": element_id,
        "element_type": element_type,
        "confidence": float(element.get("confidence", 0.0)),
        "bbox": bbox,
        "page_width": page_width,
        "page_height": page_height,
        "text": text,
        "text_char_count": len(text),
        "content_format": "plain_text",
        "crop_path": crop_path,
        "source_page_image": source_page_image,
        "source_pdf": source_pdf,
        "layout_source": "merged",
    }

    clause_number = element.get("clause_number")
    if clause_number is None and element_type == "text":
        clause_number = clause_number_from_text(text)
    if clause_number:
        chunk["clause_number"] = str(clause_number)

    article_number = element.get("article_number")
    if article_number is None and element_type == "text":
        article_number = article_number_from_text(text)
    if article_number:
        chunk["article_number"] = str(article_number)
    elif clause_number and is_article_clause_number(str(clause_number)):
        chunk["article_number"] = str(clause_number)

    merged_from = element.get("merged_from")
    if merged_from:
        chunk["merged_from"] = list(merged_from)

    if linked_caption_id:
        chunk["linked_caption_id"] = linked_caption_id

    return chunk


def expand_multi_subclause_chunks(chunks: list[dict]) -> list[dict]:
    """Split text chunks that bundle multiple '3-1.' / '2-3.' definitions."""
    expanded: list[dict] = []
    for chunk in chunks:
        if str(chunk.get("element_type", "")).lower() != "text":
            expanded.append(chunk)
            continue

        text = str(chunk.get("text", "")).strip()
        segments = split_text_by_subclauses(text)
        if len(segments) <= 1:
            expanded.append(chunk)
            continue

        base_id = str(chunk["chunk_id"])
        parent_article = chunk.get("article_number")
        for seg_idx, (seg_text, seg_clause) in enumerate(segments):
            seg_chunk = dict(chunk)
            seg_chunk["text"] = seg_text
            seg_chunk["text_char_count"] = len(seg_text)
            if seg_idx:
                seg_chunk["chunk_id"] = f"{base_id}_s{seg_idx:02d}"
                seg_chunk["element_id"] = seg_chunk["chunk_id"]
                seg_chunk["split_from"] = base_id
            if seg_clause:
                seg_chunk["clause_number"] = seg_clause
                if is_article_clause_number(seg_clause):
                    seg_chunk["article_number"] = seg_clause
            elif parent_article:
                seg_chunk["article_number"] = str(parent_article)
            expanded.append(seg_chunk)

    return expanded


def extract_text_for_element(
    element: dict,
    element_type: str,
    pdf_reader: PdfTextReader,
    page_number: int,
    page_width: int,
    page_height: int,
    caption_elements: list[dict],
    caption_text_by_id: dict[str, str],
) -> tuple[str, str | None]:
    bbox = [float(v) for v in element["bbox"]]
    body = ""
    linked_caption_id: str | None = None

    if pdf_reader.available:
        body = normalize_whitespace(
            pdf_reader.extract_bbox_text(page_number, bbox, page_width, page_height)
        )

    if element_type in FIGURE_TYPES:
        caption_el = find_caption_for_figure(bbox, caption_elements, page_height)
        if caption_el is not None:
            linked_caption_id = str(caption_el["element_id"])
            cap_text = caption_text_by_id.get(linked_caption_id, "")
            if not cap_text and pdf_reader.available:
                cap_text = normalize_whitespace(
                    pdf_reader.extract_bbox_text(
                        page_number,
                        [float(v) for v in caption_el["bbox"]],
                        page_width,
                        page_height,
                    )
                )
            parts = [part for part in (body, cap_text) if part]
            body = "\n".join(parts)

    if element_type == "table":
        # PDF clip text is the baseline; table-specific OCR/structure can be added later.
        pass

    if not body and element_type in FIGURE_TYPES:
        body = f"[{element_type} element — refer to crop image]"

    return body, linked_caption_id


def extract_chunks_for_doc(
    doc_id: str,
    layout_doc_dir: Path,
    crops_manifest_path: Path,
    pdf_reader: PdfTextReader,
    source_pdf: str,
    chunk_types: frozenset[str],
    include_headers: bool,
) -> tuple[list[dict], Counter[str], int]:
    chunks: list[dict] = []
    type_counter: Counter[str] = Counter()
    skipped_empty = 0
    crop_lookup = load_crop_lookup(crops_manifest_path)

    allowed_types = set(chunk_types)
    if include_headers:
        allowed_types |= SKIP_BY_DEFAULT

    json_files = sorted(layout_doc_dir.glob("page_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No layout pages in: {layout_doc_dir}")

    for json_file in json_files:
        page_data = json.loads(json_file.read_text(encoding="utf-8"))
        page_number = int(page_data["page_number"])
        page_width = int(page_data.get("width", 0))
        page_height = int(page_data.get("height", 0))
        page_name = str(page_data.get("page_name", json_file.name))
        source_page_image = str(page_data.get("image_path", ""))

        elements = page_data.get("elements", [])
        caption_elements = [
            el for el in elements if str(el.get("type", "")).lower() == "caption"
        ]
        caption_text_by_id: dict[str, str] = {}
        if pdf_reader.available:
            for cap in caption_elements:
                cap_id = str(cap["element_id"])
                caption_text_by_id[cap_id] = normalize_whitespace(
                    pdf_reader.extract_bbox_text(
                        page_number,
                        [float(v) for v in cap["bbox"]],
                        page_width,
                        page_height,
                    )
                )

        for element in elements:
            element_type = str(element.get("type", "unknown")).lower()
            if element_type not in allowed_types:
                continue
            if element_type == "caption":
                continue

            element_id = str(element["element_id"])
            crop_path = crop_lookup.get(element_id, "")

            text, linked_caption_id = extract_text_for_element(
                element,
                element_type,
                pdf_reader,
                page_number,
                page_width,
                page_height,
                caption_elements,
                caption_text_by_id,
            )

            if not text.strip() and element_type not in FIGURE_TYPES:
                skipped_empty += 1
                continue

            chunk = build_chunk_record(
                doc_id=doc_id,
                page_number=page_number,
                page_name=page_name,
                page_width=page_width,
                page_height=page_height,
                source_page_image=source_page_image,
                source_pdf=source_pdf,
                element=element,
                text=text,
                crop_path=crop_path,
                linked_caption_id=linked_caption_id,
            )
            chunks.append(chunk)
            type_counter[element_type] += 1

    chunks = expand_multi_subclause_chunks(chunks)
    chunks.sort(key=lambda c: (c["page_number"], c["bbox"][1], c["bbox"][0]))
    return chunks, type_counter, skipped_empty


def write_summary(
    summary_path: Path,
    doc_id: str,
    chunks: list[dict],
    type_counter: Counter[str],
    skipped_empty: int,
    source_pdf: str,
) -> None:
    lines = [
        f"Chunk Extraction Summary: {doc_id}",
        "=" * 60,
        f"Source PDF: {source_pdf or '(not available)'}",
        f"Total chunks: {len(chunks)}",
        f"Skipped (empty text, non-figure): {skipped_empty}",
        "",
        "Chunks by element_type:",
    ]
    for element_type, count in sorted(type_counter.items()):
        lines.append(f"  - {element_type}: {count}")

    text_lengths = [c["text_char_count"] for c in chunks if c.get("text")]
    if text_lengths:
        lines.extend(
            [
                "",
                f"Text length: min={min(text_lengths)}, max={max(text_lengths)}, "
                f"mean={sum(text_lengths) / len(text_lengths):.1f}",
            ]
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract RAG chunks from merged layout JSON.")
    parser.add_argument("--doc-id", type=str, required=True)
    parser.add_argument(
        "--layout-dir",
        type=Path,
        default=Path("data/processed/layout_json_merged"),
    )
    parser.add_argument(
        "--crops-dir",
        type=Path,
        default=Path("data/processed/crops_merged"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/chunks"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/pdf_manifest.csv"),
    )
    parser.add_argument("--pdf-path", type=Path, default=None)
    parser.add_argument(
        "--chunk-types",
        type=str,
        default=",".join(sorted(DEFAULT_CHUNK_TYPES)),
        help="Comma-separated element types to export as chunks",
    )
    parser.add_argument(
        "--include-headers",
        action="store_true",
        help="Include page-header and page-footer elements",
    )
    args = parser.parse_args()

    project_root = Path.cwd()
    layout_doc_dir = args.layout_dir / args.doc_id
    crops_manifest_path = args.crops_dir / args.doc_id / "crops_manifest.jsonl"
    out_doc_dir = args.out_dir / args.doc_id
    out_doc_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = out_doc_dir / "chunks.jsonl"
    summary_path = out_doc_dir / "chunks_summary.txt"

    pdf_path = resolve_pdf_path(args.doc_id, args.manifest, args.pdf_path, project_root)
    pdf_reader = PdfTextReader(pdf_path)
    if not pdf_reader.available:
        print(f"[WARN] PDF not found for {args.doc_id}; text fields may be empty.")

    chunk_types = parse_chunk_types(args.chunk_types)
    try:
        chunks, type_counter, skipped_empty = extract_chunks_for_doc(
            doc_id=args.doc_id,
            layout_doc_dir=layout_doc_dir,
            crops_manifest_path=crops_manifest_path,
            pdf_reader=pdf_reader,
            source_pdf=pdf_path.as_posix() if pdf_path else "",
            chunk_types=chunk_types,
            include_headers=args.include_headers,
        )
    finally:
        pdf_reader.close()

    with chunks_path.open("w", encoding="utf-8") as chunks_f:
        for chunk in chunks:
            chunks_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    write_summary(summary_path, args.doc_id, chunks, type_counter, skipped_empty, pdf_path.as_posix() if pdf_path else "")

    print(f"Chunks saved: {chunks_path} ({len(chunks)} records)")
    print(f"Summary saved: {summary_path}")
    print("Chunks by element_type:")
    for element_type, count in sorted(type_counter.items()):
        print(f"  - {element_type}: {count}")
    if skipped_empty:
        print(f"Skipped empty non-figure elements: {skipped_empty}")


if __name__ == "__main__":
    main()
