"""Extract structured tables and table-aware chunks from merged layout + PDF clip text."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from pdf_io import PdfTextReader, resolve_pdf_path
from table_extract_lib import (
    build_table_chunks,
    build_table_record,
    detect_parse_quality_issues,
    find_caption,
    find_section_title,
    is_toc_pseudo_table,
    normalize_whitespace,
    parse_table_structure,
    pseudo_table_penalty,
)


def load_text_chunk_map(chunks_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not chunks_path.exists():
        return mapping
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if str(rec.get("element_type", "")).lower() == "table":
                mapping[str(rec["element_id"])] = str(rec.get("text") or "")
    return mapping


def extract_tables_for_doc(
    doc_id: str,
    layout_doc_dir: Path,
    chunks_path: Path,
    pdf_reader: PdfTextReader,
    source_pdf: str,
    source_file: str,
) -> tuple[list[dict], list[dict], Counter[str]]:
    text_by_element = load_text_chunk_map(chunks_path)
    tables: list[dict] = []
    all_chunks: list[dict] = []
    chunk_type_counter: Counter[str] = Counter()
    skipped_pseudo = 0
    parse_issue_count = 0

    json_files = sorted(layout_doc_dir.glob("page_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No layout pages in: {layout_doc_dir}")

    table_seq = 0
    for json_file in json_files:
        page_data = json.loads(json_file.read_text(encoding="utf-8"))
        page_number = int(page_data["page_number"])
        page_width = int(page_data.get("width", 0))
        page_height = int(page_data.get("height", 0))
        elements = page_data.get("elements", [])

        for element in elements:
            if str(element.get("type", "")).lower() != "table":
                continue

            element_id = str(element["element_id"])
            bbox_f = [float(v) for v in element.get("bbox", [0, 0, 0, 0])]
            bbox = [int(round(v)) for v in bbox_f]
            table_id = f"{doc_id}_p{page_number:04d}_t{table_seq:03d}"
            table_seq += 1

            raw_text = text_by_element.get(element_id, "")
            if not raw_text and pdf_reader.available:
                raw_text = normalize_whitespace(
                    pdf_reader.extract_bbox_text(page_number, bbox_f, page_width, page_height)
                )
            if not raw_text or len(raw_text.strip()) < 8:
                continue

            confidence = float(element.get("confidence", 1.0))
            caption = find_caption(
                elements,
                bbox_f,
                raw_text,
                pdf_reader=pdf_reader,
                page_number=page_number,
                page_width=page_width,
                page_height=page_height,
            )
            section_title = find_section_title(
                elements,
                bbox_f,
                pdf_reader=pdf_reader,
                page_number=page_number,
                page_width=page_width,
                page_height=page_height,
            )

            is_pseudo, pseudo_reason = is_toc_pseudo_table(
                raw_text, confidence=confidence, caption=caption
            )
            if is_pseudo:
                skipped_pseudo += 1
                tables.append(
                    {
                        "doc_id": doc_id,
                        "page": page_number,
                        "table_id": table_id,
                        "element_id": element_id,
                        "is_pseudo_table": True,
                        "pseudo_reason": pseudo_reason,
                        "confidence": confidence,
                        "raw_table_text": raw_text[:500],
                    }
                )
                continue

            columns, rows, _md = parse_table_structure(raw_text)
            issues = detect_parse_quality_issues(columns=columns, rows=rows, raw_text=raw_text)
            if issues:
                parse_issue_count += 1
            quality = pseudo_table_penalty(issues, is_pseudo=False)

            table = build_table_record(
                doc_id=doc_id,
                source_file=source_file or source_pdf,
                page=page_number,
                table_id=table_id,
                element_id=element_id,
                bbox=bbox,
                raw_text=raw_text,
                caption=caption,
                section_title=section_title,
                confidence=confidence,
                is_pseudo=False,
                parse_issues=issues,
                table_quality=quality,
            )
            tables.append(table)

            for chunk in build_table_chunks(table, file_name=source_file):
                all_chunks.append(chunk)
                chunk_type_counter[str(chunk["chunk_type"])] += 1

    return tables, all_chunks, chunk_type_counter, skipped_pseudo, parse_issue_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structured tables and table chunks.")
    parser.add_argument("--doc-id", type=str, required=True)
    parser.add_argument("--layout-dir", type=Path, default=Path("data/processed/layout_json_merged"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--tables-dir", type=Path, default=Path("data/processed/tables"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/pdf_manifest.csv"))
    parser.add_argument("--pdf-path", type=Path, default=None)
    args = parser.parse_args()

    doc_id = args.doc_id
    project_root = Path(__file__).resolve().parent.parent
    layout_doc_dir = args.layout_dir / doc_id
    chunks_path = args.chunks_dir / doc_id / "chunks.jsonl"
    if not layout_doc_dir.exists():
        raise FileNotFoundError(f"Layout dir missing: {layout_doc_dir}")

    pdf_path = resolve_pdf_path(doc_id, args.manifest, args.pdf_path, project_root)
    source_pdf = str(pdf_path.resolve()) if pdf_path else ""
    source_file = pdf_path.name if pdf_path else ""

    pdf_reader = PdfTextReader(pdf_path)
    tables, table_chunks, type_counter, skipped_pseudo, parse_issues = extract_tables_for_doc(
        doc_id,
        layout_doc_dir,
        chunks_path,
        pdf_reader,
        source_pdf,
        source_file,
    )

    out_tables_dir = args.tables_dir / doc_id
    out_tables_dir.mkdir(parents=True, exist_ok=True)
    tables_path = out_tables_dir / "tables.jsonl"
    with tables_path.open("w", encoding="utf-8") as f:
        for table in tables:
            f.write(json.dumps(table, ensure_ascii=False) + "\n")

    table_chunks_path = args.chunks_dir / doc_id / "table_chunks.jsonl"
    table_chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with table_chunks_path.open("w", encoding="utf-8") as f:
        for chunk in table_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    summary_path = out_tables_dir / "table_extraction_summary.txt"
    lines = [
        f"Table extraction: {doc_id}",
        f"Tables: {len(tables)}",
        f"Table chunks: {len(table_chunks)}",
        f"Skipped pseudo-tables: {skipped_pseudo}",
        f"Tables with parse issues: {parse_issues}",
        "Chunk types:",
    ]
    for k, v in sorted(type_counter.items()):
        lines.append(f"  {k}: {v}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{doc_id}: {len(tables)} tables -> {len(table_chunks)} chunks (skipped pseudo: {skipped_pseudo})")
    for k, v in sorted(type_counter.items()):
        print(f"  {k}: {v}")
    print(f"Wrote {tables_path}")
    print(f"Wrote {table_chunks_path}")


if __name__ == "__main__":
    main()
