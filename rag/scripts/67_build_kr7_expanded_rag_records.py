#!/usr/bin/env python3
"""Build grounded RAG chunks for the incrementally expanded KR Part 7 pilot."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import importlib

snap = importlib.import_module("53_snap_tatr_to_pdf")
from hancomeqn_restore import (
    is_hancomeqn,
    iter_page_glyphs,
    load_mapping,
    point_in_bbox,
    restore_formula,
    restore_inline,
)
from pdf_io import resolve_pdf_path

DOC_ID = "kr_kr_rules_7_2025_2f2d6373"
PARSER_VERSION = "pdf-vector-region+tatr-local+hancomeqn-v2"


def crop_box_to_pdf(box: list[float], clip: fitz.Rect, crop_size: tuple[int, int]) -> fitz.Rect:
    sx, sy = clip.width / crop_size[0], clip.height / crop_size[1]
    return fitz.Rect(
        clip.x0 + box[0] * sx,
        clip.y0 + box[1] * sy,
        clip.x0 + box[2] * sx,
        clip.y0 + box[3] * sy,
    )


def region_content(page: fitz.Page, pdf_box: fitz.Rect, mapping: dict) -> dict:
    glyphs = list(iter_page_glyphs(page, pdf_box))
    raw = page.get_text("text", clip=pdf_box, sort=True).strip()
    restored, inline_unknown = restore_inline(raw, mapping)
    equation_glyphs = [glyph for glyph in glyphs if is_hancomeqn(glyph.font)]
    formula = restore_formula(equation_glyphs, mapping) if equation_glyphs else None
    return {
        "text_raw": raw,
        "text_restored": restored,
        "inline_unknown_glyphs": inline_unknown,
        "formula": formula,
    }


def chunk(
    *,
    chunk_id: str,
    table: dict,
    region_id: str,
    region_kind: str,
    text: str,
    crop_path: Path,
    chunk_type: str,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": DOC_ID,
        "source": "KR",
        "file_name": "7편_2025.pdf",
        "page_number": int(table["page"]),
        "element_type": "table",
        "element_id": f"{table['table_id']}:{region_id}",
        "chunk_type": chunk_type,
        "table_id": table["table_id"],
        "caption": table.get("caption") or "",
        "section_title": "KR 선급 규칙 7편",
        "region_id": region_id,
        "region_kind": region_kind,
        "text": text,
        "crop_path": str(crop_path.resolve()),
        "parser_version": PARSER_VERSION,
        "quality_status": "pass",
        "quality_score": 1.0,
    }


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
    parser.add_argument(
        "--mapping",
        type=Path,
        default=ROOT / "data/config/hancomeqn_maps/7_2025_bdc15136d686.json",
    )
    parser.add_argument(
        "--chunks-root",
        type=Path,
        default=ROOT / "data/processed/chunks_kr7_expanded_pilot",
    )
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    mapping = load_mapping(args.mapping)
    pdf = resolve_pdf_path(DOC_ID, ROOT / "data/manifests/pdf_manifest.csv", None, ROOT)
    if pdf is None:
        raise FileNotFoundError(DOC_ID)
    records = []
    audit = []
    doc = fitz.open(pdf)
    try:
        for table in manifest["tables"]:
            table = dict(table)
            table["caption"] = restore_inline(str(table.get("caption") or ""), mapping)[0]
            table_id = table["table_id"]
            record_start = len(records)
            source_table = snap.find_table(ROOT / "data/processed/tables_v2", table_id)
            page = doc[int(table["page"]) - 1]
            table_dir = args.pilot_root / DOC_ID / table_id
            split_dir = table_dir / "region_split_v1"
            region_data = json.loads((split_dir / "regions.json").read_text(encoding="utf-8"))
            crop_size = tuple(int(value) for value in region_data["crop_size"])
            clip = snap.render_clip(
                page,
                source_table["bbox"],
                snap.layout_size(ROOT / "data/processed/layout_json_merged", DOC_ID, int(table["page"])),
            )
            table_region_texts = []
            table_unknown = set()
            for region in region_data["regions"]:
                targets = [region, *region.get("children", [])]
                for target in targets:
                    region_id = target["region_id"]
                    pdf_box = crop_box_to_pdf(target["bbox_crop"], clip, crop_size)
                    content = region_content(page, pdf_box, mapping)
                    panel_label = ""
                    if target.get("panel_bbox_crop"):
                        panel_box = list(target["panel_bbox_crop"])
                        panel_box[3] = min(panel_box[3], target["bbox_crop"][1])
                        panel_pdf_box = crop_box_to_pdf(panel_box, clip, crop_size)
                        panel_text = restore_inline(
                            page.get_text("text", clip=panel_pdf_box, sort=True), mapping
                        )[0]
                        lines = [line.strip() for line in panel_text.splitlines() if line.strip()]
                        panel_label = next(
                            (line for line in lines if "트랜스버스" in line or "스트링거" in line),
                            next(
                                (
                                    line
                                    for line in lines
                                    if any("가" <= char <= "힣" for char in line) and len(line) <= 40
                                ),
                                lines[0] if lines else "",
                            ),
                        )
                    formula = content["formula"]
                    unknown = set(content["inline_unknown_glyphs"])
                    if formula:
                        unknown.update(formula["unknown_glyphs"])
                    table_unknown.update(unknown)
                    parts = [
                        f"{table.get('caption') or '표'} (7편_2025.pdf, {table['page']}쪽)",
                        f"영역: {region_id} / {target['kind']}",
                        f"상위 패널 제목: {panel_label}" if panel_label else "",
                        content["text_restored"],
                        (
                            "단위 주의: C1과 C2는 무차원 계수이다."
                            if target.get("kind") == "nested_grid"
                            and "C1" in content["text_restored"]
                            and "C2" in content["text_restored"]
                            else ""
                        ),
                    ]
                    # Native PDF reading order is the primary RAG evidence.  The
                    # geometric formula serializer remains audit metadata only;
                    # on stacked fractions it can be fully mapped yet ordered
                    # ambiguously, so indexing it beside correct native text can
                    # mislead the answer model.
                    text = "\n".join(part for part in parts if part).strip()
                    # Some wide tables are split into a title/header region and
                    # a following data grid.  Preserve the two immediately
                    # preceding regions as inherited context for row evidence.
                    prior_region_context = table_region_texts[-2:]
                    table_region_texts.append(text)
                    crop_path = split_dir / f"{region_id}.png"
                    records.append(
                        chunk(
                            chunk_id=f"{table_id}:{region_id}",
                            table=table,
                            region_id=region_id,
                            region_kind=target["kind"],
                            text=text,
                            crop_path=crop_path,
                            chunk_type="table_formula" if formula else "table_definition",
                        )
                    )

                    local_tatr = split_dir / f"{region_id}_tatr" / "structure.json"
                    if target.get("requires_local_structure") and local_tatr.exists():
                        local_structure = json.loads(local_tatr.read_text(encoding="utf-8"))
                        row_boundaries = snap.object_boundaries(
                            local_structure.get("detections", []), "table row", "y"
                        )
                        column_boundaries = snap.object_boundaries(
                            local_structure.get("detections", []), "table column", "x"
                        )
                        local_spans = snap.build_spans(
                            local_structure.get("detections", []),
                            row_boundaries,
                            column_boundaries,
                        )
                        local_height = max(float(local_structure["crop_size"][1]), 1.0)
                        local_width = max(float(local_structure["crop_size"][0]), 1.0)
                        tx0, ty0, tx1, ty1 = [float(value) for value in target["bbox_crop"]]
                        cell_rows: list[list[str]] = []
                        if len(column_boundaries) >= 2:
                            for local_y0, local_y1 in zip(row_boundaries, row_boundaries[1:]):
                                cells = []
                                for local_x0, local_x1 in zip(
                                    column_boundaries, column_boundaries[1:]
                                ):
                                    cell_parent_box = [
                                        tx0 + local_x0 / local_width * (tx1 - tx0),
                                        ty0 + local_y0 / local_height * (ty1 - ty0),
                                        tx0 + local_x1 / local_width * (tx1 - tx0),
                                        ty0 + local_y1 / local_height * (ty1 - ty0),
                                    ]
                                    cell_content = region_content(
                                        page,
                                        crop_box_to_pdf(cell_parent_box, clip, crop_size),
                                        mapping,
                                    )
                                    cells.append(" ".join(cell_content["text_restored"].split()))
                                cell_rows.append(cells)

                        header_bottom = max(
                            (
                                float(item["bbox_crop"][3])
                                for item in local_structure.get("detections", [])
                                if item.get("label") == "table column header"
                            ),
                            default=row_boundaries[1] if len(row_boundaries) >= 2 else 0.0,
                        )
                        header_row_indexes = {
                            index
                            for index, (local_y0, local_y1) in enumerate(
                                zip(row_boundaries, row_boundaries[1:])
                            )
                            if (local_y0 + local_y1) / 2 <= header_bottom
                        }
                        merged_group_labels = []
                        for cell_row_index, cells in enumerate(cell_rows):
                            for value in cells[:2]:
                                match = re.search(r"\b\d+\s*조\b", value)
                                if match:
                                    merged_group_labels.append(
                                        (cell_row_index, " ".join(match.group(0).split()))
                                    )
                        for row_index, (local_y0, local_y1) in enumerate(
                            zip(row_boundaries, row_boundaries[1:]), 1
                        ):
                            parent_box = [
                                tx0,
                                ty0 + local_y0 / local_height * (ty1 - ty0),
                                tx1,
                                ty0 + local_y1 / local_height * (ty1 - ty0),
                            ]
                            row_content = region_content(
                                page, crop_box_to_pdf(parent_box, clip, crop_size), mapping
                            )
                            row_text = row_content["text_restored"].strip()
                            if not row_text:
                                continue
                            row_parts = [
                                f"{table.get('caption') or '표'} (7편_2025.pdf, {table['page']}쪽)",
                                f"영역: {region_id} / 행 {row_index}",
                                f"상위 패널 제목: {panel_label}" if panel_label else "",
                                (
                                    "앞선 표 영역 문맥:\n" + "\n\n".join(prior_region_context)
                                    if prior_region_context
                                    else ""
                                ),
                                row_text,
                            ]
                            if cell_rows:
                                if merged_group_labels:
                                    nearest_group = min(
                                        merged_group_labels,
                                        key=lambda item: abs(item[0] - (row_index - 1)),
                                    )[1]
                                    row_parts.append(
                                        f"병합 행 머리글(최근접 중심 배정): {nearest_group}"
                                    )
                                current_zero_index = row_index - 1
                                inherited_previous_rows = set()
                                for span in local_spans:
                                    if (
                                        span["row"] < current_zero_index
                                        <= span["row"] + span["rowspan"] - 1
                                    ):
                                        inherited_previous_rows.update(
                                            range(span["row"], current_zero_index)
                                        )
                                relevant_rows = sorted(
                                    header_row_indexes
                                    | {current_zero_index}
                                    | inherited_previous_rows
                                )
                                grid_lines = []
                                for grid_row_index in relevant_rows:
                                    if grid_row_index >= len(cell_rows):
                                        continue
                                    serialized_cells = " | ".join(
                                        f"열{column_index}={value or '∅'}"
                                        for column_index, value in enumerate(
                                            cell_rows[grid_row_index], 1
                                        )
                                    )
                                    grid_lines.append(
                                        f"셀 좌표 행{grid_row_index + 1}: {serialized_cells}"
                                    )
                                if grid_lines:
                                    row_parts.append(
                                        "TATR 열 경계+PyMuPDF 텍스트 셀 정렬:\n"
                                        + "\n".join(grid_lines)
                                    )
                                    if inherited_previous_rows:
                                        row_parts.append(
                                            "세로 병합 문맥: 병합 그룹의 시작 행부터 현재 행까지 "
                                            "동일한 열을 함께 읽어야 하며 다른 열의 값과 섞지 않는다."
                                        )
                            records.append(
                                chunk(
                                    chunk_id=f"{table_id}:{region_id}:ROW{row_index:02d}",
                                    table=table,
                                    region_id=f"{region_id}:ROW{row_index:02d}",
                                    region_kind="local_table_row",
                                    text="\n".join(part for part in row_parts if part),
                                    crop_path=crop_path,
                                    chunk_type="table_row_aux",
                                )
                            )

            summary_text = "\n\n".join(
                [
                    f"{table.get('caption') or '표'} (7편_2025.pdf, {table['page']}쪽)",
                    *table_region_texts,
                ]
            )
            records.append(
                chunk(
                    chunk_id=f"{table_id}:SUMMARY",
                    table=table,
                    region_id="SUMMARY",
                    region_kind="table_summary",
                    text=summary_text,
                    crop_path=table_dir / "crop.png",
                    chunk_type="table_summary",
                )
            )
            audit.append(
                {
                    "table_id": table_id,
                    "page": table["page"],
                    "caption": table.get("caption") or "",
                    "region_chunks": len(table_region_texts),
                    "unknown_glyphs": sorted(table_unknown),
                    "indexable": not table_unknown,
                }
            )
            # Never contaminate the production index with unresolved formula
            # glyphs. Keep the table in the audit report, but remove only the
            # records accumulated for that table so the remaining document can
            # still be built and tested.
            if table_unknown:
                del records[record_start:]
    finally:
        doc.close()

    output_dir = args.chunks_root / DOC_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = output_dir / "chunks.jsonl"
    chunks_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    audit_path = args.audit_output or args.manifest.with_name(f"{args.manifest.stem}_rag_records.json")
    audit_path.write_text(
        json.dumps(
            {
                "summary": {
                    "tables": len(audit),
                    "chunks": len(records),
                    "indexable_tables": sum(row["indexable"] for row in audit),
                },
                "tables": audit,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"chunks": len(records), "output": str(chunks_path.resolve()), "audit": str(audit_path.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
