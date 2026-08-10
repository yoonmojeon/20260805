#!/usr/bin/env python3
"""Build a resumable TATR+PyMuPDF+HancomEQN table corpus for many PDFs.

The pipeline never guesses an unknown private-use glyph. Tables containing PUA
characters are indexed only when a reviewed mapping exists and every glyph in
the table can be restored. Rejected tables remain in the audit report.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import fitz

ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(_SCRIPT_DIR))
if str(ROOT.parent) not in sys.path:
    # Repo root (…/20260805) when this file lives under rag/scripts.
    sys.path.insert(0, str(ROOT.parent))

from hancomeqn_restore import PUA_RE, load_mapping, restore_inline
from pdf_io import resolve_pdf_path

cropper = importlib.import_module("49_vlm_table_pilot")
snap = importlib.import_module("53_snap_tatr_to_pdf")

PARSER_VERSION = "tatr-v1.1+pdf-vector-snap+pymupdf-text+reviewed-hancomeqn-v1"
DEFAULT_STAGES = "preprocess,prepare,tatr,snap,restore,segment,region-tatr,chunks,index"


def short_work_dir(work_root: Path, doc_id: str, table_id: str) -> Path:
    """Avoid Windows MAX_PATH failures from very long doc/table ids."""
    import hashlib
    import re

    doc_leaf = doc_id.rsplit("_", 1)[-1][:16] or hashlib.md5(doc_id.encode()).hexdigest()[:12]
    match = re.search(r"(p\d{4}_t\d{3})$", table_id)
    table_leaf = match.group(1) if match else hashlib.md5(table_id.encode()).hexdigest()[:12]
    return work_root / doc_leaf / table_leaf


def table_work_dir(item: dict[str, Any], work_root: Path) -> Path:
    crop = str(item.get("crop_path") or "")
    if crop:
        return Path(crop).parent
    return short_work_dir(work_root, str(item["doc_id"]), str(item["table_id"]))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def load_docs(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "doc_id" not in rows[0]:
        raise ValueError(f"doc-list must contain doc_id: {path}")
    return rows


def load_registry(path: Path) -> dict[str, dict[str, Path]]:
    if not path.exists():
        return {"documents": {}, "source_defaults": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        group: {
            str(key): (path.parent / value).resolve()
            for key, value in data.get(group, {}).items()
        }
        for group in ("documents", "source_defaults")
    }


def resolve_mapping(registry: dict[str, dict[str, Path]], doc_id: str, source: str) -> tuple[Path | None, str]:
    document_mapping = registry.get("documents", {}).get(doc_id)
    if document_mapping:
        return document_mapping, "document"
    source_mapping = registry.get("source_defaults", {}).get(source.upper())
    if source_mapping:
        return source_mapping, f"source:{source.upper()}"
    return None, "none"


def run(command: list[str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def pua_codes(page: fitz.Page, clip: fitz.Rect) -> list[str]:
    return [
        f"U+{ord(glyph.char):04X}"
        for glyph in importlib.import_module("hancomeqn_restore").iter_page_glyphs(page, clip)
        if PUA_RE.fullmatch(glyph.char)
    ]


def prepare(args: argparse.Namespace, docs: list[dict[str, str]], registry: dict[str, dict[str, Path]]) -> dict:
    records: list[dict[str, Any]] = []
    missing_tables: list[str] = []
    for doc_index, doc_row in enumerate(docs, 1):
        doc_id = str(doc_row["doc_id"])
        tables = read_jsonl(args.tables_root / doc_id / "tables.jsonl")
        if not tables:
            missing_tables.append(doc_id)
            continue
        pdf_path = resolve_pdf_path(doc_id, args.pdf_manifest, None, ROOT)
        if pdf_path is None:
            raise FileNotFoundError(f"PDF not found for {doc_id}")
        source = str(doc_row.get("source") or "KR").upper()
        mapping_path, mapping_scope = resolve_mapping(registry, doc_id, source)
        known = set(load_mapping(mapping_path).get("glyphs", {})) if mapping_path else set()
        document = fitz.open(pdf_path)
        try:
            for table_index, table in enumerate(tables, 1):
                if table.get("is_pseudo_table"):
                    continue
                bbox = table.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                table_id = str(table["table_id"])
                page_no = int(table["page"])
                page = document[page_no - 1]
                dims = cropper.layout_size(args.layout_root, doc_id, page_no)
                clip = snap.render_clip(page, bbox, dims)
                codes = pua_codes(page, clip)
                unknown = sorted(set(codes) - known)
                crop_path = short_work_dir(args.work_root, doc_id, table_id) / "crop.png"
                if not (args.resume and crop_path.exists()):
                    cropper.render_crop(pdf_path, list(bbox), dims, page_no, crop_path)
                records.append({
                    "doc_id": doc_id,
                    "file_name": str(doc_row.get("file_name") or pdf_path.name),
                    "source": source,
                    "table_id": table_id,
                    "page": page_no,
                    "caption": str(table.get("caption") or ""),
                    "crop_path": str(crop_path.resolve()),
                    "work_dir": str(crop_path.parent.resolve()),
                    "pua_occurrences": len(codes),
                    "pua_unique": len(set(codes)),
                    "mapping": str(mapping_path) if mapping_path else "",
                    "mapping_scope": mapping_scope,
                    "unknown_pua": unknown,
                })
                print(f"[{doc_index}/{len(docs)} {table_index}/{len(tables)}] crop {table_id}", flush=True)
        finally:
            document.close()
    payload = {
        "schema_version": 1,
        "selection_method": "all-detected-tables-in-document-list",
        "document_count": len(docs),
        "table_count": len(records),
        "missing_table_documents": missing_tables,
        "tables": records,
    }
    args.pipeline_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.pipeline_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_pipeline_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Run the prepare stage first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_tatr(args: argparse.Namespace, manifest: dict) -> None:
    batch = importlib.import_module("60_run_kr7_tatr_batch")
    pending = []
    for item in manifest["tables"]:
        output = table_work_dir(item, args.work_root) / "tatr_v1_1_all"
        if args.resume and (output / "structure.json").exists():
            continue
        pending.append(item)
    if not pending:
        print("TATR: all crops already processed", flush=True)
        return
    runtime = batch.load_runtime(batch.TATR.MODEL_ID)
    for index, item in enumerate(pending, 1):
        output = table_work_dir(item, args.work_root) / "tatr_v1_1_all"
        result = batch.infer(Path(item["crop_path"]), runtime, args.tatr_threshold, args.tatr_padding, output)
        print(f"[{index}/{len(pending)}] TATR {item['table_id']} {result['summary']['row_count']}x{result['summary']['column_count']}", flush=True)


def run_snap(args: argparse.Namespace, manifest: dict) -> None:
    for index, item in enumerate(manifest["tables"], 1):
        work = table_work_dir(item, args.work_root)
        result = work / "tatr_v1_1_all/snapped_structure.json"
        if args.resume and result.exists():
            continue
        run([sys.executable, "scripts/53_snap_tatr_to_pdf.py", "--table-id", item["table_id"],
             "--manifest", str(args.pdf_manifest), "--tables-root", str(args.tables_root),
             "--layout-root", str(args.layout_root), "--pilot-root", str(args.work_root),
             "--work-dir", str(work)])
        print(f"[{index}/{len(manifest['tables'])}] snap {item['table_id']}", flush=True)


def run_restore(args: argparse.Namespace, manifest: dict) -> None:
    for index, item in enumerate(manifest["tables"], 1):
        if not item["pua_occurrences"] or not item.get("mapping") or item.get("unknown_pua"):
            continue
        work = table_work_dir(item, args.work_root)
        result = work / "tatr_v1_1_all/restored_formulas.json"
        if args.resume and result.exists():
            continue
        run([sys.executable, "scripts/55_restore_hancomeqn.py", "--table-id", item["table_id"],
             "--mapping", item["mapping"], "--manifest", str(args.pdf_manifest),
             "--pilot-root", str(args.work_root), "--tables-root", str(args.tables_root),
             "--layout-root", str(args.layout_root), "--work-dir", str(work)])
        print(f"[{index}/{len(manifest['tables'])}] restore {item['table_id']}", flush=True)


def restored_cell_map(path: Path, mapping: dict) -> tuple[dict[str, str], set[str]]:
    values: dict[str, str] = {}
    unknown: set[str] = set()
    if not path.exists():
        return values, unknown
    payload = json.loads(path.read_text(encoding="utf-8"))
    for cell in payload.get("cells", []):
        formula = cell.get("formula") or {}
        unknown.update(formula.get("unknown_glyphs") or [])
        unknown.update(cell.get("inline_unknown_glyphs") or [])
        inline = str(cell.get("inline_text_restored") or "").strip()
        normalized = str(formula.get("normalized") or "").strip()
        values[str(cell["cell_id"])] = normalized if cell.get("formula_dominant") and normalized else inline
    return values, unknown


def serialize_rows(item: dict, structure: dict, mapping: dict, restored: dict[str, str]) -> list[dict]:
    cells = structure.get("cells") or []
    row_ids = sorted({int(cell["row"]) for cell in cells})
    rows: list[dict] = []
    inherited: dict[int, str] = {}
    for cell in cells:
        raw = restored.get(str(cell["cell_id"]), str(cell.get("text_raw") or ""))
        text = restore_inline(raw, mapping)[0] if mapping else raw
        if text and int(cell.get("rowspan", 1)) > 1:
            for row in range(int(cell["row"]), int(cell["row"]) + int(cell["rowspan"])):
                inherited[row] = text
    for row_id in row_ids:
        current = sorted((cell for cell in cells if int(cell["row"]) == row_id), key=lambda value: int(value["column"]))
        parts = []
        for cell in current:
            raw = restored.get(str(cell["cell_id"]), str(cell.get("text_raw") or ""))
            value = restore_inline(raw, mapping)[0] if mapping else raw
            header_values = cell.get("column_header_path") or []
            headers = " > ".join(restore_inline(str(value), mapping)[0] if mapping else str(value) for value in header_values)
            prefix = f"{headers}: " if headers and cell.get("type") != "header" else ""
            parts.append(f"열{int(cell['column']) + 1}={prefix}{value or '(빈 셀)'}")
        if inherited.get(row_id) and inherited[row_id] not in " ".join(parts):
            parts.insert(0, f"병합 행 머리글={inherited[row_id]}")
        rows.append({"row": row_id, "text": " | ".join(parts)})
    return rows


def crop_box_to_pdf(box: list[float], clip: fitz.Rect, crop_size: tuple[int, int]) -> fitz.Rect:
    sx, sy = clip.width / crop_size[0], clip.height / crop_size[1]
    return fitz.Rect(clip.x0 + box[0] * sx, clip.y0 + box[1] * sy,
                     clip.x0 + box[2] * sx, clip.y0 + box[3] * sy)


def local_region_rows(page: fitz.Page, parent_clip: fitz.Rect, parent_size: tuple[int, int],
                      target: dict, structure_path: Path, mapping: dict) -> list[dict]:
    """Snap a region TATR grid to PDF vectors and assign native PDF words."""
    if not structure_path.exists():
        return []
    structure = json.loads(structure_path.read_text(encoding="utf-8"))
    detections = structure.get("detections") or []
    local_size = tuple(int(value) for value in structure.get("crop_size") or (0, 0))
    if min(local_size) <= 0:
        return []
    region_clip = crop_box_to_pdf([float(value) for value in target["bbox_crop"]], parent_clip, parent_size)
    horizontal, vertical = snap.vector_boundaries(page, region_clip, local_size)
    row_proposals = snap.object_boundaries(detections, "table row", "y")
    column_proposals = snap.object_boundaries(detections, "table column", "x")
    if len(row_proposals) < 2 or len(column_proposals) < 2:
        return []
    rows = snap.snap_boundaries(row_proposals, horizontal, tolerance=max(25.0, local_size[1] * 0.08))
    columns = snap.snap_boundaries(column_proposals, vertical, tolerance=max(25.0, local_size[0] * 0.08))
    spans = snap.build_spans(detections, rows, columns)
    words = snap.extract_words(page, region_clip, local_size)
    header_rows = snap.header_row_indices(detections, rows)
    cells = snap.build_cells(rows, columns, spans, words, header_rows)
    snap.add_header_paths(cells, header_rows, len(columns) - 1)
    return serialize_rows({}, {"cells": cells}, mapping, {})


def compound_rows(args: argparse.Namespace, item: dict, mapping: dict,
                  page: fitz.Page, parent_clip: fitz.Rect, parent_size: tuple[int, int]) -> tuple[bool, list[dict]]:
    split_dir = table_work_dir(item, args.work_root) / "region_split_v1"
    regions_path = split_dir / "regions.json"
    if not regions_path.exists():
        return False, []
    payload = json.loads(regions_path.read_text(encoding="utf-8"))
    if not payload.get("compound"):
        return False, []
    output: list[dict] = []
    for region in payload.get("regions", []):
        targets = region.get("children") or ([region] if region.get("requires_local_structure") else [])
        for target in targets:
            rows = local_region_rows(page, parent_clip, parent_size, target,
                                     split_dir / f"{target['region_id']}_tatr/structure.json", mapping)
            for row in rows:
                output.append({"row": len(output), "region_id": target["region_id"],
                               "text": f"영역={target['region_id']} | {row['text']}"})
        raw = str(region.get("text_raw") or "").strip()
        if raw:
            restored = restore_inline(raw, mapping)[0] if mapping else raw
            output.append({"row": len(output), "region_id": region["region_id"],
                           "text": f"영역={region['region_id']} | {restored}"})
    return True, output


def build_chunks(args: argparse.Namespace, docs: list[dict[str, str]], manifest: dict) -> None:
    by_doc: dict[str, list[dict]] = defaultdict(list)
    audit: list[dict] = []
    documents: dict[str, fitz.Document] = {}
    try:
      for item in manifest["tables"]:
        doc_id, table_id = item["doc_id"], item["table_id"]
        table_dir = table_work_dir(item, args.work_root)
        structure_path = table_dir / "tatr_v1_1_all/snapped_structure.json"
        reasons: list[str] = []
        if not structure_path.exists():
            reasons.append("missing_snapped_structure")
        if item.get("pua_occurrences") and not item.get("mapping"):
            reasons.append("missing_reviewed_pua_mapping")
        if item.get("unknown_pua"):
            reasons.append("unmapped_pua:" + ",".join(item["unknown_pua"]))
        mapping = load_mapping(Path(item["mapping"])) if item.get("mapping") else {}
        restored_path = table_dir / "tatr_v1_1_all/restored_formulas.json"
        if item.get("pua_occurrences") and item.get("mapping") and not restored_path.exists():
            reasons.append("missing_restored_formulas")
        restored, restore_unknown = restored_cell_map(restored_path, mapping)
        if restore_unknown:
            reasons.append("restore_review:" + ",".join(sorted(restore_unknown)))
        structure = json.loads(structure_path.read_text(encoding="utf-8")) if structure_path.exists() else {}
        if int(structure.get("row_count", 0)) < 1 or int(structure.get("column_count", 0)) < 1:
            reasons.append("empty_grid")
        rows = serialize_rows(item, structure, mapping, restored) if not reasons else []
        compound = False
        if not reasons:
            doc_id = item["doc_id"]
            if doc_id not in documents:
                for opened in documents.values():
                    opened.close()
                documents.clear()
                pdf = resolve_pdf_path(doc_id, args.pdf_manifest, None, ROOT)
                if pdf is None:
                    reasons.append("missing_pdf")
                else:
                    documents[doc_id] = fitz.open(pdf)
            if doc_id in documents:
                source_table = snap.find_table(args.tables_root, table_id)
                page = documents[doc_id][int(item["page"]) - 1]
                parent_clip = snap.render_clip(page, source_table["bbox"],
                                               snap.layout_size(args.layout_root, doc_id, int(item["page"])))
                parent_size = tuple(int(value) for value in json.loads(
                    (table_dir / "tatr_v1_1_all/structure.json").read_text(encoding="utf-8"))["crop_size"])
                compound, region_rows = compound_rows(args, item, mapping, page, parent_clip, parent_size)
                if compound:
                    rows = region_rows
                    if not rows:
                        reasons.append("compound_regions_have_no_usable_rows")
        if not any(row["text"].replace("(빈 셀)", "").strip(" |") for row in rows):
            reasons.append("no_usable_text")
        indexable = not reasons
        audit.append({"doc_id": doc_id, "table_id": table_id, "page": item["page"], "compound": compound,
                      "indexable": indexable, "reasons": reasons,
                      "rows": int(structure.get("row_count", 0)), "columns": int(structure.get("column_count", 0))})
        if not indexable:
            continue
        caption = restore_inline(str(item.get("caption") or ""), mapping)[0] if mapping else str(item.get("caption") or "")
        common = {"doc_id": doc_id, "source": item.get("source", "KR"), "file_name": item.get("file_name", ""),
                  "page": item["page"], "page_number": item["page"], "element_type": "table",
                  "table_id": table_id, "caption": caption, "parser_version": PARSER_VERSION,
                  "quality_status": "pass", "quality_score": 1.0, "crop_path": item["crop_path"]}
        header = f"표: {caption or table_id}\n문서: {item.get('file_name', '')}, {item['page']}쪽"
        for row in rows:
            region_suffix = f":{row['region_id']}" if row.get("region_id") else ""
            by_doc[doc_id].append({**common, "chunk_id": f"{table_id}{region_suffix}:ROW{row['row']:03d}",
                                   "element_id": f"{table_id}{region_suffix}:ROW{row['row']:03d}", "chunk_type": "table_row",
                                   "text": header + "\n" + row["text"]})
        summary_text = header + "\n" + "\n".join(row["text"] for row in rows)
        by_doc[doc_id].append({**common, "chunk_id": f"{table_id}:SUMMARY", "element_id": f"{table_id}:SUMMARY",
                               "chunk_type": "table_summary", "text": summary_text})
        # Schema catalog chunk for stage-1 table routing (searcher expects table_schema).
        try:
            from table_schema_lib import build_table_schema_text, parse_schema_from_document

            schema = parse_schema_from_document(
                summary_text,
                {
                    "table_id": table_id,
                    "caption": caption,
                    "source_file": item.get("file_name", ""),
                    "file_name": item.get("file_name", ""),
                    "page": item["page"],
                    "doc_id": doc_id,
                },
            )
            schema["doc_id"] = doc_id
            schema["source_file"] = str(item.get("file_name") or "")
            schema["page"] = int(item["page"])
            schema["table_id"] = table_id
            schema["row_count"] = len(rows)
            if not schema.get("_raw_snippet"):
                schema["_raw_snippet"] = summary_text[:900]
            schema_text = build_table_schema_text(schema)
            by_doc[doc_id].append({
                **common,
                "chunk_id": f"{table_id}__schema",
                "element_id": f"{table_id}__schema",
                "chunk_type": "table_schema",
                "column_names": list(schema.get("column_names") or []),
                "section_title": str(schema.get("section_title") or ""),
                "text": schema_text,
            })
        except Exception as exc:
            print(f"schema skip {table_id}: {exc}", flush=True)
    finally:
        for document in documents.values():
            document.close()
    for doc in docs:
        doc_id = str(doc["doc_id"])
        write_jsonl(args.chunks_root / doc_id / "table_chunks.jsonl", by_doc.get(doc_id, []))
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps({"summary": {"tables": len(audit),
        "indexable_tables": sum(row["indexable"] for row in audit),
        "quarantined_tables": sum(not row["indexable"] for row in audit),
        "chunks": sum(len(rows) for rows in by_doc.values())}, "tables": audit}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json.loads(args.audit.read_text(encoding="utf-8"))["summary"], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-list", type=Path, default=Path("data/manifests/full_corpus_715.csv"))
    parser.add_argument("--pdf-manifest", type=Path, default=Path("data/manifests/full_corpus_715.csv"))
    parser.add_argument("--tables-root", type=Path, default=Path("data/processed/tables_v2"))
    parser.add_argument("--layout-root", type=Path, default=Path("data/processed/layout_json_merged"))
    parser.add_argument("--work-root", type=Path, default=Path("data/processed/precise_tables"))
    parser.add_argument("--chunks-root", type=Path, default=Path("data/processed/chunks_tables_precise"))
    parser.add_argument("--pipeline-manifest", type=Path, default=Path("data/processed/logs/full_corpus_715_precise_manifest.json"))
    parser.add_argument("--audit", type=Path, default=Path("data/processed/logs/full_corpus_715_precise_audit.json"))
    parser.add_argument("--mapping-registry", type=Path, default=Path("data/config/hancomeqn_maps/registry.json"))
    parser.add_argument("--collection-id", default="full_corpus_715_tables_precise_v1")
    parser.add_argument("--stages", default=DEFAULT_STAGES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tatr-threshold", type=float, default=0.5)
    parser.add_argument("--tatr-padding", type=int, default=20)
    parser.add_argument("--embedding-preset", default="e5-base")
    args = parser.parse_args()
    for name in ("doc_list", "pdf_manifest", "tables_root", "layout_root", "work_root", "chunks_root", "pipeline_manifest", "audit", "mapping_registry"):
        value = getattr(args, name)
        setattr(args, name, value if value.is_absolute() else ROOT / value)
    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    allowed = set(DEFAULT_STAGES.split(","))
    unknown = set(stages) - allowed
    if unknown:
        parser.error(f"Unknown stages: {sorted(unknown)}")
    docs = load_docs(args.doc_list)
    registry = load_registry(args.mapping_registry)
    if "preprocess" in stages:
        command = [sys.executable, "scripts/69_build_table_corpus.py", "--doc-list", str(args.doc_list),
                   "--manifest", str(args.pdf_manifest), "--skip-index"]
        if args.resume:
            command.append("--resume-completed")
        run(command)
    manifest = prepare(args, docs, registry) if "prepare" in stages else load_pipeline_manifest(args.pipeline_manifest)
    if "tatr" in stages: run_tatr(args, manifest)
    if "snap" in stages: run_snap(args, manifest)
    if "restore" in stages: run_restore(args, manifest)
    if "segment" in stages:
        run([sys.executable, "scripts/65_segment_kr7_compound_tables.py", "--manifest", str(args.pipeline_manifest),
             "--pilot-root", str(args.work_root), "--pdf-manifest", str(args.pdf_manifest),
             "--tables-root", str(args.tables_root), "--layout-root", str(args.layout_root),
             "--output", str(args.pipeline_manifest.with_name(args.pipeline_manifest.stem + "_regions.json"))])
    region_manifest = args.pipeline_manifest.with_name(args.pipeline_manifest.stem + "_regions.json")
    if "region-tatr" in stages:
        run([sys.executable, "scripts/66_run_kr7_region_tatr.py", "--manifest", str(region_manifest),
             "--pilot-root", str(args.work_root), "--reuse-existing"])
    if "chunks" in stages: build_chunks(args, docs, manifest)
    if "index" in stages:
        run([sys.executable, "scripts/10_build_unified_index.py", "--doc-list", str(args.doc_list),
             "--manifest", str(args.pdf_manifest), "--collection-id", args.collection_id,
             "--chunks-dir", str(ROOT / "data/processed/chunks"), "--table-chunks-dir", str(args.chunks_root),
             "--embedding-preset", args.embedding_preset, "--include-types", "table",
             "--structured-tables", "only", "--max-embedding-tokens", "420", "--embedding-overlap-tokens", "60"])


if __name__ == "__main__":
    main()
