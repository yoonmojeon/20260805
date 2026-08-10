"""Create verified VLM augmentations for complex tables in a small KR pilot.

The VLM is an extractor, not the retrieval embedding model: it reads a rendered
table crop and emits formulas, variable definitions, and table references as
strict JSON.  Existing coordinate-derived table rows remain the authoritative
source for numeric cell values.  Results are written separately so they can be
audited before building the pilot index.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import fitz

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from embedding_policy import validate_embedding_model
from pdf_io import resolve_pdf_path

DEFAULT_MODEL = "CohereLabs/aya-vision-8b"
FORMULA_RE = re.compile(r"(?:[=√≤≥]|\b(?:min|max|sin|cos)\b|[A-Za-z]\s*[₀-₉′']\s*\^?)")

EXTRACTION_PROMPT = """You are extracting a Korean maritime-rule table for a retrieval system.
Read only visible content. Do not infer missing values. Return one JSON object, with no Markdown.
Schema:
{
  "caption": "string", "headers": ["string"],
  "formulas": [{"display": "string", "normalized": "ASCII-safe formula or empty", "condition": "string", "unit": "string", "variables": ["string"]}],
  "definitions": [{"symbol": "string", "meaning": "string", "unit": "string"}],
  "references": [{"source_symbol": "string", "target": "table/clause reference", "relation": "lookup|definition|continuation"}],
  "warnings": ["uncertain or unreadable items"]
}
Preserve Korean text, inequalities, subscripts, roots, ranges, and units exactly when readable.
"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def layout_size(layout_root: Path, doc_id: str, page_no: int) -> tuple[float, float]:
    payload = json.loads((layout_root / doc_id / f"page_{page_no:04d}.json").read_text(encoding="utf-8"))
    return float(payload["width"]), float(payload["height"])


def render_crop(pdf_path: Path, bbox: list[float], layout_dims: tuple[float, float], page_no: int, out_path: Path) -> None:
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_no - 1]
        lx, ly = layout_dims
        sx, sy = page.rect.width / lx, page.rect.height / ly
        x0, y0, x1, y1 = bbox
        clip = fitz.Rect(max(0, x0 * sx - 3), max(0, y0 * sy - 3), min(page.rect.width, x1 * sx + 3), min(page.rect.height, y1 * sy + 3))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), clip=clip, alpha=False).save(out_path)
    finally:
        doc.close()


def complex_table(table: dict[str, Any]) -> bool:
    raw = " ".join((str(table.get(k, "")) for k in ("caption", "raw_grid", "markdown_table", "raw_table_text")))
    return table.get("quality_status") != "pass" or bool(FORMULA_RE.search(raw))


def parse_json_response(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("VLM response contains no JSON object")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("VLM response is not an object")
    return value


def load_vlm(model_name: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, AyaVisionForConditionalGeneration, BitsAndBytesConfig

    validate_embedding_model(model_name)
    try:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    except Exception as exc:
        raise RuntimeError("4-bit VLM loading requires bitsandbytes. Install it before running without --dry-run.") from exc
    processor = AutoProcessor.from_pretrained(model_name)
    model_class = AyaVisionForConditionalGeneration if "aya-vision" in model_name.lower() else AutoModelForImageTextToText
    model = model_class.from_pretrained(
        model_name, quantization_config=quant, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return processor, model


def extract_with_vlm(processor, model, image_path: Path, *, max_new_tokens: int) -> dict[str, Any]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": EXTRACTION_PROMPT}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    answer = processor.decode(generated[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return parse_json_response(answer)


def build_extra_chunks(table: dict[str, Any], extracted: dict[str, Any], file_name: str) -> list[dict[str, Any]]:
    base = {
        "doc_id": table["doc_id"], "source": "KR", "file_name": file_name,
        "page": table["page"], "page_number": table["page"], "table_id": table["table_id"],
        "caption": str(extracted.get("caption") or table.get("caption") or ""),
        "section_title": str(table.get("section_title") or ""), "column_names": extracted.get("headers") or table.get("column_names") or [],
        "element_type": "table", "element_id": table.get("element_id") or table["table_id"],
        "parser_version": "aya-vision-8b-pilot-1", "quality_status": "vlm_unverified",
    }
    chunks: list[dict[str, Any]] = []
    for i, formula in enumerate(extracted.get("formulas") or []):
        if not isinstance(formula, dict) or not formula.get("display"):
            continue
        text = "\n".join(["[표 수식]", f"표 제목: {base['caption']}", f"표시식: {formula['display']}", f"정규식: {formula.get('normalized', '')}", f"적용 조건: {formula.get('condition', '')}", f"단위: {formula.get('unit', '')}", "변수: " + ", ".join(formula.get("variables") or [])])
        chunks.append({**base, "chunk_id": f"{base['table_id']}__vlm_formula_{i:02d}", "chunk_type": "table_formula", "text": text})
    for i, definition in enumerate(extracted.get("definitions") or []):
        if not isinstance(definition, dict) or not definition.get("symbol"):
            continue
        text = f"[표 변수 정의]\n표 제목: {base['caption']}\n기호: {definition['symbol']}\n의미: {definition.get('meaning', '')}\n단위: {definition.get('unit', '')}"
        chunks.append({**base, "chunk_id": f"{base['table_id']}__vlm_definition_{i:02d}", "chunk_type": "table_definition", "text": text})
    for i, reference in enumerate(extracted.get("references") or []):
        if not isinstance(reference, dict) or not reference.get("target"):
            continue
        text = f"[표 참조]\n표 제목: {base['caption']}\n기호: {reference.get('source_symbol', '')}\n대상: {reference['target']}\n관계: {reference.get('relation', '')}"
        chunks.append({**base, "chunk_id": f"{base['table_id']}__vlm_reference_{i:02d}", "chunk_type": "table_reference", "text": text})
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-list", type=Path, default=ROOT / "data/manifests/kr_vlm_pilot_789.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/pdf_manifest.csv")
    parser.add_argument("--tables-root", type=Path, default=ROOT / "data/processed/tables_v2")
    parser.add_argument("--layout-root", type=Path, default=ROOT / "data/processed/layout_json_merged")
    parser.add_argument("--out-root", type=Path, default=ROOT / "data/processed/vlm_table_pilot")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--table-id", action="append", dest="table_ids")
    parser.add_argument("--max-tables", type=int, default=4, help="Maximum selected complex tables per document.")
    parser.add_argument("--max-new-tokens", type=int, default=420)
    parser.add_argument("--dry-run", action="store_true", help="Render selected crops without loading the VLM.")
    args = parser.parse_args()

    with args.doc_list.open(encoding="utf-8-sig", newline="") as f:
        docs = list(csv.DictReader(f))
    wanted = set(args.table_ids or [])
    selected: list[tuple[dict[str, str], dict[str, Any]]] = []
    for doc in docs:
        doc_selected: list[tuple[dict[str, str], dict[str, Any]]] = []
        for table in load_jsonl(args.tables_root / doc["doc_id"] / "tables.jsonl"):
            if wanted and table["table_id"] not in wanted:
                continue
            if not wanted and not complex_table(table):
                continue
            doc_selected.append((doc, table))
        selected.extend(doc_selected if wanted else doc_selected[:args.max_tables])
    if not selected:
        raise SystemExit("No pilot tables selected")
    processor = model = None
    if not args.dry_run:
        processor, model = load_vlm(args.model)
    summary: list[dict[str, Any]] = []
    for doc, table in selected:
        table_dir = args.out_root / doc["doc_id"] / table["table_id"]
        image_path = table_dir / "crop.png"
        render_crop(resolve_pdf_path(doc["doc_id"], args.manifest, None, ROOT), list(table["bbox"]), layout_size(args.layout_root, doc["doc_id"], int(table["page"])), int(table["page"]), image_path)
        record: dict[str, Any] = {"table_id": table["table_id"], "doc_id": doc["doc_id"], "page": table["page"], "caption": table.get("caption", ""), "crop_path": str(image_path), "status": "crop_rendered"}
        if not args.dry_run:
            extracted = extract_with_vlm(processor, model, image_path, max_new_tokens=args.max_new_tokens)
            (table_dir / "extraction.json").write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")
            write_jsonl(table_dir / "extra_chunks.jsonl", build_extra_chunks(table, extracted, doc["file_name"]))
            record["status"] = "extracted_unverified"
            record["formula_count"] = len(extracted.get("formulas") or [])
            record["warning_count"] = len(extracted.get("warnings") or [])
        summary.append(record)
        print(f"{record['status']}: {record['table_id']}", flush=True)
    (args.out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
