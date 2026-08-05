"""Build table_schema chunks from parsed table records."""
from __future__ import annotations

import json
import re
from typing import Any

from table_extract_lib import _table_topic_hints, infer_table_caption
from table_normalize_lib import (
    ENTITY_ALIASES,
    MATERIAL_GRADE_RE,
    expand_entity_aliases,
    extract_units,
    normalize_compact,
    normalize_material_grade,
    normalize_token,
)

FOOTNOTE_RE = re.compile(r"^\(비고|^\(\d+\)|footnote", re.IGNORECASE)
GENERIC_SINGLE_COL = frozenset({"content", "value", "col", "column"})
TABLE_CAPTION_RE = re.compile(r"^표\s*[\d.]+\s*", re.IGNORECASE)

CHEMICAL_ELEMENT_SYMBOLS = (
    "C", "Si", "Mn", "P", "S", "Cu", "Cr", "Ni", "Mo", "Al", "Nb", "V", "Ti", "N"
)
CHEMICAL_KOREAN_TO_SYMBOL = {
    "탄소": "C",
    "규소": "Si",
    "망간": "Mn",
    "인": "P",
    "황": "S",
    "구리": "Cu",
    "크롬": "Cr",
    "니켈": "Ni",
    "몰리브덴": "Mo",
    "알루미늄": "Al",
    "질소": "N",
}
STEEL_CHEMISTRY_CORE = ("C", "Mn", "P", "S")
STRUCTURAL_COLUMN_MARKERS = (
    ("종류", ("종류", "종\n류")),
    ("재료기호", ("재료기호", "재료\n기호", "재료 기호")),
    ("두께(mm)", ("두께", "(mm)")),
    ("탈산방법", ("탈산", "탈산방법")),
)
CANONICAL_TOPICS = {
    "chemical_composition": ("화학성분", "화학 성분", "chemical_composition", "함량", "성분"),
    "mechanical_property": ("기계적성질", "기계적 성질", "mechanical_property", "항복", "인장강도", "연신율"),
    "inspection": ("정기검사", "inspection", "reporting", "검사차수", "검사 보고"),
    "test_material": ("시험재료", "시험재", "test_material", "용접봉", "강재의 종류"),
    "dimension_or_tolerance": ("치수", "dimension", "두께", "mm", "허용차"),
    "note_lookup": ("비고", "note_lookup", "각주", "footnote"),
    "lot_treatment": ("열처리", "로트", "lot_treatment"),
}


def _normalize_column_name(col: str) -> str:
    c = normalize_token(col)
    for canon, forms in ENTITY_ALIASES.items():
        if normalize_compact(c) == normalize_compact(canon):
            return canon
        for form in forms:
            if normalize_compact(c) == normalize_compact(form):
                return canon
    return normalize_compact(c) if len(c) <= 6 and c.isascii() else c


def _columns_are_weak(columns: list[str]) -> bool:
    if not columns:
        return True
    if len(columns) == 1 and columns[0].lower() in GENERIC_SINGLE_COL:
        return True
    if all(len(str(c)) <= 2 for c in columns) and len(columns) < 4:
        return True
    return False


def _dedupe_preserve(items: list[str]) -> list[str]:
    return list(dict.fromkeys(t for t in items if t))


def _detect_chemical_elements(blob: str) -> list[str]:
    if not blob:
        return []
    found: list[str] = []
    if re.search(r"C\s*\+\s*Mn", blob, re.IGNORECASE):
        found.extend(["C", "Mn"])
    for sym in CHEMICAL_ELEMENT_SYMBOLS:
        if sym == "N" and re.search(r"N\s*/\s*mm", blob, re.IGNORECASE):
            continue
        if sym in {"C", "S", "P", "N", "V"}:
            if re.search(rf"(?<![A-Za-z]){sym}(?![A-Za-z0-9])", blob):
                found.append(sym)
        elif re.search(rf"\b{re.escape(sym)}\b", blob):
            found.append(sym)
    for ko, sym in CHEMICAL_KOREAN_TO_SYMBOL.items():
        if ko in blob:
            found.append(sym)
    return _dedupe_preserve(found)


def _detect_structural_columns(blob: str) -> list[str]:
    cols: list[str] = []
    for label, markers in STRUCTURAL_COLUMN_MARKERS:
        if any(m.replace("\n", "") in blob.replace("\n", "") for m in markers):
            cols.append(label)
    return cols


def _detect_grade_entities(blob: str) -> list[str]:
    grades: list[str] = []
    for m in MATERIAL_GRADE_RE.finditer(blob or ""):
        grades.append(normalize_material_grade(m.group(0)))
    for m in re.finditer(r"\b(AH|DH|EH|FH)\s+(\d{1,2})\b", blob or "", re.IGNORECASE):
        grades.append(f"{m.group(1).upper()}{m.group(2)}")
    return _dedupe_preserve(grades)


def _is_meaningful_row_entity(text: str) -> bool:
    r = normalize_token(text)
    if not r or len(r) > 40:
        return False
    if FOOTNOTE_RE.match(r):
        return False
    if re.search(r"\b(AH|DH|EH|FH)\s*\d", r, re.IGNORECASE):
        return True
    if r in {"연강", "고장력강", "평형수탱크", "화물창", "화물탱크", "이중저탱크", "디프탱크", "피크탱크", "기관실"}:
        return True
    if any(k in r for k in ("탱크", "화물", "구역", "검사", "선령")):
        return True
    if re.fullmatch(r"[A-Z]", r):
        return True
    if len(r) <= 3 and not re.search(r"[가-힣]{2,}", r):
        return False
    if any(noise in r for noise in ("킬드", "세미킬드", "세립킬드", "이하", "이상", "재료", "기호", "두께")):
        return False
    return len(r) >= 4


def _detect_domain_row_entities(blob: str) -> list[str]:
    rows: list[str] = []
    domain_terms = (
        "연강",
        "고장력강",
        "평형수탱크",
        "화물창",
        "화물탱크",
        "이중저탱크",
        "디프탱크",
        "피크탱크",
        "기관실",
    )
    for term in domain_terms:
        if term in blob:
            rows.append(term)
    return rows


def _infer_canonical_table_topics(
    blob: str,
    columns: list[str],
    caption: str = "",
    section: str = "",
) -> list[str]:
    text = f"{caption} {section} {blob} {' '.join(columns)}"
    topics: list[str] = []
    elems = _detect_chemical_elements(text)
    chem_hits = len(elems)
    if chem_hits >= 3 or "화학성분" in text or "화학 성분" in text:
        topics.extend(["화학성분", "chemical_composition"])
    header_blob = text[:500]
    if any(t in header_blob for t in ("항복", "인장강도", "연신율", "기계적 성질")):
        topics.extend(["기계적성질", "mechanical_property"])
    elif any(t in text for t in ("항복", "인장강도", "연신율")) and chem_hits < 3:
        topics.extend(["기계적성질", "mechanical_property"])
    if any(t in text for t in ("제1차 정기검사", "제2차 정기검사", "정기검사", "reporting")):
        topics.extend(["정기검사", "inspection"])
    if any(t in text for t in ("시험재료", "시험재", "용접봉", "강재의 종류")):
        topics.extend(["시험재료", "test_material"])
    if any(t in text for t in ("허용차", "치수")) or (
        "mm" in text.lower() and "두께" in text and chem_hits < 2
    ):
        topics.extend(["치수", "dimension_or_tolerance"])
    if "(비고)" in text or "각주" in text or "footnote" in text.lower():
        topics.append("note_lookup")
    if "열처리" in text and "로트" in text:
        topics.extend(["열처리", "lot_treatment"])
    if "선령" in text:
        topics.append("선령")
    return _dedupe_preserve(topics)


def _infer_caption_from_blob(blob: str, section: str = "") -> str:
    for line in (blob or "").splitlines()[:8]:
        line = line.strip()
        if TABLE_CAPTION_RE.match(line):
            return line[:240]
    text = f"{section} {blob[:1200]}"
    if "화학성분" in text or (
        _detect_chemical_elements(text) and _detect_grade_entities(text)
    ):
        return "화학성분"
    if "열처리" in text and "로트" in text:
        return "열처리 및 충격시험 로트"
    if "기계적 성질" in text or ("항복" in text and "인장" in text):
        return "기계적 성질"
    if "강재의 종류" in text:
        return "강재의 종류"
    if "정기검사" in text:
        return "정기검사"
    return ""


def _merge_columns(primary: list[str], fallback: list[str]) -> list[str]:
    if not fallback:
        return primary
    if _columns_are_weak(primary):
        return _dedupe_preserve(fallback)
    merged = list(primary)
    for col in fallback:
        ncol = _normalize_column_name(col)
        if ncol not in {_normalize_column_name(c) for c in merged}:
            merged.append(col)
    return merged


def _apply_schema_enrichment(
    schema: dict[str, Any],
    *,
    raw_text: str = "",
    markdown: str = "",
    columns: list[str] | None = None,
    rows: list[dict] | None = None,
) -> dict[str, Any]:
    """Fill weak schema fields from raw table text / markdown (complex merged-cell tables)."""
    blob = "\n".join(
        part
        for part in (
            schema.get("caption") or "",
            schema.get("section_title") or "",
            raw_text,
            markdown,
            " ".join(schema.get("column_names") or []),
            " ".join(schema.get("row_entities") or []),
        )
        if part
    )
    if not schema.get("caption"):
        inferred = _infer_caption_from_blob(blob, str(schema.get("section_title") or ""))
        if inferred:
            schema["caption"] = inferred
            schema["table_title"] = inferred

    structural = _detect_structural_columns(blob)
    chem_elems = _detect_chemical_elements(blob)
    header_blob = blob[:600]
    is_mechanical = any(t in header_blob for t in ("항복", "인장강도", "연신율", "기계적 성질", "N/mm"))
    if (
        len(chem_elems) >= 2
        and not is_mechanical
        and any(t in blob for t in ("연강", "고장력강", "탈산", "0.035"))
    ):
        for core in STEEL_CHEMISTRY_CORE:
            if core not in chem_elems:
                chem_elems.append(core)
    enriched_cols = _dedupe_preserve(structural + chem_elems)
    base_cols = list(columns or schema.get("column_names") or [])
    schema["column_names"] = _merge_columns(base_cols, enriched_cols)
    schema["normalized_column_names"] = [_normalize_column_name(c) for c in schema["column_names"]]

    grade_rows = _detect_grade_entities(blob)
    domain_rows = _detect_domain_row_entities(blob)
    parsed_rows: list[str] = []
    if rows and columns and not _columns_are_weak(columns):
        parsed_rows = _extract_row_entities(columns, rows)
    elif not _columns_are_weak(list(schema.get("column_names") or [])):
        parsed_rows = list(schema.get("row_entities") or [])
    merged_rows = _dedupe_preserve(domain_rows + grade_rows + parsed_rows)
    merged_rows = [
        r
        for r in merged_rows
        if _is_meaningful_row_entity(r)
    ]
    if merged_rows:
        schema["row_entities"] = merged_rows[:80]
        schema["normalized_row_entities"] = _dedupe_preserve(
            normalize_material_grade(e) if re.search(r"\b(AH|DH|EH|FH)\s*\d", e, re.I) else normalize_compact(e)
            for e in merged_rows
            if e
        )[:80]

    topics = list(schema.get("table_topics") or [])
    topics = _dedupe_preserve(
        topics
        + _infer_canonical_table_topics(
            blob,
            schema.get("column_names") or [],
            str(schema.get("caption") or ""),
            str(schema.get("section_title") or ""),
        )
        + _table_topic_hints(
            str(schema.get("caption") or ""),
            str(schema.get("section_title") or ""),
            raw_text[:1000],
        )
    )
    schema["table_topics"] = topics

    if not schema.get("units"):
        schema["units"] = extract_units(blob[:800])
    if not schema.get("notes"):
        schema["notes"] = _extract_notes(rows or [], schema.get("column_names") or [])
        if not schema["notes"] and "(비고)" in blob:
            for line in blob.splitlines():
                if line.strip().startswith("(비고)"):
                    schema["notes"] = [line.strip()[:200]]
                    break
    return schema


def _extract_row_entities(columns: list[str], rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    label_col = columns[0] if columns else ""
    entities: list[str] = []
    for row in rows:
        if not label_col:
            break
        val = normalize_token(str(row.get(label_col, "")))
        if not val or len(val) > 80:
            continue
        if FOOTNOTE_RE.match(val):
            continue
        entities.append(val)
        mg = normalize_material_grade(val)
        if mg != normalize_compact(val):
            entities.append(mg)
    # Deduplicate preserving order, cap for embedding size
    return list(dict.fromkeys(entities))[:80]


def _extract_notes(rows: list[dict[str, str]], columns: list[str]) -> list[str]:
    notes: list[str] = []
    for row in rows:
        for col in columns or list(row.keys()):
            val = str(row.get(col, "")).strip()
            if val and FOOTNOTE_RE.match(val):
                notes.append(val[:200])
    return notes[:12]


def _compute_parse_quality(
    *,
    columns: list[str],
    rows: list[dict[str, str]],
    parse_issues: list[dict],
    table_quality: float,
) -> float:
    score = float(table_quality or 1.0)
    if len(columns) == 1 and columns[0].lower() in GENERIC_SINGLE_COL:
        score *= 0.45
    if rows and columns:
        single_val_rows = sum(
            1 for r in rows if sum(1 for c in columns if str(r.get(c, "")).strip()) == 1
        )
        if single_val_rows > len(rows) * 0.6:
            score *= 0.55
    for issue in parse_issues or []:
        if issue.get("type") in ("header_data_merge", "value_shift"):
            score *= 0.75
    return round(max(0.05, min(1.0, score)), 3)


def build_table_schema_data(table: dict) -> dict[str, Any]:
    columns: list[str] = list(table.get("column_names") or [])
    rows: list[dict] = list((table.get("table_json") or {}).get("rows") or [])
    raw_text = str(table.get("raw_table_text") or "")
    caption = str(table.get("caption") or "")
    section_title = str(table.get("section_title") or "")
    if not caption:
        caption = infer_table_caption(caption, raw_text, section_title)

    normalized_columns = [_normalize_column_name(c) for c in columns]
    row_entities = _extract_row_entities(columns, rows)
    normalized_row_entities = list(
        dict.fromkeys(
            normalize_material_grade(e) if re.search(r"\b(AH|DH|EH|FH)\s*\d", e, re.I) else normalize_compact(e)
            for e in row_entities
            if e
        )
    )[:80]

    all_cell_text = " ".join(
        str(row.get(c, "")) for row in rows[:40] for c in columns[:12]
    )
    units = extract_units(all_cell_text + " " + raw_text[:500])
    notes = _extract_notes(rows, columns)
    topics = _table_topic_hints(caption, section_title, raw_text)
    if not topics and len(columns) > 3 and any("정기검사" in c for c in columns):
        topics.append("정기검사")
    if not topics and any(
        normalize_compact(c) in {"C", "S", "P", "MN", "SI"} for c in normalized_columns
    ):
        topics.append("화학성분")

    parse_issues = list(table.get("parse_issues") or [])
    parse_quality = _compute_parse_quality(
        columns=columns,
        rows=rows,
        parse_issues=parse_issues,
        table_quality=float(table.get("table_quality") or 1.0),
    )

    schema = {
        "doc_id": str(table.get("doc_id") or ""),
        "source_file": str(table.get("source_file") or ""),
        "page": int(table.get("page") or 0),
        "table_id": str(table.get("table_id") or ""),
        "caption": caption,
        "table_title": caption,
        "section_title": section_title,
        "column_names": columns,
        "normalized_column_names": normalized_columns,
        "row_entities": row_entities,
        "normalized_row_entities": normalized_row_entities,
        "units": units,
        "notes": notes,
        "table_topics": topics,
        "parse_quality": parse_quality,
        "row_count": len(rows),
        "column_count": len(columns),
    }
    markdown = str(table.get("markdown_table") or "")
    schema = _apply_schema_enrichment(
        schema,
        raw_text=raw_text,
        markdown=markdown,
        columns=columns,
        rows=rows,
    )
    if _columns_are_weak(columns) or parse_quality < 0.65:
        schema["parse_quality"] = min(parse_quality, 0.55 if _columns_are_weak(columns) else parse_quality)
    return schema


def build_table_schema_text(schema: dict[str, Any]) -> str:
    lines = [
        f"table_schema table_id={schema.get('table_id')}",
        f"doc_id={schema.get('doc_id')} page={schema.get('page')} file={schema.get('source_file')}",
    ]
    if schema.get("caption"):
        lines.append(f"caption: {schema['caption']}")
    if schema.get("section_title"):
        lines.append(f"section: {schema['section_title']}")
    if schema.get("table_topics"):
        lines.append(f"topics: {', '.join(schema['table_topics'])}")
    if schema.get("column_names"):
        lines.append(f"columns: {', '.join(str(c) for c in schema['column_names'][:16])}")
    if schema.get("normalized_column_names"):
        lines.append(f"normalized_columns: {', '.join(str(c) for c in schema['normalized_column_names'][:16])}")
    if schema.get("row_entities"):
        sample = schema["row_entities"][:24]
        lines.append(f"row_entities: {', '.join(sample)}")
    if schema.get("normalized_row_entities"):
        ns = schema["normalized_row_entities"][:24]
        lines.append(f"normalized_row_entities: {', '.join(ns)}")
    if schema.get("units"):
        lines.append(f"units: {', '.join(schema['units'])}")
    if schema.get("notes"):
        lines.append(f"notes: {' | '.join(schema['notes'][:4])}")
    lines.append(f"parse_quality: {schema.get('parse_quality', 1.0)}")
    lines.append(f"dimensions: {schema.get('row_count', 0)} rows x {schema.get('column_count', 0)} cols")
    raw_snippet = str(schema.get("_raw_snippet") or "")
    if raw_snippet:
        lines.append(f"raw_snippet: {raw_snippet[:900]}")
    return "\n".join(lines)


def _enrich_schema_from_meta(schema: dict[str, Any], meta: dict, document: str) -> dict[str, Any]:
    meta = meta or {}
    if not schema.get("caption"):
        schema["caption"] = str(meta.get("caption") or "")
    if not schema.get("section_title"):
        schema["section_title"] = str(meta.get("section_title") or "")
    if not schema.get("column_names") and meta.get("column_names"):
        cols = str(meta["column_names"]).split(",")
        schema["column_names"] = [c.strip() for c in cols if c.strip()]
        schema["normalized_column_names"] = [_normalize_column_name(c) for c in schema["column_names"]]
    if not schema.get("table_topics"):
        schema["table_topics"] = _table_topic_hints(
            schema.get("caption") or "",
            schema.get("section_title") or "",
            document or "",
        )
    cap = str(schema.get("caption") or "")
    if "화학" in cap and "화학성분" not in schema["table_topics"]:
        schema["table_topics"].append("화학성분")
    if "정기검사" in cap or "reporting" in (document or "").lower():
        if "정기검사" not in schema["table_topics"]:
            schema["table_topics"].append("정기검사")
    return schema


def parse_schema_from_document(document: str, meta: dict | None = None) -> dict[str, Any]:
    """Reconstruct schema fields from indexed document text (+ optional metadata)."""
    meta = meta or {}
    schema: dict[str, Any] = {
        "table_id": str(meta.get("table_id") or ""),
        "caption": str(meta.get("caption") or ""),
        "section_title": str(meta.get("section_title") or ""),
        "column_names": [],
        "normalized_column_names": [],
        "row_entities": [],
        "normalized_row_entities": [],
        "table_topics": [],
        "units": [],
        "parse_quality": 1.0,
        "_raw_snippet": "",
    }
    if meta.get("column_names"):
        cols = str(meta["column_names"]).split(",")
        schema["column_names"] = [c.strip() for c in cols if c.strip()]
        schema["normalized_column_names"] = [_normalize_column_name(c) for c in schema["column_names"]]

    for line in (document or "").splitlines():
        lower = line.lower()
        if line.startswith("topics:"):
            schema["table_topics"] = [t.strip() for t in line.split(":", 1)[1].split(",") if t.strip()]
        elif line.startswith("columns:"):
            schema["column_names"] = [t.strip() for t in line.split(":", 1)[1].split(",") if t.strip()]
        elif line.startswith("normalized_columns:"):
            schema["normalized_column_names"] = [
                t.strip() for t in line.split(":", 1)[1].split(",") if t.strip()
            ]
        elif line.startswith("row_entities:"):
            schema["row_entities"] = [t.strip() for t in line.split(":", 1)[1].split(",") if t.strip()]
        elif line.startswith("normalized_row_entities:"):
            schema["normalized_row_entities"] = [
                t.strip() for t in line.split(":", 1)[1].split(",") if t.strip()
            ]
        elif line.startswith("units:"):
            schema["units"] = [t.strip() for t in line.split(":", 1)[1].split(",") if t.strip()]
        elif line.startswith("parse_quality:"):
            try:
                schema["parse_quality"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif lower.startswith("caption:"):
            schema["caption"] = line.split(":", 1)[1].strip()
        elif line.startswith("raw_snippet:"):
            schema["_raw_snippet"] = line.split(":", 1)[1].strip()
    schema = _enrich_schema_from_meta(schema, meta or {}, document)
    raw_blob = schema.get("_raw_snippet") or document
    if _columns_are_weak(schema.get("column_names") or []) or not schema.get("row_entities"):
        schema = _apply_schema_enrichment(schema, raw_text=raw_blob, markdown="")
    return schema


def build_table_schema_chunk(table: dict, *, source: str = "", file_name: str = "") -> dict:
    schema = build_table_schema_data(table)
    raw_text = str(table.get("raw_table_text") or "")
    if _columns_are_weak(list(table.get("column_names") or [])) or float(schema.get("parse_quality") or 1) < 0.65:
        schema["_raw_snippet"] = raw_text[:900]
    text = build_table_schema_text(schema)
    doc_id = str(table.get("doc_id") or "")
    page = int(table.get("page") or 0)
    table_id = str(table.get("table_id") or "")
    return {
        "doc_id": doc_id,
        "page": page,
        "page_number": page,
        "table_id": table_id,
        "caption": schema.get("caption") or "",
        "section_title": schema.get("section_title") or "",
        "column_names": schema.get("column_names") or [],
        "source": source,
        "file_name": file_name or str(table.get("source_file") or ""),
        "element_type": "table",
        "element_id": str(table.get("element_id") or table_id),
        "bbox": table.get("bbox") or [],
        "chunk_id": f"{table_id}__schema",
        "chunk_type": "table_schema",
        "row_index": None,
        "text": text,
        "schema_json": schema,
    }
