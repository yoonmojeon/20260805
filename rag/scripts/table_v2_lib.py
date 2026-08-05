"""Coordinate-aware table reconstruction for the KR table QA v2 corpus.

The v1 parser linearized PDF clip text before inferring columns.  That loses the
horizontal spacing needed to distinguish cells.  This module reconstructs the
grid from PDF drawing/text coordinates with PyMuPDF and emits parent/row records
suited to hierarchical table retrieval.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

import fitz

from table_schema_lib import build_table_schema_chunk


PARSER_VERSION = "kr-table-v2-pymupdf-grid-1"
CAPTION_RE = re.compile(r"^(?:표\s*[A-Za-z0-9.-]+|Table\s+[A-Za-z0-9.-]+)\s*", re.IGNORECASE)
GENERIC_COLUMN_RE = re.compile(r"^(?:col(?:umn)?[_ ]?\d+|content|value|열[_ ]?\d+)$", re.IGNORECASE)
HEADER_TERMS = (
    "항목", "구분", "종류", "기호", "재료", "성분", "두께", "치수", "단위", "검사",
    "선령", "강도", "하중", "조건", "압력", "온도", "길이", "비고", "번호", "명칭",
)
ROW_HEADER_TERMS = ("항목", "구분", "종류", "기호", "재료", "구역", "명칭", "번호", "등급")
MARKERS = frozenset({"○", "O", "o", "△", "×", "X", "-", "—", "–"})


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).replace("\u00ad", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def private_use_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = sum(1 for ch in text if unicodedata.category(ch) == "Co")
    return bad / max(1, len(text))


def embedding_safe_text(text: str) -> str:
    """Replace unmapped formula-font glyphs while preserving the audited raw grid."""
    out: list[str] = []
    in_private_run = False
    for ch in str(text or ""):
        if unicodedata.category(ch) == "Co":
            if not in_private_run:
                out.append(" [수식기호] ")
            in_private_run = True
        else:
            out.append(ch)
            in_private_run = False
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _trim_grid(grid: list[list[str]]) -> list[list[str]]:
    while grid and not any(grid[0]):
        grid.pop(0)
    while grid and not any(grid[-1]):
        grid.pop()
    if not grid:
        return []
    width = max(len(row) for row in grid)
    grid = [row + [""] * (width - len(row)) for row in grid]
    keep = [i for i in range(width) if any(row[i] for row in grid)]
    return [[row[i] for i in keep] for row in grid]


def _dedupe_columns(columns: list[str]) -> list[str]:
    out: list[str] = []
    counts: dict[str, int] = {}
    for i, raw in enumerate(columns):
        col = clean_cell(raw) or f"열_{i + 1}"
        key = col.casefold()
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > 1:
            col = f"{col}_{counts[key]}"
        out.append(col)
    return out


def _looks_like_header_continuation(row: list[str], first: list[str]) -> bool:
    blob = " ".join(row)
    first_blob = " ".join(first)
    nonempty = sum(bool(v) for v in row)
    if not nonempty:
        return False
    header_hits = sum(term in blob for term in HEADER_TERMS)
    first_has_gaps = sum(not v for v in first) >= max(1, len(first) // 3)
    numeric_cells = sum(bool(re.fullmatch(r"[-+]?\d+(?:[.,]\d+)?(?:\s*%|\s*mm)?", v)) for v in row if v)
    marker_cells = sum(v in MARKERS for v in row if v)
    if numeric_cells + marker_cells >= max(2, nonempty // 2):
        return False
    return bool(header_hits >= 2 or (first_has_gaps and header_hits >= 1 and any(t in first_blob for t in HEADER_TERMS)))


def _header_depth(grid: list[list[str]]) -> int:
    if len(grid) < 2:
        return 1
    depth = 1
    for idx in range(1, min(3, len(grid) - 1)):
        if _looks_like_header_continuation(grid[idx], grid[0]):
            depth = idx + 1
        else:
            break
    return depth


def _flatten_headers(header_rows: list[list[str]]) -> list[str]:
    width = len(header_rows[0])
    columns: list[str] = []
    previous_group = ""
    for col_idx in range(width):
        parts: list[str] = []
        for row in header_rows:
            value = clean_cell(row[col_idx])
            if value and value not in parts:
                parts.append(value)
        name = " / ".join(parts)
        if name:
            previous_group = name
        elif col_idx == 1 and columns and any(term in columns[0] for term in ROW_HEADER_TERMS):
            name = "세부항목"
        elif previous_group and col_idx > 0:
            name = f"{previous_group} 세부항목"
        elif col_idx == 0:
            name = "항목"
        else:
            name = f"열_{col_idx + 1}"

        # A merged top-left header often contains a group label and the actual row key.
        if col_idx == 0 and " " in name:
            tokens = name.split()
            meaningful = [t for t in tokens if t in ROW_HEADER_TERMS or any(k in t for k in ROW_HEADER_TERMS)]
            if meaningful:
                name = meaningful[-1]
        columns.append(name)
    return _dedupe_columns(columns)


def _is_repeated_header(row: list[str], columns: list[str]) -> bool:
    if not any(row):
        return True
    matches = 0
    for value, col in zip(row, columns):
        v = re.sub(r"\s+", "", value)
        c = re.sub(r"\s+", "", col.split(" / ")[-1])
        if v and c and (v == c or v in c or c in v):
            matches += 1
    return matches >= max(2, len(columns) // 2)


def normalize_grid(raw_grid: Iterable[Iterable[Any]]) -> tuple[list[str], list[dict[str, str]], int]:
    grid = _trim_grid([[clean_cell(v) for v in row] for row in raw_grid])
    if not grid:
        return [], [], 0
    depth = _header_depth(grid)
    columns = _flatten_headers(grid[:depth])
    rows: list[dict[str, str]] = []
    last_context = [""] * len(columns)
    context_cols = {
        i for i, col in enumerate(columns[: min(3, len(columns))])
        if i == 0 or any(term in col for term in ROW_HEADER_TERMS)
    }
    for raw in grid[depth:]:
        if _is_repeated_header(raw, columns):
            continue
        resolved = list(raw)
        for idx in context_cols:
            if resolved[idx]:
                last_context[idx] = resolved[idx]
            elif any(resolved[j] for j in range(idx + 1, len(resolved))) and last_context[idx]:
                resolved[idx] = last_context[idx]
        row = {columns[i]: resolved[i] for i in range(len(columns))}
        if any(row.values()):
            rows.append(row)
    return columns, rows, depth


def rows_to_markdown(columns: list[str], rows: list[dict[str, str]]) -> str:
    if not columns:
        return ""
    def esc(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")
    lines = [
        "| " + " | ".join(esc(c) for c in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(esc(str(row.get(c, ""))) for c in columns) + " |")
    return "\n".join(lines)


def _caption_from_table(table: Any) -> str:
    header = getattr(table, "header", None)
    names = list(getattr(header, "names", []) or [])
    for name in names:
        value = clean_cell(name)
        if CAPTION_RE.match(value):
            return value[:300]
    return ""


def _caption_above(page: fitz.Page, rect: fitz.Rect) -> str:
    band = fitz.Rect(max(0, rect.x0 - 4), max(0, rect.y0 - 55), min(page.rect.x1, rect.x1 + 4), rect.y0)
    lines = [clean_cell(line) for line in page.get_text("text", clip=band).splitlines()]
    for line in reversed(lines):
        if CAPTION_RE.match(line):
            return line[:300]
    return ""


@dataclass
class GridResult:
    columns: list[str]
    rows: list[dict[str, str]]
    caption: str
    strategy: str
    header_depth: int
    raw_grid: list[list[str]]


def extract_coordinate_grid(page: fitz.Page, rect: fitz.Rect) -> GridResult | None:
    attempts = (
        ("lines_strict", {"strategy": "lines_strict"}),
    )
    best: tuple[float, GridResult] | None = None
    for strategy, kwargs in attempts:
        try:
            found = page.find_tables(clip=rect, **kwargs)
        except Exception:
            continue
        for table in found.tables:
            raw_grid = [[clean_cell(v) for v in row] for row in table.extract()]
            columns, rows, depth = normalize_grid(raw_grid)
            if not columns or not rows:
                continue
            nonempty = sum(bool(v) for row in rows for v in row.values())
            density = nonempty / max(1, len(rows) * len(columns))
            score = min(1.0, len(rows) / 8) + min(0.5, len(columns) / 10) + density
            if strategy == "text":
                score -= 0.55
            result = GridResult(
                columns=columns,
                rows=rows,
                caption=_caption_from_table(table) or _caption_above(page, rect),
                strategy=strategy,
                header_depth=depth,
                raw_grid=raw_grid,
            )
            if best is None or score > best[0]:
                best = (score, result)
        if best is not None:
            break
    return best[1] if best else None


def assess_quality(
    *,
    columns: list[str],
    rows: list[dict[str, str]],
    caption: str,
    strategy: str,
) -> tuple[float, str, list[str], dict[str, float]]:
    reasons: list[str] = []
    score = 1.0
    all_text = " ".join(columns + [str(v) for row in rows for v in row.values()])
    pua = private_use_ratio(all_text)
    generic_ratio = sum(bool(GENERIC_COLUMN_RE.match(c)) for c in columns) / max(1, len(columns))
    nonempty = sum(bool(v) for row in rows for v in row.values())
    density = nonempty / max(1, len(rows) * len(columns))
    long_header_ratio = sum(len(c) > 90 for c in columns) / max(1, len(columns))

    if strategy == "text":
        score -= 0.22
        reasons.append("text_strategy_fallback")
    if not caption:
        score -= 0.06
        reasons.append("missing_explicit_caption")
    if len(columns) < 2:
        score -= 0.36
        reasons.append("single_column")
    if len(rows) < 1:
        score -= 0.8
        reasons.append("no_data_rows")
    if generic_ratio:
        score -= min(0.34, generic_ratio * 0.38)
        reasons.append(f"generic_columns:{generic_ratio:.2f}")
    if density < 0.34:
        score -= min(0.3, (0.34 - density) * 1.2)
        reasons.append(f"sparse_cells:{density:.2f}")
    if long_header_ratio:
        score -= min(0.28, long_header_ratio * 0.35)
        reasons.append(f"long_headers:{long_header_ratio:.2f}")
    if pua > 0:
        if pua <= 0.06:
            score -= min(0.12, 0.04 + pua)
        elif pua <= 0.20:
            score -= 0.20
        else:
            score -= 0.28
        reasons.append(f"private_use_chars:{pua:.4f}")
    if len(columns) > 24:
        score -= 0.12
        reasons.append("excessive_column_count")
    score = round(max(0.0, min(1.0, score)), 3)
    if score >= 0.70 and len(columns) >= 2 and rows:
        status = "pass"
    elif score >= 0.58 and len(columns) >= 2 and rows and pua < 0.01:
        status = "review"
    else:
        status = "reject"
    metrics = {
        "cell_density": round(density, 4),
        "generic_column_ratio": round(generic_ratio, 4),
        "private_use_ratio": round(pua, 6),
        "long_header_ratio": round(long_header_ratio, 4),
    }
    return score, status, reasons, metrics


def _row_label(row: dict[str, str], columns: list[str]) -> str:
    labels = [str(row.get(c, "")).strip() for c in columns[: min(3, len(columns))]]
    return " / ".join(v for v in labels if v)[:280]


def _cell_fact_lines(caption: str, row: dict[str, str], columns: list[str]) -> list[str]:
    label = _row_label(row, columns)
    facts: list[str] = []
    for col in columns:
        value = str(row.get(col, "")).strip()
        if not value:
            continue
        if label and value not in label:
            facts.append(f"{label}의 {col} 값은 {value}")
        else:
            facts.append(f"{col} 값은 {value}")
    return facts


def build_v2_chunks(table: dict, *, source: str, file_name: str) -> list[dict]:
    if table.get("quality_status") != "pass":
        return []
    doc_id = str(table["doc_id"])
    page = int(table["page"])
    table_id = str(table["table_id"])
    caption = embedding_safe_text(str(table.get("caption") or ""))
    section = embedding_safe_text(str(table.get("section_title") or ""))
    raw_columns = list(table.get("column_names") or [])
    raw_rows = list((table.get("table_json") or {}).get("rows") or [])
    columns = _dedupe_columns([embedding_safe_text(c) for c in raw_columns])
    rows = [
        {columns[i]: embedding_safe_text(row.get(raw_columns[i], "")) for i in range(len(columns))}
        for row in raw_rows
    ]
    markdown = rows_to_markdown(columns, rows)
    quality_score = float(table.get("quality_score") or 0.0)
    quality_status = str(table.get("quality_status") or "")
    base = {
        "doc_id": doc_id,
        "source": source,
        "file_name": file_name,
        "page": page,
        "page_number": page,
        "table_id": table_id,
        "caption": caption,
        "section_title": section,
        "column_names": columns,
        "element_type": "table",
        "element_id": str(table.get("element_id") or table_id),
        "bbox": table.get("bbox") or [],
        "parser_version": PARSER_VERSION,
        "quality_score": quality_score,
        "quality_status": quality_status,
    }
    chunks: list[dict] = []
    schema = build_table_schema_chunk(
        {
            **table,
            "caption": caption,
            "section_title": section,
            "column_names": columns,
            "table_json": {"columns": columns, "rows": rows},
            "markdown_table": markdown,
            "table_quality": quality_score,
            "raw_table_text": markdown,
            "parse_issues": [{"type": r, "detail": r} for r in table.get("quality_reasons") or []],
        },
        source=source,
        file_name=file_name,
    )
    schema.update(base)
    schema["chunk_id"] = f"{table_id}__schema"
    chunks.append(schema)

    summary_parts = [
        "[표 개요]",
        f"문서: {file_name}",
        f"페이지: {page}",
        f"표 ID: {table_id}",
    ]
    if caption:
        summary_parts.append(f"표 제목: {caption}")
    if section:
        summary_parts.append(f"절 제목: {section}")
    summary_parts.extend([
        f"열: {', '.join(columns)}",
        f"행 수: {len(rows)}",
        "대표 행: " + " ; ".join(_row_label(r, columns) for r in rows[:5] if _row_label(r, columns)),
    ])
    chunks.append({
        **base,
        "chunk_id": f"{table_id}__summary",
        "chunk_type": "table_summary",
        "row_index": None,
        "text": "\n".join(v for v in summary_parts if not v.endswith(": ")),
    })

    # Markdown is a parent context, not the primary exact-cell representation.
    markdown_preview = rows_to_markdown(columns, rows[:40])
    chunks.append({
        **base,
        "chunk_id": f"{table_id}__markdown",
        "chunk_type": "table_markdown",
        "row_index": None,
        "text": "\n".join(v for v in (caption, markdown_preview) if v),
        "markdown_truncated": len(rows) > 40,
    })

    for idx, row in enumerate(rows):
        facts = _cell_fact_lines(caption, row, columns)
        text_lines = [
            "[표 행 및 셀 사실]",
            f"문서: {file_name}",
            f"페이지: {page}",
            f"표 ID: {table_id}",
        ]
        if caption:
            text_lines.append(f"표 제목: {caption}")
        if section:
            text_lines.append(f"절 제목: {section}")
        text_lines.extend([
            f"행 번호: {idx + 1}",
            f"행 기준: {_row_label(row, columns)}",
            "셀: " + " | ".join(f"{c}={row.get(c, '')}" for c in columns if row.get(c, "")),
        ])
        if facts:
            text_lines.append("사실: " + " ; ".join(facts))
        chunks.append({
            **base,
            "chunk_id": f"{table_id}__row_{idx:03d}",
            "chunk_type": "table_row",
            "row_index": idx,
            "row_key": _row_label(row, columns),
            "row_data": row,
            "text": "\n".join(text_lines),
        })
    return chunks


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
