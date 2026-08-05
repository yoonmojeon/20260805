"""Structured table extraction and table-aware chunk generation."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

TABLE_CAPTION_RE = re.compile(r"^(표\s*[\d\.]+|Table\s+\d+)", re.IGNORECASE)
SECTION_HEADER_RE = re.compile(r"^(\d{3,4})\.\s+|^제\s*\d+\s*[장절]")
MULTISPACE_SPLIT_RE = re.compile(r"\s{2,}|\t")
CIRCLE_MARKERS = frozenset({"○", "O", "o", "●", "◎", "-", "△", "—"})
TOC_DOT_LEADER_RE = re.compile(r"\.{3,}\s*\d+\s*$")
TOC_CHAPTER_RE = re.compile(r"^제\s*\d+\s*[장절]")
INSPECTION_HEADER_RE = re.compile(
    r"^(정기검사\s*구분|구역|제\s*\d+\s*차|정기검사|제\s*\d+\s*차\s*및|이후\s*정기검사)$"
)
AGE_RANGE_HEADER_RE = re.compile(r"선령")


def normalize_whitespace(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def vertical_gap_below(upper: list[float], lower: list[float]) -> float:
    return max(0.0, lower[1] - upper[3])


def find_section_title(
    elements: list[dict],
    table_bbox: list[float],
    *,
    pdf_reader: Any | None = None,
    page_number: int = 0,
    page_width: int = 0,
    page_height: int = 0,
) -> str:
    candidates: list[tuple[float, str]] = []
    for el in elements:
        el_type = str(el.get("type", "")).lower()
        if el_type not in ("section-header", "title", "text"):
            continue
        bbox = [float(v) for v in el.get("bbox", [0, 0, 0, 0])]
        if bbox[3] > table_bbox[1]:
            continue
        gap = table_bbox[1] - bbox[3]
        text = ""
        if pdf_reader is not None and getattr(pdf_reader, "available", False):
            text = normalize_whitespace(
                pdf_reader.extract_bbox_text(page_number, bbox, page_width, page_height)
            )
        if not text:
            continue
        first = text.split("\n", 1)[0].strip()
        if SECTION_HEADER_RE.match(first) or el_type in ("section-header", "title"):
            candidates.append((gap, first[:200]))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def find_caption(
    elements: list[dict],
    table_bbox: list[float],
    raw_text: str,
    *,
    pdf_reader: Any | None = None,
    page_number: int = 0,
    page_width: int = 0,
    page_height: int = 0,
) -> str:
    first_line = raw_text.split("\n", 1)[0].strip() if raw_text else ""
    if TABLE_CAPTION_RE.match(first_line):
        return first_line[:240]

    captions = [el for el in elements if str(el.get("type", "")).lower() == "caption"]
    best_gap = float("inf")
    best_text = ""
    for cap in captions:
        bbox = [float(v) for v in cap.get("bbox", [0, 0, 0, 0])]
        gap = abs(bbox[1] - table_bbox[3])
        if gap > page_height * 0.08 and bbox[1] < table_bbox[1]:
            gap = table_bbox[1] - bbox[3]
            if gap < 0 or gap > page_height * 0.06:
                continue
        text = ""
        if pdf_reader is not None and getattr(pdf_reader, "available", False):
            text = normalize_whitespace(
                pdf_reader.extract_bbox_text(page_number, bbox, page_width, page_height)
            )
        if text and gap < best_gap:
            best_gap = gap
            best_text = text.split("\n", 1)[0].strip()[:240]
    return best_text


def _escape_md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _rows_to_markdown(columns: list[str], rows: list[dict[str, str]]) -> str:
    if not columns:
        return ""
    header = "| " + " | ".join(_escape_md_cell(c) for c in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body_lines = []
    for row in rows:
        cells = [_escape_md_cell(str(row.get(col, "") or "")) for col in columns]
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body_lines])


def _looks_like_grid_line(line: str) -> bool:
    tokens = line.split()
    if not tokens:
        return False
    marker_count = sum(1 for t in tokens if t in CIRCLE_MARKERS or t.upper() in CIRCLE_MARKERS)
    return marker_count >= 2 or (marker_count >= 1 and len(tokens) >= 3)


def _split_columns(line: str) -> list[str]:
    parts = [p.strip() for p in MULTISPACE_SPLIT_RE.split(line) if p.strip()]
    if len(parts) >= 2:
        return parts
    tokens = line.split()
    if len(tokens) >= 2 and _looks_like_grid_line(line):
        return tokens
    return [line.strip()] if line.strip() else []


def _is_marker_token(token: str) -> bool:
    t = token.strip()
    if t in CIRCLE_MARKERS or t.upper() in CIRCLE_MARKERS:
        return True
    if re.match(r"^\d+\s*개$", t):
        return True
    if t in {"절반,", "절반", "최소한"}:
        return True
    return False


def _line_is_marker_row(line: str) -> bool:
    tokens = line.split()
    return bool(tokens) and all(_is_marker_token(t) for t in tokens)


def _is_inspection_header_line(line: str) -> bool:
    ln = line.strip()
    if not ln:
        return False
    if INSPECTION_HEADER_RE.match(ln):
        return True
    if len(ln) <= 18 and ("정기검사" in ln or ln in ("구역", "제4차 및")):
        return True
    return False


def _split_inspection_header_body(lines: list[str]) -> tuple[list[str], int]:
    """Split KR inspection-table header from body; avoid absorbing first data row."""
    header: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if _is_inspection_header_line(ln):
            header.append(ln)
            i += 1
            continue
        if _line_is_marker_row(ln):
            return header, i
        if i + 1 < len(lines) and _line_is_marker_row(lines[i + 1]):
            return header, i
        if i + 1 < len(lines) and any(_is_marker_token(t) for t in lines[i + 1].split()):
            return header, i
        if len(ln) > 24 and not _is_inspection_header_line(ln):
            return header, i
        header.append(ln)
        i += 1
    return header, i


def _header_to_columns(header_lines: list[str]) -> list[str]:
    if not header_lines:
        return []
    joined = " ".join(header_lines)
    if "정기검사" in joined:
        cols: list[str] = []
        buf: list[str] = []
        for ln in header_lines:
            if ln in ("구역", "정기검사 구분"):
                continue
            if not _is_inspection_header_line(ln) and len(ln) > 24:
                continue
            buf.append(ln)
            if "정기검사" in ln or "이후" in ln:
                cols.append(" ".join(buf).strip())
                buf = []
        if buf:
            cols.append(" ".join(buf).strip())
        if cols:
            return ["구역", *cols] if not cols[0].startswith("구역") else cols
    if any(AGE_RANGE_HEADER_RE.search(ln) for ln in header_lines):
        return [ln for ln in header_lines if AGE_RANGE_HEADER_RE.search(ln)][:4]
    return [ln for ln in header_lines if _is_inspection_header_line(ln) or AGE_RANGE_HEADER_RE.search(ln)]


def _collect_marker_values(lines: list[str], start: int, count: int) -> tuple[list[str], int]:
    values: list[str] = []
    i = start
    while i < len(lines) and len(values) < count:
        ln = lines[i]
        if _line_is_marker_row(ln):
            values.extend(ln.split())
            i += 1
        elif any(_is_marker_token(t) for t in ln.split()) and len(ln.split()) <= count - len(values) + 2:
            values.extend(ln.split())
            i += 1
        else:
            break
    return values, i


def _parse_inspection_grid_table(lines: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    header_lines, body_start = _split_inspection_header_body(lines)
    columns = _header_to_columns(header_lines)
    if not columns or columns[0] != "구역":
        columns = ["구역", "제1차 정기검사", "제2차 정기검사", "제3차 정기검사", "제4차 및 이후 정기검사"]

    rows: list[dict[str, str]] = []
    i = body_start
    n_value_cols = max(len(columns) - 1, 1)
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("(비고"):
            break
        if _line_is_marker_row(ln):
            if rows:
                vals, i = _collect_marker_values(lines, i, n_value_cols)
                prev = rows[-1]
                empty_cols = [c for c in columns[1:] if not str(prev.get(c, "")).strip()]
                for val, col in zip(vals, empty_cols):
                    prev[col] = val
            else:
                i += 1
            continue

        label_parts = [ln]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if nxt.startswith("(비고"):
                break
            if _line_is_marker_row(nxt) or any(_is_marker_token(t) for t in nxt.split()):
                break
            if _is_inspection_header_line(nxt):
                break
            label_parts.append(nxt)
            i += 1

        values, i = _collect_marker_values(lines, i, n_value_cols)
        row: dict[str, str] = {columns[0]: " ".join(label_parts).strip()}
        for j, col in enumerate(columns[1:], start=0):
            row[col] = values[j] if j < len(values) else ""
        if row.get(columns[0]):
            rows.append(row)

    return columns, rows


def _is_age_range_row_label(line: str) -> bool:
    s = line.strip()
    if s in {"평형수탱크", "화물창", "화물탱크6)", "화물탱크"}:
        return True
    return bool(re.match(r"^화물탱크\d*\)?$", s))


def _partition_age_range_values(body: list[str], n_val: int) -> list[str]:
    """Split value lines into n_val cells; merge wrapped lines within a cell."""
    if not body:
        return [""] * n_val
    cleaned = [ln for ln in body if ln and not ln.startswith("(비고")]
    if not cleaned:
        return [""] * n_val

    if all(len(ln) < 60 for ln in cleaned):
        deduped: list[str] = []
        for ln in cleaned:
            if deduped and ln == deduped[-1]:
                continue
            deduped.append(ln)
        has_numbered = any(re.match(r"^\d+\.\s", ln) for ln in cleaned)
        if not has_numbered and len(deduped) <= n_val + 1:
            while len(deduped) < n_val:
                deduped.append(deduped[-1] if deduped else "")
            return deduped[:n_val]

    cells: list[list[str]] = [[] for _ in range(n_val)]
    idx = 0
    for ln in cleaned:
        if idx == 0 and ln in {"-", "—", "－"}:
            cells[0] = [ln]
            idx = 1
            continue
        if idx > 0 and re.match(r"^\d+\.\s", ln) and cells[idx]:
            if idx + 1 < n_val:
                idx += 1
        cells[idx].append(ln)
    out = [" ".join(part).strip() for part in cells]
    while len(out) < n_val:
        out.append("")
    return out[:n_val]


def _parse_age_range_table(lines: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    header_count = 0
    for ln in lines[:6]:
        if AGE_RANGE_HEADER_RE.search(ln):
            header_count += 1
        else:
            break
    if header_count < 2:
        return [], []

    columns = ["항목", *lines[:header_count]]
    n_val = header_count
    label_idxs = [
        i
        for i in range(header_count, len(lines))
        if _is_age_range_row_label(lines[i]) and not lines[i].startswith("(비고")
    ]
    if not label_idxs:
        return [], []

    rows: list[dict[str, str]] = []
    for li, start in enumerate(label_idxs):
        end = label_idxs[li + 1] if li + 1 < len(label_idxs) else len(lines)
        label = lines[start]
        body = lines[start + 1 : end]
        values = _partition_age_range_values(body, n_val)
        row = {columns[0]: label}
        for j, col in enumerate(columns[1:], start=0):
            row[col] = values[j] if j < len(values) else ""
        if label.strip():
            rows.append(row)
    return columns, rows


def _parse_marker_grid_table(lines: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    if any("정기검사" in ln for ln in lines[:12]) and any("○" in ln or "O" in ln for ln in lines):
        cols, rows = _parse_inspection_grid_table(lines)
        if rows:
            return cols, rows

    if any(AGE_RANGE_HEADER_RE.search(ln) for ln in lines[:4]):
        cols, rows = _parse_age_range_table(lines)
        if rows:
            return cols, rows

    body_start = 0
    for i, line in enumerate(lines):
        if _line_is_marker_row(line) or (
            len(line.split()) >= 2 and any(_is_marker_token(t) for t in line.split())
        ):
            body_start = i
            break
    header_lines = [ln for ln in lines[:body_start] if _is_inspection_header_line(ln) or AGE_RANGE_HEADER_RE.search(ln)]
    columns = _header_to_columns(header_lines if header_lines else lines[:4])
    if not columns:
        columns = ["구역", "제1차 정기검사", "제2차 정기검사", "제3차 정기검사", "제4차 및 이후 정기검사"]

    rows: list[dict[str, str]] = []
    i = body_start if body_start else len(header_lines)
    n_value_cols = max(len(columns) - 1, 1)
    while i < len(lines):
        line = lines[i]
        if _line_is_marker_row(line):
            if rows:
                vals, i = _collect_marker_values(lines, i, n_value_cols)
                prev = rows[-1]
                empty_cols = [c for c in columns[1:] if not str(prev.get(c, "")).strip()]
                for val, col in zip(vals, empty_cols):
                    prev[col] = val
            else:
                i += 1
            continue

        label_parts = [line]
        i += 1
        while i < len(lines) and not _line_is_marker_row(lines[i]) and not any(
            _is_marker_token(t) for t in lines[i].split()
        ):
            label_parts.append(lines[i])
            i += 1

        values, i = _collect_marker_values(lines, i, n_value_cols)
        row: dict[str, str] = {columns[0]: " ".join(label_parts).strip()}
        for j, col in enumerate(columns[1:], start=0):
            row[col] = values[j] if j < len(values) else ""
        if row.get(columns[0]):
            rows.append(row)

    return columns, rows


def parse_table_structure(raw_text: str) -> tuple[list[str], list[dict[str, str]], str]:
    """Best-effort parse of PDF clip text into columns, row dicts, markdown."""
    lines = [ln.strip() for ln in normalize_whitespace(raw_text).splitlines() if ln.strip()]
    if not lines:
        return [], [], ""

    while lines and lines[-1].startswith("(비고"):
        lines.pop()

    if any("○" in ln or "O" in ln for ln in lines) or any("선령" in ln for ln in lines[:4]):
        columns, rows = _parse_marker_grid_table(lines)
        if rows:
            md = _rows_to_markdown(columns, rows)
            return columns, rows, md

    grid_lines: list[list[str]] = []
    header_lines: list[str] = []
    for line in lines:
        cols = _split_columns(line)
        if _looks_like_grid_line(line) and len(cols) >= 2:
            grid_lines.append(cols)
        elif not grid_lines:
            header_lines.append(line)
        else:
            # continuation / merged cell row label
            if grid_lines:
                prev = grid_lines[-1]
                if len(cols) == 1:
                    prev[0] = f"{prev[0]} {cols[0]}".strip()
                else:
                    grid_lines.append(cols)

    if not grid_lines:
        columns = ["content"]
        rows = [{"content": ln} for ln in lines]
        md = _rows_to_markdown(columns, rows)
        return columns, rows, md

    max_cols = max(len(r) for r in grid_lines)
    norm_grid = [r + [""] * (max_cols - len(r)) for r in grid_lines]

    if header_lines:
        header_cols = _split_columns(header_lines[-1])
        if len(header_cols) >= max_cols:
            columns = header_cols[:max_cols]
        elif len(header_lines) >= 2:
            h1 = _split_columns(header_lines[0])
            h2 = _split_columns(header_lines[1]) if len(header_lines) > 1 else []
            if h1 and h2 and len(h1) + len(h2) >= max_cols:
                columns = (h1 + h2)[:max_cols]
            else:
                columns = [f"col_{i+1}" for i in range(max_cols)]
                if h1:
                    columns[0] = h1[0]
        else:
            columns = [f"col_{i+1}" for i in range(max_cols)]
            if header_cols:
                columns[0] = header_cols[0]
    else:
        columns = [f"col_{i+1}" for i in range(max_cols)]
        if norm_grid:
            first = norm_grid[0]
            if not _looks_like_grid_line(" ".join(first)):
                columns = first
                norm_grid = norm_grid[1:]

    while len(columns) < max_cols:
        columns.append(f"col_{len(columns)+1}")
    columns = columns[:max_cols]

    rows: list[dict[str, str]] = []
    for cells in norm_grid:
        row = {columns[i]: cells[i] if i < len(cells) else "" for i in range(len(columns))}
        if any(v.strip() for v in row.values()):
            rows.append(row)

    md = _rows_to_markdown(columns, rows)
    return columns, rows, md


def infer_table_caption(caption: str, raw_text: str, section_title: str) -> str:
    cap = (caption or "").strip()
    if cap:
        return cap
    for line in (raw_text or "").splitlines()[:6]:
        line = line.strip()
        if TABLE_CAPTION_RE.match(line):
            return line[:240]
    blob = f"{section_title} {raw_text[:600]}"
    if "화학성분" in blob:
        return "화학성분"
    if "열처리" in blob and "로트" in blob:
        return "열처리 및 충격시험 로트"
    if "기계적 성질" in blob or ("항복" in blob and "인장" in blob):
        return "기계적 성질"
    if "강재의 종류" in blob:
        return "강재의 종류"
    return cap


def _table_topic_hints(caption: str, section_title: str, raw_text: str) -> list[str]:
    blob = f"{caption} {section_title} {raw_text[:1000]}"
    hints: list[str] = []
    if "화학성분" in blob or re.search(r"\b[CPMS]\b", blob):
        hints.append("화학성분")
    if any(t in blob for t in ("항복", "인장강도", "연신율", "충격시험")):
        hints.append("기계적성질")
    if "열처리" in blob and "로트" in blob:
        hints.append("열처리로트")
    if "정기검사" in blob or "선령" in blob:
        hints.append("정기검사")
    if "강재의 종류" in blob or "재료기호" in blob and "적용두께" in blob:
        hints.append("강재종류")
    return hints


def summarize_table(
    *,
    doc_id: str,
    page: int,
    caption: str,
    section_title: str,
    columns: list[str],
    rows: list[dict[str, str]],
    source_file: str = "",
    raw_text: str = "",
) -> str:
    doc_label = source_file or doc_id
    parts = [
        f"문서 {doc_label} {page}페이지의 표",
    ]
    if caption:
        parts.append(f"({caption})")
    if section_title:
        parts.append(f"— {section_title} 절")
    hints = _table_topic_hints(caption, section_title, raw_text)
    if hints:
        parts.append(f"주제: {', '.join(hints)}")
    if columns:
        parts.append(f"주요 열: {', '.join(columns[:8])}")
    parts.append(f"총 {len(rows)}행.")
    sample_vals: list[str] = []
    for row in rows[:3]:
        for col in columns[:4]:
            val = str(row.get(col, "")).strip()
            if val and val not in CIRCLE_MARKERS:
                sample_vals.append(f"{col}={val}")
                break
    if sample_vals:
        parts.append("예시: " + "; ".join(sample_vals[:4]))
    return " ".join(parts)


def row_to_kv_block(
    *,
    doc_id: str,
    page: int,
    table_id: str,
    caption: str,
    columns: list[str],
    row: dict[str, str],
    row_index: int,
    source_file: str = "",
) -> str:
    doc_label = source_file or doc_id
    lines = [
        "[Table Row KV]",
        f"doc={doc_label}",
        f"page={page}",
        f"table_id={table_id}",
        f"row_index={row_index}",
    ]
    if caption:
        lines.append(f"caption={caption}")
    for col in columns:
        val = str(row.get(col, "")).strip()
        if val:
            lines.append(f"{col}={val}")
    return "\n".join(lines)


def row_to_natural_language(
    *,
    doc_id: str,
    page: int,
    caption: str,
    section_title: str,
    columns: list[str],
    row: dict[str, str],
    row_index: int,
    source_file: str = "",
) -> str:
    doc_label = source_file or doc_id
    table_ref = caption or f"표 (p.{page})"
    kv_parts: list[str] = []
    for col in columns:
        val = str(row.get(col, "")).strip()
        if not val:
            continue
        kv_parts.append(f"{col}={val}")
    body = ", ".join(kv_parts[:8])
    section_part = f" ({section_title})" if section_title else ""
    return (
        f"문서 {doc_label} p.{page}의 {table_ref}{section_part}에 따르면, "
        f"행 {row_index + 1}: {body}."
    )


def build_table_row_text(
    *,
    doc_id: str,
    page: int,
    table_id: str,
    caption: str,
    section_title: str,
    columns: list[str],
    row: dict[str, str],
    row_index: int,
    source_file: str = "",
) -> str:
    nl = row_to_natural_language(
        doc_id=doc_id,
        page=page,
        caption=caption,
        section_title=section_title,
        columns=columns,
        row=row,
        row_index=row_index,
        source_file=source_file,
    )
    kv = row_to_kv_block(
        doc_id=doc_id,
        page=page,
        table_id=table_id,
        caption=caption,
        columns=columns,
        row=row,
        row_index=row_index,
        source_file=source_file,
    )
    return f"{nl}\n\n{kv}"


def _looks_like_inspection_data_table(raw_text: str) -> bool:
    lines = [ln.strip() for ln in normalize_whitespace(raw_text).splitlines() if ln.strip()]
    if not lines:
        return False
    has_age_cols = sum(1 for ln in lines[:6] if AGE_RANGE_HEADER_RE.search(ln)) >= 2
    has_inspection = any("정기검사" in ln for ln in lines[:12])
    has_tank_row = any(
        kw in raw_text for kw in ("평형수탱크", "화물창", "화물탱크", "연료유탱크", "빌지저장")
    )
    has_markers = any("○" in ln or _line_is_marker_row(ln) for ln in lines)
    if has_age_cols and has_tank_row:
        return True
    if has_inspection and has_tank_row and has_markers:
        return True
    return False


def is_toc_pseudo_table(raw_text: str, *, confidence: float = 1.0, caption: str = "") -> tuple[bool, str]:
    """Return (is_pseudo, reason)."""
    lines = [ln.strip() for ln in normalize_whitespace(raw_text).splitlines() if ln.strip()]
    if not lines:
        return True, "empty"

    if _looks_like_inspection_data_table(raw_text):
        return False, ""

    toc_hits = sum(1 for ln in lines if TOC_DOT_LEADER_RE.search(ln) or TOC_CHAPTER_RE.match(ln))
    if len(lines) >= 5 and toc_hits / len(lines) >= 0.55:
        return True, "toc_dot_leader"

    if len(lines) >= 8 and toc_hits >= 4:
        return True, "toc_chapter_list"

    dot_chars = sum(ln.count("·") + ln.count(".") for ln in lines[:20])
    if dot_chars > len(lines) * 8 and toc_hits >= 2:
        return True, "dot_leader_dense"

    if not caption and confidence < 0.52 and len(lines) > 22 and toc_hits >= 1:
        return True, "low_confidence_no_caption"

    return False, ""


def detect_parse_quality_issues(
    *,
    columns: list[str],
    rows: list[dict[str, str]],
    raw_text: str,
) -> list[dict[str, str]]:
    """Detect header/data merge and column shift issues."""
    issues: list[dict[str, str]] = []
    lines = [ln for ln in normalize_whitespace(raw_text).splitlines() if ln.strip()]

    for col in columns:
        if len(col) > 60:
            issues.append({"type": "header_data_merge", "detail": f"long column name: {col[:80]}"})
        if any(kw in col for kw in ("화물창", "탱크", "선박")) and "정기검사" not in col and "선령" not in col:
            issues.append({"type": "header_data_merge", "detail": f"row label in columns: {col[:80]}"})

    if columns and rows:
        expected = len(columns)
        for idx, row in enumerate(rows):
            non_empty = sum(1 for c in columns if str(row.get(c, "")).strip())
            if non_empty == 0:
                issues.append({"type": "empty_row", "detail": f"row {idx} empty"})
            if expected >= 4 and non_empty == 1 and idx < len(lines) // 3:
                issues.append({"type": "value_shift", "detail": f"row {idx} only row-key populated"})

    if lines and any("정기검사" in ln for ln in lines[:10]):
        first_body = next((ln for ln in lines if "화물창" in ln or "이중저" in ln), "")
        if first_body:
            col_joined = " ".join(columns)
            if first_body in col_joined and not any(first_body[:12] in str(r.get(columns[0], "")) for r in rows):
                issues.append({"type": "header_data_merge", "detail": "first data row absorbed into header"})

    return issues


def pseudo_table_penalty(issues: list[dict], is_pseudo: bool) -> float:
    if is_pseudo:
        return 0.15
    if any(i["type"] == "header_data_merge" for i in issues):
        return 0.08
    return 1.0


def build_table_record(
    *,
    doc_id: str,
    source_file: str,
    page: int,
    table_id: str,
    element_id: str,
    bbox: list[int],
    raw_text: str,
    caption: str,
    section_title: str,
    extraction_method: str = "layout_bbox_pdf_clip",
    confidence: float = 1.0,
    is_pseudo: bool = False,
    pseudo_reason: str = "",
    parse_issues: list[dict] | None = None,
    table_quality: float = 1.0,
) -> dict:
    columns, rows, markdown = parse_table_structure(raw_text)
    caption = infer_table_caption(caption, raw_text, section_title)
    issues = parse_issues if parse_issues is not None else detect_parse_quality_issues(
        columns=columns, rows=rows, raw_text=raw_text
    )
    return {
        "doc_id": doc_id,
        "source_file": source_file,
        "page": page,
        "table_id": table_id,
        "element_id": element_id,
        "caption": caption,
        "section_title": section_title,
        "bbox": bbox,
        "extraction_method": extraction_method,
        "confidence": confidence,
        "is_pseudo_table": is_pseudo,
        "pseudo_reason": pseudo_reason,
        "parse_issues": issues,
        "table_quality": table_quality,
        "raw_table_text": raw_text,
        "markdown_table": markdown,
        "table_json": {"columns": columns, "rows": rows},
        "row_count": len(rows),
        "column_names": columns,
    }


def build_table_chunks(table: dict, *, source: str = "", file_name: str = "") -> list[dict]:
    if table.get("is_pseudo_table"):
        return []
    doc_id = str(table["doc_id"])
    page = int(table["page"])
    table_id = str(table["table_id"])
    caption = str(table.get("caption") or "")
    section_title = str(table.get("section_title") or "")
    columns: list[str] = list(table.get("column_names") or [])
    rows: list[dict] = list((table.get("table_json") or {}).get("rows") or [])
    source_file = str(table.get("source_file") or file_name or doc_id)
    markdown = str(table.get("markdown_table") or "")
    raw_text = str(table.get("raw_table_text") or "")

    base_meta = {
        "doc_id": doc_id,
        "page": page,
        "page_number": page,
        "table_id": table_id,
        "caption": caption,
        "section_title": section_title,
        "column_names": columns,
        "source": source,
        "file_name": file_name or source_file,
        "element_type": "table",
        "element_id": str(table.get("element_id") or table_id),
        "bbox": table.get("bbox") or [],
    }

    chunks: list[dict] = []

    summary_text = summarize_table(
        doc_id=doc_id,
        page=page,
        caption=caption,
        section_title=section_title,
        columns=columns,
        rows=rows,
        source_file=source_file,
        raw_text=raw_text,
    )
    chunks.append(
        {
            **base_meta,
            "chunk_id": f"{table_id}__summary",
            "chunk_type": "table_summary",
            "row_index": None,
            "text": summary_text,
        }
    )

    from table_schema_lib import build_table_schema_chunk

    schema_chunk = build_table_schema_chunk(
        {
            **table,
            "doc_id": doc_id,
            "page": page,
            "table_id": table_id,
            "caption": caption,
            "section_title": section_title,
            "column_names": columns,
            "table_json": {"columns": columns, "rows": rows},
            "markdown_table": markdown,
            "raw_table_text": raw_text,
            "parse_issues": table.get("parse_issues") or [],
            "table_quality": table.get("table_quality", 1.0),
            "element_id": table.get("element_id") or table_id,
            "bbox": table.get("bbox") or [],
            "source_file": source_file,
        },
        source=source,
        file_name=file_name or source_file,
    )
    chunks.append(schema_chunk)

    md_body = markdown or raw_text
    md_text = f"{caption}\n\n{md_body}".strip() if caption else md_body
    chunks.append(
        {
            **base_meta,
            "chunk_id": f"{table_id}__markdown",
            "chunk_type": "table_markdown",
            "row_index": None,
            "text": md_text,
        }
    )

    for idx, row in enumerate(rows):
        row_text = build_table_row_text(
            doc_id=doc_id,
            page=page,
            table_id=table_id,
            caption=caption,
            section_title=section_title,
            columns=columns,
            row=row,
            row_index=idx,
            source_file=source_file,
        )
        chunks.append(
            {
                **base_meta,
                "chunk_id": f"{table_id}__row_{idx:03d}",
                "chunk_type": "table_row",
                "row_index": idx,
                "text": row_text,
                "row_data": row,
            }
        )

    return chunks


def load_table_chunks(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_tables(path: Path) -> list[dict]:
    return load_table_chunks(path)
