"""Deterministic Markdown table rebuild from precise table_row chunks."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_CELL_RE = re.compile(r"열\s*(\d+)\s*=\s*([^|]*)")
_ROW_ID_RE = re.compile(r":ROW(\d+)\b", re.IGNORECASE)


def strip_embed_header(text: str) -> str:
    """Drop [table_row] header lines; keep the 열N=… body line(s)."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    body_lines = [ln for ln in lines if "열" in ln and "=" in ln]
    if body_lines:
        return body_lines[-1]
    return lines[-1] if lines else ""


def parse_row_cells(text: str) -> dict[int, str]:
    body = strip_embed_header(text)
    cells: dict[int, str] = {}
    for match in _CELL_RE.finditer(body):
        col = int(match.group(1))
        raw = match.group(2).strip()
        # "Header path: value" → keep value for data rows when path repeats
        if ": " in raw and not raw.startswith("("):
            head, _, tail = raw.partition(": ")
            # Prefer human value after the last path segment
            value = tail.strip() if tail.strip() else head.strip()
        else:
            value = raw
        cells[col] = value or ""
    return cells


def row_sort_key(meta: dict[str, Any], text: str, index: int) -> tuple[int, int]:
    eid = str(meta.get("element_id") or meta.get("chunk_id") or "")
    m = _ROW_ID_RE.search(eid)
    if m:
        return (0, int(m.group(1)))
    # Fallback: preserve fetch order
    return (1, index)


def display_cell_value(raw: str) -> str:
    """For markdown display: prefer 'Header: value' → value when header-like."""
    s = (raw or "").strip()
    if ": " in s:
        left, _, right = s.partition(": ")
        # Header-only rows often have no second part usage; keep full if short
        if right.strip():
            return right.strip()
        return left.strip()
    return s


def cells_for_markdown(text: str, *, as_header: bool) -> dict[int, str]:
    body = strip_embed_header(text)
    cells: dict[int, str] = {}
    for match in _CELL_RE.finditer(body):
        col = int(match.group(1))
        raw = match.group(2).strip()
        if as_header:
            # Header row: use label before ": " if present
            cells[col] = raw.split(": ", 1)[0].strip() or raw
        else:
            cells[col] = display_cell_value(raw)
    return cells


def _escape_md_cell(value: str) -> str:
    return (value or "").replace("|", "\\|").replace("\n", " ").strip()


def rows_to_markdown(
    row_docs: list[str],
    row_metas: list[dict[str, Any]],
    *,
    highlight_question: str = "",
    max_rows: int = 40,
) -> tuple[str, list[int]]:
    """Build a Markdown table from ordered table_row texts.

    Returns (markdown, highlighted_row_indices_0based_in_output).
    """
    paired = list(zip(row_docs, row_metas))
    paired.sort(key=lambda item: row_sort_key(item[1], item[0], 0))
    # Re-sort with index
    indexed = list(enumerate(paired))
    indexed.sort(key=lambda it: row_sort_key(it[1][1], it[1][0], it[0]))
    ordered = [p for _, p in indexed]

    if not ordered:
        return "", []

    # Detect header: first row often has no ": value" pattern across cells
    first_cells_raw = parse_row_cells(ordered[0][0])
    first_is_header = bool(first_cells_raw) and sum(
        1 for v in first_cells_raw.values() if ": " in (v or "")
    ) < max(1, len(first_cells_raw) // 2)

    if first_is_header:
        headers = cells_for_markdown(ordered[0][0], as_header=True)
        data_pairs = ordered[1:]
    else:
        # Synthesize Col1..ColN from max column index
        all_cols: set[int] = set()
        for doc, _ in ordered:
            all_cols.update(parse_row_cells(doc).keys())
        headers = {c: f"열{c}" for c in sorted(all_cols)}
        data_pairs = ordered

    cols = sorted(headers.keys()) or [1]
    header_vals = [_escape_md_cell(headers.get(c, f"열{c}")) for c in cols]
    lines = [
        "| " + " | ".join(header_vals) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]

    q = (highlight_question or "").strip()
    q_tokens = [t for t in re.split(r"\s+", q) if len(t) >= 2][:8]
    highlighted: list[int] = []
    truncated = False
    out_i = 0
    for doc, _meta in data_pairs:
        if out_i >= max_rows:
            truncated = True
            break
        cells = cells_for_markdown(doc, as_header=False)
        vals = []
        row_blob = " ".join(cells.get(c, "") for c in cols)
        hit = bool(q_tokens) and any(tok in row_blob or tok in doc for tok in q_tokens)
        for c in cols:
            v = _escape_md_cell(cells.get(c, ""))
            if hit and v:
                v = f"**{v}**"
            vals.append(v or "")
        lines.append("| " + " | ".join(vals) + " |")
        if hit:
            highlighted.append(out_i)
        out_i += 1

    if truncated:
        lines.append("")
        lines.append(f"_… 표시 한도 {max_rows}행. 전체는 table_id 메타데이터로 재조회 가능._")

    return "\n".join(lines), highlighted


def format_related_tables_section(
    tables: list[dict[str, Any]],
    *,
    question: str = "",
) -> str:
    """tables: [{table_id, file_name, page, caption, markdown, crop_path, highlight_rows}]"""
    if not tables:
        return ""
    parts = ["### 관련 표"]
    for i, t in enumerate(tables, 1):
        src = t.get("file_name") or t.get("doc_id") or ""
        page = t.get("page")
        caption = t.get("caption") or t.get("table_id") or ""
        cite = f"출처: {src}"
        if page not in (None, "", 0):
            cite += f", p.{page}"
        if caption:
            cite += f", Table `{caption}`" if caption != t.get("table_id") else f", `{t.get('table_id')}`"
        parts.append(f"\n**표 {i}.** {cite}")
        md = (t.get("markdown") or "").strip()
        if md:
            parts.append("")
            parts.append(md)
        hits = t.get("highlight_rows") or []
        if hits:
            parts.append("")
            parts.append(f"검색된 핵심 행(표시 기준 0-index): {hits}")
        crop = t.get("crop_path") or ""
        if crop and Path(crop).is_file():
            parts.append("")
            parts.append(f"_원본 표 crop: `{crop}`_")
    return "\n".join(parts).strip()


def extract_table_ids_from_chunks(chunks: list[Any], *, limit: int = 3) -> list[str]:
    import re

    seen: list[str] = []
    tid_re = re.compile(
        r"(?:표:\s*|table_id[=:\s]+)?([a-z0-9_]+_p\d+_t\d+)",
        re.I,
    )
    chunk_id_re = re.compile(r"^([a-z0-9_]+_p\d+_t\d+)(?::|$)", re.I)

    def _add(tid: str) -> bool:
        tid = (tid or "").strip()
        if not tid or tid in seen:
            return False
        seen.append(tid)
        return len(seen) >= limit

    for chunk in chunks:
        tid = str(getattr(chunk, "table_id", "") or "")
        if not tid and isinstance(chunk, dict):
            tid = str(chunk.get("table_id") or "")
        meta = getattr(chunk, "metadata", None) or {}
        if not tid and isinstance(meta, dict):
            tid = str(meta.get("table_id") or "")
        if tid and _add(tid):
            break

        chunk_id = str(getattr(chunk, "chunk_id", "") or "")
        if not chunk_id and isinstance(chunk, dict):
            chunk_id = str(chunk.get("chunk_id") or "")
        match = chunk_id_re.match(chunk_id)
        if match and _add(match.group(1)):
            break

        text = str(getattr(chunk, "text", "") or "")
        if not text and isinstance(chunk, dict):
            text = str(chunk.get("text") or "")
        match = tid_re.search(text)
        if match and _add(match.group(1)):
            break
    return seen
