"""HTML fragments for Gradio answer panel (Evidence Table + related table crops).

Mirrors MaritimeRAG Streamlit `15_rag_ui.py`:
- answer body with [n] citations
- Evidence Table (각주 → 문서·페이지·청크)
- table QA: PDF crop images via ``st.image(crop_path)`` equivalent
"""
from __future__ import annotations

import base64
import html
import mimetypes
from pathlib import Path
from typing import Any


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _crop_img_tag(path: str, *, max_width: str = "100%", max_bytes: int = 2_000_000) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    try:
        size = p.stat().st_size
        if size > max_bytes:
            return (
                f"<p class='evidence-note'>표 crop은 아래 파일 목록에서 확인 "
                f"(<code>{_esc(p.name)}</code>, {size // 1024}KB)</p>"
            )
        raw = p.read_bytes()
    except OSError:
        return f"<p class='evidence-note'>crop: <code>{_esc(path)}</code></p>"
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(raw).decode("ascii")
    return (
        f"<figure class='table-crop'>"
        f"<img src='data:{mime};base64,{b64}' alt='table crop' "
        f"style='max-width:{max_width};height:auto;border:1px solid #e6e8ec;"
        f"border-radius:6px;'/>"
        f"<figcaption>검색된 표/영역 원본 crop · {_esc(p.name)}</figcaption>"
        f"</figure>"
    )


def render_evidence_table_html(rows: list[dict[str, Any]] | None) -> str:
    """MaritimeRAG Evidence Table: citation_id / file / page / preview."""
    if not rows:
        return ""
    body_rows: list[str] = []
    for row in rows:
        cite = row.get("citation_id") or ""
        if cite and not str(cite).startswith("["):
            cite = f"[{cite}]"
        preview = str(row.get("chunk_preview") or row.get("text") or "")
        if len(preview) > 420:
            preview = preview[:417].rstrip() + "…"
        page = row.get("page")
        page_s = "" if page in (None, "") else str(page)
        body_rows.append(
            "<tr>"
            f"<td class='cite'>{_esc(cite)}</td>"
            f"<td>{_esc(row.get('file_name') or row.get('doc_id') or '')}</td>"
            f"<td class='page'>{_esc(page_s)}</td>"
            f"<td class='preview'>{_esc(preview)}</td>"
            "</tr>"
        )
    return (
        "<div class='evidence-block'>"
        "<div class='evidence-title'>Evidence Table "
        "<span class='evidence-hint'>답변의 [n] = 아래 각주</span></div>"
        "<div class='evidence-scroll'><table class='evidence-table'>"
        "<thead><tr>"
        "<th>각주</th><th>문서</th><th>페이지</th><th>인용 근거 청크</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div></div>"
    )


def render_related_tables_html(
    tables: list[dict[str, Any]] | None,
    *,
    markdown_to_html=None,
) -> str:
    """Show retrieved table crops like MaritimeRAG ``st.image(crop_path)``.

    Markdown table rebuild is not the primary view.
    """
    del markdown_to_html  # kept for call-site compatibility
    if not tables:
        return ""
    parts = [
        "<div class='related-tables-block'>"
        "<div class='evidence-title'>관련 표 (원본 crop) "
        "<span class='evidence-hint'>PDF에서 잘라낸 표 이미지</span></div>"
    ]
    shown = 0
    for table in tables:
        crop = str(table.get("crop_path") or "")
        img = _crop_img_tag(crop) if crop else ""
        if not img:
            continue
        shown += 1
        src = table.get("file_name") or table.get("doc_id") or ""
        page = table.get("page")
        tid = table.get("table_id") or table.get("caption") or ""
        cite_bits = [f"출처: {_esc(src)}"]
        if page not in (None, "", 0):
            cite_bits.append(f"p.{_esc(page)}")
        if tid:
            cite_bits.append(f"<code>{_esc(tid)}</code>")
        parts.append(
            f"<div class='related-table-card'>"
            f"<div class='related-table-head'><b>표 {shown}.</b> {' · '.join(cite_bits)}</div>"
            f"{img}</div>"
        )
    if shown == 0:
        return ""
    parts.append("</div>")
    return "".join(parts)
