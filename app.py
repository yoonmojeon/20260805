"""
MaritimeOpsRAG — 통합 Gradio UI
  운항 SQLite(ops) + 문서 Chroma RAG 를 질문 의도에 따라 자동 라우팅

실행:
  python app.py
"""
from __future__ import annotations

import html
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
from markdown_it import MarkdownIt

from project_paths import (
    DEFAULT_RAG_COLLECTION,
    DEFAULT_TABLE_COLLECTION,
    OPS_DB_PATH,
    RAW_PDFS_DIR,
    REPORTS_DIR,
)
from services.answer_ui import render_evidence_table_html, render_related_tables_html
from services.llm_models import DEFAULT_LLM_MODEL, LLM_MODEL_CHOICES
from services.ops_service import ops_db_ready
from services.orchestrator import handle_question
from services.rag_service import (
    rag_index_banner,
    rag_index_ready,
    rag_status_message,
)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# CommonMark + GFM-ish bits so MaritimeRAG ## / ** / lists render like Streamlit.
_MD = (
    MarkdownIt("commonmark", {"breaks": True, "html": False})
    .enable("strikethrough")
    .enable("table")
)


def _markdown_to_html(text: str) -> str:
    """Render model markdown to HTML (do not html.escape the whole body)."""
    if not (text or "").strip():
        return (
            "<div class='answer-empty'>"
            "<span class='empty-kicker'>해사 업무 AI 코파일럿</span>"
            "<h3>무엇을 확인해드릴까요?</h3>"
            "<p>운항 데이터와 선급·IMO 문서를 구분해 근거와 함께 답변합니다.</p>"
            "</div>"
        )
    return _MD.render(text)


def build_answer_html(
    query: str,
    answer: str,
    map_html: str = "",
    *,
    route_banner: str = "",
    evidence_table: list | None = None,
    related_tables: list | None = None,
) -> str:
    q = html.escape(query.strip()) if query.strip() else ""
    banner = ""
    if route_banner.strip():
        banner = (
            "<div class='route-banner'>"
            "<span class='route-dot' aria-hidden='true'></span>"
            f"<span>{html.escape(route_banner.strip())}</span>"
            "</div>"
        )
    body = _markdown_to_html(answer)
    evidence_html = render_evidence_table_html(evidence_table or [])
    related_html = render_related_tables_html(
        related_tables or [], markdown_to_html=_markdown_to_html
    )
    query_block = (
        "<div class='query-box'>"
        "<span class='query-label'>질문</span>"
        f"<span class='query-text'>{q}</span>"
        "</div>"
        if q
        else ""
    )
    answer_header = (
        "<div class='answer-header'>"
        "<div><span class='answer-mark'>AI</span><b>답변</b></div>"
        "<span class='answer-policy'>근거 확인 후 사용자 검토</span>"
        "</div>"
        if q
        else ""
    )
    map_block = (
        f"<div class='map-box'><div class='map-title'>항차 이동 경로</div>{map_html}</div>"
        if map_html
        else ""
    )
    return f"""
    <div class="answer-wrap">
      {query_block}
      {banner}
      {answer_header}
      <div class="answer-body markdown-body">{body}</div>
      {evidence_html}
      {related_html}
      {map_block}
    </div>"""


def _status_line() -> str:
    ops = "준비됨" if ops_db_ready() else "미구축 (python ops/scripts/load_hodata.py)"
    rag = "준비됨" if rag_index_ready() else "미구축"
    raw_pdf_available = RAW_PDFS_DIR.exists() and any(RAW_PDFS_DIR.rglob("*.pdf"))
    pdfs = "연결됨" if raw_pdf_available else "없음(기존 인덱스 질의 가능)"
    return (
        f"운항DB: {ops} &nbsp;|&nbsp; 문서인덱스({DEFAULT_RAG_COLLECTION}): {rag} "
        f"&nbsp;|&nbsp; raw_pdfs: {pdfs}"
    )


def _route_banner(
    route: dict | None,
    llm_model: str | None = None,
    meta: dict | None = None,
) -> str:
    """Render only user-facing Korean routing facts, never raw router reasoning."""
    if not route:
        return ""
    labels = {
        "ops": "운항 정보",
        "rag": "문서 검색",
        "chat": "사용 안내",
        "hybrid": "운항·문서 통합",
    }
    kind = labels.get(str(route.get("route") or ""), str(route.get("route") or ""))
    method_labels = {
        "llm": "LLM 자동 분류",
        "forced": "전용 경로",
        "dialogue": "대화 문맥",
        "fallback": "안전 경로",
        "heuristic": "질문 분석",
    }
    method = method_labels.get(
        str(route.get("method") or "").lower(), "자동 분류"
    )
    model_labels = {
        "gemma4:12b": "Gemma 4 12B",
        "llama3.1:8b": "Llama 3.1 8B",
    }
    bits = [
        f"확인 자료: {kind}",
        f"신뢰도 {float(route.get('confidence') or 0):.0%}",
        method,
    ]
    if llm_model:
        bits.append(model_labels.get(llm_model, llm_model))
    latency_ms = float((meta or {}).get("end_to_end_latency_ms") or 0)
    if latency_ms > 0:
        bits.append(f"응답 {latency_ms / 1000:.1f}초")
    return "  ·  ".join(bits)


def _force_from_mode(route_mode: str) -> str:
    if route_mode.startswith("자동"):
        return "auto"
    if "운항" in route_mode:
        return "ops"
    if "둘" in route_mode or "hybrid" in route_mode.lower():
        return "hybrid"
    return "rag"


def _pack_answer(user_msg: str, result: dict) -> str:
    return build_answer_html(
        user_msg,
        result.get("answer", ""),
        result.get("map_html") or "",
        route_banner=_route_banner(
            result.get("route"), result.get("llm_model"), result.get("meta")
        ),
        evidence_table=result.get("evidence_table") or [],
        related_tables=result.get("related_tables") or [],
    )


def chat_fn(
    user_msg: str,
    history: list,
    dialogue_state: dict | None,
    route_mode: str,
    llm_model: str,
):
    empty = build_answer_html("", "")
    if not (user_msg or "").strip():
        return history, dialogue_state or {}, empty, [], user_msg or ""

    result = handle_question(
        user_msg,
        history,
        force_route=_force_from_mode(route_mode),  # type: ignore[arg-type]
        use_llm_router=True,
        rag_latency_mode="fast",
        dialogue_state=dialogue_state,
        llm_model=llm_model,
    )
    files = [f for f in (result.get("files") or []) if Path(f).exists()]
    return (
        result.get("history") or history,
        result.get("dialogue_state") or dialogue_state or {},
        _pack_answer(user_msg, result),
        files or [],
        user_msg,
    )


def retry_fn(
    force_route: str,
    history: list,
    dialogue_state: dict | None,
    last_question: str,
    llm_model: str,
):
    empty = build_answer_html("", "")
    q = (last_question or "").strip()
    if not q:
        return history, dialogue_state or {}, empty, [], last_question
    hist = list(history or [])
    if len(hist) >= 2:
        hist = hist[:-2]
    result = handle_question(
        q,
        hist,
        force_route=force_route,  # type: ignore[arg-type]
        use_llm_router=True,
        rag_latency_mode="fast",
        dialogue_state=dialogue_state,
        llm_model=llm_model,
    )
    files = [f for f in (result.get("files") or []) if Path(f).exists()]
    return (
        result.get("history") or hist,
        result.get("dialogue_state") or dialogue_state or {},
        _pack_answer(q, result),
        files or [],
        q,
    )


def fixed_route_chat_fn(
    force_route: str,
    user_msg: str,
    history: list,
    dialogue_state: dict | None,
    llm_model: str,
):
    """Handle a question in a dedicated tab without running top-level routing."""
    label = "운항 DB 강제" if force_route == "ops" else "문서 RAG 강제"
    return chat_fn(user_msg, history, dialogue_state, label, llm_model)


def document_chat_fn(
    user_msg: str,
    history: list,
    dialogue_state: dict | None,
    llm_model: str,
):
    return fixed_route_chat_fn("rag", user_msg, history, dialogue_state, llm_model)


def ops_chat_fn(
    user_msg: str,
    history: list,
    dialogue_state: dict | None,
    llm_model: str,
):
    return fixed_route_chat_fn("ops", user_msg, history, dialogue_state, llm_model)


REPORT_EXTENSIONS = {".docx", ".pdf", ".xlsx", ".csv", ".html", ".md"}


def _report_inventory(search: str = "", status: str = "전체") -> list[dict]:
    """List real generated files only; report approval metadata does not yet exist."""
    needle = (search or "").strip().lower()
    if status not in {"전체", "검토 필요"} or not REPORTS_DIR.exists():
        return []
    reports: list[dict] = []
    for path in sorted(REPORTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not path.is_file() or path.suffix.lower() not in REPORT_EXTENSIONS:
            continue
        if needle and needle not in path.name.lower():
            continue
        stat = path.stat()
        reports.append(
            {
                "path": str(path),
                "name": path.name,
                "title": path.stem.replace("_", " "),
                "ext": path.suffix[1:].upper(),
                "size_kb": max(1, round(stat.st_size / 1024)),
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y.%m.%d %H:%M"),
                "status": "AI 초안 · 검토 필요",
            }
        )
    return reports


def build_report_manager_html(search: str = "", status: str = "전체") -> str:
    reports = _report_inventory(search, status)
    if not reports:
        return (
            "<div class='report-empty'>"
            "<span class='empty-icon'>▤</span>"
            "<h3>조건에 맞는 보고서가 없습니다</h3>"
            "<p>운항 정보에서 보고서를 생성하면 이곳에 자동으로 표시됩니다.</p>"
            "</div>"
        )
    cards = []
    for item in reports:
        cards.append(
            "<article class='report-card'>"
            "<div class='report-file-icon'>DOC</div>"
            "<div class='report-card-main'>"
            f"<h3>{html.escape(item['title'])}</h3>"
            f"<p>{html.escape(item['name'])}</p>"
            "<div class='report-meta'>"
            f"<span>{item['ext']}</span><span>{item['size_kb']}KB</span>"
            f"<span>{item['modified']}</span>"
            "</div></div>"
            "<div class='report-card-side'>"
            f"<span class='status-badge review'>{item['status']}</span>"
            "<span class='review-note'>사용자가 검토·확정해야 합니다</span>"
            "</div></article>"
        )
    return (
        f"<div class='report-summary'><b>{len(reports)}건</b>의 실제 생성 파일</div>"
        f"<div class='report-list'>{''.join(cards)}</div>"
    )


def refresh_reports(search: str, status: str):
    reports = _report_inventory(search, status)
    return build_report_manager_html(search, status), [item["path"] for item in reports]


def _page_intro(eyebrow: str, title: str, description: str, badge: str) -> str:
    return f"""
    <section class='page-intro'>
      <div>
        <span class='page-eyebrow'>{html.escape(eyebrow)}</span>
        <h2>{html.escape(title)}</h2>
        <p>{html.escape(description)}</p>
      </div>
      <span class='page-badge'>{html.escape(badge)}</span>
    </section>
    """


CUSTOM_CSS = """
:root {
  --navy-950: #071b2b;
  --navy-900: #0b2438;
  --navy-800: #12344c;
  --teal-600: #0e7c7b;
  --teal-500: #149b98;
  --teal-100: #dff5f2;
  --ink-900: #172735;
  --ink-700: #425466;
  --ink-500: #6d7d8b;
  --line: #dce4ea;
  --surface: #ffffff;
  --canvas: #f3f6f8;
}
body, .gradio-container {
  font-family: 'Pretendard', 'Noto Sans KR', 'Malgun Gothic', 'Segoe UI', sans-serif !important;
  background: var(--canvas) !important;
  color: var(--ink-900) !important;
}
.gradio-container {
  max-width: 1440px !important;
  width: min(1440px, calc(100% - 56px)) !important;
  margin: 0 auto !important;
  padding: 20px 28px 34px !important;
}
.app-shell-header {
  display: flex; align-items: center; justify-content: space-between; gap: 24px;
  min-height: 88px; padding: 18px 24px; color: #fff;
  background: var(--navy-950); border: 1px solid #17384e; border-radius: 14px 14px 4px 4px;
  box-shadow: 0 10px 28px rgba(7, 27, 43, .12);
}
.brand-lockup { display: flex; align-items: center; gap: 14px; min-width: 280px; }
.brand-mark {
  display: grid; place-items: center; width: 42px; height: 42px; border-radius: 11px;
  background: #0d3b4d; border: 1px solid #1d6572; color: #68d7ce; font-size: 21px;
}
.brand-copy h1 { margin: 0; color: #fff; font-size: 21px; line-height: 1.2; letter-spacing: -.02em; }
.brand-copy p { margin: 5px 0 0; color: #9fb4c2; font-size: 11px; letter-spacing: .12em; }
.vessel-context { display: flex; gap: 8px; align-items: center; color: #c6d4dc; font-size: 13px; }
.context-pill { padding: 7px 10px; background: #0e2c41; border: 1px solid #24465b; border-radius: 999px; color: #c6d4dc !important; }
.ready-state { display: inline-flex; align-items: center; gap: 8px; color: #d9f3ee; font-size: 13px; font-weight: 700; white-space: nowrap; }
.ready-state::before { content: ''; width: 8px; height: 8px; border-radius: 50%; background: #42d3a6; box-shadow: 0 0 0 4px rgba(66, 211, 166, .12); }
.control-bar { margin: 12px 0 14px !important; align-items: stretch !important; gap: 12px !important; flex-wrap: nowrap !important; }
.control-bar > div:first-child { flex: 1 1 0 !important; min-width: 0 !important; width: auto !important; }
.control-bar > div:last-child {
  flex: 0 0 340px !important; max-width: 340px !important; background: #fff !important;
  border: 1px solid var(--line) !important; border-radius: 10px !important;
}
.system-strip {
  height: 100%; min-height: 118px; display: flex; align-items: center; gap: 18px;
  padding: 13px 18px; background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
}
.system-strip strong { font-size: 13px; color: var(--ink-900); }
.system-item { display: flex; align-items: center; gap: 8px; color: var(--ink-700); font-size: 12px; }
.system-item::before { content: ''; width: 7px; height: 7px; border-radius: 50%; background: #2bb98b; }
.system-divider { width: 1px; height: 24px; background: var(--line); }
.model-control { max-width: 310px; }
.model-control label span { color: var(--ink-700) !important; font-size: 12px !important; font-weight: 700 !important; }
#workspace-tabs { background: transparent !important; }
#workspace-tabs > div:first-child, #workspace-tabs [role='tablist'] {
  gap: 4px !important; padding: 6px !important; margin-bottom: 14px !important;
  background: #e7edf1 !important; border: 1px solid #d7e0e6 !important; border-radius: 11px !important;
}
#workspace-tabs button[role='tab'] {
  min-height: 44px !important; padding: 0 18px !important; border-radius: 8px !important;
  color: #526675 !important; font-size: 14px !important; font-weight: 700 !important;
}
#workspace-tabs button[role='tab'][aria-selected='true'] {
  background: var(--surface) !important; color: var(--navy-900) !important;
  box-shadow: 0 1px 5px rgba(10, 35, 53, .12) !important;
}
.workspace-panel { padding: 0 !important; background: transparent !important; border: 0 !important; }
.page-intro {
  display: flex; align-items: flex-start; justify-content: space-between; gap: 24px;
  padding: 24px 26px 20px; margin-bottom: 12px; background: var(--surface);
  border: 1px solid var(--line); border-radius: 12px;
}
.page-eyebrow, .empty-kicker { color: var(--teal-600); font-size: 10px; font-weight: 800; letter-spacing: .14em; }
.page-intro h2 { margin: 6px 0 5px; color: var(--navy-900); font-size: 23px; letter-spacing: -.025em; }
.page-intro p { margin: 0; max-width: 760px; color: var(--ink-500); font-size: 13px; line-height: 1.6; }
.page-badge { padding: 7px 10px; color: #11645f; background: var(--teal-100); border-radius: 999px; font-size: 11px; font-weight: 800; white-space: nowrap; }
.route-control { margin: 0 0 10px !important; }
.route-control label span { font-size: 12px !important; color: var(--ink-700) !important; }
.example-label { margin: 12px 0 7px; color: var(--ink-500); font-size: 11px; font-weight: 700; letter-spacing: .04em; }
.example-row { gap: 8px !important; }
.example-row button {
  min-height: 34px !important; border: 1px solid #cedae1 !important; border-radius: 999px !important;
  background: #f9fbfc !important; color: #385164 !important; font-size: 12px !important; font-weight: 600 !important;
}
.answer-wrap {
  margin: 12px 0; padding: 20px 22px 22px; background: var(--surface);
  border: 1px solid var(--line); border-radius: 12px; box-shadow: 0 3px 12px rgba(20, 45, 61, .035);
}
.query-box { display: flex; gap: 12px; align-items: flex-start; padding: 13px 15px; margin-bottom: 10px; background: #f5f8fa; border-radius: 8px; }
.query-label { flex: 0 0 auto; color: var(--teal-600); font-size: 12px; font-weight: 800; }
.query-text { color: var(--ink-900); font-size: 14px; line-height: 1.55; }
.route-banner {
  display: flex; align-items: center; gap: 8px; padding: 8px 11px; margin-bottom: 16px;
  background: #eff9f8; border: 1px solid #c9e8e4; border-radius: 7px;
  color: #356b6b; font-size: 11px; line-height: 1.5;
}
.route-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--teal-500); flex: 0 0 auto; }
.answer-header { display: flex; align-items: center; justify-content: space-between; padding-bottom: 12px; border-bottom: 1px solid #e8edf0; }
.answer-header > div { display: flex; align-items: center; gap: 8px; }
.answer-mark { padding: 3px 6px; border-radius: 5px; background: var(--navy-900); color: #fff; font-size: 10px; font-weight: 800; }
.answer-header b { font-size: 14px; color: var(--navy-900); }
.answer-policy { color: var(--ink-500); font-size: 10px; }
.answer-body.markdown-body { padding-top: 3px; font-size: 15px; line-height: 1.82; color: #263b49; }
.answer-body.markdown-body h1, .answer-body.markdown-body h2, .answer-body.markdown-body h3 {
  margin: 1.15em 0 .5em; color: var(--navy-900); font-weight: 800; line-height: 1.38;
}
.answer-body.markdown-body h2 { padding-bottom: .32em; border-bottom: 1px solid #e7edf1; font-size: 1.18em; }
.answer-body.markdown-body h3 { font-size: 1.05em; }
.answer-body.markdown-body p { margin: 0 0 .8em; }
.answer-body.markdown-body ul, .answer-body.markdown-body ol { margin: 0 0 .9em 1.25em; padding: 0; }
.answer-body.markdown-body li { margin: .32em 0; padding-left: .1em; }
.answer-body.markdown-body strong { color: #132e42; font-weight: 800; }
.answer-body.markdown-body code { padding: .12em .38em; background: #eef3f6; border-radius: 4px; font-size: .9em; }
.answer-body.markdown-body table, .related-table-md table { border-collapse: collapse; width: 100%; margin: .7em 0 1em; font-size: 13px; }
.answer-body.markdown-body th, .answer-body.markdown-body td, .related-table-md th, .related-table-md td { border: 1px solid #dbe4e9; padding: 8px 10px; vertical-align: top; }
.answer-body.markdown-body th, .related-table-md th { background: #f4f7f9; font-weight: 700; }
.answer-empty { padding: 44px 20px 48px; text-align: center; }
.answer-empty h3 { margin: 8px 0 5px; color: var(--navy-900); font-size: 20px; }
.answer-empty p { margin: 0; color: var(--ink-500); font-size: 13px; }
.input-row { align-items: stretch !important; gap: 10px !important; padding: 10px; background: #fff; border: 1px solid #d7e1e7; border-radius: 11px; box-shadow: 0 5px 16px rgba(19, 47, 66, .045); }
.input-row textarea { min-height: 44px !important; padding: 11px 12px !important; font-size: 14px !important; }
.send-btn { min-width: 118px !important; }
.send-btn button { height: 44px !important; background: var(--teal-600) !important; border-color: var(--teal-600) !important; font-weight: 800 !important; }
.retry-row { gap: 8px !important; margin-top: 8px !important; }
.retry-row button { min-height: 34px !important; color: #3d5667 !important; border-color: #d5e0e6 !important; background: #f7f9fa !important; font-size: 11px !important; }
.artifact-files { margin-top: 12px !important; }
.map-box, .evidence-block, .related-tables-block { margin-top: 18px; border: 1px solid #d9e3e8; border-radius: 9px; overflow: hidden; background: #fff; }
.map-title, .evidence-title { padding: 10px 12px; background: #f3f7f8; border-bottom: 1px solid #e2e9ed; color: #294354; font-size: 12px; font-weight: 800; }
.evidence-hint { margin-left: 8px; color: #748692; font-size: 11px; font-weight: 500; }
.evidence-scroll { overflow-x: auto; }
.evidence-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.evidence-table th, .evidence-table td { padding: 9px 10px; border-bottom: 1px solid #e7edf0; text-align: left; vertical-align: top; }
.evidence-table th { background: #fafcfc; color: #617582; font-weight: 700; white-space: nowrap; }
.evidence-table td.cite { width: 48px; color: var(--teal-600); font-weight: 800; white-space: nowrap; }
.evidence-table td.page { width: 58px; color: #647684; white-space: nowrap; }
.evidence-table td.preview { color: #405562; line-height: 1.48; }
.related-table-card { padding: 13px 14px; border-top: 1px solid #e7edf0; }
.related-table-card:first-of-type { border-top: 0; }
.related-table-head, .evidence-note, .table-crop figcaption { color: #71838e; font-size: 11px; }
.table-crop { margin: 10px 0 0; }
.report-toolbar { align-items: flex-end !important; gap: 10px !important; padding: 15px; margin-bottom: 10px !important; background: #fff; border: 1px solid var(--line); border-radius: 10px; }
.report-summary { margin: 14px 0 8px; color: var(--ink-700); font-size: 12px; }
.report-summary b { color: var(--navy-900); font-size: 15px; }
.report-list { display: grid; gap: 9px; }
.report-card { display: flex; align-items: center; gap: 14px; padding: 16px 18px; background: #fff; border: 1px solid var(--line); border-radius: 10px; }
.report-file-icon { display: grid; place-items: center; width: 42px; height: 48px; border: 1px solid #b9dcd7; border-radius: 7px; background: #edf8f6; color: var(--teal-600); font-size: 10px; font-weight: 900; }
.report-card-main { min-width: 0; flex: 1; }
.report-card-main h3 { margin: 0 0 3px; color: var(--navy-900); font-size: 14px; }
.report-card-main p { margin: 0; overflow: hidden; color: var(--ink-500); font-size: 11px; text-overflow: ellipsis; white-space: nowrap; }
.report-meta { display: flex; gap: 7px; margin-top: 7px; }
.report-meta span { padding-right: 7px; border-right: 1px solid #dfe6ea; color: #71828d; font-size: 10px; }
.report-meta span:last-child { border: 0; }
.report-card-side { display: flex; flex-direction: column; align-items: flex-end; gap: 6px; }
.status-badge { padding: 5px 8px; border-radius: 999px; font-size: 10px; font-weight: 800; }
.status-badge.review { color: #825b10; background: #fff3cc; }
.review-note { color: #8a989f; font-size: 9px; }
.report-empty { padding: 50px 20px; text-align: center; background: #fff; border: 1px dashed #cfdbe1; border-radius: 10px; }
.report-empty .empty-icon { display: inline-grid; place-items: center; width: 38px; height: 38px; color: var(--teal-600); background: #eaf7f5; border-radius: 50%; }
.report-empty h3 { margin: 10px 0 4px; color: var(--navy-900); font-size: 16px; }
.report-empty p { margin: 0; color: var(--ink-500); font-size: 12px; }
.report-note { padding: 12px 14px; color: #586d7a; background: #f6f9fa; border-left: 3px solid var(--teal-500); border-radius: 6px; font-size: 12px; line-height: 1.6; }
.diagnostics { margin-top: 16px !important; }
footer { display: none !important; }
@media (max-width: 900px) {
  .gradio-container { width: calc(100% - 20px) !important; padding: 10px 12px 24px !important; }
  .app-shell-header { align-items: flex-start; flex-direction: column; }
  .control-bar { flex-wrap: wrap !important; }
  .control-bar > div:last-child { flex: 1 1 auto !important; max-width: none !important; }
  .vessel-context { flex-wrap: wrap; }
  .model-control { max-width: none; }
  .page-intro { flex-direction: column; }
  .report-card { align-items: flex-start; }
  .report-card-side { align-items: flex-start; }
  .answer-policy { display: none; }
}
"""

with gr.Blocks(title="MaritimeOps AI") as demo:
    all_ready = ops_db_ready() and rag_index_ready()
    ready_label = "시스템 준비" if all_ready else "일부 기능 점검 필요"
    gr.HTML(
        f"""
        <header class="app-shell-header">
          <div class="brand-lockup">
            <div class="brand-mark" aria-hidden="true">M</div>
            <div class="brand-copy">
              <h1>MaritimeOps AI</h1>
              <p>선원 업무 지원 코파일럿</p>
            </div>
          </div>
          <div class="vessel-context">
            <span class="context-pill">선박 H2521</span>
            <span class="context-pill">벌크선</span>
            <span class="context-pill">로컬 업무공간</span>
          </div>
          <span class="ready-state">{ready_label}</span>
        </header>
        """
    )

    with gr.Row(elem_classes=["control-bar"]):
        gr.HTML(
            f"""
            <div class="system-strip">
              <strong>연결 상태</strong>
              <span class="system-divider"></span>
              <span class="system-item">운항 DB {'준비됨' if ops_db_ready() else '미구축'}</span>
              <span class="system-item">문서 인덱스 {'준비됨' if rag_index_ready() else '미구축'}</span>
              <span class="system-item">원문 PDF {'연결됨' if RAW_PDFS_DIR.exists() else '없음'}</span>
            </div>
            """,
            scale=4,
        )
        llm_model = gr.Dropdown(
            choices=list(LLM_MODEL_CHOICES),
            value=DEFAULT_LLM_MODEL,
            label="답변 모델",
            info="Gemma를 기본으로 사용하며, Llama는 비교·빠른 응답용입니다.",
            scale=1,
            elem_classes=["model-control"],
        )

    with gr.Tabs(elem_id="workspace-tabs"):
        with gr.Tab("통합 질문", id="integrated", elem_classes=["workspace-panel"]):
            integrated_history = gr.State([])
            integrated_dialogue = gr.State({})
            integrated_last_question = gr.State("")

            gr.HTML(
                _page_intro(
                    "AI 업무 지원",
                    "통합 질문",
                    "한 문장으로 질문하면 운항 데이터와 문서 근거 중 필요한 자료를 자동으로 확인합니다.",
                    "자동 라우팅",
                )
            )
            integrated_route_mode = gr.Radio(
                choices=["자동 라우팅", "운항 DB 강제", "문서 RAG 강제"],
                value="자동 라우팅",
                label="확인할 자료",
                elem_classes=["route-control"],
            )
            integrated_examples = [
                "현재 운항 상태 알려줘",
                "최신 MEPC 회의 주요 내용을 정리해줘",
                "우리 CII랑 MEPC 규제 같이 알려줘",
            ]
            gr.HTML("<div class='example-label'>추천 질문</div>")
            with gr.Row(elem_classes=["example-row"]):
                integrated_example_btns = [
                    gr.Button(text, size="sm") for text in integrated_examples
                ]
            integrated_answer = gr.HTML(value=build_answer_html("", ""))
            with gr.Row(elem_classes=["input-row"]):
                integrated_input = gr.Textbox(
                    placeholder="운항·문서·혼합 질문을 입력하세요",
                    show_label=False,
                    scale=8,
                    container=False,
                )
                integrated_send = gr.Button(
                    "질문 보내기", variant="primary", scale=1, elem_classes=["send-btn"]
                )
            with gr.Row(elem_classes=["retry-row"]):
                retry_ops = gr.Button("운항만으로 다시", size="sm")
                retry_rag = gr.Button("문서만으로 다시", size="sm")
                retry_hyb = gr.Button("둘 다로 다시", size="sm")
            integrated_files = gr.File(
                label="생성 파일 · 검색된 표 원본",
                file_count="multiple",
                elem_classes=["artifact-files"],
            )

        with gr.Tab("문서 검색", id="documents", elem_classes=["workspace-panel"]):
            document_history = gr.State([])
            document_dialogue = gr.State({})
            document_last_question = gr.State("")

            gr.HTML(
                _page_intro(
                    "문서 지식 검색",
                    "문서 검색",
                    "선급 규정, IMO 회의자료, 본문과 표를 문서 RAG로 직접 검색하고 인용 근거를 확인합니다.",
                    "문서 RAG 고정",
                )
            )
            document_examples = [
                "과도한 부식의 정의는 무엇인가?",
                "구조 규칙에서 쓰는 tcorr 기호는 어떤 두께를 뜻하지?",
                "형상이 복잡하거나 한 개의 중량이 10톤을 넘는 주강품은 제품마다 시험재가 몇 개 필요한가?",
            ]
            gr.HTML("<div class='example-label'>문서 검색 예시</div>")
            with gr.Row(elem_classes=["example-row"]):
                document_example_btns = [
                    gr.Button(text, size="sm") for text in document_examples
                ]
            document_answer = gr.HTML(value=build_answer_html("", ""))
            with gr.Row(elem_classes=["input-row"]):
                document_input = gr.Textbox(
                    placeholder="규정·회의자료·표 검색 질문을 입력하세요",
                    show_label=False,
                    scale=8,
                    container=False,
                )
                document_send = gr.Button(
                    "문서 검색", variant="primary", scale=1, elem_classes=["send-btn"]
                )
            document_files = gr.File(
                label="검색된 원본 표 · 생성 파일",
                file_count="multiple",
                elem_classes=["artifact-files"],
            )

        with gr.Tab("운항 정보", id="operations", elem_classes=["workspace-panel"]):
            ops_history = gr.State([])
            ops_dialogue = gr.State({})
            ops_last_question = gr.State("")

            gr.HTML(
                _page_intro(
                    "운항 데이터",
                    "운항 정보",
                    "로컬 운항 DB의 실제 항차, 속력, 연료, 배출량, CII를 조회하고 보고서를 생성합니다.",
                    "운항 DB 고정",
                )
            )
            ops_examples = [
                "현재 운항 중인 항차 번호와 적재 상태를 알려줘.",
                "현재 Ballast 항차의 누적 운항거리는 몇 해리야?",
                "2026년 누적 잠정 CII attained, required와 등급은?",
            ]
            gr.HTML("<div class='example-label'>운항 조회 예시</div>")
            with gr.Row(elem_classes=["example-row"]):
                ops_example_btns = [
                    gr.Button(text, size="sm") for text in ops_examples
                ]
            ops_answer = gr.HTML(value=build_answer_html("", ""))
            with gr.Row(elem_classes=["input-row"]):
                ops_input = gr.Textbox(
                    placeholder="항차·연료·배출량·CII 질문을 입력하세요",
                    show_label=False,
                    scale=8,
                    container=False,
                )
                ops_send = gr.Button(
                    "운항 조회", variant="primary", scale=1, elem_classes=["send-btn"]
                )
            ops_files = gr.File(
                label="생성된 운항 보고서",
                file_count="multiple",
                elem_classes=["artifact-files"],
            )

        with gr.Tab("보고서 관리", id="reports", elem_classes=["workspace-panel"]):
            gr.HTML(
                _page_intro(
                    "업무 결과물",
                    "보고서 관리",
                    "AI가 생성한 실제 보고서 파일을 찾아 검토하고 내려받습니다. 공식 확정과 제출은 사용자가 수행합니다.",
                    "사용자 검토 필수",
                )
            )
            with gr.Row(elem_classes=["report-toolbar"]):
                report_search = gr.Textbox(
                    label="보고서 검색",
                    placeholder="파일명으로 검색",
                    scale=3,
                )
                report_status = gr.Radio(
                    choices=["전체", "작성중", "검토 필요", "완료"],
                    value="전체",
                    label="상태",
                    scale=3,
                )
                report_refresh = gr.Button("목록 새로고침", variant="secondary", scale=1)
            report_cards = gr.HTML(value=build_report_manager_html())
            report_downloads = gr.File(
                value=[item["path"] for item in _report_inventory()],
                label="보고서 다운로드",
                file_count="multiple",
                interactive=False,
                elem_classes=["artifact-files"],
            )
            gr.HTML(
                "<div class='report-note'><b>보고서 생성 방법</b><br>"
                "운항 정보 탭에서 ‘Noon Report를 생성해줘’, ‘2026년 연간 MRV 보고서를 생성해줘’처럼 요청하면 "
                "생성된 Word 파일이 이 목록에 표시됩니다.</div>"
            )

    try:
        banner = rag_index_banner(sample_size=1500)
    except Exception as exc:
        banner = f"[RAG INDEX]\n(진단 정보를 불러오지 못했습니다: {exc})"
    gr.Markdown(
        f"<small>{html.escape(rag_status_message())}<br>"
        f"본문 인덱스: `{DEFAULT_RAG_COLLECTION}` / 표 인덱스: `{DEFAULT_TABLE_COLLECTION}`<br>"
        f"운항 DB: `{OPS_DB_PATH}`</small>",
        elem_classes=["diagnostics"],
    )
    with gr.Accordion("시스템 진단 정보", open=False, elem_classes=["diagnostics"]):
        gr.Markdown(f"```\n{banner}\n```")

    for btn, text in zip(integrated_example_btns, integrated_examples):
        btn.click(lambda t=text: t, outputs=integrated_input)
    for btn, text in zip(document_example_btns, document_examples):
        btn.click(lambda t=text: t, outputs=document_input)
    for btn, text in zip(ops_example_btns, ops_examples):
        btn.click(lambda t=text: t, outputs=ops_input)

    integrated_send.click(
        chat_fn,
        inputs=[
            integrated_input,
            integrated_history,
            integrated_dialogue,
            integrated_route_mode,
            llm_model,
        ],
        outputs=[
            integrated_history,
            integrated_dialogue,
            integrated_answer,
            integrated_files,
            integrated_last_question,
        ],
    ).then(lambda: "", outputs=integrated_input)

    integrated_input.submit(
        chat_fn,
        inputs=[
            integrated_input,
            integrated_history,
            integrated_dialogue,
            integrated_route_mode,
            llm_model,
        ],
        outputs=[
            integrated_history,
            integrated_dialogue,
            integrated_answer,
            integrated_files,
            integrated_last_question,
        ],
    ).then(lambda: "", outputs=integrated_input)

    retry_ops.click(
        lambda hist, st, last_q, model: retry_fn("ops", hist, st, last_q, model),
        inputs=[
            integrated_history,
            integrated_dialogue,
            integrated_last_question,
            llm_model,
        ],
        outputs=[
            integrated_history,
            integrated_dialogue,
            integrated_answer,
            integrated_files,
            integrated_last_question,
        ],
    )
    retry_rag.click(
        lambda hist, st, last_q, model: retry_fn("rag", hist, st, last_q, model),
        inputs=[
            integrated_history,
            integrated_dialogue,
            integrated_last_question,
            llm_model,
        ],
        outputs=[
            integrated_history,
            integrated_dialogue,
            integrated_answer,
            integrated_files,
            integrated_last_question,
        ],
    )
    retry_hyb.click(
        lambda hist, st, last_q, model: retry_fn("hybrid", hist, st, last_q, model),
        inputs=[
            integrated_history,
            integrated_dialogue,
            integrated_last_question,
            llm_model,
        ],
        outputs=[
            integrated_history,
            integrated_dialogue,
            integrated_answer,
            integrated_files,
            integrated_last_question,
        ],
    )

    document_outputs = [
        document_history,
        document_dialogue,
        document_answer,
        document_files,
        document_last_question,
    ]
    document_inputs = [
        document_input,
        document_history,
        document_dialogue,
        llm_model,
    ]
    document_send.click(
        document_chat_fn,
        inputs=document_inputs,
        outputs=document_outputs,
    ).then(lambda: "", outputs=document_input)
    document_input.submit(
        document_chat_fn,
        inputs=document_inputs,
        outputs=document_outputs,
    ).then(lambda: "", outputs=document_input)

    ops_outputs = [
        ops_history,
        ops_dialogue,
        ops_answer,
        ops_files,
        ops_last_question,
    ]
    ops_inputs = [ops_input, ops_history, ops_dialogue, llm_model]
    ops_send.click(
        ops_chat_fn,
        inputs=ops_inputs,
        outputs=ops_outputs,
    ).then(lambda: "", outputs=ops_input)
    ops_input.submit(
        ops_chat_fn,
        inputs=ops_inputs,
        outputs=ops_outputs,
    ).then(lambda: "", outputs=ops_input)

    report_refresh.click(
        refresh_reports,
        inputs=[report_search, report_status],
        outputs=[report_cards, report_downloads],
        show_progress="hidden",
    )
    report_search.submit(
        refresh_reports,
        inputs=[report_search, report_status],
        outputs=[report_cards, report_downloads],
        show_progress="hidden",
    )
    report_status.change(
        refresh_reports,
        inputs=[report_search, report_status],
        outputs=[report_cards, report_downloads],
        show_progress="hidden",
    )


if __name__ == "__main__":
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    print("=" * 56)
    print("  MaritimeOps AI")
    print(f"  {_status_line().replace('&nbsp;', ' ')}")
    print("=" * 56)
    if rag_index_ready():
        print("  문서 인덱스 준비됨. 첫 RAG 질문 때 모델을 로드합니다.")
    print(f"  브라우저: http://127.0.0.1:{server_port}  (0.0.0.0 은 접속 주소가 아님)")
    print("=" * 56)
    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=False,
        theme=gr.themes.Base(primary_hue="teal", neutral_hue="gray"),
        css=CUSTOM_CSS,
    )
