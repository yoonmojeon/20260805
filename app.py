"""
MaritimeOpsRAG — 통합 Gradio UI
  운항 SQLite(ops) + 문서 Chroma RAG 를 질문 의도에 따라 자동 라우팅

실행:
  python app.py
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
from markdown_it import MarkdownIt

from project_paths import DEFAULT_RAG_COLLECTION, DEFAULT_TABLE_COLLECTION, OPS_DB_PATH, RAW_PDFS_DIR
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
            "<p class='answer-empty'>질문을 입력하세요. "
            "안내 / 운항 데이터 / 규정·회의 문서를 자동으로 구분합니다.</p>"
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
        banner = f"<div class='route-banner'>{html.escape(route_banner.strip())}</div>"
    body = _markdown_to_html(answer)
    evidence_html = render_evidence_table_html(evidence_table or [])
    related_html = render_related_tables_html(
        related_tables or [], markdown_to_html=_markdown_to_html
    )
    query_block = f"<div class='query-box'><b>질문</b> {q}</div>" if q else ""
    map_block = (
        f"<div class='map-box'><div class='map-title'>항차 이동 경로</div>{map_html}</div>"
        if map_html
        else ""
    )
    return f"""
    <div class="answer-wrap">
      {query_block}
      {banner}
      <div class="answer-body markdown-body">{body}</div>
      {evidence_html}
      {related_html}
      {map_block}
    </div>"""


def _status_line() -> str:
    ops = "OK" if ops_db_ready() else "미구축 (python ops/scripts/load_hodata.py)"
    rag = "OK" if rag_index_ready() else "미구축"
    raw_pdf_available = RAW_PDFS_DIR.exists() and any(RAW_PDFS_DIR.rglob("*.pdf"))
    pdfs = "연결됨" if raw_pdf_available else "없음(기존 인덱스 질의 가능)"
    return (
        f"운항DB: {ops} &nbsp;|&nbsp; 문서인덱스({DEFAULT_RAG_COLLECTION}): {rag} "
        f"&nbsp;|&nbsp; raw_pdfs: {pdfs}"
    )


def _route_banner(route: dict | None, llm_model: str | None = None) -> str:
    if not route:
        return ""
    labels = {
        "ops": "운항 DB (ops)",
        "rag": "문서 RAG (rag)",
        "chat": "안내 (chat)",
        "hybrid": "운항+문서 (hybrid)",
    }
    kind = labels.get(str(route.get("route") or ""), str(route.get("route") or ""))
    model_bit = f" | 모델 {llm_model}" if llm_model else ""
    return (
        f"[경로: {kind} | 신뢰도 {float(route.get('confidence') or 0):.0%} "
        f"| {route.get('method')}{model_bit}] {route.get('reason') or ''}"
    )


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
            result.get("route"), result.get("llm_model")
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


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');
body, .gradio-container {
  font-family: 'Noto Sans KR', 'Segoe UI', sans-serif !important;
  background: #fff !important;
  color: #262730 !important;
}
.gradio-container { max-width: 980px !important; margin: 0 auto !important; }
.app-header { padding: 20px 0 12px; border-bottom: 1px solid #e6e8ec; margin-bottom: 16px; }
.app-header h1 { margin: 0; font-size: 26px; font-weight: 700; }
.app-header p { margin: 4px 0 0; color: #808495; font-size: 13px; }
.example-row button { font-size: 12px !important; border-radius: 20px !important; }
.query-box { background: #f8f9fb; border-radius: 8px; padding: 10px 14px; margin-bottom: 16px; font-size: 14px; }
.query-box b { color: #ff4b4b; margin-right: 6px; }
.route-banner {
  background: #fff7f7; border: 1px solid #ffd5d5; color: #6b3030;
  border-radius: 8px; padding: 8px 12px; margin-bottom: 14px; font-size: 12px; line-height: 1.5;
}
.answer-body.markdown-body { font-size: 15px; line-height: 1.85; color: #31333f; }
.answer-body.markdown-body h1,
.answer-body.markdown-body h2,
.answer-body.markdown-body h3 {
  margin: 1.1em 0 0.45em; font-weight: 700; line-height: 1.35; color: #1f2430;
}
.answer-body.markdown-body h2 { font-size: 1.25em; border-bottom: 1px solid #eef0f4; padding-bottom: 0.25em; }
.answer-body.markdown-body h3 { font-size: 1.1em; }
.answer-body.markdown-body p { margin: 0 0 0.85em; }
.answer-body.markdown-body ul, .answer-body.markdown-body ol { margin: 0 0 0.9em 1.2em; padding: 0; }
.answer-body.markdown-body li { margin: 0.25em 0; }
.answer-body.markdown-body strong { font-weight: 700; }
.answer-body.markdown-body code {
  background: #f4f5f7; padding: 0.1em 0.35em; border-radius: 4px; font-size: 0.92em;
}
.answer-body.markdown-body table,
.related-table-md table {
  border-collapse: collapse; width: 100%; margin: 0.6em 0 1em; font-size: 13px;
}
.answer-body.markdown-body th, .answer-body.markdown-body td,
.related-table-md th, .related-table-md td {
  border: 1px solid #e0e3ea; padding: 6px 8px; vertical-align: top;
}
.answer-body.markdown-body th, .related-table-md th { background: #f8f9fb; font-weight: 600; }
.answer-empty { color: #808495; font-size: 14px; }
.map-box { margin-top: 20px; border: 1px solid #e6e8ec; border-radius: 8px; overflow: hidden; }
.map-title { padding: 8px 12px; background: #f8f9fb; font-size: 13px; font-weight: 600; color: #555; }
.evidence-block, .related-tables-block {
  margin-top: 18px; border: 1px solid #e6e8ec; border-radius: 8px; overflow: hidden;
  background: #fff;
}
.evidence-title {
  padding: 10px 12px; background: #f8f9fb; font-size: 13px; font-weight: 700; color: #333;
  border-bottom: 1px solid #eef0f4;
}
.evidence-hint { font-weight: 500; color: #808495; margin-left: 8px; font-size: 12px; }
.evidence-scroll { overflow-x: auto; }
.evidence-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
.evidence-table th, .evidence-table td {
  border-bottom: 1px solid #eef0f4; padding: 8px 10px; text-align: left; vertical-align: top;
}
.evidence-table th { background: #fafbfc; color: #555; font-weight: 600; white-space: nowrap; }
.evidence-table td.cite { font-weight: 700; color: #c62828; white-space: nowrap; width: 48px; }
.evidence-table td.page { white-space: nowrap; width: 56px; color: #555; }
.evidence-table td.preview { color: #444; line-height: 1.45; }
.related-table-card { padding: 12px 14px; border-top: 1px solid #eef0f4; }
.related-table-card:first-of-type { border-top: none; }
.related-table-head { margin-bottom: 8px; font-size: 13px; color: #444; }
.evidence-note { font-size: 12px; color: #808495; margin: 6px 0 0; }
.table-crop { margin: 10px 0 0; }
.table-crop figcaption { font-size: 12px; color: #808495; margin-top: 4px; }
.send-btn button { background: #ff4b4b !important; border-color: #ff4b4b !important; font-weight: 600 !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="MaritimeOpsRAG") as demo:
    gr.HTML(f"""
    <div class="app-header">
      <h1>MaritimeOpsRAG</h1>
      <p>운항 데이터 · 선급/IMO 문서 통합 질의 &nbsp;|&nbsp; {_status_line()}</p>
    </div>
    """)

    with gr.Row():
        llm_model = gr.Dropdown(
            choices=list(LLM_MODEL_CHOICES),
            value=DEFAULT_LLM_MODEL,
            label="사용 모델 (Ollama)",
            info=(
                "Gemma 4 12B가 기본입니다. 통합 질문에서는 라우팅과 답변에, "
                "전용 탭에서는 답변에 사용합니다."
            ),
            scale=1,
        )

    with gr.Tabs():
        with gr.Tab("통합 질문", id="integrated"):
            integrated_history = gr.State([])
            integrated_dialogue = gr.State({})
            integrated_last_question = gr.State("")

            gr.Markdown(
                "운항 DB와 문서 RAG 중 필요한 경로를 LLM이 자동으로 판단합니다. "
                "운항 정보와 규정을 함께 묻는 질문은 `hybrid`로 처리합니다."
            )
            integrated_route_mode = gr.Radio(
                choices=["자동 라우팅", "운항 DB 강제", "문서 RAG 강제"],
                value="자동 라우팅",
                label="데이터 경로",
            )
            integrated_examples = [
                "현재 운항 상태 알려줘",
                "최신 MEPC 회의 주요 내용을 정리해줘",
                "우리 CII랑 MEPC 규제 같이 알려줘",
            ]
            with gr.Row(elem_classes=["example-row"]):
                integrated_example_btns = [
                    gr.Button(text, size="sm") for text in integrated_examples
                ]
            integrated_answer = gr.HTML(value=build_answer_html("", ""))
            with gr.Row():
                integrated_input = gr.Textbox(
                    placeholder="운항·문서·혼합 질문을 입력하세요",
                    show_label=False,
                    scale=8,
                    container=False,
                )
                integrated_send = gr.Button(
                    "전송", variant="primary", scale=1, elem_classes=["send-btn"]
                )
            with gr.Row():
                retry_ops = gr.Button("운항만으로 다시", size="sm")
                retry_rag = gr.Button("문서만으로 다시", size="sm")
                retry_hyb = gr.Button("둘 다로 다시", size="sm")
            integrated_files = gr.File(
                label="생성된 보고서 / 표 crop", file_count="multiple"
            )

        with gr.Tab("문서 검색", id="documents"):
            document_history = gr.State([])
            document_dialogue = gr.State({})
            document_last_question = gr.State("")

            gr.Markdown(
                "선급 규정, IMO 회의자료, 본문과 표를 검색합니다. "
                "이 탭의 질문은 상위 라우터를 거치지 않고 항상 `rag`로 처리됩니다."
            )
            document_examples = [
                "과도한 부식의 정의는 무엇인가?",
                "구조 규칙에서 쓰는 tcorr 기호는 어떤 두께를 뜻하지?",
                "형상이 복잡하거나 한 개의 중량이 10톤을 넘는 주강품은 제품마다 시험재가 몇 개 필요한가?",
            ]
            with gr.Row(elem_classes=["example-row"]):
                document_example_btns = [
                    gr.Button(text, size="sm") for text in document_examples
                ]
            document_answer = gr.HTML(value=build_answer_html("", ""))
            with gr.Row():
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
                label="검색된 원본 표 / 생성 파일", file_count="multiple"
            )

        with gr.Tab("운항 정보", id="operations"):
            ops_history = gr.State([])
            ops_dialogue = gr.State({})
            ops_last_question = gr.State("")

            gr.Markdown(
                "현재·과거 항차, 속력, 연료, 배출량, CII와 보고서를 조회합니다. "
                "이 탭의 질문은 상위 라우터를 거치지 않고 항상 `ops`로 처리됩니다."
            )
            ops_examples = [
                "현재 운항 중인 항차 번호와 적재 상태를 알려줘.",
                "현재 Ballast 항차의 누적 운항거리는 몇 해리야?",
                "2026년 누적 잠정 CII attained, required와 등급은?",
            ]
            with gr.Row(elem_classes=["example-row"]):
                ops_example_btns = [
                    gr.Button(text, size="sm") for text in ops_examples
                ]
            ops_answer = gr.HTML(value=build_answer_html("", ""))
            with gr.Row():
                ops_input = gr.Textbox(
                    placeholder="항차·연료·배출량·CII 질문을 입력하세요",
                    show_label=False,
                    scale=8,
                    container=False,
                )
                ops_send = gr.Button(
                    "운항 조회", variant="primary", scale=1, elem_classes=["send-btn"]
                )
            ops_files = gr.File(label="생성된 운항 보고서", file_count="multiple")

    try:
        banner = rag_index_banner(sample_size=1500)
    except Exception as exc:
        banner = f"[RAG INDEX]\n(diagnostics unavailable: {exc})"
    gr.Markdown(
        f"<small>{html.escape(rag_status_message())}<br>"
        f"Text: `{DEFAULT_RAG_COLLECTION}` / Table: `{DEFAULT_TABLE_COLLECTION}`<br>"
        f"DB: `{OPS_DB_PATH}`</small>"
    )
    with gr.Accordion("RAG index diagnostics", open=False):
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


if __name__ == "__main__":
    print("=" * 56)
    print("  MaritimeOpsRAG")
    print(f"  {_status_line().replace('&nbsp;', ' ')}")
    print("=" * 56)
    if rag_index_ready():
        print("  문서 인덱스 준비됨. 첫 RAG 질문 때 모델을 로드합니다.")
    print("  브라우저: http://127.0.0.1:7860  (0.0.0.0 은 접속 주소가 아님)")
    print("=" * 56)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Base(primary_hue="red", neutral_hue="gray"),
        css=CUSTOM_CSS,
    )
