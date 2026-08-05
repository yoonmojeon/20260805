"""
Maritime Ops Agent — Gradio UI
실행: python app.py
"""
import html
import re
import sys
from pathlib import Path

import gradio as gr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import VESSEL, MODEL_NAME, CURRENT_DATE
from agent.maritime_agent import run_agent_sync, _llm_available
from agent.tools import render_voyage_map


def _paragraphs_to_html(text: str) -> str:
    if not text:
        return "<p class='answer-empty'>질문을 입력하세요. 운항 브리핑·항차 분석·보고서 생성을 지원합니다.</p>"
    parts = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    return "".join(f"<p class='answer-p'>{html.escape(p)}</p>" for p in parts)


def build_answer_html(query: str, answer: str, map_html: str = "") -> str:
    q = html.escape(query.strip()) if query.strip() else ""
    body = _paragraphs_to_html(answer)
    query_block = f"<div class='query-box'><b>질문</b> {q}</div>" if q else ""
    map_block = (
        f"<div class='map-box'><div class='map-title'>항차 이동 경로</div>{map_html}</div>"
        if map_html else ""
    )
    return f"""
    <div class="answer-wrap">
      {query_block}
      <div class="answer-body">{body}</div>
      {map_block}
    </div>"""


def chat_fn(user_msg: str, history: list):
    empty = build_answer_html("", "")
    if not user_msg.strip():
        return history, empty, []
    answer, new_history, files, show_map = run_agent_sync(user_msg, list(history or []))
    file_paths = [f for f in files if Path(f).exists()]
    map_html = render_voyage_map() if show_map else ""
    return new_history, build_answer_html(user_msg, answer, map_html), file_paths or []


llm_status = (
    f"LLM 연결됨: `{MODEL_NAME}`"
    if _llm_available
    else "Ollama 미연결"
)

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');
body, .gradio-container {
  font-family: 'Noto Sans KR', 'Segoe UI', sans-serif !important;
  background: #fff !important;
  color: #262730 !important;
}
.gradio-container { max-width: 960px !important; margin: 0 auto !important; }
.app-header { padding: 20px 0 12px; border-bottom: 1px solid #e6e8ec; margin-bottom: 16px; }
.app-header h1 { margin: 0; font-size: 26px; font-weight: 700; }
.app-header p { margin: 4px 0 0; color: #808495; font-size: 13px; }
.example-row button { font-size: 13px !important; border-radius: 20px !important; }
.query-box { background: #f8f9fb; border-radius: 8px; padding: 10px 14px; margin-bottom: 16px; font-size: 14px; }
.query-box b { color: #ff4b4b; margin-right: 6px; }
.answer-body .answer-p { margin: 0 0 14px; font-size: 15px; line-height: 1.85; color: #31333f; text-align: justify; }
.answer-empty { color: #808495; font-size: 14px; }
.map-box { margin-top: 20px; border: 1px solid #e6e8ec; border-radius: 8px; overflow: hidden; }
.map-title { padding: 8px 12px; background: #f8f9fb; font-size: 13px; font-weight: 600; color: #555; }
.send-btn button { background: #ff4b4b !important; border-color: #ff4b4b !important; font-weight: 600 !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="Maritime Ops Agent") as demo:
    gr.HTML(f"""
    <div class="app-header">
      <h1>Maritime Ops Agent</h1>
      <p>운항 브리핑 · 항차 분석 · 보고서 생성 &nbsp;|&nbsp;
         {VESSEL['name']} &nbsp;|&nbsp; 기준일 {CURRENT_DATE} &nbsp;|&nbsp; {llm_status}</p>
    </div>
    """)

    history_state = gr.State([])

    gr.Markdown(
        "**지원:** 현재 운항 상태 · 현재/이전/올해 항차 분석 · "
        "Noon Report · MRV Voyage/Annual Report (Word 다운로드)"
    )
    with gr.Row(elem_classes=["example-row"]):
        ex1 = gr.Button("현재 운항 상태 알려줘", size="sm")
        ex2 = gr.Button("이전 항차 분석해줘", size="sm")
        ex3 = gr.Button("올해 연간 실적 보여줘", size="sm")
        ex4 = gr.Button("Noon Report 생성해줘", size="sm")
        ex5 = gr.Button("MRV Voyage Report 만들어줘", size="sm")
        ex6 = gr.Button("올해 CII 등급을 알려줘", size="sm")
        ex7 = gr.Button("연간 MRV 보고서를 생성해줘", size="sm")

    answer_html = gr.HTML(value=build_answer_html("", ""))

    with gr.Row():
        user_input = gr.Textbox(
            placeholder="질문을 입력하세요 (예: 현재 운항 상태 알려줘)",
            show_label=False,
            scale=8,
            container=False,
        )
        send_btn = gr.Button("전송", variant="primary", scale=1, elem_classes=["send-btn"])

    generated_files = gr.File(label="생성된 보고서 다운로드", file_count="multiple")

    for btn, text in [
        (ex1, "현재 운항 상태 알려줘"),
        (ex2, "이전 항차 분석해줘"),
        (ex3, "올해 연간 실적 보여줘"),
        (ex4, "Noon Report 생성해줘"),
        (ex5, "MRV Voyage Report 만들어줘"),
        (ex6, "올해 CII 등급을 알려줘"),
        (ex7, "연간 MRV 보고서를 생성해줘"),
    ]:
        btn.click(lambda t=text: t, outputs=user_input)

    send_btn.click(
        chat_fn,
        inputs=[user_input, history_state],
        outputs=[history_state, answer_html, generated_files],
    ).then(lambda: "", outputs=user_input)

    user_input.submit(
        chat_fn,
        inputs=[user_input, history_state],
        outputs=[history_state, answer_html, generated_files],
    ).then(lambda: "", outputs=user_input)


if __name__ == "__main__":
    print(f"\n{'='*55}")
    print("  Maritime Ops Agent")
    print(f"  모델: {MODEL_NAME} | 기준일: {CURRENT_DATE}")
    print(f"{'='*55}\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Base(primary_hue="red", neutral_hue="gray"),
        css=CUSTOM_CSS,
    )
