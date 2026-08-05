"""
Maritime Ops Agent — Ollama 공식 Agent Loop 패턴
Reference: https://docs.ollama.com/capabilities/tool-calling

구조:
  while True:
      response = LLM(messages, tools, tool_choice="auto")
      if response.tool_calls:
          execute tools → append results → continue
      else:
          return final answer  ← LLM이 완료 판단
"""
import json
from pathlib import Path
from typing import Generator

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OLLAMA_BASE_URL, OLLAMA_API_KEY, MODEL_NAME, CURRENT_DATE, VESSEL
from agent.tools import TOOL_SCHEMAS, dispatch_tool
from agent.briefing import build_answer_from_tools

try:
    from openai import OpenAI
    _client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
    _llm_available = True
except Exception:
    _llm_available = False

MAX_ITERATIONS = 8  # 무한루프 방지

SYSTEM_PROMPT = f"""You are a maritime operations AI assistant for vessel {VESSEL['name']} (IMO: {VESSEL['imo']}). Today: {CURRENT_DATE}.

Data source: sensor_log (1-hour intervals, ho_data Excel 기반).
선박은 Oil(VLSFO/LSMGO) + Gas(LNG) 병용. RPM·Loading·항만·기상은 원본 미제공.

Respond in Korean. Use paragraph form (줄글), NOT bullet lists or numbered sections.
Always state the time reference explicitly:
- 현재 = 현재 항차 시작일 ~ 현재
- 이전 = 직전 항차
- 올해 = 해당 연도 1/1 ~ {CURRENT_DATE}

[TOOL REQUIRED]
- 현재 운항 상태 → get_current_voyage_status
- 항차 분석(현재/이전/올해) → get_voyage_analysis(period=current|previous|ytd)
- CII 등급 → calculate_cii_rating
- 배출량 상세 → calculate_emissions
- Noon Report → generate_noon_report
- MRV Voyage Report → generate_mrv_voyage_report
- MRV Annual Report → generate_mrv_annual_report

[NO TOOL] Greetings, 기능 안내, 잡담은 직접 답변.
"""


def run_agent_sync(user_message: str, history: list) -> tuple[str, list, list, bool]:
    """
    동기 에이전트 — Ollama 공식 Agent Loop 패턴
    Returns: (answer, updated_history, generated_file_paths, show_map)
    """
    if not _llm_available:
        answer = _fallback_response(user_message)
        return answer, history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer},
        ], [], False

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": user_message})

    generated_files = []
    tool_results: list[tuple[str, dict, dict]] = []
    answer = ""
    show_map = False

    for iteration in range(MAX_ITERATIONS):
        try:
            response = _client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=2048,
            )
        except Exception as e:
            answer = f"[LLM 오류] {e}"
            break

        msg = response.choices[0].message

        if not msg.tool_calls:
            answer = msg.content or ""
            break

        messages.append(msg.model_dump(exclude_unset=True))

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments or "{}")

            try:
                result_str = dispatch_tool(fn_name, fn_args)
            except Exception as e:
                result_str = json.dumps({"error": str(e)})

            try:
                r = json.loads(result_str)
                tool_results.append((fn_name, fn_args, r))
                if "file_path" in r and Path(r["file_path"]).exists():
                    generated_files.append(r["file_path"])
            except Exception:
                pass

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })
    else:
        answer = answer or "최대 반복 횟수에 도달했습니다."

    # 브리핑/리포트 툴 호출 시 Python 포맷터로 줄글 답변 생성 (KPI 누락 방지)
    formatted = build_answer_from_tools(tool_results)
    if formatted:
        answer, show_map = formatted

    new_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": answer},
    ]
    return answer, new_history, list(dict.fromkeys(generated_files)), show_map


def run_agent(user_message: str, history: list) -> Generator[str, None, None]:
    """스트리밍 래퍼 (Gradio용)"""
    answer, _, _ = run_agent_sync(user_message, history)
    yield answer


def _fallback_response(user_message: str) -> str:
    return "Ollama 서버가 실행 중이지 않습니다. 터미널에서 'ollama serve'를 실행해주세요."
