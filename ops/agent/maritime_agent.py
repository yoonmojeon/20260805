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
from prompts.ops import build_ops_system_prompt

try:
    from openai import OpenAI
    _client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
    _llm_available = True
except Exception:
    _llm_available = False

MAX_ITERATIONS = 8  # 무한루프 방지

SYSTEM_PROMPT = build_ops_system_prompt(
    vessel_name=VESSEL["name"],
    imo=str(VESSEL["imo"]),
    today=CURRENT_DATE,
)


def run_agent_sync(
    user_message: str,
    history: list,
    *,
    model: str | None = None,
) -> tuple[str, list, list, bool]:
    """
    동기 에이전트 — Ollama 공식 Agent Loop 패턴
    Returns: (answer, updated_history, generated_file_paths, show_map)
    """
    llm_model = (model or MODEL_NAME).strip() or MODEL_NAME
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
                model=llm_model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=4096,
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

    # LLM chooses the tool; structured DB results are rendered deterministically
    # so figures, scope labels and the final sentence cannot be truncated or
    # paraphrased into a different meaning.
    formatted = build_answer_from_tools(tool_results)
    if formatted:
        structured_answer, formatted_show_map = formatted
        show_map = show_map or formatted_show_map
        answer = structured_answer

    new_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": answer},
    ]
    return answer, new_history, list(dict.fromkeys(generated_files)), show_map


def run_agent(user_message: str, history: list) -> Generator[str, None, None]:
    """스트리밍 래퍼 (Gradio용)"""
    answer, _, _, _ = run_agent_sync(user_message, history)
    yield answer


def _fallback_response(user_message: str) -> str:
    return "Ollama 서버가 실행 중이지 않습니다. 터미널에서 'ollama serve'를 실행해주세요."
