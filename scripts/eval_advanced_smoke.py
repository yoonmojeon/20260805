"""Run a compact on-prem Advanced UI-path smoke suite."""
from __future__ import annotations

import contextlib
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from services.rag_service import run_rag_query  # noqa: E402


QUESTIONS = [
    "DNV-CP-0399의 형식승인(TA)은 어떤 정격의 전력 케이블과 제어·계측 회로용 케이블에 적용되나요?",
    "MEPC 회의자료에 따르면, 2026년에 발행될 예정인 해양 플라스틱 쓰레기 관련 자료에는 어떤 것들이 포함되어 있습니까?",
    "MSC 111에서 MASS Code와 관련된 핵심 결정사항을 요약하고, 향후 mandatory code 일정까지 정리해줘.",
    "RSTH 12·22·23·24 관을 확관한 후 관 끝의 허용 바깥지름은 원래 관 바깥지름의 몇 배인가?",
    "재화중량이 10만 톤 초과 15만 톤 이하인 선박의 안전사용하중은 몇 톤인가?",
]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indices",
        nargs="*",
        type=int,
        default=[],
        help="1-based question indices; omitted means all",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for a disposable test artifact",
    )
    parser.add_argument(
        "--question",
        action="append",
        default=[],
        help="Run an additional literal question through the UI-equivalent path",
    )
    args = parser.parse_args()
    selected_questions = [
        question
        for index, question in enumerate(QUESTIONS, 1)
        if not args.indices or index in set(args.indices)
    ]
    selected_questions.extend(str(value) for value in args.question if str(value).strip())
    rows = []
    for index, question in enumerate(selected_questions, 1):
        started = time.perf_counter()
        debug = io.StringIO()
        with contextlib.redirect_stdout(debug), contextlib.redirect_stderr(debug):
            result = run_rag_query(
                question,
                latency_mode="advanced",
                retrieval_mode=None,
                llm_model="gemma4:12b",
            )
        meta = result.get("meta") or {}
        rows.append(
            {
                "index": index,
                "question": question,
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "answer": result.get("answer") or "",
                "retrieval_mode": meta.get("retrieval_mode"),
                "advanced_retrieval": meta.get("advanced_retrieval") or {},
                "advanced_rerank": meta.get("advanced_rerank") or {},
                "advanced_confidence": meta.get("advanced_confidence") or {},
                "advanced_answer_review": meta.get("advanced_answer_review") or {},
                "ready": result.get("ready"),
                "error": result.get("error") or meta.get("answer_error") or "",
            }
        )
        print(
            f"[{index}/{len(selected_questions)}] {rows[-1]['elapsed_seconds']}s "
            f"mode={rows[-1]['retrieval_mode']} "
            f"rerank={bool(rows[-1]['advanced_rerank'].get('used'))}"
        )
    rendered = json.dumps(rows, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
