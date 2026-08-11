#!/usr/bin/env python3
"""Summarize the three-model balanced 100-question evaluation."""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval"
QUESTIONS = EVAL / "balanced_quality_100.jsonl"
OUTPUT_JSON = EVAL / "balanced_quality_100_comparison.json"
OUTPUT_MD = EVAL / "balanced_quality_100_report.md"

MODEL_FILES = {
    "Gemma 4 12B": EVAL / "balanced_quality_100_gemma4_12b.json",
    "Llama 3.1 8B": EVAL / "balanced_quality_100_llama3.1_8b.json",
    "Mistral Nemo 12B": EVAL / "balanced_quality_100_mistral-nemo_12b.json",
}
TYPES = ("ops", "text", "table", "hybrid")
TEXT_GROUPS = ("KR", "PILOT", "FULL", "MEETING", "HIER")


def percentile(values: list[float], ratio: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * ratio
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def strict_pass(row: dict) -> bool:
    return bool(row["route_ok"] and row["needle"] == "PASS" and row["quality"] != "BAD")


def stats(rows: list[dict]) -> dict:
    durations = [float(row["dt"]) for row in rows]
    return {
        "n": len(rows),
        "strict_pass": sum(strict_pass(row) for row in rows),
        "strict_rate": round(sum(strict_pass(row) for row in rows) / len(rows), 4),
        "partial": sum(row["needle"] == "WEAK" for row in rows),
        "fail": sum(row["needle"] == "FAIL" for row in rows),
        "route_ok": sum(bool(row["route_ok"]) for row in rows),
        "avg_s": round(statistics.mean(durations), 2),
        "median_s": round(statistics.median(durations), 2),
        "p95_s": round(percentile(durations, 0.95), 2),
        "quality": dict(Counter(row["quality"] for row in rows)),
    }


questions = [
    json.loads(line)
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
question_ids = [row["id"] for row in questions]
assert Counter(row["type"] for row in questions) == {
    "ops": 25,
    "text": 50,
    "table": 15,
    "hybrid": 10,
}

model_rows: dict[str, list[dict]] = {}
comparison: dict = {"dataset": str(QUESTIONS.relative_to(ROOT)), "models": {}}
for model, path in MODEL_FILES.items():
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["results"]
    assert [row["id"] for row in rows] == question_ids, f"result order mismatch: {model}"
    model_rows[model] = rows
    comparison["models"][model] = {
        "overall": stats(rows),
        "types": {kind: stats([row for row in rows if row["type"] == kind]) for kind in TYPES},
        "text_groups": {
            group: stats(
                [row for row in rows if row["id"].startswith(f"B100_TEXT_{group}_")]
            )
            for group in TEXT_GROUPS
        },
        "mean_router_latency_ms": payload["summary"].get("mean_router_latency_ms"),
    }

maps = {model: {row["id"]: row for row in rows} for model, rows in model_rows.items()}
shared_non_strict = [
    question_id
    for question_id in question_ids
    if all(not strict_pass(maps[model][question_id]) for model in MODEL_FILES)
]
comparison["shared_non_strict"] = {
    "n": len(shared_non_strict),
    "by_type": dict(Counter(maps["Gemma 4 12B"][qid]["type"] for qid in shared_non_strict)),
    "ids": shared_non_strict,
}
OUTPUT_JSON.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")


def percent(value: float) -> str:
    return f"{value * 100:.0f}%"


def excerpt(text: str, limit: int = 240) -> str:
    clean = " ".join((text or "").split()).replace("|", "\\|")
    return clean if len(clean) <= limit else clean[:limit].rstrip() + "…"


lines = [
    "# 3개 Ollama 모델 균형형 100문항 평가",
    "",
    "- 평가일: 2026-08-11",
    "- 실행 환경: NVIDIA GeForce RTX 5080 16GB, Intel Core Ultra 7 265KF, RAM 96GB (측정값은 이 PC 기준)",
    "- 문항 구성: 운항 DB 25, 텍스트 문서 50, 표 15, 운항+문서 혼합 10",
    "- 라우팅: LLM primary router, 동일 질문·동일 순서, 모델별 단독 순차 실행",
    "- 엄격 통과: 예상 라우트가 맞고 정답 핵심어를 모두 포함하며 BAD 답변이 아닌 경우",
    "- 주의: 핵심어 기반 자동 채점이므로 수치는 완전한 의미 정확도가 아닌 참고용 상한으로 해석해야 한다.",
    "",
    "## 전체 비교",
    "",
    "| 모델 | 엄격 통과 | 부분정답 | 실패 | 평균 | 중앙값 | P95 | 라우트 정확도 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for model in MODEL_FILES:
    s = comparison["models"][model]["overall"]
    lines.append(
        f"| {model} | {s['strict_pass']}/100 ({percent(s['strict_rate'])}) | "
        f"{s['partial']} | {s['fail']} | {s['avg_s']:.2f}초 | {s['median_s']:.2f}초 | "
        f"{s['p95_s']:.2f}초 | {s['route_ok']}/100 |"
    )

lines += ["", "## 유형별 엄격 통과와 평균 응답속도", ""]
for kind, label in (("ops", "운항 DB"), ("text", "텍스트"), ("table", "표"), ("hybrid", "혼합")):
    lines += [
        f"### {label}",
        "",
        "| 모델 | 엄격 통과 | 부분정답 | 실패 | 평균 | 중앙값 | P95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_FILES:
        s = comparison["models"][model]["types"][kind]
        lines.append(
            f"| {model} | {s['strict_pass']}/{s['n']} ({percent(s['strict_rate'])}) | "
            f"{s['partial']} | {s['fail']} | {s['avg_s']:.2f}초 | "
            f"{s['median_s']:.2f}초 | {s['p95_s']:.2f}초 |"
        )
    lines.append("")

lines += [
    "## 텍스트 문서군별 엄격 통과",
    "",
    "| 문서군 | 문항 | Gemma | Llama | Mistral |",
    "|---|---:|---:|---:|---:|",
]
for group, label in (
    ("KR", "KR 제1편 세부 정의·조항"),
    ("PILOT", "MSC·MEPC/DNV/LR 검증 질문"),
    ("FULL", "전체 코퍼스 규칙·회의"),
    ("MEETING", "MSC 회의 결정"),
    ("HIER", "대표 계층검색 질문"),
):
    cells = []
    n = 0
    for model in MODEL_FILES:
        s = comparison["models"][model]["text_groups"][group]
        n = s["n"]
        cells.append(f"{s['strict_pass']}/{n}")
    lines.append(f"| {label} | {n} | " + " | ".join(cells) + " |")

lines += [
    "",
    "## 판단",
    "",
    "- **정확도 우선 기본 모델은 Gemma 4 12B가 맞다.** 전체 63%, 텍스트 50%, 혼합 100%로 세 모델 중 가장 높았다.",
    "- **속도 우선 선택은 Llama 3.1 8B다.** 평균 5.84초로 Gemma보다 약 46% 빠르지만 전체 엄격 통과가 12%p 낮다.",
    "- **Mistral Nemo 12B는 이번 데이터에서 추천하기 어렵다.** 평균 6.52초로 Llama보다 느리고 전체 엄격 통과도 44%로 가장 낮았다.",
    "- **텍스트는 아직 전반적으로 '잘한다'고 판정하기 어렵다.** 대표 계층검색·회의질문은 강하지만 KR 세부 조항은 Gemma 7/25, Llama 3/25, Mistral 6/25에 그쳤다.",
    "- **표 전체검색은 여전히 병목이다.** 파일·페이지가 명확한 표는 잘 맞지만, 전체 표에서 셀을 찾는 질문은 세 모델 모두 반복 실패했다.",
    "- **세 모델 공통 미통과는 33문항**이며 텍스트 24, 표 7, 운항 2였다. 모델 교체보다 검색 대상 분류와 문서/페이지 필터 개선이 우선이다.",
    "",
    "## 실제 질문·모범답안·답변 예시",
    "",
]

example_ids = (
    "B100_TEXT_KR_KR1_Q002",
    "B100_TEXT_FULL_RULE_LR_01",
    "B100_TEXT_HIER_fresh_def_01",
)
for idx, qid in enumerate(example_ids, 1):
    first = maps["Gemma 4 12B"][qid]
    lines += [f"### 예시 {idx}", "", f"- 질문: {first['question']}", f"- 모범답안: {first['gold']}", ""]
    lines += ["| 모델 | 판정 | 실제 답변 발췌 |", "|---|---|---|"]
    for model in MODEL_FILES:
        row = maps[model][qid]
        lines.append(
            f"| {model} | {row['needle']}/{row['quality']} · {row['dt']:.2f}초 | {excerpt(row['answer'])} |"
        )
    lines.append("")

lines += [
    "> 수동 점검 메모: 예시 3의 Llama 답변은 자동으로 PASS 처리됐지만 핵심인 `75%` 정의를 답하지 못했다. 따라서 자동 통과율은 실제 의미 정확도를 다소 높게 잡을 수 있다.",
    "",
]

lines += [
    "## 우선 개선사항",
    "",
    "1. 일반 규정 질문이 표 검색으로 잘못 내려가는 TEXT/TABLE 하위 분류를 보정한다.",
    "2. `KR 제1편`, 조항 번호, 문서명, 페이지 표현을 메타데이터 필터로 먼저 고정한 뒤 벡터+BM25 검색한다.",
    "3. 표 질문은 파일명·페이지가 있으면 직접 crop/row를 조회하고, 없으면 표 캡션→행/열 헤더의 2단계 검색을 사용한다.",
    "4. 운항 도구에서 현재/직전/YTD scope를 구조화 필드로 강제하고, 날짜 범위·단위·정밀도를 답변 템플릿에 포함한다.",
    "5. 혼합 답변은 Gemma 품질을 유지하되 운항 결과를 LLM이 재해석하지 않고 원 수치를 그대로 삽입해 지연과 수치 환각을 줄인다.",
    "",
    "전체 질문별 실제 답변·시간·라우트·채점 근거는 모델별 JSON 결과에 보존되어 있다.",
]
OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"json": str(OUTPUT_JSON), "markdown": str(OUTPUT_MD)}, ensure_ascii=False))
