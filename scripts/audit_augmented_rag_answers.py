"""Run expert-style QA over paraphrased maritime RAG questions.

This is deliberately not presented as a real domain-expert sign-off.  The
checks encode source-backed facts, unsafe conclusions and the agreed answer
shape so a reviewer can inspect the actual generated answer and evidence next
to every score.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("RAG_DEBUG_TRACE_STDERR", "0")

from services.llm_models import DEFAULT_LLM_MODEL  # noqa: E402
from services.orchestrator import handle_question  # noqa: E402

OUT_ROOT = ROOT / "data" / "processed" / "logs" / "expert_augmented_validation"
SECTIONS = (
    "핵심 요약",
    "선박 운항/업무 영향",
    "추후 확인 필요사항",
    "관련 선급 Rule / Guidance",
)


def case(
    case_id: str,
    category: str,
    question: str,
    must: list[list[str]],
    *,
    forbidden: list[str] | None = None,
    expected_source: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "category": category,
        "question": question,
        "must": must,
        "forbidden": forbidden or [],
        "expected_source": expected_source or [],
    }


# Each inner ``must`` list is an OR group; all groups should be represented.
# These are source-derived review anchors, not canned answers.
CASES = [
    case(
        "T01",
        "최신 동향 요약",
        "MEPC 84 환경규제 안건 중 GFI·연료 전과정 규제의 핵심 변화와 아직 확정되지 않은 부분을 알려줘.",
        [[r"SFCS|지속가능연료"], [r"Fuel Life Cycle Label"], [r"regulation 36|규칙 36"], [r"LCA|전과정"], [r"초안|미확정|추가 작업"]],
        expected_source=[r"MEPC 84/7/14"],
    ),
    case(
        "T02",
        "최신 동향 요약",
        "MSC 111 최종 주요 결정 가운데 절차사항을 빼고 선박에 영향이 큰 세 가지만 정리해줘.",
        [[r"MASS"], [r"수소"], [r"항법|navigation|전파항법"]],
        forbidden=[r"MASS Code.{0,20}(?:현재|이미).{0,12}(?:강제|mandatory)"],
        expected_source=[r"MSC 111/WP\.1"],
    ),
    case(
        "T03",
        "최신 동향 요약",
        "MSC 111에서 안전·항해·자율운항 관련 채택 또는 승인 결과를 구분해 요약해줘.",
        [[r"MASS"], [r"수소|대체연료"], [r"항법|navigation|전파항법"], [r"채택|승인"]],
        expected_source=[r"MSC 111/WP\.1"],
    ),
    case(
        "T04",
        "최신 동향 요약",
        "현재 문서 모음에서 가장 최근 MEPC의 SFCS, Fuel Life Cycle Label, LCA 논의를 요약해줘.",
        [[r"SFCS"], [r"Fuel Life Cycle Label"], [r"LCA"], [r"well-to-tank|WtT|원료.*공급"], [r"tank-to-wake|TtW|선박.*사용"]],
        expected_source=[r"MEPC 84/7/14"],
    ),
    case(
        "T05",
        "최신 동향 요약",
        "MEPC 84 환경규제 결과를 확정된 규정과 작업 중인 초안으로 나눠 설명해줘.",
        [[r"초안|검토|작업"], [r"최종.*아니|채택 결과.*아니|확정.*아니|단정.*수 없"]],
        forbidden=[r"MEPC 84/7/14.{0,40}(?:최종 채택|확정 규정)"],
        expected_source=[r"MEPC 84/7/14"],
    ),
    case(
        "E01",
        "환경규제 대응",
        "MEPC 84 문서 기준 DCS 데이터 적합성 확인과 Statement of Compliance 일정은 어떻게 되나?",
        [[r"DCS|규정 27|regulation 27"], [r"적합성"], [r"Statement of Compliance"], [r"5개월"]],
        expected_source=[r"MEPC 84/6/1"],
    ),
    case(
        "E02",
        "환경규제 대응",
        "IMO DCS 제출 데이터에서 어떤 누락·오류를 품질검증했으며 선사 보고업무에 무엇이 필요한가?",
        [[r"누락|미보고"], [r"오류|비현실|중복"], [r"GISIS|DCS"], [r"품질|검증"], [r"보고|제출"]],
        expected_source=[r"MEPC 84/6/1"],
    ),
    case(
        "E03",
        "환경규제 대응",
        "MSC 111의 수소연료 선박 안전지침 결정과 설계 승인 영향을 알려줘.",
        [[r"수소"], [r"임시 안전지침"], [r"합의|승인"], [r"설계"], [r"승인|위험성 평가"]],
        expected_source=[r"MSC 111/WP\.1"],
    ),
    case(
        "E04",
        "환경규제 대응",
        "MSC 111에서 암모니아 연료 관련 IGF/IGC Code 상태와 2026년 일정을 알려줘.",
        [[r"암모니아"], [r"IGF Code"], [r"IGC Code"], [r"2026"], [r"임시지침|interim"]],
        forbidden=[r"IGF Code 개정.{0,50}(?:2026년 7월|July 2026)"],
        expected_source=[r"MSC 111/WP\.1"],
    ),
    case(
        "E05",
        "환경규제 대응",
        "MSC 111 GHG Safety Working Group 결정은 확정 규정인가? 후속 확인사항까지 정리해줘.",
        [[r"GHG Safety Working Group"], [r"설립|establish"], [r"확정 규정.*아니|최종.*확인|초안|후속"]],
        forbidden=[r"GHG Safety Working Group.{0,40}(?:확정 규정|법적 의무)"],
        expected_source=[r"MSC 111/WP\.1"],
    ),
    case(
        "A01",
        "자율운항",
        "MSC 111에서 채택된 MASS Code는 지금 강제인가?",
        [[r"비강제|non-mandatory"], [r"현재.{0,40}(?:강제.*아니|의무.*아니)|현재 확인되는 Code는 비강제|mandatory.*확정.*아니"]],
        forbidden=[r"현재.{0,20}(?:강제|mandatory).{0,12}(?:적용|요구)"],
        expected_source=[r"MSC 111/WP\.1"],
    ),
    case(
        "A02",
        "자율운항",
        "mandatory MASS Code의 2030년과 2032년 일정은 확정인가 목표인가?",
        [[r"2030"], [r"2032"], [r"목표"], [r"야심|재검토|변경|단정.*수 없"]],
        forbidden=[r"2030년.{0,20}확정", r"2032년.{0,20}확정"],
        expected_source=[r"MSC 111/WP\.1"],
    ),
    case(
        "A03",
        "자율운항",
        "MASS Code experience-building phase는 왜 필요하며 향후 일정에 어떻게 연결되나?",
        [[r"경험축적|experience-building"], [r"설계|운항|승인|적용 사례"], [r"2030|mandatory"], [r"검토|개정|개발"]],
        expected_source=[r"MSC 111/WP\.1|MSC 111-5"],
    ),
    case(
        "A04",
        "자율운항",
        "비강제 MASS Code 채택과 강제 Code 로드맵을 구분해 설명해줘.",
        [[r"비강제"], [r"채택"], [r"2030"], [r"2032"], [r"목표|재검토"]],
        forbidden=[r"비강제 MASS Code.{0,30}(?:강제 의무|mandatory requirement)"],
        expected_source=[r"MSC 111/WP\.1"],
    ),
    case(
        "A05",
        "자율운항",
        "MSC 111 MASS 결정이 현 단계 선박 설계·운항 업무에 주는 직접 영향만 알려줘.",
        [[r"비강제|현 단계"], [r"설계|운항"], [r"경험|사례|검증 자료"], [r"의무.*아니|확정.*아니|해석.*안"]],
        expected_source=[r"MSC 111/WP\.1"],
    ),
    case(
        "R01",
        "단순 Rule",
        "DNV-CG-0264의 적용범위와 주요 자율·원격운항 notation을 알려줘.",
        [[r"DNV-CG-0264"], [r"적용범위|scope"], [r"notation"], [r"자율|autonomous"], [r"원격|remote"]],
        forbidden=[r"LR Notice"],
        expected_source=[r"DNV-CG-0264"],
    ),
    case(
        "R02",
        "단순 Rule",
        "DNV-CG-0264의 minimum risk condition 원칙을 근거 조항과 설명해줘.",
        [[r"DNV-CG-0264"], [r"minimum risk condition|최소 위험 상태"], [r"p\.|clause|조항|section"]],
        forbidden=[r"LR Notice"],
        expected_source=[r"DNV-CG-0264"],
    ),
    case(
        "R03",
        "단순 Rule",
        "LR Notice No.1에서 저인화점·dual fuel 기관의 crankcase 환기 요구를 찾아줘.",
        [[r"Notice No\.1"], [r"15\.8\.2"], [r"crankcase"], [r"환기|ventilation"], [r"dual.?fuel|저인화점"]],
        forbidden=[r"DNV-CG-0264"],
        expected_source=[r"Notice No\.1"],
    ),
    case(
        "R04",
        "단순 Rule",
        "LR 문서 중 대체연료 기관 안전성 평가와 관련된 Rule/Guidance를 알려줘.",
        [[r"LR|Lloyd"], [r"대체연료|저인화점|low.?flashpoint|dual.?fuel"], [r"안전성 평가|safety assessment|안전"]],
        forbidden=[r"DNV-CG-0264"],
        expected_source=[r"Notice No\.1"],
    ),
    case(
        "R05",
        "단순 Rule",
        "DNV Smart Vessel 문서와 autonomous/remote vessel guidance를 구분해 찾아줘.",
        [[r"Smart Functions|Smart Vessel"], [r"DNV-CG-0508"], [r"DNV-CG-0264"], [r"자율|autonomous"], [r"원격|remote"]],
        forbidden=[r"LR Notice", r"DNV-CG-0557"],
        expected_source=[r"DNV|Guide.*Smart|DNV-CG-0264"],
    ),
    case(
        "C01",
        "복합 규제·선급",
        "암모니아 연료선의 개념승인을 준비한다고 가정하고, MSC 111 논의와 보유 선급 규정을 근거로 설계 검토 체크리스트와 미확정 규제를 작성해줘.",
        [
            [r"암모니아"],
            [r"임시지침|interim"],
            [r"IGC Code"],
            [r"Fuel ready|Ammonia Ready|Gas fuelled ammonia"],
            [r"탱크|배치"],
            [r"환기|검지|비상정지|위험성 평가"],
            [r"미확정|향후|추가 개정|적용범위"],
        ],
        forbidden=[r"IGF Code 개정.{0,30}(?:2026년 7월|July 2026)"],
        expected_source=[r"MSC 111-WP\.1", r"DNV-RU-SHIP-Pt6|Circular \(K\) Total_2026"],
    ),
    case(
        "C02",
        "복합 규제·선급",
        "수소 연료선의 기본설계를 검토하려고 한다. MSC 111 수소 안전지침과 DNV·KR 선급 규정을 함께 근거로 승인 체크리스트와 아직 확정되지 않은 사항을 정리해줘.",
        [
            [r"수소"],
            [r"임시.*지침|interim.*guideline"],
            [r"DNV|KR"],
            [r"탱크|배치|연료공급"],
            [r"위험|환기|검지|비상"],
            [r"미확정|향후|추가"],
        ],
        expected_source=[r"MSC 111-WP\.1", r"DNV|KR|선급"],
    ),
    case(
        "C03",
        "복합 규제·선급",
        "자율운항선 개념승인을 준비할 때 MSC 111의 MASS Code 결정·mandatory 일정과 DNV-CG-0264 및 ABS 규정을 근거로 설계 검토 체크리스트와 규제 공백을 작성해줘.",
        [
            [r"MASS Code"],
            [r"비강제|non-mandatory"],
            [r"2030"],
            [r"2032"],
            [r"DNV-CG-0264"],
            [r"ABS"],
            [r"위험|검증|원격|운항"],
        ],
        forbidden=[r"2030년.{0,20}확정", r"2032년.{0,20}확정"],
        expected_source=[r"MSC 111-WP\.1", r"DNV-CG-0264", r"RequirementsforAutonomous"],
    ),
    case(
        "C04",
        "복합 규제·선급",
        "암모니아 화물을 연료로 쓰는 가스운반선과 암모니아를 연료로만 싣는 선박의 차이를 MSC 111 논의와 DNV Fuel ready·Gas fuelled ammonia 규정으로 비교하고, 개념설계 확인사항을 정리해줘.",
        [
            [r"화물.*연료|cargo as fuel"],
            [r"연료로만|solely.*fuel|다른 가스운반선"],
            [r"Fuel ready"],
            [r"Gas fuelled ammonia"],
            [r"적용범위|적용 범위"],
            [r"설계|탱크|배치|안전"],
        ],
        expected_source=[r"MSC 111-WP\.1", r"DNV-RU-SHIP-Pt6"],
    ),
    case(
        "C05",
        "복합 규제·선급",
        "MEPC 84의 연료 전과정·GFI 논의와 보유 선급의 대체연료 Ready 규정을 함께 고려해 개념설계 검토 체크리스트와 미확정 규제를 정리해줘.",
        [
            [r"GFI"],
            [r"LCA|전과정"],
            [r"Ready|선급"],
            [r"설계|탱크|연료공급|배치"],
            [r"미확정|초안|추가 작업"],
        ],
        expected_source=[r"MEPC 84", r"DNV|KR|ABS|LR"],
    ),
]


def _section_bullets(answer: str) -> dict[str, list[str]]:
    output = {section: [] for section in SECTIONS}
    current = ""
    for raw in answer.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            current = next((section for section in SECTIONS if section in line), "")
        elif current and line.startswith(("-", "*")):
            output[current].append(line)
    return output


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, re.I | re.S) for pattern in patterns)


def audit_answer(answer: str, item: dict[str, Any], evidence: list[dict]) -> dict[str, Any]:
    sections = _section_bullets(answer)
    bullets = [line for values in sections.values() for line in values]
    uncited = [
        line
        for line in bullets
        if not re.search(r"\[\d+\]", line)
        and "확인되지" not in line
        and "식별되지" not in line
        and "해당하지" not in line
    ]
    missing_groups = [
        alternatives
        for alternatives in item["must"]
        if not _matches_any(answer, alternatives)
    ]
    forbidden_hits = []
    for pattern in item["forbidden"]:
        matches = list(re.finditer(pattern, answer, re.I | re.S))
        unsafe = False
        for match in matches:
            context = answer[match.start() : min(len(answer), match.end() + 36)]
            # Do not count an explicit negation ("확정일이 아니라", "확정되지
            # 않음") as the very overclaim this check is meant to catch.
            if re.search(r"확정(?:일)?(?:이)?\s*아니|확정되지\s*않", context):
                continue
            unsafe = True
            break
        if unsafe:
            forbidden_hits.append(pattern)
    source_text = answer + "\n" + json.dumps(evidence, ensure_ascii=False)
    missing_sources = [
        pattern
        for pattern in item["expected_source"]
        if not re.search(pattern, source_text, re.I | re.S)
    ]
    missing_sections = [section for section, values in sections.items() if not values and section not in answer]

    category_budget = (
        10
        if item["category"] in {"최신 동향 요약", "복합 규제·선급"}
        else 3
        if item["category"] == "단순 Rule"
        else 7
    )
    structure_pass = not missing_sections and len(bullets) <= category_budget
    grounding_pass = bool(evidence) and not uncited
    coverage_ratio = round((len(item["must"]) - len(missing_groups)) / max(1, len(item["must"])), 2)
    safety_pass = not forbidden_hits
    source_pass = not missing_sources
    passed = structure_pass and grounding_pass and safety_pass and source_pass and coverage_ratio == 1.0
    return {
        "passed": passed,
        "coverage_ratio": coverage_ratio,
        "missing_fact_groups": missing_groups,
        "forbidden_hits": forbidden_hits,
        "missing_sources": missing_sources,
        "uncited_bullets": uncited,
        "missing_sections": missing_sections,
        "total_bullets": len(bullets),
        "bullet_budget": category_budget,
        "structure_pass": structure_pass,
        "grounding_pass": grounding_pass,
        "safety_pass": safety_pass,
        "source_pass": source_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--latency-mode", choices=("fast", "accurate"), default="accurate")
    parser.add_argument("--only", help="comma-separated case ids such as A02,R03")
    parser.add_argument("--cold", action="store_true", help="UI 사전 워밍을 생략한다")
    args = parser.parse_args()

    only_ids = {
        value.strip().upper()
        for value in str(args.only or "").split(",")
        if value.strip()
    }
    selected = [item for item in CASES if not only_ids or item["id"] in only_ids]
    if not selected:
        raise SystemExit(f"unknown case: {args.only}")

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = OUT_ROOT / f"{stamp}_{args.latency_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    if not args.cold:
        from services.rag_service import warmup_rag_resources

        warmup_rag_resources()

    for index, item in enumerate(selected, 1):
        started = time.perf_counter()
        result = handle_question(
            item["question"],
            [],
            use_llm_router=True,
            rag_latency_mode=args.latency_mode,
            llm_model=args.model,
        )
        elapsed = round(time.perf_counter() - started, 2)
        answer = str(result.get("answer") or "")
        evidence = list(result.get("evidence_table") or [])
        meta = dict(result.get("meta") or {})
        generation = meta.get("answer_generation") or {}
        audit = audit_answer(answer, item, evidence)
        if args.latency_mode == "fast" and elapsed > 10.0:
            audit["passed"] = False
            audit.setdefault("latency_failures", []).append(
                f"Fast 응답 {elapsed}s > 10초"
            )
        record = {
            "index": index,
            **item,
            "seconds": elapsed,
            "route": (result.get("route") or {}).get("route"),
            "answer_mode": meta.get("answer_mode"),
            "answer_generation": generation,
            "scaffold_synthesis_debug": meta.get("scaffold_synthesis_debug"),
            "answer": answer,
            "evidence_table": evidence,
            **audit,
        }
        records.append(record)
        print(
            f"[{index:02d}/{len(selected):02d}] {item['id']} {elapsed:5.2f}s "
            f"coverage={audit['coverage_ratio']:.2f} "
            f"generation={generation.get('answer_source', 'unknown')} "
            f"{'PASS' if audit['passed'] else 'REVIEW'}",
            flush=True,
        )

    (out_dir / "records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# 증강 질문 전문가형 1차 검수",
        "",
        f"- mode: {args.latency_mode}",
        f"- model: {args.model}",
        "- 주의: 실제 해사 규정 전문가의 공식 검수가 아니라, 출처 기반 검수 기준을 적용한 내부 1차 평가입니다.",
        "",
    ]
    for rec in records:
        report.extend(
            [
                f"## {rec['id']} · {rec['category']} · {'PASS' if rec['passed'] else 'REVIEW'}",
                "",
                f"질문: {rec['question']}",
                f"시간: {rec['seconds']}s · route={rec['route']} · answer_mode={rec['answer_mode']}",
                f"생성: {json.dumps(rec['answer_generation'], ensure_ascii=False)}",
                f"필수사실 coverage: {rec['coverage_ratio']} · 누락={rec['missing_fact_groups']}",
                f"금지 단정: {rec['forbidden_hits']} · 출처 누락: {rec['missing_sources']}",
                f"근거/형식: grounding={rec['grounding_pass']} structure={rec['structure_pass']} bullets={rec['total_bullets']}/{rec['bullet_budget']}",
                "",
                rec["answer"],
                "",
                "### 표시 근거",
                "",
                "```json",
                json.dumps(rec["evidence_table"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    passed = sum(bool(record["passed"]) for record in records)
    print(f"wrote {out_dir} ({passed}/{len(records)} automated pass)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
