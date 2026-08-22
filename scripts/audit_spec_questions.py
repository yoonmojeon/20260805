"""Audit the demo question categories against the agreed answer contract.

The contract (docs 1.1/1.2) fixes, per question category, how many bullets the
answer may carry, that every bullet cites evidence, and that the four sections
appear in order. This script runs the questions through the same stack the UI
uses and reports where an answer breaks the contract.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.llm_models import DEFAULT_LLM_MODEL  # noqa: E402
from services.orchestrator import handle_question  # noqa: E402

OUT_ROOT = ROOT / "data" / "processed" / "logs" / "spec_audit"

SECTIONS = ("핵심 요약", "선박 운항/업무 영향", "추후 확인 필요사항", "관련 선급 Rule / Guidance")

# (category, question, section-1 min, section-1 max, whole-answer bullet budget)
CASES: list[tuple[str, str, int, int, int]] = [
    ("최신 동향 요약", "환경규제 대응과 관련된 최신 MEPC 회의 주요 내용을 정리해줘.", 3, 10, 10),
    ("최신 동향 요약", "MSC 111의 주요 결과를 3개 항목으로 요약해줘.", 3, 3, 10),
    ("환경규제 대응", "최신 MEPC 회의에서 선박 운항 및 규제 보고에 직접 영향을 주는 사항을 정리해줘.", 3, 7, 7),
    ("환경규제 대응", "MSC 111에서 대체연료·GHG 안전규제와 관련된 논의 및 결론을 요약해줘.", 3, 7, 7),
    ("자율운항", "MSC 111에서 MASS Code와 관련된 핵심 결정사항을 요약하고, 향후 mandatory code 일정까지 정리해줘.", 3, 7, 7),
    ("단순 Rule", "DNV에서 자율운항 또는 Smart Vessel 관련 Rule/Guidance를 찾아줘.", 2, 3, 3),
    ("단순 Rule", "LR에서 대체연료 관련 Rule/Guidance를 찾아줘.", 2, 3, 3),
]

# Structural checks alone allowed fluent but materially wrong answers to pass
# (for example an ammonia/IGC amendment described as an IGF amendment).  These
# anchors are deliberately outcome-level rather than exact prepared answers:
# alternative Korean/English wording is accepted, but the governing facts for
# each agreed demo question must be present and known wrong substitutions must
# be absent.
SEMANTIC_CONTRACTS: dict[str, dict[str, object]] = {
    CASES[0][1]: {
        "required": (
            (r"\bGFI\b|연료\s*집약도", "GFI 규제"),
            (r"\bSFCS\b|Fuel\s+Lifecycle\s+Label|연료\s*수명주기\s*라벨", "SFCS/FLL"),
            (r"\bLCA\b|Well[- ]to[- ]Tank|\bWtT\b|\bTtW\b", "LCA 경계"),
        ),
    },
    CASES[1][1]: {
        "required": (
            (r"MASS\s+Code", "MASS Code"),
            (r"수소.{0,80}(?:지침|승인)|hydrogen", "수소연료 안전지침"),
            (r"무선항행|전파항법|radio\s+navigation|항행\s*시스템", "무선항행 결과"),
        ),
    },
    CASES[2][1]: {
        "required": (
            (r"(?<![A-Za-z0-9])DCS(?![A-Za-z0-9])|연료유\s*소비|Annex\s+VI.{0,40}규정\s*27", "IMO DCS"),
            (r"5\s*개월|five\s+months", "5개월 확인 기한"),
            (r"GISIS|데이터.{0,40}(?:오류|품질|누락)", "보고 데이터 품질"),
        ),
    },
    CASES[3][1]: {
        "required": (
            (r"수소.{0,100}승인|승인.{0,100}수소|hydrogen.{0,80}approved", "수소 지침 승인"),
            (r"GHG\s+Safety\s+Working\s+Group", "GHG Safety Working Group"),
            (r"암모니아.{0,160}\bIGC\s+Code\b|\bIGC\s+Code\b.{0,160}암모니아", "암모니아 IGC Code"),
        ),
        "forbidden": (
            (r"암모니아.{0,180}\bIGF\s+Code\s+개정|\bIGF\s+Code\s+개정.{0,180}암모니아", "암모니아 IGC 개정을 IGF 개정으로 표기"),
        ),
    },
    CASES[4][1]: {
        "required": (
            (r"비강제|non[- ]mandatory", "현 단계 비강제 Code"),
            (r"2030.{0,60}(?:채택|adoption)|(?:채택|adoption).{0,60}2030", "2030 채택 목표"),
            (r"2032.{0,60}(?:발효|entry)|(?:발효|entry).{0,60}2032", "2032 발효 목표"),
        ),
    },
    CASES[5][1]: {
        "required": (
            (r"DNV-CG-0264", "DNV-CG-0264"),
            (r"DNV-CG-0508", "DNV-CG-0508"),
        ),
        "section4_required": (
            (r"DNV-CG-0264", "section 4의 DNV-CG-0264"),
            (r"DNV-CG-0508", "section 4의 DNV-CG-0508"),
        ),
        "forbidden": ((r"DNV-CG-0557", "무관한 DNV-CG-0557"),),
    },
    CASES[6][1]: {
        "required": (
            (r"\bLR\b|Lloyd", "LR"),
            (r"Notice\s+No\.?\s*1", "LR Notice No.1"),
            (r"15\.8\.2|Section\s+15", "Section 15/15.8.2"),
            (r"low[- ]flashpoint|저인화점|dual[- ]fuel|대체연료", "대체연료 적용 조항"),
        ),
    },
}

PLACEHOLDERS = (
    "검색 근거에서 질문에 직접 답할 내용을 확인하지 못했습니다",
    "검색 근거에서 직접 확인되는 별도 운항·업무 영향이 없습니다",
    "추가 확인 필요사항이 별도로 식별되지 않았습니다",
    "관련 선급 Rule / Guidance가 검색 근거에 없거나 해당하지 않습니다",
    "검색 근거에서 확인되지 않음",
    "검색된 문서에서 질문에 직접 답할 근거를 찾지 못했습니다",
)

MEETING_REF = re.compile(r"(MSC|MEPC)\s?\d{2,3}(?:[/\-]\s?\d+)*", re.I)
# 조항 표기는 선급 Rule(p.12, clause 4.5), IMO 규정(regulation 36, 규정 27, Annex VI),
# 회의 문서 문단번호(6.30), 결의(MEPC.279(70))가 모두 인정된다.
CLAUSE_REF = re.compile(
    r"(p\.\s?\d+|clause|regulation|reg\.|규정\s?\d+|규칙\s?\d+|제?\s?\d+장|조항|"
    r"Annex\s?[IVX]+|Section\s?\d+|resolution|MEPC\.\d+|MSC\.\d+|"
    r"(?:초안|보고서|문서|para\.?|paragraph|문단)\s?\d+\.\d+)",
    re.I,
)
CITATION = re.compile(r"\[\d+\]")


def split_sections(answer: str) -> dict[str, list[str]]:
    """Bullet lines grouped by the four contract sections."""
    current = ""
    out: dict[str, list[str]] = {name: [] for name in SECTIONS}
    for line in answer.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            heading = re.sub(r"^\d\)\s*", "", heading)
            current = next((name for name in SECTIONS if name in heading), "")
            continue
        if current and stripped.startswith("-"):
            out[current].append(stripped.lstrip("-").strip())
    return out


def normalize(text: str) -> str:
    return re.sub(r"[\s\*\[\]0-9]+", " ", text).strip()


def audit(
    answer: str,
    category: str,
    low: int,
    high: int,
    total_max: int,
    *,
    question: str = "",
) -> dict:
    sections = split_sections(answer)
    summary = sections[SECTIONS[0]]
    impact = sections[SECTIONS[1]]
    # Placeholder lines stand in for missing evidence, so they neither need a
    # citation nor count as duplicates.
    factual = [b for b in summary + impact if not any(p in b for p in PLACEHOLDERS)]
    cited = [b for b in factual if CITATION.search(b)]
    seen: dict[str, int] = {}
    for bullet in factual:
        key = normalize(bullet)
        seen[key] = seen.get(key, 0) + 1
    problems: list[str] = []

    missing = [name for name in SECTIONS if f") {name}" not in answer and name not in answer]
    if missing:
        problems.append(f"섹션 누락: {', '.join(missing)}")
    if len(summary) < low:
        problems.append(f"핵심 요약 bullet {len(summary)}개 < 최소 {low}개")
    if len(summary) > high:
        problems.append(f"핵심 요약 bullet {len(summary)}개 > 상한 {high}개")
    if len(cited) < len(factual):
        problems.append(f"인용 없는 bullet {len(factual) - len(cited)}개")
    duplicates = {k: v for k, v in seen.items() if v > 1}
    if duplicates:
        problems.append(f"중복 bullet {len(duplicates)}종")
    if not MEETING_REF.search(answer) and category != "단순 Rule":
        problems.append("회의차수·문서번호 표기 없음")
    if not CLAUSE_REF.search(answer):
        problems.append("조항·페이지 표기 없음")
    empty = [name for name in SECTIONS if any(p in "\n".join(sections[name]) for p in PLACEHOLDERS)]
    filled_placeholder = [p for p in PLACEHOLDERS if p in answer]

    total = sum(len(sections[name]) for name in SECTIONS)
    if total > total_max:
        problems.append(f"전체 bullet {total}개 > 카테고리 예산 {total_max}개")

    semantic = SEMANTIC_CONTRACTS.get(question, {})
    for pattern, label in semantic.get("required", ()):  # type: ignore[union-attr]
        if not re.search(pattern, answer, re.I | re.S):
            problems.append(f"핵심 사실 누락: {label}")
    section4_text = "\n".join(sections[SECTIONS[3]])
    for pattern, label in semantic.get("section4_required", ()):  # type: ignore[union-attr]
        if not re.search(pattern, section4_text, re.I | re.S):
            problems.append(f"관련 문서 포인터 누락: {label}")
    for pattern, label in semantic.get("forbidden", ()):  # type: ignore[union-attr]
        if re.search(pattern, answer, re.I | re.S):
            problems.append(f"금지 오류 포함: {label}")

    return {
        "bullets": {name: len(sections[name]) for name in SECTIONS},
        "summary_bullets": len(summary),
        "total_bullets": total,
        "expected_range": [low, high],
        "total_budget": total_max,
        "citation_coverage": round(len(cited) / max(1, len(factual)), 2),
        "duplicate_bullets": sorted(duplicates.values(), reverse=True),
        "placeholder_sections": len(filled_placeholder),
        "empty_sections": empty,
        "problems": problems,
        "sections": sections,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--latency-mode", default="fast")
    parser.add_argument("--only", type=int, help="1-based case index, for a single re-run")
    parser.add_argument("--repeat", type=int, default=1, help="같은 질문 반복 횟수 (답변 변동성 확인)")
    parser.add_argument("--reaudit", type=Path, help="저장된 records.json을 재채점만 한다")
    parser.add_argument(
        "--cold",
        action="store_true",
        help="UI 시작 시 수행되는 Chroma/E5/Gemma 사전 워밍을 생략한다",
    )
    args = parser.parse_args()

    if args.reaudit:
        saved = json.loads(args.reaudit.read_text(encoding="utf-8"))
        for rec in saved:
            low, high = rec["expected_range"]
            budget = rec.get("total_budget", high)
            rec.update(
                audit(
                    rec["answer"], rec["category"], low, high, budget,
                    question=rec.get("question", ""),
                )
            )
            flag = "OK" if not rec["problems"] else " / ".join(rec["problems"])
            print(f"[{rec['index']}] {rec['category']} · §1 {rec['summary_bullets']}개 · 전체 {rec['total_bullets']}개 · {flag}")
        args.reaudit.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.cold:
        # Production UI performs this bootstrap before it becomes ready.  Keep
        # the interaction timer comparable to what tomorrow's reviewer sees;
        # use --cold separately when startup cost itself is under test.
        from services.rag_service import warmup_rag_resources

        warmup_rag_resources()
    cases = [CASES[args.only - 1]] if args.only else CASES
    cases = [case for case in cases for _ in range(args.repeat)]

    records: list[dict] = []
    for idx, (category, question, low, high, budget) in enumerate(cases, start=1):
        started = time.time()
        result = handle_question(
            question,
            [],
            use_llm_router=True,
            rag_latency_mode=args.latency_mode,
            llm_model=args.model,
        )
        elapsed = round(time.time() - started, 1)
        answer = str((result or {}).get("answer") or "")
        meta = (result or {}).get("meta") or {}
        route = (result or {}).get("route") or {}
        record = {
            "index": idx,
            "category": category,
            "question": question,
            "route": route.get("route"),
            "answer_mode": meta.get("answer_mode"),
            "seconds": elapsed,
            "answer_chars": len(answer),
            "answer": answer,
            **audit(answer, category, low, high, budget, question=question),
        }
        if args.latency_mode == "fast" and elapsed > 10.0:
            record["problems"].append(f"Fast 응답 {elapsed}s > 10초")
        records.append(record)
        flag = "OK" if not record["problems"] else " / ".join(record["problems"])
        print(
            f"[{idx}/{len(cases)}] {category} · {elapsed}s · §1 {record['summary_bullets']}개 · "
            f"전체 {record['total_bullets']}개 · {flag}",
            flush=True,
        )

    (out_dir / "records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [f"# 질문 카테고리 계약 점검 {stamp} (model={args.model}, latency={args.latency_mode})", ""]
    for rec in records:
        report += [
            f"## [{rec['index']}] {rec['category']} · {rec['seconds']}s · {rec['answer_mode']}",
            f"Q: {rec['question']}",
            f"bullets: {rec['bullets']} (기대 §1 {rec['expected_range']}, 전체 예산 {rec['total_budget']})",
            f"citation_coverage: {rec['citation_coverage']} · placeholder 섹션: {rec['placeholder_sections']}",
            f"problems: {rec['problems'] or 'none'}",
            "",
            rec["answer"],
            "",
        ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
