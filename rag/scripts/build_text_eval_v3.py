"""Build the evidence-contracted TEXT RAG v3 evaluation set.

The uploaded v2 questions are treated as source material, not as gold truth.
This builder keeps the useful seeds/format/scope variants, selects lexically
diverse paraphrases, and replaces inherited counterfactual/hard-negative labels
with explicit answerability and evidence contracts.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data/eval/source/pilot_validation_questions_augmented_v2_with_gold.jsonl"
DEFAULT_CATALOG = ROOT / "data/eval/text_rag_scenarios_v3.json"
DEFAULT_OUTPUT = ROOT / "data/eval/pilot_validation_text_v3.jsonl"
DEFAULT_REVIEW = ROOT / "data/eval/pilot_validation_text_v3_review.md"

KEEP_TYPES = {"seed", "boundary", "format", "scope", "integration"}
PARAPHRASES_PER_SCENARIO = 15
NOISE_SUFFIXES = (
    "검색 후보에 같은 주제의 다른 문서가 섞여도, 지정 근거에서 확인되는 사실만 답해줘.",
    "유사한 제목이나 인용만 담긴 청크는 근거로 쓰지 말고 핵심 사실을 답해줘.",
    "관련 없어 보이는 규칙·회의 청크가 함께 제공돼도 결론을 바꾸지 말아줘.",
    "제안문과 최종 결정문이 섞이면 결정 상태를 구분하고 확정 근거만 사용해줘.",
    "근접 검색된 주변 문서보다 직접 근거 청크를 우선해 답해줘.",
)

EXTRA_INTEGRATION = {
    "V01": "MEPC 84/7/14의 GFI·SEEMP 후속작업과 MEPC 84/6/2의 2024 선대 탄소집약도 결과를 연결해 선사 준비사항을 정리해줘.",
    "V02": "MSC 111 본회의의 MASS 결과와 회기간 MASS 작업반의 2030 로드맵을 결정·후속일정으로 나눠 정리해줘.",
    "V05": "MSC MASS Code의 국제 일정과 DNV-CG-0264의 위험기반 검증 포인트를 규제·선급 준비로 구분해줘.",
    "V08": "ABS Smart Functions Guide와 Autonomous and Remote Control Requirements가 각각 다루는 대상·평가체계를 비교해줘.",
}

# Each counterfactual checks one false premise, so its completeness contract
# must contain only the evidence points needed to correct that premise.  The
# previous v2 data inherited every parent keypoint and unfairly penalised a
# concise, correct correction.
COUNTERFACTUAL_POINT_MAP = {
    "V01": [["V01-K1"], ["V01-K3"], ["V01-K4"], ["V01-K3"]],
    "V02": [["V02-K1"], ["V02-K1"], ["V02-K3"], ["V02-K2"]],
    "V03": [["V03-K1"], ["V03-K1", "V03-K3"], ["V03-K1"], ["V03-K2"]],
    "V04": [["V04-K1", "V04-K2", "V04-K3", "V04-K4"], ["V04-K4"], ["V04-K4"], ["V04-K1"]],
    "V05": [["V05-K1", "V05-K2"], ["V05-K1", "V05-K4"], ["V05-K2"], ["V05-K1"]],
    "V06": [["V06-K1"], ["V06-K1"], ["V06-K2"], ["V06-K3"]],
    "V07": [["V07-K1"], ["V07-K2"], ["V07-K3"], ["V07-K1"]],
    "V08": [["V08-K1"], ["V08-K1"], ["V08-K4"], ["V08-K1"]],
    "V09": [["V09-K1"], ["V09-K1"], ["V09-K4"], ["V09-K1"]],
}

COUNTERFACTUAL_CORRECTIONS = {
    "V01": [
        ("MEPC 80이 이 인덱스의 최신 MEPC 회의다", "대상 자료는 MEPC 84/7/14이므로 MEPC 80이 최신이라는 전제는 틀리다.", ["MEPC 84", "84/7/14"]),
        ("GFI 보고 지침이 이미 최종 확정·발효됐다", "GFI 보고·검증과 SEEMP 지침은 추가 개발 단계이며 최종 확정·발효된 상태가 아니다.", ["추가 개발", "to be developed", "SEEMP"]),
        ("LCA 지침은 선박 운항 단계 배출만 계산한다", "LCA 지침은 운항 단계뿐 아니라 연료 전과정 배출을 다룬다.", ["전과정", "well-to-tank", "WtT"]),
        ("SEEMP 개정 작업은 더 이상 필요하지 않다고 결론 났다", "SEEMP 지침 개정안은 추가 개발·검토가 필요한 후속 작업이다.", ["추가 작업", "추가 개발", "SEEMP 지침 개정"]),
    ],
    "V02": [
        ("MASS Code 관련 제안은 작업반에 회부되지 않았다", "위원회는 MASS Code 초안 최종화를 위해 관련 제안을 MASS 작업반에 회부했다.", ["작업반에 회부", "working group"]),
        ("MASS Code가 MSC 111에서 즉시 강제 발효됐다", "MSC 111에서는 MASS Code 초안 최종화를 위한 작업이 진행됐으며 즉시 강제 발효된 것이 아니다.", ["초안", "최종화", "non-mandatory"]),
        ("수소 연료 잠정 안전지침은 부결됐다", "수소 연료 사용 선박 잠정 안전지침은 편집 수정 후 승인됐다.", ["승인", "approved"]),
        ("VDES 안건은 MSC 111에서 논의되지 않았다", "VDES 성능기준과 SOLAS V장 연계 결의안이 MSC 111의 주요 결과로 다뤄졌다.", ["VDES", "SOLAS V"]),
    ],
    "V03": [
        ("문서의 보고연도는 2022년이다", "문서가 다루는 선대 보고연도는 2024년이다.", ["2024"]),
        ("2024년 공급기반 평균 탄소집약도는 2019년보다 10.8% 증가했다", "2024년 공급기반 평균 탄소집약도는 2019년 대비 최대 10.8% 감소했다.", ["10.8% 감소", "decrease"]),
        ("문서는 2022년 한 해의 지표만 비교한다", "문서는 AER·cgDIST·EEOI를 사용해 2019년부터 2024년까지의 추이를 비교한다.", ["2019", "2024", "AER", "cgDIST", "EEOI"]),
        ("AER와 cgDIST는 개별 선박 과징금을 계산하는 지표다", "AER와 cgDIST는 공급기반 탄소집약도 지표이며 개별 선박 과징금 산식이 아니다.", ["공급기반", "AER", "cgDIST"]),
    ],
    "V04": [
        ("모든 대체연료 안전코드가 MSC 111에서 전면 강제 발효됐다", "MSC 111 결과에는 승인·채택된 잠정 지침과 후속 작업계획이 함께 있으며 모든 항목이 강제 발효된 것은 아니다.", ["잠정 지침", "작업계획", "interim guidelines"]),
        ("풍력보조추진 최종 규칙이 채택됐다", "풍력보조추진 항목은 최종 규칙 채택이 아니라 후속 제출·작업 단계다.", ["후속", "wind-assisted"]),
        ("리튬이온 배터리 작업은 계획에서 삭제됐다", "리튬이온 배터리 항목은 후속 제출·작업계획 단계로 남아 있다.", ["리튬이온", "후속", "lithium-ion"]),
        ("이 결과는 MEPC 111의 환경규제 결론이다", "해당 결과는 MEPC가 아니라 MSC 111의 선박 안전 관련 결과다.", ["MSC 111"]),
    ],
    "V05": [
        ("MASS Code가 MSC 111에서 이미 강제 발효됐다", "MSC 111 단계의 MASS Code는 비강제 Code 개발·최종화 단계이며 이미 강제 발효된 것이 아니다.", ["비강제", "non-mandatory"]),
        ("MASS Code 개발이 폐지됐다", "MASS Code는 최종화와 경험축적단계·로드맵 갱신을 위한 작업이 계속되고 있다.", ["최종화", "경험축적", "로드맵"]),
        ("mandatory Code 채택 목표는 2025년이다", "mandatory MASS Code 채택 예상 일정은 2030년이다.", ["2030"]),
        ("원격운항자는 STCW 논의 대상에서 완전히 제외하기로 확정됐다", "원격운항자 훈련은 고수준 조항·후속 지침·최종 기준으로 이어지는 단계적 검토 대상이다.", ["원격운항자", "훈련", "3단계"]),
    ],
    "V06": [
        ("DNV-CG-0264는 KR 규칙이다", "DNV-CG-0264는 DNV의 자율·원격운항 선박 지침이다.", ["DNV"]),
        ("DNV-CG-0264는 IMO 강제협약이다", "DNV-CG-0264는 IMO 강제협약이 아니라 DNV의 class guideline이다.", ["class guideline", "DNV"]),
        ("이 지침은 위험평가를 요구하지 않는다", "DNV-CG-0264는 새로운 운항·기능·시스템 위험을 식별·통제하기 위한 위험평가를 요구한다.", ["위험평가", "risk assessment"]),
        ("Concept Qualification은 AROS notation과 무관하다", "Concept Qualification은 AROS notation 부여와 혁신 개념의 제3자 검증에 사용된다.", ["Concept Qualification", "AROS"]),
    ],
    "V07": [
        ("Section 15는 ABS 규정이다", "Section 15는 LR Rules and Regulations for the Classification of Ships의 조항이다.", ["LR", "Lloyd's Register", "Section 15"]),
        ("Notice No.1은 저인화점 연료를 다루지 않는다", "Notice No.1의 Section 15는 가스와 기타 저인화점 연료 엔진을 다룬다.", ["저인화점", "low-flashpoint"]),
        ("가스 엔진의 크랭크케이스 안전평가는 필요 없다", "가스·저인화점 연료 엔진은 크랭크케이스 안전에 대한 상세 평가가 필요하다.", ["상세 평가", "크랭크케이스", "crankcase"]),
        ("이 조항은 IMO MSC 회의결과 문서다", "이 조항은 IMO 회의결과가 아니라 LR 선급 규칙의 Section 15다.", ["LR", "Section 15"]),
    ],
    "V08": [
        ("Guide for Smart Functions는 자율운항 선박에만 적용된다", "Guide for Smart Functions는 모든 해양선박과 해양구조물에 적용된다.", ["모든 해양선박", "해양구조물", "marine vessels", "offshore units"]),
        ("적용 대상은 해양구조물을 제외한 상선뿐이다", "적용 대상에는 해양선박뿐 아니라 해양구조물도 포함된다.", ["해양구조물", "offshore units"]),
        ("SMART notation은 의무 강제부호다", "SMART 계열 notation은 요건을 충족한 시스템이 선택적으로 받을 수 있는 부호다.", ["선택", "optional", "SMART"]),
        ("이 Guide는 IMO MSC가 채택한 협약이다", "Guide for Smart Functions는 IMO 협약이 아니라 ABS의 선급 Guide다.", ["ABS", "Guide"]),
    ],
    "V09": [
        ("모든 자율·원격제어 기능에는 같은 위험범주가 적용된다", "각 기능은 운항감독 수준과 고장 결과에 따라 저·중·고 위험범주 중 하나를 배정받는다.", ["저", "중", "고", "risk category"]),
        ("기능 위험범주는 고장 결과와 무관하다", "기능 위험범주는 운항감독 수준과 고장 결과를 기준으로 정한다.", ["고장 결과", "failure consequence"]),
        ("상위 위험범주는 하위범주 요건을 충족할 필요가 없다", "상위 위험범주 기능은 하위 위험범주의 관련 요건도 충족해야 한다.", ["하위 위험범주", "lower risk"]),
        ("이 Requirements는 IMO 협약이다", "Requirements for Autonomous and Remote Control Functions는 IMO 협약이 아니라 ABS 선급 요구사항이다.", ["ABS", "Requirements"]),
    ],
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def normalized_ngrams(text: str, n: int = 3) -> set[str]:
    compact = re.sub(r"[^0-9a-z가-힣]+", "", text.lower())
    if len(compact) <= n:
        return {compact} if compact else set()
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def annotate_quality_gates(rows: list[dict[str, Any]]) -> None:
    """Attach an auditable AURA-QG-inspired deterministic quality gate.

    Robustness variants intentionally share a base question, so lexical
    redundancy is measured only among ordinary paraphrases of one scenario.
    """
    paraphrases: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["test_type"] == "paraphrase":
            paraphrases.setdefault(str(row["scenario_id"]), []).append(row)
    for row in rows:
        peers = paraphrases.get(str(row["scenario_id"]), [])
        nearest = 0.0
        if row["test_type"] == "paraphrase":
            current = normalized_ngrams(str(row["question"]))
            nearest = max(
                (
                    jaccard(current, normalized_ngrams(str(peer["question"])))
                    for peer in peers
                    if peer is not row
                ),
                default=0.0,
            )
        evidence_complete = bool(row["gold_chunk_ids"]) if row["answerability"] else True
        answerability_valid = bool(row["answerability"]) == bool(row["retrieval_target_required"])
        non_redundant = row["test_type"] != "paraphrase" or nearest < 0.75
        context_anchor_present = (
            row["test_type"] != "evidence_precision"
            or str(row.get("evaluation_context") or "").lower()
            in str(row.get("question") or "").lower()
        )
        gate = row.setdefault("quality_gate", {})
        gate.update(
            {
                "answerability_label_valid": answerability_valid,
                "evidence_contract_complete": evidence_complete,
                "non_redundant": non_redundant,
                "standalone_context_anchor_present": context_anchor_present,
                "nearest_paraphrase_3gram_jaccard": round(nearest, 4),
                "coverage_axis_present": all(
                    row.get(key) not in (None, "")
                    for key in ("scenario_id", "test_type", "retrieval_difficulty", "hop_count")
                ),
            }
        )
        gate["passed"] = all(
            gate[key]
            for key in (
                "answerability_label_valid",
                "evidence_contract_complete",
                "non_redundant",
                "standalone_context_anchor_present",
                "coverage_axis_present",
            )
        )


def select_diverse_paraphrases(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Greedy max-min selection prevents easy near-duplicate dominance."""
    if len(rows) <= limit:
        return list(rows)
    ordered = sorted(rows, key=lambda row: str(row["question_id"]))
    grams = {str(row["question_id"]): normalized_ngrams(str(row["question"])) for row in ordered}
    selected = [max(ordered, key=lambda row: len(grams[str(row["question_id"])]))]
    remaining = [row for row in ordered if row is not selected[0]]
    while remaining and len(selected) < limit:
        next_row = min(
            remaining,
            key=lambda row: (
                max(
                    jaccard(grams[str(row["question_id"])], grams[str(chosen["question_id"])])
                    for chosen in selected
                ),
                str(row["question_id"]),
            ),
        )
        selected.append(next_row)
        remaining.remove(next_row)
    return sorted(selected, key=lambda row: str(row["question_id"]))


def page_from_chunk_id(chunk_id: str) -> int | None:
    match = re.search(r"_p(\d{4})_", chunk_id)
    return int(match.group(1)) if match else None


def doc_from_chunk_id(chunk_id: str) -> str:
    match = re.search(r"^(.*?)_p\d{4}(?:_|$)", chunk_id)
    return match.group(1) if match else ""


def evidence_points(scenario: dict[str, Any], *, limit: int | None = None) -> list[dict[str, Any]]:
    points = list(scenario["keypoints"])
    return points[:limit] if limit else points


def evidence_contract(points: list[dict[str, Any]], scenario_by_doc: dict[str, dict[str, Any]]) -> tuple[list[dict], list[str], list[int], list[str]]:
    evidence: list[dict[str, Any]] = []
    chunk_ids: list[str] = []
    pages: list[int] = []
    doc_ids: list[str] = []
    for point in points:
        for chunk_id in point["chunk_ids"]:
            doc_id = doc_from_chunk_id(chunk_id)
            scenario = scenario_by_doc.get(doc_id, {})
            page = page_from_chunk_id(chunk_id)
            item = {
                "evidence_id": f"{point['id']}:{len(evidence) + 1}",
                "point_id": point["id"],
                "doc_id": doc_id,
                "source": scenario.get("source", ""),
                "chunk_id": chunk_id,
                "page": page,
                "claim": point["text"],
            }
            evidence.append(item)
            if chunk_id not in chunk_ids:
                chunk_ids.append(chunk_id)
            if page is not None and page not in pages:
                pages.append(page)
            if doc_id and doc_id not in doc_ids:
                doc_ids.append(doc_id)
    return evidence, chunk_ids, sorted(pages), doc_ids


def integration_secondary_id(
    question: str,
    scenario: dict[str, Any],
) -> str:
    """Resolve the second evidence scenario from the source named in the question."""
    q = question.upper()
    primary_source = str(scenario.get("source") or "").upper()
    if "DNV" in q and primary_source != "DNV":
        return "V06"
    if re.search(r"(?<![A-Z])LR(?![A-Z])|LLOYD", q) and primary_source != "LR":
        return "V07"
    if "ABS" in q and primary_source != "ABS":
        return "V09"
    if "MSC" in q and primary_source != "MSC":
        return "V05" if "MASS" in q else "V04"
    if "MEPC" in q and primary_source != "MEPC":
        return "V03" if re.search(r"CII|FLEET|선대|탄소집약도", q) else "V01"
    return str(scenario["secondary_scenario"])


def format_contract(question: str) -> dict[str, Any]:
    required_sections: list[str] = []
    if all(term in question for term in ("핵심", "업무", "추후", "Rule")):
        required_sections = ["핵심 요약", "업무 영향", "추후 확인", "관련 Rule"]
    exact_items = None
    match = re.search(r"정확히\s*(\d+)개|주요 결과\s*(\d+)개", question)
    if match:
        exact_items = int(match.group(1) or match.group(2))
    bullet_range = None
    match = re.search(r"(\d+)\s*[~～-]\s*(\d+)개\s*bullet", question, re.I)
    if match:
        bullet_range = [int(match.group(1)), int(match.group(2))]
    return {
        "required_sections": required_sections,
        "exact_summary_items": exact_items,
        "summary_bullet_range": bullet_range,
        "citation_required": "citation" in question.lower() or "근거" in question,
    }


def make_row(
    *,
    question_id: str,
    legacy_question_id: str | None,
    parent_id: str,
    question: str,
    test_type: str,
    scenario: dict[str, Any],
    scenarios: dict[str, dict[str, Any]],
    scenario_by_doc: dict[str, dict[str, Any]],
    target_point_ids: list[str] | None = None,
    false_premise: str | None = None,
    counterfactual_correction: tuple[str, list[str]] | None = None,
) -> dict[str, Any]:
    answerability = test_type != "negative_rejection"
    expected_behavior = {
        "noise_robustness": "ignore_noise",
        "negative_rejection": "reject_missing_evidence",
        "counterfactual_robustness": "correct_false_premise",
        "integration": "integrate_all_sources",
        "format": "follow_format",
    }.get(test_type, "answer_from_evidence")

    primary_points = evidence_points(scenario)
    points = primary_points
    required_sources = [scenario["source"]]
    if test_type == "integration":
        secondary = scenarios[integration_secondary_id(question, scenario)]
        points = evidence_points(scenario, limit=2) + evidence_points(secondary, limit=2)
        required_sources = list(dict.fromkeys([scenario["source"], secondary["source"]]))
    if test_type == "evidence_precision":
        index_match = re.search(r"P(\d{2})$", question_id)
        point_index = (int(index_match.group(1)) - 1) if index_match else 0
        # P01-P04 isolate one gold proposition. P05 deliberately combines
        # the scenario's propositions to test within-document integration.
        points = [primary_points[point_index]] if point_index < len(primary_points) else primary_points
    if test_type == "counterfactual_robustness" and target_point_ids:
        selected = [point for point in primary_points if point["id"] in target_point_ids]
        if selected:
            points = selected

    evidence, chunk_ids, pages, doc_ids = evidence_contract(points, scenario_by_doc)
    acceptable_doc_ids = (
        list(dict.fromkeys([*doc_ids, *scenario.get("acceptable_doc_ids", [])]))
        if answerability
        else list(scenario.get("acceptable_doc_ids") or [scenario["doc_id"]])
    )

    if answerability:
        if test_type == "counterfactual_robustness" and counterfactual_correction:
            correction_text, correction_aliases = counterfactual_correction
            gold_points = [{
                "point_id": f"{question_id}-CORRECTION",
                "text": correction_text,
                "aliases": correction_aliases,
                "evidence_chunk_ids": list(chunk_ids),
            }]
            for item in evidence:
                item["claim"] = correction_text
        else:
            gold_points = [
                {
                    "point_id": point["id"],
                    "text": point["text"],
                    "aliases": point["aliases"],
                    "evidence_chunk_ids": list(point["chunk_ids"]),
                }
                for point in points
            ]
        gold_answer = "\n".join(f"- {point['text']}" for point in points)
    else:
        gold_points = [
            {
                "point_id": f"{parent_id}-REJECT",
                "text": "지정된 문서 근거에서는 요청한 세부 정보를 확인할 수 없다고 명시해야 한다.",
                "aliases": ["확인할 수 없습니다", "근거가 없습니다", "제공된 문서에서 확인되지 않습니다"],
                "evidence_chunk_ids": [],
            }
        ]
        gold_answer = "지정된 문서 근거에서는 요청한 정보를 확인할 수 없습니다. 추정하지 말고 추가 자료를 요청해야 합니다."
        evidence, chunk_ids, pages, doc_ids = [], [], [], []

    hop_count = 2 if test_type == "integration" else int(scenario.get("default_hop_count") or 1)
    retrieval_difficulty = (
        "hard"
        if test_type in {"noise_robustness", "negative_rejection", "counterfactual_robustness", "integration"}
        else "medium"
        if test_type in {"paraphrase", "boundary", "scope"}
        else "easy"
    )
    rgb_ability = {
        "noise_robustness": "noise_robustness",
        "negative_rejection": "negative_rejection",
        "counterfactual_robustness": "counterfactual_robustness",
        "integration": "information_integration",
    }.get(test_type)
    return {
        "schema_version": "text-rag-eval-v3",
        "question_id": question_id,
        "legacy_question_id": legacy_question_id,
        "scenario_id": parent_id,
        "parent_id": parent_id,
        "category": scenario["category"],
        "evaluation_context": scenario["doc_label"],
        "test_type": test_type,
        "augment_type": test_type,
        "rgb_ability": rgb_ability,
        "hop_count": hop_count,
        "retrieval_difficulty": retrieval_difficulty,
        "question": question,
        "answerability": answerability,
        "expected_behavior": expected_behavior,
        "false_premise": false_premise,
        "gold_answer": gold_answer,
        "gold_answer_points": gold_points,
        "must_cover": [point["text"] for point in gold_points],
        "gold_evidence": evidence,
        "acceptable_doc_ids": acceptable_doc_ids,
        "gold_source": scenario["source"],
        "gold_doc_id": (doc_ids[0] if doc_ids else "") if answerability else "",
        "gold_doc_ids": doc_ids,
        "gold_chunk_ids": chunk_ids,
        "gold_pages": pages,
        "gold_page": pages[0] if pages else None,
        "hard_negative_chunk_ids": list(scenario["hard_negative_chunk_ids"]),
        "forbidden_claims": list(scenario["forbidden_claims"]),
        "forbid_claims": list(scenario["forbidden_claims"]),
        "source_constraints": {
            "required": required_sources,
            "excluded": [],
            "only": required_sources if test_type in {"scope", "negative_rejection"} else [],
        },
        "retrieval_target_required": answerability,
        "format_contract": format_contract(question),
        "required_sections": format_contract(question)["required_sections"],
        "evaluation_dimensions": [
            "retrieval_recall",
            "completeness",
            "faithfulness",
            "answer_relevance",
            "hallucination",
            "irrelevance",
        ],
        "context_perturbation": (
            {
                "type": "inject_hard_negatives",
                "chunk_ids": list(scenario["hard_negative_chunk_ids"]),
                "position": "before_gold",
            }
            if test_type == "noise_robustness"
            else None
        ),
        "quality_gate": {
            "answerable_from_corpus": answerability,
            "single_interpretation": True,
            "evidence_contract_complete": bool(evidence) if answerability else True,
            "parent_gold_inherited_unconditionally": False,
        },
    }


def build_rows(source_rows: list[dict[str, Any]], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios: dict[str, dict[str, Any]] = catalog["scenarios"]
    scenario_by_doc = {scenario["doc_id"]: scenario for scenario in scenarios.values()}
    output: list[dict[str, Any]] = []

    for parent_id, scenario in scenarios.items():
        grouped = [row for row in source_rows if row.get("parent_id") == parent_id]
        paraphrases = select_diverse_paraphrases(
            [row for row in grouped if row.get("augment_type") == "paraphrase"],
            PARAPHRASES_PER_SCENARIO,
        )
        selected = [row for row in grouped if row.get("augment_type") in KEEP_TYPES]
        selected.extend(paraphrases)
        selected.sort(key=lambda row: str(row["question_id"]))
        for source_row in selected:
            test_type = str(source_row["augment_type"])
            output.append(
                make_row(
                    question_id=f"T3-{source_row['question_id']}",
                    legacy_question_id=str(source_row["question_id"]),
                    parent_id=parent_id,
                    question=str(source_row["question"]),
                    test_type=test_type,
                    scenario=scenario,
                    scenarios=scenarios,
                    scenario_by_doc=scenario_by_doc,
                )
            )

        for index, question in enumerate(scenario["precision_questions"], 1):
            output.append(
                make_row(
                    question_id=f"T3-{parent_id}-P{index:02d}",
                    legacy_question_id=None,
                    parent_id=parent_id,
                    question=f"{scenario['doc_label']}를 기준으로, {question}",
                    test_type="evidence_precision",
                    scenario=scenario,
                    scenarios=scenarios,
                    scenario_by_doc=scenario_by_doc,
                )
            )
        for index, suffix in enumerate(NOISE_SUFFIXES, 1):
            seed = next(row for row in grouped if row.get("augment_type") == "seed")
            output.append(
                make_row(
                    question_id=f"T3-{parent_id}-N{index:02d}",
                    legacy_question_id=None,
                    parent_id=parent_id,
                    question=f"{seed['question']} {suffix}",
                    test_type="noise_robustness",
                    scenario=scenario,
                    scenarios=scenarios,
                    scenario_by_doc=scenario_by_doc,
                )
            )
        for index, target in enumerate(scenario["unanswerable_targets"], 1):
            rejection_row = make_row(
                    question_id=f"T3-{parent_id}-R{index:02d}",
                    legacy_question_id=None,
                    parent_id=parent_id,
                    question=f"{scenario['doc_label']}에서 {target}을 찾아 문서 근거와 함께 알려줘.",
                    test_type="negative_rejection",
                    scenario=scenario,
                    scenarios=scenarios,
                    scenario_by_doc=scenario_by_doc,
                )
            rejection_row["unanswerable_target"] = target
            output.append(rejection_row)
        counterfactuals = COUNTERFACTUAL_CORRECTIONS[parent_id]
        for index, (premise, correction_text, correction_aliases) in enumerate(counterfactuals, 1):
            target_ids = COUNTERFACTUAL_POINT_MAP.get(parent_id, [])
            mapped_ids = target_ids[index - 1] if index <= len(target_ids) else None
            output.append(
                make_row(
                    question_id=f"T3-{parent_id}-C{index:02d}",
                    legacy_question_id=None,
                    parent_id=parent_id,
                    question=(
                        f"{scenario['doc_label']}를 기준으로 '{premise}'라는 전제가 맞는지 "
                        "검증하고, 틀리면 문서 근거로 바로잡아줘."
                    ),
                    test_type="counterfactual_robustness",
                    scenario=scenario,
                    scenarios=scenarios,
                    scenario_by_doc=scenario_by_doc,
                    target_point_ids=mapped_ids,
                    false_premise=premise,
                    counterfactual_correction=(correction_text, correction_aliases),
                )
            )

        if parent_id in EXTRA_INTEGRATION:
            output.append(
                make_row(
                    question_id=f"T3-{parent_id}-IX01",
                    legacy_question_id=None,
                    parent_id=parent_id,
                    question=EXTRA_INTEGRATION[parent_id],
                    test_type="integration",
                    scenario=scenario,
                    scenarios=scenarios,
                    scenario_by_doc=scenario_by_doc,
                )
            )
    output = sorted(output, key=lambda row: str(row["question_id"]))
    annotate_quality_gates(output)
    return output


def validate_rows(rows: list[dict[str, Any]], expected_count: int = 405) -> list[str]:
    errors: list[str] = []
    ids = [str(row["question_id"]) for row in rows]
    questions = [re.sub(r"\s+", " ", str(row["question"]).strip().lower()) for row in rows]
    if len(rows) != expected_count:
        errors.append(f"row_count={len(rows)} expected={expected_count}")
    if len(ids) != len(set(ids)):
        errors.append("duplicate question_id")
    if len(questions) != len(set(questions)):
        errors.append("duplicate normalized question")
    for row in rows:
        if row["answerability"] and not row["gold_answer_points"]:
            errors.append(f"{row['question_id']}: missing gold points")
        if row["retrieval_target_required"] and not row["gold_chunk_ids"]:
            errors.append(f"{row['question_id']}: missing gold chunks")
        if row["test_type"] == "noise_robustness" and not row["hard_negative_chunk_ids"]:
            errors.append(f"{row['question_id']}: missing hard negatives")
        if not (row.get("quality_gate") or {}).get("passed"):
            errors.append(f"{row['question_id']}: quality gate failed")
        if row["test_type"] == "negative_rejection" and row["retrieval_target_required"]:
            errors.append(f"{row['question_id']}: rejection must not require gold retrieval")
    return errors


def review_markdown(rows: list[dict[str, Any]]) -> str:
    by_type = Counter(row["test_type"] for row in rows)
    by_scenario = Counter(row["scenario_id"] for row in rows)
    difficulty = Counter(row["retrieval_difficulty"] for row in rows)
    lines = [
        "# TEXT RAG 평가셋 v3 검토",
        "",
        f"- 총 문항: **{len(rows)}개**",
        "- 원본 v2는 변경하지 않고, 유용한 문항을 선별해 증거 계약을 새로 부여했습니다.",
        "- 모든 답변 가능 문항은 `gold_answer_points`와 실제 `gold_chunk_ids`를 가집니다.",
        "- 근거 없음 문항은 부모 gold를 상속하지 않고 `reject_missing_evidence`로 채점합니다.",
        "",
        "## 유형 분포",
        "",
        "| 유형 | 문항 수 |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in sorted(by_type.items()))
    lines.extend(["", "## 시나리오 분포", "", "| 시나리오 | 문항 수 |", "|---|---:|"])
    lines.extend(f"| {key} | {value} |" for key, value in sorted(by_scenario.items()))
    lines.extend(["", "## 검색 난이도", "", "| 난이도 | 문항 수 |", "|---|---:|"])
    lines.extend(f"| {key} | {value} |" for key, value in sorted(difficulty.items()))
    lines.extend(
        [
            "",
            "## 품질 게이트",
            "",
            "- ID·정규화 질문 중복 없음",
            "- 답변 가능 문항은 최소 1개 이상의 실제 정답 청크 보유",
            "- noise 문항은 실제 hard-negative 청크 주입 계약 보유",
            "- negative rejection 문항은 정답 검색률과 분리해 거절 정확도로 평가",
            "- integration 문항은 두 시나리오의 근거를 각각 포함",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()

    source_rows = load_jsonl(args.source)
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    rows = build_rows(source_rows, catalog)
    errors = validate_rows(rows)
    if errors:
        raise SystemExit("validation failed:\n- " + "\n- ".join(errors))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    args.review.write_text(review_markdown(rows), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
