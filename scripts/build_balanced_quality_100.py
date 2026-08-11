#!/usr/bin/env python3
"""Build the reproducible 100-question balanced end-to-end evaluation set.

Distribution:
  - operations database: 25
  - text retrieval:      50
  - table retrieval:     15
  - hybrid:              10

The text/table questions are selected from the checked-in gold sets.  Operations
and hybrid questions use the deterministic H2521 fixture currently shipped with
the project, so every question has a concrete, machine-checkable reference.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval"
OUTPUT = EVAL / "balanced_quality_100.jsonl"


def load_jsonl(name: str) -> list[dict]:
    path = EVAL / name
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def number_pattern(value: str) -> str:
    """Accept an optional thousands comma while keeping the decimal exact."""
    whole, dot, frac = value.partition(".")
    if len(whole) > 3:
        groups = []
        while whole:
            groups.append(whole[-3:])
            whole = whole[:-3]
        whole_re = ",?".join(reversed(groups))
    else:
        whole_re = whole
    return rf"{whole_re}\.{frac}" if dot else whole_re


def op(question_id: str, question: str, gold: str, *needles: str) -> dict:
    return {
        "id": f"B100_OPS_{question_id}",
        "type": "ops",
        "subtype": "operations_db",
        "route": "ops",
        "question": question,
        "gold": gold,
        "needles": list(needles),
    }


OPS = [
    op("01", "현재 운항 중인 항차 번호와 적재 상태를 알려줘.", "H2521_V21, Ballast", r"H2521(?:_V21)?", r"Ballast|밸러스트"),
    op("02", "현재 항차의 시작 시각과 마지막 데이터 시각은 언제야?", "2026-06-08 00:00 ~ 2026-06-28 04:00", r"2026[-.]?06[-.]?08", r"2026[-.]?06[-.]?28"),
    op("03", "현재 항차는 데이터 기준으로 며칠 동안 운항했어?", "20.2일", r"20\.2\s*일"),
    op("04", "현재 Ballast 항차의 누적 운항거리는 몇 해리야?", "5,847.4 nm", number_pattern("5847.4"), r"nm|해리"),
    op("05", "현재 선박의 최신 위도와 경도를 알려줘.", "위도 27.973, 경도 0.0", r"27\.973", r"0\.0|0(?:\D|$)"),
    op("06", "지금 선박의 대지속력(SOG)은 몇 노트야?", "18.6 kn", r"18\.6", r"kn|노트|SOG"),
    op("07", "현재 항차 전체의 평균 SOG는 얼마야?", "14.0 kn", r"14(?:\.0)?", r"kn|노트|SOG"),
    op("08", "최신 주기관 출력은 몇 kW로 기록됐어?", "19,740 kW", r"19,?740", r"kW|킬로와트"),
    op("09", "현재 시점의 오일계 연료 소비율은 시간당 몇 톤이야?", "0.1083 MT/h", r"0\.1083", r"MT/h|톤/시간|시간당"),
    op("10", "현재 시점의 가스계 연료 소비율은 얼마야?", "3.2426 MT/h", r"3\.2426", r"MT/h|톤/시간|시간당"),
    op("11", "현재 CO2 배출률은 시간당 몇 톤이야?", "9.3105 MT/h", r"9\.3105", r"CO2|CO₂"),
    op("12", "현재 항차에서 누적 사용한 오일계 연료는 얼마야?", "71.19 MT", r"71\.19", r"MT|톤"),
    op("13", "현재 항차에서 누적 사용한 가스계 연료는 얼마야?", "917.23 MT", r"917\.23", r"MT|톤"),
    op("14", "현재 항차의 누적 CO2 배출량을 알려줘.", "2,751.67 MT", number_pattern("2751.67"), r"CO2|CO₂"),
    op("15", "현재 항차의 누적 CH4 배출량은 얼마야?", "53.0901 MT", r"53\.0901", r"CH4|CH₄|메탄"),
    op("16", "현재 항차의 누적 CO2e는 얼마야?", "4,238.2 MTCO2e", number_pattern("4238.2"), r"CO2e|CO₂e|이산화탄소환산"),
    op("17", "현재 항차의 거리당 CO2 배출량(CII 지표)은 얼마야?", "470.5805 kg CO2/nm", r"470\.5805|470\.58", r"kg.*(?:nm|해리)|CII"),
    op("18", "직전 Laden 항차의 운항거리는 몇 해리였어?", "H2521_V20_Laden, 5,470.5 nm", r"V20|Laden|적재", number_pattern("5470.5")),
    op("19", "직전 적재 항차의 평균 속력은 얼마였어?", "17.8 kn", r"17\.8", r"kn|노트"),
    op("20", "직전 Laden 항차에서 배출한 CO2 총량은 얼마야?", "4,662.82 MT", number_pattern("4662.82"), r"CO2|CO₂"),
    op("21", "직전 적재 항차의 하루 평균 총 연료 사용량은 얼마야?", "93.49 MT/day", r"93\.49", r"MT/day|톤/일|하루"),
    op("22", "직전 항차의 잠정 attained CII와 required CII, 등급을 같이 알려줘.", "attained 10.497, required 3.73149, E", r"10\.497", r"3\.73149|3\.73", r"E\s*등급|등급.{0,10}E|rating.{0,10}E"),
    op("23", "2026년 누적 운항 항차 수와 총 거리는 얼마야?", "11항차, 60,848.6 nm", r"11\s*(?:개\s*)?항차|항차.{0,10}11", number_pattern("60848.6")),
    op("24", "2026년 누적 총 연료 사용량과 오일·가스별 사용량을 알려줘.", "총 11,299.01 MT (오일 1,216.07, 가스 10,082.94)", number_pattern("11299.01"), number_pattern("1216.07"), number_pattern("10082.94")),
    op("25", "2026년 누적 잠정 CII attained, required와 등급은?", "attained 6.39567, required 3.73149, E", r"6\.39567|6\.39|6\.40|6\.4", r"3\.73149|3\.73", r"E\s*등급|등급.{0,10}E|rating.{0,10}E"),
]


def first_and_fact_patterns(row: dict) -> list[str]:
    terms = list(row.get("must_cover") or row.get("expected_keywords") or [])
    terms = [str(term).strip() for term in terms if str(term).strip()]
    if not terms:
        return [r"근거|문서|규칙"]
    first = re.escape(terms[0]).replace(r"\ ", r"\s+")
    if len(terms) == 1:
        return [first]
    facts = "|".join(re.escape(term).replace(r"\ ", r"\s+") for term in terms[1:])
    return [first, facts]


def text_row(prefix: str, row: dict) -> dict:
    source_id = str(row.get("id") or row.get("question_id"))
    needles = list(row.get("needles") or first_and_fact_patterns(row))
    gold_terms = row.get("must_cover") or row.get("expected_keywords") or row.get("gold")
    if isinstance(gold_terms, list):
        gold = ", ".join(map(str, gold_terms))
    else:
        gold = str(gold_terms or row.get("note") or "문서 근거 답변")
    return {
        "id": f"B100_TEXT_{prefix}_{source_id}",
        "type": "text",
        "subtype": row.get("category") or row.get("type") or prefix.lower(),
        "route": "rag",
        "question": row["question"],
        "gold": gold,
        "needles": needles,
        "gold_doc_id": row.get("gold_doc_id"),
        "gold_page": row.get("gold_page") or row.get("gold_pages"),
    }


kr = load_jsonl("kr_1_2025_questions.jsonl")[:25]
pilot = load_jsonl("pilot_validation_questions.jsonl")
_prior_questions = {row["question"] for row in kr + pilot}
full = [
    row
    for row in load_jsonl("full_corpus_questions.jsonl")
    if row["question"] not in _prior_questions
][:10]
meeting = [
    row
    for row in load_jsonl("meeting_questions.jsonl")
    if row.get("question_id") in {"MQ002", "MQ003", "MQ004"}
]
hier_ids = {"fresh_meet_01", "fresh_meet_02", "fresh_meet_03", "fresh_def_01", "fresh_def_02"}
hier = [row for row in load_jsonl("hierarchical_retrieval_20.jsonl") if row.get("id") in hier_ids]
TEXT = (
    [text_row("KR", row) for row in kr]
    + [text_row("PILOT", row) for row in pilot]
    + [text_row("FULL", row) for row in full]
    + [text_row("MEETING", row) for row in meeting]
    + [text_row("HIER", row) for row in hier]
)


def table_row(row: dict) -> dict:
    raw_needles = list(row.get("needles") or [])
    # The original open-table set often contains one full literal plus its key
    # cells.  Requiring the final two cell patterns is strict without being
    # sensitive to prose or whitespace around the answer.
    needles = raw_needles[-2:] if len(raw_needles) >= 2 else raw_needles
    return {
        "id": f"B100_TABLE_{row['id']}",
        "type": "table",
        "subtype": "open_table_cell",
        "route": "rag",
        "question": row["question"],
        "gold": row.get("gold"),
        "needles": needles,
        "gold_file": row.get("gold_file"),
        "gold_page": row.get("gold_page"),
    }


# Fixed, evenly spread indices prevent cherry-picking only the first/easiest rows.
table_pool = load_jsonl("quality_50_open_mix.jsonl")
table_indices = [0, 1, 2, 3, 5, 8, 11, 14, 17, 20, 23, 27, 32, 33, 36]
TABLE = [table_row(table_pool[index]) for index in table_indices]


def hybrid(question_id: str, question: str, gold: str, *needles: str) -> dict:
    return {
        "id": f"B100_HYBRID_{question_id}",
        "type": "hybrid",
        "subtype": "operations_plus_document",
        "route": "hybrid",
        "question": question,
        "gold": gold,
        "needles": list(needles),
    }


HYBRID = [
    hybrid("01", "우리 선박의 2026년 누적 CII 수치와 등급을 조회하고, IMO·MEPC의 CII 관리 요구사항과 함께 평가해줘.", "6.39567 / 3.73149 / E와 CII 관리 요구", r"6\.39567|6\.39|6\.40|6\.4", r"E\s*등급|등급.{0,10}E", r"CII|탄소집약"),
    hybrid("02", "현재 항차의 CO2 총배출량을 먼저 계산해 보여주고, MEPC 온실가스 감축 문서가 요구하는 운항상 시사점을 함께 설명해줘.", "2,751.67 MT와 MEPC GHG 감축", number_pattern("2751.67"), r"MEPC", r"GHG|온실가스|배출"),
    hybrid("03", "직전 적재 항차의 CO2 배출량과 평균 속력을 조회한 뒤, CII 규정 관점에서 어떤 관리가 필요한지 문서 근거로 답해줘.", "4,662.82 MT, 17.8 kn과 CII 관리", number_pattern("4662.82"), r"17\.8", r"CII|탄소집약"),
    hybrid("04", "현재 항차의 가스계 연료 사용량과 CO2e를 보여주고, 대체연료 안전에 관한 MSC 111 논의와 연결해 설명해줘.", "917.23 MT, 4,238.2 MTCO2e와 MSC 111 대체연료 안전", r"917\.23", number_pattern("4238.2"), r"MSC\s*111|대체연료|alternative\s+fuel"),
    hybrid("05", "현재 Ballast 항차의 기간과 거리를 조회하고, SEEMP·운항 탄소집약도 보고에서 확인할 사항을 문서 근거로 정리해줘.", "20.2일, 5,847.4 nm와 SEEMP/CII 보고", r"20\.2", number_pattern("5847.4"), r"SEEMP|CII|보고|reporting"),
    hybrid("06", "2026년 누적 11개 항차의 총거리와 CO2e를 조회한 뒤, IMO 온실가스 감축 회의자료 관점의 의미를 설명해줘.", "11항차, 60,848.6 nm, 40,337.8 MTCO2e와 IMO GHG", r"11\s*(?:개\s*)?항차|항차.{0,10}11", number_pattern("60848.6"), r"IMO|GHG|온실가스"),
    hybrid("07", "현재 선박 위치와 속력을 먼저 알려주고, DNV 자율·원격운항 지침에서 연결성과 안전 측면에 어떤 고려가 필요한지 같이 답해줘.", "27.973/0.0, 18.6 kn과 DNV-CG-0264 connectivity/safety", r"27\.973", r"18\.6", r"DNV|connectivity|연결성|자율|원격"),
    hybrid("08", "현재 주기관 출력과 시간당 연료 소비를 조회하고, LR 저인화점 연료 엔진 규칙의 안전 고려사항과 함께 설명해줘.", "19,740 kW, 오일 0.1083 또는 가스 3.2426 MT/h와 LR low-flashpoint fuel", r"19,?740", r"0\.1083|3\.2426", r"LR|low.?flashpoint|저인화점"),
    hybrid("09", "직전 항차의 잠정 CII 등급과 attained 값을 조회하고, 등급이 낮을 때 필요한 시정·관리 조치를 문서에서 찾아 정리해줘.", "E, 10.497과 CII 시정관리", r"10\.497", r"E\s*등급|등급.{0,10}E", r"시정|corrective|SEEMP|CII"),
    hybrid("10", "현재 항차가 Ballast인지 Laden인지와 누적 연료를 확인하고, 관련 운항 효율·배출 보고 요구를 MEPC 문서 근거와 함께 알려줘.", "Ballast, 오일 71.19 MT, 가스 917.23 MT와 MEPC 보고", r"Ballast|밸러스트", r"71\.19", r"917\.23", r"MEPC|보고|reporting|CII"),
]


rows = OPS + TEXT + TABLE + HYBRID
counts = {kind: sum(row["type"] == kind for row in rows) for kind in ("ops", "text", "table", "hybrid")}
assert counts == {"ops": 25, "text": 50, "table": 15, "hybrid": 10}, counts
assert len(rows) == 100
assert len({row["id"] for row in rows}) == 100
assert len({row["question"] for row in rows}) == 100
assert all(row["needles"] for row in rows)

OUTPUT.write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    encoding="utf-8",
)
print(json.dumps({"output": str(OUTPUT), "counts": counts, "n": len(rows)}, ensure_ascii=False))
