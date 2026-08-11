"""Unit checks for rule_lookup_answer repair pipeline."""
from __future__ import annotations

from dataclasses import dataclass

from rule_lookup_answer import (
    build_direct_rule_fact_answer,
    finalize_rule_lookup_answer,
    is_direct_rule_fact_question,
)


@dataclass
class _Chunk:
    chunk_id: str
    file_name: str
    text: str
    page_number: int = 1
    doc_id: str = "kr_1_2025"
    source: str = "KR"
    clause_number: str = ""
    is_catalog_table: bool = False


def _sample_chunks() -> list[_Chunk]:
    return [
        _Chunk("1", "DNV-CG-0264.pdf", "2 Objective guidance for safe implementation of autoremote vessel functions and AROS notations.", 9),
        _Chunk("2", "DNV-CG-0264.pdf", "8.1 General principles for autoremote vessel design fault tolerance.", 24),
        _Chunk("3", "DNV-RU-OU-0103.pdf", "23.4 Application Smart notation may be applied to units in operation.", 106),
        _Chunk("4", "DNV-RU-UWT-Pt5.pdf", "Supporting underwater technology rules reference.", 172),
    ]


def test_strips_hallucination_and_rebuilds_section4() -> None:
    raw = """## 1) 핵심 요약
- **DNV-CG-0264.pdf**: autoremote guidance [1]
- **DNV-RU-OU-0103.pdf**: Smart notation [3]

## 2) 선박 운항/업무 영향
- **DNV-CG-0264.pdf**: autoremote guidance work process [1]
- AROS notation 정의 필요 [2]

## 3) 추후 확인 필요사항
- DNV-RU-SHIP Pt.6 Ch.12 Sec.2 확인 필요

## 4) 관련 선급 Rule / Guidance
- **DNV-RU-SHIP Pt.6 Ch.12**: autoremote [9]
- **DNV-CG-0508**: Smart notation [6]
"""
    chunks = _sample_chunks()
    out, notes = finalize_rule_lookup_answer(raw, chunks)

    assert "DNV-RU-SHIP" not in out
    assert "DNV-CG-0508" not in out
    assert "DNV-RU-UWT-Pt5.pdf" in out
    assert "## 4)" in out
    assert any("duplicate" in n.lower() or "dedupe" in n.lower() or "§2" in n for n in notes)
    assert "DNV-CG-0264.pdf" not in out.split("## 4)")[1] or "본 검색 context" in out


def test_direct_rule_fact_selects_topic_specific_clause() -> None:
    chunks = [
        _Chunk(
            "wrong",
            "1편_2025.pdf",
            "902 선박설비의 정비 및 사이버 유지관리 절차를 회사가 관리하여야 한다.",
            84,
            clause_number="902",
        ),
        _Chunk(
            "right",
            "1편_2025.pdf",
            "902 탈급 1 등록된 선박이 선급검사를 받지 아니한 경우 선급위원회의 결의로 탈급한다. "
            "선박소유자에게 통지하고 6개월 이내에 필요한 조치를 하도록 한다.",
            22,
            clause_number="902",
        ),
    ]
    answer, selected, debug = build_direct_rule_fact_answer(
        "902절 탈급(선급등록 취소)의 적용 대상과 절차는?", chunks
    )

    assert selected is chunks[1]
    assert "탈급" in (answer or "")
    assert "6개월" in (answer or "")
    assert debug["reason"] == "direct_fact_extract"


def test_direct_rule_fact_selects_exact_test_inspection_phrase() -> None:
    chunks = [
        _Chunk("wrong", "1편_2025.pdf", "검사는 정기적으로 시행하여야 한다.", 44),
        _Chunk(
            "right",
            "1편_2025.pdf",
            "시험 및 검사는 원칙적으로 우리 선급 검사원의 입회하에 시행하여야 한다.",
            9,
        ),
    ]
    answer, selected, _debug = build_direct_rule_fact_answer(
        "시험 및 검사는 원칙적으로 어떻게 시행해야 하는가?", chunks
    )

    assert selected is chunks[1]
    assert "검사원의 입회" in (answer or "")


def test_direct_rule_fact_does_not_capture_broad_discovery() -> None:
    assert not is_direct_rule_fact_question(
        "DNV에서 자율운항 또는 Smart Vessel 관련 Rule/Guidance를 찾아줘."
    )


def test_direct_rule_fact_keeps_kr_part1_scope_for_kr_term() -> None:
    chunks = [
        _Chunk(
            "dnv",
            "DNV-RU-SHIP.pdf",
            "Additional class notations DPS indicate requirements for dynamic positioning systems.",
            11,
            doc_id="dnv_rules",
            source="DNV",
        ),
        _Chunk(
            "kr",
            "1편_2025.pdf",
            "선급부호(class notations)는 특정규칙요건을 만족하는 선박의 특징을 나타낸다.",
            7,
        ),
    ]
    answer, selected, _debug = build_direct_rule_fact_answer(
        "선급부호(class notations)는 무엇을 나타내는가?", chunks
    )

    assert selected is chunks[1]
    assert "특정규칙요건" in (answer or "")


def test_direct_rule_fact_joins_split_clause_heading_and_body() -> None:
    chunks = [
        _Chunk(
            "heading",
            "1편_2025.pdf",
            "801. 검사원의 권한 【지침 참조】",
            19,
            clause_number="801",
        ),
        _Chunk(
            "body",
            "1편_2025.pdf",
            "2. 검사준비를 하지 아니할 때 또는 입회자가 없을 때는 검사를 중지할 수 있다.",
            19,
            clause_number="2",
        ),
    ]
    answer, selected, _debug = build_direct_rule_fact_answer(
        "801절에서 검사 준비가 안 되었거나 입회자가 없을 때 검사원은 어떻게 할 수 있는가?",
        chunks,
    )

    assert selected is not None
    assert "801" in (answer or "")
    assert "검사를 중지" in (answer or "")


def test_direct_rule_fact_normalizes_korean_verb_ending() -> None:
    chunks = [
        _Chunk(
            "wrong",
            "1편_2025.pdf",
            "첫 번째 선급은 현재 선박이 등록되어 있는 선급을 말한다.",
            8,
        ),
        _Chunk(
            "right",
            "1편_2025.pdf",
            "등록된 선박은 선급을 유지하기 위하여 규정된 선급검사를 받아야 한다.",
            9,
        ),
    ]
    answer, selected, _debug = build_direct_rule_fact_answer(
        "등록된 선박이 선급을 유지하려면 무엇을 해야 하는가?", chunks
    )

    assert selected is chunks[1]
    assert "선급검사" in (answer or "")


def test_direct_rule_fact_prioritizes_exact_construction_contract_term() -> None:
    chunks = [
        _Chunk("wrong", "1편_2025.pdf", "등록 선박은 계선 사실을 선급에 통보하여야 한다.", 73),
        _Chunk(
            "right",
            "1편_2025.pdf",
            "검사신청자는 검사신청 시 선급에 건조계약일과 선번을 통보하여야 한다.",
            7,
        ),
    ]
    answer, selected, _debug = build_direct_rule_fact_answer(
        "건조계약일 신청 시 선급에 무엇을 통보해야 하는가?", chunks
    )

    assert selected is chunks[1]
    assert "선번" in (answer or "")


def test_direct_rule_fact_prioritizes_exact_test_and_inspection_term() -> None:
    chunks = [
        _Chunk("wrong", "1편_2025.pdf", "검사는 원칙적으로 정해진 주기에 시행한다.", 73),
        _Chunk(
            "right",
            "1편_2025.pdf",
            "시험 및 검사는 원칙적으로 우리 선급 검사원의 입회하에 시행하여야 한다.",
            9,
        ),
    ]
    answer, selected, _debug = build_direct_rule_fact_answer(
        "시험 및 검사는 원칙적으로 어떻게 시행해야 하는가?", chunks
    )

    assert selected is chunks[1]
    assert "검사원의 입회" in (answer or "")


if __name__ == "__main__":
    test_strips_hallucination_and_rebuilds_section4()
    print("ok")
