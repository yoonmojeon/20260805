from __future__ import annotations

from types import SimpleNamespace

from services.rag_answer_guard import guard_rag_answer


def chunk(file_name: str, page: int, text: str, number: int = 1):
    return SimpleNamespace(
        chunk_id=f"c-{number}",
        file_name=file_name,
        page_number=page,
        source="ABS" if file_name.startswith(("Guide", "Requirements")) else "MEPC",
        text=text,
    )


def payload(*chunks):
    return {"search_out": {"retrieval_pool": list(chunks), "retrieved": list(chunks)}}


def test_negative_record_lookup_rejects_instead_of_describing_document():
    evidence = chunk(
        "GuideforSmartFunctionsforMarineVesselsandOffshoreUnits-v8.pdf",
        3,
        "ABS introduced this Guide to provide technical and survey requirements.",
    )
    result = guard_rag_answer(
        "ABS Guide에서 IMO MASS Code 의무발효일을 찾아줘.",
        "일반 문서 소개",
        payload(evidence),
        model="gemma4:12b",
    )
    assert result.mode == "negative_rejection"
    assert "확인할 수 없습니다" in result.answer
    assert "대체하지 않습니다" in result.answer


def test_abs_risk_category_is_rebuilt_from_exact_clause():
    evidence = chunk(
        "RequirementsforAutonomousandRemoteControlFunctions-v4.pdf",
        40,
        "Risk levels are assigned based on the Operations Supervision Level and "
        "Consequences of Failure. The matrix assigns a risk category L: Low, "
        "M: Medium, H: High.",
    )
    result = guard_rag_answer(
        "ABS Requirements에서 기능 위험범주는 어떻게 정해져?",
        "## 1) 핵심 요약\n\n## 2) 선박 운항/업무 영향\n\n## 3) 추후 확인 필요사항\n\n## 4) 관련 선급 Rule / Guidance",
        payload(evidence),
        model="gemma4:12b",
    )
    assert result.mode == "exact_fact_extract"
    assert "운항감독 수준" in result.answer
    assert "저위험(Low)·중위험(Medium)·상위험(High)" in result.answer
    assert result.evidence_table and result.evidence_table[0]["page"] == 40


def test_abs_risk_false_premise_gets_explicit_verdict():
    evidence = chunk(
        "RequirementsforAutonomousandRemoteControlFunctions-v4.pdf",
        40,
        "2.3 Risk Matrix TABLE 3 Risk Category. Risk levels are assigned based on "
        "the Operations Supervision Level and Consequences of Failure. "
        "L: Low, M: Medium, H: High.",
    )
    result = guard_rag_answer(
        "ABS Requirements에서 모든 자율·원격제어 기능에는 같은 위험범주가 적용된다는 전제가 맞는지 검증해줘.",
        "일반 설명",
        payload(evidence),
        model="gemma4:12b",
    )
    assert "전제는 틀렸습니다" in result.answer


def test_abs_smart_scope_is_rebuilt_from_exact_clause():
    scope = chunk(
        "GuideforSmartFunctionsforMarineVesselsandOffshoreUnits-v8.pdf",
        17,
        "This Guide is applicable to all marine vessels and offshore units. "
        "It covers SF categories SHM and MHM within optional Smart Function "
        "class notations SMART (INF), SMART (SHM) and SMART (MHM).",
    )
    result = guard_rag_answer(
        "ABS Guide for Smart Functions의 적용 대상과 포함 SF 범위는 무엇이야?",
        "영문 원문 근거는 확인했지만 한국어 변환을 완료하지 못했습니다.",
        payload(scope),
        model="gemma4:12b",
    )
    assert result.mode == "exact_fact_extract"
    assert "모든 해양선박과 해양구조물" in result.answer
    assert "SHM과 MHM" in result.answer


def test_sfcs_deadline_is_copied_from_current_evidence():
    evidence = chunk(
        "MEPC 84-7-14 - Report of ISWG-GHG 20.pdf",
        6,
        "The recognized list of sustainable fuels certification schemes "
        "(SFCS) should be published by 1 March 2027.",
    )
    result = guard_rag_answer(
        "MEPC 84/7/14에서 SFCS 인정 목록 공표 시한은 언제야?",
        "관련 GFI 논의입니다.",
        payload(evidence),
        model="gemma4:12b",
    )
    assert result.mode == "exact_fact_extract"
    assert "2027년 3월 1일" in result.answer


def test_blank_sections_are_filled_without_an_llm_retry():
    answer = (
        "## 1) 핵심 요약\n\n- 확인된 사실입니다. [1]\n\n"
        "## 2) 선박 운항/업무 영향\n\n"
        "## 3) 추후 확인 필요사항\n\n"
        "## 4) 관련 선급 Rule / Guidance\n\n- 문서입니다. [1]"
    )
    evidence = chunk("document.pdf", 1, "확인된 사실입니다.")
    result = guard_rag_answer(
        "일반 질문",
        answer,
        payload(evidence),
        model="gemma4:12b",
    )
    assert "검색 근거에서 직접 확인되는 별도 운항·업무 영향이 없습니다" in result.answer
    assert "추가 확인 필요사항이 별도로 식별되지 않았습니다" in result.answer


def test_compact_exact_fact_is_not_expanded_back_to_four_sections():
    answer = (
        "## 1) 핵심 요약\n\n- 전력 케이블은 1 kV 및 3 kV급입니다. [1]\n\n"
        "## 4) 관련 선급 Rule / Guidance\n\n- DNV-CP-0399, p.5 [1]"
    )
    evidence = chunk("DNV-CP-0399.pdf", 5, "power cables for rated voltages 1 kV and 3 kV")
    exact_payload = payload(evidence)
    exact_payload["answer_out"] = {
        "verification_summary": {
            "answer_length_contract": {"answer_profile": "exact_rule_fact"}
        }
    }

    result = guard_rag_answer(
        "DNV-CP-0399은 어떤 정격 케이블에 적용되나요?",
        answer,
        exact_payload,
        model="gemma4:12b",
    )

    assert "## 2)" not in result.answer
    assert "## 3)" not in result.answer
