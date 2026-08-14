from __future__ import annotations

from services.korean_output import english_prose_leak_lines


def test_detects_english_source_sentence_in_answer_body() -> None:
    answer = """## 1) 핵심 요약

- This Guide provides technical and survey requirements for vessels fitted with Smart Functions and may grant optional class notations. [1]

## 2) 선박 운항/업무 영향

- 설계와 검사 업무에 반영해야 합니다. [1]

## 4) 관련 선급 Rule / Guidance

- Guide for Smart Functions for Marine Vessels and Offshore Units [1]
"""
    leaks = english_prose_leak_lines(answer)
    assert len(leaks) == 1
    assert leaks[0].startswith("- This Guide")


def test_allows_korean_prose_and_english_document_title() -> None:
    answer = """## 1) 핵심 요약

- 이 Guide는 Smart Functions가 설치된 선박의 기술·검사 요건을 제시합니다. [1]

## 4) 관련 선급 Rule / Guidance

- Guide for Smart Functions for Marine Vessels and Offshore Units [1]
"""
    assert english_prose_leak_lines(answer) == []
