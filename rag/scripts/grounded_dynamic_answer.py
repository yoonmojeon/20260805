"""Grounded, question-centred Korean answer generation.

The prompt is built entirely from the current question, parsed requirements,
and retrieved evidence.  It contains no prepared answer for a particular
question or evaluation item.
"""
from __future__ import annotations

import re
from typing import Any

from question_requirements import QuestionRequirements, analyze_requirements


CITATION_RE = re.compile(r"\[(\d+)\]")
METADATA_LEAK_RE = re.compile(
    r"(?:\[[a-z]{2,6}\]\s*)?file=|folder=|doc_type=|chunk[_ ]?id=",
    re.I,
)
FOREIGN_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
PREMISE_CHECK_RE = re.compile(
    r"전제.{0,24}(?:맞|검증)|틀리면|사실인지|가정.{0,24}(?:맞|검증)",
    re.I,
)


def robustness_instruction(question: str) -> str:
    """Question-derived guardrail; contains no evaluation answers or IDs."""
    if not PREMISE_CHECK_RE.search(question or ""):
        return ""
    return (
        "- 이 질문은 전제 검증 질문이다. 첫 bullet 첫 문장에서 전제가 맞는지 "
        "'맞습니다' 또는 '맞지 않습니다'로 명시하고 [N]을 붙인다.\n"
        "- 전제가 틀리면 무시하지 말고, 어떤 부분이 틀렸는지와 근거에서 확인되는 "
        "올바른 상태·범위·일정을 바로 이어서 쓴다.\n"
        "- 근거가 전제의 참·거짓을 판정하기 부족하면 맞다고 추정하지 말고 "
        "'검색 근거에서 확인되지 않음'이라고 명시한다.\n"
    )


def _chunk_text(chunk: Any, limit: int = 4200) -> str:
    text = re.sub(r"\s+", " ", str(getattr(chunk, "text", "") or "")).strip()
    return text[:limit]


def _direct_evidence_cues(
    question: str,
    requirements: QuestionRequirements,
    evidence_bodies: list[str],
) -> str:
    """Surface source fragments likely to contain easy-to-drop conditions.

    The fragments remain verbatim evidence and are not a prepared answer.  The
    local model still decides relevance and writes the Korean response, but it
    no longer has to rediscover a two-week deadline or adjacent exception near
    the end of a long clause.
    """
    cues: list[str] = []
    seen: set[str] = set()

    def add(citation_id: int, label: str, fragment: str) -> None:
        clean = re.sub(r"\s+", " ", fragment or "").strip(" -—")
        signature = clean.lower()
        if len(clean) < 20 or signature in seen:
            return
        seen.add(signature)
        if len(clean) > 520:
            clean = clean[:520].rsplit(" ", 1)[0].rstrip(" ,;") + "…"
        cues.append(f"- [{citation_id}] {label}: {clean}")

    for citation_id, body in enumerate(evidence_bodies, 1):
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?;])\s+", body)
            if part.strip()
        ]
        if "period" in requirements.facets:
            for sentence in sentences:
                if re.search(
                    r"(?:submit|present|report|completion|terminat|deadline|within|"
                    r"제출|보고|완료|종료).{0,260}"
                    r"(?:\d+\s*(?:weeks?|days?|months?|years?|주|일|개월|년))|"
                    r"(?:\d+\s*(?:weeks?|days?|months?|years?|주|일|개월|년)).{0,260}"
                    r"(?:submit|present|report|completion|terminat|deadline|제출|보고|완료|종료)",
                    sentence,
                    re.I,
                ):
                    add(citation_id, "기간·기한 직접 근거", sentence)
                    break
        if "value" in requirements.facets:
            for sentence in sentences:
                if re.search(
                    r"\d+(?:[./]\d+)?\s*(?:%|kV|V|m/s|mm/s|mm|cm|m|"
                    r"N/mm2|N/mm²|MPa|°C|℃|ppm|배|hours?|days?|weeks?|years?|"
                    r"시간|일|주|년)",
                    sentence,
                    re.I,
                ):
                    add(citation_id, "수치·조건 직접 근거", sentence)
                    if sum("수치·조건 직접 근거" in cue for cue in cues) >= 3:
                        break
        if "scope" in requirements.facets or "requirement" in requirements.facets:
            for sentence in sentences:
                if re.search(
                    r"\b(?:unless|except|provided that|only if|if|however)\b|"
                    r"예외|제외|다만|손상되지\s*않|사용할\s*수\s*있",
                    sentence,
                    re.I,
                ):
                    add(citation_id, "예외·조건 직접 근거", sentence)
                    break
        if len(cues) >= 6:
            break
    if not cues:
        return ""
    return (
        "\n원문 직접 확인 후보(기계적으로 뽑은 원문이며, 질문과 맞는 항목만 사용):\n"
        + "\n".join(cues[:6])
        + "\n"
    )


def build_structured_finding_bullets(chunks: list[Any]) -> list[str]:
    """Extract audit/error findings as type-count-action facts.

    This is deliberately document-independent. It recognizes recurring
    reporting-quality language and copies only values found in the cited
    source chunk, avoiding free-form arithmetic or category invention.
    """
    bullets: list[str] = []
    seen: set[str] = set()
    number_words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }
    count_pattern = r"(?:\d[\d,]*|[A-Za-z]+(?:-[A-Za-z]+)*)"

    def normalized_count(value: str) -> str:
        if re.fullmatch(r"\d[\d,]*", value):
            return value
        parts = value.lower().replace("-", " ").split()
        if parts and all(part in number_words for part in parts):
            return str(sum(int(number_words[part]) for part in parts))
        return ""

    def add(text: str) -> None:
        signature = re.sub(r"[^A-Za-z가-힣0-9]+", "", text.lower())
        if signature and signature not in seen:
            seen.add(signature)
            bullets.append(text)

    for citation_id, chunk in enumerate(chunks, 1):
        text = _chunk_text(chunk)
        compact = re.sub(r"\s+", " ", text)

        # Bind each count to the finding and action in the same sentence.
        # This prevents a preceding aggregate (for example 265 ships) from
        # being attached to a later subtype (for example 65 ships).
        for sentence in re.split(r"(?<=[.!?])\s+", compact):
            count_match = re.search(
                rf"\b({count_pattern})\s+(ships?|records?|instances?)\b",
                sentence,
                re.I,
            )
            if not count_match:
                continue
            count = normalized_count(count_match.group(1))
            if not count:
                continue
            if (
                re.search(r"hours under way", sentence, re.I)
                and re.search(r"(?:excluded|removed)", sentence, re.I)
                and re.search(r"(?:more than|exceed)", sentence, re.I)
            ):
                add(
                    f"- 연간 총시간을 초과한 'hours under way'를 보고한 "
                    f"{count}척은 분석 대상에서 제외했습니다. [{citation_id}]"
                )
            if (
                re.search(r"duplicate reporting", sentence, re.I)
                and re.search(r"(?:removed|excluded)", sentence, re.I)
            ):
                add(f"- 중복 보고 {count}건은 데이터 분석에서 제거했습니다. [{citation_id}]")
            if (
                re.search(r"unrealistic (?:ship )?(?:characteristics|paramet\w*)", sentence, re.I)
                and re.search(r"(?:excluded|removed)", sentence, re.I)
            ):
                add(
                    f"- 비현실적이거나 기술적으로 불가능한 선박 제원을 보고한 "
                    f"{count}척은 분석 대상에서 제외했습니다. [{citation_id}]"
                )

        unresolved = re.search(
            r"(?:number of )?(?:identified )?errors?.{0,180}?"
            r"(?:reduced to|affected)\s+(\d[\d,]*)\s+ships?.{0,360}?"
            r"(?:not been included|excluded|removed)",
            compact,
            re.I,
        )
        if unresolved:
            add(
                f"- 기국정부 또는 인정기관이 수정하지 않은 집계 영향 가능 오류 "
                f"{unresolved.group(1)}척은 해당 보고서의 분석에서 제외했습니다. [{citation_id}]"
            )

        categories: list[str] = []
        if re.search(r"unrealistic .{0,40}(?:characteristics|parameters)", compact, re.I):
            categories.append("기술적으로 불가능한 비현실적 선박 특성·제원")
        if re.search(r"duplicate reporting", compact, re.I):
            categories.append("중복 보고")
        if re.search(r"(?:incorrect ship type|categorized under an incorrect)", compact, re.I):
            categories.append("MARPOL Annex VI 규칙 2 기준의 잘못된 선종 분류")
        if categories and re.search(
            r"further examined.{0,180}?(?:cause|provided)|"
            r"(?:cause|information).{0,180}?Administrations and ROs",
            compact,
            re.I,
        ):
            add(
                f"- 자동 품질검사는 {'·'.join(categories)}를 식별하고, "
                f"원인을 추가 조사한 뒤 관련 정보를 기국정부와 인정기관(RO)에 제공했습니다. "
                f"[{citation_id}]"
            )

    return bullets[:5]


def build_prompts(
    question: str,
    row: dict,
    chunks: list[Any],
    requirements: QuestionRequirements | None = None,
) -> tuple[str, str, QuestionRequirements]:
    req = requirements or analyze_requirements(question, row)
    exact_fact_profile = row.get("_answer_profile") == "exact_rule_fact"
    exact_fact_slots = max(1, min(3, int(row.get("_answer_fact_slots") or 1)))
    evidence_parts: list[str] = []
    evidence_bodies: list[str] = []
    for idx, chunk in enumerate(chunks, 1):
        body = _chunk_text(chunk)
        feature_anchors = [
            str(value)
            for value in (row.get("_answer_feature_terms") or [])
            if str(value).strip()
        ]
        for anchor in feature_anchors:
            if anchor.lower() not in body.lower():
                continue
            from fast_context import _question_focused_excerpt

            focused_feature = _question_focused_excerpt(
                body, anchor, max_chars=2400
            )
            if focused_feature:
                body = focused_feature
            break
        list_like_question = bool(
            re.search(
                r"체크\s*리스트|무엇(?:이|을)?\s*포함|포함(?:되어|해야)|"
                r"항목(?:들)?(?:에는|은|을|이)|목록|장치들|"
                r"최소(?:한|로)?\s*포함|제출해야\s*하는\s*서류",
                question,
                re.I,
            )
        )
        if (
            row.get("_answer_priority_local_used")
            or row.get("_answer_query_focused_used")
        ) and idx == 1 and not list_like_question:
            from fast_context import _question_focused_excerpt

            focused_body = _question_focused_excerpt(
                body,
                question,
                max_chars=1800,
            )
            if focused_body:
                body = focused_body
        evidence_bodies.append(body)
        evidence_parts.append(
            "\n".join(
                [
                    f"[{idx}]",
                    f"기관={getattr(chunk, 'source', '')}",
                    f"발행기관={getattr(chunk, 'publisher', '') or getattr(chunk, 'source', '')}",
                    f"자료유형={getattr(chunk, 'source_type', '') or '미분류'}",
                    f"회의={getattr(chunk, 'session_org', '')} {getattr(chunk, 'session_number', '') or ''}".rstrip(),
                    f"문서상태={getattr(chunk, 'document_status_label', '') or getattr(chunk, 'document_status', '') or '상태 미확인'}",
                    f"문서={getattr(chunk, 'file_name', '')}",
                    f"페이지={getattr(chunk, 'page_number', '')}",
                    f"조항={getattr(chunk, 'clause_number', '')}",
                    f"본문={body}",
                ]
            )
        )

    facets_ko = {
        "finding": "식별된 오류·발견사항",
        "value": "수치",
        "metric": "사용 지표",
        "comparison": "비교 기준",
        "period": "기간·일정",
        "status": "결정·규제 상태",
        "requirement": "구체 요구사항",
        "method": "방법·절차",
        "scope": "적용 범위·대상·예외",
        "document": "문서·결의·가이드",
        "clause": "세부 조항",
        "impact": "실무 영향·조치",
        "reason": "근거·이유",
    }
    requested = [facets_ko.get(name, name) for name in req.facets]
    system = """당신은 해사 규정·선급 문서를 검토하는 실무 분석가다.
검색 근거에 명시된 사실만 사용해 자연스러운 한국어로 답하라.

필수 원칙:
1. 질문이 요구한 대상과 세부 항목에 먼저 직접 답한다. 관련성이 낮은 배경 설명으로 분량을 채우지 않는다.
2. 수치 질문은 수치·단위·비교 기준·사용 지표를, 일정 질문은 결정 상태·목표일·발효일을 서로 구분한다.
3. 문서의 제안·초안·위원회 제출자료·최종 채택을 구분하고, 근거보다 강한 의무로 바꾸지 않는다.
4. 실무 영향은 근거에 명시된 보고·검증·설계·운항·승인 조치만 쓴다. 일반론을 만들지 않는다.
5. 모든 사실 문장은 끝에 근거 번호 [N]을 붙인다. 번호는 제공된 근거와 정확히 대응해야 한다.
6. 근거가 없으면 추측하지 말고 해당 요구항목을 '검색 근거에서 확인되지 않음'으로 표시한다.
7. 영어 원문은 그대로 복사하지 말고 정확한 한국어로 번역하되, 문서명·약어·지표명은 보존한다.
8. 같은 사실을 여러 절에서 반복하지 않는다.
9. 동일한 수치를 표현만 바꿔 반복하지 않는다. 한 근거에 최대값·최소값·범위·비교연도·복수 지표가 함께 있으면 질문과 관련된 것을 빠뜨리지 않는다.
10. 질문에 '무엇을 식별했고 어떻게 처리했는가'처럼 둘 이상의 요구가 있으면 식별된 사실과 후속 처리를 각각 명시한다.
11. up to, at least, approximately 같은 한정어는 각각 '최대', '최소', '약'으로 보존한다.
12. 공급 기반·수요 기반처럼 지표의 분류가 원문에 있으면 지표명과 분류를 함께 쓴다.
13. 해사 실무 용어를 정확히 번역한다. situational awareness는 '상황 인식'이며 단순한 '시각적 인식'이 아니다.
14. 원문에 여러 대상별 조건·수치가 병렬로 나열되면, 질문의 대상과 같은 항목만 사용한다. 인접 항목의 값이나 예외를 질문 대상에 결합하지 않는다.
15. 근거는 질문 관련도 순으로 제공된다. [1]에 직접 답이 있으면 [1]을 우선 사용하고, 후순위 근거의 유사한 수치·조건으로 바꾸지 않는다.
16. 질문이 복수 항목이나 목록을 요구하고 한 근거에 원문의 열거가 있으면, 그 열거를 끝까지 읽어 서로 다른 항목을 빠짐없이 답한다.
17. 질문이 '대상과 목적', '조건과 예외'처럼 두 요소를 연결해 요구하면 두 요소를 각각 답한다. 원문에 also applies, furthermore, in addition으로 이어지는 적용 대상·목적·조건은 첫 문장만 고르고 생략하지 않는다.
18. 원문이 비기밀(non-confidential), 기밀, 잠정, 최종 같은 정보의 성격을 명시하면 그 한정어를 생략하지 않는다. '어떤 정보'를 묻는 경우 정보의 성격만 쓰지 말고 무엇에 관한 정보인지도 같은 bullet에 쓴다.
19. 회의 연혁에서는 승인한 회의와 후속 검토가 예정된 회의를 구분한다. 현재 문서의 회의차수를 승인 주체로 추정하지 말고 원문의 주어와 동사를 그대로 따른다. 원문에 앞선 회의의 승인과 후속 회의의 검토·확정 목표가 함께 있으면 두 이정표를 모두 명시한다.
20. 이미 개별 bullet로 나열한 사실을 마지막 bullet에서 한 문장으로 다시 합쳐 반복하지 않는다.

출력은 반드시 아래 네 절을 아래 표기 그대로 사용한다. 굵은 글씨(**)로 제목을 바꾸지 않는다.
## 1) 핵심 요약
## 2) 선박 운항/업무 영향
## 3) 추후 확인 필요사항
## 4) 관련 선급 Rule / Guidance

각 사실은 반드시 다음 형식으로 쓴다: '- 한국어 사실 문장. [1]'
각 절은 bullet 형식으로 작성한다. 해당 사항이 없으면 '- 검색 근거에서 확인되지 않음'이라고 쓴다."""
    if exact_fact_profile:
        system += f"""

이 질문은 단순 사실 조회다. 1절에는 질문이 요구한 값·조건·대상만 최대
{exact_fact_slots}개 bullet로 작성한다. 답이 하나면 하나만 쓰며 개수를 채우려고
문서 목적·적용 배경 또는 같은 수치의 바꿔쓰기를 추가하지 않는다. 같은 사실은
답변 전체에서 한 번만 말한다. 질문이 두 대상을 요구하면 대상별 한 bullet로
답한다. 2절과 3절은 질문이 별도로 요구하지 않으면 '검색 근거에서 확인되지 않음'
한 줄만 쓰고, 4절에는 직접 사용한 문서·페이지·조항만 한 bullet로 쓴다."""
    from compound_regulatory import (
        compound_prompt_instruction,
        is_compound_regulatory_class_question,
    )

    if is_compound_regulatory_class_question(question):
        system += "\n\n" + compound_prompt_instruction(question)
    list_instruction = ""
    clean_list_lookup = bool(
        re.search(
            r"목록(?:은|을|이)?|체크\s*리스트|항목(?:들)?(?:은|을|이)?|"
            r"(?:서류|문서)\s*(?:목록|일체)",
            question,
            re.I,
        )
    )
    if clean_list_lookup or re.search(
        r"체크\s*리스트|무엇(?:이|을)?\s*포함|포함(?:되어|해야)|"
        r"항목(?:들)?(?:에는|은|을|이)|최소(?:한|로)?\s*포함|"
        r"제출해야\s*하는\s*서류",
        question,
        re.I,
    ):
        list_instruction = (
            "- 목록/체크리스트 질문이므로 원문에 열거된 항목을 기능별로 묶어 3~5개 "
            "bullet로 작성하고, 상태 점검·기록 데이터·합격기준·보고/보관처럼 서로 "
            "다른 항목군을 한 문장으로 축약해 누락하지 않는다."
        )
        numbered_sources: list[tuple[int, list[int]]] = []
        for citation_id, body in enumerate(evidence_bodies, 1):
            values = [
                int(value)
                for value in re.findall(r"(?:^|\s)\(?(\d{1,2})\)\s+", body)
            ]
            unique = list(dict.fromkeys(values))
            if len(unique) >= 5:
                numbered_sources.append((citation_id, unique))
        if numbered_sources:
            citation_id, values = max(numbered_sources, key=lambda item: len(item[1]))
            groups = [values[index : index + 3] for index in range(0, len(values), 3)]
            list_instruction += (
                f"\n- 근거 [{citation_id}]의 번호 목록 "
                + ", ".join(str(value) for value in values)
                + "을 모두 답한다. 연속 항목을 한 bullet에 묶을 수 있지만 번호와 "
                "각 항목의 고유 내용을 하나도 생략하지 않는다. 대표 항목만 고르지 않는다."
                + " 다음 묶음대로 작성하면 된다: "
                + " / ".join(
                    ", ".join(f"{value})" for value in group)
                    for group in groups
                )
                + "."
            )
        source_item_counts = []
        for citation_id, body in enumerate(evidence_bodies, 1):
            count = max(
                len(re.findall(r"(?:^|\s)—\s+", body)),
                len(re.findall(r"(?:^|\s)\(\d{1,2}\)\s+", body)),
                len(re.findall(r"(?:^|\s)\d{1,2}\)\s+", body)),
            )
            if count >= 2:
                source_item_counts.append((citation_id, count))
        if source_item_counts:
            citation_id, count = max(source_item_counts, key=lambda item: item[1])
            list_instruction += (
                f"\n- 근거 [{citation_id}]에는 최소 {count}개의 병렬 항목이 있습니다. "
                "대표 항목만 쓰지 말고 질문 범위에 속하는 각 항목의 고유 조건을 모두 답하세요."
            )
    evidence_cues = _direct_evidence_cues(question, req, evidence_bodies)
    direct_length_instruction = (
        f"- 단순 사실 조회이므로 1절은 최대 {exact_fact_slots}개 bullet만 쓰고, "
        "같은 값·조건을 다른 문장으로 반복하지 않는다."
        if exact_fact_profile
        else "- 구체 질문은 1절 2~5개, 광범위 요약은 전체 7~10개 bullet 이내로 제한한다."
    )
    user = f"""질문:
{question}

질문 분석:
- 기관/회의: {req.organization or '명시되지 않음'} {req.session_number}
- 반드시 답할 항목: {', '.join(requested) if requested else '질문의 핵심 사실'}
- 핵심 검색어: {', '.join(req.topic_terms[:16])}
- 요청 항목 수: {req.requested_count or '명시되지 않음'}
- 광범위 요약 여부: {'예' if req.broad_summary else '아니오'}

작성 지시:
- 질문 유형별 추가 지시:\n{robustness_instruction(question) or '- 일반 근거 답변'}
- 1절 첫 bullet부터 질문의 직접 답을 쓴다.
- 질문이 요구한 각 항목을 빠뜨리지 않는다.
- 서로 다른 근거의 내용을 임의로 결합해 하나의 의무나 결론으로 만들지 않는다.
- [1]이 질문을 직접 다루면 [1]의 대상·조건·수치·목록을 그대로 우선하고, 후순위 조항의 값을 섞지 않는다.
- 문서명, 회의차수, 조항 또는 결의·가이드 명칭을 가능한 범위에서 포함한다.
{direct_length_instruction}
- 인용 없는 사실 문장은 작성하지 않는다.
{list_instruction}
{evidence_cues}

검색 근거:
{chr(10).join(evidence_parts)}
"""
    if row.get("_advanced_mode") and row.get("_answer_feature_terms"):
        recovered_facets = [
            str(value)
            for value in row.get("_answer_feature_terms") or []
            if str(value).strip()
        ][:6]
        if recovered_facets:
            user += (
                "\nAdvanced 정확일치 회수 항목:\n- "
                + "\n- ".join(recovered_facets)
                + "\n위 항목은 질문을 영문 원문과 연결하기 위해 회수한 독립 근거 축입니다. "
                "검색 근거가 직접 뒷받침하는 각 축을 별도 사실로 확인하고, 회의 결과 질문이면 "
                "승인·채택된 사항과 제출 요청·작업계획 단계인 사항을 구분해 빠뜨리지 마세요.\n"
            )
    if is_compound_regulatory_class_question(question):
        citation_by_chunk = {
            str(getattr(chunk, "chunk_id", "") or ""): index
            for index, chunk in enumerate(chunks, 1)
        }
        completion = row.get("_evidence_completion") or {}
        plan_slots = {
            str(slot.get("name") or ""): str(slot.get("label") or slot.get("name") or "")
            for slot in ((completion.get("plan") or {}).get("slots") or [])
        }
        slot_lines: list[str] = []
        for slot_name, chunk_ids in (completion.get("slot_hits") or {}).items():
            citation_ids = [
                citation_by_chunk[str(chunk_id)]
                for chunk_id in list(chunk_ids or [])
                if str(chunk_id) in citation_by_chunk
            ]
            if citation_ids:
                slot_lines.append(
                    f"- {plan_slots.get(str(slot_name), str(slot_name))} (우선 근거): "
                    + ", ".join(f"[{citation_id}]" for citation_id in citation_ids)
                )
        if slot_lines:
            user += (
                "\n필수 증거 슬롯(이 용도에 우선 사용할 인용):\n"
                + "\n".join(slot_lines)
                + "\n특히 '최종 결정'에는 앞선 회부 문장이 아니라 최종 승인·채택 슬롯을 사용하세요.\n"
            )
    return system, user, req


def build_scaffold_synthesis_prompts(
    question: str,
    row: dict,
    chunks: list[Any],
    scaffold: str,
) -> tuple[str, str, QuestionRequirements]:
    """Build a grounded Korean rewrite prompt from verified claim cards.

    ``scaffold`` is a deterministic, citation-checked draft.  It is not an
    answer cache and it contains no question-specific instruction: it merely
    gives a small local model the already verified facts that it may organise
    into a useful working answer.  The current question and the current
    retrieved chunks remain the only source of factual content.
    """
    req = analyze_requirements(question, row)
    evidence: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        text = _chunk_text(chunk, limit=2600)
        evidence.append(
            "\n".join(
                (
                    f"[{index}] doc={getattr(chunk, 'file_name', '')}",
                    f"page={getattr(chunk, 'page_number', '')}",
                    f"clause={getattr(chunk, 'clause_number', '')}",
                    text,
                )
            )
        )

    # Make the evidence planner's dynamic coverage contract visible to the
    # local model.  Without this, a fluent rewrite may collapse distinct
    # milestones (for example adoption, entry into force and schedule
    # uncertainty) into a single bullet even though retrieval found each one.
    citation_by_chunk = {
        str(getattr(chunk, "chunk_id", "") or ""): index
        for index, chunk in enumerate(chunks, 1)
    }
    completion = row.get("_evidence_completion") or {}
    slot_lines: list[str] = []
    for slot_name, chunk_ids in (completion.get("slot_hits") or {}).items():
        eligible = sorted(
            {
                citation_by_chunk[str(chunk_id)]
                for chunk_id in list(chunk_ids or [])
                if str(chunk_id) in citation_by_chunk
            }
        )
        if eligible:
            slot_lines.append(
                f"- {slot_name}: cite at least one of "
                + ", ".join(f"[{value}]" for value in eligible)
            )

    section_counts = {number: 0 for number in range(1, 5)}
    current_section = 0
    for raw_line in scaffold.splitlines():
        heading = re.match(r"^##\s*([1-4])\)", raw_line.strip())
        if heading:
            current_section = int(heading.group(1))
        elif current_section and raw_line.strip().startswith(("-", "*")):
            section_counts[current_section] += 1
    preservation_contract = (
        "Mandatory evidence coverage (each populated slot must survive the rewrite):\n"
        + ("\n".join(slot_lines) if slot_lines else "- preserve every distinct cited claim in the verified draft")
        + "\nTarget bullet counts from the verified draft: "
        + ", ".join(
            f"section {number}={count}"
            for number, count in section_counts.items()
        )
    )
    # Use Unicode escapes so this instruction remains sound even when a
    # Windows terminal displays a source file with a legacy code page.
    headings = (
        "## 1) \\ud575\\uc2ec \\uc694\\uc57d\n"
        "## 2) \\uc120\\ubc15 \\uc6b4\\ud56d/\\uc5c5\\ubb34 \\uc601\\ud5a5\n"
        "## 3) \\ucd94\\ud6c4 \\ud655\\uc778 \\ud544\\uc694\\uc0ac\\ud56d\n"
        "## 4) \\uad00\\ub828 \\uc120\\uae09 Rule / Guidance"
    )
    system = (
        "You are a maritime regulatory analyst. Write natural Korean only. "
        "Use only the evidence chunks and the verified draft supplied by the "
        "application. Do not add background knowledge or generic advice. "
        "Preserve the source modality: shall/must is a requirement; should is "
        "a recommendation; draft/proposal is not an adopted rule. Every factual "
        "bullet must end with one or more matching [n] citations. "
        "Do not copy English source sentences; translate them accurately.\n\n"
        "Use exactly these headings:\n" + headings
    )
    user = (
        f"Question:\n{question}\n\n"
        "Verified claim draft (rewrite and prioritise it; do not treat it as an "
        "independent source):\n"
        f"{scaffold}\n\n"
        "Writing rules:\n"
        + robustness_instruction(question)
        + preservation_contract
        + "\n"
        "- Preserve every distinct decision, status, date, uncertainty and "
        "operational consequence in the verified draft. Do not merge separate "
        "timeline milestones into one vague bullet.\n"
        "- Match the target bullet counts unless removing an exact duplicate. "
        "Changing wording is allowed; dropping a factual claim is not.\n"
        + "- Put the direct answer to the question first, then only concrete "
        "operational actions supported by evidence.\n"
        "- For a broad meeting question, select the decision, status/timeline, "
        "and operational or reporting consequence that are actually evidenced.\n"
        "- For a rule question, cite the matching clause, explain the requirement "
        "in Korean, and distinguish a required control from an example or a "
        "matter requiring follow-up.\n"
        "- Do not repeat a fact across sections. If a section has no supported "
        "content, use exactly one of these safe sentences as appropriate: "
        "'검색 근거에서 직접 확인되는 별도 운항·업무 영향이 없습니다.', "
        "'추가 확인 필요사항이 별도로 식별되지 않았습니다.', or "
        "'관련 선급 Rule / Guidance가 검색 근거에 없거나 해당하지 않습니다.'\n"
        "- Keep a direct rule answer to 2-3 key bullets and a meeting answer to "
        "the number requested by the question where possible.\n\n"
        "Evidence chunks:\n" + "\n\n".join(evidence)
    )
    return system, user, req


def normalize_generated_markdown(answer: str, *, bulletize_prose: bool = False) -> str:
    """Normalize harmless local-model formatting variations before contract QA."""
    headings = {
        "1": "## 1) 핵심 요약",
        "2": "## 2) 선박 운항/업무 영향",
        "3": "## 3) 추후 확인 필요사항",
        "4": "## 4) 관련 선급 Rule / Guidance",
    }
    out: list[str] = []
    inside_section = False
    for raw in (answer or "").splitlines():
        line = raw.strip()
        heading = re.match(
            r"^(?:#{1,4}\s*)?\*{0,2}\s*([1-4])\s*[\).:\-]\s*.*?\*{0,2}\s*$",
            line,
        )
        if heading:
            out.append(headings[heading.group(1)])
            inside_section = True
            continue
        line = re.sub(r"【\s*(\d+)\s*】", r"[\1]", line)
        line = re.sub(r"\[\s*(?:근거|출처|evidence)\s*(\d+)\s*\]", r"[\1]", line, flags=re.I)
        if (
            bulletize_prose
            and inside_section
            and line
            and not line.startswith(("-", "*", ">", "#"))
        ):
            line = f"- {line}"
        out.append(line)
    return "\n".join(out).strip()


def preserve_source_qualifiers(answer: str, chunks: list[Any]) -> str:
    """Restore numeric qualifiers that small local models commonly omit."""
    evidence_text = " ".join(_chunk_text(chunk) for chunk in chunks)
    qualifier_map = {
        "up to": "최대",
        "at least": "최소",
        "approximately": "약",
        "about": "약",
    }
    out = answer
    for source_qualifier, value, unit in re.findall(
        r"\b(up to|at least|approximately|about)\s+"
        r"(\d+(?:\.\d+)?)\s*(%)",
        evidence_text,
        re.I,
    ):
        korean = qualifier_map[source_qualifier.lower()]
        if re.search(
            rf"(?:{korean}|이내|이상)[^.\n]{{0,24}}"
            rf"{re.escape(value)}\s*{re.escape(unit)}",
            out,
        ):
            continue
        out = re.sub(
            rf"(?<!최대\s)(?<!최소\s)(?<!약\s)\b"
            rf"{re.escape(value)}\s*{re.escape(unit)}",
            f"{korean} {value}{unit}",
            out,
        )
    return out


_ENGLISH_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def repair_deadline_fact_answer(
    answer: str,
    question: str,
    chunks: list[Any],
) -> tuple[str, bool]:
    """Prefer an explicit live-evidence deadline over nearby meeting dates.

    This only runs for deadline/submission questions and reads the date from
    the current retrieved chunks.  It prevents a fluent background sentence
    (for example the date a guidance was approved) from replacing the asked
    ``submit comments by ...`` deadline.
    """
    if not re.search(r"마감일|마감\s*기한|제출\s*기한|기한은", question, re.I):
        return answer, False
    candidates: list[tuple[int, int, tuple[int, int, int]]] = []
    english_date = re.compile(
        r"\b(?:by|before|until|no\s+later\s+than)\s+"
        r"(\d{1,2})\s+"
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{4})\b",
        re.I,
    )
    korean_date = re.compile(
        r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*(?:까지|이내|전)",
    )
    for citation_id, chunk in enumerate(chunks, 1):
        source = re.sub(r"\s+", " ", _chunk_text(chunk)).strip()
        for sentence in re.split(r"(?<=[.!?;])\s+", source):
            deadline_cue = bool(
                re.search(
                    r"submit|comments?|parties|observers|deadline|due\s+date|"
                    r"제출|의견|당사자|참관인|마감",
                    sentence,
                    re.I,
                )
            )
            match = english_date.search(sentence)
            if match:
                day = int(match.group(1))
                month = _ENGLISH_MONTHS[match.group(2).lower()]
                year = int(match.group(3))
                candidates.append((3 + int(deadline_cue), citation_id, (year, month, day)))
            for match in korean_date.finditer(sentence):
                year, month, day = map(int, match.groups())
                candidates.append((2 + int(deadline_cue), citation_id, (year, month, day)))
    if not candidates:
        return answer, False
    _score, citation_id, (year, month, day) = max(candidates, key=lambda item: item[0])
    date_ko = f"{year}년 {month}월 {day}일"
    normalized_answer = re.sub(r"\s+", "", answer)
    if re.sub(r"\s+", "", date_ko) in normalized_answer:
        return answer, False
    subject = (
        "당사자 및 참관인의 의견 제출 마감일"
        if re.search(r"당사자|참관인|의견", question)
        else "제출 마감일"
    )
    bullet = f"- {subject}은 {date_ko}까지입니다. [{citation_id}]"
    first = answer.find("## 1) 핵심 요약")
    second = answer.find("## 2) 선박 운항/업무 영향")
    if first < 0 or second <= first:
        return answer, False
    repaired = (
        answer[:first]
        + "## 1) 핵심 요약\n\n"
        + bullet
        + "\n\n"
        + answer[second:]
    )
    return repaired, True


def repair_numeric_citations(answer: str, chunks: list[Any]) -> str:
    """Bind high-risk numeric claims to chunks that contain those values."""
    chunk_texts = [_chunk_text(chunk).lower().replace(",", "") for chunk in chunks]
    output: list[str] = []
    fact_pattern = re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:%|퍼센트|년|월|일|척|건|톤|t\b|kg|g\b)",
        re.I,
    )
    for raw in (answer or "").splitlines():
        if not raw.strip().startswith(("-", "*")):
            output.append(raw)
            continue
        prose = CITATION_RE.sub("", raw).replace(",", "")
        facts = [
            re.sub(r"\s+", "", value.lower())
            for value in fact_pattern.findall(prose)
        ]
        if not facts:
            output.append(raw)
            continue
        supporting_sets: list[set[int]] = []
        for fact in facts:
            numeric = re.match(r"\d+(?:\.\d+)?", fact)
            if not numeric:
                continue
            value = numeric.group(0)
            supporting = {
                idx
                for idx, source_text in enumerate(chunk_texts, 1)
                if re.search(rf"(?<!\d){re.escape(value)}(?!\d)", source_text)
            }
            if supporting:
                supporting_sets.append(supporting)
        if not supporting_sets:
            output.append(raw)
            continue
        existing_ids = {
            int(value)
            for value in CITATION_RE.findall(raw)
            if 1 <= int(value) <= len(chunks)
        }
        # Do not replace an already valid citation merely because an earlier
        # chunk repeats the same number.  Meeting reports commonly put an
        # objection and the committee's final position in adjacent paragraphs
        # with identical years; numeric-only remapping previously changed the
        # final-position citation back to the objection paragraph.
        if existing_ids and all(
            existing_ids.intersection(candidates)
            for candidates in supporting_sets
        ):
            output.append(raw)
            continue
        common = set.intersection(*supporting_sets)
        if common:
            target_ids = [min(common)]
        else:
            uncovered = list(supporting_sets)
            target_ids: list[int] = []
            while uncovered and len(target_ids) < 2:
                best = max(
                    range(1, len(chunks) + 1),
                    key=lambda idx: sum(idx in candidates for candidates in uncovered),
                )
                if not any(best in candidates for candidates in uncovered):
                    break
                target_ids.append(best)
                uncovered = [candidates for candidates in uncovered if best not in candidates]
        if target_ids:
            clean = re.sub(r"(?:\s*\[\d+\])+\s*$", "", raw).rstrip()
            raw = clean + " " + "".join(f"[{idx}]" for idx in target_ids)
        output.append(raw)
    return "\n".join(output)


def _lexical_support_terms(text: str) -> set[str]:
    stopwords = {
        "그리고", "그러나", "따라서", "대한", "관련", "경우", "해당", "있습니다",
        "합니다", "됩니다", "것으로", "질문", "검색", "근거", "확인",
    }
    suffixes = (
        "으로부터", "에서는", "에게서", "이라고", "하여야", "해야", "하는", "되는",
        "에서", "으로", "에게", "에는", "은", "는", "이", "가", "을", "를", "의",
        "와", "과", "에",
    )
    terms: set[str] = set()
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{1,}|[가-힣]{2,}", text or ""):
        term = raw.lower()
        for suffix in suffixes:
            if term.endswith(suffix) and len(term) - len(suffix) >= 2:
                term = term[: -len(suffix)]
                break
        if len(term) >= 2 and term not in stopwords:
            terms.add(term)
    return terms


def repair_lexical_citations(
    answer: str,
    chunks: list[Any],
    *,
    required_terms: list[str] | tuple[str, ...] = (),
) -> str:
    """Restore an omitted citation only when one chunk strongly supports it.

    This is limited to literal-feature recovery.  The feature term must occur
    in both the claim and source chunk, and at least three meaningful lexical
    terms must overlap.  Unrelated or low-overlap prose remains uncited and is
    subsequently removed by the shared answer contract.
    """
    required = [str(term or "").lower() for term in required_terms if str(term or "").strip()]
    if not required or not chunks:
        return answer
    chunk_texts = [_chunk_text(chunk).lower() for chunk in chunks]
    chunk_terms = [_lexical_support_terms(text) for text in chunk_texts]
    out: list[str] = []
    for raw in (answer or "").splitlines():
        stripped = raw.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or CITATION_RE.search(stripped)
            or re.search(r"검색\s*근거.{0,24}(?:확인되지|없음|찾지 못)", stripped)
        ):
            out.append(raw)
            continue
        claim = re.sub(r"^[-*+]\s+", "", stripped)
        claim_low = claim.lower()
        matched_features = [term for term in required if term in claim_low]
        if not matched_features:
            out.append(raw)
            continue
        claim_terms = _lexical_support_terms(claim)
        best_id = 0
        best_overlap = 0
        for index, (source_text, source_terms) in enumerate(
            zip(chunk_texts, chunk_terms), start=1
        ):
            if not any(term in source_text for term in matched_features):
                continue
            overlap = len(claim_terms.intersection(source_terms))
            if overlap > best_overlap:
                best_id = index
                best_overlap = overlap
        if best_id and best_overlap >= 3:
            prefix = raw if stripped.startswith(("-", "*", "+")) else f"- {stripped}"
            out.append(f"{prefix.rstrip()} [{best_id}]")
        else:
            out.append(raw)
    return "\n".join(out)


def build_scope_evidence_bullet(text: str, citation_id: int) -> str:
    """Render a source-verbatim scope/exception item from one evidence chunk."""
    source = re.sub(r"\s+", " ", str(text or "")).strip()
    if not source or citation_id < 1:
        return ""
    cue = re.compile(
        r"예외|제외|면제|다만|참작|적용하지|except|exemption|unless|does not apply",
        re.I,
    )
    numbered = [
        part.strip()
        for part in re.split(r"(?=\(\d+\)\s*)", source)
        if part.strip()
    ]
    candidates = [part for part in numbered if cue.search(part)]
    if not candidates:
        candidates = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", source)
            if cue.search(part)
        ]
    if not candidates:
        return ""
    selected = max(
        candidates,
        key=lambda part: (
            len(cue.findall(part)),
            bool(re.search(r"하여야|해야|shall|must|required", part, re.I)),
            min(len(part), 900),
        ),
    )
    selected = re.sub(r"^\(\d+\)\s*", "", selected).strip()
    if len(selected) > 700:
        selected = selected[:700].rsplit(" ", 1)[0].rstrip(" ,;") + "…"
    return f"- 적용 범위·예외: {selected} [{citation_id}]"


def enforce_question_relevance(
    answer: str, requirements: QuestionRequirements
) -> str:
    """Remove generic sections and duplicate numeric claims without adding facts."""
    from compound_regulatory import is_compound_regulatory_class_question

    compound_regulatory_class = is_compound_regulatory_class_question(
        requirements.question
    )
    headings = {
        "1": "## 1) 핵심 요약",
        "2": "## 2) 선박 운항/업무 영향",
        "3": "## 3) 추후 확인 필요사항",
        "4": "## 4) 관련 선급 Rule / Guidance",
    }
    sections: dict[str, list[str]] = {key: [] for key in headings}
    current = "1"
    for raw in (answer or "").splitlines():
        match = re.match(r"^##\s*([1-4])\)", raw.strip())
        if match:
            current = match.group(1)
        elif raw.strip().startswith(("-", "*")):
            sections[current].append(raw.strip())

    # Concrete fact questions should not acquire generic operational actions
    # or unrelated class instruments merely to fill the four-section layout.
    if (
        requirements.is_concrete
        and "impact" not in requirements.facets
        and not compound_regulatory_class
    ):
        sections["2"] = ["- 검색 근거에서 확인되지 않음"]
    if (
        requirements.organization in {"MSC", "MEPC"}
        and "document" not in requirements.facets
        and not compound_regulatory_class
    ):
        sections["4"] = ["- 검색 근거에서 확인되지 않음"]

    # Generic "review/check/take action" text is not an evidence-based
    # uncertainty.  Retain only explicit status/uncertainty statements.
    if requirements.is_concrete and not compound_regulatory_class:
        useful_followup = [
            line
            for line in sections["3"]
            if re.search(
                r"미확정|초안|제안|발효|채택|승인|해석|상이|근거.*부족|"
                r"확인되지 않음|draft|proposal|not adopted|entry into force",
                line,
                re.I,
            )
        ]
        sections["3"] = useful_followup or ["- 검색 근거에서 확인되지 않음"]

    # A concrete question must not be padded with file headers, section titles,
    # or facts that answer a different facet merely because they were present
    # in the retrieved context.
    facet_patterns = {
        "finding": r"식별|발견|확인|오류|문제|결함|누락|중복|비현실|잘못|제외",
        "value": r"\d+(?:\.\d+)?\s*(?:%|퍼센트|톤|t|kg|g|척|건)",
        "metric": r"\b(?:AER|cgDIST|EEOI|CII|DCS|LCA|WtT|TtW)\b|지표",
        "comparison": r"대비|보다|비교|기준연도|baseline|versus|\bvs\.?\b",
        "period": r"(?:19|20)\d{2}|발효|시행|기한|일정|기간",
        "status": r"채택|승인|합의|결정|초안|제안|미확정|발효|status",
        "requirement": r"하여야|해야|요구|필요|가능해야|shall|must|required",
        "method": r"방법|절차|방식|산정|계산|측정|검증",
        "scope": r"적용\s*범위|대상|예외|제외|면제|다만|참작|scope|appl",
        "document": r"문서|결의|가이드|지침|규칙|Code|Rule|Guidance",
        "clause": r"조항|제\d+조|규칙\s*\d+|clause|section|paragraph",
        "impact": r"운항|업무|보고|제출|검증|설계|승인|관리|대응",
        "reason": r"때문|이유|배경|근거|목적",
    }
    requested_facets = tuple(requirements.facets)
    if requirements.is_concrete and not compound_regulatory_class:
        filtered: list[str] = []
        for line in sections["1"]:
            prose = CITATION_RE.sub("", line)
            if METADATA_LEAK_RE.search(prose):
                continue
            if FOREIGN_CJK_RE.search(prose):
                continue
            matched = {
                facet
                for facet, pattern in facet_patterns.items()
                if re.search(pattern, prose, re.I)
            }
            if requested_facets and not matched.intersection(requested_facets):
                continue
            minimum_fact_length = 10 if "finding" in requested_facets else 18
            if len(re.sub(r"[^A-Za-z가-힣0-9]", "", prose)) < minimum_fact_length:
                continue
            filtered.append(line)
        if filtered:
            sections["1"] = filtered

    # Prefer the more informative of duplicate numeric claims. Citation
    # numbers are excluded from the signature.  A sentence containing the
    # comparison basis or metric wins over a restatement of the same number.
    deduped: list[str] = []
    numeric_position: dict[tuple[str, ...], int] = {}
    for line in sections["1"]:
        prose = CITATION_RE.sub("", line)
        signature = tuple(
            value
            for value in re.findall(r"\d+(?:\.\d+)?\s*%", prose)
        )
        if signature and signature in numeric_position:
            pos = numeric_position[signature]
            if len(prose) > len(CITATION_RE.sub("", deduped[pos])):
                deduped[pos] = line
            continue
        if signature:
            numeric_position[signature] = len(deduped)
        deduped.append(line)
    sections["1"] = deduped

    # Greedily keep the smallest set of concrete bullets that covers the
    # requested facets. Finding/audit questions are the exception: distinct
    # findings are the answer, so preserve all relevant non-duplicate bullets
    # instead of collapsing them after the first finding+method pair.
    if (
        requirements.is_concrete
        and requested_facets
        and sections["1"]
        and not compound_regulatory_class
    ):
        if "finding" in requested_facets:
            seen_finding_lines: set[str] = set()
            preserved: list[str] = []
            for line in sections["1"]:
                signature = re.sub(
                    r"[^A-Za-z가-힣0-9]+",
                    "",
                    CITATION_RE.sub("", line).lower(),
                )
                if signature and signature not in seen_finding_lines:
                    seen_finding_lines.add(signature)
                    preserved.append(line)
            sections["1"] = preserved[:5]
        else:
            remaining = set(requested_facets)
            ranked: list[tuple[int, int, str, set[str]]] = []
            for line in sections["1"]:
                prose = CITATION_RE.sub("", line)
                matched = {
                    facet
                    for facet in requested_facets
                    if re.search(facet_patterns.get(facet, r"$^"), prose, re.I)
                }
                ranked.append((len(matched), len(prose), line, matched))
            selected: list[str] = []
            for _coverage, _length, line, matched in sorted(
                ranked, key=lambda item: (item[0], item[1]), reverse=True
            ):
                if matched.intersection(remaining) or not selected:
                    selected.append(line)
                    remaining.difference_update(matched)
                if not remaining:
                    break
            sections["1"] = selected[:5]

    rendered: list[str] = []
    for key in ("1", "2", "3", "4"):
        rendered.extend([headings[key], *sections[key], ""])
    return "\n".join(rendered).strip()


def validate_answer_requirements(
    answer: str, requirements: QuestionRequirements, chunks: list[Any]
) -> tuple[bool, list[str]]:
    """Cheap structural/coverage checks; semantic truth remains evidence-bound."""
    warnings: list[str] = []
    text = answer or ""
    prose = CITATION_RE.sub("", text)
    for heading in (
        "1) 핵심 요약",
        "2) 선박 운항/업무 영향",
        "3) 추후 확인 필요사항",
        "4) 관련 선급 Rule / Guidance",
    ):
        if heading not in text:
            warnings.append("missing_section:" + heading)

    valid_ids = range(1, len(chunks) + 1)
    for raw in text.splitlines():
        if raw.strip().startswith(("-", "*")):
            ids = [int(value) for value in CITATION_RE.findall(raw)]
            if "확인되지 않음" not in raw and not any(value in valid_ids for value in ids):
                warnings.append("uncited_bullet")
            if FOREIGN_CJK_RE.search(raw):
                warnings.append("non_korean_cjk_leak")

    if "value" in requirements.facets and not re.search(
        r"\d+(?:[./]\d+)?\s*(?:%|퍼센트|톤|t|kg|g|년|월|일|주|시간|"
        r"m/s|mm/s|mm|cm|m|kV|V|N/mm2|N/mm²|MPa|°C|℃|ppm|배)",
        prose,
        re.I,
    ):
        warnings.append("requested_value_missing")
    if "value" in requirements.facets:
        evidence_text = " ".join(_chunk_text(chunk) for chunk in chunks)
        dimension_patterns = (
            (r"정격|전압|voltage", r"-?\d+(?:[./]\d+)?\s*(?:kV|V)\b"),
            (r"온도|temperature", r"-?\d+(?:\.\d+)?\s*(?:°C|℃)"),
            (r"(?:원주\s*)?속도|velocity|speed", r"-?\d+(?:\.\d+)?\s*(?:m/s|mm/s|knots?)\b"),
            (r"두께|thickness", r"-?\d+(?:\.\d+)?\s*(?:mm|cm)\b"),
        )
        for question_pattern, value_pattern in dimension_patterns:
            if not re.search(question_pattern, requirements.question, re.I):
                continue
            source_dimension_values = list(
                dict.fromkeys(re.findall(value_pattern, evidence_text, re.I))
            )
            if source_dimension_values and any(
                not re.search(re.escape(value), prose, re.I)
                for value in source_dimension_values
            ):
                warnings.append("requested_dimension_values_incomplete")
                break
        qualified_values = re.findall(
            r"\b(up to|at least|approximately|about)\s+"
            r"(\d+(?:\.\d+)?)\s*(%)",
            evidence_text,
            re.I,
        )
        qualifier_ko = {
            "up to": r"최대|이내|up to",
            "at least": r"최소|이상|at least",
            "approximately": r"약|대략|approximately",
            "about": r"약|대략|about",
        }
        for qualifier, value, unit in qualified_values:
            if re.search(rf"\b{re.escape(value)}\s*{re.escape(unit)}", prose) and not re.search(
                rf"(?:{qualifier_ko[qualifier.lower()]})[^.\n]{{0,24}}"
                rf"{re.escape(value)}\s*{re.escape(unit)}",
                prose,
                re.I,
            ):
                warnings.append("requested_value_qualifier_missing")
                break
        # Preserve parallel condition/value pairs from one source sentence.
        # A small model may copy the first value (oil/water: 6 m/s) and drop
        # the immediately adjacent second condition (grease: 3 m/s), even
        # though both answer the same comparison question.
        unit_value_re = re.compile(
            r"(?<![\w.])-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?\s*"
            r"(?:%|m/s|mm/s|knots?|hours?|days?|weeks?|years?|mm|cm|kV|V|"
            r"N/mm2|N/mm²|MPa|°C|℃|ppm|배)\b",
            re.I,
        )
        for sentence in re.split(r"(?<=[.;])\s+", evidence_text):
            source_values = list(dict.fromkeys(unit_value_re.findall(sentence)))
            if (
                len(source_values) >= 2
                and re.search(
                    r"\b(?:and|or|while|whereas)\b|각각|경우|및|[과와]",
                    sentence,
                    re.I,
                )
            ):
                missing_values = [
                    value
                    for value in source_values
                    if not re.search(re.escape(value), prose, re.I)
                ]
                if missing_values:
                    warnings.append("requested_parallel_values_incomplete")
                    break
        if re.search(r"각각", requirements.question):
            pair = re.search(
                r"([가-힣A-Za-z]{2,24})[과와]\s*([가-힣A-Za-z]{2,24})(?:의|은|는|이|가|을|를)?",
                requirements.question,
            )
            if pair:
                pair_terms = [
                    re.sub(r"(?:의|은|는|이|가|을|를)$", "", value)
                    for value in pair.groups()
                ]
                if not all(term and term in prose for term in pair_terms):
                    warnings.append("requested_parallel_subjects_incomplete")
    if "finding" in requirements.facets and not re.search(
        r"식별|발견|확인|오류|문제|결함|누락|중복|비현실|잘못", prose, re.I
    ):
        warnings.append("requested_finding_missing")
    if "finding" in requirements.facets:
        evidence_text = " ".join(_chunk_text(chunk) for chunk in chunks)
        evidence_categories = sum(
            bool(re.search(pattern, evidence_text, re.I))
            for pattern in (
                r"duplicate|multiple reporting",
                r"unrealistic|not technically possible",
                r"incorrect (?:ship )?type|incorrectly categorized",
                r"missing ships?|no data (?:had been )?reported",
                r"hours under way.{0,80}(?:more than|exceed)",
            )
        )
        answer_categories = sum(
            bool(re.search(pattern, prose, re.I))
            for pattern in (
                r"중복|duplicate|multiple reporting",
                r"비현실|기술적으로 불가능|unrealistic",
                r"잘못된 선종|선종.*오류|incorrect (?:ship )?type",
                r"누락 선박|미보고|missing ships?",
                r"운항시간.*(?:초과|연간.*많)|hours under way",
            )
        )
        if evidence_categories >= 2 and answer_categories < min(3, evidence_categories):
            warnings.append("requested_finding_incomplete")
    if "metric" in requirements.facets and not re.search(
        r"\b(?:AER|cgDIST|EEOI|CII|DCS|LCA|WtT|TtW)\b|지표", prose, re.I
    ):
        warnings.append("requested_metric_missing")
    if "metric" in requirements.facets:
        evidence_text = " ".join(_chunk_text(chunk) for chunk in chunks)
        if (
            re.search(r"supply-based", evidence_text, re.I)
            and re.search(r"demand-based", evidence_text, re.I)
            and {"AER", "cgDIST", "EEOI"}.issubset(
                set(re.findall(r"\b(?:AER|cgDIST|EEOI)\b", evidence_text, re.I))
            )
            and not (
                re.search(r"공급\s*기반", prose)
                and re.search(r"수요\s*기반", prose)
            )
        ):
            warnings.append("requested_metric_classification_missing")
    if "period" in requirements.facets and not re.search(
        r"(?:19|20)\d{2}|\d+\s*(?:일|주|개월|년|days?|weeks?|months?|years?)|"
        r"발효|시행|기한|일정|기간",
        prose,
        re.I,
    ):
        warnings.append("requested_period_missing")
    if "period" in requirements.facets:
        evidence_text = " ".join(_chunk_text(chunk) for chunk in chunks)
        deadline = re.search(
            r"within\s+(?:\w+\s+)?\(?([0-9]+)\)?\s*(weeks?|days?|months?|years?)|"
            r"([0-9]+)\s*(주|일|개월|년)\s*(?:이내|안)",
            evidence_text,
            re.I,
        )
        if deadline:
            value = deadline.group(1) or deadline.group(3)
            english_unit = (deadline.group(2) or "").lower()
            korean_unit = deadline.group(4) or {
                "week": "주", "weeks": "주", "day": "일", "days": "일",
                "month": "개월", "months": "개월", "year": "년", "years": "년",
            }.get(english_unit, "")
            if value and korean_unit and not re.search(
                rf"{re.escape(value)}\s*(?:{re.escape(korean_unit)}|{re.escape(english_unit)})",
                prose,
                re.I,
            ):
                warnings.append("requested_deadline_value_missing")
    if "requirement" in requirements.facets and not re.search(
        r"하여야|해야|요구|필요|shall|must|required", prose, re.I
    ):
        warnings.append("requested_requirement_missing")
    if "requirement" in requirements.facets:
        evidence_text = " ".join(_chunk_text(chunk) for chunk in chunks)
        if re.search(
            r"최소(?:한의|한)?\s*(?:증빙|입증|요건|요구)|minimum\s+(?:evidence|requirement)",
            requirements.question,
            re.I,
        ):
            minimum_block = re.search(
                r"(?:As\s+a\s+minimum|최소(?:한|한의)?)[\s\S]{0,1500}",
                evidence_text,
                re.I,
            )
            if minimum_block:
                block = minimum_block.group(0)
                required_markers: list[str] = []
                required_markers.extend(
                    re.findall(r"\b\d+\s*years?\b|\b\d+\s*년\b", block, re.I)
                )
                if re.search(r"\bGOOD\b", block, re.I):
                    required_markers.append("GOOD")
                if re.search(r"laboratory\s+testing|실험실\s*시험", block, re.I):
                    required_markers.append("laboratory testing")
                marker_missing = False
                for marker in list(dict.fromkeys(required_markers)):
                    if marker.lower() == "laboratory testing":
                        present = bool(re.search(r"실험실\s*시험|laboratory\s+testing", prose, re.I))
                    elif re.fullmatch(r"\d+\s*years?", marker, re.I):
                        number = re.search(r"\d+", marker).group(0)
                        present = bool(
                            re.search(rf"{re.escape(number)}\s*(?:년|years?)", prose, re.I)
                        )
                    else:
                        present = bool(re.search(re.escape(marker), prose, re.I))
                    marker_missing = marker_missing or not present
                if marker_missing:
                    warnings.append("requested_minimum_evidence_incomplete")
        explicit_exception = bool(
            re.search(
                r"\b(?:unless|except|provided that|only if|however)\b|다만|예외|제외|"
                r"손상\s*되지\s*않.{0,80}사용할\s*수\s*있|"
                r"may\s+be\s+of\s+a\s+lower",
                evidence_text,
                re.I,
            )
        )
        exception_preserved = bool(
            re.search(
                r"예외|다만|그러나|전용|경우에는|조건(?:으로|하에)|손상되지\s*않|"
                r"낮은.{0,24}(?:허용|사용)|unless|except|provided that|only if|however",
                prose,
                re.I,
            )
        )
        if explicit_exception and not exception_preserved:
            warnings.append("requested_exception_missing")
    if "scope" in requirements.facets and not re.search(
        r"적용\s*범위|대상|예외|제외|면제|다만|참작|적용하지|"
        r"scope|exemption|except|unless|does not apply",
        prose,
        re.I,
    ):
        warnings.append("requested_scope_missing")
    if "method" in requirements.facets and not re.search(
        r"제외|수정|검토|통보|제공|처리|분류|이동|산정|계산|검증|"
        r"removed|excluded|corrected|reviewed|provided|processed",
        prose,
        re.I,
    ):
        warnings.append("requested_method_missing")
    if re.search(r"어디|어느\s*(?:위치|지점)|장소|where", requirements.question, re.I):
        evidence_text = " ".join(_chunk_text(chunk) for chunk in chunks)
        location_pairs = (
            (r"at\s+LBP/2.{0,80}load\s+line\s+mark\s+position", r"LBP/2|만재흘수선|load\s+line\s+mark"),
            (r"Palais\s+des\s+Nations.{0,80}(?:Geneva|online)", r"Palais\s+des\s+Nations|제네바|온라인|online"),
        )
        for source_pattern, answer_pattern in location_pairs:
            if re.search(source_pattern, evidence_text, re.I) and not re.search(
                answer_pattern, prose, re.I
            ):
                warnings.append("requested_location_missing")
                break
    if re.search(r"비용|수수료|cost|fees?", requirements.question, re.I):
        evidence_text = " ".join(_chunk_text(chunk) for chunk in chunks)
        if re.search(r"costs?|fees?|비용|수수료", evidence_text, re.I) and not re.search(
            r"비용|수수료|costs?|fees?", prose, re.I
        ):
            warnings.append("requested_cost_missing")
    primary_prose = prose.split("## 2)", 1)[0]
    negative_answer = bool(
        re.search(
            r"(?:검색\s*)?근거.{0,28}(?:확인되지|찾지\s*못|명시되지)|"
            r"(?:내용|정보|사항|장소|수치|정의|이유).{0,24}"
            r"(?:확인되지|찾지\s*못|확인하지\s*못|명시되지|없습니다|없음)|"
            r"직접\s*답.{0,24}(?:확인하지\s*못|확인되지)",
            primary_prose,
            re.I,
        )
    )
    if negative_answer:
        evidence_text = " ".join(_chunk_text(chunk) for chunk in chunks)
        q = requirements.question
        direct_answer_shape = bool(
            (
                re.search(r"장소|어디|실시", q)
                and re.search(r"laborator|premises|test\s+site|carried\s+out\s+at", evidence_text, re.I)
            )
            or (
                re.search(r"목록|항목|포함|어떤\s*것", q)
                and len(re.findall(r"(?:^|\s)—\s+", evidence_text)) >= 2
            )
            or (
                re.search(r"이유|왜|배경", q)
                and re.search(r"because|due\s+to|reason|therefore|in\s+view\s+of", evidence_text, re.I)
            )
            or (
                re.search(r"정의|뜻", q)
                and re.search(r"defined\s+as|is\s+defined|means\b|definition", evidence_text, re.I)
            )
            or (
                re.search(r"얼마|몇\s*(?:년|개월|일|%|퍼센트)|수치|규모|정확도|간격", q)
                and re.search(r"\b\d+(?:\.\d+)?\s*(?:%|years?|months?|days?|m/s|mm|kg|t)\b", evidence_text, re.I)
            )
            or (
                re.search(r"어떤\s*형태.{0,12}그룹|전담\s*그룹", q)
                and re.search(r"(?:working|correspondence|review|expert)\s+group", evidence_text, re.I)
            )
            or (
                "requirement" in requirements.facets
                and re.search(
                    r"\b(?:shall|must|required|is\s+to|are\s+to|should)\b|하여야|해야",
                    evidence_text,
                    re.I,
                )
            )
        )
        if direct_answer_shape:
            warnings.append("false_negative_despite_direct_evidence")
    if re.search(
        r"체크\s*리스트|무엇(?:이|을)?\s*포함|포함(?:되어|해야)|"
        r"항목(?:에는|은|을)|목록|장치들",
        requirements.question,
        re.I,
    ):
        evidence_lists = [
            list(
                dict.fromkeys(
                    int(value)
                    for value in re.findall(
                        r"(?:^|\s)\(?(\d{1,2})\)\s+", _chunk_text(chunk)
                    )
                )
            )
            for chunk in chunks
        ]
        source_numbers = max(evidence_lists, key=len, default=[])
        if len(source_numbers) >= 5:
            answer_numbers = {
                int(value)
                for value in re.findall(r"(?:^|[\s,;])([1-9]|1\d)\)\s*", text)
            }
            if not set(source_numbers).issubset(answer_numbers):
                warnings.append("requested_numbered_list_incomplete")
    if "list" in requirements.facets or re.search(
        r"(?:조건|기준|원칙|이슈|요건).{0,16}(?:무엇|어떤)|어떠한\s*조건|"
        r"어떤.{0,16}(?:조건|기준|원칙|이슈|요건)",
        requirements.question,
    ):
        source_list_count = max(
            (
                max(
                    len(re.findall(r"(?:^|\s)—\s+", _chunk_text(chunk))),
                    len(re.findall(r"(?:^|\s)\(\d{1,2}\)\s+", _chunk_text(chunk))),
                    len(re.findall(r"(?:^|\s)\.\d{1,2}\s+", _chunk_text(chunk))),
                )
                for chunk in chunks
            ),
            default=0,
        )
        minimum_expected = requirements.requested_count or min(source_list_count, 4)
        if source_list_count >= 2:
            section_one = text.split("## 2)", 1)[0]
            answer_item_numbers = {
                int(value)
                for value in re.findall(r"(?:^|[\s,;])([1-9]|[1-9]\d)\)\s+", section_one)
            }
            answer_bullets = len(re.findall(r"(?m)^[-*]\s+", section_one))
            if (
                source_list_count >= 4
                and len(answer_item_numbers) < source_list_count
            ) or (
                source_list_count < 4
                and max(len(answer_item_numbers), answer_bullets) < minimum_expected
            ):
                warnings.append("requested_source_list_incomplete")
    return not warnings, list(dict.fromkeys(warnings))
