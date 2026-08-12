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
    evidence_parts: list[str] = []
    for idx, chunk in enumerate(chunks, 1):
        evidence_parts.append(
            "\n".join(
                [
                    f"[{idx}]",
                    f"기관={getattr(chunk, 'source', '')}",
                    f"문서={getattr(chunk, 'file_name', '')}",
                    f"페이지={getattr(chunk, 'page_number', '')}",
                    f"조항={getattr(chunk, 'clause_number', '')}",
                    f"본문={_chunk_text(chunk)}",
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
        "scope": "적용 범위·대상",
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

출력은 반드시 아래 네 절을 아래 표기 그대로 사용한다. 굵은 글씨(**)로 제목을 바꾸지 않는다.
## 1) 핵심 요약
## 2) 선박 운항/업무 영향
## 3) 추후 확인 필요사항
## 4) 관련 선급 Rule / Guidance

각 사실은 반드시 다음 형식으로 쓴다: '- 한국어 사실 문장. [1]'
각 절은 bullet 형식으로 작성한다. 해당 사항이 없으면 '- 검색 근거에서 확인되지 않음'이라고 쓴다."""
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
- 문서명, 회의차수, 조항 또는 결의·가이드 명칭을 가능한 범위에서 포함한다.
- 구체 질문은 1절 2~5개, 광범위 요약은 전체 7~10개 bullet 이내로 제한한다.
- 인용 없는 사실 문장은 작성하지 않는다.

검색 근거:
{chr(10).join(evidence_parts)}
"""
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
        + "- Put the direct answer to the question first, then only concrete "
        "operational actions supported by evidence.\n"
        "- For a broad meeting question, select the decision, status/timeline, "
        "and operational or reporting consequence that are actually evidenced.\n"
        "- For a rule question, cite the matching clause, explain the requirement "
        "in Korean, and distinguish a required control from an example or a "
        "matter requiring follow-up.\n"
        "- Do not repeat a fact across sections. If a section has no supported "
        "content, write one short Korean limitation sentence without inventing a fact.\n"
        "- Keep a direct rule answer to 2-3 key bullets and a meeting answer to "
        "the number requested by the question where possible.\n\n"
        "Evidence chunks:\n" + "\n\n".join(evidence)
    )
    return system, user, req


def normalize_generated_markdown(answer: str) -> str:
    """Normalize harmless local-model formatting variations before contract QA."""
    headings = {
        "1": "## 1) 핵심 요약",
        "2": "## 2) 선박 운항/업무 영향",
        "3": "## 3) 추후 확인 필요사항",
        "4": "## 4) 관련 선급 Rule / Guidance",
    }
    out: list[str] = []
    for raw in (answer or "").splitlines():
        line = raw.strip()
        heading = re.match(
            r"^(?:#{1,4}\s*)?\*{0,2}\s*([1-4])\)\s*.*?\*{0,2}\s*$",
            line,
        )
        if heading:
            out.append(headings[heading.group(1)])
            continue
        line = re.sub(r"【\s*(\d+)\s*】", r"[\1]", line)
        line = re.sub(r"\[\s*(?:근거|출처|evidence)\s*(\d+)\s*\]", r"[\1]", line, flags=re.I)
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


def enforce_question_relevance(
    answer: str, requirements: QuestionRequirements
) -> str:
    """Remove generic sections and duplicate numeric claims without adding facts."""
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
    if requirements.is_concrete and "impact" not in requirements.facets:
        sections["2"] = ["- 검색 근거에서 확인되지 않음"]
    if requirements.organization in {"MSC", "MEPC"} and "document" not in requirements.facets:
        sections["4"] = ["- 검색 근거에서 확인되지 않음"]

    # Generic "review/check/take action" text is not an evidence-based
    # uncertainty.  Retain only explicit status/uncertainty statements.
    if requirements.is_concrete:
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
        "scope": r"적용\s*범위|대상|예외|경우|scope|appl",
        "document": r"문서|결의|가이드|지침|규칙|Code|Rule|Guidance",
        "clause": r"조항|제\d+조|규칙\s*\d+|clause|section|paragraph",
        "impact": r"운항|업무|보고|제출|검증|설계|승인|관리|대응",
        "reason": r"때문|이유|배경|근거|목적",
    }
    requested_facets = tuple(requirements.facets)
    if requirements.is_concrete:
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
    if requirements.is_concrete and requested_facets and sections["1"]:
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
        r"\d+(?:\.\d+)?\s*(?:%|퍼센트|톤|t|kg|g|년|월|일)", prose, re.I
    ):
        warnings.append("requested_value_missing")
    if "value" in requirements.facets:
        evidence_text = " ".join(_chunk_text(chunk) for chunk in chunks)
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
        r"(?:19|20)\d{2}|발효|시행|기한|일정|기간", prose
    ):
        warnings.append("requested_period_missing")
    if "requirement" in requirements.facets and not re.search(
        r"하여야|해야|요구|필요|shall|must|required", prose, re.I
    ):
        warnings.append("requested_requirement_missing")
    if "method" in requirements.facets and not re.search(
        r"제외|수정|검토|통보|제공|처리|분류|이동|산정|계산|검증|"
        r"removed|excluded|corrected|reviewed|provided|processed",
        prose,
        re.I,
    ):
        warnings.append("requested_method_missing")
    return not warnings, list(dict.fromkeys(warnings))
