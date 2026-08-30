"""Shared policy for questions that combine IMO meetings and class rules.

These questions have two independent evidence lanes.  Treating them as a
meeting summary silently drops the class-rule lane; treating them as a rule
lookup drops the committee decision and regulatory-status lane.
"""
from __future__ import annotations

import re
from typing import Any


CLASS_RULE_SOURCES = ("DNV", "KR", "ABS", "LR")
MEETING_SOURCES = ("MSC", "MEPC", "IMO")

_MEETING_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:MSC|MEPC|IMO)(?:\s*[-/]?\s*\d{1,3})?(?![A-Za-z0-9])|"
    r"국제해사기구|위원회\s*회의|회의\s*(?:논의|결과|결정)",
    re.I,
)
_CLASS_RE = re.compile(
    r"선급|보유\s*(?:선급\s*)?(?:규정|Rule)|분류협회|classification\s+societ|"
    r"class\s*(?:rule|guidance|requirement)|"
    r"(?<![A-Za-z0-9])(?:DNV|ABS|LR|KR)(?![A-Za-z0-9])",
    re.I,
)
_CHECKLIST_RE = re.compile(
    r"체크\s*리스트|검토\s*목록|설계\s*검토|개념\s*승인|원칙\s*승인|"
    r"개념\s*설계.{0,12}확인|approval\s+in\s+principle|\bAIP\b|준비\s*사항",
    re.I,
)
_UNCERTAINTY_RE = re.compile(
    r"미확정|불확실|추후\s*확인|향후\s*확인|규제\s*공백|pending|uncertain|"
    r"future\s+revision|해석\s*논란|상이한\s*요구",
    re.I,
)
_PREMISE_VERIFICATION_RE = re.compile(
    r"(?:전제가\s*맞는지|전제(?:를|가)?\s*검증|사실인지\s*검증|"
    r"틀리면\s*(?:문서\s*)?근거로\s*바로잡)",
    re.I,
)

DESIGN_ANCHOR_GROUPS = (
    (r"도면|제출\s*자료|승인\s*자료|검토\(Examined|승인\(Approved", r"drawing|documentation|examined|approved|도면|제출\s*자료"),
    (r"Fuel ready|Gas fuelled|선급\s*인증|부기부호|notation", r"fuel ready|gas fuelled|class notation|부기부호"),
    (r"탱크|저장", r"fuel tank|containment|storage|탱크|저장"),
    (r"벙커링|배관|연료공급", r"bunkering|pipe routing|piping|fuel supply|벙커링|배관|연료공급"),
    (r"환기|팬|환기구", r"ventilation|fan|outlet|환기|팬|환기구"),
    (r"검지|감지", r"gas detection|detector|detect|검지|감지"),
    (r"비상정지|원격정지|ESD", r"emergency shutdown|remote stop|\bESD\b|비상정지|원격정지"),
    (r"위험성\s*평가|가스\s*확산|안전\s*거리|\bHAZID\b|\bQRA\b|화재.?폭발", r"risk assessment|gas dispersion|safety distance|\bHAZID\b|\bQRA\b|fire and explosion|위험성\s*평가|가스\s*확산|안전\s*거리"),
    (r"수분무|워터\s*스프레이|수분\s*분사", r"water spray|water screen|수분무|워터\s*스프레이"),
    # Plain "화재" is also used in "화재·폭발 분석".  Treat only an
    # explicit protection/extinguishing claim as a separate fire-system
    # requirement; otherwise the risk-analysis anchor already covers it.
    (r"화재\s*방호|방화|소화", r"fire protection|fire extinguish|화재\s*방호|방화|소화"),
    (r"제어|감시|안전\s*시스템", r"control|monitoring|safety system|제어|감시|안전\s*시스템"),
    (r"운용\s*개념|운항\s*개념|CONOPS", r"operational concept|concept of operations|\bCONOPS\b|운용\s*개념"),
    (r"원격\s*운영|원격\s*운항|원격\s*선박|자율.?원격|\bROC\b|연결\s*링크", r"remote operations centre|remotely operated|autoremote vessel functions|\bROC\b|connectivity link|원격"),
    (r"결함\s*허용|고장.*복구|FDIR", r"fault tolerance|fault detection|isolation and recovery|\bFDIR\b|결함\s*허용"),
    (r"검증.?확인\s*\(?(?:V&V)?\)?|\bV&V\b|시뮬레이션", r"verification|validation|\bV&V\b|simulation|검증"),
    (r"상황\s*인식|개입\s*시점", r"situational awareness|human intervention|상황\s*인식"),
    (r"개념\s*승인|원칙\s*승인|\bAIP\b|Concept\s+Qualification|\bCQ\b", r"concept qualification|approval in principle|approval process|\bAIP\b|\bCQ\b|개념\s*승인|원칙\s*승인"),
    (r"고온\s*표면|냉각|불활성", r"high temperature surface|cooling|inerting|inert gas|고온|불활성"),
    (r"압력\s*방출|압력\s*해소", r"pressure relief|압력\s*방출"),
    (r"복합재|FRP|Type\s*4", r"fibre reinforced|\bFRP\b|type 4 pressure vessel|composite tank"),
)


def is_compound_regulatory_class_question(question: str) -> bool:
    """Return True only when both meeting and class evidence are requested."""
    q = str(question or "")
    # A false-premise check such as ``DNV-CG-0264는 IMO 협약이다`` mentions
    # both a class society and IMO, but asks for the nature of one named class
    # document.  The compound lane would bypass the dedicated verifier.
    if _PREMISE_VERIFICATION_RE.search(q):
        return False
    integration = bool(
        _CHECKLIST_RE.search(q)
        or re.search(
            r"함께|동시에|각각|근거로|비교|차이|대조|고려|연계|"
            r"회의.{0,32}(?:선급|class)|논의.{0,32}(?:규정|rule|guidance)",
            q,
            re.I,
        )
    )
    return bool(_MEETING_RE.search(q) and _CLASS_RE.search(q) and integration)


def requests_design_checklist(question: str) -> bool:
    return bool(_CHECKLIST_RE.search(str(question or "")))


def requests_concept_approval(question: str) -> bool:
    return bool(
        re.search(
            r"개념\s*승인|원칙\s*승인|approval\s+in\s+principle|\bAIP\b",
            str(question or ""),
            re.I,
        )
    )


def requests_uncertainty_analysis(question: str) -> bool:
    return bool(_UNCERTAINTY_RE.search(str(question or "")))


def requested_class_sources(question: str) -> list[str]:
    """Return explicitly named societies, or every indexed class corpus."""
    q = str(question or "")
    named = [
        source
        for source in CLASS_RULE_SOURCES
        if re.search(
            rf"(?<![A-Za-z0-9]){re.escape(source)}(?![A-Za-z0-9])",
            q,
            re.I,
        )
    ]
    return named or list(CLASS_RULE_SOURCES)


def explicitly_requested_class_sources(question: str) -> list[str]:
    """Return only societies literally named by the user."""
    q = str(question or "")
    return [
        source
        for source in CLASS_RULE_SOURCES
        if re.search(
            rf"(?<![A-Za-z0-9]){re.escape(source)}(?![A-Za-z0-9])",
            q,
            re.I,
        )
    ]


def compound_topic_terms(question: str) -> tuple[str, ...]:
    """Small bilingual domain vocabulary used by retrieval, never answer text."""
    q = str(question or "").lower()
    groups = (
        (("암모니아", "ammonia"), ("ammonia", "암모니아")),
        (("수소", "hydrogen"), ("hydrogen", "수소")),
        (("메탄올", "methanol"), ("methanol", "메탄올")),
        (("mass", "자율운항", "autonomous"), ("MASS", "autonomous", "자율운항")),
        (("대체연료", "alternative fuel"), ("alternative fuel", "대체연료", "low-flashpoint")),
        (("ghg", "온실가스"), ("GHG", "greenhouse gas", "온실가스")),
        (
            ("gfi", "전과정", "전 과정", "lca", "life cycle"),
            (
                "GFI",
                "LCA",
                "life cycle",
                "Fuel Life Cycle Label",
                "SFCS",
                "전과정",
            ),
        ),
    )
    output: list[str] = []
    for triggers, terms in groups:
        if any(trigger.lower() in q for trigger in triggers):
            output.extend(terms)
    return tuple(dict.fromkeys(output)) or ("ship", "vessel", "선박")


def compound_exact_phrases(question: str) -> tuple[str, ...]:
    """Return selective corpus phrases for bounded exact recovery.

    A bare fuel noun can occur hundreds of times in a large rulebook and a
    fixed Chroma ``get`` limit then keeps whichever rows were inserted first.
    Instrument/notation phrases are far less frequent and recover the actual
    governing section without a full-corpus lexical scan.
    """
    q = str(question or "").lower()
    phrases: list[str] = []
    if "수소" in q or "hydrogen" in q:
        phrases.extend(
            (
                "Gas fuelled hydrogen",
                "Fuel ready(Hydrogen",
                "hydrogen as fuel",
                "hydrogen fuel approval",
            )
        )
    if "암모니아" in q or "ammonia" in q:
        phrases.extend(
            (
                "Gas fuelled ammonia",
                "Fuel ready(Ammonia",
                "ammonia cargo as fuel",
                "Ammonia Ready",
            )
        )
    if re.search(r"\bMASS\b|자율운항|autonomous", question or "", re.I):
        phrases.extend(
            (
                "MASS Code",
                "autonomous and remotely operated",
                "Concept Qualification",
                "AROS",
            )
        )
    if re.search(r"대체\s*연료|alternative\s+fuel|\bGFI\b", question or "", re.I):
        phrases.extend(("Fuel ready", "alternative fuel", "low-flashpoint"))
    return tuple(dict.fromkeys(phrases))


def build_class_search_query(question: str) -> str:
    """Rewrite only the class lane so meeting words do not dominate embedding."""
    q = re.sub(
        r"(?<![A-Za-z0-9])(?:MSC|MEPC|IMO)\s*[-/]?\s*\d{0,3}(?![A-Za-z0-9])",
        " ",
        str(question or ""),
        flags=re.I,
    )
    topic = " ".join(
        (*compound_topic_terms(question), *compound_exact_phrases(question))
    )
    return re.sub(
        r"\s+",
        " ",
        f"{q} {topic} class rules guidance approval in principle AIP class notation "
        "concept design scope application fuel ready design arrangement fuel tank "
        "bunkering piping hazardous toxic area ventilation detection emergency shutdown "
        "risk assessment fire protection",
    ).strip()


def compound_prompt_instruction(question: str) -> str:
    checklist = requests_design_checklist(question)
    uncertainty = requests_uncertainty_analysis(question)
    autonomous = bool(re.search(r"\bMASS\b|자율운항|autonomous", question, re.I))
    comparison = bool(re.search(r"차이|비교|대조|difference|compare", question, re.I))
    checklist_examples = (
        "CONOPS/운용범위, 자율·원격 기능과 ROC/통신, 고장·fallback/최소위험상태, "
        "위험성평가와 V&V·시험"
        if autonomous
        else "탱크·배치, 연료공급/배관, 위험·독성구역, 환기/가스검지/ESD, "
        "화재·위험성평가·승인자료"
    )
    return f"""복합 규제 질문 추가 계약:
- 이 질문은 'IMO 회의 결정'과 '선급 Rule/Guidance'라는 서로 독립된 두 근거 축을 요구한다. 두 축을 모두 인용하지 못하면 완성 답변으로 간주하지 않는다.
- 1절에서는 위원회가 실제 승인·채택·합의한 사항, 아직 초안·추가작업인 사항, 적용대상을 분리한다. IGC Code와 IGF Code, 승인일·채택예정일·발효일을 서로 바꾸지 않는다.
- {'질문이 비교한 두 대상의 적용범위 차이를 1절에서 각각 명시한다. 한쪽만 설명하면 안 된다.' if comparison else '질문이 특정 적용대상을 나누지 않았다면 근거에 없는 비교 대상을 만들지 않는다.'}
- 같은 안건에 대해 앞선 근거가 '승인을 위해 회부'라고 하고 뒤의 근거가 '위원회가 승인'이라고 하면, 뒤의 최종 결정을 우선한다. 이미 승인된 사항을 '승인 추진 중'이라고 쓰지 않는다.
- 2절은 {'최소 4개의 구체적인 설계 검토 체크리스트로 작성한다' if checklist else '질문이 요구한 실무 검토 항목을 구체적으로 작성한다'}. {checklist_examples} 중 근거가 있는 항목을 우선한다. 각 bullet은 선급 근거를 인용한다.
- 2절의 각 bullet은 '프로젝트에서 무엇을 제출·확인·검증·시험해야 하는지'를 행동형 문장으로 쓴다. 규제 필요성이나 일반 배경만 설명한 문장은 체크리스트로 세지 않는다.
- notation 변경이나 문서의 EX/Approved 상태만 설명하는 항목은 4절에 두고, 2절의 설계 체크리스트 개수에는 포함하지 않는다.
- 3절은 확정 규정과 미확정·추가개정·적용범위 공백을 분리한다. {'미확정 항목을 반드시 하나 이상 근거와 함께 쓴다.' if uncertainty else '근거에 있는 미확정 상태만 쓴다.'}
- 3절에는 회의 근거에 명시된 미확정 일정·법적 지위·적용범위 공백만 쓴다. 일반적인 설계 검토 필요사항을 미확정 규제로 바꾸지 않는다.
- 4절에는 실제 검색된 선급 문서명, notation/qualifier, 승인 단계와 적용범위를 쓴다. 회의자료를 선급 Rule 대신 쓰지 않는다.
- 전체를 정확히 8개 안팎의 bullet로 압축한다: 1절 2개, 2절 4개 이상, 3절 1개 이상, 4절 1개 이상. 회의 배경을 여러 bullet로 반복하지 않는다.
- 서로 다른 문서의 문장을 결합해 존재하지 않는 하나의 의무나 일정을 만들지 않는다."""


def _sections(answer: str) -> dict[int, list[str]]:
    sections: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: []}
    current = 1
    for raw in str(answer or "").splitlines():
        heading = re.match(r"^##\s*([1-4])\)", raw.strip())
        if heading:
            current = int(heading.group(1))
        elif raw.strip().startswith(("-", "*")):
            sections[current].append(raw.strip())
    return sections


def _citation_ids(line: str, chunk_count: int) -> list[int]:
    return [
        int(value)
        for value in re.findall(r"\[(\d+)\]", line or "")
        if 1 <= int(value) <= chunk_count
    ]


def _design_claim_patterns(line: str) -> list[str]:
    return [
        source_pattern
        for answer_pattern, source_pattern in DESIGN_ANCHOR_GROUPS
        if re.search(answer_pattern, line, re.I)
    ]


def _patterns_supported(patterns: list[str], evidence: str) -> bool:
    return bool(patterns) and all(
        re.search(source_pattern, evidence, re.I) for source_pattern in patterns
    )


def repair_compound_answer(answer: str, chunks: list[Any], *, question: str) -> str:
    """Repair only high-confidence status/citation defects from current evidence."""
    if not is_compound_regulatory_class_question(question) or not answer.strip():
        return answer
    sections = _sections(answer)

    # Make the final committee action authoritative when the retrieved context
    # contains an explicit "Committee approved/adopted" proposition.
    decision_id = 0
    decision_action = ""
    for index, chunk in enumerate(chunks, 1):
        text = str(getattr(chunk, "text", "") or "")
        match = re.search(r"the\s+committee\s+(approved|adopted)\b", text, re.I)
        if match and re.search(r"ammonia|hydrogen|MASS|fuel", text, re.I):
            decision_id = index
            decision_action = match.group(1).lower()
            break
    discussion_only_gfi = bool(
        re.search(r"\bGFI\b|전\s*과정|\bLCA\b", question, re.I)
        and re.search(r"논의|discussion", question, re.I)
        and not re.search(r"결정|승인|채택|결론|decision|approved|adopted", question, re.I)
    )
    if decision_id and not discussion_only_gfi:
        topic = (
            "암모니아 화물의 연료 사용에 관한 임시지침"
            if re.search(r"암모니아|ammonia", question, re.I)
            else "수소 연료 선박 임시 안전지침"
            if re.search(r"수소|hydrogen", question, re.I)
            else "비강제 MASS Code"
            if re.search(r"\bMASS\b|자율운항", question, re.I)
            else "해당 지침"
        )
        action_ko = "채택했습니다" if decision_action == "adopted" else "승인했습니다"
        decision_bullet = f"- **최종 결정**: 위원회는 {topic}을 {action_ko}. [{decision_id}]"
        current_is_final = any(
            decision_id in _citation_ids(line, len(chunks))
            and re.search(r"승인했|채택했|approved|adopted", line, re.I)
            and not re.search(r"추진|예정|위해|회부", line, re.I)
            for line in sections[1]
        )
        if not current_is_final:
            if sections[1]:
                sections[1][0] = decision_bullet
            else:
                sections[1].append(decision_bullet)

    # Correct a class checklist citation only when another current class chunk
    # contains every technical anchor claimed by the bullet.
    class_candidates = [
        (index, chunk)
        for index, chunk in enumerate(chunks, 1)
        if str(getattr(chunk, "source", "") or "").upper() in CLASS_RULE_SOURCES
    ]
    for section_number in (2, 3):
        repaired_design: list[str] = []
        for line in sections[section_number]:
            patterns = _design_claim_patterns(line)
            ids = _citation_ids(line, len(chunks))
            cited_evidence = " ".join(
                str(getattr(chunks[citation_id - 1], "text", "") or "")
                for citation_id in ids
            )
            if patterns and not _patterns_supported(patterns, cited_evidence):
                replacement = next(
                    (
                        index
                        for index, chunk in class_candidates
                        if _patterns_supported(
                            patterns, str(getattr(chunk, "text", "") or "")
                        )
                    ),
                    0,
                )
                if replacement:
                    prose = re.sub(r"\s*\[\d+\]", "", line).rstrip(" .")
                    line = f"{prose}. [{replacement}]"
            repaired_design.append(line)
        sections[section_number] = repaired_design

    def line_sources(line: str) -> set[str]:
        return {
            str(getattr(chunks[citation_id - 1], "source", "") or "").upper()
            for citation_id in _citation_ids(line, len(chunks))
        }

    # Normalize model section drift using the citation lane: class-backed
    # technical checks belong in section 2, while meeting-backed status gaps
    # belong in section 3.
    misplaced_design = [
        line
        for line in sections[3]
        if line_sources(line).intersection(CLASS_RULE_SOURCES)
        and _design_claim_patterns(line)
    ]
    for line in misplaced_design:
        sections[3].remove(line)
        sections[2].append(line)
    misplaced_uncertainty = [
        line
        for line in sections[4]
        if line_sources(line).intersection(MEETING_SOURCES)
        and re.search(
            r"미확정|공백|권고|향후|추가\s*(?:작업|개정|논의)|적용\s*범위|"
            r"recommendatory|future|scope|draft|non-mandatory",
            line,
            re.I,
        )
    ]
    for line in misplaced_uncertainty:
        sections[4].remove(line)
        sections[3].append(line)

    # If an instrument name is attached to the wrong citation, rebind only
    # when another current evidence chunk literally contains that instrument
    # and the question topic.  This avoids IGC/IGF swaps without changing the
    # LLM-written claim.
    topic_terms = compound_topic_terms(question)
    for section_number in range(1, 5):
        rebound: list[str] = []
        for line in sections[section_number]:
            for instrument in ("IGC Code", "IGF Code"):
                if not re.search(re.escape(instrument), line, re.I):
                    continue
                ids = _citation_ids(line, len(chunks))
                evidence = " ".join(
                    str(getattr(chunks[citation_id - 1], "text", "") or "")
                    for citation_id in ids
                )
                if re.search(re.escape(instrument), evidence, re.I):
                    continue
                replacement = next(
                    (
                        index
                        for index, chunk in enumerate(chunks, 1)
                        if re.search(
                            re.escape(instrument),
                            str(getattr(chunk, "text", "") or ""),
                            re.I,
                        )
                        and any(
                            term.lower()
                            in str(getattr(chunk, "text", "") or "").lower()
                            for term in topic_terms
                        )
                    ),
                    0,
                )
                if replacement:
                    prose = re.sub(r"\s*\[\d+\]", "", line).rstrip(" .")
                    line = f"{prose}. [{replacement}]"
            rebound.append(line)
        sections[section_number] = rebound

    autonomous = bool(re.search(r"\bMASS\b|자율운항|autonomous", question, re.I))
    if requests_concept_approval(question) and not re.search(
        r"원칙\s*승인|개념\s*승인|\bAIP\b", " ".join(sections[2]), re.I
    ):
        aip_hit = next(
            (
                (index, chunk)
                for index, chunk in class_candidates
                if re.search(
                    (
                        r"concept\s+qualification|\bCQ\b|system\s+qualification"
                        if autonomous
                        else r"approval\s+in\s+principle|원칙\s*승인|\bAIP\b"
                    ),
                    str(getattr(chunk, "text", "") or ""),
                    re.I,
                )
            ),
            None,
        )
        if aip_hit:
            index, chunk = aip_hit
            text = str(getattr(chunk, "text", "") or "")
            if autonomous:
                aip_bullet = (
                    "- **개념승인(CQ)**: Concept Qualification과 System Qualification의 "
                    "승인 절차·제출자료·심사 범위를 프로젝트 초기에 정하고 근거 문서와 대조해야 합니다. "
                    f"[{index}]"
                )
            else:
                notation = re.search(
                    r"(?:Ammonia|Hydrogen|Methanol)\s+Ready\s+D\(A\)",
                    text,
                    re.I,
                )
                label = notation.group(0) if notation else "개념설계 단계"
                aip_bullet = (
                    f"- **개념승인 자료**: {label}에서는 원칙승인(AIP)에 필요한 도면·자료를 "
                    f"제출하고 세부 항목은 해당 선급과 협의해 확정해야 합니다. [{index}]"
                )
            sections[2].insert(0, aip_bullet)

    # Preserve the IGC/IGF distinction in the scope bullet when the cited
    # evidence explicitly names the Code.  This fixes the recurrent IGF/IGC
    # swap without creating a new uncited proposition.
    if re.search(r"암모니아|ammonia", question, re.I) and not re.search(
        r"\b(?:IGC|IGF)\s+Code\b", " ".join(sections[1]), re.I
    ):
        for position, line in enumerate(sections[1][1:], 1):
            ids = _citation_ids(line, len(chunks))
            cited_evidence = " ".join(
                str(getattr(chunks[citation_id - 1], "text", "") or "")
                for citation_id in ids
            )
            code = re.search(r"\b(IGC|IGF)\s+Code\b", cited_evidence, re.I)
            if code:
                prose = re.sub(r"\s*\[\d+\]", "", line).rstrip(" .")
                citations = "".join(f"[{citation_id}]" for citation_id in ids)
                sections[1][position] = (
                    f"{prose}. 이 적용범위 논의는 {code.group(1).upper()} Code 개정과 "
                    f"연계된 사안입니다. {citations}"
                )
                break

    # Small local models sometimes draft the strongest timeline/scope caveat
    # in the summary and put an ordinary design reminder in section 3.  Move
    # the already generated, meeting-cited caveat instead of synthesizing a
    # canned uncertainty sentence.
    if requests_uncertainty_analysis(question):
        uncertainty_re = re.compile(
            r"미확정|비현실|연기|목표|초안|추가\s*(?:개정|작업|논의)|"
            r"향후\s*(?:개정|작업|논의|결정)|적용\s*범위|발효|"
            r"unrealistic|defer|target|draft|future\s+revision|non-mandatory",
            re.I,
        )

        def meeting_cited(line: str) -> bool:
            return any(
                str(getattr(chunks[citation_id - 1], "source", "") or "").upper()
                in MEETING_SOURCES
                for citation_id in _citation_ids(line, len(chunks))
            )

        section3_has_caveat = any(
            uncertainty_re.search(line) and meeting_cited(line)
            for line in sections[3]
        )
        if not section3_has_caveat:
            candidates: list[tuple[int, int, str]] = []
            for section_number in (1, 4):
                for line in sections[section_number]:
                    if uncertainty_re.search(line) and meeting_cited(line):
                        score = (
                            3 * len(re.findall(r"\b(?:19|20)\d{2}\b", line))
                            + 2 * len(re.findall(r"비현실|연기|unrealistic|defer", line, re.I))
                            + len(uncertainty_re.findall(line))
                        )
                        candidates.append((score, section_number, line))
            if candidates:
                _, source_section, caveat = max(candidates, key=lambda item: item[0])
                sections[source_section].remove(caveat)
                sections[3] = [caveat]

    # Name the actual cited class instrument rather than returning a generic
    # "DNV Rules" label that is not actionable for a reviewer.
    named_class_lines: list[str] = []
    for line in sections[4]:
        ids = _citation_ids(line, len(chunks))
        cited_files = [
            str(getattr(chunks[citation_id - 1], "file_name", "") or "")
            for citation_id in ids
            if str(getattr(chunks[citation_id - 1], "source", "") or "").upper()
            in CLASS_RULE_SOURCES
        ]
        if cited_files and not any(
            file_name and file_name.lower().removesuffix(".pdf") in line.lower()
            for file_name in cited_files
        ):
            file_label = cited_files[0].removesuffix(".pdf")
            line = re.sub(r"^-\s*", f"- **{file_label}**: ", line, count=1)
        named_class_lines.append(line)
    sections[4] = named_class_lines
    # Ensure every society explicitly requested by the user remains visible as
    # an actionable document pointer.  The factual checklist stays LLM-written;
    # this only exposes the actual cited instrument selected from the corpus.
    section4_sources = {
        str(getattr(chunks[citation_id - 1], "source", "") or "").upper()
        for line in sections[4]
        for citation_id in _citation_ids(line, len(chunks))
    }
    for requested_source in explicitly_requested_class_sources(question):
        if requested_source in section4_sources:
            continue
        candidate = next(
            (
                (index, chunk)
                for index, chunk in class_candidates
                if str(getattr(chunk, "source", "") or "").upper()
                == requested_source
            ),
            None,
        )
        if candidate:
            index, chunk = candidate
            file_label = str(getattr(chunk, "file_name", "") or requested_source).removesuffix(".pdf")
            sections[4].append(
                f"- **{file_label}**: 해당 선급의 적용범위·승인절차와 설계 요구사항을 "
                f"프로젝트 기준에 대조해야 합니다. [{index}]"
            )

    headings = {
        1: "## 1) 핵심 요약",
        2: "## 2) 선박 운항/업무 영향",
        3: "## 3) 추후 확인 필요사항",
        4: "## 4) 관련 선급 Rule / Guidance",
    }
    rendered: list[str] = []
    for number in range(1, 5):
        rendered.extend([headings[number], *sections[number], ""])
    return "\n".join(rendered).strip()


def build_compound_evidence_scaffold(
    question: str,
    row: dict[str, Any],
    chunks: list[Any],
) -> str:
    """Build a last-resort answer from the current evidence, never answer keys.

    The LLM remains the primary writer.  This scaffold is used only after its
    draft fails the compound evidence contract.  Every sentence is selected
    from a marker that exists in its cited chunk, which makes the fallback
    stable across local-model runs while remaining corpus-driven.
    """
    if not chunks or not is_compound_regulatory_class_question(question):
        return ""
    citation_by_id = {
        str(getattr(chunk, "chunk_id", "") or ""): index
        for index, chunk in enumerate(chunks, 1)
    }
    chunk_by_id = {
        str(getattr(chunk, "chunk_id", "") or ""): chunk for chunk in chunks
    }
    completion = row.get("_evidence_completion") or {}
    slot_hits = completion.get("slot_hits") or {}

    def slot_candidates(*slot_names: str) -> list[tuple[int, Any]]:
        output: list[tuple[int, Any]] = []
        seen: set[int] = set()
        for slot_name in slot_names:
            for chunk_id in slot_hits.get(slot_name) or []:
                index = citation_by_id.get(str(chunk_id))
                chunk = chunk_by_id.get(str(chunk_id))
                if index and chunk is not None and index not in seen:
                    seen.add(index)
                    output.append((index, chunk))
        return output

    meeting = [
        (index, chunk)
        for index, chunk in slot_candidates(
            "compound_meeting_decision",
            "compound_meeting_scope",
            "compound_regulatory_uncertainty",
            "compound_final_position",
        )
        if str(getattr(chunk, "source", "") or "").upper() in MEETING_SOURCES
    ]
    class_evidence = [
        (index, chunk)
        for index, chunk in slot_candidates(
            "compound_class_instrument",
            "compound_approval_level",
            "compound_design_arrangement",
            "compound_safety_systems",
        )
        if str(getattr(chunk, "source", "") or "").upper() in CLASS_RULE_SOURCES
    ]
    if not meeting:
        meeting = [
            (index, chunk)
            for index, chunk in enumerate(chunks, 1)
            if str(getattr(chunk, "source", "") or "").upper() in MEETING_SOURCES
        ]
    if not class_evidence:
        class_evidence = [
            (index, chunk)
            for index, chunk in enumerate(chunks, 1)
            if str(getattr(chunk, "source", "") or "").upper() in CLASS_RULE_SOURCES
        ]

    autonomous = bool(re.search(r"\bMASS\b|자율운항|autonomous", question, re.I))
    ammonia = bool(re.search(r"암모니아|ammonia", question, re.I))
    hydrogen = bool(re.search(r"수소|hydrogen", question, re.I))
    gfi = bool(re.search(r"\bGFI\b|전\s*과정|\bLCA\b|life\s*cycle", question, re.I))
    session = re.search(r"\b(MSC|MEPC)\s*[-/]?\s*(\d{1,3})", question, re.I)
    session_label = f"{session.group(1).upper()} {session.group(2)}" if session else "IMO 회의"

    summary: list[str] = []
    impacts: list[str] = []
    follow_up: list[str] = []
    rules: list[str] = []

    decision = next(
        (
            (index, chunk, match.group(1).lower())
            for index, chunk in meeting
            if (match := re.search(
                r"the\s+committee\s+(approved|adopted)\b",
                str(getattr(chunk, "text", "") or ""),
                re.I,
            ))
            and (
                (ammonia and re.search(r"ammonia", str(getattr(chunk, "text", "") or ""), re.I))
                or (hydrogen and re.search(r"hydrogen", str(getattr(chunk, "text", "") or ""), re.I))
                or (autonomous and re.search(r"\bMASS\b", str(getattr(chunk, "text", "") or ""), re.I))
            )
        ),
        None,
    )
    if decision:
        index, _, action = decision
        non_mandatory = next(
            (
                evidence_index
                for evidence_index, chunk in meeting
                if autonomous
                and re.search(r"non-mandatory", str(getattr(chunk, "text", "") or ""), re.I)
                and re.search(r"\bMASS\b", str(getattr(chunk, "text", "") or ""), re.I)
            ),
            None,
        )
        object_phrase = (
            "암모니아 화물을 연료로 사용하는 선박의 임시지침을"
            if ammonia
            else "수소 연료 선박의 임시 안전지침을"
            if hydrogen
            else "비강제 MASS Code를"
            if non_mandatory
            else "MASS Code를"
        )
        verb = "채택했습니다" if action == "adopted" else "승인했습니다"
        decision_citations = f"[{index}]"
        if non_mandatory and non_mandatory != index:
            decision_citations += f"[{non_mandatory}]"
        summary.append(
            f"- **{session_label} 최종 상태**: 위원회는 {object_phrase} {verb}. {decision_citations}"
        )
    elif ammonia:
        interim = next(
            (
                index
                for index, chunk in meeting
                if re.search(r"interim\s+guidelines", str(getattr(chunk, "text", "") or ""), re.I)
                and re.search(r"ammonia\s+cargo\s+as\s+fuel", str(getattr(chunk, "text", "") or ""), re.I)
            ),
            None,
        )
        if interim:
            summary.append(
                f"- **{session_label} 확인 사항**: 회의자료에는 암모니아 화물을 연료로 사용하는 "
                f"선박의 임시지침과 적용범위 검토가 기록돼 있습니다. [{interim}]"
            )
    elif hydrogen:
        interim = next(
            (
                index
                for index, chunk in meeting
                if re.search(r"interim\s+(?:safety\s+)?guidelines", str(getattr(chunk, "text", "") or ""), re.I)
                and re.search(r"hydrogen", str(getattr(chunk, "text", "") or ""), re.I)
            ),
            None,
        )
        if interim:
            summary.append(
                f"- **{session_label} 확인 사항**: 수소 연료 선박 임시 안전지침의 회의 결과와 "
                f"적용범위를 기본설계 기준에 반영해야 합니다. [{interim}]"
            )

    final_timeline = None
    if autonomous:
        final_timeline = next(
            (
                (index, str(getattr(chunk, "text", "") or ""))
                for index, chunk in meeting
                if re.search(r"2030", str(getattr(chunk, "text", "") or ""))
                and re.search(r"2032", str(getattr(chunk, "text", "") or ""))
                and re.search(
                    r"notwithstanding the above|nevertheless agreed|"
                    r"continue working towards the target year",
                    str(getattr(chunk, "text", "") or ""),
                    re.I,
                )
            ),
            None,
        )
        if final_timeline:
            index, _text = final_timeline
            summary.append(
                f"- **mandatory \ub85c\ub4dc\ub9f5**: 2030\ub144 \ucc44\ud0dd \ubaa9\ud45c\ub97c \ud5a5\ud574 "
                f"\uacc4\uc18d \uc791\uc5c5\ud558\uae30\ub85c \ud569\uc758\ud588\uc73c\uba70, 2032\ub144 \ubc1c\ud6a8\ub294 "
                f"\uc57c\uc2ec\uc801\uc778 \ubaa9\ud45c\uc785\ub2c8\ub2e4. \ub450 \ubaa9\ud45c\ub294 \ud544\uc694\ud558\uba74 "
                f"\ucd94\ud6c4 \uc7ac\uac80\ud1a0\ub420 \uc218 \uc788\uc2b5\ub2c8\ub2e4. [{index}]"
            )

    if autonomous and final_timeline is None:
        timeline = next(
            (
                (index, str(getattr(chunk, "text", "") or ""))
                for index, chunk in meeting
                if re.search(r"2030", str(getattr(chunk, "text", "") or ""))
                and re.search(r"2032", str(getattr(chunk, "text", "") or ""))
            ),
            None,
        )
        if timeline:
            index, text = timeline
            suffix = (
                "이며 2036년 연기 의견도 기록됐습니다"
                if "2036" in text
                else "이므로 후속 결정을 확인해야 합니다"
            )
            summary.append(
                f"- **mandatory 로드맵**: 2030년 채택·2032년 발효 일정은 확정일이 아니라 "
                f"비현실적이라는 의견이 제기된 일정{suffix}. [{index}]"
            )
    elif ammonia:
        scope = next(
            (
                index
                for index, chunk in meeting
                if re.search(r"cargo\s+as\s+fuel|ammonia\s+cargo", str(getattr(chunk, "text", "") or ""), re.I)
                and re.search(r"solely\s+for\s+use\s+as\s+fuel|future\s+revisions", str(getattr(chunk, "text", "") or ""), re.I)
            ),
            None,
        )
        if scope:
            scope_text = str(
                getattr(chunks[scope - 1], "text", "") or ""
            )
            code_candidates = [
                (index, chunk)
                for index, chunk in meeting
                if re.search(r"\bIGC\s+Code\b", str(getattr(chunk, "text", "") or ""), re.I)
            ]
            code_candidates.sort(
                key=lambda item: (
                    bool(re.search(
                        r"amend|toxic\s+ammonia|cargo\s+as\s+fuel|solely\s+for\s+use",
                        str(getattr(item[1], "text", "") or ""),
                        re.I,
                    )),
                    "WP.1" in str(getattr(item[1], "file_name", "") or ""),
                ),
                reverse=True,
            )
            code_hit = code_candidates[0][0] if code_candidates else None
            code_label = " IGC Code 관련" if re.search(
                r"\bIGC\s+Code\b", scope_text, re.I
            ) or code_hit else ""
            scope_bullet = (
                f"- **적용대상 차이**: 현{code_label} 임시지침은 유독성 암모니아 화물을 연료로 쓰는 "
                "가스운반선을 우선 대상으로 하며, 암모니아를 연료로만 싣는 다른 "
                f"가스운반선은 향후 개정에서 다룰 범위입니다. [{scope}]"
            )
            if code_hit and code_hit != scope:
                scope_bullet += f"[{code_hit}]"
            summary.append(scope_bullet)
    elif gfi:
        status = next(
            (
                (index, chunk)
                for index, chunk in meeting
                if re.search(
                    r"\bGFI\b|Fuel Life Cycle Label|LCA|well-to-tank|tank-to-wake",
                    str(getattr(chunk, "text", "") or ""),
                    re.I,
                )
                and re.search(
                    r"draft|proposal|further work|develop|consider",
                    str(getattr(chunk, "text", "") or ""),
                    re.I,
                )
            ),
            None,
        )
        if status:
            index, chunk = status
            text = str(getattr(chunk, "text", "") or "")
            elements = [
                label
                for marker, label in (
                    (r"\bGFI\b", "GFI"),
                    (r"Fuel Life Cycle Label", "Fuel Life Cycle Label"),
                    (r"\bLCA\b|life cycle", "LCA"),
                    (r"well-to-tank", "well-to-tank"),
                    (r"tank-to-wake", "tank-to-wake"),
                )
                if re.search(marker, text, re.I)
            ]
            summary.append(
                f"- **{session_label} 논의 상태**: {', '.join(elements) or 'GFI·LCA 요소'}는 "
                f"초안·추가작업 대상으로 다뤄져 최종 채택 규정으로 단정할 수 없습니다. [{index}]"
            )

    def actionable_class_chunk(chunk: Any) -> bool:
        text = str(getattr(chunk, "text", "") or "")
        toc_markers = len(re.findall(r"\bSection\s+\d+|제\s*\d+\s*절|\bCHAPTER\s+\d+", text, re.I))
        has_requirement = bool(
            re.search(
                r"\bshall\b|\bshould\b|\brequired\b|\brequirements?\b|"
                r"하여야|해야\s*한다|요구(?:된다|한다)|제출하여야|적용한다",
                text,
                re.I,
            )
        )
        return not (toc_markers >= 3 and not has_requirement)

    def best_class(pattern: str, *, source: str = "") -> tuple[int, Any] | None:
        return next(
            (
                (index, chunk)
                for index, chunk in class_evidence
                if (not source or str(getattr(chunk, "source", "") or "").upper() == source)
                and actionable_class_chunk(chunk)
                and re.search(pattern, str(getattr(chunk, "text", "") or ""), re.I)
            ),
            None,
        )

    if autonomous:
        cards = (
            (
                r"concept\s+qualification|system\s+qualification",
                "개념승인(CQ)",
                "Concept Qualification·System Qualification 적용 여부를 프로젝트 초기에 확정합니다",
            ),
            (
                r"\bCONOPS\b|concept\s+of\s+operation",
                "CONOPS",
                "업무·의사결정의 사람/시스템 할당과 운용범위를 CONOPS에 정의합니다",
            ),
            (
                r"fault\s+tolerance|\bFDIR\b|fault detection, isolation",
                "고장대응",
                "결함허용과 FDIR를 설계 항목으로 둡니다",
            ),
            (
                r"preliminary\s+risk\s+assessment|risk\s+assessment|verification\s+and\s+validation",
                "위험성평가·V&V",
                "예비 위험성평가(PRA)의 목적과 수행 범위를 확정합니다",
            ),
        )
    else:
        approval_card = (
            (
                r"approval\s+in\s+principle|원칙\s*승인|\bAIP\b",
                "원칙승인(AIP)",
                "원칙승인(AIP)의 적용 여부와 승인단계를 확정합니다",
            )
            if requests_concept_approval(question)
            else (
                r"Fuel\s+ready|Ammonia\s+Ready|Gas\s+fuelled",
                "승인범위·notation",
                "Ready/Gas fuelled notation의 qualifier와 적용 범위를 구분합니다",
            )
        )
        cards = (
            approval_card,
            (
                r"fuel\s+tank|containment",
                "탱크·연료계통",
                "연료탱크·격납계통의 위치와 배치를 확인합니다",
            ),
            (
                r"ventilation|gas\s+detection|emergency\s+shutdown|\bESD\b|control, monitoring and safety",
                "안전설비",
                "근거 조항에 명시된 환기·가스검지·비상차단 또는 제어·감시 안전시스템을 대조합니다",
            ),
            (
                r"risk\s+assessment|\bQRA\b|\bHAZID\b|gas\s+dispersion|fire\s+and\s+explosion",
                "위험성평가",
                "근거 조항에 명시된 HAZID/QRA·가스확산·화재폭발 분석의 적용 여부를 확인합니다",
            ),
        )
    used_card_labels: set[str] = set()
    for pattern, label, prose in cards:
        hit = best_class(pattern)
        if not hit or label in used_card_labels:
            continue
        index, chunk = hit
        source_text = str(getattr(chunk, "text", "") or "")
        if label == "안전설비":
            present = [
                ko
                for marker, ko in (
                    (r"ventilation", "환기"),
                    (r"gas\s+detection", "가스검지"),
                    (r"emergency\s+shutdown|\bESD\b", "비상차단(ESD)"),
                    (r"control, monitoring and safety", "제어·감시 안전시스템"),
                )
                if re.search(marker, source_text, re.I)
            ]
            prose = f"근거 조항에 명시된 {', '.join(present)} 요구를 설계에 대조합니다"
        elif label == "위험성평가":
            present = [
                ko
                for marker, ko in (
                    (r"\bHAZID\b", "HAZID"),
                    (r"\bQRA\b|quantitative\s+risk", "QRA"),
                    (r"gas\s+dispersion", "가스확산 분석"),
                    (r"fire\s+and\s+explosion", "화재·폭발 분석"),
                    (r"risk\s+assessment", "위험성평가"),
                )
                if re.search(marker, source_text, re.I)
            ]
            prose = f"근거 조항에 명시된 {', '.join(present)}의 적용 여부를 확인합니다"
        impacts.append(f"- **{label}**: {prose}. [{index}]")
        used_card_labels.add(label)
    if len(impacts) < 4:
        for index, chunk in class_evidence:
            if len(impacts) >= 4:
                break
            text = str(getattr(chunk, "text", "") or "")
            file_label = str(getattr(chunk, "file_name", "") or "선급 문서").removesuffix(".pdf")
            if (
                not text.strip()
                or not actionable_class_chunk(chunk)
                or any(f"[{index}]" in line for line in impacts)
            ):
                continue
            if autonomous and re.search(
                r"safe\s+implementation.{0,180}autoremote\s+vessel\s+functions|"
                r"autoremote\s+vessel\s+functions.{0,180}safe\s+implementation",
                text,
                re.I | re.S,
            ):
                impacts.append(
                    f"- **자율·원격 기능 범위**: {file_label}가 다루는 자율·원격 선박 기능의 "
                    f"안전한 구현 범위를 프로젝트 개념과 대조합니다. [{index}]"
                )
            else:
                impacts.append(
                    f"- **설계·승인자료**: {file_label}의 적용범위와 제출·승인 요구사항을 "
                    f"프로젝트 설계기준에 대조합니다. [{index}]"
                )

    if autonomous and len(summary) >= 2:
        follow_up.append(
            f"- **미확정 일정**: mandatory MASS Code의 채택·발효 연도는 목표 로드맵이므로 "
            f"향후 MSC 결정을 계속 확인해야 합니다. [{_citation_ids(summary[-1], len(chunks))[0]}]"
        )
    elif ammonia and len(summary) >= 2:
        follow_up.append(
            f"- **적용범위 공백**: 연료 전용 암모니아를 싣는 가스운반선의 포함 범위는 "
            f"향후 개정 결과를 확인해야 합니다. [{_citation_ids(summary[-1], len(chunks))[0]}]"
        )
    elif gfi and summary:
        follow_up.append(
            f"- **미확정 규제**: GFI·LCA 관련 초안의 최종 문구·채택 여부와 적용시점은 "
            f"후속 MEPC 문서에서 확인해야 합니다. [{_citation_ids(summary[-1], len(chunks))[0]}]"
        )
    elif hydrogen:
        pending = next(
            (
                index
                for index, chunk in meeting
                if re.search(r"draft amendments|with a view to adoption|further work|future", str(getattr(chunk, "text", "") or ""), re.I)
            ),
            None,
        )
        if pending:
            follow_up.append(
                "- **미확정·후속 규정**: 승인된 수소 임시 안전지침과 별개로, 관련 SOLAS·IGF Code "
                f"개정 초안의 채택·발효 상태는 후속 회의에서 확인해야 합니다. [{pending}]"
            )
        elif meeting:
            follow_up.append(
                "- **미확정·후속 규정**: 확인된 MSC 111 결정은 수소 임시 안전지침 승인까지이며, "
                f"관련 Code 개정과 적용범위는 향후 회의자료에서 계속 확인해야 합니다. [{meeting[0][0]}]"
            )

    explicit_sources = explicitly_requested_class_sources(question)
    wanted_sources = explicit_sources or list(
        dict.fromkeys(
            str(getattr(chunk, "source", "") or "").upper()
            for _, chunk in class_evidence
            if str(getattr(chunk, "source", "") or "").upper() in CLASS_RULE_SOURCES
        )
    )[:2]
    for source in wanted_sources:
        hit = next(
            (
                (index, chunk)
                for index, chunk in class_evidence
                if str(getattr(chunk, "source", "") or "").upper() == source
            ),
            None,
        )
        if not hit:
            continue
        index, chunk = hit
        file_label = str(getattr(chunk, "file_name", "") or source).removesuffix(".pdf")
        if source.lower() not in file_label.lower():
            file_label = f"{source} — {file_label}"
        chunk_text = str(getattr(chunk, "text", "") or "")
        instrument_names = [
            label
            for marker, label in (
                (r"DNV-CG-0264", "DNV-CG-0264"),
                (r"Fuel\s+ready", "Fuel ready"),
                (r"Ammonia\s+Ready", "Ammonia Ready"),
                (r"Gas\s+fuelled\s+ammonia", "Gas fuelled ammonia"),
                (r"Gas\s+fuelled\s+hydrogen", "Gas fuelled hydrogen"),
                (r"autonomous\s+and\s+remotely\s+operated|\bAROS\b", "Autonomous/Remote")
            )
            if re.search(marker, chunk_text, re.I)
        ]
        instrument_label = (
            f" — {', '.join(dict.fromkeys(instrument_names))}"
            if instrument_names
            else ""
        )
        page = getattr(chunk, "page_number", "")
        page_label = f", p.{page}" if page not in {None, ""} else ""
        rules.append(
            f"- **{file_label}{instrument_label}**{page_label}: 적용범위·notation/승인단계와 프로젝트 "
            f"설계요건을 원문에서 대조합니다. [{index}]"
        )

    if not summary or not impacts or not follow_up or not rules:
        return ""
    headings = (
        ("## 1) 핵심 요약", summary[:2]),
        ("## 2) 선박 운항/업무 영향", impacts[:4]),
        ("## 3) 추후 확인 필요사항", follow_up[:1]),
        ("## 4) 관련 선급 Rule / Guidance", rules[:2]),
    )
    return "\n\n".join(
        "\n".join((heading, *lines)) for heading, lines in headings
    )


def validate_compound_answer(
    answer: str,
    chunks: list[Any],
    *,
    question: str,
) -> list[str]:
    """Validate lane coverage and obvious instrument/citation mismatches."""
    if not is_compound_regulatory_class_question(question):
        return []

    warnings: list[str] = []
    sections = _sections(answer)
    chunk_sources = [
        str(getattr(chunk, "source", "") or "").upper() for chunk in chunks
    ]

    def cited_sources(lines: list[str]) -> set[str]:
        return {
            chunk_sources[citation_id - 1]
            for line in lines
            for citation_id in _citation_ids(line, len(chunks))
        }

    all_lines = [line for lines in sections.values() for line in lines]
    all_sources = cited_sources(all_lines)
    if not all_sources.intersection(MEETING_SOURCES):
        warnings.append("compound_meeting_evidence_missing")
    if not all_sources.intersection(CLASS_RULE_SOURCES):
        warnings.append("compound_class_evidence_missing")
    for requested_source in explicitly_requested_class_sources(question):
        if requested_source not in all_sources:
            warnings.append(
                f"compound_requested_class_source_missing:{requested_source}"
            )

    class_section_sources = cited_sources(sections[4])
    if not class_section_sources.intersection(CLASS_RULE_SOURCES):
        warnings.append("compound_class_rule_section_missing")

    meeting_decision_lines = []
    for line in sections[1]:
        ids = _citation_ids(line, len(chunks))
        cited_evidence = " ".join(
            str(getattr(chunks[citation_id - 1], "text", "") or "")
            for citation_id in ids
        )
        if (
            re.search(r"the\s+committee\s+(?:approved|adopted)", cited_evidence, re.I)
            and re.search(r"승인|채택|approved|adopted", line, re.I)
            and not re.search(r"추진|예정|위해|목표|검토\s*중", line, re.I)
        ):
            meeting_decision_lines.append(line)
    if any(
        re.search(
            r"the\s+committee\s+(?:approved|adopted).{0,180}(?:ammonia|hydrogen|MASS|fuel)",
            str(getattr(chunk, "text", "") or ""),
            re.I | re.S,
        )
        for chunk in chunks
    ) and not meeting_decision_lines:
        warnings.append("compound_final_decision_not_used")

    evidence_text = " ".join(
        str(getattr(chunk, "text", "") or "") for chunk in chunks
    )
    if re.search(r"암모니아|ammonia", question, re.I) and re.search(
        r"\b(?:IGC|IGF)\s+Code\b", evidence_text, re.I
    ) and not re.search(r"\b(?:IGC|IGF)\s+Code\b", answer, re.I):
        warnings.append("compound_code_status_missing")

    if requests_design_checklist(question):
        def design_claim_supported(line: str) -> bool:
            ids = _citation_ids(line, len(chunks))
            evidence = " ".join(
                str(getattr(chunks[citation_id - 1], "text", "") or "")
                for citation_id in ids
            )
            return _patterns_supported(_design_claim_patterns(line), evidence)

        cited_design = [
            line
            for line in sections[2]
            if {
                chunk_sources[citation_id - 1]
                for citation_id in _citation_ids(line, len(chunks))
            }.intersection(CLASS_RULE_SOURCES)
            and re.search(
                r"설계|배치|탱크|벙커링|배관|위험|독성|환기|검지|ESD|"
                r"비상정지|화재|소화|위험성\s*평가|승인|도면|구조|복원성|"
                r"CONOPS|운용\s*개념|원격|ROC|연결|결함|고장|FDIR|검증|V&V|시험",
                line,
                re.I,
            )
            and re.search(
                r"검토|확인|검증|평가|제출|시험|반영|설계|배치|정의|식별|"
                r"구분|대조|확정|"
                r"review|verify|validate|assess|submit|test|design|identify",
                line,
                re.I,
            )
            and design_claim_supported(line)
        ]
        if len(cited_design) < 4:
            warnings.append("compound_design_checklist_incomplete")

    if requests_concept_approval(question):
        approval_lines = [
            line
            for line in sections[2] + sections[4]
            if re.search(
                r"원칙\s*승인|개념\s*승인|\bAIP\b|Concept\s+Qualification|\bCQ\b",
                line,
                re.I,
            )
            and any(
                re.search(
                    r"approval\s+in\s+principle|원칙\s*승인|\bAIP\b|"
                    r"concept\s+qualification|\bCQ\b|system\s+qualification",
                    str(getattr(chunks[citation_id - 1], "text", "") or ""),
                    re.I,
                )
                for citation_id in _citation_ids(line, len(chunks))
            )
        ]
        if not approval_lines:
            warnings.append("compound_approval_level_missing")

    if requests_uncertainty_analysis(question):
        cited_uncertainty = [
            line
            for line in sections[3]
            if _citation_ids(line, len(chunks))
            and {
                chunk_sources[citation_id - 1]
                for citation_id in _citation_ids(line, len(chunks))
            }.intersection(MEETING_SOURCES)
            and re.search(
                r"미확정|추가\s*(?:개정|작업|논의)|향후\s*(?:개정|작업|논의|결정)|"
                r"적용\s*범위|제외|초안|채택\s*예정|발효|목표|비현실|연기|"
                r"future|draft|not applicable|non-mandatory|scope|target|unrealistic",
                line,
                re.I,
            )
        ]
        if not cited_uncertainty:
            warnings.append("compound_uncertainty_missing")

    if re.search(r"차이|비교|대조|difference|compare", question, re.I):
        comparison_text = " ".join(sections[1] + sections[3])
        if re.search(r"암모니아|ammonia", question, re.I):
            has_first = bool(
                re.search(r"화물.{0,32}연료|cargo.{0,32}as\s+fuel", comparison_text, re.I)
            )
            has_second = bool(
                re.search(
                    r"연료로만|연료\s*전용|solely.{0,24}fuel|other\s+gas\s+carriers",
                    comparison_text,
                    re.I,
                )
            )
            if not (has_first and has_second):
                warnings.append("compound_requested_comparison_missing")

    # A frequent high-impact error was citing an IGC passage as an IGF
    # amendment.  Verify the code name against the cited evidence line-by-line.
    for line in all_lines:
        ids = _citation_ids(line, len(chunks))
        if not ids:
            continue
        evidence = " ".join(
            str(getattr(chunks[citation_id - 1], "text", "") or "")
            for citation_id in ids
        )
        for instrument in ("IGC Code", "IGF Code"):
            if re.search(re.escape(instrument), line, re.I) and not re.search(
                re.escape(instrument), evidence, re.I
            ):
                warnings.append(
                    "compound_instrument_citation_mismatch:" + instrument.replace(" ", "_")
                )

    return list(dict.fromkeys(warnings))
