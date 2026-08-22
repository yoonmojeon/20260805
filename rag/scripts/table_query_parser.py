"""Parse table QA questions into structured lookup slots."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from table_normalize_lib import (
    ENTITY_ALIASES,
    MATERIAL_GRADE_RE,
    expand_entity_aliases,
    extract_paren_aliases,
    extract_units,
    normalize_material_grade,
    normalize_token,
)

QUERY_TYPES = (
    "table_lookup",
    "row_lookup",
    "cell_lookup",
    "note_lookup",
    "condition_lookup",
)

ROW_DOMAIN_TERMS = (
    "평형수탱크",
    "화물창",
    "화물탱크",
    "연료유탱크",
    "빌지저장탱크",
    "이중저탱크",
    "디프탱크",
    "피크탱크",
    "기관실",
    "주위벽",
    "호퍼탱크",
    "이중선측",
    "수평거더",
    "구명정",
    "개방갑판",
    "안전대피구역",
    "탱크",
    "구역",
)

INSPECTION_TERMS = (
    "제1차 정기검사",
    "제2차 정기검사",
    "제3차 정기검사",
    "제4차 및 이후 정기검사",
    "정기검사",
    "reporting",
    "검사차수",
    "검사 보고",
)

MECHANICAL_TERMS = ("항복", "인장", "연신", "충격", "흡수에너지", "기계적 성질", "n/mm")
CHEMISTRY_TERMS = ("화학", "함량", "허용한도", "성분", "탈산")
NOTE_TERMS = ("비고", "주석", "footnote", "각주")
CONDITION_TERMS = ("선령", "두께", "조건", "구간", "년", "mm", "이상", "이하", "초과", "미만")
SUMMARY_TERMS = ("구조", "설명", "주요", "개요", "매트릭스")
QUOTED_RE = re.compile(r"['‘“]([^'’”]{1,240})['’”]")
ASCII_ALIAS_RE_TEMPLATE = r"(?<![0-9A-Za-z]){}(?![0-9A-Za-z])"
MAIN_TOPIC_PARTICLE_RE = re.compile(
    r"^(.{2,220})(?:에는|에서는|은|는)(?:\s+|[?？.!…]|$)"
)
# "…행의 허용기준은?" / "…행의 적용두께(mm)는?" — common explicit cell slots.
ROW_OF_ATTRIBUTE_RE = re.compile(
    r"(.+?)\s*행의\s+([0-9A-Za-z가-힣·/()%~∼\- ]{2,50}?)"
    r"(?:은|는|이|가|을|를|와|과)?\s*[?？.!…]?\s*$"
)
TECHNICAL_SERIES_RE = re.compile(
    r"\b[A-Z]{1,8}(?:[- ]?\d+[A-Z]?)?(?:\s*[·,/]\s*[A-Z0-9-]{1,10})+"
)
QUESTION_FILLERS = {
    "무엇인가", "어떤", "어느", "얼마인가", "몇", "필요한가", "요구되는가",
    "적용하는가", "평가하는가", "의미하는가", "사용해야", "표시해야", "있는가",
}
# Possessive tails that are glossary/prose asks, not table column headers.
NON_COLUMN_ATTRIBUTES = {
    "정의", "의미", "취지", "목적", "개요", "설명", "내용", "배경", "요지",
}
ATTRIBUTE_TERMS = (
    "평가 방법", "구조평가 방법", "적용 규정", "설계하중 시나리오", "자동 작동",
    "안전사용하중", "설치비율", "허용응력", "허용 차이", "허용 바깥지름",
    "허용기준", "적용두께", "강종", "등급", "시험전압", "시험압력", "시험재의 수",
    "분류번호", "보호등급", "회전속도", "절단하중", "판정기준", "설계온도",
    "공칭 두께", "시험규격", "표시 장소", "동력 빌지펌프", "운항거리 제한",
    "방화 보존성", "최소 용접 다리 길이", "최소 각장", "용접 다리 길이",
    "CMS 통일명칭", "통일명칭", "격벽 위치", "제조법 승인 적용 장",
    "설계 적용 장", "시험압력수두", "최종강도 검토", "최종강도",
    "부식추가", "공칭값 오차", "외판으로부터 거리",
)


@dataclass
class ParsedTableQuery:
    raw_question: str
    query_type: str = "cell_lookup"
    row_entities: list[str] = field(default_factory=list)
    column_entities: list[str] = field(default_factory=list)
    table_topic_candidates: list[str] = field(default_factory=list)
    unit_candidates: list[str] = field(default_factory=list)
    condition_candidates: list[str] = field(default_factory=list)
    keyword_terms: list[str] = field(default_factory=list)
    subject_candidates: list[str] = field(default_factory=list)
    attribute_candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(t for t in items if t))


def _infer_column_entities(question: str) -> list[str]:
    cols: list[str] = []
    inspection_match = re.search(
        r"제\s*(\d+)\s*차\s*(?:(및\s*)?이후\s*)?정기검사",
        question,
    )
    if inspection_match:
        number = inspection_match.group(1)
        if "이후" in inspection_match.group(0):
            cols.extend(
                [
                    f"제{number}차 및 이후 정기검사",
                    f"제{number}차 이후 정기검사",
                ]
            )
        else:
            cols.append(f"제{number}차 정기검사")
    quoted = [normalize_token(v) for v in QUOTED_RE.findall(question)]
    if len(quoted) >= 2 and any(term in question for term in ("값", "해당", "행")):
        cols.append(quoted[1])
    for term in INSPECTION_TERMS:
        if term.lower() in question.lower() or term in question:
            cols.append(term)
    for _base, aliases in extract_paren_aliases(question):
        cols.extend(aliases[:3])
    for canon, forms in ENTITY_ALIASES.items():
        if len(canon) <= 3 and canon.isalpha():
            for form in forms:
                escaped = re.escape(str(form))
                form_s = str(form)
                # Short ASCII aliases ("age", "c") must be whole tokens —
                # otherwise "age" matches inside "agenda".
                if form_s.isascii() and len(form_s) <= 8:
                    matched = re.search(
                        rf"(?<![0-9A-Za-z]){escaped}(?![0-9A-Za-z])",
                        question,
                        re.I,
                    )
                elif len(form_s) == 1:
                    matched = re.search(
                        rf"(?<![0-9A-Za-z가-힣]){escaped}(?![0-9A-Za-z가-힣])",
                        question,
                        re.I,
                    )
                else:
                    matched = re.search(escaped, question, re.I)
                if matched:
                    cols.append(canon)
                    break
    # Age-range column patterns
    age_patterns = [
        (r"15\s*년\s*(을\s*)?(초과|넘)", "15년< 선령"),
        (r"10\s*년\s*초과.*15\s*년|10\s*[~\-–]\s*15\s*년", "10년< 선령≤15년"),
        (r"5\s*년\s*초과.*10\s*년|5\s*[~\-–]\s*10\s*년", "5년< 선령≤10년"),
    ]
    for pat, label in age_patterns:
        if re.search(pat, question):
            cols.append(label)
    return _dedupe(cols)


def _natural_slots(question: str) -> tuple[list[str], list[str]]:
    """Extract conservative subject/attribute phrases from ordinary Korean questions."""
    rows: list[str] = []
    cols: list[str] = []
    q = question.rstrip(" ?.!")

    row_of_attr = ROW_OF_ATTRIBUTE_RE.search(q)
    if row_of_attr:
        row_phrase = normalize_token(row_of_attr.group(1)).strip(" ,/")
        col_phrase = normalize_token(row_of_attr.group(2)).strip(" ,")
        # Drop leading file/page anchors from the row subject.
        row_phrase = re.sub(
            r"^[0-9A-Za-z가-힣_.\-]+\.pdf\s+\d+\s*페이지\s*표에서\s*",
            "",
            row_phrase,
        ).strip(" ,/")
        if 2 <= len(row_phrase) <= 180:
            rows.append(row_phrase)
        if 2 <= len(col_phrase) <= 80 and col_phrase not in NON_COLUMN_ATTRIBUTES:
            cols.append(col_phrase)

    # The last topic-marked noun phrase is usually the lookup subject.  If it
    # contains a possessive construction, the part after the final '의' is
    # normally the requested attribute and the left side is the row subject.
    matches = list(MAIN_TOPIC_PARTICLE_RE.finditer(q))
    if matches:
        subject = normalize_token(matches[-1].group(1)).strip(" ,")
        if "의 " in subject:
            left, right = subject.rsplit("의 ", 1)
            right = right.strip()
            if right in NON_COLUMN_ATTRIBUTES:
                # "…의 정의/의미" is glossary prose, not a table subject/column.
                pass
            elif right in ATTRIBUTE_TERMS or re.search(
                r"(?:등급|강종|압력|온도|두께|지름|속도|거리|설치비율|"
                r"분류번호|보호등급|시험전압|판정기준|허용기준|"
                r"설계하중\s*시나리오|시험재(?:의\s*)?수|통일명칭)$",
                right,
            ):
                if len(left.strip()) >= 2:
                    rows.append(left.strip())
                if 2 <= len(right) <= 80:
                    cols.append(right)
            elif 2 <= len(subject) <= 180:
                # Internal possessive phrase ("한 개의 중량") belongs to the
                # row description and must not be split into a fake column.
                rows.append(subject)
        elif 2 <= len(subject) <= 180:
            rows.append(subject)

    # Explicit engineering code series are strong row keys even when Korean
    # particles obscure the surrounding noun phrase.
    rows.extend(normalize_token(m.group(0)) for m in TECHNICAL_SERIES_RE.finditer(q))

    # Attribute wording that commonly appears verbatim in table headers.
    attribute_patterns = (
        r"([0-9A-Za-z가-힣·/()\- ]{2,70}?(?:등급|강종|압력|온도|각도|질량|두께|지름|속도|거리|설치비율|분류번호|보호등급|시험전압|판정기준|허용기준|설계하중 시나리오))\s*(?:은|는|이|가|을|를)?\s*(?:몇|무엇|어느|어떤|얼마|$)",
        r"어떤\s+([0-9A-Za-z가-힣·/()\- ]{2,35}?방법)으로",
        r"어느\s+([0-9A-Za-z가-힣·/()\- ]{1,25}?(?:장|규정|위치))",
    )
    for pattern in attribute_patterns:
        for match in re.finditer(pattern, q):
            value = normalize_token(match.group(1)).strip(" ,")
            if value and len(value) <= 40:
                cols.append(value)

    for term in ATTRIBUTE_TERMS:
        if term in q:
            cols.append(term)

    if "자동 작동" in q:
        cols.append("자동 작동")
    if "평가" in q and any(t in q for t in ("방법", "손상모드", "대상")):
        cols.append("평가 방법")
    if "적용" in q and "규정" in q:
        cols.append("적용 규정")
    if "운항거리 제한" in q:
        cols.append("항해범위 제한부호")
    if "어디" in q and "표시" in q:
        cols.append("표시 장소")
    if "방화" in q and ("보존" in q or "등급" in q):
        cols.append("방화 보존성")
    if "시험재" in q and re.search(r"몇\s*개", q):
        cols.append("시험재의 수")
    # Cross-language KR-rule tables often use English row/column labels.  Keep
    # compact semantic slots so alias expansion can bridge the natural Korean
    # wording to those cells without knowing the source file or page.
    if "기관실 격벽" in q and "횡격벽" in q:
        rows.extend(["기관실 격벽", "engine room bulkhead"])
        cols.extend(["격벽 위치", "최전방 수밀 횡격벽"])
    if "적층제조" in q and "제조법 승인" in q:
        rows.extend(["적층제조 최종 재료", "AM 최종 재료", "AM 최종 재료에 대한 제조법 승인"])
        cols.extend(["제조법 승인 적용 장", "이 지침에서 적용되는 장 또는 하위 번호"])
    if "구명정" in q and "승정구역" in q:
        rows.extend(["구명정 승정 구역", "개방갑판", "임시 안전 대피 구역"])
    if re.search(r"ESP\s*[·./-]?\s*EXP", q, re.IGNORECASE) and re.search(
        r"Oil\s*/\s*Bulk\s*/\s*Ore\s+Carrier", q, re.IGNORECASE
    ):
        rows.extend(["Oil/Bulk/Ore Carrier 'ESP'(EXP)", "ESP EXP"])
        cols.extend(["Design", "설계 적용 장"])
    if "이중저 늑판" in q:
        rows.extend(["이중저 늑판", "Double bottom floors"])
    if "체인로커" in q and ("선수격벽" in q or "후방" in q or "뒤" in q):
        rows.extend(["체인로커(선수격벽 후방에 있는 경우)", "선수격벽 후방 체인로커"])
        cols.extend(["시험압력수두", "시험압력수두(m)"])
    if "Chemical Carrier" in q or "Chemical Tanker" in q:
        rows.extend(["Chemical Carrier", "Chemical Tanker"])
        if "운송화물명" in q or "화물명" in q:
            rows.extend(["운송화물명", "Name of Chemical primarily carried"])
        if "설계" in q and ("장" in q or "규정" in q):
            cols.extend(["Design", "설계 적용 장"])
    if "주요 지지부재" in q:
        rows.extend(["주요 지지부재", "Primary supporting members"])
        if "최종강도" in q:
            cols.extend(["최종강도 검토", "Ultimate strength check"])
    if "부식추가" in q or "부식 추가" in q:
        cols.extend(["부식추가", "tc1 또는 tc2", "corrosion addition"])
        for compartment in (
            "평형수탱크",
            "평형수 탱크",
            "빌지탱크",
            "드레인 저장탱크",
            "체인로커",
        ):
            if compartment.replace(" ", "") in q.replace(" ", ""):
                rows.append(compartment)
    if ("비선수미 격벽" in q or "선수미 격벽 이외" in q) and "외판" in q:
        rows.extend(["비선수미 격벽", "선수미 격벽이외의 격벽"])
        cols.extend(["외판으로부터 거리", "거리"])
    if "선급" in q and "표시" in q and "경보" in q:
        rows.extend([
            "표시·경보항목",
            "선급이 필요하다고 인정하는 항목",
            "기관에 따라 우리 선급이 필요하다고 인정하는 항목",
        ])
        cols.extend(["표시 장소", "표 시 장 소"])
    if "소선" in q and "지름" in q:
        rows.extend(["소선지름", "소선의 공칭지름", "소선 지름"])
        if "차이" in q or "허용" in q:
            cols.extend(["허용 차이", "최대인 것과 최소인 것의 차"])
    if "광물" in q and "함유" in q:
        rows.extend(["광물 함유량", "광물 함유", "적층용 수지"])
        if "공칭값" in q or "%" in q:
            cols.extend(["공칭값 오차", "요건"])
    if "안덮개" in q or "안 덮개" in q:
        rows.extend(["안덮개", "주갑판 아래" if "주갑판" in q else ""])
        if "비율" in q:
            cols.append("설치비율")
    if "창구" in q and "맨홀" in q:
        rows.extend([
            "창구, 맨홀",
            "창구, 맨홀 (덮개, 코밍 포함, 피팅류 제외)",
        ])
        if "2차방벽" in q.replace(" ", "") and "방벽간" in q.replace(" ", ""):
            cols.extend([
                "2차방벽 및 방벽간 구역",
                "2차 방벽 및 방벽간 구역",
            ])
    if ("용접" in q and ("다리" in q or "각장" in q)) or "최소 각장" in q:
        cols.extend(["최소 용접 다리 길이", "Minimum length, in mm", "최소 각장"])
    pump_match = re.search(
        r"([가-힣A-Za-z]+(?:\s+[가-힣A-Za-z]+){0,3})(?:은|는|이|가)\s*몇",
        q,
    )
    if pump_match:
        cols.append(normalize_token(pump_match.group(1)))
    return _dedupe(rows), _dedupe(cols)


def _short_domain_from_long_rows(rows: list[str]) -> list[str]:
    """Pull short domain anchors out of long subject phrases for schema overlap."""
    # Too-generic tails that match almost every tank/space table.
    skip = {"탱크", "구역", "갑판", "격벽"}
    extras: list[str] = []
    for row in rows:
        text = str(row or "")
        if len(text) < 12:
            continue
        for term in ROW_DOMAIN_TERMS:
            if term in skip:
                continue
            if term in text and term != text:
                extras.append(term)
        for m in re.finditer(
            r"[가-힣A-Za-z0-9]{2,}(?:탱크|거더|갑판|구역|격벽|외판|웨브)",
            text,
        ):
            tok = m.group(0)
            if tok not in skip and len(tok) >= 4:
                extras.append(tok)
    return extras


def _infer_row_entities(question: str) -> list[str]:
    rows: list[str] = []
    quoted = [normalize_token(v) for v in QUOTED_RE.findall(question)]
    if len(quoted) >= 2 and any(term in question for term in ("값", "해당", "행")):
        rows.append(quoted[0])
    for m in MATERIAL_GRADE_RE.finditer(question):
        rows.append(normalize_material_grade(m.group(0)))
        rows.append(normalize_token(m.group(0)))
    # Prefer longest domain hits so bare "탱크" does not fire inside "호퍼탱크".
    domain_hits = [term for term in ROW_DOMAIN_TERMS if term in question]
    domain_hits.sort(key=len, reverse=True)
    kept: list[str] = []
    for term in domain_hits:
        if any(term != other and term in other for other in domain_hits):
            # Skip strict substring of a longer matched domain term.
            continue
        kept.append(term)
    rows.extend(kept)
    return _dedupe(rows)


def _infer_table_topics(question: str, cols: list[str], rows: list[str]) -> list[str]:
    topics: list[str] = []
    q = question.lower()
    if any(t in question for t in CHEMISTRY_TERMS) or any(
        c in {"C", "S", "P", "MN", "SI"} for c in cols
    ):
        topics.extend(["화학성분", "chemical_composition"])
    if any(t in question for t in MECHANICAL_TERMS):
        topics.extend(["기계적성질", "mechanical_properties"])
    if any(t in question for t in INSPECTION_TERMS) or "reporting" in q:
        topics.extend(["정기검사", "inspection", "reporting"])
    if "선령" in question:
        topics.extend(["선령", "age_range"])
    if "열처리" in question or "로트" in question:
        topics.extend(["열처리", "lot_treatment"])
    if "용접" in question:
        topics.extend(["용접", "welding", "leg size", "minimum leg"])
    if "시험재" in question:
        topics.extend(["시험재료", "test_material", "test specimen"])
    if "치수" in question or "두께" in question:
        topics.extend(["치수", "dimension"])
    if "방화" in question or "보존성" in question:
        topics.extend(["방화", "fire integrity", "방화 보존성"])
    if "평가" in question and "방법" in question:
        topics.extend(["평가방법", "assessment method", "구조평가"])
    if "호퍼" in question or "이중선측" in question or "거더" in question:
        topics.extend(["구조요소", "structural member", "평가방법"])
    if any(t in question for t in NOTE_TERMS):
        topics.append("비고")
    if not topics and any(t in question for t in SUMMARY_TERMS):
        topics.append("table_overview")
    return _dedupe(topics)


def _infer_query_type(question: str, cols: list[str], rows: list[str], topics: list[str]) -> str:
    if any(t in question for t in NOTE_TERMS):
        return "note_lookup"
    if any(t in question for t in SUMMARY_TERMS) and not cols and not rows:
        return "table_lookup"
    if rows and not cols:
        return "row_lookup"
    if any(t in question for t in CONDITION_TERMS) and "선령" in question:
        return "condition_lookup"
    if cols and rows:
        return "cell_lookup"
    if cols:
        return "column_lookup" if "column_lookup" in QUERY_TYPES else "cell_lookup"
    if rows:
        return "row_lookup"
    if topics:
        return "table_lookup"
    return "cell_lookup"


def parse_table_query(question: str) -> ParsedTableQuery:
    q = normalize_token(question)
    natural_rows, natural_cols = _natural_slots(q)
    row_entities = _dedupe(_infer_row_entities(q) + natural_rows)
    row_entities = _dedupe(row_entities + _short_domain_from_long_rows(row_entities))
    column_entities = _dedupe(_infer_column_entities(q) + natural_cols)
    # Do not treat the C in a temperature unit (°C/℃) as the chemistry
    # column alias for carbon.  That false alias can redirect an otherwise
    # exact insulation-temperature lookup into chemical-composition tables.
    if re.search(r"(?:°\s*C|℃)", q, re.I):
        column_entities = [
            value for value in column_entities if str(value).strip().upper() != "C"
        ]
    table_topic_candidates = _infer_table_topics(q, column_entities, row_entities)
    unit_candidates = extract_units(q)
    condition_candidates = [t for t in CONDITION_TERMS if t in q]
    lexical_words = [
        t.strip("?.,()[]")
        for t in q.split()
        if len(t.strip("?.,()[]")) >= 2 and t.strip("?.,()[]") not in QUESTION_FILLERS
    ]
    keyword_terms = _dedupe(
        row_entities
        + column_entities
        + table_topic_candidates
        + lexical_words[:18]
    )
    # Bilingual aliases help schema/BM25 rank English KR-rule cells.
    alias_extra: list[str] = []
    for term in list(keyword_terms)[:24]:
        alias_extra.extend(expand_entity_aliases(term)[:8])
    keyword_terms = _dedupe(keyword_terms + alias_extra)[:40]
    query_type = _infer_query_type(q, column_entities, row_entities, table_topic_candidates)
    return ParsedTableQuery(
        raw_question=q,
        query_type=query_type,
        row_entities=row_entities,
        column_entities=column_entities,
        table_topic_candidates=table_topic_candidates,
        unit_candidates=unit_candidates,
        condition_candidates=condition_candidates,
        keyword_terms=keyword_terms,
        subject_candidates=natural_rows,
        attribute_candidates=natural_cols,
    )


def build_embed_query(parsed: ParsedTableQuery) -> str:
    alias_bits: list[str] = []
    for term in (
        list(parsed.row_entities)[:6]
        + list(parsed.column_entities)[:6]
        + list(parsed.table_topic_candidates)[:6]
    ):
        alias_bits.extend(expand_entity_aliases(term)[:6])
    parts = (
        parsed.table_topic_candidates[:6]
        + parsed.row_entities[:6]
        + parsed.column_entities[:6]
        + parsed.unit_candidates[:3]
        + alias_bits[:18]
        + [parsed.raw_question]
    )
    return " ".join(dict.fromkeys(p for p in parts if p)).strip()
