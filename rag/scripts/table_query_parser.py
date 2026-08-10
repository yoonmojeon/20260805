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
            else:
                if len(left.strip()) >= 2:
                    rows.append(left.strip())
                if 2 <= len(right) <= 80:
                    cols.append(right)
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
    pump_match = re.search(
        r"([가-힣A-Za-z]+(?:\s+[가-힣A-Za-z]+){0,3})(?:은|는|이|가)\s*몇",
        q,
    )
    if pump_match:
        cols.append(normalize_token(pump_match.group(1)))
    return _dedupe(rows), _dedupe(cols)


def _infer_row_entities(question: str) -> list[str]:
    rows: list[str] = []
    quoted = [normalize_token(v) for v in QUOTED_RE.findall(question)]
    if len(quoted) >= 2 and any(term in question for term in ("값", "해당", "행")):
        rows.append(quoted[0])
    for m in MATERIAL_GRADE_RE.finditer(question):
        rows.append(normalize_material_grade(m.group(0)))
        rows.append(normalize_token(m.group(0)))
    for term in ROW_DOMAIN_TERMS:
        if term in question:
            rows.append(term)
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
    if "용접" in question or "시험재" in question:
        topics.extend(["용접", "시험재료", "welding"])
    if "치수" in question or "두께" in question:
        topics.extend(["치수", "dimension"])
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
    column_entities = _dedupe(_infer_column_entities(q) + natural_cols)
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
    parts = (
        parsed.table_topic_candidates[:6]
        + parsed.row_entities[:6]
        + parsed.column_entities[:6]
        + parsed.unit_candidates[:3]
        + [parsed.raw_question]
    )
    return " ".join(dict.fromkeys(p for p in parts if p)).strip()
