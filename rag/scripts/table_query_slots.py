"""Slot extraction for table-first retrieval (domain-agnostic heuristics)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MATERIAL_GRADE_RE = re.compile(
    r"\b(AH|DH|EH|FH)\s*(\d{1,2})\b",
    re.IGNORECASE,
)
MATERIAL_CODE_RE = re.compile(
    r"\b(RL\s*\d+[A-Z]?|RPV\s*\d+|RSB\s*\d+|RSBC\s*\d+|RBH\s*\d+)\b",
    re.IGNORECASE,
)
PLAIN_GRADE_RE = re.compile(r"\b([ABDE])\s*급\b")

CHEMICAL_ELEMENT_MAP = {
    "황": "S",
    "인": "P",
    "탄소": "C",
    "망간": "Mn",
    "규소": "Si",
    "니켈": "Ni",
    "크롬": "Cr",
    "몰리브덴": "Mo",
    "알루미늄": "Al",
    "질소": "N",
}

CHEMISTRY_MARKERS = (
    "화학",
    "함량",
    "허용",
    "한도",
    "이하",
    "이상",
    "성분",
    "탈산",
)
MECHANICAL_MARKERS = (
    "항복",
    "인장",
    "연신",
    "충격",
    "흡수에너지",
    "n/mm",
    "N/mm",
)
INSPECTION_MARKERS = (
    "정기검사",
    "reporting",
    "선령",
    "검사차수",
    "제1차",
    "제2차",
)
LOT_MARKERS = (
    "로트",
    "열처리",
    "tmcp",
    "노멀라이징",
    "압연 그대로",
)

CHEMISTRY_PENALIZE = (
    "시험재",
    "용접",
    "butt welding",
    "예열",
    "재료계수",
    "피로강도",
    "preheat",
    "용접 시험",
)
MECHANICAL_PENALIZE = (
    "정기검사",
    "reporting",
    "선령 구간",
)


@dataclass
class TableQuerySlots:
    material_grades: list[str] = field(default_factory=list)
    material_codes: list[str] = field(default_factory=list)
    chemical_elements: list[str] = field(default_factory=list)
    attribute_terms: list[str] = field(default_factory=list)
    intent: str = "general_table"
    penalize_terms: list[str] = field(default_factory=list)
    boost_terms: list[str] = field(default_factory=list)


def _norm_grade(prefix: str, num: str) -> str:
    return f"{prefix.upper()} {num}".strip()


def extract_table_query_slots(question: str) -> TableQuerySlots:
    q = question.strip()
    lower = q.lower()
    slots = TableQuerySlots()

    for m in MATERIAL_GRADE_RE.finditer(q):
        slots.material_grades.append(_norm_grade(m.group(1), m.group(2)))
    for m in MATERIAL_CODE_RE.finditer(q):
        slots.material_codes.append(re.sub(r"\s+", " ", m.group(1)).upper())
    for m in PLAIN_GRADE_RE.finditer(q):
        slots.material_codes.append(f"{m.group(1).upper()}급")

    for ko, sym in CHEMICAL_ELEMENT_MAP.items():
        if ko in q or f"({sym})" in q or re.search(rf"\b{sym}\b", q, re.I):
            if sym not in slots.chemical_elements:
                slots.chemical_elements.append(sym)
            if ko not in slots.attribute_terms:
                slots.attribute_terms.append(ko)

    # intent
    if any(m in q for m in INSPECTION_MARKERS) or "○" in q:
        slots.intent = "inspection"
        slots.boost_terms.extend(["정기검사", "reporting", "선령"])
    elif slots.chemical_elements or any(m in q for m in CHEMISTRY_MARKERS):
        slots.intent = "chemistry"
        slots.boost_terms.extend(["화학성분", "화학 성분", "탈산", "함량"])
        slots.penalize_terms.extend(CHEMISTRY_PENALIZE)
    elif any(m in q for m in MECHANICAL_MARKERS):
        slots.intent = "mechanical"
        slots.boost_terms.extend(["항복강도", "인장강도", "연신율", "충격시험", "기계적 성질"])
        slots.penalize_terms.extend(MECHANICAL_PENALIZE)
    elif any(m in lower for m in LOT_MARKERS):
        slots.intent = "lot_treatment"
        slots.boost_terms.extend(["열처리", "로트", "TMCP", "충격시험"])

    for grade in slots.material_grades:
        slots.boost_terms.append(grade)
    for code in slots.material_codes:
        slots.boost_terms.append(code)
    if "고장력강" in q or "고장력" in q:
        slots.boost_terms.append("고장력강")

    # de-dupe preserve order
    slots.boost_terms = list(dict.fromkeys(t for t in slots.boost_terms if t))
    slots.penalize_terms = list(dict.fromkeys(t for t in slots.penalize_terms if t))
    slots.material_grades = list(dict.fromkeys(slots.material_grades))
    slots.material_codes = list(dict.fromkeys(slots.material_codes))
    slots.chemical_elements = list(dict.fromkeys(slots.chemical_elements))
    return slots


def build_table_first_embed_query(question: str, slots: TableQuerySlots) -> str:
    """Enrich embedding query for table routing (not keyword gating)."""
    parts: list[str] = []
    parts.extend(slots.boost_terms[:12])
    if slots.intent == "chemistry":
        parts.extend(["화학성분 표", "재료기호"])
        parts.extend(slots.chemical_elements)
    elif slots.intent == "mechanical":
        parts.extend(["기계적 성질", "인장시험", "충격시험"])
    elif slots.intent == "inspection":
        parts.extend(["정기검사 reporting 표"])
    elif slots.intent == "lot_treatment":
        parts.extend(["열처리 충격시험 로트"])
    parts.append(question)
    return " ".join(dict.fromkeys(p for p in parts if p)).strip()
