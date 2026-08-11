"""Normalization and alias expansion for table entities (columns, rows, values)."""
from __future__ import annotations

import re
from typing import Iterable

# Extensible alias map: canonical key -> list of surface forms (lowercase for lookup).
ENTITY_ALIASES: dict[str, list[str]] = {
    "C": ["탄소", "carbon", "탄소(c)", "c"],
    "S": ["황", "sulfur", "황(s)", "s"],
    "P": ["인", "phosphorus", "인(p)", "p"],
    "MN": ["망간", "manganese", "망간(mn)", "mn"],
    "SI": ["규소", "silicon", "규소(si)", "si"],
    "NI": ["니켈", "nickel", "니켈(ni)", "ni"],
    "CR": ["크롬", "chromium", "크롬(cr)", "cr"],
    "MO": ["몰리브덴", "molybdenum", "몰리브덴(mo)", "mo"],
    "AL": ["알루미늄", "aluminum", "알루미늄(al)", "al"],
    "N": ["질소", "nitrogen", "질소(n)", "n"],
    "제1차정기검사": ["제1차 정기검사", "1차 정기검사", "1차 검사", "제1차검사", "first survey"],
    "제2차정기검사": ["제2차 정기검사", "2차 정기검사", "2차 검사"],
    "제3차정기검사": ["제3차 정기검사", "3차 정기검사", "3차 검사"],
    "제4차정기검사": ["제4차 및 이후 정기검사", "4차 정기검사", "4차 및 이후"],
    "화학성분": ["화학 성분", "chemical composition", "chemical_composition", "함량", "성분"],
    "기계적성질": ["기계적 성질", "mechanical properties", "항복", "인장", "연신", "충격"],
    "정기검사": ["inspection", "reporting", "검사차수", "검사"],
    "선령": ["age", "ship age", "선박연령"],
    # Open-table bilingual bridges (KR ask ↔ EN cell/caption).
    "화물창": ["cargo hold", "cargo hold region", "화물창 구역", "hold region"],
    "최소용접다리길이": [
        "최소 용접 다리 길이",
        "용접 다리",
        "최소 각장",
        "minimum length",
        "minimum leg",
        "leg size",
        "minimum leg size",
    ],
    "방화보존성": ["방화 보존성", "방화보존성", "fire integrity", "fire resistance", "방화"],
    "평가방법": ["평가 방법", "구조평가 방법", "assessment method", "evaluation method", "SP-A", "SP-B"],
    "호퍼탱크": ["호퍼 탱크", "hopper tank", "hopper"],
    "이중선측": ["이중 선측", "double side", "double-side"],
    "수평거더": ["수평 거더", "horizontal girder"],
    "구명정": ["lifeboat", "life boat"],
    "안전대피구역": ["임시 안전대피구역", "임시 안전 대피 구역", "temporary refuge", "refuge"],
    "기관실격벽": ["기관실 격벽", "engine room bulkhead"],
    "격벽위치": ["격벽 위치", "최전방 수밀 횡격벽", "foremost watertight transverse bulkhead"],
    "적층제조": ["additive manufacturing", "AM"],
    "적층제조최종재료": ["적층제조 최종 재료", "AM 최종 재료", "AM final material"],
    "제조법승인적용장": [
        "제조법 승인 적용 장",
        "이 지침에서 적용되는 장 또는 하위 번호",
        "applicable chapter or subsection",
    ],
    "구명정승정구역": ["구명정 승정구역", "구명정 승정 구역", "lifeboat embarkation area"],
    "개방갑판": ["open deck", "open-deck"],
    "OIL/BULK/ORE CARRIER": ["Oil/Bulk/Ore Carrier", "OBO carrier"],
    "ESP EXP": ["ESP·EXP", "ESP(EXP)", "'ESP'(EXP)"],
    "DESIGN": ["설계", "design"],
    "이중저늑판": ["이중저 늑판", "double bottom floors", "double bottom floor"],
    "체인로커": ["chain locker", "체인로커"],
    "선수격벽후방체인로커": [
        "선수격벽 뒤에 있는 체인로커",
        "체인로커(선수격벽 후방에 있는 경우)",
        "chain locker aft of collision bulkhead",
    ],
    "시험압력수두": ["시험압력수두", "시험압력수두(m)", "test pressure head"],
}

MATERIAL_GRADE_RE = re.compile(r"\b(AH|DH|EH|FH)\s*(\d{1,2})\b", re.IGNORECASE)
PAREN_SYMBOL_RE = re.compile(r"([가-힣A-Za-z]+)\s*\(([A-Za-z]{1,3})\)")
UNIT_RE = re.compile(
    r"(%|％|(?<![A-Za-z])(?:mm|cm|N/mm²|N/mm2|n/mm²|n/mm2|MPa|kN|°C)(?![A-Za-z])|"
    r"(?<![A-Za-z])m\b|℃|개|년)",
    re.IGNORECASE,
)

_ALIAS_LOOKUP: dict[str, str] | None = None


def _build_alias_lookup() -> dict[str, str]:
    global _ALIAS_LOOKUP
    # Rebuild when alias map grows (module reload / hot edit).
    expected = sum(1 + len(v) for v in ENTITY_ALIASES.values())
    if _ALIAS_LOOKUP is not None and len(_ALIAS_LOOKUP) >= expected:
        return _ALIAS_LOOKUP
    lookup: dict[str, str] = {}
    for canonical, forms in ENTITY_ALIASES.items():
        lookup[normalize_compact(canonical)] = canonical
        for form in forms:
            lookup[normalize_compact(form)] = canonical
    _ALIAS_LOOKUP = lookup
    return lookup


def normalize_compact(text: str) -> str:
    """Remove spaces, unify case for matching keys."""
    return re.sub(r"\s+", "", (text or "").strip()).upper()


def normalize_token(text: str) -> str:
    """Trim and collapse internal whitespace."""
    return re.sub(r"\s+", " ", (text or "").strip())


def normalize_material_grade(text: str) -> str:
    m = MATERIAL_GRADE_RE.search(text or "")
    if m:
        return f"{m.group(1).upper()}{m.group(2)}"
    return normalize_compact(text)


def expand_entity_aliases(text: str) -> list[str]:
    """Return original + compact + alias forms for an entity string."""
    forms: list[str] = []
    raw = normalize_token(text)
    if not raw:
        return forms
    forms.append(raw)
    compact = normalize_compact(raw)
    if compact and compact not in forms:
        forms.append(compact)
    lookup = _build_alias_lookup()
    if compact in lookup:
        canon = lookup[compact]
        forms.append(canon)
        forms.append(normalize_compact(canon))
        for form in ENTITY_ALIASES.get(canon, []):
            forms.append(form)
            forms.append(normalize_compact(form))
    # Substring alias hits for long natural phrases ("화물창 구역의 …").
    compact_lower = compact
    for canon, alias_forms in ENTITY_ALIASES.items():
        canon_c = normalize_compact(canon)
        # One/two-letter material symbols (C/S/P/N/MN/SI...) are exact aliases,
        # not safe substrings: otherwise CMS/System becomes a chemistry query.
        if len(canon_c) >= 3 and canon_c in compact_lower:
            forms.append(canon)
            forms.extend(alias_forms)
            continue
        for form in alias_forms:
            form_c = normalize_compact(form)
            short_ascii = form_c.isascii() and form_c.isalpha() and len(form_c) <= 4
            if form_c and len(form_c) >= 3 and not short_ascii and form_c in compact_lower:
                forms.append(canon)
                forms.extend(alias_forms)
                break
    for m in PAREN_SYMBOL_RE.finditer(raw):
        ko, sym = m.group(1), m.group(2).upper()
        forms.extend([ko, sym, f"{ko}({sym})", normalize_compact(f"{ko}{sym}")])
    return list(dict.fromkeys(f for f in forms if f))


def extract_paren_aliases(question: str) -> list[tuple[str, list[str]]]:
    """황(S) -> ('황', ['황','S','황(S)', ...])"""
    out: list[tuple[str, list[str]]] = []
    for m in PAREN_SYMBOL_RE.finditer(question):
        base = m.group(0)
        out.append((base, expand_entity_aliases(base)))
    return out


def extract_units(text: str) -> list[str]:
    found = list(dict.fromkeys(m.group(0) for m in UNIT_RE.finditer(text or "")))
    if "허용" in text or "한도" in text or "함량" in text:
        if "%" not in found and "％" not in found:
            found.append("%")
    return found


def split_value_and_unit(value: str) -> tuple[str, str]:
    s = normalize_token(value)
    m = UNIT_RE.search(s)
    if not m:
        return s, ""
    unit = m.group(0)
    num = s[: m.start()].strip()
    return num, unit


def entity_matches(query_forms: Iterable[str], target_blob: str) -> bool:
    blob = target_blob or ""
    blob_compact = normalize_compact(blob)
    blob_lower = blob.lower()
    for q in query_forms:
        if not q:
            continue
        qc = normalize_compact(q)
        if q in blob or qc in blob_compact:
            return True
        if len(q) >= 2 and q.lower() in blob_lower:
            return True
    return False


def best_entity_overlap(query_entities: list[str], target_entities: list[str]) -> float:
    """Fraction of query entities matched in target set (0..1)."""
    if not query_entities:
        return 0.0
    target_blob = " | ".join(target_entities)
    hits = sum(1 for e in query_entities if entity_matches(expand_entity_aliases(e), target_blob))
    return hits / len(query_entities)
