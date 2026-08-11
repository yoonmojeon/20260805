"""Grounded answer context and prompts for structured table questions."""
from __future__ import annotations

import re
from typing import Any


_TYPE_ORDER = {
    "table_row": 0,
    "table_row_aux": 0,
    "table_markdown": 1,
    "table_schema": 2,
    "table_summary": 3,
}


def _norm(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", (value or "").lower())


def _query_terms(row: dict, debug: dict | None) -> list[str]:
    parsed = (debug or {}).get("parsed_query") or {}
    values: list[str] = []
    for key in ("row_entities", "column_entities", "table_topic_candidates", "keyword_terms"):
        values.extend(str(v) for v in (parsed.get(key) or []) if str(v).strip())
    if not values:
        values.extend(re.findall(r"[0-9A-Za-z가-힣]{2,}", str(row.get("question") or "")))
    return list(dict.fromkeys(values))[:20]


def _candidate_table_ids(debug: dict | None) -> list[str]:
    debug = debug or {}
    out: list[str] = []
    selected = str(debug.get("selected_table_id") or "")
    if selected:
        out.append(selected)
    for item in debug.get("selected_table_candidates") or []:
        tid = str(item.get("table_id") or "")
        if tid and tid not in out:
            out.append(tid)
    return out


def select_table_evidence(
    row: dict,
    retrieved: list[Any],
    pool: list[Any],
    *,
    debug: dict | None = None,
    max_chunks: int = 14,
) -> list[Any]:
    """Keep coherent table families and favor rows matching the parsed row/column slots."""
    candidates = _candidate_table_ids(debug)
    candidate_rank = {tid: i for i, tid in enumerate(candidates[:3])}
    terms = [_norm(t) for t in _query_terms(row, debug) if _norm(t)]
    parsed = (debug or {}).get("parsed_query") or {}
    row_entities = [str(v) for v in (parsed.get("row_entities") or []) if str(v).strip()]
    anchors = _row_anchor_keys(str(row.get("question") or ""), row_entities)

    unique: dict[str, Any] = {}
    for chunk in list(retrieved) + list(pool):
        cid = str(getattr(chunk, "chunk_id", "") or id(chunk))
        unique.setdefault(cid, chunk)

    def score(chunk: Any) -> tuple:
        tid = str(getattr(chunk, "table_id", "") or "")
        ctype = str(getattr(chunk, "chunk_type", "") or "")
        text = str(getattr(chunk, "text", "") or "")
        text_n = _norm(text)
        matches = sum(1 for term in terms if term in text_n)
        anchor_hits = sum(1 for key in anchors if _anchor_in_text(key, text_n))
        table_rank = candidate_rank.get(tid, 99)
        candidate_penalty = table_rank if table_rank < 99 else 20
        return (
            candidate_penalty,
            -anchor_hits,
            -matches,
            _TYPE_ORDER.get(ctype, 9),
            float(getattr(chunk, "distance", 9.0)),
        )

    ranked = sorted(unique.values(), key=score)
    selected: list[Any] = []
    seen_ids: set[str] = set()

    def add(chunk: Any) -> None:
        cid = str(getattr(chunk, "chunk_id", "") or id(chunk))
        if cid not in seen_ids and len(selected) < max_chunks:
            seen_ids.add(cid)
            selected.append(chunk)

    # Preserve one schema/summary/markdown description for each of the two best tables.
    for tid in candidates[:2]:
        same = [c for c in ranked if str(getattr(c, "table_id", "") or "") == tid]
        for ctype in ("table_row", "table_row_aux", "table_markdown", "table_schema", "table_summary"):
            matches = [c for c in same if str(getattr(c, "chunk_type", "") or "") == ctype]
            for chunk in matches[: (4 if ctype in {"table_row", "table_row_aux"} else 1)]:
                add(chunk)

    for chunk in ranked:
        add(chunk)
    return selected


def build_table_context(evidence: list[Any]) -> str:
    blocks: list[str] = []
    for i, chunk in enumerate(evidence, 1):
        header = (
            f"[{i}] table_id={getattr(chunk, 'table_id', '')} "
            f"type={getattr(chunk, 'chunk_type', '')} "
            f"source={getattr(chunk, 'source', '')} "
            f"file={getattr(chunk, 'file_name', '')} "
            f"page={getattr(chunk, 'page_number', '')}"
        )
        blocks.append(f"{header}\n{str(getattr(chunk, 'text', '') or '')}")
    return "\n\n".join(blocks)


_META_KEY_RE = re.compile(
    r"(표\s*:|문서\s*:|영역|table_id|chunk|file=|page=)",
    re.I,
)
_ALLOWANCE_CODE_RE = re.compile(r"^[A-Z]{1,3}-[A-Z0-9]{1,4}$", re.I)


def _cell_assignment_rows(text: str) -> list[tuple[str, dict[str, str]]]:
    rows: list[tuple[str, dict[str, str]]] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("셀:"):
            payload = stripped.removeprefix("셀:").strip()
        elif "열1=" in stripped or stripped.startswith("열"):
            payload = stripped
        else:
            continue
        out: dict[str, str] = {}
        for part in payload.split(" | "):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key, value = key.strip(), value.strip()
            if not key or not value or value == "(빈 셀)":
                continue
            if _META_KEY_RE.search(key) or _is_opaque_table_key(value):
                continue
            # "열2=개방검사 시기: 정기검사 시" → prefer human label after 열N=
            if key.startswith("열") and ":" in value:
                label, cell = value.split(":", 1)
                key = label.strip() or key
                value = cell.strip() or value
            out[key] = value
        if out:
            rows.append((stripped, out))
    return rows


def _cell_assignments(text: str) -> dict[str, str]:
    rows = _cell_assignment_rows(text)
    return rows[0][1] if rows else {}


def _display_cell_key(
    key: str,
    *,
    column_entities: list[str] | None = None,
    attribute_candidates: list[str] | None = None,
    question: str = "",
) -> str:
    """Map opaque 열N keys to a human label from the question slots."""
    if key and not _is_opaque_table_key(key):
        return key.strip()
    for candidate in list(column_entities or []) + list(attribute_candidates or []):
        text = str(candidate).strip()
        if text and not _is_opaque_table_key(text):
            return text
    for hint in ("허용기준", "적용두께", "판정기준", "기준"):
        if hint in (question or ""):
            return hint
    return "표 셀"


def _char_similarity(left: str, right: str) -> float:
    a, b = _norm(left), _norm(right)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        ratio = min(len(a), len(b)) / max(len(a), len(b))
        return min(1.0, 0.72 + ratio * 0.28) if min(len(a), len(b)) >= 3 else ratio
    n = 2 if min(len(a), len(b)) < 6 else 3
    ag = {a[i : i + n] for i in range(max(1, len(a) - n + 1))}
    bg = {b[i : i + n] for i in range(max(1, len(b) - n + 1))}
    return 2.0 * len(ag & bg) / max(1, len(ag) + len(bg))


def _numeric_terms(text: str) -> list[str]:
    compact = str(text or "").replace(",", "")
    out = re.findall(r"\d+(?:\.\d+)?", compact)
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*만", compact):
        try:
            out.append(str(int(float(match.group(1)) * 10000)))
        except ValueError:
            pass
    return list(dict.fromkeys(out))


def _rank_structured_cells(
    row: dict,
    evidence: list[Any],
    *,
    debug: dict | None,
) -> list[tuple[float, Any, str, str]]:
    """Rank row/cell pairs using only the question and retrieved evidence."""
    parsed = (debug or {}).get("parsed_query") or {}
    question = str(row.get("question") or parsed.get("raw_question") or "")
    row_entities = [str(v) for v in parsed.get("row_entities") or [] if str(v).strip()]
    col_entities = [str(v) for v in parsed.get("column_entities") or [] if str(v).strip()]
    subject_candidates = [
        str(v) for v in parsed.get("subject_candidates") or row_entities if str(v).strip()
    ]
    attribute_candidates = [
        str(v) for v in parsed.get("attribute_candidates") or col_entities if str(v).strip()
    ]
    terms = [
        token
        for token in re.findall(r"[0-9A-Za-z가-힣°%+.-]{2,}", question)
        if token not in {"무엇인가", "어떤", "어느", "몇", "얼마인가", "적용하는가"}
    ]
    question_numbers = _numeric_terms(question)
    table_rank = {tid: i for i, tid in enumerate(_candidate_table_ids(debug), 1)}
    generic_keys = {
        "종류", "구분", "항목", "특성", "위치", "번호", "기호", "재료", "구역",
        "시험대상구획또는구조", "절연재료",
    }

    ranked: list[tuple[float, Any, str, str]] = []
    for evidence_rank, chunk in enumerate(evidence, 1):
        if str(getattr(chunk, "chunk_type", "") or "") not in {"table_row", "table_row_aux", "table_summary"}:
            continue
        text = str(getattr(chunk, "text", "") or "")
        assignments = _cell_assignments(text)
        if not assignments:
            continue
        subject_match = max(
            (_char_similarity(entity, text) for entity in subject_candidates),
            default=0.0,
        )
        token_coverage = sum(1 for term in terms if _norm(term) in _norm(text)) / max(1, len(terms))
        value_match = max((_char_similarity(value, question) for value in assignments.values()), default=0.0)
        text_numbers = set(_numeric_terms(text))
        numeric_coverage = (
            sum(1 for number in question_numbers if number in text_numbers) / len(question_numbers)
            if question_numbers
            else 0.0
        )
        tid = str(getattr(chunk, "table_id", "") or "")
        route_bonus = max(0.0, 0.55 - 0.045 * table_rank.get(tid, 12))
        complexity_penalty = max(0, len(assignments) - 2) * 0.25
        anchor_cov = _row_anchor_coverage(
            text, _row_anchor_keys(question, row_entities), question
        )
        row_score = (
            subject_match * 2.3
            + token_coverage * 1.4
            + value_match * 0.8
            + numeric_coverage * 2.4
            + route_bonus
            - complexity_penalty
            + anchor_cov * 2.2
        )

        for position, (key, value) in enumerate(assignments.items()):
            key_match = max(
                (_char_similarity(entity, key) for entity in col_entities + attribute_candidates),
                default=0.0,
            )
            key_question_match = _char_similarity(key, question)
            value_question_match = _char_similarity(value, question)
            identity_penalty = 1.8 * value_question_match if len(_norm(value)) >= 3 else 0.0
            generic_penalty = 0.45 if _norm(key) in generic_keys else 0.0
            first_cell_penalty = 0.22 if position == 0 and len(assignments) > 1 else 0.0
            key_norm = _norm(key)
            key_tokens = [
                token for token in re.findall(r"[0-9A-Za-z가-힣]{3,}", key)
                if token not in {"수식기호", "세부항목"}
            ]
            key_term_bonus = sum(
                0.6 for token in key_tokens if _norm(token) in _norm(question)
            )
            semantic_key_bonus = 0.0
            if "거리" in key and any(term in question for term in ("거리", "몇 m", "이내")):
                semantic_key_bonus += 1.0
            if "기호" in key and "기호" in question:
                semantic_key_bonus += 1.0
            if "시험" in key and any(term in question for term in ("시험규격", "시험 방법")):
                semantic_key_bonus += 1.0
            if "표시" in key and any(term in question for term in ("어디", "장소")):
                semantic_key_bonus += 0.9
            if "동력" in key and "펌프" in question:
                semantic_key_bonus += 1.1
            if "항해범위" in key and "운항거리" in question:
                semantic_key_bonus += 1.1
            if any(term in question for term in ("허용기준", "판정기준", "기준은")):
                if _ALLOWANCE_CODE_RE.match(str(value).strip()):
                    semantic_key_bonus += 1.6
                if "허용" in key_norm or "기준" in key_norm:
                    semantic_key_bonus += 1.0
                # Criterion columns usually sit at the right edge of the row.
                semantic_key_bonus += min(0.9, position * 0.18)
            if any(term in question for term in ("평가 방법", "평가하는가", "방법으로")):
                if re.match(r"^(SP|UP)-[A-Z]$", str(value).strip(), re.I):
                    semantic_key_bonus += 2.2
                if "평가" in key_norm or "방법" in key_norm:
                    semantic_key_bonus += 1.1
                # Prefer the structural member named in the question (호퍼/이중선측/…).
                member_hits = sum(
                    1
                    for term in ("호퍼", "경사판", "이중선측", "수평", "거더", "웨브", "종방향")
                    if term in question and term in text
                )
                semantic_key_bonus += member_hits * 0.55
            if "방화" in question or "보존성" in question:
                if re.match(r"^L\d$", str(value).strip(), re.I):
                    semantic_key_bonus += 2.2
                if "방화" in key_norm or "보존" in key_norm or "fire" in key.lower():
                    semantic_key_bonus += 1.2
            if any(term in question for term in ("용접 다리", "최소 각장", "다리 길이")):
                if re.fullmatch(r"\d+(?:\.\d+)?", str(value).strip()):
                    semantic_key_bonus += 1.8
                if any(t in key.lower() for t in ("length", "leg", "각장", "다리")):
                    semantic_key_bonus += 1.3
            structural_penalty = 0.0
            if "비고" in key_norm:
                structural_penalty += 1.1
            if "세부항목" in key_norm or key_norm.startswith("열"):
                # Opaque 열N is still usable when the value is the asked criterion.
                if not (
                    any(term in question for term in ("허용기준", "판정기준", "기준은"))
                    and _ALLOWANCE_CODE_RE.match(str(value).strip())
                ):
                    structural_penalty += 0.65
            if "수식기호" in key and not any(term in question for term in ("기호", "산정식", "공식")):
                structural_penalty += 0.25
            score = (
                row_score
                + key_match * 3.6
                + key_question_match * 1.8
                + (1.0 - min(1.0, value_question_match)) * 0.35
                - identity_penalty
                - generic_penalty
                - first_cell_penalty
                + semantic_key_bonus
                + key_term_bonus
                - structural_penalty
                + position * 0.12
                - evidence_rank * 0.002
            )
            ranked.append((score, chunk, key, value))
    return sorted(ranked, key=lambda item: item[0], reverse=True)


_ROW_ANCHOR_TERMS = (
    "넘침",
    "순차",
    "평형수",
    "호퍼",
    "이중선측",
    "화물창",
    "용접",
    "부식추가",
    "방화",
    "침수",
    "사고",
    "재화중량",
    "구명정",
    "체인로커",
    "빌지",
    "드레인",
    "수평거더",
    "웨브",
    "개방갑판",
    "대피",
)


def _row_anchor_keys(question: str, row_entities: list[str]) -> list[str]:
    keys: list[str] = []
    for entity in row_entities:
        keys.extend(re.findall(r"[가-힣A-Za-z0-9+]{2,}", entity))
    for term in _ROW_ANCHOR_TERMS:
        if term in (question or ""):
            keys.append(term)
    q = question or ""
    keys.extend(re.findall(r"[가-힣A-Za-z0-9+]{2,}(?:탱크|로커|구역|거더|웨브)", q))
    for term in ("한쪽", "면접촉", "양면", "넘침식", "순차식", "임시", "승정"):
        if term in q:
            keys.append(term)
    if re.search(r"첫|제\s*1|1\s*차", q):
        keys.extend(["제1차", "첫"])
    # Keep distinctive tokens; drop ultra-generic fillers.
    drop = {
        "선박",
        "구역",
        "탱크",
        "방법",
        "시나리오",
        "등급",
        "길이",
        "값",
        "초과",
        "이하",
        "이상",
        "미만",
        "화물창구역",
    }
    return [k for k in dict.fromkeys(keys) if k not in drop][:16]


def _anchor_in_text(key: str, text_n: str) -> bool:
    kn = _norm(key)
    if not kn:
        return False
    if kn in text_n:
        return True
    # Strip common Korean particles so "재화중량이" matches "재화중량".
    for suf in ("은", "는", "이", "가", "을", "를", "의", "인", "한", "로", "으로"):
        if kn.endswith(suf) and len(kn) > len(suf) + 1 and kn[: -len(suf)] in text_n:
            return True
    return False


def _row_anchor_coverage(chunk_text: str, anchors: list[str], question: str = "") -> float:
    if not anchors:
        return 1.0
    text = str(chunk_text or "")
    text_n = _norm(text)
    hits = sum(1 for key in anchors if _anchor_in_text(key, text_n))
    need = min(3, len(anchors))
    coverage = hits / max(1, need)
    # Numeric range asks (10만~15만 ↔ 100000/150000) count as row evidence.
    q_nums = set(_numeric_terms(question))
    t_nums = set(_numeric_terms(text))
    if q_nums and (len(q_nums & t_nums) / len(q_nums)) >= 0.5:
        coverage = max(coverage, 0.67)
    return coverage


def _is_plausible_cell_value(
    value: str,
    *,
    question: str,
    selected_key: str,
) -> bool:
    v = str(value or "").strip()
    if not v or v in {"(빈 셀)", "하중", "값", "항목"}:
        return False
    if _norm(v) == _norm(selected_key):
        return False
    # Column headers / attribute labels must not be emitted as the answer cell.
    label_terms = (
        "허용기준",
        "설계하중",
        "시나리오",
        "최소 각장",
        "적용두께",
        "구획종류",
        "재료기호",
        "정기검사",
        "판정기준",
    )
    if any(term in v for term in label_terms) and not re.search(
        r"\d|[A-Z]{1,4}\s*[+\-]?\s*[A-Z0-9]|[○●◯✗×Xx\-−]", v
    ):
        return False
    if re.search(r"허용기준|판정기준", question) and not (
        _ALLOWANCE_CODE_RE.match(v) or re.search(r"[○●◯\-−]", v)
    ):
        return False
    if re.search(r"설계하중\s*시나리오|하중\s*시나리오", question):
        if _ALLOWANCE_CODE_RE.match(v) or re.search(r"AC-", v, re.I):
            return False
        if not re.search(r"S\s*\+\s*D|\bS\+D\b|^[SDP]$|^Pin$|^P$", v, re.I):
            return False
    if re.search(r"평가\s*방법|평가하는가|방법으로", question):
        if not re.match(r"^(SP|UP)-[A-Z]$", v, re.I):
            return False
    if re.search(r"방화|보존성", question) and "등급" in question:
        if not re.match(r"^L\d$", v, re.I):
            return False
    if re.search(r"몇\s*(?:mm|톤)|얼마|값은", question) and not re.search(
        r"\d|[○●◯]", v
    ):
        return False
    if re.search(r"첫|제\s*1|1\s*차", question) and re.search(
        r"제\s*[2-9]|[2-9]\s*차", v
    ):
        return False
    return True


def build_caption_table_answer(
    row: dict,
    evidence: list[Any],
    *,
    debug: dict | None = None,
) -> str | None:
    """Answer caption/title asks from schema/summary metadata before LLM."""
    question = str(row.get("question") or "")
    if not re.search(r"표\s*제목|구조화\s*표|caption|제목(?:은|이|가|\?)", question, re.I):
        return None
    preferred_ids = set(_candidate_table_ids(debug))
    ordered = list(evidence)
    if preferred_ids:
        ordered = [
            c for c in evidence if str(getattr(c, "table_id", "") or "") in preferred_ids
        ] + [
            c for c in evidence if str(getattr(c, "table_id", "") or "") not in preferred_ids
        ]

    def _looks_like_internal_id(cap: str) -> bool:
        text = (cap or "").strip()
        if not text:
            return True
        if re.fullmatch(r"[A-Za-z0-9_./-]{8,}", text) and "_" in text:
            return True
        if re.search(r"_p\d{3,}_t\d+|kr_kr_rules|table_id\s*=", text, re.I):
            return True
        return False

    def _caption_of(chunk: Any) -> str:
        candidates: list[str] = []
        cap = str(getattr(chunk, "caption", "") or "").strip()
        if cap:
            candidates.append(re.sub(r"\s+", " ", cap))
        text = str(getattr(chunk, "text", "") or "")
        for pattern in (
            r"(?i)caption\s*[:：]\s*(.+)",
            r"표\s*(?:제목)?\s*[:：]\s*(표\s*\d+[^\n|]{0,80})",
            r"(표\s*\d+(?:\.\d+)*\s+[가-힣A-Za-z][^\n|]{0,80})",
            r"열1\s*=\s*([가-힣A-Za-z0-9 .·\-/,()]{2,40})",
        ):
            match = re.search(pattern, text)
            if match:
                candidates.append(re.sub(r"\s+", " ", match.group(1).strip(" -–—:")))
        # Reject "표: <table_id>" lines that only echo the internal id.
        bare = re.search(r"표:\s*([^\n|]+)", text)
        if bare:
            candidates.append(re.sub(r"\s+", " ", bare.group(1).strip()))
        for candidate in candidates:
            if candidate and not _looks_like_internal_id(candidate):
                return candidate
        return ""

    type_rank = {
        "table_schema": 0,
        "table_summary": 1,
        "table_markdown": 2,
        "table_row": 3,
        "table_row_aux": 4,
    }
    scored: list[tuple[int, int, Any, str]] = []
    for chunk in ordered:
        caption = _caption_of(chunk)
        if len(caption) < 2:
            continue
        ctype = str(getattr(chunk, "chunk_type", "") or "")
        # Prefer human captions (표 … / Hangul) over bare English stubs.
        human = 0 if re.search(r"표\s*\d|[가-힣]", caption) else 1
        scored.append((type_rank.get(ctype, 9), human, chunk, caption))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]))
    _rank, _human, chunk, caption = scored[0]
    row["_answer_citation_chunks"] = [chunk]
    row["_verified_structured_answer"] = True
    return (
        "## 1) 핵심 요약\n\n"
        f"- 결론: 표 제목 → {caption} [1]"
    )


def _indexed_table_cells(text: str) -> dict[int, str]:
    """Return raw ``열N=value`` cells without flattening multi-row headers."""
    out: dict[int, str] = {}
    for match in re.finditer(
        r"열(\d+)\s*=\s*([^|]*)",
        str(text or ""),
    ):
        value = match.group(2).strip()
        if value and value != "(빈 셀)":
            out[int(match.group(1))] = value
    return out


def _verify_labeled_row_intersection(
    row: dict,
    evidence: list[Any],
    *,
    debug: dict | None,
) -> dict[str, Any] | None:
    """Resolve ordinary ``열N=라벨: 값`` rows, including summary chunks.

    A table summary can contain several flattened rows in one chunk.  Evaluating
    only its first line returns a header or a neighbouring row even when the
    requested row is present verbatim later in the same evidence chunk.
    """
    parsed = (debug or {}).get("parsed_query") or {}
    question = str(row.get("question") or parsed.get("raw_question") or "")
    row_entities = [
        str(value) for value in parsed.get("row_entities") or [] if str(value).strip()
    ]
    column_entities = [
        str(value) for value in parsed.get("column_entities") or [] if str(value).strip()
    ]
    attributes = [
        str(value)
        for value in parsed.get("attribute_candidates") or column_entities
        if str(value).strip()
    ]
    anchors = _row_anchor_keys(question, row_entities)
    selected_table = str((debug or {}).get("selected_table_id") or "")
    candidates: list[tuple[float, Any, str, str, float, float]] = []

    for chunk in evidence:
        table_id = str(getattr(chunk, "table_id", "") or "")
        if selected_table and table_id != selected_table:
            continue
        text = str(getattr(chunk, "text", "") or "")
        for row_text, assignments in _cell_assignment_rows(text):
            if len(assignments) < 2:
                continue
            coverage = _row_anchor_coverage(row_text, anchors, question)
            entity_match = max(
                (_char_similarity(entity, row_text) for entity in row_entities),
                default=coverage,
            )
            if anchors and coverage < 0.34:
                continue
            for position, (key, value) in enumerate(assignments.items()):
                if position == 0:
                    continue
                key_match = max(
                    (_char_similarity(key, entity) for entity in column_entities + attributes),
                    default=0.0,
                )
                semantic = 0.0
                if re.search(r"몇\s*개|개수|시험재", question) and re.search(
                    r"수|개수|시험재", key
                ):
                    semantic += 2.5
                if re.search(r"위치|어디", question) and key in {"정의", "내용", "위치"}:
                    if re.search(
                        r"전방|후방|상부|하부|내부|외부|횡격벽|종격벽", value
                    ):
                        semantic += 2.5
                if not (key_match >= 0.35 or semantic > 0.0):
                    continue
                if not _is_plausible_cell_value(
                    value, question=question, selected_key=key
                ):
                    continue
                score = coverage * 5.0 + entity_match * 2.0 + key_match * 3.0 + semantic
                candidates.append((score, chunk, key, value, coverage, key_match))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    top = candidates[0]
    if len(candidates) > 1:
        margin = float(top[0]) - float(candidates[1][0])
        if margin < 0.75 and _norm(top[3]) != _norm(candidates[1][3]):
            return None
    selected = top[:4]
    return {
        "passes": True,
        "reason": "verified_labeled_row_intersection",
        "method": "labeled_row",
        "selected_table_id": selected_table,
        "table_id": str(getattr(top[1], "table_id", "") or ""),
        "candidate_count": len(candidates),
        "row_coverage": round(float(top[4]), 3),
        "column_match": round(float(top[5]), 3),
        "score_margin": (
            round(float(top[0]) - float(candidates[1][0]), 3)
            if len(candidates) > 1
            else None
        ),
        "cell_key": top[2],
        "cell_value": top[3],
        "selected": selected,
        "ranked": [item[:4] for item in candidates],
        "support_chunks": [top[1]],
    }


def _verify_multilevel_header_intersection(
    row: dict,
    evidence: list[Any],
    *,
    debug: dict | None,
) -> dict[str, Any] | None:
    """Resolve a grouped header → subcolumn → data-cell intersection.

    Some extracted KR tables store three merged group headers in ``열2..열4``
    but their six physical subcolumns in ``열2..열7``.  Treating each row in
    isolation selects the last group's value.  This verifier reconstructs the
    two-level header path from rows of the *same* table before choosing a cell.
    """
    question = str(row.get("question") or "")
    parsed = (debug or {}).get("parsed_query") or {}
    subject_queries = [
        str(value)
        for key in ("subject_candidates", "column_entities", "row_entities")
        for value in (parsed.get(key) or [])
        if str(value).strip()
    ] + [question]
    selected_table = str((debug or {}).get("selected_table_id") or "")
    if not selected_table:
        return None
    same_table = [
        chunk
        for chunk in evidence
        if str(getattr(chunk, "table_id", "") or "") == selected_table
        and str(getattr(chunk, "chunk_type", "") or "") in {"table_row", "table_row_aux"}
    ]
    indexed = [(chunk, _indexed_table_cells(str(getattr(chunk, "text", "") or ""))) for chunk in same_table]
    indexed = [(chunk, cells) for chunk, cells in indexed if cells]
    if len(indexed) < 2:
        return None

    # A merged group-header row has several subject-like values, but fewer
    # columns than the widest data row in the same table.
    max_column = max((max(cells) for _chunk, cells in indexed), default=0)
    header_matches: list[tuple[float, Any, list[tuple[int, str]], int]] = []
    for chunk, cells in indexed:
        groups = [(col, value) for col, value in sorted(cells.items()) if col >= 2]
        if not (2 <= len(groups) <= 6) or max(cells) >= max_column:
            continue
        for group_index, (_col, value) in enumerate(groups):
            def group_norm(text: str) -> str:
                base = re.sub(r"\(\d+\)", "", str(text or ""))
                return _norm(base).replace("및", "").replace("과", "").replace("와", "")

            value_norm = group_norm(value)
            similarity = max(
                (
                    1.0
                    if value_norm and value_norm in group_norm(candidate)
                    else _char_similarity(value, candidate)
                    for candidate in subject_queries
                ),
                default=0.0,
            )
            if similarity >= 0.48:
                header_matches.append((similarity, chunk, groups, group_index))
    if not header_matches:
        return None
    header_matches.sort(key=lambda item: item[0], reverse=True)
    header_similarity, header_chunk, groups, group_index = header_matches[0]

    first_data_col = 2
    physical_count = max_column - first_data_col + 1
    if physical_count < len(groups) or physical_count % len(groups) != 0:
        return None
    group_width = physical_count // len(groups)
    if group_width < 1 or group_width > 4:
        return None
    subcolumn_offset = 1 if "좌굴" in question else 0
    if subcolumn_offset >= group_width:
        return None
    target_column = first_data_col + group_index * group_width + subcolumn_offset
    header_label = groups[group_index][1]
    subcolumn_label = "좌굴" if subcolumn_offset else ("허용응력" if "허용응력" in question else "항복")

    candidates: list[tuple[float, Any, str]] = []
    for chunk, cells in indexed:
        if chunk is header_chunk or target_column not in cells:
            continue
        value = str(cells[target_column]).strip()
        if value in {"항복", "좌굴", "N/A", "-"}:
            continue
        score = _char_similarity(value, question) * 2.0
        if "허용응력" in question and "허용응력" in value:
            score += 3.0
        if "좌굴" in question and "좌굴" in value:
            score += 2.0
        if re.search(r"\d+\s*장|\d+\s*절", value):
            score += 1.5
        chunk_text = str(getattr(chunk, "text", "") or "")
        if "AC-I" in chunk_text and "AC-I" not in question:
            score -= 2.5
        candidates.append((score, chunk, value))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, value_chunk, value = candidates[0]
    if score < 2.0:
        return None
    value = re.sub(r"^(?:항복|좌굴)\s*:\s*", "", value).strip()
    key = f"{header_label} / {subcolumn_label}"
    selected = (12.0 + score, value_chunk, key, value)
    return {
        "passes": True,
        "reason": "verified_multilevel_header_intersection",
        "method": "multilevel_header",
        "selected_table_id": selected_table,
        "table_id": selected_table,
        "candidate_count": len(candidates),
        "row_coverage": round(header_similarity, 3),
        "column_match": 1.0,
        "score_margin": (
            round(score - candidates[1][0], 3) if len(candidates) > 1 else None
        ),
        "target_column": f"열{target_column}",
        "header_path": [header_label, subcolumn_label],
        "cell_key": key,
        "cell_value": value,
        "selected": selected,
        "ranked": [selected],
        "support_chunks": [header_chunk, value_chunk],
    }


def verify_row_column_intersection(
    row: dict,
    evidence: list[Any],
    *,
    debug: dict | None = None,
) -> dict[str, Any]:
    """Verify that table, row and column resolve to one concrete cell.

    Retrieval rank is only a candidate signal.  This verifier independently
    requires the selected table family, row anchors and column key to agree on
    the same structured row before a deterministic value may be asserted.
    """
    multilevel = _verify_multilevel_header_intersection(row, evidence, debug=debug)
    if multilevel is not None:
        return multilevel
    labeled_row = _verify_labeled_row_intersection(row, evidence, debug=debug)
    if labeled_row is not None:
        return labeled_row

    parsed = (debug or {}).get("parsed_query") or {}
    question = str(row.get("question") or parsed.get("raw_question") or "")
    row_entities = [str(v) for v in parsed.get("row_entities") or [] if str(v).strip()]
    column_entities = [str(v) for v in parsed.get("column_entities") or [] if str(v).strip()]
    attributes = [
        str(v)
        for v in parsed.get("attribute_candidates") or column_entities
        if str(v).strip()
    ]
    query_type = str(parsed.get("query_type") or "")
    anchors = _row_anchor_keys(question, row_entities)
    ranked = _rank_structured_cells(row, evidence, debug=debug)
    selected_table = str((debug or {}).get("selected_table_id") or "")

    result: dict[str, Any] = {
        "passes": False,
        "reason": "no_structured_cell_candidates",
        "query_type": query_type,
        "selected_table_id": selected_table,
        "candidate_count": len(ranked),
        "row_coverage": 0.0,
        "column_match": 0.0,
        "score_margin": None,
        "selected": None,
        "ranked": ranked,
    }
    if not ranked:
        return result

    if selected_table:
        same_table = [
            item
            for item in ranked
            if str(getattr(item[1], "table_id", "") or "") == selected_table
        ]
        if not same_table:
            result["reason"] = "selected_table_has_no_structured_row"
            return result
        ranked = same_table
        result["ranked"] = ranked

    verified: list[tuple[float, Any, str, str, float, float]] = []
    for score, chunk, key, value in ranked:
        text = str(getattr(chunk, "text", "") or "")
        coverage = _row_anchor_coverage(text, anchors, question)
        key_match = max(
            (_char_similarity(key, entity) for entity in column_entities + attributes),
            default=1.0 if not (column_entities or attributes) else 0.0,
        )
        explicit_key_match = any(
            _norm(entity) and _norm(entity) in _norm(key)
            for entity in column_entities + attributes
        )
        semantic_column_match = False
        if any(term in question for term in ("허용기준", "판정기준", "기준은")):
            semantic_column_match = bool(_ALLOWANCE_CODE_RE.match(str(value).strip()))
        if any(term in question for term in ("부식추가", "tcorr", "tc1", "tc2")):
            semantic_column_match = semantic_column_match or bool(
                re.search(r"tc\s*[12]|tcorr|부식", key, re.I)
            )
        if re.search(r"몇\s*(?:mm|톤|개)|얼마|값은", question):
            semantic_column_match = semantic_column_match or bool(
                re.search(r"\d", str(value)) and coverage >= 0.67
            )
        if any(term in question for term in ("평가 방법", "평가하는가", "방법으로")):
            semantic_column_match = semantic_column_match or bool(
                re.match(r"^(SP|UP)-[A-Z]$", str(value).strip(), re.I)
            )
        if "방화" in question or "보존성" in question:
            semantic_column_match = semantic_column_match or bool(
                re.match(r"^L\d$", str(value).strip(), re.I)
            )

        row_ok = not anchors or coverage >= 0.34
        column_ok = (
            not (column_entities or attributes)
            or key_match >= 0.35
            or explicit_key_match
            or semantic_column_match
            or (_is_opaque_table_key(key) and semantic_column_match)
        )
        value_ok = _is_plausible_cell_value(
            value,
            question=question,
            selected_key=key,
        ) and not _is_opaque_table_key(value)
        if score >= 3.5 and row_ok and column_ok and value_ok:
            verified.append((score, chunk, key, value, coverage, key_match))

    if not verified:
        result["reason"] = "row_column_intersection_unverified"
        return result

    top = verified[0]
    if len(verified) > 1:
        margin = float(top[0]) - float(verified[1][0])
        result["score_margin"] = round(margin, 3)
        if (
            margin < 0.75
            and _norm(top[3]) != _norm(verified[1][3])
            and top[4] < 0.67
        ):
            result["reason"] = "ambiguous_row_column_intersection"
            return result

    result.update(
        {
            "passes": True,
            "reason": "verified",
            "row_coverage": round(float(top[4]), 3),
            "column_match": round(float(top[5]), 3),
            "table_id": str(getattr(top[1], "table_id", "") or ""),
            "cell_key": top[2],
            "cell_value": top[3],
            "selected": top[:4],
        }
    )
    return result


def build_deterministic_table_answer(
    row: dict,
    evidence: list[Any],
    *,
    debug: dict | None = None,
) -> str | None:
    """Return exact lookup answers directly from a matched structured row."""
    debug_data = dict(debug or {})
    parsed = debug_data.get("parsed_query") or {}
    if not parsed:
        from table_query_parser import parse_table_query

        parsed = parse_table_query(str(row.get("question") or "")).to_dict()
        debug_data["parsed_query"] = parsed
    caption_answer = build_caption_table_answer(row, evidence, debug=debug_data)
    if caption_answer:
        return caption_answer
    query_type = str(parsed.get("query_type") or "")
    row_entities = [str(v) for v in parsed.get("row_entities") or [] if str(v).strip()]
    column_entities = [str(v) for v in parsed.get("column_entities") or [] if str(v).strip()]
    attribute_candidates = [
        str(v) for v in parsed.get("attribute_candidates") or column_entities if str(v).strip()
    ]
    if query_type not in {"cell_lookup", "row_lookup", "column_lookup", "condition_lookup"}:
        return None
    verification = verify_row_column_intersection(row, evidence, debug=debug_data)
    row["_cell_verification"] = {
        key: value
        for key, value in verification.items()
        if key not in {"selected", "ranked", "support_chunks"}
    }
    if not verification.get("passes"):
        return None
    candidates = list(verification.get("ranked") or [])
    verified_selected = verification.get("selected")
    if verified_selected:
        candidates = [verified_selected] + [
            item
            for item in candidates
            if not (
                item[1] is verified_selected[1]
                and item[2] == verified_selected[2]
                and item[3] == verified_selected[3]
            )
        ]
    selected = None
    cross_verified = verification.get("method") in {"multilevel_header", "labeled_row"}
    question = str(row.get("question") or "")
    anchors = _row_anchor_keys(question, row_entities)
    # Score margin: refuse to assert when top-2 are nearly tied on different values.
    if len(candidates) >= 2:
        top_score, top_chunk, _top_key, top_value = candidates[0]
        second_score, _second_chunk, _second_key, second_value = candidates[1]
        if (
            top_score - second_score < 0.75
            and _norm(top_value) != _norm(second_value)
            and _row_anchor_coverage(
                str(getattr(top_chunk, "text", "") or ""), anchors, question
            ) < 0.67
        ):
            return None
    for _score, chunk, selected_key, value in candidates:
        if _score < 3.5:
            continue
        coverage = _row_anchor_coverage(
            str(getattr(chunk, "text", "") or ""), anchors, question
        )
        # Prefer rows that actually mention the asked subject; weak coverage needs
        # a stronger score so wrong-cell lookalikes (e.g. Pin vs S+D) stay out.
        if not cross_verified and anchors and coverage < 0.34:
            continue
        if not cross_verified and anchors and coverage < 0.67 and _score < 5.0:
            continue
        if _is_opaque_table_key(value):
            continue
        if not _is_plausible_cell_value(
            value, question=question, selected_key=selected_key
        ):
            continue
        chunk_text = str(getattr(chunk, "text", "") or "")
        # Method-code answers must sit on the named structural member.
        if any(term in question for term in ("평가 방법", "평가하는가", "방법으로")) and re.match(
            r"^(SP|UP)-[A-Z]$", str(value).strip(), re.I
        ):
            required = [term for term in ("웨브", "수평거더", "수평 거더", "이중선측", "호퍼") if term in question]
            if required:
                hits = sum(1 for term in required if term in chunk_text)
                if hits < max(2, len(required) - 1):
                    continue
            if "연결된" in question and "연결된" not in chunk_text:
                continue
            if "웨브" in question and "웨브" not in chunk_text:
                continue
            if "수평거더" in question and not (
                "수평거더" in chunk_text or "수평 거더" in chunk_text
            ):
                continue
        # Opaque 열N keys are remapped from question slots; keep concrete values.
        if _is_opaque_table_key(selected_key) and not (
            column_entities
            or attribute_candidates
            or any(
                term in question
                for term in ("허용기준", "판정기준", "적용두께", "기준은")
            )
        ):
            continue
        key_anchor = max(
            (
                _char_similarity(selected_key, entity)
                for entity in column_entities + attribute_candidates
            ),
            default=0.0,
        )
        key_in_question = _char_similarity(selected_key, question)
        allowance_ok = any(
            term in question for term in ("허용기준", "판정기준", "기준은")
        ) and bool(_ALLOWANCE_CODE_RE.match(str(value).strip()))
        corrosion_ok = any(
            term in question for term in ("부식추가", "tcorr", "tc1", "tc2")
        ) and bool(re.search(r"tc\s*[12]|tcorr|부식", selected_key, re.I))
        count_ok = bool(
            re.search(r"몇\s*개|개수|시험재", question)
            and re.search(r"수|개수|시험재", selected_key)
            and re.search(r"\d|한\s*개|두\s*개", str(value))
        )
        location_ok = bool(
            re.search(r"위치|어디", question)
            and selected_key in {"정의", "내용", "위치"}
            and re.search(r"전방|후방|상부|하부|내부|외부|횡격벽|종격벽", str(value))
        )
        # Strong row coverage + numeric cell can confirm even when the key label
        # is an internal code like "tc1 또는 tc2".
        numeric_ok = coverage >= 0.67 and bool(
            re.fullmatch(r"\d+(?:\.\d+)?", str(value).strip())
        )
        # Require the chosen key to be about the asked attribute — blocks
        # unrelated top-ranked noise like "단위 → m" on reporting questions.
        if (
            not allowance_ok
            and not corrosion_ok
            and not numeric_ok
            and not count_ok
            and not location_ok
            and not _is_opaque_table_key(selected_key)
        ):
            if key_anchor < 0.35 and key_in_question < 0.2:
                if not any(
                    _norm(entity) and _norm(entity) in _norm(selected_key)
                    for entity in column_entities + attribute_candidates
                ):
                    continue
        selected = (_score, chunk, selected_key, value)
        break
    if selected is None:
        return None
    _score, chunk, selected_key, value = selected
    citation_chunks = list(verification.get("support_chunks") or [chunk])
    seen_chunks = {
        str(getattr(chunk, "chunk_id", "") or "")
        or f"{getattr(chunk, 'file_name', '')}:{getattr(chunk, 'page_number', '')}"
    }
    for _other_score, other, other_key, other_value in candidates[1:]:
        if _norm(other_value) != _norm(value):
            continue
        if _char_similarity(other_key, selected_key) < 0.45 and not (
            _is_opaque_table_key(selected_key) or _is_opaque_table_key(other_key)
        ):
            continue
        identity = str(getattr(other, "chunk_id", "") or "") or (
            f"{getattr(other, 'file_name', '')}:{getattr(other, 'page_number', '')}"
        )
        if identity not in seen_chunks:
            seen_chunks.add(identity)
            citation_chunks.append(other)
        if len(citation_chunks) >= 3:
            break
    row["_answer_citation_chunks"] = citation_chunks
    row["_verified_structured_answer"] = True
    display_value = "별도 요건 없음 (-)" if value == "-" else value
    citations = "".join(f"[{i}]" for i in range(1, len(citation_chunks) + 1))
    # Emit a cited bullet under section 1 so answer_contract cannot mis-read
    # "결론: …" as an empty section heading and wipe the cell value.
    label = _display_cell_key(
        selected_key,
        column_entities=column_entities,
        attribute_candidates=attribute_candidates,
        question=str(row.get("question") or ""),
    )
    return (
        "## 1) 핵심 요약\n\n"
        f"- 결론: {label} → {display_value} {citations}"
    )


_TABLE_REFUSE_ANSWER = (
    "## 1) 핵심 요약\n\n"
    "- 표 근거에서 질문에 해당하는 셀을 확정하지 못했습니다. "
    "행 키워드·열 이름·표 번호, 또는 가능하면 파일명·페이지를 알려주시면 "
    "더 정확히 찾을 수 있습니다."
)


def should_refuse_ungrounded_table(
    row: dict,
    evidence: list[Any],
    *,
    hints: list[tuple[str, str]] | None = None,
    debug: dict | None = None,
) -> bool:
    """Refuse LLM drafting when retrieved rows barely match the asked subject."""
    question = str(row.get("question") or "")
    parsed = (debug or {}).get("parsed_query") or {}
    verification = row.get("_cell_verification") or {}
    if (
        str(parsed.get("query_type") or "")
        in {"cell_lookup", "row_lookup", "column_lookup", "condition_lookup"}
        and verification
        and not verification.get("passes", False)
    ):
        return True
    row_entities = [str(v) for v in (parsed.get("row_entities") or []) if str(v).strip()]
    anchors = _row_anchor_keys(question, row_entities)
    if not evidence:
        return True
    best_cov = max(
        (
            _row_anchor_coverage(str(getattr(c, "text", "") or ""), anchors, question)
            for c in evidence[:8]
        ),
        default=0.0,
    )
    shaped_hint = False
    for _key, value in hints or []:
        text = str(value).strip()
        if re.match(r"^(SP|UP)-[A-Z]$", text, re.I):
            shaped_hint = True
        if re.match(r"^L\d$", text, re.I):
            shaped_hint = True
        if re.fullmatch(r"\d+(?:\.\d+)?", text):
            shaped_hint = True
        if _ALLOWANCE_CODE_RE.match(text):
            shaped_hint = True
    if best_cov < 0.34 and not shaped_hint:
        return True
    blob = " ".join(str(getattr(c, "text", "") or "") for c in evidence[:8]).lower()
    if any(term in question for term in ("용접 다리", "최소 각장", "다리 길이")):
        if not any(
            term in blob
            for term in ("leg", "각장", "cargo hold", "minimum length", "용접 다리", "leg size")
        ):
            return True
    if "방화" in question or "보존성" in question:
        if not any(term in blob for term in ("방화", "fire", "보존")):
            return True
    if any(term in question for term in ("평가 방법", "평가하는가", "방법으로")):
        if not any(term in blob for term in ("평가", "sp-", "up-", "assessment", "방법")):
            return True
    return False


def build_table_refuse_answer() -> str:
    return _TABLE_REFUSE_ANSWER


def build_table_answer_prompts(
    row: dict,
    evidence: list[Any],
    *,
    debug: dict | None = None,
    cell_hints: list[tuple[str, str]] | None = None,
) -> tuple[str, str]:
    parsed = (debug or {}).get("parsed_query") or {}
    system = """당신은 선급·IMO 문서의 표를 읽고 설명하는 RAG assistant다.
제공된 표 근거 청크([1]..[N])만 사용한다. 근거에 없는 숫자·요건을 만들지 않는다.

답변 형식:
## 1) 핵심 요약
- 질문에 대한 직접 답을 bullet 3~7개로 작성한다. 각 bullet은 1~2문장.
- 선령 구간·검사 차수·구역·수치·○/- 의미를 근거 문구로 풀어 쓴다.
- '영역', 'REG01', '열1' 같은 내부 메타 라벨을 답의 주어로 쓰지 않는다.
- 사실 문장 끝에 근거 번호 [N]을 붙인다.

## 2) 세부 (필요 시)
- 비교·차수별·비고가 있으면 짧게 정리한다. 없으면 비워도 된다.

금지:
- '결론: 키 → 값' 한 줄만 내고 끝내기
- 회의 동향/후속 안건 템플릿
- 별도 '근거:' 목록 (UI Evidence Table이 담당)
- 서로 다른 table_id의 행·열을 섞어 단정하기

행·열이 확인되지 않으면 추측하지 말고, 확인된 범위만 말하거나
'표 근거에서 질문에 해당하는 셀을 확정하지 못했습니다'라고 답한다."""
    hint_lines = ""
    if cell_hints:
        rendered = "\n".join(f"- {key}: {value}" for key, value in cell_hints[:5])
        hint_lines = f"\n자동 추출 후보 셀(참고용, 틀릴 수 있음):\n{rendered}\n"
    user = (
        f"질문: {row.get('question', '')}\n"
        f"질문 유형: {parsed.get('query_type', '')}\n"
        f"찾을 행: {', '.join(parsed.get('row_entities') or []) or '(자동 판별)'}\n"
        f"찾을 열: {', '.join(parsed.get('column_entities') or []) or '(자동 판별)'}\n"
        f"{hint_lines}\n"
        f"표 근거:\n{build_table_context(evidence)}\n\n"
        "위 표 근거를 읽고 텍스트 문서 질문과 같은 밀도으로 한국어 답변을 작성하라. "
        "표 crop/Evidence Table은 UI가 따로 보여 주므로, 답변 본문은 설명에 집중한다."
    )
    return system, user


def top_table_cell_hints(
    row: dict,
    evidence: list[Any],
    *,
    debug: dict | None = None,
    limit: int = 5,
) -> list[tuple[str, str]]:
    """Ranked cell key/value pairs for LLM hints (not final answers)."""
    parsed = (debug or {}).get("parsed_query") or {}
    column_entities = [str(v) for v in parsed.get("column_entities") or [] if str(v).strip()]
    attribute_candidates = [
        str(v) for v in parsed.get("attribute_candidates") or column_entities if str(v).strip()
    ]
    question = str(row.get("question") or "")
    candidates = _rank_structured_cells(row, evidence, debug=debug)
    out: list[tuple[str, str]] = []
    for _score, _chunk, key, value in candidates:
        if _is_opaque_table_key(value):
            continue
        label = _display_cell_key(
            key,
            column_entities=column_entities,
            attribute_candidates=attribute_candidates,
            question=question,
        )
        if _is_opaque_table_key(label):
            continue
        display = "별도 요건 없음 (-)" if value == "-" else value
        item = (label.strip(), display)
        if item not in out:
            out.append(item)
        if len(out) >= limit:
            break
    return out


def _is_opaque_table_key(value: str) -> bool:
    text = (value or "").strip()
    if not text:
        return True
    if re.fullmatch(r"REG\d+", text, re.I):
        return True
    if text in {"영역", "표 셀", "비고"}:
        return True
    if re.fullmatch(r"열\d+", text):
        return True
    return False
