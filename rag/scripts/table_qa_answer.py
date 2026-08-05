"""Grounded answer context and prompts for structured table questions."""
from __future__ import annotations

import re
from typing import Any


_TYPE_ORDER = {
    "table_row": 0,
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

    unique: dict[str, Any] = {}
    for chunk in list(retrieved) + list(pool):
        cid = str(getattr(chunk, "chunk_id", "") or id(chunk))
        unique.setdefault(cid, chunk)

    def score(chunk: Any) -> tuple:
        tid = str(getattr(chunk, "table_id", "") or "")
        ctype = str(getattr(chunk, "chunk_type", "") or "")
        text = _norm(str(getattr(chunk, "text", "") or ""))
        matches = sum(1 for term in terms if term in text)
        table_rank = candidate_rank.get(tid, 99)
        candidate_penalty = table_rank if table_rank < 99 else 20
        return (
            candidate_penalty,
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
        for ctype in ("table_row", "table_markdown", "table_schema", "table_summary"):
            matches = [c for c in same if str(getattr(c, "chunk_type", "") or "") == ctype]
            for chunk in matches[: (4 if ctype == "table_row" else 1)]:
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


def _cell_assignments(text: str) -> dict[str, str]:
    for line in str(text or "").splitlines():
        if not line.startswith("셀:"):
            continue
        out: dict[str, str] = {}
        for part in line.removeprefix("셀:").strip().split(" | "):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            if key.strip() and value.strip():
                out[key.strip()] = value.strip()
        return out
    return {}


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
        if str(getattr(chunk, "chunk_type", "") or "") != "table_row":
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
        row_score = (
            subject_match * 2.3
            + token_coverage * 1.4
            + value_match * 0.8
            + numeric_coverage * 2.4
            + route_bonus
            - complexity_penalty
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
            structural_penalty = 0.0
            if "비고" in key_norm:
                structural_penalty += 1.1
            if "세부항목" in key_norm or key_norm.startswith("열"):
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
    query_type = str(parsed.get("query_type") or "")
    row_entities = [str(v) for v in parsed.get("row_entities") or [] if str(v).strip()]
    column_entities = [str(v) for v in parsed.get("column_entities") or [] if str(v).strip()]
    if query_type not in {"cell_lookup", "row_lookup", "column_lookup", "condition_lookup"}:
        return None
    candidates = _rank_structured_cells(row, evidence, debug=debug_data)
    if not candidates:
        return None
    _score, chunk, selected_key, value = candidates[0]
    citation_chunks = [chunk]
    seen_chunks = {
        str(getattr(chunk, "chunk_id", "") or "")
        or f"{getattr(chunk, 'file_name', '')}:{getattr(chunk, 'page_number', '')}"
    }
    for _other_score, other, other_key, other_value in candidates[1:]:
        if _norm(other_value) != _norm(value):
            continue
        if _char_similarity(other_key, selected_key) < 0.45:
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
    display_value = "별도 요건 없음 (-)" if value == "-" else value
    citations = "".join(f"[{i}]" for i in range(1, len(citation_chunks) + 1))
    return f"결론: {display_value}입니다. {citations}"


def build_table_answer_prompts(
    row: dict,
    evidence: list[Any],
    *,
    debug: dict | None = None,
) -> tuple[str, str]:
    parsed = (debug or {}).get("parsed_query") or {}
    system = """당신은 선급 규칙의 구조화 표를 판독하는 QA 전문가다.
오직 제공된 표 근거만 사용하고, 질문의 행과 열이 동시에 일치하는 셀을 우선한다.
서로 다른 table_id의 행과 열을 결합하지 않는다.
○는 해당 검사 차수의 대상, '-'는 별도 요건 없음으로 해석하되 표에 있는 경우에만 단정한다.
첫 줄은 반드시 '결론: '으로 시작해 질문에 직접 답한다.
그 다음 필요한 경우 차수·선령 구간별 내용을 짧은 bullet로 정리한다.
모든 사실 문장을 한 문장씩 분리하고 각 문장 끝에 근거 번호 [N]을 붙인다.
문서명·페이지·원문 청크는 UI Evidence Table에서 표시하므로 별도의 '근거:' 목록은 쓰지 않는다.
회의 결과, 후속 회의, 최신 동향 형식은 사용하지 않는다.
행·열이 함께 확인되지 않으면 추측하지 말고 '표 근거에서 정확한 셀을 확인하지 못했습니다'라고 답한다."""
    user = (
        f"질문: {row.get('question', '')}\n"
        f"질문 유형: {parsed.get('query_type', '')}\n"
        f"찾을 행: {', '.join(parsed.get('row_entities') or []) or '(자동 판별)'}\n"
        f"찾을 열: {', '.join(parsed.get('column_entities') or []) or '(자동 판별)'}\n\n"
        f"구조화 표 근거:\n{build_table_context(evidence)}\n\n"
        "위 근거에서 질문에 해당하는 표·행·열을 확인하여 간결한 한국어 답변을 작성하라."
    )
    return system, user
