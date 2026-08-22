"""Alternative-fuel Rule lookup: must-cover keywords and clause theme extraction."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rule_lookup_context import strip_metadata_prefix

ALT_FUEL_QUERY_RE = re.compile(
    r"대체연료|저인화점|alternative\s+fuel|low[- ]flashpoint|dual\s+fuel|lng|암모니아|메탄올",
    re.I,
)

# (display label, match terms)
ALT_FUEL_MUST_COVER: list[tuple[str, list[str]]] = [
    ("low-flashpoint fuel", ["low-flashpoint", "low flashpoint"]),
    ("alternative fuel", ["alternative fuel"]),
    ("dual fuel", ["dual fuel", "dual-fuel"]),
    ("fuel storage", ["fuel storage", "storage of fuel"]),
    ("fuel supply", ["fuel supply", "supplied with"]),
    ("engine requirements", ["engine", "engines supplied", "crankcase"]),
    ("Section 15", ["section 15"]),
    ("IGF Code", ["igf code", "igf"]),
    ("IGC Code", ["igc code", "igc"]),
    ("methanol", ["methanol"]),
    ("ammonia", ["ammonia"]),
    ("hydrogen", ["hydrogen"]),
    ("LNG", ["lng", "liquefied natural gas"]),
]

_SCORE_WEIGHTS: list[tuple[float, list[str]]] = [
    (4.0, ["section 15"]),
    (3.5, ["low-flashpoint", "low flashpoint"]),
    (3.0, ["dual fuel", "dual-fuel"]),
    (2.5, ["alternative fuel"]),
    (2.0, ["engines supplied", "engine"]),
    (1.8, ["fuel storage", "fuel supply", "supplied with"]),
    (1.5, ["crankcase", "lel"]),
    (1.5, ["igf", "igc"]),
    (1.2, ["methanol", "ammonia", "hydrogen", "lng"]),
    (1.0, ["survey", "amend", "2025"]),
]


@dataclass
class ClauseTheme:
    theme_id: str
    title: str
    doc_type: str
    relevance_ko: str
    content_ko: str
    confirmation: str
    file_name: str
    page: int | None
    clause_ref: str
    citation_ids: list[int] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)


def is_alt_fuel_question(question: str) -> bool:
    return bool(ALT_FUEL_QUERY_RE.search(question or ""))


def _body_lower(chunk: Any) -> str:
    return strip_metadata_prefix(getattr(chunk, "text", "")).lower()


def score_chunk_for_rule_lookup(body: str, question: str = "") -> float:
    low = (body or "").lower()
    score = 0.0
    for weight, terms in _SCORE_WEIGHTS:
        if any(t in low for t in terms):
            score += weight
    if is_alt_fuel_question(question) and ("대체" in question or "연료" in question):
        if "notice" in low or "section" in low:
            score += 0.5
    return score


def pick_citation_id(
    chunk: Any,
    citation_chunks: list[Any],
    citation_fallback: list[Any] | None,
) -> list[int]:
    # A page can contain several independently chunked clauses.  Matching only
    # file/page can therefore attach a claim about clause 15.8.2 to a different
    # clause on the same page.  Prefer the stable chunk id and use file/page
    # only for legacy chunks that have no id.
    chunk_id = str(getattr(chunk, "chunk_id", "") or "")
    if chunk_id:
        for i, candidate in enumerate(citation_chunks, start=1):
            if str(getattr(candidate, "chunk_id", "") or "") == chunk_id:
                return [i]
    fn = str(getattr(chunk, "file_name", "") or "")
    page = getattr(chunk, "page_number", None)
    for i, c in enumerate(citation_chunks, start=1):
        if str(getattr(c, "file_name", "")) != fn:
            continue
        if page is not None and getattr(c, "page_number", None) == page:
            return [i]
    for i, c in enumerate(citation_chunks, start=1):
        if str(getattr(c, "file_name", "")) == fn:
            return [i]
    return []


def select_alt_fuel_work_chunks(
    pool: list[Any],
    retrieved: list[Any],
    *,
    question: str,
    max_chunks: int = 16,
) -> list[Any]:
    scored: list[tuple[float, Any]] = []
    for c in pool:
        body = strip_metadata_prefix(getattr(c, "text", ""))
        s = score_chunk_for_rule_lookup(body, question)
        if s > 0:
            scored.append((s, c))
    scored.sort(key=lambda x: (-x[0], getattr(x[1], "page_number", 0) or 0))
    out: list[Any] = []
    seen: set[tuple[str, int | None, str]] = set()
    for _, c in scored:
        body = strip_metadata_prefix(getattr(c, "text", ""))
        key = (str(getattr(c, "file_name", "")), getattr(c, "page_number", None), body[:100])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= max_chunks:
            break
    return out if out else list(retrieved)


def _matched_keywords(body: str) -> list[str]:
    low = body.lower()
    found: list[str] = []
    for label, terms in ALT_FUEL_MUST_COVER:
        if any(t in low for t in terms):
            found.append(label)
    return found


def _clause_ref(body: str) -> str:
    m = re.search(r"section\s+(\d+(?:\.\d+)*)", body, re.I)
    if m:
        return f"Section {m.group(1)}"
    m = re.search(r"(\d+\.\d+(?:\.\d+)*)", body[:80])
    if m:
        return m.group(1)
    return ""


def _notice_doc_label(file_name: str) -> str:
    if "notice" in file_name.lower():
        return "LR Notice No.1"
    return file_name.replace(".pdf", "")


def extract_clause_themes(
    chunks: list[Any],
    *,
    question: str,
    citation_chunks: list[Any],
    citation_fallback: list[Any] | None = None,
) -> list[ClauseTheme]:
    if not is_alt_fuel_question(question):
        return []

    themes: list[ClauseTheme] = []
    used_ids: set[str] = set()

    def add_theme(
        theme_id: str,
        *,
        predicate,
        title: str,
        doc_type: str,
        relevance_ko: str,
        content_ko: str,
        confirmation: str = "관련 조항 후보 / 추가 확인 필요",
    ) -> None:
        if theme_id in used_ids:
            return
        best: tuple[float, Any, list[str]] | None = None
        for c in chunks:
            body = strip_metadata_prefix(getattr(c, "text", ""))
            low = body.lower()
            if not predicate(low, body):
                continue
            score = score_chunk_for_rule_lookup(body, question)
            kw = _matched_keywords(body)
            if best is None or score > best[0]:
                best = (score, c, kw)
        if best is None:
            return
        _, chunk, kw = best
        fn = str(getattr(chunk, "file_name", "") or "")
        notice = _notice_doc_label(fn)
        resolved_title = title.replace("{notice}", notice)
        themes.append(
            ClauseTheme(
                theme_id=theme_id,
                title=resolved_title,
                doc_type=doc_type,
                relevance_ko=relevance_ko,
                content_ko=content_ko,
                confirmation=confirmation,
                file_name=fn,
                page=getattr(chunk, "page_number", None),
                clause_ref=_clause_ref(strip_metadata_prefix(getattr(chunk, "text", ""))),
                citation_ids=pick_citation_id(chunk, citation_chunks, citation_fallback),
                matched_keywords=kw,
            )
        )
        used_ids.add(theme_id)

    add_theme(
        "section_15_low_flashpoint",
        predicate=lambda low, _: "section 15" in low
        and ("low-flashpoint" in low or "low flashpoint" in low or "engines supplied" in low),
        title="{notice} — Section 15 / low-flashpoint fuel 관련 조항 후보",
        doc_type="LR Rules 후보",
        relevance_ko="low-flashpoint fuel engine requirements 및 dual/alternative fuel arrangement 관련 조항이 검색됨",
        content_ko="저인화점 연료·가스 등을 사용하는 기관의 적용 범위, 엔진 안전성 평가, dual fuel arrangement와 관련",
    )

    add_theme(
        "dual_fuel_engines",
        predicate=lambda low, _: "dual fuel" in low or "dual-fuel" in low,
        title="{notice} — dual fuel engine 관련 조항 후보",
        doc_type="LR Rules 후보",
        relevance_ko="dual fuel engine의 crankcase ventilation·안전 arrangement 관련 조항이 검색됨",
        content_ko="trunk piston dual fuel engine의 crankcase ventilation, 저인화점 연료 사용 시 환기·안전 arrangement 검토와 관련",
    )

    # Upgrade the two primary LR cards from topic labels to the operative
    # source conditions.  Keeping them as two compact cards preserves the
    # agreed 2-3 bullet Rule-answer budget while still exposing the actual
    # design/monitoring duties.
    for theme in themes:
        if theme.theme_id == "section_15_low_flashpoint":
            theme.title = f"{_notice_doc_label(theme.file_name)} — Section 15 / low-flashpoint fuel"
            theme.doc_type = "LR Rules"
            theme.relevance_ko = "가스·저인화점 연료 기관의 crankcase 안전성 상세 평가 요구"
            theme.content_ko = (
                "가스 또는 저인화점 연료 기관은 crankcase 가스 농도가 별도 조치 없이 "
                "LEL 미만으로 유지되거나, 특정 조치로 폭발 위험이 감소함을 상세 안전성 "
                "평가로 입증해야 함"
            )
        elif theme.theme_id == "dual_fuel_engines":
            theme.title = f"{_notice_doc_label(theme.file_name)} — dual fuel engine crankcase ventilation"
            theme.doc_type = "LR Rules"
            theme.relevance_ko = "trunk piston dual fuel engine의 crankcase 환기 예외와 안전 조건"
            theme.content_ko = (
                "외부 공기 유입 환기는 원칙적으로 금지되지만 가스·저인화점 연료를 쓰는 "
                "trunk piston dual fuel engine에는 crankcase 환기를 제공해야 하며, 폭발위험 "
                "비증가 입증·작동 모니터링·고장 시 자동 안전조치 또는 위험완화조치가 요구됨"
            )

    add_theme(
        "engine_safety_evaluation",
        predicate=lambda low, _: ("low flashpoint" in low or "low-flashpoint" in low)
        and ("engine" in low or "crankcase" in low)
        and (
            "lower explosive limit" in low
            or re.search(r"\blel\b", low) is not None
            or "explosion" in low
        )
        and "section 15" not in low[:120],
        title="{notice} — 저인화점 연료 기관 안전성 평가 조항 후보",
        doc_type="LR Rules 후보",
        relevance_ko="gas·low-flashpoint fuel 기관의 crankcase 안전성 상세 평가 요건이 검색됨",
        content_ko="저인화점 연료·가스 기관의 crankcase LEL·폭발 위험 저감 조치에 대한 상세 평가(survey 전제)와 관련",
    )

    add_theme(
        "notice_2025_amendment",
        predicate=lambda low, _: "amend" in low and ("2025" in low or "notice" in low),
        title="IMO IGF/IGC Code 정합 관련 Notice 개정 후보",
        doc_type="Rule 개정/Notice 후보",
        relevance_ko="LR Notice 개정·적용 시점(2025) 관련 문구가 검색됨 — IMO IGF/IGC Code 정합 여부 추가 확인 필요",
        content_ko="2025 Notice 개정 적용 시점·이전 Notice와의 병행 적용; IMO IGF/IGC Code와의 정합은 본문 추가 확인 필요",
    )

    return themes


ANSWER_HINTS: dict[str, list[str]] = {
    "low-flashpoint fuel": ["low-flashpoint", "low flashpoint", "저인화점"],
    "alternative fuel": ["alternative fuel", "alternative", "대체연료"],
    "dual fuel": ["dual fuel", "dual-fuel"],
    "fuel storage": ["fuel storage", "연료 저장", "storage"],
    "fuel supply": ["fuel supply", "연료 공급", "공급"],
    "engine requirements": ["engine", "engines", "기관", "crankcase"],
    "Section 15": ["section 15"],
    "IGF Code": ["igf"],
    "IGC Code": ["igc"],
    "methanol": ["methanol", "메탄올"],
    "ammonia": ["ammonia", "암모니아"],
    "hydrogen": ["hydrogen", "수소"],
    "LNG": ["lng", "liquefied natural gas"],
}


def priority_section4_themes(themes: list[ClauseTheme]) -> list[ClauseTheme]:
    by_id = {t.theme_id: t for t in themes}
    out: list[ClauseTheme] = []
    if "section_15_low_flashpoint" in by_id:
        out.append(by_id["section_15_low_flashpoint"])
    for key in ("notice_2025_amendment", "dual_fuel_engines", "engine_safety_evaluation"):
        if key in by_id and len(out) < 2:
            out.append(by_id[key])
    return out[:2]


def compute_alt_fuel_must_cover(
    chunks: list[Any],
    answer: str = "",
) -> list[dict]:
    chunk_text = "\n".join(_body_lower(c) for c in chunks)
    ans = (answer or "").lower()
    rows: list[dict] = []
    for label, terms in ALT_FUEL_MUST_COVER:
        in_chunks = any(t in chunk_text for t in terms)
        hints = ANSWER_HINTS.get(label, terms)
        in_answer = any(h.lower() in ans for h in hints) or label.lower() in ans
        rows.append(
            {
                "must_cover": label,
                "found_in_chunks": "Yes" if in_chunks else "No",
                "included_in_answer": "Yes" if in_answer else ("No" if answer else "—"),
            }
        )
    return rows


def format_theme_citations(theme: ClauseTheme) -> str:
    return "".join(f"[{i}]" for i in theme.citation_ids)


def format_theme_grounds(theme: ClauseTheme) -> str:
    page = f"p.{theme.page}" if theme.page is not None else "p.?"
    cite = format_theme_citations(theme)
    parts = [theme.file_name or "—", page]
    if cite:
        parts.append(cite.strip())
    return ", ".join(parts)
