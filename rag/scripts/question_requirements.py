"""Question-centred requirements for retrieval and answer generation.

This module intentionally contains no evaluation ids, document names, page
numbers, or prepared answers.  It extracts what the user is asking for and
turns that into reusable search facets.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


SESSION_RE = re.compile(r"(?<![A-Za-z0-9])(MSC|MEPC)\s*[-/]?\s*(\d{1,3})(?!\d)", re.I)
DOCUMENT_ID_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:MSC|MEPC)\s*[-/]?\s*\d{1,3}"
    r"(?:\s*[-/]\s*[A-Za-z0-9.]+){1,5}(?![A-Za-z0-9])",
    re.I,
)
ORG_RE = re.compile(
    r"(?<![A-Za-z0-9])(MSC|MEPC|DNV|LR|ABS|KR)(?![A-Za-z0-9])", re.I
)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{1,}|[가-힣]{2,}")

STOPWORDS = {
    "관련", "대한", "무엇", "어떤", "정리", "요약", "알려줘", "찾아줘",
    "내용", "사항", "질문", "그리고", "에서", "으로", "했다", "했나",
    "what", "which", "find", "show", "tell", "about", "related", "please",
}

FACET_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "finding",
        r"어떤\s+(?:오류|문제|결함|위반|사실|항목|결과)|"
        r"무엇을\s+(?:식별|발견|확인)|"
        r"(?:식별|발견|확인)(?:했|한|된)\s*(?:오류|문제|결함|사항)?|"
        r"which\s+(?:errors?|issues?|findings?)|what\s+was\s+(?:identified|found)",
    ),
    ("value", r"수치|값|얼마|몇\s*%|증가율|감소율|개선율|percentage|value|how much"),
    ("metric", r"지표|기준으로 측정|산정\s*방식|metric|indicator|measure(?:d|ment)?"),
    ("comparison", r"대비|비교|기준연도|baseline|compared|versus|\bvs\.?\b"),
    ("period", r"기간|연도|언제|일정|기한|발효|시행|timeline|deadline|when|year"),
    ("status", r"상태|채택|승인|합의|결정|초안|확정|status|adopt|approve|agree|decision|draft"),
    ("requirement", r"요구사항|요건|해야|하여야|shall|must|required?|requirement"),
    ("method", r"어떻게|방법|절차|방식|how|method|procedure|process"),
    ("scope", r"적용\s*범위|대상|예외|scope|application|applies|exemption"),
    ("document", r"문서|결의|가이드|지침|규칙|rule|guidance|resolution|circular|document"),
    # Do not use the bare syllable ``항``: it occurs inside ``자율운항`` and
    # incorrectly turns broad instrument searches into clause questions.
    ("clause", r"조항|제\s*\d+\s*항|제\s*\d+\s*절|clause|section|paragraph|regulation"),
    # ``운항`` is a topic word in 자율운항/원격운항.  Count it as an impact
    # request only when it is not part of those compound nouns.
    (
        "impact",
        r"영향|대응|준비|실무|업무|(?<!자율)(?<!원격)운항|설계|승인|검증|보고|제출|"
        r"impact|operation|compliance",
    ),
    ("reason", r"이유|배경|왜|근거|reason|why|rationale"),
)

TOPIC_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        r"탄소\s*집약도|carbon\s+intensity|\bCII\b",
        ("carbon intensity", "CII", "AER", "cgDIST", "EEOI", "transport work"),
    ),
    (
        r"연료\s*소비|fuel\s+(?:oil\s+)?consumption|\bDCS\b|\bGISIS\b",
        ("fuel oil consumption", "IMO DCS", "GISIS", "submitted data", "reporting"),
    ),
    (
        r"온실가스|\bGHG\b|net[- ]zero",
        ("GHG", "greenhouse gas", "net-zero", "MARPOL Annex VI"),
    ),
    (
        r"전과정|배출계수|\bLCA\b|\bWtT\b|\bTtW\b",
        ("LCA", "life cycle", "WtT", "TtW", "emission factor"),
    ),
    (
        r"자율\s*운항|원격\s*운항|smart\s*vessel|\bMASS\b",
        ("autonomous", "remotely operated", "remote operation", "MASS", "smart vessel"),
    ),
    (
        r"\bROC\b|remote\s+operation\s+cent",
        ("ROC", "remote operations centre", "remote operator"),
    ),
    (
        r"상황\s*인식|situational\s+awareness",
        ("situational awareness", "CCTV", "monitoring", "operator"),
    ),
    (
        r"대체\s*연료|저인화점|alternative\s+fuel|low[- ]flashpoint",
        ("alternative fuel", "low-flashpoint fuel", "IGF Code", "dual fuel"),
    ),
    (r"암모니아|ammonia", ("ammonia",)),
    (r"수소|hydrogen", ("hydrogen",)),
    (
        r"보고|제출|검증|report(?:ing)?|submission|verification",
        ("reporting", "submission", "verification", "quality control"),
    ),
    (
        r"오류|품질\s*(?:검사|관리|검증)|error|quality\s*(?:check|control)",
        (
            "error",
            "duplicate reporting",
            "invalid",
            "quality control",
            "excluded",
            "not included",
        ),
    ),
)


@dataclass(frozen=True)
class QuestionRequirements:
    question: str
    organization: str = ""
    session_number: str = ""
    document_identifiers: tuple[str, ...] = ()
    facets: tuple[str, ...] = ()
    topic_terms: tuple[str, ...] = ()
    requested_count: int = 0
    broad_summary: bool = False

    @property
    def is_concrete(self) -> bool:
        return bool(self.facets or self.topic_terms) and not self.broad_summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "organization": self.organization,
            "session_number": self.session_number,
            "document_identifiers": list(self.document_identifiers),
            "facets": list(self.facets),
            "topic_terms": list(self.topic_terms),
            "requested_count": self.requested_count,
            "broad_summary": self.broad_summary,
            "is_concrete": self.is_concrete,
        }

    def search_queries(self) -> list[str]:
        scope = " ".join(
            part for part in (self.organization, self.session_number) if part
        )
        if self.document_identifiers:
            scope = " ".join(self.document_identifiers)
        topics = " ".join(self.topic_terms[:10])
        facet_terms = {
            "finding": "identified finding error issue defect missing obvious potential",
            "value": "value percentage increase decrease improvement result",
            "metric": "metric indicator AER cgDIST EEOI measurement methodology",
            "comparison": "baseline compared with change trend",
            "period": "year date timeline deadline entry into force",
            "status": "adopted approved agreed decided draft status",
            "requirement": "shall must requirement required",
            "method": "method procedure process calculation",
            "scope": "scope application applies exemption",
            "document": "official document resolution guideline rule",
            "clause": "clause section paragraph regulation",
            "impact": "operational compliance reporting verification action",
            "reason": "reason rationale basis",
        }
        queries: list[str] = []
        for facet in self.facets:
            suffix = facet_terms.get(facet, facet)
            queries.append(" ".join(part for part in (scope, topics, suffix) if part))
        if not queries:
            queries.append(" ".join(part for part in (scope, topics, self.question) if part))
        return list(dict.fromkeys(q.strip() for q in queries if q.strip()))[:6]


def _requested_count(question: str) -> int:
    match = re.search(r"(\d+)\s*(?:개|가지|건|항목|결과)", question)
    return max(1, min(10, int(match.group(1)))) if match else 0


def analyze_requirements(question: str, row: dict | None = None) -> QuestionRequirements:
    row = row or {}
    q = re.sub(r"\s+", " ", question or "").strip()
    session = SESSION_RE.search(q)
    org = session or ORG_RE.search(q)
    organization = org.group(1).upper() if org else ""
    session_number = session.group(2) if session else ""
    document_identifiers = tuple(
        dict.fromkeys(
            re.sub(r"\s*[-/]\s*", "/", match.group(0)).strip().upper()
            for match in DOCUMENT_ID_RE.finditer(q)
        )
    )

    facets = [
        name for name, pattern in FACET_PATTERNS if re.search(pattern, q, re.I)
    ]
    # Clean semantic pass for contemporary Korean input. This turns a request
    # for error types into a finding/QA task instead of a broad summary.
    if re.search(
        r"오류|누락|중복|비현실|잘못된|발견|식별|"
        r"\berrors?\b|\bfindings?\b|\bmissing\b|\bduplicate\b|"
        r"\bunrealistic\b|\bincorrect\b",
        q,
        re.I,
    ):
        facets.append("finding")
    if re.search(r"품질\s*검증|품질관리|quality\s*(?:control|verification)", q, re.I):
        facets.append("method")
    requested_count = _requested_count(q)
    broad_summary = bool(
        re.search(r"최신|주요\s*내용|동향|전반|종합|overview|latest|overall", q, re.I)
        and not any(name in facets for name in ("value", "metric", "clause", "requirement"))
    )

    topic_terms: list[str] = []
    for pattern, aliases in TOPIC_ALIASES:
        if re.search(pattern, q, re.I):
            topic_terms.extend(aliases)

    for token in WORD_RE.findall(q):
        low = token.lower()
        if low in STOPWORDS or token in STOPWORDS or len(token) < 2:
            continue
        if re.fullmatch(r"(?:MSC|MEPC)\d*", token, re.I):
            continue
        topic_terms.append(token)

    # Row metadata can scope retrieval, but never supplies an answer.
    if not organization:
        source = str(row.get("source") or row.get("society") or "").upper()
        if source in {"MSC", "MEPC", "DNV", "LR", "ABS", "KR"}:
            organization = source

    return QuestionRequirements(
        question=q,
        organization=organization,
        session_number=session_number,
        document_identifiers=document_identifiers,
        facets=tuple(dict.fromkeys(facets)),
        topic_terms=tuple(dict.fromkeys(topic_terms))[:24],
        requested_count=requested_count,
        broad_summary=broad_summary,
    )
