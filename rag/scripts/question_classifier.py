"""Question category (A–D) and broad multi-document summary mode detection."""
from __future__ import annotations

import re

from retrieval_verification import detect_narrow_doc_id

# A: 최신 동향 요약
TREND_PATTERNS = [
    r"최신\s*(MEPC|MSC)",
    r"주요\s*(내용|결과)",
    r"동향",
    r"요약해",
    r"정리해",
    r"핵심\s*요약",
    r"회의\s*주요",
]

# B: 환경규제 대응
ENV_PATTERNS = [
    # Concrete reporting/data-quality questions are environmental compliance
    # questions even when the user does not say "environmental regulation".
    r"(?:IMO\s*)?DCS|GISIS",
    r"(?:보고|제출)\s*데이터.{0,20}(?:품질|검증|오류|누락|중복)",
    r"(?:품질\s*검증|품질관리|quality\s*(?:control|verification)).{0,20}(?:오류|유형|finding)",
    r"환경규제",
    r"대체\s*연료|암모니아|수소\s*연료|저인화점|\b(?:IGF|IGC)\s*Code\b",
    r"규제\s*보고",
    r"선박\s*운항.*영향",
    r"GHG|CII|SEEMP|GFI|EEXI|Net-?Zero|MARPOL",
    r"탄소\s*(?:집약도|강도)(?:\s*(?:지수|등급))?",
    r"배출|에너지효율",
    r"운항\s*및\s*규제",
]

# C: 자율운항
AUTONOMOUS_PATTERNS = [
    r"MASS",
    r"자율\s*운항",
    r"mandatory\s*code",
    r"degree\s*of\s*autonomy",
]

# D: Rule lookup
RULE_PATTERNS = [
    # A society name is a source constraint, not a document-discovery intent.
    # Treating every "DNV 규정에 따라 ..." fact question as ``rule_lookup``
    # sends it to the catalogue renderer, which cannot answer clause values or
    # exceptions.  Explicit Rule/Guidance discovery wording is handled below.
    r"(?:DNV|LR\b|ABS|KR).{0,60}(?:관련\s*)?(?:Rule|Guidance|규칙|지침).{0,30}(?:찾|조회|목록|어떤\s*문서)",
    r"Rule/Guidance|Guidance를\s*찾",
    # Exact document names constrain retrieval, but do not by themselves mean
    # "list relevant documents".  A clause/value question in DNV-CG-xxxx or
    # Notice No.1 must still use grounded generation.
    # Short symbols and class-rule load-case codes are poor semantic-embedding
    # targets.  Route them to the narrow rule path before the generic trend
    # fallback; table questions are still caught by ``_table_qa`` above.
    r"\btc\s*(?:orr|1|2)\b|\bAC-(?:S|SD|A|T)\b",
    r"(?:구조|선급|강선)\s*규칙.{0,30}(?:기호|정의|뜻)",
    # KR Part 1 and equivalent class-rule prose.  Without these signals the
    # generic fallback labels short clause/definition asks as trend summaries.
    r"(?:\d{3,4})\s*(?:절|조|항)",
    r"제\s*\d{1,2}\s*편\s*규칙",
    r"선급(?:등록|부호|검사|기술규칙)|공동선급선|중복선급선|동형선|"
    r"선박소유자|지적사항|불가항력|풍우밀|과도한\s*부식|쇠모한도|"
    r"건조계약일|문서준수확인서|탈급|양자\s*협정|등록된\s*선박|"
    r"시험\s*및\s*검사|제조중등록검사|검사\s*신청",
    r"dual\s+class\s+vessel|double\s+class\s+vessel|sister\s+ship|"
    r"condition\s+of\s+class|force\s+majeure|weathertight|"
    r"substantial\s+corrosion",
]

MEPC_BROAD_KEYWORDS = [
    "최신 mepc",
    "환경규제",
    "ghg",
    "배출",
    "에너지효율",
    "cii",
    "seemp",
    "eexi",
    "gfi",
    "net-zero",
    "net zero",
    "marpol annex vi",
    "규제 보고",
    "선박 운항",
    "회의 주요",
    "핵심 요약",
    "동향",
    "운항 및 규제",
    "운항.*규제",
]

MSC_BROAD_KEYWORDS = [
    "msc 111",
    "최신 msc",
    "주요 결과",
    "mass code",
    "mandatory code",
    "자율운항",
    "mass wg",
]

CATEGORY_LABELS_KO = {
    "table_qa": "표 질의응답",
    "trend_summary": "최신 동향 요약",
    "meeting_outcome": "회의 결과 요약",
    "env_regulation": "환경규제 대응",
    "autonomous": "자율운항",
    "rule_lookup": "단순 Rule 질문",
}


def _is_latest_meeting_trend_summary(question: str) -> bool:
    """Distinguish a broad latest-MEPC briefing from an impact request.

    The agreed category examples intentionally place
    ``환경규제 대응과 관련된 최신 MEPC ... 주요 내용`` in the latest-trend
    category.  The phrase ``환경규제`` used to win first and route that broad
    briefing to the narrower operational-response profile.  Direct ship-work,
    reporting and compliance asks remain environmental-regulation questions.
    """
    q = str(question or "")
    return bool(
        re.search(r"최신|최근|latest|current", q, re.I)
        and re.search(r"\b(?:MEPC|MSC)\b|회의|위원회", q, re.I)
        and re.search(r"주요\s*(?:내용|결과)|동향|요약|정리", q, re.I)
        and not re.search(
            r"선박\s*운항|규제\s*보고|직접\s*영향|보고|제출|검증|"
            r"업무|실무|대응\s*(?:사항|방안|조치)",
            q,
            re.I,
        )
    )


def classify_question_category(question: str, row: dict) -> str:
    """Return trend_summary | env_regulation | autonomous | rule_lookup."""
    if row.get("_table_qa"):
        return "table_qa"
    explicit = str(row.get("category") or "").strip()
    if explicit in CATEGORY_LABELS_KO:
        return explicit

    q = question.lower()
    if any(re.search(p, q, re.I) for p in RULE_PATTERNS):
        return "rule_lookup"
    # A multi-domain MSC outcome briefing must not collapse onto whichever
    # single topic happens to appear first (usually MASS).  Two or more named
    # domains plus a summary/result ask means a cross-agenda trend summary.
    if re.search(r"(?<![A-Za-z0-9])MSC\s*\d{1,3}(?!\d)", question, re.I) and re.search(
        r"결과|채택|승인|구분|요약|정리", question, re.I
    ):
        domain_count = sum(
            bool(re.search(pattern, question, re.I))
            for pattern in (
                r"안전|safety|대체\s*연료|수소|암모니아",
                r"항해|항법|navigation",
                r"자율\s*운항|MASS|autonomous",
            )
        )
        if domain_count >= 2:
            return "trend_summary"
    if any(re.search(p, q, re.I) for p in AUTONOMOUS_PATTERNS):
        return "autonomous"
    if _is_latest_meeting_trend_summary(question):
        return "trend_summary"
    if any(re.search(p, q, re.I) for p in ENV_PATTERNS):
        return "env_regulation"
    if any(re.search(p, q, re.I) for p in TREND_PATTERNS):
        return "trend_summary"
    sources = {s.upper() for s in row.get("retrieval_sources") or []}
    if "DNV" in sources or "LR" in sources:
        return "rule_lookup"
    if "MSC" in sources and "MEPC" not in sources:
        return "autonomous" if "mepc" not in q else "env_regulation"
    return "trend_summary"


def _mepc_broad_signals(question: str) -> bool:
    ql = question.lower()
    return any(kw in ql for kw in MEPC_BROAD_KEYWORDS) or any(
        re.search(p, ql, re.I) for p in TREND_PATTERNS + ENV_PATTERNS
    )


def _msc_broad_signals(question: str) -> bool:
    ql = question.lower()
    return any(kw in ql for kw in MSC_BROAD_KEYWORDS) or any(
        re.search(p, ql, re.I) for p in AUTONOMOUS_PATTERNS + TREND_PATTERNS
    )


def detect_broad_summary_mode(question: str, row: dict, category: str | None = None) -> bool:
    """True → multi-document summarization pipeline (not simple top-k RAG)."""
    if detect_narrow_doc_id(question, row):
        return False
    cat = category or classify_question_category(question, row)
    if cat == "rule_lookup":
        return False

    sources = {s.upper() for s in row.get("retrieval_sources") or []}
    if "MEPC" in sources and cat in {"trend_summary", "env_regulation"}:
        return _mepc_broad_signals(question) or cat == "trend_summary"
    if "MSC" in sources and cat in {"autonomous", "trend_summary", "env_regulation"}:
        return _msc_broad_signals(question)
    return cat in {"trend_summary", "env_regulation"} and _mepc_broad_signals(question)


def category_label_ko(category: str) -> str:
    return CATEGORY_LABELS_KO.get(category, category)
