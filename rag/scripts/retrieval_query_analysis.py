"""Query intent and IMO session signals for retrieval boosting."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

IMO_SESSION_RE = re.compile(
    r"\b(MSC|MEPC)\s*[-/]?\s*(\d{1,3})\b|\b(\d{1,3})\s*차\b",
    re.IGNORECASE,
)

# This is part of the versioned corpus contract, not a claim about the live IMO
# website.  Update it when a newer session is added to full_corpus_715_v1.
LATEST_CORPUS_SESSION = {"MEPC": 84, "MSC": 111}
LATEST_SESSION_RE = re.compile(r"최신|최근|latest|current", re.I)

CLASS_SOCIETY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"DNV(?:에서|의)?|\bDNV\b", re.I), "DNV"),
    (re.compile(r"LR(?:에서|의)?|\bLR\b", re.I), "LR"),
    (re.compile(r"ABS(?:에서|의)?|\bABS\b", re.I), "ABS"),
    (re.compile(r"KR(?:에서|의)?|\bKR\b", re.I), "KR"),
)

CLASS_RULE_SOURCES = frozenset({"DNV", "LR", "ABS", "KR"})

RULE_DOC_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"DNV\s*[-–]?\s*CG\s*[-–]?\s*0264", re.I), "DNV-CG-0264"),
    (
        re.compile(
            r"Guide\s+for\s+Smart\s+Functions|ABS.{0,40}Smart\s+Functions?.{0,20}Guide",
            re.I,
        ),
        "ABS-Smart-Functions-Guide",
    ),
    (
        re.compile(r"Guidance\s+Notes?\s+on\s+Smart\s+Function\s+Implementation", re.I),
        "ABS-Smart-Implementation",
    ),
    (
        re.compile(
            r"Requirements?\s+for\s+Autonomous\s+and\s+Remote\s+Control\s+Functions|"
            r"ABS.{0,80}(?:autonomous|remote\s+control|자율.{0,4}원격).{0,40}(?:Requirements?|요건|규정)",
            re.I,
        ),
        "ABS-Autonomous-Remote-Requirements",
    ),
    (re.compile(r"Smart\s*Vessel|자율운항|autonomous", re.I), "autonomous"),
    (re.compile(r"Notice\s*No\.?\s*1|LR.*Rule", re.I), "Notice No.1"),
)

SOURCE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(MEPC|MSC|DNV|LR|ABS|KR)(?![A-Za-z0-9])",
    re.I,
)
EXCLUSION_AFTER_RE = re.compile(
    r"^\s*(?:문서|자료|가이드|guide|documents?)?\s*(?:는|은|를|을|에서|의)?\s*"
    r"(?:제외|빼고|말고|아닌|아니라|제외하고|제외한|exclude(?:d|ing)?|without|except)",
    re.I,
)
EXCLUSION_BEFORE_RE = re.compile(
    r"(?:exclude(?:d|ing)?|without|except)\s*$",
    re.I,
)

TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "mass": ("mass", "mass code", "자율운항", "maritime autonomous", "autonomous surface"),
    "ghg": ("ghg", "greenhouse", "온실가스", "reduction of ghg", "배출"),
    "alt_fuel": ("대체연료", "alternative fuel", "low-flashpoint", "low flashpoint", "lng", "ammonia", "methanol", "암모니아"),
    "cii": (
        "cii",
        "carbon intensity",
        "탄소집약도",
        "탄소 집약도",
        "탄소강도",
        "탄소 강도",
        "data collection",
        "보고",
    ),
    "marpol": ("marpol", "annex vi", "regulation 12"),
    "igc": ("igc code", "igc", "gas carrier"),
}

MEETING_OUTCOME_INTENT_RE = re.compile(
    r"주요\s*결과|\b결과\b|\boutcome\b|\bsummary\b|key\s*outcomes?|\badopted\b|\bapproved\b|\bdecision\b|"
    r"결정\s*사항|채택|승인|요약해|정리(?:해|됐|되었|된)|논의|개정",
    re.IGNORECASE,
)

CLASS_RULE_PROSE_RE = re.compile(
    r"(?:\d{3,4})\s*(?:절|조|항)|제\s*\d{1,2}\s*편\s*규칙|"
    r"선급(?:등록|부호|검사|기술규칙)|공동선급선|중복선급선|동형선|"
    r"선박소유자|지적사항|불가항력|풍우밀|과도한\s*부식|쇠모한도|"
    r"건조계약일|문서준수확인서|탈급|양자\s*협정|등록된\s*선박|"
    r"시험\s*및\s*검사|제조중등록검사|검사\s*신청|증서의\s*재교부|"
    r"dual\s+class\s+vessel|double\s+class\s+vessel|sister\s+ship|"
    r"condition\s+of\s+class|force\s+majeure|weathertight|"
    r"substantial\s+corrosion",
    re.IGNORECASE,
)


@dataclass
class QuerySignals:
    session_codes: list[tuple[str, int]] = field(default_factory=list)
    wants_report: bool = False
    wants_agenda: bool = False
    wants_rule_lookup: bool = False
    wants_outcome: bool = False
    wants_summary: bool = False
    meeting_outcome_question: bool = False
    topics: set[str] = field(default_factory=set)
    rule_doc_hints: list[str] = field(default_factory=list)
    expanded_terms: list[str] = field(default_factory=list)
    class_society_hint: str = ""
    named_sources: list[str] = field(default_factory=list)
    excluded_sources: list[str] = field(default_factory=list)
    constrained_sources: list[str] = field(default_factory=list)


def detect_excluded_sources(question: str) -> list[str]:
    """Return explicitly negated source names in occurrence order.

    This prevents queries such as ``MEPC는 제외하고 MASS만`` from being
    hard-filtered to the very source the user asked us to omit.
    """
    q = question or ""
    out: list[str] = []
    for match in SOURCE_TOKEN_RE.finditer(q):
        source = match.group(1).upper()
        before = q[max(0, match.start() - 24) : match.start()]
        after = q[match.end() : match.end() + 32]
        if EXCLUSION_AFTER_RE.search(after) or EXCLUSION_BEFORE_RE.search(before):
            if source not in out:
                out.append(source)
    return out


def detect_only_sources(question: str) -> list[str]:
    """Return source names constrained by Korean ``만`` or English ``only``."""
    q = question or ""
    out: list[str] = []
    matches = list(SOURCE_TOKEN_RE.finditer(q))
    for match in matches:
        source = match.group(1).upper()
        before = q[max(0, match.start() - 36) : match.start()]
        after = q[match.end() : match.end() + 44]
        only_after = re.search(
            r"^(?:.{0,32}?)(?:만(?:\s|으로|에서|보고|근거|사용|대상|가지고)|\bonly\b)",
            after,
            re.I,
        )
        only_before = re.search(r"\bonly\s*$", before, re.I)
        if only_after or only_before:
            if source not in out:
                out.append(source)
    return out


def detect_named_sources(question: str) -> list[str]:
    """Return active source names, respecting explicit only/exclude language."""
    q = question or ""
    excluded = set(detect_excluded_sources(q))
    only = [source for source in detect_only_sources(q) if source not in excluded]
    if only:
        return only
    out: list[str] = []
    for match in SOURCE_TOKEN_RE.finditer(q):
        source = match.group(1).upper()
        if source not in excluded and source not in out:
            out.append(source)
    return out


def detect_class_society_hint(question: str) -> str:
    """When the user names one class society (e.g. 'DNV에서'), prefer that source in rule lookup."""
    unique = [source for source in detect_named_sources(question) if source in CLASS_RULE_SOURCES]
    if len(unique) == 1:
        return unique[0]
    return ""


def detect_meeting_source_hint(question: str) -> str:
    """Explicit IMO meeting acronym in the question → Chroma ``source`` filter."""
    meetings = [source for source in detect_named_sources(question) if source in {"MEPC", "MSC"}]
    if len(meetings) == 1:
        return meetings[0]
    return ""


def detect_table_source_hint(question: str, *, default_society: str = "KR") -> str:
    """Table QA source: meeting acronym > named society > corpus default (KR)."""
    meeting = detect_meeting_source_hint(question)
    if meeting:
        return meeting
    society = detect_class_society_hint(question)
    if society:
        return society
    return default_society


def is_meeting_outcome_question(question: str, row: dict | None = None) -> bool:
    if row and str(row.get("category") or "") == "meeting_outcome":
        return True
    q = question.strip()
    if not q:
        return False
    lower = q.lower()
    has_session = bool(IMO_SESSION_RE.search(q)) or bool(
        re.search(r"(msc|mepc)\s*(\d{1,3})", lower)
    )
    if not has_session and LATEST_SESSION_RE.search(q):
        has_session = "mepc" in lower or "msc" in lower
    has_outcome = bool(MEETING_OUTCOME_INTENT_RE.search(q))
    if has_session and has_outcome:
        return True
    if has_session and re.search(r"의제|안건|agenda|provisional", q, re.IGNORECASE):
        return True
    if has_session and any(k in lower for k in ("주요", "핵심", "highlight", "key ")):
        return True
    if has_session and "요약" in q:
        return True
    return False


def analyze_query(query: str) -> QuerySignals:
    q = query.strip()
    lower = q.lower()
    signals = QuerySignals()
    signals.named_sources = detect_named_sources(q)
    signals.excluded_sources = detect_excluded_sources(q)
    signals.constrained_sources = detect_only_sources(q)

    for m in IMO_SESSION_RE.finditer(q):
        body = m.group(1)
        num = m.group(2) or m.group(3)
        if body and num:
            if body.upper() not in signals.excluded_sources:
                signals.session_codes.append((body.upper(), int(num)))
        elif num and ("회의" in q or "차" in m.group(0)):
            if "mepc" in lower:
                signals.session_codes.append(("MEPC", int(num)))
            elif "msc" in lower:
                signals.session_codes.append(("MSC", int(num)))

    if (
        "mepc" in lower
        and "MEPC" not in signals.excluded_sources
        and not any(s[0] == "MEPC" for s in signals.session_codes)
    ):
        m = re.search(r"mepc\s*(\d{1,3})", lower)
        if m:
            signals.session_codes.append(("MEPC", int(m.group(1))))
    if (
        "msc" in lower
        and "MSC" not in signals.excluded_sources
        and not any(s[0] == "MSC" for s in signals.session_codes)
    ):
        m = re.search(r"msc\s*(\d{1,3})", lower)
        if m:
            signals.session_codes.append(("MSC", int(m.group(1))))

    # "최신 MEPC" must not float across every historical session.  Resolve it
    # to the newest session represented by this index version before retrieval.
    if LATEST_SESSION_RE.search(q):
        for body, number in LATEST_CORPUS_SESSION.items():
            if (
                body.lower() in lower
                and body not in signals.excluded_sources
                and not any(s[0] == body for s in signals.session_codes)
            ):
                signals.session_codes.append((body, number))

    signals.wants_report = any(
        k in lower
        for k in (
            "report",
            "주요 결과",
            "결과",
            "동향",
            "요약",
            "outcome",
            "resolution",
            "decision",
        )
    )
    signals.wants_agenda = any(k in lower for k in ("agenda", "의제", "annotation", "provisional"))
    signals.wants_outcome = any(
        k in lower
        for k in (
            "outcome",
            "결과",
            "conclusion",
            "결론",
            "adopted",
            "approved",
            "decision",
            "key outcomes",
        )
    )
    signals.wants_summary = any(
        k in lower for k in ("주요", "정리", "동향", "요약", "summary", "highlight")
    )
    if "igc" in lower or "igc code" in lower:
        signals.topics.add("igc")
    if signals.wants_summary:
        signals.wants_report = True
    signals.wants_rule_lookup = bool(CLASS_RULE_PROSE_RE.search(q)) or any(
        k in lower for k in ("rule", "guidance", "찾아", "찾아줘", "notice", "cg-", "규칙")
    )
    signals.class_society_hint = detect_class_society_hint(q)

    for topic, keys in TOPIC_KEYWORDS.items():
        if any(k in lower for k in keys):
            signals.topics.add(topic)

    if any(k in lower for k in ("환경규제", "환경 규제", "environmental regulation")):
        signals.topics.update({"ghg", "marpol"})

    # A named meeting body plus a concrete topic is normally asking about the
    # newest meeting represented by this versioned corpus, even when the user
    # omits the session number (for example, "MSC MASS Code 일정").
    if not signals.session_codes and (signals.topics or signals.wants_report):
        for source in signals.named_sources:
            if source in LATEST_CORPUS_SESSION:
                signals.session_codes.append((source, LATEST_CORPUS_SESSION[source]))

    for pattern, hint in RULE_DOC_PATTERNS:
        if pattern.search(q):
            signals.rule_doc_hints.append(hint)

    # Generalize exact DNV document routing beyond the original CG-0264 case.
    for match in re.finditer(r"\bDNV\s*[-–]?\s*(CG|RP|RU)\s*[-–]?\s*([A-Z0-9-]+)\b", q, re.I):
        code = f"DNV-{match.group(1).upper()}-{match.group(2).upper()}"
        if code not in signals.rule_doc_hints:
            signals.rule_doc_hints.append(code)

    from meeting_outcome_retrieval import expand_meeting_outcome_queries

    signals.meeting_outcome_question = is_meeting_outcome_question(q)
    if signals.meeting_outcome_question:
        signals.wants_outcome = True
        signals.wants_report = True
        signals.wants_summary = True

    signals.expanded_terms = _build_expanded_terms(signals, q)
    if signals.meeting_outcome_question:
        for term in expand_meeting_outcome_queries(q, signals):
            if term not in signals.expanded_terms:
                signals.expanded_terms.append(term)
    return signals


def _build_expanded_terms(signals: QuerySignals, query: str) -> list[str]:
    terms: list[str] = []
    # Exact-document facts are placed first because the interactive embedding
    # query intentionally caps enrichment terms.  This improves recall inside
    # the named document without another vector search or reranker call.
    if re.search(r"MEPC\s*84\s*[/_-]\s*6\s*[/_-]\s*2", query, re.I):
        terms.extend(
            [
                "MEPC 84-6-2",
                "2019 to 2024",
                "up to 10.8%",
                "supply-based AER cgDIST",
                "demand-based EEOI",
            ]
        )
    if "ABS-Smart-Functions-Guide" in signals.rule_doc_hints:
        terms.extend(
            [
                "all marine vessels and offshore units",
                "optional class notation SMART INF SHM MHM",
                "risk-informed verification validation",
            ]
        )
    if "ABS-Autonomous-Remote-Requirements" in signals.rule_doc_hints:
        terms.extend(
            [
                "operations supervision level consequences of failure",
                "low medium high risk category",
                "additional verification validation",
            ]
        )
    for body, num in signals.session_codes:
        terms.extend(
            [
                f"{body} {num}",
                f"{body}-{num}",
                f"{body}/{num}",
                f"{body} {num}-",
            ]
        )
    if signals.wants_report:
        terms.extend(["report", "outcome", "resolution", "substantive"])
    if signals.wants_agenda:
        terms.extend(["agenda", "annotations", "provisional"])
    if "mass" in signals.topics:
        terms.extend(["MASS Code", "maritime autonomous", "111-5"])
    if "ghg" in signals.topics:
        terms.extend(["GHG", "reduction", "intersessional working group", "84-7"])
    if "alt_fuel" in signals.topics:
        terms.extend(["low-flashpoint", "alternative fuel", "Section 15", "engines supplied"])
    if "cii" in signals.topics:
        terms.extend(
            [
                "CII",
                "carbon intensity indicator",
                "carbon intensity",
                "operational carbon intensity rating",
                "fuel oil consumption",
            ]
        )
    if re.search(r"\btc\s*orr\b|\bt_corr\b", query, re.I):
        # The symbol is often repeated in worked formulae.  Definition phrases
        # pull the document-scoped reranker toward the defining clause instead.
        terms.extend(
            [
                "국부 부식추가",
                "부식추가",
                "corrosion addition",
                "6장 3.2",
                "정의",
            ]
        )
    if "DNV-CG-0264" in signals.rule_doc_hints:
        terms.extend(["DNV-CG-0264", "autonomous", "remotely operated"])
    if signals.class_society_hint == "DNV":
        terms.extend(["DNV-CG-0264", "DNV-RP-C205", "DNV-RP-C206", "Smart Vessel", "autonomous"])
    if "Notice No.1" in signals.rule_doc_hints:
        terms.extend(["Notice No.1", "low-flashpoint", "Section 15"])
    rule_term_expansions = (
        (r"위험\s*범주|risk\s*categor", ("risk category", "operations supervision", "consequences of failure")),
        (r"위험\s*정보|risk[- ]?informed|검증\s*활동", ("risk-informed", "verification and validation")),
        (r"foundational|기반\s*요건", ("foundational requirements", "connectivity data software")),
        (r"상위.{0,12}하위", ("higher risk category", "lower risk category")),
        (r"적용\s*(?:대상|범위)", ("scope", "applicable to")),
        (r"concept\s*qualification|개념\s*(?:검증|적격)", ("concept qualification", "AROS notation")),
        (r"초기\s*위험|preliminary\s*risk|\bPRA\b", ("preliminary risk assessment", "showstoppers")),
        (r"크랭크케이스|crankcase|\bLEL\b", ("crankcase", "below the LEL", "crankcase explosion")),
        (r"저인화점|low[- ]?flashpoint", ("low-flashpoint fuel", "Section 15")),
    )
    for pattern, expansions in rule_term_expansions:
        if re.search(pattern, query, re.I):
            terms.extend(expansions)
    pairs = {
        "환경규제": "environmental regulation MARPOL GHG emissions",
        "대체연료": "alternative fuel low-flashpoint",
        "자율운항": "autonomous MASS remotely operated",
        "선박 운항": "operational reporting CII",
    }
    for ko, en in pairs.items():
        if ko in query:
            terms.extend(en.split())
    return list(dict.fromkeys(t for t in terms if t))


def session_file_prefixes(signals: QuerySignals) -> list[str]:
    out: list[str] = []
    for body, num in signals.session_codes:
        out.append(f"{body.lower()} {num}-")
        out.append(f"{body.lower()}-{num}")
        out.append(f"{body.lower()} {num}/")
    return out


def topic_agenda_prefixes(signals: QuerySignals) -> list[str]:
    prefixes: list[str] = []
    for body, num in signals.session_codes:
        if "mass" in signals.topics and body == "MSC":
            prefixes.append(f"{body.lower()} {num}-5")
        if "ghg" in signals.topics and body == "MEPC":
            prefixes.append(f"{body.lower()} {num}-7")
        if "alt_fuel" in signals.topics and body == "MSC":
            prefixes.append(f"{body.lower()} {num}-12")
        if "igc" in signals.topics and body == "MSC":
            prefixes.append(f"{body.lower()} {num}-14")
    return prefixes
