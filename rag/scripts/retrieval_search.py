"""Clause-aware hybrid retrieval (dense + lexical + metadata boost)."""
from __future__ import annotations

import math
import os
import re
from typing import Any

from clause_parse import is_article_clause_number
from imo_doc_classify import classify_imo_filename, tier_for_query
from imo_doc_registry import exact_doc_ids_for_query, priority_doc_ids_for_signals
from retrieval_query_analysis import (
    CLASS_RULE_SOURCES,
    QuerySignals,
    analyze_query,
    session_file_prefixes,
    topic_agenda_prefixes,
)

CLAUSE_IN_QUERY_RE = re.compile(
    r"(?:제?\s*)?(\d{3,4})\s*절|(\d{3,4})절|(?:^|\s)(\d{3,4})(?:\s*절|\s|$)",
    re.IGNORECASE,
)

TOKEN_RE = re.compile(r"[\w가-힣]+", re.UNICODE)

CLAUSE_EXACT_BOOST = 0.22
CLAUSE_IN_TEXT_BOOST = 0.08
LEXICAL_BOOST_SCALE = 0.18
REFERENCE_LINE_BOOST = 0.06
SESSION_FILE_BOOST = 0.14
TOPIC_PREFIX_BOOST = 0.20
RULE_FILENAME_BOOST = 0.24
PRIORITY_DOC_BOOST = 0.28
EXPANDED_TERM_BOOST = 0.06
SUBCOMM_PENALTY = 0.14
DOC_CODE_CROSSREF_RE = re.compile(r"document\s+code.*title", re.I)
DNV_RULE_CODE_RE = re.compile(
    r"^DNV-(?:CG|RP|RU|OS|CP|SI)-[A-Z0-9]+(?:-[A-Z0-9]+)*$", re.I
)
KR_RULE_PART_RE = re.compile(
    r"(?:제\s*)?(\d{1,2})\s*편(?:\s*[_-]?\s*(20\d{2})|\s*년판)?",
    re.IGNORECASE,
)
KR_PART1_CONTEXT_RE = re.compile(
    r"선급(?:등록|부호|검사|기술규칙)|공동선급선|중복선급선|동형선|"
    r"선박소유자|지적사항|불가항력|풍우밀|과도한\s*부식|쇠모한도|"
    r"건조계약일|문서준수확인서|탈급|양자\s*협정|등록된\s*선박|"
    r"시험\s*및\s*검사|제조중등록검사|"
    r"dual\s+class\s+vessel|double\s+class\s+vessel|sister\s+ship|"
    r"condition\s+of\s+class|force\s+majeure|weathertight|"
    r"substantial\s+corrosion",
    re.IGNORECASE,
)
EXACT_IDENTIFIER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"t(?:corr|c\s*[12])|"
    r"AC-[A-Z0-9+.-]+|"
    r"CII|EEXI|SEEMP|MARPOL|"
    r"DNV-(?:CG|RP|RU|OS|CP|SI)-[A-Z0-9-]+|"
    r"(?:MEPC|MSC)\s*\d{1,3}(?:\s*[/.-]\s*[A-Z0-9]+)+"
    r")(?![A-Za-z0-9])|\d+\s*장\s*\d+\s*절",
    re.I,
)
SPARSE_TOKEN_RE = re.compile(
    r"[A-Za-z]+(?:[A-Za-z0-9]*)(?:[-./][A-Za-z0-9]+)*|\d+(?:\.\d+)?|[가-힣]{2,}",
    re.UNICODE,
)
SPARSE_STOPWORDS = {
    "관련", "기준", "문서", "규정", "주요", "어떤", "무엇", "알려줘", "찾아줘",
    "에서", "으로", "대한", "the", "and", "for", "with", "what", "which",
}
FEATURE_TERM_STOPWORDS = SPARSE_STOPWORDS | {
    "내용", "정의", "의미", "설명", "요건", "예외", "조건", "방법", "절차",
    "적용", "대상", "범위", "차이", "비교", "요약", "확인", "질문", "결과",
    "최신", "해당", "경우", "제품마다", "알려주", "설명해줘", "정리해줘",
}
KOREAN_QUERY_SUFFIXES = (
    "으로부터", "에서는", "에게서", "이라고", "라는", "하는가", "되는가",
    "해야", "하고", "에서", "으로", "에게", "에는", "은", "는", "이", "가",
    "을", "를", "의", "와", "과", "에",
)
FEATURE_NOUN_ENDINGS = (
    "조치", "장치", "설비", "시스템", "선박", "선급", "검사", "시험", "시험재",
    "두께", "재료", "계수", "하중", "기관", "강도", "구조", "용접", "부식",
    "위험", "기능", "기준", "절차", "기호", "인증서", "방식", "보호", "제어",
)
SHORT_TECHNICAL_FEATURE_TERMS = {
    # Short fuel and safety nouns are highly distinctive in this corpus but
    # were excluded by the generic five-Hangul-character threshold.
    "암모니아", "수소", "메탄올", "에탄올", "프로판", "부탄", "독성",
    "케이블", "평형수", "농도",
    "ammonia", "hydrogen", "methanol", "ethanol", "propane", "butane",
    # Distinctive maritime/environment acronyms and compact study terms.  These
    # are intentionally searched only when absent/weak in the dense candidate
    # pool, through bounded Chroma ``where_document`` lookups.
    "goc", "lng", "wtt", "ttw", "bwms", "co2", "vdr", "eca", "ebp",
    "cslip", "upstream", "evidence",
}
FEATURE_LITERAL_TRANSLATIONS = {
    # Exact fallback queries must use the language that appears in the source
    # PDF.  The UI question is usually Korean while IMO/Class documents are
    # frequently English; without these bounded variants ``수소`` cannot
    # recover a clause containing only ``hydrogen``.
    "수소": ("hydrogen",),
    "암모니아": ("ammonia",),
    "메탄올": ("methanol",),
    "에탄올": ("ethanol",),
    "프로판": ("propane",),
    "부탄": ("butane",),
    "독성": ("toxic", "toxicity"),
}
FEATURE_VERB_ENDINGS = (
    "하거나", "하거나요", "되는지", "하는지", "했는지", "알려줘", "찾아줘",
    "설명해줘", "정리해줘", "비교해줘", "확인해줘", "넘는", "되는", "하는",
    "되어", "하여", "하면", "한가", "인가", "있나", "없나",
)
DOCUMENT_ROUTE_MAX_DOCS = 4
DOCUMENT_ROUTE_MAX_DOCS_BROAD = 6
DOCUMENT_ROUTE_STAGE2_FETCH = 72
DOCUMENT_ROUTE_BOOSTS = (0.20, 0.13, 0.08, 0.04, 0.02, 0.01)
SCOPED_SPARSE_MAX_ROWS = 6000

ABS_RULE_DOC_IDS = {
    "ABS-Smart-Functions-Guide": (
        "abs_abs_rules_guideforsmartfunctionsformarinevesselsandoffshoreunits_v8_bbfd9d9e"
    ),
    "ABS-Smart-Implementation": (
        "abs_abs_rules_guidancenotesonsmartfunctionimplementation_v1_b275249c"
    ),
    "ABS-Autonomous-Remote-Requirements": (
        "abs_abs_rules_requirementsforautonomousandremotecontrolfunctions_v4_1d89b7bb"
    ),
}


def _priority_rule_file_names(signals: QuerySignals, query: str = "") -> list[str]:
    """Return exact class-rule filenames worth querying alongside dense hits.

    The global dense search can miss a short document code when the question is
    phrased by topic.  Exact metadata routing is cheap and keeps interactive
    retrieval fast without scanning the full BM25 corpus.
    """
    names: list[str] = []
    for hint in signals.rule_doc_hints:
        code = str(hint or "").strip()
        if DNV_RULE_CODE_RE.fullmatch(code):
            names.append(f"{code.upper()}.pdf")

    autonomous_hint = any(
        str(hint or "").lower() == "autonomous" for hint in signals.rule_doc_hints
    )
    if (
        signals.wants_rule_lookup
        and str(signals.class_society_hint or "").upper() == "DNV"
        and (autonomous_hint or "mass" in signals.topics)
    ):
        names.append("DNV-CG-0264.pdf")
    if (
        signals.wants_rule_lookup
        and str(signals.class_society_hint or "").upper() == "DNV"
        and re.search(r"smart\s+vessel", query or "", re.I)
    ):
        names.append("DNV-CG-0508.pdf")
    part = KR_RULE_PART_RE.search(query or "")
    if part:
        volume = int(part.group(1))
        year = part.group(2) or "2025"
        names.append(f"{volume}편_{year}.pdf")
    elif KR_PART1_CONTEXT_RE.search(query or ""):
        names.append("1편_2025.pdf")
    return list(dict.fromkeys(names))


def _direct_priority_rule_doc_ids(query: str, signals: QuerySignals) -> list[str]:
    """Known corpus identities for strongly named rule documents."""
    out: list[str] = []
    q = query or ""
    part = KR_RULE_PART_RE.search(q)
    unnamed_korean_clause = bool(
        not part
        and not signals.class_society_hint
        and re.search(r"[가-힣]", q)
        and re.search(r"(?:^|\s)\d{3,4}\s*(?:절|조|항)(?:\s|에서|의|$)", q)
    )
    if (part and int(part.group(1)) == 1 and (part.group(2) or "2025") == "2025") or (
        not part and KR_PART1_CONTEXT_RE.search(q)
    ) or unnamed_korean_clause:
        out.append("kr_1_2025")
    if "Notice No.1" in signals.rule_doc_hints:
        out.append("lr_notice_no_1_2025")
    for hint, doc_id in ABS_RULE_DOC_IDS.items():
        if hint in signals.rule_doc_hints:
            out.append(doc_id)
    if (
        str(signals.class_society_hint or "").upper() == "ABS"
        and any(str(h).lower() == "autonomous" for h in signals.rule_doc_hints)
        and not any(hint in signals.rule_doc_hints for hint in ABS_RULE_DOC_IDS)
    ):
        out.append(
            "abs_abs_rules_requirementsforautonomousandremotecontrolfunctions_v4_1d89b7bb"
        )
    return list(dict.fromkeys(out))


def infer_query_narrow_doc_id(query: str, signals: QuerySignals) -> str | None:
    """Return a single high-confidence document identity stated by the query."""
    manifest_ids = exact_doc_ids_for_query(query)
    if len(manifest_ids) == 1:
        return manifest_ids[0]
    direct = _direct_priority_rule_doc_ids(query, signals)
    return direct[0] if len(direct) == 1 else None


def _resolve_priority_rule_doc_ids(
    collection, signals: QuerySignals, query: str = ""
) -> list[str]:
    """Resolve exact rule filenames to Chroma document ids."""
    direct = _direct_priority_rule_doc_ids(query, signals)
    names = _priority_rule_file_names(signals, query)
    if not names:
        return direct
    where = {"file_name": names[0]} if len(names) == 1 else {"file_name": {"$in": names}}
    try:
        raw = collection.get(where=where, include=["metadatas"], limit=100)
    except Exception:
        return direct
    doc_ids: list[str] = list(direct)
    for meta in raw.get("metadatas") or []:
        doc_id = str((meta or {}).get("doc_id") or "").strip()
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)
    return doc_ids


def resolve_explicit_query_doc_id(
    collection, query: str, signals: QuerySignals | None = None
) -> str | None:
    """Resolve one explicitly named document to its Chroma document id.

    This is deliberately narrower than topic routing: only a manifest-known
    document or a literal class document code is hard-selected.  Generic
    society/topic questions keep the normal Accurate document-ranking path.
    """
    signals = signals or analyze_query(query)
    excluded = {str(source).upper() for source in signals.excluded_sources}
    constrained = {str(source).upper() for source in signals.constrained_sources}
    explicit_identifier_sources = {
        match.group(1).upper()
        for match in re.finditer(
            r"(?<![A-Za-z0-9])(DNV|LR|ABS|KR|MEPC|MSC)"
            r"(?=\s*[-–/]\s*(?:[A-Z]{1,4}\s*[-–/]\s*)?\d)",
            query or "",
            re.I,
        )
    }
    disallowed_identifier = any(
        source in excluded or (constrained and source not in constrained)
        for source in explicit_identifier_sources
    )
    if not disallowed_identifier:
        direct = infer_query_narrow_doc_id(query, signals)
        if direct:
            return direct
    else:
        # The named code may appear only to say "exclude it".  A positively
        # named known document (for example ABS Autonomous Requirements) can
        # still be selected; otherwise keep the normal constrained search.
        positive_direct = _direct_priority_rule_doc_ids(query, signals)
        if len(positive_direct) == 1:
            return positive_direct[0]
    explicit_codes = [
        str(hint or "").strip().upper()
        for hint in signals.rule_doc_hints
        if DNV_RULE_CODE_RE.fullmatch(str(hint or "").strip())
        and "DNV" not in excluded
        and (not constrained or "DNV" in constrained)
    ]
    if len(explicit_codes) != 1:
        return None
    resolved = _resolve_priority_rule_doc_ids(collection, signals, query)
    return resolved[0] if len(resolved) == 1 else None


def _hierarchical_text_enabled() -> bool:
    return os.environ.get("MARITIME_TEXT_HIERARCHICAL", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def extract_exact_identifiers(query: str) -> list[str]:
    """Extract terms for which literal retrieval is safer than embeddings."""
    out: list[str] = []
    for match in EXACT_IDENTIFIER_RE.finditer(query or ""):
        value = re.sub(r"\s*([/.-])\s*", r"\1", match.group(0).strip())
        value = re.sub(r"\s+", " ", value)
        if value and value.lower() not in {item.lower() for item in out}:
            out.append(value)
    return out[:4]


def _korean_query_stem(token: str) -> str:
    stem = str(token or "").strip().lower()
    for suffix in KOREAN_QUERY_SUFFIXES:
        if stem.endswith(suffix) and len(stem) - len(suffix) >= 2:
            return stem[: -len(suffix)]
    return stem


def extract_sparse_feature_terms(query: str, *, limit: int = 1) -> list[str]:
    """Pick distinctive Korean nouns for zero-hit literal recovery.

    This is intentionally conservative.  Exact-code recovery already handles
    Latin identifiers, while generic Korean intent words (``요건``, ``예외``,
    ``설명``) would match too much of the 271k-chunk corpus.  Short compounds
    are accepted only when they have a technical noun ending; otherwise a term
    must be at least five Hangul characters long.
    """
    ranked: list[tuple[int, int, int, str]] = []
    seen: set[str] = set()
    for position, raw in enumerate(re.findall(r"[가-힣]{2,}", query or "")):
        low = raw.lower()
        if any(low.endswith(ending) for ending in FEATURE_VERB_ENDINGS):
            continue
        term = _korean_query_stem(low)
        technical_short = term in SHORT_TECHNICAL_FEATURE_TERMS
        if (
            term in FEATURE_TERM_STOPWORDS
            or term in seen
            or (len(term) < 3 and not technical_short)
        ):
            continue
        noun_ending = any(term.endswith(ending) for ending in FEATURE_NOUN_ENDINGS)
        if len(term) < 5 and not noun_ending and not technical_short:
            continue
        seen.add(term)
        ranked.append((2 if technical_short else 1 if noun_ending else 0, len(term), -position, term))
    ranked.sort(reverse=True)
    return [item[3] for item in ranked[: max(0, limit)]]


def extract_sparse_latin_terms(query: str, *, limit: int = 2) -> list[str]:
    """Return distinctive Latin technical phrases for zero-hit recovery.

    Korean compound recovery cannot help queries such as ``minimum risk
    condition`` or ``crankcase ventilation``.  Keep this conservative: remove
    document codes and request-language tokens, prefer 2–4 word phrases, and
    allow only long technical single tokens as a final fallback.
    """
    cleaned = EXACT_IDENTIFIER_RE.sub(" ", query or "")
    stop = {
        *SPARSE_STOPWORDS,
        "code", "rule", "guidance", "notice", "section", "clause",
        "dnv", "mepc", "msc", "abs",
    }
    candidates: list[str] = []
    for named_pattern in (
        r"statement\s+of\s+compliance",
        r"mass\s+roc\s+record",
        r"ammonia\s+fuel\s+preparation\s+room",
    ):
        match = re.search(named_pattern, cleaned, re.I)
        if match:
            candidates.append(re.sub(r"\s+", " ", match.group(0).lower()))
    candidates.extend(
        [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{1,}", cleaned)
        if token.lower() in SHORT_TECHNICAL_FEATURE_TERMS
        ]
    )
    # Preserve boundaries created by Korean text and punctuation.  Flattening
    # every Latin token made artificial phrases such as "smart vessel
    # autonomous remote" outrank the actual named term "Smart Vessel".
    spans = re.findall(
        r"[A-Za-z][A-Za-z0-9-]{2,}(?:\s+[A-Za-z][A-Za-z0-9-]{2,}){0,3}",
        cleaned,
    )
    singletons: list[str] = []
    for span in spans:
        words = [word.lower() for word in span.split() if word.lower() not in stop]
        if len(words) >= 2:
            for size in range(min(4, len(words)), 1, -1):
                for start in range(len(words) - size + 1):
                    phrase = " ".join(words[start : start + size])
                    if len(phrase) >= 12:
                        candidates.append(phrase)
        singletons.extend(word for word in words if len(word) >= 9)
    candidates.extend(singletons)
    return list(dict.fromkeys(candidates))[: max(0, limit)]


def extract_translated_feature_terms(query: str, *, limit: int = 2) -> list[str]:
    """Map distinctive Korean maritime concepts to literal source phrases.

    The embedding model is multilingual, but rare Korean compound nouns can
    still miss an English class-rule paragraph completely.  Accurate recovery
    needs only one selective Chroma ``where_document`` lookup, so use phrases
    that are substantially narrower than generic words such as ``report`` or
    ``test``.  These are retrieval aliases, never answer text.
    """
    bundles = (
        (r"운용\s*경험.{0,16}부족|선박.{0,10}운용\s*경험", "insufficient experience from service on board ships"),
        (r"선상\s*측정.{0,24}(?:최종\s*)?보고서|최종\s*보고서.{0,20}측정", "within two (2) weeks after the job is terminated"),
        (r"선상\s*측정.{0,24}(?:최종\s*)?보고서|최종\s*보고서.{0,20}측정", "in addition to the measured values, the original scantlings, the minimum thickness and the substantial corrosion limits"),
        (r"선상\s*측정.{0,24}(?:최종\s*)?보고서|최종\s*보고서.{0,20}측정", "The report shall include a copy of the certificate of approval of the firm"),
        (r"정격\s*전압.{0,20}(?:제어|계측)|제어.{0,8}계측.{0,16}케이블", "cables for control and instrumentation circuits"),
        (r"소음.{0,8}진동.{0,20}(?:운영자|자격)|측정\s*운영자.{0,12}자격", "category I certificate"),
        (r"소재.{0,24}(?:승인\s*보고서|기록|데이터)|starting\s*material", "manufacture of the starting material"),
        (r"내측.{0,12}접근.{0,18}(?:불가능|추가)|(?:ROV|잠수).{0,12}검사", "not accessible from the inside"),
        (r"유체\s*평형|구조적\s*덕트|교차\s*침수", "cross-flooding by structural ducts"),
        (r"세척.{0,20}(?:효율|시험\s*계획)|미세.{0,12}거대\s*오염", "areas selected for assessing the efficiency of removing"),
        (r"(?:선실\s*구역|객실).{0,28}(?:열\s*감지기|화재\s*감지기|감지기\s*종류)", "Heat detectors shall be used in cabins"),
        (r"언코일링|\bW\s*인증.{0,16}\bNV\s*인증", "If the coils are delivered with W-certificate"),
        (r"크랭크샤프트.{0,24}(?:품질\s*관리|시편)|피로\s*시험.{0,16}시편", "Prior to fatigue testing, the crankshaft"),
        (r"평판\s*제품.{0,24}(?:용접성|항복)|용접성\s*시험", "minimum specified yield stress"),
        (r"대체\s*시스템.{0,24}(?:부식|승인|증빙)|MSC\.288\(87\)", "target useful life of 15 years"),
        (r"윤활\s*방식.{0,20}(?:원주\s*)?속도|원주\s*속도", "Circumferential velocity should be"),
        (r"\bSCR\b.{0,24}(?:형식\s*시험|제출\s*서류)", "exhaust composition, mass flow and temperatures"),
        (r"고소음|소음.{0,12}(?:매우\s*)?높은.{0,12}(?:경보|공간)", "beacon light or similar device"),
        (r"머드가스\s*분리기|액봉.{0,12}트랩", "일반적 굴착운전상태에 대하여는 3 m 이상"),
        (r"\bFRP\b.{0,28}(?:보호용\s*코팅|침식\s*하중)|(?:보호용\s*코팅|침식\s*하중).{0,28}\bFRP\b", "FRP structures with heavy"),
        (r"(?:CGS.{0,40}(?:설치\s*후|대체\s*전력|독립적인\s*전원)|(?:설치\s*후|대체\s*전력|독립적인\s*전원).{0,40}CGS)", "tested at normal seagoing load"),
        (r"최대\s*중간\s*흘수|maximum\s+draught\s+midship", "Checking only the equivalent draught"),
        (r"소프트웨어\s*구성\s*요소.{0,28}(?:기술|설계\s*문서)|software\s+components", "hardware allocation of the software components"),
        (r"로봇\s*용접|완전\s*자동화.{0,16}용접", "shall be re-qualified"),
        (r"데이터\s*수집\s*인프라.{0,24}(?:인증|구성\s*요소)", "individual components or the complete infrastructure"),
        (r"선미관.{0,16}밀봉|stern\s*tube.{0,12}seal", "stern tube sealing devices"),
        (r"(?:전통적인|표준).{0,16}(?:C\s*)?가스\s*운반선|design\s+accelerations", "Alternatively, for standard gas carriers, the envelope accelerations"),
        (r"(?:CA\s*챔버|가스\s*제거).{0,28}(?:진입|질소|완료)", "procedures for checking completed gas freeing prior to entry"),
        (r"구형\s*쉘.{0,24}(?:편평도|곡률)|out-of-roundness", "local outside curvature radius of Ro,l = 1.3"),
        (r"클러치.{0,60}(?:다른\s*시험|시험\s*절차|특수|새로운\s*설계)", "special or new designs may be applied based on a case by case approval"),
        (r"(?:해상|Maritime)\s*LAN.{0,34}(?:다른\s*규칙|준수|인증)", "will not confirm compliance with requirements in other parts of the rules"),
        (r"(?:탱크\s*지지대|hardwood).{0,30}(?:등급|grade|정의)", "Defined by density, lamination"),
        (r"(?:탱크\s*지지대|hardwood).{0,30}(?:등급|grade|정의|변형|variants)", "variants: different numbers of plies"),
        (r"(?:배터리|EES).{0,28}25\s*MWh", "engineering analysis in accordance with IMO MSC/Circ.1002"),
        (r"케이블\s*타이|cable\s+ties", "This CP does not set the design requirements to the cable ties"),
        (r"무거운\s*부품|heavy\s+parts", "120% of the static forces at the maximum amplitude"),
        (r"무거운\s*부품|heavy\s+parts", "Roll or pitch of 15° at 10 sec is assumed to include the load effect from heave, sway and surge motions"),
        (r"무거운\s*부품|heavy\s+parts", "As wind loads are not included"),
        (r"(?:DRILL|시추\s*설비).{0,32}(?:구성|승인|포함)", "approval of complete drilling plant"),
        (r"충격\s*시험.{0,28}(?:요구되지|10\s*mm)|강종.{0,20}10\s*mm", "All steel grades without specified impact toughness requirements"),
        (r"(?:선박평형수처리장치|BWMS).{0,34}(?:생물학적|독성).{0,16}(?:시설|시험)", "Biological efficacy testing and/or toxicity testing shall be carried out by a test facility approved"),
        (r"열수축성\s*튜브|heat\s+shrinkable\s+tubing", "This CP does not set the design requirements to the heat shrinkable tubing"),
        (r"마모\s*저항성\s*코팅|abrasion\s+resistant\s+coatings", "Variants: colour variants, thinned variants and similar"),
        (r"\bRSCS\+\b.{0,34}\bbq\b|횡방향.{0,18}가속도\s*계수", "shall be reduced by a route reduction factor, froute"),
        (r"\bIHM\b.{0,32}유지관리\s*매뉴얼|maintenance\s+manual.{0,24}minimum", "policy about how MD and/or SDOC is handled for spare parts"),
        (r"와이어\s*인발용.{0,20}(?:결정립|알루미늄|니오븀|바나듐)", "austenite grain size of the bar used for drawing of wire"),
        (r"서비스\s*문서.{0,30}(?:책임|제3자|인증)|300,?000\s*USD", "under any circumstance be limited to 300,000 USD"),
        (r"\bBWMS\b.{0,28}(?:시험\s*기간|기록해야|기록\s*항목)", "The test facility shall maintain a record of"),
        (r"강성.{0,12}(?:상대\s*)?감쇠|dynamic\s+stiffness.{0,16}damping", "Dynamic stiffness (K) in compression and/or shear and relative damping"),
        (r"satisfaction\s+of\s+the\s+Administration|이행\s*지침.{0,20}후속\s*일정", "biennial agenda of the III Sub-Committee for the 2028-2029 biennium"),
        (r"(?:수중\s*방사\s*소음|\bURN\b).{0,32}(?:이해관계자|관점|효과)", "nothing herein should be construed as implying complete consensus"),
        (r"해양\s*플라스틱\s*쓰레기.{0,28}2026|2026.{0,20}(?:VGMFG|플라스틱)", "Other resources under development for publication in 2026 include an e-learning"),
        (
            r"(?:MSC\s*111.{0,40})?연료.{0,20}(?:안전|위험평가).{0,20}(?:관련|결과|추려)",
            "interim guidelines for the safety of ships using hydrogen as fuel",
        ),
        (
            r"(?:MSC\s*111.{0,40})?연료.{0,20}(?:안전|위험평가).{0,20}(?:관련|결과|추려)",
            "Revised Interim Recommendations for carriage of liquefied hydrogen in bulk",
        ),
        (
            r"(?:MSC\s*111.{0,40})?연료.{0,20}(?:안전|위험평가).{0,20}(?:관련|결과|추려)",
            "approve the draft work plan",
        ),
        (
            r"(?:MSC\s*111.{0,40})?연료.{0,20}(?:안전|위험평가).{0,20}(?:관련|결과|추려)",
            "wind-assisted propulsion",
        ),
        (r"Uniting\s+Efforts\s+for\s+a\s+Quieter\s+Ocean", "Canada and Panama"),
        (r"화석\s*LNG.{0,28}(?:기본|WtT|배출\s*계수)", "must reflect the full diversity of global LNG production and trade flows"),
        (r"북동\s*대서양.{0,24}(?:ECA|배출통제구역).{0,60}(?:승인|회의)", "to MARPOL Annex VI relating to the North-East Atlantic ECA"),
        (r"INTERTANKO.{0,28}(?:수질|데이터베이스|항구별)", "These include practical parameters associated with system bypasses"),
        (r"RenovaCalc.{0,32}(?:사탕수수|운송|재할당)", "average transport radius of 17.4 miles (26.85 km)"),
        (r"(?:IMO\s*선박\s*연료|Ship\s+Fuel\s+Oil).{0,36}(?:\bRO\b|권한|익명화)", "the IMO number and ship's Administration flag or any duly authorized organization"),
        (r"러시아.{0,80}MASS.{0,40}(?:환경|기술)|러시아.{0,80}환경.{0,60}MASS", "The KOPORYE is a newly built bilge water removal ship"),
        (r"ICS\s+CII.{0,32}(?:표본|정확도|제출)", "Submission of data to the ICS CII Data Collection System is voluntary"),
        (r"inTank\s+BWTS.{0,30}(?:살균제|biocide|공급)", "two means of delivering the biocide (sodium hypochlorite)"),
        (r"해상\s*기반.{0,26}플라스틱.{0,26}(?:단일\s*연구|글로벌\s*평가)", "recommended adopting a step-wise approach and to gradually build up"),
        (r"2024년.{0,24}\bCII\b.{0,28}(?:보고하지|미보고|사유)", "Some Administrations informed the Secretariat on the status of ships, for which CII data had not been reported"),
        (r"2024년.{0,24}(?:대기\s*중\s*)?CO2.{0,24}(?:농도|전\s*산업화)", "423.9 ± 0.2 parts per million"),
        (r"LNG\s*운반선.{0,30}(?:상류|upstream).{0,22}(?:가정|영향)", "limited scope to adjust fuel use or operational profiles"),
        (r"크루즈\s*여객선.{0,28}\bcgHRS\b|\bcgHRS\b.{0,24}(?:개발|단계|목표)", "further consideration and finalization of the development of the cgHRS metric"),
        (r"평형수\s*관리\s*시스템.{0,28}갱신\s*검사|renewal\s+survey.{0,18}(?:years|간격)", "A renewal survey at intervals specified by the Administration, but not exceeding five years"),
        (r"\bBWRB\b.{0,30}최종\s*총량|final\s+total\s+quantity.{0,24}(?:평형수|tank)", "aggregated volume across all ballast water tanks or only the remaining volume"),
        (r"대형\s*원양.{0,30}(?:항구\s*기반|비상\s*조치)|port-based.{0,20}mandatory\s+contingency", "should not be considered a mandatory contingency measure for large ocean going ships"),
        (r"총\s*운송\s*작업.{0,26}(?:즉시|CII|2026)|total\s+transport\s+work.{0,24}(?:CII|immediate)", "Such an application was neither discussed nor agreed upon by the Committee"),
        (r"\bWAEMU\b.{0,28}(?:Council|Assembly|협력)", "A 34 endorsed the Council's decision to approve the request for cooperation"),
        (r"(?:RNDN|Dorsal\s+de\s+Nasca).{0,30}(?:PSSA|근거|지정)", "Given its uniqueness, relative isolation, and associated degree of endemism"),
        (r"중간\s*재배\s*옥수수.{0,30}(?:환경|에탄올|특성)|intermediate\s+crop\s+corn", "planted immediately after the primary soybean harvest, utilizing the same land"),
        (r"Technical\s+Body.{0,30}(?:최소|며칠|장소)", "at the Palais des Nations in Geneva and/or online"),
        (
            r"\bBBNJ\b.{0,80}(?:2026|2027|협의|조정|의무|메커니즘)|"
            r"(?:협의|조정)\s*의무.{0,60}\bBBNJ\b",
            "articulate and operationalize a clear internal mechanism",
        ),
        (r"IACS\s+Rec\.165.{0,30}(?:대상|목적|적용)", "addresses designers, shipyards, technical managers responsible for calculations"),
        (r"IACS\s+Rec\.165.{0,30}(?:대상|목적|적용)", "It also applies to deviations from CSR and other requirements"),
        (r"IACS\s+Rec\.165.{0,30}(?:대상|목적|적용)", "addressed to Classification Societies"),
        (r"IAPH.{0,32}(?:Cyber\s+Resilience|사이버\s*복원력).{0,24}(?:배경|개발)", "recognizing the vast opportunities and risks of operating in an increasingly digital landscape"),
        (r"\bEBP\b.{0,30}(?:전담\s*그룹|수집된\s*데이터|분석)|MASS\s+Data\s+Review\s+Group", "MASS Data Review Group' in the form of a working group (WG) or correspondence group (CG)"),
        (r"(?:신기술|대체\s*연료).{0,30}(?:안전\s*규제|작업\s*계획|GHG)", "approve the draft work plan"),
        (r"소프트웨어\s*업데이트.{0,34}(?:MASS\s+Record|MASS\s+ROC\s+Record)", "the feasibility of software lifecycle maintenance shall be considered"),
        (r"MASS\s+Code.{0,30}(?:제4장|정의|Definitions).{0,20}(?:상태|작업)", "chapter 4 (Definitions) of the draft MASS Code has not yet been finalized"),
        (r"(?:원자력\s*FPU|nuclear\s+FPU).{0,32}(?:즉시|비강제적|안전\s*지침)", "construction of one FPU takes about four years"),
        (r"감사된\s*규칙.{0,30}(?:GBS|검증\s*프로세스)|audited\s+rules.{0,24}GBS", "should not be addressed within the framework of the GBS verification process"),
        (r"Factual\s+Statements.{0,32}(?:기항국|RO|활용|목적)", "as a component of their RO oversight programme"),
        (r"암모니아\s*연료\s*준비실.{0,70}(?:일본|문구|수정)", "replace the words 'the two valves' with 'the two shut-off valves'"),
        (r"(?:연료|오일)\s*필터.{0,34}(?:엔진\s*정지|SOLAS|세척|정비)", "future amendment to SOLAS regulation II-1/27 could be considered"),
        (r"BC-A.{0,50}(?:최대\s*화물\s*밀도|3\.0\s*t/m)", "BC-A(Hold Nos."),
        (r"펌프타워.{0,28}(?:유한\s*요소|모델링|경계조건)", "펌프타워 구조는 판, 쉘(shell) 혹은 보 요소"),
        (r"(?:특수\s*구역|Ro-Ro\s*공간).{0,36}(?:상부와\s*하부|A-30|램프|도어)", "this deck shall be of \"A-30\" standard while any ramps and doors"),
        (r"가청경보장치.{0,28}전원공급", "전원공급원은 2개 이상이어야 하며"),
        (r"간헐적으로\s*침수.{0,24}(?:도선\s*연결부|보호\s*등급)", "간헐적으로 침수가 되는 갑판상의 연결부는 최소 요건으로 IEC 60529에 따른 IP 67"),
        (r"\bDHT\(CSR\).{0,34}(?:종방향|두께\s*측정|Longitudinal)", "For DHT(CSR)"),
        (r"준설선.{0,28}연차검사.{0,24}(?:감소된\s*건현|확인)", "호퍼도어 개방 및 준설밸브 폐쇄를 위한 비상제어장치"),
        (r"선체\s*거더.{0,28}(?:국부\s*하중|전단|굽힘\s*모멘트)", "The following local loads are to be applied for the calculation of hull girder shear"),
        (r"(?:구조적\s*구성|화물창마다).{0,90}(?:3D|유한요소|화물창)|structural\s+configuration.{0,40}cargo\s+spaces", "Two or more 3D F.E. models may be required"),
        (r"(?:화물창|펌프룸).{0,30}빌지\s*수위.{0,20}(?:모니터링|알람)", "Cargo holds are to be fitted with a bilge water-level monitoring system"),
        (r"\bSCR\b.{0,30}(?:엔진\s*성능|설치\s*및\s*운용|배압|호환)", "Installation and operation of an SCR system is to be compatible with the engine"),
        (r"(?:최종\s*통합|선상\s*테스트).{0,34}(?:자율|원격\s*제어|수동\s*제어권)", "possible for the Operator to retake control of the action from the autonomous function at all times"),
        (r"(?:상태\s*모니터링.*포함되지\s*않은|Planned\s+Maintenance).{0,34}(?:이행\s*검사|확인)", "The onboard personnel are familiar with the PM Program"),
        (r"앵커\s*핸들링\s*윈치.{0,32}(?:제출|설계\s*정보|계산)|anchor\s+handling\s+winch.{0,22}(?:information|calculations)", "including maximum line pull, winch brake holding capacity, rendering load"),
    )
    out: list[str] = []
    for pattern, phrase in bundles:
        if re.search(pattern, query or "", re.I):
            out.append(phrase)
            if len(out) >= max(0, limit):
                break
    return out


def feature_fallback_relevance_score(
    query: str, document: str, feature_term: str = ""
) -> float:
    """Reward exact feature hits that also cover the requested facets."""
    question = str(query or "")
    text = str(document or "")
    score = 1.35
    if feature_term and " " in feature_term.strip():
        # A named multiword phrase is far more selective than a short acronym
        # (``Gulf of California`` versus ``LNG``).  Preserve that distinction
        # when both exact-literal lookups enter the same document route.
        score += min(0.45, 0.12 + len(feature_term.split()) * 0.08)
    if re.search(r"요건|요구|조건|기준|하여야|해야", question):
        if re.search(r"요건|요구|조건|기준|하여야|해야|필요", text):
            score += 0.16
    if re.search(r"예외|제외|면제|적용하지", question):
        if re.search(r"예외|면제", text):
            score += 0.24
        elif re.search(r"다만|제외|참작|경감|감소할 수|적용하지", text):
            score += 0.18
    if feature_term and re.search(
        rf"(?:^|\n)\s*(?:조문\s*\d+절[^\n]*\n\s*)?"
        rf"(?:\d+[.)]?\s*)?{re.escape(feature_term)}(?:\s|$|\()",
        text,
    ):
        score += 0.55
    return score


def _literal_variants(identifier: str) -> list[str]:
    variants = [identifier]
    if re.match(r"^(?:MEPC|MSC)\s*\d", identifier, re.I):
        variants.extend(
            [
                identifier.replace("/", "-"),
                identifier.replace("-", "/"),
                re.sub(r"\s+", " ", identifier),
            ]
        )
    if re.match(r"^t(?:corr|c\s*[12])$", identifier, re.I):
        variants.extend([identifier.lower(), identifier.upper()])
    return list(dict.fromkeys(v for v in variants if v))[:4]


def _identifier_matches_filename(identifier: str, file_name: str) -> bool:
    """Match document codes while tolerating IMO slash/hyphen filename forms."""
    imo_identifier = re.fullmatch(
        r"\s*(MEPC|MSC)\s*(\d{1,3})(?:\s*[/.-]\s*([A-Z0-9]+))+(?:\s*)",
        identifier or "",
        re.I,
    )
    if imo_identifier:
        ident_parts = re.findall(r"[A-Za-z]+|\d+", identifier or "")
        if not re.match(r"\s*(MEPC|MSC)\b", file_name or "", re.I):
            return False
        file_parts = re.findall(r"[A-Za-z]+|\d+", (file_name or "")[:100])
        return [part.lower() for part in ident_parts] == [
            part.lower() for part in file_parts[: len(ident_parts)]
        ]
    dnv_identifier = re.fullmatch(
        r"DNV-(?:CG|RP|RU|OS|CP|SI)-[A-Z0-9-]+", identifier or "", re.I
    )
    if dnv_identifier:
        prefix = re.match(
            r"\s*(DNV-(?:CG|RP|RU|OS|CP|SI)-[A-Z0-9-]+)", file_name or "", re.I
        )
        return bool(prefix and prefix.group(1).lower() == identifier.lower())
    ident = re.sub(r"[^a-z0-9]+", "", (identifier or "").lower())
    name = re.sub(r"[^a-z0-9]+", "", (file_name or "").lower())
    return bool(ident and len(ident) >= 3 and ident in name)


def _query_exact_identifier_hits(
    collection,
    query: str,
    *,
    where: dict | None,
    candidate_documents: list[str] | None = None,
    limit_per_term: int = 40,
) -> tuple[dict[str, Any], dict[str, float], list[str], list[str]]:
    """Use Chroma's existing documents for cheap exact-code recall.

    This is deliberately limited to distinctive identifiers.  It avoids the
    271k-row global BM25 path and does not create or modify any embedding.
    """
    identifiers = extract_exact_identifiers(query)
    feature_fallback_terms: list[str] = []
    korean_terms = extract_sparse_feature_terms(query, limit=1)
    latin_terms = extract_sparse_latin_terms(query, limit=3)
    signals = analyze_query(query)
    expansion_noise = {
        "report", "outcome", "resolution", "substantive", "mass code",
        "maritime autonomous", "low-flashpoint", "alternative fuel",
        "section 15", "engines supplied", "intersessional working group",
    }
    expanded_phrases: list[str] = []
    for raw_term in signals.expanded_terms:
        term = re.sub(r"\s+", " ", str(raw_term or "").strip())
        term_lower = term.lower()
        if (
            term_lower in expansion_noise
            or len(term) < 10
            or " " not in term
            or re.search(r"\b(?:mepc|msc)\s*[-/]?\s*\d", term, re.I)
        ):
            continue
        expanded_phrases.append(term)
        if len(expanded_phrases) >= 2:
            break
    # A parenthesized/named multiword English term is usually the literal used
    # in the source PDF (for example ``supply-based carbon intensity``).  It is
    # more selective than a Korean request token such as ``감소했습니까`` and
    # must therefore get the single bounded fallback lookup first.  Keep
    # curated query expansions next, then Korean compounds, then long Latin
    # singletons.  This ordering changes only the zero/weak-hit rescue path;
    # normal dense retrieval keeps its original cost.
    multiword_latin = [term for term in latin_terms if " " in term.strip()]
    singleton_latin = [term for term in latin_terms if " " not in term.strip()]
    premise_verification = bool(
        re.search(
            r"전제.{0,40}(?:맞는지|검증|확인|틀리)|"
            r"(?:맞는지|사실인지).{0,24}(?:검증|확인)|"
            r"(?:부결|승인|채택|거절).{0,40}(?:맞는지|전제)",
            query or "",
            re.I,
        )
    )
    translated_limit = (
        3
        if re.search(
            r"(?:MSC\s*111.{0,40})?연료.{0,20}(?:안전|위험평가).{0,20}(?:관련|결과|추려)",
            query,
            re.I,
        )
        else 2
    )
    translated_terms = extract_translated_feature_terms(query, limit=translated_limit)
    fallback_candidates = (
        [*korean_terms, *translated_terms, *multiword_latin, *expanded_phrases, *singleton_latin]
        if premise_verification
        else [*translated_terms, *multiword_latin, *expanded_phrases, *korean_terms, *singleton_latin]
    )
    for term in fallback_candidates:
        # Dense presence is a useful shortcut for Korean recovery.  Named
        # Latin phrases still get a literal score even when they appear deep
        # in the raw candidate set; otherwise they can be ranked away before
        # evidence planning sees them.
        if any(term.lower() in str(document or "").lower() for document in candidate_documents or []):
            if term in korean_terms and not premise_verification:
                continue
        feature_fallback_terms.append(term)
        # One distinctive phrase is enough to recover a missed concept and
        # matches the latency contract: this is exactly one extra document
        # lookup in the failure case, rather than a second sparse search mode.
        if len(feature_fallback_terms) >= 1:
            break
    out: dict[str, list[list]] = {
        "ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]
    }
    scores: dict[str, float] = {}
    seen: set[str] = set()
    positions: dict[str, int] = {}
    literal_targets = [(identifier, False) for identifier in identifiers]
    literal_targets.extend((term, True) for term in feature_fallback_terms)
    for identifier, is_feature_fallback in literal_targets:
        if is_feature_fallback:
            variants = [identifier]
            original = re.search(re.escape(identifier), query, re.I)
            if original:
                variants.append(original.group(0))
            variants.extend(FEATURE_LITERAL_TRANSLATIONS.get(identifier.lower(), ()))
            # Curated multiword expansions preserve the source's intended case
            # and should cost one lookup.  Case variants remain useful for a
            # short identifier/acronym directly supplied by the user.
            if " " not in identifier.strip():
                variants.extend([identifier.title(), identifier.upper()])
            variants = list(dict.fromkeys(value for value in variants if value))[:4]
        else:
            variants = _literal_variants(identifier)
        if re.fullmatch(r"tc\s*orr", identifier, re.I) and re.search(
            r"(?:기호.{0,12}(?:뜻|의미)|무엇을\s*뜻|무슨\s*뜻|정의)",
            query,
            re.I,
        ):
            variants = list(
                dict.fromkeys(
                    variants
                    + [
                        "국부 부식추가 tcorr",
                        "국부 부식추가",
                        "tcorr :",
                    ]
                )
            )
        for variant in variants:
            try:
                kwargs: dict[str, Any] = {
                    "where_document": {"$contains": variant},
                    "limit": max(limit_per_term, 80) if is_feature_fallback else limit_per_term,
                    "include": ["metadatas", "documents"],
                }
                if where:
                    kwargs["where"] = where
                raw = collection.get(**kwargs)
            except Exception:
                continue
            for cid, meta, document in zip(
                raw.get("ids") or [],
                raw.get("metadatas") or [],
                raw.get("documents") or [],
            ):
                text = str(document or "")
                exact = variant.lower() in text.lower()
                file_match = (not is_feature_fallback) and _identifier_matches_filename(
                    identifier, str((meta or {}).get("file_name") or "")
                )
                literal_score = (
                    feature_fallback_relevance_score(query, text, identifier)
                    if is_feature_fallback and exact
                    else 1.8
                    if file_match
                    else 1.0
                    if exact
                    else 0.7
                )
                scores[cid] = max(scores.get(cid, 0.0), literal_score)
                literal_distance = (
                    max(0.08, 0.16 - (literal_score - 1.35) * 0.20)
                    if is_feature_fallback and exact
                    else 0.08
                    if file_match
                    else 0.22
                    if exact
                    else 0.35
                )
                if cid in seen:
                    pos = positions[cid]
                    out["distances"][0][pos] = min(
                        float(out["distances"][0][pos]), literal_distance
                    )
                    continue
                seen.add(cid)
                positions[cid] = len(out["ids"][0])
                out["ids"][0].append(cid)
                # Synthetic distance only seeds the candidate pool; final order
                # is recalculated with metadata, document and sparse scores.
                out["distances"][0].append(literal_distance)
                out["metadatas"][0].append(meta or {})
                out["documents"][0].append(text)
    return out, scores, identifiers, feature_fallback_terms


def _document_route_candidates(
    *,
    query: str,
    ids: list[str],
    distances: dict[str, float],
    metadatas: dict[str, dict],
    documents: dict[str, str],
    signals: QuerySignals,
    clause_hints: list[str],
    priority_doc_ids: set[str],
    exact_scores: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate chunk evidence into a ranked document shortlist."""
    grouped: dict[str, dict[str, Any]] = {}
    for global_rank, cid in enumerate(ids, 1):
        meta = metadatas.get(cid) or {}
        doc_id = str(meta.get("doc_id") or "").strip()
        if not doc_id:
            continue
        adjusted = adjusted_distance(
            distances[cid],
            query=query,
            document=documents.get(cid, ""),
            meta=meta,
            clause_hints=clause_hints,
            signals=signals,
            priority_doc_ids=priority_doc_ids,
        )
        item = grouped.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "file_name": str(meta.get("file_name") or ""),
                "source": str(meta.get("source") or ""),
                "best_distance": adjusted,
                "raw_best_distance": float(distances[cid]),
                "first_rank": global_rank,
                "hit_count": 0,
                "exact_hit_score": 0.0,
            },
        )
        item["hit_count"] += 1
        item["exact_hit_score"] = max(
            float(item["exact_hit_score"]),
            float((exact_scores or {}).get(cid, 0.0)),
        )
        item["best_distance"] = min(float(item["best_distance"]), adjusted)
        item["raw_best_distance"] = min(
            float(item["raw_best_distance"]), float(distances[cid])
        )
        item["first_rank"] = min(int(item["first_rank"]), global_rank)

    ranked: list[dict[str, Any]] = []
    for item in grouped.values():
        file_overlap = lexical_overlap(
            query,
            f"{item['file_name']} {item['doc_id']}",
            signals.expanded_terms,
        )
        support = min(0.10, math.log1p(int(item["hit_count"])) * 0.025)
        priority = item["doc_id"] in priority_doc_ids
        identifier_file_match = any(
            _identifier_matches_filename(identifier, str(item["file_name"]))
            for identifier in extract_exact_identifiers(query)
        )
        route_score = (
            float(item["best_distance"])
            - support
            - file_overlap * 0.28
            - (0.22 if priority else 0.0)
            - min(1.60, float(item["exact_hit_score"]) * 0.70)
            - (0.45 if identifier_file_match else 0.0)
        )
        item.update(
            {
                "score": route_score,
                "file_overlap": file_overlap,
                "priority": priority,
                "identifier_file_match": identifier_file_match,
            }
        )
        ranked.append(item)
    ranked.sort(key=lambda item: (float(item["score"]), int(item["first_rank"])))
    return ranked


def _sparse_query_terms(query: str, signals: QuerySignals) -> list[tuple[str, float]]:
    weighted: dict[str, float] = {}
    exact = {value.lower() for value in extract_exact_identifiers(query)}
    for raw in SPARSE_TOKEN_RE.findall(query or ""):
        term = raw.lower().strip()
        if len(term) < 2 or term in SPARSE_STOPWORDS:
            continue
        weight = 8.0 if term in exact else (3.2 if re.search(r"\d|[-./]", term) else 1.0)
        if raw.isupper() and len(raw) >= 3:
            weight = max(weight, 2.0)
        elif len(term) >= 5:
            weight = max(weight, 1.25)
        weighted[term] = max(weighted.get(term, 0.0), weight)
        if re.fullmatch(r"[가-힣]{3,}", raw):
            for suffix in KOREAN_QUERY_SUFFIXES:
                if term.endswith(suffix) and len(term) - len(suffix) >= 2:
                    stem = term[: -len(suffix)]
                    weighted[stem] = max(weighted.get(stem, 0.0), weight * 1.1)
                    break
    for raw in signals.expanded_terms[:16]:
        term = str(raw or "").lower().strip()
        if len(term) >= 3:
            weighted[term] = max(weighted.get(term, 0.0), 0.55)
    return list(weighted.items())[:32]


def rank_scoped_sparse_rows(
    query: str,
    signals: QuerySignals,
    ids: list[str],
    metadatas: list[dict],
    documents: list[str],
    *,
    top_k: int = 18,
) -> list[tuple[float, str, dict, str]]:
    """Rank chunks inside routed documents using lightweight exact terms.

    The same section number can occur in several chapters of a rule book.  A
    plain term count therefore overvalues repeated identifiers (for example
    ``902``) and generic words such as "procedure".  Document-local IDF,
    heading matches, and short query phrases make the topic next to the section
    number decisive without requiring a new embedding index.
    """
    terms = _sparse_query_terms(query, signals)
    if not terms:
        return []
    row_texts = [
        f"{(meta or {}).get('file_name', '')} {document or ''}".lower()
        for meta, document in zip(metadatas, documents)
    ]
    row_count = max(1, len(row_texts))
    doc_frequency = {
        term: sum(1 for text in row_texts if term in text)
        for term, _weight in terms
    }

    # Phrase matching keeps one-syllable connectors such as Korean "및" even
    # though they are intentionally excluded from unigram scoring.
    raw_tokens = [
        token.lower()
        for token in re.findall(
            r"[A-Za-z]+(?:[A-Za-z0-9]*)(?:[-./][A-Za-z0-9]+)*|\d+(?:\.\d+)?|[가-힣]+",
            query or "",
        )
    ]
    phrases: list[tuple[str, float]] = []
    phrase_values: set[str] = set()
    for size, weight in ((3, 10.0), (2, 2.5)):
        for start in range(max(0, len(raw_tokens) - size + 1)):
            phrase = " ".join(raw_tokens[start : start + size]).strip()
            if len(phrase) >= 5 and phrase not in phrase_values:
                phrases.append((phrase, weight))
                phrase_values.add(phrase)

    topic_anchors = []
    for match in re.finditer(
        r"\d{3,4}\s*(?:section|clause|절|조)\s*([A-Za-z가-힣][A-Za-z가-힣0-9_-]{1,24})",
        query or "",
        re.IGNORECASE,
    ):
        anchor = match.group(1).lower().strip()
        if anchor not in {"검사", "inspection", "requirements", "요건"}:
            topic_anchors.append(anchor)

    scored: list[tuple[float, str, dict, str]] = []
    for cid, meta, document, text in zip(ids, metadatas, documents, row_texts):
        heading = text[:220]
        score = 0.0
        hits = 0
        for term, weight in terms:
            if term in text:
                hits += 1
                frequency = doc_frequency.get(term, row_count)
                idf = math.log(1.0 + (row_count + 1.0) / (frequency + 1.0))
                effective_weight = weight * min(2.8, 0.75 + 0.38 * idf)
                score += effective_weight
                if " " in term or re.search(r"\d|[-./]", term):
                    score += effective_weight * 0.35
                if term in heading:
                    score += effective_weight * 0.45
        for phrase, phrase_weight in phrases:
            if phrase in text:
                score += phrase_weight * (1.45 if phrase in heading else 1.0)
        for anchor in topic_anchors:
            if anchor in heading:
                score += 12.0
            elif anchor in text:
                score += 4.0
        if hits:
            score += min(0.8, hits * 0.08)
            scored.append((score, cid, meta or {}, document or ""))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def _load_scoped_sparse_rows(
    collection,
    doc_ids: list[str],
    *,
    base_where: dict | None,
) -> tuple[list[str], list[dict], list[str]]:
    if not doc_ids:
        return [], [], []
    ids: list[str] = []
    metadatas: list[dict] = []
    documents: list[str] = []
    per_doc_limit = max(700, SCOPED_SPARSE_MAX_ROWS // max(1, len(doc_ids)))
    for doc_id in doc_ids:
        where = _merge_where(base_where, {"doc_id": doc_id})
        try:
            raw = collection.get(
                where=where,
                limit=per_doc_limit,
                include=["metadatas", "documents"],
            )
        except Exception:
            continue
        ids.extend(raw.get("ids") or [])
        metadatas.extend(raw.get("metadatas") or [])
        documents.extend(raw.get("documents") or [])
    return ids, metadatas, documents


def extract_clause_hints(query: str) -> list[str]:
    hints: list[str] = []
    for groups in CLAUSE_IN_QUERY_RE.finditer(query):
        for g in groups.groups():
            if g and g.isdigit() and len(g) >= 3:
                hints.append(g)
    seen: set[str] = set()
    out: list[str] = []
    for h in hints:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def _meta_clause(meta: dict) -> str:
    return str(meta.get("clause_number") or meta.get("article_number") or "").strip()


def lexical_overlap(query: str, document: str, extra_terms: list[str] | None = None) -> float:
    q_tokens = {t for t in TOKEN_RE.findall(query.lower()) if len(t) > 1}
    if extra_terms:
        for term in extra_terms:
            q_tokens.update(t for t in TOKEN_RE.findall(term.lower()) if len(t) > 1)
    if not q_tokens:
        return 0.0
    d_tokens = {t for t in TOKEN_RE.findall(document.lower()) if len(t) > 1}
    return len(q_tokens & d_tokens) / len(q_tokens)


def _file_name(meta: dict) -> str:
    return str(meta.get("file_name") or meta.get("doc_id") or "").lower()


def _doc_type(meta: dict) -> str:
    dt = str(meta.get("doc_type") or "").strip()
    if dt:
        return dt
    return classify_imo_filename(str(meta.get("file_name") or ""))


def _metadata_boosts(
    *,
    meta: dict,
    document: str,
    signals: QuerySignals,
    priority_doc_ids: set[str],
    query: str = "",
) -> tuple[float, float]:
    boost = 0.0
    penalty = 0.0
    fname = _file_name(meta)
    doc_id = str(meta.get("doc_id") or "")
    doc_head = (document or "")[:800].lower()
    combined = f"{fname} {doc_head}"
    doc_type = _doc_type(meta)

    if doc_id and doc_id in priority_doc_ids:
        boost += PRIORITY_DOC_BOOST

    tier = tier_for_query(
        doc_type,
        wants_summary=signals.wants_summary,
        wants_outcome=signals.wants_outcome,
        wants_agenda=signals.wants_agenda,
    )
    if tier >= 0:
        boost += tier
    else:
        penalty += abs(tier)

    for prefix in session_file_prefixes(signals):
        if prefix.replace("/", "-") in fname.replace("/", "-") or prefix in fname:
            boost += SESSION_FILE_BOOST
            break

    for prefix in topic_agenda_prefixes(signals):
        if prefix in fname:
            boost += TOPIC_PREFIX_BOOST
            break

    if doc_type == "subcommittee_report" and (signals.wants_summary or signals.wants_outcome):
        penalty += SUBCOMM_PENALTY

    if signals.wants_agenda:
        if doc_type == "agenda":
            boost += 0.12

    if signals.wants_rule_lookup:
        ql = (query or "").lower()
        if "dnv-cg-0264" in fname or "cg-0264" in fname or "cg 0264" in fname:
            boost += RULE_FILENAME_BOOST + 0.12
        if "notice no.1" in fname or "notice no. 1" in fname:
            boost += RULE_FILENAME_BOOST
        if any(k in fname for k in ("autonomous", "remotely-operated", "remotely operated", "smart-vessel", "smart vessel")):
            boost += 0.14
        if any(k in combined for k in ("autonomous", "remotely operated", "smart vessel", "notation")):
            boost += 0.08
        if re.search(r"rp-c\d", fname) and "cg-0264" not in fname:
            if any(k in ql for k in ("smart", "autonomous", "자율", "vessel", "mass")):
                penalty += 0.10
        if DOC_CODE_CROSSREF_RE.search(doc_head):
            penalty += 0.16
        society = str(signals.class_society_hint or "").upper()
        chunk_source = str(meta.get("source") or "").upper()
        if society and chunk_source in CLASS_RULE_SOURCES:
            if chunk_source == society:
                boost += 0.20
            else:
                penalty += 0.22

    for hint in signals.rule_doc_hints:
        if hint.lower() in combined:
            boost += EXPANDED_TERM_BOOST

    for term in signals.expanded_terms:
        t = term.lower()
        if len(t) >= 4 and t in combined:
            boost += EXPANDED_TERM_BOOST

    if "mass" in signals.topics and "mass" in combined:
        boost += EXPANDED_TERM_BOOST
    if "ghg" in signals.topics and "ghg" in combined:
        boost += EXPANDED_TERM_BOOST
    if "alt_fuel" in signals.topics:
        if any(k in combined for k in ("low-flashpoint", "low flashpoint", "section 15", "alternative fuel", "igf")):
            boost += TOPIC_PREFIX_BOOST
    if "igc" in signals.topics and "igc" in combined:
        boost += EXPANDED_TERM_BOOST

    if signals.meeting_outcome_question:
        from meeting_outcome_retrieval import is_comparison_question, meeting_outcome_metadata_adjustment

        mo_boost, mo_penalty = meeting_outcome_metadata_adjustment(
            meta=meta,
            document=document,
            signals=signals,
            question=query,
            is_comparison=is_comparison_question(query),
        )
        boost += mo_boost
        penalty += mo_penalty

    return boost, penalty


def adjusted_distance(
    distance: float,
    *,
    query: str,
    document: str,
    meta: dict,
    clause_hints: list[str],
    signals: QuerySignals | None = None,
    priority_doc_ids: set[str] | None = None,
) -> float:
    score = float(distance)
    meta_clause = _meta_clause(meta)
    doc_head = (document or "")[:400]
    sig = signals or analyze_query(query)
    prio = priority_doc_ids or set()

    for hint in clause_hints:
        if meta_clause == hint:
            score -= CLAUSE_EXACT_BOOST
        elif hint in doc_head and is_article_clause_number(hint):
            score -= CLAUSE_IN_TEXT_BOOST
        if f"{hint}." in doc_head or f"{hint}절" in doc_head:
            score -= REFERENCE_LINE_BOOST

    meta_boost, meta_penalty = _metadata_boosts(
        meta=meta, document=document, signals=sig, priority_doc_ids=prio, query=query
    )
    score -= meta_boost
    score += meta_penalty
    score -= LEXICAL_BOOST_SCALE * lexical_overlap(query, document, sig.expanded_terms)
    score -= _definition_clause_boost(query, document)
    return score


def _definition_clause_boost(query: str, document: str) -> float:
    """Prefer defining clauses over later formula references for symbol asks."""
    q = str(query or "")
    if not re.search(
        r"(?:기호.{0,12}(?:뜻|의미)|무엇을\s*뜻|무슨\s*뜻|정의|what.{0,12}mean)",
        q,
        re.I,
    ):
        return 0.0
    body = str(document or "")
    boost = 0.0
    if re.search(
        r"(?:정의(?:된|한다|는)|이라\s*함은|을\s*말한다|means|is\s+defined\s+as)",
        body,
        re.I,
    ):
        boost += 0.10
    if re.search(r"\btc\s*orr\b|\bt_corr\b", q, re.I):
        if re.search(r"국부\s*부식추가.{0,40}tc\s*orr|tc\s*orr.{0,40}국부\s*부식추가", body, re.I):
            boost += 0.30
        elif re.search(r"tc\s*orr\s*[:=：]", body, re.I):
            boost += 0.20
    return min(0.40, boost)


def _merge_where(*clauses: dict | None) -> dict | None:
    parts = [c for c in clauses if c]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}


def _meta_matches_where(meta: dict, where: dict | None) -> bool:
    if not where:
        return True
    meta = meta or {}
    if "$and" in where:
        return all(_meta_matches_where(meta, clause) for clause in where["$and"])
    if "$or" in where:
        return any(_meta_matches_where(meta, clause) for clause in where["$or"])
    for key, expected in where.items():
        actual = meta.get(key)
        if isinstance(expected, dict):
            for op, val in expected.items():
                if op == "$in":
                    if actual not in val:
                        return False
                elif op == "$eq":
                    if actual != val:
                        return False
                elif op == "$ne":
                    if actual == val:
                        return False
                else:
                    return False
        elif actual != expected:
            return False
    return True


def _filter_chroma_raw(raw: dict, where: dict | None) -> dict:
    if not where or not raw.get("ids"):
        return raw
    out: dict[str, list[list]] = {
        "ids": [],
        "distances": [],
        "metadatas": [],
        "documents": [],
    }
    # Chroma returns one result row per query embedding.  Accurate mode uses
    # both the expanded and the original question embeddings, so preserve and
    # filter every row rather than silently discarding all but the first.
    row_count = len(raw.get("ids") or [])
    for row_index in range(row_count):
        out["ids"].append([])
        out["distances"].append([])
        out["metadatas"].append([])
        out["documents"].append([])
        for cid, dist, meta, doc in zip(
            (raw.get("ids") or [[]])[row_index],
            (raw.get("distances") or [[]])[row_index],
            (raw.get("metadatas") or [[]])[row_index],
            (raw.get("documents") or [[]])[row_index],
        ):
            if _meta_matches_where(meta or {}, where):
                out["ids"][row_index].append(cid)
                out["distances"][row_index].append(dist)
                out["metadatas"][row_index].append(meta)
                out["documents"][row_index].append(doc)
    return out


def safe_chroma_query(
    collection,
    *,
    query_embeddings: list[list[float]],
    n_results: int,
    where: dict | None = None,
) -> dict[str, Any]:
    """Chroma query with retries when the vector index returns stale/missing ids."""
    attempts: list[tuple[int, dict | None]] = [
        (n_results, where),
        (min(n_results, 40), where),
        (min(n_results, 40), None),
    ]
    last_exc: Exception | None = None
    for n, clause in attempts:
        try:
            kwargs: dict[str, Any] = {
                "query_embeddings": query_embeddings,
                "n_results": max(1, n),
            }
            if clause:
                kwargs["where"] = clause
            raw = collection.query(**kwargs)
            if clause is None and where is not None:
                return _filter_chroma_raw(raw, where)
            return raw
        except Exception as exc:
            last_exc = exc
            if "error finding id" not in str(exc).lower():
                raise
    if last_exc is not None:
        raise last_exc
    return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}


def query_with_hybrid_ranking(
    collection,
    query: str,
    query_vector: list[float],
    *,
    alternate_query_vectors: list[list[float]] | None = None,
    top_k: int = 5,
    fetch_k: int | None = None,
    source: str | None = None,
    doc_id: str | None = None,
    timing=None,
) -> dict[str, Any]:
    """Hierarchical text retrieval over the existing Chroma embeddings.

    Stage 1 retrieves broad chunks and aggregates them to documents. Stage 2
    searches only the routed documents and optionally injects exact identifier
    hits plus a lightweight document-scoped sparse rerank. No index rebuild is
    performed.
    """
    clause_hints = extract_clause_hints(query)
    signals = analyze_query(query)
    priority_ids = exact_doc_ids_for_query(query)
    for priority_id in priority_doc_ids_for_signals(signals):
        if priority_id not in priority_ids:
            priority_ids.append(priority_id)
    if doc_id is None:
        for priority_id in _resolve_priority_rule_doc_ids(collection, signals, query):
            if priority_id not in priority_ids:
                priority_ids.append(priority_id)
    priority_set = set(priority_ids)
    n_fetch = fetch_k or max(top_k * 15, 80)
    base_where = _merge_where(
        {"source": source.upper()} if source else None,
        {"doc_id": doc_id} if doc_id else None,
    )

    merged_ids: list[str] = []
    merged_dist: dict[str, float] = {}
    merged_meta: dict[str, dict] = {}
    merged_doc: dict[str, str] = {}
    exact_scores: dict[str, float] = {}
    scoped_sparse_ratios: dict[str, float] = {}

    def absorb(raw: dict) -> None:
        if not raw.get("ids"):
            return
        for ids, distances, metadatas, documents in zip(
            raw.get("ids") or [],
            raw.get("distances") or [],
            raw.get("metadatas") or [],
            raw.get("documents") or [],
        ):
            for cid, dist, meta, doc in zip(ids, distances, metadatas, documents):
                if cid in merged_dist:
                    merged_dist[cid] = min(merged_dist[cid], float(dist))
                else:
                    merged_ids.append(cid)
                    merged_dist[cid] = float(dist)
                    merged_meta[cid] = meta or {}
                    merged_doc[cid] = doc or ""

    if timing is not None and hasattr(timing, "mark"):
        timing.mark("t_vector_search_start")

    raw_vector = safe_chroma_query(
        collection,
        query_embeddings=[query_vector, *(alternate_query_vectors or [])],
        n_results=min(n_fetch, 150),
        where=base_where,
    )
    absorb(raw_vector)

    # A priority registry used to affect ranking only when a document happened
    # to survive the global vector top-N.  Topic-heavy sessions contain many
    # near-duplicate submissions, so authoritative reports such as MSC 111/12
    # could be absent and therefore impossible to boost.  Fetch a small number
    # of best chunks inside each routed priority document before ranking.  This
    # reuses the existing embeddings and adds no indexing step.
    if doc_id is None:
        for priority_doc_id in priority_ids[:8]:
            priority_where = _merge_where(base_where, {"doc_id": priority_doc_id})
            priority_raw = safe_chroma_query(
                collection,
                query_embeddings=[query_vector],
                n_results=3,
                where=priority_where,
            )
            absorb(priority_raw)

    hierarchy_enabled = _hierarchical_text_enabled()
    if hierarchy_enabled:
        exact_raw, exact_scores, exact_identifiers, feature_fallback_terms = _query_exact_identifier_hits(
            collection,
            query,
            where=base_where,
            candidate_documents=list(merged_doc.values()),
        )
        absorb(exact_raw)
    else:
        exact_identifiers = []
        feature_fallback_terms = []

    if timing is not None and hasattr(timing, "mark"):
        timing.mark("t_vector_search_end")
        timing.mark("t_metadata_filter_start")

    if priority_ids:
        batch = priority_ids[:20]
        try:
            routed = safe_chroma_query(
                collection,
                query_embeddings=[query_vector],
                n_results=min(20, n_fetch),
                where=_merge_where(base_where, {"doc_id": {"$in": batch}}),
            )
            absorb(routed)
        except Exception:
            for pid in batch[:8]:
                try:
                    one = safe_chroma_query(
                        collection,
                        query_embeddings=[query_vector],
                        n_results=8,
                        where=_merge_where(base_where, {"doc_id": pid}),
                    )
                    absorb(one)
                except Exception:
                    pass

    if clause_hints:
        for hint in clause_hints[:3]:
            try:
                clause_where = {
                    "$or": [
                        {"clause_number": hint},
                        {"article_number": hint},
                    ]
                }
                filtered = safe_chroma_query(
                    collection,
                    query_embeddings=[query_vector],
                    n_results=min(15, n_fetch),
                    where=_merge_where(base_where, clause_where),
                )
                absorb(filtered)
            except Exception:
                pass

    document_route: dict[str, Any] = {
        "enabled": False,
        "selected_doc_ids": [],
        "candidates": [],
        "confidence": 0.0,
        "exact_identifiers": exact_identifiers,
        "feature_fallback_terms": feature_fallback_terms,
        "scoped_sparse_used": False,
    }
    selected_doc_ids: list[str] = []
    if hierarchy_enabled and doc_id and (clause_hints or signals.wants_rule_lookup):
        sparse_ids, sparse_metas, sparse_docs = _load_scoped_sparse_rows(
            collection,
            [doc_id],
            base_where=base_where,
        )
        sparse_ranked = rank_scoped_sparse_rows(
            query,
            signals,
            sparse_ids,
            sparse_metas,
            sparse_docs,
            top_k=max(24, top_k * 4),
        )
        best_sparse_score = max((item[0] for item in sparse_ranked), default=1.0)
        doc_best = min(merged_dist.values(), default=0.58)
        for sparse_score, cid, meta, document in sparse_ranked:
            synthetic_distance = max(
                0.0,
                doc_best + 0.05 - min(0.28, sparse_score * 0.03),
            )
            scoped_sparse_ratios[cid] = max(
                scoped_sparse_ratios.get(cid, 0.0),
                sparse_score / max(best_sparse_score, 1e-9),
            )
            absorb(
                {
                    "ids": [[cid]],
                    "distances": [[synthetic_distance]],
                    "metadatas": [[meta]],
                    "documents": [[document]],
                }
            )
        document_route.update(
            {
                "enabled": True,
                "selected_doc_ids": [doc_id],
                "confidence": 1.0,
                "scoped_sparse_used": bool(sparse_ranked),
            }
        )
    if hierarchy_enabled and not doc_id and merged_ids:
        doc_candidates = _document_route_candidates(
            query=query,
            ids=merged_ids,
            distances=merged_dist,
            metadatas=merged_meta,
            documents=merged_doc,
            signals=signals,
            clause_hints=clause_hints,
            priority_doc_ids=priority_set,
            exact_scores=exact_scores,
        )
        max_docs = (
            DOCUMENT_ROUTE_MAX_DOCS_BROAD
            if signals.wants_summary or signals.wants_outcome or signals.meeting_outcome_question
            else DOCUMENT_ROUTE_MAX_DOCS
        )
        selected_doc_ids = [str(item["doc_id"]) for item in doc_candidates[:max_docs]]
        if selected_doc_ids:
            stage2 = safe_chroma_query(
                collection,
                query_embeddings=[query_vector],
                n_results=min(
                    max(DOCUMENT_ROUTE_STAGE2_FETCH, top_k * 10),
                    120,
                ),
                where=_merge_where(
                    base_where,
                    {"doc_id": {"$in": selected_doc_ids}},
                ),
            )
            absorb(stage2)

            # Only pay the scoped lexical cost for exact-code/clause or rule
            # questions, where dense embeddings are known to miss literal terms.
            use_scoped_sparse = bool(
                exact_identifiers
                or feature_fallback_terms
                or clause_hints
                or signals.wants_rule_lookup
            )
            if use_scoped_sparse:
                sparse_ids, sparse_metas, sparse_docs = _load_scoped_sparse_rows(
                    collection,
                    selected_doc_ids,
                    base_where=base_where,
                )
                sparse_ranked = rank_scoped_sparse_rows(
                    query,
                    signals,
                    sparse_ids,
                    sparse_metas,
                    sparse_docs,
                    top_k=max(18, top_k * 3),
                )
                best_sparse_score = max((item[0] for item in sparse_ranked), default=1.0)
                best_by_doc = {
                    str(item["doc_id"]): float(item["raw_best_distance"])
                    for item in doc_candidates
                }
                for sparse_score, cid, meta, document in sparse_ranked:
                    routed_doc = str((meta or {}).get("doc_id") or "")
                    synthetic_distance = max(
                        0.0,
                        best_by_doc.get(routed_doc, 0.58)
                        + 0.05
                        - min(0.20, sparse_score * 0.025),
                    )
                    scoped_sparse_ratios[cid] = max(
                        scoped_sparse_ratios.get(cid, 0.0),
                        sparse_score / max(best_sparse_score, 1e-9),
                    )
                    # A sparse-exact hit may already be present in the dense
                    # pool with a poor distance.  Re-absorb it so ``min`` can
                    # lower that distance; otherwise the lexical signal is
                    # recorded but cannot overcome the stale dense score.
                    absorb(
                        {
                            "ids": [[cid]],
                            "distances": [[synthetic_distance]],
                            "metadatas": [[meta]],
                            "documents": [[document]],
                        }
                    )
                document_route["scoped_sparse_used"] = bool(sparse_ranked)

        margin = (
            float(doc_candidates[1]["score"]) - float(doc_candidates[0]["score"])
            if len(doc_candidates) > 1
            else 0.25
        )
        first_priority = bool(doc_candidates and doc_candidates[0].get("priority"))
        confidence = min(1.0, 0.42 + max(0.0, min(0.30, margin)) + (0.22 if first_priority else 0.0))
        document_route.update(
            {
                "enabled": True,
                "selected_doc_ids": selected_doc_ids,
                "confidence": round(confidence, 3),
                "candidates": [
                    {
                        "doc_id": item["doc_id"],
                        "file_name": item["file_name"],
                        "source": item["source"],
                        "score": round(float(item["score"]), 4),
                        "hit_count": int(item["hit_count"]),
                        "priority": bool(item["priority"]),
                        "exact_hit_score": round(float(item["exact_hit_score"]), 3),
                        "identifier_file_match": bool(item["identifier_file_match"]),
                    }
                    for item in doc_candidates[:max_docs]
                ],
            }
        )

    if timing is not None and hasattr(timing, "mark"):
        timing.mark("t_metadata_filter_end")
        timing.mark("t_rerank_start")

    doc_route_rank = {value: rank for rank, value in enumerate(selected_doc_ids)}

    def final_distance(cid: str) -> float:
        score = adjusted_distance(
            merged_dist[cid],
            query=query,
            document=merged_doc[cid],
            meta=merged_meta[cid],
            clause_hints=clause_hints,
            signals=signals,
            priority_doc_ids=priority_set,
        )
        routed_doc = str((merged_meta.get(cid) or {}).get("doc_id") or "")
        rank = doc_route_rank.get(routed_doc)
        if rank is not None and rank < len(DOCUMENT_ROUTE_BOOSTS):
            score -= DOCUMENT_ROUTE_BOOSTS[rank]
        score -= min(0.90, exact_scores.get(cid, 0.0) * 0.38)
        # IDF and phrase matches can make several lexical scores exceed a
        # fixed cap.  A within-document ratio preserves the decisive gap
        # between the strongest clause and a generic introduction.
        score -= scoped_sparse_ratios.get(cid, 0.0) * 0.32
        return score

    ranked_all = sorted(merged_ids, key=final_distance)
    ranked = ranked_all[:top_k]

    # The fallback lookup is only issued when dense retrieval needs literal
    # recovery.  Retain a small quota of those literal hits in the final list;
    # otherwise many generic chunks from the correctly routed document can
    # still fill top-k and hide the very clause the extra lookup recovered.
    if feature_fallback_terms and top_k:
        def is_feature_hit(cid: str) -> bool:
            text = str(merged_doc.get(cid) or "").lower()
            for term in feature_fallback_terms:
                variants = [term, *FEATURE_LITERAL_TRANSLATIONS.get(term.lower(), ())]
                if any(str(variant).lower() in text for variant in variants):
                    return True
            return False

        feature_ranked = [cid for cid in ranked_all if is_feature_hit(cid)]
        if feature_ranked:
            premise_question = bool(
                re.search(r"전제|맞는지.{0,24}(?:검증|확인)|사실인지", query or "", re.I)
            )
            quota = min(4 if premise_question else 2, top_k, len(feature_ranked))
            retained = feature_ranked[:quota]
            ranked = (retained + [cid for cid in ranked_all if cid not in retained])[:top_k]
            document_route["feature_fallback_retained"] = retained

    # An explicitly named multi-document comparison must retain evidence from
    # every named document.  Global top-k can otherwise be filled by several
    # near-duplicate intro chunks from only one ABS guide.  The candidates are
    # already loaded above; this is a zero-I/O quota over the existing ranking.
    direct_rule_ids = _direct_priority_rule_doc_ids(query, signals)
    if len(direct_rule_ids) >= 2 and top_k >= len(direct_rule_ids):
        per_doc = max(1, top_k // len(direct_rule_ids))
        forced: set[str] = set()
        for direct_doc_id in direct_rule_ids:
            matches = [
                cid
                for cid in ranked_all
                if str((merged_meta.get(cid) or {}).get("doc_id") or "")
                == direct_doc_id
            ]
            forced.update(matches[:per_doc])
        if forced:
            selected = [cid for cid in ranked_all if cid in forced]
            selected.extend(cid for cid in ranked_all if cid not in forced)
            ranked = selected[:top_k]
            document_route["direct_document_quota"] = {
                "doc_ids": direct_rule_ids,
                "per_doc": per_doc,
                "retained": {
                    direct_doc_id: sum(
                        1
                        for cid in ranked
                        if str((merged_meta.get(cid) or {}).get("doc_id") or "")
                        == direct_doc_id
                    )
                    for direct_doc_id in direct_rule_ids
                },
            }

    if timing is not None and hasattr(timing, "mark"):
        timing.mark("t_rerank_end")

    return {
        "ids": [ranked],
        "distances": [[merged_dist[cid] for cid in ranked]],
        "metadatas": [[merged_meta[cid] for cid in ranked]],
        "documents": [[merged_doc[cid] for cid in ranked]],
        "clause_hints": clause_hints,
        "query_signals": signals,
        "priority_doc_ids": priority_ids,
        "document_route": document_route,
        "final_scores": [[final_distance(cid) for cid in ranked]],
    }


def _enrich_query_standard(query: str) -> str:
    """Base query enrichment without meeting-outcome branch."""
    hints = extract_clause_hints(query)
    signals = analyze_query(query)
    parts: list[str] = []
    if hints:
        parts.extend(f"{h}절 {h}." for h in hints[:2])
    if signals.expanded_terms:
        parts.extend(signals.expanded_terms[:12])
    if re.search(r"minimum[- ]risk\s+condition|\bMRC\b", query or "", re.I):
        parts.extend(
            [
                "fallback state",
                "operational envelope",
                "acceptable risk",
                "last resort fallback state",
            ]
        )
    if signals.wants_summary or signals.wants_outcome:
        parts.extend(["outcome", "executive summary", "report", "resolution", "decision"])
    if not parts:
        return query
    return f"{' '.join(parts)} {query}".strip()


def enrich_query_for_embedding(query: str, model_name: str) -> str:
    """Prepend clause hints and cross-lingual/session terms for E5."""
    signals = analyze_query(query)
    if signals.meeting_outcome_question:
        from meeting_outcome_retrieval import enrich_meeting_outcome_query

        return enrich_meeting_outcome_query(query, model_name)
    from table_retrieval import enrich_table_query_for_embedding, is_table_question

    if is_table_question(query):
        return enrich_table_query_for_embedding(_enrich_query_standard(query))
    return _enrich_query_standard(query)
