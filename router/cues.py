"""Keyword cues, overlap rules, and speech-act patterns."""
from __future__ import annotations

import re

_AC = r"(?<![A-Za-z0-9_])"
_AZ = r"(?![A-Za-z0-9_])"

OPS_PATTERNS: list[tuple[str, float]] = [
    (r"운항\s*상태|현재\s*항차|이전\s*항차|이번\s*항차|올해\s*연간|연간\s*실적", 3.0),
    (r"지금\s*(배|선박|위치)|배\s*(가\s*)?(어디|위치)|어디\s*(야|있음|떠)|항해\s*중", 2.8),
    (rf"{_AC}CII{_AZ}|씨아이아이|시아이아이|씨아아이|탄소집약|attained|required\s*cii|등급\s*[A-E]", 2.5),
    (r"Noon\s*Report|눈\s*리포트|눈리포트|MRV|배출량|CO2e?|CH4|FOC|FGC|연료\s*소모", 2.5),
    (r"기름값|기름\s*(얼마나|얼마|썼|소비|소모)|연비|연료\s*(얼마나|소비|사용|썼|소모)", 2.3),
    (rf"스피드|속도|{_AC}SOG{_AZ}|속력|위도|경도|항적|항차\s*분석|sensor|센서", 2.0),
    (
        rf"{_AC}YTD{_AZ}|연초\s*누적|운항\s*거리|항해\s*거리|항차\s*수|항차수|"
        r"Ballast|Laden|H2521|voyage|distance_nm",
        2.0,
    ),
    (r"유류|LNG\s*(소모|소비|사용|연료)|가스\s*소모|oil_flow|gas_flow", 1.5),
    (r"보고서\s*(만들|생성|뽑아)|워드|docx|브리핑", 1.2),
    (r"우리\s*(배|선박|호선)|이\s*선박|온보드|선내\s*(데이터|로그)|올해\s*(운항|항차|거리|배출)", 1.0),
]

RAG_PATTERNS: list[tuple[str, float]] = [
    (
        rf"{_AC}(?:MEPC|MSC|MASS){_AZ}|멥시|엠이피시|아이모|"
        rf"{_AC}IMO{_AZ}|회의\s*(결과|주요|동향|결정)|"
        rf"{_AC}GHG{_AZ}|온실가스|중기조치|well-?to-?wake|Strategy",
        3.0,
    ),
    (
        rf"선급|Rule/?Guidance|{_AC}(?:DNV|ABS|LR){_AZ}|디엔브이|"
        rf"KR\s*(?:규칙|Rule|\d+\s*편|표)|{_AC}KR{_AZ}\s*(?:Rule|\d)|"
        rf"RU\s*-?\s*SHIP|rules?\s+for\s+steel",
        3.0,
    ),
    (
        r"규정|지침|요건|조항|\bclause\b|\bchapter\b|Guidance|가이던스|가이드|"
        r"guidance\s*note|규제|스마트\s*(?:십|기능)|원격\s*검사|remote\s*survey",
        2.0,
    ),
    (r"MARPOL|마르폴|SOLAS|Net-?Zero|GFI|SEEMP|EEXI|DCS|GISIS", 2.0),
    (
        r"표\s*(에서|에|의|질의|검색|기준)|표에|정기검사|평형수|밸러스트\s*탱크|선령|"
        r"검사\s*주기|검사주기|검사\s*(범위|기준|표|일반)|"
        r"(?:survey\s*)?interval|intermediate\s*survey|annual\s*survey|"
        r"docking\s*survey|special\s*survey|continuous\s*survey|class\s*survey|"
        r"tank\s*inspection|개방검사|두께계측",
        2.5,
    ),
    # 선급 표/구조 계산형 — MEPC·KR 단어가 없어도 문서(rag)다.
    (
        r"최소\s*두께|판두께|요구(?:되는)?\s*(?:최소\s*)?두께|부식추가|\btcorr\b|"
        r"선박\s*길이|\bL\s*[<>≤≥=]|L이\s*\d|항복\s*(?:응력|강도)|인장\s*강도|"
        r"화학성분|재료기호|용접용?\s*재료|기계적\s*성질|용접강|"
        r"\d+\s*m\s*(?:미만|이상|이하)|미만일\s*때|이상일\s*때|"
        r"N\s*/?\s*mm|표\s*\d+(?:\.\d+)*|"
        r"화물창|화물탱크|평형수탱크|reporting\s*요건",
        2.6,
    ),
    (r"문서|PDF|회의록|circular|resolution|WP\.?\d", 1.5),
    (r"자율운항|대체연료.{0,12}안전|환경규제\s*대응|최신\s*동향", 2.0),
    (r"규칙\s*(이|은|뭐|어디)|뭐라고\s*(돼|되어)|요건이\s*뭐|기준이\s*뭐", 1.8),
    # Term/definition lookups belong in documents, not chat clarify.
    (
        r"(?:의\s*)?정의(?:는|가|란)?|무슨\s*뜻|의미(?:는|가)|용어\s*(?:정의|설명)|"
        r"substantial\s*corrosion|과도한\s*부식",
        2.2,
    ),
    (r"이사회|총회|워킹그룹|작년에\s*.*회의", 1.5),
]

GREET_PATTERN = re.compile(r"^(안녕|헬로|hello|\bhi\b)[\s!?.]*$", flags=re.IGNORECASE)
# 봇 능력·소개. 도메인 단어가 섞여도 chat으로 고정할 때 사용.
# '둘 다 알려줘'(내용 요청)와 구분되도록 '가능/할 수' 단서를 요구한다.
CAPABILITY_PATTERN = re.compile(
    r"(뭐|무엇)\s*할\s*수|할\s*수\s*있|기능\s*(알려|소개|설명)|도움말|\bhelp\b|"
    r"(운항|문서).{0,20}가능|둘\s*다.{0,10}가능|가능.{0,12}(운항|문서|둘)|"
    r"사용법|뭘\s*물어보면|범위가\s*뭐|할\s*수\s*없는|처음인데|어떻게\s*써|"
    r"어떤\s*데이터(?:를\s*)?보니|데이터를\s*보니",
    flags=re.IGNORECASE,
)
IDENTITY_PATTERN = re.compile(
    r"너\s*누구|누구야|너는\s*누구|너는\s*뭐|너\s*뭐야|자기소개|너에\s*대해|"
    r"뭐\s*할\s*수|할\s*수\s*있|기능\s*(알려|소개|설명)|도움말|\bhelp\b|"
    r"이\s*봇|이\s*에이전트|정체가\s*뭐|이름이\s*뭐|누가\s*만들|너는\s*사람|"
    r"사용법|뭘\s*물어보면|범위가\s*뭐|할\s*수\s*없는|처음인데|어떻게\s*써|"
    r"(운항|문서).{0,20}가능|둘\s*다.{0,10}가능",
    flags=re.IGNORECASE,
)
META_PATTERN = re.compile(
    r"라우터|의도\s*분류|데이터\s*경로|자동\s*라우팅|"
    r"뭘로\s*구분|어떻게\s*구분|어떻게\s*나뉘|어떤\s*경로|"
    r"시스템\s*(구조|설명|뭐)|어떻게\s*동작|DB가\s*몇|몇\s*개\s*DB|엔진이\s*뭐",
    flags=re.IGNORECASE,
)
THANKS_PATTERN = re.compile(
    r"^(고마워(?:요)?|감사합니다|감사|땡큐|thanks)[\s!?.]*$",
    flags=re.IGNORECASE,
)
FOLLOWUP_PATTERN = re.compile(
    r"^(그럼|그래서|그거|그건|그것도|더|자세히|이어서|그리고|또|계속|응|네)|"
    r"좀\s*더|자세히\s*(알려|설명|봐)|그건\s*뭐|그거\s*뭐",
    flags=re.IGNORECASE,
)
DEIXIS_PATTERN = re.compile(r"그거|그건|그\s*규정|그\s*문서|그\s*회의|그\s*등급|그\s*기준|이어서")
SWITCH_OPS_PATTERN = re.compile(
    r"운항(\s*(쪽|으로|데이터|DB))?|우리\s*배로|숫자로|항차로\b",
    flags=re.IGNORECASE,
)
SWITCH_RAG_PATTERN = re.compile(
    r"문서(\s*(쪽|으로))?|규정으로|선급으로|회의로|표로|\bRAG\b",
    flags=re.IGNORECASE,
)
SWITCH_HYBRID_PATTERN = re.compile(
    r"둘\s*다|같이\s*(봐|알려|정리)|양쪽|hybrid",
    flags=re.IGNORECASE,
)
DUAL_MARK_PATTERN = re.compile(r"같이|둘\s*다|동시에|양쪽")
COMPARE_PATTERN = re.compile(
    r"기준으로\s*(우리|이\s*선박|우리\s*배)|규정\s*기준으로|"
    r"우리\s*(배|선박).{0,12}(규정|규제|기준|준수)|"
    r"(규정|규제|회의).{0,12}우리\s*(배|선박)|대비해서|준수하",
    flags=re.IGNORECASE,
)
OOS_PATTERN = re.compile(
    r"날씨\s*(어때|좋|나쁘)|환율|주식|비트코인|요리|레시피|축구\s*경기|야구\s*점수|"
    r"영화\s*추천|농담\s*해|심심해|점심\s*뭐|기분\s*어때",
    flags=re.IGNORECASE,
)
# Rule/table-shaped questions that should not fall through to chat clarify
# when keyword cue scores are still zero (LLM / fallback safety net).
TECHNICAL_RAG_SHAPE_PATTERN = re.compile(
    r"최소\s*두께|판두께|요구(?:되는)?\s*(?:최소\s*)?두께|부식추가|\btcorr\b|"
    r"선박\s*길이|\bL\s*[<>≤≥=]|L이\s*\d|항복\s*(?:응력|강도)|인장\s*강도|"
    r"화학성분|재료기호|용접용?\s*재료|기계적\s*성질|용접강|AH\s*\d{2}|"
    r"화물창|화물탱크|평형수\s*탱크|reporting|정기검사|선령|"
    r"표\s*\d+|N\s*/?\s*mm|\d+\s*m\s*(?:미만|이상|이하)|미만일\s*때|"
    r"검사\s*(범위|선정|요건|주기)|두께계측|개방검사|"
    r"yield|tensile|corrosion\s*addition|min(?:imum)?\s*thickness|"
    r"ship\s*length|ballast\s*tank|cargo\s*(?:hold|tank)|"
    # Definition / glossary shapes (class-rule terms, not ship ops).
    r"(?:의\s*)?정의(?:는|가|란)?|무슨\s*뜻|의미(?:는|가)|용어|"
    r"substantial\s*corrosion|과도한\s*부식|허용\s*부식|부식\s*여유|"
    r"glossary|what\s+is\s+(?:substantial\s+)?corrosion",
    flags=re.IGNORECASE,
)
# Soft ops shape: live ship ops without needing strong keyword hit.
TECHNICAL_OPS_SHAPE_PATTERN = re.compile(
    r"지금\s*(배|선박|위치|스피드)|배\s*어디|기름\s*얼마나|연료\s*(소모|소비|썼)|"
    r"올해\s*(CII|항차|운항)|현재\s*항차|Noon|MRV",
    flags=re.IGNORECASE,
)
DOC_FRAME_PATTERN = re.compile(
    r"MEPC|MSC|선급|규정|규제|조항|회의|Rule|지침|가이드|가이던스|동향|"
    r"문서에서|PDF|circular|MARPOL|SOLAS|resolution|워킹그룹|이사회|총회|"
    rf"{_AC}(?:IMO|GHG|KR){_AZ}|대체연료|원격\s*검사|survey|표에|표\s*에서|"
    r"최소\s*두께|판두께|선박\s*길이|부식추가|화학성분|재료기호|용접강|"
    r"정기검사|화물창|화물탱크|reporting|"
    r"(?:의\s*)?정의|substantial\s*corrosion|과도한\s*부식|용어",
    flags=re.IGNORECASE,
)
SHIP_FRAME_PATTERN = re.compile(
    r"우리|현재|올해|항차|계산|등급|온보드|이\s*선박|잘\s*되고|지키고|준수|"
    r"알려줘|얼마|얼마나",
    flags=re.IGNORECASE,
)
SHIP_STRONG_PATTERN = re.compile(
    r"우리\s*(배|선박|호선)|이\s*선박|온보드|현재\s*항차|올해\s*(우리|CII|배출)|"
    r"잘\s*되고|지키고\s*있|준수하",
    flags=re.IGNORECASE,
)
REPORT_OPS_PATTERN = re.compile(r"보고서|워드|docx|브리핑|Noon|눈\s*리포트|MRV", flags=re.IGNORECASE)
DOC_REPORT_PATTERN = re.compile(r"회의|MEPC|MSC|문서|규정|선급|circular|동향", flags=re.IGNORECASE)
OVERLAP_TERM_PATTERN = re.compile(
    r"CII|씨아이아이|시아이아이|씨아아이|탄소집약|SEEMP|배출|EEXI|DCS",
    flags=re.IGNORECASE,
)

TOPIC_PATTERNS: list[tuple[str, str]] = [
    ("cii", r"CII|탄소집약|attained|required\s*cii"),
    ("voyage", r"항차|운항\s*상태|\bvoyage\b|YTD|운항\s*거리"),
    ("fuel", r"연료|기름|FOC|LNG|연비|유류"),
    ("position", r"위치|어디|위도|경도|스피드|SOG|속력"),
    ("mepc", r"MEPC|GHG|온실가스"),
    ("msc", r"MSC|\bMASS\b"),
    ("class", r"선급|DNV|ABS|\bLR\b|KR\s*Rule|KR\s*규칙|KR\s*\d+\s*편|RU\s*-?\s*SHIP"),
    (
        "table",
        r"표|검사\s*주기|평형수|밸러스트|선령|정기검사|survey|개방검사|"
        r"최소\s*두께|판두께|선박\s*길이|부식추가|tcorr|화학성분|재료기호",
    ),
    ("seemp", r"SEEMP|EEXI|MARPOL|SOLAS"),
    ("report", r"Noon|MRV|보고서"),
]

ENTITY_PATTERNS = [
    r"H2521",
    r"MEPC\s*\d+",
    r"MSC\s*\d+",
    r"선령\s*\d+\s*년?",
    r"올해|작년|현재\s*항차|이전\s*항차",
]


def _score(question: str, patterns: list[tuple[str, float]]) -> float:
    q = question or ""
    total = 0.0
    for pat, weight in patterns:
        if re.search(pat, q, flags=re.IGNORECASE):
            total += weight
    return total


def score_question(question: str) -> tuple[float, float]:
    ops, rag = _score(question, OPS_PATTERNS), _score(question, RAG_PATTERNS)
    return adjust_overlap(question, ops, rag)


def adjust_overlap(question: str, ops: float, rag: float) -> tuple[float, float]:
    q = question or ""
    doc_frame = bool(DOC_FRAME_PATTERN.search(q))
    ship_strong = bool(SHIP_STRONG_PATTERN.search(q))
    ship_frame = bool(SHIP_FRAME_PATTERN.search(q))
    overlap = bool(OVERLAP_TERM_PATTERN.search(q))

    if REPORT_OPS_PATTERN.search(q) and DOC_REPORT_PATTERN.search(q):
        if not re.search(r"Noon|눈\s*리포트|MRV|운항", q, flags=re.IGNORECASE):
            ops = max(0.0, ops - 1.2)
            rag += 1.2

    if overlap and doc_frame and not ship_strong:
        rag += 1.5
        ops = max(0.0, ops - 1.0)
    elif overlap and ship_strong and not doc_frame:
        ops += 1.5
        rag = max(0.0, rag - 1.0)
    elif overlap and ship_frame and not doc_frame:
        ops += 1.0

    if re.search(r"문서에서", q) and overlap:
        rag += 1.0
        ops = max(0.0, ops - 1.5)

    # 연료 단어(LNG 등)가 문서·안전 맥락이면 ops가 아니라 rag.
    if re.search(r"대체연료|안전\s*관련|가이드|가이던스|문서", q, flags=re.IGNORECASE) and doc_frame:
        if not re.search(r"소모|소비|FOC|FGC|얼마나\s*썼|연비", q, flags=re.IGNORECASE):
            rag += 1.2
            ops = max(0.0, ops - 1.5)

    # 명시적 dual 표지 + 양쪽 소스 단어면 점수를 채워 hybrid 후보로 올린다.
    if has_dual_mark(q):
        if re.search(r"운항|항차|Noon|MRV|배출|CII|온보드|이\s*선박", q, flags=re.IGNORECASE):
            ops = max(ops, 1.5)
        if re.search(r"문서|규정|선급|회의|표|MEPC|MSC|가이드", q, flags=re.IGNORECASE):
            rag = max(rag, 1.5)

    return ops, rag


def has_dual_mark(question: str) -> bool:
    return bool(DUAL_MARK_PATTERN.search(question or ""))


def has_compare_frame(question: str) -> bool:
    return bool(COMPARE_PATTERN.search(question or ""))


def extract_topics(question: str) -> list[str]:
    q = question or ""
    found: list[str] = []
    for name, pat in TOPIC_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE):
            found.append(name)
    return found


def extract_entities(question: str) -> list[str]:
    q = question or ""
    found: list[str] = []
    for pat in ENTITY_PATTERNS:
        found.extend(m.group(0) for m in re.finditer(pat, q, flags=re.IGNORECASE))
    return found


def is_followup_text(question: str) -> bool:
    q = (question or "").strip()
    if FOLLOWUP_PATTERN.search(q) or DEIXIS_PATTERN.search(q):
        return True
    return len(q) <= 12 and bool(re.search(r"그거|그건|그럼|더|자세히|응|네", q))


def looks_like_technical_rag(question: str) -> bool:
    """True when the utterance is a class-rule / table / structural lookup."""
    q = (question or "").strip()
    if not q or OOS_PATTERN.search(q):
        return False
    if CAPABILITY_PATTERN.search(q) or IDENTITY_PATTERN.search(q) or GREET_PATTERN.search(q):
        return False
    if META_PATTERN.search(q) or THANKS_PATTERN.search(q):
        return False
    return bool(TECHNICAL_RAG_SHAPE_PATTERN.search(q))


def looks_like_technical_ops(question: str) -> bool:
    q = (question or "").strip()
    if not q or OOS_PATTERN.search(q):
        return False
    if looks_like_technical_rag(q) and not TECHNICAL_OPS_SHAPE_PATTERN.search(q):
        return False
    return bool(TECHNICAL_OPS_SHAPE_PATTERN.search(q))
