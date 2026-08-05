"""
질문 의도 라우터 — chat / ops / rag.

문장 전체를 외우지 않는다. 단서(cue)만 보고 경로를 고른다.
모르는 입력은 RAG로 보내지 않고 chat에서 되묻는다.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Literal

from prompts.router_prompt import ROUTER_SYSTEM_PROMPT, build_router_user_prompt

RouteKind = Literal["ops", "rag", "chat", "hybrid", "ambiguous"]
FinalRoute = Literal["ops", "rag", "chat", "hybrid"]
ChatMode = Literal["identity", "greeting", "meta", "clarify", "dual", "oos"]
PersistentRoute = Literal["ops", "rag", "hybrid"]

# \b is Unicode-aware in Python 3, so "MEPC에서" / "CII규제" would miss.
_AC = r"(?<![A-Za-z0-9_])"
_AZ = r"(?![A-Za-z0-9_])"

OPS_PATTERNS: list[tuple[str, float]] = [
    (r"운항\s*상태|현재\s*항차|이전\s*항차|올해\s*연간|연간\s*실적", 3.0),
    (r"지금\s*(배|선박|위치)|배\s*(가\s*)?(어디|위치)|어디\s*(야|있음|떠)|항해\s*중", 2.8),
    (rf"{_AC}CII{_AZ}|탄소집약|attained|required\s*cii|등급\s*[A-E]", 2.5),
    (r"Noon\s*Report|눈\s*리포트|눈리포트|MRV|배출량|CO2e?|CH4|FOC|FGC|연료\s*소모", 2.5),
    (r"기름\s*(얼마나|얼마|썼|소비|소모)|연비|연료\s*(얼마나|소비|사용|썼|소모)", 2.3),
    (rf"스피드|속도|{_AC}SOG{_AZ}|속력|위도|경도|항적|항차\s*분석|sensor|센서", 2.0),
    (r"Ballast|Laden|H2521|voyage|항해\s*거리|distance_nm", 1.5),
    (r"유류|LNG|가스\s*소모|oil_flow|gas_flow", 1.5),
    (r"보고서\s*(만들|생성|뽑아)|워드|docx|브리핑", 1.2),
    (r"우리\s*(배|선박|호선)|이\s*선박|온보드|선내\s*(데이터|로그)", 1.0),
]

RAG_PATTERNS: list[tuple[str, float]] = [
    (rf"{_AC}(?:MEPC|MSC|MASS){_AZ}|IMO\s*회의|회의\s*(결과|주요|동향|결정)", 3.0),
    (rf"선급|Rule/?Guidance|{_AC}(?:DNV|ABS|LR){_AZ}|{_AC}KR{_AZ}\s*Rule|KR\s*규칙", 3.0),
    (r"규정|지침|요건|조항|\bclause\b|\bchapter\b|Guidance|규제", 2.0),
    (r"MARPOL|SOLAS|Net-?Zero|GFI|SEEMP|EEXI|DCS|GISIS", 2.0),
    (r"표\s*(에서|질의|검색|기준)|정기검사|평형수|밸러스트\s*탱크|선령|검사\s*(주기|범위|기준|표)", 2.5),
    (r"문서|PDF|회의록|circular|resolution|WP\.?\d", 1.5),
    (r"자율운항|대체연료\s*안전|환경규제\s*대응|최신\s*동향", 2.0),
    (r"규칙\s*(이|은|뭐|어디)|뭐라고\s*(돼|되어)|요건이\s*뭐|기준이\s*뭐", 1.8),
    (r"이사회|총회|워킹그룹|작년에\s*.*회의", 1.5),
]

GREET_PATTERN = re.compile(r"^(안녕|헬로|hello|\bhi\b)[\s!?.]*$", flags=re.IGNORECASE)
IDENTITY_PATTERN = re.compile(
    r"너\s*누구|누구야|너는\s*누구|너는\s*뭐|너\s*뭐야|자기소개|너에\s*대해|"
    r"뭐\s*할\s*수|할\s*수\s*있|기능\s*(알려|소개|설명)|도움말|\bhelp\b|"
    r"이\s*봇|이\s*에이전트|정체가\s*뭐",
    flags=re.IGNORECASE,
)
META_PATTERN = re.compile(
    r"라우터|의도\s*분류|데이터\s*경로|자동\s*라우팅|"
    r"뭘로\s*구분|어떻게\s*구분|어떻게\s*나뉘|어떤\s*경로|"
    r"시스템\s*(구조|설명|뭐)|어떻게\s*동작|DB가\s*몇|몇\s*개\s*DB|엔진이\s*뭐",
    flags=re.IGNORECASE,
)
FOLLOWUP_PATTERN = re.compile(
    r"^(그럼|그래서|그거|그건|그것도|더|자세히|이어서|그리고|또|계속|응|네)|"
    r"좀\s*더|자세히\s*(알려|설명|봐)|그건\s*뭐|그거\s*뭐",
    flags=re.IGNORECASE,
)
SWITCH_OPS_PATTERN = re.compile(
    r"운항(\s*(쪽|으로|데이터|DB))?|우리\s*배(로|는|는요)?|숫자로|항차로\b",
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
OOS_PATTERN = re.compile(
    r"날씨\s*(어때|좋|나쁘)|환율|주식|비트코인|요리|레시피|축구\s*경기|야구\s*점수|"
    r"영화\s*추천|농담\s*해|심심해",
    flags=re.IGNORECASE,
)


@dataclass
class RouteDecision:
    route: RouteKind
    confidence: float
    ops_score: float
    rag_score: float
    reason: str
    method: str  # "rules" | "llm" | "rules+llm" | "manual"
    chat_mode: ChatMode | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _score(question: str, patterns: list[tuple[str, float]]) -> float:
    q = question or ""
    total = 0.0
    for pat, weight in patterns:
        if re.search(pat, q, flags=re.IGNORECASE):
            total += weight
    return total


def score_question(question: str) -> tuple[float, float]:
    ops, rag = _score(question, OPS_PATTERNS), _score(question, RAG_PATTERNS)
    return _adjust_overlap(question, ops, rag)


def _adjust_overlap(question: str, ops: float, rag: float) -> tuple[float, float]:
    """CII/배출 등 겹치는 용어는 소스 단서로 기울인다."""
    q = question or ""
    overlap = re.search(r"CII|탄소집약|SEEMP|배출|EEXI|DCS", q, flags=re.IGNORECASE)
    if not overlap:
        return ops, rag
    has_reg = bool(re.search(r"MEPC|MSC|선급|규정|규제|조항|회의|Rule|지침|동향", q, flags=re.I))
    has_ship = bool(
        re.search(r"우리|현재|올해|항차|알려줘|계산|등급|보고서|이\s*선박|온보드", q, flags=re.I)
    )
    if has_reg and not has_ship:
        rag += 1.5
        ops = max(0.0, ops - 1.0)
    elif has_ship and not has_reg:
        ops += 1.0
    return ops, rag


def _chat_decision(
    reason: str,
    *,
    chat_mode: ChatMode,
    confidence: float = 0.9,
    ops_score: float = 0.0,
    rag_score: float = 0.0,
) -> RouteDecision:
    return RouteDecision(
        route="chat",
        confidence=confidence,
        ops_score=ops_score,
        rag_score=rag_score,
        reason=reason,
        method="rules",
        chat_mode=chat_mode,
    )


def _rules_route(question: str, margin: float = 1.0) -> RouteDecision:
    q = (question or "").strip()
    if META_PATTERN.search(q):
        return _chat_decision("시스템·라우터 안내 질문이라 chat 경로입니다.", chat_mode="meta")
    if IDENTITY_PATTERN.search(q):
        return _chat_decision("자기소개·기능 안내는 chat 경로입니다.", chat_mode="identity")
    if GREET_PATTERN.search(q):
        return _chat_decision("인사는 chat 경로입니다.", chat_mode="greeting")

    ops, rag = score_question(q)
    if (
        ops > 0
        and rag > 0
        and re.search(r"같이|둘\s*다|동시에|그리고", q)
    ):
        return RouteDecision(
            route="hybrid",
            confidence=0.8,
            ops_score=ops,
            rag_score=rag,
            reason="운항 단서와 문서 단서를 같이 물어 hybrid로 보냅니다.",
            method="rules",
        )
    if ops == 0 and rag == 0:
        if OOS_PATTERN.search(q):
            return _chat_decision(
                "운항/문서 범위 밖 질문이라 chat에서 안내합니다.",
                chat_mode="oos",
                confidence=0.8,
            )
        return RouteDecision(
            route="ambiguous",
            confidence=0.0,
            ops_score=ops,
            rag_score=rag,
            reason="운항/문서 단서가 없어 분류가 불명확합니다.",
            method="rules",
            chat_mode="clarify",
        )
    if ops - rag >= margin:
        return RouteDecision(
            route="ops",
            confidence=min(1.0, (ops - rag) / max(ops, 1.0)),
            ops_score=ops,
            rag_score=rag,
            reason="운항·선박 데이터 단서가 더 강합니다.",
            method="rules",
        )
    if rag - ops >= margin:
        return RouteDecision(
            route="rag",
            confidence=min(1.0, (rag - ops) / max(rag, 1.0)),
            ops_score=ops,
            rag_score=rag,
            reason="규정·회의·선급 문서 단서가 더 강합니다.",
            method="rules",
        )
    return RouteDecision(
        route="hybrid",
        confidence=0.55,
        ops_score=ops,
        rag_score=rag,
        reason=f"운항/문서 단서가 비슷해 hybrid로 함께 봅니다(ops={ops:.1f}, rag={rag:.1f}).",
        method="rules",
    )


def _llm_classify(question: str) -> FinalRoute | None:
    """Optional Ollama JSON classifier for ambiguous questions."""
    base = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("MODEL_NAME", "llama3.1:8b")
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": build_router_user_prompt(question)},
            ],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        content = (payload.get("message") or {}).get("content") or ""
        data = json.loads(content)
        route = str(data.get("route", "")).strip().lower()
        if route in {"ops", "rag", "chat", "hybrid"}:
            return route  # type: ignore[return-value]
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError):
        return None
    return None


def _resolve_ambiguous(decision: RouteDecision) -> RouteDecision:
    if decision.ops_score == 0 and decision.rag_score == 0:
        decision.route = "chat"
        decision.chat_mode = "clarify"
        decision.reason += " 추측하지 않고 되묻습니다. 문서 RAG로 보내지 않습니다."
        decision.confidence = 0.55
        return decision

    decision.route = "hybrid"
    decision.chat_mode = None
    decision.reason += " 한쪽만 고르지 않고 ops와 rag를 함께 실행합니다."
    decision.confidence = 0.55
    return decision


def _is_followup(question: str) -> bool:
    q = (question or "").strip()
    if FOLLOWUP_PATTERN.search(q):
        return True
    return len(q) <= 12 and bool(re.search(r"그거|그건|그럼|더|자세히|응|네", q))


def _multiturn_override(question: str, last_route: str | None) -> RouteDecision | None:
    if last_route not in {"ops", "rag", "hybrid"}:
        return None
    q = (question or "").strip()
    ops, rag = score_question(q)

    if SWITCH_HYBRID_PATTERN.search(q):
        return RouteDecision(
            route="hybrid",
            confidence=0.85,
            ops_score=ops,
            rag_score=rag,
            reason="이전 대화 맥락에서 양쪽을 같이 요청했습니다.",
            method="multiturn",
        )
    switch_ops = bool(SWITCH_OPS_PATTERN.search(q))
    switch_rag = bool(SWITCH_RAG_PATTERN.search(q))
    if switch_ops and not switch_rag and rag - ops < 1.0:
        return RouteDecision(
            route="ops",
            confidence=0.85,
            ops_score=max(ops, 1.0),
            rag_score=rag,
            reason="후속 질문이 운항 쪽으로 전환되었습니다.",
            method="multiturn",
        )
    if switch_rag and not switch_ops and ops - rag < 1.0:
        return RouteDecision(
            route="rag",
            confidence=0.85,
            ops_score=ops,
            rag_score=max(rag, 1.0),
            reason="후속 질문이 문서 쪽으로 전환되었습니다.",
            method="multiturn",
        )

    if not _is_followup(q):
        return None
    if ops - rag >= 1.0 and last_route != "ops":
        return None
    if rag - ops >= 1.0 and last_route != "rag":
        return None
    return RouteDecision(
        route=last_route,  # type: ignore[arg-type]
        confidence=0.75,
        ops_score=ops,
        rag_score=rag,
        reason=f"짧은 후속 질문이라 이전 경로({last_route})를 유지합니다.",
        method="multiturn",
    )


def route_question(
    question: str,
    *,
    use_llm_fallback: bool = True,
    force_route: RouteKind | None = None,
    last_route: str | None = None,
) -> RouteDecision:
    """
    Return where the question should be answered from.
    force_route: UI override ("ops" | "rag" | "chat" | "hybrid")
    last_route: previous persistent route for multi-turn follow-ups
    """
    if force_route in {"ops", "rag", "chat", "hybrid"}:
        return RouteDecision(
            route=force_route,
            confidence=1.0,
            ops_score=0.0,
            rag_score=0.0,
            reason="사용자가 경로를 직접 지정했습니다.",
            method="manual",
            chat_mode="clarify" if force_route == "chat" else None,
        )

    q = (question or "").strip()
    if not (
        META_PATTERN.search(q) or IDENTITY_PATTERN.search(q) or GREET_PATTERN.search(q)
    ):
        multiturn = _multiturn_override(q, last_route)
        if multiturn:
            return multiturn

    decision = _rules_route(question)
    if decision.route != "ambiguous":
        return decision

    if not use_llm_fallback:
        return _resolve_ambiguous(decision)

    llm_route = _llm_classify(question)
    if llm_route in {"ops", "rag", "hybrid"}:
        return RouteDecision(
            route=llm_route,
            confidence=0.7,
            ops_score=decision.ops_score,
            rag_score=decision.rag_score,
            reason=f"규칙 단서가 애매해 LLM이 '{llm_route}'로 분류했습니다.",
            method="rules+llm",
        )
    if llm_route == "chat":
        return RouteDecision(
            route="chat",
            confidence=0.65,
            ops_score=decision.ops_score,
            rag_score=decision.rag_score,
            reason="규칙 단서가 애매해 LLM이 chat으로 분류했습니다.",
            method="rules+llm",
            chat_mode="clarify",
        )

    fallback = _resolve_ambiguous(decision)
    fallback.reason += " LLM 분류 불가라 되묻기로 처리했습니다."
    return fallback
