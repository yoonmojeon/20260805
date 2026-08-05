"""
질문 의도 라우터 — 운항 DB(ops) vs 규정/회의 문서 RAG(rag).

1) 규칙 기반 점수 (빠름, 결정론적)
2) 점수가 비슷하면 Ollama로 한 번 더 분류 (선택)
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Literal

RouteKind = Literal["ops", "rag", "ambiguous"]

OPS_PATTERNS: list[tuple[str, float]] = [
    (r"운항\s*상태|현재\s*항차|이전\s*항차|올해\s*연간|연간\s*실적", 3.0),
    (r"\bCII\b|탄소집약|attained|required\s*cii|등급\s*[A-E]", 2.5),
    (r"Noon\s*Report|MRV|배출량|CO2e?|CH4|FOC|FGC|연료\s*소모", 2.5),
    (r"\bSOG\b|속력|위도|경도|항적|항차\s*분석|sensor|센서", 2.0),
    (r"Ballast|Laden|H2521|voyage|항해\s*거리|distance_nm", 1.5),
    (r"보고서\s*생성|워드|docx|브리핑", 1.2),
    (r"유류|LNG|가스\s*소모|oil_flow|gas_flow", 1.5),
]

RAG_PATTERNS: list[tuple[str, float]] = [
    (r"\bMEPC\b|\bMSC\b|\bMASS\b|IMO\s*회의|회의\s*(결과|주요|동향)", 3.0),
    (r"선급|Rule/?Guidance|\bDNV\b|\bABS\b|\bLR\b|\bKR\b\s*Rule|KR\s*규칙", 3.0),
    (r"규정|지침|요건|조항|\bclause\b|\bchapter\b|Guidance", 2.0),
    (r"MARPOL|SOLAS|Net-?Zero|GFI|SEEMP|EEXI|DCS|GISIS", 2.0),
    (r"표\s*(에서|질의|검색)|정기검사|평형수탱크|선령", 2.5),
    (r"문서|PDF|회의록|circular|resolution|WP\.?\d", 1.5),
    (r"자율운항|대체연료\s*안전|환경규제\s*대응|최신\s*동향", 2.0),
]


@dataclass
class RouteDecision:
    route: RouteKind
    confidence: float
    ops_score: float
    rag_score: float
    reason: str
    method: str  # "rules" | "llm" | "rules+llm"

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
    return _score(question, OPS_PATTERNS), _score(question, RAG_PATTERNS)


def _rules_route(question: str, margin: float = 1.0) -> RouteDecision:
    ops, rag = score_question(question)
    if ops == 0 and rag == 0:
        return RouteDecision(
            route="ambiguous",
            confidence=0.0,
            ops_score=ops,
            rag_score=rag,
            reason="운항/문서 키워드가 없어 분류가 불명확합니다.",
            method="rules",
        )
    if ops - rag >= margin:
        return RouteDecision(
            route="ops",
            confidence=min(1.0, (ops - rag) / max(ops, 1.0)),
            ops_score=ops,
            rag_score=rag,
            reason="운항·CII·센서·보고서 관련 표현이 더 강합니다.",
            method="rules",
        )
    if rag - ops >= margin:
        return RouteDecision(
            route="rag",
            confidence=min(1.0, (rag - ops) / max(rag, 1.0)),
            ops_score=ops,
            rag_score=rag,
            reason="IMO/선급/규정 문서 관련 표현이 더 강합니다.",
            method="rules",
        )
    winner = "ops" if ops >= rag else "rag"
    return RouteDecision(
        route="ambiguous",
        confidence=0.35,
        ops_score=ops,
        rag_score=rag,
        reason=f"점수가 비슷합니다(ops={ops:.1f}, rag={rag:.1f}). 기본 후보={winner}.",
        method="rules",
    )


def _llm_classify(question: str) -> RouteKind | None:
    """Optional Ollama JSON classifier for ambiguous questions."""
    base = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("MODEL_NAME", "llama3.1:8b")
    prompt = (
        "Classify the user question into exactly one label.\n"
        "- ops: vessel voyage/sensor KPIs, CII rating from ship data, FOC/emissions, "
        "Noon/MRV reports from onboard logs\n"
        "- rag: class society rules, IMO MEPC/MSC meeting documents, regulations, table QA from PDFs\n"
        'Return JSON only: {"route":"ops"|"rag"}\n\n'
        f"Question: {question}"
    )
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": "You are a strict router. JSON only."},
                {"role": "user", "content": prompt},
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
        if route in {"ops", "rag"}:
            return route  # type: ignore[return-value]
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError):
        return None
    return None


def route_question(
    question: str,
    *,
    use_llm_fallback: bool = True,
    force_route: RouteKind | None = None,
) -> RouteDecision:
    """
    Return where the question should be answered from.
    force_route: UI override ("ops" | "rag")
    """
    if force_route in {"ops", "rag"}:
        return RouteDecision(
            route=force_route,
            confidence=1.0,
            ops_score=0.0,
            rag_score=0.0,
            reason="사용자가 경로를 직접 지정했습니다.",
            method="manual",
        )

    decision = _rules_route(question)
    if decision.route != "ambiguous" or not use_llm_fallback:
        if decision.route == "ambiguous":
            # Prefer ops when both empty? Prefer rag for regulatory default?
            # Safer default for empty keywords: ask-facing ops if greeting else rag.
            q = (question or "").strip()
            if re.search(r"안녕|헬로|기능|도움|help|뭐\s*할", q, re.I):
                decision.route = "ops"
                decision.reason += " 인사/안내는 운항 에이전트로 보냅니다."
                decision.confidence = 0.5
            elif decision.ops_score >= decision.rag_score and decision.ops_score > 0:
                decision.route = "ops"
            elif decision.rag_score > 0:
                decision.route = "rag"
            else:
                decision.route = "rag"
                decision.reason += " 기본값은 문서 RAG입니다."
                decision.confidence = 0.4
        return decision

    llm_route = _llm_classify(question)
    if llm_route:
        return RouteDecision(
            route=llm_route,
            confidence=0.7,
            ops_score=decision.ops_score,
            rag_score=decision.rag_score,
            reason=f"규칙 점수가 비슷해 LLM이 '{llm_route}'로 분류했습니다.",
            method="rules+llm",
        )

    # LLM unavailable: break tie toward stronger score, else rag
    tie = "ops" if decision.ops_score > decision.rag_score else "rag"
    if decision.ops_score == decision.rag_score:
        tie = "rag"
    return RouteDecision(
        route=tie,  # type: ignore[arg-type]
        confidence=0.45,
        ops_score=decision.ops_score,
        rag_score=decision.rag_score,
        reason=decision.reason + " LLM 분류 불가로 점수 우위로 결정했습니다.",
        method="rules",
    )
