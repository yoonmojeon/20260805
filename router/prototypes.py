"""Lightweight character n-gram prototype vote. No embedding model download."""
from __future__ import annotations

import re
from functools import lru_cache

PROTOTYPES: dict[str, list[str]] = {
    "chat": [
        "너 누구야",
        "자기소개 해봐",
        "뭐 할 수 있어",
        "안녕",
        "라우터가 뭘로 구분되어있어",
        "시스템 구조 뭐야",
        "사용법 알려줘",
        "오늘 날씨 어때",
        "영화 추천해줘",
        "아무 말",
        "고마워",
        "처음인데 어떻게 써요",
    ],
    "ops": [
        "현재 운항 상태 알려줘",
        "지금 배 어디야",
        "올해 CII 등급을 알려줘",
        "기름 얼마나 썼어",
        "지금 스피드 얼마야",
        "Noon Report 생성해줘",
        "올해 배출량 알려줘",
        "이전 항차 실적 보여줘",
        "우리 배 CII 얼마야",
        "올해 SEEMP 잘 되고 있어",
    ],
    "rag": [
        "최신 MEPC 회의 주요 내용을 정리해줘",
        "DNV에서 자율운항 관련 Rule/Guidance를 찾아줘",
        "검사 주기 표 어디 있어",
        "작년에 회의에서 CII 어떻게 됐대",
        "MARPOL Annex VI 요건",
        "선급 규정 확인해줘",
        "회의 내용으로 보고서 만들어줘",
        "문서에서 CII 규제 찾아줘",
        "KR 규칙 검사 요건이 뭐야",
        "최신 동향 요약해줘",
    ],
    "hybrid": [
        "우리 CII랑 MEPC 규제 같이 알려줘",
        "올해 배출량이랑 환경규제 동시에 설명해줘",
        "그 규정 기준으로 우리 배는 어때",
        "항차 분석이랑 선급 규칙 같이 봐줘",
        "운항 숫자와 규정 둘 다 정리해줘",
    ],
}


def _ngrams(text: str) -> dict[str, int]:
    s = re.sub(r"\s+", "", (text or "").lower())
    counts: dict[str, int] = {}
    for n in (2, 3):
        for i in range(max(0, len(s) - n + 1)):
            gram = s[i : i + n]
            counts[gram] = counts.get(gram, 0) + 1
    return counts


def _cosine(a: dict[str, int], b: dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(av * b[k] for k, av in a.items() if k in b)
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@lru_cache(maxsize=1)
def _prototype_vectors() -> dict[str, list[dict[str, int]]]:
    return {route: [_ngrams(text) for text in texts] for route, texts in PROTOTYPES.items()}


def prototype_scores(question: str) -> dict[str, float]:
    qv = _ngrams(question)
    scores: dict[str, float] = {}
    for route, vectors in _prototype_vectors().items():
        scores[route] = max((_cosine(qv, vec) for vec in vectors), default=0.0)
    return scores


def prototype_vote(question: str) -> tuple[str | None, float, dict[str, float]]:
    scores = prototype_scores(question)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_route, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best - second
    if best >= 0.30 and margin >= 0.05:
        return best_route, margin, scores
    return None, margin, scores
