"""안내(chat) 경로 — DB/문서 검색 없이 정체성·되묻기만 한다."""

from __future__ import annotations

from typing import Literal

ChatMode = Literal["identity", "greeting", "meta", "clarify", "dual", "oos", "thanks"]

CHAT_SYSTEM_PROMPT = """너는 MaritimeOpsRAG의 안내 경로다.
운항 SQLite도, 선급/IMO 문서 검색도 하지 않는다.
인사·자기소개·기능 안내·라우터 설명·되묻기만 짧게 한국어로 답한다.
규정 조항이나 운항 수치를 지어내지 않는다.
모르는 질문이면 거절하지 말고, 운항 데이터인지 문서 검색인지 되묻는다.
"""

CHAT_IDENTITY = """안녕하세요. 저는 **MaritimeOpsRAG**입니다.

운항 데이터와 선급·IMO 문서를 구분해 찾아 답하는 통합 어시스턴트입니다.

- 운항: 항차 상태, CII, 배출량, Noon/MRV
- 문서: MEPC·MSC 동향, 선급 Rule, 표 질의

예: `지금 배 어디야?` / `올해 CII 등급` / `최신 MEPC 회의 주요 내용`
"""

CHAT_GREETING = """안녕하세요. MaritimeOpsRAG입니다.

운항 수치와 규정/회의 문서를 나눠 찾습니다.
운항 쪽을 볼까요, 문서 쪽을 볼까요?
"""

CHAT_ROUTER = """질문은 **선택한 Ollama 모델의 의미 라우터**가 필요한 데이터 소스를 판단해 네 경로 중 하나로 보냅니다.

**1. chat (안내)** — 인사, 소개, 시스템 설명, 애매하면 되묻기. 검색 없음.
**2. ops (운항 DB)** — SQLite + CII/Noon/MRV 툴. 예: `지금 배 어디야`, `기름 얼마나 썼어`
**3. rag (문서)** — 선급·IMO 검색. 예: `MEPC에서 뭐 결정됐어`, `검사 주기 표`
**4. hybrid** — 운항 숫자와 문서를 같이 물어보면 양쪽 답을 출처별로 붙입니다.

인사·감사·기능 안내 같은 확실한 화행만 모델 호출 전에 처리합니다.
짧은 후속 질문은 이전 질문을 의미적으로 펼쳐 다시 판단하고, 모델 호출 실패나 저신뢰 때만 기존 규칙·프로토타입으로 복구합니다.
"""

CHAT_CLARIFY = """운항 정보와 규정·회의 문서는 **둘 다 찾아볼 수 있습니다**. 다만 이번 질문은 어느 쪽의 실제 결과가 필요한지 확실하지 않습니다.

- **운항 수치**가 필요하면: 항차 상태, CII, 연료/배출, Noon/MRV 처럼 말씀해 주세요.
- **문서 검색**이 필요하면: MEPC/MSC, 선급 Rule, 검사 기준/표 처럼 말씀해 주세요.

예: `올해 CII 등급` / `최신 MEPC 동향`
"""

CHAT_DUAL = """이 질문에는 운항 데이터 단서와 규정/회의 문서 단서가 같이 보입니다.

어느 쪽을 먼저 볼까요?
- 우리 선박 숫자(CII, 배출, 항차) → 운항 DB
- 규정·회의·선급 문서 → 문서 검색

한 가지만 집어서 다시 물어보시면 바로 해당 경로로 갑니다.
"""

CHAT_THANKS = """도움이 되었다니 다행입니다.

이어서 운항 수치를 볼까요, 규정·회의 문서를 볼까요?
"""

CHAT_OOS = """그 주제는 이 시스템이 다루는 범위가 아닙니다.

대신 아래는 찾을 수 있습니다.
- 운항: 항차 상태, CII, 배출, Noon/MRV
- 문서: MEPC/MSC 동향, 선급 Rule, 표 질의

운항 수치를 볼까요, 규정·회의 문서를 볼까요?
"""


def render_chat_answer(question: str, chat_mode: ChatMode | str | None = None) -> str:
    mode = (chat_mode or "").strip().lower()
    templates = {
        "identity": CHAT_IDENTITY,
        "greeting": CHAT_GREETING,
        "meta": CHAT_ROUTER,
        "clarify": CHAT_CLARIFY,
        "dual": CHAT_DUAL,
        "oos": CHAT_OOS,
        "thanks": CHAT_THANKS,
    }
    if mode in templates:
        return templates[mode].strip()
    return CHAT_CLARIFY.strip()
