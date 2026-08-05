"""최상위 의도 분류기용 프롬프트. 답변을 쓰지 않고 경로만 고른다."""

from __future__ import annotations


ROUTER_SYSTEM_PROMPT = (
    "You are a strict intent router for MaritimeOpsRAG. Return JSON only. "
    "Never answer the user question. Prefer chat when unsure rather than guessing. "
    "Do not invent voyage numbers or regulation text."
)


def build_router_user_prompt(
    question: str,
    *,
    last_route: str | None = None,
    last_topic: str | None = None,
    last_question: str | None = None,
    ops_score: float = 0.0,
    rag_score: float = 0.0,
    expanded_question: str | None = None,
) -> str:
    return (
        "Classify the user question into exactly one label.\n"
        "- chat: identity, greeting, how the system works, off-topic chit-chat, unclear\n"
        "- hybrid: user wants BOTH this-ship numbers AND regulation/meeting docs, "
        "or compares this ship against a rule\n"
        "- ops: THIS vessel's live/historical voyage KPIs, speed/fuel/position, "
        "CII/emissions calculated from onboard logs, Noon/MRV generation\n"
        "- rag: class society rules, IMO MEPC/MSC documents, regulations, "
        "inspection tables from PDFs. Meeting summaries requested as a report stay rag.\n"
        "Colloquial ops examples: 지금 배 어디야, 기름 얼마나 썼어, 스피드 얼마야\n"
        "Colloquial rag examples: 작년에 회의에서 CII 어떻게 됐대, 검사 주기 표\n"
        "If CII/SEEMP/emissions appear with MEPC/MSC/rule/meeting only → rag.\n"
        "If they appear with this ship / this year / current voyage → ops.\n"
        "If both ship-data and regulation cues are strong, or compare-frame → hybrid.\n"
        "Close scores without a clear dual/compare intent → chat.\n"
        "Return JSON only: "
        '{"route":"chat"|"ops"|"rag"|"hybrid","ops_query":"","rag_query":"",'
        '"expanded_question":"","confidence":0.0}\n\n'
        f"Question: {question}\n"
        f"Expanded: {expanded_question or question}\n"
        f"Rule scores: ops={ops_score:.1f}, rag={rag_score:.1f}\n"
        f"Last route: {last_route or '-'}\n"
        f"Last topic: {last_topic or '-'}\n"
        f"Last question: {last_question or '-'}\n"
    )
