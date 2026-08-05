"""최상위 의도 분류기용 프롬프트. 답변을 쓰지 않고 경로만 고른다."""

ROUTER_SYSTEM_PROMPT = (
    "You are a strict intent router for MaritimeOpsRAG. Return JSON only. "
    "Never answer the user question. Prefer chat when unsure rather than guessing."
)


def build_router_user_prompt(question: str) -> str:
    return (
        "Classify the user question into exactly one label.\n"
        "- chat: identity, greeting, how the system works, off-topic chit-chat\n"
        "- hybrid: user clearly wants BOTH this-ship numbers AND regulation/meeting docs\n"
        "- ops: THIS vessel's live/historical voyage KPIs, speed/fuel/position, "
        "CII/emissions calculated from onboard logs, Noon/MRV generation\n"
        "- rag: class society rules, IMO MEPC/MSC documents, regulations, "
        "inspection tables from PDFs\n"
        "Colloquial ops examples: 지금 배 어디야, 기름 얼마나 썼어, 스피드 얼마야\n"
        "Colloquial rag examples: 작년에 회의에서 CII 어떻게 됐대, 검사 주기 표\n"
        "If CII/SEEMP/emissions appear with MEPC/MSC/rule/meeting only → rag.\n"
        "If they appear with this ship / this year / current voyage → ops.\n"
        "If both ship-data and regulation cues are strong → hybrid.\n"
        'Return JSON only: {"route":"chat"|"ops"|"rag"|"hybrid"}\n\n'
        f"Question: {question}"
    )
