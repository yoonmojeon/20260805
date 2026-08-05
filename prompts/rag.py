"""문서 RAG 경로 — 규정/회의 근거만. 운항 DB 계산 금지."""

RAG_ROUTE_IDENTITY = (
    "너는 MaritimeOpsRAG의 문서 검색(RAG) 경로다. "
    "선급 Rule/Guidance와 IMO MEPC·MSC 등 검색된 문서 근거만 사용한다. "
    "운항 SQLite 수치, 우리 선박 CII 계산, Noon/MRV 생성은 하지 않는다. "
    "자기소개·잡담·현재 운항 상태 질문이면 문서로 추측하지 말고, "
    "운항 데이터 경로로 다시 물어보라고 짧게 안내한다."
)


def apply_rag_identity(system_prompt: str) -> str:
    text = (system_prompt or "").strip()
    identity = RAG_ROUTE_IDENTITY.strip()
    if not text:
        return identity
    if identity in text:
        return text
    return f"{identity}\n\n{text}"
