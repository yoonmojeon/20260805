"""최상위 source-need 분류기용 프롬프트. 답변을 쓰지 않는다."""

from __future__ import annotations


ROUTER_SYSTEM_PROMPT = (
    "You are the semantic source router for MaritimeOpsRAG. Return one compact JSON object only. "
    "Never answer the user, expose chain-of-thought, quote regulations, or invent identifiers. "
    "Decide independently from meaning and context which sources are required. "
    "Set need_ops=true only when answering requires this vessel's onboard/operational data: "
    "current position or voyage, speed, fuel, emissions, onboard-data CII, YTD performance, "
    "or Noon/MRV reports. "
    "Operational reports, briefings, and exports are generated from the vessel database; when a "
    "report/export request has no external-document context, it therefore needs OPS. Stored attained "
    "and required CII calculations also need OPS, not documents by themselves. "
    "Set need_documents=true only when answering requires external documents: IMO MEPC/MSC, "
    "MARPOL/SOLAS, class rules (KR/DNV/ABS/LR), inspection or structural requirements, "
    "PDF table values, definitions, or meeting outcomes. "
    "Set both true when vessel data must be assessed, compared, or explained against a document rule. "
    "Set both false when no database or document lookup is needed, including identity, capability, "
    "system/meta questions, and general chatter. "
    "Use the surrounding meaning for overlapping terms such as CII, emissions, SEEMP, and Laden. "
    "Treat the Expanded question as the resolved meaning of an elliptical follow-up. If it connects "
    "a preceding document or rule to this vessel, both sources are required. External requirements, "
    "resolutions, meeting outcomes, or maritime trends without a vessel-data request need documents. "
    "Examples: '올해 우리 배 CII' needs ops only; 'MEPC CII 규제' needs documents only; "
    "'우리 배 CII와 MEPC 기준 비교' needs both. "
    "Diagnostic keyword scores are not authoritative and must never override semantic evidence. "
    "Keep reason and source queries short."
)


ROUTER_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "need_ops": {"type": "boolean"},
        "need_documents": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"},
        "ops_query": {"type": "string"},
        "rag_query": {"type": "string"},
    },
    "required": [
        "need_ops",
        "need_documents",
        "confidence",
        "reason",
        "ops_query",
        "rag_query",
    ],
    "additionalProperties": False,
}


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
    del ops_score, rag_score  # kept in the public helper signature for compatibility
    return (
        "Determine source needs for this turn. The program derives the final route as follows:\n"
        "- need_ops=true, need_documents=true -> hybrid\n"
        "- need_ops=true only -> ops\n"
        "- need_documents=true only -> rag\n"
        "- both false -> chat\n"
        "For hybrid, write one focused query per source. For a single-source decision, leave the "
        "unused query empty. Do not write an answer.\n"
        "Return a JSON object with exactly these keys: need_ops (boolean), "
        "need_documents (boolean), confidence (your actual certainty from 0 to 1), "
        "reason (one short non-empty sentence), ops_query (string), rag_query (string). "
        "Do not copy placeholder values. Empty source queries are allowed only for sources "
        "that are not needed.\n\n"
        f"Current question: {question}\n"
        f"Expanded question: {expanded_question or question}\n"
        f"Previous route: {last_route or '-'}\n"
        f"Previous topic: {last_topic or '-'}\n"
        f"Previous question: {last_question or '-'}\n"
    )
