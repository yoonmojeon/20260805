"""Document scope classification for Fast mode (pre-LLM, rule-based)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag_answer_lib import RetrievedChunk
from imo_doc_classify import meeting_outcome_scope
from retrieval_query_analysis import analyze_query

SCOPE_LABELS = {
    "session_final_outcome": "회의 최종 결과 문서",
    "reference_body_outcome": "특정 회의에 보고되는 참고 문서(다른 body 결과)",
    "cross_committee_reference": "다른 IMO body 결과 보고 문서",
    "agenda_proposal": "의제 제안 문서",
    "information_note": "통보/정보 제공 문서",
    "resolution_or_guidance": "결의안/규정/가이드라인 문서",
    "working_group_report": "워킹그룹/분과위 보고 문서",
    "unknown": "미분류",
}

OUTCOME_OF_BODY_RE = re.compile(
    r"outcome\s+of\s+(c\s*\d+|tc\s*\d+|a\s*\d+|msc\s*\d+|mepc\s*\d+|mepces)",
    re.I,
)
COUNCIL_ASSEMBLY_RE = re.compile(r"\b(c|a)\s*(\d{1,3})\b", re.I)


@dataclass
class DocumentScopeResult:
    scope_type: str
    scope_label_ko: str
    primary_titles: list[str] = field(default_factory=list)
    referenced_bodies: list[str] = field(default_factory=list)
    agenda_items: list[str] = field(default_factory=list)
    scope_mismatch: bool = False
    mismatch_reason: str = ""
    scope_correction_sentence: str = ""
    session_in_question: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scope_type": self.scope_type,
            "scope_label_ko": self.scope_label_ko,
            "primary_titles": self.primary_titles,
            "referenced_bodies": self.referenced_bodies,
            "agenda_items": self.agenda_items,
            "scope_mismatch": self.scope_mismatch,
            "mismatch_reason": self.mismatch_reason,
            "scope_correction_sentence": self.scope_correction_sentence,
            "session_in_question": [f"{b} {n}" for b, n in self.session_in_question],
        }


def _bodies_from_outcome_title(file_name: str) -> list[str]:
    fn = file_name or ""
    bodies: list[str] = []
    m = re.search(r"outcome\s+of\s+(.+?)(?:\s*\(|\.pdf|$)", fn, re.I)
    if not m:
        return bodies
    segment = m.group(1)
    for bm in re.finditer(r"\b(C|A|TC|MSC|MEPC)\s*(\d{1,3})\b", segment, re.I):
        label = bm.group(1).upper()
        num = bm.group(2)
        if label == "C":
            bodies.append(f"C {num}")
        elif label == "A":
            bodies.append(f"A {num}")
        elif label == "TC":
            bodies.append(f"TC {num}")
        elif label == "MSC":
            bodies.append(f"MSC {num}")
        elif label == "MEPC":
            bodies.append(f"MEPC {num}")
    return list(dict.fromkeys(bodies))


def _classify_title(file_name: str) -> tuple[str, list[str]]:
    scope = meeting_outcome_scope(file_name)
    bodies = _bodies_from_outcome_title(file_name) if scope == "reference_body_outcome" else []
    if scope == "session_final_report":
        return "session_final_outcome", bodies
    if scope == "reference_body_outcome":
        return "reference_body_outcome", bodies
    if scope == "working_group_report":
        return "working_group_report", bodies

    fn = (file_name or "").lower()
    if "proposal for" in fn or "annotations to" in fn or "provisional agenda" in fn:
        return "agenda_proposal", bodies
    if "-inf." in fn or re.search(r"\binf\.\d", fn):
        return "information_note", bodies
    if "report of the" in fn and ("working group" in fn or "sub-committee" in fn or "intersessional" in fn):
        return "working_group_report", bodies
    if "outcome" in fn and re.search(r"msc\s*\d+-2\b|mepc\s*\d+-2\b", fn):
        return "session_final_outcome", bodies
    if re.search(r"\b(cg-|ru-|notice no|code of)", fn):
        return "resolution_or_guidance", bodies
    if "report of the" in fn:
        return "working_group_report", bodies
    return "unknown", bodies


def _extract_agenda_from_title(file_name: str) -> str:
    m = re.search(r"\b(msc|mepc)\s*(\d{1,3})-(\d{1,2})", file_name or "", re.I)
    if m:
        return f"{m.group(1).upper()} {m.group(2)}-{m.group(3)}"
    return ""


def _question_asks_session_outcome(question: str) -> bool:
    q = question.lower()
    return any(
        k in q
        for k in (
            "주요 결과",
            "결과를",
            "outcome",
            "key outcomes",
            "최종 결과",
            "회의 결과",
        )
    )


def classify_document_scope(
    chunks: list[RetrievedChunk],
    question: str,
) -> DocumentScopeResult:
    signals = analyze_query(question)
    titles = list(dict.fromkeys(c.file_name for c in chunks if c.file_name))
    scope_type = "unknown"
    all_bodies: list[str] = []
    agendas: list[str] = []

    for title in titles:
        st, bodies = _classify_title(title)
        if scope_type == "unknown" or st != "unknown":
            scope_type = st
        all_bodies.extend(bodies)
        ag = _extract_agenda_from_title(title)
        if ag:
            agendas.append(ag)

    all_bodies = list(dict.fromkeys(all_bodies))
    agendas = list(dict.fromkeys(agendas))

    result = DocumentScopeResult(
        scope_type=scope_type,
        scope_label_ko=SCOPE_LABELS.get(scope_type, scope_type),
        primary_titles=titles[:5],
        referenced_bodies=all_bodies,
        agenda_items=agendas,
        session_in_question=signals.session_codes,
    )

    asks_outcome = _question_asks_session_outcome(question)
    ref_types = {
        "reference_body_outcome",
        "cross_committee_reference",
        "information_note",
        "agenda_proposal",
    }

    if asks_outcome and scope_type in ref_types and signals.session_codes:
        body, num = signals.session_codes[0]
        session_label = f"{body} {num}"
        bodies_txt = " 및 ".join(result.referenced_bodies) if result.referenced_bodies else "다른 IMO body"
        result.scope_mismatch = True
        result.mismatch_reason = (
            f"질문은 {session_label} 회의 자체 결과를 요청했으나, "
            f"근거 문서는 {bodies_txt} 결과 참고 문서입니다."
        )
        result.scope_correction_sentence = (
            f"검색된 문서 기준으로 보면, 이 문서는 {session_label} 자체의 최종 결과가 아니라 "
            f"{session_label}에서 참고할 {bodies_txt}의 결과를 정리한 문서입니다."
        )
    elif asks_outcome and scope_type == "working_group_report" and signals.session_codes:
        body, num = signals.session_codes[0]
        result.scope_mismatch = True
        result.mismatch_reason = "질문은 회의 전체 결과이나 근거는 분과/워킹그룹 보고서입니다."
        result.scope_correction_sentence = (
            f"검색된 문서 기준으로 보면, 이는 {body} {num} 전체 최종 결과가 아니라 "
            "특정 안건/워킹그룹 보고 내용입니다."
        )

    return result
