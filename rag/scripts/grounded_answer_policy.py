"""Shared evidence policy for deterministic RAG answers.

The module deliberately uses conservative checks.  A document reference is not
treated as a committee decision merely because it contains words such as
"adopted" in background text, and every factual bullet must point to evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class DocumentStatus:
    code: str
    label_ko: str
    authority: int
    supports_final_decision: bool


STATUSES = {
    "class_rule": DocumentStatus("class_rule", "선급 Rule/Guidance", 5, True),
    "adopted_instrument": DocumentStatus("adopted_instrument", "채택 결의·협약 문서", 5, True),
    "committee_decision": DocumentStatus("committee_decision", "위원회 결정·최종 보고", 4, True),
    "draft_outcome": DocumentStatus("draft_outcome", "회의 결과 초안", 3, False),
    "proposal": DocumentStatus("proposal", "제안·제출 문서", 2, False),
    "action_request": DocumentStatus("action_request", "위원회 조치 요청", 1, False),
    "background": DocumentStatus("background", "배경·참고 자료", 1, False),
    "committee_submission": DocumentStatus("committee_submission", "위원회 제출·보고 자료", 2, False),
    "unknown": DocumentStatus("unknown", "상태 미확인", 0, False),
}


FINAL_CLAIM_RE = re.compile(
    r"채택|승인|확정|결정|발효|의무화|adopt(?:ed|ion)?|approv(?:ed|al)?|"
    r"decid(?:ed|ed that)|finali[sz]ed|entry into force|mandatory",
    re.I,
)
NONFINAL_STATUS_RE = re.compile(
    r"연기|보류|미채택|승인\s*전|확정되지|최종\s*(?:결과|보고서)가\s*아니|"
    r"adjourn(?:ed|ment)?|postpon(?:ed|ement)?|defer(?:red|ral)?|not\s+(?:yet\s+)?adopted",
    re.I,
)
PROPOSAL_FILE_RE = re.compile(
    r"proposal|proposed|submission|comments? by|consideration of|information paper|inf[. _-]?\d",
    re.I,
)
DRAFT_FILE_RE = re.compile(r"draft report|draft guideline|draft code|wp[. _-]?1\b", re.I)
REPORT_FILE_RE = re.compile(r"final report|report of the|committee report|session report", re.I)
MEETING_SUBMISSION_RE = re.compile(r"^(?:MEPC|MSC)\s+\d{1,3}[-/]\d", re.I)
ACTION_REQUEST_RE = re.compile(r"action requested (?:of|by) the (?:committee|sub-committee)", re.I)
ADOPTION_RE = re.compile(
    r"\b(?:the committee\s+)?(?:adopted|approved|agreed|decided|endorsed|finali[sz]ed)\b|"
    r"\bADOPTS\b|entry into force",
    re.I,
)
RESOLUTION_RE = re.compile(r"\b(?:MEPC|MSC)\.\d+\(\d+\)|\bresolution\b", re.I)
CLASS_SOURCES = {"DNV", "LR", "ABS", "KR"}


def _chunk_blob(chunk: Any) -> tuple[str, str, str]:
    file_name = str(getattr(chunk, "file_name", "") or "")
    source = str(getattr(chunk, "source", "") or "").upper()
    text = str(getattr(chunk, "text", "") or "")
    return file_name, source, text


def classify_document_status(chunk: Any) -> DocumentStatus:
    """Classify what the retrieved passage is allowed to prove."""
    file_name, source, text = _chunk_blob(chunk)
    name = file_name.lower()
    low = text.lower()

    if source in CLASS_SOURCES:
        return STATUSES["class_rule"]
    # Filename/purpose has priority: a proposal may quote an adopted resolution,
    # but that does not turn the proposal itself into the decision record.
    if PROPOSAL_FILE_RE.search(name) or re.search(r"submitted by\s+", low[:700], re.I):
        return STATUSES["proposal"]
    if ACTION_REQUEST_RE.search(name) or ACTION_REQUEST_RE.search(low[:900]):
        return STATUSES["action_request"]
    if DRAFT_FILE_RE.search(name):
        return STATUSES["draft_outcome"]
    if RESOLUTION_RE.search(name) and ADOPTION_RE.search(text):
        return STATUSES["adopted_instrument"]
    # Numbered agenda papers may quote earlier adoptions in their background.
    # The quotation does not make the current submission a final decision.
    if source in {"MEPC", "MSC"} and MEETING_SUBMISSION_RE.search(file_name):
        return STATUSES["committee_submission"]
    if REPORT_FILE_RE.search(name) and ADOPTION_RE.search(text):
        return STATUSES["committee_decision"]
    if ADOPTION_RE.search(text):
        return STATUSES["committee_decision"]
    if RESOLUTION_RE.search(name) or RESOLUTION_RE.search(text[:500]):
        return STATUSES["background"]
    return STATUSES["unknown"]


HEADER_NOISE_RE = re.compile(
    r"copyright|all rights reserved|electronic pdf|standard disclaimer|"
    r"class guideline\s*[—-].{0,100}(?:edition|december|january)|"
    r"^\s*(?:mepc|msc)\s+\d+(?:/\d+)*\s*$",
    re.I,
)
REFERENCE_ONLY_RE = re.compile(r"doi\.org|https?://|references?\s*$|bibliograph", re.I)
CLAUSE_SIGNAL_RE = re.compile(
    r"\b(?:scope|objective|application|requirements?|shall|must|should|"
    r"adopted|approved|agreed|decided|entry into force|work plan|timeline|"
    r"notation|design|operation|verification|assessment|reporting)\b",
    re.I,
)


def is_substantive_chunk(chunk: Any) -> bool:
    text = re.sub(r"\s+", " ", str(getattr(chunk, "text", "") or "")).strip()
    if len(text) < 90:
        return False
    if HEADER_NOISE_RE.search(text) and not CLAUSE_SIGNAL_RE.search(text):
        return False
    if text.count("....") >= 3:
        return False
    if REFERENCE_ONLY_RE.search(text) and not CLAUSE_SIGNAL_RE.search(text):
        return False
    words = re.findall(r"[A-Za-z가-힣0-9]+", text.lower())
    if len(set(words)) < 14:
        return False
    return True


def _query_terms(question: str) -> set[str]:
    stop = {
        "관련", "정리", "요약", "알려줘", "찾아줘", "무엇", "어떤", "대한",
        "the", "and", "for", "with", "from", "what", "find", "rule", "guidance",
    }
    terms = {
        x.lower()
        for x in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{2,}|[가-힣]{2,}", question or "")
        if x.lower() not in stop
    }
    return terms


def select_key_clause_chunks(
    question: str,
    chunks: Iterable[Any],
    *,
    limit: int = 12,
    outcome_query: bool = False,
) -> list[Any]:
    """Prefer substantive, query-matching clauses and authoritative status."""
    qterms = _query_terms(question)
    ranked: list[tuple[float, int, Any]] = []
    for pos, chunk in enumerate(chunks):
        if not is_substantive_chunk(chunk):
            continue
        status = classify_document_status(chunk)
        file_name, _, body = _chunk_blob(chunk)
        blob = f"{file_name} {body}".lower()
        overlap = sum(1 for term in qterms if term in blob)
        clause_signals = len(CLAUSE_SIGNAL_RE.findall(body[:1800]))
        score = overlap * 2.0 + min(clause_signals, 5) * 0.35
        score += status.authority * (1.25 if outcome_query else 0.35)
        if outcome_query and status.code in {"proposal", "action_request"}:
            score -= 4.0
        ranked.append((score, -pos, chunk))
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [chunk for _, _, chunk in ranked[:limit]]


TOPIC_ALIASES = {
    "mass": ("mass", "autonomous", "자율운항", "원격운항"),
    "cii": ("cii", "carbon intensity", "탄소집약도"),
    "seemp": ("seemp",),
    "ghg": ("ghg", "greenhouse gas", "온실가스"),
    "fuel": ("fuel", "연료"),
    "safety": ("safety", "안전"),
    "report": ("report", "보고", "제출"),
    "verification": ("verification", "검증"),
    "design": ("design", "설계"),
    "operation": ("operation", "운항", "운용"),
    "notation": ("notation", "부기부호"),
    "remote_control": (
        "remote operation",
        "remote operator",
        "remote control",
        "원격 운항",
        "원격운항",
        "원격 운영",
        "원격 제어",
    ),
    "situational_awareness": (
        "situational awareness",
        "상황 인식",
        "상황인식",
    ),
    "monitoring": (
        "monitoring",
        "observe",
        "operational status",
        "감시",
        "관찰",
        "운영 상태",
    ),
}
# Product codes only (DNV-CG-0264). Do not treat "LR Notice" prose as a hard code.
DOC_CODE_RE = re.compile(
    r"\b(?:MEPC|MSC)\.\d+\(\d+\)|"
    r"\b(?:DNV|LR|ABS|KR)-(?:CG|RU|RP|GL|CLASS)?[A-Z0-9][A-Z0-9._-]*",
    re.I,
)
DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b\d{1,2}\s+(?:January|February|March|April|May|June|"
    r"July|August|September|October|November|December)\s+(?:19|20)\d{2}\b",
    re.I,
)
PRESCRIPTIVE_CLAIM_RE = re.compile(
    r"해야\s*합니다|하여야\s*합니다|필요합니다|준비해야|추적해야|맞춰야|"
    r"관리\s*대상입니다|검증해야|대조해야|반영해야",
    re.I,
)
NORMATIVE_EVIDENCE_RE = re.compile(
    r"\bshall\b|\bmust\b|\bis required to\b|\bare required to\b|"
    r"\bshould\b|\bhas to\b|\bhave to\b|의무|하여야\s*한다|해야\s*한다|"
    r"요구된다|필수",
    re.I,
)


def _topic_keys(text: str) -> set[str]:
    low = (text or "").lower()
    return {key for key, aliases in TOPIC_ALIASES.items() if any(alias in low for alias in aliases)}


def _unsupported_reason(claim: str, evidence: str, statuses: list[DocumentStatus]) -> str | None:
    claim_clean = CITATION_RE.sub("", claim)
    if (
        "추가 확인" in claim_clean
        or "확인되지" in claim_clean
        or "근거 부족" in claim_clean
        or "단정할 수 없" in claim_clean
        or "아니므로" in claim_clean
        or "아닙니다" in claim_clean
    ):
        return None

    for code in DOC_CODE_RE.findall(claim_clean):
        if re.sub(r"[\s_-]", "", code).lower() not in re.sub(r"[\s_-]", "", evidence).lower():
            return f"document_code_not_in_evidence:{code}"
        # A resolution number and a topic appearing somewhere in the same
        # multi-paragraph chunk is not enough.  The topic must occur in the
        # sentence that actually mentions that resolution.
        code_pos = evidence.lower().find(code.lower())
        if code_pos >= 0 and re.match(r"(?:MEPC|MSC)\.\d+\(\d+\)", code, re.I):
            left = max(evidence.rfind(".", 0, code_pos), evidence.rfind("\n", 0, code_pos))
            right_candidates = [x for x in (evidence.find(".", code_pos), evidence.find("\n", code_pos)) if x >= 0]
            right = min(right_candidates) if right_candidates else min(len(evidence), code_pos + 360)
            local_evidence = evidence[left + 1:right + 1]
            specific_claim_topics = _topic_keys(claim_clean) - {"report", "operation"}
            if specific_claim_topics and not specific_claim_topics.intersection(_topic_keys(local_evidence)):
                return f"resolution_topic_not_in_same_sentence:{code}"
    for date in DATE_RE.findall(claim_clean):
        if date.lower() not in evidence.lower():
            return f"date_not_in_evidence:{date}"

    # A descriptive report passage does not support converting a finding into
    # an operator/company obligation.  Recommendations are allowed only when
    # the cited passage itself contains normative language.
    if PRESCRIPTIVE_CLAIM_RE.search(claim_clean) and not NORMATIVE_EVIDENCE_RE.search(evidence):
        return "prescriptive_inference_not_in_evidence"

    if FINAL_CLAIM_RE.search(claim_clean) and not statuses.count(STATUSES["class_rule"]):
        draft_attributed = "초안 기록상" in claim_clean and any(
            status.code == "draft_outcome" for status in statuses
        )
        nonfinal_attributed = (
            (
                "초안" in claim_clean
                or "요청" in claim_clean
                or "보고" in claim_clean
                or "기록" in claim_clean
                or "작업반" in claim_clean
                or (
                    NONFINAL_STATUS_RE.search(claim_clean) is not None
                    and NONFINAL_STATUS_RE.search(evidence) is not None
                )
            )
            and any(
                status.code in {"draft_outcome", "committee_submission", "background", "unknown"}
                for status in statuses
            )
        )
        if not any(status.supports_final_decision for status in statuses) and not (draft_attributed or nonfinal_attributed):
            return "final_decision_claim_from_nonfinal_document"
        # Draft-attributed bullets may cite WP.1 / WG reports that use agreed/approved
        # without a formal "adopted" resolution string in the same clip.
        soft_decision = bool(
            re.search(r"\b(?:agreed|approved|endorsed|noted|invited|requested)\b", evidence, re.I)
            or "mandatory" in evidence.lower()
            or "mass" in evidence.lower()
        )
        if (
            not ADOPTION_RE.search(evidence)
            and not soft_decision
            and not (draft_attributed or nonfinal_attributed)
        ):
            return "decision_action_not_in_evidence"

    claim_topics = _topic_keys(claim_clean)
    evidence_topics = _topic_keys(evidence)
    if claim_topics and not claim_topics.intersection(evidence_topics):
        return "topic_not_in_evidence"
    return None


def verify_claim_citations(answer: str, citation_chunks: list[Any]) -> tuple[str, list[dict], list[str]]:
    """Remove factual bullets whose cited passages do not support the claim."""
    rows: list[dict] = []
    warnings: list[str] = []
    output: list[str] = []
    section_had_kept_bullet = False
    pending_heading = False

    for raw in (answer or "").splitlines():
        stripped = raw.strip()
        if stripped.startswith("## "):
            if pending_heading and not section_had_kept_bullet:
                output.append("- 검색 근거에서 직접 확인되는 내용이 없어 답변에서 제외했습니다.")
                output.append("")
            output.append(raw)
            pending_heading = True
            section_had_kept_bullet = False
            continue
        if not stripped.startswith("- "):
            output.append(raw)
            continue

        claim = stripped[2:].strip()
        cite_ids = sorted({int(x) for x in CITATION_RE.findall(claim)})
        is_disclaimer = any(
            x in claim
            for x in ("추가 확인", "확인되지", "근거 부족", "근거에서 직접", "아니므로", "아닙니다")
        )
        valid_ids = [i for i in cite_ids if 1 <= i <= len(citation_chunks)]
        reason: str | None = None
        meaningful = re.sub(r"\*|_|`|\[[0-9]+\]|[^0-9A-Za-z가-힣]+", "", claim)
        if not meaningful:
            reason = "empty_claim"
        elif not cite_ids and not is_disclaimer:
            reason = "citation_missing"
        elif cite_ids and len(valid_ids) != len(cite_ids):
            reason = "citation_out_of_range"
        elif valid_ids:
            evidence_chunks = [citation_chunks[i - 1] for i in valid_ids]
            evidence = " ".join(
                f"{getattr(c, 'file_name', '')} {getattr(c, 'text', '')}" for c in evidence_chunks
            )
            statuses = [classify_document_status(c) for c in evidence_chunks]
            reason = _unsupported_reason(claim, evidence, statuses)

        supported = reason is None
        rows.append(
            {
                "claim": CITATION_RE.sub("", claim).strip(),
                "citations": [f"[{i}]" for i in valid_ids],
                "supported": supported,
                "reason": reason or "supported",
                "document_status": [
                    classify_document_status(citation_chunks[i - 1]).code for i in valid_ids
                ],
            }
        )
        if supported:
            output.append(raw)
            section_had_kept_bullet = True
        else:
            warnings.append(f"unsupported_claim_removed:{reason}")

    if pending_heading and not section_had_kept_bullet:
        output.append("- 검색 근거에서 직접 확인되는 내용이 없어 답변에서 제외했습니다.")

    # Collapse excessive blank lines left by removed bullets.
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(output)).strip()
    return cleaned, rows, list(dict.fromkeys(warnings))


HIGH_RISK_REASONS = {
    "prescriptive_inference_not_in_evidence",
    "final_decision_claim_from_nonfinal_document",
    "decision_action_not_in_evidence",
    "citation_missing",
    "citation_out_of_range",
    "empty_claim",
}


def verify_high_risk_claims(
    answer: str,
    citation_chunks: list[Any],
) -> tuple[str, list[dict], list[str]]:
    """Block only objectively dangerous claim/evidence mismatches.

    Korean paraphrases of English source text make pure lexical entailment
    unreliable.  This boundary therefore blocks dates or document codes absent
    from evidence, unsupported prescriptive duties, and final decisions claimed
    from non-final documents, while leaving low-confidence topic-overlap checks
    as diagnostics.
    """
    checked, rows, warnings = verify_claim_citations(answer, citation_chunks)
    high_risk_prefixes = (
        "date_not_in_evidence:",
        "document_code_not_in_evidence:",
        "resolution_topic_not_in_same_sentence:",
    )
    high_risk_claims = {
        row["claim"]
        for row in rows
        if row.get("reason") in HIGH_RISK_REASONS
        or str(row.get("reason") or "").startswith(high_risk_prefixes)
    }
    if not high_risk_claims:
        return answer, rows, []

    output: list[str] = []
    removed: list[str] = []
    for raw in (answer or "").splitlines():
        stripped = raw.strip()
        if not stripped.startswith("- "):
            output.append(raw)
            continue
        claim = CITATION_RE.sub("", stripped[2:]).strip()
        if claim in high_risk_claims:
            reason = next(
                (
                    str(row.get("reason"))
                    for row in rows
                    if row.get("claim") == claim
                ),
                "unsupported",
            )
            removed.append(f"high_risk_claim_removed:{reason}")
            continue
        output.append(raw)
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(output)).strip()
    return cleaned or checked, rows, list(dict.fromkeys(removed))
