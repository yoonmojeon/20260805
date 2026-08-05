"""Pre-answer confidence checks for Fast mode."""
from __future__ import annotations

import re
from dataclasses import dataclass

from meeting_summary_intent import validate_meeting_summary_answer
from fast_context import FastEvidence
from fast_question_classifier import FastQuestionType
from fast_retrieval import _infer_must_cover
from rag_answer_lib import RetrievedChunk

TOKEN_RE = re.compile(r"[\w가-힣]{2,}", re.UNICODE)


@dataclass
class FastConfidenceResult:
    score: float
    low_confidence: bool
    keyword_hit_ratio: float
    has_citation_evidence: bool
    must_cover_hits: int
    must_cover_total: int
    has_table_evidence: bool
    reasons: list[str]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower())


def _question_keywords(question: str, row: dict) -> list[str]:
    terms: list[str] = []
    for key in ("expected_keywords", "expected_topics", "must_cover"):
        for t in row.get(key) or []:
            if t and str(t) not in terms:
                terms.append(str(t))
    for tok in TOKEN_RE.findall(question):
        if len(tok) >= 3 and tok.lower() not in {"에서", "관련", "주요", "결과", "요약"}:
            terms.append(tok)
    return terms[:12]


def assess_meeting_summary_answer_quality(answer: str) -> tuple[bool, list[str]]:
    return validate_meeting_summary_answer(answer)


def assess_fast_confidence(
    question: str,
    row: dict,
    evidence: list[FastEvidence],
    *,
    fast_type: FastQuestionType,
) -> FastConfidenceResult:
    chunks = [ev.chunk for ev in evidence]
    corpus = _norm(" ".join(c.text or "" for c in chunks))
    keywords = _question_keywords(question, row)
    hits = sum(1 for k in keywords if _norm(k) in corpus)
    kw_ratio = hits / max(len(keywords), 1)

    has_citation = any(
        (c.file_name or c.doc_id) and c.page_number is not None for c in chunks
    )
    if not has_citation:
        has_citation = any(c.file_name or c.doc_id for c in chunks)

    must_terms = _infer_must_cover(question, row) if fast_type in {
        "meeting_summary",
        "meeting_outcome_question",
    } else list(row.get("must_cover") or [])
    must_hits = sum(1 for t in must_terms if _norm(t) in corpus)

    has_table = any(
        c.chunk_type in {"table_row", "table_markdown", "table_summary"} or c.table_id
        for c in chunks
    )

    reasons: list[str] = []
    score = 1.0
    if kw_ratio < 0.25:
        score -= 0.35
        reasons.append("keyword_coverage_low")
    if not has_citation:
        score -= 0.25
        reasons.append("no_citation_evidence")
    if fast_type in {"meeting_summary", "meeting_outcome_question"} and must_terms and must_hits < max(
        1, len(must_terms) // 2
    ):
        score -= 0.3
        reasons.append("must_cover_insufficient")
    if fast_type == "table_question" and not has_table:
        score -= 0.35
        reasons.append("no_table_evidence")

    score = max(0.0, min(1.0, score))
    low = score < 0.55 or "no_citation_evidence" in reasons

    return FastConfidenceResult(
        score=round(score, 3),
        low_confidence=low,
        keyword_hit_ratio=round(kw_ratio, 3),
        has_citation_evidence=has_citation,
        must_cover_hits=must_hits,
        must_cover_total=len(must_terms),
        has_table_evidence=has_table,
        reasons=reasons,
    )
