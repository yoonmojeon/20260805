"""Deterministic, evidence-bound document cards for broad Rule discovery."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from answer_depth_guidance import join_four_sections
from document_profile_catalog import load_document_profiles
from fast_context import question_focus_score
from rule_lookup_document_analysis import summarize_scope_ko


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}|[가-힣]{2,}")
STOP = {
    "관련", "찾아줘", "찾아주세요", "문서", "규칙", "지침", "선급",
    "rule", "rules", "guidance", "guide", "find", "related", "에서",
}
CONTRAST_RE = re.compile(r"(.{1,100}?)(?:가|이)?\s*아니라|(.{1,100}?)\s*말고", re.I)


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(text or "")
        if token.lower() not in STOP
    }


def _profile_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
        / "index"
        / "unified_full_corpus_715_v1"
        / "document_profiles_v1.json"
    )


def _doc_score(question: str, chunks: list[Any], profile: dict[str, Any]) -> float:
    file_name = str(getattr(chunks[0], "file_name", "") or "")
    blob = f"{file_name} {profile.get('display_code', '')}".lower()
    q_tokens = _tokens(question)
    score = sum(min(len(token), 12) for token in q_tokens if token in blob) * 1.3
    score += max(
        question_focus_score(str(getattr(chunk, "text", "") or ""), question)
        for chunk in chunks
    )
    normalized_q = re.sub(r"[^a-z0-9]", "", question.lower())
    normalized_code = re.sub(r"[^a-z0-9]", "", str(profile.get("display_code") or "").lower())
    if len(normalized_code) >= 6 and normalized_code in normalized_q:
        score += 40.0
    ql = question.lower()
    if "requirements" in ql and "requirements" in blob:
        score += 18.0
    if re.search(r"smart\s*function|스마트\s*기능", question, re.I) and "smartfunction" in re.sub(r"[^a-z]", "", blob):
        score += 16.0
    if re.search(r"autonomous|remote|자율|원격", question, re.I) and re.search(
        r"autonomous|remote|0264", blob, re.I
    ):
        score += 14.0
    if re.search(r"smart\s*vessel", question, re.I) and "0508" in blob:
        score += 16.0
    contrast = CONTRAST_RE.search(question or "")
    if contrast:
        excluded_terms = _tokens(next(value for value in contrast.groups() if value))
        overlap = sum(1 for token in excluded_terms if token in blob)
        score -= overlap * 14.0
    return score


def build_rule_document_cards(
    question: str,
    chunks: list[Any],
    *,
    max_documents: int = 3,
) -> tuple[str, list[Any]]:
    profiles = load_document_profiles(str(_profile_path().resolve()))
    by_doc: dict[str, list[Any]] = {}
    order: list[str] = []
    for chunk in chunks:
        doc_id = str(getattr(chunk, "doc_id", "") or "")
        file_name = str(getattr(chunk, "file_name", "") or "")
        if not doc_id or not file_name or len(str(getattr(chunk, "text", "") or "").strip()) < 50:
            continue
        if doc_id not in by_doc:
            order.append(doc_id)
        by_doc.setdefault(doc_id, []).append(chunk)
    ranked: list[tuple[float, int, str]] = []
    for index, doc_id in enumerate(order):
        ranked.append(
            (_doc_score(question, by_doc[doc_id], profiles.get(doc_id, {})), -index, doc_id)
        )
    ranked.sort(reverse=True)
    best_score = ranked[0][0] if ranked else 0.0

    selected: list[Any] = []
    cards: list[str] = []
    refs: list[str] = []
    for _score, _order, doc_id in ranked:
        # A retrieval pool intentionally contains near-neighbour documents.
        # A guide answer should not display every neighbour as equally
        # applicable; keep documents close to the best direct match only.
        if selected and (_score < 4.0 or _score < best_score - 18.0):
            continue
        doc_chunks = by_doc[doc_id]
        representative = max(
            doc_chunks,
            key=lambda chunk: question_focus_score(
                str(getattr(chunk, "text", "") or ""), question
            ),
        )
        profile = profiles.get(doc_id, {})
        file_name = str(getattr(representative, "file_name", "") or "")
        code = str(profile.get("display_code") or Path(file_name).stem)
        family = str(profile.get("document_family") or "Rule/Guidance")
        purpose = str(profile.get("purpose") or "질문 주제의 적용범위와 요구사항을 확인하는 문서")
        when_to_use = str(profile.get("when_to_use") or "설계·승인 검토 시")
        scope = summarize_scope_ko(
            str(getattr(representative, "text", "") or ""), file_name, family
        )
        selected.append(representative)
        cite = len(selected)
        page = getattr(representative, "page_number", "?")
        clause = str(getattr(representative, "clause_number", "") or "")
        locator = f"p.{page}" + (f", clause {clause}" if clause else "")
        cards.append(
            f"- **{code}** ({file_name}, {locator}) — **성격**: {purpose}; "
            f"**주요 범위**: {scope}; **활용 시점**: {when_to_use}. [{cite}]"
        )
        refs.append(f"**{code}**, {locator} [{cite}]")
        if len(selected) >= max_documents:
            break

    if not cards:
        return "", []
    answer = join_four_sections(
        {
            "1": "\n".join(cards),
            "2": "",
            "3": "",
            "4": "",
        }
    )
    return answer, selected
