"""Group and format Rule/Guidance analyses for user-facing answers."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rule_lookup_document_analysis import (
    AUTONOMOUS_QUERY_RE,
    DocAnalysis,
    analyze_documents,
    doc_code_in_corpus,
    format_citations,
)
MAX_CONFIRMED = 2
MAX_CANDIDATE_GROUPS = 1

RU_OU_FAMILY_RE = re.compile(r"RU[- ]?OU", re.I)
SOCIETY_FROM_QUESTION_RE = re.compile(r"\b(DNV|LR|ABS|KR)\b", re.I)


@dataclass
class CandidateGroup:
    """Bundled non-confirmed documents with the same disqualifying reason."""
    group_id: str
    label: str
    members: list[DocAnalysis]
    confirmation: str
    doc_type: str
    relevance_ko: str
    reason_ko: str
    citation_ids: list[int] = field(default_factory=list)
    file_names: list[str] = field(default_factory=list)

    @property
    def doc_codes(self) -> list[str]:
        return [m.doc_code for m in self.members]


@dataclass
class RuleLookupPresentation:
    question: str
    society: str
    confirmed: list[DocAnalysis]
    candidate_groups: list[CandidateGroup]
    catalog_codes: list[str]
    all_analyses: list[DocAnalysis]
    clause_themes: list[Any] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def society_from_question(question: str, docs: list[DocAnalysis]) -> str:
    m = SOCIETY_FROM_QUESTION_RE.search(question or "")
    if m:
        return m.group(1).upper()
    for d in docs:
        if d.source:
            return str(d.source).upper()
    return ""


def _family_group_id(d: DocAnalysis, question: str) -> str | None:
    is_ru_ou = bool(RU_OU_FAMILY_RE.search(d.doc_code))
    autonomous_q = bool(AUTONOMOUS_QUERY_RE.search(question))
    if is_ru_ou and autonomous_q and d.confirmation != "확정":
        return "ru_ou_negative" if d.negative_applicability else "ru_ou_family"
    if d.negative_applicability and is_ru_ou:
        return "ru_ou_negative"
    if d.negative_applicability:
        return "negative_other"
    if d.confirmation == "후보" and is_ru_ou:
        return "ru_ou_weak"
    return None


def _merge_citations(members: list[DocAnalysis], *, max_cites: int = 2) -> list[int]:
    seen: list[int] = []
    for m in members:
        for cid in m.citation_ids:
            if cid not in seen:
                seen.append(cid)
            if len(seen) >= max_cites:
                return seen
    return seen


def _build_candidate_group(group_id: str, members: list[DocAnalysis], question: str) -> CandidateGroup:
    members = sorted(members, key=lambda d: -d.relevance_score)
    codes = "/".join(m.doc_code for m in members[:4])
    if len(members) > 4:
        codes += " 등"
    cites = _merge_citations(members)

    if group_id == "ru_ou_negative" or group_id == "ru_ou_family":
        label = "DNV-RU-OU 계열 Smart notation 문서"
        dtype = "Smart notation 관련 Rule/Class notation 후보"
        rel = (
            "Smart notation 관련 문서로 검색되었으나, "
            "자율·원격운항 자산 적용 제외 문구가 있어 직접 관련성은 낮음"
        )
        has_negative = any(m.negative_applicability for m in members)
        if has_negative:
            reason = (
                f"{codes}는 Smart notation 후보로 검색되었으나, "
                "자율·원격운항 자산 적용 제외 문구가 있어 확정 Rule로 보기 어렵습니다."
            )
            conf = "추가 확인 필요"
        else:
            reason = f"{codes}는 Smart notation 관련 후보로 검색되었으나 적용 범위 추가 확인이 필요합니다."
            conf = "후보"
    elif group_id == "ru_ou_weak":
        label = "DNV-RU-OU 계열 문서"
        dtype = "Rule/Class notation 후보"
        rel = "키워드는 유사하나 적용 범위·본문 정합 추가 확인 필요"
        reason = f"{codes}는 관련 후보로 검색되었으나 확정 Rule로 단정하기 어렵습니다."
        conf = "후보"
    else:
        label = f"{members[0].doc_code} 외 유사 후보"
        dtype = "Rule/Class notation 후보"
        rel = members[0].relevance_ko
        reason = f"{codes}는 적용 범위 확인이 필요한 후보입니다."
        conf = "추가 확인 필요"

    return CandidateGroup(
        group_id=group_id,
        label=label,
        members=members,
        confirmation=conf,
        doc_type=dtype,
        relevance_ko=rel,
        reason_ko=reason,
        citation_ids=cites,
        file_names=[m.file_name for m in members],
    )


def build_presentation(
    analyses: list[DocAnalysis],
    *,
    question: str,
    catalog_codes: list[str] | None = None,
    clause_themes: list[Any] | None = None,
) -> RuleLookupPresentation:
    catalog_codes = catalog_codes or []
    society = society_from_question(question, analyses)

    confirmed = [d for d in analyses if d.confirmation == "확정"][:MAX_CONFIRMED]
    non_confirmed = [d for d in analyses if d.confirmation != "확정"]

    # Stricter confirmed: must link doc_code ↔ file_name
    strict_confirmed: list[DocAnalysis] = []
    demoted: list[DocAnalysis] = []
    for d in confirmed:
        if doc_code_in_corpus(d.doc_code, {d.file_name}) or d.doc_code.replace("-", "") in d.file_name.replace("-", ""):
            strict_confirmed.append(d)
        else:
            d.confirmation = "후보"
            demoted.append(d)
    confirmed = strict_confirmed[:MAX_CONFIRMED]
    non_confirmed = demoted + [d for d in non_confirmed if d not in demoted]

    groups_by_id: dict[str, list[DocAnalysis]] = {}
    loose: list[DocAnalysis] = []
    for d in non_confirmed:
        gid = _family_group_id(d, question)
        if gid:
            # Merge ru_ou_family into ru_ou_negative bucket when both appear.
            bucket = "ru_ou_negative" if gid in ("ru_ou_negative", "ru_ou_family") else gid
            groups_by_id.setdefault(bucket, []).append(d)
        else:
            loose.append(d)

    candidate_groups: list[CandidateGroup] = []
    priority = ("ru_ou_negative", "ru_ou_weak", "negative_other")
    for gid in priority:
        if gid in groups_by_id and len(candidate_groups) < MAX_CANDIDATE_GROUPS:
            candidate_groups.append(_build_candidate_group(gid, groups_by_id[gid], question))
            del groups_by_id[gid]

    if len(candidate_groups) < MAX_CANDIDATE_GROUPS and loose:
        candidate_groups.append(_build_candidate_group("misc_weak", loose[:3], question))

    warnings: list[str] = []
    if any(d.negative_applicability for d in analyses):
        warnings.append("negative_applicability_clause")
    if candidate_groups or non_confirmed:
        warnings.append("candidate_not_confirmed")
    if len(non_confirmed) > 3 or len(groups_by_id) > 1:
        warnings.append("too_many_candidates")
    if sum(1 for d in analyses if d.confirmation != "확정") >= 2:
        warnings.append("weak_relevance")

    return RuleLookupPresentation(
        question=question,
        society=society,
        confirmed=confirmed,
        candidate_groups=candidate_groups[:MAX_CANDIDATE_GROUPS],
        catalog_codes=catalog_codes,
        all_analyses=analyses,
        clause_themes=list(clause_themes or []),
        warnings=warnings,
    )


def confirmed_relevance_ko(d: DocAnalysis, question: str) -> str:
    low = d.file_name.lower()
    if "cg-0264" in low and AUTONOMOUS_QUERY_RE.search(question):
        return "자율운항 및 원격운항 선박의 설계, 운용, 승인, 검증 절차를 다루는 핵심 Guidance"
    if "notice" in low and ("fuel" in question.lower() or "연료" in question):
        return "대체연료·저인화점 연료 관련 기관·저장·공급 class 요건을 다루는 Rule"
    if d.doc_type == "Class Guideline":
        return "질문 주제와 직접 연결되는 class guideline"
    return "검색된 본문과 문서명이 일치하는 핵심 Rule/Guidance"
