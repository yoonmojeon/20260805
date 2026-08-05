"""Per-document analysis for Rule/Guidance structured answers."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from bm25_index import extract_document_codes
from rule_lookup_context import doc_code_in_corpus, strip_metadata_prefix

NEGATIVE_APPLICABILITY_RE = re.compile(
    r"(?:does|do)\s+not\s+apply\s+to\s+(?:remotely\s+operated|autonomous)"
    r"|not\s+applicable\s+to\s+(?:remotely|autonomous)"
    r"|excluding\s+remotely\s+operated"
    r"|does\s+not\s+cover\s+remotely\s+operated",
    re.I,
)
AUTONOMOUS_QUERY_RE = re.compile(
    r"자율|autonomous|smart\s*vessel|remotely|원격\s*운항|mass",
    re.I,
)
ENGLISH_SENTENCE_RE = re.compile(r"[A-Za-z][A-Za-z\s,;:'\"()-]{40,}")
DOC_CODE_FROM_NAME_RE = re.compile(
    r"(DNV|LR|ABS|KR)[-_ ]?(?:CG|RP|RU|CP|NV|OU)[-_ ]?[\w.-]+",
    re.I,
)


@dataclass
class DocAnalysis:
    file_name: str
    source: str
    doc_code: str
    doc_type: str
    page: int | None
    clause: str
    citation_ids: list[int]
    body: str
    negative_applicability: bool
    negative_snippet: str
    relevance_score: float
    confirmation: str  # 확정 | 후보 | 추가 확인 필요
    relevance_ko: str
    scope_ko: str
    summary_ko: str
    is_catalog_only: bool = False
    warnings: list[str] = field(default_factory=list)


def doc_code_from_file_name(file_name: str) -> str:
    fn = file_name or ""
    m = DOC_CODE_FROM_NAME_RE.search(fn.replace(".pdf", ""))
    if m:
        return re.sub(r"\s+", "-", m.group(0).strip())
    for code in extract_document_codes(fn):
        return code
    return fn.replace(".pdf", "").strip()


def infer_doc_type(file_name: str, body: str = "") -> str:
    blob = f"{file_name} {body[:300]}".upper()
    if "NOTICE" in blob and ("RULE" in blob or "REGULATION" in blob):
        return "LR Rules 후보" if "LR" in blob or "NOTICE NO" in blob else "Rule/Notice 후보"
    if "CG-" in blob or "GUIDELINE" in blob:
        return "Class Guideline"
    if "RP-" in blob:
        return "Guidance"
    if re.search(r"RU[- ]?OU", blob, re.I):
        return "Class Notation"
    if "NOTATION" in blob.lower():
        return "Class Notation"
    if "guidance" in blob.lower():
        return "Guidance"
    return "Rule"


def detect_negative_applicability(text: str) -> tuple[bool, str]:
    body = text or ""
    m = NEGATIVE_APPLICABILITY_RE.search(body)
    if m:
        snippet = re.sub(r"\s+", " ", m.group(0)).strip()[:120]
        return True, snippet
    return False, ""


def _topic_hits(text: str, question: str) -> int:
    ql = question.lower()
    body = text.lower()
    keys = []
    if AUTONOMOUS_QUERY_RE.search(ql):
        keys = ["autonomous", "remotely", "remote", "smart", "notation", "mass", "auto"]
    if "연료" in ql or "fuel" in ql:
        keys.extend(["fuel", "low-flashpoint", "alternative", "dual fuel", "section 15"])
    if not keys:
        keys = ["rule", "guidance", "requirement", "scope"]
    return sum(1 for k in keys if k in body or k in ql)


def relevance_score(body: str, file_name: str, question: str) -> float:
    base = _topic_hits(body, question) / max(len(question.split()), 1)
    fn = file_name.lower()
    ql = question.lower()
    if "cg-0264" in fn and AUTONOMOUS_QUERY_RE.search(question):
        base += 2.0
    if "autonomous" in body.lower() or "remotely" in body.lower():
        base += 0.5
    if ("연료" in ql or "fuel" in ql) and "notice" in fn:
        low = body.lower()
        if "section 15" in low or "low-flashpoint" in low or "low flashpoint" in low:
            base += 2.5
        elif "dual fuel" in low or "alternative fuel" in low:
            base += 1.5
    return base


def confirmation_status(
    *,
    doc_code: str,
    file_name: str,
    negative: bool,
    question: str,
    relevance: float,
    is_catalog: bool,
) -> str:
    if is_catalog:
        return "후보"
    if negative:
        return "추가 확인 필요"
    linked = doc_code_in_corpus(doc_code, {file_name}) or doc_code.replace("-", "") in file_name.replace("-", "")
    if not linked:
        return "후보"
    if relevance < 1.2:
        return "후보"
    return "확정"


def summarize_scope_ko(body: str, file_name: str, doc_type: str) -> str:
    low = body.lower()
    if "autonomous" in low and "remotely" in low:
        return "자율운항·원격운항(remotely operated/autonomous) 선박"
    if "autonomous" in low:
        return "자율운항(autonomous) 관련 선박·시스템"
    if "remotely operated" in low or "remote" in low:
        return "원격운항(remotely operated) 선박"
    if "smart" in low or "notation" in low:
        return "Smart/autonomous notation 적용 대상 선박"
    if "low-flashpoint" in low or "alternative fuel" in low:
        return "저인화점·대체연료(low-flashpoint/alternative fuel) 사용 선박"
    if "fuel" in low:
        return "연료 저장·공급·기관 설계 관련 선박"
    if doc_type == "Class Guideline":
        return "DNV class guideline 적용 대상 (본문 scope 참조)"
    return "해당 Rule/Guidance의 적용 범위 (본문 참조)"


def summarize_content_ko(
    body: str,
    file_name: str,
    *,
    negative: bool,
    negative_snippet: str,
) -> str:
    low = body.lower()
    if negative:
        return (
            "본문에 자율·원격운항 자산에 **적용 제외** 문구가 있어, "
            "질문 주제의 확정 Rule로 단정하기 어렵다."
        )
    if "cg-0264" in file_name.lower() or "autonomous and remotely" in low:
        return (
            "자율·원격운항 선박의 scope, autonomy level, notation(AUTO/REMO 등) "
            "및 class 승인·검증 요건을 규정한다."
        )
    if "notation" in low and ("smart" in low or "ou-" in file_name.lower()):
        return "Smart/autonomous 관련 class notation·기술 요건을 다루나, 적용 제외 범위를 본문에서 확인해야 한다."
    if "section 15" in low or "low-flashpoint" in low:
        return "저인화점 연료·기관·저장·공급 arrangement 관련 class 요건을 규정한다."
    if "scope" in low[:400]:
        return "문서 scope·적용 한계 및 관련 notation·승인 절차를 정의한다."
    return "검색된 본문에서 핵심 요건·적용 범위를 확인할 수 있다."


def summarize_relevance_ko(
    confirmation: str,
    *,
    negative: bool,
    question: str,
    doc_code: str,
) -> str:
    if negative and AUTONOMOUS_QUERY_RE.search(question):
        return f"{doc_code}는 Smart/autonomous notation 문서이나, 본문에 원격·자율운항 **적용 제외** 문구가 있어 질문과 직접 정합성이 낮음"
    if confirmation == "확정":
        if "cg-0264" in doc_code.lower() and AUTONOMOUS_QUERY_RE.search(question):
            return "자율운항 및 원격운항 선박의 설계, 운용, 승인, 검증 절차를 다루는 핵심 Guidance"
        if AUTONOMOUS_QUERY_RE.search(question):
            return "질문 주제와 직접 연결되는 핵심 Rule/Guidance"
        return "질문 주제와 문서명·본문 키워드가 부합함"
    if confirmation == "후보":
        return "관련 키워드는 있으나 확정 Rule로 단정하기엔 근거·적용 범위 확인 필요"
    return "적용 제외·범위 불명확 — 추가 본문 검토 필요"


def pick_citations(
    chunks: list[Any],
    file_name: str,
    *,
    max_cites: int = 2,
    fallback_pool: list[Any] | None = None,
) -> list[int]:
    ids: list[int] = []
    for i, c in enumerate(chunks, start=1):
        if str(getattr(c, "file_name", "")) != file_name:
            continue
        if getattr(c, "is_catalog_table", False):
            continue
        ids.append(i)
    if ids:
        return ids[:max_cites]
    if fallback_pool:
        for i, c in enumerate(fallback_pool, start=1):
            if str(getattr(c, "file_name", "")) == file_name:
                return [i]
    return []


def analyze_documents(
    chunks: list[Any],
    *,
    question: str,
    citation_chunks: list[Any] | None = None,
    citation_fallback: list[Any] | None = None,
    max_docs: int = 5,
) -> list[DocAnalysis]:
    cite_source = citation_chunks or chunks
    by_file: dict[str, list[Any]] = {}
    for c in chunks:
        fn = str(getattr(c, "file_name", "") or "")
        if not fn:
            continue
        by_file.setdefault(fn, []).append(c)

    analyses: list[DocAnalysis] = []
    for fn, group in by_file.items():
        best = max(
            group,
            key=lambda c: len(strip_metadata_prefix(getattr(c, "text", ""))),
        )
        body = strip_metadata_prefix(getattr(best, "text", ""))
        negative, neg_snip = detect_negative_applicability(body)
        doc_code = doc_code_from_file_name(fn)
        dtype = infer_doc_type(fn, body)
        rel = relevance_score(body, fn, question)
        catalog = any(getattr(c, "is_catalog_table", False) for c in group)
        conf = confirmation_status(
            doc_code=doc_code,
            file_name=fn,
            negative=negative,
            question=question,
            relevance=rel,
            is_catalog=catalog,
        )
        cites = pick_citations(
            cite_source, fn, max_cites=2, fallback_pool=citation_fallback
        )
        warnings: list[str] = []
        if negative:
            warnings.append("negative_applicability_clause")
        if rel < 1.0 and conf != "확정" and AUTONOMOUS_QUERY_RE.search(question):
            warnings.append("weak_relevance")

        analyses.append(
            DocAnalysis(
                file_name=fn,
                source=str(getattr(best, "source", "") or ""),
                doc_code=doc_code,
                doc_type=dtype,
                page=getattr(best, "page_number", None),
                clause=str(getattr(best, "clause_number", "") or getattr(best, "caption", "") or ""),
                citation_ids=cites,
                body=body,
                negative_applicability=negative,
                negative_snippet=neg_snip,
                relevance_score=rel,
                confirmation=conf,
                relevance_ko=summarize_relevance_ko(
                    conf, negative=negative, question=question, doc_code=doc_code
                ),
                scope_ko=summarize_scope_ko(body, fn, dtype),
                summary_ko=summarize_content_ko(
                    body, fn, negative=negative, negative_snippet=neg_snip
                ),
                is_catalog_only=catalog and len(body) < 120,
                warnings=warnings,
            )
        )

    analyses.sort(
        key=lambda d: (
            {"확정": 0, "추가 확인 필요": 1, "후보": 2}.get(d.confirmation, 3),
            -d.relevance_score,
        )
    )
    return analyses[:max_docs]


def format_citations(ids: list[int]) -> str:
    return "".join(f"[{i}]" for i in ids)


def detect_answer_quality_warnings(answer: str, analyses: list[DocAnalysis]) -> list[str]:
    flags: list[str] = []
    for a in analyses:
        flags.extend(a.warnings)
    if ENGLISH_SENTENCE_RE.search(answer):
        flags.append("raw_chunk_leak")
    cite_counts = re.findall(r"\[(\d+)\]", answer)
    if len(cite_counts) > 8:
        flags.append("too_many_citations")
    if any(a.negative_applicability for a in analyses) and "negative_applicability_clause" not in flags:
        flags.append("negative_applicability_clause")
    if sum(1 for a in analyses if a.confirmation != "확정") >= 2:
        flags.append("weak_relevance")
    return list(dict.fromkeys(flags))
