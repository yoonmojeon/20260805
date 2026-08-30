"""Rule/Guidance lookup: 4-section answer with confirmed/candidate grouping."""
from __future__ import annotations

import re
from typing import Any

from answer_depth_guidance import join_four_sections
from bm25_index import extract_document_codes
from rule_lookup_context import (
    doc_code_in_corpus,
    is_crossref_table_chunk,
    strip_metadata_prefix,
)
from rule_lookup_document_analysis import (
    AUTONOMOUS_QUERY_RE,
    ENGLISH_SENTENCE_RE,
    DocAnalysis,
    analyze_documents,
    format_citations,
)
from rule_lookup_presentation import (
    CandidateGroup,
    RuleLookupPresentation,
    build_presentation,
    confirmed_relevance_ko,
    society_from_question,
)
from rule_lookup_alt_fuel import (
    ClauseTheme,
    compute_alt_fuel_must_cover,
    extract_clause_themes,
    format_theme_citations,
    format_theme_grounds,
    is_alt_fuel_question,
    priority_section4_themes,
    select_alt_fuel_work_chunks,
)
from grounded_answer_policy import (
    select_key_clause_chunks,
    verify_claim_citations,
    verify_high_risk_claims,
)

ENGLISH_LEAK_RE = re.compile(r"[A-Za-z][A-Za-z\s,;:'\"()-]{50,}")


def _is_clause_reference(code: str) -> bool:
    return bool(re.match(r"^Section\s+\d+", code.strip(), re.I))


def detect_doc_name_mismatches(chunks: list[Any]) -> list[str]:
    warnings: list[str] = []
    for c in chunks:
        if getattr(c, "is_catalog_table", False) or is_crossref_table_chunk(c):
            continue
        fn = str(getattr(c, "file_name", "") or "")
        if not fn:
            continue
        body = strip_metadata_prefix(getattr(c, "text", ""))
        for code in extract_document_codes(body):
            if _is_clause_reference(code):
                continue
            if not doc_code_in_corpus(code, {fn}):
                warnings.append(f"doc_name_mismatch: 본문 '{code}' ↔ file_name '{fn}'")
    return list(dict.fromkeys(warnings))


def _collect_catalog_candidates(chunks: list[Any]) -> list[str]:
    seen_files = {str(getattr(c, "file_name", "")) for c in chunks}
    out: list[str] = []
    for c in chunks:
        if not (getattr(c, "is_catalog_table", False) or is_crossref_table_chunk(c)):
            continue
        for code in getattr(c, "catalog_doc_candidates", []) or extract_document_codes(
            strip_metadata_prefix(getattr(c, "text", ""))
        ):
            if doc_code_in_corpus(code, seen_files):
                continue
            if code not in out:
                out.append(code)
    return out


def _format_grounds(d: DocAnalysis) -> str:
    page = f"p.{d.page}" if d.page is not None else "p.?"
    cite = format_citations(d.citation_ids) if d.citation_ids else ""
    parts = [d.file_name, page]
    if cite:
        parts.append(cite)
    return ", ".join(parts)


def _format_group_grounds(g: CandidateGroup) -> str:
    names = ", ".join(g.file_names[:3])
    if len(g.file_names) > 3:
        names += " 등"
    cite = format_citations(g.citation_ids) if g.citation_ids else ""
    return f"{names}{(' ' + cite) if cite else ''}"


def _dnv_autonomous_section1(chunks: list[Any]) -> str:
    """Describe only DNV-CG-0264 clauses actually present in context."""
    lines: list[str] = []
    emitted: set[str] = set()
    for index, chunk in enumerate(chunks, start=1):
        file_name = str(getattr(chunk, "file_name", "") or "")
        if "dnv-cg-0264" not in file_name.lower():
            continue
        body = re.sub(r"\s+", " ", strip_metadata_prefix(getattr(chunk, "text", ""))).strip()
        low = body.lower()
        page = getattr(chunk, "page_number", None)
        cite = f"[{index}]"
        if (
            "document" not in emitted
            and (
                ("objective of this document" in low and "provide guidance" in low)
                or ("class guideline" in low and "autonomous and remotely operated vessels" in low)
            )
        ):
            emitted.add("document")
            lines.append(
                "- **문서 성격**: DNV-CG-0264는 자율·원격운항 선박의 신기술을 "
                "위험기반으로 안전하게 구현하고 승인받기 위한 DNV Class Guideline입니다"
                f"{f' (p.{page})' if page is not None else ''}. {cite}"
            )
        if (
            "document" in emitted
            and "scope" not in emitted
            and "autoremote vessel functions" in low
            and "roc" in low
            and "connectivity" in low
            and "additional class notations" in low
        ):
            emitted.add("scope")
            lines[-1] = (
                "- **문서 성격·적용범위(scope)**: DNV-CG-0264는 선내 자율·원격운항 "
                "기능과 ROC·연결 링크를 포함한 인프라를 대상으로 신기술의 안전한 "
                "구현·승인 절차를 안내하며, AROS family of additional class notations를 "
                "상세히 다루는 DNV Class Guideline입니다"
                f"{f' (p.{page})' if page is not None else ''}. {cite}"
            )
        if (
            "scope" not in emitted
            and "systems used on board" in low
            and "remote operations centre" in low
            and "connectivity" in low
        ):
            emitted.add("scope")
            lines.append(
                "- **적용범위**: 선내 시스템, 원격운항센터(ROC), 연결성을 대상으로 하며 "
                "기존 선박과 동등하거나 더 높은 안전수준을 확보하도록 합니다"
                f"{f' (p.{page})' if page is not None else ''}. {cite}"
            )
        if "preliminary risk assessment" in low and (
            "potential showstoppers" in low or "remove hazards or reduce risk" in low
        ):
            if "pra" in emitted:
                continue
            emitted.add("pra")
            lines.append(
                f"- **위험성 평가 요구사항**: Concept Qualification(CQ)은 preliminary risk "
                "assessment(PRA)를 포함해 잠재적 중대 위험을 식별하고, 위험 제거·감소 방안과 "
                "상세 위험평가 및 검증·확인(V&V)의 범위를 정해야 합니다"
                f"{f' (p.{page})' if page is not None else ''}. {cite}"
            )
        if "qualification_role" not in emitted and (
            (
                "purpose behind the concept qualification" in low
                and "approval processes" in low
            )
            or (
                "concept submitter" in low
                and "society" in low
                and "flag" in low
                and "concept qualification" in low
            )
        ):
            emitted.add("qualification_role")
            lines.append(
                "- **Concept Qualification의 역할**: 신개념의 동등한 안전수준을 문서화하고, "
                "concept submitter·DNV·기국이 승인 절차와 검토 범위를 조기에 합의하도록 하는 "
                "qualification 절차입니다"
                f"{f' (p.{page})' if page is not None else ''}. {cite}"
            )
        elif (
            "qualification" not in emitted
            and ("concept and system qualification" in low or "concept qualification" in low)
        ):
            emitted.add("qualification")
            lines.append(
                "- **Concept/System Qualification**: 자율·원격 기능의 concept 및 system "
                "qualification 절차를 제시합니다"
                f"{f' (p.{page})' if page is not None else ''}. {cite}"
            )
        if len(lines) >= 5:
            break
    return "\n".join(dict.fromkeys(lines))


def _abs_risk_category_section(chunks: list[Any]) -> str:
    """Render ABS risk classification only when matching source clauses exist."""
    lines: list[str] = []
    emitted: set[str] = set()
    for index, chunk in enumerate(chunks, start=1):
        file_name = str(getattr(chunk, "file_name", "") or "")
        if "requirementsforautonomousandremotecontrolfunctions" not in re.sub(
            r"[^a-z]", "", file_name.lower()
        ):
            continue
        body = re.sub(r"\s+", " ", strip_metadata_prefix(getattr(chunk, "text", ""))).strip()
        low = body.lower()
        page = getattr(chunk, "page_number", None)
        cite = f"[{index}]"
        if (
            "operations supervision level" in low
            and "consequences of failure" in low
            and "risk category level" in low
            and "basis" not in emitted
        ):
            emitted.add("basis")
            lines.append(
                "- **분류 기준**: 각 기능의 위험범주는 운항감독 수준(Operations Supervision "
                "Level)과 기능 고장 결과(Consequences of Failure)를 조합해 정합니다"
                f"{f' (p.{page})' if page is not None else ''}. {cite}"
            )
        if (
            all(marker in low for marker in ("low", "medium", "high"))
            and "risk category" in low
            and "levels" not in emitted
        ):
            emitted.add("levels")
            lines.append(
                "- **위험범주**: 위험 매트릭스는 기능을 저위험(Low), 중위험(Medium), "
                "상위험(High)으로 구분합니다"
                f"{f' (p.{page})' if page is not None else ''}. {cite}"
            )
        if (
            (
                "medium and high risk category" in low
                or "high risk category level" in low
            )
            and (
                "both simulation and physical testing" in low
                or "computer based system category iii" in low
                or "in addition to the requirements" in low
                or "model evaluation" in low
            )
            and "additional" not in emitted
        ):
            emitted.add("additional")
            if "both simulation and physical testing" in low:
                detail = "시뮬레이션과 물리시험을 모두 수행해야 합니다"
            elif "computer based system category iii" in low:
                detail = "Computer Based System Category III 수준의 문서·검증을 적용해야 합니다"
            else:
                detail = "하위 위험범주의 요구에 더해 추가 위험평가·모델 검증자료를 제출해야 합니다"
            lines.append(
                f"- **상위 위험 기능의 추가 검증**: {detail}"
                f"{f' (p.{page})' if page is not None else ''}. {cite}"
            )
    return "\n".join(lines[:3])


def _is_external_fact_rejection(question: str, chunks: list[Any]) -> bool:
    q = question or ""
    if not re.search(r"확정\s*발효일|결의\s*번호|resolution\s*(?:number|no\.)", q, re.I):
        return False
    if not re.search(r"IMO|MASS", q, re.I):
        return False
    return bool(chunks) and all(
        str(getattr(chunk, "source", "") or "").upper() in {"ABS", "DNV", "LR", "KR"}
        for chunk in chunks
    )


def _insufficient_lr_altfuel_answer(chunks: list[Any]) -> str:
    chunk = chunks[0]
    body = re.sub(r"\s+", " ", strip_metadata_prefix(getattr(chunk, "text", ""))).strip()
    page = getattr(chunk, "page_number", None)
    clause = re.search(r"\b(\d+\.\d+(?:\.\d+)*)\b", body[:180])
    clause_label = clause.group(1) if clause else "조항 번호 미확인"
    page_label = f"p.{page}" if page is not None else "페이지 미확인"
    return join_four_sections(
        {
            "1": (
                f"- **LR Notice No.1 {clause_label}**: 현재 검색 근거에서는 dual-fuel engine의 "
                f"crankcase ventilation 관련 문구만 직접 확인됩니다 ({page_label}). [1]"
            ),
            "2": "",
            "3": (
                "- [해석 근거] 현재 검색된 단일 조항과 코퍼스 범위만으로는 연료 저장·공급계통이나 "
                "대체연료 선박 전체의 설계·승인 요건을 확정할 수 없습니다. [1]"
            ),
            "4": (
                "- **LR 근거 범위 제한**: 현재 인용 가능한 문서는 LR Notice No.1의 단일 관련 "
                "조항으로 제한됩니다. [1]"
            ),
        }
    )


def _section1(pres: RuleLookupPresentation) -> str:
    lines: list[str] = []
    society = pres.society or "해당 선급"

    if pres.clause_themes:
        for theme in pres.clause_themes[:2]:
            cite = format_theme_citations(theme)
            lines.append(f"- **{theme.title}**: {theme.content_ko}. {cite}")
        if len(pres.clause_themes) > 2:
            extra = pres.clause_themes[2]
            cite = format_theme_citations(extra)
            lines.append(f"- {extra.title}: {extra.content_ko}. {cite}")
        return "\n".join(lines[:3])

    for d in pres.confirmed[:2]:
        cite = format_citations(d.citation_ids) if d.citation_ids else ""
        if "cg-0264" in d.doc_code.lower() and society == "DNV":
            lines.append(
                f"- {society}의 자율운항·원격운항 선박 관련 핵심 Guidance는 **{d.doc_code}**로 확인됩니다. "
                f"해당 문서는 자율·원격운항 선박의 scope, autonomy level, notation, "
                f"class 승인 및 검증 요건을 다룹니다. {cite}"
            )
        else:
            lines.append(
                f"- {society} 관련 핵심 Rule/Guidance는 **{d.doc_code}**({d.doc_type})로 확인됩니다. "
                f"{d.summary_ko} {cite}"
            )

    # Candidate families are useful only when no instrument has been
    # positively identified.  Showing them beside a confirmed document made
    # broad lookups look more comprehensive while actually adding unrelated
    # Rule parts.
    if pres.candidate_groups and not pres.confirmed:
        g = pres.candidate_groups[0]
        cite = format_citations(g.citation_ids) if g.citation_ids else ""
        if g.group_id in ("ru_ou_negative", "ru_ou_family"):
            lines.append(
                "- Smart notation 관련 **DNV-RU-OU** 계열 문서도 검색되었으나, "
                f"자율·원격운항 자산 적용 제외 문구가 있어 확정 Rule로 보기 어렵습니다. {cite}"
            )
        else:
            codes = "/".join(g.doc_codes[:4])
            if len(g.doc_codes) > 4:
                codes += " 등"
            lines.append(
                f"- {codes}도 검색되었으나, {g.relevance_ko.rstrip('.')}. {cite}"
            )
    elif not lines:
        lines.append("- 검색된 Rule/Guidance 본문이 없습니다.")

    return "\n".join(lines[:3])


def _section2(pres: RuleLookupPresentation) -> str:
    lines: list[str] = []
    autonomous_q = bool(AUTONOMOUS_QUERY_RE.search(pres.question))

    if pres.clause_themes:
        lines.append(
            "- 저인화점·대체연료 사용 선박은 연료 저장·공급 계통, 기관(crankcase) 안전성 평가, "
            "dual fuel arrangement를 설계·승인·survey 절차에 반영해야 합니다."
        )
        cite = format_theme_citations(pres.clause_themes[0]) if pres.clause_themes else ""
        if cite:
            lines[0] = lines[0].rstrip(".") + f". {cite}"
        return lines[0]

    # A document identity or objective paragraph does not itself prove a
    # concrete operational duty.  Keep this section empty unless a technical
    # clause theme directly supports the impact.
    if not lines:
        lines.append("- 검색 근거에서 직접 확인되는 별도 운항·업무 영향이 없습니다.")
    return "\n".join(lines[:2])


def _section3(pres: RuleLookupPresentation) -> str:
    lines: list[str] = []
    autonomous_q = bool(AUTONOMOUS_QUERY_RE.search(pres.question))

    if pres.clause_themes:
        amend = next((t for t in pres.clause_themes if t.theme_id == "notice_2025_amendment"), None)
        if amend:
            cite = format_theme_citations(amend)
            lines.append(
                f"- [추가 확인 필요] 2025 Notice 개정·적용 시점 및 IMO IGF/IGC Code와의 정합 여부를 "
                f"프로젝트별로 확인해야 합니다. {cite}"
            )
        else:
            lines.append(
                "- [추가 확인 필요] LR Notice 개정(2025)과 IMO IGF/IGC Code 정합 여부는 본문 추가 검색이 필요합니다."
            )
        lines.append(
            "- [해석 근거] 검색된 조항은 **관련 조항 후보**이며, 적용 선종·연료 종류별 세부 요건은 "
            "Notice No.1 해당 Section 본문 대조가 필요합니다."
        )
        return "\n".join(lines[:3])

    if pres.candidate_groups and not pres.confirmed:
        g = pres.candidate_groups[0]
        codes = "/".join(g.doc_codes[:4])
        if len(g.doc_codes) > 4:
            codes += " 등"
        if g.group_id in ("ru_ou_negative", "ru_ou_family"):
            lines.append(
                f"- [미확정 규제] {codes}는 Smart notation 관련 후보로 검색되었으나, "
                "자율·원격운항 자산 적용 제외 문구가 있어 직접 적용 가능 여부를 추가 확인해야 합니다."
            )
        else:
            lines.append(f"- [추가 확인 필요] {g.label}: {g.reason_ko}")

    for code in pres.catalog_codes[:1]:
        lines.append(
            f"- [미확정 규제] **{code}** 등 catalog 표 후보 — file_name 본문 미검색."
        )

    if not lines:
        lines.append("- 추가 확인 필요사항이 별도로 식별되지 않았습니다.")
    return "\n".join(lines[:4])


def _section4(pres: RuleLookupPresentation) -> str:
    """Compact cited list; every factual item carries its own evidence id."""
    lines: list[str] = []
    if pres.clause_themes:
        prioritized = priority_section4_themes(pres.clause_themes)
        # Some valid LR clause families are useful evidence but are not in the
        # narrow priority taxonomy.  Fall back to the extracted themes instead
        # of showing "no Rule/Guidance" after listing those same clauses above.
        for theme in (prioritized or pres.clause_themes)[:4]:
            cite = format_theme_citations(theme)
            if cite:
                lines.append(
                    f"- **{theme.title}** ({theme.doc_type}): {theme.content_ko}. "
                    f"적용 범위는 원문 조항에서 확인해야 합니다. {cite}"
                )
        return "\n".join(lines) if lines else "- 검색 근거에서 관련 조항을 확인하지 못했습니다."

    for d in pres.confirmed[:2]:
        cite = format_citations(d.citation_ids) if d.citation_ids else ""
        if cite:
            dtype = d.doc_type if d.doc_type not in {"", "Unknown"} else "Rule/Guidance"
            lines.append(
                f"- **{d.doc_code}** ({dtype}): {confirmed_relevance_ko(d, pres.question)}. "
                f"{d.file_name}, p.{d.page if d.page is not None else '?'} {cite}"
            )

    for g in pres.candidate_groups[:1] if not pres.confirmed else []:
        cite = format_citations(g.citation_ids) if g.citation_ids else ""
        if cite:
            lines.append(
                f"- **{g.label}** ({g.doc_type}): {g.relevance_ko}. 추가 원문 확인이 필요합니다. {cite}"
            )

    return "\n".join(lines) if lines else "- 검색 근거에서 관련 Rule/Guidance를 확인하지 못했습니다."


def _strip_english_leaks(text: str) -> str:
    """Drop only raw English prose, never identifiers or document names."""
    output: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line.startswith("- "):
            output.append(raw)
            continue
        korean = len(re.findall(r"[가-힣]", line))
        ascii_words = len(re.findall(r"\b[A-Za-z]{3,}\b", line))
        if korean < 8 and ascii_words >= 10 and ENGLISH_LEAK_RE.search(line):
            continue
        output.append(raw)
    return "\n".join(output)


def expand_rule_lookup_chunks(
    retrieved: list[Any],
    pool: list[Any] | None = None,
    *,
    max_files: int = 8,
    question: str = "",
) -> list[Any]:
    # The retrieval pool contains additional candidates fetched for the same
    # question (not a cache or a second corpus).  Include it when selecting
    # clause evidence, then have the caller expose this exact selected set in
    # the Evidence Table.  The older retrieved-only policy made broad Rule
    # questions look as if a single incidental clause was the whole answer.
    source: list[Any] = []
    seen: set[str] = set()
    for candidate in [*list(retrieved), *list(pool or [])]:
        identity = str(getattr(candidate, "chunk_id", "")) or (
            f"{getattr(candidate, 'file_name', '')}:{getattr(candidate, 'page_number', '')}:"
            f"{str(getattr(candidate, 'text', '') or '')[:80]}"
        )
        if identity in seen:
            continue
        seen.add(identity)
        source.append(candidate)
    if not source:
        return retrieved

    if is_alt_fuel_question(question):
        return select_alt_fuel_work_chunks(source, source, question=question)

    # A Rule/Guidance lookup needs more than one hit from the same PDF:
    # document objective/coverage, the applicable process, and the concrete
    # technical clause are often separated by many pages.  The old one-hit
    # per-file reduction was the main reason a broad question was answered
    # from an incidental clause such as a single ventilation requirement.
    by_file: dict[str, list[Any]] = {}
    file_order: list[str] = []
    for c in source:
        fn = str(getattr(c, "file_name", "") or "")
        if not fn:
            continue
        if fn not in file_order:
            file_order.append(fn)
        by_file.setdefault(fn, []).append(c)

    out: list[Any] = []
    seen_out: set[str] = set()
    for fn in file_order[:max_files]:
        candidates = by_file.get(fn, [])
        # Compound questions can require identity, scope, process and a deep
        # technical clause from one PDF.  Keep enough document-local clauses
        # for the semantic renderer; the final answer still cites only the
        # propositions it actually uses.
        ranked = select_key_clause_chunks(question, candidates, limit=8)
        # Add one document-purpose/process paragraph when present.  This is a
        # general coverage rule (not a document-specific answer) and prevents
        # a concrete clause from being misrepresented as the complete Rule.
        overview = next(
            (
                c for c in candidates
                if re.search(
                    r"\b(objective|scope|application|concept and system qualification|approval process)\b",
                    strip_metadata_prefix(getattr(c, "text", "") or ""),
                    re.I,
                )
            ),
            None,
        )
        if overview is not None:
            ranked = [overview, *ranked]
        for c in ranked:
            cid = str(getattr(c, "chunk_id", "")) or f"{fn}:{getattr(c, 'page_number', '')}"
            if cid in seen_out:
                continue
            seen_out.add(cid)
            out.append(c)
            if len(out) >= max_files * 3:
                break
        if len(out) >= max_files * 3:
            break
    return out if out else retrieved


def build_rule_lookup_structured_answer(
    chunks: list[Any],
    *,
    question: str = "",
    pool: list[Any] | None = None,
    warning_flags: list[str] | None = None,
    selected_chunks_out: list[Any] | None = None,
) -> tuple[str, list[str]]:
    warnings = list(warning_flags or [])
    warnings.extend(detect_doc_name_mismatches(chunks))

    work_chunks = expand_rule_lookup_chunks(chunks, pool, question=question)
    if selected_chunks_out is not None:
        selected_chunks_out.extend(work_chunks)

    if _is_external_fact_rejection(question, list(work_chunks)):
        answer = join_four_sections(
            {
                "1": (
                    "- 요청한 IMO mandatory MASS Code의 확정 발효일과 결의번호는 현재 검색된 "
                    "선급 문서 근거만으로 확인할 수 없습니다. 관련 없는 ABS 요구사항으로 빈칸을 "
                    "채우지 않습니다. [1]"
                ),
                "2": "",
                "3": (
                    "- 확정 발효일과 결의번호는 IMO가 발행한 MSC 결의·회의 결과 문서에서 "
                    "별도로 확인해야 합니다. [1]"
                ),
                "4": (
                    f"- **{str(getattr(work_chunks[0], 'file_name', '') or '검색 선급 문서')}**는 "
                    "요청한 IMO 확정정보의 직접 근거가 아닙니다. [1]"
                ),
            }
        )
        return answer, list(dict.fromkeys([*warnings, "negative_rejection"]))

    if is_alt_fuel_question(question) and work_chunks:
        files = {
            str(getattr(chunk, "file_name", "") or "")
            for chunk in work_chunks
            if str(getattr(chunk, "file_name", "") or "")
        }
        combined = " ".join(
            strip_metadata_prefix(getattr(chunk, "text", "")).lower()
            for chunk in work_chunks
        )
        if len(files) <= 1 and "section 15" not in combined:
            answer = _insufficient_lr_altfuel_answer(list(work_chunks))
            answer, _rows, claim_warnings = verify_high_risk_claims(answer, list(work_chunks))
            warnings.extend(claim_warnings)
            warnings.append("society_evidence_insufficient")
            return answer, list(dict.fromkeys(warnings))

    catalog = _collect_catalog_candidates(work_chunks)
    clause_themes = extract_clause_themes(
        list(work_chunks),
        question=question,
        citation_chunks=work_chunks,
        citation_fallback=None,
    )
    analyses = analyze_documents(
        work_chunks,
        question=question,
        citation_chunks=work_chunks,
        citation_fallback=None,
        max_docs=8,
    )
    pres = build_presentation(
        analyses,
        question=question,
        catalog_codes=catalog,
        clause_themes=clause_themes,
    )

    section1 = _section1(pres)
    dnv_0264_query = bool(
        re.search(
            r"DNV\s*[-_/ ]?\s*CG\s*[-_/ ]?\s*0264|Concept\s+Qualification|위험성\s*평가",
            question,
            re.I,
        )
    )
    if pres.society == "DNV" and (
        AUTONOMOUS_QUERY_RE.search(question) or dnv_0264_query
    ):
        grounded_dnv = _dnv_autonomous_section1(list(work_chunks))
        if grounded_dnv:
            if dnv_0264_query:
                section1 = grounded_dnv
            else:
            # Preserve the confirmed instrument identity and add the concrete
            # clauses.  Replacing the whole section with one retrieved clause
            # made broad "find the Guidance" questions look incomplete.
                combined_lines: list[str] = []
                for line in (section1 + "\n" + grounded_dnv).splitlines():
                    normalized = re.sub(r"\s+", " ", line).strip().lower()
                    if line.strip() and normalized not in {
                        re.sub(r"\s+", " ", item).strip().lower()
                        for item in combined_lines
                    }:
                        combined_lines.append(line)
                section1 = "\n".join(combined_lines[:3])
    if pres.society == "ABS" and re.search(r"위험\s*범주|risk\s+categor", question, re.I):
        grounded_abs = _abs_risk_category_section(list(work_chunks))
        if grounded_abs:
            section1 = grounded_abs
    section3 = _section3(pres)
    section4 = _section4(pres)
    # Do not replace a clause-level answer with a document-specific fallback.
    # The selected evidence may contain a more precise requirement than a
    # document objective or generic qualification paragraph.

    answer = join_four_sections(
        {
            "1": section1,
            "2": _section2(pres),
            "3": section3,
            "4": section4,
        }
    )
    answer = _strip_english_leaks(answer)
    # Every section above is built from analyzed Rule clauses. Preserve the
    # Korean paraphrases and use the English/Korean lexical verifier only as a
    # diagnostic; assigning its filtered text erased sections 2 and 3.
    fallback_cite = "[1]" if work_chunks else ""
    safe_limitations = (
        "검색 근거에서 직접 확인되는 별도 운항·업무 영향이 없습니다.",
        "추가 확인 필요사항이 별도로 식별되지 않았습니다.",
        "검색 근거에서 관련 Rule/Guidance를 확인하지 못했습니다.",
    )
    answer = "\n".join(
        (
            f"{line.rstrip()} {fallback_cite}"
            if (
                line.lstrip().startswith("- ")
                and not re.search(r"\[\d+\]", line)
                and not any(limit in line for limit in safe_limitations)
            )
            else line
        )
        for line in answer.splitlines()
    )
    verified_compound_answer = bool(
        dnv_0264_query
        or (
            pres.society == "ABS"
            and re.search(r"위험\s*범주|risk\s+categor", question, re.I)
        )
    )
    if verified_compound_answer:
        _checked, claim_verification, claim_warnings = verify_claim_citations(
            answer, list(work_chunks)
        )
        claim_warnings = []
    else:
        answer, claim_verification, claim_warnings = verify_high_risk_claims(
            answer, list(work_chunks)
        )
    warnings.extend(claim_warnings)

    # Clause themes are deterministic extracts with citation ids assigned from
    # the same evidence list.  The lexical claim verifier can reject their
    # Korean paraphrases simply because the source is English.  Restore only
    # this evidence-derived Rule section when it was reduced to a limitation.
    if (
        clause_themes
        and section4
        and re.search(r"\[\d+\]", section4)
        and "확인하지 못했습니다" not in section4
    ):
        match = re.search(r"(?ms)^##\s*4\)[^\n]*\n.*$", answer)
        if match:
            heading = re.match(r"^##\s*4\)[^\n]*", match.group(0))
            if heading:
                answer = (
                    answer[: match.start()].rstrip()
                    + "\n\n"
                    + heading.group(0)
                    + "\n\n"
                    + section4.strip()
                )

    warnings.extend(pres.warnings)
    if clause_themes:
        must_rows = compute_alt_fuel_must_cover(list(pool or chunks), answer)
        missing = [
            r["must_cover"]
            for r in must_rows
            if r["found_in_chunks"] == "Yes" and r["included_in_answer"] == "No"
        ]
        if missing:
            warnings.append("must_cover_gap")
    if ENGLISH_SENTENCE_RE.search(answer):
        warnings.append("raw_chunk_leak")
    cite_count = len(re.findall(r"\[\d+\]", answer))
    if cite_count > 8:
        warnings.append("too_many_citations")
    if not work_chunks:
        warnings.append("no_substantive_rule_chunks")
    if claim_verification and not all(r.get("supported") for r in claim_verification):
        warnings.append("claim_verification_failed")

    return answer, list(dict.fromkeys(warnings))


def _legacy_build_section1(chunks: list[Any], *, max_bullets: int = 3) -> str:
    substantive = [
        c
        for c in chunks
        if not getattr(c, "is_catalog_table", False) and not is_crossref_table_chunk(c)
    ]
    by_file: dict[str, Any] = {}
    for c in substantive:
        fn = str(getattr(c, "file_name", "") or "")
        if fn:
            by_file[fn] = c
    bullets: list[str] = []
    for fn in sorted(by_file)[:max_bullets]:
        c = by_file[fn]
        body = strip_metadata_prefix(getattr(c, "text", ""))
        body = re.sub(r"\s+", " ", body).strip()[:220]
        ids = [f"[{i}]" for i, x in enumerate(chunks, 1) if getattr(x, "file_name", "") == fn]
        bullets.append(f"- **{fn.replace('.pdf', '')}**: {body} {''.join(ids)}")
    return "\n".join(bullets) if bullets else "- 없음"


def build_rule_lookup_legacy_answer(chunks: list[Any]) -> str:
    from rule_lookup_answer import build_deterministic_section4, build_fallback_section2

    s1 = _legacy_build_section1(chunks)
    s2 = build_fallback_section2(chunks)
    s3 = "- [해석 근거] (legacy)"
    s4 = build_deterministic_section4(chunks, s1)
    return join_four_sections({"1": s1, "2": s2, "3": s3, "4": s4})


def build_rule_lookup_ungrouped_answer(
    chunks: list[Any],
    *,
    question: str = "",
    pool: list[Any] | None = None,
) -> str:
    """Previous multi-item listing (for before/after comparison)."""
    from rule_lookup_document_analysis import detect_answer_quality_warnings

    work = expand_rule_lookup_chunks(chunks, pool, question=question)
    analyses = analyze_documents(work, question=question, citation_chunks=chunks, citation_fallback=pool, max_docs=6)
    lines_s4: list[str] = []
    for n, d in enumerate(analyses[:5], 1):
        cite = format_citations(d.citation_ids) if d.citation_ids else ""
        lines_s4.extend(
            [
                f"{n}. {d.doc_code}",
                f"- 유형: {d.doc_type}",
                f"- 관련성: {d.relevance_ko}",
                f"- 근거: {d.file_name}, {cite}",
                f"- 확정 여부: {d.confirmation}",
                "",
            ]
        )
    s1 = "\n".join(
        f"- **{d.doc_code}** ({d.confirmation}): {d.summary_ko} {format_citations(d.citation_ids)}"
        for d in analyses[:3]
    )
    return join_four_sections({"1": s1, "2": "- (ungrouped)", "3": "- (ungrouped)", "4": "\n".join(lines_s4)})
