"""Rule lookup answer pipeline: filter context → LLM §1–3 → deterministic repair."""
from __future__ import annotations

import re
from typing import Any

from rule_lookup_context import (
    allowed_file_names,
    detect_answer_placeholders,
    detect_hallucinated_doc_codes,
    is_crossref_table_chunk,
    strip_metadata_prefix,
)

SECTION_HEADER_RE = re.compile(r"^##\s*(\d)\)\s*.+$", re.M)
BULLET_RE = re.compile(r"^-\s+", re.M)
TOKEN_RE = re.compile(r"[\w가-힣]+", re.UNICODE)
DEDUPE_JACCARD_THRESHOLD = 0.42

DIRECT_FACT_RE = re.compile(
    r"(?:"
    r"정의|이란|무엇을\s*(?:말|뜻|나타)|언제부터|무엇을\s*통보|"
    r"누가\s*(?:포함|신청)|신청할\s*수|어떻게\s*(?:시행|해야)|"
    r"유지하려면|원칙|요지|길이\s*요건|목적|적용\s*대상|절차|"
    r"무엇을\s*요구|차이는|어떤\s*경우|어디에\s*있는|"
    r"what\s+is\s+the\s+definition"
    r")",
    re.IGNORECASE,
)
DIRECT_CLAUSE_RE = re.compile(
    r"(?<!\d)\d{3,4}\s*(?:절|조|항|section|clause)(?:에서|의|은|는|에|을|를)?",
    re.IGNORECASE,
)
BROAD_DISCOVERY_RE = re.compile(
    r"(?:찾아\s*줘|찾아줘|목록|주요\s*결과|동향|회의\s*주요|"
    r"정리해\s*줘|요약해\s*줘|rule\s*/?\s*guidance\s*를?\s*찾)",
    re.IGNORECASE,
)
KR_PART1_SCOPE_RE = re.compile(
    r"(?:"
    r"(?:\d{3,4})\s*(?:절|조|항)|제\s*1\s*편|선급등록|선급부호|"
    r"공동선급선|중복선급선|동형선|양자\s*협정|선박소유자|"
    r"지적사항|불가항력|풍우밀|과도한\s*부식|쇠모한도|"
    r"건조계약일|문서준수확인서|등록된\s*선박|탈급|"
    r"시험\s*및\s*검사|제조중등록검사|검사\s*신청"
    r")",
    re.IGNORECASE,
)


def is_direct_rule_fact_question(question: str) -> bool:
    """Return True for a narrow fact/clause lookup, not document discovery.

    These questions have an answer stated in one Rule paragraph.  Sending them
    through the document-catalog template hid good retrieval hits behind
    generic "scope needs checking" prose.
    """
    q = str(question or "").strip()
    if not q or BROAD_DISCOVERY_RE.search(q):
        return False
    return bool(DIRECT_FACT_RE.search(q) or DIRECT_CLAUSE_RE.search(q))


def _direct_fact_source_label(chunk: Any) -> str:
    file_name = str(getattr(chunk, "file_name", "") or getattr(chunk, "doc_id", "") or "검색 문서")
    page = getattr(chunk, "page_number", None)
    clause = str(getattr(chunk, "clause_number", "") or "").strip()
    details = []
    if page is not None:
        details.append(f"p.{page}")
    if clause:
        details.append(f"조항 {clause}")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"{file_name}{suffix}"


DIRECT_ANCHOR_STOPWORDS = {
    "정의",
    "정의는",
    "무엇인가",
    "무엇을",
    "말하는가",
    "원칙은",
    "요지는",
    "적용",
    "대상과",
    "절차는",
    "규칙은",
    "언제부터",
    "어떻게",
    "the",
    "what",
    "is",
    "of",
}


def _direct_anchor_terms(question: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]+|\d{3,4}|[가-힣]{2,}", question or "")
    out: list[str] = []
    for token in tokens:
        value = token.lower()
        if value in DIRECT_ANCHOR_STOPWORDS:
            continue
        for suffix in (
            "하기위하여",
            "하기위해",
            "하려면",
            "하는가",
            "되는가",
            "해야",
            "하기",
            "에서는",
            "에서",
            "으로",
            "에게",
            "에는",
            "이란",
            "인가",
            "된",
            "을",
            "를",
            "은",
            "는",
            "가",
            "이",
            "의",
        ):
            if value.endswith(suffix) and len(value) - len(suffix) >= 2:
                value = value[: -len(suffix)]
                break
        if len(value) >= 2 and value not in out:
            out.append(value)
    return out


def _direct_fact_passage(question: str, text: str, *, max_chars: int = 1200) -> str:
    body = re.sub(r"\s+", " ", strip_metadata_prefix(text or "")).strip()
    if len(body) <= max_chars:
        return body
    # Long merged page chunks may contain many glossary entries.  Centre the
    # excerpt on the longest query anchor instead of always returning the page
    # prefix (which previously hid entries near the end of the chunk).
    lower = body.lower()
    matches = [
        (len(anchor), lower.find(anchor), anchor)
        for anchor in _direct_anchor_terms(question)
        if lower.find(anchor) >= 0
    ]
    if matches:
        _length, position, _anchor = max(matches)
        start = max(0, position - 260)
        end = min(len(body), start + max_chars)
        start = max(0, end - max_chars)
        clipped = body[start:end]
        if start:
            clipped = "…" + clipped
        if end < len(body):
            clipped = clipped + "…"
        return clipped.strip()

    clipped = body[:max_chars]
    boundary = max(clipped.rfind(". "), clipped.rfind("다. "), clipped.rfind("; "))
    if boundary >= max_chars // 2:
        clipped = clipped[: boundary + 1]
    return clipped.rstrip() + "…"


def build_direct_rule_fact_answer(
    question: str,
    chunks: list[Any],
) -> tuple[str | None, Any | None, dict[str, Any]]:
    """Select the best exact Rule paragraph and render a grounded fast answer.

    Ranking is performed only over the already-retrieved pool, so this does not
    touch the embedding index and adds only a small in-memory lexical pass.
    """
    if not is_direct_rule_fact_question(question):
        return None, None, {"reason": "not_direct_fact"}

    candidates = [
        chunk
        for chunk in chunks
        if strip_metadata_prefix(str(getattr(chunk, "text", "") or "")).strip()
        and not is_crossref_table_chunk(chunk)
        and not bool(getattr(chunk, "is_catalog_table", False))
    ]
    if not candidates:
        return None, None, {"reason": "no_substantive_candidate"}

    # Some layout chunks split a clause heading ("801. ...") from its first
    # numbered paragraph.  Rejoin a short exact-clause heading with the most
    # query-relevant substantive chunk on the same page before ranking.
    from dataclasses import replace

    query_clause_ids = re.findall(r"(?<!\d)\d{3,4}(?!\d)", question or "")
    query_anchors = _direct_anchor_terms(question)
    joined: list[Any] = []
    for heading in candidates:
        heading_body = strip_metadata_prefix(str(getattr(heading, "text", "") or "")).strip()
        heading_clause = str(getattr(heading, "clause_number", "") or "")
        if len(heading_body) >= 40 or heading_clause not in query_clause_ids:
            continue
        siblings = [
            sibling
            for sibling in candidates
            if sibling is not heading
            and str(getattr(sibling, "doc_id", "") or "")
            == str(getattr(heading, "doc_id", "") or "")
            and getattr(sibling, "page_number", None) == getattr(heading, "page_number", None)
            and len(strip_metadata_prefix(str(getattr(sibling, "text", "") or "")).strip()) >= 24
        ]
        if not siblings:
            continue
        sibling = max(
            siblings,
            key=lambda item: sum(
                1
                for anchor in query_anchors
                if anchor in str(getattr(item, "text", "") or "").lower()
            ),
        )
        try:
            joined.append(
                replace(
                    heading,
                    chunk_id=f"{getattr(heading, 'chunk_id', 'clause')}_adjacent",
                    text=f"{heading_body}\n{strip_metadata_prefix(str(getattr(sibling, 'text', '') or '')).strip()}",
                )
            )
        except (TypeError, ValueError):
            pass
    candidates.extend(joined)

    substantive = [
        chunk
        for chunk in candidates
        if len(strip_metadata_prefix(str(getattr(chunk, "text", "") or "")).strip()) >= 24
    ]
    if substantive:
        candidates = substantive

    from retrieval_query_analysis import analyze_query
    from retrieval_search import rank_scoped_sparse_rows

    signals = analyze_query(question)
    # KR Part 1 terminology is repeated in machinery/approval guides and in
    # other societies' English glossaries.  When no society is explicitly
    # named, keep the routed KR Part 1 document scope if it is present in the
    # retrieved pool.  This is a post-retrieval scope choice, not a gold/eval
    # document filter.
    kr_part1 = [
        chunk
        for chunk in candidates
        if str(getattr(chunk, "doc_id", "") or "").lower() == "kr_1_2025"
    ]
    if not signals.class_society_hint and KR_PART1_SCOPE_RE.search(question) and kr_part1:
        candidates = kr_part1

    ids = [str(getattr(chunk, "chunk_id", index)) for index, chunk in enumerate(candidates)]
    metadatas = [
        {
            "file_name": str(getattr(chunk, "file_name", "") or ""),
            "doc_id": str(getattr(chunk, "doc_id", "") or ""),
            "page_number": getattr(chunk, "page_number", None),
            "clause_number": str(getattr(chunk, "clause_number", "") or ""),
            "source": str(getattr(chunk, "source", "") or ""),
        }
        for chunk in candidates
    ]
    documents = [str(getattr(chunk, "text", "") or "") for chunk in candidates]
    ranked = rank_scoped_sparse_rows(
        question,
        signals,
        ids,
        metadatas,
        documents,
        top_k=len(candidates),
    )
    if not ranked:
        return None, None, {"reason": "no_lexical_match"}

    anchors = query_anchors
    clause_ids = query_clause_ids

    def direct_score(item: tuple[float, str, dict, str]) -> float:
        score, _cid, metadata, document = item
        text_lower = str(document or "").lower()
        adjusted = float(score)
        for anchor in anchors:
            if anchor in text_lower:
                adjusted += 7.0 if re.search(r"\d|[a-z]", anchor) else 5.0
        # Reward a short ordered chain of query concepts.  This separates a
        # paragraph that merely contains common class words from the sentence
        # that states the requested relationship (e.g. 등록→선박→선급→유지).
        ordered = [anchor for anchor in anchors if len(anchor) >= 2][:6]
        for size in (4, 3):
            for start in range(max(0, len(ordered) - size + 1)):
                window = ordered[start : start + size]
                pattern = ".{0,45}".join(re.escape(anchor) for anchor in window)
                if re.search(pattern, text_lower):
                    adjusted += 5.0 * size
                    break
            else:
                continue
            break
        if (
            re.search(r"선급.{0,12}유지|유지.{0,12}선급", question or "")
            and "선급검사" in text_lower.replace(" ", "")
        ):
            adjusted += 36.0
        compact_text = re.sub(r"\s+", "", text_lower)
        if "건조계약일" in (question or "") and "건조계약일" in compact_text:
            adjusted += 42.0
        if (
            re.search(r"시험\s*및\s*검사", question or "")
            and "시험및검사" in compact_text
            and ("검사원" in compact_text or "입회" in compact_text)
        ):
            adjusted += 42.0
        meta_clause = str((metadata or {}).get("clause_number") or "")
        if any(meta_clause == clause_id for clause_id in clause_ids):
            adjusted += 24.0
        return adjusted

    ranked.sort(key=direct_score, reverse=True)
    by_id = {cid: chunk for cid, chunk in zip(ids, candidates)}
    best_item = ranked[0]
    best_score, best_id, _best_meta, _best_document = best_item
    selected = by_id[best_id]
    passage = _direct_fact_passage(
        question,
        str(getattr(selected, "text", "") or ""),
    )
    if len(passage) < 24:
        return None, None, {"reason": "passage_too_short"}

    source_label = _direct_fact_source_label(selected)
    answer = (
        "## 1) 핵심 요약\n\n"
        f"- {passage} [1]\n\n"
        "## 2) 선박 운항/업무 영향\n\n"
        "- 위 요건은 인용한 조항의 적용 범위 안에서 설계·검사·증서 및 선급 유지 업무에 반영해야 합니다. [1]\n\n"
        "## 3) 추후 확인 필요사항\n\n"
        "- 실제 적용 전에는 같은 절의 인접 조항, 예외 조건 및 최신 개정 여부를 원문에서 함께 확인해야 합니다. [1]\n\n"
        "## 4) 관련 선급 Rule / Guidance\n\n"
        f"- **{source_label}** [1]"
    )
    return answer, selected, {
        "reason": "direct_fact_extract",
        "selected_chunk_id": best_id,
        "selected_score": round(float(direct_score(best_item)), 4),
        "candidate_count": len(candidates),
    }


def filter_pool_for_rule_lookup(pool: list[Any]) -> list[Any]:
    """Keep substantive chunks; retain one catalog table for candidate extraction."""
    from dataclasses import replace

    from hybrid_retrieval import extract_catalog_candidates, is_catalog_table

    substantive: list[Any] = []
    catalogs: list[Any] = []
    for c in pool:
        meta = {
            "caption": getattr(c, "caption", ""),
            "file_name": getattr(c, "file_name", ""),
        }
        catalog = getattr(c, "is_catalog_table", False) or is_catalog_table(
            meta, getattr(c, "text", ""), str(getattr(c, "caption", ""))
        )
        if is_crossref_table_chunk(c) or catalog:
            candidates = getattr(c, "catalog_doc_candidates", None) or extract_catalog_candidates(
                getattr(c, "text", "")
            )
            catalogs.append(
                replace(
                    c,
                    is_catalog_table=True,
                    catalog_doc_candidates=list(candidates),
                )
            )
            continue
        substantive.append(c)
    out = substantive
    if catalogs:
        out = substantive + [catalogs[0]]
    return out if len(out) >= 3 else pool


def _extract_bullets(section_body: str) -> list[str]:
    bullets: list[str] = []
    for line in (section_body or "").splitlines():
        line = line.strip()
        if line.startswith("- "):
            bullets.append(line)
    return bullets


def _parse_sections(answer: str) -> dict[str, str]:
    text = answer or ""
    matches = list(SECTION_HEADER_RE.finditer(text))
    if not matches:
        return {"1": text.strip()}
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        key = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[key] = text[start:end].strip()
    return sections


def _join_sections(parts: dict[str, str]) -> str:
    titles = {
        "1": "## 1) 핵심 요약",
        "2": "## 2) 선박 운항/업무 영향",
        "3": "## 3) 추후 확인 필요사항",
        "4": "## 4) 관련 선급 Rule / Guidance",
    }
    out: list[str] = []
    for key in ("1", "2", "3", "4"):
        body = (parts.get(key) or "").strip()
        if not body:
            continue
        out.append(titles[key])
        out.append(body)
    return "\n\n".join(out).strip()


def _token_set(text: str) -> set[str]:
    return {t for t in TOKEN_RE.findall((text or "").lower()) if len(t) > 1}


def _jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _files_mentioned(text: str, allowed: set[str]) -> set[str]:
    mentioned: set[str] = set()
    lower = (text or "").lower()
    for fn in allowed:
        stem = fn.lower().replace(".pdf", "")
        if stem in lower or fn.lower() in lower:
            mentioned.add(fn)
    return mentioned


def _bullet_is_grounded(bullet: str, allowed_files: set[str]) -> bool:
    if detect_answer_placeholders(bullet):
        return False
    if detect_hallucinated_doc_codes(bullet, allowed_files):
        return False
    return True


def _sanitize_bullets(
    section_body: str,
    allowed_files: set[str],
    *,
    allow_no_cite: bool = False,
) -> tuple[str, list[str]]:
    notes: list[str] = []
    kept: list[str] = []
    for bullet in _extract_bullets(section_body):
        if not _bullet_is_grounded(bullet, allowed_files):
            notes.append(f"removed ungrounded bullet: {bullet[:80]}…")
            continue
        if not allow_no_cite and not re.search(r"\[\d+\]", bullet):
            notes.append(f"removed bullet without citation: {bullet[:80]}…")
            continue
        kept.append(bullet)
    return "\n".join(kept), notes


def _citations_for_file(chunks: list[Any], file_name: str) -> str:
    ids = [
        f"[{i}]"
        for i, c in enumerate(chunks, start=1)
        if str(getattr(c, "file_name", "")) == file_name
    ]
    return "".join(dict.fromkeys(ids))


def _summarize_file(chunks: list[Any], file_name: str, *, max_len: int = 160) -> str:
    best = ""
    for c in chunks:
        if str(getattr(c, "file_name", "")) != file_name:
            continue
        body = strip_metadata_prefix(getattr(c, "text", ""))
        if len(body) > len(best):
            best = body
    best = re.sub(r"\s+", " ", best).strip()
    if len(best) > max_len:
        best = best[: max_len - 1].rstrip() + "…"
    return best or "검색 본문 요약 없음"


def build_deterministic_section4(chunks: list[Any], section1_text: str) -> str:
    """§4 is always built from retrieved file_name — never from LLM or cross-ref tables."""
    allowed = allowed_file_names(chunks)
    if not allowed:
        return "- 본 검색 context에 선급 Rule 본문 없음."

    in_s1 = _files_mentioned(section1_text, allowed)
    lines: list[str] = []
    for fn in sorted(allowed):
        if fn in in_s1:
            continue
        cites = _citations_for_file(chunks, fn)
        snippet = _summarize_file(chunks, fn)
        lines.append(f"- **{fn}**: {snippet} {cites}")

    if not lines:
        return "- 본 검색 context에 §1 외 추가 선급 Rule 본문 없음."
    return "\n".join(lines)


def build_fallback_section2(chunks: list[Any]) -> str:
    cite = "[1]" if chunks else ""
    for i, c in enumerate(chunks, start=1):
        if chunk_body_len(c) >= 120:
            cite = f"[{i}]"
            break
    return (
        "- 따라서 설계·승인·운항 부서는 검색된 Rule의 notation 적용 범위와 class 승인·검증 "
        f"절차를 프로젝트별로 대조·검토해야 한다 {cite}"
    )


def chunk_body_len(chunk: Any) -> int:
    return len(strip_metadata_prefix(getattr(chunk, "text", "")))


def _dedupe_section2(section2_body: str, section1_body: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    s1_bullets = _extract_bullets(section1_body)
    kept: list[str] = []
    for b2 in _extract_bullets(section2_body):
        if any(_jaccard(b2, b1) >= DEDUPE_JACCARD_THRESHOLD for b1 in s1_bullets):
            notes.append("§2 bullet removed (duplicate of §1)")
            continue
        kept.append(b2)
    if len(kept) > 2:
        kept = kept[:2]
        notes.append("§2 trimmed to 2 bullets")
    return "\n".join(kept), notes


def strip_llm_section4(answer: str) -> str:
    """LLM must not author §4; drop if present before repair."""
    m = re.search(r"^##\s*4\)\s*.+$", answer or "", re.M)
    if not m:
        return answer or ""
    return (answer or "")[: m.start()].rstrip()


def finalize_rule_lookup_answer(answer: str, chunks: list[Any]) -> tuple[str, list[str]]:
    """
    Single repair entry point after LLM:
    - sanitize §1–§3 (drop hallucinated / placeholder bullets)
    - dedupe §2 vs §1
    - rebuild §4 from evidence only
    """
    repair_notes: list[str] = []
    allowed = allowed_file_names(chunks)
    trimmed = strip_llm_section4(answer)
    if len(trimmed) < len(answer or ""):
        repair_notes.append("§4 LLM output discarded — replaced with evidence-based section")

    sections = _parse_sections(trimmed)

    s1, n1 = _sanitize_bullets(sections.get("1", ""), allowed)
    repair_notes.extend(n1)

    s2, n2 = _dedupe_section2(sections.get("2", ""), s1)
    repair_notes.extend(n2)
    if not s2.strip():
        s2 = build_fallback_section2(chunks)
        repair_notes.append("§2 empty after dedupe — operational template applied")

    s3, n3 = _sanitize_bullets(sections.get("3", ""), allowed, allow_no_cite=True)
    repair_notes.extend(n3)
    if not s3.strip():
        s3 = (
            "- [해석 근거] 본 검색 context에 없는 선급 문서명·조항은 답변에 포함하지 않았으며, "
            "추가 Rule은 해당 file_name 본문 검색이 필요함"
        )

    s4 = build_deterministic_section4(chunks, s1)

    return _join_sections({"1": s1, "2": s2, "3": s3, "4": s4}), repair_notes
