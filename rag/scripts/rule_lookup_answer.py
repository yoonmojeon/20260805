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
