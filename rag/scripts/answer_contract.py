"""One user-facing answer contract shared by every RAG answer mode.

The generators may use different retrieval and drafting strategies, but the UI
must always receive the same two artifacts:

1. a concise answer whose factual sentences end in numeric citations; and
2. an evidence table containing only the chunks cited by that answer.

This module never invents a citation.  It may propagate citations already
attached to a multi-sentence bullet to each sentence in that same bullet, but
uncited factual lines are removed instead of being assigned an arbitrary chunk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


CITATION_RE = re.compile(r"\[(\d+)\]")
LIST_PREFIX_RE = re.compile(r"^(\s*(?:[-*+]|\d+[.)])\s+)(.*)$")
# A period after a single capital letter belongs to an abbreviation such as the
# equipment name "S.W. Service Pump", not to the end of a sentence.  Splitting
# there tore the answer "CMS 통일명칭 → S.W. Service Pump" into two bullets and
# left the second half without its subject.
SENTENCE_RE = re.compile(r".+?(?:(?<![A-Z])[.!?](?:\s*\[\d+\])*(?=\s+|$)|$)", re.S)
PLACEHOLDER_RE = re.compile(
    r"검색 근거에서 직접 확인되는 내용이 없어 답변에서 제외|"
    r"답변에서 제외했습니다|^근거\s*:\s*$|^근거\s*:\s*없음\s*$",
    re.I,
)
WORD_CHAR_RE = re.compile(r"[0-9A-Za-z가-힣]")
NO_EVIDENCE_RE = re.compile(
    r"근거(?:가|를)?\s*(?:없|찾지 못)|검색 결과에서.*찾지 못|직접 관련되는 근거.*없",
    re.I,
)


NO_VERIFIED_CLAIM_NOTICE = (
    "> 검색된 문서에서 질문에 직접 답할 근거를 찾지 못했습니다. "
    "문서명·조항·항목명을 함께 넣어 다시 질문해 주세요."
)


def has_no_verified_claim(answer: str) -> bool:
    """True when the contract stripped every claim and only the notice remains.

    Several callers rescue such answers with an extractive summary.  They used
    to grep for the notice wording, which silently broke whenever the wording
    was reworded for users; ask this function instead.
    """
    return NO_VERIFIED_CLAIM_NOTICE.lstrip("> ") in str(answer or "")


@dataclass(frozen=True)
class AnswerContractResult:
    answer: str
    evidence_table: list[dict]
    warnings: list[str]
    cited_ids: list[int]
    valid: bool


def _ordered_citation_ids(text: str, *, limit: int) -> list[int]:
    ids: list[int] = []
    for match in CITATION_RE.finditer(text or ""):
        value = int(match.group(1))
        if 1 <= value <= limit and value not in ids:
            ids.append(value)
    return ids


def _format_citations(ids: list[int]) -> str:
    return "".join(f"[{value}]" for value in ids)


def _canonical_heading(line: str) -> str | None:
    # Content bullets can legitimately contain "Rule/Guidance", "결론", or
    # "확인 필요".  They are facts, not headings.  Classifying them as headings
    # silently deleted cited Rule identities during normalization.
    if LIST_PREFIX_RE.match(line):
        return None
    text = re.sub(r"^#+\s*", "", line.strip()).strip()
    text = re.sub(r"^\d+[.)]\s*", "", text).strip()
    # Table/rule generators emit "결론: <fact> [n]".  That is an answer line,
    # not a section title — treating it as a heading wiped the entire claim.
    fact_prefix = re.match(r"^(결론|확인\s*필요)\s*[:：]\s*(.+)$", text)
    if fact_prefix and len(fact_prefix.group(2).strip()) >= 2:
        return None
    text = text.rstrip(":：").strip()
    low = text.lower()
    if any(key in low for key in ("핵심 요약", "핵심 답변")) or low in {"결론", "핵심"}:
        return "## 1) 핵심 요약"
    if any(key in low for key in ("선박 운항/업무 영향", "선박 운항·업무 영향", "실무 영향")):
        return "## 2) 선박 운항/업무 영향"
    if any(
        key in low
        for key in ("추후 확인 필요사항", "추후 확인 필요", "후속 확인 필요", "추가 확인")
    ) or low in {"확인 필요", "확인필요"}:
        return "## 3) 추후 확인 필요사항"
    if "근거 조항" in low:
        return "## 핵심 조항"
    if "관련 선급" in low or "rule / guidance" in low or "rule/guidance" in low:
        return "## 4) 관련 선급 Rule / Guidance"
    if line.lstrip().startswith("#"):
        return f"## {text}" if text else None
    if line.strip().endswith((":", "：")) and len(text) <= 40:
        return f"## {text}" if text else None
    return None


def _clean_sentence(sentence: str, fallback_ids: list[int]) -> tuple[str, list[int]]:
    own_ids: list[int] = []
    for raw_id in CITATION_RE.findall(sentence):
        value = int(raw_id)
        if value not in own_ids:
            own_ids.append(value)
    cite_ids = own_ids or fallback_ids
    prose = CITATION_RE.sub("", sentence)
    prose = re.sub(r"\s+([,.!?])", r"\1", prose)
    prose = re.sub(r"\s{2,}", " ", prose).strip()
    return prose, cite_ids


def _merge_unbalanced_bold(parts: list[str]) -> list[str]:
    """Rejoin pieces that were split in the middle of a bold span.

    A list ordinal ends in a period, so "**1. 선체 거더**" split into "**1." and
    "선체 거더**" — two bullets, both with broken markup and the second missing
    its subject.  An odd number of "**" marks an unfinished span.
    """
    merged: list[str] = []
    for part in parts:
        if merged and merged[-1].count("**") % 2 == 1:
            merged[-1] = f"{merged[-1]} {part}".strip()
            continue
        merged.append(part)
    return merged


def _split_cited_line(content: str, *, valid_ids: list[int]) -> list[str]:
    if not valid_ids:
        return []
    parts = [part.strip() for part in SENTENCE_RE.findall(content) if part.strip()]
    parts = _merge_unbalanced_bold(parts)
    if not parts:
        parts = [content.strip()]
    out: list[str] = []
    for part in parts:
        prose, sentence_ids = _clean_sentence(part, valid_ids)
        if not prose or PLACEHOLDER_RE.search(prose):
            continue
        # Sentence splitting can leave a fragment that is only punctuation, which
        # used to surface as an empty "- , [1][2]" bullet.
        if not WORD_CHAR_RE.search(prose):
            continue
        usable = [value for value in sentence_ids if value in valid_ids]
        if not usable:
            continue
        out.append(f"{prose} {_format_citations(usable)}".strip())
    return out


def normalize_answer(answer: str, citation_chunks: list[Any]) -> tuple[str, list[str]]:
    """Normalize headings, split bullets by sentence, and block uncited claims."""
    limit = len(citation_chunks)
    warnings: list[str] = []
    normalized: list[str] = []
    saw_no_evidence = False

    for raw in (answer or "").splitlines():
        stripped = raw.strip()
        if not stripped:
            if normalized and normalized[-1] != "":
                normalized.append("")
            continue

        heading = _canonical_heading(stripped)
        if heading:
            if normalized and normalized[-1] != "":
                normalized.append("")
            normalized.append(heading)
            normalized.append("")
            continue

        list_match = LIST_PREFIX_RE.match(raw)
        content = list_match.group(2).strip() if list_match else stripped
        if PLACEHOLDER_RE.search(content) or not re.sub(r"[\[\]\d\s*`_]", "", content):
            warnings.append("placeholder_or_citation_only_line_removed")
            continue

        valid_ids = _ordered_citation_ids(content, limit=limit)
        cited_sentences = _split_cited_line(content, valid_ids=valid_ids)
        if cited_sentences:
            normalized.extend(f"- {sentence}" for sentence in cited_sentences)
            continue

        if NO_EVIDENCE_RE.search(content):
            saw_no_evidence = True
            continue

        # A factual sentence without a citation is not allowed through the
        # user-facing boundary.  Do not guess which retrieved chunk supports it.
        warnings.append("uncited_sentence_removed")

    # Remove headings that ended up with no cited content below them.
    compact: list[str] = []
    i = 0
    while i < len(normalized):
        line = normalized[i]
        if line.startswith("## "):
            j = i + 1
            while j < len(normalized) and not normalized[j].startswith("## "):
                if normalized[j].startswith("- "):
                    break
                j += 1
            if j >= len(normalized) or not normalized[j].startswith("- "):
                warnings.append("empty_section_removed")
                i += 1
                while i < len(normalized) and not normalized[i].startswith("## "):
                    i += 1
                continue
        compact.append(line)
        i += 1

    while compact and not compact[-1].strip():
        compact.pop()
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(compact)).strip()
    if text and not text.startswith("## 1) 핵심 요약"):
        text = f"## 1) 핵심 요약\n\n{text}"
    if not text:
        # Whether the generator said "no evidence" or produced only uncited
        # prose, the user-facing outcome is the same: nothing was verifiable.
        # The distinction stays in ``warnings`` for diagnosis.
        _ = saw_no_evidence
        text = f"## 1) 핵심 요약\n\n{NO_VERIFIED_CLAIM_NOTICE}"
    return text, list(dict.fromkeys(warnings))


def build_cited_evidence_table(answer: str, citation_chunks: list[Any]) -> list[dict]:
    """Build the compact UI table in first-citation order."""
    rows: list[dict] = []
    for citation_id in _ordered_citation_ids(answer, limit=len(citation_chunks)):
        chunk = citation_chunks[citation_id - 1]
        raw_text = str(getattr(chunk, "text", "") or "")
        evidence = re.sub(r"\s+", " ", raw_text).strip()
        if len(evidence) > 1600:
            evidence = evidence[:1599].rstrip() + "…"
        rows.append(
            {
                "citation_id": f"[{citation_id}]",
                "file_name": str(
                    getattr(chunk, "file_name", "")
                    or getattr(chunk, "doc_id", "")
                    or "(문서명 없음)"
                ),
                "page": getattr(chunk, "page_number", None),
                "chunk_id": str(getattr(chunk, "chunk_id", "") or ""),
                "chunk_preview": evidence,
                # Hidden full-enough source for the Advanced local auditor.
                # The visible UI preview remains compact.
                "review_text": raw_text[:4200],
            }
        )
    return rows


ALLOWED_SUB_HEADINGS = ("## 핵심 조항",)


def _ensure_required_sections(answer: str) -> str:
    """Keep the four-section UI contract and never expose an empty heading.

    Structured generators and the UI both assume the same four sections.  If
    claim verification removes a section body, replace it with a transparent
    limitation instead of dropping the section or preserving a blank shell.

    An LLM asked for one summary section still invents its own headings such as
    "## 종 방향 구조 (REG03)".  Those are dropped rather than kept as body text,
    because a heading inside section 1 makes the rendered answer look like it
    has eight top-level sections instead of four.
    """
    defaults = {
        "## 1) 핵심 요약": "> 검색 근거에서 질문에 직접 답할 내용을 확인하지 못했습니다.",
        "## 2) 선박 운항/업무 영향": "> 검색 근거에서 직접 확인되는 별도 운항·업무 영향이 없습니다.",
        "## 3) 추후 확인 필요사항": "> 추가 확인 필요사항이 별도로 식별되지 않았습니다.",
        "## 4) 관련 선급 Rule / Guidance": "> 관련 선급 Rule / Guidance가 검색 근거에 없거나 해당하지 않습니다.",
    }
    sections: dict[str, list[str]] = {}
    current = ""
    for line in (answer or "").splitlines():
        if line in defaults:
            current = line
            sections.setdefault(current, [])
        elif line.startswith("#") and not line.startswith(ALLOWED_SUB_HEADINGS):
            continue
        elif current:
            sections[current].append(line)
    out: list[str] = []
    for heading in (
        "## 1) 핵심 요약",
        "## 2) 선박 운항/업무 영향",
        "## 3) 추후 확인 필요사항",
        "## 4) 관련 선급 Rule / Guidance",
    ):
        body = "\n".join(sections.get(heading, [])).strip()
        out.extend([heading, "", body or defaults[heading], ""])
    return "\n".join(out).strip()


def apply_answer_contract(answer: str, citation_chunks: list[Any]) -> AnswerContractResult:
    normalized, warnings = normalize_answer(answer, citation_chunks)
    normalized = _ensure_required_sections(normalized)
    cited_ids = _ordered_citation_ids(normalized, limit=len(citation_chunks))
    evidence = build_cited_evidence_table(normalized, citation_chunks)
    factual_lines = [line for line in normalized.splitlines() if line.strip().startswith("- ")]
    valid = all(CITATION_RE.search(line) for line in factual_lines)
    if factual_lines and len(evidence) != len(set(cited_ids)):
        valid = False
        warnings.append("evidence_table_mismatch")
    return AnswerContractResult(
        answer=normalized,
        evidence_table=evidence,
        warnings=list(dict.fromkeys(warnings)),
        cited_ids=cited_ids,
        valid=valid,
    )
