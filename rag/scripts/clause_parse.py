from __future__ import annotations

import re

# 절 번호: "902. 탈급", "1201. 정부규정"
ARTICLE_CLAUSE_RE = re.compile(r"^\s*(\d{3,})\.\s*\S")

# 항·호: "14. 과도한", "3-1. 유조선", "2-3. 자가"
SUBCLAUSE_CLAUSE_RE = re.compile(r"^\s*(\d+(?:-\d+)?)\.\s*\S")

# 단순 십진: "1. 선급등록" (절 번호와 구분: 3자리 미만 또는 비-절 맥락)
DECIMAL_CLAUSE_RE = re.compile(r"^\s*(\d{1,2})\.\s*\S")

MULTI_SUBCLAUSE_SPLIT_RE = re.compile(
    r"(?=\n\s*\d+(?:-\d+)?\.\s*\S)",
    re.MULTILINE,
)


def first_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def article_number_from_text(text: str) -> str | None:
    match = ARTICLE_CLAUSE_RE.match(first_line(text))
    return match.group(1) if match else None


def subclause_number_from_text(text: str) -> str | None:
    match = SUBCLAUSE_CLAUSE_RE.match(first_line(text))
    return match.group(1) if match else None


def decimal_clause_number_from_text(text: str) -> str | None:
    match = DECIMAL_CLAUSE_RE.match(first_line(text))
    return match.group(1) if match else None


def clause_number_from_text(text: str) -> str | None:
    """Prefer article (902) > subclause (3-1) > decimal item (14)."""
    article = article_number_from_text(text)
    if article:
        return article
    sub = subclause_number_from_text(text)
    if sub:
        return sub
    return decimal_clause_number_from_text(text)


def is_article_clause_number(clause_no: str | None) -> bool:
    if not clause_no or not clause_no.isdigit():
        return False
    return len(clause_no) >= 3


def is_subsection_clause_number(clause_no: str | None) -> bool:
    if not clause_no:
        return False
    if "-" in clause_no:
        return True
    return clause_no.isdigit() and len(clause_no) <= 2


def should_split_merge_blocks(
    current_text: str,
    current_clause: str | None,
    candidate_text: str,
    candidate_clause: str | None,
) -> bool:
    """Whether to end the current merge block before appending candidate."""
    current_article = article_number_from_text(current_text) or (
        current_clause if is_article_clause_number(current_clause) else None
    )
    candidate_article = article_number_from_text(candidate_text)

    if candidate_article and current_article and candidate_article != current_article:
        return True

    if candidate_article and current_article is None and current_clause:
        return True

    if not candidate_clause or not current_clause:
        if candidate_article and current_article == candidate_article:
            return False
        return bool(candidate_article and current_article)

    if candidate_clause == current_clause:
        return False

    if current_article and is_subsection_clause_number(candidate_clause):
        return False

    if is_article_clause_number(current_clause) and is_subsection_clause_number(candidate_clause):
        return False

    return True


def split_text_by_subclauses(text: str) -> list[tuple[str, str | None]]:
    """Split a merged block into (text, clause_number) segments at 3-1. / 2-3. boundaries."""
    normalized = text.strip()
    if not normalized:
        return []

    parts = MULTI_SUBCLAUSE_SPLIT_RE.split(normalized)
    segments: list[tuple[str, str | None]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        segments.append((part, clause_number_from_text(part)))

    if len(segments) <= 1:
        return [(normalized, clause_number_from_text(normalized))]

    first_article = article_number_from_text(segments[0][0])
    if first_article and len(segments) > 1:
        second_clause = segments[1][1]
        if second_clause and is_subsection_clause_number(second_clause):
            merged_text = "\n".join(seg for seg, _ in segments)
            return [(merged_text, first_article)]

    return segments
