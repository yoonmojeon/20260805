"""Question-driven presentation formats for already grounded answers.

Generation and citation validation continue to use the established four
section contract.  This module only removes empty template padding after that
contract has passed, so it cannot create or rewrite factual claims.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerFormatDecision:
    kind: str
    reason: str


PREMISE_QUESTION_RE = re.compile(
    r"전제.{0,30}(?:맞는지|검증)|틀리면.{0,40}바로잡",
    re.I,
)
EXPLICIT_VERDICT_RE = re.compile(
    r"전제.{0,30}(?:맞습니다|맞지\s*않|틀렸|틀립|옳지\s*않)|"
    r"검색\s*근거만으로는\s*전제를\s*확인할\s*수\s*없",
    re.I,
)
NEGATIVE_EVIDENCE_RE = re.compile(
    r"포함(?:하고)?\s*있지\s*않|확인되지\s*않|"
    r"채택(?:되|하)지\s*않|발효(?:되|하)지\s*않|"
    r"확정(?:되|하)지\s*않|폐지(?:되|하)지\s*않",
    re.I,
)


def choose_answer_format(question: str, row: dict | None = None) -> AnswerFormatDecision:
    row = row or {}
    question = str(question or "")
    if row.get("_table_qa") or row.get("question_type") or row.get("gold_table_id"):
        return AnswerFormatDecision("table_answer", "table route owns its presentation")
    if row.get("_specific_lookup_verification"):
        return AnswerFormatDecision("short_factual", "verified exact lookup or conservative rejection")

    from rag_query_router import is_rule_guidance_lookup
    from rule_guidance_accurate import is_exact_rule_fact_question

    if PREMISE_QUESTION_RE.search(question):
        return AnswerFormatDecision(
            "short_factual", "explicit premise verification"
        )
    if str((row.get("_question_profile") or {}).get("answer_style") or "") == "document_cards":
        return AnswerFormatDecision(
            "document_guide", "document-oriented Rule/Guidance guide"
        )
    if is_rule_guidance_lookup(
        question,
        row,
        category=str(row.get("category") or ""),
    ) and not is_exact_rule_fact_question(question):
        return AnswerFormatDecision("rule_guide", "rule/guidance discovery intent")
    if is_exact_rule_fact_question(question) or row.get("_answer_profile") == "exact_rule_fact":
        return AnswerFormatDecision("short_factual", "single clause/value/condition lookup")
    if re.search(
        r"(?:MEPC|MSC)\s*\d+.{0,80}(?:일정|timeline|mandatory|발효|채택|흐름)|"
        r"(?:회의|회차).{0,50}(?:일정|결정|결론|채택|발효)|"
        r"mandatory\s+(?:MASS\s+)?code",
        question,
        re.I,
    ):
        return AnswerFormatDecision("meeting_timeline", "meeting decision/timeline intent")
    return AnswerFormatDecision("regulation_summary", "broad regulatory/operational summary")


def _parse_sections(answer: str) -> dict[str, list[str]]:
    sections = {str(i): [] for i in range(1, 5)}
    current = ""
    for raw in str(answer or "").splitlines():
        match = re.match(r"^##\s*([1-4])\)", raw.strip())
        if match:
            current = match.group(1)
            continue
        line = raw.strip()
        if current and line.startswith(("-", "*")):
            sections[current].append("- " + line.lstrip("-* ").strip())
    return sections


def _claim_signature(line: str) -> str:
    return re.sub(
        r"[^0-9A-Za-z가-힣]+",
        "",
        re.sub(r"\[\d+\]", "", line).lower(),
    )


def _dedupe(lines: list[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if "검색 근거에서 확인되지 않" in line or "별도로 식별되지 않" in line:
            continue
        signature = _claim_signature(line)
        if not signature or signature in seen:
            continue
        seen.add(signature)
        # Small local models occasionally close a bold span without opening
        # it ("title**:").  Repair only Markdown punctuation, never prose.
        if line.count("**") % 2 == 1 and line.startswith("- "):
            line = "- **" + line[2:]
        out.append(line)
        if len(out) >= limit:
            break
    return out


def render_answer_format(answer: str, decision: AnswerFormatDecision) -> str:
    """Shape a validated answer without changing any factual sentence."""
    if decision.kind in {"regulation_summary", "meeting_timeline", "table_answer"}:
        return answer
    sections = _parse_sections(answer)
    if decision.kind == "short_factual":
        claims = _dedupe(sections["1"], 3)
        if not claims:
            return answer
        return "## 답변\n\n" + "\n".join(claims)
    if decision.kind == "document_guide":
        groups = (
            ("### 문서 안내", _dedupe(sections["1"], 50)),
            ("### 활용 시점 / 실무 적용", _dedupe(sections["2"], 50)),
            ("### 확인 사항", _dedupe(sections["3"], 50)),
            ("### 관련 참조 문서", _dedupe(sections["4"], 50)),
        )
        rendered = ["## 관련 Rule / Guidance"]
        for heading, claims in groups:
            if not claims:
                continue
            rendered.extend(["", heading, "", *claims])
        return "\n".join(rendered) if len(rendered) > 1 else answer
    if decision.kind == "rule_guide":
        claims = _dedupe([*sections["1"], *sections["4"]], 3)
        if not claims:
            return answer
        return "## 관련 Rule / Guidance\n\n" + "\n".join(claims)
    return answer


def ensure_explicit_premise_verdict(
    question: str, answer: str, verdict_hint: str = ""
) -> str:
    """Canonicalize an already cited negative finding into an explicit verdict."""
    if not PREMISE_QUESTION_RE.search(question or "") or EXPLICIT_VERDICT_RE.search(answer or ""):
        return answer
    lines = str(answer or "").splitlines()
    hinted = re.search(
        r"전제는\s*(맞습니다|맞지\s*않습니다)|"
        r"검색\s*근거만으로는\s*전제를\s*확인할\s*수\s*없습니다",
        str(verdict_hint or ""),
        re.I,
    )
    if hinted:
        if hinted.group(1):
            verdict = f"전제는 {hinted.group(1)}."
        else:
            verdict = "검색 근거만으로는 전제를 확인할 수 없습니다."
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("-", "*")) and re.search(r"\[\d+\]", stripped):
                prefix = line[: len(line) - len(line.lstrip())]
                body = stripped.lstrip("-* ").strip()
                lines[index] = f"{prefix}- {verdict} {body}"
                return "\n".join(lines)
    if NEGATIVE_EVIDENCE_RE.search(str(verdict_hint or "")):
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("-", "*")) and re.search(r"\[\d+\]", stripped):
                prefix = line[: len(line) - len(line.lstrip())]
                body = stripped.lstrip("-* ").strip()
                lines[index] = f"{prefix}- 전제는 맞지 않습니다. {body}"
                return "\n".join(lines)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped.startswith(("-", "*"))
            and re.search(r"\[\d+\]", stripped)
            and NEGATIVE_EVIDENCE_RE.search(stripped)
        ):
            prefix = line[: len(line) - len(line.lstrip())]
            body = stripped.lstrip("-* ").strip()
            lines[index] = f"{prefix}- 전제는 맞지 않습니다. {body}"
            return "\n".join(lines)
    return answer
