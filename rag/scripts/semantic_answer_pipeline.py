"""Question-independent semantic answer planning and grounded post-processing.

The module deliberately does not know evaluation question ids, document names,
or page numbers.  It converts a free-form question into a reusable plan, turns
retrieved chunks into typed evidence units, and filters/ranks a drafted answer
against those units.  It can also add narrowly allowed operational implications
when a passage directly contains a deadline, reporting, verification, design,
or approval requirement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from grounded_answer_policy import classify_document_status


CITATION_RE = re.compile(r"\[(\d+)\]")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{1,}|[가-힣]{2,}")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
ACTION_RE = re.compile(
    r"\b(?:the\s+committee\s+)?(adopted|approved|agreed|decided|endorsed|"
    r"instructed|requested|noted)\b|채택|승인|합의|결정|지시",
    re.I,
)
NORMATIVE_RE = re.compile(
    r"\b(?:shall|must|is required to|are required to|should)\b|"
    r"하여야\s*한다|해야\s*한다|요구된다|필수",
    re.I,
)
DEADLINE_RE = re.compile(
    r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+(?:19|20)\d{2}\b|"
    r"not later than|within\s+\d+\s+(?:day|month|year)s?|"
    r"entry into force|target(?:\s+year)?\s*(?:19|20)\d{2}|"
    r"(?:19|20)\d{2}\s*년?\s*(?:채택|발효|시행|목표)|발효|이내",
    re.I,
)
REPORT_RE = re.compile(
    r"\breport(?:ing|ed)?\b|submitted data|submission|IMO DCS|GISIS|"
    r"Statement of Compliance|보고|제출",
    re.I,
)
VERIFY_RE = re.compile(
    r"\bverification\b|quality control|assessment|audit|survey|"
    r"검증|품질관리|심사|검사",
    re.I,
)
DESIGN_RE = re.compile(
    r"\bdesign\b|approval|qualification|notation|application|scope|"
    r"설계|승인|적용범위|부기부호",
    re.I,
)
UNCERTAINTY_RE = re.compile(
    r"\bdraft\b|proposal|proposed|target|ambitious|revisit|defer|"
    r"초안|제안|목표|재검토|미확정",
    re.I,
)

STOPWORDS = {
    "관련", "내용", "질문", "정리", "요약", "알려줘", "찾아줘", "무엇",
    "대한", "주요", "최신", "최근", "the", "and", "for", "with", "from",
    "what", "find", "summary", "rule", "guidance",
}


@dataclass(frozen=True)
class QuestionPlan:
    task: str
    organization: str = ""
    session: str = ""
    topics: tuple[str, ...] = ()
    required_evidence: tuple[str, ...] = ()
    requested_count: int = 0
    bullet_min: int = 2
    bullet_max: int = 7
    require_operational_impact: bool = False
    require_rule_guidance: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "organization": self.organization,
            "session": self.session,
            "topics": list(self.topics),
            "required_evidence": list(self.required_evidence),
            "requested_count": self.requested_count,
            "bullet_min": self.bullet_min,
            "bullet_max": self.bullet_max,
            "require_operational_impact": self.require_operational_impact,
            "require_rule_guidance": self.require_rule_guidance,
        }


@dataclass(frozen=True)
class EvidenceUnit:
    citation_id: int
    evidence_type: str
    text: str
    file_name: str
    page: int | None
    clause: str
    authority: int
    status: str
    source: str = ""
    topic_terms: tuple[str, ...] = ()
    practical_score: float = 0.0


@dataclass
class SemanticAnswerResult:
    answer: str
    plan: QuestionPlan
    units: list[EvidenceUnit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    coverage: dict[str, bool] = field(default_factory=dict)
    claim_mappings: list[dict[str, Any]] = field(default_factory=list)
    answer_scope_status: str = "complete"


def _terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(text or "")
        if token.lower() not in STOPWORDS
    }


def analyze_question(question: str, row: dict | None = None) -> QuestionPlan:
    """Build one reusable plan from meaning-bearing query signals."""
    row = row or {}
    q = question or ""
    ql = q.lower()
    session_match = re.search(r"\b(MSC|MEPC)\s*[-/]?\s*(\d{1,3})\b", q, re.I)
    org_match = session_match or re.search(
        r"(?<![A-Za-z0-9])(MSC|MEPC|DNV|LR|ABS|KR)(?![A-Za-z0-9])",
        q,
        re.I,
    )
    organization = org_match.group(1).upper() if org_match else ""
    session = session_match.group(2) if session_match else ""
    count_match = re.search(r"(\d+)\s*개\s*(?:항목|결과|사항)?", q)
    requested = int(count_match.group(1)) if count_match else 0

    rule = bool(re.search(r"rule|guidance|class guideline|notice|requirement|요구사항|선급|규칙|지침", q, re.I))
    impact = bool(
        re.search(
            r"영향|대응|준비|업무|(?<!자율)(?<!원격)운항|보고|제출|실무",
            q,
            re.I,
        )
    )
    timeline = bool(re.search(r"일정|발효|채택|mandatory|timeline|entry into force", q, re.I))
    outcome = bool(re.search(r"결과|결정|채택|승인|합의|outcome|adopt|approve", q, re.I))
    latest = bool(re.search(r"최신|최근|latest|current", q, re.I))

    internal_intent = str(
        row.get("_internal_intent") or row.get("internal_intent") or ""
    )
    if rule:
        task = "rule_lookup"
        required = ("class_rule", "scope", "requirement")
        bullet_min, bullet_max = 2, 3
    elif timeline:
        task = "timeline"
        required = ("decision", "deadline", "document_status")
        bullet_min, bullet_max = 5, 7
    elif outcome or requested:
        task = "meeting_outcome"
        required = ("decision", "document_status")
        bullet_min, bullet_max = (requested or 3), (requested or 7)
    elif latest and not re.search(r"운항|업무|보고|제출|직접\s*영향", q, re.I):
        task = "trend_summary"
        required = ("decision", "requirement", "document_status")
        bullet_min, bullet_max = 7, 10
    elif impact or internal_intent in {"env_regulation", "altfuel_ghg_safety"}:
        task = "operational_impact"
        required = ("requirement", "deadline", "verification")
        bullet_min, bullet_max = 5, 7
    else:
        task = "fact_lookup"
        required = ("requirement",)
        bullet_min, bullet_max = 2, 5

    topic_terms = tuple(sorted(_terms(q)))
    return QuestionPlan(
        task=task,
        organization=organization,
        session=session,
        topics=topic_terms,
        required_evidence=required,
        requested_count=requested,
        bullet_min=bullet_min,
        bullet_max=bullet_max,
        require_operational_impact=impact or task in {"operational_impact", "timeline"},
        require_rule_guidance=rule,
    )


def _evidence_type(sentence: str, source: str) -> str:
    if ACTION_RE.search(sentence):
        return "decision"
    if NORMATIVE_RE.search(sentence):
        return "requirement"
    if DEADLINE_RE.search(sentence):
        return "deadline"
    if REPORT_RE.search(sentence):
        return "reporting"
    if VERIFY_RE.search(sentence):
        return "verification"
    if DESIGN_RE.search(sentence):
        return "scope" if re.search(r"scope|application|적용범위", sentence, re.I) else "design"
    if source.upper() in {"DNV", "LR", "ABS", "KR"}:
        return "class_rule"
    if UNCERTAINTY_RE.search(sentence):
        return "document_status"
    return "fact"


def extract_evidence_units(chunks: list[Any], plan: QuestionPlan) -> list[EvidenceUnit]:
    """Extract sentence-level, typed evidence while preserving citation ids."""
    units: list[EvidenceUnit] = []
    seen: set[tuple[int, str]] = set()
    for citation_id, chunk in enumerate(chunks, 1):
        raw = re.sub(r"^\[[^\]]+\]\s*", "", str(getattr(chunk, "text", "") or ""))
        source = str(getattr(chunk, "source", "") or "")
        status = classify_document_status(chunk)
        for sentence in SENTENCE_RE.split(raw):
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if len(sentence) < 45:
                continue
            etype = _evidence_type(sentence, source)
            terms = _terms(sentence)
            overlap = len(set(plan.topics).intersection(terms))
            practical = (
                overlap * 1.4
                + status.authority * 0.45
                + (2.0 if etype in plan.required_evidence else 0.0)
                + (1.5 if etype in {"deadline", "requirement", "verification", "scope", "design"} else 0.0)
            )
            key = (citation_id, sentence[:180].lower())
            if key in seen:
                continue
            seen.add(key)
            units.append(
                EvidenceUnit(
                    citation_id=citation_id,
                    evidence_type=etype,
                    text=sentence,
                    file_name=str(getattr(chunk, "file_name", "") or ""),
                    page=getattr(chunk, "page_number", None),
                    clause=str(getattr(chunk, "clause_number", "") or ""),
                    authority=status.authority,
                    status=status.code,
                    source=source.upper(),
                    topic_terms=tuple(sorted(terms)),
                    practical_score=practical,
                )
            )
    units.sort(key=lambda unit: unit.practical_score, reverse=True)
    return units


def evidence_coverage(plan: QuestionPlan, units: list[EvidenceUnit]) -> dict[str, bool]:
    available = {unit.evidence_type for unit in units}
    if any(
        unit.status == "class_rule"
        or unit.source in {"DNV", "LR", "ABS", "KR"}
        for unit in units
    ):
        available.add("class_rule")
    return {name: name in available for name in plan.required_evidence}


def _claim_score(line: str, plan: QuestionPlan, units: list[EvidenceUnit]) -> tuple[float, bool]:
    ids = [int(value) for value in CITATION_RE.findall(line)]
    if not ids:
        return (-999.0, False)
    claim_terms = _terms(CITATION_RE.sub("", line))
    cited = [unit for unit in units if unit.citation_id in ids]
    if not cited:
        return (-999.0, False)
    best_overlap = max(
        (len(claim_terms.intersection(set(unit.topic_terms))) for unit in cited),
        default=0,
    )
    evidence_types = {unit.evidence_type for unit in cited}
    practical = max((unit.practical_score for unit in cited), default=0.0)
    score = best_overlap * 1.5 + practical
    if evidence_types.intersection(plan.required_evidence):
        score += 2.5
    if re.search(r"일반적으로|통상적으로|전반적으로|중요합니다|필요가 있습니다", line):
        score -= 3.0
    supported = best_overlap >= 1 or any(unit.status == "class_rule" for unit in cited) or bool(
        evidence_types.intersection({"decision", "deadline", "requirement", "class_rule"})
    )
    return score, supported


def _practical_impacts(
    core_lines: list[str],
    units: list[EvidenceUnit],
    *,
    task: str,
    limit: int = 2,
) -> list[str]:
    """Translate cited facts into narrowly bounded work items.

    The original factual payload and citation are preserved.  Only the label
    changes, so the transformation cannot introduce an uncited obligation.
    """
    labels = {
        "deadline": "규제 일정 관리 기준",
        "reporting": "보고업무 기준",
        "verification": "데이터 QA·검증 기준",
        "requirement": "요구사항 대조 기준",
        "scope": "적용범위 확인 기준",
        "design": "설계·승인 검토 기준",
        "class_rule": "선급 검토 기준",
    }
    impacts: list[str] = []
    used: set[str] = set()
    allowed = (
        {"deadline", "requirement"}
        if task == "timeline"
        else set(labels)
    )
    for line in core_lines:
        # Governance decisions describe committee organization, not a ship or
        # company reporting duty.  Do not relabel them as operational work.
        if re.search(r"working group|작업반|소위원회|위원회", line, re.I):
            continue
        ids = [int(value) for value in CITATION_RE.findall(line)]
        cited_units = [unit for unit in units if unit.citation_id in ids]
        chosen = None
        for unit in cited_units:
            if unit.evidence_type not in labels or unit.evidence_type not in allowed:
                continue
            if unit.evidence_type == "reporting" and not re.search(
                r"report|보고|제출|submitted|GISIS|DCS|Statement of Compliance",
                line,
                re.I,
            ):
                continue
            chosen = unit
            break
        if chosen is None:
            continue
        label = labels[chosen.evidence_type]
        if label in used:
            continue
        payload = re.sub(r"^-\s*", "", line).strip()
        payload = re.sub(r"^\*\*[^*]+\*\*:\s*", "", payload)
        citation_suffix = "".join(
            f"[{value}]" for value in dict.fromkeys(ids)
        )
        prose = CITATION_RE.sub("", payload).strip()
        first_sentence = re.split(r"(?<=[.!?])\s+", prose, maxsplit=1)[0].strip()
        impacts.append(f"- **{label}**: {first_sentence} {citation_suffix}".strip())
        used.add(label)
        if len(impacts) >= limit:
            break
    return impacts


def _parse_sections(answer: str) -> dict[str, list[str]]:
    sections = {"1": [], "2": [], "3": [], "4": []}
    current = "1"
    for raw in (answer or "").splitlines():
        heading = re.match(r"\s*#+\s*([1-4])\)", raw)
        if heading:
            current = heading.group(1)
            continue
        if raw.strip().startswith("- "):
            sections[current].append(raw.strip())
    return sections


def _rule_identity_line(units: list[EvidenceUnit]) -> str:
    """Build a generic, cited instrument identity from retrieved metadata."""
    candidates = [
        unit
        for unit in units
        if unit.status == "class_rule"
        or unit.evidence_type == "class_rule"
        or unit.source in {"DNV", "LR", "ABS", "KR"}
    ]
    if not candidates:
        return ""
    unit = max(candidates, key=lambda item: (item.authority, item.practical_score))
    name = re.sub(r"\.pdf$", "", unit.file_name or "", flags=re.I).strip()
    if not name:
        return ""
    return (
        f"- **{name}** (Rule/Guidance): 질문 주제와 직접 연결되는 조항이 "
        f"검색 근거에서 확인됩니다. [{unit.citation_id}]"
    )


def _render_sections(sections: dict[str, list[str]]) -> str:
    headings = {
        "1": "## 1) 핵심 요약",
        "2": "## 2) 선박 운항/업무 영향",
        "3": "## 3) 추후 확인 필요사항",
        "4": "## 4) 관련 선급 Rule / Guidance",
    }
    parts: list[str] = []
    for key in ("1", "2", "3", "4"):
        parts.extend([headings[key], "", "\n".join(sections.get(key) or []), ""])
    return "\n".join(parts).strip()


def refine_answer(
    question: str,
    row: dict,
    answer: str,
    citation_chunks: list[Any],
) -> SemanticAnswerResult:
    """Remove generic/weak claims and retain the most useful grounded claims."""
    plan = analyze_question(question, row)
    units = extract_evidence_units(citation_chunks, plan)
    coverage = evidence_coverage(plan, units)
    sections = _parse_sections(answer)
    warnings: list[str] = []
    claim_mappings: list[dict[str, Any]] = []

    for key in sections:
        ranked: list[tuple[float, str]] = []
        for line in sections[key]:
            if key == "4" and re.search(
                r"별도\s*(?:선급\s*)?문서\s*검색|별도로\s*확인", line, re.I
            ):
                warnings.append("uncited_rule_search_advice_removed")
                continue
            score, supported = _claim_score(line, plan, units)
            ids = [int(value) for value in CITATION_RE.findall(line)]
            cited_units = [unit for unit in units if unit.citation_id in ids]
            claim_terms = _terms(CITATION_RE.sub("", line))
            overlap = max(
                (
                    len(claim_terms.intersection(set(unit.topic_terms)))
                    for unit in cited_units
                ),
                default=0,
            )
            claim_mappings.append(
                {
                    "section": key,
                    "claim": CITATION_RE.sub("", line).strip(),
                    "citation_ids": ids,
                    "supported": supported,
                    "lexical_overlap": overlap,
                    "evidence_types": sorted(
                        {unit.evidence_type for unit in cited_units}
                    ),
                }
            )
            if not supported:
                warnings.append("semantic_claim_removed")
                continue
            ranked.append((score, line))
        ranked.sort(key=lambda item: item[0], reverse=True)
        sections[key] = [line for _, line in ranked]

    if plan.require_rule_guidance and not sections["4"]:
        identity = _rule_identity_line(units)
        if identity:
            sections["4"] = [identity]

    missing_evidence = [
        name for name in plan.required_evidence if not coverage.get(name, False)
    ]
    if plan.require_rule_guidance and missing_evidence and units:
        best = max(units, key=lambda item: (item.authority, item.practical_score))
        labels = {
            "class_rule": "Rule/Guidance 식별",
            "scope": "적용범위",
            "requirement": "세부 요구사항",
        }
        missing_labels = "·".join(labels.get(name, name) for name in missing_evidence)
        sections["3"] = [
            f"- **근거 범위 제한**: 현재 검색 근거에는 {missing_labels} 정보가 "
            f"부족하므로, 아래 결과를 해당 선급의 전체 요구사항으로 확대 해석할 수 "
            f"없습니다. [{best.citation_id}]"
        ]

    if plan.require_operational_impact and not sections["2"]:
        sections["2"] = _practical_impacts(
            sections["1"], units, task=plan.task
        )

    # Keep the requested count for explicit N-item questions.  Otherwise use
    # the category ceiling without padding the answer with generic statements.
    if plan.requested_count:
        sections["1"] = sections["1"][: plan.requested_count]
    else:
        sections["1"] = sections["1"][: plan.bullet_max]

    total_max = plan.bullet_max + (1 if plan.requested_count else 0)
    while sum(len(lines) for lines in sections.values()) > total_max:
        # Operational implications are derived from section 1 and therefore
        # the first candidates to trim.  Never discard the only follow-up
        # caveat or the only Rule/Guidance identification to satisfy length.
        if len(sections["2"]) > 1:
            sections["2"].pop()
        elif len(sections["1"]) > max(1, plan.requested_count):
            sections["1"].pop()
        elif len(sections["2"]) == 1:
            sections["2"].pop()
        else:
            break

    # Rule answers must not pretend that a single incidental clause represents
    # the society's full rule set.
    if plan.require_rule_guidance and not sections["4"]:
        warnings.append("rule_guidance_evidence_missing")

    missing = [name for name, found in coverage.items() if not found]
    scope_status = "complete"
    if missing:
        warnings.append("missing_evidence:" + ",".join(missing))
        scope_status = "partial"

    # A cited sentence can be correct while the answer as a whole is
    # incomplete.  Make that distinction machine-readable so the UI and
    # downstream users never confuse a single incidental clause with a
    # complete Rule/Guidance survey.
    if plan.require_rule_guidance:
        substantive_core = len(sections["1"]) + len(sections["4"])
        if substantive_core < 2 or not all(
            coverage.get(name, False)
            for name in ("class_rule", "scope", "requirement")
        ):
            scope_status = "insufficient"
            warnings.append("question_scope_not_satisfied")

    return SemanticAnswerResult(
        answer=_render_sections(sections),
        plan=plan,
        units=units,
        warnings=list(dict.fromkeys(warnings)),
        coverage=coverage,
        claim_mappings=claim_mappings,
        answer_scope_status=scope_status,
    )
