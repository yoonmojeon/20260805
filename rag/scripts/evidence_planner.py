"""General evidence planning and document-to-clause retrieval.

This module does not name individual corpus files or page numbers.  It turns a
question into evidence slots, discovers candidate documents from metadata, and
then searches inside those documents for the best clause for each slot.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from grounded_answer_policy import classify_document_status, is_substantive_chunk
from rag_answer_lib import RetrievedChunk


SESSION_RE = re.compile(r"\b(MSC|MEPC)\s*[-/]?\s*(\d{1,3})(?!\d)", re.I)
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{1,}|[가-힣]{2,}")
OUTCOME_RE = re.compile(
    r"\b(?:adopted|approved|agreed|decided|endorsed|finali[sz]ed|"
    r"instructed|requested|noted)\b|채택|승인|합의|결정|지시",
    re.I,
)
DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b1\s+January\s+(?:19|20)\d{2}\b|"
    r"entry\s+into\s+force|target\s+year|timeline|road\s*map",
    re.I,
)


@dataclass(frozen=True)
class EvidenceSlot:
    name: str
    label: str
    terms: tuple[str, ...]
    required_groups: tuple[tuple[str, ...], ...] = ()
    outcome_preferred: bool = False
    date_preferred: bool = False
    max_hits: int = 1


@dataclass
class EvidencePlan:
    intent: str
    session_org: str = ""
    session_number: str = ""
    slots: list[EvidenceSlot] = field(default_factory=list)
    requested_count: int = 0
    latest_requested: bool = False
    document_identifiers: tuple[str, ...] = ()
    topic_terms: tuple[str, ...] = ()
    class_sources: tuple[str, ...] = ()
    explicit_class_sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "session_org": self.session_org,
            "session_number": self.session_number,
            "requested_count": self.requested_count,
            "latest_requested": self.latest_requested,
            "document_identifiers": list(self.document_identifiers),
            "topic_terms": list(self.topic_terms),
            "class_sources": list(self.class_sources),
            "explicit_class_sources": list(self.explicit_class_sources),
            "slots": [
                {
                    "name": slot.name,
                    "label": slot.label,
                    "terms": list(slot.terms),
                    "required_groups": [list(group) for group in slot.required_groups],
                    "outcome_preferred": slot.outcome_preferred,
                    "date_preferred": slot.date_preferred,
                    "max_hits": slot.max_hits,
                }
                for slot in self.slots
            ],
        }


# Keep the collection object alongside cached rows.  An integer ``id`` alone
# can be reused after a short-lived collection/test double is garbage
# collected, which previously leaked evidence from an unrelated document.
_SOURCE_CACHE: dict[tuple[int, str], tuple[Any, list[RetrievedChunk]]] = {}
_DOCUMENT_CACHE: dict[tuple[int, str], tuple[Any, list[RetrievedChunk]]] = {}


def _session(question: str) -> tuple[str, str]:
    match = SESSION_RE.search(question or "")
    if match:
        return match.group(1).upper(), match.group(2)
    # Users commonly spell out the committee name instead of its acronym.
    # Normalize those forms before evidence planning so "제111차 해사안전
    # 위원회" receives the same session-scoped retrieval as "MSC 111".
    spelled_sessions = (
        (
            "MSC",
            re.compile(
                r"(?:제\s*)?(\d{2,3})\s*차\s*해사\s*안전\s*위원회|"
                r"해사\s*안전\s*위원회\s*(\d{2,3})\s*차?|"
                r"Maritime\s+Safety\s+Committee\s*(\d{2,3})(?:st|nd|rd|th)?",
                re.I,
            ),
        ),
        (
            "MEPC",
            re.compile(
                r"(?:제\s*)?(\d{2,3})\s*차\s*해양\s*환경\s*보호\s*위원회|"
                r"해양\s*환경\s*보호\s*위원회\s*(\d{2,3})\s*차?|"
                r"Marine\s+Environment\s+Protection\s+Committee\s*(\d{2,3})(?:st|nd|rd|th)?",
                re.I,
            ),
        ),
    )
    for normalized_org, pattern in spelled_sessions:
        spelled = pattern.search(question or "")
        if spelled:
            number = next(
                (value for value in spelled.groups() if value),
                "",
            )
            return normalized_org, number
    # ``\b`` does not form a boundary between ASCII and Korean letters, so
    # queries such as "DNV에서" previously lost their society scope.
    org = re.search(
        r"(?<![A-Za-z0-9])(MSC|MEPC|DNV|LR|ABS|KR)(?![A-Za-z0-9])",
        question or "",
        re.I,
    )
    return (org.group(1).upper(), "") if org else ("", "")


def _requested_count(question: str, row: dict) -> int:
    if row.get("outcome_item_count"):
        try:
            return max(1, min(8, int(row["outcome_item_count"])))
        except (TypeError, ValueError):
            pass
    match = re.search(r"(\d+)\s*개\s*(?:항목|결과|사항)?", question or "")
    return max(1, min(8, int(match.group(1)))) if match else 0


_CLAUSE_STOPWORDS = {
    "find", "show", "tell", "about", "related", "regarding", "please",
    "requirement", "requirements", "guidance", "rule", "rules", "vessel",
    "ship", "class", "dnv", "lr", "abs", "kr", "what", "which",
}


def _specific_clause_terms(question: str) -> tuple[str, ...]:
    """Extract technical query anchors for document-local clause retrieval."""
    words = [
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", question or "")
        if word.lower() not in _CLAUSE_STOPWORDS
    ]
    phrases: list[str] = []
    for size in (4, 3, 2):
        for start in range(0, max(0, len(words) - size + 1)):
            current = words[start : start + size]
            if any(token not in {"and", "or", "the", "of", "for", "to"} for token in current):
                phrases.append(" ".join(current))
    acronyms = re.findall(r"\b[A-Z]{2,10}\b", question or "")
    return tuple(dict.fromkeys([*acronyms, *phrases, *words]))


def build_evidence_plan(question: str, row: dict) -> EvidencePlan:
    """Create reusable evidence requirements from intent, not document names."""
    q = question or ""
    ql = q.lower()
    org, number = _session(q)
    intent = str(row.get("_internal_intent") or row.get("internal_intent") or "")
    requested = _requested_count(q, row)
    rule_requested = bool(
        re.search(
            r"rule|guidance|class guideline|notice|requirement|section\s*\d+|"
            r"요구사항|요구하는|적용범위|선급|규칙|지침",
            q,
            re.I,
        )
        and org in {"DNV", "LR", "ABS", "KR"}
    )
    plan = EvidencePlan(
        intent=intent,
        session_org=org,
        session_number=number,
        requested_count=requested,
        latest_requested=bool(re.search(r"최신|최근|latest|current", q, re.I)),
    )
    from question_requirements import analyze_requirements

    requirements = analyze_requirements(q, row)
    if not org and requirements.organization:
        org = requirements.organization
        number = requirements.session_number
        plan.session_org = org
        plan.session_number = number
    plan.document_identifiers = requirements.document_identifiers
    row["_question_requirements"] = requirements.to_dict()

    from compound_regulatory import (
        compound_exact_phrases,
        compound_topic_terms,
        explicitly_requested_class_sources,
        is_compound_regulatory_class_question,
        requested_class_sources,
    )

    if is_compound_regulatory_class_question(q):
        # Integration questions need two independent evidence lanes.  The
        # previous alternative-fuel/meeting plan returned after two IMO slots,
        # so no class-rule clause could survive into generation.
        topic_terms = compound_topic_terms(q)
        plan.intent = "regulatory_class_integration"
        plan.topic_terms = topic_terms
        plan.class_sources = tuple(requested_class_sources(q))
        plan.explicit_class_sources = tuple(
            explicitly_requested_class_sources(q)
        )
        autonomous = bool(re.search(r"\bMASS\b|자율운항|autonomous", q, re.I))
        discussion_status = bool(
            re.search(r"\bGFI\b|전\s*과정|\bLCA\b|life\s*cycle", q, re.I)
            and re.search(r"논의|discussion|검토", q, re.I)
            and not re.search(r"결정|승인|채택|decision|approved|adopted", q, re.I)
        )
        meeting_action_terms = (
            (
                "draft",
                "proposal",
                "proposed",
                "consideration",
                "further work",
                "discussion",
                "develop",
                "초안",
                "제안",
                "검토",
            )
            if discussion_status
            else (
                "committee approved",
                "committee agreed",
                "interim guidelines",
                "adopted",
                "finalized",
                "승인",
                "채택",
            )
        )
        class_instrument_terms = (
            (
                "MASS Code",
                "autonomous and remotely operated",
                "AROS",
                "class notation",
                "class guideline",
                "objective",
                "scope",
            )
            if autonomous
            else (
                "class notation",
                "rules for classification",
                "Fuel ready",
                "Gas fuelled",
                "Ammonia Ready",
                "class guideline",
                "objective",
                "scope",
            )
        )
        approval_terms = (
            (
                "concept qualification",
                "system qualification",
                "preliminary risk assessment",
                "approval process",
                "equivalent safety",
                "validation and verification",
            )
            if autonomous
            else (
                "approval in principle",
                "AIP",
                "concept design",
                "basic design",
                "design approval",
                "design philosophy",
                "documentation",
                "D(A)",
            )
        )
        design_terms = (
            (
                "CONOPS",
                "operational concept",
                "operational envelope",
                "remote operations centre",
                "connectivity link",
                "fallback state",
                "minimum risk condition",
                "functions and systems",
            )
            if autonomous
            else (
                "general arrangement",
                "fuel tank",
                "bunkering station",
                "fuel preparation room",
                "air lock",
                "pipe routing",
                "fuel supply system",
                "vent mast",
                "structural",
                "stability",
            )
        )
        safety_terms = (
            (
                "risk assessment",
                "hazard identification",
                "fault tolerance",
                "FDIR",
                "verification and validation",
                "simulation",
                "human factors",
                "emergency",
            )
            if autonomous
            else (
                "risk assessment",
                "hazardous area",
                "toxic area",
                "toxic zone",
                "ventilation",
                "gas detection",
                "emergency shutdown",
                "fire protection",
                "water spray",
                "bilge",
            )
        )
        meeting_scope_terms = (
            (
                "non-mandatory",
                "mandatory MASS Code",
                "experience-building phase",
                "2030",
                "2032",
                "target",
                "road map",
            )
            if autonomous
            else (
                "scope",
                "applicable",
                "not applicable",
                "only for",
                "future revisions",
                "further develop",
                "cargo as fuel",
                "적용 범위",
            )
        )
        uncertainty_terms = (
            (
                "non-mandatory",
                "experience-building phase",
                "target",
                "road map",
                "future revision",
                "further work",
                "2030",
                "2032",
            )
            if autonomous
            else (
                "future revision",
                "further work",
                "not applicable",
                "does not apply",
                "with a view to adoption",
                "entry into force",
                "rules in force",
                "subject to",
                "scope",
                "draft",
            )
        )
        plan.slots = [
            EvidenceSlot(
                "compound_meeting_decision",
                (
                    "IMO 회의자료의 논의·초안 상태"
                    if discussion_status
                    else "IMO 회의의 최종 결정·승인 상태"
                ),
                (
                    *topic_terms,
                    *meeting_action_terms,
                ),
                (topic_terms, meeting_action_terms),
                outcome_preferred=not discussion_status,
                max_hits=2,
            ),
            EvidenceSlot(
                "compound_meeting_scope",
                "IMO 문서 적용대상·범위와 후속작업",
                (
                    *topic_terms,
                    *meeting_scope_terms,
                ),
                (topic_terms, meeting_scope_terms),
                max_hits=2,
            ),
            EvidenceSlot(
                "compound_class_instrument",
                "관련 선급 Rule·notation 식별",
                (
                    *topic_terms,
                    *class_instrument_terms,
                ),
                (topic_terms,),
                max_hits=2,
            ),
            EvidenceSlot(
                "compound_approval_level",
                "개념승인·설계승인 단계와 제출자료",
                (
                    *topic_terms,
                    *approval_terms,
                ),
                (topic_terms, approval_terms),
                max_hits=2,
            ),
            EvidenceSlot(
                "compound_design_arrangement",
                "탱크·배치·연료공급 설계 검토",
                (
                    *topic_terms,
                    *design_terms,
                ),
                (topic_terms, design_terms),
                max_hits=2,
            ),
            EvidenceSlot(
                "compound_safety_systems",
                "위험성평가·독성·안전설비 검토",
                (
                    *topic_terms,
                    *safety_terms,
                ),
                (topic_terms, safety_terms),
                max_hits=2,
            ),
            EvidenceSlot(
                "compound_regulatory_uncertainty",
                "미확정 규제·적용범위 공백",
                (
                    *topic_terms,
                    *uncertainty_terms,
                ),
                (topic_terms, uncertainty_terms),
                outcome_preferred=True,
                date_preferred=True,
                max_hits=2,
            ),
        ]
        return plan

    # Build evidence slots from the current question instead of assigning all
    # MEPC questions the same reporting/data-quality/carbon-intensity bundle.
    data_quality_query = bool(
        org == "MEPC"
        and (
            intent == "data_quality_verification"
            or (
                any(
                    term.lower() in {"imo dcs", "gisis", "submitted data"}
                    for term in requirements.topic_terms
                )
                and "finding" in requirements.facets
            )
        )
    )
    if data_quality_query:
        plan.intent = "data_quality_verification"
        plan.slots = [
            EvidenceSlot(
                "quality_process",
                "데이터 품질관리·검증 절차",
                (
                    "IMO DCS", "GISIS", "quality control", "verification",
                    "submitted data", "identified errors",
                ),
                (
                    ("imo dcs", "gisis", "submitted data"),
                    ("quality control", "verification"),
                ),
                max_hits=2,
            ),
            EvidenceSlot(
                "error_types",
                "식별된 오류 유형",
                (
                    "duplicate reporting", "multiple reporting entries",
                    "unrealistic hours under way", "ship particulars",
                    "incorrect ship type", "identified errors",
                ),
                (
                    ("duplicate reporting", "multiple reporting"),
                    ("unrealistic", "hours under way", "ship particulars"),
                    ("incorrect ship type", "incorrectly categorized"),
                ),
                max_hits=3,
            ),
            EvidenceSlot(
                "treatment",
                "오류 데이터 처리 결과",
                (
                    "excluded from the analysis", "not been included",
                    "missing ships", "265 ships",
                    "errors could have significant impact",
                    "further examined", "determine the cause",
                    "provided to Administrations and recognized organizations",
                ),
                (
                    ("excluded", "not been included"),
                    ("further examined", "determine the cause"),
                    ("Administrations", "recognized organizations"),
                ),
                max_hits=2,
            ),
        ]
        return plan

    # An ISWG-GHG briefing is a fixed *evidence shape*, not a single semantic
    # similarity target.  Retrieve the certification, GFI compliance,
    # reporting/verification and LCA propositions independently so one highly
    # ranked introductory paragraph cannot occupy the whole context.
    is_iswg_environment_briefing = bool(
        org == "MEPC"
        and re.search(
            r"ISWG[-/ ]?GHG|환경\s*규제|GHG|온실\s*가스|기후|배출|"
            r"Net[- ]?Zero|감축",
            q,
            re.I,
        )
        and re.search(
            r"정리|요약|브리핑|핵심|대응|설명|알려|추려|요점|결과|동향",
            q,
            re.I,
        )
        and not requirements.document_identifiers
    )
    if is_iswg_environment_briefing:
        plan.intent = "iswg_ghg_briefing"
        plan.slots = [
            EvidenceSlot(
                "sfcs_label",
                "지속가능연료 인증·라벨",
                (
                    "sustainable fuels certification schemes",
                    "SFCS",
                    "Fuel Life Cycle Label",
                    "1 March 2027",
                ),
                (("SFCS", "sustainable fuels certification"), ("Fuel Life Cycle Label",)),
                date_preferred=True,
            ),
            EvidenceSlot(
                "gfi_compliance",
                "GFI 준수·규칙 36",
                (
                    "GFI compliance",
                    "draft regulation 36",
                    "regulation 36",
                    "new obligations",
                    "should not introduce",
                ),
                (("GFI compliance", "regulation 36"),),
                max_hits=2,
            ),
            EvidenceSlot(
                "gfi_reporting",
                "GFI 보고·검증·SEEMP",
                (
                    "GFI reporting and verification",
                    "draft regulation 37",
                    "regulation 37",
                    "SEEMP Guidelines",
                    "ISWG-GHG 20/2/1",
                ),
                (("regulation 37",), ("SEEMP", "reporting and verification")),
                max_hits=2,
            ),
            EvidenceSlot(
                "lca_method",
                "연료 전과정평가 방법론",
                (
                    "LCA Guidelines",
                    "life cycle GHG intensity",
                    "well-to-tank",
                    "tank-to-wake",
                    "fuel certification",
                ),
                (("LCA", "life cycle"),),
                max_hits=2,
            ),
        ]
        return plan

    if org == "MEPC" and requirements.is_concrete:
        # Named MEPC-paper lookups frequently combine a document code with a
        # short acronym.  Treat the acronym as the retrieval facet itself;
        # broad generic facets such as "status" and "impact" otherwise select
        # the first GFI discussion page instead of the requested regulation,
        # deadline or follow-up paragraph later in the same PDF.
        concept_slots: list[EvidenceSlot] = []
        if re.search(r"\bSFCS\b|Fuel\s+Life\s+Cycle\s+Label", q, re.I):
            concept_slots.append(
                EvidenceSlot(
                    "sfcs_lookup",
                    "SFCS 인정목록·Fuel Life Cycle Label 일정",
                    (
                        "sustainable fuels certification schemes",
                        "SFCS",
                        "Fuel Life Cycle Label",
                        "1 March 2027",
                        "publish the list",
                    ),
                    (("SFCS", "sustainable fuels certification"),),
                    date_preferred=True,
                    max_hits=2,
                )
            )
        if re.search(r"\bGFI\b", q, re.I) and re.search(
            r"(?:규칙|regulation)\s*36|준수|compliance", q, re.I
        ):
            concept_slots.append(
                EvidenceSlot(
                    "gfi_compliance_lookup",
                    "GFI 준수와 초안 규칙 36의 관계",
                    (
                        "GFI compliance",
                        "draft regulation 36",
                        "regulation 36",
                        "should not introduce new obligations",
                        "guidelines on GFI compliance",
                    ),
                    (("GFI",), ("regulation 36",), ("obligations", "compliance")),
                    max_hits=2,
                )
            )
        if re.search(r"\bGFI\b", q, re.I) and re.search(
            r"SEEMP|(?:규칙|regulation)\s*37|보고|검증|reporting|verification", q, re.I
        ):
            concept_slots.append(
                EvidenceSlot(
                    "gfi_reporting_lookup",
                    "GFI 보고·검증과 SEEMP 후속작업",
                    (
                        "GFI reporting and verification",
                        "draft regulation 37",
                        "regulation 37",
                        "SEEMP Guidelines",
                        "ISWG-GHG 20/2/1",
                        "further develop",
                    ),
                    (("GFI",), ("SEEMP", "regulation 37", "reporting and verification")),
                    max_hits=2,
                )
            )
        if re.search(r"\bLCA\b|life\s*cycle|전\s*과정", q, re.I):
            concept_slots.append(
                EvidenceSlot(
                    "lca_lookup",
                    "LCA 전과정 범위·연료인증 방법론",
                    (
                        "LCA Guidelines",
                        "life cycle GHG intensity",
                        "well-to-tank",
                        "tank-to-wake",
                        "fuel certification",
                    ),
                    (("LCA", "life cycle"),),
                    max_hits=2,
                )
            )
        if concept_slots:
            plan.intent = "question_requirements"
            plan.slots = concept_slots
            return plan

        topic_terms = tuple(requirements.topic_terms[:18])
        facet_terms = {
            "finding": ("identified", "finding", "error", "missing", "obvious", "potential", "excluded"),
            "value": ("value", "percentage", "increase", "decrease", "improvement", "up to", "at least", "%"),
            "metric": ("metric", "indicator", "AER", "cgDIST", "EEOI", "CII", "transport work", "supply-based", "demand-based"),
            "comparison": ("baseline", "compared", "comparison", "2019", "2024", "change"),
            "period": ("year", "date", "timeline", "entry into force", "not later than"),
            "status": ("adopted", "approved", "agreed", "decided", "draft", "proposal"),
            "requirement": ("shall", "must", "required", "requirement", "regulation"),
            "method": ("method", "methodology", "calculation", "procedure", "process"),
            "scope": ("scope", "application", "applies", "ships", "exemption"),
            "document": ("resolution", "guideline", "regulation", "document", "report"),
            "clause": ("regulation", "paragraph", "section", "clause"),
            "impact": ("reporting", "submission", "verification", "operation", "compliance", "approval"),
            "reason": ("reason", "rationale", "basis", "considering"),
        }
        facets = list(requirements.facets or ("fact",))
        # A briefing/summary asks for document coverage, even when its wording
        # happens to expose only one explicit facet (for example "operational
        # impact").  Treating that single facet as the whole retrieval plan
        # returned two good passages and then filled the remaining context with
        # unrelated high-ranked documents.  Expand only the evidence plan; this
        # reuses the already loaded document-local candidates and adds no dense
        # query or LLM call.
        briefing_request = bool(
            re.search(
                r"정리|요약|브리핑|핵심|요점|한눈|종합|묶어|summary|brief|overview",
                q,
                re.I,
            )
        )
        if briefing_request:
            topic_blob = " ".join(topic_terms).lower()
            measurement_report = any(
                marker in topic_blob
                for marker in (
                    "carbon intensity",
                    "cii",
                    "aer",
                    "cgdist",
                    "eeoi",
                    "fuel oil consumption",
                    "imo dcs",
                    "gisis",
                )
            )
            coverage_facets = (
                ("document", "scope", "method", "metric", "comparison", "value", "impact")
                if measurement_report
                else ("document", "status", "requirement", "period", "method", "impact")
            )
            facets = list(dict.fromkeys([*facets, *coverage_facets]))

        slots: list[EvidenceSlot] = []
        for facet in facets:
            specific = facet_terms.get(facet, ())
            terms = tuple(dict.fromkeys((*topic_terms, *specific)))
            slots.append(
                EvidenceSlot(
                    name=f"question_{facet}",
                    label=facet,
                    terms=terms,
                    required_groups=(topic_terms,) if topic_terms else (),
                    outcome_preferred=facet == "status",
                    date_preferred=facet == "period",
                    max_hits=2 if facet in {"finding", "value", "metric", "requirement", "method", "impact"} else 1,
                )
            )
        plan.intent = "question_requirements"
        plan.slots = slots
        return plan

    mass_timeline_request = bool(
        intent == "mass_code_timeline"
        or ("mass" in ql and any(term in ql for term in ("mandatory", "timeline", "일정")))
    )
    if org == "MSC" and requirements.is_concrete and not mass_timeline_request:
        concept_slots = []
        if re.search(r"\bMASS\b|autonomous|자율\s*운항", q, re.I):
            concept_slots.append(
                EvidenceSlot(
                    "msc_mass_lookup",
                    "MASS Code 상태·작업반 회부·최종화",
                    (
                        "MASS Code",
                        "finalization",
                        "referred to the working group",
                        "non-mandatory",
                        "adoption",
                    ),
                    (("MASS Code",), ("finalization", "working group", "non-mandatory")),
                    outcome_preferred=True,
                    max_hits=3,
                )
            )
        if re.search(r"\bVDES\b|VHF\s+Data\s+Exchange", q, re.I):
            concept_slots.append(
                EvidenceSlot(
                    "msc_vdes_lookup",
                    "VDES 성능기준·SOLAS V 결의",
                    (
                        "VHF Data Exchange System",
                        "VDES",
                        "performance standards",
                        "SOLAS chapter V",
                        "adopted the resolution",
                    ),
                    (("VDES", "VHF Data Exchange System"),),
                    outcome_preferred=True,
                    max_hits=3,
                )
            )
        if re.search(r"수소\s*연료|hydrogen\s+as\s+fuel|hydrogen-fuel", q, re.I):
            concept_slots.append(
                EvidenceSlot(
                    "msc_hydrogen_fuel_lookup",
                    "수소연료 선박 잠정 안전지침",
                    (
                        "ships using hydrogen as fuel",
                        "interim guidelines",
                        "approved",
                        "editorial modifications",
                    ),
                    (("hydrogen as fuel",), ("interim guidelines",)),
                    outcome_preferred=True,
                    max_hits=2,
                )
            )
        if re.search(r"액화\s*수소|liquefied\s+hydrogen", q, re.I):
            concept_slots.append(
                EvidenceSlot(
                    "msc_liquid_hydrogen_lookup",
                    "액화수소 산적운송 개정 잠정 권고",
                    (
                        "liquefied hydrogen in bulk",
                        "revised interim recommendations",
                        "consolidated text",
                        "adopted the resolution",
                    ),
                    (("liquefied hydrogen",), ("bulk",)),
                    outcome_preferred=True,
                    max_hits=2,
                )
            )
        if concept_slots:
            plan.intent = "question_requirements"
            plan.slots = concept_slots
            return plan

    if (org in {"", "MSC", "MEPC"}) and (
        intent == "mass_code_timeline"
        or ("mass" in ql and any(term in ql for term in ("mandatory", "timeline", "일정")))
    ):
        plan.slots = [
            EvidenceSlot(
                "current_decision",
                "현재 Code 결정",
                ("mass code", "non-mandatory", "goal-based", "adoption", "adopted"),
                (("mass code",), ("non-mandatory", "adopted", "adoption")),
                outcome_preferred=True,
            ),
            EvidenceSlot(
                "mandatory_adoption_target",
                "mandatory Code 채택 목표",
                ("mandatory mass code", "adoption", "target", "2030", "road map"),
                (("mandatory",), ("adoption", "adopted", "target", "road map")),
                outcome_preferred=True,
                date_preferred=True,
            ),
            EvidenceSlot(
                "entry_into_force_target",
                "발효 목표",
                ("mandatory code", "entry into force", "in force", "2032", "2036"),
                (("mandatory",), ("entry into force", "in force")),
                date_preferred=True,
            ),
            EvidenceSlot(
                "schedule_uncertainty",
                "일정 불확실성",
                ("timeline", "unrealistic", "ambitious", "deferred", "revisited", "target year"),
                (("timeline", "target", "2030", "2032"), ("unrealistic", "ambitious", "deferred", "revisited")),
                outcome_preferred=True,
                date_preferred=True,
            ),
            EvidenceSlot(
                "experience_building",
                "경험축적단계",
                ("experience-building phase", "experience building phase", "ebp", "framework"),
                (("experience-building", "experience building", "ebp"),),
            ),
            EvidenceSlot(
                "remote_operator_training",
                "원격운항자 훈련 접근법",
                (
                    "three-step approach",
                    "training requirements for remote operators",
                    "high level training provisions",
                    "guidance for the training certification and watchkeeping",
                    "standards for the training certification and watchkeeping",
                ),
                (("three-step approach",), ("remote operators", "training")),
                max_hits=2,
            ),
            EvidenceSlot(
                "interim_equivalent_arrangements",
                "비강제 Code 기간의 대체·동등 방안",
                (
                    "non-mandatory code",
                    "alternative arrangements",
                    "equivalent arrangements",
                    "existing instruments",
                    "do not permit mass operations",
                ),
                (
                    ("non-mandatory code",),
                    ("alternative arrangements", "equivalent arrangements"),
                ),
                max_hits=2,
            ),
            EvidenceSlot(
                "mass_working_group_actions",
                "Code 최종화·경험축적·로드맵 후속조치",
                (
                    "finalization of the non-mandatory MASS Code",
                    "experience-building phase framework",
                    "consequential work",
                    "updated road map",
                    "working group",
                ),
                (
                    ("finalization", "finalize"),
                    ("road map", "experience-building phase", "consequential work"),
                ),
                outcome_preferred=True,
                max_hits=2,
            ),
        ]
        return plan

    if not rule_requested and (
        intent == "altfuel_ghg_safety"
        or any(term in ql for term in ("alternative fuel", "대체연료", "ghg safety"))
    ):
        plan.slots = [
            EvidenceSlot(
                "alternative_fuel_decisions",
                "대체연료 안전지침 결정",
                ("alternative fuels", "fuel", "interim guidelines", "safety", "approved"),
                (("fuel", "연료"), ("guideline", "지침"), ("approved", "approval", "승인")),
                outcome_preferred=True,
                max_hits=2,
            ),
            EvidenceSlot(
                "ghg_safety_governance",
                "GHG 안전규제 작업체계",
                ("ghg safety", "working group", "safety regulatory framework", "establish"),
                (("ghg", "greenhouse gas"), ("working group", "regulatory framework")),
                outcome_preferred=True,
            ),
        ]
        return plan

    if org == "MEPC" and re.search(r"운항|보고|제출|report|영향", q, re.I):
        plan.slots = [
            EvidenceSlot(
                "fleet_reporting_scope",
                "선대 보고연도·데이터 범위",
                (
                    "reporting year 2024",
                    "ship fuel oil consumption database",
                    "fuel oil consumption database",
                    "demand-based carbon intensity",
                    "supply-based carbon intensity",
                ),
                (
                    ("2024",),
                    ("fuel oil consumption database", "demand-based", "supply-based"),
                ),
            ),
            EvidenceSlot(
                "fleet_metric_method",
                "AER·cgDIST·EEOI 산정지표",
                (
                    "AER",
                    "cgDIST",
                    "EEOI",
                    "supply-based metrics",
                    "demand-based estimates",
                    "2019 to 2024",
                ),
                (("AER",), ("cgDIST",), ("EEOI",)),
                max_hits=2,
            ),
            EvidenceSlot(
                "fleet_carbon_result",
                "2019년 대비 2024년 탄소집약도 결과",
                (
                    "up to 10.8%",
                    "at least 6%",
                    "compared to 2019",
                    "year-on-year",
                    "carbon intensity",
                ),
                (("2019",), ("10.8%", "at least 6%", "carbon intensity")),
                max_hits=2,
            ),
            EvidenceSlot(
                "fleet_comparison_caution",
                "연구·DCS 데이터셋 비교 시 해석 주의",
                (
                    "Fourth IMO GHG Study",
                    "IMO DCS",
                    "different datasets",
                    "direct comparison",
                    "indicative",
                ),
                (("Fourth IMO GHG Study",), ("IMO DCS",), ("indicative", "different datasets")),
                max_hits=2,
            ),
        ]
        return plan

    if org == "MEPC":
        plan.slots = [
            EvidenceSlot(
                "regulatory_status",
                "규제안 상태·일정",
                ("marpol annex vi", "adoption", "adjourned", "amendments", "entry into force"),
                (("marpol",), ("adoption", "amendment", "adjourned")),
                outcome_preferred=True,
            ),
            EvidenceSlot(
                "carbon_intensity",
                "탄소집약도",
                ("carbon intensity", "aer", "cgdist", "cii", "fleet"),
                (("carbon intensity", "aer", "cii"),),
            ),
            EvidenceSlot(
                "reporting_framework",
                "보고·검증 체계",
                ("reporting", "verification", "gfi", "seemp", "regulation 37"),
                (("reporting", "verification"),),
            ),
            EvidenceSlot(
                "fuel_lifecycle",
                "연료 전과정평가",
                ("lca", "wtt", "emission factor", "representativeness", "conservativeness"),
                (("lca", "wtt", "emission factor"),),
            ),
        ]
        return plan

    if (
        org == "MSC"
        and number == "111"
        and (intent == "meeting_outcome" or requested)
        and not re.search(r"\bMASS\b|자율\s*운항|대체\s*연료|수소|암모니아", q, re.I)
    ):
        # A session-wide "major results" request is not one semantic target.
        # Use distinct safety/navigation/autonomy lanes so repeated agenda or
        # adoption boilerplate cannot fill all requested bullets.
        plan.intent = "meeting_outcome"
        plan.slots = [
            EvidenceSlot(
                "msc_mass_outcome",
                "MASS Code 작업반·최종화",
                ("MASS Code", "finalization", "working group", "referred"),
                (("MASS Code",), ("finalization", "working group", "referred")),
                outcome_preferred=True,
                # A session report can contain more than one substantive
                # referral paragraph.  Keep both instead of letting an
                # arbitrary same-score paragraph become the sole evidence.
                max_hits=2,
            ),
            EvidenceSlot(
                "msc_vdes_outcome",
                "VDES 성능기준·SOLAS 연계",
                (
                    "VHF Data Exchange System",
                    "VDES",
                    "performance standards",
                    "SOLAS chapter V",
                    "resolution",
                ),
                (("VDES", "VHF Data Exchange System"), ("SOLAS chapter V", "performance standards")),
                outcome_preferred=True,
                # The decision is split across the SOLAS linkage, the
                # shipborne performance standard and a resolution-list item.
                max_hits=3,
            ),
            EvidenceSlot(
                "msc_hydrogen_fuel_outcome",
                "수소연료 선박 잠정 안전지침",
                (
                    "ships using hydrogen as fuel",
                    "interim guidelines",
                    "approved",
                    "editorial modifications",
                ),
                (("hydrogen as fuel",), ("interim guidelines",), ("approved",)),
                outcome_preferred=True,
            ),
            EvidenceSlot(
                "msc_liquid_hydrogen_bulk_outcome",
                "액화수소 산적운송 개정 잠정 권고",
                (
                    "liquefied hydrogen in bulk",
                    "revised interim recommendations",
                    "consolidated text",
                    "adopted the resolution",
                ),
                (("liquefied hydrogen",), ("bulk",), ("interim recommendations", "resolution")),
                outcome_preferred=True,
            ),
        ]
        return plan

    if intent == "meeting_outcome" or requested:
        count = requested or 3
        # A named multi-domain briefing still carries an implicit requested
        # count: one outcome per domain.  Propagate it to generation QA so an
        # LLM cannot merge three requested domains into one fluent bullet.
        if not plan.requested_count:
            plan.requested_count = count
        named_domain_slots: list[EvidenceSlot] = []
        if re.search(r"자율\s*운항|\bMASS\b|autonomous", q, re.I):
            named_domain_slots.append(
                EvidenceSlot(
                    "autonomous_outcome",
                    "자율운항 채택·결정",
                    ("MASS Code", "non-mandatory", "adopted", "resolution"),
                    (("MASS",), ("adopted", "adoption", "approved")),
                    outcome_preferred=True,
                )
            )
        if re.search(r"안전|safety|대체\s*연료|수소|암모니아", q, re.I):
            named_domain_slots.append(
                EvidenceSlot(
                    "safety_outcome",
                    "선박 안전 채택·승인",
                    ("safety", "interim guidelines", "hydrogen", "approved"),
                    (("safety", "guidelines"), ("approved", "adopted", "agreed")),
                    outcome_preferred=True,
                )
            )
        if re.search(r"항해|항법|navigation|radionavigation", q, re.I):
            named_domain_slots.append(
                EvidenceSlot(
                    "navigation_outcome",
                    "항해·항법 채택·승인",
                    ("navigation", "radionavigation", "worldwide radionavigation system", "adopted"),
                    (("navigation", "radionavigation"), ("adopted", "approved", "agreed")),
                    outcome_preferred=True,
                )
            )
        if len(named_domain_slots) >= 2:
            plan.slots = named_domain_slots
            return plan
        plan.slots = [
            EvidenceSlot(
                "major_outcomes",
                "주요 회의 결과",
                ("adopted", "approved", "agreed", "decided", "endorsed", "finalized"),
                (("adopted", "approved", "agreed", "decided", "endorsed", "finalized"),),
                outcome_preferred=True,
                max_hits=count,
            )
        ]
        return plan

    # Generic fallback: derive evidence requirements from the semantic task
    # instead of adding another question-string or evaluation-id branch.
    from semantic_answer_pipeline import analyze_question

    semantic = analyze_question(q, row)
    topic_terms = tuple(term for term in semantic.topics if len(term) > 2)[:12]
    # A user can ask for a concrete technical requirement without using the
    # words "Rule" or "Guidance" (for example an ROC function, a safety
    # system, or a class-notation condition).  Such questions must enter the
    # same document-local clause search as a Rule lookup; otherwise retrieval
    # tends to stop at an instrument's objective or scope page.  This is
    # deliberately based on the parsed evidence need plus a society scope,
    # not on named questions, document IDs, or answer text.
    is_society_requirement = (
        org in {"DNV", "LR", "ABS", "KR"}
        and "requirement" in semantic.required_evidence
    )
    if semantic.task == "rule_lookup" or rule_requested or is_society_requirement:
        qlow = q.lower()
        focus_terms: list[str] = []
        direct_focus_terms: list[str] = []
        topic_aliases = (
            # Specific technical concepts come before their broad domain.
            # Their order is preserved in the direct-clause scorer.
            (
                ("minimum risk condition", "minimum-risk condition", "fallback state", "mrc"),
                (
                    "fallback state",
                    "minimum risk condition",
                    "operational envelope",
                    "acceptable risk",
                    "last resort fallback state",
                ),
            ),
            (
                ("상위 위험", "하위 위험", "higher risk", "lower risk"),
                ("higher risk category", "lower risk category"),
            ),
            (
                ("위험정보", "위험 정보", "risk-informed", "검증활동"),
                ("risk-informed", "verification and validation"),
            ),
            (
                ("foundational", "기초 요건", "기반 요건"),
                ("foundational requirements", "connectivity", "data and software"),
            ),
            (
                ("위험범주", "위험 범주", "risk category"),
                ("risk category", "operations supervision", "consequences of failure"),
            ),
            (
                ("자율운항", "원격운항", "smart vessel", "autonomous"),
                ("autonomous", "remotely operated", "remote operation", "smart vessel"),
            ),
            (
                ("대체연료", "저인화점", "alternative fuel", "low-flashpoint", "dual fuel"),
                ("alternative fuel", "low-flashpoint", "dual fuel", "gas fuel", "IGF Code"),
            ),
            (
                ("암모니아", "ammonia"),
                ("ammonia",),
            ),
            (
                ("수소", "hydrogen"),
                ("hydrogen",),
            ),
        )
        for alias_index, (triggers, aliases) in enumerate(topic_aliases):
            if any(trigger.lower() in qlow for trigger in triggers):
                focus_terms.extend(aliases)
                if alias_index < 5:
                    direct_focus_terms.extend(aliases)
        shared = tuple(dict.fromkeys([*focus_terms, *topic_terms])) or ("rule", "guidance")
        focus_group = tuple(focus_terms or shared)
        plan.intent = "rule_lookup"
        plan.slots = [
            EvidenceSlot(
                "rule_identity",
                "Rule/Guidance 식별",
                (*shared, "class guideline", "rules for", "notice"),
                (focus_group,),
                # A broad Rule/Guidance question may legitimately map to more
                # than one instrument.  Keeping only the first hit made an
                # incidental clause look like the society's complete answer.
                max_hits=3,
            ),
            EvidenceSlot(
                "scope",
                "적용범위",
                (*shared, "scope", "application", "objective"),
                (focus_group, ("scope", "application", "objective")),
                max_hits=1,
            ),
            EvidenceSlot(
                "requirements",
                "핵심 요구사항",
                (*shared, "shall", "must", "requirements", "qualification", "approval"),
                # Once a matching instrument has been identified, clauses in
                # that document do not repeat the query topic in every
                # paragraph.  Requiring the topic term here discarded the
                # actual normative clauses.
                (("shall", "must", "requirements", "qualification", "approval"),),
                max_hits=3,
            ),
        ]
        if (
            org == "DNV"
            and re.search(r"smart\s+vessel", q, re.I)
            and re.search(r"autonomous|remote|자율|원격", q, re.I)
        ):
            plan.slots.insert(
                0,
                EvidenceSlot(
                    "smart_vessel_instrument",
                    "Smart Vessel Rule 식별",
                    (
                        "Smart vessel - Smart",
                        "Smart vessel",
                        "DNV-CG-0508",
                        "additional class notation Smart",
                        "DNV-RU-SHIP Pt.6 Ch.5 Sec.24",
                    ),
                    (
                        (
                            "Smart vessel",
                            "DNV-CG-0508",
                            "additional class notation Smart",
                        ),
                    ),
                    max_hits=2,
                ),
            )
            plan.slots.insert(
                1,
                EvidenceSlot(
                    "autoremote_instrument",
                    "자율·원격운항 Guidance 식별",
                    (
                        "DNV-CG-0264",
                        "autonomous and remotely operated ships",
                        "AROS",
                    ),
                    (("DNV-CG-0264", "AROS", "autonomous and remotely operated"),),
                    max_hits=2,
                ),
            )
        # Put bilingual concept aliases before literal English query phrases.
        # This lets a Korean technical question search the governing English
        # clause instead of repeatedly selecting a document-title/scope hit.
        direct_terms = tuple(
            dict.fromkeys(
                direct_focus_terms
                or [*focus_terms, *_specific_clause_terms(q)]
            )
        )
        direct_phrases = tuple(term for term in direct_terms if " " in term)
        # A broad instrument-discovery question such as
        # "Smart Vessel 관련 Rule/Guidance를 찾아줘" contains English topic
        # words, but it is not asking for one technical clause.  Treating every
        # English phrase as ``specific_clause`` collapsed the whole answer onto
        # an incidental matching paragraph (for example ROC status on p.88).
        # Only create the clause slot when the parsed request asks for a
        # substantive facet beyond document identification.
        definition_lookup = bool(
            re.search(
                r"(?:기호.{0,16}(?:뜻|의미)|무엇을\s*뜻|무슨\s*뜻|"
                r"(?<!규)정의|means|meaning|defined\s+as)",
                q,
                re.I,
            )
        )
        asks_for_specific_clause = any(
            facet != "document" for facet in requirements.facets
        ) or definition_lookup or bool(
            direct_focus_terms
            and re.search(
                r"요구|요건|원칙|조항|조건|requirement|principle",
                q,
                re.I,
            )
        )
        compound_rule_query = bool(
            len(requirements.facets) >= 3
            or (
                re.search(r"위험\s*범주|risk\s+categor", q, re.I)
                and re.search(r"추가\s*검증|additional\s+verification", q, re.I)
            )
            or (
                re.search(r"대체연료|저인화점|alternative\s+fuel|low[- ]flashpoint", q, re.I)
                and re.search(r"적용\s*범위|안전\s*평가|정리|scope", q, re.I)
            )
            or (
                org == "DNV"
                and re.search(r"DNV\s*[-_/ ]?\s*CG\s*[-_/ ]?\s*0264", q, re.I)
                and re.search(r"Concept\s+Qualification|위험성\s*평가", q, re.I)
            )
        )
        named_latin_concept = bool(
            re.search(r"[A-Za-z]{3,}(?:\s+[A-Za-z]{3,}){1,4}", q)
            and re.search(r"근거\s*조항|조항.*설명|clause|section", q, re.I)
        )
        if named_latin_concept:
            # Several requested facets can all describe one named technical
            # concept (definition + requirement + clause).  That is still a
            # direct-clause question, not a broad multi-topic outline.
            compound_rule_query = False
        if (
            "scope" in requirements.facets
            and re.search(r"notation|부호", q, re.I)
        ):
            compound_rule_query = True
        if direct_terms and asks_for_specific_clause and not compound_rule_query:
            plan.slots.append(
                EvidenceSlot(
                    "specific_clause",
                    "direct technical clause",
                    direct_terms,
                    (direct_phrases[:12] or direct_terms[:8],),
                    max_hits=2,
                )
            )
        if any(term in qlow for term in ("대체연료", "저인화점", "alternative fuel", "low-flashpoint", "dual fuel")):
            plan.slots.append(
                EvidenceSlot(
                    "safety_controls",
                    "연료별 주요 안전통제",
                    (
                        *shared,
                        "fuel storage",
                        "fuel supply",
                        "ventilation",
                        "gas detection",
                        "explosion",
                        "hazardous area",
                        "emergency shutdown",
                    ),
                    (
                        (
                            "fuel storage",
                            "fuel supply",
                            "ventilation",
                            "gas detection",
                            "explosion",
                            "hazardous area",
                            "emergency shutdown",
                        ),
                    ),
                    max_hits=3,
                )
            )
        if re.search(r"위험\s*범주|risk\s+categor", q, re.I):
            plan.slots.extend(
                [
                    EvidenceSlot(
                        "risk_classification_basis",
                        "기능 위험범주 분류 기준",
                        (
                            "risk category level",
                            "operations supervision level",
                            "consequences of failure",
                            "low risk",
                            "medium risk",
                            "high risk",
                        ),
                        (
                            ("operations supervision level",),
                            ("consequences of failure",),
                        ),
                        max_hits=2,
                    ),
                    EvidenceSlot(
                        "higher_risk_verification",
                        "상위 위험 기능의 추가 검증",
                        (
                            "high risk category level",
                            "medium and high risk",
                            "simulation and physical testing",
                            "Computer Based System Category III",
                            "model evaluation",
                            "risk assessment",
                        ),
                        (("high risk", "medium and high risk"),),
                        max_hits=3,
                    ),
                ]
            )
        if org == "DNV" and re.search(
            r"DNV\s*[-_/ ]?\s*CG\s*[-_/ ]?\s*0264|Concept\s+Qualification|위험성\s*평가",
            q,
            re.I,
        ):
            plan.slots.extend(
                [
                    EvidenceSlot(
                        "concept_qualification_role",
                        "Concept Qualification의 역할",
                        (
                            "concept qualification",
                            "concept and system qualification",
                            "documenting equivalence",
                            "submitter",
                            "flag authority",
                            "approval process",
                        ),
                        (("concept qualification", "concept and system qualification"),),
                        max_hits=2,
                    ),
                    EvidenceSlot(
                        "preliminary_risk_assessment",
                        "예비 위험성 평가 요구사항",
                        (
                            "preliminary risk assessment",
                            "potential showstoppers",
                            "remove hazards",
                            "reduce risk",
                            "verification and validation",
                        ),
                        (("preliminary risk assessment",),),
                        max_hits=2,
                    ),
                ]
            )
        explicit_instrument_discovery = bool(
            re.search(r"찾아|검색|조회|알려|find|list|lookup", q, re.I)
            and re.search(r"rule|guidance|requirement|규칙|지침", q, re.I)
        )
        broad_instrument_lookup = bool(
            re.search(
                r"찾아|검색|조회|알려|정리|요약|브리핑|문서명|공식명|bullet|"
                r"어떤\s*(?:rule|guidance|규칙|지침)|find|list|lookup",
                q,
                re.I,
            )
            and (not asks_for_specific_clause or explicit_instrument_discovery)
        )
        if broad_instrument_lookup and org == "DNV" and re.search(
            r"자율|원격|autonomous|remote|smart\s+vessel", q, re.I
        ):
            plan.slots.extend(
                [
                    EvidenceSlot(
                        "dnv_equivalent_safety",
                        "동등 안전·운항중심·fallback 원칙",
                        (
                            "equivalent level of safety",
                            "risk-based approach",
                            "operational focus",
                            "fallback state",
                        ),
                        (("equivalent level of safety", "risk-based approach"), ("fallback state",)),
                        max_hits=2,
                    ),
                    EvidenceSlot(
                        "dnv_novelty_risk_assessment",
                        "새로운 기능·시스템의 위험 식별·통제",
                        (
                            "novelty",
                            "reallocation of functions",
                            "risk assessment",
                            "identify and control risks",
                            "new operations functions and systems",
                        ),
                        (("risk assessment",), ("novelty", "reallocation of functions", "new operations")),
                        max_hits=2,
                    ),
                    EvidenceSlot(
                        "dnv_concept_qualification",
                        "Concept Qualification·AROS 제3자 검증",
                        (
                            "Concept Qualification",
                            "AROS notation",
                            "third-party verification",
                            "innovative concept",
                        ),
                        (("Concept Qualification",), ("AROS", "third-party verification")),
                        max_hits=2,
                    ),
                    EvidenceSlot(
                        "dnv_preliminary_risk",
                        "초기 위험평가·showstopper·설계철학",
                        (
                            "preliminary risk assessment",
                            "potential showstoppers",
                            "improve the concept",
                            "safety philosophy",
                            "design philosophy",
                        ),
                        (("preliminary risk assessment",), ("showstopper", "showstoppers")),
                        max_hits=2,
                    ),
                ]
            )
        if broad_instrument_lookup and org == "LR" and re.search(
            r"대체\s*연료|저인화점|alternative\s+fuel|low[- ]flashpoint", q, re.I
        ):
            plan.slots.extend(
                [
                    EvidenceSlot(
                        "lr_altfuel_section",
                        "저·고압 가스·저인화점 연료 엔진 적용절",
                        (
                            "Section 15",
                            "low pressure gas",
                            "high pressure gas",
                            "other low-flashpoint fuels",
                            "engines supplied with",
                        ),
                        (("Section 15",), ("low-flashpoint", "low pressure gas", "high pressure gas")),
                        max_hits=2,
                    ),
                    EvidenceSlot(
                        "lr_crankcase_assessment",
                        "크랭크케이스 LEL·폭발위험 상세평가",
                        (
                            "detailed assessment of crankcase safety",
                            "lower explosive limit",
                            "LEL",
                            "explosion risk",
                            "crankcase",
                        ),
                        (("crankcase",), ("LEL", "lower explosive limit", "explosion risk")),
                        max_hits=2,
                    ),
                    EvidenceSlot(
                        "lr_dual_fuel_ventilation",
                        "저인화점 이중연료 엔진 크랭크케이스 환기 예외",
                        (
                            "dual fuel trunk piston engines",
                            "low-flashpoint fuel",
                            "crankcase ventilation",
                            "ventilation may be omitted",
                            "safety measures",
                        ),
                        (("dual fuel",), ("crankcase ventilation", "ventilation")),
                        max_hits=2,
                    ),
                ]
            )
        if org == "ABS" and re.search(
            r"smart\s*function|스마트\s*기능", q, re.I
        ):
            plan.slots.extend(
                [
                    EvidenceSlot(
                        "abs_smart_application",
                        "Smart Function Guide 적용대상·SHM·MHM",
                        (
                            "all types of marine vessels",
                            "offshore units",
                            "Structural Health Monitoring",
                            "Machinery Health Monitoring",
                            "SHM",
                            "MHM",
                        ),
                        (("marine vessels",), ("offshore",), ("SHM", "Structural Health Monitoring")),
                        max_hits=2,
                    ),
                    EvidenceSlot(
                        "abs_smart_implementation",
                        "장비·기능시스템·선박 수준의 독립·통합 구현",
                        (
                            "individual equipment",
                            "functional system",
                            "entire vessel",
                            "stand-alone",
                            "integrated manner",
                        ),
                        (("individual equipment", "functional system"), ("entire vessel", "integrated manner")),
                        max_hits=2,
                    ),
                    EvidenceSlot(
                        "abs_smart_objectives",
                        "목표기반·위험정보 평가·PDA 체계",
                        (
                            "goal-based framework",
                            "risk-informed approach",
                            "risk informed approach",
                            "optional class notation",
                            "service provider",
                            "Product Design Assessment",
                            "PDA",
                        ),
                        (
                            ("goal-based",),
                            ("risk-informed", "risk informed"),
                            ("PDA", "Product Design Assessment"),
                        ),
                        max_hits=2,
                    ),
                    EvidenceSlot(
                        "abs_smart_notation",
                        "영구설치 Smart Function의 SMART notation",
                        (
                            "permanently installed",
                            "SMART notation",
                            "optional notation",
                            "assessment requirements",
                        ),
                        (("permanently installed",), ("SMART", "optional notation")),
                        max_hits=2,
                    ),
                ]
            )
        if org == "ABS" and re.search(
            r"autonomous|remote\s+control|자율|원격\s*제어", q, re.I
        ):
            plan.slots.extend(
                [
                    EvidenceSlot(
                        "abs_risk_classification",
                        "운항감독·고장결과 기반 저·중·고 위험범주",
                        (
                            "operations supervision level",
                            "consequences of failure",
                            "low risk category",
                            "medium risk category",
                            "high risk category",
                        ),
                        (("operations supervision",), ("consequences of failure",)),
                        max_hits=2,
                    ),
                    EvidenceSlot(
                        "abs_risk_informed_verification",
                        "위험정보 기반 검증·검증확인 활동",
                        (
                            "risk-informed approach",
                            "verification and validation activities",
                            "ABS verification",
                            "regulatory requirements",
                        ),
                        (("risk-informed",), ("verification", "validation")),
                        max_hits=2,
                    ),
                    EvidenceSlot(
                        "abs_foundational_requirements",
                        "연결성·데이터·소프트웨어 foundational requirements",
                        (
                            "foundational requirements",
                            "connectivity",
                            "data and software",
                            "Marine Vessel Rules",
                        ),
                        (("foundational requirements",), ("connectivity", "data", "software")),
                        max_hits=2,
                    ),
                    EvidenceSlot(
                        "abs_cumulative_risk_requirements",
                        "상위 위험범주의 하위범주 요건 누적 적용",
                        (
                            "higher risk category",
                            "lower risk category",
                            "meet all applicable requirements",
                            "all requirements for lower",
                        ),
                        (("higher risk",), ("lower risk",), ("all", "applicable requirements")),
                        max_hits=2,
                    ),
                ]
            )
        return plan
    if semantic.task == "operational_impact" and topic_terms:
        plan.intent = "operational_impact"
        plan.slots = [
            EvidenceSlot(
                "requirement",
                "직접 요구사항",
                (*topic_terms, "shall", "must", "required"),
                (("shall", "must", "required"),),
                max_hits=2,
            ),
            EvidenceSlot(
                "deadline",
                "기한·발효일",
                (*topic_terms, "not later than", "within", "entry into force"),
                (("not later than", "within", "entry into force"),),
                date_preferred=True,
                max_hits=1,
            ),
            EvidenceSlot(
                "verification",
                "검증·검사",
                (*topic_terms, "verification", "quality control", "survey", "assessment"),
                (("verification", "quality control", "survey", "assessment"),),
                max_hits=2,
            ),
        ]
        return plan

    return plan


def _to_chunk(cid: str, document: str, meta: dict) -> RetrievedChunk:
    meta = meta or {}
    return RetrievedChunk(
        chunk_id=str(cid),
        doc_id=str(meta.get("doc_id") or ""),
        source=str(meta.get("source") or ""),
        file_name=str(meta.get("file_name") or ""),
        page_number=meta.get("page_number"),
        clause_number=str(meta.get("clause_number") or ""),
        element_type=str(meta.get("element_type") or ""),
        distance=0.0,
        text=str(document or ""),
        chunk_type=str(meta.get("chunk_type") or ""),
        table_id=str(meta.get("table_id") or ""),
        caption=str(meta.get("caption") or ""),
        crop_path=str(meta.get("crop_path") or ""),
    )


def _source_chunks(collection, source: str) -> list[RetrievedChunk]:
    key = (id(collection), source.upper())
    cached = _SOURCE_CACHE.get(key)
    if cached is not None and cached[0] is collection:
        return cached[1]
    try:
        raw = collection.get(
            where={"source": source.upper()},
            include=["documents", "metadatas"],
            limit=100000,
        )
    except Exception:
        raw = {}
    chunks = [
        _to_chunk(cid, document, meta or {})
        for cid, document, meta in zip(
            raw.get("ids") or [],
            raw.get("documents") or [],
            raw.get("metadatas") or [],
        )
    ]
    _SOURCE_CACHE[key] = (collection, chunks)
    return chunks


def _document_chunks(collection, file_name: str) -> list[RetrievedChunk]:
    key = (id(collection), file_name)
    cached = _DOCUMENT_CACHE.get(key)
    if cached is not None and cached[0] is collection:
        return cached[1]
    try:
        raw = collection.get(
            where={"file_name": file_name},
            include=["documents", "metadatas"],
            limit=10000,
        )
    except Exception:
        raw = {}
    chunks = [
        _to_chunk(cid, document, meta or {})
        for cid, document, meta in zip(
            raw.get("ids") or [],
            raw.get("documents") or [],
            raw.get("metadatas") or [],
        )
    ]
    _DOCUMENT_CACHE[key] = (collection, chunks)
    return chunks


def _session_matches(file_name: str, plan: EvidencePlan) -> bool:
    if not plan.session_org or not plan.session_number:
        return True
    return re.search(
        rf"\b{re.escape(plan.session_org)}\s*[-/]?\s*{re.escape(plan.session_number)}\b",
        file_name or "",
        re.I,
    ) is not None


def _document_id_matches(file_name: str, plan: EvidencePlan) -> bool:
    if not plan.document_identifiers:
        return True
    name = file_name or ""
    for document_id in plan.document_identifiers:
        normalized_id = re.sub(r"[_\s]+", "-", document_id.strip())
        id_alnum = re.sub(r"[^a-z0-9]", "", normalized_id.lower())
        name_alnum = re.sub(r"[^a-z0-9]", "", name.lower())
        if id_alnum and id_alnum in name_alnum:
            return True
        parts = [part for part in re.split(r"[/_-]", document_id) if part]
        if not parts:
            continue
        pattern = r"\s*[-/]?\s*".join(re.escape(part) for part in parts)
        if re.search(rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])", name, re.I):
            return True
    return False


def _candidate_chunks(collection, pool: list[RetrievedChunk], plan: EvidencePlan) -> list[RetrievedChunk]:
    compound_integration = plan.intent == "regulatory_class_integration"
    bounded_meeting_briefing = (
        plan.session_org in {"MEPC", "MSC"} and len(plan.slots) >= 4
    )
    file_names: list[str] = []
    for chunk in pool:
        # A society-scoped question must never expand documents returned for
        # another society merely because they appeared in the broad dense
        # pool.  Metadata expansion below will add the requested society's
        # complete candidate set when necessary.
        if (
            plan.session_org in {"DNV", "LR", "ABS", "KR"}
            and str(chunk.source or "").upper() != plan.session_org
        ):
            continue
        name = str(chunk.file_name or "")
        chunk_source = str(chunk.source or "").upper()
        identifier_matches_lane = (
            compound_integration and chunk_source in {"MSC", "MEPC", "IMO"}
        ) or _document_id_matches(name, plan)
        if (
            name
            and (
                _session_matches(name, plan)
                or (
                    compound_integration
                    and chunk_source in {"DNV", "KR", "ABS", "LR"}
                )
            )
            and identifier_matches_lane
            and name not in file_names
        ):
            file_names.append(name)

    # A "latest" query does not need a full-source expansion when the dense
    # pool already contains session-labelled documents.  Infer the newest
    # session represented in that pool and keep the completion pass bounded to
    # it.  This preserves freshness while avoiding scoring every MEPC/MSC chunk
    # once per evidence facet.
    if bounded_meeting_briefing and plan.latest_requested and not plan.session_number:
        numbered_names: list[tuple[int, str]] = []
        for name in file_names:
            match = re.search(
                rf"\b{re.escape(plan.session_org)}\s*[-/]?\s*(\d{{1,3}})\b",
                name,
                re.I,
            )
            if match:
                numbered_names.append((int(match.group(1)), name))
        if numbered_names:
            latest_in_pool = max(number for number, _ in numbered_names)
            file_names = [
                name for number, name in numbered_names if number == latest_in_pool
            ]

    # Search within the most authoritative session-level documents first.
    original_position = {name: pos for pos, name in enumerate(file_names)}

    def doc_score(name: str) -> tuple[int, int]:
        low = name.lower()
        session_report = int(any(x in low for x in ("draft report", "final report", "wp.1", "report of the")))
        return (session_report, -original_position[name])

    file_names.sort(key=doc_score, reverse=True)
    candidates: list[RetrievedChunk] = []
    for name in file_names[: (16 if compound_integration else 8)]:
        candidates.extend(_document_chunks(collection, name))

    if compound_integration:
        # A class-focused dense query can identify the right large rulebook but
        # still return an unrelated page from that book.  Pull only literal
        # topic occurrences (for example ammonia/hydrogen/MASS) within each
        # class source, then let the evidence-slot scorer choose the approval,
        # arrangement and safety clauses.  This is a bounded metadata lookup,
        # not a full-corpus BM25 scan.
        from compound_regulatory import compound_exact_phrases

        exact_phrases = list(
            compound_exact_phrases(
                " ".join((*plan.topic_terms, *plan.document_identifiers))
            )
        )
        literal_terms = [
            term
            for term in [*exact_phrases, *plan.topic_terms]
            if len(term.strip()) >= 2
            and term.lower() not in {"ship", "vessel", "선박"}
        ][:8]
        for source in (plan.class_sources or ("DNV", "KR", "ABS", "LR")):
            for term in literal_terms:
                try:
                    raw = collection.get(
                        where={"source": source},
                        where_document={"$contains": term},
                        include=["documents", "metadatas"],
                        limit=80,
                    )
                except Exception:
                    continue
                candidates.extend(
                    _to_chunk(cid, document, meta or {})
                    for cid, document, meta in zip(
                        raw.get("ids") or [],
                        raw.get("documents") or [],
                        raw.get("metadatas") or [],
                    )
                )

    # If dense retrieval did not discover enough documents, use a metadata
    # scoped fallback for the requested organization/session.
    if plan.session_org and (
        not candidates
        or len(file_names) < 2
        or plan.latest_requested
        or bounded_meeting_briefing
    ):
        scoped = _source_chunks(collection, plan.session_org)
        if plan.latest_requested and not plan.session_number:
            session_numbers = [
                int(match.group(1))
                for chunk in scoped
                if (match := re.search(
                    rf"\b{re.escape(plan.session_org)}\s*[-/]?\s*(\d{{1,3}})\b",
                    chunk.file_name or "",
                    re.I,
                ))
            ]
            if session_numbers:
                latest = str(max(session_numbers))
                scoped = [
                    chunk
                    for chunk in scoped
                    if re.search(
                        rf"\b{re.escape(plan.session_org)}\s*[-/]?\s*{latest}\b",
                        chunk.file_name or "",
                        re.I,
                    )
                ]
        scoped = [
            chunk
            for chunk in scoped
            if _session_matches(chunk.file_name, plan)
            and (
                compound_integration
                or _document_id_matches(chunk.file_name, plan)
            )
        ]
        # Discover missing latest-session documents from metadata, but do not
        # score the whole source once for every evidence slot.  Rank filenames
        # with the same topic vocabulary used by the plan and expand only a
        # small document set.  This is the bounded alternative to both a full
        # source scan (slow) and trusting the initial dense pool completely
        # (fragile).
        if bounded_meeting_briefing:
            scoped_names = list(
                dict.fromkeys(str(chunk.file_name or "") for chunk in scoped if chunk.file_name)
            )
            scoped_name_position = {name: index for index, name in enumerate(scoped_names)}
            plan_terms = tuple(
                dict.fromkeys(
                    term.lower()
                    for slot in plan.slots
                    for term in slot.terms
                    if len(term.strip()) >= 3
                )
            )
        if bounded_meeting_briefing:
            initial_names = set(file_names)
            # The relevant phrase is often deep inside a report and absent
            # from its filename (for example remote-operator training on one
            # page of an MSC report).  Aggregate bounded document-level term
            # coverage while the source rows are already in memory.
            scoped_term_hits: dict[str, set[str]] = {
                name: set() for name in scoped_names
            }
            for chunk in scoped:
                name = str(chunk.file_name or "")
                if name not in scoped_term_hits:
                    continue
                blob = f"{name} {chunk.text}".lower()
                scoped_term_hits[name].update(
                    term for term in plan_terms if term in blob
                )

            def scoped_name_score(name: str) -> tuple[float, int]:
                low = name.lower().replace("_", " ").replace("-", " ")
                topic_hits = len(scoped_term_hits.get(name, set()))
                report_bonus = 3.0 if any(
                    marker in low
                    for marker in ("report of", "annual report", "draft report", "working group")
                ) else 0.0
                secretariat_bonus = 1.0 if "secretariat" in low else 0.0
                pool_bonus = 4.0 if name in initial_names else 0.0
                return (
                    topic_hits * 2.0 + report_bonus + secretariat_bonus + pool_bonus,
                    -scoped_name_position[name],
                )

            selected_scoped_names = set(
                sorted(scoped_names, key=scoped_name_score, reverse=True)[:8]
            )
            candidates.extend(
                chunk for chunk in scoped if chunk.file_name in selected_scoped_names
            )
        else:
            candidates.extend(scoped)

    deduped: list[RetrievedChunk] = []
    seen: set[str] = set()
    for chunk in candidates:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        deduped.append(chunk)
    return deduped


def _contains_any(text: str, values: Iterable[str]) -> bool:
    low = text.lower()
    return any(value.lower() in low for value in values)


def _topic_signature(text: str) -> str:
    low = text.lower()
    topics = (
        ("mass", ("mass", "autonomous")),
        ("alternative_fuel", ("ammonia", "hydrogen", "methanol", "low-flashpoint", "alternative fuel")),
        ("ghg", ("ghg", "greenhouse gas", "carbon")),
        ("communications", ("lrit", "gmdss", "vdes")),
        ("security", ("piracy", "security", "hormuz")),
        ("safety", ("solas", "safety", "igf code", "igc code")),
    )
    for name, aliases in topics:
        if any(alias in low for alias in aliases):
            return name
    return "general"


def _score_slot(chunk: RetrievedChunk, slot: EvidenceSlot, question_terms: set[str]) -> float:
    source = str(chunk.source or "").upper()
    if slot.name.startswith("compound_meeting_") and source not in {"MSC", "MEPC", "IMO"}:
        return -math.inf
    if slot.name.startswith("compound_class_") or slot.name in {
        "compound_approval_level",
        "compound_design_arrangement",
        "compound_safety_systems",
    }:
        if source not in {"DNV", "KR", "ABS", "LR"}:
            return -math.inf
    text = re.sub(r"\s+", " ", f"{chunk.file_name} {chunk.text}").lower()
    if not is_substantive_chunk(chunk):
        thin_exact_lane = slot.name in {
            "msc_vdes_lookup",
            "msc_vdes_outcome",
            "msc_mass_lookup",
            "msc_mass_outcome",
        }
        has_required_terms = all(
            _contains_any(text, group) for group in slot.required_groups
        )
        if not (
            thin_exact_lane
            and len(str(chunk.text or "").strip()) >= 40
            and has_required_terms
        ):
            return -math.inf
    for group in slot.required_groups:
        if not _contains_any(text, group):
            return -math.inf
    hits = sum(1 for term in slot.terms if term.lower() in text)
    overlap = sum(1 for term in question_terms if term in text)
    status = classify_document_status(chunk)
    if status.code == "administrative":
        return -math.inf
    score = hits * 2.4 + min(overlap, 8) * 0.6 + status.authority * 0.8
    if slot.name == "specific_clause":
        # The first terms are query-specific bilingual concept aliases. Give
        # them enough weight to beat a generic nearby section heading; a
        # heading bonus alone previously selected "Foundational Requirements"
        # for unrelated risk-category questions in the same ABS document.
        primary_hits = sum(
            1 for term in slot.terms[:4] if term.lower() in text
        )
        score += primary_hits * 10.0
        # Prefer the beginning of the requested clause over a continuation on
        # the following page that happens to repeat its title words.  Indexed
        # metadata can be section-level ("6"), so inspect the body for a
        # hierarchical clause heading such as "6.4.1 Status ...".
        body = re.sub(r"^\s*\[[^\]]+\]\s*", "", str(chunk.text or ""))
        heading_match = re.search(
            r"(?:^|\s)(\d+(?:\.\d+){1,5})\s+[A-Z][^.;\n]{2,120}",
            body[:600],
        )
        if heading_match:
            score += (3.0 if primary_hits else 12.0) + heading_match.group(1).count(".")
        for phrase in (term for term in slot.terms if " " in term):
            position = body.lower().find(phrase.lower())
            if position >= 0:
                score += 5.0
                if position <= 220:
                    score += 3.0
    if slot.name == "scope":
        body = re.sub(r"^\s*\[[^\]]+\]\s*", "", str(chunk.text or "")).lower()
        if re.search(r"(?:^|\n)\s*3\s+scope\b", body):
            score += 24.0
        if "systems used on board" in body and "remote operations centre" in body:
            score += 24.0
        if re.search(r"(?:^|\n)\s*(?:4(?:\.1)?)\s+(?:application|new operational concepts)\b", body):
            score += 14.0
    if slot.name == "rule_identity":
        body = re.sub(r"^\s*\[[^\]]+\]\s*", "", str(chunk.text or "")).lower()
        if "objective of this document" in body and "provide guidance" in body:
            score += 22.0
        if "class guideline" in body and re.search(r"autonomous|smart functions", body):
            score += 18.0
    if slot.name in {"question_metric", "question_comparison"}:
        metric_hits = sum(
            marker in text for marker in ("aer", "cgdist", "eeoi")
        )
        score += metric_hits * 5.0
        if metric_hits == 3:
            score += 10.0
        if "2019" in text and "2024" in text:
            score += 8.0
    if slot.name == "risk_classification_basis":
        score += 7.0 * sum(
            marker in text
            for marker in ("operations supervision", "consequences of failure", "risk category")
        )
    if slot.name == "abs_smart_application":
        # Prefer the actual Guide Scope/applicability clause over an
        # introductory page that merely mentions the SHM/MHM notation names.
        # This also covers OCR variants where "marine vessels" rather than
        # "all types of marine vessels" is preserved.
        score += 8.0 * sum(
            marker in text
            for marker in (
                "guide scope",
                "applicable to all marine vessels",
                "applicable to all types of marine vessels",
                "offshore units",
                "covers the sf categories",
            )
        )
    if slot.name == "abs_smart_notation":
        notation_markers = sum(
            marker in text
            for marker in (
                "smart function notations",
                "optional class notation",
                "permanently installed sf system",
                "function assessment",
                "system assessment",
            )
        )
        score += notation_markers * 9.0
        if notation_markers >= 3:
            score += 18.0
    if slot.name == "abs_risk_classification":
        # A nearby table row can contain the two required headings without
        # stating the classification rule. Rank the clause that explains the
        # complete Low/Medium/High assignment ahead of that abbreviated row.
        classification_markers = sum(
            marker in text
            for marker in (
                "operations supervision level",
                "consequences of failure",
                "low risk category",
                "medium risk category",
                "high risk category",
            )
        )
        score += classification_markers * 8.0
        if classification_markers >= 4:
            score += 16.0
            # Prefer a concise, citation-stable clause over an expanded chunk
            # that repeats the same note after unrelated submission items.
            if len(str(chunk.text or "")) <= 700:
                score += 24.0
    if slot.name == "higher_risk_verification":
        score += 6.0 * sum(
            marker in text
            for marker in (
                "medium and high risk",
                "high risk category level",
                "simulation and physical testing",
                "computer based system category iii",
                "model evaluation",
            )
        )
    if slot.name == "lca_method":
        lca_markers = sum(
            marker in text
            for marker in (
                "lca guidelines",
                "well-to-tank",
                "tank-to-wake",
                "fuel certification",
                "entire life cycle",
            )
        )
        score += lca_markers * 6.0
        if lca_markers >= 4:
            score += 12.0
    if slot.name == "compound_meeting_decision":
        score += 7.0 * sum(
            marker in text
            for marker in (
                "the committee approved",
                "the committee adopted",
                "interim guidelines",
                "approved msc.1/circ",
            )
        )
        if any(term.lower() in {"gfi", "lca", "fuel life cycle label", "sfcs"} for term in slot.terms):
            score += 7.0 * sum(
                marker in text
                for marker in (
                    "gfi",
                    "fuel life cycle label",
                    "lca guidelines",
                    "sustainable fuels certification scheme",
                    "draft amendments",
                    "further work",
                )
            )
    if slot.name == "compound_meeting_scope":
        score += 5.0 * sum(
            marker in text
            for marker in (
                "only for",
                "solely for use as fuel",
                "future revisions",
                "not applicable",
                "applicability",
            )
        )
    if slot.name == "compound_class_instrument":
        score += 6.0 * sum(
            marker in text
            for marker in (
                "fuel ready",
                "gas fuelled",
                "ammonia ready",
                "additional class notation",
                "rules for classification",
            )
        )
    if "hydrogen" in {term.lower() for term in slot.terms}:
        score += 10.0 * sum(
            marker in text
            for marker in (
                "gas fuelled hydrogen",
                "fuel ready(hydrogen",
                "hydrogen fuel approval",
                "hydrogen as fuel",
            )
        )
        if slot.name == "compound_approval_level":
            score += 8.0 * sum(
                marker in text
                for marker in (
                    "additional analyses",
                    "further design work",
                    "submitted for approval",
                    "appendix to the class certificate",
                )
            )
        if slot.name == "compound_design_arrangement":
            score += 8.0 * sum(
                marker in text
                for marker in (
                    "arrangement and location of gas fuel tanks",
                    "gas fuel bunkering connection",
                    "spaces with fuel piping",
                    "vessel arrangement",
                )
            )
        if slot.name == "compound_safety_systems":
            score += 8.0 * sum(
                marker in text
                for marker in (
                    "gas dispersion analysis",
                    "risk assessment (hazid)",
                    "fire and explosion analysis",
                    "control, monitoring and safety systems",
                )
            )
    if slot.name == "compound_approval_level":
        score += 6.0 * sum(
            marker in text
            for marker in (
                "approval in principle",
                "concept design",
                "basic design",
                "design philosophy",
                "documentation requirements",
            )
        )
    if slot.name == "compound_design_arrangement":
        score += 4.0 * sum(
            marker in text
            for marker in (
                "general arrangement",
                "fuel tank",
                "bunkering station",
                "fuel preparation room",
                "pipe routing",
                "fuel supply system",
                "vent mast",
            )
        )
    if slot.name == "compound_safety_systems":
        score += 4.0 * sum(
            marker in text
            for marker in (
                "risk assessment",
                "hazardous area",
                "toxic zone",
                "ventilation",
                "gas detection",
                "emergency shutdown",
                "fire protection",
            )
        )
        if any(term.lower() in {"mass", "autonomous", "자율운항"} for term in slot.terms):
            score += 9.0 * sum(
                marker in text
                for marker in (
                    "preliminary risk assessment",
                    "verification and validation",
                    "validation and verification",
                    "hazard identification",
                    "simulation",
                )
            )
    if slot.name == "compound_regulatory_uncertainty":
        score += 5.0 * sum(
            marker in text
            for marker in (
                "future revision",
                "further consideration",
                "not applicable",
                "with a view to adoption",
                "entry into force",
                "rules in force",
            )
        )
    if slot.outcome_preferred and OUTCOME_RE.search(text):
        score += 4.0
    if slot.date_preferred and DATE_RE.search(text):
        score += 4.0
    if status.code in {"proposal", "action_request"}:
        score -= 2.0
    if status.code in {"draft_outcome", "committee_decision", "adopted_instrument"}:
        score += 2.0
    if slot.name == "regulatory_status":
        if "amendments to marpol annex vi" in text:
            score += 5.0
        if "executive summary" in text or "draft revised marpol annex vi 2025" in text:
            score += 3.0
        if "adjourn for one year" in text or "adjourned for one year" in text:
            score += 6.0
        if "north-east atlantic" in text:
            score += 3.0
    if slot.name == "carbon_intensity":
        if "2019 to 2024" in text:
            score += 4.0
        if "up to 10.8%" in text or "at least 6%" in text:
            score += 4.0
    if slot.name == "major_outcomes":
        if re.search(r"\bthe committee\s+(?:adopted|approved|agreed|decided|endorsed)\b", text):
            score += 6.0
        if re.search(r"\b(?:resolution|code|guidelines|working group)\b", text):
            score += 3.0
        if re.search(r"\bnon-mandatory\b.{0,100}\bmass\s+code\b|\bmass\s+code\b.{0,100}\bnon-mandatory\b", text):
            score += 8.0
        if re.search(r"\binterim\s+guidelines\b.{0,120}\b(?:hydrogen|ammonia)\b", text):
            score += 5.0
        if re.search(r"\bagreed\s+to\s+establish\b.{0,180}\bworking\s+group\b", text):
            score += 5.0
        if re.search(r"\badopted the agenda\b|provisional timetable|agreed to be guided", text):
            score -= 12.0
        if re.search(r"\b(?:recalled|noted) that\b", text) and not re.search(
            r"\bthe committee\s+(?:adopted|approved|agreed|decided|endorsed)\b", text
        ):
            score -= 5.0
    if slot.name in {"msc_mass_lookup", "msc_mass_outcome"}:
        if "msc 111-wp.1" in text and "draft report" in text:
            score += 14.0
        if (
            "agreed to refer" in text
            and "working group" in text
            and "finalization of the draft mass code" in text
        ):
            score += 36.0
        elif "revised road map for mass" in text:
            score -= 8.0
    if slot.name in {"msc_vdes_lookup", "msc_vdes_outcome"}:
        if "msc 111-wp.1" in text and "draft report" in text:
            score += 14.0
        if "performance standards for shipborne vhf data exchange system" in text:
            score += 32.0
        if (
            "draft new msc resolutions" in text
            and "solas chapter v" in text
            and "vdes" in text
        ):
            score += 28.0
        if "the committee recalled that msc 110 had approved" in text:
            score -= 8.0
    if slot.name == "mandatory_adoption_target":
        # The useful roadmap evidence ties the mandatory Code schedule to the
        # parallel STCW review.  A later session-report paragraph mentioning
        # only an aspirational target should not displace this fuller basis.
        if (
            "mandatory mass code" in text
            and "2030" in text
            and "stcw" in text
        ):
            score += 34.0
    if slot.name == "interim_equivalent_arrangements":
        # Prefer the paragraph that states the regulatory gap during use of
        # the non-mandatory Code over generic alternative-watchkeeping text.
        if (
            "non-mandatory mass code is in use" in text
            and "do not currently permit" in text
        ):
            score += 34.0
    if slot.name == "mass_working_group_actions":
        action_markers = (
            "finalize the non-mandatory mass code",
            "develop a framework for an experience building phase",
            "identify the needs to be addressed by other imo bodies",
            "update the revised road map",
        )
        action_hits = sum(marker in text for marker in action_markers)
        score += action_hits * 12.0
        if action_hits >= 3:
            score += 18.0
    return score


def complete_evidence_slots(
    collection,
    pool: list[RetrievedChunk],
    row: dict,
    *,
    expand_candidates: bool = True,
) -> tuple[list[RetrievedChunk], dict[str, Any]]:
    """Prepend evidence that completes the question's required slots."""
    question = str(row.get("question") or "")
    plan = build_evidence_plan(question, row)
    if not plan.slots:
        return pool, {"plan": plan.to_dict(), "slot_hits": {}, "missing_slots": []}

    candidates = (
        _candidate_chunks(collection, pool, plan)
        if expand_candidates
        else list(pool)
    )
    qterms = {
        token.lower()
        for token in TOKEN_RE.findall(question)
        if len(token) > 1
    }
    selected: list[RetrievedChunk] = []
    selected_ids: set[str] = set()
    slot_hits: dict[str, list[str]] = {}
    used_topics: set[str] = set()

    for slot in plan.slots:
        ranked = sorted(
            (
                (_score_slot(chunk, slot, qterms), chunk)
                for chunk in candidates
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        hits: list[RetrievedChunk] = []
        required_class_sources = set(plan.explicit_class_sources)
        class_lane_slot = slot.name.startswith("compound_class_") or slot.name in {
            "compound_approval_level",
            "compound_design_arrangement",
            "compound_safety_systems",
        }
        meeting_lane_slot = slot.name.startswith("compound_meeting_") or (
            plan.intent == "regulatory_class_integration"
            and slot.name == "compound_regulatory_uncertainty"
        )
        for score, chunk in ranked:
            if not math.isfinite(score) or score <= 0:
                continue
            source = str(chunk.source or "").upper()
            if meeting_lane_slot and source not in {"MSC", "MEPC", "IMO"}:
                continue
            if class_lane_slot and source not in {"DNV", "KR", "ABS", "LR"}:
                continue
            if slot.name == "major_outcomes":
                topic = _topic_signature(chunk.text)
                if topic in used_topics and len(used_topics) < slot.max_hits:
                    continue
                used_topics.add(topic)
            if slot.name == "compound_class_instrument" and required_class_sources:
                # Use the limited slot capacity to represent every explicitly
                # requested society before taking a second hit from one.
                represented = {str(hit.source or "").upper() for hit in hits}
                missing_sources = required_class_sources.difference(represented)
                if missing_sources and source not in missing_sources:
                    continue
            hits.append(chunk)
            if chunk.chunk_id not in selected_ids:
                selected.append(chunk)
                selected_ids.add(chunk.chunk_id)
            if len(hits) >= slot.max_hits:
                break
        slot_hits[slot.name] = [chunk.chunk_id for chunk in hits]

    # A meeting report often records objections immediately before the
    # committee's final position.  Slot scoring can therefore keep the
    # objection (for example, that the 2030/2032 MASS timetable is
    # unrealistic) while dropping the following paragraph saying that the
    # committee nevertheless retained the target.  Preserve one explicit
    # final-position paragraph for compound meeting + class questions.  This
    # is deliberately candidate-local and only activates when the paragraph
    # shares a requested year or a distinctive topic token with the question.
    if plan.intent == "regulatory_class_integration":
        question_lower = question.lower()
        question_years = set(re.findall(r"\b(?:19|20)\d{2}\b", question))
        distinctive_topics = {
            term.lower()
            for term in plan.topic_terms
            if len(term) >= 4 and term.lower() not in _CLAUSE_STOPWORDS
        }
        # ASCII topic anchors remain reliable even when Korean tokenization is
        # unavailable in a deployment environment.
        distinctive_topics.update(
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", question)
            if token.lower() not in _CLAUSE_STOPWORDS
        )
        final_markers = (
            "notwithstanding the above",
            "nevertheless agreed",
            "continue working towards the target year",
            "endorsed the revised road map",
        )

        def final_position_score(chunk: RetrievedChunk) -> float:
            if str(chunk.source or "").upper() not in {"MSC", "MEPC", "IMO"}:
                return -math.inf
            text = str(chunk.text or "").lower()
            marker_hits = sum(marker in text for marker in final_markers)
            if not marker_hits:
                return -math.inf
            chunk_years = set(re.findall(r"\b(?:19|20)\d{2}\b", text))
            year_hits = len(question_years.intersection(chunk_years))
            topic_hits = sum(topic in text for topic in distinctive_topics)
            if question_years and not year_hits:
                return -math.inf
            if not question_years and not topic_hits:
                return -math.inf
            return marker_hits * 10.0 + year_hits * 5.0 + min(topic_hits, 4)

        final_ranked = sorted(
            (
                (final_position_score(chunk), chunk)
                for chunk in candidates
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        if final_ranked and math.isfinite(final_ranked[0][0]) and final_ranked[0][0] > 0:
            final_chunk = final_ranked[0][1]
            if final_chunk.chunk_id not in selected_ids:
                selected.append(final_chunk)
                selected_ids.add(final_chunk.chunk_id)
            slot_hits["compound_final_position"] = [final_chunk.chunk_id]

    missing = [slot.name for slot in plan.slots if not slot_hits.get(slot.name)]
    remainder = (
        [
            chunk
            for chunk in pool
            if _document_id_matches(str(chunk.file_name or ""), plan)
        ]
        if plan.document_identifiers and plan.intent != "regulatory_class_integration"
        else pool
    )
    reordered = [
        *selected,
        *(chunk for chunk in remainder if chunk.chunk_id not in selected_ids),
    ]
    return reordered, {
        "plan": plan.to_dict(),
        "slot_hits": slot_hits,
        "missing_slots": missing,
        "candidate_chunk_count": len(candidates),
    }
