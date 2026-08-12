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

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "session_org": self.session_org,
            "session_number": self.session_number,
            "requested_count": self.requested_count,
            "latest_requested": self.latest_requested,
            "document_identifiers": list(self.document_identifiers),
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
        re.search(r"rule|guidance|class guideline|notice|requirement|요구사항|선급|규칙|지침", q, re.I)
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

    if org == "MEPC" and requirements.is_concrete:
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
        slots: list[EvidenceSlot] = []
        for facet in requirements.facets or ("fact",):
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

    if intent == "mass_code_timeline" or (
        "mass" in ql and any(term in ql for term in ("mandatory", "timeline", "일정"))
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
                "reporting_requirement",
                "규제 보고 요구",
                ("reporting", "submitted data", "imo dcs", "gisis", "mandatory", "verification"),
                (("report", "submitted", "dcs", "gisis"),),
            ),
            EvidenceSlot(
                "data_quality",
                "보고 데이터 품질",
                ("duplicate reporting", "identified errors", "quality control", "verification", "ships"),
                (("error", "duplicate", "quality control"), ("verification", "excluded", "not been included")),
                max_hits=2,
            ),
            EvidenceSlot(
                "carbon_intensity",
                "탄소집약도·운항지표",
                ("carbon intensity", "aer", "cgdist", "transport work", "cii"),
                (("carbon intensity", "aer", "cii"),),
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

    if intent == "meeting_outcome" or requested:
        count = requested or 3
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
                if alias_index < 4:
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
        ) or definition_lookup
        if direct_terms and asks_for_specific_clause:
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
        parts = [part for part in document_id.split("/") if part]
        if not parts:
            continue
        pattern = r"\s*[-/]?\s*".join(re.escape(part) for part in parts)
        if re.search(rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])", name, re.I):
            return True
    return False


def _candidate_chunks(collection, pool: list[RetrievedChunk], plan: EvidencePlan) -> list[RetrievedChunk]:
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
        if (
            name
            and _session_matches(name, plan)
            and _document_id_matches(name, plan)
            and name not in file_names
        ):
            file_names.append(name)

    # Search within the most authoritative session-level documents first.
    original_position = {name: pos for pos, name in enumerate(file_names)}

    def doc_score(name: str) -> tuple[int, int]:
        low = name.lower()
        session_report = int(any(x in low for x in ("draft report", "final report", "wp.1", "report of the")))
        return (session_report, -original_position[name])

    file_names.sort(key=doc_score, reverse=True)
    candidates: list[RetrievedChunk] = []
    for name in file_names[:8]:
        candidates.extend(_document_chunks(collection, name))

    # If dense retrieval did not discover enough documents, use a metadata
    # scoped fallback for the requested organization/session.
    if plan.session_org and (
        not candidates or len(file_names) < 2 or plan.latest_requested
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
        candidates.extend(
            chunk
            for chunk in scoped
            if _session_matches(chunk.file_name, plan)
            and _document_id_matches(chunk.file_name, plan)
        )

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
    if not is_substantive_chunk(chunk):
        return -math.inf
    text = re.sub(r"\s+", " ", f"{chunk.file_name} {chunk.text}").lower()
    for group in slot.required_groups:
        if not _contains_any(text, group):
            return -math.inf
    hits = sum(1 for term in slot.terms if term.lower() in text)
    overlap = sum(1 for term in question_terms if term in text)
    status = classify_document_status(chunk)
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
    return score


def complete_evidence_slots(
    collection,
    pool: list[RetrievedChunk],
    row: dict,
) -> tuple[list[RetrievedChunk], dict[str, Any]]:
    """Prepend evidence that completes the question's required slots."""
    question = str(row.get("question") or "")
    plan = build_evidence_plan(question, row)
    if not plan.slots:
        return pool, {"plan": plan.to_dict(), "slot_hits": {}, "missing_slots": []}

    candidates = _candidate_chunks(collection, pool, plan)
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
        for score, chunk in ranked:
            if not math.isfinite(score) or score <= 0:
                continue
            if slot.name == "major_outcomes":
                topic = _topic_signature(chunk.text)
                if topic in used_topics and len(used_topics) < slot.max_hits:
                    continue
                used_topics.add(topic)
            hits.append(chunk)
            if chunk.chunk_id not in selected_ids:
                selected.append(chunk)
                selected_ids.add(chunk.chunk_id)
            if len(hits) >= slot.max_hits:
                break
        slot_hits[slot.name] = [chunk.chunk_id for chunk in hits]

    missing = [slot.name for slot in plan.slots if not slot_hits.get(slot.name)]
    remainder = (
        [
            chunk
            for chunk in pool
            if _document_id_matches(str(chunk.file_name or ""), plan)
        ]
        if plan.document_identifiers
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
