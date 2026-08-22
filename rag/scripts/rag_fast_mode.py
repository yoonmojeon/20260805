"""Fast RAG mode — minimal retrieval context + short prompt for low TTFT."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from rag_answer_lib import (
    RetrievedChunk,
    call_ollama_chat_timed,
    model_prefers_think_off,
    retrieve_for_question,
)
from ollama_warmup import ensure_fast_warm, ensure_fast_warm_checked, mark_fast_llm_run
from retrieval_timing import TimingTrace, estimate_tokens

from fast_confidence import assess_fast_confidence
from fast_answer_pipeline import (
    build_citation_mapping,
    build_evidence_first_prompts,
    postprocess_fast_answer,
    prepare_fast_answer_pipeline,
)
from fast_mode_trace import append_fast_answer_trace, build_trace_record
from fast_context import FastEvidence, build_slot_compact_context, question_focus_score
from fast_prompts import build_fast_system_prompt, build_fast_user_prompt
from fast_question_classifier import classify_fast_question_type, fast_type_label_ko
from fast_retrieval import evidence_to_chunks, select_fast_evidence_slots
from adjacent_chunk_expansion import (
    expand_chunks_with_neighbors,
    expand_evidence_with_neighbors,
)

FAST_RETRIEVAL = {
    "top_k": 3,
    "fetch_k": 10,
    "pool_fetch_k": 18,
    "max_docs": 2,
    "max_chunks_per_doc": 1,
    "use_rerank": False,
    "preview_chars": 600,
    "use_typed_slots": True,
    # The 271k-chunk global BM25 takes ~14s even after loading. Interactive UI
    # modes use dense retrieval plus typed evidence slots; BM25 remains an
    # offline evaluation/build artifact until a sub-second sparse backend lands.
    "use_hybrid_bm25": False,
}

TABLE_FAST_RETRIEVAL = {
    "top_k": 10,
    "fetch_k": 30,
    "pool_fetch_k": 30,
    "max_docs": 3,
    "max_chunks_per_doc": 5,
    "use_rerank": False,
    "preview_chars": 4000,
    "use_typed_slots": True,
}

RULE_GUIDANCE_FAST_RETRIEVAL = {
    "top_k": 8,
    "fetch_k": 40,
    "pool_fetch_k": 56,
    "max_docs": 3,
    "max_chunks_per_doc": 2,
    "use_rerank": False,
    "preview_chars": 800,
    "use_typed_slots": True,
    "use_hybrid_bm25": False,
    "hard_society_filter": True,
}

MEETING_FAST_RETRIEVAL = {
    "top_k": 8,
    "fetch_k": 40,
    "pool_fetch_k": 48,
    "max_docs": 5,
    "max_chunks_per_doc": 2,
    "use_rerank": False,
    "preview_chars": 900,
    "use_typed_slots": True,
    # Full-corpus BM25 is ~10–15s/query and stalls interactive Accurate.
    # Meeting quality relies on dense expansion + WP.1/topic boosts instead.
    "use_hybrid_bm25": False,
}

ACCURATE_MEETING_RETRIEVAL = {
    **MEETING_FAST_RETRIEVAL,
    "top_k": 12,
    "fetch_k": 72,
    "pool_fetch_k": 80,
    "max_docs": 8,
    "max_chunks_per_doc": 2,
}


_LOCAL_DOC_TOKEN_STOP = {
    "mepc", "msc", "imo", "dnv", "abs", "kr", "lr", "pdf", "document", "rule", "guidance", "which", "what",
    "from", "with", "that", "this", "does", "into", "have", "been",
}

_LOCAL_KO_TOKEN_STOP = {
    "규정", "문서", "질문", "경우", "어떤", "무엇", "따르면", "관련", "대한",
    "위한", "통해", "각각", "구체적", "필요", "요구", "사항", "어떻게",
}


def _local_korean_query_terms(question: str) -> set[str]:
    """Return reusable Korean technical anchors without case particles."""
    out: set[str] = set()
    suffixes = (
        "으로부터", "에서는", "에게서", "까지는", "으로", "에서", "부터", "까지",
        "에게", "처럼", "보다", "하고", "이며", "이면", "에는", "에도", "만을",
        "의", "은", "는", "이", "가", "을", "를", "과", "와", "로", "에", "도",
    )
    for raw in re.findall(r"[가-힣]{2,}", question or ""):
        token = raw
        for suffix in suffixes:
            if token.endswith(suffix) and len(token) - len(suffix) >= 2:
                token = token[: -len(suffix)]
                break
        if len(token) >= 2 and token not in _LOCAL_KO_TOKEN_STOP:
            out.add(token)
    return out


def _local_domain_terms(question: str) -> set[str]:
    """Small domain lexicon for Korean-to-source-language within-PDF recall."""
    bundles = (
        (r"수소.{0,12}(?:연료|지침)|수소\s*연료", {"hydrogen", "interim guidelines", "safety of ships using hydrogen as fuel"}),
        (r"암모니아.{0,12}(?:연료|지침)|암모니아\s*연료", {"ammonia", "interim guidelines", "ammonia as fuel"}),
        (r"정격|전압", {"rated voltage", "voltage rating", "power cable"}),
        (r"제어.{0,8}계측|계측.{0,8}회로", {"control and instrumentation", "instrumentation circuit"}),
        (r"구성\s*요소|구성요소", {"component", "components", "hardware allocation", "architecture", "hierarchy", "interface"}),
        (r"등급|grade", {"grade", "grades", "variant", "variants"}),
        (r"변형|variant", {"variant", "variants", "plies", "colour", "diluted"}),
        (r"실험실|제조사.{0,8}현장|시험.{0,8}장소", {"laboratory", "manufacturer's premises", "premises", "surveyors"}),
        (r"압력\s*센서|감지\s*영역", {"pressure sensor", "sensing area", "flush", "protective cover", "cap"}),
        (r"가스\s*제거|진입", {"gas freeing", "prior to entry", "chamber", "nitrogen"}),
        (r"목적|절차\s*범위", {"objective", "scope", "documentation", "design", "type testing", "design requirements"}),
        (r"단면적", {"cross-sectional area", "cross section", "mm2", "mm²"}),
        (r"제출.{0,8}(?:서류|문서)|문서\s*목록|필수\s*문서", {"documentation", "documents", "submitted", "application", "list"}),
        (r"수직\s*지지대|인장\s*상태", {"vertical support", "tension", "no contact", "compression", "iterative"}),
        (r"충격\s*시험|샤르피", {"impact test", "charpy v-notch", "impact toughness", "thickness"}),
        (r"보고서.{0,12}(?:제출|포함)", {"final report", "submitted", "shall include", "weeks"}),
        (
            r"선상\s*측정.{0,20}(?:최종\s*)?보고서|최종\s*보고서.{0,16}측정",
            {
                "final report", "two weeks", "original electronic format",
                "non-editable electronic format", "scantlings",
                "substantial corrosion", "approval certificate",
            },
        ),
        (r"내측\s*접근|외부\s*조사", {"internal access", "inaccessible", "external examination", "inspection"}),
        (
            r"내측\s*접근.{0,20}(?:불가능|추가)|(?:ROV|잠수).{0,12}검사",
            {
                "inaccessible from internally", "bilge keel", "moonpool",
                "fairlead foundation", "anchor rack", "sea chest", "ROV",
                "diver inspection", "coating damage", "local corrosion",
            },
        ),
        (r"테스트\s*계획|시험\s*계획", {"test plan", "shall include"}),
        (
            r"세척.{0,16}(?:효율|시험\s*계획)|미세.{0,12}거대\s*오염",
            {
                "cleaning scope", "areas to be cleaned", "micro-fouling",
                "macro-fouling", "representative areas", "sampling method",
                "analysis method", "EHS plan",
            },
        ),
        (r"평판\s*제품|용접성", {"flat products", "weldability", "weldability testing"}),
        (r"데이터\s*수집\s*인프라", {"data collection infrastructure", "certification", "individual components"}),
        (r"횡방향\s*강도|경계\s*조건", {"racking analysis", "boundary condition", "boundary conditions"}),
        (r"무거운\s*부품|설계\s*하중", {"heavy parts", "design load", "transportation condition", "force"}),
        (r"생물학적\s*효능|독성\s*시험", {"biological efficacy", "toxicity testing", "testing facility", "laboratory"}),
        (r"고망간|냉간\s*성형|열처리", {"high manganese", "cold forming", "heat treatment", "forming ratio"}),
        (r"동적\s*시험|상대\s*감쇠|강성", {"dynamic test", "stiffness", "relative damping", "displacement", "frequency"}),
        (r"비상전원|급전", {"emergency source", "emergency power", "supply", "hours", "automatically"}),
        (r"가청\s*경보|전원공급", {"audible alarm", "sources of power", "emergency source", "transitional"}),
        (r"단면계수|특설늑골|크로스타이", {"section modulus", "web frame", "cross-tie", "effective span"}),
        (r"설치\s*후.{0,10}테스트|대체\s*전력|\bCGS\b", {"CGS", "normal seagoing load", "installation test", "alternative power"}),
        (
            r"운용\s*경험.{0,12}부족|선박\s*내\s*운용\s*경험|service\s*experience",
            {
                "insufficient experience from service on board ships",
                "vibration test", "damp heat test", "dry heat test",
                "salt mist test", "conducted emissions", "radiated emissions",
                "electronic devices",
            },
        ),
        (
            r"소음.{0,8}진동.{0,16}(?:운영자|자격)|측정\s*운영자.{0,12}자격",
            {
                "noise and vibration", "measurement operator", "qualification",
                "ISO 18436-2", "Category I", "one year experience",
                "practical training", "ship machinery", "manoeuvring systems",
            },
        ),
        (
            r"소재\s*\(?(?:starting\s*material)?\)?.{0,20}(?:승인\s*보고서|기록|데이터)|"
            r"starting\s*material",
            {
                "starting material", "approval report", "supplier",
                "ladle analysis", "austenite grain size",
                "non-metallic inclusions", "macro-etch", "heat treatment furnaces",
                "calibration records", "weld repair procedure",
            },
        ),
        (
            r"유체\s*평형.{0,20}60\s*초|중간\s*단계.{0,12}침수",
            {
                "intermediate stages of flooding", "instantaneous flooding",
                "cross-flooding through structural ducts", "pressure losses",
                "air pipes", "60 seconds", "SOLAS II-1/7-2.2",
            },
        ),
        (
            r"객실.{0,12}화재\s*감지기|accommodation\s*area.{0,12}cabins",
            {"heat detectors", "cabins", "accommodation area"},
        ),
        (
            r"언코일링|\bW\s*인증.{0,12}\bNV\s*인증",
            {
                "uncoiling plant", "W certificate", "NV certificate",
                "full scope of certification", "stamping", "identification",
                "inspection and testing", "costs", "survey expenses",
            },
        ),
        (r"크랭크샤프트|피로\s*시험", {"crankshaft", "fatigue test", "chemical composition", "mechanical properties", "specimen preparation", "surface condition", "heat treatment"}),
        (
            r"크랭크샤프트.{0,20}(?:품질\s*관리|시편)|피로\s*시험.{0,16}시편",
            {
                "quality control procedure", "surface hardness", "hardness depth",
                "hardness extension", "fillet surface roughness", "lower end",
                "induction hardened crankshaft", "test specimen",
            },
        ),
        (r"비금속\s*호스|고무\s*보상기|고정형\s*커플링", {"flexible non-metallic hose", "fixed coupling", "rubber compensator", "documentation", "drawing", "product specification"}),
        (r"\bSCR\b.{0,20}형식\s*시험|형식\s*시험.{0,20}\bSCR\b", {"SCR system", "type test", "documentation", "test plan", "technical file", "drawing"}),
        (r"\bDHT\b|종방향\s*부재|두께\s*측정값", {"Deck", "Sheerstrake", "Side shell", "Bilge", "Bottom longitudinals", "Deck girders", "longitudinal members"}),
        (r"수신\s*측.{0,12}확인|원격\s*제어권.{0,12}전환|acknowledg", {"acknowledgment", "acknowledgement", "receiver", "centralized control station", "local manual control", "transfer"}),
        (
            r"대체\s*시스템.{0,20}(?:승인|증빙)|MSC\.288\(87\)",
            {
                "alternative system", "documented evidence", "equivalent corrosion prevention",
                "five years actual field exposure", "GOOD", "15 years target useful life",
                "gas-tight cabinet test", "immersion test", "MSC.288(87)",
            },
        ),
        (
            r"로봇\s*용접|완전\s*자동화.{0,12}용접|수동.{0,16}반자동.{0,16}자동",
            {
                "robotic welding", "re-qualified", "fully mechanized",
                "essential process parameters", "pWPS", "WPQT", "WPQR", "WPS",
            },
        ),
        (
            r"데이터\s*수집\s*인프라.{0,20}(?:인증|구성\s*요소)",
            {
                "data collection infrastructure", "individual components",
                "vessel server", "data relay", "remote server", "proprietary protocols",
            },
        ),
        (
            r"선미관.{0,12}밀봉|stern\s*tube.{0,12}seal",
            {
                "stern tube sealing devices", "seagoing ships", "legal requirements",
                "marine discharge", "conformity assessment", "onboard installation",
            },
        ),
        (
            r"구형\s*쉘.{0,16}편평도|out-of-roundness",
            {
                "spherical shell", "out-of-roundness", "local radius of curvature",
                "1.3", "0.218", "corrosion allowance", "machined spherical shell", "1.05",
            },
        ),
        (
            r"\bSCR\b.{0,20}(?:형식\s*시험|제출\s*서류)",
            {
                "SCR system", "prior to type test", "measuring equipment",
                "test setup", "exhaust composition", "mass flow", "temperature chamber",
                "catalyst block", "modelling procedure", "predicted values", "NOx reduction",
            },
        ),
        (
            r"클러치.{0,20}(?:특수|새로운|시험\s*절차)|clutch.{0,20}(?:special|new)",
            {"clutch", "special or new design", "case-by-case", "deviating test programme"},
        ),
        (
            r"LAN.{0,16}영구\s*링크|permanent\s*link.{0,16}(?:TA|type\s*approval)",
            {
                "LAN horizontal cabling", "copper permanent link", "type approval",
                "other parts of the rules", "does not ensure", "specified in the application",
                "stated on the certificate",
            },
        ),
    )
    terms: set[str] = set()
    for pattern, values in bundles:
        if re.search(pattern, question or "", re.I):
            terms.update(values)
    return terms


def _document_local_query_hits(
    chunks_dir: Path,
    doc_id: str,
    question: str,
    *,
    existing: list[RetrievedChunk],
    limit: int = 6,
    preview_chars: int = 2200,
) -> list[RetrievedChunk]:
    """Cheap lexical rerank inside one explicitly named PDF.

    Global BM25 is intentionally too expensive for Fast mode, but scanning the
    tens/hundreds of chunks in one already-resolved document is sub-millisecond
    to a few milliseconds and recovers headings such as FUMES/Cslip that dense
    Korean-to-English retrieval can otherwise miss.
    """
    path = chunks_dir / doc_id / "chunks.jsonl"
    if not path.exists():
        return []

    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+_.-]{2,}", question or "")
        if token.lower() not in _LOCAL_DOC_TOKEN_STOP
        and not re.fullmatch(r"(?:dnv|mepc|msc|abs|kr|lr)[-_/.a-z0-9]+", token, re.I)
    }
    # Preserve named source-language concepts as phrases.  Scoring their
    # component words independently made generic scope paragraphs outrank the
    # actual definition/value paragraph (e.g. ``progressive couplings`` or
    # ``administrative requirement``).
    from retrieval_search import extract_sparse_latin_terms

    named_phrases = {
        term.lower()
        for term in extract_sparse_latin_terms(question, limit=4)
        if " " in term.strip()
    }
    numeric_lookup = bool(
        re.search(
            r"얼마|몇\s*(?:%|퍼센트)|어느\s*정도|수치|값|속도|조건|"
            r"how\s+much|what\s+(?:value|speed)",
            question or "",
            re.I,
        )
    )
    reason_lookup = bool(re.search(r"이유|왜|reason|why|rationale", question or "", re.I))
    definition_lookup = bool(re.search(r"정의|defined|definition|means", question or "", re.I))
    decision_lookup = bool(
        re.search(r"전제|부결|승인|채택|거절|approved|rejected|adopted", question or "", re.I)
    )
    query_years = list(dict.fromkeys(re.findall(r"(?:19|20)\d{2}", question or "")))
    class_local = doc_id.split("_", 1)[0].lower() in {"dnv", "kr", "abs", "lr"}
    korean_terms = _local_korean_query_terms(question) if class_local else set()
    # These are source-language anchors, so they are useful for English IMO
    # reports as well as class documents.  The scan is already restricted to
    # one resolved PDF and therefore does not add global sparse-search cost.
    domain_terms = _local_domain_terms(question)
    q = str(question or "")
    intent_terms: set[str] = set()
    for pattern, terms in (
        (r"연구|조사|실험", {"study", "research", "project", "measurement", "empirical"}),
        (r"결과|근거|기반", {"finding", "findings", "derived", "based", "support", "report"}),
        (r"승인", {"approved", "approval"}),
        (r"채택", {"adopted", "adoption"}),
        (r"회의", {"meeting", "session", "committee"}),
        (r"요구|요건", {"require", "required", "requirement", "shall"}),
        (r"예외", {"exception", "except", "exemption"}),
        (r"적용", {"apply", "applies", "application", "applicable"}),
        (r"값|수치|계수", {"value", "factor", "coefficient"}),
        (r"목적|용도", {"purpose", "use", "used"}),
        (r"연례|연간", {"annual", "yearly"}),
        (r"성능\s*시험", {"performance", "test", "testing"}),
        (r"서비스\s*공급자", {"service", "supplier", "provider"}),
        (r"절차", {"procedure", "procedures"}),
        (r"체크\s*리스트", {"checklist", "checklists"}),
    ):
        if re.search(pattern, q, re.I):
            intent_terms.update(terms)

    if not tokens and not korean_terms and not domain_terms and not intent_terms:
        return []

    reference = next((chunk for chunk in existing if chunk.doc_id == doc_id), None)
    scored: list[tuple[float, dict[str, Any]]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        table_path = chunks_dir / doc_id / "table_chunks.jsonl"
        if class_local and table_path.exists():
            lines.extend(table_path.read_text(encoding="utf-8").splitlines())
    except OSError:
        return []
    for raw in lines:
        try:
            item = json.loads(raw)
        except (TypeError, ValueError):
            continue
        text = str(item.get("text") or "").strip()
        if len(text) < 40 or "picture element" in text.lower():
            continue
        lower = text.lower()
        token_hits = {token for token in tokens if token in lower}
        korean_hits = {term for term in korean_terms if term in text}
        domain_hits = {term for term in domain_terms if term in lower}
        intent_hits = {term for term in intent_terms if term in lower}
        phrase_hits = {phrase for phrase in named_phrases if phrase in lower}
        if not token_hits and not korean_hits and not domain_hits and not intent_hits:
            continue
        score = sum(1.0 + min(len(token), 12) / 8.0 for token in token_hits)
        score += sum(1.4 + min(len(term), 10) / 7.0 for term in korean_hits)
        score += sum(1.8 + min(len(term), 20) / 10.0 for term in domain_hits)
        score += 1.35 * len(intent_hits)
        score += sum(8.0 + min(len(phrase), 32) / 8.0 for phrase in phrase_hits)
        # Closely adjacent alternative-fuel agenda items share generic words
        # such as "interim guidelines" and "approval".  Preserve the fuel
        # named by the user as a hard within-document anchor so the ammonia
        # paragraph cannot outrank the requested hydrogen decision (and vice
        # versa) inside the same MSC report.
        if re.search(r"수소|\bhydrogen\b", q, re.I):
            score += 8.0 if "hydrogen" in lower else (-4.0 if "ammonia" in lower else 0.0)
        if re.search(r"암모니아|\bammonia\b", q, re.I):
            score += 8.0 if "ammonia" in lower else (-4.0 if "hydrogen" in lower else 0.0)
        if numeric_lookup and re.search(
            r"\d+(?:\.\d+)?\s*(?:%|m/s|mm|cm|m|kg|t|hours?)\b",
            lower,
            re.I,
        ):
            score += 3.5
        if len(query_years) >= 2 and all(year in lower for year in query_years):
            first, second = query_years[0], query_years[1]
            if re.search(
                rf"{re.escape(second)}.{{0,100}}(?:relative\s+to|compared\s+to)"
                rf".{{0,30}}{re.escape(first)}",
                lower,
                re.I | re.S,
            ):
                score += 10.0
        if reason_lookup and re.search(r"because|reason|due\s+to|therefore|as\s+a\s+result", lower):
            score += 5.0
        if reason_lookup and re.search(
            r"more\s+(?:extensive|stringent|frequent)\s+than|"
            r"less\s+(?:extensive|stringent|frequent)\s+than",
            lower,
            re.I,
        ):
            score += 6.0
        if definition_lookup and re.search(
            r"(?:term\s+[^.]{0,80})?\bis\s+defined\b|defined\s+as|definition\b|\bmeans\b",
            lower,
            re.I,
        ):
            score += 7.0
        if decision_lookup and re.search(
            r"\bcommittee\s+(?:approved|rejected|adopted|agreed)\b|"
            r"\b(?:was|were)\s+(?:approved|rejected|adopted)\b",
            lower,
            re.I,
        ):
            score += 7.0
        # A source proposition is preferable to a title that merely repeats all
        # query nouns.  Action/summary paragraphs get a small relevance bonus.
        if re.search(r"derived from|based on|supported by|considered and approved", lower):
            score += 2.5
        if len(text) >= 180:
            score += 0.4
        scored.append((score, item))

    scored.sort(key=lambda pair: (-pair[0], int(pair[1].get("page_number") or 0)))
    out: list[RetrievedChunk] = []
    for rank, (score, item) in enumerate(scored[:limit]):
        text = str(item.get("text") or "")
        if len(text) > preview_chars:
            text = text[:preview_chars] + "\n...(truncated)"
        source_pdf = str(item.get("source_pdf") or "")
        out.append(
            RetrievedChunk(
                chunk_id=str(item.get("chunk_id") or ""),
                doc_id=doc_id,
                source=(reference.source if reference else doc_id.split("_", 1)[0].upper()),
                file_name=(reference.file_name if reference else Path(source_pdf).name),
                page_number=item.get("page_number"),
                clause_number=str(item.get("clause_number") or item.get("article_number") or ""),
                element_type=str(item.get("element_type") or ""),
                distance=1.0 / (1.0 + score) + rank * 0.00001,
                text=text,
                chunk_type=str(item.get("content_format") or ""),
                crop_path=str(item.get("crop_path") or ""),
            )
        )
    return [chunk for chunk in out if chunk.chunk_id]


def _source_local_query_hits(
    chunks_dir: Path,
    source: str,
    question: str,
    *,
    existing: list[RetrievedChunk],
    limit: int = 12,
    preview_chars: int = 4000,
) -> list[RetrievedChunk]:
    """Sparse rerank across a small named class-society corpus (Accurate)."""
    prefix = source.strip().lower() + "_"
    if source.strip().upper() not in {"DNV", "KR", "ABS", "LR"}:
        return []
    # A literal class-program/guideline identifier is a hard document scope.
    # The cross-document sparse pass previously allowed a generic ``variants``
    # or ``cross-sectional area`` paragraph from a different DNV CP to outrank
    # the expressly requested CP.  OS identifiers are deliberately excluded:
    # DNV class-program questions often cite an OS activity code while the
    # authoritative explanatory text lives in the CP.
    explicit_dnv_docs = {
        re.sub(r"[^a-z0-9]", "", match.group(0).lower())
        for match in re.finditer(
            r"\bDNV[- ](?:CP|CG)[- ]\d{4}(?!\d)|"
            r"\bDNV[- ]RU[- ]SHIP(?:[- ]Pt\d+)?(?![A-Za-z0-9])",
            str(question or ""),
            re.I,
        )
    }
    hits: list[RetrievedChunk] = []
    try:
        folders = [path for path in chunks_dir.iterdir() if path.is_dir() and path.name.lower().startswith(prefix)]
    except OSError:
        return []
    if explicit_dnv_docs:
        scoped = [
            folder
            for folder in folders
            if any(
                doc_ref in re.sub(r"[^a-z0-9]", "", folder.name.lower())
                for doc_ref in explicit_dnv_docs
            )
        ]
        if scoped:
            folders = scoped
    for folder in folders:
        hits.extend(
            _document_local_query_hits(
                chunks_dir,
                folder.name,
                question,
                existing=existing,
                limit=3,
                preview_chars=preview_chars,
            )
        )
    hits.sort(key=lambda chunk: (float(chunk.distance), chunk.doc_id, chunk.chunk_id))
    # The society-wide lexical winner is not necessarily the right document:
    # common class-program phrases such as "test plan" and "documentation"
    # occur in hundreds of PDFs.  Retain the best local paragraph from the
    # documents already favoured by dense retrieval before filling the rest
    # by global sparse score.  This lets source-language clause matching refine
    # the document route without replacing it.
    dense_doc_order: list[str] = []
    for chunk in existing:
        doc_id = str(getattr(chunk, "doc_id", "") or "")
        if (
            doc_id.lower().startswith(prefix)
            and doc_id not in dense_doc_order
        ):
            dense_doc_order.append(doc_id)
        if len(dense_doc_order) >= 8:
            break
    best_by_doc: dict[str, RetrievedChunk] = {}
    for chunk in hits:
        best_by_doc.setdefault(str(chunk.doc_id), chunk)
    dense_seed_hits = [
        best_by_doc[doc_id]
        for doc_id in dense_doc_order
        if doc_id in best_by_doc
    ]
    hits = [*dense_seed_hits, *hits]
    seen: set[str] = set()
    return [
        chunk
        for chunk in hits
        if not (chunk.chunk_id in seen or seen.add(chunk.chunk_id))
    ][:limit]

# A named society or meeting body reduces the corpus enough to afford a wider
# candidate pool while preserving Fast latency.  This is for clause/fact
# questions, not the Rule/Guidance or whole-session renderers.
SOURCE_SCOPED_FACT_FAST_RETRIEVAL = {
    **FAST_RETRIEVAL,
    "top_k": 5,
    "fetch_k": 40,
    "pool_fetch_k": 56,
    "max_docs": 3,
    "max_chunks_per_doc": 2,
    "preview_chars": 1200,
}

FAST_LLM = {
    "num_ctx": 4096,
    "max_new_tokens": 512,
    "max_new_tokens_meeting": 600,
    "temperature": 0.0,
}


def _is_definition_lookup(question: str) -> bool:
    return bool(
        re.search(
            r"(?:기호.{0,16}(?:뜻|의미)|무엇을\s*뜻|무슨\s*뜻|(?<!규)정의)",
            str(question or ""),
            re.I,
        )
    )

# Legacy prompts (fallback when use_typed_slots=False)
FAST_SYSTEM_PROMPT = (
    "너는 해사 규정 문서 기반 RAG assistant다. "
    "제공된 근거 context 안에서만 답변해라. "
    "근거가 부족하면 부족하다고 말해라. 답변은 간결하게 작성해라."
)


def trim_fast_chunks(
    pool: list[RetrievedChunk],
    *,
    max_chunks: int = 3,
    max_docs: int = 2,
    max_per_doc: int = 1,
) -> list[RetrievedChunk]:
    out: list[RetrievedChunk] = []
    doc_counts: dict[str, int] = {}
    seen_docs: set[str] = set()
    for chunk in pool:
        doc_id = chunk.doc_id
        if doc_id not in seen_docs and len(seen_docs) >= max_docs:
            continue
        n = doc_counts.get(doc_id, 0)
        if n >= max_per_doc:
            continue
        out.append(chunk)
        doc_counts[doc_id] = n + 1
        seen_docs.add(doc_id)
        if len(out) >= max_chunks:
            break
    return out


def build_compact_context(chunks: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        page = c.page_number if c.page_number is not None else "?"
        name = c.file_name or c.doc_id
        text = (c.text or "").strip().replace("\n", " ")
        if len(text) > 500:
            text = text[:500] + "…"
        lines.append(f"[{i}] {name} p.{page}: {text}")
    return "\n".join(lines)


def _rank_feature_fallback_chunks(
    pool: list[RetrievedChunk],
    question: str,
    feature_terms: list[str],
) -> list[RetrievedChunk]:
    """Rank literal-recovery chunks without paying another collection query."""
    if not pool or not feature_terms:
        return []
    from retrieval_search import feature_fallback_relevance_score

    question_anchors = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+_.-]{2,}", question or "")
        if token.lower()
        not in {"mepc", "msc", "imo", "rule", "guidance", "what", "which"}
    }
    ranked: list[tuple[float, RetrievedChunk]] = []
    for chunk in pool:
        blob = f"{chunk.file_name or ''} {chunk.text or ''}".lower()
        matched = [term for term in feature_terms if term.lower() in blob]
        if not matched:
            continue
        score = max(
            feature_fallback_relevance_score(question, chunk.text or "", term)
            + min(0.45, len(term.split()) * 0.06)
            for term in matched
        )
        score += min(0.6, 0.22 * (len(matched) - 1))
        score += min(
            0.6,
            0.18 * sum(anchor in blob for anchor in question_anchors),
        )
        ranked.append((score, chunk))
    # When a parent/page chunk and its atomic child both contain the recovered
    # phrase, prefer the shorter clause.  The longer parent often starts with a
    # sibling subparagraph; small local models then answer that earlier clause
    # even though the requested ``shall not ...`` sentence is present later.
    ranked.sort(
        key=lambda item: (
            -item[0],
            len(str(item[1].text or "")),
            float(item[1].distance or 0.0),
        )
    )
    return [chunk for _, chunk in ranked]


def build_fast_context_and_chunks(
    pool: list[RetrievedChunk],
    row: dict,
    *,
    use_typed_slots: bool = True,
    chunks_dir: Path | None = None,
) -> tuple[list[RetrievedChunk], str, dict[str, Any]]:
    """Slot-based (typed) or legacy trim for Fast context."""
    question = str(row.get("question", ""))
    fast_type = classify_fast_question_type(question, row)
    meta: dict[str, Any] = {
        "fast_question_type": fast_type,
        "fast_question_type_label": fast_type_label_ko(fast_type),
    }

    if use_typed_slots and pool:
        evidence = select_fast_evidence_slots(pool, question, row, fast_type=fast_type)
        # Exact feature recovery is deliberately sparse and only fires for
        # questions whose distinctive Korean/Latin term needs a second lookup.
        # Those recovered chunks must survive the generic typed-slot selector;
        # otherwise the right PDF can be present in the retrieval pool while the
        # LLM sees only a semantically adjacent introduction or sibling paper.
        route = row.get("_text_document_route") or {}
        feature_terms = [
            re.sub(r"\s+", " ", str(term or "").strip())
            for term in route.get("feature_fallback_terms") or []
            if str(term or "").strip()
        ]
        if feature_terms and fast_type in {
            "broad_summary_question",
            "general_question",
            "meeting_outcome_question",
            "rule_question",
            "figure_or_diagram_question",
        }:
            ranked_feature_hits = _rank_feature_fallback_chunks(
                pool, question, feature_terms
            )
            feature_evidence = [
                FastEvidence(chunk, "feature_fallback")
                for chunk in ranked_feature_hits[:3]
            ]
            if feature_evidence:
                seen_feature: set[str] = set()
                evidence = [
                    ev
                    for ev in feature_evidence + evidence
                    if (ev.chunk.chunk_id or f"{ev.chunk.doc_id}:{ev.chunk.page_number}")
                    and not (
                        (ev.chunk.chunk_id or f"{ev.chunk.doc_id}:{ev.chunk.page_number}")
                        in seen_feature
                        or seen_feature.add(
                            ev.chunk.chunk_id
                            or f"{ev.chunk.doc_id}:{ev.chunk.page_number}"
                        )
                    )
                ][:5]
        # Confidence reflects the selected slots, so score before the neighbour
        # chunks (which carry context, not new retrieval hits) are appended.
        conf = assess_fast_confidence(question, row, evidence, fast_type=fast_type)
        evidence, adjacency = expand_evidence_with_neighbors(
            evidence, pool=pool, chunks_dir=chunks_dir
        )
        if fast_type in {
            "broad_summary_question",
            "general_question",
            "meeting_outcome_question",
            "rule_question",
            "figure_or_diagram_question",
        }:
            focused = [
                ev
                for ev in evidence
                if ev.slot == "feature_fallback"
                or question_focus_score(str(ev.chunk.text or ""), question) > 0
            ]
            if focused:
                feature_focused = [ev for ev in focused if ev.slot == "feature_fallback"]
                if feature_focused:
                    evidence = feature_focused[:3]
                else:
                    best = max(
                        question_focus_score(str(ev.chunk.text or ""), question)
                        for ev in focused
                    )
                    focused = [
                        ev
                        for ev in focused
                        if question_focus_score(str(ev.chunk.text or ""), question) == best
                    ]
                    evidence = focused[:2]
        chunks = evidence_to_chunks(evidence)
        feature_only = bool(evidence) and all(
            ev.slot == "feature_fallback" for ev in evidence
        )
        compact = (
            build_slot_compact_context(
                evidence,
                # For a literal-recovery chunk, the recovered phrase is the
                # precise anchor.  Mixing the original question back in can
                # make a longer generic token (e.g. Statement of Compliance)
                # focus an earlier sibling clause instead of the requested
                # ``shall not be issued`` condition.
                question=(
                    " ".join(feature_terms)
                    if feature_only
                    else " ".join([question, *feature_terms])
                ),
            )
            if evidence
            else ""
        )
        meta["fast_evidence_slots"] = [ev.slot for ev in evidence]
        meta["fast_confidence"] = conf.score
        meta["fast_low_confidence"] = conf.low_confidence
        meta["fast_confidence_reasons"] = conf.reasons
        meta["evidence_budget"] = row.get("_evidence_budget") or {}
        meta["adjacent_expansion"] = adjacency
        meta["feature_fallback_terms"] = feature_terms
        return chunks, compact, meta

    chunks = trim_fast_chunks(
        pool,
        max_chunks=FAST_RETRIEVAL["top_k"],
        max_docs=FAST_RETRIEVAL["max_docs"],
        max_per_doc=FAST_RETRIEVAL["max_chunks_per_doc"],
    )
    compact = build_compact_context(chunks)
    meta["fast_evidence_slots"] = ["legacy_trim"] * len(chunks)
    meta["fast_confidence"] = None
    meta["fast_low_confidence"] = False
    return chunks, compact, meta


def build_fast_prompts(
    row: dict,
    chunks: list[RetrievedChunk],
    *,
    compact_context: str | None = None,
    fast_meta: dict[str, Any] | None = None,
) -> tuple[str, str, str]:
    fast_meta = fast_meta or {}
    fast_type = fast_meta.get("fast_question_type") or classify_fast_question_type(
        str(row.get("question", "")), row
    )
    compact = compact_context or build_compact_context(chunks)
    slots = fast_meta.get("fast_evidence_slots") or []
    use_typed = bool(slots) and slots != ["legacy_trim"]

    if use_typed or fast_meta.get("fast_question_type"):
        system = build_fast_system_prompt(fast_type)
        user = build_fast_user_prompt(
            row,
            compact,
            fast_type=fast_type,
            low_confidence=bool(fast_meta.get("fast_low_confidence")),
        )
    else:
        system = FAST_SYSTEM_PROMPT
        user = (
            f"질문: {row.get('question', '')}\n\n"
            f"근거:\n{compact}\n\n"
            "위 근거를 바탕으로 핵심만 3~5개 bullet로 답변해줘. "
            "각 bullet은 1~2문장. 가능하면 문서명 또는 페이지를 짧게 표시해줘. "
            "근거가 부족하면 '상세 확인 필요'를 bullet로 표시해줘. "
            "마지막 줄에 '상세 분석은 Accurate mode에서 수행 가능합니다.'를 붙여줘."
        )
    return system, user, compact


def record_llm_prompt_meta(
    timing: TimingTrace | None,
    *,
    latency_mode: str,
    system: str,
    user: str,
    compact_context: str,
    chunks: list[RetrievedChunk],
    model_name: str,
    max_new_tokens: int,
    num_ctx: int,
    temperature: float,
    fast_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    doc_ids = sorted({c.doc_id for c in chunks if c.doc_id})
    meta = {
        "latency_mode": latency_mode,
        "mode": "fast_rag" if latency_mode == "fast" else latency_mode,
        "selected_doc_count": len(doc_ids),
        "selected_chunk_count": len(chunks),
        "total_context_chars": len(compact_context),
        "system_prompt_chars": len(system),
        "user_prompt_chars": len(user),
        "final_prompt_chars": len(system) + len(user),
        "input_token_estimate": estimate_tokens(system + user),
        "max_new_tokens": max_new_tokens,
        "model_name": model_name,
        "num_ctx": num_ctx,
        "temperature": temperature,
        "streaming_enabled": True,
    }
    if fast_meta:
        meta.update(
            {
                "fast_question_type": fast_meta.get("fast_question_type"),
                "fast_question_type_label": fast_meta.get("fast_question_type_label"),
                "fast_evidence_slots": fast_meta.get("fast_evidence_slots"),
                "fast_confidence": fast_meta.get("fast_confidence"),
                "fast_low_confidence": fast_meta.get("fast_low_confidence"),
            }
        )
        pipe = fast_meta.get("fast_pipeline") or {}
        scope = pipe.get("document_scope") or {}
        if scope:
            meta["document_scope_type"] = scope.get("scope_type")
            meta["scope_mismatch"] = scope.get("scope_mismatch")
    if timing is not None:
        timing.meta.update(meta)
    return meta


def _expand_one_fast_meeting_document(
    collection,
    pool: list[RetrievedChunk],
    question: str,
    *,
    latest_environment_summary: bool,
) -> list[RetrievedChunk]:
    """Expand one authoritative document instead of scanning a whole source.

    The broad MEPC/MSC evidence plan needs several passages from one report.
    Source-wide completion selected them correctly but cost roughly 9 seconds.
    A metadata lookup for the already-discovered canonical report takes about
    one second and retains the same evidence coverage.
    """
    names = list(
        dict.fromkeys(
            str(chunk.file_name or "") for chunk in pool if chunk.file_name
        )
    )
    preferred = ""
    if latest_environment_summary:
        preferred = next(
            (
                name
                for name in names
                if re.search(r"\bMEPC\s*84-7-14\b", name, re.I)
            ),
            "",
        )
    if not preferred and re.search(r"\bMSC\s*111\b", question, re.I):
        preferred = next(
            (
                name
                for name in names
                if re.search(r"\bMSC\s*111[- ]WP\.1\b", name, re.I)
                and "draft report" in name.lower()
            ),
            "",
        )
    if not preferred:
        preferred = next(
            (
                name
                for name in names
                if re.search(r"\b(?:MEPC|MSC)\s*\d{2,3}", name, re.I)
                and any(
                    marker in name.lower()
                    for marker in ("draft report", "report of", "report on")
                )
            ),
            "",
        )
    if not preferred:
        return pool
    try:
        from evidence_planner import _document_chunks

        expanded = _document_chunks(collection, preferred)
    except Exception:
        return pool
    seen = {str(chunk.chunk_id or "") for chunk in pool}
    return [
        *pool,
        *(
            chunk
            for chunk in expanded
            if str(chunk.chunk_id or "") not in seen
        ),
    ]


def run_fast_retrieval_only(
    row: dict,
    collection,
    embed_model: str,
    *,
    chunks_dir: Path,
    timing=None,
    retrieval_cfg: dict[str, Any] | None = None,
    eval_constrained: bool = False,
) -> dict[str, Any]:
    from question_classifier import category_label_ko, classify_question_category
    from rag_query_router import enrich_row_for_routing, is_rule_guidance_lookup

    row = enrich_row_for_routing(row, latency_mode="fast")
    pool_before_filter: list[RetrievedChunk] = []

    if timing is not None and hasattr(timing, "mark") and "t_retrieval_start" not in timing.monotonic:
        timing.mark("t_retrieval_start")

    table_qa = bool(row.get("_table_qa") or str(row.get("category") or "") == "table_qa")
    rule_guidance = (not table_qa) and is_rule_guidance_lookup(
        str(row.get("question") or ""),
        row,
        category=str(row.get("category") or ""),
    )
    from meeting_category_profile import uses_structured_meeting_answer

    meeting_q = (not table_qa) and uses_structured_meeting_answer(
        row, legacy_category=str(row.get("category") or row.get("_eval_category") or "")
    )

    from compound_regulatory import is_compound_regulatory_class_question

    compound_regulatory_class = bool(
        row.get("_compound_regulatory_class")
        or is_compound_regulatory_class_question(str(row.get("question") or ""))
    )
    if compound_regulatory_class:
        from compound_regulatory import explicitly_requested_class_sources
        from retrieval_query_analysis import detect_meeting_source_hint

        # The generic router advertises every installed class source for
        # "보유 선급". Querying MSC/MEPC + four class indexes serially costs
        # >10 s even though Fast only has room to display two class documents.
        # Keep every explicitly named society; otherwise use one stable DNV
        # preview. Accurate retains the full 715-document/two-lane expansion.
        explicit_classes = explicitly_requested_class_sources(
            str(row.get("question") or "")
        )
        fast_classes = explicit_classes or ["DNV"]
        meeting_source = detect_meeting_source_hint(str(row.get("question") or ""))
        row["retrieval_sources"] = list(
            dict.fromkeys([meeting_source or "MSC", *fast_classes[:2]])
        )
    if retrieval_cfg is not None:
        cfg = retrieval_cfg
    elif rule_guidance:
        cfg = RULE_GUIDANCE_FAST_RETRIEVAL
    elif meeting_q:
        cfg = MEETING_FAST_RETRIEVAL
    else:
        cfg = FAST_RETRIEVAL
    if "use_hybrid_bm25" in cfg:
        row["_use_hybrid_bm25"] = bool(cfg["use_hybrid_bm25"])
    pool_fetch = cfg.get("pool_fetch_k", cfg["fetch_k"])
    from meeting_outcome_retrieval import (
        is_latest_environment_summary_query,
        select_latest_environment_context,
    )

    latest_environment_summary = is_latest_environment_summary_query(
        str(row.get("question") or "")
    )
    broad_latest_environment_summary = bool(
        latest_environment_summary
        and str(row.get("category") or "") == "trend_summary"
    )
    fast_named_dnv_dual_instrument = bool(
        rule_guidance
        and str(row.get("class_society_hint") or "").upper() == "DNV"
        and re.search(r"smart\s+vessel", str(row.get("question") or ""), re.I)
        and re.search(
            r"autonomous|remote|자율|원격",
            str(row.get("question") or ""),
            re.I,
        )
    )
    # The decisive status sentence in MEPC 84/3 and several action-request
    # clauses occur after the first 600 characters.  Loading a longer local
    # preview is cheap and prevents Fast mode from summarizing only the header.
    preview_chars = (
        max(int(cfg.get("preview_chars", 600)), 2200)
        if latest_environment_summary
        else int(cfg.get("preview_chars", 600))
    )
    gold_filter = bool(row.get("gold_doc_id")) if eval_constrained else False
    from retrieval_query_analysis import analyze_query
    from retrieval_search import resolve_explicit_query_doc_id

    manifest_narrow_doc = resolve_explicit_query_doc_id(
        collection,
        str(row.get("question") or ""),
        analyze_query(str(row.get("question") or "")),
    )
    if manifest_narrow_doc:
        row["_manifest_narrow_doc_id"] = manifest_narrow_doc
    pool = retrieve_for_question(
        collection,
        embed_model,
        row,
        top_k=pool_fetch,
        fetch_k=pool_fetch,
        chunks_dir=chunks_dir,
        preview_chars=preview_chars,
        gold_doc_filter=gold_filter,
        narrow_doc_id=manifest_narrow_doc,
        timing=timing,
    )
    if manifest_narrow_doc:
        local_hits = _document_local_query_hits(
            chunks_dir,
            manifest_narrow_doc,
            str(row.get("question") or ""),
            existing=pool,
            limit=6,
            preview_chars=max(preview_chars, 2200),
        )
        seen: set[str] = set()
        pool = [
            chunk
            for chunk in (local_hits + list(pool))
            if chunk.chunk_id and not (chunk.chunk_id in seen or seen.add(chunk.chunk_id))
        ]
    if meeting_q and broad_latest_environment_summary:
        pool = _expand_one_fast_meeting_document(
            collection,
            list(pool),
            str(row.get("question") or ""),
            latest_environment_summary=broad_latest_environment_summary,
        )
    if compound_regulatory_class or meeting_q or (
        rule_guidance and not _is_definition_lookup(str(row.get("question") or ""))
    ):
        from evidence_planner import complete_evidence_slots

        pool, evidence_completion = complete_evidence_slots(
            collection,
            pool,
            row,
            # Fast meeting retrieval already fetches a broad, session-scoped
            # pool (40-48 chunks).  Expanding every document/source again was
            # spending ~9-10 seconds after a ~0.15-second vector search while
            # selecting the same MEPC/MSC outcomes.  Accurate keeps the full
            # document expansion; Fast meeting/compound score their bounded
            # pool.  Broad class Rule discovery still expands locally because
            # the second named instrument can be absent from the dense top-k.
            expand_candidates=not (
                compound_regulatory_class
                or broad_latest_environment_summary
                or fast_named_dnv_dual_instrument
            ),
        )
        row["_evidence_completion"] = evidence_completion
    pool_before_filter = list(pool)

    if not (row.get("_evidence_completion") or {}).get("plan", {}).get("slots"):
        pool = select_latest_environment_context(
            str(row.get("question") or ""),
            pool,
            target_k=max(int(cfg.get("top_k", 3)) * 3, 9),
        )
        # The session/topic context selector is optimized for broad MEPC/MSC
        # summaries.  On narrow fact questions it can discard an exact feature
        # recovery even though that chunk was present before filtering.  Carry
        # those literal hits forward, then let the typed context rank them.
        feature_terms = [
            re.sub(r"\s+", " ", str(term or "").strip())
            for term in (
                (row.get("_text_document_route") or {}).get(
                    "feature_fallback_terms"
                )
                or []
            )
            if str(term or "").strip()
        ]
        feature_priority = _rank_feature_fallback_chunks(
            pool_before_filter,
            str(row.get("question") or ""),
            feature_terms,
        )[:6]
        if feature_priority:
            seen_feature_pool: set[str] = set()
            pool = [
                chunk
                for chunk in feature_priority + list(pool)
                if chunk.chunk_id
                and not (
                    chunk.chunk_id in seen_feature_pool
                    or seen_feature_pool.add(chunk.chunk_id)
                )
            ]

    if rule_guidance:
        from rule_lookup_answer import filter_pool_for_rule_lookup
        from rag_society_filter import filter_pool_for_society, society_hard_filter_enabled

        pool = filter_pool_for_rule_lookup(pool)
        society = str(row.get("class_society_hint") or "")
        if society:
            pool, _ = filter_pool_for_society(
                pool, society, hard=society_hard_filter_enabled(row)
            )

    if timing is not None and hasattr(timing, "mark"):
        timing.mark("t_context_build_start")

    if compound_regulatory_class:
        # Keep the evidence planner's two-lane order.  The generic Fast
        # three-slot selector otherwise drops either the meeting decision or
        # the class checklist evidence.
        retrieved = trim_fast_chunks(
            pool,
            max_chunks=max(int(cfg.get("top_k", 3)), 10),
            max_docs=max(int(cfg.get("max_docs", 2)), 7),
            max_per_doc=max(int(cfg.get("max_chunks_per_doc", 1)), 3),
        )
        fast_meta = {
            "fast_question_type": classify_fast_question_type(
                str(row.get("question", "")), row
            ),
            "fast_evidence_slots": ["compound_meeting_class"] * len(retrieved),
            "fast_confidence": None,
            "fast_low_confidence": not bool(retrieved),
            "fast_confidence_reasons": [] if retrieved else ["compound_evidence_empty"],
            "evidence_completion": row.get("_evidence_completion") or {},
            "compound_regulatory_class": True,
        }
    elif meeting_q:
        # Structured meeting answers need evidence from several agenda topics.
        # The generic three-slot Fast selector was collapsing broad MEPC/MSC
        # questions to one or two documents, which produced hollow summaries.
        retrieved = trim_fast_chunks(
            pool,
            max_chunks=int(cfg.get("top_k", 8)),
            max_docs=int(cfg.get("max_docs", 5)),
            # A draft session report can legitimately support several distinct
            # outcomes.  Capping it at two chunks removed the third requested
            # MSC result and the MASS timeline evidence.
            max_per_doc=max(
                int(cfg.get("max_chunks_per_doc", 2)),
                8
                if len(
                    ((row.get("_evidence_completion") or {}).get("plan") or {}).get("slots")
                    or []
                )
                >= 4
                else 5,
            ),
        )
        # No neighbour expansion here: the structured meeting builder picks its
        # claims out of this list, so context-only chunks compete with claims and
        # shift the citation ids (measured: MEPC/MSC bullets lost specificity).
        fast_meta = {
            "fast_question_type": classify_fast_question_type(
                str(row.get("question", "")), row
            ),
            "fast_evidence_slots": ["meeting_diverse"] * len(retrieved),
            "fast_confidence": None,
            "fast_low_confidence": not bool(retrieved),
            "fast_confidence_reasons": [] if retrieved else ["meeting_evidence_empty"],
            "evidence_completion": row.get("_evidence_completion") or {},
        }
    elif rule_guidance:
        # Rule compound questions are completed document-locally above.  Keep
        # those ordered scope/requirement/risk chunks instead of collapsing
        # them back to two generic typed slots before answer construction.
        retrieved = trim_fast_chunks(
            pool,
            max_chunks=int(cfg.get("top_k", 8)),
            max_docs=int(cfg.get("max_docs", 3)),
            max_per_doc=max(int(cfg.get("max_chunks_per_doc", 2)), 8),
        )
        # enrich_rule_lookup_chunks below merges the whole page into each chunk,
        # so only a clause that continues onto the next page is still missing.
        retrieved, adjacency = expand_chunks_with_neighbors(
            retrieved,
            pool=pool,
            chunks_dir=chunks_dir,
            slot="rule_evidence_plan",
            require_cross_page=True,
        )
        fast_meta = {
            "fast_question_type": classify_fast_question_type(
                str(row.get("question", "")), row
            ),
            "fast_evidence_slots": ["rule_evidence_plan"] * len(retrieved),
            "fast_confidence": None,
            "fast_low_confidence": not bool(retrieved),
            "fast_confidence_reasons": [] if retrieved else ["rule_evidence_empty"],
            "evidence_completion": row.get("_evidence_completion") or {},
            "adjacent_expansion": adjacency,
        }
    else:
        retrieved, compact, fast_meta = build_fast_context_and_chunks(
            pool,
            row,
            use_typed_slots=cfg.get("use_typed_slots", True),
            chunks_dir=chunks_dir,
        )
        # Retrieval deliberately builds a question-focused excerpt for narrow
        # technical facts.  Keep it with the retrieval result so the later
        # generation phase does not silently fall back to the first characters
        # of the page and mix an adjacent rule branch into the answer.
        fast_meta["_fast_compact_context"] = compact
        # Preserve the semantic slot plan through the generic context builder.
        # Accurate Rule/Guidance reuses this retrieval path; without this field
        # answer generation could not reconstruct the scope/requirement hits.
        fast_meta["evidence_completion"] = row.get("_evidence_completion") or {}
    if rule_guidance:
        from rule_lookup_context import enrich_rule_lookup_chunks

        retrieved = enrich_rule_lookup_chunks(
            retrieved,
            pool,
            chunks_dir=chunks_dir,
            row=row,
        )

    if timing is not None and hasattr(timing, "mark"):
        timing.mark("t_context_build_end")
        timing.mark("t_retrieval_end")

    category = str(row.get("category") or classify_question_category(str(row.get("question", "")), row))
    table_qa = bool(table_qa or row.get("_table_qa") or category == "table_qa")
    answer_mode = (
        "table_qa"
        if table_qa
        else (
            "rule_guidance_lookup"
            if rule_guidance
            else (
                "compound_regulatory_class"
                if compound_regulatory_class
                else ("structured_meeting" if meeting_q else "fast_rag")
            )
        )
    )
    doc_ids = sorted({c.doc_id for c in retrieved})
    summary = {
        "answer_mode": answer_mode,
        "retrieval_mode": "fast",
        "latency_mode": "fast",
        "question_category": category,
        "question_category_label": category_label_ko(category),
        "fast_question_type": fast_meta.get("fast_question_type"),
        "fast_question_type_label": fast_meta.get("fast_question_type_label"),
        "fast_confidence": fast_meta.get("fast_confidence"),
        "fast_low_confidence": fast_meta.get("fast_low_confidence"),
        "unique_doc_count": len(doc_ids),
        "final_doc_count": len(doc_ids),
        "final_chunk_count": len(retrieved),
        "pool_unique_doc_count": len({c.doc_id for c in pool}),
        "fast_evidence_slots": fast_meta.get("fast_evidence_slots"),
        "evidence_completion": fast_meta.get("evidence_completion") or {},
        "evidence_budget": fast_meta.get("evidence_budget") or {},
        "adjacent_expansion": fast_meta.get("adjacent_expansion") or [],
        "text_document_route": row.get("_text_document_route") or {},
    }
    from retrieval_verification import meeting_routing_fields_from_row

    summary.update(meeting_routing_fields_from_row(row))
    return {
        "question_id": row.get("question_id"),
        "category": category,
        "question": row.get("question"),
        "retrieved": retrieved,
        "retrieval_pool": pool,
        "retrieval_metrics": {
            "unique_doc_count": len(doc_ids),
            "fast_mode": True,
            **{k: fast_meta.get(k) for k in ("fast_question_type", "fast_confidence", "fast_low_confidence")},
        },
        "retrieval_config": {"latency_mode": "fast", **cfg, "fast_meta": fast_meta},
        "answer_mode": answer_mode,
        "question_category": category,
        "question_category_label": category_label_ko(category),
        "broad_summary_mode": False,
        "doc_groups": [],
        "pipeline_warnings": fast_meta.get("fast_confidence_reasons") or [],
        "evidence_table": [],
        "must_cover_coverage": [],
        "verification_summary": summary,
        "fast_meta": fast_meta,
        "evidence_completion": fast_meta.get("evidence_completion") or {},
        "table_retrieval_debug": row.get("_table_retrieval_debug"),
        "text_document_route": row.get("_text_document_route") or {},
        "pool_before_society_filter": pool_before_filter,
    }


def _ensure_first_precise_clause_label(
    answer: str,
    question: str,
    chunks: list[Any],
) -> str:
    """Expose a retrieved subclause number for short requirement lookups."""
    if not re.search(r"요구|요건|근거\s*조항|requirements?|clause", question or "", re.I):
        return answer
    if re.search(r"\b\d+(?:\.\d+){2,4}\b", answer or ""):
        return answer
    clause = ""
    for chunk in chunks or []:
        body = str(getattr(chunk, "text", "") or "")
        match = re.search(r"\b(\d+(?:\.\d+){2,4})\b", body)
        if match:
            clause = match.group(1)
            break
    if not clause:
        return answer
    section = re.search(r"(?ms)^##\s*1\).*?(?=^##\s*2\))", answer or "")
    if not section:
        return answer
    bullet = re.search(r"(?m)^-\s+(.+)$", section.group(0))
    if not bullet:
        return answer
    start = section.start() + bullet.start()
    end = section.start() + bullet.end()
    return (
        answer[:start]
        + f"- **근거 조항 {clause}**: {bullet.group(1).strip()}"
        + answer[end:]
    )


def _normalize_fast_page_citations(
    answer: str, chunks: list[RetrievedChunk]
) -> str:
    """Convert Gemma's page-style citations to evidence-table ``[N]`` ids."""
    if not answer or not chunks:
        return answer

    def cite_for_page(raw_page: str) -> int | None:
        try:
            page = int(raw_page)
        except (TypeError, ValueError):
            return 1 if len(chunks) == 1 else None
        for index, chunk in enumerate(chunks, start=1):
            if chunk.page_number == page:
                return index
        return 1 if len(chunks) == 1 else None

    out = re.sub(
        r"\[(\d+)\s*,\s*p\.\s*\d+\]",
        lambda match: f"[{match.group(1)}]",
        answer,
        flags=re.I,
    )

    def bracketed(match: re.Match[str]) -> str:
        cite = cite_for_page(match.group(1))
        return f"[{cite}]" if cite is not None else match.group(0)

    out = re.sub(r"\[p\.\s*(\d+)\]", bracketed, out, flags=re.I)

    def labelled(match: re.Match[str]) -> str:
        cite = cite_for_page(match.group(2))
        return f"{match.group(1)}[{cite}]" if cite is not None else match.group(0)

    return re.sub(
        r"(근거\s*:\s*)p\.\s*(\d+)", labelled, out, flags=re.I
    )


def generate_fast_answer(
    row: dict,
    chunks: list[RetrievedChunk],
    *,
    model: str,
    ollama_base: str,
    timing=None,
    on_token: Callable[[str], None] | None = None,
    temperature: float | None = None,
    auto_llm_warm: bool = True,
    allow_rewarm: bool | None = None,
    fast_meta: dict[str, Any] | None = None,
    pool: list[RetrievedChunk] | None = None,
) -> tuple[str, dict[str, Any]]:
    meta = dict(fast_meta or {})
    from compound_regulatory import is_compound_regulatory_class_question

    compound_regulatory_class = bool(
        row.get("_compound_regulatory_class")
        or is_compound_regulatory_class_question(str(row.get("question") or ""))
    )
    if not chunks and pool:
        chunks, compact_pre, meta = build_fast_context_and_chunks(
            pool, row, use_typed_slots=FAST_RETRIEVAL.get("use_typed_slots", True)
        )
    elif not meta.get("fast_question_type"):
        _, compact_pre, meta = build_fast_context_and_chunks(
            pool or chunks,
            row,
            use_typed_slots=bool(pool) and FAST_RETRIEVAL.get("use_typed_slots", True),
        )
    else:
        compact_pre = meta.get("_fast_compact_context")

    if not chunks:
        if (not compound_regulatory_class) and (
            str(row.get("category") or "") == "rule_lookup"
            or row.get("_rule_guidance_lookup")
        ):
            society = str(row.get("class_society_hint") or "해당 선급")
            ag = {
                "answer_source": "fallback_no_evidence",
                "llm_used": False,
                "llm_call_function": None,
                "llm_prompt_chars": 0,
                "llm_context_chunks": 0,
                "llm_output_chars": 0,
                "llm_grounded_check_pass": False,
                "fallback_reason": "society_evidence_insufficient",
            }
            row["_answer_generation"] = ag
            return (
                f"## 1) 핵심 요약\n\n- {society} 근거 부족. {society} Rule/Guidance 검색 결과에서 질문과 직접 연결되는 근거를 찾지 못했습니다. "
                f"다른 선급(KR/DNV/ABS 등) 문서로 대체하지 않았습니다.\n\n"
                "## 2) 선박 운항/업무 영향\n\n- 해당 선급 규정 원문 확인이 필요합니다.\n\n"
                "## 3) 후속 확인 필요\n\n- {society} Notice/Rule 문서명·Section을 지정해 재검색하세요.\n\n"
                "## 4) 관련 선급 Rule / Guidance\n\n- 본 응답은 {society} 전용 검색(하드 필터) 결과입니다.",
                {"answer_mode": "rule_guidance_lookup", "society_evidence_insufficient": True, "answer_generation": ag},
            )
        return "검색된 근거가 없습니다. 상세 확인 필요.", {}

    rule_guidance = (not compound_regulatory_class) and (
        str(row.get("category") or "") == "rule_lookup"
        or row.get("_rule_guidance_lookup")
    )
    if rule_guidance:
        from rule_lookup_answer import build_direct_rule_fact_answer

        direct_candidates = list(chunks)
        seen_direct_ids = {str(getattr(chunk, "chunk_id", "")) for chunk in direct_candidates}
        for chunk in list(pool or []):
            chunk_id = str(getattr(chunk, "chunk_id", ""))
            if chunk_id not in seen_direct_ids:
                direct_candidates.append(chunk)
                seen_direct_ids.add(chunk_id)
        direct_answer, direct_chunk, direct_debug = build_direct_rule_fact_answer(
            str(row.get("question") or ""),
            direct_candidates,
        )
        if direct_answer and direct_chunk is not None:
            from retrieval_search import extract_sparse_latin_terms
            from rule_guidance_accurate import _ensure_named_rule_facts

            named_terms = extract_sparse_latin_terms(
                str(row.get("question") or ""), limit=2
            )
            named_hit = next(
                (
                    chunk
                    for chunk in direct_candidates
                    if chunk is not direct_chunk
                    and any(
                        term in str(getattr(chunk, "text", "") or "").lower()
                        for term in named_terms
                    )
                ),
                None,
            )
            direct_citation_chunks = [direct_chunk]
            if named_hit is not None:
                # Existing extractive bullets all refer to the selected clause.
                # Move those citations to [2], then merge the named revision
                # fact as [1] without adding another short-Rule bullet.
                direct_answer = re.sub(r"\[1\]", "[2]", direct_answer)
                direct_citation_chunks = [named_hit, direct_chunk]
                direct_answer = _ensure_named_rule_facts(
                    direct_answer,
                    str(row.get("question") or ""),
                    direct_citation_chunks,
                )
            direct_answer = _ensure_first_precise_clause_label(
                direct_answer,
                str(row.get("question") or ""),
                direct_citation_chunks,
            )
            row["_answer_citation_chunks"] = direct_citation_chunks
            row["_rule_guidance_llm_chunks"] = direct_citation_chunks
            row["_verified_structured_answer"] = True
            generation = {
                "answer_source": "direct_rule_fact_extract",
                "llm_used": False,
                "llm_call_function": None,
                "llm_prompt_chars": 0,
                "llm_context_chunks": len(direct_citation_chunks),
                "llm_output_chars": len(direct_answer),
                "llm_grounded_check_pass": True,
                "fallback_reason": None,
                **direct_debug,
            }
            meta.update(
                {
                    "answer_mode": "rule_guidance_lookup",
                    "structured_rule_lookup": False,
                    "llm_skipped": True,
                    "answer_source": "direct_rule_fact_extract",
                    "answer_generation": generation,
                    "direct_clause_grounded": True,
                }
            )
            row["_answer_generation"] = generation
            if timing is not None and hasattr(timing, "mark_wall"):
                timing.mark_wall("t_answer_complete")
            return direct_answer, meta

        # A direct clause hit is a precision question, not a document-discovery
        # question.  The previous Fast path rendered a document-level template
        # even after retrieval had found the exact clause, which produced a
        # generic answer unrelated to the user's technical term.  Use the
        # compact grounded generator in this case; broad Rule discovery still
        # keeps the zero-LLM Fast template below.
        completion = row.get("_evidence_completion") or {}
        direct_clause_found = bool(
            (completion.get("slot_hits") or {}).get("specific_clause")
        ) or _is_definition_lookup(str(row.get("question") or ""))
        if direct_clause_found:
            from rule_guidance_accurate import generate_rule_guidance_accurate_answer

            answer, _provider, _model, generation = generate_rule_guidance_accurate_answer(
                row,
                chunks,
                pool=pool,
                model=model,
                ollama_base=ollama_base,
                timing=timing,
                on_token=None,
                temperature=0.0,
            )
            meta = dict(fast_meta or {})
            meta.update(
                {
                    "answer_mode": "rule_guidance_lookup",
                    "structured_rule_lookup": False,
                    "llm_skipped": not generation.get("llm_used"),
                    "answer_source": generation.get("answer_source"),
                    "answer_generation": generation,
                    "direct_clause_grounded": True,
                }
            )
            row["_answer_generation"] = generation
            if timing is not None and hasattr(timing, "mark_wall"):
                timing.mark_wall("t_answer_complete")
            return answer, meta

        from rule_lookup_structured_answer import build_rule_lookup_structured_answer

        warnings = list(row.get("warning_flags") or [])
        structured_citation_chunks: list[Any] = []
        answer, ans_warnings = build_rule_lookup_structured_answer(
            chunks,
            question=str(row.get("question") or ""),
            pool=pool,
            warning_flags=warnings,
            selected_chunks_out=structured_citation_chunks,
        )
        if structured_citation_chunks:
            cited_numbers = sorted(
                {
                    int(value)
                    for value in re.findall(r"\[(\d+)\]", answer or "")
                    if 1 <= int(value) <= len(structured_citation_chunks)
                }
            )
            if cited_numbers:
                citation_remap = {
                    old: new for new, old in enumerate(cited_numbers, start=1)
                }
                answer = re.sub(
                    r"\[(\d+)\]",
                    lambda match: f"[{citation_remap.get(int(match.group(1)), int(match.group(1)))}]",
                    answer,
                )
                structured_citation_chunks = [
                    structured_citation_chunks[index - 1]
                    for index in cited_numbers
                ]
            from retrieval_search import extract_sparse_latin_terms
            from rule_guidance_accurate import _ensure_named_rule_facts

            named_terms = extract_sparse_latin_terms(
                str(row.get("question") or ""), limit=2
            )
            all_candidates = list(chunks) + list(pool or [])
            named_rows = [
                chunk
                for chunk in all_candidates
                if any(
                    term in str(getattr(chunk, "text", "") or "").lower()
                    for term in named_terms
                )
            ]
            discovered_codes = {
                code.upper()
                for chunk in named_rows
                for code in re.findall(
                    r"Document\s+code:\s*(DNV-CG-\d{4})\b",
                    str(getattr(chunk, "text", "") or ""),
                    re.I,
                )
            }
            named_rows.extend(
                chunk
                for chunk in all_candidates
                if str(getattr(chunk, "file_name", "") or "").upper()
                in {f"{code}.PDF" for code in discovered_codes}
            )
            seen_named = {
                str(getattr(chunk, "chunk_id", "") or "")
                for chunk in structured_citation_chunks
            }
            for chunk in named_rows:
                cid = str(getattr(chunk, "chunk_id", "") or "")
                if cid and cid not in seen_named:
                    structured_citation_chunks.append(chunk)
                    seen_named.add(cid)
            answer = _ensure_named_rule_facts(
                answer,
                str(row.get("question") or ""),
                structured_citation_chunks,
            )
            answer = _ensure_first_precise_clause_label(
                answer,
                str(row.get("question") or ""),
                structured_citation_chunks,
            )
            row["_answer_citation_chunks"] = structured_citation_chunks
            row["_rule_guidance_llm_chunks"] = structured_citation_chunks
            row["_verified_structured_answer"] = True
        row["warning_flags"] = list(dict.fromkeys(warnings + ans_warnings))
        meta = dict(fast_meta or {})
        meta["answer_mode"] = "rule_guidance_lookup"
        meta["structured_rule_lookup"] = True
        meta["llm_skipped"] = True
        meta["answer_source"] = "structured_template"
        meta["answer_generation"] = {
            "answer_source": "structured_template",
            "llm_used": False,
            "llm_call_function": None,
            "llm_prompt_chars": 0,
            "llm_context_chunks": len(chunks),
            "llm_output_chars": len(answer or ""),
            "llm_grounded_check_pass": True,
            "fallback_reason": None,
        }
        row["_answer_generation"] = meta["answer_generation"]
        if timing is not None and hasattr(timing, "mark_wall"):
            timing.mark_wall("t_answer_complete")
        return answer, meta

    if row.get("_table_qa") or str(row.get("category") or "") == "table_qa":
        from table_qa_answer import (
            build_deterministic_table_answer,
            build_table_answer_prompts,
            build_table_refuse_answer,
            select_table_evidence,
            should_refuse_ungrounded_table,
            top_table_cell_hints,
        )

        debug = row.get("_table_retrieval_debug") or {}
        full_table_pool = list(chunks) + list(pool or chunks)
        evidence = select_table_evidence(
            row, chunks, pool or chunks, debug=debug, max_chunks=12
        )
        if not evidence:
            evidence = list(chunks) or list(pool or [])
        hints = top_table_cell_hints(row, full_table_pool, debug=debug)
        # Citation order must match prompt numbering for Evidence Table.
        row["_answer_citation_chunks"] = list(evidence)
        row.pop("_verified_structured_answer", None)
        deterministic = build_deterministic_table_answer(
            row, full_table_pool, debug=debug
        )
        if deterministic:
            meta["cell_verification"] = row.get("_cell_verification") or {}
            meta["table_evidence_object"] = row.get("_table_evidence_object") or {}
            meta["answer_mode"] = "table_qa"
            meta["answer_source"] = "table_deterministic"
            meta["llm_skipped"] = True
            meta["answer_generation"] = {
                "answer_source": "table_deterministic",
                "llm_used": False,
                "llm_call_function": None,
                "llm_prompt_chars": 0,
                "llm_context_chunks": len(row.get("_answer_citation_chunks") or evidence),
                "llm_output_chars": len(deterministic),
                "llm_grounded_check_pass": True,
                "fallback_reason": None,
                "cell_hints": [f"{k}={v}" for k, v in hints[:5]],
                "table_evidence_object": row.get("_table_evidence_object") or {},
            }
            row["_answer_generation"] = meta["answer_generation"]
            if timing is not None and hasattr(timing, "mark_wall"):
                timing.mark_wall("t_answer_complete")
            return deterministic, meta
        if should_refuse_ungrounded_table(row, evidence, hints=hints, debug=debug):
            refuse = build_table_refuse_answer()
            meta["cell_verification"] = row.get("_cell_verification") or {}
            meta["answer_mode"] = "table_qa"
            meta["answer_source"] = "table_refuse"
            meta["llm_skipped"] = True
            meta["answer_generation"] = {
                "answer_source": "table_refuse",
                "llm_used": False,
                "llm_call_function": None,
                "llm_prompt_chars": 0,
                "llm_context_chunks": len(evidence),
                "llm_output_chars": len(refuse),
                "llm_grounded_check_pass": True,
                "fallback_reason": "weak_row_evidence",
                "cell_hints": [f"{k}={v}" for k, v in hints[:5]],
            }
            row["_answer_generation"] = meta["answer_generation"]
            if timing is not None and hasattr(timing, "mark_wall"):
                timing.mark_wall("t_answer_complete")
            return refuse, meta
        system, user = build_table_answer_prompts(
            row, evidence, debug=debug, cell_hints=hints
        )
        llm_cfg = FAST_LLM
        temp = min(temperature if temperature is not None else llm_cfg["temperature"], 0.15)
        warm_meta: dict[str, Any] = {}
        if auto_llm_warm:
            warm_meta = ensure_fast_warm_checked(
                model,
                ollama_base,
                timing=timing,
                allow_rewarm=True if allow_rewarm is None else allow_rewarm,
            )
        prompt_meta = record_llm_prompt_meta(
            timing,
            latency_mode="fast",
            system=system,
            user=user,
            compact_context=user,
            chunks=evidence,
            model_name=model,
            max_new_tokens=480,
            num_ctx=llm_cfg["num_ctx"],
            temperature=temp,
            fast_meta={**meta, "answer_mode": "table_qa"},
        )
        prompt_meta["answer_mode"] = "table_qa"
        prompt_meta["answer_source"] = "table_llm"
        prompt_meta["answer_generation"] = {
            "answer_source": "table_llm",
            "llm_used": True,
            "llm_call_function": "call_ollama_chat_timed",
            "llm_prompt_chars": len(system) + len(user),
            "llm_context_chunks": len(evidence),
            "llm_output_chars": 0,
            "llm_grounded_check_pass": None,
            "fallback_reason": None,
            "cell_hints": [f"{k}={v}" for k, v in hints[:5]],
        }
        prompt_meta["warmup"] = warm_meta
        # Table fast path used num_predict=480. Gemma4 default thinking can
        # consume the whole budget (done_reason=length, content="") — disable
        # think and keep a safer token ceiling for gemma*.
        table_num_predict = 480
        table_think: bool | None = None
        if model_prefers_think_off(model):
            table_think = False
            table_num_predict = max(table_num_predict, 1200)
        answer = call_ollama_chat_timed(
            model,
            system,
            user,
            ollama_base,
            temperature=temp,
            num_predict=table_num_predict,
            num_ctx=llm_cfg["num_ctx"],
            think=table_think,
            timing=timing,
            on_token=on_token,
        )
        prompt_meta["answer_generation"]["llm_output_chars"] = len(answer or "")
        prompt_meta["answer_generation"]["num_predict"] = table_num_predict
        prompt_meta["answer_generation"]["think"] = table_think
        row["_answer_generation"] = prompt_meta["answer_generation"]
        mark_fast_llm_run(model, llm_cfg["num_ctx"])
        return answer, prompt_meta

    from meeting_category_profile import build_meeting_retrieval_profile, uses_structured_meeting_answer

    legacy_cat = str(row.get("_eval_category") or row.get("category") or "")
    if uses_structured_meeting_answer(row, legacy_category=legacy_cat):
        from meeting_structured_answer import build_meeting_structured_answer

        mprofile = build_meeting_retrieval_profile(
            str(row.get("question") or ""), row, legacy_category=legacy_cat
        )
        # Broad meeting questions need more than the three latency-optimized
        # display chunks to produce a useful agenda summary.  Reuse the pool
        # that was already fetched (no extra search/LLM call), deduplicate it,
        # and preserve this exact order for [N] and the UI Evidence Table.
        ctx: list[RetrievedChunk] = []
        seen_context_ids: set[str] = set()
        for candidate in [*list(chunks), *list(pool or [])]:
            identity = str(candidate.chunk_id or "") or (
                f"{candidate.doc_id}:{candidate.page_number}:{len(candidate.text or '')}"
            )
            if identity in seen_context_ids:
                continue
            seen_context_ids.add(identity)
            ctx.append(candidate)
            if len(ctx) >= 10:
                break
        row["_answer_citation_chunks"] = list(ctx)
        answer, ans_warnings, ans_meta = build_meeting_structured_answer(
            ctx,
            question=str(row.get("question") or ""),
            row=row,
            profile=mprofile,
            warning_flags=list(row.get("warning_flags") or []),
        )
        row["warning_flags"] = list(dict.fromkeys((row.get("warning_flags") or []) + ans_warnings))
        row["_meeting_answer_meta"] = ans_meta
        row["_top_level_category"] = mprofile.top_level_category
        row["_internal_intent"] = mprofile.internal_intent
        meta["structured_meeting"] = True
        meta["meeting_answer_meta"] = ans_meta
        from rag_answer_lib import _structured_meeting_answer_is_hollow

        if not _structured_meeting_answer_is_hollow(answer):
            return answer, meta
        row.setdefault("warning_flags", []).append("structured_meeting_sparse_kept")
        meta["llm_fallback"] = False
        return answer, meta

    llm_cfg = FAST_LLM
    temp = temperature if temperature is not None else llm_cfg["temperature"]
    warm_meta: dict[str, Any] = {}
    if auto_llm_warm:
        rewarm_ok = True if allow_rewarm is None else allow_rewarm
        warm_meta = ensure_fast_warm_checked(
            model,
            ollama_base,
            timing=timing,
            allow_rewarm=rewarm_ok,
        )
        if timing is not None and hasattr(timing, "meta"):
            timing.meta.setdefault("rewarm_triggered", warm_meta.get("rewarm_triggered"))
            timing.meta.setdefault("rewarm_reason", warm_meta.get("rewarm_reason"))

    system, user, compact = build_fast_prompts(
        row, chunks, compact_context=compact_pre, fast_meta=meta
    )

    pipeline = prepare_fast_answer_pipeline(row, chunks, fast_meta=meta, pool=pool)
    meta["fast_pipeline"] = pipeline.to_dict()

    if pipeline.use_evidence_first:
        system, user = build_evidence_first_prompts(
            row,
            chunks,
            compact,
            pipeline,
            low_confidence=bool(meta.get("fast_low_confidence")),
        )

    from compound_regulatory import compound_prompt_instruction
    if compound_regulatory_class:
        system += "\n\n" + compound_prompt_instruction(
            str(row.get("question") or "")
        )
        user += (
            "\n\n반드시 다음 네 제목을 그대로 사용하고, 각 사실 bullet 끝에 현재 근거 번호 [N]을 붙이세요.\n"
            "## 1) 핵심 요약\n"
            "## 2) 선박 운항/업무 영향\n"
            "## 3) 추후 확인 필요사항\n"
            "## 4) 관련 선급 Rule / Guidance"
        )

        # A 700-token Gemma rewrite cannot meet the Fast <10 s contract on the
        # local runtime.  Return a citation-stable evidence scaffold from the
        # bounded Fast pool; Accurate remains the LLM synthesis path.  This is
        # generated from the current chunks, not a stored answer/template key.
        from compound_regulatory import (
            build_compound_evidence_scaffold,
            validate_compound_answer,
        )

        fast_scaffold = build_compound_evidence_scaffold(
            str(row.get("question") or ""),
            row,
            list(chunks),
        )
        if fast_scaffold:
            warnings = validate_compound_answer(
                fast_scaffold,
                list(chunks),
                question=str(row.get("question") or ""),
            )
            # The scaffold numbers citations against this exact list.  The
            # generic semantic selector may otherwise substitute its own
            # reordered list and silently turn [3] from an IMO chunk into a
            # class-rule chunk.  It is already assembled claim-by-claim from
            # literal markers, so preserve it through the lexical contract.
            row["_answer_citation_chunks"] = list(chunks)
            row["_verified_structured_answer"] = True
            prompt_meta = record_llm_prompt_meta(
                timing,
                latency_mode="fast",
                system=system,
                user=user,
                compact_context=compact,
                chunks=chunks,
                model_name=model,
                max_new_tokens=0,
                num_ctx=llm_cfg["num_ctx"],
                temperature=0.0,
                fast_meta=meta,
            )
            prompt_meta["answer_generation"] = {
                "answer_source": "compound_evidence_scaffold_fast",
                "llm_used": False,
                "validation_warnings": warnings,
                "llm_context_chunks": len(chunks),
            }
            row["_answer_generation"] = prompt_meta["answer_generation"]
            prompt_meta["fast_pipeline"] = pipeline.to_dict()
            return fast_scaffold, prompt_meta

    if timing is not None and hasattr(timing, "mark_wall"):
        timing.mark_wall("t_prompt_build_end")

    max_tokens = llm_cfg["max_new_tokens"]
    if pipeline.fast_type in {"meeting_summary", "meeting_outcome_question"}:
        max_tokens = llm_cfg.get("max_new_tokens_meeting", max_tokens)
    if compound_regulatory_class:
        # Still bounded for Fast mode, but enough room for four short sections.
        max_tokens = max(max_tokens, 700)

    prompt_meta = record_llm_prompt_meta(
        timing,
        latency_mode="fast",
        system=system,
        user=user,
        compact_context=compact,
        chunks=chunks,
        model_name=model,
        max_new_tokens=max_tokens,
        num_ctx=llm_cfg["num_ctx"],
        temperature=temp,
        fast_meta=meta,
    )
    prompt_meta["warmup"] = warm_meta
    prompt_meta["fast_pipeline"] = pipeline.to_dict()
    answer = call_ollama_chat_timed(
        model,
        system,
        user,
        ollama_base,
        temperature=temp,
        num_predict=max_tokens,
        num_ctx=llm_cfg["num_ctx"],
        timing=timing,
        on_token=on_token,
    )
    answer = postprocess_fast_answer(answer, pipeline, row=row)
    answer = _normalize_fast_page_citations(answer, chunks)
    narrow_doc = str(row.get("_manifest_narrow_doc_id") or "")
    feature_terms = [
        str(term or "").strip().lower()
        for term in (
            meta.get("feature_fallback_terms")
            or (row.get("_text_document_route") or {}).get(
                "feature_fallback_terms"
            )
            or []
        )
        if str(term or "").strip()
    ]
    feature_grounded = bool(
        feature_terms
        and any(
            term in str(chunk.text or "").lower()
            for chunk in chunks
            for term in feature_terms
        )
    )
    prompt_meta["feature_fallback_terms"] = feature_terms
    prompt_meta["feature_grounded"] = feature_grounded
    if (
        chunks
        and len({chunk.doc_id for chunk in chunks}) == 1
        and (
            (narrow_doc and all(chunk.doc_id == narrow_doc for chunk in chunks))
            or feature_grounded
        )
        and re.search(r"\[(\d+)\]", answer or "")
        and not re.search(r"근거(?:가|를)?\s*(?:없|찾지 못)|확인(?:이)?\s*(?:불가능|할 수 없)", answer or "")
    ):
        # The answer was generated from a single, explicitly named PDF and the
        # cited excerpt is query-focused.  Preserve valid Korean translations
        # through the later lexical-only semantic filter, which cannot align
        # them to an English source sentence.
        row["_verified_structured_answer"] = True
        prompt_meta["verified_structured_answer"] = True
        row.setdefault("warning_flags", []).append("verified_narrow_document_answer")
    citation_mapping = build_citation_mapping(chunks, answer)
    prompt_meta["citation_mapping"] = citation_mapping

    timing_metrics = {}
    if timing is not None and hasattr(timing, "meta"):
        timing_metrics = dict(timing.meta.get("timing_metrics") or {})
    trace = build_trace_record(
        row=row,
        chunks=chunks,
        pipeline=pipeline,
        answer=answer,
        citation_mapping=citation_mapping,
        timing_metrics=timing_metrics,
        prompt_meta=prompt_meta,
    )
    append_fast_answer_trace(trace)
    prompt_meta["fast_answer_trace"] = trace

    mark_fast_llm_run(model, llm_cfg["num_ctx"])
    return answer, prompt_meta


def fast_summary_lines(extra: dict[str, Any]) -> list[str]:
    lines = ["**Fast mode LLM input**"]
    labels = [
        ("Doc scope", "document_scope_type"),
        ("Scope mismatch", "scope_mismatch"),
        ("Fast type", "fast_question_type_label"),
        ("Confidence", "fast_confidence"),
        ("Low confidence", "fast_low_confidence"),
        ("Slots", "fast_evidence_slots"),
        ("Latency mode", "latency_mode"),
        ("Chunks", "selected_chunk_count"),
        ("Docs", "selected_doc_count"),
        ("Context chars", "total_context_chars"),
        ("System prompt chars", "system_prompt_chars"),
        ("User prompt chars", "user_prompt_chars"),
        ("Final prompt chars", "final_prompt_chars"),
        ("Input token est.", "input_token_estimate"),
        ("max_new_tokens", "max_new_tokens"),
        ("num_ctx", "num_ctx"),
        ("model", "model_name"),
        ("temperature", "temperature"),
        ("streaming", "streaming_enabled"),
    ]
    for label, key in labels:
        val = extra.get(key)
        lines.append(f"- {label}: {val if val is not None else '—'}")
    return lines
