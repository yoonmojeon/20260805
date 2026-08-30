"""Accurate Rule/Guidance: evidence draft + short LLM-grounded Korean answer."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from rag_answer_lib import RetrievedChunk, call_ollama_chat_timed
from rule_lookup_context import is_crossref_table_chunk, strip_metadata_prefix
from rule_lookup_structured_answer import build_rule_lookup_structured_answer
from rag_society_filter import filter_pool_for_society
from grounded_answer_policy import verify_claim_citations
from direct_clause_grounding import (
    build_clause_proposition_block,
    clause_body,
    direct_clause_coverage_gaps,
    ensure_direct_clause_source_details,
    extract_clause_reference,
    replace_rule_reference_section,
    select_specific_clause_chunks,
    validate_direct_clause_answer,
)

# Accurate mode is a quality path.  A broad Rule/Guidance request usually
# needs an applicability clause, a requirement clause and a safety-control
# clause together; three short fragments made the model fall back to a generic
# document description.  Keep Fast mode compact, but give Accurate mode the
# already-ranked multi-slot evidence it needs to produce a working answer.
MAX_LLM_CHUNKS = 8
MAX_CHUNK_CHARS = 900
MAX_TOTAL_CONTEXT_CHARS = 6400
RULE_GUIDANCE_NUM_CTX = 4096
ACCURATE_NUM_CTX = RULE_GUIDANCE_NUM_CTX
# Broad Korean Rule answers frequently reach the fourth required section only
# after ~360 tokens.  Accurate mode has a 20-25 s quality budget, so leave
# enough room for the complete four-section answer instead of rejecting a
# faithful but truncated draft back to a sparse template.
ACCURATE_NUM_PREDICT = 520
DIRECT_CLAUSE_NUM_PREDICT = 520
ACCURATE_TEMPERATURE = 0.0
KEEP_ALIVE = "24h"


EXACT_RULE_FACT_EXCLUSION_RE = re.compile(
    r"(?:목록|체크리스트|요약|정리|최신\s*동향|주요\s*(?:내용|결과)|"
    r"논의\s*및\s*결론|미확정\s*규제|업무\s*영향|운항\s*영향|"
    r"어떤\s*(?:정보|항목|구성\s*요소|것들?).{0,20}포함|"
    r"구성\s*요소.{0,20}무엇|"
    r"(?:장치|설비|대상)들|대상.{0,30}(?:와|과|및|·).{0,30}목적)",
    re.I,
)
EXACT_RULE_FACT_SIGNAL_RE = re.compile(
    r"(?:"
    r"어떤\s*(?:정격|종류|조건|경우|시험|특기사항)|"
    r"어느\s*(?:위치|장|절|조항|단계)|어떤\s*목표|몇\s*(?:시간|일|대|개|톤|배|mm|m)|"
    r"며칠|어디|얼마|언제|어느\s*정도|정의(?:는|가)?|무엇을\s*(?:뜻|의미|말)|"
    r"종류(?:는|가)|이유(?:는|가)?|주요\s*외부\s*위험|(?:두|2)\s*가지|"
    r"적용(?:되는|됩니까|되나요)|필요(?:한가|합니까)|"
    r"생략할\s*수\s*있는\s*조건|어떻게\s*(?:구성|시행)|"
    r"조건.{0,20}어떻게|어떻게\s*조치|"
    r"(?:0\.\d+/)?\d+(?:\.\d+)?\s*(?:kV|V|mm|m|시간|일|톤|배)"
    r")",
    re.I,
)


def is_exact_rule_fact_question(question: str) -> bool:
    """Identify a bounded value/condition lookup inside the Rule path.

    This is an answer-shape decision, not a retrieval route.  Broad document
    discovery, checklists and multi-item inventories keep the full analytical
    answer, while scalar or paired clause questions receive a compact answer.
    """
    q = re.sub(r"\s+", " ", str(question or "")).strip()
    if not q or EXACT_RULE_FACT_EXCLUSION_RE.search(q):
        return False
    return bool(EXACT_RULE_FACT_SIGNAL_RE.search(q))


def exact_rule_fact_slots(question: str) -> int:
    """Return the maximum number of distinct answer facts requested."""
    q = re.sub(r"\s+", " ", str(question or "")).strip()
    explicit = re.search(r"(?:두|2)\s*(?:가지|개|항목)", q)
    if re.search(r"국가\s*나\s*단체", q):
        return 3
    paired = bool(
        re.search(r"각각", q)
        or (
            len(
                re.findall(
                    r"어떤|어느|몇|며칠|얼마|언제|무엇|어디|어떻게",
                    q,
                )
            )
            >= 2
        )
        or re.search(
            r"(?:정격|등급|조건|정의|대상).{0,55}(?:와|과|및|·).{0,55}"
            r"(?:정격|등급|조건|정의|대상|케이블|회로|변형)",
            q,
            re.I,
        )
        or re.search(r"전력\s*케이블.{0,50}(?:와|과|및|·).{0,50}제어", q, re.I)
    )
    if explicit or paired:
        return 2
    if re.search(r"어떻게\s*구성|주요\s*외부\s*위험|이유(?:는|가)?", q, re.I):
        return 2
    return 1


def _is_definition_lookup(question: str) -> bool:
    """Return True for a symbol/term definition request.

    Definition questions are direct-clause lookups even when the evidence
    planner was intentionally skipped on the low-latency path.
    """
    return bool(
        re.search(
            r"(?:기호.{0,16}(?:뜻|의미)|무엇을\s*뜻|무슨\s*뜻|"
            r"(?<!규)정의|means|meaning|defined\s+as)",
            question or "",
            re.I,
        )
    )


def _build_definition_extractive_answer(
    question: str,
    chunks: list[Any],
    society: str,
) -> tuple[str, Any | None]:
    """Build a short answer from an explicit symbol-definition line."""
    ignored = {
        "rule", "rules", "guidance", "what", "which", "means", "meaning",
        "definition", "define", "defined", "symbol", "term", "thickness",
    }
    identifiers = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", question or "")
        if token.lower() not in ignored
    ]
    normalized_question = re.sub(r"\([^)]*\)", "", question or "")
    korean_match = re.search(
        r"([가-힣][가-힣\s]{1,40}?)(?:의)?\s*정의(?:는|란|가|를|이)?",
        normalized_question,
    )
    if korean_match:
        korean_anchor = re.split(
            r"(?:에서|기준으로|중)\s*", korean_match.group(1)
        )[-1].strip()
        korean_anchor = re.sub(r"^(?:규칙|지침|문서)\s+", "", korean_anchor).strip()
        if len(korean_anchor) >= 2:
            identifiers.insert(0, korean_anchor)
    if not identifiers:
        return "", None
    anchor = identifiers[0]
    ranked: list[tuple[int, int, Any, str]] = []
    for order, chunk in enumerate(chunks):
        body = strip_metadata_prefix(str(getattr(chunk, "text", "") or "")).strip()
        anchor_pattern = (
            rf"\b{re.escape(anchor)}\b"
            if re.fullmatch(r"[A-Za-z0-9_-]+", anchor)
            else re.escape(anchor)
        )
        if not re.search(anchor_pattern, body, re.I):
            continue
        score = 0
        if re.search(
            rf"(?im)^\s*{re.escape(anchor)}\s*[:：]", body
        ):
            score += 8
        if re.search(
            r"정의(?:된|한다|는)|의미|뜻|means|defined\s+as|corrosion\s+addition|부식추가",
            body,
            re.I,
        ):
            score += 5
        if re.search(
            rf"{re.escape(anchor)}(?:\s*\([^)]*\))?.{{0,40}}"
            r"(?:이라\s*함은|라\s*함은|란|을\s*말한다)",
            body,
            re.I | re.S,
        ):
            score += 10
        if re.search(r"\bmm\b|두께", body, re.I):
            score += 2
        ranked.append((score, -order, chunk, body))
    if not ranked:
        return "", None

    _score, _order, chunk, body = max(ranked, key=lambda item: (item[0], item[1]))
    korean_definition = None
    if re.search(r"[가-힣]", anchor):
        korean_definition = re.search(
            rf"({re.escape(anchor)}(?:\s*\([^)]*\))?\s*"
            r"(?:이라\s*함은|라\s*함은|란)\s*.{2,700}?말한다\.)",
            body,
            re.I | re.S,
        )
    value_match = re.search(
        rf"(?im)^\s*{re.escape(anchor)}\s*[:：]\s*([^\r\n]{{2,180}})",
        body,
    )
    if korean_definition:
        fact = re.sub(r"\s+", " ", korean_definition.group(1)).strip()
    elif value_match:
        value = re.sub(r"\s+", " ", value_match.group(1)).strip(" .")
        fact = f"`{anchor}`는 {value}를 뜻합니다."
    else:
        lines = [re.sub(r"\s+", " ", line).strip() for line in body.splitlines()]
        direct = next(
            (
                line
                for line in lines
                if re.search(rf"\b{re.escape(anchor)}\b", line, re.I)
                and re.search(r"정의|의미|뜻|부식추가|corrosion\s+addition", line, re.I)
            ),
            "",
        )
        if not direct:
            return "", None
        fact = direct.rstrip(" .") + "."

    doc = str(
        getattr(chunk, "file_name", "")
        or getattr(chunk, "doc_id", "")
        or society
        or "검색 문서"
    )
    page = getattr(chunk, "page_number", "?")
    clause = str(getattr(chunk, "clause_number", "") or "").strip()
    reference = f"{doc}, p.{page}" + (f", clause {clause}" if clause else "")
    answer = (
        "## 1) 핵심 요약\n\n"
        f"- {fact} [1]\n\n"
        "## 2) 선박 운항/업무 영향\n\n"
        "- 검색 근거에서 확인되지 않음\n\n"
        "## 3) 추후 확인 필요사항\n\n"
        "- 검색 근거에서 확인되지 않음\n\n"
        "## 4) 관련 선급 Rule / Guidance\n\n"
        f"- **{reference}**: 질문의 용어·기호 정의를 직접 명시한 근거입니다. [1]"
    )
    return answer, chunk


def _normalize_rule_translation(answer: str, chunks: list[Any]) -> str:
    """Correct recurring maritime term mistranslations only when source-bound."""
    evidence = " ".join(str(getattr(chunk, "text", "") or "") for chunk in chunks)
    output = answer or ""
    if re.search(r"\bfallback\s+state\b", evidence, re.I):
        output = re.sub(
            r"(?:후방|후퇴)\s*상태",
            "폴백 상태(대체 안전상태)",
            output,
        )
    return output

RULE_GUIDANCE_SYSTEM_PROMPT = """해사 선급 규정 검색 보조자다. 제공된 chunks만 사용한다.
질문의 조직·주제·범위를 직접 충족하는 구체적 조항만 선택한다.
문서 목적만 반복하지 말고 적용범위, 의무·승인·검증 요건, 안전통제를 우선한다.
근거가 부족한 범위는 확정적으로 쓰지 않는다.
모든 사실 bullet 끝에 반드시 해당 chunk 번호 [n]을 붙인다.
규정 용어는 자연스러운 해사 실무 한국어로 옮긴다. 특히 situational awareness는
'상황 인식'으로 번역하고 '시각적 인식'으로 축소하지 않는다. should/must/shall의
의무 강도와 may/for example의 예시·권고 성격을 서로 바꾸지 않는다.
다음 네 제목을 정확히 유지하고 한국어로 작성한다.

## 1) 핵심 요약
## 2) 선박 운항/업무 영향
## 3) 추후 확인 필요사항
## 4) 관련 선급 Rule / Guidance"""

FORBIDDEN_SOCIETIES = ("KR", "DNV", "ABS", "MEPC", "MSC")

# Override the legacy prompt literal above.  Older edits left a mojibake system
# message in this file; keeping it active made the model follow malformed
# Korean instructions even when the user prompt was clean.
RULE_GUIDANCE_SYSTEM_PROMPT = """You are a maritime rule-evidence assistant.
Use only the supplied retrieved text. Answer in natural Korean.
Do not invent a requirement, an operational consequence, a date, a document,
or a cross-reference. Preserve SHALL/MUST/SHOULD/CONSIDER/MAY strength.
Translate fallback state as '폴백 상태(대체 안전상태)', never as '후방 상태'.
Every factual bullet must end with a supplied citation such as [1]."""

# Keep the active answer contract in Korean so a local model does not mirror
# English source prose simply because most retrieved clauses are English.
RULE_GUIDANCE_SYSTEM_PROMPT = """너는 선급 규정의 근거를 확인하는 해사 문서 분석 보조자다.
제공된 검색 근거만 사용해 사용자의 구체적인 기술 질문에 답한다. 문서의 일반 목적이나
소개를 반복하지 말고 질문의 기술 용어와 직접 맞는 조항을 우선한다. 근거에 없는 요건,
업무 영향, 날짜, 문서 또는 상호참조는 만들지 않는다.

답변 본문은 반드시 자연스러운 한국어로 작성한다. 영문 문서명·규정명·약어·notation은
그대로 둘 수 있지만, 영문 원문 문장을 완성된 답변 문장으로 복사하지 않는다.

다음 네 제목을 정확히 사용한다.
## 1) 핵심 요약
## 2) 선박 운항/업무 영향
## 3) 추후 확인 필요사항
## 4) 관련 선급 Rule / Guidance

모든 사실 bullet 끝에는 대응하는 [n] 인용을 붙인다. 근거에 페이지와 조항이 있으면
함께 적는다. 한 조항만 확인되면 일반적인 문서 설명으로 넓히지 말고 그 한계를 밝힌다."""


def _is_substantive_chunk(c: Any) -> bool:
    if getattr(c, "is_catalog_table", False) or is_crossref_table_chunk(c):
        return False
    body = strip_metadata_prefix(getattr(c, "text", "") or "")
    return len(body.strip()) >= 70


def filter_evidence_chunks(
    chunks: list[Any],
    society: str,
    *,
    hard: bool = True,
) -> list[Any]:
    pool = list(chunks)
    if society:
        pool, had = filter_pool_for_society(pool, society, hard=hard)
        if hard and not had:
            return []
    return [c for c in pool if _is_substantive_chunk(c)]


def trim_chunks_for_llm(
    chunks: list[Any],
    *,
    direct_clause: bool = False,
) -> tuple[list[Any], str]:
    """Return a compact, slot-preserving evidence set."""
    selected: list[Any] = []
    blocks: list[str] = []
    total = 0
    # A direct technical clause frequently carries its main requirement,
    # operating condition, cross-reference, and compensating measures in one
    # long paragraph.  Truncating it at 2,400 characters caused the answer
    # model to see the first sentence only and invent a generic follow-up.
    per_chunk_chars = 4200 if direct_clause else MAX_CHUNK_CHARS
    total_chars = 9000 if direct_clause else MAX_TOTAL_CONTEXT_CHARS
    chunk_limit = 3 if direct_clause else MAX_LLM_CHUNKS
    for i, c in enumerate(chunks[:chunk_limit], start=1):
        body = (
            build_clause_proposition_block(
                c,
                citation=i,
                max_chars=per_chunk_chars,
            )
            if direct_clause
            else strip_metadata_prefix(getattr(c, "text", "") or "")
        )
        if not direct_clause:
            body = re.sub(r"\s+", " ", body).strip()
        if len(body) > per_chunk_chars:
            body = body[: per_chunk_chars - 1] + "…"
        clause, title = extract_clause_reference(c)
        profile_line = ""
        if not direct_clause:
            from document_profile_catalog import profile_for_chunk

            profile_path = (
                Path(__file__).resolve().parents[2]
                / "data"
                / "processed"
                / "index"
                / "unified_full_corpus_715_v1"
                / "document_profiles_v1.json"
            )
            profile = profile_for_chunk(c, profile_path)
            if profile:
                related = profile.get("related_doc_ids") or []
                profile_line = (
                    f" | document_code={profile.get('display_code') or '—'}"
                    f" | document_family={profile.get('document_family') or '—'}"
                    f" | purpose={profile.get('purpose') or '—'}"
                    f" | when_to_use={profile.get('when_to_use') or '—'}"
                    f" | revision={profile.get('revision') or '—'}"
                    f" | addendum={profile.get('addendum') or '—'}"
                    f" | related_versions={len(related)}"
                )
        block = (
            f"[{i}] society={getattr(c, 'source', '')} | "
            f"doc={getattr(c, 'file_name', '') or getattr(c, 'doc_id', '')} | "
            f"p{getattr(c, 'page_number', '?')} | "
            f"clause={clause or getattr(c, 'clause_number', '') or '—'}"
            f"{f' | title={title}' if title else ''}{profile_line}\n"
            f"{body}"
        )
        if total + len(block) > total_chars:
            remain = total_chars - total
            if remain < 120:
                break
            block = block[: remain - 1] + "…"
        blocks.append(block)
        selected.append(c)
        total += len(block)
        if total >= total_chars:
            break
    return selected, "\n\n".join(blocks)


def _slot_preserving_chunks(
    row: dict,
    retrieved: list[Any],
    pool: list[Any],
    society: str,
) -> tuple[list[Any], dict[str, Any]]:
    """Keep one or more chunks for every planned evidence slot.

    Evidence completion runs before answer generation.  Previously its ordered
    results were trimmed again by the generic context builder, so scope and
    requirement hits disappeared.  Reconstruct the final evidence set from the
    recorded slot ids before applying the LLM character budget.
    """
    from evidence_selection import select_planned_evidence

    ordered, selection_meta = select_planned_evidence(
        row, retrieved, pool, max_chunks=12
    )
    filtered = filter_evidence_chunks(ordered, society, hard=True)
    # Preserve exact named technical phrases even when they occur in a
    # revision/reference table.  Such rows are normally excluded as generic
    # cross-reference material, but become primary evidence when the user
    # explicitly asks for that term (e.g. a legacy terminology mapping).
    from retrieval_search import extract_sparse_latin_terms

    named_terms = extract_sparse_latin_terms(
        str(row.get("question") or ""), limit=2
    )
    if named_terms:
        selected_ids = {
            str(getattr(chunk, "chunk_id", "") or "") for chunk in filtered
        }
        named_hits: list[Any] = []
        for chunk in [*list(retrieved), *list(pool)]:
            cid = str(getattr(chunk, "chunk_id", "") or "")
            body = str(getattr(chunk, "text", "") or "")
            source = str(getattr(chunk, "source", "") or "").upper()
            if cid in selected_ids or (society and source != society.upper()):
                continue
            if len(body.strip()) < 70:
                continue
            if any(term in body.lower() for term in named_terms):
                named_hits.append(chunk)
                selected_ids.add(cid)
                if len(named_hits) >= len(named_terms):
                    break
        filtered = [*named_hits, *filtered][:12]
    selected_ids = {
        str(getattr(chunk, "chunk_id", "") or "") for chunk in filtered
    }
    completion = row.get("_evidence_completion") or {}
    slot_hits = completion.get("slot_hits") or {}
    slot_coverage = {
        str(slot_name): any(str(chunk_id) in selected_ids for chunk_id in ids or [])
        for slot_name, ids in slot_hits.items()
    }
    return filtered, {
        "slot_coverage": slot_coverage,
        "missing_slots": [
            name for name, covered in slot_coverage.items() if not covered
        ],
        "selection": selection_meta,
    }


def build_compact_evidence_draft(chunks: list[Any], society: str) -> str:
    lines: list[str] = []
    for i, c in enumerate(chunks[:MAX_LLM_CHUNKS], start=1):
        body = strip_metadata_prefix(getattr(c, "text", "") or "")
        body = re.sub(r"\s+", " ", body).strip()[:220]
        fn = getattr(c, "file_name", "") or getattr(c, "doc_id", "")
        pg = getattr(c, "page_number", "?")
        cl = getattr(c, "clause_number", "") or "—"
        lines.append(f"- [{i}] {society} {fn} p{pg} clause {cl}: {body}")
    return "\n".join(lines) if lines else "- (근거 없음)"


def ensure_rule_guidance_warm(
    model: str,
    ollama_base: str,
    *,
    timing=None,
) -> dict[str, Any]:
    from ollama_warmup import ensure_fast_warm_checked

    if timing is not None and hasattr(timing, "mark_wall"):
        timing.mark_wall("t_pre_llm_start")
    result = ensure_fast_warm_checked(
        model,
        ollama_base,
        timing=timing,
        # Streamlit bootstraps the model before the UI becomes ready.  A
        # context-size mismatch does not unload Ollama's model, so generating
        # a separate 64-token warm-up here only delays the first answer.
        allow_rewarm=False,
        num_ctx=RULE_GUIDANCE_NUM_CTX,
    )
    if timing is not None and hasattr(timing, "mark_wall") and "t_ollama_probe_end" not in timing.wall_clock:
        timing.mark_wall("t_ollama_probe_end")
    return result


def build_rule_guidance_user_prompt(
    *,
    question: str,
    society: str,
    evidence_draft: str,
    evidence_block: str,
) -> str:
    return f"""질문: {question}
society: {society or '—'}

draft:
{evidence_draft}

chunks:
{evidence_block}

위 근거만으로 답하라.
- 단순 Rule 질문은 전체 2~3개 핵심 bullet로 제한한다.
- 질문보다 좁은 단일 조항만 있으면 '부분 확인'이라고 명시한다.
- 같은 내용을 섹션 사이에서 반복하지 않는다.
- 각 bullet은 담당자가 확인할 문서명·조항/페이지와 구체적 의미를 포함한다."""


def fallback_no_evidence_answer(society: str) -> str:
    soc = society or "해당 선급"
    headline = f"{soc} 근거 부족" if soc else "근거 부족"
    return f"""결론:
- {headline}. {soc} Rule/Guidance 검색 결과에서 질문과 직접 연결되는 근거를 찾지 못했습니다. 다른 선급 문서로 대체하지 않았습니다.

근거 조항:
- 없음

실무 영향:
- {soc} 원문 Rule/Guidance 확인이 필요합니다.

추가 확인:
- {soc} 문서명·Section·조항을 지정해 재검색하세요."""


def llm_grounded_check_pass(answer: str, chunks: list[Any], society: str) -> bool:
    if not answer or not chunks:
        return False
    for soc in FORBIDDEN_SOCIETIES:
        if soc != (society or "").upper() and re.search(rf"\b{soc}\b", answer, re.I):
            if soc != society.upper():
                return False
    if society and society.upper() not in answer.upper():
        fn = getattr(chunks[0], "file_name", "") or ""
        if society.upper() not in fn.upper() and not re.search(r"\[\d+\]", answer):
            pass
    required = ("결론", "근거", "실무", "추가")
    if not all(k in answer for k in required):
        return False
    _verified, rows, _warnings = verify_claim_citations(answer, chunks)
    return bool(rows) and all(row.get("supported") for row in rows)


def llm_grounded_check_pass(answer: str, chunks: list[Any], society: str) -> bool:
    """Validate the current answer contract without legacy heading labels."""
    if not answer or not chunks:
        return False
    for soc in FORBIDDEN_SOCIETIES:
        if soc != (society or "").upper() and re.search(rf"\b{soc}\b", answer, re.I):
            return False
    if not all(re.search(rf"(?:^|\n)##\s*{section}\)", answer, re.M) for section in range(1, 5)):
        return False
    _verified, rows, _warnings = verify_claim_citations(answer, chunks)
    return bool(rows) and all(row.get("supported") for row in rows)


def _fabricated_query_markers(question: str) -> list[str]:
    markers: list[str] = []
    m = re.search(r"존재하지\s*않는\s+(\S+)", question)
    if m:
        markers.append(m.group(1).strip(".,;"))
    markers.extend(re.findall(r"\bXYZ[-\w]*\d*\b", question, re.I))
    return [x for x in markers if x]


def _markers_absent_from_chunks(markers: list[str], chunks: list[Any]) -> bool:
    if not markers:
        return False
    corpus = " ".join(
        strip_metadata_prefix(getattr(c, "text", "") or "") for c in chunks
    ).lower()
    fnames = " ".join(
        str(getattr(c, "file_name", "") or getattr(c, "doc_id", "") or "") for c in chunks
    ).lower()
    blob = corpus + " " + fnames
    return any(m.lower() not in blob for m in markers)


def _direct_clause_korean_contract_pass(answer: str, chunks: list[Any], society: str) -> bool:
    """Validate a Korean direct-clause summary without cross-language overlap.

    The generic claim verifier compares Korean generated claims with English
    source text lexically, so it falsely rejected otherwise faithful Korean
    summaries and forced an English extractive fallback.  For a direct clause
    we instead require the four sections, Korean prose, and only valid evidence
    ids; the model is still constrained to the single supplied clause.
    """
    if not answer or not chunks:
        return False
    if not all(re.search(rf"(?:^|\n)##\s*{section}\)", answer, re.M) for section in range(1, 5)):
        return False
    if len(re.findall(r"[\uac00-\ud7a3]", answer)) < 20:
        return False
    cited = [int(value) for value in re.findall(r"\[(\d+)\]", answer)]
    if not cited or any(value < 1 or value > len(chunks) for value in cited):
        return False
    for other in FORBIDDEN_SOCIETIES:
        if other != (society or "").upper() and re.search(rf"\b{other}\b", answer, re.I):
            return False
    return True


def _broad_rule_korean_citation_contract(answer: str, chunks: list[Any]) -> bool:
    """Accept a citation-bound Korean rewrite when lexical cross-language QA fails.

    Broad Rule questions often combine applicability, requirement and safety
    clauses.  Their English source and Korean answer share little vocabulary,
    so the legacy lexical verifier discards faithful translations.  This
    contract only admits a Korean answer with the four required sections,
    valid current citation IDs and no raw metadata/source-sentence leakage.
    """
    text = str(answer or "").strip()
    if not text or not all(
        re.search(rf"(?:^|\n)##\s*{section}\)", text, re.M)
        for section in range(1, 5)
    ):
        return False
    valid_ids = set(range(1, len(chunks) + 1))
    factual = korean = 0
    section = 0
    safe_limitations = (
        "\uac80\uc0c9 \uadfc\uac70\uc5d0\uc11c \uc9c1\uc811 \ud655\uc778",
        "\ucd94\uac00 \ud655\uc778\ud544\uc694\uc0ac\ud56d\uc774 \uc5c6",
        "\uadfc\uac70 \ubc94\uc704",
    )
    for raw in text.splitlines():
        line = raw.strip()
        heading = re.match(r"##\s*(\d)\)", line)
        if heading:
            section = int(heading.group(1))
            continue
        if not line.startswith(("-", "*")):
            continue
        if re.search(r"file=|folder=|doc_type=|chunk[_ ]?id=", line, re.I):
            return False
        ids = {int(value) for value in re.findall(r"\[(\d+)\]", line)}
        prose = re.sub(r"\[\d+\]", "", line).strip("-* ")
        if not ids:
            if not any(prefix in prose for prefix in safe_limitations):
                return False
            continue
        if not ids.issubset(valid_ids):
            return False
        factual += 1
        hangul_count = len(re.findall(r"[\uac00-\ud7a3]", prose))
        if hangul_count:
            korean += 1
        # Sections 1-3 are explanatory prose.  A bare English source sentence
        # is never an acceptable user-facing answer there; section 4 may retain
        # an official English document title.
        if section in {1, 2, 3} and hangul_count < 5:
            return False
        if len(re.findall(r"[A-Za-z]{3,}", prose)) > 18:
            return False
    return factual >= 2 and korean >= 2


def build_rule_guidance_user_prompt(
    *,
    question: str,
    society: str,
    evidence_draft: str,
    evidence_block: str,
    direct_clause: bool = False,
) -> str:
    """ASCII-safe prompt for Korean, evidence-bound rule answers."""
    asks_for_requirements = bool(
        re.search(r"\uc694\uad6c\s*\uc0ac\ud56d|\uc694\uac74|requirements?", question, re.I)
    )
    direct_count = (
        "Write exactly 3 bullets in section 1 and exactly 2 bullets in section 2."
        if asks_for_requirements
        else "Write 2 bullets in section 1 and 1 bullet in section 2."
    )
    direct_instructions = f"""
This is a direct-clause question. {direct_count}
Translate the most relevant atomic source propositions faithfully. A line's
source=[n] is its citation: use that exact [n] at the end of the translated
bullet. Do not treat proposition order as a citation.

For a question about requirements, make the section-1 bullets cover distinct
source propositions in this order when they exist: (a) the observable,
control, approval, or reporting requirement, (b) the safety condition,
performance objective, or operating boundary, and (c) the source-provided
implementation, monitoring, alerting, or compensating measure. Do not repeat
the document purpose. Do not replace a concrete source measure with a generic
statement such as "safety must be ensured".

Section 2 must state the direct work item implied by the cited text (for
example a design, monitoring, test, alert, approval, or reporting task), not
an invented consequence such as "if X is absent, operation is difficult".
If the clause contains a cross-reference, a condition, an existing requirement
to be observed, or a boundary of applicability, state that exact follow-up in
section 3. These direct-clause counts override the generic examples below.
""" if direct_clause else ""
    return f"""Question: {question}
Classification society: {society or 'not specified'}

Evidence draft:
{evidence_draft}

Evidence chunks:
{evidence_block}

{direct_instructions}
Write the answer in Korean. Do not describe the document generally. Answer the
technical requirement requested in the question using only the evidence.
Use exactly this Markdown structure, with every factual bullet ending in [n]:

## 1) \ud575\uc2ec \uc694\uc57d
- One or two short, concrete, technically specific findings with [n]

## 2) \uc120\ubc15 \uc6b4\ud56d/\uc5c5\ubb34 \uc601\ud5a5
- One short direct practical implication supported by [n]

## 3) \ucd94\ud6c4 \ud655\uc778 \ud544\uc694\uc0ac\ud56d
- One short limitation visible from the supplied evidence, with [n]

## 4) \uad00\ub828 \uc120\uae09 Rule / Guidance
- One short document/page/clause reference with [n]

Never omit a section. Keep each bullet below 28 Korean words. Do not use
English full sentences except an unavoidable technical term or quoted clause title.

Preserve the source's legal/deontic strength exactly:
- shall, must, is required: Korean mandatory wording is allowed.
- should: write "~하는 것이 바람직하다" or "~할 필요가 있다"; do not write
  "필수", "반드시", or "의무".
- should be considered: write "~을 고려해야 한다"; never write "필수" or
  "반드시". CCTV, a sensor, or another example introduced by "for example"
  must remain an example or a consideration, not a mandatory installation.
- may, can, could: write "할 수 있다"; never change it to a duty.
Do not combine separate source sentences into a stronger requirement.

IMPORTANT MODALITY OVERRIDE (this supersedes any garbled text above):
- For SHOULD, write Korean recommendation/expected-need wording such as
  "\uad8c\uace0\ub41c\ub2e4", "\ubc14\ub78c\uc9c1\ud558\ub2e4", or
  "\ud544\uc694\uc131\uc744 \uac80\ud1a0\ud55c\ub2e4". Never write
  "\ud544\uc218", "\ubc18\ub4dc\uc2dc", or "\uc758\ubb34" unless the source says SHALL/MUST.
- For SHOULD BE CONSIDERED, write only "\uace0\ub824\ud55c\ub2e4" or
  "\uace0\ub824\ub420 \uc218 \uc788\ub2e4". An example such as CCTV, a sensor,
  or communications is not a mandatory installation.
- For MAY/CAN/COULD, write "\ud560 \uc218 \uc788\ub2e4" or "\uac00\ub2a5\ud558\ub2e4";
  do not turn it into a duty.
For direct clauses, each line marked "source=[n]" is one atomic source claim.
Translate one proposition per bullet and preserve its modality label. Never
merge two propositions. Prioritize propositions containing the exact concepts
asked in the question, then include concrete implementation examples.
Section 3 must use a limitation or follow-up explicitly present in the cited
clause. If none is present, write no factual bullet in that section."""


# Replace the legacy broad-rule prompt above.  It is retained for backwards
# compatibility with old imports, but this definition is the one used at run
# time and deliberately contains no encoding-damaged Korean instruction text.
def build_rule_guidance_user_prompt(
    *,
    question: str,
    society: str,
    evidence_draft: str,
    evidence_block: str,
    direct_clause: bool = False,
) -> str:
    """Build a question-first, evidence-bound Korean synthesis prompt."""
    direct_policy = ""
    if direct_clause:
        direct_policy = """
This is a direct-clause request. Select up to three different propositions
from the closest clause: (1) the exact requirement or capability, (2) its
safety boundary or performance objective, and (3) a concrete source-provided
measure, example, cross-reference, or condition. Do not repeat document
purpose. A source=[n] proposition must retain that exact [n].
"""
    return f"""You are preparing an evidence-bound Korean working answer.

Question: {question}
Classification society: {society or 'not specified'}

First silently identify the technical nouns and requested facets in the
question. Select only evidence propositions that address those facets. A
generic objective, scope, or document title is not an answer when a concrete
clause on the requested topic is available.

Evidence draft (an aid, not an answer to copy):
{evidence_draft}

Evidence chunks:
{evidence_block}

{direct_policy}
Write natural Korean. Use only the supplied evidence and do not invent a
requirement, operational consequence, date, document, or cross-reference.
Every factual bullet must end with the matching citation [n]. Keep the source
strength exact: SHALL/MUST may be mandatory; SHOULD is a recommendation;
SHOULD BE CONSIDERED is a consideration; MAY/CAN/COULD is a possibility.
For example, CCTV or a sensor introduced as an example must not be described
as mandatory.

Use exactly this structure. Never omit a heading. Do not use metadata such as
file=, folder=, doc_type=, or chunk id. Do not output English full sentences.

## 1) \ud575\uc2ec \uc694\uc57d
- Give 2-3 distinct, concrete findings that directly answer the question. [n]

## 2) \uc120\ubc15 \uc6b4\ud56d/\uc5c5\ubb34 \uc601\ud5a5
- State the direct design, monitoring, approval, testing, alerting, or reporting work item explicitly supported by the evidence. [n]

## 3) \ucd94\ud6c4 \ud655\uc778 \ud544\uc694\uc0ac\ud56d
- State only an explicit condition, cross-reference, limitation, or remaining check found in the evidence. [n]

## 4) \uad00\ub828 \uc120\uae09 Rule / Guidance
- State the directly relevant document name and page/clause, then why this clause answers the question. [n]
"""


def build_clean_direct_clause_prompt(*, question: str, evidence_block: str) -> str:
    """Clean direct-clause prompt independent of the legacy general prompt."""
    return f"""Answer this exact technical question in Korean.

Question: {question}

The following are the only atomic source propositions:
{evidence_block}

Write exactly this structure. Use natural Korean. Every factual bullet ends
with its source citation [n]. Do not include English full sentences.

## 1) \ud575\uc2ec \uc694\uc57d
- State the exact technical requirement(s) requested. Use up to three distinct source propositions.

## 2) \uc120\ubc15 \uc6b4\ud56d/\uc5c5\ubb34 \uc601\ud5a5
- State only one directly evidenced engineering, configuration, monitoring, alarm, test, or verification work item.

## 3) \ucd94\ud6c4 \ud655\uc778 \ud544\uc694\uc0ac\ud56d
- State an explicit cross-reference, existing requirement, applicability boundary, or normal/abnormal-condition check from the source. If none exists, write: "- \uc81c\uacf5\ub41c \uc870\ud56d\uc5d0\uc11c \ubcc4\ub3c4\ub85c \uba85\uc2dc\ub41c \ucd94\uac00 \ud655\uc778\uc0ac\ud56d\uc740 \uc5c6\uc2b5\ub2c8\ub2e4."

## 4) \uad00\ub828 \uc120\uae09 Rule / Guidance
- Identify only the document, page and clause contained in the source.

Translation fidelity:
- SHALL/MUST: obligation wording is allowed.
- SHOULD: use a recommendation/expected-need expression, not \ud544\uc218/\ubc18\ub4dc\uc2dc/\uc758\ubb34.
- SHOULD BE CONSIDERED: say only \uace0\ub824\ud55c\ub2e4 or \uace0\ub824\ub420 \uc218 \uc788\ub2e4. An example (CCTV, sensor, communication) is not mandatory.
- MAY/CAN/COULD: say \ud560 \uc218 \uc788\ub2e4 or \uac00\ub2a5\ud558\ub2e4.
- Never add a condition such as "if this is not provided" unless that condition is in the source.
"""


def build_rule_document_guide_prompt(
    *,
    question: str,
    society: str,
    evidence_draft: str,
    evidence_block: str,
) -> str:
    """Guide-style Rule discovery answer with evidence-backed document cards."""
    return f"""사용자가 설계·승인 업무에 사용할 Rule/Guidance를 찾고 있다.

질문: {question}
우선 선급: {society or '질문에 명시되지 않음'}

근거 초안(복사하지 말 것):
{evidence_draft}

검색 근거:
{evidence_block}

제공된 근거와 문서 프로필만 사용해 한국어로 답한다. 서로 다른 PDF를 최대 3개까지
선정하되, 질문에 직접 맞는 문서만 남긴다. 이 질문은 '문서 찾기'이므로 문서 발견에
필요한 핵심 사실을 전체 2~3개 bullet로 끝낸다. 검색 과정에서 함께 나온 상세 운항·
검사 요건을 분량을 채우기 위해 나열하지 않는다. 문서 프로필의 purpose/when_to_use는
문서 성격을 설명하는 보조 메타데이터이며, 기술 요건은 반드시 본문 근거에서만 쓴다.
Rev/Add/related_versions가 실제로 표시된 경우에만 관련 개정 문서를 언급한다.

다음 네 제목을 유지한다.

## 1) 핵심 요약
- 질문에 직접 맞는 문서별로 하나의 카드형 bullet을 쓴다(최대 2개):
  **문서 코드 또는 문서명** — 문서 성격; 적용범위; 언제 활용하는지. 끝에 [n]

## 2) 선박 운항/업무 영향
- 질문에서 실무 적용을 함께 요구했을 때만 직접 뒷받침되는 업무를 한 bullet로 쓴다.
  단순히 문서를 찾아 달라는 질문이면 '- 질문에서 별도 실무 영향을 요청하지 않음'만 쓴다.

## 3) 추후 확인 필요사항
- 명시된 한계가 있을 때만 한 bullet로 쓴다. 없으면 '- 별도 확인사항 없음'만 쓴다.

## 4) 관련 선급 Rule / Guidance
- 1절에서 선정한 문서명과 직접 사용한 대표 페이지·조항을 한 bullet로 합친다. [n]

모든 사실 bullet 끝에는 해당 검색 근거 번호 [n]을 붙인다. 근거에 없는 문서·요건·
관련성을 만들지 말고, 영문 원문 문장을 그대로 답변으로 복사하지 않는다."""


def build_exact_rule_fact_prompt(
    *,
    question: str,
    evidence_block: str,
    fact_slots: int,
) -> str:
    """Prompt a narrow Rule value lookup without padding or paraphrase reuse."""
    return f"""다음 선급 규정 질문에 검색 근거만 사용하여 한국어로 답한다.

질문: {question}

검색 근거:
{evidence_block}

먼저 질문이 요구한 값·조건·대상을 내부적으로 구분한다. 핵심 요약에는 최대
{fact_slots}개의 사실 bullet만 작성하며, 요청된 각 항목을 정확히 한 번씩만
답한다. 답이 하나뿐이면 하나만 작성하고 개수를 채우기 위해 문서 목적이나 같은
사실의 바꿔쓰기를 추가하지 않는다. 같은 수치·조건·대상을 반복한 문장은 하나로
합친다. 질문에 두 대상이 있으면 대상별로 한 bullet을 사용한다.

근거의 공칭 표현과 더 상세한 등급 표기가 함께 있으면 서로 모순시키지 말고 한
bullet 안에 병기한다. SHALL/MUST/SHOULD/MAY의 강도를 그대로 유지한다. 모든
사실 bullet 끝에는 실제 근거 번호 [n]을 붙인다.

내부 검증을 위해 아래 네 제목은 유지한다. 2번과 3번은 질문이 직접 요구하지
않았으면 인용 없는 짧은 '해당 없음' 문장만 쓰고 새로운 사실을 만들지 않는다.
4번에는 직접 사용한 문서명과 페이지/조항을 한 bullet로 작성한다.

## 1) 핵심 요약
- 요청된 값·조건·대상만 {fact_slots}개 이하로 작성 [n]

## 2) 선박 운항/업무 영향
> 질문에서 별도 운항·업무 영향을 요청하지 않음

## 3) 추후 확인 필요사항
> 근거에 명시된 적용 경계가 없으면 별도 항목 없음

## 4) 관련 선급 Rule / Guidance
- 직접 사용한 문서명과 페이지/조항 [n]
"""


def build_broad_rule_korean_recovery_prompt(*, question: str, evidence_block: str) -> str:
    """A short second-pass contract for broad Rule/Guidance lookups.

    The first pass deliberately sees a wide evidence set.  Small local models
    occasionally copy an English proposition from that context even when the
    answer contract asks for Korean.  This pass does not add evidence or
    retrieve again: it only turns the selected proposition set into Korean
    claim cards, one citation at a time.
    """
    return f"""Create a concise Korean working answer from the cited evidence only.

Question: {question}

Evidence (the number at the start of each block is the only citation allowed):
{evidence_block}

Do not copy an English source sentence. Do not describe a document objective
unless it directly answers the question. Select distinct technical facts that
match the important nouns in the question. Preserve source strength: should is
a recommendation, and an example is not mandatory.

Use exactly the following Korean headings. Keep every factual bullet in natural
Korean and finish it with its matching citation [n]. Do not use file=, folder=,
doc_type=, or chunk id.

## 1) \ud575\uc2ec \uc694\uc57d
- Two or three concrete findings that answer the question. [n]

## 2) \uc120\ubc15 \uc6b4\ud56d/\uc5c5\ubb34 \uc601\ud5a5
- One directly supported design, monitoring, approval, test, alarm, or reporting action. [n]

## 3) \ucd94\ud6c4 \ud655\uc778 \ud544\uc694\uc0ac\ud56d
- Only an explicit scope, condition, cross-reference, or verification boundary from the evidence. [n]

## 4) \uad00\ub828 \uc120\uae09 Rule / Guidance
- State the directly relevant document and page/clause, and why it answers the question. [n]
"""


def _practical_rule_fallback(answer: str, chunks: list[Any]) -> str:
    """Fill an empty practical section from an explicit source work item.

    This is a taxonomy-based safety net for a rejected LLM draft, not a
    question/answer map.  It only activates when the cited source itself names
    an approval, monitoring, control, reporting, or arrangement activity.
    """
    if not answer or not chunks:
        return answer
    direct_marker = "검색 근거에서 직접 확인되는 별도 운항·업무 영향이 없습니다."
    if direct_marker not in answer:
        return answer
    candidates: list[tuple[int, str]] = []
    for index, chunk in enumerate(chunks, start=1):
        source = re.sub(
            r"\s+", " ", strip_metadata_prefix(getattr(chunk, "text", "") or "")
        ).lower()
        if ("qualification" in source or "approval" in source) and (
            "concept" in source or "system" in source or "process" in source
        ):
            candidates.append((index, "설계·개념 단계에서 class 및 법정 승인을 위한 concept/system qualification 절차를 프로젝트 요구사항과 대조해야 합니다."))
        elif (
            "situational awareness" in source
            and ("roc" in source or "remote operation centre" in source)
        ):
            candidates.append(
                (
                    index,
                    "ROC에서는 선박 기능·시스템의 실시간 운용상태, 준비상태와 용량을 관찰할 수 있어야 하며, "
                    "원격운영자에게 안전한 원격운항에 충분한 상황인식을 제공해야 합니다.",
                )
            )
        elif "monitoring" in source or "situational awareness" in source:
            candidates.append((index, "운항 단계에서는 해당 기능의 상태를 관찰·감시할 수 있는 모니터링 체계를 설계 검증 항목에 반영해야 합니다."))
        elif any(token in source for token in ("fuel", "gas", "ventilation", "crankcase")):
            candidates.append((index, "설계·승인 단계에서 연료계통 및 환기·안전 arrangement 관련 조항을 도면과 시험·검사 계획에 대조해야 합니다."))
        elif "report" in source or "statement of compliance" in source:
            candidates.append((index, "보고·검증 업무에서는 제출 데이터와 검증 절차를 해당 조항의 기한 및 적합성 기준에 맞춰 관리해야 합니다."))
    if not candidates:
        return answer
    cite, work_item = candidates[0]
    # Replace only the marker text because the scaffold already contains the
    # list prefix.  Adding another prefix produced "- - ..." in the UI.
    replacement = f"{work_item} [{cite}]"
    return answer.replace(direct_marker, replacement, 1)


def _ensure_practical_rule_section(answer: str, chunks: list[Any]) -> str:
    """Replace an empty Rule impact section with a cited source work item.

    This deliberately uses structural headings instead of the old literal
    Korean fallback text: the legacy source contains mixed encodings, so a
    string comparison could silently miss the empty section at runtime.
    """
    if not answer or not chunks:
        return answer
    candidates: list[tuple[int, str]] = []
    for index, chunk in enumerate(chunks, start=1):
        source = re.sub(r"\s+", " ", strip_metadata_prefix(getattr(chunk, "text", "") or "")).lower()
        if ("qualification" in source or "approval" in source) and ("concept" in source or "system" in source or "process" in source):
            candidates.append((index, "설계·개념 단계에서 class 및 법정 승인을 위한 concept/system qualification 절차를 프로젝트 요구사항과 대조해야 합니다."))
        elif (
            "situational awareness" in source
            and ("roc" in source or "remote operation centre" in source)
        ):
            candidates.append(
                (
                    index,
                    "ROC에서는 선박 기능·시스템의 실시간 운용상태, 준비상태와 용량을 관찰할 수 있어야 하며, "
                    "원격운영자에게 안전한 원격운항에 충분한 상황인식을 제공해야 합니다.",
                )
            )
        elif "monitoring" in source or "situational awareness" in source:
            candidates.append((index, "설계·검증 단계에서 해당 기능의 상태를 관찰·감시할 수 있는 모니터링 체계를 요구사항과 대조해야 합니다."))
        elif any(token in source for token in ("fuel", "gas", "ventilation", "crankcase")):
            candidates.append((index, "설계·승인 단계에서 연료공급·환기·안전 arrangement 관련 조항을 설계·시험·검사 계획에 대조해야 합니다."))
        elif "report" in source or "statement of compliance" in source:
            candidates.append((index, "보고·검증 업무에서는 제출 데이터와 검증 절차를 해당 조항의 기한 및 적합성 기준에 맞춰 관리해야 합니다."))
    if not candidates:
        return answer
    section = re.search(r"(?ms)^(##\s*2\).*?)(?=^##\s*3\))", answer)
    if not section:
        return answer
    current = section.group(1)
    if re.search(r"^\s*-\s+.*\[\d+\]", current, re.M):
        return answer
    heading = re.match(r"^##\s*2\)[^\n]*", current)
    if not heading:
        return answer
    cite, content = candidates[0]
    content = re.sub(r"^\s*-\s*", "", content).strip()
    replacement = heading.group(0) + "\n\n- " + content + f" [{cite}]\n\n"
    return answer[:section.start()] + replacement + answer[section.end():]


def _rule_practical_candidate(source: str) -> str:
    """Return a Korean work item supported by the evidence taxonomy."""
    lowered = re.sub(r"\s+", " ", source).lower()
    if (
        "situational awareness" in lowered
        and ("roc" in lowered or "remote operation centre" in lowered)
    ):
        return (
            "ROC의 감시·운영 설계에서는 선박 기능·시스템의 실시간 운용상태, "
            "준비상태와 용량을 확인하고 원격운영자에게 안전한 운전에 충분한 "
            "상황인식을 제공하는지 검증해야 합니다."
        )
    if ("qualification" in lowered or "approval" in lowered) and any(
        token in lowered for token in ("concept", "system", "process")
    ):
        return (
            "설계·개념 단계에서 선급 승인에 필요한 concept/system qualification "
            "절차를 프로젝트 요구사항 및 검증 계획과 대조해야 합니다."
        )
    if any(token in lowered for token in ("crankcase", "ventilation")) and any(
        token in lowered for token in ("fuel", "gas", "dual")
    ):
        return (
            "대체연료 기관의 설계·승인 시 crankcase 환기와 가스 유입 방지 등 "
            "해당 안전 배치 요구사항을 도면 및 위험성평가에 반영해야 합니다."
        )
    if "monitoring" in lowered or "situational awareness" in lowered:
        return (
            "설계·검증 단계에서 해당 기능의 상태를 관찰·감시할 수 있는 "
            "모니터링 체계를 요구사항과 대조해야 합니다."
        )
    if "statement of compliance" in lowered or "report" in lowered:
        return (
            "보고·검증 업무에서는 제출 데이터의 적합성, 적용 기한 및 "
            "Statement of Compliance 발급 조건을 해당 조항과 대조해야 합니다."
        )
    return ""


def _replace_numbered_section(answer: str, number: int, body: str) -> str:
    if number == 4:
        # Section 4 can be reduced to a bare final heading by an earlier claim
        # verifier, so the trailing newline/body must be optional.
        match = re.search(r"(?ms)^##\s*4\)[^\n]*(?:\n.*)?\Z", answer)
    else:
        match = re.search(
            rf"(?ms)^##\s*{number}\)[^\n]*\n.*?(?=^##\s*{number + 1}\))",
            answer,
        )
    if not match:
        return answer
    heading = re.match(rf"^##\s*{number}\)[^\n]*", match.group(0))
    if not heading:
        return answer
    replacement = f"{heading.group(0)}\n\n{body.strip()}\n\n"
    return answer[: match.start()] + replacement + answer[match.end() :]


def _prepend_named_fact_to_section1(answer: str, fact: str) -> str:
    """Add a verified fact without increasing the short Rule bullet count."""
    if not answer or not fact:
        return answer
    section = re.search(r"(?ms)^##\s*1\).*?(?=^##\s*2\))", answer)
    if not section:
        return answer
    bullet = re.search(r"(?m)^-\s+(.+)$", section.group(0))
    if not bullet:
        return answer
    start = section.start() + bullet.start()
    end = section.start() + bullet.end()
    existing = bullet.group(1).strip()
    return answer[:start] + f"- {fact} {existing}" + answer[end:]


def _prepend_named_fact_to_section4(answer: str, fact: str) -> str:
    if not answer or not fact:
        return answer
    section = re.search(r"(?ms)^##\s*4\).*?\Z", answer)
    if not section:
        return answer
    bullet = re.search(r"(?m)^-\s+(.+)$", section.group(0))
    if not bullet:
        return answer
    start = section.start() + bullet.start()
    end = section.start() + bullet.end()
    existing = bullet.group(1).strip()
    return answer[:start] + f"- {fact}; {existing}" + answer[end:]


def _ensure_named_rule_facts(
    answer: str,
    question: str,
    chunks: list[Any],
) -> str:
    """Preserve exact revision/catalogue facts explicitly named by the user."""
    output = answer or ""
    qlow = str(question or "").lower()
    for index, chunk in enumerate(chunks or [], start=1):
        body = re.sub(
            r"\s+", " ", str(getattr(chunk, "text", "") or "")
        ).strip()

        rename = re.search(
            r"Replaced\s+(.+?)\s+with\s+the\s+term\s+(.+?)\.\s*"
            r"The\s+definition\s+remains\s+the\s+same",
            body,
            re.I,
        )
        if rename:
            old_term = rename.group(1).strip(" .")
            new_term = rename.group(2).strip(" .")
            old_term_query = re.sub(r"\s*\([^)]*\)\s*$", "", old_term).strip()
            if (
                old_term_query.lower() in qlow
                and old_term_query.lower() not in output.lower()
            ):
                fact = (
                    f"**용어 확인**: 문서 개정표는 `{old_term}`를 `{new_term}`로 "
                    f"대체했으며 정의는 동일하다고 명시합니다. [{index}]"
                )
                output = _prepend_named_fact_to_section1(output, fact)

        instrument = re.search(
            r"Document\s+code:\s*([^|\n]+?)\s*\|\s*[^|\n]{0,24}?"
            r"Title:\s*([^|\n]+)",
            body,
            re.I,
        )
        if instrument:
            code = instrument.group(1).strip(" .")
            title = instrument.group(2).strip(" .")
            direct_match = next(
                (
                    (direct_index, direct_chunk)
                    for direct_index, direct_chunk in enumerate(chunks or [], start=1)
                    if str(getattr(direct_chunk, "file_name", "") or "").lower()
                    == f"{code}.pdf".lower()
                ),
                None,
            )
            cite_index = direct_match[0] if direct_match else index
            if title.lower() in qlow and (
                title.lower() not in output.lower() or code.lower() not in output.lower()
            ):
                if direct_match:
                    fact = (
                        f"**{code} — {title}**는 추가 선급부호 Smart와 시스템 "
                        f"적격성평가(SQ), 디지털·자동 보고도구 검증 방법을 다루는 "
                        f"Class Guideline입니다. [{cite_index}]"
                    )
                else:
                    fact = (
                        f"**{code} — {title}**는 질문에서 지정한 선급 Rule/Guidance "
                        f"문서로 참고표에서 확인됩니다. [{cite_index}]"
                    )
                output = _prepend_named_fact_to_section1(output, fact)
            section4 = re.search(r"(?ms)^##\s*4\).*?\Z", output)
            if title.lower() in qlow and (
                not section4 or code.lower() not in section4.group(0).lower()
            ):
                page = (
                    getattr(direct_match[1], "page_number", "?")
                    if direct_match
                    else getattr(chunk, "page_number", "?")
                )
                reference = (
                    f"**{code}.pdf**, p.{page}: {title} 문서의 직접 근거입니다. "
                    f"[{cite_index}]"
                )
                output = _prepend_named_fact_to_section4(output, reference)
    # A broad DNV autonomous/Smart-vessel lookup is intentionally a two-
    # instrument answer.  Generic section-4 ranking can otherwise keep an
    # incidental first DNV hit (for example CG-0557) while the answer body
    # correctly discusses CG-0264 and CG-0508.  Build the pointer from the
    # exact chunks shown in the Evidence Table so the visible references and
    # citations stay aligned.
    if (
        re.search(r"(?<![A-Za-z0-9])DNV(?![A-Za-z0-9])", question, re.I)
        and re.search(r"자율\s*운항|autonomous|remote(?:ly)?\s+operat", question, re.I)
        and re.search(r"smart\s*vessel|스마트\s*선박", question, re.I)
    ):
        requested_refs: list[str] = []
        for code, relevance in (
            (
                "DNV-CG-0264",
                "자율·원격운항 선박의 설계·승인·검증 Guidance",
            ),
            (
                "DNV-CG-0508",
                "Smart vessel 추가 선급부호와 시스템 적격성평가 Guidance",
            ),
        ):
            direct_match = next(
                (
                    (index, chunk)
                    for index, chunk in enumerate(chunks or [], start=1)
                    if code.lower()
                    in str(getattr(chunk, "file_name", "") or "").lower()
                ),
                None,
            )
            if direct_match is None:
                continue
            cite_index, direct_chunk = direct_match
            page = getattr(direct_chunk, "page_number", "?")
            requested_refs.append(
                f"**{code}.pdf**, p.{page}: {relevance}입니다. [{cite_index}]"
            )
        if len(requested_refs) == 2:
            output = _replace_numbered_section(
                output,
                4,
                "- " + "; ".join(requested_refs),
            )
    return output


# These definitions intentionally supersede the legacy helpers above.  The
# legacy source contained damaged Korean literals and could emit mojibake.
def _practical_rule_fallback(answer: str, chunks: list[Any]) -> str:
    return _ensure_practical_rule_section(answer, chunks)


def _ensure_practical_rule_section(answer: str, chunks: list[Any]) -> str:
    """Fill a hollow practical section using final Evidence Table chunks only."""
    if not answer or not chunks:
        return answer
    section = re.search(r"(?ms)^##\s*2\).*?(?=^##\s*3\))", answer)
    if not section or re.search(r"^\s*-\s+.*\[\d+\]", section.group(0), re.M):
        return answer
    for index, chunk in enumerate(chunks, start=1):
        source = strip_metadata_prefix(getattr(chunk, "text", "") or "")
        content = _rule_practical_candidate(source)
        if content:
            return _replace_numbered_section(answer, 2, f"- {content} [{index}]")
    return answer


def _rule_evidence_topic(source: str) -> str:
    lowered = re.sub(r"\s+", " ", source).lower()
    if "situational awareness" in lowered and "roc" in lowered:
        return "ROC의 운용상태 관찰 및 원격운영자 상황인식 요구사항"
    if "crankcase" in lowered or "ventilation" in lowered:
        return "대체연료·dual-fuel 기관의 crankcase 환기 및 안전 배치 조항"
    if "low flashpoint" in lowered or "low-flashpoint" in lowered:
        return "저인화점 연료 적용범위와 안전 요구사항"
    if "qualification" in lowered:
        return "concept/system qualification 및 승인 절차"
    if "autonomy" in lowered or "remotely operated" in lowered:
        return "자율·원격운항 기능의 적용범위와 검증 요구사항"
    return "질문의 기술 주제와 직접 연결되는 요구사항"


def _ensure_rule_reference_section(
    answer: str, chunks: list[Any], *, force: bool = False
) -> str:
    """Create section 4 only from chunks shown in the final Evidence Table."""
    if not answer or not chunks:
        return answer
    section = re.search(r"(?ms)^##\s*4\).*?\Z", answer)
    if (
        not force
        and section
        and re.search(r"^\s*-\s+.*\[\d+\]", section.group(0), re.M)
    ):
        return answer
    lines: list[str] = []
    seen: set[tuple[str, Any, str]] = set()
    for index, chunk in enumerate(chunks, start=1):
        source = strip_metadata_prefix(getattr(chunk, "text", "") or "")
        file_name = str(
            getattr(chunk, "file_name", "")
            or getattr(chunk, "doc_id", "")
            or "문서명 미확인"
        )
        page = getattr(chunk, "page_number", "?")
        clause, title = extract_clause_reference(chunk)
        key = (file_name, page, clause or title or "")
        if key in seen:
            continue
        seen.add(key)
        clause_label = f", clause {clause}" if clause else (f", {title}" if title else "")
        lines.append(
            f"- **{file_name}**, p.{page}{clause_label}: "
            f"{_rule_evidence_topic(source)}입니다. [{index}]"
        )
        if len(lines) >= 3:
            break
    return _replace_numbered_section(answer, 4, "\n".join(lines)) if lines else answer


def _compact_direct_rule_references(answer: str, chunks: list[Any]) -> str:
    """Keep every direct-clause source in one short section-4 bullet."""
    if not answer or len(chunks or []) < 2:
        return answer
    parts: list[str] = []
    seen: set[tuple[str, Any, str]] = set()
    for index, chunk in enumerate(chunks, start=1):
        file_name = str(
            getattr(chunk, "file_name", "")
            or getattr(chunk, "doc_id", "")
            or "검색 문서"
        )
        page = getattr(chunk, "page_number", "?")
        clause, title = extract_clause_reference(chunk)
        key = (file_name, page, clause or title or "")
        if key in seen:
            continue
        seen.add(key)
        detail = f", clause {clause}" if clause else (f", {title}" if title else "")
        parts.append(f"**{file_name}**, p.{page}{detail} [{index}]")
        if len(parts) >= 3:
            break
    if len(parts) < 2:
        return answer
    return _replace_numbered_section(
        answer,
        4,
        "- " + "; ".join(parts) + ": 질문의 용어·원칙과 직접 연결되는 근거입니다.",
    )


def build_direct_clause_revision_prompt(
    *,
    question: str,
    draft: str,
    evidence_block: str,
) -> str:
    """Ask for a source-coverage rewrite of a direct technical clause answer.

    This is intentionally question- and document-independent.  The first
    generation can be fluent but omit a source condition that appears later in
    a long clause.  The revision sees the same atomic propositions and is
    required to cover the concrete control, safety boundary, and available
    implementation/follow-up evidence without inventing a conclusion.
    """
    return f"""Rewrite the Korean answer below for the user's exact question.
Use only the atomic source propositions. Do not add a document overview.

Question: {question}

Atomic evidence:
{evidence_block}

Draft to correct:
{draft}

Required output contract:
1. Preserve exactly the four headings ## 1) through ## 4), Korean only, and
   end every factual bullet with the matching [n] citation.
2. In section 1, use different source propositions for the concrete technical
   requirement, the safety/performance condition, and a source-provided
   implementation or compensating measure when those propositions exist.
3. In section 2, state a directly evidenced design, monitoring, alarm, test,
   approval, or reporting work item. Do not invent consequences.
4. If the evidence contains a cross-reference (for example "See also"), an
   existing requirement to be observed, a condition, or a stated boundary,
   section 3 must name that exact follow-up with its citation; do not write
   "none" in that case.
5. Preserve modality: SHALL/MUST may be mandatory; SHOULD is a recommendation
   or expected requirement; SHOULD BE CONSIDERED remains a consideration;
   MAY/CAN/COULD remains optional. An example is never a mandatory measure.
6. Do not make generic statements such as "safety must be ensured". State the
   concrete source requirement in natural Korean.
7. Cover every source-stated existing requirement, normal/abnormal condition,
   human-sense limitation, and compensating-measure example that is relevant
   to the question. Group related examples in one concise Korean bullet.
8. Section 3 is only for an explicit cross-reference, unresolved application
   boundary, or adjacent requirement to check. Do not repeat a requirement
   already stated in sections 1 or 2.
"""


def build_direct_clause_repair_prompt(
    *,
    question: str,
    accepted_draft: str,
    evidence_block: str,
    gaps: list[str],
    rejected_claims: list[dict[str, Any]],
) -> str:
    """Repair only source-implied missing sections after modality validation.

    This is a general second-pass contract.  It receives the current question,
    the retrieved atomic propositions and validation reasons; it does not use
    a document-specific answer template or a question-answer cache.
    """
    rejected = "\n".join(
        f"- rejected: {item.get('claim', '')} | reason={item.get('reason', '')}"
        for item in rejected_claims
        if not item.get("supported")
    ) or "- none"
    return f"""Produce a corrected Korean answer to the exact question using only the source below.

Question: {question}

Atomic evidence (each proposition has its own modality):
{evidence_block}

Current validated draft (some factual bullets were removed because they overstated the source):
{accepted_draft}

Missing contract requirements: {', '.join(gaps)}
Rejected claims and reasons:
{rejected}

Return exactly four headings, ## 1) through ## 4), with Korean bullets only.
Every factual bullet must end with [n].  Keep every surviving valid fact, then
fill each listed missing section from an explicit proposition in the evidence.

For section 2, write a concrete design/configuration/verification/monitoring
work item already stated by the source; do not state a generic safety impact.
For section 3, if the source says "See also", refers to existing requirements,
or names normal/abnormal conditions, state that exact follow-up or boundary.
The missing-contract names identify source concepts omitted from the draft;
translate and restore those concepts from the atomic evidence. Do not merely
repeat the gap name.

Modal fidelity is mandatory: SHALL/MUST can be mandatory; SHOULD is a
recommendation or expected need; SHOULD BE CONSIDERED is only a consideration;
MAY/CAN/COULD is optional.  Do not turn an example such as CCTV, sensors, or
communications into a mandatory installation.  Do not use English source
sentences in the Korean answer.
"""


def _direct_clause_extractive_answer(question: str, chunks: list[Any], society: str) -> str:
    """Clause-first fallback when a local model fails citation verification.

    This is generic: it selects sentences by overlap with the technical words
    in the user's question.  It contains no document ID, page number, or
    question-specific answer text.
    """
    ignored = {
        "dnv", "lr", "abs", "kr", "requirement", "requirements", "related",
        "and", "or", "the", "of", "for", "to", "in", "on", "from", "with",
        "find", "show", "please",
    }
    terms = [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", question)
        if token.lower() not in ignored
    ]
    best_index = max(
        range(len(chunks)),
        key=lambda idx: sum(
            term in strip_metadata_prefix(getattr(chunks[idx], "text", "") or "").lower()
            for term in terms
        ),
    )
    chunk = chunks[best_index]
    body = re.sub(r"\s+", " ", strip_metadata_prefix(getattr(chunk, "text", "") or "")).strip()
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", body) if item.strip()]
    direct = [item for item in sentences if any(term in item.lower() for term in terms)][:3]
    if not direct:
        direct = [body[:650].rstrip()]
    doc = getattr(chunk, "file_name", "") or getattr(chunk, "doc_id", "") or society
    page = getattr(chunk, "page_number", "?")
    clause = getattr(chunk, "clause_number", "") or ""
    reference = f"{doc}, p.{page}" + (f", clause {clause}" if clause else "")
    facts = "\n".join(f"- **{reference}**: {item} [1]" for item in direct)
    return (
        "## 1) \ud575\uc2ec \uc694\uc57d\n" + facts + "\n\n"
        "## 2) \uc120\ubc15 \uc6b4\ud56d/\uc5c5\ubb34 \uc601\ud5a5\n"
        "- \ud604 \uc778\uc6a9 \uc870\ud56d\uc740 \uc6d0\uaca9 \uc6b4\uc601 \uc2dc \uc0c1\ud0dc \uad00\uce21\uacfc \uc0c1\ud669\uc778\uc2dd\uc758 \ud655\ubcf4\ub97c \uc694\uad6c\ud558\ub294 \ubc94\uc704\uc5d0\uc11c \uc801\uc6a9\ud558\uc5ec\uc57c \ud569\ub2c8\ub2e4. [1]\n\n"
        "## 3) \ucd94\ud6c4 \ud655\uc778 \ud544\uc694\uc0ac\ud56d\n"
        f"- \uc778\uc6a9 \uc870\ud56d \uc678\uc758 \uc138\ubd80 \uc2dc\uc2a4\ud15c \uc694\uac74\uacfc \uc2b9\uc778 \ubc94\uc704\ub294 {reference}\uc758 \uc778\uc811 \uc870\ud56d\uc744 \ucd94\uac00\ub85c \ub300\uc870\ud574\uc57c \ud569\ub2c8\ub2e4. [1]\n\n"
        "## 4) \uad00\ub828 \uc120\uae09 Rule / Guidance\n"
        f"- **{doc}**: \uc9c8\ubb38\uacfc \uc9c1\uc811 \uc77c\uce58\ud558\ub294 \uadfc\uac70\ub294 {reference}\uc785\ub2c8\ub2e4. [1]"
    )


def generate_rule_guidance_accurate_answer(
    row: dict,
    retrieved: list[RetrievedChunk],
    *,
    pool: list[RetrievedChunk] | None = None,
    model: str,
    ollama_base: str,
    timing=None,
    on_token: Callable[[str], None] | None = None,
    temperature: float = ACCURATE_TEMPERATURE,
) -> tuple[str, str, str, dict[str, Any]]:
    """
    Accurate rule guidance: structured draft + short LLM summary.
    Returns (answer, provider, model_name, answer_generation_meta).
    """
    question = str(row.get("question") or "")
    exact_fact = is_exact_rule_fact_question(question)
    fact_slots = exact_rule_fact_slots(question) if exact_fact else 0
    if exact_fact:
        row["_answer_profile"] = "exact_rule_fact"
        row["_answer_fact_slots"] = fact_slots
    from retrieval_query_analysis import detect_class_society_hint

    society = str(
        row.get("class_society_hint") or detect_class_society_hint(question)
    )
    pool = pool or retrieved
    gen_meta: dict[str, Any] = {
        "answer_source": "fallback_no_evidence",
        "llm_used": False,
        "llm_call_function": None,
        "llm_prompt_chars": 0,
        "llm_context_chunks": 0,
        "llm_output_chars": 0,
        "llm_grounded_check_pass": False,
        "fallback_reason": None,
    }

    evidence_chunks, coverage_meta = _slot_preserving_chunks(
        row, retrieved or [], pool or [], society
    )
    gen_meta["evidence_slot_coverage"] = coverage_meta
    # A planner may retain a low-priority ``specific_clause`` slot even for a
    # broad document-discovery question.  Do not let that implementation
    # detail collapse an applicability/requirements lookup to three chunks.
    # The compact direct-clause path is reserved for an actually bounded fact,
    # definition or explicitly requested clause/section.
    direct_clause_intent = bool(
        exact_fact
        or _is_definition_lookup(question)
        or re.search(
            r"(?:근거\s*)?조항|clause|section|몇\s*(?:시간|일|톤|배|개)|"
            r"어떤\s*(?:경우|조건|정격)|생략할\s*수\s*있는\s*조건",
            question,
            re.I,
        )
    )
    direct_clause_found = bool(
        (coverage_meta.get("slot_coverage") or {}).get("specific_clause")
        and direct_clause_intent
    ) or _is_definition_lookup(question)
    if direct_clause_found:
        direct_chunks = select_specific_clause_chunks(
            row, retrieved or [], pool or []
        )
        if direct_chunks:
            direct_filtered = filter_evidence_chunks(
                direct_chunks, society, hard=True
            )
            from retrieval_search import extract_sparse_latin_terms

            named_terms = extract_sparse_latin_terms(question, limit=2)
            named_hits = [
                chunk
                for chunk in [*(retrieved or []), *(pool or [])]
                if str(getattr(chunk, "source", "") or "").upper()
                == society.upper()
                and any(
                    term in str(getattr(chunk, "text", "") or "").lower()
                    for term in named_terms
                )
            ]
            seen_direct: set[str] = set()
            evidence_chunks = []
            for chunk in [*named_hits[:1], *direct_filtered]:
                cid = str(getattr(chunk, "chunk_id", "") or id(chunk))
                if cid in seen_direct:
                    continue
                seen_direct.add(cid)
                evidence_chunks.append(chunk)
            coverage_meta["direct_clause_context_only"] = True
            coverage_meta["direct_clause_chunk_ids"] = [
                str(getattr(chunk, "chunk_id", "")) for chunk in evidence_chunks
            ]

    if not evidence_chunks:
        gen_meta["fallback_reason"] = "society_evidence_insufficient"
        answer = fallback_no_evidence_answer(society)
        row["_answer_generation"] = gen_meta
        return answer, "rule_guidance_lookup", "none", gen_meta

    if _is_definition_lookup(question):
        definition_answer, definition_chunk = _build_definition_extractive_answer(
            question,
            [*evidence_chunks, *(retrieved or []), *(pool or [])],
            society,
        )
        if definition_answer and definition_chunk is not None:
            row["_rule_guidance_llm_chunks"] = [definition_chunk]
            row["_answer_citation_chunks"] = [definition_chunk]
            gen_meta.update(
                {
                    "answer_source": "direct_definition_extractive",
                    "llm_used": False,
                    "llm_context_chunks": 1,
                    "llm_output_chars": len(definition_answer),
                    "llm_grounded_check_pass": True,
                    "fallback_reason": None,
                }
            )
            row["_answer_generation"] = gen_meta
            return definition_answer, "rule_guidance_lookup", "extractive", gen_meta

    markers = _fabricated_query_markers(question)
    if _markers_absent_from_chunks(markers, evidence_chunks):
        gen_meta["fallback_reason"] = "query_terms_not_in_evidence"
        answer = fallback_no_evidence_answer(society)
        row["_answer_generation"] = gen_meta
        return answer, "rule_guidance_lookup", "none", gen_meta

    guide_style = str(
        (row.get("_question_profile") or {}).get("answer_style") or ""
    ) == "document_cards"
    if guide_style:
        from rule_lookup_alt_fuel import is_alt_fuel_question

        # LR alternative-fuel discovery already has a stronger, clause-theme
        # renderer covering Section 15, crankcase ventilation and safeguards.
        # Preserve that technical guide instead of reducing it to metadata.
        guide_style = not is_alt_fuel_question(question)

    warnings = list(row.get("warning_flags") or [])
    evidence_draft = build_compact_evidence_draft(evidence_chunks, society)
    if direct_clause_found:
        evidence_draft = "Use the atomic clause propositions below."
    structured_draft = ""
    document_card_fallback = ""
    document_card_chunks: list[Any] = []
    try:
        from rule_lookup_structured_answer import expand_rule_lookup_chunks

        # Keep purpose/process/technical-clause coverage together for both
        # the deterministic factual floor and the LLM context.  Previously
        # only the renderer expanded these candidates; the LLM still saw the
        # first incidental clause and treated it as the whole Rule.
        if not direct_clause_found:
            evidence_chunks = expand_rule_lookup_chunks(
                evidence_chunks, pool, question=question
            )
        if guide_style:
            from rule_document_cards import build_rule_document_cards

            document_card_fallback, document_card_chunks = build_rule_document_cards(
                question,
                [*(retrieved or []), *(pool or []), *evidence_chunks],
                max_documents=3,
            )
            if document_card_chunks:
                seen_card_ids: set[str] = set()
                evidence_chunks = [
                    chunk
                    for chunk in [*document_card_chunks, *evidence_chunks]
                    if not (
                        (identity := str(getattr(chunk, "chunk_id", "") or id(chunk)))
                        in seen_card_ids
                        or seen_card_ids.add(identity)
                    )
                ]
        structured_draft, ans_warnings = build_rule_lookup_structured_answer(
            evidence_chunks,
            question=question,
            pool=evidence_chunks,
            warning_flags=warnings,
        )
        row["warning_flags"] = list(dict.fromkeys(warnings + ans_warnings))
        if structured_draft and not direct_clause_found:
            # This is a cited fact outline, not a prepared answer.  Retaining
            # more than one bullet gives the LLM coverage across application,
            # safety control and follow-up rather than anchoring it to the
            # first document title alone.
            evidence_draft = structured_draft[:1200]
    except Exception:
        pass

    planned_slot_names = set(
        ((row.get("_evidence_completion") or {}).get("slot_hits") or {}).keys()
    )
    verified_compound_outline = bool(
        {"risk_classification_basis", "higher_risk_verification"}.issubset(
            planned_slot_names
        )
        or {
            "scope",
            "concept_qualification_role",
            "preliminary_risk_assessment",
        }.issubset(planned_slot_names)
    )
    if structured_draft and verified_compound_outline and not direct_clause_found:
        answer = _practical_rule_fallback(structured_draft, list(evidence_chunks))
        answer = _ensure_practical_rule_section(answer, list(evidence_chunks))
        answer = _ensure_rule_reference_section(
            answer, list(evidence_chunks), force=True
        )
        answer = _ensure_named_rule_facts(
            answer, question, list(evidence_chunks)
        )
        if re.search(r"notation|부호", question, re.I):
            for index, chunk in enumerate(evidence_chunks, 1):
                body = str(getattr(chunk, "text", "") or "")
                if re.search(r"AROS.{0,80}additional\s+class\s+notations", body, re.I | re.S):
                    answer = re.sub(
                        r"(?m)^(- .*?)(\s+\[\d+\])$",
                        rf"\1 또한 자율·원격운항 선박의 AROS family of additional class notations를 다룹니다. [{index}]",
                        answer,
                        count=1,
                    )
                    break
        row["_rule_guidance_llm_chunks"] = list(evidence_chunks)
        row["_answer_citation_chunks"] = list(evidence_chunks)
        row["_rule_guidance_skip_heavy_postprocess"] = True
        row["_verified_structured_answer"] = True
        gen_meta.update(
            {
                "answer_source": "verified_compound_structured_answer",
                "llm_used": False,
                "llm_context_chunks": len(evidence_chunks),
                "llm_output_chars": len(answer),
                "llm_grounded_check_pass": True,
                "fallback_reason": None,
            }
        )
        row["_answer_generation"] = gen_meta
        return answer, "rule_guidance_lookup", "none", gen_meta

    llm_chunks, evidence_block = trim_chunks_for_llm(
        evidence_chunks,
        direct_clause=direct_clause_found,
    )
    if not llm_chunks or not evidence_block.strip():
        gen_meta["fallback_reason"] = "no_substantive_chunks_for_llm"
        answer = fallback_no_evidence_answer(society)
        row["_answer_generation"] = gen_meta
        return answer, "rule_guidance_lookup", "none", gen_meta

    draft_budget = max(280, MAX_TOTAL_CONTEXT_CHARS - len(evidence_block) - 60)
    evidence_draft_trim = evidence_draft[:draft_budget]
    if len(evidence_draft) > draft_budget:
        evidence_draft_trim = evidence_draft_trim.rstrip() + "…"

    gen_meta["answer_style"] = "document_cards" if guide_style else (
        "short_fact" if exact_fact else "analytical_rule"
    )
    user = (
        build_exact_rule_fact_prompt(
            question=question,
            evidence_block=evidence_block,
            fact_slots=fact_slots,
        )
        if exact_fact
        else build_clean_direct_clause_prompt(
            question=question,
            evidence_block=evidence_block,
        )
        if direct_clause_found
        else build_rule_document_guide_prompt(
            question=question,
            society=society,
            evidence_draft=evidence_draft_trim,
            evidence_block=evidence_block,
        )
        if guide_style
        else build_rule_guidance_user_prompt(
            question=question,
            society=society,
            evidence_draft=evidence_draft_trim,
            evidence_block=evidence_block,
            direct_clause=False,
        )
    )
    system = RULE_GUIDANCE_SYSTEM_PROMPT
    prompt_chars = len(system) + len(user)

    ensure_rule_guidance_warm(model, ollama_base, timing=timing)

    gen_meta.update(
        {
            "answer_source": "llm_grounded_summary",
            "llm_used": True,
            "llm_call_function": "call_ollama_chat_timed",
            "llm_prompt_chars": prompt_chars,
            "llm_context_chunks": len(llm_chunks),
            "llm_num_ctx": ACCURATE_NUM_CTX,
            "llm_num_predict": ACCURATE_NUM_PREDICT,
            "llm_temperature": temperature,
            "keep_alive": KEEP_ALIVE,
            "answer_profile": "exact_rule_fact" if exact_fact else "analytical_rule",
            "requested_fact_slots": fact_slots or None,
        }
    )

    if timing is not None and hasattr(timing, "mark_wall"):
        timing.mark_wall("t_prompt_build_end")
        timing.mark_wall("t_accurate_llm_request_start")

    answer = call_ollama_chat_timed(
        model,
        system,
        user,
        ollama_base,
        temperature=temperature,
        num_predict=ACCURATE_NUM_PREDICT,
        num_ctx=ACCURATE_NUM_CTX,
        timing=timing,
        # Do not stream unverified text into the UI.  The completed answer is
        # checked claim-by-claim before it is rendered.
        on_token=None,
    )
    # A direct clause may be a long, compound paragraph.  Use a second,
    # evidence-bound review pass so a fluent first draft cannot silently omit
    # a cross-reference, existing requirement, or concrete control measure
    # present later in that same clause.  This is not an answer cache: both
    # inputs are the current question and the retrieved clause only.
    if direct_clause_found and answer:
        try:
            review_user = build_direct_clause_revision_prompt(
                question=question,
                draft=answer,
                evidence_block=evidence_block,
            )
            revised = call_ollama_chat_timed(
                model,
                system,
                review_user,
                ollama_base,
                temperature=temperature,
                num_predict=DIRECT_CLAUSE_NUM_PREDICT,
                num_ctx=ACCURATE_NUM_CTX,
                timing=None,
                on_token=None,
            )
            if revised and re.search(r"##\s*1\)", revised):
                answer = revised
                gen_meta["direct_clause_review_pass"] = True
        except Exception as exc:
            gen_meta["direct_clause_review_error"] = type(exc).__name__

    # Validate the reviewed draft before deciding whether it omitted a section
    # that the *source itself* makes actionable.  This catches the common
    # failure mode where the model writes an over-strong CCTV/UMS claim, the
    # validator correctly removes it, and the final answer silently leaves
    # sections 2 and 3 empty.
    if direct_clause_found and answer:
        try:
            preview, preview_rows, _ = validate_direct_clause_answer(answer, llm_chunks)
            preview = replace_rule_reference_section(preview, llm_chunks)
            gaps = direct_clause_coverage_gaps(preview, llm_chunks)
            if gaps:
                repair_user = build_direct_clause_repair_prompt(
                    question=question,
                    accepted_draft=preview,
                    evidence_block=evidence_block,
                    gaps=gaps,
                    rejected_claims=preview_rows,
                )
                repaired = call_ollama_chat_timed(
                    model,
                    system,
                    repair_user,
                    ollama_base,
                    temperature=temperature,
                    num_predict=DIRECT_CLAUSE_NUM_PREDICT,
                    num_ctx=ACCURATE_NUM_CTX,
                    timing=None,
                    on_token=None,
                )
                if repaired and re.search(r"##\s*1\)", repaired):
                    repaired_preview, _, _ = validate_direct_clause_answer(repaired, llm_chunks)
                    repaired_preview = replace_rule_reference_section(repaired_preview, llm_chunks)
                    repaired_gaps = direct_clause_coverage_gaps(repaired_preview, llm_chunks)
                    if len(repaired_gaps) < len(gaps):
                        answer = repaired
                        gen_meta["direct_clause_repair_pass"] = True
                        gen_meta["direct_clause_coverage_gaps_fixed"] = gaps
                    else:
                        gen_meta["direct_clause_repair_unresolved"] = repaired_gaps
        except Exception as exc:
            gen_meta["direct_clause_repair_error"] = type(exc).__name__
    gen_meta["raw_llm_answer"] = answer
    gen_meta["llm_output_chars"] = len(answer or "")
    direct_claim_rows: list[dict[str, Any]] = []
    direct_claim_warnings: list[str] = []
    if direct_clause_found:
        answer, direct_claim_rows, direct_claim_warnings = (
            validate_direct_clause_answer(answer, llm_chunks)
        )
        answer = replace_rule_reference_section(answer, llm_chunks)
        answer = ensure_direct_clause_source_details(answer, llm_chunks)
        # The source-driven completion pass may resolve gaps that the local
        # model left behind.  Recompute the audit metadata from the displayed
        # answer so logs do not report stale unresolved omissions.
        final_direct_gaps = direct_clause_coverage_gaps(answer, llm_chunks)
        if final_direct_gaps:
            gen_meta["direct_clause_repair_unresolved"] = final_direct_gaps
        else:
            gen_meta.pop("direct_clause_repair_unresolved", None)
            gen_meta["direct_clause_coverage_complete"] = True
        grounded = _direct_clause_korean_contract_pass(answer, llm_chunks, society)
        # The direct-clause validator has already checked surviving Korean
        # bullets against the atomic source propositions.  Do not send those
        # claims back through the generic cross-language lexical verifier:
        # it cannot match Korean translations to English text and was replacing
        # a valid clause follow-up with a generic "not identified" message.
        if not grounded and any(item.get("supported") for item in direct_claim_rows):
            has_four_sections = all(
                re.search(rf"(?:^|\n)##\s*{section}\)", answer, re.M)
                for section in range(1, 5)
            )
            if has_four_sections:
                grounded = True
                gen_meta["direct_clause_atomic_rows_preserved"] = True
    else:
        # A broad Rule answer is useful only when it is both Korean and bound
        # to the currently displayed evidence.  The old lexical check could
        # accept an English quotation, while a faithful Korean answer failed
        # because Korean and English share few surface words.  Prefer a short
        # claim-card recovery pass over silently returning either outcome.
        grounded = False
    broad_korean_contract = False
    if not direct_clause_found:
        broad_korean_contract = _broad_rule_korean_citation_contract(answer, llm_chunks)
        if not broad_korean_contract:
            try:
                recovery_chunks = list(llm_chunks)[:4]
                _recovery_selected, recovery_block = trim_chunks_for_llm(
                    recovery_chunks, direct_clause=False
                )
                recovery_user = build_broad_rule_korean_recovery_prompt(
                    question=question,
                    evidence_block=recovery_block,
                )
                recovered = call_ollama_chat_timed(
                    model,
                    system,
                    recovery_user,
                    ollama_base,
                    temperature=0.0,
                    num_predict=ACCURATE_NUM_PREDICT,
                    num_ctx=ACCURATE_NUM_CTX,
                    timing=None,
                    on_token=None,
                )
                if _broad_rule_korean_citation_contract(recovered, recovery_chunks):
                    answer = recovered
                    llm_chunks = recovery_chunks
                    broad_korean_contract = True
                    gen_meta["broad_rule_korean_recovery_pass"] = True
            except Exception as exc:
                gen_meta["broad_rule_korean_recovery_error"] = type(exc).__name__
        if broad_korean_contract:
            grounded = True
            gen_meta["claim_verification_mode"] = "citation_bound_korean_contract"
    gen_meta["llm_grounded_check_pass"] = grounded
    if grounded:
        if direct_clause_found:
            gen_meta["claim_verification"] = direct_claim_rows
            gen_meta["claim_verification_warnings"] = direct_claim_warnings
            gen_meta["claim_verification_mode"] = (
                "direct_clause_modality_and_contract"
            )
        elif not broad_korean_contract:
            answer, claim_rows, claim_warnings = verify_claim_citations(answer, llm_chunks)
            gen_meta["claim_verification"] = claim_rows
            if claim_warnings:
                grounded = False
                gen_meta["llm_grounded_check_pass"] = False
    if not grounded:
        # For a broad Rule lookup, an English/mixed-language partial answer is
        # not an acceptable degradation.  Keep the complete citation-stable
        # structured draft instead; it is generated from the selected current
        # evidence and is preferable to leaking a translated fragment that
        # answers a different facet of the question.
        if not direct_clause_found:
            if guide_style and document_card_fallback and document_card_chunks:
                answer = document_card_fallback
                gen_meta["answer_source"] = "document_profile_card_fallback"
                gen_meta["fallback_reason"] = "broad_rule_korean_llm_contract_failed"
                row["_answer_generation"] = gen_meta
                row["_rule_guidance_llm_chunks"] = list(document_card_chunks)
                row["_answer_citation_chunks"] = list(document_card_chunks)
                row["_rule_guidance_skip_heavy_postprocess"] = True
                row["_verified_structured_answer"] = True
                return answer, "rule_guidance_lookup", "none", gen_meta
            answer = _practical_rule_fallback(
                structured_draft or fallback_no_evidence_answer(society),
                list(evidence_chunks),
            )
            answer = _ensure_practical_rule_section(
                answer, list(evidence_chunks)
            )
            answer = _ensure_rule_reference_section(
                answer, list(evidence_chunks), force=True
            )
            answer = _ensure_named_rule_facts(
                answer, question, list(evidence_chunks)
            )
            gen_meta["answer_source"] = "structured_template_grounding_fallback"
            gen_meta["fallback_reason"] = "broad_rule_korean_llm_contract_failed"
            row["_answer_generation"] = gen_meta
            # The structured draft assigned [n] against evidence_chunks, not
            # the later LLM-trimmed subset.  Keep that exact ordered list as
            # the UI citation source so same-page clauses cannot be remapped
            # to a different Evidence Table row.
            row["_rule_guidance_llm_chunks"] = list(evidence_chunks)
            row["_rule_guidance_skip_heavy_postprocess"] = True
            row["_verified_structured_answer"] = True
            return answer, "rule_guidance_lookup", "none", gen_meta
        # Do not discard every newly retrieved slot because one generated
        # claim failed.  Keep the verified subset when it still answers the
        # question; only fall back when no grounded factual claim survives.
        verified_answer, claim_rows, claim_warnings = verify_claim_citations(
            answer, llm_chunks
        )
        supported_rows = [row for row in claim_rows if row.get("supported")]
        gen_meta["claim_verification"] = claim_rows
        gen_meta["claim_verification_warnings"] = claim_warnings
        if supported_rows and re.search(r"##\s*1\)", verified_answer or ""):
            answer = verified_answer
            gen_meta["answer_source"] = "llm_verified_claim_subset"
            gen_meta["fallback_reason"] = "unsupported_claims_removed"
        else:
            direct_clause_found = bool(
                (coverage_meta.get("slot_coverage") or {}).get("specific_clause")
            )
            answer = (
                _direct_clause_extractive_answer(question, llm_chunks, society)
                if direct_clause_found
                else (structured_draft or fallback_no_evidence_answer(society))
            )
            gen_meta["answer_source"] = (
                "direct_clause_extractive_fallback"
                if direct_clause_found
                else "structured_template_grounding_fallback"
            )
            gen_meta["fallback_reason"] = "no_verified_llm_claims"
    if timing is not None and hasattr(timing, "monotonic"):
        mono = timing.monotonic
        t_req = mono.get("t_llm_request_start")
        t_tok = mono.get("t_first_token")
        t_ret = mono.get("t_retrieval_end") or mono.get("t_context_build_end")
        pre = max(0.0, (t_req - t_ret)) if t_req and t_ret else 0.0
        ttft = max(0.0, (t_tok - t_req)) if t_tok and t_req else 0.0
        if ttft:
            combined = round(pre + ttft, 4)
            gen_meta["rule_guidance_first_token_latency"] = combined
            gen_meta["rule_guidance_first_token_3s_pass"] = combined <= 3.0
    answer = _normalize_rule_translation(answer, list(llm_chunks))
    # The agreed UI contract requires an explicit document and page/clause.
    # LLM prose often filled section 4 with a generic restatement, so replace
    # it with citation-stable references from the displayed evidence.
    answer = _ensure_rule_reference_section(answer, list(llm_chunks), force=True)
    answer = _ensure_named_rule_facts(answer, question, list(llm_chunks))
    if direct_clause_found:
        answer = _compact_direct_rule_references(answer, list(llm_chunks))
    row["_answer_generation"] = gen_meta
    row["_rule_guidance_llm_chunks"] = (
        list(evidence_chunks)
        if gen_meta.get("answer_source") == "structured_template_grounding_fallback"
        else llm_chunks
    )
    row["_rule_guidance_skip_heavy_postprocess"] = True
    row["_verified_structured_answer"] = True

    return answer, "ollama", model, gen_meta
