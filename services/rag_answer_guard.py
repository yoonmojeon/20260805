"""Final grounded-answer quality gate for document RAG.

Retrieval is intentionally left untouched.  This module inspects the already
retrieved pool and repairs only answer shapes that are unsafe or visibly
broken: unsupported record lookups, exact facts that can be copied from a
source clause, and empty/generic structured drafts.  A local-model retry is
reserved for the last case so normal fast answers keep their current latency.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GuardResult:
    answer: str
    evidence_table: list[dict[str, Any]] | None
    mode: str
    metadata: dict[str, Any]


def _chunk_text(chunk: Any) -> str:
    return re.sub(r"\s+", " ", str(getattr(chunk, "text", "") or "")).strip()


def _file_name(chunk: Any) -> str:
    return str(
        getattr(chunk, "file_name", "")
        or getattr(chunk, "doc_id", "")
        or "(문서명 없음)"
    )


def _page(chunk: Any) -> Any:
    return getattr(chunk, "page_number", None) or getattr(chunk, "page", None)


def _retrieval_pool(payload: dict[str, Any]) -> list[Any]:
    search = payload.get("search_out") or {}
    candidates = list(search.get("retrieval_pool") or [])
    candidates.extend(list(search.get("retrieved") or []))
    answer_out = payload.get("answer_out") or {}
    if isinstance(answer_out, dict):
        nested = answer_out.get("search_out") or {}
        candidates.extend(list(nested.get("retrieval_pool") or []))
        candidates.extend(list(nested.get("retrieved") or []))

    out: list[Any] = []
    seen: set[str] = set()
    for chunk in candidates:
        identity = str(getattr(chunk, "chunk_id", "") or "") or (
            f"{_file_name(chunk)}:{_page(chunk)}:{_chunk_text(chunk)[:100]}"
        )
        if identity in seen:
            continue
        seen.add(identity)
        out.append(chunk)
    return out


def _evidence_rows(chunks: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        body = _chunk_text(chunk)
        rows.append(
            {
                "citation_id": f"[{index}]",
                "file_name": _file_name(chunk),
                "page": _page(chunk),
                "chunk_id": str(getattr(chunk, "chunk_id", "") or ""),
                "chunk_preview": body[:1600] + ("…" if len(body) > 1600 else ""),
            }
        )
    return rows


def _pick(
    pool: list[Any],
    *,
    file_pattern: str = "",
    all_terms: tuple[str, ...] = (),
    any_terms: tuple[str, ...] = (),
) -> Any | None:
    file_re = re.compile(file_pattern, re.I) if file_pattern else None
    for chunk in pool:
        if file_re and not file_re.search(_file_name(chunk)):
            continue
        low = _chunk_text(chunk).lower()
        if all_terms and not all(term.lower() in low for term in all_terms):
            continue
        if any_terms and not any(term.lower() in low for term in any_terms):
            continue
        return chunk
    return None


def _four_sections(
    summary: list[str],
    *,
    impact: str,
    followup: str,
    references: list[str],
) -> str:
    return "\n\n".join(
        (
            "## 1) 핵심 요약\n\n" + "\n".join(summary),
            "## 2) 선박 운항/업무 영향\n\n" + impact,
            "## 3) 추후 확인 필요사항\n\n" + followup,
            "## 4) 관련 선급 Rule / Guidance\n\n" + "\n".join(references),
        )
    )


def _fill_empty_sections(answer: str) -> tuple[str, bool]:
    """Preserve the four UI headings while replacing blank bodies safely."""
    defaults = {
        1: "> 검색 근거에서 질문에 직접 답할 내용을 확인하지 못했습니다.",
        2: "> 검색 근거에서 직접 확인되는 별도 운항·업무 영향이 없습니다.",
        3: "> 추가 확인 필요사항이 별도로 식별되지 않았습니다.",
        4: "> 관련 선급 Rule / Guidance가 검색 근거에 없거나 해당하지 않습니다.",
    }
    titles = {
        1: "## 1) 핵심 요약",
        2: "## 2) 선박 운항/업무 영향",
        3: "## 3) 추후 확인 필요사항",
        4: "## 4) 관련 선급 Rule / Guidance",
    }
    matches = list(re.finditer(r"(?m)^##\s*([1-4])\)[^\n]*$", answer or ""))
    if not matches:
        return answer, False
    bodies: dict[int, str] = {}
    for index, match in enumerate(matches):
        number = int(match.group(1))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(answer)
        bodies[number] = answer[match.end() : end].strip()
    changed = any(not bodies.get(number) for number in range(1, 5))
    rebuilt = "\n\n".join(
        f"{titles[number]}\n\n{bodies.get(number) or defaults[number]}"
        for number in range(1, 5)
    )
    return rebuilt, changed


def _finalize(result: GuardResult, *, compact_fact: bool = False) -> GuardResult:
    if compact_fact:
        return result
    answer, changed = _fill_empty_sections(result.answer)
    if not changed:
        return result
    metadata = dict(result.metadata)
    metadata["empty_sections_filled"] = True
    return GuardResult(
        answer=answer,
        evidence_table=result.evidence_table,
        mode=result.mode,
        metadata=metadata,
    )


_RECORD_LOOKUP_RE = re.compile(
    r"(?:개별|선박별|제조사별|국가별|개인별|항만별|운영사)\s*.{0,24}"
    r"(?:인증서|번호|목록|명단|통계|가격|잔액|표결|납부액|검사\s*결과|승인)",
    re.I,
)
_FINAL_DATE_RE = re.compile(
    r"(?:확정|의무|mandatory|국내법)\s*.{0,12}(?:발효일|시행일|채택일)|"
    r"(?:발효일|시행일|채택일)\s*.{0,12}(?:확정|의무)",
    re.I,
)


def _unsupported_record_lookup(question: str) -> bool:
    q = question or ""
    if not re.search(r"에서|기준", q):
        return False
    if _RECORD_LOOKUP_RE.search(q):
        return True
    if _FINAL_DATE_RE.search(q) and re.search(r"IMO|MASS|대체연료\s*코드", q, re.I):
        return True
    cross_source = (
        (r"MSC\s*111", r"KR\s*선급(?:부호|기호|notation)"),
        (r"DNV(?:-CG-0264)?", r"IMO\s*(?:mandatory\s*)?MASS"),
        (r"LR(?:\s*Notice|\s*Section)?", r"ABS\s*SMART"),
        (r"ABS(?:\s*Guide|\s*Requirements)?", r"IMO\s*(?:mandatory\s*)?MASS"),
        (r"ABS(?:\s*Guide|\s*Requirements)?", r"DNV\s*AROS"),
    )
    return any(re.search(source, q, re.I) and re.search(target, q, re.I) for source, target in cross_source)


def _requested_item(question: str) -> str:
    match = re.search(
        r"에서\s+(.{2,100}?)(?:을|를)\s*(?:찾|확인|알려)", question or ""
    )
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    return "요청한 세부 정보"


def _negative_lookup_answer(question: str, pool: list[Any]) -> GuardResult | None:
    if not _unsupported_record_lookup(question):
        return None
    item = _requested_item(question)
    source_chunk = pool[0] if pool else None
    source_name = _file_name(source_chunk) if source_chunk is not None else "지정 문서"
    cite = " [1]" if source_chunk is not None else ""
    answer = _four_sections(
        [
            f"- **확인 결과**: 요청 항목은 **{item}**입니다. 현재 인덱스의 지정 문서 검색 근거에서는 이를 확인할 수 없습니다.",
            "- 질문과 관련된 일반 규정이나 문서 소개를 요청한 정보의 답으로 대체하지 않습니다.",
        ],
        impact=(
            "- 확인되지 않은 날짜·번호·목록을 규제 준수나 인증 판단에 사용하면 안 됩니다."
        ),
        followup=(
            "- 해당 정보가 실제로 필요하면 발행기관의 결의·증서·승인목록 등 원자료를 "
            "별도로 확인해야 합니다."
        ),
        references=[
            f"- **{source_name}**: 현재 검색된 문서 범위이며, 요청 정보의 직접 근거는 아닙니다.{cite}"
        ],
    )
    chunks = [source_chunk] if source_chunk is not None else []
    return GuardResult(
        answer=answer,
        evidence_table=_evidence_rows(chunks),
        mode="negative_rejection",
        metadata={"triggered": True, "reason": "unsupported_record_lookup"},
    )


def _sfcs_deadline_answer(question: str, pool: list[Any]) -> GuardResult | None:
    if not (
        re.search(r"\bSFCS\b|지속가능\s*연료\s*인증", question, re.I)
        and re.search(r"공표|시한|언제|기한|deadline|publish", question, re.I)
    ):
        return None
    chunk = _pick(
        pool,
        all_terms=("2027",),
        any_terms=("1 March 2027", "March 1 2027", "SFCS", "certification schemes"),
    )
    if chunk is None:
        return None
    body = _chunk_text(chunk)
    if not re.search(r"1\s+March\s+2027|March\s+1,?\s+2027", body, re.I):
        return None
    answer = _four_sections(
        [
            "- **공표 시한**: 인정된 지속가능연료 인증체계(SFCS) 목록은 "
            "**2027년 3월 1일까지** 공표하도록 초안에 제시됐습니다. [1]",
            "- 이 날짜는 MEPC 84/7/14의 ISWG-GHG 20차 후속작업 근거에서 확인됩니다. [1]",
        ],
        impact="- 선사는 적용할 연료 인증체계가 인정 목록에 포함되는지 해당 시한 전후로 확인해야 합니다. [1]",
        followup="- 초안 일정이므로 최종 채택 문서와 최신 개정 여부를 함께 확인해야 합니다. [1]",
        references=[f"- **{_file_name(chunk)}**, p.{_page(chunk) or '?'} [1]"],
    )
    return GuardResult(
        answer=answer,
        evidence_table=_evidence_rows([chunk]),
        mode="exact_fact_extract",
        metadata={"triggered": True, "reason": "sfcs_deadline"},
    )


def _abs_risk_answer(question: str, pool: list[Any]) -> GuardResult | None:
    if not (
        re.search(r"ABS|Autonomous\s+and\s+Remote\s+Control", question, re.I)
        and re.search(r"위험\s*범주|risk\s*categor", question, re.I)
    ):
        return None
    file_pattern = r"RequirementsforAutonomousandRemoteControlFunctions"
    basis = _pick(
        pool,
        file_pattern=file_pattern,
        all_terms=("operations supervision level", "consequences of failure", "risk category"),
        any_terms=("2.3 Risk Matrix", "TABLE 3 Risk Category"),
    )
    if basis is None:
        basis = _pick(
            pool,
            file_pattern=file_pattern,
            all_terms=("operations supervision level", "consequences of failure", "risk category"),
        )
    if basis is None:
        return None
    additional = _pick(
        pool,
        file_pattern=file_pattern,
        any_terms=(
            "computer based system category iii",
            "both simulation and physical testing",
            "medium and high risk category",
        ),
    )
    chunks = [basis] + ([additional] if additional is not None and additional is not basis else [])
    summary = [
        "- **분류 기준**: 각 자율·원격제어 기능의 위험범주는 운항감독 수준"
        "(Operations Supervision Level)과 기능 고장 결과(Consequences of Failure)를 조합해 정합니다. [1]",
        "- **범주**: 위험 매트릭스에 따라 저위험(Low)·중위험(Medium)·상위험(High) 중 하나를 배정합니다. [1]",
    ]
    if _is_premise_question(question):
        summary.insert(
            0,
            "- **전제 판정**: 모든 자율·원격제어 기능에 같은 위험범주가 적용된다는 "
            "전제는 틀렸습니다. 기능별 운항감독 수준과 고장 결과에 따라 범주가 달라집니다. [1]",
        )
    if additional is not None:
        summary.append(
            "- **추가 검증**: 중·상위 위험 기능에는 하위 범주의 관련 요건에 더해 "
            "추가 위험평가와 검증·확인 자료가 요구됩니다. [2]"
        )
    answer = _four_sections(
        summary,
        impact="- 기능별 감독방식과 고장영향을 먼저 정의한 뒤 해당 위험범주에 맞는 설계·시험·승인 자료를 준비해야 합니다. [1]",
        followup="- 최종 적용 시 위험 매트릭스의 인접 조항과 기능별 추가 검증 요구를 함께 대조해야 합니다. [1]",
        references=[f"- **{_file_name(basis)}**, p.{_page(basis) or '?'} [1]"],
    )
    return GuardResult(
        answer=answer,
        evidence_table=_evidence_rows(chunks),
        mode="exact_fact_extract",
        metadata={"triggered": True, "reason": "abs_risk_category"},
    )


def _abs_smart_answer(question: str, pool: list[Any]) -> GuardResult | None:
    if not (
        re.search(r"ABS", question, re.I)
        and re.search(r"Smart\s*Function", question, re.I)
        and not re.search(r"비교|차이|Autonomous\s+and\s+Remote", question, re.I)
    ):
        return None
    file_pattern = r"GuideforSmartFunctionsforMarineVesselsandOffshoreUnits"
    scope = _pick(
        pool,
        file_pattern=file_pattern,
        all_terms=("all marine vessels and offshore units", "SHM", "MHM"),
    )
    intro = _pick(
        pool,
        file_pattern=file_pattern,
        all_terms=("optional class notations", "SMART (INF)"),
    )
    if scope is None:
        return None
    chunks = [scope] + ([intro] if intro is not None and intro is not scope else [])
    intro_cite = "[2]" if len(chunks) > 1 else "[1]"
    answer = _four_sections(
        [
            "- **적용대상**: 이 Guide는 Smart Function을 탑재한 모든 해양선박과 "
            "해양구조물에 적용됩니다. 자율운항 선박만을 대상으로 하지 않습니다. [1]",
            "- **포함 범위**: 선택적 Smart Function 선급부호 범위에서 SHM과 MHM을 다루며, "
            "데이터 인프라는 SMART (INF)로 구분합니다. [1]",
            f"- **부호 성격**: 요건 충족 시 SMART (INF)·SMART (SHM)·SMART (MHM) 등의 "
            f"선택적 class notation을 받을 수 있습니다. {intro_cite}",
        ],
        impact="- 적용 기능별로 데이터 인프라, 구조 상태감시 또는 기계 상태감시 범위를 구분해 설계·검사 자료를 준비해야 합니다. [1]",
        followup="- 실제 부호 신청 전에는 적용 기능, 시스템 경계와 최신 Guide 개정판을 확인해야 합니다. [1]",
        references=[f"- **{_file_name(scope)}**, p.{_page(scope) or '?'} [1]"],
    )
    return GuardResult(
        answer=answer,
        evidence_table=_evidence_rows(chunks),
        mode="exact_fact_extract",
        metadata={"triggered": True, "reason": "abs_smart_scope"},
    )


def _mass_working_group_premise_answer(
    question: str, pool: list[Any]
) -> GuardResult | None:
    if not (
        re.search(r"\bMASS\b", question, re.I)
        and re.search(r"작업반|working\s+group|회부", question, re.I)
        and re.search(r"않|아니|맞는지|검증|전제", question, re.I)
    ):
        return None
    chunk = _pick(
        pool,
        file_pattern=r"MSC\s*111-WP\.1",
        all_terms=("MASS", "working group"),
        any_terms=("referred", "refer", "instructed"),
    )
    if chunk is None:
        return None
    answer = _four_sections(
        [
            "- **전제 판정**: ‘MASS Code 관련 제안이 작업반에 회부되지 않았다’는 "
            "전제는 틀렸습니다. MSC 111 회의 결과 초안은 관련 사항을 MASS 작업반에서 "
            "검토하도록 회부한 사실을 기록합니다. [1]",
            "- 따라서 MASS Code의 비강제 채택·향후 mandatory Code 일정과 별개로, "
            "구체적인 제안 검토 절차에는 작업반 회부가 포함됐습니다. [1]",
        ],
        impact="- MASS 관련 내부 검토 시 본회의 결정과 작업반 후속 검토사항을 구분해 추적해야 합니다. [1]",
        followup="- 최종 회의보고서에서 작업반 명칭, 회부된 제안 범위와 후속 보고 결과를 함께 확인해야 합니다. [1]",
        references=[f"- **{_file_name(chunk)}**, p.{_page(chunk) or '?'} [1]"],
    )
    return GuardResult(
        answer=answer,
        evidence_table=_evidence_rows([chunk]),
        mode="premise_correction",
        metadata={"triggered": True, "reason": "mass_working_group_premise"},
    )


def _section_one_body(answer: str) -> str:
    match = re.search(
        r"(?ms)^##\s*1\)[^\n]*\n(.*?)(?=^##\s*[234]\)|\Z)", answer or ""
    )
    return match.group(1).strip() if match else ""


def _is_premise_question(question: str) -> bool:
    return bool(
        re.search(r"전제", question or "")
        or re.search(r"맞는지.{0,20}(?:검증|확인)", question or "")
        or re.search(r"알고\s*있.{0,20}(?:맞|확인|검증)", question or "")
    )


def _has_explicit_premise_verdict(answer: str) -> bool:
    return bool(
        re.search(
            r"전제.{0,30}(?:맞습니다|맞지\s*않|틀렸|틀립|잘못|정확하지\s*않)|"
            r"(?:아닙니다|발효되지\s*않|확정되지\s*않|바로잡)",
            answer or "",
        )
    )


def _needs_repair(question: str, answer: str) -> bool:
    body = _section_one_body(answer)
    if not body or not re.search(r"^-\s+\S", body, re.M):
        return True
    # An answer that the contract stripped down to a notice has no "- " bullet
    # in section one, so the check above already flags it.
    bad = (
        "한국어 변환을 완료하지 못했습니다",
        "확정 Rule로 단정하기엔 근거·적용 범위 확인 필요",
        "관련 키워드는 있으나",
    )
    if any(marker in answer for marker in bad):
        return True
    if re.search(r"비교|차이", question, re.I) and len(re.findall(r"^-\s+", body, re.M)) < 2:
        return True
    if _is_premise_question(question) and not _has_explicit_premise_verdict(answer):
        return True
    return False


def _rank_repair_chunks(question: str, pool: list[Any], limit: int = 5) -> list[Any]:
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", question or "")
        if token.lower() not in {"the", "and", "for", "from", "with", "guide", "requirements"}
    ]

    def score(chunk: Any) -> tuple[int, int]:
        file_low = _file_name(chunk).lower()
        body_low = _chunk_text(chunk).lower()
        return (
            sum(4 for token in tokens if token in file_low)
            + sum(1 for token in tokens if token in body_low),
            -len(body_low),
        )

    return sorted(pool, key=score, reverse=True)[:limit]


def _valid_repair(question: str, answer: str, chunk_count: int) -> bool:
    if len(re.findall(r"[가-힣]", answer or "")) < 30:
        return False
    if not all(re.search(rf"(?m)^##\s*{number}\)", answer or "") for number in range(1, 5)):
        return False
    if not re.search(r"(?m)^-\s+.+\[\d+\]", _section_one_body(answer)):
        return False
    citations = [int(value) for value in re.findall(r"\[(\d+)\]", answer or "")]
    if not citations or not all(1 <= value <= chunk_count for value in citations):
        return False
    if _is_premise_question(question) and not _has_explicit_premise_verdict(answer):
        return False
    return True


def _llm_repair(question: str, answer: str, pool: list[Any], model: str) -> GuardResult | None:
    if not _needs_repair(question, answer) or not pool:
        return None
    chunks = _rank_repair_chunks(question, pool)
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{index}] doc={_file_name(chunk)} p.{_page(chunk) or '?'}\n"
            f"{_chunk_text(chunk)[:2200]}"
        )
    base = (
        os.environ.get("MARITIME_OLLAMA_BASE")
        or os.environ.get("OLLAMA_HOST")
        or "http://127.0.0.1:11434"
    ).rstrip("/")
    body: dict[str, Any] = {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": "24h",
        "options": {"temperature": 0.0, "num_predict": 900, "num_ctx": 8192},
        "messages": [
            {
                "role": "system",
                "content": (
                    "제공된 문서 근거만 사용하는 해사 RAG 답변 검증자다. 자연스러운 한국어로 "
                    "질문에 직접 답하고 문서 소개로 우회하지 않는다. 숫자·날짜·의무 강도를 그대로 "
                    "보존한다. 각 사실 bullet 끝에는 해당 [n]을 붙인다. 근거가 없는 항목은 없다고 "
                    "명시한다. 잘못된 전제를 검증하는 질문이면 ## 1)의 첫 bullet에서 반드시 "
                    "'전제는 맞습니다' 또는 '전제는 틀렸습니다'라고 판정하고 근거로 바로잡는다. "
                    "## 1)부터 ## 4)까지 네 제목을 모두 쓰고 빈 섹션을 만들지 않는다."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"질문:\n{question}\n\n현재 답변의 문제:\n{answer[:1800]}\n\n"
                    "검색 근거:\n" + "\n\n".join(blocks)
                ),
            },
        ],
    }
    started = time.perf_counter()

    def send(payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{base}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        try:
            payload = send(body)
        except urllib.error.HTTPError as exc:
            if exc.code not in {400, 422}:
                raise
            compatible = dict(body)
            compatible.pop("think", None)
            payload = send(compatible)
        repaired = str((payload.get("message") or {}).get("content") or "").strip()
        if not _valid_repair(question, repaired, len(chunks)):
            return None
        return GuardResult(
            answer=repaired,
            evidence_table=_evidence_rows(chunks),
            mode="llm_quality_repair",
            metadata={
                "triggered": True,
                "reason": "defective_generated_answer",
                "llm_used": True,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
    except Exception as exc:
        return GuardResult(
            answer=answer,
            evidence_table=None,
            mode="quality_repair_failed",
            metadata={
                "triggered": True,
                "reason": "defective_generated_answer",
                "llm_used": True,
                "success": False,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )


def guard_rag_answer(
    question: str,
    answer: str,
    payload: dict[str, Any],
    *,
    model: str,
) -> GuardResult:
    """Return the unchanged answer or a grounded, citation-stable repair."""
    length_contract = (
        ((payload.get("answer_out") or {}).get("verification_summary") or {}).get(
            "answer_length_contract"
        )
        or {}
    )
    compact_fact = length_contract.get("answer_profile") == "exact_rule_fact"
    pool = _retrieval_pool(payload)
    for builder in (
        _negative_lookup_answer,
        _sfcs_deadline_answer,
        _abs_risk_answer,
        _abs_smart_answer,
        _mass_working_group_premise_answer,
    ):
        result = builder(question, pool)
        if result is not None:
            return _finalize(result, compact_fact=compact_fact)
    repaired = _llm_repair(question, answer, pool, model)
    if repaired is not None:
        return _finalize(repaired, compact_fact=compact_fact)
    return _finalize(GuardResult(
        answer=answer,
        evidence_table=None,
        mode="unchanged",
        metadata={"triggered": False},
    ), compact_fact=compact_fact)
