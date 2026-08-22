"""On-premise Advanced RAG: listwise evidence reranking and answer review.

This module deliberately uses the already installed local Ollama model.  It
does not call an external API and it never makes Advanced a dependency of the
existing Fast/Accurate paths.  All public helpers fail closed: callers keep
the established Accurate result when the local model is unavailable or its
structured output is invalid.
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable


DEFAULT_ADVANCED_MODEL = (
    os.environ.get("MARITIME_ADVANCED_RERANK_MODEL")
    or os.environ.get("MARITIME_OLLAMA_MODEL")
    or "gemma4:12b"
).strip()
DEFAULT_OLLAMA_BASE = (
    os.environ.get("MARITIME_OLLAMA_BASE") or "http://127.0.0.1:11434"
).strip().rstrip("/")

_CITATION_RE = re.compile(r"\[(\d+)\]")
_CITATION_GROUP_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")
_DOC_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"(?:DNV|LR|ABS|KR)[-–/ ](?:CG|RP|RU|CP|OS|SI|NV)[-–/ ]?[A-Z0-9.-]+|"
    r"(?:MEPC|MSC)\s*\d{1,3}(?:\s*[-–/.]\s*[A-Z0-9]+)+|"
    r"(?:MEPC|MSC)\.\d+\(\d+\)"
    r")(?![A-Za-z0-9])",
    re.I,
)
_FACT_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_SAFE_UNCITED_RE = re.compile(
    r"확인(?:되지|할 수 없)|추가 확인|근거 부족|해당하지 않|없습니다|"
    r"원문 확인|미확정|해석 주의",
    re.I,
)


@dataclass(frozen=True)
class AdvancedRerankConfig:
    candidate_limit: int = 36
    output_k: int = 18
    preview_chars: int = 760
    num_ctx: int = 32768
    num_predict: int = 900


@dataclass(frozen=True)
class AdvancedRetrievalPlan:
    facets: tuple[str, ...] = ()
    covered: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    followup_queries: tuple[str, ...] = ()
    confidence: float = 0.0
    used_llm: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "facets": list(self.facets),
            "covered": list(self.covered),
            "missing": list(self.missing),
            "followup_queries": list(self.followup_queries),
            "confidence": self.confidence,
            "used_llm": self.used_llm,
        }


_ADVANCED_CACHE_MAX = 128
_ADVANCED_JSON_CACHE: OrderedDict[str, tuple[dict[str, Any] | None, dict[str, Any]]] = OrderedDict()
_ADVANCED_CACHE_LOCK = threading.Lock()


def clear_advanced_cache() -> None:
    with _ADVANCED_CACHE_LOCK:
        _ADVANCED_JSON_CACHE.clear()


def _cache_key(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cached_ollama_json(*, cache_scope: str, **kwargs: Any) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    key = _cache_key(cache_scope, kwargs)
    with _ADVANCED_CACHE_LOCK:
        cached = _ADVANCED_JSON_CACHE.get(key)
        if cached is not None:
            _ADVANCED_JSON_CACHE.move_to_end(key)
            payload, meta = cached
            return payload, {**meta, "cache_hit": True, "elapsed_seconds": 0.0}
    payload, meta = _ollama_json(**kwargs)
    if meta.get("ok"):
        with _ADVANCED_CACHE_LOCK:
            _ADVANCED_JSON_CACHE[key] = (payload, dict(meta))
            _ADVANCED_JSON_CACHE.move_to_end(key)
            while len(_ADVANCED_JSON_CACHE) > _ADVANCED_CACHE_MAX:
                _ADVANCED_JSON_CACHE.popitem(last=False)
    return payload, {**meta, "cache_hit": False}


def _value(obj: Any, name: str, default: Any = "") -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _chunk_text(chunk: Any) -> str:
    return re.sub(r"\s+", " ", str(_value(chunk, "text", "") or "")).strip()


def _chunk_id(chunk: Any) -> str:
    return str(_value(chunk, "chunk_id", "") or "")


def _file_name(chunk: Any) -> str:
    return str(_value(chunk, "file_name", "") or "")


def _doc_id(chunk: Any) -> str:
    return str(_value(chunk, "doc_id", "") or "")


def _page(chunk: Any) -> Any:
    return _value(chunk, "page_number", None) or _value(chunk, "page", None)


def _source(chunk: Any) -> str:
    return str(_value(chunk, "source", "") or "").upper()


def _status(chunk: Any) -> str:
    return str(_value(chunk, "document_status", "") or "unknown")


def _score_value(chunk: Any, name: str) -> float | None:
    value = _value(chunk, name, None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S).strip()
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(raw[start : end + 1])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def _ollama_json(
    *,
    model: str,
    system: str,
    user: str,
    num_ctx: int,
    num_predict: int,
    timeout: int = 240,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    started = time.perf_counter()
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": "24h",
        "format": "json",
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "repeat_penalty": 1.05,
        },
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    def send(body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{DEFAULT_OLLAMA_BASE}/api/chat",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        try:
            response = send(payload)
        except urllib.error.HTTPError as exc:
            # Older Ollama builds may not accept either ``format`` or ``think``.
            if exc.code not in {400, 404, 422}:
                raise
            compatible = dict(payload)
            compatible.pop("format", None)
            compatible.pop("think", None)
            response = send(compatible)
        content = str((response.get("message") or {}).get("content") or "")
        parsed = _extract_json_object(content)
        return parsed, {
            "ok": parsed is not None,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "model": model,
            "eval_count": response.get("eval_count"),
            "done_reason": response.get("done_reason"),
        }
    except Exception as exc:
        return None, {
            "ok": False,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "model": model,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _candidate_union(
    retrieved: Iterable[Any], pool: Iterable[Any], *, limit: int
) -> list[Any]:
    """Preserve Accurate order while adding document-diverse sparse candidates."""
    ordered: list[Any] = []
    seen: set[str] = set()
    for chunk in [*list(retrieved), *list(pool)]:
        cid = _chunk_id(chunk)
        key = cid or f"{_file_name(chunk)}:{_page(chunk)}:{len(ordered)}"
        if key in seen or not _chunk_text(chunk):
            continue
        seen.add(key)
        ordered.append(chunk)

    substantive = [chunk for chunk in ordered if len(_chunk_text(chunk)) >= 45]
    if len(substantive) >= min(8, limit):
        ordered = substantive

    if len(ordered) <= limit:
        return ordered

    # Give the model the strong prefix and at least one opportunity from as
    # many documents as possible.  This avoids a single repetitive PDF using
    # the complete listwise budget.
    prefix_n = min(18, max(10, limit // 2))
    selected = list(ordered[:prefix_n])
    selected_ids = {_chunk_id(chunk) for chunk in selected}
    seen_docs = {_doc_id(chunk) or _file_name(chunk) for chunk in selected}
    for chunk in ordered[prefix_n:]:
        doc_key = _doc_id(chunk) or _file_name(chunk)
        if doc_key in seen_docs:
            continue
        selected.append(chunk)
        seen_docs.add(doc_key)
        selected_ids.add(_chunk_id(chunk))
        if len(selected) >= limit:
            return selected
    for chunk in ordered[prefix_n:]:
        if _chunk_id(chunk) in selected_ids:
            continue
        selected.append(chunk)
        if len(selected) >= limit:
            break
    return selected


_OUTCOME_QUESTION_RE = re.compile(
    r"결과|결정|승인|채택|결론|outcomes?|decisions?|approved|adopted|agreed",
    re.I,
)
_OUTCOME_DOCUMENT_RE = re.compile(
    r"WP\.?\s*1|Draft\s+Report|Report\s+of\s+the\s+"
    r"(?:eleventh|twelfth|\d+\w*)\s+session|Resolution",
    re.I,
)
_OUTCOME_ACTION_RE = re.compile(
    r"\b(?:approved?|adopted?|agreed|endorsed?|invited|instructed|requested)\b|"
    r"\b(?:draft\s+)?resolution\b",
    re.I,
)


def _protected_outcome_literal_candidates(
    question: str,
    retrieved: Iterable[Any],
    pool: Iterable[Any],
) -> list[Any]:
    """Keep one authoritative decision paragraph per recovered literal facet.

    Advanced follow-up retrieval can find the exact English paragraph while a
    listwise small model still prefers a nearby discussion paragraph.  For a
    question explicitly asking for meeting outcomes, protect the strongest
    official outcome paragraph for each narrow bilingual recovery phrase.
    This is a selection guard only: it never manufactures evidence and it is
    inactive for proposal/comment questions or ordinary Rule queries.
    """
    if not _OUTCOME_QUESTION_RE.search(str(question or "")):
        return []
    try:
        from retrieval_search import extract_translated_feature_terms

        literals = extract_translated_feature_terms(question, limit=4)
    except (ImportError, AttributeError):
        return []
    if not literals:
        return []

    all_candidates = _candidate_union(retrieved, pool, limit=100000)
    protected: list[Any] = []
    protected_ids: set[str] = set()
    for literal in literals:
        needle = re.sub(r"\s+", " ", str(literal or "")).strip().lower()
        if not needle:
            continue

        def normalized_action(text: str) -> str:
            value = re.sub(r"\s+", " ", str(text or "")).lower()
            return re.sub(r"\bapproved\b|\bapproving\b", "approve", value)

        normalized_needle = normalized_action(needle)
        matches = [
            chunk
            for chunk in all_candidates
            if normalized_needle in normalized_action(_chunk_text(chunk))
            and _OUTCOME_DOCUMENT_RE.search(_file_name(chunk))
        ]
        if not matches:
            continue

        def score(chunk: Any) -> tuple[int, int, int]:
            text = _chunk_text(chunk)
            name = _file_name(chunk)
            return (
                1 if _OUTCOME_ACTION_RE.search(text) else 0,
                1 if re.search(r"WP\.?\s*1|Draft\s+Report|Resolution", name, re.I) else 0,
                -len(text),
            )

        best = max(matches, key=score)
        cid = _chunk_id(best)
        if cid not in protected_ids:
            protected.append(best)
            protected_ids.add(cid)
    return protected


_COMPLEX_QUESTION_RE = re.compile(
    r"(?:체크리스트|미확정|향후\s*(?:일정|계획)|추후\s*확인|운항.업무\s*영향|"
    r"설계\s*검토|비교|각각|동시에|관련\s*선급|mandatory|timeline|"
    r"어떤\s*것들|무엇들이|모두\s*(?:정리|열거)|목록|"
    r"결정.{0,20}일정|일정.{0,20}결정|"
    r"연료.{0,20}(?:안전|위험평가).{0,20}(?:관련|결과|추려)|"
    r"(?:요건|조건|예외).{0,30}(?:요건|조건|예외))",
    re.I,
)


def should_plan_retrieval(question: str) -> bool:
    """Use a planning call only when a single retrieval intent is insufficient."""
    text = re.sub(r"\s+", " ", str(question or "")).strip()
    if len(text) >= 95:
        return True
    if _COMPLEX_QUESTION_RE.search(text):
        return True
    conjunctions = len(re.findall(r"(?:및|그리고|또는|와|과|,)", text))
    asks = len(
        re.findall(
            r"(?:알려|정리|요약|작성|찾아|비교|설명|무엇|언제|어떻게)",
            text,
        )
    )
    return conjunctions >= 2 and asks >= 2


def _clean_plan_strings(values: Any, *, limit: int, max_chars: int) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if isinstance(raw, dict):
            raw = raw.get("query") or raw.get("name") or raw.get("facet")
        value = re.sub(r"\s+", " ", str(raw or "")).strip(" -•\t\r\n")
        if not value or len(value) > max_chars:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
        if len(out) >= limit:
            break
    return tuple(out)


def plan_retrieval_followups(
    question: str,
    retrieved: list[Any],
    pool: list[Any],
    *,
    model: str = DEFAULT_ADVANCED_MODEL,
) -> tuple[AdvancedRetrievalPlan, dict[str, Any]]:
    """Identify uncovered facets and produce at most two narrow search queries."""
    if not should_plan_retrieval(question):
        return AdvancedRetrievalPlan(confidence=1.0), {
            "used": False,
            "reason": "simple_question",
        }
    candidates = _candidate_union(retrieved, pool, limit=18)
    evidence_rows = [
        {
            "id": index,
            "document": _file_name(chunk),
            "page": _page(chunk),
            "clause": _value(chunk, "clause_number", ""),
            "status": _status(chunk),
            "text": _chunk_text(chunk)[:900],
        }
        for index, chunk in enumerate(candidates, 1)
    ]
    system = (
        "당신은 온프레미스 해사 규정 RAG의 검색 계획 모델이다. 답변을 작성하지 않는다. "
        "질문을 서로 중복되지 않는 필수 근거 항목으로 나누고, 제공된 후보가 각 항목을 "
        "직접 뒷받침하는지 판정한다. 숫자·단위·조건·예외·회의 결정 상태·일정을 각각 "
        "독립 항목으로 본다. 누락된 항목에 대해서만 최대 2개의 짧고 구체적인 검색어를 "
        "작성한다. 문서번호·선급·회의차수·제외 조건을 바꾸지 않는다. 한국어 질문과 영문 "
        "PDF를 연결할 수 있도록 검색어에는 원문의 예상 영문 기술용어도 함께 포함한다. "
        "추론 설명 없이 JSON만 반환한다."
    )
    user = (
        f"질문:\n{question}\n\n현재 후보(JSON):\n"
        + json.dumps(evidence_rows, ensure_ascii=False)
        + "\n\n반환 형식: "
        '{"facets":["필수 항목"],"covered":["직접 근거가 있는 항목"],'
        '"missing":["직접 근거가 없는 항목"],"followup_queries":["검색어"],'
        '"confidence":0.0}'
    )
    parsed, meta = _cached_ollama_json(
        cache_scope="retrieval_plan_v1",
        model=model,
        system=system,
        user=user,
        num_ctx=24576,
        num_predict=700,
    )
    if not parsed:
        return AdvancedRetrievalPlan(), {**meta, "used": False, "reason": "invalid_plan"}
    facets = _clean_plan_strings(parsed.get("facets"), limit=8, max_chars=180)
    covered = _clean_plan_strings(parsed.get("covered"), limit=8, max_chars=180)
    # A small model occasionally returns candidate ordinals ("1", "2")
    # instead of facet names.  Treat that as unknown coverage rather than
    # allowing the confidence gate to overstate completeness.
    if covered and all(re.fullmatch(r"\d+", value) for value in covered):
        covered = ()
    missing = _clean_plan_strings(parsed.get("missing"), limit=6, max_chars=180)
    followups = _clean_plan_strings(
        parsed.get("followup_queries"), limit=2, max_chars=240
    )
    # Gemma can emit candidate ordinals in ``missing`` while still producing
    # useful follow-up queries.  Never expose those ordinals as factual gaps;
    # use the actual narrow query labels as the conservative missing facets.
    if missing and all(re.fullmatch(r"\d+", value) for value in missing):
        missing = tuple(followups)
    if not missing:
        followups = ()
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    plan = AdvancedRetrievalPlan(
        facets=facets,
        covered=covered,
        missing=missing,
        followup_queries=followups,
        confidence=confidence,
        used_llm=True,
    )
    return plan, {
        **meta,
        "used": True,
        "plan": plan.to_dict(),
        "candidate_count": len(candidates),
    }


def retrieval_confidence(
    question: str,
    retrieved: list[Any],
    *,
    plan: AdvancedRetrievalPlan | None = None,
    rerank_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose a conservative evidence-confidence gate without suppressing facts."""
    rows = [chunk for chunk in retrieved if len(_chunk_text(chunk)) >= 45]
    direct_docs = len({_doc_id(chunk) or _file_name(chunk) for chunk in rows})
    named_ok = not any(_DOC_CODE_RE.findall(question or "")) or any(
        _explicitly_named_candidate(question, chunk) for chunk in rows
    )
    missing = list((rerank_meta or {}).get("missing") or [])
    if plan is not None and plan.missing:
        covered_lower = " ".join((rerank_meta or {}).get("coverage") or []).lower()
        missing.extend(item for item in plan.missing if item.lower() not in covered_lower)
    missing = list(dict.fromkeys(str(item) for item in missing if str(item).strip()))
    score = 0.25
    score += min(0.35, len(rows) * 0.035)
    score += 0.15 if named_ok else -0.25
    score += min(0.15, direct_docs * 0.03)
    score -= min(0.45, len(missing) * 0.15)
    score = round(max(0.0, min(1.0, score)), 3)
    if missing:
        score = min(score, 0.69)
    level = "high" if score >= 0.75 else "medium" if score >= 0.5 else "low"
    return {
        "score": score,
        "level": level,
        "named_document_satisfied": named_ok,
        "substantive_evidence_count": len(rows),
        "document_count": direct_docs,
        "missing": missing,
        "answer_policy": "answer_with_caveat" if level == "low" else "answer",
    }


def _explicitly_named_candidate(question: str, chunk: Any) -> bool:
    named = {
        re.sub(r"[^a-z0-9]", "", value.lower())
        for value in _DOC_CODE_RE.findall(question or "")
        if value
    }
    if not named:
        return False
    blob = re.sub(
        r"[^a-z0-9]",
        "",
        f"{_doc_id(chunk)} {_file_name(chunk)} {_chunk_text(chunk)[:300]}".lower(),
    )
    return any(value and value in blob for value in named)


def _parse_ranked_ids(payload: dict[str, Any] | None, count: int) -> list[int]:
    if not payload:
        return []
    raw = payload.get("ranked_ids") or payload.get("ranking") or payload.get("ids") or []
    ids: list[int] = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict):
            item = item.get("id") or item.get("candidate_id")
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= value <= count and value not in ids:
            ids.append(value)
    return ids


def listwise_rerank(
    question: str,
    retrieved: list[Any],
    pool: list[Any],
    *,
    model: str = DEFAULT_ADVANCED_MODEL,
    config: AdvancedRerankConfig | None = None,
    cross_encoder_query: str = "",
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """Use local Gemma as an actual listwise model over fused candidates."""
    cfg = config or AdvancedRerankConfig()
    protected_outcomes = _protected_outcome_literal_candidates(
        question, retrieved, pool
    )
    candidates = _candidate_union(
        [*protected_outcomes, *list(retrieved)],
        pool,
        limit=cfg.candidate_limit,
    )
    original_pool = _candidate_union(retrieved, pool, limit=max(len(pool), len(retrieved)))
    target_k = max(1, min(cfg.output_k, len(candidates)))
    if len(candidates) < 3:
        return list(retrieved), original_pool, {
            "used": False,
            "reason": "too_few_candidates",
            "candidate_count": len(candidates),
        }

    cross_encoder_meta: dict[str, Any] = {}
    cross_encoder_scores: list[float] = []
    try:
        from local_cross_encoder_reranker import score_candidates

        cross_encoder_scores, cross_encoder_meta = score_candidates(
            cross_encoder_query or question,
            candidates,
        )
    except Exception as exc:
        cross_encoder_meta = {
            "used": False,
            "reason": "cross_encoder_import_error",
            "error": f"{type(exc).__name__}: {exc}",
        }

    rows: list[str] = []
    for index, chunk in enumerate(candidates, 1):
        scores = {
            key: value
            for key, value in {
                "dense": _score_value(chunk, "dense_score"),
                "bm25": _score_value(chunk, "bm25_score"),
                "rrf": _score_value(chunk, "rrf_score"),
            }.items()
            if value is not None
        }
        rows.append(
            json.dumps(
                {
                    "id": index,
                    "source": _source(chunk),
                    "document": _file_name(chunk),
                    "document_id": _doc_id(chunk),
                    "page": _page(chunk),
                    "clause": _value(chunk, "clause_number", ""),
                    "status": _status(chunk),
                    "retrieval_scores": scores,
                    "cross_encoder_score": (
                        round(cross_encoder_scores[index - 1], 6)
                        if len(cross_encoder_scores) == len(candidates)
                        else None
                    ),
                    "text": _chunk_text(chunk)[: cfg.preview_chars],
                },
                ensure_ascii=False,
            )
        )
    system = (
        "당신은 선박 규정 PDF RAG의 근거 순위 모델이다. 후보 전체를 서로 비교하는 "
        "listwise reranker로만 행동한다. 답변을 작성하지 말고 JSON만 반환한다. "
        "질문에 직접 답하는 정의·요건·예외·수치·결정 문장을 우선한다. 문서번호와 선급 "
        "포함/제외 조건을 지킨다. 최종 결정 질문에서는 Resolution/공식 회의결과를 "
        "Proposal/Comments/INF/J보다 우선하고, 제안 자체를 묻는 질문에서는 제안을 "
        "유지한다. 여러 항목을 요구하면 한 문서의 반복 청크보다 요구 항목을 함께 "
        "충족하는 근거 구성을 우선한다. 표의 정확한 셀과 인접 헤더는 함께 보존한다."
    )
    user = (
        f"질문:\n{question}\n\n후보(JSONL):\n"
        + "\n".join(rows)
        + f"\n\n가장 유용한 후보 {target_k}개를 순서대로 고르세요. "
        '형식: {"ranked_ids":[정수,...],"coverage":["충족한 질문 항목"],'
        '"missing":["후보에서 확인되지 않은 항목"]}'
    )
    parsed, call_meta = _cached_ollama_json(
        cache_scope="listwise_rerank_v2",
        model=model,
        system=system,
        user=user,
        num_ctx=cfg.num_ctx,
        num_predict=cfg.num_predict,
    )
    ranked_ids = _parse_ranked_ids(parsed, len(candidates))
    if len(ranked_ids) < min(6, target_k):
        return list(retrieved), original_pool, {
            **call_meta,
            "used": False,
            "reason": "invalid_or_short_ranking",
            "candidate_count": len(candidates),
            "valid_rank_count": len(ranked_ids),
            "cross_encoder": cross_encoder_meta,
        }

    ranked = [candidates[value - 1] for value in ranked_ids]
    ranked_chunk_ids = {_chunk_id(chunk) for chunk in ranked}
    # Exact official outcome paragraphs recovered for separate question facets
    # are hard evidence guards.  The listwise model may reorder them but may
    # not discard them in favour of nearby discussion/proposal paragraphs.
    for chunk in reversed(protected_outcomes):
        cid = _chunk_id(chunk)
        if cid in ranked_chunk_ids:
            ranked = [item for item in ranked if _chunk_id(item) != cid]
        ranked.insert(0, chunk)
        ranked_chunk_ids.add(cid)
    # An explicitly named document is a hard query constraint, not a preference
    # the model may discard.  Insert its strongest existing hit if necessary.
    named = [chunk for chunk in candidates if _explicitly_named_candidate(question, chunk)]
    for chunk in reversed(named[:3]):
        if _chunk_id(chunk) not in ranked_chunk_ids:
            ranked.insert(0, chunk)
            ranked_chunk_ids.add(_chunk_id(chunk))

    remainder = [
        chunk for chunk in original_pool if _chunk_id(chunk) not in ranked_chunk_ids
    ]
    final_pool = [*ranked, *remainder]
    final_retrieved = final_pool[:target_k]
    for rank, chunk in enumerate(final_retrieved, 1):
        try:
            setattr(chunk, "reranker_score", 1.0 / rank)
        except Exception:
            pass
    return final_retrieved, final_pool, {
        **call_meta,
        "used": True,
        "backend": "ollama_listwise",
        "candidate_count": len(candidates),
        "output_count": len(final_retrieved),
        "ranked_candidate_ids": ranked_ids,
        "protected_outcome_chunk_ids": [
            _chunk_id(chunk) for chunk in protected_outcomes
        ],
        "cross_encoder": cross_encoder_meta,
        "coverage": list((parsed or {}).get("coverage") or []),
        "missing": list((parsed or {}).get("missing") or []),
        "model": model,
    }


def rerank_retrieval_result(
    question: str,
    result: dict[str, Any],
    *,
    model: str = DEFAULT_ADVANCED_MODEL,
    cross_encoder_query: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    retrieved = list(result.get("retrieved") or [])
    pool = list(result.get("retrieval_pool") or retrieved)
    reranked, reranked_pool, meta = listwise_rerank(
        question,
        retrieved,
        pool,
        model=model,
        cross_encoder_query=cross_encoder_query,
    )
    if meta.get("used"):
        result = dict(result)
        result["retrieved"] = reranked
        result["retrieval_pool"] = reranked_pool
        config = dict(result.get("retrieval_config") or {})
        config["advanced_listwise_rerank"] = meta
        result["retrieval_config"] = config
    return result, meta


def _review_excerpt(raw_text: str, question: str, *, limit: int = 2600) -> str:
    text = str(raw_text or "")
    if not text:
        return ""
    try:
        from retrieval_search import extract_translated_feature_terms
        from fast_context import _question_focused_excerpt

        for anchor in extract_translated_feature_terms(question, limit=4):
            if str(anchor).lower() not in text.lower():
                continue
            focused = _question_focused_excerpt(text, str(anchor), max_chars=limit)
            if focused:
                return focused[:limit]
    except (ImportError, AttributeError):
        pass
    return text[:limit]


def _evidence_payload(
    evidence_table: list[dict[str, Any]], question: str = ""
) -> tuple[list[dict[str, Any]], set[int]]:
    rows: list[dict[str, Any]] = []
    allowed: set[int] = set()
    for position, row in enumerate(evidence_table[:20], 1):
        marker = str(row.get("citation_id") or row.get("citation") or f"[{position}]")
        found = _CITATION_RE.search(marker)
        citation_id = int(found.group(1)) if found else position
        allowed.add(citation_id)
        rows.append(
            {
                "citation": citation_id,
                "document": row.get("file_name") or row.get("document") or "",
                "page": row.get("page") or row.get("page_number"),
                "clause": row.get("clause_number") or row.get("clause") or "",
                "document_status": row.get("document_status") or "unknown",
                "evidence": _review_excerpt(
                    str(
                        row.get("review_text")
                        or row.get("chunk_preview")
                        or row.get("evidence")
                        or row.get("text")
                        or ""
                    ),
                    question,
                ),
            }
        )
    return rows, allowed


def _normalized_term(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", str(value or "").lower())


def _required_term_present(answer: str, raw_term: str) -> bool:
    normalized_answer = _normalized_term(answer)
    alternatives = [
        _normalized_term(value) for value in str(raw_term or "").split("|")
    ]
    alternatives = [value for value in alternatives if len(value) >= 3]
    if alternatives and any(value in normalized_answer for value in alternatives):
        return True
    # Accept a harmless wording change such as ``LPG 및 에탄`` for the
    # auditor's ``LPG 또는 에탄(ethane)`` while still requiring both entities.
    base = re.sub(r"\([^)]*\)", " ", str(raw_term or ""))
    normalized_base = _normalized_term(base)
    if len(normalized_base) >= 3 and normalized_base in normalized_answer:
        return True
    stop = {"또는", "그리고", "관련", "대한", "해당", "문서", "내용", "및"}
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9.-]{1,}|[가-힣]{2,}", base)
        if token.lower() not in stop
    ]
    return len(tokens) >= 2 and all(_normalized_term(token) in normalized_answer for token in tokens)


def _ground_uncited_bullets(text: str, evidence: list[dict[str, Any]]) -> str:
    """Attach a citation only when an uncited bullet has strong lexical support."""
    stop = {
        "검색", "근거", "관련", "문서", "해당", "사항", "필요", "선박", "운항",
        "경우", "대한", "위해", "통해", "합니다", "됩니다", "있습니다",
    }

    def terms(value: str) -> set[str]:
        return {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9.-]{2,}|[가-힣]{2,}", value)
            if token.lower() not in stop
        }

    evidence_terms = [
        (int(row.get("citation") or 0), terms(str(row.get("evidence") or "")))
        for row in evidence
        if int(row.get("citation") or 0) > 0
    ]
    output: list[str] = []
    for line in str(text or "").splitlines():
        if (
            not _FACT_BULLET_RE.match(line)
            or _SAFE_UNCITED_RE.search(line)
            or _CITATION_RE.search(line)
            or re.search(r":\s*$", line.strip())
        ):
            output.append(line)
            continue
        line_terms = terms(line)
        ranked = sorted(
            (
                (len(line_terms.intersection(candidate_terms)), citation)
                for citation, candidate_terms in evidence_terms
            ),
            reverse=True,
        )
        best_score, best_citation = ranked[0] if ranked else (0, 0)
        threshold = 2 if len(line_terms) <= 7 else 3
        if best_citation and best_score >= threshold:
            line = line.rstrip() + f" [{best_citation}]"
        output.append(line)
    return "\n".join(output)


def _drop_uncited_factual_bullets(text: str) -> str:
    """Remove unsupported auditor additions instead of rejecting a sound edit.

    Group labels ending in a colon and the fixed no-evidence placeholders are
    structural, not claims.  Every other factual bullet must already carry an
    atomic citation after ``_ground_uncited_bullets`` has had a chance to link
    it.  Dropping an ungrounded optional sentence is safer than discarding the
    auditor's entire otherwise useful checklist revision.
    """
    output: list[str] = []
    for line in str(text or "").splitlines():
        if (
            _FACT_BULLET_RE.match(line)
            and not _SAFE_UNCITED_RE.search(line)
            and not _CITATION_RE.search(line)
            and not re.search(r":\s*$", line.strip())
        ):
            continue
        output.append(line)
    return "\n".join(output)


def _normalize_bullet_citations(text: str) -> str:
    """Canonicalize model citation groups and remove duplicate markers.

    The rest of the service deliberately accepts only atomic ``[n]`` markers
    so each answer claim can be joined to one displayed Evidence Table row.
    Gemma sometimes emits ``[1, 3] [1]``; convert that to ``[1][3]`` without
    changing any factual text.
    """
    output: list[str] = []
    for line in str(text or "").splitlines():
        if not _FACT_BULLET_RE.match(line):
            output.append(line)
            continue
        ids: list[int] = []
        for group in _CITATION_GROUP_RE.findall(line):
            for value in re.findall(r"\d+", group):
                number = int(value)
                if number not in ids:
                    ids.append(number)
        if not ids:
            output.append(line)
            continue
        clean = _CITATION_GROUP_RE.sub("", line)
        clean = re.sub(r"\s+([,.;:])", r"\1", clean)
        clean = re.sub(r"\s{2,}", " ", clean).rstrip()
        # Avoid a punctuation fragment such as ``요건. . [1]`` after removing
        # a second citation group that followed its own full stop.
        clean = re.sub(r"(?:\s*[.]\s*){2,}$", ".", clean).rstrip()
        output.append(clean + " " + "".join(f"[{value}]" for value in ids))
    return "\n".join(output)


def _canonicalize_section_headings(text: str) -> str:
    """Render every Advanced answer with the four user-approved headings."""
    headings = {
        "1": "## 1) 핵심 요약",
        "2": "## 2) 선박 운항/업무 영향",
        "3": "## 3) 추후 확인 필요사항",
        "4": "## 4) 관련 선급 Rule / Guidance",
    }
    output: list[str] = []
    for line in str(text or "").splitlines():
        match = re.match(r"^##\s*([1-4])\)[^\n]*$", line.strip())
        output.append(headings[match.group(1)] if match else line)
    return "\n".join(output)


def _ensure_advanced_four_section_shell(text: str) -> str:
    """Wrap a validated short answer in the formal Advanced UI contract."""
    canonical = _canonicalize_section_headings(text)
    present = set(re.findall(r"(?m)^##\s*([1-4])\)", canonical))
    if present == {"1", "2", "3", "4"}:
        return canonical
    # Only wrap genuinely compact renderers.  Partial numbered drafts remain
    # untouched so the auditor can restore their missing original sections.
    if present or len(canonical) > 1200:
        return canonical
    body = re.sub(r"(?m)^##\s*(?:답변|관련 Rule / Guidance)\s*$", "", canonical).strip()
    if not body:
        return canonical
    return "\n\n".join(
        (
            "## 1) 핵심 요약\n" + body,
            "## 2) 선박 운항/업무 영향\n- 검색 근거에서 직접 확인되는 별도 운항·업무 영향이 없음",
            "## 3) 추후 확인 필요사항\n- 추가 확인 필요사항이 별도로 식별되지 않음",
            "## 4) 관련 선급 Rule / Guidance\n- 검색 근거에서 관련 선급 Rule / Guidance가 확인되지 않음",
        )
    )


def _bulletize_section_prose(text: str) -> str:
    """Turn auditor prose inside numbered sections into verifiable bullets."""
    output: list[str] = []
    inside_section = False
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if re.match(r"^##\s*[1-4]\)", line):
            inside_section = True
            output.append(raw)
            continue
        if (
            inside_section
            and line
            and not line.startswith(("-", "*", ">", "#"))
        ):
            output.append("- " + line)
        else:
            output.append(raw)
    return "\n".join(output)


def _ensure_advanced_premise_verdict(
    question: str, text: str, *, signal_text: str = ""
) -> str:
    """Put an explicit verdict first when the grounded answer refutes a premise."""
    if not re.search(r"전제.{0,40}(?:맞는지|검증)|틀리면.{0,40}바로잡", question, re.I):
        return text
    if re.search(r"전제는\s*(?:맞습니다|맞지\s*않습니다|틀렸습니다)", text, re.I):
        return text
    signal = f"{text}\n{signal_text}"
    if not re.search(
        r"과징금.{0,60}(?:명시|정의|산정|계산).{0,30}(?:않|없)|"
        r"전제.{0,30}(?:틀|옳지\s*않|맞지\s*않)",
        signal,
        re.I,
    ):
        return text
    lines = str(text or "").splitlines()
    in_summary = False
    for index, raw in enumerate(lines):
        if re.match(r"^##\s*1\)", raw.strip()):
            in_summary = True
            continue
        if in_summary and re.match(r"^##\s*[2-4]\)", raw.strip()):
            break
        if in_summary and _FACT_BULLET_RE.match(raw) and _CITATION_RE.search(raw):
            prefix = raw[: len(raw) - len(raw.lstrip())]
            body = raw.lstrip().lstrip("-* ")
            lines[index] = f"{prefix}- 전제는 맞지 않습니다. {body}"
            return "\n".join(lines)
    return text


def _compact_simple_rule_lookup_answer(text: str) -> str:
    """Keep document-discovery answers inside the agreed 2-3 fact budget.

    A broad evidence pool is useful for finding the right Guide, but it can
    tempt the final auditor to append survey and maintenance clauses that the
    user did not ask for.  This formatter only removes those optional details:
    it keeps up to two already cited document/scope bullets and one cited
    Rule/Guidance reference.  It never creates a factual claim.
    """
    section_pattern = re.compile(
        r"(?ms)^##\s*([1-4])\)[^\n]*\n(.*?)(?=^##\s*[1-4]\)|\Z)"
    )
    sections = {number: body.strip() for number, body in section_pattern.findall(text or "")}
    if "1" not in sections or "4" not in sections:
        return text

    def bullets(body: str) -> list[str]:
        return [
            match.group(0).strip()
            for match in re.finditer(
                r"(?ms)^\s*[-*]\s+.*?(?=^\s*[-*]\s+|\Z)", body or ""
            )
        ]

    candidates = [
        value for value in bullets(sections["1"]) if _CITATION_RE.search(value)
    ]
    document_candidates: list[str] = []
    seen_documents: set[str] = set()
    for value in candidates:
        match = re.search(
            r"\b(?:DNV|ABS|LR|KR)[-_ ]?(?:CG|CP|RU|GUIDE|RULE)[-_ A-Z0-9.]*\d\b",
            value,
            re.I,
        )
        if not match:
            continue
        identity = re.sub(r"\s+", "", match.group(0).upper())
        if identity in seen_documents:
            continue
        seen_documents.add(identity)
        document_candidates.append(value)
        if len(document_candidates) == 2:
            break
    direct = document_candidates[:2]
    for value in candidates:
        if len(direct) >= 2:
            break
        if value not in direct:
            direct.append(value)
    references = [value for value in bullets(sections["4"]) if _CITATION_RE.search(value)][:1]
    if not direct or not references:
        return text
    if len(direct) >= 2:
        titles: list[str] = []
        citation_ids: list[int] = []
        for value in direct:
            code = re.search(
                r"\b(?:DNV|ABS|LR|KR)[-_ ]?(?:CG|CP|RU|GUIDE|RULE)[-_ A-Z0-9.]*?\d\b"
                r"(?:\s+[A-Za-z][A-Za-z ]{1,80})?",
                value,
                re.I,
            )
            if code:
                titles.append(re.sub(r"\s+", " ", code.group(0)).strip(" *-"))
            for raw_id in _CITATION_RE.findall(value):
                citation_id = int(raw_id)
                if citation_id not in citation_ids:
                    citation_ids.append(citation_id)
        if len(titles) == 2 and citation_ids:
            references = [
                "- 직접 관련 문서: "
                + "; ".join(titles)
                + ". "
                + "".join(f"[{value}]" for value in citation_ids)
            ]
    return "\n\n".join(
        (
            "## 1) 핵심 요약\n" + "\n".join(direct),
            "## 2) 선박 운항/업무 영향\n- 질문에서 별도 운항·업무 영향을 요청하지 않음",
            "## 3) 추후 확인 필요사항\n- 검색 근거에서 별도 확인 필요사항이 식별되지 않음",
            "## 4) 관련 선급 Rule / Guidance\n" + references[0],
        )
    ).strip()


def _restore_required_sections(original: str, revised: str) -> str:
    """Restore only missing UI sections from the already validated answer."""
    pattern = re.compile(
        r"(?ms)(^##\s*(\d)\)[^\n]*\n.*?)(?=^##\s*\d\)|\Z)"
    )
    original_blocks = {
        number: block.strip() for block, number in pattern.findall(original or "")
    }
    revised_blocks = {
        number: block.strip() for block, number in pattern.findall(revised or "")
    }
    if len(original_blocks) < 3 or not revised_blocks:
        return revised
    if any(number not in original_blocks for number in revised_blocks):
        return revised
    return "\n\n".join(
        revised_blocks.get(number) or original_blocks[number]
        for number in sorted(original_blocks, key=int)
    ).strip()


def _valid_revised_answer(
    original: str,
    revised: str,
    allowed: set[int],
    required_terms: Iterable[str] = (),
) -> tuple[bool, str]:
    text = str(revised or "").strip()
    if len(text) < 80:
        return False, "too_short"
    if len(text) > max(8000, len(original) * 2 + 1200):
        return False, "too_long"
    citations = {int(value) for value in _CITATION_RE.findall(text)}
    original_citations = {int(value) for value in _CITATION_RE.findall(original or "")}
    if citations - allowed:
        return False, "out_of_range_citation"
    if original_citations and not citations:
        return False, "citations_removed"
    original_sections = set(re.findall(r"(?m)^##\s*(\d)\)", original or ""))
    revised_sections = set(re.findall(r"(?m)^##\s*(\d)\)", text))
    if any(section not in {"1", "2", "3", "4"} for section in revised_sections):
        return False, "unexpected_sections"
    if len(original_sections) >= 3 and not original_sections.issubset(revised_sections):
        return False, "required_sections_removed"
    for raw_term in required_terms:
        if not _required_term_present(text, str(raw_term or "")):
            return False, f"required_term_missing:{raw_term}"
    for line in text.splitlines():
        if not _FACT_BULLET_RE.match(line) or _SAFE_UNCITED_RE.search(line):
            continue
        if re.search(r":\s*$", line.strip()):
            # A grouping label such as ``- **mandatory MASS Code**:`` has no
            # factual payload; its indented child bullets carry citations.
            continue
        if not _CITATION_RE.search(line):
            return False, "uncited_fact_bullet"
    return True, "accepted"


def review_answer(
    question: str,
    answer: str,
    evidence_table: list[dict[str, Any]],
    *,
    model: str = DEFAULT_ADVANCED_MODEL,
    confidence: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Audit one generated answer and accept only a citation-safe revision."""
    original_answer = str(answer or "")
    answer = _ensure_advanced_four_section_shell(
        _normalize_bullet_citations(original_answer)
    )
    evidence, allowed = _evidence_payload(evidence_table, question)
    if not answer.strip() or not evidence:
        return answer, {
            "used": False,
            "reason": "missing_answer_or_evidence",
            "evidence_count": len(evidence),
        }
    scope_rule_lookup = bool(
        re.search(r"(?:Rule|Guidance|Guide|CG(?:-\d+)?|규칙|지침)", question, re.I)
        and re.search(r"명칭.{0,20}범위.{0,20}알려", question, re.I)
    )
    simple_rule_lookup = bool(
        re.search(r"(?:Rule|Guidance|Guide|CG(?:-\d+)?|규칙|지침)", question, re.I)
        and (
            re.search(r"찾아\s*줘|찾아줘|조회해\s*줘|목록을?\s*알려", question, re.I)
            or scope_rule_lookup
        )
        and not re.search(
            r"제안문|최종\s*결정|확정\s*근거|비교|차이|검토\s*체크|"
            r"요건|조건|예외|절차|목적|영향",
            question,
            re.I,
        )
    )
    profile_instruction = (
        " 이 질문은 단순 선급 문서 탐색이다. 답변의 핵심 사실은 문서명·범위·대표 "
        "근거를 합쳐 총 2~3개 bullet로 제한한다. 질문하지 않은 세부 검사·유지보수·"
        "서비스 평가를 추가하지 않는다. 현재 답변이 이미 직접 관련 문서를 정확히 "
        "제시하면 누락으로 간주해 주변 조항을 늘리지 않는다."
        if simple_rule_lookup
        else ""
    )
    if scope_rule_lookup:
        profile_instruction += (
            " 이 질문은 명칭과 범위를 함께 요구한다. Smart Vessel 문서와 자율운항 "
            "문서를 각각 한 bullet로 쓰고, 각 bullet 안에 정확한 CG 명칭과 그 문서의 "
            "적용 범위를 함께 적는다. 주변 참고 CG는 추가하지 않는다."
        )
    system = (
        "당신은 온프레미스 해사 규정 RAG의 최종 답변 감사자다. 검색 근거에 없는 사실을 "
        "추가하지 않는다. 질문의 모든 요구 항목, 숫자·단위·조건·예외, 문서의 권위와 "
        "Proposal/Report/Outcome/Resolution 상태, 회의 차수, 선급 범위를 점검한다. "
        "keep을 선택하기 전에 질문이 요구한 하위 항목을 하나씩 대조한다. 특히 일정 질문은 "
        "근거에 있는 채택·발효·시행 목표 연도를 빠뜨리지 않고, 자료 목록 질문은 해당 연도에 "
        "발행 예정이라고 직접 적힌 자료만 답한다. 하나의 근거 청크에 번호가 다른 여러 문단과 "
        "서로 다른 연도가 함께 있으면, 각 연도는 그 연도를 직접 서술한 문단에만 결속한다. "
        "앞 문단의 연도로 뒤 문단의 명시적 연도를 부정하거나 바꾸지 않는다. 질문이 특정 연도의 "
        "목록을 요구하고 근거가 '2026 ... include A; B; and C.'처럼 열거하면, 문장 앞의 "
        "공통 연도·범위는 마침표 전 A·B·C 모두에 적용된다. 각 항목에 연도가 반복되지 "
        "않았다는 이유로 B나 C를 제외하면 안 된다. 그 연도와 범위를 답변 첫 문장에 "
        "명시하고 열거된 모든 항목을 빠짐없이 각각 구분해 답한다. "
        "질문이 회의 '결과·결정·승인·채택'을 요구하면 공식 Report/WP/Resolution의 "
        "approved·adopted·agreed 문장을 결과로 우선한다. Proposal/Comments 문서의 제안이나 "
        "토론 문구를 최종 결정처럼 쓰지 않는다. 공식 결과 근거에 승인·채택 사항과 함께 "
        "'다음 회의 제출 요청', '작업계획 반영'처럼 아직 후속 단계인 관련 항목이 열거되어 "
        "있으면, 질문 범위에 속하는 각 항목을 승인·채택과 후속 단계로 구분해 누락 없이 "
        "요약한다. 근거에 없는 운항 영향을 늘리기 위해 핵심 결과를 빼지 않는다. "
        "검색 신뢰도가 low이고 missing 항목이 있으면 그 항목을 추정해 채우지 말고, "
        "답변의 추후 확인 필요사항에 무엇이 확인되지 않았는지 구체적으로 남긴다. "
        "출력 섹션은 1) 핵심 요약, 2) 선박 운항/업무 영향, 3) 추후 확인 필요사항, "
        "4) 관련 선급 Rule / Guidance의 네 개만 허용한다. 일정이나 별도 분석 제목을 "
        "5번째 섹션으로 만들지 말고 1) 또는 3)에 포함한다. "
        "답변이 충분하면 keep을 선택한다. 명백한 누락·왜곡이 있을 때만 revise하고, 기존 "
        "인용 번호를 근거와 정확히 연결해 유지한다. revise라면 누락을 실제로 복원했는지 "
        "검증할 수 있도록 revised_answer에 반드시 들어가야 할 원문 핵심어를 required_terms에 "
        "넣는다. 한영 대체어는 'Joint Group of Experts|공동 전문가 그룹'처럼 |로 묶는다. "
        "삭제해야 할 용어는 required_terms에 넣지 않는다."
        + profile_instruction
        + " 추론 과정 없이 JSON만 반환한다."
    )
    user = (
        f"질문:\n{question}\n\n현재 답변:\n{answer}\n\n"
        "검색 근거 신뢰도(JSON):\n"
        + json.dumps(confidence or {}, ensure_ascii=False)
        + "\n\n"
        "답변 유형: "
        + ("단순 Rule/Guidance 문서 탐색(핵심 사실 2~3개 bullet)" if simple_rule_lookup else "일반 규정 분석")
        + "\n\n허용된 근거(JSON):\n"
        + json.dumps(evidence, ensure_ascii=False)
        + "\n\n반환 형식: "
        '{"decision":"keep 또는 revise","issues":["문제"],'
        '"required_terms":["수정안에 반드시 포함될 핵심어"],'
        '"revised_answer":"revise일 때만 완성된 한국어 Markdown 답변"}'
    )
    parsed, call_meta = _cached_ollama_json(
        cache_scope="answer_review_v2",
        model=model,
        system=system,
        user=user,
        num_ctx=32768,
        num_predict=1800,
    )
    decision = str((parsed or {}).get("decision") or "keep").strip().lower()
    issues = list((parsed or {}).get("issues") or [])
    required_terms = _clean_plan_strings(
        (parsed or {}).get("required_terms"), limit=10, max_chars=120
    )
    if decision != "revise":
        final_answer = (
            _compact_simple_rule_lookup_answer(answer)
            if simple_rule_lookup
            else answer
        )
        return final_answer, {
            **call_meta,
            "used": bool(parsed),
            "decision": "keep",
            "issues": issues,
            "revision_accepted": False,
            "simple_rule_compacted": final_answer != answer,
        }
    revised = _restore_required_sections(
        answer,
        _canonicalize_section_headings(
            _normalize_bullet_citations(
                _drop_uncited_factual_bullets(
                    _ground_uncited_bullets(
                        _bulletize_section_prose(
                            str((parsed or {}).get("revised_answer") or "").strip()
                        ),
                        evidence,
                    )
                )
            )
        ),
    )
    valid, reason = _valid_revised_answer(
        answer, revised, allowed, required_terms=required_terms
    )
    repair_meta: dict[str, Any] = {}
    if not valid and (
        reason
        in {
            "uncited_fact_bullet",
            "citations_removed",
            "out_of_range_citation",
            "unexpected_sections",
            "required_sections_removed",
        }
        or reason.startswith("required_term_missing:")
    ):
        repair_user = (
            user
            + "\n\n직전 수정안은 안전 검사에서 거절되었습니다. "
            + f"거절 사유: {reason}. 모든 사실 bullet 끝에 허용된 [n]을 붙이고, "
            "허용 근거에 없는 문장은 삭제하고 섹션은 1)~4)만 사용한 뒤 JSON 형식으로 다시 작성하세요.\n"
            + (
                "반드시 포함할 핵심어: " + " | ".join(required_terms) + "\n"
                if required_terms
                else ""
            )
            + f"직전 수정안:\n{revised}"
        )
        repaired_payload, repair_meta = _cached_ollama_json(
            cache_scope="answer_review_repair_v2",
            model=model,
            system=system,
            user=repair_user,
            num_ctx=32768,
            num_predict=1800,
        )
        repaired = _restore_required_sections(
            answer,
            _canonicalize_section_headings(
                _normalize_bullet_citations(
                    _drop_uncited_factual_bullets(
                        _ground_uncited_bullets(
                            _bulletize_section_prose(
                                str((repaired_payload or {}).get("revised_answer") or "").strip()
                            ),
                            evidence,
                        )
                    )
                )
            ),
        )
        repaired_valid, repaired_reason = _valid_revised_answer(
            answer, repaired, allowed, required_terms=required_terms
        )
        if repaired_valid:
            revised = repaired
            valid = True
            reason = "accepted_after_citation_repair"
            issues = list((repaired_payload or {}).get("issues") or issues)
        else:
            reason = f"{reason};repair:{repaired_reason}"
    if not valid:
        final_answer = (
            _compact_simple_rule_lookup_answer(answer)
            if simple_rule_lookup
            else answer
        )
        return final_answer, {
            **call_meta,
            "used": True,
            "decision": "revise",
            "issues": issues,
            "revision_accepted": False,
            "rejection_reason": reason,
            "revised_preview": revised[:1200],
            "repair_attempt": repair_meta,
            "simple_rule_compacted": final_answer != answer,
        }
    final_answer = (
        _compact_simple_rule_lookup_answer(revised)
        if simple_rule_lookup
        else revised
    )
    final_answer = _ensure_advanced_premise_verdict(
        question,
        final_answer,
        signal_text=str((parsed or {}).get("revised_answer") or "") + "\n" + answer,
    )
    return final_answer, {
        **call_meta,
        "used": True,
        "decision": "revise",
        "issues": issues,
        "revision_accepted": True,
        "validation": reason,
        "repair_attempt": repair_meta,
        "required_terms": list(required_terms),
        "simple_rule_compacted": final_answer != revised,
    }
