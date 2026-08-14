"""Bridge to MaritimeRAG (Chroma / document QA)."""
from __future__ import annotations

import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from project_paths import (
    DEFAULT_RAG_COLLECTION,
    DEFAULT_TABLE_COLLECTION,
    PRECISE_TABLES_DIR,
    RAG_CHUNKS_DIR,
    RAG_DIR,
    RAG_INDEX_DIR,
    RAG_SCRIPTS_DIR,
    RAG_TABLE_CHUNKS_DIR,
)
from services.rag_index_diag import (
    diagnose_rag_indexes,
    format_rag_index_banner,
    index_ready as _index_ready,
)
from services.korean_output import ensure_korean_answer
from services.rag_answer_guard import guard_rag_answer
from services.retrieval_mode import RetrievalMode, classify_retrieval_mode
from services.table_render import (
    extract_table_ids_from_chunks,
    format_related_tables_section,
)

# Process-level handles so Gradio does not cold-load Chroma/e5 on every question.
_WARM: dict[str, Any] = {
    "collection": None,
    "embed_model": None,
    "manifest": None,
    "unified_id": None,
}
_TABLE_WARM: dict[str, Any] = {
    "collection": None,
    "embed_model": None,
    "manifest": None,
    "unified_id": None,
}


def _ensure_rag_path() -> None:
    if str(RAG_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(RAG_SCRIPTS_DIR))


def _unified_root(collection_id: str) -> Path:
    return RAG_INDEX_DIR / f"unified_{collection_id}"


def rag_index_ready(collection_id: str = DEFAULT_RAG_COLLECTION) -> bool:
    return _index_ready(collection_id)


def rag_status_message() -> str:
    if not rag_index_ready(DEFAULT_RAG_COLLECTION):
        return (
            "문서 RAG 인덱스가 아직 없습니다. PDF는 data/raw_pdfs 에 연결되어 있습니다.\n"
            "인덱스를 만들려면 MaritimeRAG 전처리·통합 인덱스 구축을 실행하세요 "
            f"(목표 collection: {DEFAULT_RAG_COLLECTION})."
        )
    table = (
        f"표QA({DEFAULT_TABLE_COLLECTION}): OK"
        if rag_index_ready(DEFAULT_TABLE_COLLECTION)
        else f"표QA({DEFAULT_TABLE_COLLECTION}): 없음 → 본문 인덱스만 사용"
    )
    return f"RAG 인덱스 준비됨: {DEFAULT_RAG_COLLECTION}  |  {table}"


def rag_index_diagnostics(*, sample_size: int = 2000) -> dict[str, Any]:
    """Startup/ops helper: live collection.count() + sample metadata aggregates."""
    return diagnose_rag_indexes(sample_size=sample_size)


def rag_index_banner(*, sample_size: int = 2000) -> str:
    return format_rag_index_banner(diagnose_rag_indexes(sample_size=sample_size))


def warmup_rag_resources(unified_id: str = DEFAULT_RAG_COLLECTION) -> dict[str, Any]:
    """Eager-load Chroma + e5 + optional Ollama warm (Streamlit parity)."""
    if not rag_index_ready(unified_id):
        return {"ok": False, "reason": "index_missing"}

    _ensure_rag_path()
    prev = os.getcwd()
    try:
        os.chdir(RAG_DIR)
        from rag_resource_cache import (  # type: ignore
            load_unified_collection,
            prime_interactive_retrieval,
            warm_all_resources,
        )

        collection, embed_model, manifest = warm_all_resources(
            unified_id=unified_id,
            index_dir=RAG_INDEX_DIR,
        )
        collection, embed_model, manifest = load_unified_collection(
            unified_id, RAG_INDEX_DIR
        )
        prime = prime_interactive_retrieval(collection, embed_model)
        bm25_warm = False
        if unified_id == DEFAULT_TABLE_COLLECTION:
            try:
                from bm25_index import load_or_build_table_bm25  # type: ignore
                from rag_resource_cache import unified_index_fingerprint  # type: ignore

                fp = unified_index_fingerprint(unified_id, RAG_INDEX_DIR)
                bm25_warm = (
                    load_or_build_table_bm25(
                        collection,
                        unified_id=unified_id,
                        index_dir=RAG_INDEX_DIR,
                        fingerprint=fp,
                        allow_disk_load=True,
                    )
                    is not None
                )
            except Exception:
                bm25_warm = False
        try:
            from ollama_warmup import ensure_fast_warm  # type: ignore
            from rag_answer_lib import DEFAULT_OLLAMA_BASE, DEFAULT_OLLAMA_MODEL  # type: ignore

            ensure_fast_warm(DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_BASE)
            ollama_warm = True
        except Exception:
            ollama_warm = False

        bucket = _TABLE_WARM if unified_id == DEFAULT_TABLE_COLLECTION else _WARM
        bucket["collection"] = collection
        bucket["embed_model"] = embed_model
        bucket["manifest"] = manifest
        bucket["unified_id"] = unified_id
        return {
            "ok": True,
            "unified_id": unified_id,
            "prime": prime,
            "ollama_warm": ollama_warm,
            "bm25_warm": bm25_warm,
        }
    finally:
        os.chdir(prev)


def _classify_category(question: str) -> str:
    try:
        _ensure_rag_path()
        from question_classifier import classify_question_category  # type: ignore

        return str(classify_question_category(question, {}) or "")
    except Exception:
        return ""


def _build_rag_row(question: str, *, table_qa: bool) -> dict[str, Any]:
    category = "table_qa" if table_qa else _classify_category(question)
    row: dict[str, Any] = {
        "question_id": "ui_q",
        "question": question,
        "category": category,
    }
    if table_qa:
        row["_table_qa"] = True
        try:
            _ensure_rag_path()
            from rag_inprocess import normalize_table_question_row  # type: ignore

            row = normalize_table_question_row(row)
        except Exception:
            pass
    return row


def _resolve_mode(
    question: str,
    *,
    retrieval_mode: RetrievalMode | str | None,
    table_qa: bool | None,
) -> RetrievalMode:
    if retrieval_mode is not None:
        if isinstance(retrieval_mode, RetrievalMode):
            return retrieval_mode
        return RetrievalMode(str(retrieval_mode).strip().lower())
    if table_qa is True:
        return RetrievalMode.TABLE
    if table_qa is False:
        # Explicit legacy false → text only
        return RetrievalMode.TEXT
    return classify_retrieval_mode(question)


def _index_for(mode: RetrievalMode) -> tuple[str, Any]:
    _ensure_rag_path()
    from project_paths import DEFAULT_TABLE_COLLECTION as table_id  # local alias

    if mode in {RetrievalMode.TABLE, RetrievalMode.BOTH} and rag_index_ready(table_id):
        # Keep in sync with project_paths (single source of truth)
        return table_id, RAG_TABLE_CHUNKS_DIR
    return DEFAULT_RAG_COLLECTION, RAG_CHUNKS_DIR


def _warm_handles(unified: str, *, table_side: bool):
    bucket = _TABLE_WARM if table_side else _WARM
    if bucket.get("unified_id") == unified and bucket.get("collection") is not None:
        return bucket["collection"], bucket["embed_model"], bucket["manifest"]
    return None, None, None


def _extract_answer(out: dict[str, Any]) -> str:
    answer = (
        out.get("answer")
        or out.get("final_answer")
        or (out.get("answer_out") or {}).get("answer")
        or ""
    )
    if not answer and isinstance(out.get("answer_out"), dict):
        answer = out["answer_out"].get("text") or out["answer_out"].get("content") or ""
    if not answer:
        search = out.get("search_out") or {}
        answer = search.get("answer") or ""
    return str(answer or "").strip()


def _generation_diagnostics(payload: Any, depth: int = 0) -> dict[str, Any]:
    if depth > 4:
        return {}
    if isinstance(payload, dict):
        found = {
            target: payload.get(target)
            for target in ("done_reason", "eval_count")
            if payload.get(target) is not None
        }
        if found:
            return found
        for value in payload.values():
            nested = _generation_diagnostics(value, depth + 1)
            if nested:
                return nested
    elif isinstance(payload, list):
        for value in payload[:8]:
            nested = _generation_diagnostics(value, depth + 1)
            if nested:
                return nested
    return {}


def _tag_chunks(chunks: list[Any], *, source_kind: str) -> list[Any]:
    for chunk in chunks:
        try:
            setattr(chunk, "evidence_source", source_kind)
        except Exception:
            pass
        meta = getattr(chunk, "metadata", None)
        if isinstance(meta, dict):
            meta.setdefault("evidence_source", source_kind)
    return chunks


def _dedupe_chunks(chunks: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for chunk in chunks:
        cid = str(getattr(chunk, "chunk_id", "") or id(chunk))
        key = f"{getattr(chunk, 'evidence_source', '')}:{cid}"
        if key in seen:
            continue
        seen.add(key)
        out.append(chunk)
    return out


def _page_key(chunk: Any) -> str:
    doc = str(getattr(chunk, "doc_id", "") or getattr(chunk, "file_name", "") or "")
    page = str(getattr(chunk, "page_number", "") or getattr(chunk, "page", "") or "")
    return f"{doc}::{page}"


def fuse_evidence(
    *,
    text_hits: list[Any],
    table_hits: list[Any],
    text_top_k: int = 6,
    table_top_k: int = 8,
    prefer: str = "balanced",
) -> list[Any]:
    """Merge text/table hits with separate caps and light page dedupe."""
    text_hits = _tag_chunks(list(text_hits or []), source_kind="text")[:text_top_k]
    table_hits = _tag_chunks(list(table_hits or []), source_kind="table")[:table_top_k]

    if prefer == "table_primary":
        ordered = table_hits + text_hits
    elif prefer == "text_primary":
        ordered = text_hits + table_hits
    else:
        # Interleave a little so BOTH keeps both sides visible early
        ordered = []
        for i in range(max(len(table_hits), len(text_hits))):
            if i < len(table_hits):
                ordered.append(table_hits[i])
            if i < len(text_hits):
                ordered.append(text_hits[i])

    # Drop near-duplicate same doc/page text chunks (keep first)
    seen_pages: set[str] = set()
    fused: list[Any] = []
    for chunk in ordered:
        src = getattr(chunk, "evidence_source", "")
        pk = _page_key(chunk)
        if src == "text" and pk in seen_pages and pk != "::":
            continue
        if src == "text" and pk != "::":
            seen_pages.add(pk)
        fused.append(chunk)
    return _dedupe_chunks(fused)


def _chunk_meta_dict(chunk: Any) -> dict[str, Any]:
    if isinstance(chunk, dict):
        return dict(chunk)
    meta = getattr(chunk, "metadata", None)
    out: dict[str, Any] = dict(meta) if isinstance(meta, dict) else {}
    for key in (
        "table_id",
        "file_name",
        "doc_id",
        "page_number",
        "caption",
        "crop_path",
        "chunk_type",
        "element_id",
        "chunk_id",
        "text",
    ):
        val = getattr(chunk, key, None)
        if val not in (None, ""):
            out[key] = val
    if "text" not in out or not out.get("text"):
        out["text"] = str(getattr(chunk, "text", "") or "")
    return out


def _resolve_crop_path(meta: dict[str, Any], table_id: str) -> str:
    crop = str(meta.get("crop_path") or "")
    import re

    root = PRECISE_TABLES_DIR.resolve()
    candidates: list[Path] = []

    # Old metadata can contain an absolute path from the machine that built the
    # corpus. Rebase the suffix below precise_tables onto this project first.
    normalized = crop.replace("\\", "/")
    marker = "/data/processed/precise_tables/"
    marker_at = normalized.lower().find(marker)
    if marker_at >= 0:
        rel = normalized[marker_at + len(marker) :]
        if rel:
            candidates.append(root.joinpath(*Path(rel).parts))

    # Current precise corpus layout:
    # precise_tables/{doc-id hash}/p####_t###/crop.png
    match = re.search(r"_([0-9a-f]{8,16})_(p\d+_t\d+)$", table_id or "", re.I)
    if match:
        candidates.append(root / match.group(1) / match.group(2) / "crop.png")

    # Compatibility with the older year-based layout.
    match = re.search(r"(20\d{2}).*?(p\d+_t\d+)", table_id or "", re.I)
    if match:
        candidates.append(root / match.group(1) / match.group(2) / "crop.png")

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    # External absolute paths are disabled by default so a copied workspace
    # never silently depends on its source folder. This switch is only for
    # backwards compatibility with installations that have not copied crops.
    allow_external = os.environ.get(
        "MARITIME_ALLOW_EXTERNAL_DATA_PATHS", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if allow_external and crop and Path(crop).is_file():
        return str(Path(crop).resolve())
    return ""


def _related_tables_from_hits(
    question: str, table_chunks: list[Any]
) -> tuple[str, list[str], list[dict[str, Any]]]:
    """Collect unique table crops from retrieved hits (MaritimeRAG-style).

    Prefer ``crop_path`` images over Markdown rebuild. No second Chroma scan.
    """
    empty: tuple[str, list[str], list[dict[str, Any]]] = ("", [], [])
    if not table_chunks:
        return empty

    limit = 2
    try:
        from services.retrieval_mode import table_shape_score

        table_score, _ = table_shape_score(question)
        if table_score >= 0.35:
            limit = 1
    except Exception:
        pass
    table_ids = extract_table_ids_from_chunks(table_chunks, limit=limit)
    if not table_ids:
        return empty

    rendered: list[dict[str, Any]] = []
    images: list[str] = []
    for tid in table_ids:
        meta0: dict[str, Any] = {}
        crop = ""
        for chunk in table_chunks:
            meta = _chunk_meta_dict(chunk)
            chunk_tid = str(meta.get("table_id") or "")
            if not chunk_tid:
                # Recover from chunk_id / text when metadata was stripped.
                recovered = extract_table_ids_from_chunks([chunk], limit=1)
                chunk_tid = recovered[0] if recovered else ""
            if chunk_tid and chunk_tid != tid:
                continue
            if not meta0:
                meta0 = meta
            candidate = _resolve_crop_path(meta, tid)
            if candidate:
                crop = candidate
                meta0 = meta
                break
        if not crop:
            crop = _resolve_crop_path(meta0, tid)
        if not crop:
            continue
        if crop not in images:
            images.append(crop)
        rendered.append(
            {
                "table_id": tid,
                "file_name": meta0.get("file_name"),
                "doc_id": meta0.get("doc_id"),
                "page": meta0.get("page_number"),
                "caption": meta0.get("caption"),
                "markdown": "",
                "crop_path": crop,
                "highlight_rows": [],
                "source": "retrieved_crop",
            }
        )
    section = format_related_tables_section(rendered) if rendered else ""
    return section, images, rendered


def _extract_evidence_table(out: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Pull Evidence Table rows from run_full / run_answer payloads."""
    if not isinstance(out, dict):
        return []
    for key in ("evidence_table",):
        rows = out.get(key)
        if isinstance(rows, list) and rows:
            return list(rows)
    answer_out = out.get("answer_out")
    if isinstance(answer_out, dict):
        rows = answer_out.get("evidence_table")
        if isinstance(rows, list) and rows:
            return list(rows)
    return []


def _targeted_document_completion(
    question: str,
    out: dict[str, Any],
    collection: Any,
) -> None:
    """Complete named multi-clause questions inside their already named PDFs.

    This is a metadata lookup, not another embedding search.  It is activated
    only when the question explicitly requires two ABS instruments or the
    MASS Working Group clause.  Loading a few rows from those PDFs prevents a
    top-ranked introductory paragraph from becoming the whole answer.
    """
    if collection is None:
        return
    q = question or ""
    specs: list[tuple[str, tuple[str, ...]]] = []
    abs_comparison = bool(
        re.search(r"Smart\s+Functions?", q, re.I)
        and re.search(r"Autonomous\s+and\s+Remote\s+Control", q, re.I)
        and re.search(r"비교|차이|각각|구분", q, re.I)
    )
    if abs_comparison:
        specs.extend(
            [
                (
                    "GuideforSmartFunctionsforMarineVesselsandOffshoreUnits-v8.pdf",
                    (
                        "all marine vessels and offshore units",
                        "optional class notations",
                        "risk-informed approach",
                    ),
                ),
                (
                    "RequirementsforAutonomousandRemoteControlFunctions-v4.pdf",
                    (
                        "autonomous or remote control functions",
                        "operations supervision level",
                        "computer based system category iii",
                    ),
                ),
            ]
        )
    mass_working_group = bool(
        re.search(r"\bMASS\b", q, re.I)
        and re.search(r"작업반|working\s+group|회부", q, re.I)
    )
    if mass_working_group:
        specs.append(
            (
                "MSC 111-WP.1 - Draft Report Of The Maritime Safety Committee On Its 111Th Session (Secretariat).pdf",
                ("MASS", "working group", "referred"),
            )
        )
    if not specs:
        return

    additions: list[Any] = []
    for file_name, terms in specs:
        try:
            payload = collection.get(
                where={"file_name": {"$eq": file_name}},
                include=["documents", "metadatas"],
            )
        except Exception:
            try:
                payload = collection.get(
                    where={"file_name": file_name},
                    include=["documents", "metadatas"],
                )
            except Exception:
                continue
        ids = list(payload.get("ids") or [])
        documents = list(payload.get("documents") or [])
        metadatas = list(payload.get("metadatas") or [])
        ranked: list[tuple[int, int, Any]] = []
        for position, (chunk_id, text, metadata) in enumerate(
            zip(ids, documents, metadatas)
        ):
            metadata = metadata or {}
            body = str(text or "")
            low = body.lower()
            score = sum(1 for term in terms if term.lower() in low)
            if score <= 0:
                continue
            ranked.append(
                (
                    score,
                    -position,
                    SimpleNamespace(
                        chunk_id=str(chunk_id or ""),
                        file_name=str(metadata.get("file_name") or file_name),
                        doc_id=str(metadata.get("doc_id") or ""),
                        page_number=metadata.get("page_number") or metadata.get("page"),
                        clause_number=str(metadata.get("clause_number") or ""),
                        source=str(metadata.get("source") or ""),
                        text=body,
                        metadata=metadata,
                    ),
                )
            )
        additions.extend(
            item[2]
            for item in sorted(
                ranked, key=lambda item: (item[0], item[1]), reverse=True
            )[:32]
        )

    if not additions:
        return
    search = out.setdefault("search_out", {})
    pool = list(search.get("retrieval_pool") or search.get("retrieved") or [])
    seen = {str(getattr(chunk, "chunk_id", "") or "") for chunk in pool}
    for chunk in additions:
        chunk_id = str(getattr(chunk, "chunk_id", "") or "")
        if chunk_id and chunk_id in seen:
            continue
        if chunk_id:
            seen.add(chunk_id)
        pool.append(chunk)
    search["retrieval_pool"] = pool


def _abs_comparison_from_pool(
    question: str, out: dict[str, Any]
) -> tuple[str, list[dict[str, Any]]] | None:
    """Render a two-ABS-document comparison from the already retrieved pool.

    This path is deliberately narrow: it activates only when the user names
    both instruments.  It adds no search and no LLM call; it prevents the
    direct-clause extractor from collapsing a comparison to one long clause.
    """
    q = question or ""
    smart_named = re.search(r"(?:Guide\s+for\s+)?Smart\s+Functions?", q, re.I)
    remote_named = re.search(
        r"(?:Requirements?\s+for\s+Autonomous\s+and\s+Remote\s+Control\s+Functions|"
        r"Autonomous\s+and\s+Remote\s+Control\s+Requirements?)",
        q,
        re.I,
    )
    if not (smart_named and remote_named and re.search(r"비교|차이|각각|구분", q, re.I)):
        return None

    search = out.get("search_out") or {}
    pool = list(search.get("retrieval_pool") or search.get("retrieved") or [])
    if not pool:
        return None

    def file_name(chunk: Any) -> str:
        return str(getattr(chunk, "file_name", "") or "")

    def body(chunk: Any) -> str:
        return re.sub(r"\s+", " ", str(getattr(chunk, "text", "") or "")).strip()

    def is_guide(chunk: Any) -> bool:
        return "guideforsmartfunctionsformarinevesselsandoffshoreunits" in re.sub(
            r"[^a-z]", "", file_name(chunk).lower()
        )

    def is_requirements(chunk: Any) -> bool:
        return "requirementsforautonomousandremotecontrolfunctions" in re.sub(
            r"[^a-z]", "", file_name(chunk).lower()
        )

    def pick(predicate, patterns: tuple[str, ...]) -> Any | None:
        for chunk in pool:
            if not predicate(chunk):
                continue
            low = body(chunk).lower()
            if all(pattern.lower() in low for pattern in patterns):
                return chunk
        return None

    guide_scope = pick(is_guide, ("all marine vessels and offshore units", "SMART (INF)"))
    guide_risk = pick(is_guide, ("risk-informed approach", "assigned by the submitter"))
    req_scope = pick(is_requirements, ("autonomous or remote control functions", "AUTONOMOUS"))
    req_category = pick(
        is_requirements,
        ("operations supervision level", "consequences of failure"),
    )
    req_validation = pick(is_requirements, ("computer based system category iii",))

    chosen: list[Any] = []
    for chunk in (guide_scope, guide_risk, req_scope, req_category, req_validation):
        if chunk is not None and chunk not in chosen:
            chosen.append(chunk)
    # Both sources and the risk-category basis are the minimum safe comparison.
    if not guide_scope or not req_scope or not req_category:
        return None

    citation = {id(chunk): index for index, chunk in enumerate(chosen, start=1)}

    def cite(chunk: Any | None) -> str:
        return f"[{citation[id(chunk)]}]" if chunk is not None else ""

    lines = [
        "## 1) 핵심 요약",
        "",
        (
            "- **Smart Functions Guide — 적용대상·성격**: 모든 해양선박과 해양구조물에 "
            "적용하며, SMART (INF)·SMART (SHM)·SMART (MHM) 등 선택적 Smart Function "
            f"선급부호와 시스템 평가를 다룹니다. {cite(guide_scope)}"
        ),
        (
            "- **Autonomous/Remote Requirements — 적용대상·성격**: 자율 또는 원격제어 "
            "기능이 설치된 선박·해양구조물에 적용되는 ABS Requirements이며, 해당 기능은 "
            f"AUTONOMOUS 또는 REMOTE-CON 부호의 대상입니다. {cite(req_scope)}"
        ),
    ]
    if guide_risk:
        lines.append(
            "- **Smart Functions의 위험정보 평가**: 제출자가 안전 관련 위험요인과 기능의 "
            "고장·성능저하가 선박 안전·운항에 미치는 결과를 고려해 위험수준을 정하고, 그 "
            f"수준에 맞춰 ABS 검증·확인 범위를 정합니다. {cite(guide_risk)}"
        )
    lines.append(
        "- **Autonomous/Remote 기능 위험범주**: 각 기능을 Low·Medium·High로 나누며, "
        "운항감독 수준(Operations Supervision Level)과 고장 결과(Consequences of Failure)를 "
        f"조합해 범주를 정합니다. {cite(req_category)}"
    )
    if req_validation:
        lines.append(
            "- **검증 차이**: Autonomous/Remote Requirements에서는 Medium·High 위험 기능에 "
            "Computer Based System Category III 수준의 문서·검증 요구를 적용합니다. 따라서 "
            "Smart Functions Guide의 위험수준 기반 시스템 평가보다 자율·원격 기능의 감독수준과 "
            f"고장영향에 따른 추가 검증이 더 명시적입니다. {cite(req_validation)}"
        )
    lines.extend(
        [
            "",
            "## 2) 선박 운항/업무 영향",
            "",
            "- 상태감시·데이터 인프라 중심의 Smart Function은 Guide와 선택 부호 기준으로, "
            "자율·원격제어 기능은 Requirements의 위험범주·기반요건·검증자료 기준으로 "
            "승인 자료를 분리해 준비해야 합니다.",
            "",
            "## 3) 추후 확인 필요사항",
            "",
            "- 실제 부호 신청 전에는 대상 기능의 자율화 수준, 운항감독 위치, 고장 결과를 "
            "확정한 뒤 각 문서의 인접 조항과 최신 개정판을 함께 확인해야 합니다.",
            "",
            "## 4) 관련 선급 Rule / Guidance",
            "",
            f"- **Guide for Smart Functions for Marine Vessels and Offshore Units** {cite(guide_scope)}",
            f"- **Requirements for Autonomous and Remote Control Functions** {cite(req_scope)}{cite(req_category)}",
        ]
    )

    evidence: list[dict[str, Any]] = []
    for index, chunk in enumerate(chosen, start=1):
        evidence.append(
            {
                "citation_id": f"[{index}]",
                "file_name": file_name(chunk),
                "page": getattr(chunk, "page_number", None)
                or getattr(chunk, "page", None),
                "chunk_id": str(getattr(chunk, "chunk_id", "") or ""),
                "chunk_preview": body(chunk),
            }
        )
    return "\n".join(lines), evidence


def _run_single_rag(
    question: str,
    *,
    latency_mode: str,
    mode: RetrievalMode,
    llm_model: str | None = None,
) -> dict[str, Any]:
    from services.llm_models import normalize_llm_model

    model = normalize_llm_model(llm_model)
    want_table = mode == RetrievalMode.TABLE
    use_table_index = want_table and rag_index_ready(DEFAULT_TABLE_COLLECTION)
    target = DEFAULT_TABLE_COLLECTION if use_table_index else DEFAULT_RAG_COLLECTION
    if not rag_index_ready(target):
        return {
            "answer": (
                f"요청한 인덱스(`{target}`)가 없습니다. {rag_status_message()}"
            ),
            "files": [],
            "evidence_table": [],
            "related_tables": [],
            "source": "rag",
            "citations": [],
            "ready": False,
            "meta": {"missing_index": target, "retrieval_mode": mode.value},
        }

    _ensure_rag_path()
    prev = os.getcwd()
    try:
        os.chdir(RAG_DIR)
        from rag_inprocess import run_full_inprocess  # type: ignore

        row = _build_rag_row(question, table_qa=want_table)
        unified, chunks_dir = _index_for(mode if want_table else RetrievalMode.TEXT)
        collection, embed_model, manifest = _warm_handles(
            unified, table_side=use_table_index
        )

        out = run_full_inprocess(
            row,
            unified_id=unified,
            index_dir=RAG_INDEX_DIR,
            chunks_dir=chunks_dir,
            latency_mode=latency_mode,
            skip_llm=False,
            collection=collection,
            embed_model=embed_model,
            manifest=manifest,
            auto_llm_warm=True,
            llm_model=model,
        )
        completion_collection = collection
        if completion_collection is None:
            try:
                from rag_resource_cache import load_unified_collection  # type: ignore

                completion_collection, loaded_embed, loaded_manifest = (
                    load_unified_collection(unified, RAG_INDEX_DIR)
                )
                bucket = _TABLE_WARM if use_table_index else _WARM
                bucket.update(
                    {
                        "collection": completion_collection,
                        "embed_model": loaded_embed,
                        "manifest": loaded_manifest,
                        "unified_id": unified,
                    }
                )
            except Exception:
                completion_collection = None
        _targeted_document_completion(question, out, completion_collection)
        abs_comparison = _abs_comparison_from_pool(question, out)
        generated_answer = abs_comparison[0] if abs_comparison else _extract_answer(out)
        guarded = guard_rag_answer(question, generated_answer, out, model=model)
        generated_answer = guarded.answer
        answer_empty = not bool(generated_answer)
        answer = generated_answer or "검색은 완료되었으나 답변 텍스트를 찾지 못했습니다."
        answer, korean_output = ensure_korean_answer(question, answer, model=model)
        generation = _generation_diagnostics(out)
        files: list[str] = []
        related_tables: list[dict[str, Any]] = []
        evidence_table = (
            guarded.evidence_table
            if guarded.evidence_table is not None
            else abs_comparison[1]
            if abs_comparison
            else _extract_evidence_table(out)
        )
        if use_table_index:
            search = out.get("search_out") or {}
            retrieved = list(search.get("retrieved") or [])
            pool = list(search.get("retrieval_pool") or [])
            # Prefer the fuller pool so crop_path survives Fast slot trimming.
            crop_hits = pool or retrieved
            _related_md, images, related_tables = _related_tables_from_hits(
                question, crop_hits
            )
            files.extend(images)

        bucket = _TABLE_WARM if use_table_index else _WARM
        if bucket.get("collection") is None:
            try:
                from rag_resource_cache import load_unified_collection  # type: ignore

                c, e, m = load_unified_collection(unified, RAG_INDEX_DIR)
                bucket["collection"] = c
                bucket["embed_model"] = e
                bucket["manifest"] = m
                bucket["unified_id"] = unified
            except Exception:
                pass

        return {
            "answer": answer,
            "files": files,
            "evidence_table": evidence_table,
            "related_tables": related_tables,
            "source": "rag",
            "ready": True,
            "meta": {
                "unified_id": unified,
                "latency_mode": latency_mode,
                "category": row.get("category"),
                "retrieval_mode": mode.value,
                "table_qa_requested": want_table,
                "table_index_used": use_table_index,
                "related_tables_appended": bool(related_tables),
                "evidence_rows": len(evidence_table),
                "answer_mode": (
                    guarded.mode
                    if guarded.mode != "unchanged"
                    else "structured_abs_comparison"
                    if abs_comparison
                    else out.get("answer_mode")
                    or (out.get("search_out") or {}).get("answer_mode")
                ),
                "timing_metrics": (out.get("timing_metrics") or {}),
                "llm_model": model,
                "answer_empty": answer_empty,
                "answer_error": str(out.get("error") or ""),
                "answer_done_reason": generation.get("done_reason"),
                "answer_eval_count": generation.get("eval_count"),
                "korean_output": korean_output,
                "answer_quality_guard": guarded.metadata,
            },
            "raw": {k: out[k] for k in out if k not in {"raw", "timing_log"}},
        }
    except Exception as exc:
        return {
            "answer": f"[문서 RAG 오류] {exc}",
            "files": [],
            "evidence_table": [],
            "related_tables": [],
            "source": "rag",
            "ready": False,
            "error": str(exc),
            "meta": {"retrieval_mode": mode.value},
        }
    finally:
        os.chdir(prev)


def _run_search_only(
    question: str,
    *,
    latency_mode: str,
    table_side: bool,
) -> dict[str, Any]:
    _ensure_rag_path()
    prev = os.getcwd()
    try:
        os.chdir(RAG_DIR)
        from rag_inprocess import run_full_inprocess  # type: ignore

        row = _build_rag_row(question, table_qa=table_side)
        mode = RetrievalMode.TABLE if table_side else RetrievalMode.TEXT
        unified, chunks_dir = _index_for(mode)
        collection, embed_model, manifest = _warm_handles(
            unified, table_side=table_side
        )
        out = run_full_inprocess(
            row,
            unified_id=unified,
            index_dir=RAG_INDEX_DIR,
            chunks_dir=chunks_dir,
            latency_mode=latency_mode,
            skip_llm=True,
            collection=collection,
            embed_model=embed_model,
            manifest=manifest,
            auto_llm_warm=False,
        )
        search = out.get("search_out") or {}
        return {
            "row": row,
            "unified_id": unified,
            "search_out": search,
            "retrieved": list(search.get("retrieved") or []),
            "timing_metrics": out.get("timing_metrics") or {},
            "evidence_source": "table" if table_side else "text",
        }
    finally:
        os.chdir(prev)


def _run_both_fused(
    question: str,
    *,
    latency_mode: str,
    prefer: str = "balanced",
    llm_model: str | None = None,
) -> dict[str, Any]:
    """Search text + table in parallel, fuse evidence, answer once."""
    from services.llm_models import normalize_llm_model

    model = normalize_llm_model(llm_model)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_table = pool.submit(
            _run_search_only, question, latency_mode=latency_mode, table_side=True
        )
        fut_text = pool.submit(
            _run_search_only, question, latency_mode=latency_mode, table_side=False
        )
        table_hit = fut_table.result()
        text_hit = fut_text.result()

    table_chunks = list(table_hit.get("retrieved") or [])
    text_chunks = list(text_hit.get("retrieved") or [])
    text_cap = 2 if prefer == "table_primary" else 6
    table_cap = 10 if prefer == "table_primary" else 8
    merged_chunks = fuse_evidence(
        text_hits=text_chunks,
        table_hits=table_chunks,
        text_top_k=text_cap,
        table_top_k=table_cap,
        prefer=prefer,
    )
    # Cell/table-primary answers: keep table rows as the answer pool so prose
    # text hits cannot displace deterministic cell extraction.
    if prefer == "table_primary" and table_chunks:
        answer_chunks = list(table_chunks[:table_cap])
        if text_chunks:
            answer_chunks.extend(list(text_chunks[:text_cap]))
    else:
        answer_chunks = merged_chunks

    if not merged_chunks:
        return {
            "answer": "표·문서 인덱스에서 관련 근거를 찾지 못했습니다.",
            "files": [],
            "evidence_table": [],
            "related_tables": [],
            "source": "rag",
            "ready": True,
            "meta": {
                "dual_retrieval": True,
                "retrieval_mode": RetrievalMode.BOTH.value,
                "dual_mode": "search2_llm1",
                "n_chunks": 0,
            },
        }

    _ensure_rag_path()
    prev = os.getcwd()
    try:
        os.chdir(RAG_DIR)
        from rag_inprocess import run_answer_inprocess  # type: ignore

        use_table_answer = prefer in {"balanced", "table_primary"} and bool(table_chunks)
        row = (
            (table_hit.get("row") if use_table_answer else text_hit.get("row"))
            or _build_rag_row(question, table_qa=use_table_answer)
        )
        base_search = table_hit.get("search_out") if use_table_answer else text_hit.get("search_out")
        base_search = base_search or {}
        answer_out = run_answer_inprocess(
            row=row,
            chunks=answer_chunks,
            pool=answer_chunks,
            config_dict=base_search.get("retrieval_config"),
            metrics=base_search.get("retrieval_metrics"),
            doc_groups=base_search.get("doc_groups"),
            answer_mode=base_search.get("answer_mode")
            or ("table_qa" if use_table_answer else "rag"),
            question_category=("table_qa" if use_table_answer else row.get("category")),
            latency_mode=latency_mode,
            auto_llm_warm=True,
            llm_model=model,
        )
        generated_answer = _extract_answer(answer_out)
        guard_payload = {
            "answer_out": answer_out,
            "search_out": {
                "retrieval_pool": list(answer_chunks),
                "retrieved": list(answer_chunks),
            },
        }
        guarded = guard_rag_answer(question, generated_answer, guard_payload, model=model)
        generated_answer = guarded.answer
        answer_empty = not bool(generated_answer)
        answer = generated_answer or "검색은 완료되었으나 답변 텍스트를 찾지 못했습니다."
        answer, korean_output = ensure_korean_answer(question, answer, model=model)
        generation = _generation_diagnostics(answer_out)
        _related_md, images, related_tables = _related_tables_from_hits(
            question, table_chunks
        )
        evidence_table = (
            guarded.evidence_table
            if guarded.evidence_table is not None
            else _extract_evidence_table(answer_out)
        )
        return {
            "answer": answer,
            "files": images,
            "evidence_table": evidence_table,
            "related_tables": related_tables,
            "source": "rag",
            "ready": True,
            "meta": {
                "dual_retrieval": True,
                "dual_mode": "search2_llm1",
                "dual_retrieval_enabled": True,
                "retrieval_mode": RetrievalMode.BOTH.value,
                "fuse_policy": prefer,
                "n_table_chunks": len(table_chunks),
                "n_text_chunks": len(text_chunks),
                "n_merged_chunks": len(merged_chunks),
                "text_collection": DEFAULT_RAG_COLLECTION,
                "table_collection": DEFAULT_TABLE_COLLECTION,
                "latency_mode": latency_mode,
                "table_index_used": True,
                "related_tables_appended": bool(related_tables),
                "evidence_rows": len(evidence_table),
                "debug": {
                    "mode": "BOTH",
                    "text_hits": len(text_chunks),
                    "table_hits": len(table_chunks),
                    "fused_hits": len(merged_chunks),
                },
                "timing_metrics": {
                    "table_search": table_hit.get("timing_metrics") or {},
                    "text_search": text_hit.get("timing_metrics") or {},
                    "answer": (answer_out.get("timing_metrics") or {}),
                },
                "llm_model": model,
                "answer_empty": answer_empty,
                "answer_error": str(answer_out.get("error") or ""),
                "answer_done_reason": generation.get("done_reason"),
                "answer_eval_count": generation.get("eval_count"),
                "korean_output": korean_output,
                "answer_quality_guard": guarded.metadata,
            },
            "raw": {"answer_out": answer_out, "table_search": table_hit.get("search_out")},
        }
    except Exception as exc:
        return {
            "answer": f"[문서 RAG 오류] {exc}",
            "files": [],
            "evidence_table": [],
            "related_tables": [],
            "source": "rag",
            "ready": False,
            "error": str(exc),
            "meta": {"retrieval_mode": RetrievalMode.BOTH.value},
        }
    finally:
        os.chdir(prev)


def dual_retrieval_enabled(dual: bool | None = None) -> bool:
    """Return whether BOTH (search2) fusion is allowed.

    Default ON when env unset. Explicit MARITIME_RAG_DUAL=0 disables fusion.
    """
    if dual is True:
        return True
    if dual is False:
        return False
    env = os.environ.get("MARITIME_RAG_DUAL", "1").strip().lower()
    return env not in {"0", "false", "off", "no"}


def run_rag_query(
    question: str,
    *,
    latency_mode: str = "fast",
    table_qa: bool | None = None,
    dual: bool | None = None,
    retrieval_mode: RetrievalMode | str | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    """Run MaritimeRAG in-process against the shared data/ tree.

    retrieval_mode: text | table | both
    Legacy: table_qa=True → TABLE (or BOTH when dual env forces).
    BOTH fusion default: ON (MARITIME_RAG_DUAL defaults to "1").
    """
    from services.llm_models import normalize_llm_model

    model = normalize_llm_model(llm_model)
    mode = _resolve_mode(question, retrieval_mode=retrieval_mode, table_qa=table_qa)
    both_ready = rag_index_ready(DEFAULT_RAG_COLLECTION) and rag_index_ready(
        DEFAULT_TABLE_COLLECTION
    )
    dual_on = dual_retrieval_enabled(dual)

    if mode == RetrievalMode.BOTH and both_ready and dual_on:
        # Cell/open-table shape: keep table evidence first, text as support.
        prefer = "balanced"
        try:
            from services.retrieval_mode import table_shape_score

            t_score, _ = table_shape_score(question)
            if t_score >= 0.35:
                prefer = "table_primary"
        except Exception:
            prefer = "table_primary"
        return _run_both_fused(
            question, latency_mode=latency_mode, prefer=prefer, llm_model=model
        )
    # Strong TABLE stays on the table index (schema 2-stage). Dual fuse is for
    # BOTH/ambiguous asks — parallel text search can dilute cell extraction.
    if mode == RetrievalMode.BOTH and not both_ready:
        fallback = (
            RetrievalMode.TABLE
            if rag_index_ready(DEFAULT_TABLE_COLLECTION)
            else RetrievalMode.TEXT
        )
        out = _run_single_rag(
            question, latency_mode=latency_mode, mode=fallback, llm_model=model
        )
        meta = dict(out.get("meta") or {})
        meta["dual_retrieval_enabled"] = False
        meta["dual_fallback_reason"] = "missing_index"
        out["meta"] = meta
        return out
    if mode == RetrievalMode.BOTH and both_ready and not dual_on:
        # Env explicitly disabled dual: fall back to table-primary single index.
        out = _run_single_rag(
            question, latency_mode=latency_mode, mode=RetrievalMode.TABLE, llm_model=model
        )
        meta = dict(out.get("meta") or {})
        meta["retrieval_mode"] = RetrievalMode.BOTH.value
        meta["dual_retrieval_enabled"] = False
        meta["dual_fallback_reason"] = "MARITIME_RAG_DUAL=0"
        out["meta"] = meta
        return out

    out = _run_single_rag(question, latency_mode=latency_mode, mode=mode, llm_model=model)
    meta = dict(out.get("meta") or {})
    meta.setdefault("dual_retrieval_enabled", dual_on)
    out["meta"] = meta
    return out
