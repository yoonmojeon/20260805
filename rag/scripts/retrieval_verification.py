"""Retrieval verification: question modes, evidence tables, coverage, trace logs."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_retrieval_metrics import content_preview

CITATION_RE = re.compile(r"\[(\d+)\]")

BROAD_PATTERNS = [
    r"최신",
    r"주요\s*내용",
    r"동향",
    r"정리해",
    r"환경규제\s*대응",
    r"미치는\s*영향",
    r"overview",
    r"summarize",
    r"요약해",
]

NARROW_SIGNAL_PATTERNS = [
    r"p\.?\s*\d+",
    r"page\s+\d+",
    r"\d+\.\d+절",
    r"section\s+\d+",
    r"문서에서",
    r"문서의",
    r"in document",
    r"from document",
    r"해당\s*문서",
]

DOC_REF_PATTERNS = [
    r"mepc\s*\d+[-/]\d+",
    r"dnv-cg-\d+",
    r"lr\s+notice",
    r"notice\s+no",
    r"abs\s+",
    r"kr\s+rule",
    r"84[-/]7[-/]14",
    r"0264",
]

MUST_COVER_BY_QID: dict[str, list[str]] = {
    "V01": [
        "IMO Net-Zero Framework",
        "GHG",
        "GFI",
        "CII",
        "SEEMP",
        "MARPOL Annex VI",
        "guidelines",
        "reporting",
        "verification",
        "ship operation impact",
    ],
    "V07": [
        "low-flashpoint fuel",
        "alternative fuel",
        "dual fuel",
        "fuel storage",
        "fuel supply",
        "engine requirements",
        "Section 15",
        "IGF Code",
        "IGC Code",
        "methanol",
        "ammonia",
        "hydrogen",
        "LNG",
    ],
}

MUST_COVER_BROAD_MEPC = [
    "MEPC",
    "GHG",
    "emissions",
    "MARPOL",
    "CII",
    "SEEMP",
    "reporting",
    "guidelines",
    "verification",
    "ship operation",
]

CLASS_RULE_SOURCES = frozenset({"DNV", "LR", "KR", "ABS"})
VALID_CATEGORIES = frozenset(
    {"table_qa", "trend_summary", "env_regulation", "autonomous", "rule_lookup"}
)


def effective_question_category(question: str, row: dict) -> str:
    explicit = str(row.get("category") or "").strip()
    if explicit in VALID_CATEGORIES:
        return explicit
    from question_classifier import classify_question_category

    return classify_question_category(question, row)


@dataclass
class RetrievalRunConfig:
    question_mode: str = "broad"
    question_category: str = "trend_summary"
    broad_summary_mode: bool = False
    answer_mode: str = "standard_rag"
    use_gold_filter: bool = False
    eval_constrained_mode: bool = False
    top_k: int = 8
    fetch_k: int = 40
    max_chunks_per_doc: int = 2
    max_chunks_per_page: int = 1
    max_docs: int = 4
    use_diversity_rerank: bool = True
    narrow_doc_id: str | None = None
    supplement_gold_pages: bool = False
    retrieval_profile_id: str = ""
    retrieval_profile_label: str = ""
    retrieval_profile_notes: list[str] = field(default_factory=list)
    # Query-focused clauses recovered by the bounded document/source sparse
    # pass.  These IDs are execution metadata, not gold labels: carrying them
    # into the answer phase lets a small local model see the best direct clause
    # before semantically adjacent sibling provisions.
    priority_local_chunk_ids: list[str] = field(default_factory=list)
    priority_local_scope: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def detect_narrow_doc_id(question: str, row: dict) -> str | None:
    from imo_doc_registry import exact_doc_ids_for_query
    from retrieval_query_analysis import analyze_query

    manifest_ids = exact_doc_ids_for_query(question)
    signals = analyze_query(question)
    excluded = {str(source).upper() for source in signals.excluded_sources}
    constrained = {str(source).upper() for source in signals.constrained_sources}
    disallowed_named_code = any(
        (
            match.group(1).upper() in excluded
            or (
                constrained
                and match.group(1).upper() not in constrained
            )
        )
        for match in re.finditer(
            r"(?<![A-Za-z0-9])(DNV|LR|ABS|KR|MEPC|MSC)"
            r"(?=\s*[-–/]\s*(?:[A-Z]{1,4}\s*[-–/]\s*)?\d)",
            question or "",
            re.I,
        )
    )
    if disallowed_named_code:
        manifest_ids = []
    if len(manifest_ids) == 1:
        return manifest_ids[0]
    q = question
    ql = q.lower()
    has_narrow_signal = any(re.search(p, ql, re.I) for p in NARROW_SIGNAL_PATTERNS)
    has_doc_ref = any(re.search(p, ql, re.I) for p in DOC_REF_PATTERNS)
    if not has_narrow_signal and not has_doc_ref:
        return None
    gold = str(row.get("gold_doc_id") or "")
    if gold:
        file_hint = str(row.get("gold_file_hint") or "")
        hints = [h for h in [file_hint, "84-7-14", "0264", "notice no", "mass code"] if h]
        if has_doc_ref and any(h.lower() in ql for h in hints):
            return gold
        if has_narrow_signal and effective_question_category(question, row) == "rule_lookup":
            return gold
    return None


def classify_question_mode(question: str, row: dict) -> str:
    if detect_narrow_doc_id(question, row):
        return "narrow"
    cat = effective_question_category(question, row)
    if cat == "rule_lookup":
        return "narrow"
    if any(re.search(p, question, re.I) for p in BROAD_PATTERNS):
        return "broad"
    if cat in {"trend_summary", "env_regulation"}:
        return "broad"
    if cat == "autonomous":
        from question_classifier import _msc_broad_signals

        return "broad" if _msc_broad_signals(question) else "narrow"
    return "broad"


def resolve_retrieval_run_config(
    row: dict,
    *,
    eval_constrained_mode: bool = False,
    gold_doc_filter: bool | None = None,
    question_mode: str | None = None,
    top_k: int = 8,
    fetch_k: int = 40,
    max_chunks_per_doc: int | None = None,
    max_chunks_per_page: int = 1,
    max_docs: int = 4,
    use_diversity_rerank: bool = True,
) -> RetrievalRunConfig:
    question = str(row.get("question", ""))
    mode = question_mode or classify_question_mode(question, row)
    narrow_doc = detect_narrow_doc_id(question, row)

    if gold_doc_filter is None:
        use_gold = bool(eval_constrained_mode)
    else:
        use_gold = gold_doc_filter

    if mode == "broad":
        mcpd = 2 if max_chunks_per_doc is None else max_chunks_per_doc
    else:
        mcpd = 5 if max_chunks_per_doc is None else max_chunks_per_doc

    return RetrievalRunConfig(
        question_mode=mode,
        use_gold_filter=use_gold,
        eval_constrained_mode=eval_constrained_mode,
        top_k=top_k,
        fetch_k=fetch_k,
        max_chunks_per_doc=mcpd,
        max_chunks_per_page=max_chunks_per_page,
        max_docs=max_docs,
        use_diversity_rerank=use_diversity_rerank,
        narrow_doc_id=narrow_doc,
        supplement_gold_pages=eval_constrained_mode,
    )


def get_must_cover_items(row: dict) -> list[str]:
    qid = str(row.get("question_id", ""))
    if qid in MUST_COVER_BY_QID:
        return list(MUST_COVER_BY_QID[qid])
    sources = {s.upper() for s in row.get("retrieval_sources") or []}
    if "MEPC" in sources and classify_question_mode(str(row.get("question", "")), row) == "broad":
        return list(MUST_COVER_BROAD_MEPC)
    ref = row.get("must_cover")
    if ref:
        return [str(x) for x in ref]
    return [str(k) for k in (row.get("expected_keywords") or [])]


def parse_citation_ids(text: str) -> set[int]:
    return {int(m.group(1)) for m in CITATION_RE.finditer(text or "")}


def _term_in_text(term: str, text: str) -> bool:
    return term.lower() in (text or "").lower()


def compute_must_cover_coverage(
    must_cover: list[str],
    chunks: list[Any],
    answer: str = "",
) -> list[dict]:
    chunk_text = "\n".join(getattr(c, "text", "") or "" for c in chunks)
    rows = []
    for term in must_cover:
        in_chunks = _term_in_text(term, chunk_text)
        in_answer = _term_in_text(term, answer) if answer else False
        rows.append(
            {
                "must_cover": term,
                "found_in_chunks": "Yes" if in_chunks else "No",
                "included_in_answer": "Yes" if in_answer else ("No" if answer else "—"),
            }
        )
    return rows


def build_evidence_table(
    retrieved: list[Any],
    answer: str = "",
    *,
    score_lookup: dict[str, dict] | None = None,
) -> list[dict]:
    used = parse_citation_ids(answer) if answer else set()
    lookup = score_lookup or {}
    rows = []
    for i, c in enumerate(retrieved, start=1):
        preview = getattr(c, "content_preview", "") or content_preview(getattr(c, "text", ""), 120)
        if answer:
            used_flag = "Yes" if i in used else "No"
        else:
            used_flag = "—"
        cid = str(getattr(c, "chunk_id", "") or "")
        extra = lookup.get(cid, {})
        bm25_score = getattr(c, "bm25_score", None)
        if bm25_score is None:
            bm25_score = extra.get("bm25_score")
        dense_score = getattr(c, "dense_score", None)
        if dense_score is None:
            dense_score = extra.get("dense_score")
        rrf_score = getattr(c, "rrf_score", None)
        if rrf_score is None:
            rrf_score = extra.get("rrf_score")
        rows.append(
            {
                "rank": i,
                "citation_id": f"[{i}]",
                "source": getattr(c, "source", ""),
                "doc_id": getattr(c, "doc_id", ""),
                "file_name": getattr(c, "file_name", "") or getattr(c, "doc_id", ""),
                "page": getattr(c, "page_number", None),
                "chunk_type": getattr(c, "chunk_type", "") or getattr(c, "element_type", ""),
                "table_id": getattr(c, "table_id", ""),
                "caption": getattr(c, "caption", ""),
                "row_index": getattr(c, "row_index", None),
                "matched_columns": ", ".join(getattr(c, "matched_columns", []) or []),
                "score": round(float(getattr(c, "distance", 0.0)), 4),
                "dense_score": dense_score,
                "bm25_score": bm25_score,
                "rrf_score": rrf_score,
                "bm25_rank": extra.get("bm25_rank"),
                "dense_rank": extra.get("dense_rank"),
                "metadata_boost": getattr(c, "metadata_boost", None),
                "source_priority_score": getattr(c, "source_priority_score", None),
                "is_catalog_table": getattr(c, "is_catalog_table", False),
                "used_in_answer": used_flag,
                "chunk_preview": preview,
            }
        )
    return rows


def hybrid_score_lookup(hybrid_log: dict | None) -> dict[str, dict]:
    """Map chunk_id → dense/bm25/rrf scores from hybrid retrieval log."""
    if not hybrid_log:
        return {}
    out: dict[str, dict] = {}
    for key in ("fused_results", "fused_top10"):
        for row in hybrid_log.get(key) or []:
            cid = str(row.get("chunk_id") or "")
            if not cid:
                continue
            out[cid] = {
                "dense_score": row.get("dense_score"),
                "bm25_score": row.get("bm25_score"),
                "rrf_score": row.get("rrf_score"),
                "bm25_rank": row.get("bm25_rank"),
                "dense_rank": row.get("dense_rank"),
            }
    for row in hybrid_log.get("bm25_top20") or hybrid_log.get("bm25_results") or []:
        cid = str(row.get("chunk_id") or "")
        if not cid:
            continue
        slot = out.setdefault(cid, {})
        if slot.get("bm25_score") is None:
            slot["bm25_score"] = row.get("score")
        if slot.get("bm25_rank") is None:
            slot["bm25_rank"] = row.get("rank")
    return out


def build_answer_citation_mapping(answer: str, retrieved: list[Any]) -> list[dict]:
    rows = []
    for line in (answer or "").splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        bullet = line[2:].strip()
        cite_ids = sorted(parse_citation_ids(bullet))
        if not cite_ids:
            rows.append(
                {
                    "answer_point": bullet[:240],
                    "citations": [],
                    "evidence_docs": "",
                    "pages": "",
                }
            )
            continue
        docs: list[str] = []
        pages: list[str] = []
        cites: list[str] = []
        for cid in cite_ids:
            if cid < 1 or cid > len(retrieved):
                continue
            c = retrieved[cid - 1]
            cites.append(f"[{cid}]")
            docs.append(getattr(c, "file_name", "") or getattr(c, "doc_id", ""))
            pn = getattr(c, "page_number", None)
            pages.append(f"p{pn}" if pn is not None else "?")
        rows.append(
            {
                "answer_point": bullet[:240],
                "citations": cites,
                "evidence_docs": ", ".join(dict.fromkeys(docs)),
                "pages": ", ".join(dict.fromkeys(pages)),
            }
        )
    return rows


def unique_doc_count(chunks: list[Any], k: int | None = None) -> int:
    subset = chunks[:k] if k else chunks
    return len({getattr(c, "doc_id", "") for c in subset if getattr(c, "doc_id", "")})


def has_class_rule_chunks(chunks: list[Any]) -> bool:
    return any(getattr(c, "source", "").upper() in CLASS_RULE_SOURCES for c in chunks)


def build_verification_summary(
    *,
    config: RetrievalRunConfig,
    retrieved: list[Any],
    pool: list[Any],
    answer: str = "",
    must_cover_rows: list[dict] | None = None,
    metrics: dict | None = None,
    row: dict | None = None,
) -> dict:
    used = parse_citation_ids(answer) if answer else set()
    unique_docs = unique_doc_count(retrieved)
    single_doc_skew = unique_docs <= 1 and len(retrieved) >= 3
    must_rows = must_cover_rows or []
    covered = sum(1 for r in must_rows if r.get("found_in_chunks") == "Yes")
    answer_covered = sum(1 for r in must_rows if r.get("included_in_answer") == "Yes")
    class_rules = has_class_rule_chunks(retrieved)

    warnings: list[str] = []
    if single_doc_skew and config.question_mode == "broad":
        warnings.append(
            "본 검색 결과는 특정 문서에 편중됨 — broad 질문이므로 추가 MEPC 문서 검색 권장"
        )
    if must_rows and covered < len(must_rows) * 0.5:
        warnings.append(f"must-cover 검색 커버리지 낮음 ({covered}/{len(must_rows)})")
    if answer and used and max(used, default=0) > len(retrieved):
        warnings.append("답변 인용 번호가 검색 청크 범위를 초과함")

    retrieval_mode_label = (
        "Constrained (gold doc)" if config.use_gold_filter else "Open retrieval / Multi-doc"
    )
    answer_mode_label = (
        "Structured Meeting (4섹션, LLM 없음)"
        if getattr(config, "answer_mode", "") == "structured_meeting"
        else (
            "Rule/Guidance Lookup (구조화, LLM 없음)"
            if getattr(config, "answer_mode", "") == "rule_guidance_lookup"
            else (
                "Multi-document Summary"
                if getattr(config, "answer_mode", "") == "multi_doc_summary"
                else "일반 RAG"
            )
        )
    )

    result = {
        "retrieval_mode": retrieval_mode_label,
        "answer_mode": answer_mode_label,
        "question_category": getattr(config, "question_category", ""),
        "question_mode": config.question_mode,
        "broad_summary_mode": getattr(config, "broad_summary_mode", False),
        "retrieval_profile_id": getattr(config, "retrieval_profile_id", ""),
        "retrieval_profile_label": getattr(config, "retrieval_profile_label", ""),
        "retrieval_profile_notes": list(getattr(config, "retrieval_profile_notes", []) or []),
        "use_gold_filter": config.use_gold_filter,
        "eval_constrained_mode": config.eval_constrained_mode,
        "unique_doc_count": unique_docs,
        "pool_unique_doc_count": unique_doc_count(pool),
        "final_chunk_count": len(retrieved),
        "pool_chunk_count": len(pool),
        "citations_used_count": len(used),
        "single_doc_skew": single_doc_skew,
        "must_cover_in_chunks": f"{covered}/{len(must_rows)}" if must_rows else "—",
        "must_cover_in_answer": f"{answer_covered}/{len(must_rows)}" if answer and must_rows else "—",
        "class_rule_in_retrieval": class_rules,
        "class_rule_note": (
            "포함됨" if class_rules else "No — 별도 검색 필요"
        ),
        "warnings": warnings,
        "doc_recall_at_5": metrics.get("doc_recall_at_5") if metrics else None,
        "page_recall_at_5": metrics.get("page_recall_at_5") if metrics else None,
    }
    row = row or {}
    mprof = row.get("_meeting_retrieval_profile") or {}
    if row.get("_top_level_category") or mprof:
        result["top_level_category"] = row.get("_top_level_category") or mprof.get("top_level_category", "")
        result["internal_intent"] = row.get("_internal_intent") or mprof.get("internal_intent", "")
        result["retrieval_profile"] = mprof.get("profile_id") or mprof.get("retrieval_profile", "")
        result["use_dense"] = mprof.get("use_dense")
        result["use_bm25"] = mprof.get("use_bm25")
        result["use_rrf"] = mprof.get("use_rrf")
        result["use_source_tier"] = mprof.get("use_source_tier")
    meta = row.get("_meeting_answer_meta") or {}
    if meta.get("coverage_check"):
        result["coverage_check"] = meta["coverage_check"]
        result["coverage_pass"] = all(meta["coverage_check"].values())
    if "claim_verification_pass" in meta:
        result["claim_verification_pass"] = bool(meta["claim_verification_pass"])
        result["claim_verification"] = list(meta.get("claim_verification") or [])
        result["document_status"] = list(meta.get("document_status") or [])
        result["unsupported_claims_blocked"] = int(meta.get("unsupported_claims_blocked") or 0)
    if row.get("warning_flags"):
        result["warning_flags"] = list(row["warning_flags"])
    return result


def meeting_routing_fields_from_row(row: dict, *, answer: str = "") -> dict:
    """Extract meeting routing panel fields for UI (from row after retrieval/answer)."""
    mprof = row.get("_meeting_retrieval_profile") or {}
    out: dict = {}
    if row.get("_top_level_category") or mprof:
        out["top_level_category"] = row.get("_top_level_category") or mprof.get("top_level_category", "")
        out["internal_intent"] = row.get("_internal_intent") or mprof.get("internal_intent", "")
        out["retrieval_profile"] = mprof.get("profile_id") or mprof.get("retrieval_profile", "")
        out["use_dense"] = mprof.get("use_dense")
        out["use_bm25"] = mprof.get("use_bm25")
        out["use_rrf"] = mprof.get("use_rrf")
        out["use_source_tier"] = mprof.get("use_source_tier")
    meta = row.get("_meeting_answer_meta") or {}
    if meta.get("coverage_check"):
        out["coverage_check"] = meta["coverage_check"]
        out["coverage_pass"] = all(meta["coverage_check"].values())
    if "claim_verification_pass" in meta:
        out["claim_verification_pass"] = bool(meta["claim_verification_pass"])
    if row.get("warning_flags"):
        out["warning_flags"] = list(row["warning_flags"])
    return out


def chunk_to_trace_dict(chunk: Any, rank: int) -> dict:
    return {
        "rank": rank,
        "chunk_id": getattr(chunk, "chunk_id", ""),
        "doc_id": getattr(chunk, "doc_id", ""),
        "source": getattr(chunk, "source", ""),
        "file_name": getattr(chunk, "file_name", ""),
        "page_number": getattr(chunk, "page_number", None),
        "distance": round(float(getattr(chunk, "distance", 0.0)), 4),
        "text_preview": content_preview(getattr(chunk, "text", ""), 300),
    }


def build_retrieval_trace(
    *,
    row: dict,
    config: RetrievalRunConfig,
    pool: list[Any],
    retrieved: list[Any],
    metrics: dict,
    answer: str = "",
    llm_provider: str = "",
    llm_model: str = "",
) -> dict:
    must_cover = get_must_cover_items(row)
    must_rows = compute_must_cover_coverage(must_cover, retrieved, answer)
    evidence = build_evidence_table(
        retrieved,
        answer,
        score_lookup=hybrid_score_lookup(row.get("_hybrid_retrieval_log")),
    )
    mapping = build_answer_citation_mapping(answer, retrieved) if answer else []
    summary = build_verification_summary(
        config=config,
        retrieved=retrieved,
        pool=pool,
        answer=answer,
        must_cover_rows=must_rows,
        metrics=metrics,
    )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question_id": row.get("question_id"),
        "query": row.get("question"),
        "mode": config.question_mode,
        "source": (row.get("retrieval_sources") or [None])[0],
        "use_gold_filter": config.use_gold_filter,
        "eval_constrained_mode": config.eval_constrained_mode,
        "narrow_doc_id": config.narrow_doc_id,
        "top_k": config.top_k,
        "fetch_k": config.fetch_k,
        "rerank": config.use_diversity_rerank,
        "max_chunks_per_doc": config.max_chunks_per_doc,
        "max_chunks_per_page": config.max_chunks_per_page,
        "max_docs": config.max_docs,
        "retrieved_pool": [chunk_to_trace_dict(c, i) for i, c in enumerate(pool, 1)],
        "final_context_chunks": [chunk_to_trace_dict(c, i) for i, c in enumerate(retrieved, 1)],
        "retrieval_metrics": metrics,
        "evidence_table": evidence,
        "answer_citation_mapping": mapping,
        "must_cover_coverage": must_rows,
        "verification_summary": summary,
        "used_citations": sorted(parse_citation_ids(answer)),
        "answer": answer,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "unique_doc_count": summary["unique_doc_count"],
        "warnings": summary["warnings"],
    }


def append_retrieval_trace_log(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def serialize_chunk_list(chunks: list[Any]) -> list[dict]:
    from dataclasses import asdict, is_dataclass

    out = []
    for c in chunks:
        if is_dataclass(c):
            out.append(asdict(c))
        elif isinstance(c, dict):
            out.append(c)
        else:
            out.append(chunk_to_trace_dict(c, 0))
    return out
