"""Table question detection and table-aware retrieval (additive to baseline search)."""
from __future__ import annotations

import re
from typing import Any

from embedding_policy import embed_texts_local
from retrieval_search import _merge_where, enrich_query_for_embedding, safe_chroma_query

TABLE_QUESTION_KEYWORDS = [
    "몇 년",
    "선령",
    "percentage",
    "최소 두께",
    "판두께",
    "선박 길이",
    "부식추가",
    "tcorr",
    "화학성분",
    "재료기호",
    "용접강",
    "항복",
    "인장",
    "화물창",
    "화물탱크",
]

TABLE_CHUNK_PRIORITY_DEFAULT = ("table_summary", "table_row", "table_markdown")
TABLE_AWARE_CHUNK_TYPES = frozenset({"table_summary", "table_markdown", "table_row", "table_schema"})
TABLE_CHUNK_BOOST_DEFAULT = {
    "table_summary": 0.12,
    "table_row": 0.10,
    "table_markdown": 0.06,
}

PAGE_HINT_RE = re.compile(r"(\d+)\s*페이지")
AGE_COLUMN_PATTERNS = (
    (re.compile(r"15\s*년\s*(을\s*)?(초과|넘)"), "15년< 선령"),
    (re.compile(r"10\s*년\s*초과.*15\s*년|10\s*[~\-–]\s*15\s*년|10\s*년.*15\s*년\s*이하"), "10년< 선령≤15년"),
    (re.compile(r"5\s*년\s*초과.*10\s*년|5\s*[~\-–]\s*10\s*년|5\s*년.*10\s*년\s*이하"), "5년< 선령≤10년"),
    (re.compile(r"5\s*년\s*미만"), "5년< 선령≤10년"),
)
ROW_ENTITY_TERMS = (
    "평형수탱크",
    "화물창",
    "화물탱크",
    "연료유탱크",
    "빌지저장탱크",
    "이중저탱크",
    "기관실",
    "주기관",
    "보조기관",
    "개방검사",
)
INSPECTION_COLUMN_TERMS = (
    "제1차 정기검사",
    "제2차 정기검사",
    "제3차 정기검사",
    "제4차 및 이후 정기검사",
)


def is_table_question(question: str) -> bool:
    q = question.lower()
    # One-syllable substring checks (``표``, ``행``, ``열``, ``값``) produced
    # broad false positives such as 발표, 통행, 실행 and 영향.  Likewise,
    # generic English words such as requirement/regulation/reporting occur in
    # ordinary rule prose.  Require an explicit tabular expression, while
    # retaining the domain-specific fields that are predominantly stored in
    # rule tables.
    if re.search(r"\btable\b", q, re.IGNORECASE):
        return True
    if re.search(
        r"(?<![가-힣])표(?:\s|$|\d|에서|의|를|가|는|로|에|상|내)|"
        r"(?<![가-힣])(?:행|열)(?:\s|$|\d|에서|의|을|이|별)|"
        r"(?:몇|제?\d+)\s*(?:행|열)\b|셀(?:\s|$|에서|의|값)",
        q,
    ):
        return True
    for kw in TABLE_QUESTION_KEYWORDS:
        if kw.lower() in q:
            return True
    return False


def extract_page_hints(question: str) -> list[int]:
    return list(dict.fromkeys(int(m) for m in PAGE_HINT_RE.findall(question)))


def infer_age_column_hints(question: str) -> list[str]:
    hints: list[str] = []
    for pattern, col in AGE_COLUMN_PATTERNS:
        if pattern.search(question):
            hints.append(col)
    return hints


def enrich_table_query_for_embedding(question: str) -> str:
    """Add age-band and entity terms so embedding hits the right table without page numbers."""
    parts: list[str] = []
    for hint in infer_age_column_hints(question):
        parts.append(hint)
    if infer_age_column_hints(question):
        parts.append("선령 구간")
    for term in ROW_ENTITY_TERMS:
        if term in question:
            parts.append(term)
    if "reporting" in question.lower() or "정기검사" in question:
        parts.append("정기검사 reporting")
    if "개방검사" in question or "주기관" in question:
        parts.append("주기관/보조기관수 개방검사 시기")
    if not parts:
        return question
    return f"{' '.join(dict.fromkeys(parts))} {question}".strip()


def classify_table_query_mode(question: str) -> str:
    q = question
    if any(k in q for k in ("요약", "주요 열", "열과", "검사 주기")):
        return "table_summary"
    if any(k in q for k in ("비교", "compare")):
        return "column_comparison"
    if any(k in q for k in ("행", "항목은", "요건")) and not any(
        k in q for k in ("셀", "값", "표시", "몇")
    ):
        return "row_lookup"
    return "cell_lookup"


def table_chunk_priority_for_question(question: str) -> tuple[str, ...]:
    mode = classify_table_query_mode(question)
    if mode == "table_summary":
        return ("table_summary", "table_markdown", "table_row")
    return ("table_row", "table_markdown", "table_summary")


def table_chunk_boosts_for_question(question: str) -> dict[str, float]:
    mode = classify_table_query_mode(question)
    if mode == "table_summary":
        return {"table_summary": 0.20, "table_markdown": 0.12, "table_row": 0.08}
    if mode in ("cell_lookup", "row_lookup", "column_comparison"):
        return {"table_row": 0.26, "table_markdown": 0.14, "table_summary": 0.06}
    return dict(TABLE_CHUNK_BOOST_DEFAULT)


def merge_limits_for_question(question: str) -> dict[str, int]:
    mode = classify_table_query_mode(question)
    if mode == "table_summary":
        return {"table_summary": 6, "table_row": 2, "table_markdown": 3}
    return {"table_summary": 2, "table_row": 8, "table_markdown": 3}


def _matched_columns(question: str, column_names: str) -> list[str]:
    if not column_names:
        return []
    cols = [c.strip() for c in column_names.split(",") if c.strip()]
    q_lower = question.lower()
    hits = []
    for col in cols:
        if col.lower() in q_lower or any(tok in q_lower for tok in col.lower().split() if len(tok) > 2):
            hits.append(col)
    for hint in infer_age_column_hints(question):
        if hint in cols and hint not in hits:
            hits.append(hint)
    for col in INSPECTION_COLUMN_TERMS:
        if col in question and col in cols and col not in hits:
            hits.append(col)
    return hits


def table_relevance_boost(
    question: str,
    meta: dict,
    document: str,
    chunk_type: str,
) -> float:
    """Extra score reduction (distance subtract) for table-aware merge."""
    boost = 0.0
    page_hints = extract_page_hints(question)
    page = meta.get("page_number")
    try:
        page_i = int(page) if page is not None else None
    except (TypeError, ValueError):
        page_i = None
    if page_hints and page_i in page_hints:
        boost += 0.24

    blob = f"{document} {meta.get('caption', '')} {meta.get('section_title', '')} {meta.get('column_names', '')}"
    col_names = str(meta.get("column_names") or "")
    for col in _matched_columns(question, col_names):
        if col in blob:
            boost += 0.14

    for term in ROW_ENTITY_TERMS:
        if term in question and term in blob:
            boost += 0.12

    if "정기검사" in question and "정기검사" in col_names:
        boost += 0.10
    if "선령" in question and "선령" in col_names:
        boost += 0.10

    age_hints = infer_age_column_hints(question)
    if age_hints:
        if any(h in col_names for h in age_hints):
            boost += 0.22
        elif any(h in blob for h in age_hints):
            boost += 0.16
        # Penalize inspection-cycle-only tables when question is age-band specific.
        if "선령" not in col_names and any(t in col_names for t in INSPECTION_COLUMN_TERMS):
            boost -= 0.18

    mode = classify_table_query_mode(question)
    if mode == "table_summary" and chunk_type == "table_summary":
        boost += 0.08
    elif mode in ("cell_lookup", "row_lookup", "column_comparison") and chunk_type == "table_row":
        boost += 0.10

    return boost


def query_table_chunks(
    collection,
    question: str,
    model_name: str,
    *,
    top_k: int = 8,
    doc_id: str | None = None,
    source: str | None = None,
    timing=None,
) -> dict[str, list[tuple[str, float, dict, str]]]:
    """Fetch table chunks by type. Returns {chunk_type: [(id, dist, meta, doc), ...]}."""
    embed_query = enrich_table_query_for_embedding(enrich_query_for_embedding(question, model_name))
    vector = embed_texts_local([embed_query], model_name, for_query=True, timing=timing)[0]
    base_where = _merge_where(
        {"source": source.upper()} if source else None,
        {"doc_id": doc_id} if doc_id else None,
    )

    priority = table_chunk_priority_for_question(question)
    by_type: dict[str, list[tuple[str, float, dict, str]]] = {t: [] for t in priority}
    for chunk_type in priority:
        try:
            where = _merge_where(base_where, {"chunk_type": chunk_type})
            raw = safe_chroma_query(
                collection,
                query_embeddings=[vector],
                n_results=min(top_k * 4, 48),
                where=where,
            )
        except Exception:
            continue
        if not raw.get("ids") or not raw["ids"][0]:
            continue
        hits = list(
            zip(
                raw["ids"][0],
                raw["distances"][0],
                raw["metadatas"][0],
                raw["documents"][0],
            )
        )
        boosts = table_chunk_boosts_for_question(question)
        rescored: list[tuple[str, float, dict, str]] = []
        for cid, dist, meta, doc in hits:
            meta = meta or {}
            type_boost = boosts.get(chunk_type, 0.0)
            rel_boost = table_relevance_boost(question, meta, doc or "", chunk_type)
            adj = float(dist) - type_boost - rel_boost
            rescored.append((cid, adj, meta, doc or ""))
        rescored.sort(key=lambda x: x[1])
        by_type[chunk_type] = rescored[:top_k]
    return by_type


def merge_table_aware_into_raw(
    baseline_raw: dict,
    table_by_type: dict[str, list[tuple[str, float, dict, str]]],
    *,
    summary_k: int | None = None,
    row_k: int | None = None,
    markdown_k: int | None = None,
    top_k: int | None = None,
    question: str = "",
) -> dict:
    """Merge table chunk hits with baseline via score boost (does not evict strong text hits)."""
    limits = merge_limits_for_question(question) if question else {
        "table_summary": summary_k or 3,
        "table_row": row_k or 5,
        "table_markdown": markdown_k or 2,
    }
    if summary_k is not None:
        limits["table_summary"] = summary_k
    if row_k is not None:
        limits["table_row"] = row_k
    if markdown_k is not None:
        limits["table_markdown"] = markdown_k

    priority = table_chunk_priority_for_question(question) if question else TABLE_CHUNK_PRIORITY_DEFAULT
    boosts = table_chunk_boosts_for_question(question) if question else TABLE_CHUNK_BOOST_DEFAULT

    pool: dict[str, tuple[float, dict, str]] = {}
    for cid, dist, meta, doc in zip(
        baseline_raw.get("ids", [[]])[0],
        baseline_raw.get("distances", [[]])[0],
        baseline_raw.get("metadatas", [[]])[0],
        baseline_raw.get("documents", [[]])[0],
    ):
        pool[cid] = (float(dist), meta or {}, doc or "")

    for chunk_type in priority:
        for cid, dist, meta, doc in table_by_type.get(chunk_type, [])[: limits[chunk_type]]:
            type_boost = boosts.get(chunk_type, 0.0)
            rel_boost = table_relevance_boost(question, meta or {}, doc or "", chunk_type) if question else 0.0
            adj = float(dist) - type_boost - rel_boost
            prev = pool.get(cid)
            if prev is None or adj < prev[0]:
                pool[cid] = (adj, meta or {}, doc or "")

    ranked = sorted(pool.items(), key=lambda x: x[1][0])
    if top_k is not None:
        ranked = ranked[:top_k]

    return {
        "ids": [[cid for cid, _ in ranked]],
        "distances": [[score for _, (score, _, _) in ranked]],
        "metadatas": [[meta for _, (_, meta, _) in ranked]],
        "documents": [[doc for _, (_, _, doc) in ranked]],
        "table_aware": True,
    }


def annotate_matched_columns(question: str, meta: dict) -> list[str]:
    return _matched_columns(question, str(meta.get("column_names") or ""))


_TOKEN_RE = re.compile(r"[\w가-힣]+", re.UNICODE)


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _contains_text(hay: str, needle: str) -> bool:
    if not needle:
        return True
    return _norm_text(needle) in _norm_text(hay)


def evaluate_table_qa_retrieval(retrieved: list, row: dict, k: int = 10) -> dict[str, bool]:
    """Retrieval metrics aligned with scripts/26_table_qa_benchmark.py."""
    gold_table = str(row.get("gold_table_id") or "")
    gold_page = int(row.get("gold_page") or -1)
    gold_doc = str(row.get("gold_doc_id") or "")
    row_key = str(row.get("gold_row_key") or "")
    col = str(row.get("gold_column") or "")
    answer = str(row.get("gold_answer") or "")

    raw_gold_cells = row.get("gold_cells") or []
    gold_cells = [cell for cell in raw_gold_cells if isinstance(cell, dict)]
    if not gold_cells and (row_key or col or answer):
        gold_cells = [
            {
                "table_id": gold_table,
                "doc_id": gold_doc,
                "page": gold_page,
                "row_key": row_key,
                "column": col,
                "answer": answer,
            }
        ]

    table_recall = False
    row_recall = False
    cell_match = False
    citation_match = False
    matched_rows = [False] * len(gold_cells)
    matched_cells = [False] * len(gold_cells)
    matched_citations = [False] * len(gold_cells)

    for c in retrieved[:k]:
        chunk_type = str(getattr(c, "chunk_type", "") or "")
        table_id = str(getattr(c, "table_id", "") or "")
        page = getattr(c, "page_number", None)
        doc_id = str(getattr(c, "doc_id", "") or "")
        text = str(getattr(c, "text", "") or "")

        if not table_recall:
            if gold_table and table_id == gold_table:
                table_recall = True
            elif chunk_type.startswith("table") and doc_id == gold_doc and page == gold_page:
                if not gold_table or not table_id or table_id == gold_table:
                    table_recall = True

        if not row_recall and row_key and chunk_type == "table_row":
            if gold_table and table_id and table_id != gold_table:
                continue
            if _contains_text(text, row_key):
                row_recall = True

        if not cell_match:
            if row_key and not _contains_text(text, row_key):
                continue
            if col and not _contains_text(text, col):
                continue
            if _contains_text(text, answer):
                cell_match = True

        if not citation_match and doc_id == gold_doc:
            if page == gold_page:
                citation_match = True
            elif table_id and table_id == gold_table:
                citation_match = True

        for index, cell in enumerate(gold_cells):
            cell_table = str(cell.get("table_id") or gold_table)
            cell_doc = str(cell.get("doc_id") or gold_doc)
            try:
                cell_page = int(cell.get("page") or gold_page)
            except (TypeError, ValueError):
                cell_page = gold_page
            cell_row = str(cell.get("row_key") or "")
            cell_column = str(cell.get("column") or "")
            cell_answer = str(cell.get("answer") or "")
            same_table = not cell_table or not table_id or table_id == cell_table
            if same_table and chunk_type == "table_row" and _contains_text(text, cell_row):
                matched_rows[index] = True
            if (
                same_table
                and (not cell_row or _contains_text(text, cell_row))
                and (not cell_column or _contains_text(text, cell_column))
                and (not cell_answer or _contains_text(text, cell_answer))
            ):
                matched_cells[index] = True
            if doc_id == cell_doc and (page == cell_page or (cell_table and table_id == cell_table)):
                matched_citations[index] = True

    if gold_cells:
        row_recall = all(matched_rows) if any(str(cell.get("row_key") or "") for cell in gold_cells) else table_recall
        cell_match = all(matched_cells)
        citation_match = all(matched_citations)

    if not gold_cells and not row_key:
        row_recall = table_recall
    if not gold_cells and not answer:
        cell_match = table_recall

    return {
        "table_recall@k": table_recall,
        "row_recall@k": row_recall,
        "cell_exact_match": cell_match,
        "citation_match": citation_match,
        "gold_cell_count": len(gold_cells),
        "matched_cell_count": sum(matched_cells),
    }
