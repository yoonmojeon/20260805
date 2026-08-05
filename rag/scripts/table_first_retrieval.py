"""Two-stage table-first retrieval: route tables, then fetch rows/markdown."""
from __future__ import annotations

from typing import Any

from embedding_policy import embed_texts_local
from retrieval_search import _merge_where, enrich_query_for_embedding, safe_chroma_query
from table_query_slots import TableQuerySlots, build_table_first_embed_query, extract_table_query_slots


ROUTE_TABLE_K = 5
SUMMARY_FETCH = 48
MARKDOWN_PER_TABLE = 1
ROW_PER_TABLE = 8


def _blob(meta: dict, document: str) -> str:
    meta = meta or {}
    return " ".join(
        [
            document or "",
            str(meta.get("caption") or ""),
            str(meta.get("section_title") or ""),
            str(meta.get("file_name") or ""),
            str(meta.get("column_names") or ""),
        ]
    )


def _grade_in_text(grade: str, text: str) -> bool:
    compact = text.replace(" ", "")
    g_compact = grade.replace(" ", "")
    return grade in text or g_compact in compact


def _scan_tables_for_grades(
    collection,
    grades: list[str],
    doc_ids: list[str],
    *,
    slots: TableQuerySlots | None = None,
) -> list[str]:
    """Literal grade scan inside routed doc(s) — needed when row embeddings are too sparse."""
    if not grades or not doc_ids:
        return []

    scored: dict[str, float] = {}
    chem_table: dict[str, bool] = {}
    for did in dict.fromkeys(doc_ids):
        try:
            raw = collection.get(
                where={"$and": [{"doc_id": did}, {"chunk_type": "table_row"}]},
                include=["metadatas", "documents"],
            )
        except Exception:
            continue
        for meta, doc in zip(raw["metadatas"], raw["documents"]):
            meta = meta or {}
            text = doc or ""
            if not any(_grade_in_text(g, text) for g in grades):
                continue
            tid = str(meta.get("table_id") or "")
            if not tid:
                continue
            rank = scored.get(tid, 1.0)
            if any(t in text for t in ("탈산", "0.035", "화학성분", "화학 성분")):
                rank -= 0.6
                chem_table[tid] = True
            section = str(meta.get("section_title") or "")
            if slots and slots.intent == "chemistry" and (
                "용접" in section or "용접" in text or "인장강도" in text
            ):
                rank += 0.7
            if "고장력" in text:
                rank -= 0.15
            scored[tid] = min(rank, scored.get(tid, rank))

    def _sort_key(tid: str) -> tuple:
        chem = chem_table.get(tid, False)
        return (0 if chem else 1, scored.get(tid, 1.0))

    return [tid for tid, _ in sorted(scored.items(), key=lambda x: _sort_key(x[0]))]


def route_score_adjust(slots: TableQuerySlots, meta: dict, document: str, distance: float) -> float:
    """Lower adjusted distance = better hit."""
    blob = _blob(meta, document)
    blob_lower = blob.lower()
    boost = 0.0

    for term in slots.boost_terms:
        if term.lower() in blob_lower or term in blob:
            boost += 0.14

    for elem in slots.chemical_elements:
        if elem in blob or any(ko for ko, sym in [("황", "S"), ("인", "P"), ("탄소", "C")] if sym == elem and ko in blob):
            boost += 0.10

    if slots.intent == "chemistry":
        if any(t in blob for t in ("화학성분", "화학 성분", "탈산")):
            boost += 0.28
        if "선체 구조용" in blob or "301." in blob or "301 " in blob:
            boost += 0.12
        if "고장력강" in blob or "고장력" in blob:
            boost += 0.18
        if any(t in blob for t in ("제1차 정기검사", "제2차 정기검사", "reporting", "정기검사 구역", "정기검사")):
            boost -= 0.55
        section = str(meta.get("section_title") or "")
        if "용접" in section or "용접용재료" in blob:
            boost -= 0.40
        if "탈산" in blob and "재료" in blob and "기호" in blob:
            boost += 0.25
        if "기계적 성질" in blob and "화학" not in blob:
            boost -= 0.10
        for grade in slots.material_grades:
            g_compact = grade.replace(" ", "")
            if grade in blob or g_compact in blob.replace(" ", ""):
                boost += 0.35
        for elem in slots.chemical_elements:
            if elem in blob:
                boost += 0.12
        fn = str(meta.get("file_name") or "")
        if slots.material_grades and "2편" in fn:
            boost += 0.10
        if slots.material_grades and any(x in fn for x in ("저인화점", "7편", "빙해")):
            boost -= 0.25
    elif slots.intent == "mechanical":
        if any(t in blob for t in ("항복", "인장", "연신", "충격")):
            boost += 0.22
        if "화학성분" in blob and not slots.chemical_elements:
            boost -= 0.06
    elif slots.intent == "inspection":
        if "정기검사" in blob or "선령" in blob:
            boost += 0.22

    for pen in slots.penalize_terms:
        if pen.lower() in blob_lower or pen in blob:
            boost -= 0.18

    # Prefer table summaries with real captions
    if str(meta.get("chunk_type") or "") == "table_summary" and str(meta.get("caption") or "").strip():
        boost += 0.06

    return float(distance) - boost


def _query_typed(
    collection,
    vector: list[float],
    *,
    chunk_type: str,
    n_results: int,
    doc_id: str | None = None,
    source: str | None = None,
    table_ids: list[str] | None = None,
) -> list[tuple[str, float, dict, str]]:
    base_where = _merge_where(
        {"chunk_type": chunk_type},
        {"source": source.upper()} if source else None,
        {"doc_id": doc_id} if doc_id else None,
        {"table_id": {"$in": table_ids}} if table_ids else None,
    )
    raw = safe_chroma_query(
        collection,
        query_embeddings=[vector],
        n_results=n_results,
        where=base_where,
    )
    if not raw.get("ids") or not raw["ids"][0]:
        return []
    return list(
        zip(
            raw["ids"][0],
            raw["distances"][0],
            raw["metadatas"][0],
            raw["documents"][0],
        )
    )


def route_tables(
    collection,
    question: str,
    model_name: str,
    slots: TableQuerySlots,
    *,
    doc_id: str | None = None,
    source: str | None = None,
    timing=None,
) -> list[tuple[str, float, dict, str]]:
    """Return ranked (chunk_id, adj_dist, meta, doc) for table_summary hits (one per table_id)."""
    embed_q = build_table_first_embed_query(
        question, slots
    )
    embed_q = enrich_query_for_embedding(embed_q, model_name)
    vector = embed_texts_local([embed_q], model_name, for_query=True, timing=timing)[0]

    hits = _query_typed(
        collection,
        vector,
        chunk_type="table_summary",
        n_results=SUMMARY_FETCH,
        doc_id=doc_id,
        source=source,
    )

    by_table: dict[str, tuple[str, float, dict, str]] = {}
    for cid, dist, meta, doc in hits:
        meta = meta or {}
        table_id = str(meta.get("table_id") or "")
        if not table_id:
            continue
        adj = route_score_adjust(slots, meta, doc or "", float(dist))
        prev = by_table.get(table_id)
        if prev is None or adj < prev[1]:
            by_table[table_id] = (cid, adj, meta, doc or "")

    ranked = sorted(by_table.values(), key=lambda x: x[1])
    return ranked[:ROUTE_TABLE_K]


def _discover_tables_via_markdown(
    collection,
    vector: list[float],
    slots: TableQuerySlots,
    *,
    doc_id: str | None = None,
    source: str | None = None,
    limit: int = 8,
) -> list[str]:
    hits = _query_typed(
        collection,
        vector,
        chunk_type="table_markdown",
        n_results=40,
        doc_id=doc_id,
        source=source,
    )
    by_table: dict[str, float] = {}
    for _cid, dist, meta, doc in hits:
        meta = meta or {}
        table_id = str(meta.get("table_id") or "")
        if not table_id:
            continue
        adj = route_score_adjust(slots, meta, doc or "", float(dist))
        prev = by_table.get(table_id)
        if prev is None or adj < prev:
            by_table[table_id] = adj
    return [tid for tid, _ in sorted(by_table.items(), key=lambda x: x[1])[:limit]]


def _discover_tables_via_rows(
    collection,
    vector: list[float],
    slots: TableQuerySlots,
    *,
    doc_id: str | None = None,
    source: str | None = None,
    limit: int = 8,
) -> list[str]:
    """Find tables by material-grade / chemistry signals in row chunks (caption-less tables)."""
    if not slots.material_grades and slots.intent != "chemistry":
        return []

    hits = _query_typed(
        collection,
        vector,
        chunk_type="table_row",
        n_results=60,
        doc_id=doc_id,
        source=source,
    )
    by_table: dict[str, float] = {}
    for _cid, dist, meta, doc in hits:
        meta = meta or {}
        table_id = str(meta.get("table_id") or "")
        if not table_id:
            continue
        adj = route_score_adjust(slots, meta, doc or "", float(dist))
        blob = doc or ""
        for grade in slots.material_grades:
            g_compact = grade.replace(" ", "")
            if grade in blob or g_compact in blob.replace(" ", ""):
                adj -= 0.45
        if slots.intent == "chemistry" and any(t in blob for t in ("탈산", "0.035", "이하")):
            adj -= 0.12
        prev = by_table.get(table_id)
        if prev is None or adj < prev:
            by_table[table_id] = adj
    return [tid for tid, _ in sorted(by_table.items(), key=lambda x: x[1])[:limit]]


def fetch_within_tables(
    collection,
    question: str,
    model_name: str,
    slots: TableQuerySlots,
    table_ids: list[str],
    *,
    doc_id: str | None = None,
    source: str | None = None,
    timing=None,
) -> list[tuple[str, float, dict, str, str]]:
    """Fetch markdown + row chunks scoped to routed table_ids."""
    if not table_ids:
        return []

    embed_q = enrich_query_for_embedding(build_table_first_embed_query(question, slots), model_name)
    vector = embed_texts_local([embed_q], model_name, for_query=True, timing=timing)[0]

    out: list[tuple[str, float, dict, str, str]] = []
    per_table_row: dict[str, int] = {}

    for chunk_type, per_table_limit in (("table_markdown", MARKDOWN_PER_TABLE), ("table_row", ROW_PER_TABLE)):
        hits = _query_typed(
            collection,
            vector,
            chunk_type=chunk_type,
            n_results=min(len(table_ids) * per_table_limit * 4, 80),
            doc_id=doc_id,
            source=source,
            table_ids=table_ids,
        )
        rescored: list[tuple[str, float, dict, str]] = []
        for cid, dist, meta, doc in hits:
            adj = route_score_adjust(slots, meta or {}, doc or "", float(dist))
            if chunk_type == "table_row":
                adj -= 0.06
                blob = doc or ""
                for grade in slots.material_grades:
                    g_compact = grade.replace(" ", "")
                    if grade in blob or g_compact in blob.replace(" ", ""):
                        adj -= 0.35
            rescored.append((cid, adj, meta or {}, doc or ""))
        rescored.sort(key=lambda x: x[1])

        for cid, adj, meta, doc in rescored:
            tid = str(meta.get("table_id") or "")
            if tid not in table_ids:
                continue
            if chunk_type == "table_row":
                n = per_table_row.get(tid, 0)
                if n >= ROW_PER_TABLE:
                    continue
                per_table_row[tid] = n + 1
            out.append((cid, adj, meta, doc, chunk_type))

    return out


def build_table_first_raw(
    collection,
    question: str,
    model_name: str,
    *,
    top_k: int,
    doc_id: str | None = None,
    source: str | None = None,
    timing=None,
) -> dict[str, Any]:
    """Build Chroma-style raw dict for retrieve_for_question."""
    slots = extract_table_query_slots(question)
    embed_q = enrich_query_for_embedding(build_table_first_embed_query(question, slots), model_name)
    vector = embed_texts_local([embed_q], model_name, for_query=True, timing=timing)[0]

    routed = route_tables(
        collection,
        question,
        model_name,
        slots,
        doc_id=doc_id,
        source=source,
        timing=timing,
    )
    table_ids: list[str] = []
    routed_doc_ids: list[str] = []
    for _cid, _adj, meta, _doc in routed:
        tid = str(meta.get("table_id") or "")
        if tid and tid not in table_ids:
            table_ids.append(tid)
        did = str(meta.get("doc_id") or "")
        if did and did not in routed_doc_ids:
            routed_doc_ids.append(did)

    # Markdown/row fallback discovers tables whose summaries lack captions (e.g. 2편 p.17).
    if slots.intent in ("chemistry", "mechanical", "lot_treatment", "general_table"):
        for discover in (_discover_tables_via_rows, _discover_tables_via_markdown):
            for tid in discover(collection, vector, slots, doc_id=doc_id, source=source):
                if tid not in table_ids:
                    table_ids.append(tid)
        if slots.material_grades:
            scan_docs = [doc_id] if doc_id else routed_doc_ids
            for tid in _scan_tables_for_grades(
                collection, slots.material_grades, scan_docs, slots=slots
            ):
                if tid not in table_ids:
                    table_ids.insert(0, tid)
        table_ids = table_ids[: max(ROUTE_TABLE_K, 12)]

    pool: dict[str, tuple[float, dict, str]] = {}

    # Stage 1: routed table summaries (strong priority)
    for cid, adj, meta, doc in routed:
        pool[cid] = (adj - 0.15, meta, doc)

    # Stage 2: markdown + rows within routed tables
    for cid, adj, meta, doc, _ctype in fetch_within_tables(
        collection,
        question,
        model_name,
        slots,
        table_ids,
        doc_id=doc_id,
        source=source,
        timing=timing,
    ):
        prev = pool.get(cid)
        if prev is None or adj < prev[0]:
            pool[cid] = (adj, meta, doc)

    ranked = sorted(pool.items(), key=lambda x: x[1][0])[:top_k]
    return {
        "ids": [[cid for cid, _ in ranked]],
        "distances": [[score for _, (score, _, _) in ranked]],
        "metadatas": [[meta for _, (_, meta, _) in ranked]],
        "documents": [[doc for _, (_, _, doc) in ranked]],
        "table_first": True,
        "routed_table_ids": table_ids,
        "table_query_slots": {
            "intent": slots.intent,
            "material_grades": slots.material_grades,
            "chemical_elements": slots.chemical_elements,
        },
    }
