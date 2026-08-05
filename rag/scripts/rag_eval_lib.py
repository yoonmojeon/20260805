"""Shared retrieval evaluation helpers for MaritimeRAG."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from embedding_policy import DEFAULT_EMBEDDING_PRESET, embed_texts_local, resolve_embedding_config
from retrieval_search import enrich_query_for_embedding, query_with_hybrid_ranking


def load_manifest(index_doc_dir: Path) -> dict:
    manifest_path = index_doc_dir / "index_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Index manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_questions(path: Path) -> list[dict]:
    questions: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            questions.append(row)
    return questions


def load_chunks(chunks_path: Path) -> list[dict]:
    chunks: list[dict] = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_chunk_text_map(chunks_path: Path) -> dict[str, str]:
    mapping = {c["chunk_id"]: c.get("text") or "" for c in load_chunks(chunks_path)}
    table_path = chunks_path.parent / "table_chunks.jsonl"
    if table_path.exists():
        for c in load_chunks(table_path):
            mapping[c["chunk_id"]] = c.get("text") or ""
    return mapping


def gold_chunk_ids_for_label(chunks: list[dict], doc_id: str, page: int, clause: str) -> list[str]:
    clause_str = str(clause).strip()
    ids: list[str] = []
    for chunk in chunks:
        if str(chunk.get("doc_id", "")) != doc_id:
            continue
        if int(chunk.get("page_number", -1)) != page:
            continue
        chunk_clause = str(chunk.get("clause_number", "")).strip()
        chunk_article = str(chunk.get("article_number", "")).strip()
        if clause_str and chunk_clause != clause_str and chunk_article != clause_str:
            continue
        if not clause_str:
            continue
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if chunk_id:
            ids.append(chunk_id)
    return ids


def resolve_gold_chunk_ids(question_row: dict, chunks: list[dict]) -> list[str]:
    explicit = question_row.get("gold_chunk_ids")
    if explicit:
        return [str(x).strip() for x in explicit if str(x).strip()]
    return gold_chunk_ids_for_label(
        chunks,
        str(question_row["gold_doc_id"]),
        int(question_row["gold_page"]),
        normalize_clause(question_row.get("gold_clause")),
    )


def normalize_clause(value: object) -> str:
    return str(value).strip() if value is not None else ""


def metadata_matches_gold(
    meta: dict,
    gold_page: int,
    gold_clause: str,
    *,
    gold_doc_id: str | None = None,
) -> bool:
    if gold_doc_id and str(meta.get("doc_id", "")) != gold_doc_id:
        return False
    try:
        page = int(meta.get("page_number", -1))
    except (TypeError, ValueError):
        return False
    if page != gold_page:
        return False
    gold = normalize_clause(gold_clause)
    if not gold:
        return True
    if normalize_clause(meta.get("clause_number")) == gold:
        return True
    return normalize_clause(meta.get("article_number")) == gold


def keyword_hits(text: str, keywords: list[str]) -> tuple[int, int]:
    if not keywords:
        return 0, 0
    lower = text.lower()
    hit = sum(1 for kw in keywords if kw.lower() in lower)
    return hit, len(keywords)


def page_band_label(page: int, band_size: int = 50) -> str:
    start = ((page - 1) // band_size) * band_size + 1
    end = start + band_size - 1
    return f"p{start:03d}-p{end:03d}"


def compute_page_coverage(results: list[QuestionResult], band_size: int = 50) -> dict:
    bands: dict[str, dict] = {}
    for r in results:
        label = page_band_label(r.gold_page, band_size)
        entry = bands.setdefault(label, {"questions": 0, "hits": 0, "pages": set()})
        entry["questions"] += 1
        entry["pages"].add(r.gold_page)
        if r.hit_at_k:
            entry["hits"] += 1
    serializable = {
        band: {"questions": v["questions"], "hits": v["hits"], "pages": sorted(v["pages"])}
        for band, v in sorted(bands.items())
    }
    return serializable


@dataclass
class QuestionResult:
    question_id: str
    question: str
    gold_page: int
    gold_clause: str
    page_band: str = ""
    gold_chunk_ids: list[str] = field(default_factory=list)
    gold_indexed: bool = False
    hit_at_k: bool = False
    hit_rank: int | None = None
    hit_chunk_id: str | None = None
    page_hit_at_k: bool = False
    keyword_hits: int = 0
    keyword_total: int = 0
    top_results: list[dict] = field(default_factory=list)
    note: str = ""


def evaluate_question(
    collection,
    question_row: dict,
    query_vector: list[float],
    chunks: list[dict],
    indexed_ids: set[str],
    top_k: int,
    *,
    question_text: str | None = None,
    source: str | None = None,
) -> QuestionResult:
    question_id = str(question_row["question_id"])
    question = question_text if question_text is not None else str(question_row["question"])
    gold_doc_id = str(question_row["gold_doc_id"])
    gold_page = int(question_row["gold_page"])
    gold_clause = normalize_clause(question_row.get("gold_clause"))
    expected_keywords = list(question_row.get("expected_keywords") or [])
    note = str(question_row.get("note", ""))
    page_band = str(question_row.get("page_band") or page_band_label(gold_page))

    gold_ids = resolve_gold_chunk_ids(question_row, chunks)
    gold_indexed = any(cid in indexed_ids for cid in gold_ids) if gold_ids else False

    raw = query_with_hybrid_ranking(
        collection,
        question,
        query_vector,
        top_k=top_k,
        source=source,
    )

    ids = raw["ids"][0]
    distances = raw["distances"][0]
    metadatas = raw["metadatas"][0]
    documents = raw["documents"][0]

    top_results: list[dict] = []
    hit_at_k = False
    hit_rank: int | None = None
    hit_chunk_id: str | None = None
    page_hit_at_k = False
    keyword_hits_best = 0
    keyword_total = len(expected_keywords)

    for rank, (chunk_id, distance, meta, doc) in enumerate(
        zip(ids, distances, metadatas, documents), start=1
    ):
        meta = meta or {}
        doc_text = doc or ""
        clause_match = metadata_matches_gold(
            meta, gold_page, gold_clause, gold_doc_id=gold_doc_id
        )
        chunk_match = chunk_id in gold_ids if gold_ids else False
        gold_match = clause_match or chunk_match
        if (
            not gold_match
            and str(meta.get("doc_id", "")) == gold_doc_id
            and expected_keywords
        ):
            kh, kt = keyword_hits(doc_text, expected_keywords)
            if kt and kh >= max(2, (kt + 1) // 2):
                gold_match = True
        page_match = False
        try:
            page_match = int(meta.get("page_number", -1)) == gold_page
        except (TypeError, ValueError):
            pass

        kh, _ = keyword_hits(doc_text, expected_keywords)
        top_results.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "distance": distance,
                "page_number": meta.get("page_number"),
                "doc_id": meta.get("doc_id"),
                "clause_number": meta.get("clause_number"),
                "element_type": meta.get("element_type"),
                "gold_match": gold_match,
                "clause_match": clause_match,
                "chunk_match": chunk_match,
                "page_match": page_match,
                "keyword_hits": kh,
                "keyword_total": keyword_total,
                "preview": doc_text.replace("\n", " ")[:200],
            }
        )

        if gold_match and not hit_at_k:
            hit_at_k = True
            hit_rank = rank
            hit_chunk_id = chunk_id
            keyword_hits_best = kh

        if page_match and str(meta.get("doc_id", "")) == gold_doc_id:
            page_hit_at_k = True

    return QuestionResult(
        question_id=question_id,
        question=question,
        gold_page=gold_page,
        gold_clause=gold_clause,
        page_band=page_band,
        gold_chunk_ids=gold_ids,
        gold_indexed=gold_indexed,
        hit_at_k=hit_at_k,
        hit_rank=hit_rank,
        hit_chunk_id=hit_chunk_id,
        page_hit_at_k=page_hit_at_k,
        keyword_hits=keyword_hits_best,
        keyword_total=keyword_total,
        top_results=top_results,
        note=note,
    )


def write_text_report(path: Path, summary: dict, results: list[QuestionResult]) -> None:
    lines = [
        f"Retrieval evaluation — {summary['doc_id']}",
        f"Generated: {summary['generated_at']}",
        f"Questions: {summary['num_questions']} | top_k: {summary['top_k']}",
        f"Model: {summary['embedding_model']}",
        f"Questions file: {summary.get('questions_path', '')}",
        "",
        "Metrics:",
        f"  Recall@{summary['top_k']} (gold chunk in top-k): {summary['recall_at_k']:.1%} "
        f"({summary['hits_at_k']}/{summary['num_questions']})",
        f"  Page hit@{summary['top_k']}:          {summary['page_recall_at_k']:.1%}",
        f"  Mean keyword recall (on hit):        {summary['mean_keyword_recall_on_hit']:.1%}",
        f"  Gold labels in index:                {summary['gold_in_index']}/{summary['num_questions']}",
        f"  Gold page range:                     p{summary.get('gold_page_min')} – p{summary.get('gold_page_max')}",
        "",
        "Page band coverage (50-page bands):",
    ]
    for band, stats in (summary.get("page_band_coverage") or {}).items():
        q = stats["questions"]
        h = stats["hits"]
        pages = stats.get("pages") or []
        page_span = f"{min(pages)}-{max(pages)}" if pages else "-"
        lines.append(f"  {band}: {h}/{q} hits | pages {page_span}")
    lines.extend(["", "Per question:"])

    for r in results:
        status = "HIT" if r.hit_at_k else "MISS"
        rank_part = f" rank={r.hit_rank}" if r.hit_rank else ""
        idx_part = " indexed" if r.gold_indexed else " NOT_IN_INDEX"
        lines.append(
            f"  [{status}] {r.question_id} {r.page_band} p{r.gold_page} "
            f"clause={r.gold_clause or '-'}{rank_part}{idx_part}"
        )
        lines.append(f"    Q: {r.question}")
        if r.gold_chunk_ids:
            lines.append(f"    gold_chunks: {', '.join(r.gold_chunk_ids)}")
        if r.hit_chunk_id:
            lines.append(f"    retrieved: {r.hit_chunk_id}")
        if r.keyword_total:
            lines.append(f"    keywords: {r.keyword_hits}/{r.keyword_total}")
        if not r.hit_at_k and r.top_results:
            top = r.top_results[0]
            lines.append(
                f"    top1: {top['chunk_id']} p{top['page_number']} "
                f"clause={top.get('clause_number')}"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_retrieval_eval(
    *,
    doc_id: str,
    questions_path: Path,
    top_k: int = 5,
    index_dir: Path = Path("data/processed/index"),
    chunks_dir: Path = Path("data/processed/chunks"),
    output_dir: Path = Path("data/processed/logs"),
    embedding_preset: str | None = None,
) -> tuple[dict, list[QuestionResult], Path, Path]:
    questions = load_questions(questions_path)
    if not questions:
        raise ValueError(f"No questions in {questions_path}")

    index_doc_dir = index_dir / doc_id
    manifest = load_manifest(index_doc_dir)
    preset = embedding_preset or manifest.get("embedding_preset", DEFAULT_EMBEDDING_PRESET)
    embed_config = resolve_embedding_config(preset, str(manifest.get("embedding_model", "")))
    model_name = str(embed_config["model"])

    chunks_path = chunks_dir / doc_id / "chunks.jsonl"
    chunks = load_chunks(chunks_path)

    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=manifest["chroma_path"],
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(manifest["collection_name"])
    indexed_ids = set(collection.get(include=[])["ids"])

    eval_rows = [row for row in questions if str(row.get("gold_doc_id", doc_id)) == doc_id]
    if not eval_rows:
        raise ValueError(f"No questions for doc_id={doc_id}")

    query_texts = [
        enrich_query_for_embedding(str(row["question"]), model_name) for row in eval_rows
    ]
    query_vectors = embed_texts_local(query_texts, model_name, for_query=True)

    results: list[QuestionResult] = []
    for row, query_vector in zip(eval_rows, query_vectors):
        results.append(
            evaluate_question(
                collection,
                row,
                query_vector,
                chunks,
                indexed_ids,
                top_k,
                question_text=str(row["question"]),
            )
        )

    hits = sum(1 for r in results if r.hit_at_k)
    page_hits = sum(1 for r in results if r.page_hit_at_k)
    gold_in_index = sum(1 for r in results if r.gold_indexed)
    gold_pages = [r.gold_page for r in results]
    keyword_scores = [
        r.keyword_hits / r.keyword_total
        for r in results
        if r.hit_at_k and r.keyword_total > 0
    ]
    mean_kw = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0.0

    summary = {
        "doc_id": doc_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "questions_path": questions_path.resolve().as_posix(),
        "top_k": top_k,
        "embedding_preset": preset,
        "embedding_model": model_name,
        "num_questions": len(results),
        "hits_at_k": hits,
        "recall_at_k": hits / len(results),
        "page_hits_at_k": page_hits,
        "page_recall_at_k": page_hits / len(results),
        "gold_in_index": gold_in_index,
        "mean_keyword_recall_on_hit": mean_kw,
        "gold_page_min": min(gold_pages),
        "gold_page_max": max(gold_pages),
        "page_band_coverage": compute_page_coverage(results),
    }

    report_json = {
        "summary": summary,
        "results": [
            {
                "question_id": r.question_id,
                "question": r.question,
                "gold_page": r.gold_page,
                "gold_clause": r.gold_clause,
                "page_band": r.page_band,
                "gold_chunk_ids": r.gold_chunk_ids,
                "gold_indexed": r.gold_indexed,
                "hit_at_k": r.hit_at_k,
                "hit_rank": r.hit_rank,
                "hit_chunk_id": r.hit_chunk_id,
                "page_hit_at_k": r.page_hit_at_k,
                "keyword_hits": r.keyword_hits,
                "keyword_total": r.keyword_total,
                "note": r.note,
                "top_results": r.top_results,
            }
            for r in results
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{doc_id}_retrieval_eval.json"
    txt_path = output_dir / f"{doc_id}_retrieval_eval.txt"
    json_path.write_text(json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8")
    write_text_report(txt_path, summary, results)
    return summary, results, json_path, txt_path


def run_unified_retrieval_eval(
    *,
    unified_id: str,
    questions_path: Path,
    top_k: int = 5,
    index_dir: Path = Path("data/processed/index"),
    chunks_dir: Path = Path("data/processed/chunks"),
    output_dir: Path = Path("data/processed/logs"),
    embedding_preset: str | None = None,
) -> tuple[dict, list[QuestionResult], Path, Path]:
    questions = load_questions(questions_path)
    if not questions:
        raise ValueError(f"No questions in {questions_path}")

    index_root = index_dir / f"unified_{unified_id}"
    manifest = load_manifest(index_root)
    preset = embedding_preset or manifest.get("embedding_preset", DEFAULT_EMBEDDING_PRESET)
    embed_config = resolve_embedding_config(preset, str(manifest.get("embedding_model", "")))
    model_name = str(embed_config["model"])

    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=manifest["chroma_path"],
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(manifest["collection_name"])
    indexed_ids = set(collection.get(include=[])["ids"])

    chunks_cache: dict[str, list[dict]] = {}

    def chunks_for(doc_id: str) -> list[dict]:
        if doc_id not in chunks_cache:
            chunks_path = chunks_dir / doc_id / "chunks.jsonl"
            chunks_cache[doc_id] = load_chunks(chunks_path) if chunks_path.exists() else []
        return chunks_cache[doc_id]

    query_texts = [
        enrich_query_for_embedding(str(row["question"]), model_name) for row in questions
    ]
    query_vectors = embed_texts_local(query_texts, model_name, for_query=True)

    results: list[QuestionResult] = []
    for row, query_vector in zip(questions, query_vectors):
        gold_doc_id = str(row["gold_doc_id"])
        chunks = chunks_for(gold_doc_id)
        source = row.get("gold_source")
        results.append(
            evaluate_question(
                collection,
                row,
                query_vector,
                chunks,
                indexed_ids,
                top_k,
                question_text=str(row["question"]),
                source=str(source) if source else None,
            )
        )

    hits = sum(1 for r in results if r.hit_at_k)
    page_hits = sum(1 for r in results if r.page_hit_at_k)
    gold_in_index = sum(1 for r in results if r.gold_indexed)
    gold_pages = [r.gold_page for r in results]
    keyword_scores = [
        r.keyword_hits / r.keyword_total
        for r in results
        if r.hit_at_k and r.keyword_total > 0
    ]
    mean_kw = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0.0

    by_category: dict[str, dict] = {}
    for row, res in zip(questions, results):
        cat = str(row.get("category") or "general")
        entry = by_category.setdefault(cat, {"questions": 0, "hits": 0})
        entry["questions"] += 1
        if res.hit_at_k:
            entry["hits"] += 1

    summary = {
        "doc_id": f"unified_{unified_id}",
        "unified_id": unified_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "questions_path": questions_path.resolve().as_posix(),
        "top_k": top_k,
        "embedding_preset": preset,
        "embedding_model": model_name,
        "num_questions": len(results),
        "hits_at_k": hits,
        "recall_at_k": hits / len(results),
        "page_hits_at_k": page_hits,
        "page_recall_at_k": page_hits / len(results),
        "gold_in_index": gold_in_index,
        "mean_keyword_recall_on_hit": mean_kw,
        "gold_page_min": min(gold_pages) if gold_pages else 0,
        "gold_page_max": max(gold_pages) if gold_pages else 0,
        "page_band_coverage": compute_page_coverage(results),
        "by_category": {
            cat: {**stats, "recall_at_k": stats["hits"] / stats["questions"]}
            for cat, stats in sorted(by_category.items())
        },
    }

    report_json = {
        "summary": summary,
        "results": [
            {
                "question_id": r.question_id,
                "question": r.question,
                "gold_page": r.gold_page,
                "gold_clause": r.gold_clause,
                "page_band": r.page_band,
                "gold_chunk_ids": r.gold_chunk_ids,
                "gold_indexed": r.gold_indexed,
                "hit_at_k": r.hit_at_k,
                "hit_rank": r.hit_rank,
                "hit_chunk_id": r.hit_chunk_id,
                "page_hit_at_k": r.page_hit_at_k,
                "keyword_hits": r.keyword_hits,
                "keyword_total": r.keyword_total,
                "note": r.note,
                "top_results": r.top_results,
            }
            for r in results
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"unified_{unified_id}_retrieval_eval.json"
    txt_path = output_dir / f"unified_{unified_id}_retrieval_eval.txt"
    json_path.write_text(json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8")
    write_text_report(txt_path, summary, results)
    return summary, results, json_path, txt_path
