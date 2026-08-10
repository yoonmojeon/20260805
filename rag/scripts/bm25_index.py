"""BM25 sparse index: build, save/load, search with metadata-enriched documents."""
from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None  # type: ignore

DOC_CODE_RE = re.compile(
    r"\b("
    r"(?:DNV|LR|ABS|KR|MSC|MEPC|IGF|MARPOL|MASS)"
    r"(?:[- ]?(?:CG|RP|RU|CP|NV))?[- ]?\d[\w.-]*"
    r"|Notice\s+No\.?\s*\d+"
    r")",
    re.I,
)
WORD_RE = re.compile(r"[\w가-힣]+", re.UNICODE)
MANIFEST_NAME = "bm25_manifest.json"
INDEX_NAME = "bm25_index.pkl"
TABLE_INDEX_DIR_NAME = "table_bm25"
TABLE_TOKENIZER_VERSION = "table-bm25-char3-v1"
# SQLite IN-clause limit (~999); keep batches small for metadatas+documents fetch.
CHROMA_GET_BATCH_SIZE = 250
_BM25_CACHE: dict[str, tuple[str, "BM25Index"]] = {}
_TABLE_BM25_CACHE: dict[str, tuple[str, "BM25Index"]] = {}


def iter_collection_records(
    collection,
    *,
    batch_size: int = CHROMA_GET_BATCH_SIZE,
    include: list[str] | None = None,
):
    """Yield (id, metadata, document) tuples without exceeding SQLite variable limits."""
    include = include or ["metadatas", "documents"]
    offset = 0
    while True:
        raw = collection.get(include=include, limit=batch_size, offset=offset)
        ids = raw.get("ids") or []
        if not ids:
            break
        metas = raw.get("metadatas") or [None] * len(ids)
        docs = raw.get("documents") or [None] * len(ids)
        for cid, meta, doc in zip(ids, metas, docs):
            yield str(cid), meta or {}, doc or ""
        offset += len(ids)
        if len(ids) < batch_size:
            break


def extract_document_codes(text: str) -> list[str]:
    return list(dict.fromkeys(m.group(1).strip() for m in DOC_CODE_RE.finditer(text or "")))


def enrich_query_for_bm25(query: str) -> tuple[str, list[str]]:
    """Expand Korean Rule/Guidance queries with English retrieval terms for BM25."""
    from retrieval_query_analysis import analyze_query

    signals = analyze_query(query)
    parts: list[str] = [query]
    if signals.expanded_terms:
        parts.extend(signals.expanded_terms[:16])
    ql = query.lower()
    if "대체연료" in query or "연료" in query or "fuel" in ql:
        parts.extend(
            [
                "alternative fuel",
                "low-flashpoint fuel",
                "low flashpoint",
                "dual fuel",
                "fuel storage",
                "fuel supply",
                "engine requirements",
                "Section 15",
                "engines supplied",
                "IGF Code",
                "IGC Code",
                "methanol",
                "ammonia",
                "hydrogen",
                "LNG",
                "Notice No.1",
            ]
        )
    if signals.class_society_hint:
        parts.append(signals.class_society_hint)
    if "자율" in query or "autonomous" in ql or "smart vessel" in ql:
        parts.extend(
            ["autonomous", "remotely operated", "Smart Vessel", "DNV-CG-0264", "notation"]
        )
    enriched = " ".join(dict.fromkeys(p.strip() for p in parts if p.strip()))
    tokens = tokenize_for_bm25(enriched)
    return enriched, tokens


def bm25_alt_fuel_terms_present(tokens: list[str]) -> dict[str, bool]:
    blob = " ".join(tokens).lower()
    checks = {
        "alternative fuel": "alternative" in blob and "fuel" in blob,
        "low-flashpoint fuel": "low-flashpoint" in blob or "low" in blob and "flashpoint" in blob,
        "dual fuel": "dual" in blob and "fuel" in blob,
        "IGF Code": "igf" in blob,
        "IGC Code": "igc" in blob,
        "methanol": "methanol" in blob,
        "ammonia": "ammonia" in blob,
        "hydrogen": "hydrogen" in blob,
        "LNG": "lng" in blob,
    }
    return checks


def tokenize_for_bm25(text: str) -> list[str]:
    """Preserve doc codes; split hyphenated codes into extra tokens."""
    raw = text or ""
    extras: list[str] = []
    for code in extract_document_codes(raw):
        extras.append(code.replace(" ", "-"))
        extras.extend(re.split(r"[-_/.\s]+", code))
    normalized = DOC_CODE_RE.sub(lambda m: f" {m.group(1).replace(' ', '-')} ", raw)
    tokens = [t for t in WORD_RE.findall(normalized.lower()) if len(t) > 1]
    for e in extras:
        el = e.lower().strip()
        if len(el) > 1 and el not in tokens:
            tokens.append(el)
    return tokens


def tokenize_for_table_bm25(text: str) -> list[str]:
    """Tokenize table text with word, phrase, and character features.

    Korean particles and OCR spacing often make an otherwise exact table row
    invisible to word-only BM25. Character 2/3-grams recover those variants,
    while prefixed word bigrams retain useful multi-word engineering phrases.
    """
    words = [t.lower() for t in WORD_RE.findall(text or "") if len(t) > 1]
    tokens = list(words)
    for left, right in zip(words, words[1:]):
        if len(left) + len(right) <= 48:
            tokens.append(f"w2:{left}_{right}")
    for word in words:
        compact = re.sub(r"[^0-9a-z가-힣]", "", word)
        if not (2 <= len(compact) <= 64):
            continue
        # Bigrams help short labels/codes; trigrams reduce noise in prose.
        tokens.extend(f"c2:{compact[i:i + 2]}" for i in range(len(compact) - 1))
        if len(compact) >= 3:
            tokens.extend(f"c3:{compact[i:i + 3]}" for i in range(len(compact) - 2))
    return tokens


def build_bm25_document(*, text: str, meta: dict) -> str:
    parts = [
        str(meta.get("source") or ""),
        str(meta.get("file_name") or ""),
        str(meta.get("doc_type") or ""),
        str(meta.get("document_code") or ""),
        str(meta.get("caption") or ""),
        str(meta.get("section_title") or ""),
        str(meta.get("clause_number") or meta.get("article_number") or ""),
        text or "",
    ]
    return " ".join(p for p in parts if p).strip()


@dataclass
class BM25Hit:
    chunk_id: str
    score: float
    rank: int
    meta: dict = field(default_factory=dict)
    document: str = ""


@dataclass
class BM25Index:
    chunk_ids: list[str]
    metas: list[dict]
    documents: list[str]
    corpus_tokens: list[list[str]]
    bm25: Any = None
    unified_id: str = ""
    fingerprint: str = ""
    tokenizer_mode: str = "generic"

    def _matches_society(self, meta: dict, society: str | None) -> bool:
        if not society:
            return True
        s = society.upper()
        src = str(meta.get("source") or "").upper()
        fn = str(meta.get("file_name") or "").lower()
        return src == s or society.lower() in fn

    def search(
        self,
        query: str,
        *,
        top_k: int = 50,
        source: str | None = None,
        file_name_contains: str | None = None,
        doc_id: str | None = None,
        table_ids: set[str] | None = None,
        chunk_types: set[str] | None = None,
    ) -> list[BM25Hit]:
        if not self.bm25 or not self.chunk_ids:
            return []
        q_tokens = (
            tokenize_for_table_bm25(query)
            if self.tokenizer_mode == TABLE_TOKENIZER_VERSION
            else tokenize_for_bm25(query)
        )
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        hits: list[BM25Hit] = []
        for idx in ranked:
            sc = float(scores[idx])
            if sc <= 0:
                continue
            meta = self.metas[idx]
            if source and not self._matches_society(meta, source):
                continue
            if doc_id and str(meta.get("doc_id") or "") != doc_id:
                continue
            if table_ids is not None and str(meta.get("table_id") or "") not in table_ids:
                continue
            if chunk_types is not None and str(meta.get("chunk_type") or "") not in chunk_types:
                continue
            if file_name_contains and file_name_contains.lower() not in str(meta.get("file_name") or "").lower():
                if source and not self._matches_society(meta, source):
                    continue
            hits.append(
                BM25Hit(
                    chunk_id=self.chunk_ids[idx],
                    score=sc,
                    rank=len(hits) + 1,
                    meta=meta,
                    document=self.documents[idx],
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def search_with_fallback(
        self,
        query: str,
        *,
        top_k: int = 50,
        source: str | None = None,
        hard_source_filter: bool = False,
    ) -> tuple[list[BM25Hit], list[str]]:
        warnings: list[str] = []
        hits = self.search(query, top_k=top_k, source=source, file_name_contains=source)
        if source and not hard_source_filter and len(hits) < min(5, top_k // 2):
            warnings.append("source_filter_fallback")
            hits = self.search(query, top_k=top_k)
        elif source and hard_source_filter and len(hits) < min(5, top_k // 2):
            warnings.append("society_filter_strict_no_fallback")
        return hits, warnings

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunk_ids": self.chunk_ids,
            "metas": self.metas,
            "documents": self.documents,
            "corpus_tokens": self.corpus_tokens,
            "unified_id": self.unified_id,
            "fingerprint": self.fingerprint,
            "tokenizer_mode": self.tokenizer_mode,
        }
        with (out_dir / INDEX_NAME).open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        (out_dir / MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "unified_id": self.unified_id,
                    "fingerprint": self.fingerprint,
                    "chunk_count": len(self.chunk_ids),
                    "tokenizer_mode": self.tokenizer_mode,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, out_dir: Path) -> BM25Index | None:
        path = out_dir / INDEX_NAME
        if not path.exists() or BM25Okapi is None:
            return None
        try:
            size = path.stat().st_size
            if size < 64:
                return None
            with path.open("rb") as f:
                payload = pickle.load(f)
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "BM25 pickle load failed (%s): %s", path, exc
            )
            return None
        if not isinstance(payload, dict) or "chunk_ids" not in payload:
            return None
        inst = cls(
            chunk_ids=payload["chunk_ids"],
            metas=payload["metas"],
            documents=payload["documents"],
            corpus_tokens=payload["corpus_tokens"],
            unified_id=payload.get("unified_id", ""),
            fingerprint=payload.get("fingerprint", ""),
            tokenizer_mode=payload.get("tokenizer_mode", "generic"),
        )
        inst.bm25 = BM25Okapi(inst.corpus_tokens)
        return inst


def bm25_index_dir(index_dir: Path, unified_id: str) -> Path:
    return index_dir / f"unified_{unified_id}" / "bm25"


def table_bm25_index_dir(index_dir: Path, unified_id: str) -> Path:
    return index_dir / f"unified_{unified_id}" / TABLE_INDEX_DIR_NAME


def build_bm25_from_collection(
    collection,
    *,
    unified_id: str,
    index_dir: Path,
    fingerprint: str = "",
    batch_size: int = CHROMA_GET_BATCH_SIZE,
    tokenizer_mode: str = "generic",
    out_dir: Path | None = None,
    chunk_types: set[str] | None = None,
) -> BM25Index:
    if BM25Okapi is None:
        raise ImportError("rank_bm25 is required: pip install rank_bm25")

    chunk_ids: list[str] = []
    meta_list: list[dict] = []
    doc_list: list[str] = []
    tokens_list: list[list[str]] = []
    for cid, meta, doc in iter_collection_records(collection, batch_size=batch_size):
        if chunk_types is not None and str((meta or {}).get("chunk_type") or "") not in chunk_types:
            continue
        enriched = build_bm25_document(text=doc, meta=meta)
        toks = (
            tokenize_for_table_bm25(enriched)
            if tokenizer_mode == TABLE_TOKENIZER_VERSION
            else tokenize_for_bm25(enriched)
        )
        if not toks:
            continue
        chunk_ids.append(cid)
        meta_list.append(meta)
        doc_list.append(doc)
        tokens_list.append(toks)
    if not chunk_ids:
        raise ValueError("No tokenizable chunks found in Chroma collection")
    bm25 = BM25Okapi(tokens_list)
    inst = BM25Index(
        chunk_ids=chunk_ids,
        metas=meta_list,
        documents=doc_list,
        corpus_tokens=tokens_list,
        bm25=bm25,
        unified_id=unified_id,
        fingerprint=fingerprint,
        tokenizer_mode=tokenizer_mode,
    )
    inst.save(out_dir or bm25_index_dir(index_dir, unified_id))
    return inst


def load_or_build_bm25(
    collection,
    *,
    unified_id: str,
    index_dir: Path,
    fingerprint: str = "",
    rebuild: bool = False,
) -> BM25Index | None:
    out_dir = bm25_index_dir(index_dir, unified_id)
    cache_key = str(out_dir.resolve())
    cached = _BM25_CACHE.get(cache_key)
    if not rebuild and cached and (not fingerprint or cached[0] == fingerprint):
        return cached[1]
    if not rebuild:
        loaded = BM25Index.load(out_dir)
        if loaded and (not fingerprint or loaded.fingerprint == fingerprint or not loaded.fingerprint):
            _BM25_CACHE[cache_key] = (fingerprint, loaded)
            return loaded
    try:
        built = build_bm25_from_collection(
            collection, unified_id=unified_id, index_dir=index_dir, fingerprint=fingerprint
        )
        _BM25_CACHE[cache_key] = (fingerprint, built)
        return built
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning("BM25 index build failed: %s", exc)
        return None


def clear_bm25_caches() -> None:
    """Drop process-level lexical index caches."""
    _BM25_CACHE.clear()
    _TABLE_BM25_CACHE.clear()


def peek_table_bm25_cache(
    *,
    unified_id: str,
    index_dir: Path,
    fingerprint: str = "",
) -> BM25Index | None:
    """Return an already-warmed table BM25 handle, or None (no disk I/O)."""
    out_dir = table_bm25_index_dir(index_dir, unified_id)
    cache_key = str(out_dir.resolve())
    cached = _TABLE_BM25_CACHE.get(cache_key)
    if cached and (not fingerprint or cached[0] == fingerprint):
        return cached[1]
    return None


def load_or_build_table_bm25(
    collection,
    *,
    unified_id: str,
    index_dir: Path,
    fingerprint: str = "",
    rebuild: bool = False,
    allow_disk_load: bool = True,
) -> BM25Index | None:
    """Load the table-only lexical index.

    Interactive RAG must not rebuild this on failure — the precise table
    corpus is ~190k chunks and a silent rebuild can hang the UI for minutes.
    The on-disk pickle is also ~1.3GB (~25s to load); set allow_disk_load=False
    to use only an already-warm process cache (dense-only otherwise).
    Offline rebuild: ``python scripts/35_build_bm25_index.py --table --rebuild``.
    """
    out_dir = table_bm25_index_dir(index_dir, unified_id)
    cache_key = str(out_dir.resolve())
    cached = _TABLE_BM25_CACHE.get(cache_key)
    if not rebuild and cached and (not fingerprint or cached[0] == fingerprint):
        return cached[1]
    if not rebuild:
        if not allow_disk_load:
            return None
        loaded = BM25Index.load(out_dir)
        if not loaded:
            return None
        # Prefer dense-only over a multi-minute online rebuild when the on-disk
        # tokenizer is incompatible. Fingerprint drift (schema upserts bump the
        # Chroma fp without a BM25 rebuild) is soft-accepted and still cached —
        # otherwise every query reloads ~240MB from disk (~4–5s) with no reuse.
        if loaded.tokenizer_mode != TABLE_TOKENIZER_VERSION:
            return loaded
        _TABLE_BM25_CACHE[cache_key] = (fingerprint or loaded.fingerprint or "", loaded)
        return loaded
    try:
        built = build_bm25_from_collection(
            collection,
            unified_id=unified_id,
            index_dir=index_dir,
            fingerprint=fingerprint,
            tokenizer_mode=TABLE_TOKENIZER_VERSION,
            out_dir=out_dir,
        )
        _TABLE_BM25_CACHE[cache_key] = (fingerprint, built)
        return built
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning("Table BM25 index build failed: %s", exc)
        return None
