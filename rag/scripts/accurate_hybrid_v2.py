"""Feature-flagged Accurate dense + FTS5 sparse retrieval.

The module is intentionally additive.  It does not alter embeddings, Chroma,
Fast mode, table retrieval, or the legacy Accurate path.  When the sidecar
index is missing or stale callers can fall back to legacy retrieval.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from bm25_index import iter_collection_records
from embedding_policy import embed_texts_local, validate_embedding_model
from hybrid_retrieval import FusedHit, is_catalog_table, extract_catalog_candidates
from retrieval_query_analysis import QuerySignals, analyze_query
from retrieval_search import extract_exact_identifiers, query_with_hybrid_ranking


SPARSE_SCHEMA_VERSION = "maritime-fts5-v2"
SPARSE_DB_NAME = "accurate_sparse_fts5_v2.sqlite3"
TRUE_VALUES = {"1", "true", "yes", "on"}
CLASS_SOURCES = {"DNV", "KR", "ABS", "LR"}

DOC_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"(?:DNV|LR|ABS|KR)[-–/ ](?:CG|RP|RU|CP|OS|SI|NV)[-–/ ]?[A-Z0-9.-]+|"
    r"(?:MEPC|MSC)\s*\d{1,3}(?:\s*[-–/.]\s*[A-Z0-9]+)+|"
    r"(?:MEPC|MSC)\.\d+\(\d+\)"
    r")(?![A-Za-z0-9])",
    re.I,
)
CLAUSE_RE = re.compile(r"(?<!\d)\d{1,4}(?:\.\d{1,4}){1,5}(?!\d)")
TOKEN_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9]{1,}(?:[-./][A-Za-z0-9]+)*|"
    r"\d+(?:\.\d+)?|[가-힣]{2,}",
    re.UNICODE,
)
SESSION_RE = re.compile(r"(?<![A-Za-z0-9])(MEPC|MSC)\s*[-_/ ]?\s*(\d{2,3})", re.I)
RANGE_SESSION_RE = re.compile(r"\d{2,3}\s*(?:~|부터|에서)\s*\d{2,3}|회차별|흐름|timeline", re.I)

STOPWORDS = {
    "관련", "대한", "어떤", "무엇", "알려줘", "설명", "정리", "요약",
    "질문", "내용", "경우", "해당", "그리고", "에서", "으로", "the",
    "and", "for", "with", "what", "which", "from", "that", "this",
}
KOREAN_SUFFIXES = (
    "으로부터", "이라고", "이라는", "에서는", "에게서", "으로", "에서",
    "에는", "에게", "까지", "부터", "만으로", "만", "은", "는", "이",
    "가", "을", "를", "의", "와", "과", "에", "도",
)
ANSWER_MARKERS = {
    "exception": re.compile(r"예외|제외|면제|다만|unless|except|shall not apply|not required", re.I),
    "condition": re.compile(r"조건|경우|요건|requirement|provided that|subject to|when|if ", re.I),
    "numeric": re.compile(r"수치|얼마|몇\s|최소|최대|정격|압력|전압|기간|bar|kpa|mpa|kv|mm|ton", re.I),
    "definition": re.compile(r"정의|뜻|의미|무엇인가|means\b|is defined as", re.I),
    "decision": re.compile(r"결정|결론|채택|승인|확정|outcome|adopted|approved|agreed|decided", re.I),
    "requirement": re.compile(r"요구|요건|하여야|해야|shall|must|required", re.I),
}
NOISE_RE = re.compile(
    r"table of contents|contents\s*$|copyright|all rights reserved|"
    r"changes\s*[–—-]\s*current|^\s*introduction\s*$|bibliograph|references?\s*$",
    re.I,
)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES


def _env_int(name: str, default: int, *, low: int, high: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(low, min(high, value))


@dataclass(frozen=True)
class AccurateHybridConfig:
    enabled: bool = False
    reranker_enabled: bool = False
    dense_top_k: int = 60
    sparse_top_k: int = 60
    rrf_k: int = 60
    rrf_top_k: int = 50
    rerank_top_k: int = 14
    protected_dense_k: int = 24
    output_top_k: int = 80
    reranker_model: str = ""

    @classmethod
    def from_env(cls, *, embedding_model: str = "") -> "AccurateHybridConfig":
        return cls(
            enabled=_env_bool("MARITIME_ACCURATE_HYBRID_V2"),
            reranker_enabled=_env_bool("MARITIME_ACCURATE_RERANKER"),
            dense_top_k=_env_int("MARITIME_DENSE_TOP_K", 60, low=10, high=200),
            sparse_top_k=_env_int("MARITIME_SPARSE_TOP_K", 60, low=10, high=200),
            rrf_k=_env_int("MARITIME_RRF_K", 60, low=1, high=500),
            rrf_top_k=_env_int("MARITIME_RRF_TOP_K", 50, low=10, high=150),
            rerank_top_k=_env_int("MARITIME_RERANK_TOP_K", 14, low=6, high=30),
            protected_dense_k=_env_int(
                "MARITIME_PROTECTED_DENSE_K", 24, low=8, high=60
            ),
            output_top_k=_env_int("MARITIME_HYBRID_OUTPUT_K", 80, low=30, high=120),
            reranker_model=os.environ.get("MARITIME_RERANKER_MODEL", "").strip()
            or embedding_model,
        )

    @classmethod
    def advanced(cls, *, embedding_model: str = "") -> "AccurateHybridConfig":
        """Wider on-prem candidate union used only by the Advanced UI mode.

        The old optional semantic reranker is intentionally disabled here.  It
        is a bi-encoder re-score, whereas Advanced applies a real local
        listwise model after the protected Dense+BM25+RRF union.
        """
        return cls(
            enabled=True,
            reranker_enabled=False,
            dense_top_k=80,
            sparse_top_k=80,
            rrf_k=60,
            rrf_top_k=80,
            rerank_top_k=18,
            protected_dense_k=24,
            output_top_k=120,
            reranker_model=embedding_model,
        )


@dataclass
class SparseHit:
    chunk_id: str
    rank: int
    score: float
    raw_bm25: float
    meta: dict[str, Any] = field(default_factory=dict)
    document: str = ""


class SparseIndexUnavailable(RuntimeError):
    pass


def sparse_index_path(index_dir: Path, unified_id: str) -> Path:
    return index_dir / f"unified_{unified_id}" / SPARSE_DB_NAME


def _strip_korean_suffix(token: str) -> str:
    value = str(token or "").strip().lower()
    for suffix in KOREAN_SUFFIXES:
        if value.endswith(suffix) and len(value) - len(suffix) >= 2:
            return value[: -len(suffix)]
    return value


def _code_variants(value: str) -> list[str]:
    raw = re.sub(r"\s+", " ", str(value or "").strip().lower())
    spaced = re.sub(r"[-–/_.]+", " ", raw)
    spaced = re.sub(r"\s+", " ", spaced).strip()
    compact = re.sub(r"[^0-9a-z가-힣]", "", raw)
    return list(dict.fromkeys(item for item in (raw, spaced, compact) if len(item) >= 2))


def normalize_sparse_document(text: str, meta: dict[str, Any]) -> str:
    """Create a stable mixed Korean/English FTS representation."""
    parts = [
        str(meta.get("source") or ""),
        str(meta.get("file_name") or ""),
        str(meta.get("doc_id") or ""),
        str(meta.get("clause_number") or meta.get("article_number") or ""),
        str(meta.get("caption") or ""),
        str(text or ""),
    ]
    blob = " ".join(part for part in parts if part).lower()
    extras: list[str] = []
    for match in DOC_CODE_RE.finditer(blob):
        extras.extend(_code_variants(match.group(0)))
    for match in CLAUSE_RE.finditer(blob):
        extras.extend(_code_variants(match.group(0)))
    for token in re.findall(r"[가-힣]{2,}", blob):
        stem = _strip_korean_suffix(token)
        if stem != token and len(stem) >= 2:
            extras.append(stem)
    return f"{blob} {' '.join(dict.fromkeys(extras))}".strip()


def _publisher(source: str) -> str:
    source = str(source or "").upper()
    if source in {"MEPC", "MSC", "IMO"}:
        return "IMO"
    if source in CLASS_SOURCES:
        return source
    return source or "unknown"


def _source_type(source: str, file_name: str, text: str) -> str:
    source = str(source or "").upper()
    blob = f"{file_name} {text[:800]}".lower()
    if source in CLASS_SOURCES:
        return "class_rule"
    from imo_doc_classify import classify_imo_filename

    role = classify_imo_filename(file_name)
    if role in {"administrative", "resolution", "amendments"}:
        return role
    if re.search(r"\b(?:mepc|msc)\.\d+\(\d+\)|\bresolution\b", blob, re.I):
        return "resolution"
    if source in {"MEPC", "MSC"} and SESSION_RE.search(file_name):
        return "meeting_record"
    if re.search(r"marpol|solas|colreg|stcw|convention", blob, re.I):
        return "imo_convention"
    if re.search(r"guideline|guidance|guide\b", blob, re.I):
        return "guidance"
    return "unknown"


def _sidecar_metadata(meta: dict[str, Any], text: str) -> dict[str, Any]:
    from grounded_answer_policy import classify_document_status

    out = dict(meta or {})
    source = str(out.get("source") or "").upper()
    file_name = str(out.get("file_name") or "")
    session = SESSION_RE.search(file_name)
    status = classify_document_status(
        SimpleNamespace(source=source, file_name=file_name, text=str(text or "")[:1800])
    )
    out.update(
        {
            "publisher": _publisher(source),
            "source_type": _source_type(source, file_name, text),
            "session_org": session.group(1).upper() if session else "",
            "session_number": int(session.group(2)) if session else None,
            "document_status": status.code if status else "unknown",
            "document_status_label": status.label_ko if status else "상태 미확인",
        }
    )
    return out


def build_sparse_index(
    collection,
    *,
    out_path: Path,
    fingerprint: str,
    expected_count: int | None = None,
    progress_every: int = 20000,
) -> dict[str, Any]:
    """Build the FTS sidecar from existing Chroma records, then atomically swap."""
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix(out_path.suffix + ".building")
    if temp_path.exists():
        temp_path.unlink()
    connection = sqlite3.connect(temp_path)
    started = time.perf_counter()
    count = 0
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            PRAGMA cache_size=-200000;
            CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE chunks (
                rowid INTEGER PRIMARY KEY,
                chunk_id TEXT NOT NULL UNIQUE,
                doc_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                source TEXT NOT NULL,
                publisher TEXT NOT NULL,
                page_number INTEGER,
                clause_number TEXT,
                source_type TEXT NOT NULL,
                session_org TEXT,
                session_number INTEGER,
                document_status TEXT NOT NULL,
                document TEXT NOT NULL,
                search_text TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE INDEX chunks_doc_idx ON chunks(doc_id);
            CREATE INDEX chunks_source_idx ON chunks(source);
            CREATE INDEX chunks_session_idx ON chunks(session_org, session_number);
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                search_text,
                content='chunks',
                content_rowid='rowid',
                tokenize='unicode61 remove_diacritics 2'
            );
            """
        )
        batch: list[tuple[Any, ...]] = []
        for chunk_id, raw_meta, document in iter_collection_records(
            collection, batch_size=250
        ):
            meta = _sidecar_metadata(raw_meta, document)
            batch.append(
                (
                    chunk_id,
                    str(meta.get("doc_id") or ""),
                    str(meta.get("file_name") or ""),
                    str(meta.get("source") or "").upper(),
                    str(meta.get("publisher") or "unknown"),
                    meta.get("page_number"),
                    str(meta.get("clause_number") or meta.get("article_number") or ""),
                    str(meta.get("source_type") or "unknown"),
                    str(meta.get("session_org") or ""),
                    meta.get("session_number"),
                    str(meta.get("document_status") or "unknown"),
                    str(document or ""),
                    normalize_sparse_document(document, meta),
                    json.dumps(meta, ensure_ascii=False, default=str),
                )
            )
            if len(batch) >= 1000:
                connection.executemany(
                    """INSERT INTO chunks(
                        chunk_id, doc_id, file_name, source, publisher,
                        page_number, clause_number, source_type, session_org,
                        session_number, document_status, document, search_text,
                        metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    batch,
                )
                count += len(batch)
                batch.clear()
                if progress_every and count % progress_every < 1000:
                    print(f"[sparse-index] {count} chunks", flush=True)
        if batch:
            connection.executemany(
                """INSERT INTO chunks(
                    chunk_id, doc_id, file_name, source, publisher,
                    page_number, clause_number, source_type, session_org,
                    session_number, document_status, document, search_text,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                batch,
            )
            count += len(batch)
        connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        metadata = {
            "schema_version": SPARSE_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "chunk_count": str(count),
            "created_at_unix": str(time.time()),
        }
        connection.executemany(
            "INSERT INTO index_meta(key, value) VALUES (?, ?)", metadata.items()
        )
        connection.commit()
    finally:
        connection.close()
    if expected_count is not None and count != int(expected_count):
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Sparse index count mismatch: built={count}, expected={expected_count}"
        )
    os.replace(temp_path, out_path)
    return {
        "path": str(out_path),
        "chunk_count": count,
        "schema_version": SPARSE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "build_seconds": round(time.perf_counter() - started, 3),
        "size_bytes": out_path.stat().st_size,
    }


def _query_features(query: str, signals: QuerySignals) -> list[str]:
    features: list[str] = []
    for identifier in extract_exact_identifiers(query):
        features.extend(_code_variants(identifier))
    for match in DOC_CODE_RE.finditer(query or ""):
        features.extend(_code_variants(match.group(0)))
    for match in CLAUSE_RE.finditer(query or ""):
        features.extend(_code_variants(match.group(0)))
    for match in re.finditer(
        r"(?<![A-Za-z0-9])(?:[A-Za-z][A-Za-z0-9-]*\s+){1,5}"
        r"[A-Za-z][A-Za-z0-9-]*(?![A-Za-z0-9])",
        query or "",
    ):
        phrase = re.sub(r"\s+", " ", match.group(0)).strip().lower()
        if not DOC_CODE_RE.fullmatch(phrase):
            features.insert(0, phrase)
    for match in re.finditer(
        r"(?<!\d)\d+(?:[./]\d+)*(?:\s*(?:k?V|bar|kPa|MPa|mm|cm|m|kg|t|ton|%|°C))",
        query or "",
        re.I,
    ):
        features[0:0] = _code_variants(match.group(0))
    tokens = TOKEN_RE.findall(query or "")
    for token in tokens:
        low = token.lower().strip()
        if re.fullmatch(r"[가-힣]{2,}", low):
            low = _strip_korean_suffix(low)
        if len(low) < 2 or low in STOPWORDS:
            continue
        features.append(low)
    # Query expansion is useful for cross-language recall, but sparse search
    # should remain literal.  Keep only a few short technical expansions.
    for term in signals.expanded_terms[:8]:
        cleaned = re.sub(r"\s+", " ", str(term or "").strip().lower())
        if 3 <= len(cleaned) <= 64:
            features.append(cleaned)
    unique: list[str] = []
    seen: set[str] = set()
    for feature in features:
        normalized = re.sub(r"\s+", " ", feature).strip()
        key = normalized.lower()
        if len(normalized) < 2 or key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    unique.sort(
        key=lambda value: (
            0 if DOC_CODE_RE.search(value) or CLAUSE_RE.fullmatch(value) else 1,
            -len(value),
        )
    )
    return unique[:18]


def _fts_match_query(features: Iterable[str]) -> str:
    clauses: list[str] = []
    for feature in features:
        # FTS phrases are parameter values, not SQL fragments.  Doubling quote
        # characters is the FTS5 phrase escape.
        escaped = str(feature).replace('"', '""').strip()
        if not escaped:
            continue
        clauses.append(f'"{escaped}"')
    return " OR ".join(clauses)


def _single_session_constraint(query: str, signals: QuerySignals) -> tuple[str, int] | None:
    if RANGE_SESSION_RE.search(query or ""):
        return None
    unique = list(dict.fromkeys(signals.session_codes))
    if len(unique) == 1 and unique[0][0] in {"MEPC", "MSC"}:
        return unique[0][0], int(unique[0][1])
    return None


class SparseFTSIndex:
    def __init__(self, path: Path, *, expected_fingerprint: str = "") -> None:
        self.path = path.resolve()
        if not self.path.exists():
            raise SparseIndexUnavailable(f"Sparse FTS index is missing: {self.path}")
        uri = f"file:{self.path.as_posix()}?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        meta = {
            str(row["key"]): str(row["value"])
            for row in self.connection.execute("SELECT key, value FROM index_meta")
        }
        if meta.get("schema_version") != SPARSE_SCHEMA_VERSION:
            raise SparseIndexUnavailable(
                f"Sparse schema mismatch: {meta.get('schema_version')}"
            )
        if expected_fingerprint and meta.get("fingerprint") != expected_fingerprint:
            raise SparseIndexUnavailable("Sparse index fingerprint is stale")
        self.meta = meta

    def close(self) -> None:
        self.connection.close()

    def search(
        self,
        query: str,
        *,
        top_k: int,
        source: str | None = None,
        doc_id: str | None = None,
        excluded_sources: Iterable[str] | None = None,
        signals: QuerySignals | None = None,
    ) -> tuple[list[SparseHit], dict[str, Any]]:
        signals = signals or analyze_query(query)
        features = _query_features(query, signals)
        if doc_id:
            # The forced document already supplies the identifier constraint.
            # Repeating its filename in every row gives all chunks the same
            # BM25 boost and hides the answer-bearing clause inside that PDF.
            identifier_features: set[str] = set()
            for match in DOC_CODE_RE.finditer(query or ""):
                identifier_features.update(_code_variants(match.group(0)))
            features = [
                feature
                for feature in features
                if feature.lower() not in identifier_features
            ]
        match_query = _fts_match_query(features)
        if not match_query:
            return [], {"features": [], "match_query": "", "elapsed_seconds": 0.0}
        sql = """
            SELECT c.chunk_id, c.document, c.metadata_json,
                   bm25(chunks_fts, 1.0) AS sparse_bm25
            FROM chunks_fts
            JOIN chunks c ON c.rowid = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
        """
        params: list[Any] = [match_query]
        if source:
            sql += " AND c.source = ?"
            params.append(str(source).upper())
        if doc_id:
            sql += " AND c.doc_id = ?"
            params.append(str(doc_id))
        excluded = sorted(
            {str(value).upper() for value in (excluded_sources or []) if value}
        )
        if excluded:
            placeholders = ",".join("?" for _ in excluded)
            sql += f" AND c.source NOT IN ({placeholders})"
            params.extend(excluded)
        session = _single_session_constraint(query, signals)
        if session and not doc_id:
            sql += " AND c.session_org = ? AND c.session_number = ?"
            params.extend([session[0], session[1]])
        sql += " ORDER BY sparse_bm25 ASC LIMIT ?"
        params.append(max(1, int(top_k)))
        started = time.perf_counter()
        rows = list(self.connection.execute(sql, params))
        elapsed = time.perf_counter() - started
        hits: list[SparseHit] = []
        for rank, row in enumerate(rows, 1):
            raw_score = float(row["sparse_bm25"] or 0.0)
            hits.append(
                SparseHit(
                    chunk_id=str(row["chunk_id"]),
                    rank=rank,
                    score=max(0.0, -raw_score),
                    raw_bm25=raw_score,
                    meta=json.loads(str(row["metadata_json"])),
                    document=str(row["document"] or ""),
                )
            )
        return hits, {
            "features": features,
            "match_query": match_query,
            "elapsed_seconds": round(elapsed, 6),
            "session_filter": list(session) if session else None,
            "source_filter": str(source or ""),
            "doc_id_filter": str(doc_id or ""),
            "excluded_sources": excluded,
        }


_SPARSE_CACHE: dict[tuple[str, str], SparseFTSIndex] = {}


def get_sparse_index(
    path: Path, *, expected_fingerprint: str = ""
) -> SparseFTSIndex:
    key = (str(path.resolve()), expected_fingerprint)
    cached = _SPARSE_CACHE.get(key)
    if cached is None:
        cached = SparseFTSIndex(path, expected_fingerprint=expected_fingerprint)
        _SPARSE_CACHE[key] = cached
    return cached


def clear_sparse_index_cache() -> None:
    for index in _SPARSE_CACHE.values():
        index.close()
    _SPARSE_CACHE.clear()


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _answerability_score(question: str, hit: FusedHit) -> float:
    text = str(hit.document or "")
    low = text.lower()
    score = 0.0
    for intent, pattern in ANSWER_MARKERS.items():
        if pattern.search(question or "") and pattern.search(text):
            score += 0.18 if intent in {"exception", "numeric", "decision"} else 0.12
    q_numbers = set(re.findall(r"\d+(?:\.\d+)?", question or ""))
    if q_numbers:
        overlap = q_numbers.intersection(re.findall(r"\d+(?:\.\d+)?", text))
        score += min(0.24, len(overlap) * 0.08)
    identifiers = extract_exact_identifiers(question)
    if any(identifier.lower() in low for identifier in identifiers):
        score += 0.28
    if re.search(r"\b(?:shall|must|required|provided that|unless)\b|하여야|해야|다만", text, re.I):
        score += 0.08
    if NOISE_RE.search(text[:500]) and len(text.strip()) < 1200:
        score -= 0.30
    if len(text.strip()) < 45:
        score -= 0.18
    return score


def semantic_rerank(
    query: str,
    candidates: list[FusedHit],
    *,
    model_name: str,
    top_k: int,
) -> tuple[list[FusedHit], dict[str, Any]]:
    if not candidates:
        return [], {"elapsed_seconds": 0.0, "model": model_name, "results": []}
    validate_embedding_model(model_name)
    started = time.perf_counter()
    query_vector = embed_texts_local([query], model_name, for_query=True)[0]
    passages = [str(hit.document or "")[:2200] for hit in candidates]
    passage_vectors = embed_texts_local(passages, model_name, for_query=False)
    semantic_scores = [
        float(sum(q * p for q, p in zip(query_vector, vector)))
        for vector in passage_vectors
    ]
    rrf_scores = _minmax([float(hit.rrf_score or 0.0) for hit in candidates])
    heuristic_scores = [_answerability_score(query, hit) for hit in candidates]
    heuristic_norm = _minmax(heuristic_scores)
    scored: list[tuple[float, int, FusedHit]] = []
    result_rows: list[dict[str, Any]] = []
    for index, (hit, semantic, rrf_norm, heuristic, heuristic_scaled) in enumerate(
        zip(candidates, semantic_scores, rrf_scores, heuristic_scores, heuristic_norm)
    ):
        final = semantic * 0.68 + rrf_norm * 0.20 + heuristic_scaled * 0.12
        scored.append((final, index, hit))
        result_rows.append(
            {
                "chunk_id": hit.chunk_id,
                "pre_rank": index + 1,
                "semantic_score": round(semantic, 6),
                "answerability_score": round(heuristic, 6),
                "reranker_score": round(final, 6),
            }
        )
    scored.sort(key=lambda item: (-item[0], item[1]))
    reranked: list[FusedHit] = []
    score_by_id = {row["chunk_id"]: row for row in result_rows}
    for rank, (score, _, hit) in enumerate(scored[:top_k], 1):
        hit.final_score = float(score)
        hit.distance = 1.0 - float(score)
        hit.meta = dict(hit.meta or {})
        hit.meta["reranker_score"] = round(float(score), 6)
        hit.meta["reranker_rank"] = rank
        score_by_id[hit.chunk_id]["post_rank"] = rank
        reranked.append(hit)
    return reranked, {
        "model": model_name,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "input_count": len(candidates),
        "output_count": len(reranked),
        "results": result_rows,
    }


def _dense_rows(raw: dict[str, Any]) -> tuple[list[str], dict[str, tuple[float, dict, str]]]:
    ids = list((raw.get("ids") or [[]])[0])
    distances = list((raw.get("distances") or [[]])[0])
    metadatas = list((raw.get("metadatas") or [[]])[0])
    documents = list((raw.get("documents") or [[]])[0])
    return ids, {
        str(cid): (float(distance), dict(meta or {}), str(document or ""))
        for cid, distance, meta, document in zip(ids, distances, metadatas, documents)
    }


def fuse_dense_sparse(
    dense_raw: dict[str, Any],
    sparse_hits: list[SparseHit],
    *,
    rrf_k: int,
    top_k: int,
) -> list[FusedHit]:
    dense_ids, dense_by_id = _dense_rows(dense_raw)
    dense_rank = {chunk_id: rank for rank, chunk_id in enumerate(dense_ids, 1)}
    sparse_by_id = {hit.chunk_id: hit for hit in sparse_hits}
    all_ids = list(dict.fromkeys([*dense_ids, *[hit.chunk_id for hit in sparse_hits]]))
    fused: list[FusedHit] = []
    from imo_doc_classify import classify_imo_filename

    for chunk_id in all_ids:
        dense = dense_by_id.get(chunk_id)
        sparse = sparse_by_id.get(chunk_id)
        d_rank = dense_rank.get(chunk_id)
        s_rank = sparse.rank if sparse else None
        rrf = (1.0 / (rrf_k + d_rank) if d_rank else 0.0) + (
            1.0 / (rrf_k + s_rank) if s_rank else 0.0
        )
        if dense:
            distance, meta, document = dense
        else:
            distance, meta, document = 1.0, dict(sparse.meta), sparse.document
        if classify_imo_filename(str((meta or {}).get("file_name") or "")) == "administrative":
            continue
        meta = dict(meta or {})
        if sparse:
            # Sidecar fields enrich Chroma metadata without mutating Chroma.
            for key in (
                "publisher", "source_type", "session_org", "session_number",
                "document_status", "document_status_label",
            ):
                if sparse.meta.get(key) not in (None, ""):
                    meta[key] = sparse.meta[key]
        catalog = is_catalog_table(meta, document, str(meta.get("caption") or ""))
        fused.append(
            FusedHit(
                chunk_id=chunk_id,
                dense_score=round(1.0 - distance, 6) if dense else None,
                bm25_score=round(sparse.score, 6) if sparse else None,
                dense_rank=d_rank,
                bm25_rank=s_rank,
                rrf_score=round(rrf, 8),
                final_score=rrf,
                distance=-rrf,
                meta=meta,
                document=document,
                is_catalog_table=catalog,
                catalog_candidates=extract_catalog_candidates(document) if catalog else [],
            )
        )
    fused.sort(
        key=lambda hit: (
            -float(hit.rrf_score),
            hit.dense_rank or 10**9,
            hit.bm25_rank or 10**9,
        )
    )
    return fused[:top_k]


def protect_dense_candidates(
    dense_raw: dict[str, Any],
    ranked: list[FusedHit],
    *,
    protected_k: int,
    top_k: int,
) -> list[FusedHit]:
    """Union hybrid candidates while preserving the proven Dense prefix.

    Pure RRF can move a lexically repetitive but wrong PDF ahead of a correct
    cross-language Dense hit.  Accurate therefore protects the first Dense
    candidates and uses BM25/RRF/reranking to add evidence, not replace it.
    """
    dense_ids, dense_by_id = _dense_rows(dense_raw)
    ranked_by_id = {hit.chunk_id: hit for hit in ranked}
    out: list[FusedHit] = []
    seen: set[str] = set()

    def append(hit: FusedHit) -> None:
        if hit.chunk_id not in seen and len(out) < top_k:
            seen.add(hit.chunk_id)
            out.append(hit)

    for chunk_id in dense_ids[:protected_k]:
        hit = ranked_by_id.get(chunk_id)
        if hit is None:
            distance, meta, document = dense_by_id[chunk_id]
            hit = FusedHit(
                chunk_id=chunk_id,
                dense_score=round(1.0 - distance, 6),
                dense_rank=dense_ids.index(chunk_id) + 1,
                bm25_score=None,
                bm25_rank=None,
                rrf_score=0.0,
                final_score=1.0 - distance,
                distance=distance,
                meta=dict(meta or {}),
                document=document,
            )
        append(hit)
    for hit in ranked:
        append(hit)
    for chunk_id in dense_ids[protected_k:]:
        if len(out) >= top_k:
            break
        hit = ranked_by_id.get(chunk_id)
        if hit is not None:
            append(hit)
            continue
        distance, meta, document = dense_by_id[chunk_id]
        append(
            FusedHit(
                chunk_id=chunk_id,
                dense_score=round(1.0 - distance, 6),
                dense_rank=dense_ids.index(chunk_id) + 1,
                bm25_score=None,
                bm25_rank=None,
                rrf_score=0.0,
                final_score=1.0 - distance,
                distance=distance,
                meta=dict(meta or {}),
                document=document,
            )
        )
    return out


def _raw_from_fused(fused: list[FusedHit]) -> dict[str, Any]:
    return {
        "ids": [[hit.chunk_id for hit in fused]],
        "distances": [[float(hit.distance) for hit in fused]],
        "metadatas": [[dict(hit.meta or {}) for hit in fused]],
        "documents": [[str(hit.document or "") for hit in fused]],
    }


def accurate_hybrid_search(
    collection,
    query: str,
    query_vector: list[float],
    *,
    index_dir: Path,
    unified_id: str,
    fingerprint: str,
    embedding_model: str,
    source: str | None = None,
    doc_id: str | None = None,
    excluded_sources: Iterable[str] | None = None,
    alternate_query_vectors: list[list[float]] | None = None,
    advanced: bool = False,
    timing=None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run Accurate B/C retrieval and return Chroma-shaped output plus trace."""
    config = (
        AccurateHybridConfig.advanced(embedding_model=embedding_model)
        if advanced
        else AccurateHybridConfig.from_env(embedding_model=embedding_model)
    )
    if not config.enabled:
        raise SparseIndexUnavailable("Accurate Hybrid V2 feature flag is disabled")
    signals = analyze_query(query)
    dense_started = time.perf_counter()
    dense_raw = query_with_hybrid_ranking(
        collection,
        query,
        query_vector,
        alternate_query_vectors=alternate_query_vectors,
        top_k=config.dense_top_k,
        fetch_k=max(config.dense_top_k * 3, 120),
        source=source,
        doc_id=doc_id,
        timing=timing,
    )
    dense_elapsed = time.perf_counter() - dense_started
    index = get_sparse_index(
        sparse_index_path(index_dir, unified_id),
        expected_fingerprint=fingerprint,
    )
    sparse_hits, sparse_log = index.search(
        query,
        top_k=config.sparse_top_k,
        source=source,
        doc_id=doc_id,
        excluded_sources=excluded_sources,
        signals=signals,
    )
    fused = fuse_dense_sparse(
        dense_raw,
        sparse_hits,
        rrf_k=config.rrf_k,
        top_k=max(config.rrf_top_k, config.output_top_k),
    )
    reranker_log: dict[str, Any] | None = None
    if config.reranker_enabled:
        reranked, reranker_log = semantic_rerank(
            query,
            fused[: config.rrf_top_k],
            model_name=config.reranker_model,
            top_k=config.rerank_top_k,
        )
        reranked_ids = {hit.chunk_id for hit in reranked}
        fused = [*reranked, *(hit for hit in fused if hit.chunk_id not in reranked_ids)]
    fused = protect_dense_candidates(
        dense_raw,
        fused,
        protected_k=config.protected_dense_k,
        top_k=config.output_top_k,
    )
    log = {
        "mode": (
            "accurate_dense_sparse_rrf_reranker"
            if config.reranker_enabled
            else "accurate_dense_sparse_rrf"
        ),
        "config": {
            "dense_top_k": config.dense_top_k,
            "sparse_top_k": config.sparse_top_k,
            "rrf_k": config.rrf_k,
            "rrf_top_k": config.rrf_top_k,
            "rerank_top_k": config.rerank_top_k,
            "protected_dense_k": config.protected_dense_k,
            "output_top_k": config.output_top_k,
            "reranker_model": config.reranker_model if config.reranker_enabled else "",
        },
        "dense_seconds": round(dense_elapsed, 6),
        "sparse_seconds": sparse_log.get("elapsed_seconds"),
        "reranker_seconds": (reranker_log or {}).get("elapsed_seconds", 0.0),
        "sparse_query": sparse_log,
        "dense_results": [
            {
                "chunk_id": chunk_id,
                "rank": rank,
                "distance": distance,
                "file_name": (meta or {}).get("file_name"),
                "page": (meta or {}).get("page_number"),
            }
            for rank, (chunk_id, distance, meta) in enumerate(
                zip(
                    (dense_raw.get("ids") or [[]])[0],
                    (dense_raw.get("distances") or [[]])[0],
                    (dense_raw.get("metadatas") or [[]])[0],
                ),
                1,
            )
        ],
        "sparse_results": [
            {
                "chunk_id": hit.chunk_id,
                "rank": hit.rank,
                "score": round(hit.score, 6),
                "raw_bm25": round(hit.raw_bm25, 6),
                "file_name": hit.meta.get("file_name"),
                "page": hit.meta.get("page_number"),
            }
            for hit in sparse_hits
        ],
        "fused_results": [
            {
                "chunk_id": hit.chunk_id,
                "document_id": hit.meta.get("doc_id"),
                "page": hit.meta.get("page_number"),
                "dense_rank": hit.dense_rank,
                "dense_score": hit.dense_score,
                "sparse_rank": hit.bm25_rank,
                "sparse_score": hit.bm25_score,
                "bm25_rank": hit.bm25_rank,
                "bm25_score": hit.bm25_score,
                "rrf_score": hit.rrf_score,
                "reranker_score": hit.meta.get("reranker_score"),
                "document_status": hit.meta.get("document_status"),
                "file_name": hit.meta.get("file_name"),
            }
            for hit in fused
        ],
        "reranker": reranker_log,
        "warning_flags": [],
    }
    if timing is not None and hasattr(timing, "meta"):
        timing.meta["accurate_hybrid_v2"] = True
        timing.meta["accurate_sparse_time"] = log["sparse_seconds"]
        timing.meta["accurate_rrf_candidates"] = len(log["fused_results"])
        timing.meta["accurate_reranker_time"] = log["reranker_seconds"]
    raw = _raw_from_fused(fused)
    if dense_raw.get("document_route"):
        raw["document_route"] = dense_raw["document_route"]
    return raw, log
