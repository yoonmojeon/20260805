"""Shared chunk filtering and embedding text preparation for single/multi-doc indexes."""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

from clause_parse import article_number_from_text, is_article_clause_number
from imo_doc_classify import classify_imo_filename

REFERENCE_CLAUSE_RE = re.compile(r"(\d{3,})\.\s*의\s*규정")
PICTURE_PLACEHOLDER_MARKERS = ("[picture element", "refer to crop image")

DEFAULT_INDEX_TYPES = frozenset({"text", "table", "picture"})
TABLE_CHUNK_TYPES = frozenset(
    {"table_summary", "table_markdown", "table_row", "table_row_aux", "table_schema"}
)
MIN_INDEX_TEXT_CHARS = 10
# Body text (excluding the 2-line table header) must clear this for row/summary chunks.
MIN_TABLE_BODY_CHARS = 40


@lru_cache(maxsize=4)
def load_quarantine_table_ids(path: str | None = None) -> frozenset[str]:
    """Optional JSON side-file: {"table_ids": [...]} excluded from the table index."""
    candidates = []
    if path:
        candidates.append(Path(path))
    env = os.environ.get("TABLE_QUARANTINE_IDS")
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).resolve().parents[1] / "data/processed/logs/precise_table_quarantine_ids.json")
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        ids = payload.get("table_ids") if isinstance(payload, dict) else payload
        return frozenset(str(x) for x in (ids or []))
    return frozenset()


def _table_body_text(text: str) -> str:
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    if len(lines) > 2:
        return "\n".join(lines[2:])
    return "\n".join(lines)


def _is_tocish_table_text(text: str) -> bool:
    body = _table_body_text(text)
    if "Surveys..." in body or body.count("....") >= 2:
        return True
    if body.count("|") <= 1 and ("...." in body or "Surveys" in body):
        return True
    return False


def load_chunks(chunks_path: Path) -> list[dict]:
    chunks: list[dict] = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_chunk_ids_from_suspicious_csv(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    import csv

    ids: set[str] = set()
    with csv_path.open(encoding="utf-8", newline="") as csv_f:
        reader = csv.DictReader(csv_f)
        for row in reader:
            chunk_id = row.get("chunk_id", "").strip()
            if chunk_id:
                ids.add(chunk_id)
    return ids


def is_placeholder_picture(text: str) -> bool:
    lower = text.lower()
    return any(m in lower for m in PICTURE_PLACEHOLDER_MARKERS)


def text_for_embedding_text_chunk(chunk: dict) -> str:
    """Enrich text chunks: article headers, cross-refs."""
    text = str(chunk.get("text", "")).strip()
    if not text:
        return text

    article = str(chunk.get("article_number") or chunk.get("clause_number") or "")
    if not article or not is_article_clause_number(article):
        inferred = article_number_from_text(text)
        if inferred:
            article = inferred

    parts: list[str] = []
    if article and is_article_clause_number(article):
        parts.append(f"조문 {article}절 {article}.")

    if len(text) < 200 and article and is_article_clause_number(article):
        first_line = text.split("\n", 1)[0].strip()
        if not first_line.startswith(f"{article}."):
            text = f"{article}. {text}"

    ref = REFERENCE_CLAUSE_RE.search(text)
    if ref:
        ref_no = ref.group(1)
        if ref_no != article:
            parts.append(f"참조 {ref_no}절 {ref_no}.")

    if parts:
        prefix = " ".join(parts)
        if not text.startswith(prefix):
            return f"{prefix} {text}"
    return text


def embedding_text_for_table_chunk(chunk: dict, *, source: str = "", file_name: str = "") -> tuple[str, str]:
    """Embedding text for structured table chunks (summary / markdown / row)."""
    chunk_type = str(chunk.get("chunk_type", "")).lower()
    caption = str(chunk.get("caption") or "")[:160]
    section = str(chunk.get("section_title") or "")[:100]
    src_part = f"source={source}" if source else "source=unknown"
    header = f"[{chunk_type}] {src_part}"
    if file_name:
        header += f" file={file_name}"
    if caption:
        header += f"\ncaption: {caption}"
    if section:
        header += f"\nsection: {section}"
    column_names = chunk.get("column_names") or []
    if column_names:
        cols = ", ".join(str(c)[:60] for c in column_names[:10])[:280]
        header += f"\ncolumns: {cols}"
    body = str(chunk.get("text", "")).strip()
    mode = f"table_{chunk_type.removeprefix('table_')}" if chunk_type else "table_structured"
    return f"{header}\n{body}", mode


def embedding_text_and_mode(
    chunk: dict,
    *,
    source: str = "",
    file_name: str = "",
    folder: str = "",
) -> tuple[str, str]:
    """Type-specific text for E5 + embedding_mode label."""
    chunk_type = str(chunk.get("chunk_type", "")).lower()
    if chunk_type in TABLE_CHUNK_TYPES:
        return embedding_text_for_table_chunk(chunk, source=source, file_name=file_name)

    element_type = str(chunk.get("element_type", "text")).lower()
    doc_id = str(chunk.get("doc_id", ""))
    page = chunk.get("page_number", 0)
    src_part = f"source={source}" if source else "source=unknown"

    if element_type == "table":
        body = str(chunk.get("text", "")).strip()
        first_line = body.split("\n", 1)[0].strip()[:160]
        header = f"[table] {src_part}"
        if file_name:
            header += f" file={file_name}"
        if first_line:
            header += f"\ntable_title: {first_line}"
        return f"{header}\n{body}", "table_linearized"

    if element_type in ("picture", "figure"):
        body = str(chunk.get("text", "")).strip()
        header = f"[figure] {src_part}"
        if file_name:
            header += f" file={file_name}"
        mode = "figure_caption" if body and not is_placeholder_picture(body) else "figure_placeholder"
        if chunk.get("linked_caption_id"):
            header += " linked_caption=1"
        return f"{header}\n{body}", mode

    body = text_for_embedding_text_chunk(chunk)
    header = _passage_header(source=source, file_name=file_name, folder=folder)
    section = str(chunk.get("section_title") or "").strip()
    clause = str(chunk.get("article_number") or chunk.get("clause_number") or "").strip()
    if section:
        header = f"{header}\nsection: {section}" if header else f"section: {section}"
    if clause:
        header = f"{header}\nclause: {clause}" if header else f"clause: {clause}"
    if header:
        return f"{header}\n{body}", "text_native"
    return body, "text_native"


def _passage_header(*, source: str, file_name: str, folder: str) -> str:
    """Prepend source/file/doc_type so vector search can distinguish IMO documents."""
    src = (source or "").upper()
    parts: list[str] = []
    if src:
        parts.append(f"[{src.lower()}]")
    if file_name:
        parts.append(f"file={file_name}")
    if folder:
        parts.append(f"folder={folder}")
    if src in ("MSC", "MEPC"):
        doc_type = classify_imo_filename(file_name)
        if doc_type != "unknown":
            parts.append(f"doc_type={doc_type}")
    elif src in ("DNV", "LR", "KR", "ABS"):
        parts.append("doc_type=rule")
    if not parts:
        return ""
    return " ".join(parts)


def should_index_table_chunk(chunk: dict, min_chars: int = 8) -> bool:
    chunk_type = str(chunk.get("chunk_type", "")).lower()
    if chunk_type not in TABLE_CHUNK_TYPES:
        return False
    table_id = str(chunk.get("table_id") or "")
    if table_id and table_id in load_quarantine_table_ids():
        return False
    text = str(chunk.get("text", "")).strip()
    if len(text) < min_chars:
        return False
    if _is_tocish_table_text(text):
        return False
    body = _table_body_text(text)
    # Keep short schema/header-only noise out of the vector index.
    if chunk_type in {"table_row", "table_row_aux", "table_summary"} and len(body) < MIN_TABLE_BODY_CHARS:
        # Allow rows that still carry structured cell separators with some content.
        if body.count("|") < 1 or len(body.replace("(빈 셀)", "").strip(" |")) < 12:
            return False
    return True


def should_index_chunk(
    chunk: dict,
    include_types: frozenset[str],
    skip_ids: set[str],
    min_chars: int,
) -> bool:
    chunk_type = str(chunk.get("chunk_type", "")).lower()
    if chunk_type in TABLE_CHUNK_TYPES:
        return should_index_table_chunk(chunk, min_chars=min(8, min_chars))

    chunk_id = str(chunk.get("chunk_id", ""))
    if chunk_id in skip_ids:
        text = str(chunk.get("text", "")).strip()
        article = str(chunk.get("article_number") or chunk.get("clause_number") or "")
        if not (is_article_clause_number(article) and len(text) >= 80):
            return False

    element_type = str(chunk.get("element_type", "")).lower()
    if element_type not in include_types:
        return False

    text = str(chunk.get("text", "")).strip()
    text_len = len(text)

    if element_type == "picture":
        if chunk.get("linked_caption_id") and text_len >= min_chars:
            return True
        if not is_placeholder_picture(text) and text_len >= min_chars:
            return True
        return False

    return text_len >= min_chars


def filter_chunks_for_index(
    chunks: list[dict],
    include_types: frozenset[str],
    skip_ids: set[str],
    min_chars: int,
    structured_table_mode: str = "include",
) -> list[dict]:
    if structured_table_mode not in {"include", "exclude", "only"}:
        raise ValueError(f"Unknown structured_table_mode: {structured_table_mode}")
    selected: list[dict] = []
    for chunk in chunks:
        is_structured_table = str(chunk.get("chunk_type", "")).lower() in TABLE_CHUNK_TYPES
        if structured_table_mode == "exclude" and is_structured_table:
            continue
        if structured_table_mode == "only" and not is_structured_table:
            continue
        if should_index_chunk(chunk, include_types, skip_ids, min_chars):
            selected.append(chunk)
    return selected


def folder_from_path(file_path: str) -> str:
    p = Path(file_path.replace("\\", "/"))
    parts = p.parts
    try:
        idx = parts.index("raw_pdfs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    except ValueError:
        pass
    return ""


def chunk_metadata(
    chunk: dict,
    *,
    source: str,
    file_name: str,
    folder: str,
    embedding_mode: str,
) -> dict:
    doc_type = classify_imo_filename(file_name) if str(source or "").upper() in ("MSC", "MEPC") else ""
    chunk_type = str(chunk.get("chunk_type", "") or "")
    row_index = chunk.get("row_index")
    column_names = chunk.get("column_names") or []
    meta = {
        "doc_id": str(chunk.get("doc_id", "")),
        "source": str(source or "UNKNOWN"),
        "folder": str(folder or ""),
        "file_name": str(file_name or ""),
        "doc_type": doc_type,
        "page_number": int(chunk.get("page_number", chunk.get("page", 0))),
        "element_type": str(chunk.get("element_type", "")),
        "element_id": str(chunk.get("element_id", "")),
        "clause_number": str(chunk.get("clause_number", "")),
        "article_number": str(chunk.get("article_number", "")),
        "embedding_mode": embedding_mode,
        "crop_path": str(chunk.get("crop_path", "")),
        "source_page_image": str(chunk.get("source_page_image", "")),
    }
    if chunk_type:
        meta["chunk_type"] = chunk_type
        meta["table_id"] = str(chunk.get("table_id", ""))
        meta["caption"] = str(chunk.get("caption", ""))
        meta["section_title"] = str(chunk.get("section_title", ""))
        if row_index is not None:
            meta["row_index"] = int(row_index)
        if column_names:
            meta["column_names"] = ",".join(str(c)[:120] for c in column_names[:12])[:1200]
        if chunk.get("parser_version"):
            meta["parser_version"] = str(chunk.get("parser_version"))
        if chunk.get("quality_status"):
            meta["quality_status"] = str(chunk.get("quality_status"))
        if chunk.get("quality_score") is not None:
            meta["quality_score"] = float(chunk.get("quality_score", 0.0))
    if chunk.get("split_from"):
        meta["split_from"] = str(chunk.get("split_from"))
        meta["embedding_split_index"] = int(chunk.get("embedding_split_index", 0))
        meta["embedding_split_count"] = int(chunk.get("embedding_split_count", 0))
    if chunk.get("embedding_token_count") is not None:
        meta["embedding_token_count"] = int(chunk.get("embedding_token_count", 0))
    if chunk.get("chunk_policy_version"):
        meta["chunk_policy_version"] = str(chunk.get("chunk_policy_version"))
    return meta


def build_chroma_index(
    chunks: list[dict],
    documents: list[str],
    metadatas: list[dict],
    embeddings: list[list[float]],
    out_dir: Path,
    collection_name: str,
) -> None:
    import chromadb
    from chromadb.config import Settings

    out_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(out_dir), settings=Settings(anonymized_telemetry=False))

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    ids = [str(c["chunk_id"]) for c in chunks]
    batch_size = 64
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )
