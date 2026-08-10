#!/usr/bin/env python
"""Document why Text Chroma has 714 docs while the corpus CSV lists 715 PDFs."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import DATA_DIR, DEFAULT_RAG_COLLECTION, RAG_CHUNKS_DIR, RAG_INDEX_DIR, RAW_PDFS_DIR

try:
    import fitz
except Exception:  # pragma: no cover
    fitz = None


def main() -> int:
    corpus_path = DATA_DIR / "manifests" / "full_corpus_715.csv"
    with corpus_path.open(encoding="utf-8-sig", newline="") as fh:
        corpus = list(csv.DictReader(fh))
    corpus_ids = {r["doc_id"] for r in corpus}

    man = json.loads(
        (RAG_INDEX_DIR / f"unified_{DEFAULT_RAG_COLLECTION}" / "index_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    indexed = set(man.get("doc_ids") or [])
    missing = list(man.get("missing_chunks_doc_ids") or [])
    raw_pdfs = list(RAW_PDFS_DIR.rglob("*.pdf")) + list(RAW_PDFS_DIR.rglob("*.PDF"))
    raw_n = len({p.resolve() for p in raw_pdfs})

    details = []
    for doc_id in missing:
        row = next((r for r in corpus if r["doc_id"] == doc_id), {})
        pdf = Path(row.get("file_path") or "")
        if not pdf.exists():
            hits = list(RAW_PDFS_DIR.rglob(row.get("file_name") or ""))
            pdf = hits[0] if hits else pdf
        chunk_path = RAG_CHUNKS_DIR / doc_id / "chunks.jsonl"
        info = {
            "doc_id": doc_id,
            "file_name": row.get("file_name"),
            "source": row.get("source"),
            "file_path": str(pdf),
            "pdf_exists": pdf.exists(),
            "pdf_bytes": pdf.stat().st_size if pdf.exists() else None,
            "chunks_jsonl_bytes": chunk_path.stat().st_size if chunk_path.exists() else None,
            "pages": None,
            "page_text_len": None,
            "reason": None,
        }
        if pdf.exists() and fitz is not None:
            doc = fitz.open(pdf)
            info["pages"] = doc.page_count
            text_len = sum(len(doc[i].get_text("text") or "") for i in range(doc.page_count))
            info["page_text_len"] = text_len
            doc.close()
            if info["pdf_bytes"] is not None and info["pdf_bytes"] < 4096 and text_len == 0:
                info["reason"] = (
                    "empty_withdrawn_stub_pdf: preprocess produced 0 text chunks; "
                    "index build skipped via missing_chunks_doc_ids (not a silent drop)"
                )
            elif text_len == 0:
                info["reason"] = "pdf_has_no_extractable_text; chunks empty"
            else:
                info["reason"] = "chunks_missing_despite_text; investigate preprocess"
        else:
            info["reason"] = "pdf_missing_or_pymupdf_unavailable"
        details.append(info)

    only_in_corpus = sorted(corpus_ids - indexed)
    report = {
        "raw_pdfs_rglob": raw_n,
        "corpus_csv_rows": len(corpus),
        "text_collection": DEFAULT_RAG_COLLECTION,
        "text_indexed_documents": len(indexed),
        "missing_from_text_index": details,
        "doc_ids_in_csv_not_indexed": only_in_corpus,
    }
    out = DATA_DIR / "eval" / "text_corpus_coverage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
