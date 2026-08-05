"""Shared project root and data paths for the unified MaritimeOpsRAG app."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OPS_DIR = PROJECT_ROOT / "ops"
RAG_DIR = PROJECT_ROOT / "rag"
RAG_SCRIPTS_DIR = RAG_DIR / "scripts"

HO_DATA_DIR = DATA_DIR / "ho_data"
OPS_DB_PATH = DATA_DIR / "maritime.db"
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "output"

# MaritimeRAG collections (Chroma under data/processed/index)
RAG_INDEX_DIR = PROCESSED_DIR / "index"
RAG_CHUNKS_DIR = PROCESSED_DIR / "chunks"
RAG_TABLE_CHUNKS_DIR = PROCESSED_DIR / "chunks_v2"
DEFAULT_RAG_COLLECTION = "full_corpus_715_v1"
DEFAULT_TABLE_COLLECTION = "kr_tables_v2"
