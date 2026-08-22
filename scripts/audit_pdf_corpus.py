"""Audit expected maritime document families using the Accurate FTS sidecar."""
from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/processed/index/unified_full_corpus_715_v1/accurate_sparse_fts5_v2.sqlite3"
OUT_CSV = ROOT / "reports/corpus_audit_20260821.csv"
OUT_MD = ROOT / "reports/corpus_audit_20260821.md"

EXPECTED = [
    ("IMO", "MARPOL / Annex VI", r"marpol|annex\s*vi"),
    ("IMO", "SOLAS", r"solas"),
    ("IMO", "COLREG", r"colreg"),
    ("IMO", "STCW", r"stcw"),
    ("IMO", "BWM Convention", r"ballast|bwm"),
    ("IMO", "IMO GHG Strategy", r"ghg.*strategy|strategy.*ghg"),
    ("IMO", "CII / EEXI / SEEMP", r"\bcii\b|\beexi\b|\bseemp\b"),
    ("IMO", "MASS", r"\bmass\b|autonomous surface"),
    ("MEPC", "MEPC 80", r"mepc[ _.-]*80"),
    ("MEPC", "MEPC 81", r"mepc[ _.-]*81"),
    ("MEPC", "MEPC 82", r"mepc[ _.-]*82"),
    ("MEPC", "MEPC 83", r"mepc[ _.-]*83"),
    ("MEPC", "MEPC 84", r"mepc[ _.-]*84"),
    ("MSC", "MSC 107", r"msc[ _.-]*107"),
    ("MSC", "MSC 108", r"msc[ _.-]*108"),
    ("MSC", "MSC 109", r"msc[ _.-]*109"),
    ("MSC", "MSC 110", r"msc[ _.-]*110"),
    ("MSC", "MSC 111", r"msc[ _.-]*111"),
    ("DNV", "DNV documents", r"\bdnv\b|dnv[-_]"),
    ("KR", "KR documents", r"한국선급|\bkr\b|kr[-_]"),
    ("TOPIC", "Autonomous / MASS", r"autonomous|remotely operated|\bmass\b"),
    ("TOPIC", "Environment / GHG / Energy", r"environment|greenhouse|\bghg\b|energy efficiency"),
]


def main() -> None:
    if not DB.exists():
        raise FileNotFoundError(DB)
    with sqlite3.connect(DB) as connection:
        docs = connection.execute(
            """SELECT doc_id, MIN(file_name), MAX(source), MAX(session_org),
                      MAX(session_number), MAX(source_type), MAX(document_status)
               FROM chunks GROUP BY doc_id"""
        ).fetchall()
    records = []
    for publisher, family, pattern in EXPECTED:
        regex = re.compile(pattern, re.I)
        matches = [row for row in docs if regex.search(f"{row[0]} {row[1]}")]
        examples = [row[1] for row in matches[:5]]
        records.append(
            {
                "publisher": publisher,
                "document_family": family,
                "found": bool(matches),
                "document_count": len(matches),
                "example_files": " | ".join(examples),
            }
        )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = [
        "# PDF Corpus Audit",
        "",
        f"- sidecar distinct documents: **{len(docs)}**",
        "- This is a filename/metadata inventory, not a completeness or legal-currentness guarantee.",
        "",
        "| Publisher | Document family | Found | Documents |",
        "|---|---|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['publisher']} | {row['document_family']} | "
            f"{'yes' if row['found'] else 'no'} | {row['document_count']} |"
        )
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"documents": len(docs), "families": records}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
