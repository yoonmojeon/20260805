from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import pandas as pd


HIGH_PRIORITY_KEYWORDS = (
    "DNV-RU-GEN-0050",
    "DNV-RU-GEN-0587",
    "RU-SHIP",
    "LR-RU-001",
)


def normalize_token(value: str) -> str:
    # 공백/괄호/특수문자를 underscore로 치환하고 중복 underscore를 정리합니다.
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", value)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized.lower() if sanitized else "unknown"


def infer_source(pdf_path: Path) -> str:
    path_lower = pdf_path.as_posix().lower()
    if "abs" in path_lower:
        return "ABS"
    if "dnv" in path_lower:
        return "DNV"
    if "kr" in path_lower:
        return "KR"
    if "lr rules" in path_lower:
        return "LR"
    if re.search(r"[/\\]mepc[/\\]", path_lower):
        return "MEPC"
    if re.search(r"[/\\]msc[/\\]", path_lower) or re.search(r"[/\\]imo[/\\]msc[/\\]", path_lower):
        return "MSC"
    return "UNKNOWN"


def infer_category(pdf_path: Path, input_dir: Path, source: str) -> str:
    rel_parts = [p.lower() for p in pdf_path.relative_to(input_dir).parts[:-1]]

    source_key = source.lower()
    if source != "UNKNOWN" and source_key in rel_parts:
        idx = rel_parts.index(source_key)
        if idx + 1 < len(rel_parts):
            return normalize_token(rel_parts[idx + 1]).upper()

    filename_upper = pdf_path.stem.upper()
    filename_rules = [
        ("RU", "RULES"),
        ("GUIDE", "GUIDE"),
        ("GUIDANCE", "GUIDANCE"),
        ("NOTE", "NOTE"),
        ("STANDARD", "STANDARD"),
        ("REQ", "REQUIREMENT"),
        ("REQUIREMENT", "REQUIREMENT"),
    ]
    for key, category in filename_rules:
        if key in filename_upper:
            return category
    return "GENERAL"


def infer_priority(file_name: str) -> int:
    file_name_upper = file_name.upper()
    for kw in HIGH_PRIORITY_KEYWORDS:
        if kw in file_name_upper:
            return 1
    return 3


# 파일럿·eval과 호환되는 고정 ID (상대 경로 기준, 소문자)
CANONICAL_DOC_ID_ALIASES: dict[str, str] = {
    "kr rules/1편_2025.pdf": "kr_1_2025",
    "lr rules/notice no.1 rules and regulations for the classification of ships, july 2025.pdf": "lr_notice_no_1_2025",
}


def path_hash_suffix(rel_posix: str, length: int = 8) -> str:
    return hashlib.sha1(rel_posix.encode("utf-8")).hexdigest()[:length]


def make_doc_id(source: str, pdf_path: Path, input_dir: Path) -> str:
    rel_posix = pdf_path.relative_to(input_dir).as_posix().replace("\\", "/").lower()
    if rel_posix in CANONICAL_DOC_ID_ALIASES:
        return CANONICAL_DOC_ID_ALIASES[rel_posix]

    rel = pdf_path.relative_to(input_dir)
    parts = [normalize_token(source)]
    parts.extend(normalize_token(part) for part in rel.with_suffix("").parts if part)
    doc_id = "_".join(p for p in parts if p)
    if not doc_id:
        doc_id = normalize_token(source)
    return f"{doc_id}_{path_hash_suffix(rel_posix)}"


def collect_pdfs(input_dir: Path) -> list[dict]:
    rows: list[dict] = []
    assigned: dict[str, str] = {}

    for pdf_path in sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"):
        source = infer_source(pdf_path)
        file_name = pdf_path.name
        file_path = pdf_path.resolve().as_posix()
        doc_id = make_doc_id(source, pdf_path, input_dir)

        if doc_id in assigned and assigned[doc_id] != file_path:
            rel_posix = pdf_path.relative_to(input_dir).as_posix().replace("\\", "/").lower()
            doc_id = f"{doc_id}_{path_hash_suffix(rel_posix)}"

        assigned[doc_id] = file_path
        rows.append(
            {
                "doc_id": doc_id,
                "source": source,
                "category": infer_category(pdf_path, input_dir, source),
                "file_path": file_path,
                "file_name": file_name,
                "status": "pending",
                "priority": infer_priority(file_name),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PDF manifest from data/raw_pdfs recursively.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw_pdfs"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/pdf_manifest.csv"))
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    rows = collect_pdfs(args.input_dir)
    columns = ["doc_id", "source", "category", "file_path", "file_name", "status", "priority"]
    df = pd.DataFrame(rows, columns=columns)

    df = df.drop_duplicates(subset=["file_path"]).reset_index(drop=True)

    dup_ids = df[df["doc_id"].duplicated(keep=False)]
    if not dup_ids.empty:
        sample = dup_ids.sort_values("doc_id").head(20)[["doc_id", "file_path"]]
        raise ValueError(
            "Duplicate doc_id values remain after path-based IDs. "
            f"Sample:\n{sample.to_string(index=False)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        df.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"No PDF files found. Empty manifest created: {args.output}")
        return

    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Manifest written: {args.output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
