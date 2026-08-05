from __future__ import annotations

import csv
import json
from pathlib import Path


STALE_PATH_MARKERS = (
    "OneDrive",
    "Desktop/MaritimeRAG",
    "Desktop\\MaritimeRAG",
)


def project_root_from_cwd() -> Path:
    return Path.cwd().resolve()


def extract_relative_data_path(path_str: str) -> str | None:
    normalized = path_str.replace("\\", "/")
    lower = normalized.lower()
    key = "maritimerag/"
    idx = lower.find(key)
    if idx < 0:
        return None
    rel = normalized[idx + len(key) :]
    if rel.lower().startswith("data/"):
        return rel
    return None


def rewrite_stale_path(path_str: str, project_root: Path | None = None) -> tuple[str, bool]:
    if not path_str:
        return path_str, False

    if not any(marker in path_str for marker in STALE_PATH_MARKERS):
        return path_str, False

    root = project_root or project_root_from_cwd()
    rel = extract_relative_data_path(path_str)
    if rel is None:
        return path_str, False

    return (root / rel).resolve().as_posix(), True


def resolve_page_image_path(
    doc_id: str,
    page_name: str,
    pages_root: Path | None = None,
) -> str:
    root = pages_root or Path("data/processed/pages")
    image_path = (root / doc_id / page_name).resolve()
    return image_path.as_posix()


def normalize_page_json_paths(page_data: dict, pages_root: Path | None = None) -> dict:
    doc_id = str(page_data.get("doc_id", ""))
    page_number = int(page_data.get("page_number", 1))
    page_name = str(page_data.get("page_name", f"page_{page_number:04d}.png"))
    if not doc_id:
        return page_data

    root = project_root_from_cwd()
    out = dict(page_data)
    out["page_name"] = page_name
    out["image_path"] = resolve_page_image_path(doc_id, page_name, pages_root)

    merge_meta = out.get("merge_meta")
    if isinstance(merge_meta, dict):
        merge_meta = dict(merge_meta)
        pdf_path = merge_meta.get("pdf_path")
        if isinstance(pdf_path, str):
            new_pdf, _ = rewrite_stale_path(pdf_path, root)
            merge_meta["pdf_path"] = new_pdf
        out["merge_meta"] = merge_meta

    return out


def fix_layout_json_dir(layout_doc_dir: Path, pages_root: Path | None = None) -> int:
    if not layout_doc_dir.exists():
        raise FileNotFoundError(f"Layout directory not found: {layout_doc_dir}")

    updated = 0
    for json_file in sorted(layout_doc_dir.glob("page_*.json")):
        page_data = json.loads(json_file.read_text(encoding="utf-8"))
        fixed = normalize_page_json_paths(page_data, pages_root)
        if json.dumps(fixed, sort_keys=True) != json.dumps(page_data, sort_keys=True):
            updated += 1
        json_file.write_text(json.dumps(fixed, ensure_ascii=False, indent=2), encoding="utf-8")
    return updated


def fix_crops_manifest(manifest_path: Path, project_root: Path | None = None) -> int:
    if not manifest_path.exists():
        return 0

    root = project_root or project_root_from_cwd()
    crops_folder = manifest_path.parent.parent.name
    doc_id_dir = manifest_path.parent

    updated = 0
    out_lines: list[str] = []
    with manifest_path.open(encoding="utf-8") as manifest_f:
        for line in manifest_f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            doc_id = str(rec["doc_id"])
            page_number = int(rec["page_number"])
            element_type = str(rec["element_type"])
            crop_name = Path(str(rec.get("crop_path", ""))).name
            page_name = f"page_{page_number:04d}.png"

            new_crop = (root / "data/processed" / crops_folder / doc_id / element_type / crop_name).resolve()
            new_source = (root / "data/processed/pages" / doc_id / page_name).resolve()

            new_rec = dict(rec)
            changed = False
            if new_rec.get("crop_path") != new_crop.as_posix():
                new_rec["crop_path"] = new_crop.as_posix()
                changed = True
            if new_rec.get("source_image_path") != new_source.as_posix():
                new_rec["source_image_path"] = new_source.as_posix()
                changed = True

            if changed:
                updated += 1
            out_lines.append(json.dumps(new_rec, ensure_ascii=False))

    manifest_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return updated


def fix_csv_path_columns(csv_path: Path, path_columns: tuple[str, ...], project_root: Path | None = None) -> int:
    if not csv_path.exists():
        return 0

    root = project_root or project_root_from_cwd()
    with csv_path.open(encoding="utf-8", newline="") as csv_f:
        reader = csv.DictReader(csv_f)
        if reader.fieldnames is None:
            return 0
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    updated_cells = 0
    for row in rows:
        for col in path_columns:
            if col not in row or not row[col]:
                continue
            new_path, changed = rewrite_stale_path(str(row[col]), root)
            if changed:
                row[col] = new_path
                updated_cells += 1

    with csv_path.open("w", encoding="utf-8", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return updated_cells


def find_stale_path_files(processed_root: Path) -> list[Path]:
    hits: list[Path] = []
    patterns = ("*.json", "*.jsonl", "*.csv", "*.txt")
    for pattern in patterns:
        for path in processed_root.rglob(pattern):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if any(marker in text for marker in STALE_PATH_MARKERS):
                hits.append(path)
    return sorted(set(hits))
