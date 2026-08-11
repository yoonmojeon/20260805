"""Rewrite tracked PDF manifests to portable paths under data/raw_pdfs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw_pdfs"
MANIFEST_ROOT = ROOT / "data" / "manifests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="변경하지 않고 필요한 수정 수만 출력")
    return parser.parse_args()


def pdf_lookup() -> dict[str, Path]:
    by_name: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in RAW_ROOT.rglob("*.pdf"):
        key = path.name.casefold()
        if key in by_name:
            duplicates.add(key)
        else:
            by_name[key] = path
    if duplicates:
        names = ", ".join(sorted(duplicates)[:10])
        raise ValueError(f"중복 PDF 파일명은 자동 변환할 수 없습니다: {names}")
    return by_name


def localize_manifest(path: Path, lookup: dict[str, Path], *, check: bool) -> tuple[int, list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "file_path" not in fieldnames or "file_name" not in fieldnames:
        return 0, []

    changed = 0
    unresolved: list[str] = []
    for row in rows:
        file_name = str(row.get("file_name") or "").strip()
        if not file_name:
            continue
        local = lookup.get(file_name.casefold())
        if local is None:
            unresolved.append(file_name)
            continue
        portable = local.relative_to(ROOT).as_posix()
        if row.get("file_path") != portable:
            row["file_path"] = portable
            changed += 1

    if changed and not check:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    return changed, unresolved


def main() -> int:
    args = parse_args()
    lookup = pdf_lookup()
    total_changed = 0
    unresolved: list[str] = []
    for manifest in sorted(MANIFEST_ROOT.glob("*.csv")):
        changed, missing = localize_manifest(manifest, lookup, check=args.check)
        total_changed += changed
        unresolved.extend(f"{manifest.name}: {name}" for name in missing)
        if changed:
            verb = "needs" if args.check else "updated"
            print(f"{manifest.relative_to(ROOT).as_posix()}: {verb} {changed}")
    print(f"PDF files: {len(lookup)}")
    print(f"Manifest rows {'needing changes' if args.check else 'updated'}: {total_changed}")
    if unresolved:
        print(f"Unresolved rows: {len(unresolved)}")
        for item in unresolved[:20]:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
