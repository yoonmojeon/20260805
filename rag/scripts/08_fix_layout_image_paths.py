from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from layout_paths import (
    find_stale_path_files,
    fix_crops_manifest,
    fix_csv_path_columns,
    fix_layout_json_dir,
    project_root_from_cwd,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite stale absolute paths (e.g. OneDrive) to current project root."
    )
    parser.add_argument("--doc-id", type=str, default=None, help="Fix one doc subfolder only")
    parser.add_argument(
        "--layout-dir",
        type=Path,
        action="append",
        default=None,
        help="Layout root(s) to fix (repeatable). Default: layout_json and layout_json_merged",
    )
    parser.add_argument(
        "--pages-root",
        type=Path,
        default=Path("data/processed/pages"),
        help="Page images root directory",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
        help="Processed data root to scan for remaining stale paths",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Skip crops_manifest.jsonl path fixes",
    )
    args = parser.parse_args()

    project_root = project_root_from_cwd()
    layout_roots = args.layout_dir or [
        Path("data/processed/layout_json"),
        Path("data/processed/layout_json_merged"),
    ]

    layout_files_updated = 0
    for layout_root in layout_roots:
        if not layout_root.exists():
            print(f"[SKIP] Not found: {layout_root}")
            continue

        if args.doc_id:
            targets = [layout_root / args.doc_id]
        else:
            targets = sorted(p for p in layout_root.iterdir() if p.is_dir())

        for layout_doc_dir in targets:
            if not layout_doc_dir.is_dir():
                print(f"[SKIP] Not a directory: {layout_doc_dir}")
                continue
            changed = fix_layout_json_dir(layout_doc_dir, args.pages_root)
            layout_files_updated += changed
            status = "updated" if changed else "ok"
            print(f"Layout [{status}] {changed} file(s): {layout_doc_dir}")

    manifest_rows_updated = 0
    if not args.skip_manifest:
        manifest_glob = "*/crops_manifest.jsonl"
        if args.doc_id:
            patterns = [
                args.processed_root / "crops" / args.doc_id / "crops_manifest.jsonl",
                args.processed_root / "crops_merged" / args.doc_id / "crops_manifest.jsonl",
            ]
        else:
            patterns = list(args.processed_root.glob("crops*/**/crops_manifest.jsonl"))

        for manifest_path in patterns:
            if not manifest_path.exists():
                continue
            changed = fix_crops_manifest(manifest_path, project_root)
            manifest_rows_updated += changed
            print(f"Manifest updated {changed} row(s): {manifest_path}")

    csv_cells_updated = 0
    logs_root = args.processed_root / "logs"
    if logs_root.exists():
        for csv_path in sorted(logs_root.glob("*.csv")):
            changed = fix_csv_path_columns(csv_path, ("crop_path",), project_root)
            if changed:
                csv_cells_updated += changed
                print(f"CSV updated {changed} cell(s): {csv_path}")

    remaining = find_stale_path_files(args.processed_root)
    if args.doc_id:
        doc_key = args.doc_id.replace("\\", "/")
        remaining = [p for p in remaining if doc_key in p.as_posix()]

    print("")
    print(f"Project root: {project_root.as_posix()}")
    print(f"Layout JSON files rewritten: {layout_files_updated}")
    print(f"Manifest rows rewritten: {manifest_rows_updated}")
    print(f"CSV path cells rewritten: {csv_cells_updated}")
    if remaining:
        print(f"Remaining files with stale markers ({len(remaining)}):")
        for path in remaining:
            print(f"  - {path.as_posix()}")
    else:
        print("No remaining stale paths under data/processed.")


if __name__ == "__main__":
    main()
