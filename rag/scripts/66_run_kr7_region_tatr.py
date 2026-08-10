#!/usr/bin/env python3
"""Run one cached TATR model over grid regions from the KR7 compound splitter."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
batch = importlib.import_module("60_run_kr7_tatr_batch")
snap = importlib.import_module("53_snap_tatr_to_pdf")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/manifests/kr7_expanded_region_results.json",
    )
    parser.add_argument(
        "--pilot-root",
        type=Path,
        default=ROOT / "data/processed/kr7_expanded_table_pilot",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--padding", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = []
    for table in manifest["tables"]:
        doc_id = str(table.get("doc_id") or snap.doc_id_from_table_id(table["table_id"]))
        if table.get("output"):
            split_dir = Path(table["output"]).parent
        elif table.get("work_dir"):
            split_dir = Path(table["work_dir"]) / "region_split_v1"
        elif table.get("crop_path"):
            split_dir = Path(table["crop_path"]).parent / "region_split_v1"
        else:
            split_dir = args.pilot_root / doc_id / table["table_id"] / "region_split_v1"
        regions_path = split_dir / "regions.json"
        if not regions_path.exists():
            print(f"skip missing regions.json: {table['table_id']} -> {regions_path}", flush=True)
            continue
        regions = json.loads(regions_path.read_text(encoding="utf-8"))["regions"]
        for region in regions:
            children = region.get("children", [])
            targets = children or ([region] if region.get("requires_local_structure") else [])
            for target in targets:
                jobs.append((table, split_dir, target))

    runtime = batch.load_runtime(batch.TATR.MODEL_ID)
    rows = []
    for index, (table, split_dir, target) in enumerate(jobs, 1):
        region_id = target["region_id"]
        result_path = split_dir / f"{region_id}_tatr" / "structure.json"
        if args.reuse_existing and result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result = batch.infer(
                split_dir / f"{region_id}.png", runtime, args.threshold,
                args.padding, split_dir / f"{region_id}_tatr",
            )
        summary = result["summary"]
        row = {
            "table_id": table["table_id"],
            "doc_id": str(table.get("doc_id") or snap.doc_id_from_table_id(table["table_id"])),
            "page": table["page"],
            "region_id": region_id,
            "region_kind": target["kind"],
            "row_count": summary["row_count"],
            "column_count": summary["column_count"],
            "column_header_count": summary["column_header_count"],
            "spanning_cell_count": summary["spanning_cell_count"],
        }
        rows.append(row)
        print(
            f"[{index:02d}/{len(jobs)}] p.{table['page']} {region_id} "
            f"rows={row['row_count']} cols={row['column_count']}",
            flush=True,
        )

    output = args.output or args.manifest.with_name(f"{args.manifest.stem}_tatr_results.json")
    output.write_text(
        json.dumps(
            {
                "model": batch.TATR.MODEL_ID,
                "summary": {"jobs": len(jobs), "successful": len(rows)},
                "regions": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output.resolve()), "jobs": len(jobs)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
