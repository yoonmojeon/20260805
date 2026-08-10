"""Snap all expanded-pilot TATR grids to PDF vectors and summarize agreement."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_ID = "kr_kr_rules_7_2025_2f2d6373"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT/"data/manifests/kr7_expanded_table_pilot.json")
    parser.add_argument("--pilot-root", type=Path, default=ROOT/"data/processed/kr7_expanded_table_pilot")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(); manifest = json.loads(args.manifest.read_text(encoding="utf-8")); rows=[]
    for index, table in enumerate(manifest["tables"], 1):
        command = [sys.executable, str(ROOT/"scripts/53_snap_tatr_to_pdf.py"), "--table-id", table["table_id"],
                   "--pilot-root", str(args.pilot_root)]
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        result_path = args.pilot_root/DOC_ID/table["table_id"]/"tatr_v1_1_all/snapped_structure.json"
        if process.returncode or not result_path.exists():
            rows.append({"table_id":table["table_id"],"status":"error","error":process.stderr[-800:]}); continue
        result=json.loads(result_path.read_text(encoding="utf-8")); comparison=result["coordinate_parser_comparison"]
        row={"table_id":table["table_id"],"page":table["page"],"caption":table["caption"],"status":"snapped",
             "tatr_rows":result["row_count"],"tatr_columns":result["column_count"],
             "legacy_rows":comparison["raw_grid_rows"],"legacy_columns":comparison["raw_grid_columns"],
             "row_match_legacy":comparison["row_count_match"],"column_match_legacy":comparison["column_count_match"],
             "spanning_cells":len(result["spanning_cells"]),
             "pdf_horizontal_candidates":len(result["vector_candidates"]["horizontal"]),
             "pdf_vertical_candidates":len(result["vector_candidates"]["vertical"])}
        rows.append(row); print(f"[{index:02d}/{len(manifest['tables'])}] {table['table_id']} "
            f"TATR={row['tatr_rows']}x{row['tatr_columns']} legacy={row['legacy_rows']}x{row['legacy_columns']}",flush=True)
    successful=[r for r in rows if r["status"]=="snapped"]
    summary={"tables":len(rows),"successful":len(successful),"errors":len(rows)-len(successful),
             "row_match_legacy":sum(r["row_match_legacy"] for r in successful),
             "column_match_legacy":sum(r["column_match_legacy"] for r in successful),
             "full_grid_match_legacy":sum(r["row_match_legacy"] and r["column_match_legacy"] for r in successful)}
    output=args.output or args.manifest.with_name(f"{args.manifest.stem}_snap_results.json")
    output.write_text(json.dumps({"summary":summary,"tables":rows},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({**summary,"output":str(output.resolve())},ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
