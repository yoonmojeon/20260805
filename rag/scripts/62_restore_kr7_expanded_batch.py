"""Run HancomEQN restoration over all snapped expanded-pilot tables."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DOC_ID="kr_kr_rules_7_2025_2f2d6373"


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=ROOT/"data/manifests/kr7_expanded_table_pilot.json")
    parser.add_argument("--pilot-root",type=Path,default=ROOT/"data/processed/kr7_expanded_table_pilot")
    parser.add_argument("--mapping",type=Path,default=ROOT/"data/config/hancomeqn_maps/7_2025_bdc15136d686.json")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(); manifest=json.loads(args.manifest.read_text(encoding="utf-8")); rows=[]; unknown=Counter()
    for index,table in enumerate(manifest["tables"],1):
        command=[sys.executable,str(ROOT/"scripts/55_restore_hancomeqn.py"),"--table-id",table["table_id"],
                 "--mapping",str(args.mapping),"--pilot-root",str(args.pilot_root)]
        process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding="utf-8",errors="replace")
        result_path=args.pilot_root/DOC_ID/table["table_id"]/"tatr_v1_1_all/restored_formulas.json"
        if process.returncode or not result_path.exists():
            rows.append({"table_id":table["table_id"],"status":"error","error":process.stderr[-800:]}); continue
        result=json.loads(result_path.read_text(encoding="utf-8")); formula_cells=result["cells"]
        review=[cell for cell in formula_cells if cell["formula"]["needs_review"]]
        dominant=[cell for cell in formula_cells if cell.get("formula_dominant")]
        for cell in review:
            unknown.update(cell["formula"]["unknown_glyphs"])
        row={"table_id":table["table_id"],"page":table["page"],"status":"restored",
             "hancomeqn_cells":len(formula_cells),"formula_dominant_cells":len(dominant),
             "needs_review_cells":len(review),"fully_mapped_cells":len(formula_cells)-len(review),
             "unknown_glyphs":sorted({code for cell in review for code in cell["formula"]["unknown_glyphs"]})}
        rows.append(row); print(f"[{index:02d}/{len(manifest['tables'])}] {table['table_id']} "
            f"cells={row['hancomeqn_cells']} review={row['needs_review_cells']}",flush=True)
    successful=[r for r in rows if r["status"]=="restored"]
    total_cells=sum(r["hancomeqn_cells"] for r in successful); review_cells=sum(r["needs_review_cells"] for r in successful)
    summary={"tables":len(rows),"successful":len(successful),"hancomeqn_cells":total_cells,
             "fully_mapped_cells":total_cells-review_cells,"needs_review_cells":review_cells,
             "fully_mapped_cell_rate":round((total_cells-review_cells)/total_cells,4) if total_cells else 1.0,
             "unknown_unique":len(unknown),"top_unknown":unknown.most_common(30)}
    output=args.output or args.manifest.with_name(f"{args.manifest.stem}_formula_results.json")
    output.write_text(json.dumps({"summary":summary,"tables":rows},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({**summary,"output":str(output.resolve())},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
