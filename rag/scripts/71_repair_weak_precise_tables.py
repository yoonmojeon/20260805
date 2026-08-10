"""Repair weak precise tables and quarantine ones that cannot be fixed overnight.

Same TATR+snap methodology as script 70, but only for failing/thin candidates.
PUA-mapping failures are quarantined (kept on disk, excluded from index).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def body_text(text: str) -> str:
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    if len(lines) > 2:
        return "\n".join(lines[2:])
    return "\n".join(lines)


def is_tocish(text: str) -> bool:
    t = body_text(text)
    if "Surveys..." in t or t.count("....") >= 2:
        return True
    if t.count("|") <= 1 and ("...." in t or "Surveys" in t):
        return True
    return False


def is_thin_body(text: str, min_chars: int = 40) -> bool:
    return len(body_text(text).strip()) < min_chars


def collect_candidates(args: argparse.Namespace) -> dict:
    audit = load_json(args.audit)
    items = audit.get("tables") if isinstance(audit, dict) else audit
    manifest = load_json(args.pipeline_manifest)
    by_id = {str(t["table_id"]): t for t in manifest.get("tables") or []}

    # Judge thinness on SUMMARY chunks only (rows are often short by design).
    thin_summary: set[str] = set()
    tocish: set[str] = set()
    emptyish_summary: set[str] = set()
    for path in args.chunks_root.glob("*/table_chunks.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            tid = str(row.get("table_id") or "")
            ctype = str(row.get("chunk_type") or "").lower()
            text = str(row.get("text") or "")
            if not tid or ctype != "table_summary":
                continue
            body = body_text(text)
            if is_tocish(text):
                tocish.add(tid)
            if is_thin_body(text, min_chars=80):
                thin_summary.add(tid)
            usable = body.replace("(빈 셀)", "").replace("|", " ")
            usable = " ".join(usable.split())
            if len(usable) < 24 and body.count("(빈 셀)") >= 2:
                emptyish_summary.add(tid)

    not_indexable: dict[str, list[str]] = {}
    empty_grid: set[str] = set()
    missing_structure: set[str] = set()
    for item in items:
        tid = str(item.get("table_id") or "")
        reasons = [str(r) for r in (item.get("reasons") or [])]
        if not item.get("indexable"):
            not_indexable[tid] = reasons
        if any(r.startswith("empty_grid") for r in reasons):
            empty_grid.add(tid)
        if any("missing_snapped_structure" in r or "missing_structure" in r for r in reasons):
            missing_structure.add(tid)

    # Repairable tonight: structural failures TATR/snap can actually fix.
    # Quarantine: PUA mapping gaps / TOC-like noise (exclude from index).
    repairable: list[str] = []
    quarantine: list[str] = []
    reason_counts: Counter[str] = Counter()
    per_table_tags: dict[str, list[str]] = {}

    all_ids = set(not_indexable) | thin_summary | tocish | empty_grid | emptyish_summary | missing_structure
    for tid in sorted(all_ids):
        reasons = not_indexable.get(tid, [])
        tags: set[str] = set()
        if tid in empty_grid or any(r.startswith("empty_grid") for r in reasons):
            tags.add("empty_grid")
        if tid in missing_structure:
            tags.add("missing_structure")
        if tid in thin_summary:
            tags.add("thin_summary")
        if tid in emptyish_summary:
            tags.add("emptyish_summary")
        if tid in tocish:
            tags.add("tocish")
        if any(
            "pua" in r.lower() or "mapping" in r.lower() or "formula" in r.lower() or "restore_review" in r.lower()
            for r in reasons
        ):
            tags.add("pua_or_mapping")
        if any(r.startswith("no_usable_text") for r in reasons):
            tags.add("no_usable_text")
        for tag in tags:
            reason_counts[tag] += 1
        per_table_tags[tid] = sorted(tags)

        structural = {"empty_grid", "missing_structure", "thin_summary", "emptyish_summary"}
        # TOC false positives: quarantine (don't burn GPU)
        if "tocish" in tags and not (tags & (structural - {"thin_summary"})):
            quarantine.append(tid)
            continue
        # Pure PUA/mapping: quarantine until glyph maps exist
        if tags and tags <= {"pua_or_mapping", "no_usable_text"}:
            quarantine.append(tid)
            continue
        if "pua_or_mapping" in tags and not (tags & {"empty_grid", "missing_structure"}):
            quarantine.append(tid)
            continue
        if tid in by_id and (tags & structural or ("no_usable_text" in tags and "pua_or_mapping" not in tags)):
            repairable.append(tid)
        elif tid in by_id and tags:
            quarantine.append(tid)

    repairable = [tid for tid in repairable if tid in by_id]
    quarantine = sorted(set(quarantine))
    payload = {
        "summary": {
            "audit_not_indexable": len(not_indexable),
            "thin_summary_tables": len(thin_summary),
            "emptyish_summary_tables": len(emptyish_summary),
            "tocish_tables": len(tocish),
            "empty_grid": len(empty_grid),
            "missing_structure": len(missing_structure),
            "repairable": len(repairable),
            "quarantine": len(quarantine),
            "reason_counts": dict(reason_counts),
        },
        "repairable_table_ids": repairable,
        "quarantine_table_ids": quarantine,
        "repairable_tables": [by_id[tid] for tid in repairable],
        "tags_by_table": {tid: per_table_tags[tid] for tid in repairable + quarantine if tid in per_table_tags},
    }
    write_json(args.report, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    return payload


def clear_outputs(item: dict, work_root: Path) -> None:
    work = Path(item.get("work_dir") or "")
    if not work.exists():
        # fallback via crop_path
        crop = Path(item.get("crop_path") or "")
        work = crop.parent if crop else work_root
    tatr = work / "tatr_v1_1_all"
    if tatr.exists():
        for name in (
            "structure.json",
            "snapped_structure.json",
            "snapped_overlay.png",
            "restored_formulas.json",
        ):
            path = tatr / name
            if path.exists():
                path.unlink()
    split = work / "region_split_v1"
    if split.exists():
        shutil.rmtree(split, ignore_errors=True)


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-manifest", type=Path, default=ROOT / "data/processed/logs/full_corpus_715_precise_manifest.json")
    parser.add_argument("--audit", type=Path, default=ROOT / "data/processed/logs/full_corpus_715_precise_audit.json")
    parser.add_argument("--chunks-root", type=Path, default=ROOT / "data/processed/chunks_tables_precise")
    parser.add_argument("--work-root", type=Path, default=ROOT / "data/processed/precise_tables")
    parser.add_argument("--report", type=Path, default=ROOT / "data/processed/logs/precise_table_repair_plan.json")
    parser.add_argument("--repair-manifest", type=Path, default=ROOT / "data/processed/logs/precise_table_repair_manifest.json")
    parser.add_argument("--quarantine-list", type=Path, default=ROOT / "data/processed/logs/precise_table_quarantine_ids.json")
    parser.add_argument("--doc-list", type=Path, default=ROOT / "data/manifests/full_corpus_715.csv")
    parser.add_argument("--pdf-manifest", type=Path, default=ROOT / "data/manifests/full_corpus_715.csv")
    parser.add_argument("--tables-root", type=Path, default=ROOT / "data/processed/tables")
    parser.add_argument("--collection-id", default="full_corpus_715_tables_precise_v1")
    parser.add_argument("--skip-repair-run", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()

    plan = collect_candidates(args)
    write_json(args.quarantine_list, {"table_ids": plan["quarantine_table_ids"]})

    repair_tables = plan["repairable_tables"]
    repair_manifest = {
        "schema_version": 1,
        "selection_method": "weak-or-failed-precise-tables-repair",
        "document_count": len({t["doc_id"] for t in repair_tables}),
        "table_count": len(repair_tables),
        "tables": repair_tables,
    }
    write_json(args.repair_manifest, repair_manifest)

    if not repair_tables:
        print("No repairable tables; quarantine list written.", flush=True)
    elif not args.skip_repair_run:
        for item in repair_tables:
            clear_outputs(item, args.work_root)
        # Re-run same methodology only on the filtered manifest.
        py = sys.executable
        # Use 70 internals via stages against filtered manifest: inject by replacing pipeline manifest temporarily.
        # Safer: call tatr/snap through 70 with prepare skipped and custom manifest path.
        # Do NOT run chunks here: 70 overwrites every doc's jsonl from the active
        # manifest only, which would wipe non-repair docs if given a subset.
        run([
            py, "scripts/70_build_precise_table_corpus.py",
            "--doc-list", str(args.doc_list),
            "--pdf-manifest", str(args.pdf_manifest),
            "--tables-root", str(args.tables_root),
            "--work-root", str(args.work_root),
            "--chunks-root", str(args.chunks_root),
            "--pipeline-manifest", str(args.repair_manifest),
            "--audit", str(ROOT / "data/processed/logs/precise_table_repair_audit.json"),
            "--collection-id", args.collection_id,
            "--stages", "tatr,snap,restore,segment,region-tatr",
            "--resume",
        ])

    if not args.skip_index:
        # Rebuild all chunks from the full pipeline manifest, then reindex.
        # Quarantine IDs are applied by index_build_lib via env/JSON side file.
        run([
            sys.executable, "scripts/70_build_precise_table_corpus.py",
            "--doc-list", str(args.doc_list),
            "--pdf-manifest", str(args.pdf_manifest),
            "--tables-root", str(args.tables_root),
            "--work-root", str(args.work_root),
            "--chunks-root", str(args.chunks_root),
            "--pipeline-manifest", str(args.pipeline_manifest),
            "--audit", str(args.audit),
            "--collection-id", args.collection_id,
            "--stages", "chunks,index",
            "--resume",
        ])


if __name__ == "__main__":
    main()
