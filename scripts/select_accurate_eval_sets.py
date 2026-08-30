"""Create immutable Accurate A/B/C evaluation subsets from the 405-row v3 set.

The selector never synthesizes or edits questions.  It copies complete JSONL
rows from the existing corpus and records the source hash and selected ids so
that every retrieval variant sees exactly the same questions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCE = Path("data/eval/pilot_validation_text_v3.jsonl")
DEFAULT_LEGACY_RECORDS = Path(
    "data/processed/logs/text_rag_eval_v3/"
    "retrieval_20260813_postfix_fast_rerun/records.jsonl"
)
DEFAULT_FIXED = Path("data/eval/accurate_eval_150.jsonl")
DEFAULT_PILOT = Path("data/eval/accurate_pilot_30.jsonl")
DEFAULT_MANIFEST = Path("data/eval/accurate_eval_selection_manifest.json")
DEFAULT_SEED = 260821


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    return {
        "total": len(materialized),
        "category": dict(sorted(Counter(str(r.get("category") or "") for r in materialized).items())),
        "test_type": dict(sorted(Counter(str(r.get("test_type") or "") for r in materialized).items())),
        "answerability": dict(
            sorted(Counter(str(bool(r.get("answerability"))).lower() for r in materialized).items())
        ),
    }


def _largest_remainder(groups: dict[tuple[str, str], list[dict[str, Any]]], total: int) -> dict[tuple[str, str], int]:
    population = sum(len(items) for items in groups.values())
    raw = {key: total * len(items) / population for key, items in groups.items()}
    allocated = {key: min(len(groups[key]), math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(allocated.values())
    order = sorted(
        groups,
        key=lambda key: (raw[key] - math.floor(raw[key]), len(groups[key]), key),
        reverse=True,
    )
    while remaining:
        progressed = False
        for key in order:
            if allocated[key] < len(groups[key]):
                allocated[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise RuntimeError("Could not allocate requested stratified sample")
    return allocated


def select_fixed(rows: list[dict[str, Any]], *, size: int, seed: int) -> list[dict[str, Any]]:
    seed_rows = [r for r in rows if str(r.get("test_type")) == "seed"]
    if len(seed_rows) > size:
        raise ValueError("Fixed set is smaller than the original seed question set")
    remaining_rows = [r for r in rows if str(r.get("test_type")) != "seed"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in remaining_rows:
        groups[(str(row.get("category") or ""), str(row.get("test_type") or ""))].append(row)
    allocation = _largest_remainder(groups, size - len(seed_rows))
    rng = random.Random(seed)
    selected = list(seed_rows)
    for key in sorted(groups):
        items = sorted(groups[key], key=lambda r: str(r.get("question_id") or ""))
        selected.extend(rng.sample(items, allocation[key]))
    rng.shuffle(selected)
    return selected


def _legacy_scores(path: Path) -> dict[str, tuple[int, float, float]]:
    if not path.exists():
        return {}
    scores: dict[str, tuple[int, float, float]] = {}
    for row in _read_jsonl(path):
        qid = str(row.get("question_id") or "")
        doc_fail = int(row.get("answerability") is not False and not bool(row.get("final_doc_any")))
        point = float(row.get("final_point_recall") or 0.0)
        semantic = float(row.get("final_semantic_point_recall") or 0.0)
        scores[qid] = (doc_fail, point, semantic)
    return scores


def select_pilot(
    rows: list[dict[str, Any]],
    *,
    size: int,
    seed: int,
    legacy_records: Path,
) -> list[dict[str, Any]]:
    """Mix original demos, legacy failures, hard cases and negatives."""
    by_id = {str(r.get("question_id")): r for r in rows}
    selected: list[dict[str, Any]] = [r for r in rows if str(r.get("test_type")) == "seed"]
    selected_ids = {str(r.get("question_id")) for r in selected}
    legacy = _legacy_scores(legacy_records)
    rng = random.Random(seed + 30)

    answerable = [r for r in rows if r.get("answerability") is not False and str(r.get("question_id")) not in selected_ids]
    # Lower legacy point recall is more valuable; document failures come first.
    answerable.sort(
        key=lambda r: (
            -legacy.get(str(r.get("question_id")), (0, 1.0, 1.0))[0],
            legacy.get(str(r.get("question_id")), (0, 1.0, 1.0))[1],
            legacy.get(str(r.get("question_id")), (0, 1.0, 1.0))[2],
            rng.random(),
        )
    )
    # Round-robin across categories prevents a single weak scenario dominating.
    category_queues: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in answerable:
        category_queues[str(row.get("category") or "")].append(row)
    categories = sorted(category_queues)
    difficult_target = max(0, size - len(selected) - 3)
    while len(selected) < len([r for r in rows if str(r.get("test_type")) == "seed"]) + difficult_target:
        progressed = False
        for category in categories:
            queue = category_queues[category]
            while queue and str(queue[0].get("question_id")) in selected_ids:
                queue.pop(0)
            if queue:
                row = queue.pop(0)
                selected.append(row)
                selected_ids.add(str(row.get("question_id")))
                progressed = True
                if len(selected) >= size - 3:
                    break
        if not progressed:
            break

    negatives = [
        r for r in rows
        if r.get("answerability") is False and str(r.get("question_id")) not in selected_ids
    ]
    for row in rng.sample(negatives, min(3, len(negatives))):
        selected.append(row)
        selected_ids.add(str(row.get("question_id")))
    if len(selected) < size:
        remainder = [r for qid, r in by_id.items() if qid not in selected_ids]
        selected.extend(rng.sample(remainder, size - len(selected)))
    return selected[:size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--legacy-records", type=Path, default=DEFAULT_LEGACY_RECORDS)
    parser.add_argument("--fixed-output", type=Path, default=DEFAULT_FIXED)
    parser.add_argument("--pilot-output", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fixed-size", type=int, default=150)
    parser.add_argument("--pilot-size", type=int, default=30)
    args = parser.parse_args()

    rows = _read_jsonl(args.source)
    if len(rows) != 405:
        raise ValueError(f"Expected the existing 405-row set, found {len(rows)}")
    fixed = select_fixed(rows, size=args.fixed_size, seed=args.seed)
    pilot = select_pilot(
        rows,
        size=args.pilot_size,
        seed=args.seed,
        legacy_records=args.legacy_records,
    )
    _write_jsonl(args.fixed_output, fixed)
    _write_jsonl(args.pilot_output, pilot)
    manifest = {
        "schema_version": "maritime-accurate-eval-selection-v1",
        "source": str(args.source),
        "source_sha256": _sha256(args.source),
        "source_rows": len(rows),
        "seed": args.seed,
        "selection_policy": {
            "fixed": "all 9 seed rows plus proportional category/test_type stratified sample",
            "pilot": "all 9 seed rows plus legacy low-recall round-robin cases and 3 negatives",
            "questions_modified_or_generated": False,
        },
        "fixed": {
            "path": str(args.fixed_output),
            "question_ids": [str(r.get("question_id")) for r in fixed],
            "distribution": _distribution(fixed),
        },
        "pilot": {
            "path": str(args.pilot_output),
            "question_ids": [str(r.get("question_id")) for r in pilot],
            "distribution": _distribution(pilot),
            "legacy_records": str(args.legacy_records),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fixed": _distribution(fixed), "pilot": _distribution(pilot)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
