"""End-to-end table QA evaluation over distinct PDFs through the UI service.

The curated set contains three questions for each of 22 documents.  By
default this runner selects one question per PDF, generates the actual answer,
and checks both the gold value and whether the cited evidence comes from the
gold PDF/page.  It is intentionally separate from retrieval-only table probes.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_QUESTIONS = (
    ROOT / "data" / "eval" / "table_questions_22docs_practical_v1_curated.jsonl"
)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _norm(value: str) -> str:
    value = str(value or "")
    value = (
        value.replace("●", " black-circle ")
        .replace("○", " white-circle ")
        .replace("◯", " white-circle ")
    )
    value = re.sub(r"\((?:주\s*)?\d{1,2}\)", "", value)
    return re.sub(r"[^0-9a-z가-힣.%≤≥<>+-]+", "", value.lower())


def _select_distinct(rows: list[dict], per_doc: int) -> list[dict]:
    counts: dict[str, int] = {}
    selected: list[dict] = []
    for row in rows:
        doc_id = str(row.get("gold_doc_id") or "")
        if counts.get(doc_id, 0) >= per_doc:
            continue
        counts[doc_id] = counts.get(doc_id, 0) + 1
        selected.append(row)
    return selected


def _evidence(result: dict) -> list[dict]:
    meta = result.get("meta") or {}
    for key in ("evidence_table", "retrieved_sources", "sources"):
        value = result.get(key)
        if not value:
            value = meta.get(key)
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--per-doc", type=int, default=1)
    parser.add_argument(
        "--qids",
        default="",
        help="Optional comma-separated question ids for focused regression runs.",
    )
    parser.add_argument("--latency-mode", choices=("fast", "accurate"), default="fast")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = _select_distinct(_read_jsonl(args.questions), max(1, args.per_doc))
    if args.qids.strip():
        wanted = {value.strip() for value in args.qids.split(",") if value.strip()}
        rows = [row for row in rows if str(row.get("qid") or "") in wanted]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from services.rag_service import run_rag_query

    records: list[dict] = []
    with (args.out_dir / "records.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            started = time.perf_counter()
            try:
                result = run_rag_query(
                    str(row.get("question") or ""),
                    latency_mode=args.latency_mode,
                )
                answer = str(result.get("answer") or "")
                meta = result.get("meta") or {}
                evidence = _evidence(result)
                error = None
            except Exception as exc:  # keep the broad audit running
                answer, meta, evidence, error = "", {}, [], repr(exc)
            seconds = time.perf_counter() - started
            gold = str(row.get("gold_answer") or "")
            value_hit = bool(_norm(gold)) and _norm(gold) in _norm(answer)
            gold_file = str(row.get("gold_file_name") or "").lower()
            gold_page = int(row.get("gold_page") or -1)
            evidence_file_hit = any(
                str(item.get("file_name") or item.get("source") or "").lower()
                == gold_file
                for item in evidence
            )
            evidence_page_hit = any(
                str(item.get("file_name") or item.get("source") or "").lower()
                == gold_file
                and int(item.get("page") or item.get("page_number") or -1) == gold_page
                for item in evidence
            )
            record = {
                "qid": row.get("qid"),
                "question": row.get("question"),
                "gold_answer": gold,
                "gold_doc_id": row.get("gold_doc_id"),
                "gold_file_name": row.get("gold_file_name"),
                "gold_page": gold_page,
                "answer": answer,
                "value_hit": value_hit,
                "evidence_file_hit": evidence_file_hit,
                "evidence_page_hit": evidence_page_hit,
                "answer_mode": meta.get("answer_mode"),
                "seconds": round(seconds, 4),
                "under_10s": seconds <= 10.0,
                "error": error,
            }
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                f"[{index}/{len(rows)}] {record['qid']} "
                f"value={'HIT' if value_hit else 'MISS'} {seconds:.2f}s",
                flush=True,
            )

    valid = [record for record in records if not record["error"]]
    summary = {
        "n": len(records),
        "distinct_docs": len({record["gold_doc_id"] for record in records}),
        "errors": len(records) - len(valid),
        "latency_mode": args.latency_mode,
        "value_hit_rate": (
            sum(record["value_hit"] for record in valid) / len(valid) if valid else 0.0
        ),
        "evidence_file_hit_rate": (
            sum(record["evidence_file_hit"] for record in valid) / len(valid)
            if valid
            else 0.0
        ),
        "evidence_page_hit_rate": (
            sum(record["evidence_page_hit"] for record in valid) / len(valid)
            if valid
            else 0.0
        ),
        "under_10s_rate": (
            sum(record["under_10s"] for record in valid) / len(valid) if valid else 0.0
        ),
        "mean_seconds": (
            statistics.fmean(record["seconds"] for record in valid) if valid else 0.0
        ),
        "median_seconds": (
            statistics.median(record["seconds"] for record in valid) if valid else 0.0
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
