"""Does the single-column extraction defect predict table QA failures?

Runs a sample of the curated table set end to end and splits the pass rate by
whether the gold table kept its column structure.  A large gap means the
remaining failures come from chunking, not from the ranker, and tells us that
tuning retrieval further would not help those questions.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVAL = ROOT / "data" / "eval" / "table_questions_22docs_practical_v1_curated.jsonl"
CHUNK_ROOT = ROOT / "data" / "processed" / "chunks"
DEGENERATE = {"content", "p", "col1", "열1", ""}


def norm(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def table_widths(table_ids: set[str]) -> dict[str, int]:
    """Column count per table id, read from the indexed schema chunks."""
    widths: dict[str, int] = {}
    docs = {tid.rsplit("_p", 1)[0] for tid in table_ids}
    for path in CHUNK_ROOT.rglob("table_chunks.jsonl"):
        if not any(path.parent.name == doc for doc in docs):
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if "__schema" not in line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                chunk_id = str(rec.get("chunk_id") or "")
                if not chunk_id.endswith("__schema"):
                    continue
                table_id = chunk_id[: -len("__schema")]
                if table_id not in table_ids:
                    continue
                for row in str(rec.get("text") or "").splitlines():
                    if row.startswith("columns:"):
                        cols = [c.strip() for c in row.split(":", 1)[1].split(",") if c.strip()]
                        collapsed = len(cols) <= 1 and (
                            not cols or cols[0].lower() in DEGENERATE
                        )
                        widths[table_id] = 1 if collapsed else len(cols)
    return widths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()

    rows = [json.loads(line) for line in EVAL.read_text(encoding="utf-8").splitlines() if line]
    sample = rows[: args.limit]
    widths = table_widths({str(r.get("gold_table_id") or "") for r in sample})

    from services.rag_service import run_rag_query  # noqa: PLC0415

    results = []
    for index, row in enumerate(sample, 1):
        question = str(row.get("question") or "")
        gold = str(row.get("gold_answer") or "")
        table_id = str(row.get("gold_table_id") or "")
        width = widths.get(table_id, 0)
        start = time.time()
        try:
            res = run_rag_query(question, latency_mode="fast")
            answer = str(res.get("answer") or "")
            mode = str((res.get("meta") or {}).get("answer_mode") or "")
        except Exception as exc:  # keep the batch going
            answer, mode = f"<error {exc}>", "error"
        hit = norm(gold) in norm(answer)
        results.append(
            {
                "qid": row.get("qid"),
                "question": question,
                "gold": gold,
                "hit": hit,
                "collapsed": width <= 1,
                "width": width,
                "mode": mode,
                "seconds": round(time.time() - start, 1),
                "answer": answer,
            }
        )
        print(f"[{index}/{len(sample)}] {row.get('qid')} {'HIT' if hit else 'MISS'} w={width} {mode}")

    collapsed = [r for r in results if r["collapsed"]]
    intact = [r for r in results if not r["collapsed"]]

    def rate(items: list[dict]) -> str:
        if not items:
            return "n/a"
        return f"{sum(1 for i in items if i['hit'])}/{len(items)}"

    lines = [
        "# 표 QA 정확도 vs 열 구조 손실",
        "",
        f"표본 {len(results)}문항",
        f"- 전체 정답: {rate(results)}",
        f"- 열 구조 정상 표: {rate(intact)}",
        f"- 단일열로 붕괴된 표: {rate(collapsed)}",
        "",
        "## 문항별",
    ]
    for r in results:
        flag = "HIT " if r["hit"] else "MISS"
        shape = "붕괴" if r["collapsed"] else f"{r['width']}열"
        lines.append(f"- {flag} [{shape}] {r['qid']} gold={r['gold']!r} mode={r['mode']} {r['seconds']}s")
        lines.append(f"    Q: {r['question']}")

    out_dir = ROOT / "outputs"
    (out_dir / "_table_accuracy.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "_table_accuracy.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("wrote", out_dir / "_table_accuracy.md")


if __name__ == "__main__":
    main()
