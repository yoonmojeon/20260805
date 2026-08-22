"""How many indexed tables lost their column structure during extraction?

A multi-column PDF table that extracts as a single "content" column turns every
per-row KV chunk into a fragment ("content=잔류응력 측정(해당되는 경우") with the
value stranded in a different row.  Row-level retrieval cannot answer a
row x column lookup against such a table, no matter how good the ranker is.
This separates that chunking defect from a retrieval defect.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNK_ROOT = ROOT / "data" / "processed" / "chunks"

DEGENERATE = {"content", "p", "col1", "열1", ""}


def main() -> None:
    per_table_columns: dict[str, list[str]] = {}
    per_table_doc: dict[str, str] = {}
    row_counts: Counter[str] = Counter()

    for path in sorted(CHUNK_ROOT.rglob("table_chunks.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                chunk_id = str(rec.get("chunk_id") or "")
                text = str(rec.get("text") or "")
                if chunk_id.endswith("__schema"):
                    table_id = chunk_id[: -len("__schema")]
                    for row in text.splitlines():
                        if row.startswith("columns:"):
                            cols = [
                                c.strip() for c in row.split(":", 1)[1].split(",") if c.strip()
                            ]
                            per_table_columns[table_id] = cols
                    meta = rec.get("metadata") or {}
                    per_table_doc[table_id] = str(
                        meta.get("file_name") or rec.get("file_name") or path.parent.name
                    )
                elif "__row_" in chunk_id:
                    row_counts[chunk_id.split("__row_")[0]] += 1

    total = len(per_table_columns)
    degenerate = {
        tid: cols
        for tid, cols in per_table_columns.items()
        if len(cols) <= 1 and (not cols or cols[0].strip().lower() in DEGENERATE)
    }
    wide_degenerate = {tid: c for tid, c in degenerate.items() if row_counts.get(tid, 0) >= 10}

    lines = [
        "# 표 열 구조 손실 진단",
        "",
        f"인덱싱된 표: {total}개",
        f"단일 열('content'/'P' 등)로 추출된 표: {len(degenerate)}개 "
        f"({len(degenerate) / max(1, total):.1%})",
        f"그중 10행 이상(=실제로 다열이었을 표): {len(wide_degenerate)}개 "
        f"({len(wide_degenerate) / max(1, total):.1%})",
        "",
        "## 정상 추출 표의 열 개수 분포",
    ]
    healthy = Counter(
        len(cols) for tid, cols in per_table_columns.items() if tid not in degenerate
    )
    for width, count in sorted(healthy.items()):
        lines.append(f"- {width}열: {count}개")

    lines += ["", "## 손실이 큰 표 예시 (행 수 상위 15개)"]
    for tid, _cols in sorted(
        wide_degenerate.items(), key=lambda kv: row_counts.get(kv[0], 0), reverse=True
    )[:15]:
        lines.append(f"- {tid}  행 {row_counts.get(tid, 0)}개  doc={per_table_doc.get(tid, '')}")

    dest = ROOT / "outputs" / "_table_column_diagnosis.md"
    dest.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
