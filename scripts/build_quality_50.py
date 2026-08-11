#!/usr/bin/env python3
"""Build data/eval/quality_50_open_mix.jsonl (open-table heavy + route mix)."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def needle_from_gold(ans: str) -> list[str]:
    a = str(ans or "").strip()
    if not a:
        return []
    if re.match(r"^(SP|UP)-[A-Z]$", a, re.I):
        return [re.escape(a), a.replace("-", r"[- ]?")]
    if re.match(r"^L\d$", a, re.I):
        return [re.escape(a)]
    if re.match(r"^S\+D$", a, re.I):
        return [r"S\+D|S\s*\+\s*D"]
    if re.fullmatch(r"\d+(?:\.\d+)?", a):
        return [re.escape(a)]
    if "○" in a or a in {"O", "○"}:
        return [r"○|O|대상|reporting"]
    toks = re.findall(r"[A-Za-z0-9가-힣.+\-/]{2,}", a)
    needles: list[str] = []
    for t in toks:
        if re.search(r"\d|[A-Z]{2,}", t):
            needles.append(re.escape(t))
    if not needles and toks:
        needles = [re.escape(t) for t in toks[:3]]
    if len(a) <= 40:
        needles.insert(0, re.escape(a[:30]))
    out: list[str] = []
    seen: set[str] = set()
    for n in needles:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out[:4]


def main() -> int:
    rows: list[dict] = []
    cur = ROOT / "data" / "eval" / "table_questions_22docs_practical_v1_curated.jsonl"
    for line in cur.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        q = o["question"]
        if re.search(r"\.pdf|페이지", q):
            continue
        needles = needle_from_gold(o.get("gold_answer") or "")
        if not needles:
            continue
        rows.append(
            {
                "id": f"open_{o['qid']}",
                "type": "table_open",
                "route": "rag",
                "question": q,
                "gold": o.get("gold_answer"),
                "needles": needles,
                "gold_file": o.get("gold_file_name"),
                "gold_page": o.get("gold_page"),
            }
        )

    q30 = ROOT / "data" / "eval" / "quality_30_types.jsonl"
    others: list[dict] = []
    seen_q = {r["question"] for r in rows}
    for line in q30.read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        if o["question"] in seen_q:
            continue
        others.append(
            {
                "id": o["id"],
                "type": o.get("type") or "other",
                "route": o.get("route") or "rag",
                "question": o["question"],
                "gold": o.get("gold"),
                "needles": o.get("needles") or [],
            }
        )
        seen_q.add(o["question"])

    final: list[dict] = rows[:32]
    want_ids = [
        "table_01",
        "table_02",
        "table_03",
        "table_04",
        "table_11",
        "chat_01",
        "ops_01",
        "ops_02",
        "meet_01",
        "def_01",
        "rule_01",
        "hyb_01",
        "oos_01",
        "ops_04",
        "meet_03",
        "def_02",
        "chat_02",
        "ops_05",
    ]
    by_id = {r["id"]: r for r in others}
    for wid in want_ids:
        r = by_id.get(wid)
        if not r:
            continue
        if r["question"] in {x["question"] for x in final}:
            continue
        final.append(r)
        if len(final) >= 50:
            break
    for r in others:
        if len(final) >= 50:
            break
        if r["question"] not in {x["question"] for x in final}:
            final.append(r)

    final = final[:50]
    out = ROOT / "data" / "eval" / "quality_50_open_mix.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    open_n = sum(1 for r in final if r["type"] == "table_open" or str(r["type"]).startswith("table"))
    print(f"wrote {out} n={len(final)} tableish={open_n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
