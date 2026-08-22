"""Separate chunking failures from retrieval failures, then from answer failures.

For table questions the evaluation set records the exact chunk that holds the
answer, so three checks can be made independently:

  chunk_present  - the gold chunk exists in the chunk store and still contains
                   the gold value (otherwise chunking/extraction lost it)
  retrieved_rank - the gold chunk came back from search (otherwise retrieval
                   ranks it out even though the chunk exists)
  answer_hit     - the generated answer carries the gold value

A question that fails only the third check is a generation problem, not a
retrieval one.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import RAG_CHUNKS_DIR, RAG_TABLE_CHUNKS_DIR  # noqa: E402

TABLE_SET = ROOT / "data" / "eval" / "table_questions_22docs_practical_v1_curated.jsonl"
TEXT_SET = ROOT / "data" / "eval" / "pilot_validation_text_v3.jsonl"
OUT_ROOT = ROOT / "data" / "processed" / "logs" / "rag_diagnosis"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def norm(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def load_doc_chunks(doc_id: str, *, table_side: bool) -> list[dict]:
    root = RAG_TABLE_CHUNKS_DIR if table_side else RAG_CHUNKS_DIR
    out: list[dict] = []
    for name in ("table_chunks.jsonl", "chunks.jsonl"):
        path = root / doc_id / name
        if path.exists():
            out.extend(read_jsonl(path))
    return out


def chunk_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return str(chunk.get("text") or "")
    return str(getattr(chunk, "text", "") or "")


def chunk_id_of(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return str(chunk.get("chunk_id") or "")
    return str(getattr(chunk, "chunk_id", "") or "")


def diagnose_table_row(row: dict, retrieved: list[Any]) -> dict:
    gold = str(row.get("gold_answer") or "")
    gold_chunk_id = str(row.get("gold_row_chunk_id") or "")
    gold_table_id = str(row.get("gold_table_id") or "")
    doc_id = str(row.get("gold_doc_id") or "")

    stored = load_doc_chunks(doc_id, table_side=True)
    by_id = {str(c.get("chunk_id") or ""): c for c in stored}
    gold_chunk = by_id.get(gold_chunk_id)
    same_table = [c for c in stored if gold_table_id and gold_table_id in str(c.get("chunk_id") or "")]

    chunk_present = gold_chunk is not None
    gold_in_chunk = bool(gold_chunk and norm(gold) in norm(chunk_text(gold_chunk)))
    gold_in_table = any(norm(gold) in norm(chunk_text(c)) for c in same_table)
    # The evaluation set records row-level chunk ids from an earlier chunking
    # run, so a missing id proves nothing by itself; what matters is whether the
    # value survived anywhere in this document's chunks.
    value_in_doc = any(norm(gold) in norm(chunk_text(c)) for c in stored)
    value_in_doc_text = any(
        norm(gold) in norm(chunk_text(c)) for c in load_doc_chunks(doc_id, table_side=False)
    )

    ranks = [
        idx
        for idx, c in enumerate(retrieved, start=1)
        if gold_table_id and gold_table_id in chunk_id_of(c)
    ]
    exact_rank = next(
        (idx for idx, c in enumerate(retrieved, start=1) if chunk_id_of(c) == gold_chunk_id),
        None,
    )
    value_rank = next(
        (idx for idx, c in enumerate(retrieved, start=1) if norm(gold) in norm(chunk_text(c))),
        None,
    )

    if value_rank is not None:
        verdict = "검색 정상"
    elif not value_in_doc and not value_in_doc_text:
        verdict = "청킹: 정답 값이 청크에 없음"
    elif not value_in_doc and value_in_doc_text:
        verdict = "청킹: 표 청크 누락 (본문 청크에만 존재)"
    elif ranks:
        verdict = "검색: 같은 표는 회수했으나 정답 행 누락"
    else:
        verdict = "검색: 정답 청크 미회수"

    return {
        "chunk_present": chunk_present,
        "gold_in_chunk": gold_in_chunk,
        "gold_in_same_table": gold_in_table,
        "value_in_doc_table_chunks": value_in_doc,
        "value_in_doc_text_chunks": value_in_doc_text,
        "stored_chunks_in_doc": len(stored),
        "same_table_chunks": len(same_table),
        "gold_table_rank": ranks[0] if ranks else None,
        "gold_chunk_rank": exact_rank,
        "gold_value_rank": value_rank,
        "n_retrieved": len(retrieved),
        "retrieved_preview": [
            {
                "rank": idx,
                "chunk_id": chunk_id_of(c),
                "has_gold": norm(gold) in norm(chunk_text(c)),
                "head": re.sub(r"\s+", " ", chunk_text(c))[:120],
            }
            for idx, c in enumerate(retrieved[:8], start=1)
        ],
        "verdict": verdict,
    }


def run_retrieval(question: str, *, table_side: bool) -> list[Any]:
    import services.rag_service as rs

    out = rs._run_search_only(question, latency_mode="fast", table_side=table_side)
    return list(out.get("retrieved") or [])


def phase_a(limit: int, out_dir: Path, qids: list[str] | None = None) -> list[dict]:
    """Retrieval-only pass over table questions (no LLM)."""
    rows = read_jsonl(TABLE_SET)
    rows = [r for r in rows if str(r.get("qid")) in set(qids)] if qids else rows[:limit]
    results: list[dict] = []
    for idx, row in enumerate(rows, start=1):
        question = str(row.get("question") or "")
        started = time.time()
        retrieved = run_retrieval(question, table_side=True)
        diag = diagnose_table_row(row, retrieved)
        diag.update(
            {
                "qid": row.get("qid"),
                "question": question,
                "gold_answer": row.get("gold_answer"),
                "gold_file_name": row.get("gold_file_name"),
                "gold_page": row.get("gold_page"),
                "search_seconds": round(time.time() - started, 1),
            }
        )
        results.append(diag)
        print(
            f"[A {idx}/{len(rows)}] {diag['qid']} · {diag['verdict']} · "
            f"표 rank {diag['gold_table_rank']} · 값 rank {diag['gold_value_rank']} · "
            f"{diag['search_seconds']}s",
            flush=True,
        )
    (out_dir / "phase_a_table_retrieval.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results


def phase_b(questions: list[dict], out_dir: Path, model: str) -> list[dict]:
    """Full answers through the same stack the UI uses."""
    from services.orchestrator import handle_question

    results: list[dict] = []
    for idx, item in enumerate(questions, start=1):
        question = item["question"]
        started = time.time()
        result = handle_question(
            question, [], use_llm_router=True, rag_latency_mode="fast", llm_model=model
        )
        elapsed = round(time.time() - started, 1)
        answer = str((result or {}).get("answer") or "")
        meta = (result or {}).get("meta") or {}
        needle = str(item.get("needle") or "")
        hit = bool(needle) and norm(needle) in norm(answer)
        record = {
            "label": item.get("label", ""),
            "question": question,
            "needle": needle,
            "answer_hit": hit,
            "route": ((result or {}).get("route") or {}).get("route"),
            "answer_mode": meta.get("answer_mode"),
            "latency_promoted": meta.get("latency_mode_promoted"),
            "seconds": elapsed,
            "answer": answer,
        }
        results.append(record)
        print(
            f"[B {idx}/{len(questions)}] {record['label']} · {elapsed}s · "
            f"needle {'HIT' if hit else ('MISS' if needle else '-')} · {record['answer_mode']}",
            flush=True,
        )
    (out_dir / "phase_b_answers.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = ["# 답변 확인", ""]
    for rec in results:
        report += [
            f"## {rec['label']} · {rec['seconds']}s · {rec['answer_mode']} · "
            f"needle {'HIT' if rec['answer_hit'] else ('MISS' if rec['needle'] else '-')}",
            f"Q: {rec['question']}",
            f"기대 값: {rec['needle'] or '-'}",
            "",
            rec["answer"],
            "",
        ]
    (out_dir / "phase_b_report.md").write_text("\n".join(report), encoding="utf-8")
    return results


def build_phase_b_questions() -> list[dict]:
    """Demo questions, UI examples, and known-weak asks across every category."""
    table_rows = {r["qid"]: r for r in read_jsonl(TABLE_SET)}
    # First four are the demo picks (regression guard); the rest are the
    # retrieval failures found in phase A, to see how the answer behaves when
    # the gold row never reaches the context.
    picks = [
        "TC22_013",
        "TC22_001",
        "TC22_002",
        "TC22_004",
        "TC22_005",
        "TC22_011",
        "TC22_019",
    ]
    items = [
        {
            "label": f"표 {qid}",
            "question": table_rows[qid]["question"],
            "needle": table_rows[qid]["gold_answer"],
        }
        for qid in picks
        if qid in table_rows
    ]
    items += [
        {"label": "UI 예시 부식 정의", "question": "과도한 부식의 정의는 무엇인가?", "needle": ""},
        {"label": "UI 예시 tcorr", "question": "구조 규칙에서 쓰는 tcorr 기호는 어떤 두께를 뜻하지?", "needle": ""},
        {"label": "문서 MEPC 동향", "question": "환경규제 대응과 관련된 최신 MEPC 회의 주요 내용을 정리해줘.", "needle": "SFCS"},
        {"label": "문서 수소 전제", "question": "MSC 111-WP.1 — MSC 111 본회의 보고서 초안을 기준으로 '수소 연료 잠정 안전지침은 부결됐다'라는 전제가 맞는지 검증하고, 틀리면 문서 근거로 바로잡아줘.", "needle": "승인"},
        {"label": "문서 DNV CQ", "question": "DNV-CG-0264 — Autonomous and remotely operated vessels를 기준으로, Concept Qualification은 어떤 두 목적에 쓰여?", "needle": ""},
        {"label": "문서 LR 대체연료", "question": "LR에서 대체연료 관련 Rule/Guidance를 찾아줘.", "needle": "Notice No.1"},
        {"label": "문서 ABS Smart 목적", "question": "ABS Guide for Smart Functions for Marine Vessels and Offshore Units v8를 기준으로, ABS Smart Functions Guide가 제시하는 네 가지 목적을 정리해줘.", "needle": ""},
        {"label": "문서 MSC 111 요약", "question": "MSC 111의 주요 결과를 3개 항목으로 요약해줘.", "needle": "MASS"},
        {"label": "문서 MASS 일정", "question": "MSC 111에서 MASS Code와 관련된 핵심 결정사항을 요약하고, 향후 mandatory code 일정까지 정리해줘.", "needle": "2032"},
        {"label": "문서 DCS 데이터오류", "question": "최신 MEPC 회의에서 IMO DCS 제출 데이터의 오류나 누락은 어떻게 다뤘어?", "needle": ""},
    ]
    return items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["a", "b", "ab"], default="ab")
    parser.add_argument("--table-limit", type=int, default=20)
    parser.add_argument("--qids", help="쉼표로 구분한 표 질문 qid (예: TC22_005,TC22_011)")
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    from services.llm_models import DEFAULT_LLM_MODEL

    model = args.model or DEFAULT_LLM_MODEL
    out_dir = OUT_ROOT / datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase in {"a", "ab"}:
        qids = [q.strip() for q in (args.qids or "").split(",") if q.strip()] or None
        rows = phase_a(args.table_limit, out_dir, qids)
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
        print("\n[A 요약]", json.dumps(counts, ensure_ascii=False), flush=True)

    if args.phase in {"b", "ab"}:
        phase_b(build_phase_b_questions(), out_dir, model)

    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
