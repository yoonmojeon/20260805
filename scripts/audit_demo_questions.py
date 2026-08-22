"""Audit a curated DOCX question list through the same RAG entry point as the UI.

The DOCX contains questions only.  Gold answers/evidence are resolved from the
reviewed broad-PDF benchmark by normalized question text, then each question is
sent through ``services.orchestrator.handle_question``.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.orchestrator import handle_question
from services.rag_service import warmup_rag_resources

WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

TABLE_QID_BY_POSITION = {
    41: "TC22_001",
    42: "TC22_004",
    43: "TC22_007",
    44: "TC22_013",
    45: "TC22_016",
    46: "TC22_019",
    47: "TC22_025",
    48: "TC22_034",
    49: "TC22_052",
    50: "TC22_064",
}

TABLE_ANSWER_POINT_ALIASES = {
    "TC22_001": [["1.14"]],
    "TC22_004": [["기관실 전방 또는 후방의 횡격벽", "기관실 전방", "기관실 후방"]],
    "TC22_007": [["좌굴해석을 고려하는 요소 판 패널", "좌굴해석을 고려"]],
    "TC22_013": [["250"]],
    "TC22_016": [["7편 6장", "제7편 제6장", "7 편 6 장"]],
    "TC22_019": [["Y(2)", "Y (2)"]],
    "TC22_025": [["1.0"]],
    "TC22_034": [["1.2 m", "1.2m"]],
    "TC22_052": [["1대", "1 대"]],
    "TC22_064": [["DIN 4102"]],
}

MANUAL_POINT_ALIASES_BY_POSITION = {
    4: [["구조적 덕트", "교차 침수"], ["공기 파이프", "압력 손실"]],
    11: [["특별하거나 새로운 설계", "특수하거나 새로운 설계"]],
    25: [["필요한 시험을 결정", "시험을 결정해야"]],
    31: [["해양 활동", "해상 활동"], ["끌려가는 닻", "끌린 닻"]],
    36: [["최소 2일", "최소 2 일"], ["Palais des Nations", "제네바", "온라인"]],
    39: [["화물창에는 빌지 수위 모니터링"], ["펌프룸", "섹션 5"]],
}

# The same BC COP-17 deadline is reproduced verbatim in both MEPC 84-INF.5
# and MEPC 84-INF.7.  Either official meeting document is valid evidence.
MANUAL_ACCEPTABLE_EVIDENCE_BY_POSITION = {
    21: {
        "doc_ids": [
            "mepc_mepc_mepc_84_inf_5_outcome_of_the_seventeenth_meeting_of_the_conference_of_the_parties_to_the_basel_conventio_secretariat_c6317e25"
        ],
        "pages": [1, 2, 3],
    }
}


def normalize(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣%]+", "", (text or "").lower())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_docx_questions(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    questions: list[str] = []
    for paragraph in root.findall(".//w:body/w:p", WORD_NS):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", WORD_NS)).strip()
        if (
            not text
            or text == "질문리스트"
            or text.startswith("텍스트·규정·회의 질문")
            or text.startswith("표 기반 질문")
        ):
            continue
        text = re.sub(r"^\s*\d+\s*[.)]\s*", "", text).strip()
        if text:
            questions.append(text)
    if len(questions) != 50:
        raise RuntimeError(f"Expected 50 questions in {path}, found {len(questions)}")
    return questions


def resolve_rows(
    questions: list[str],
    gold_rows: list[dict[str, Any]],
    table_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    by_question = {normalize(str(row.get("question") or "")): row for row in gold_rows}
    table_by_qid = {
        str(row.get("qid") or ""): row for row in (table_rows or [])
    }
    resolved: list[dict[str, Any]] = []
    missing: list[str] = []
    for position, question in enumerate(questions, 1):
        table_qid = TABLE_QID_BY_POSITION.get(position)
        row = table_by_qid.get(table_qid) if table_qid else by_question.get(normalize(question))
        if row is None:
            missing.append(f"{position}. {question}")
            continue
        resolved_row = {**row, "question": question, "demo_position": position}
        manual_points = MANUAL_POINT_ALIASES_BY_POSITION.get(position)
        if manual_points:
            resolved_row["gold_answer_points"] = [
                {
                    "point_id": f"MANUAL-{position}-{index}",
                    "text": aliases[0],
                    "aliases": aliases,
                }
                for index, aliases in enumerate(manual_points, 1)
            ]
        acceptable_evidence = MANUAL_ACCEPTABLE_EVIDENCE_BY_POSITION.get(position)
        if acceptable_evidence:
            resolved_row["acceptable_doc_ids"] = list(
                dict.fromkeys(
                    [
                        *list(resolved_row.get("acceptable_doc_ids") or []),
                        *list(acceptable_evidence["doc_ids"]),
                    ]
                )
            )
            resolved_row["gold_pages"] = sorted(
                {
                    *[int(value) for value in resolved_row.get("gold_pages") or []],
                    *[int(value) for value in acceptable_evidence["pages"]],
                }
            )
        if table_qid:
            point_aliases = TABLE_ANSWER_POINT_ALIASES[table_qid]
            resolved_row["gold_answer_points"] = [
                {
                    "point_id": f"{table_qid}_answer_{index}",
                    "text": aliases[0],
                    "aliases": aliases,
                }
                for index, aliases in enumerate(point_aliases, 1)
            ]
            resolved_row["gold_pages"] = [resolved_row.get("gold_page")]
        resolved.append(resolved_row)
    if missing:
        raise RuntimeError("Gold benchmark match failed:\n" + "\n".join(missing))
    return resolved


def _aliases_for_point(point: dict[str, Any]) -> list[str]:
    aliases = [str(item) for item in point.get("aliases") or [] if str(item).strip()]
    text = str(point.get("text") or "").strip()
    if text:
        aliases.append(text)
    return aliases


def score_answer(row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    answer = str(result.get("answer") or "")
    answer_norm = normalize(answer)
    points = list(row.get("gold_answer_points") or [])
    point_hits: list[str] = []
    point_misses: list[str] = []
    for point in points:
        aliases = _aliases_for_point(point)
        hit = any(normalize(alias) and normalize(alias) in answer_norm for alias in aliases)
        target = point_hits if hit else point_misses
        target.append(str(point.get("point_id") or point.get("text") or ""))

    evidence = list(result.get("evidence_table") or [])
    gold_docs = {
        str(value)
        for value in (
            list(row.get("gold_doc_ids") or [])
            + list(row.get("acceptable_doc_ids") or [])
            + [row.get("gold_doc_id")]
        )
        if str(value or "").strip()
    }
    gold_file = str(row.get("gold_file_name") or "").strip().lower()
    gold_pages = {int(value) for value in row.get("gold_pages") or [] if value is not None}
    source_hit = False
    page_hit = False
    for item in evidence:
        file_name = str(item.get("file_name") or "").strip().lower()
        chunk_id = str(item.get("chunk_id") or "")
        doc_hit = bool(gold_docs and any(chunk_id.startswith(doc_id) for doc_id in gold_docs))
        file_hit = bool(gold_file and file_name == gold_file)
        if doc_hit or file_hit:
            source_hit = True
            page = item.get("page")
            try:
                page_hit = page_hit or not gold_pages or int(page) in gold_pages
            except (TypeError, ValueError):
                pass

    forbidden = [
        claim
        for claim in row.get("forbidden_claims") or []
        if normalize(str(claim)) and normalize(str(claim)) in answer_norm
    ]
    completeness = len(point_hits) / len(points) if points else 0.0
    citation_ok = bool(re.search(r"\[\d+\]", answer))
    strict_pass = (
        completeness == 1.0
        and source_hit
        and page_hit
        and citation_ok
        and not forbidden
        and not str(result.get("meta", {}).get("answer_error") or "")
    )
    return {
        "strict_pass": strict_pass,
        "completeness": completeness,
        "point_hits": point_hits,
        "point_misses": point_misses,
        "gold_source_hit": source_hit,
        "gold_page_hit": page_hit,
        "citation_ok": citation_ok,
        "forbidden_violations": forbidden,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docx", type=Path, required=True)
    parser.add_argument(
        "--gold",
        type=Path,
        default=ROOT / "data/eval/broad_pdf_150_final.jsonl",
    )
    parser.add_argument(
        "--table-gold",
        type=Path,
        default=ROOT / "data/eval/table_questions_22docs_practical_v1_curated.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data/processed/logs/demo_questions_50/live_accurate",
    )
    parser.add_argument("--latency-mode", choices=("fast", "accurate"), default="accurate")
    parser.add_argument("--model", default="gemma4:12b")
    parser.add_argument("--position", type=int, action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args()

    rows = resolve_rows(
        read_docx_questions(args.docx),
        load_jsonl(args.gold),
        load_jsonl(args.table_gold),
    )
    if args.position:
        selected = set(args.position)
        rows = [row for row in rows if int(row["demo_position"]) in selected]
    if args.limit is not None:
        rows = rows[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.out_dir / "records.partial.jsonl"
    partial_path.write_text("", encoding="utf-8")
    if not args.skip_warmup:
        warmup_rag_resources()

    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        started = time.perf_counter()
        result = handle_question(
            str(row["question"]),
            history=[],
            force_route="auto",
            use_llm_router=True,
            rag_latency_mode=args.latency_mode,
            llm_model=args.model,
        )
        elapsed = time.perf_counter() - started
        score = score_answer(row, result)
        record = {
            "demo_position": row["demo_position"],
            "question_id": row.get("question_id"),
            "qid": row.get("qid"),
            "question": row["question"],
            "gold_answer": row.get("gold_answer"),
            "gold_file_name": row.get("gold_file_name"),
            "gold_pages": row.get("gold_pages"),
            "answer": result.get("answer"),
            "evidence_table": result.get("evidence_table") or [],
            "related_table_count": len(result.get("related_tables") or []),
            "route": (result.get("route") or {}).get("route"),
            "retrieval_mode": (result.get("meta") or {}).get("retrieval_mode"),
            "category": (result.get("meta") or {}).get("category"),
            "answer_mode": (result.get("meta") or {}).get("answer_mode"),
            "llm_model": (result.get("meta") or {}).get("llm_model"),
            "answer_model": (result.get("meta") or {}).get("answer_model"),
            "latency_mode": (result.get("meta") or {}).get("latency_mode"),
            "answer_source": (result.get("meta") or {}).get("answer_source"),
            "answer_generation": (result.get("meta") or {}).get("answer_generation"),
            "elapsed_seconds": round(elapsed, 3),
            **score,
        }
        records.append(record)
        with partial_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[{index}/{len(rows)}] #{row['demo_position']} "
            f"{'PASS' if score['strict_pass'] else 'FAIL'} "
            f"mode={record['retrieval_mode']} t={elapsed:.1f}s",
            flush=True,
        )

    with (args.out_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "n": len(records),
        "strict_pass": sum(bool(record["strict_pass"]) for record in records),
        "strict_pass_rate": (
            sum(bool(record["strict_pass"]) for record in records) / len(records)
            if records
            else 0.0
        ),
        "mean_seconds": statistics.fmean(record["elapsed_seconds"] for record in records)
        if records
        else 0.0,
        "median_seconds": statistics.median(record["elapsed_seconds"] for record in records)
        if records
        else 0.0,
        "fail_positions": [record["demo_position"] for record in records if not record["strict_pass"]],
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
