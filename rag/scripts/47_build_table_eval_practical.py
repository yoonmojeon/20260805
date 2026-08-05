"""Build a page-blind, practical table-QA draft for the 22 KR documents."""
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

_legacy_builder = importlib.import_module("41_build_table_eval_22")
cell_candidates = _legacy_builder.cell_candidates
compact = _legacy_builder.compact
load_doc_tables = _legacy_builder.load_doc_tables


DEFAULT_MANIFEST = ROOT / "data/processed/index/unified_kr_tables_v2/index_manifest.json"
DEFAULT_CHUNKS = ROOT / "data/processed/chunks_v2"
DEFAULT_OUT = ROOT / "data/eval/table_questions_22docs_practical_v1_draft.jsonl"
DEFAULT_REVIEW = ROOT / "data/eval/table_questions_22docs_practical_v1_review.md"

CAPTION_PREFIX_RE = re.compile(r"^(?:표|table)\s*[0-9A-Za-z.()가-힣-]+\s*", re.I)
YEAR_SUFFIX_RE = re.compile(r"[_ ](?:20\d{2})$")
PAGE_LEAK_RE = re.compile(r"\d+\s*페이지|\bp\.?\s*\d+\b", re.I)
TITLE_LOOKUP_RE = re.compile(r"표(?:의)?\s*제목|주요\s*열|열\s*하나")


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def topic_from(candidate: dict) -> str:
    caption = compact(candidate.get("caption"))
    topic = CAPTION_PREFIX_RE.sub("", caption).strip(" -–—:")
    topic = re.sub(r"\s*\((?:계속|20\d{2})\)\s*$", "", topic, flags=re.I).strip()
    section = compact(candidate.get("section_title"))
    section = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", section).strip()
    if len(topic) < 4 or topic in {"화학성분", "기계적 성질", "종류", "요건"}:
        topic = f"{section} {topic}".strip()
    return topic or "해당 규정"


def document_label(file_name: str) -> str:
    stem = Path(file_name).stem
    stem = YEAR_SUFFIX_RE.sub("", stem).replace("_", " ").strip()
    replacements = {
        "Circular (K) Total": "KR 기술회보",
    }
    return replacements.get(stem, stem)


def practical_score(candidate: dict) -> float:
    row = compact(candidate.get("row_key"))
    column = compact(candidate.get("column"))
    answer = compact(candidate.get("answer"))
    topic = topic_from(candidate)
    score = float(candidate.get("score") or 0)
    score += 0.7 if 3 <= len(row) <= 45 else -0.6
    score += 0.4 if 2 <= len(column) <= 35 else -0.4
    score += 0.3 if 1 <= len(answer) <= 35 else -0.3
    score += 0.4 if 4 <= len(topic) <= 55 else -0.4
    score -= 0.4 * row.count(" / ")
    score -= 0.5 if "세부항목" in column else 0.0
    score -= 2.0 if "[수식기호]" in f"{row} {column} {answer} {topic}" else 0.0
    score -= 0.7 if re.search(r"_\d+$", column) else 0.0
    return score


def enrich_candidates(doc_id: str, chunks_path: Path) -> list[dict]:
    schemas, rows = load_doc_tables(chunks_path)
    candidates = cell_candidates(doc_id, schemas, rows)
    for candidate in candidates:
        schema = schemas.get(candidate["table_id"]) or {}
        record = schema.get("_record") or {}
        candidate["section_title"] = compact(schema.get("section_title") or record.get("section_title"))
        candidate["practical_score"] = practical_score(candidate)
    return sorted(candidates, key=lambda c: (-c["practical_score"], c["page"], c["table_id"]))


def clean_candidate(candidate: dict) -> bool:
    row = compact(candidate.get("row_key"))
    column = compact(candidate.get("column"))
    answer = compact(candidate.get("answer"))
    topic = topic_from(candidate)
    combined = f"{row} {column} {answer} {topic}"
    return (
        2 <= len(row) <= 65
        and row.count(" / ") <= 1
        and 2 <= len(column) <= 45
        and "세부항목" not in column
        and "[수식기호]" not in combined
        and not re.search(r"_\d+$", column)
        and 1 <= len(answer) <= 60
        and 4 <= len(topic) <= 70
    )


def choose_singles(candidates: list[dict], count: int = 2) -> list[dict]:
    selected: list[dict] = []
    used_tables: set[str] = set()
    used_pages: set[int] = set()
    for candidate in candidates:
        if not clean_candidate(candidate):
            continue
        if candidate["table_id"] in used_tables or candidate["page"] in used_pages:
            continue
        selected.append(candidate)
        used_tables.add(candidate["table_id"])
        used_pages.add(candidate["page"])
        if len(selected) == count:
            break
    return selected


def choose_comparison(candidates: list[dict], excluded_tables: set[str]) -> tuple[dict, dict] | None:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for candidate in candidates:
        if clean_candidate(candidate):
            grouped[(candidate["table_id"], candidate["column"])].append(candidate)
    pairs: list[tuple[float, dict, dict]] = []
    for (table_id, _), values in grouped.items():
        unique: list[dict] = []
        seen_rows: set[str] = set()
        for value in values:
            if value["row_key"] not in seen_rows:
                unique.append(value)
                seen_rows.add(value["row_key"])
        for left_index, left in enumerate(unique):
            for right in unique[left_index + 1 :]:
                if left["answer"] == right["answer"]:
                    continue
                bonus = 0.5 if table_id not in excluded_tables else 0.0
                pairs.append((left["practical_score"] + right["practical_score"] + bonus, left, right))
                break
            if pairs and pairs[-1][1] is left:
                break
    if not pairs:
        return None
    _, left, right = max(pairs, key=lambda item: item[0])
    return left, right


def gold_cell(candidate: dict) -> dict:
    return {
        "doc_id": candidate["doc_id"],
        "file_name": candidate["file_name"],
        "page": candidate["page"],
        "table_id": candidate["table_id"],
        "row_key": candidate["row_key"],
        "row_column": candidate["row_column"],
        "column": candidate["column"],
        "answer": candidate["answer"],
        "row_chunk_id": candidate["row_chunk_id"],
    }


def base_row(qid: str, question: str, qtype: str, scope: str, candidate: dict, cells: list[dict]) -> dict:
    return {
        "qid": qid,
        "question": question,
        "question_type": qtype,
        "eval_scope": scope,
        "gold_doc_id": candidate["doc_id"],
        "gold_file_name": candidate["file_name"],
        "gold_page": candidate["page"],
        "gold_table_id": candidate["table_id"],
        "gold_row_key": candidate["row_key"],
        "gold_row_column": candidate["row_column"],
        "gold_column": candidate["column"],
        "gold_answer": candidate["answer"],
        "gold_row_chunk_id": candidate["row_chunk_id"],
        "gold_cells": cells,
        "table_caption": candidate["caption"],
        "section_title": candidate.get("section_title") or "",
        "table_topics": candidate["topics"],
        "generator_version": "table_eval_practical_v1_draft",
        "human_verified": False,
    }


def build_questions(manifest: dict, chunks_dir: Path) -> tuple[list[dict], list[dict]]:
    questions: list[dict] = []
    audit: list[dict] = []
    qnum = 1
    for doc_id in manifest.get("doc_ids") or []:
        path = chunks_dir / doc_id / "table_chunks.jsonl"
        candidates = enrich_candidates(doc_id, path)
        singles = choose_singles(candidates)
        comparison = choose_comparison(candidates, {c["table_id"] for c in singles})
        if len(singles) < 2 or comparison is None:
            audit.append({"doc_id": doc_id, "error": "insufficient practical candidates", "candidates": len(candidates)})
            continue

        first, second = singles
        topic = topic_from(first)
        question = f"KR {topic} 기준에서 {first['row_key']}에 해당하는 {first['column']} 값을 알려줘."
        questions.append(base_row(f"TP22_{qnum:03d}", question, "direct_cell_lookup", "open_corpus", first, [gold_cell(first)]))
        qnum += 1

        label = document_label(second["file_name"])
        question = f"KR {label} 기준에서 {second['row_key']}에 해당하는 {second['column']} 값을 알려줘."
        questions.append(base_row(f"TP22_{qnum:03d}", question, "document_context_lookup", "document_context", second, [gold_cell(second)]))
        qnum += 1

        left, right = comparison
        topic = topic_from(left)
        question = f"KR {topic} 기준에서 {left['row_key']}, {right['row_key']} 두 항목의 {left['column']} 값을 비교해줘."
        row = base_row(
            f"TP22_{qnum:03d}",
            question,
            "two_row_comparison",
            "open_corpus",
            left,
            [gold_cell(left), gold_cell(right)],
        )
        row["gold_answer"] = f"{left['row_key']}: {left['answer']}; {right['row_key']}: {right['answer']}"
        questions.append(row)
        qnum += 1
        audit.append(
            {
                "doc_id": doc_id,
                "candidate_count": len(candidates),
                "selected_tables": [first["table_id"], second["table_id"], left["table_id"]],
            }
        )
    return questions, audit


def leakage_summary(rows: list[dict]) -> dict:
    return {
        "questions": len(rows),
        "page_hint": sum(bool(PAGE_LEAK_RE.search(row["question"])) for row in rows),
        "pdf_file_name": sum(".pdf" in row["question"].lower() for row in rows),
        "title_or_column_trivia": sum(bool(TITLE_LOOKUP_RE.search(row["question"])) for row in rows),
        "human_verified": sum(bool(row.get("human_verified")) for row in rows),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_review(path: Path, rows: list[dict], audit: list[dict]) -> None:
    leak = leakage_summary(rows)
    lines = [
        "# KR tables practical QA v1 — human review draft",
        "",
        "> 이 파일은 자동 생성 초안이다. PDF와 대조해 질문의 실무성·정답·인용 위치를 사람이 승인하기 전에는 운영 성능 수치로 사용하지 않는다.",
        "",
        f"- 문항: {len(rows)} (문서당 직접 조회 1 + 문서 맥락 조회 1 + 두 행 비교 1)",
        f"- 페이지 힌트: {leak['page_hint']}",
        f"- 정확한 PDF 파일명: {leak['pdf_file_name']}",
        f"- 표 제목/열 이름 맞히기: {leak['title_or_column_trivia']}",
        f"- 사람 검수 완료: {leak['human_verified']}",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['qid']} — {row['question_type']}",
                "",
                f"- [ ] 질문이 실제 업무에서 자연스럽다",
                f"- [ ] PDF 원문과 정답·단위가 일치한다",
                f"- [ ] 질문만으로 답이 하나로 특정된다",
                f"- 질문: {row['question']}",
                f"- 숨은 근거: `{row['gold_file_name']}` p.{row['gold_page']} / `{row['gold_table_id']}`",
                f"- 모범답안: {row['gold_answer']}",
                "- 근거 셀:",
            ]
        )
        for cell in row["gold_cells"]:
            lines.append(f"  - {cell['row_key']} / {cell['column']} = {cell['answer']}")
        lines.append("")
    errors = [item for item in audit if item.get("error")]
    if errors:
        lines.extend(["## 생성 오류", "", "```json", json.dumps(errors, ensure_ascii=False, indent=2), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()
    rows, audit = build_questions(read_manifest(args.manifest), args.chunks_dir)
    write_jsonl(args.out, rows)
    write_review(args.review, rows, audit)
    print(json.dumps(leakage_summary(rows), ensure_ascii=False, indent=2))
    print(f"covered_docs={len({row['gold_doc_id'] for row in rows})}")
    print(f"wrote {args.out}")
    print(f"wrote {args.review}")


if __name__ == "__main__":
    main()
