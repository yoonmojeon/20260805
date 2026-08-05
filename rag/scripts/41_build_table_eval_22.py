"""Build a balanced, evidence-verifiable table QA set across all kr_tables_v1 docs."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/processed/index/unified_kr_tables_v1/index_manifest.json"
DEFAULT_CHUNKS = ROOT / "data/processed/chunks"
DEFAULT_OUT = ROOT / "data/eval/table_questions_22docs_v1.jsonl"
DEFAULT_REVIEW = ROOT / "data/eval/table_questions_22docs_v1_review.md"

BAD_VALUE_RE = re.compile(r"^(?:nan|none|null|n/?a)?$", re.I)
FORMULA_NOISE_RE = re.compile(r"[\ue000-\uf8ff�]|[∑√∫]{2,}")
TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣○△%./+\-]+")
GENERIC_COLUMN_RE = re.compile(r"^(?:col(?:umn)?[_ ]?\d+|content|value|항목\s*\d*)$", re.I)


def compact(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def has_bad_chars(value: str) -> bool:
    return bool(FORMULA_NOISE_RE.search(value or ""))


def usable_label(value: str, *, max_len: int) -> bool:
    value = compact(value)
    if not value or len(value) > max_len or has_bad_chars(value):
        return False
    if BAD_VALUE_RE.fullmatch(value):
        return False
    return bool(TOKEN_RE.search(value))


def meaningful_column(value: str) -> bool:
    value = compact(value)
    return usable_label(value, max_len=55) and not GENERIC_COLUMN_RE.fullmatch(value)


def meaningful_row_key(value: str) -> bool:
    value = compact(value)
    if not usable_label(value, max_len=55) or value in {"-", "○", "O", "△"}:
        return False
    tokens = re.findall(r"[0-9A-Za-z가-힣]+", value)
    if len(tokens) > 12:
        return False
    if len(value) > 35 and sum(ch.isdigit() for ch in value) / len(value) > 0.35:
        return False
    return True


def usable_answer(value: str) -> bool:
    value = compact(value)
    if not usable_label(value, max_len=70):
        return False
    if len(value) == 1 and value not in {"○", "O", "-", "P", "S", "N", "△"}:
        return False
    return True


def valid_inspection_value(value: str) -> bool:
    value = compact(value)
    return bool(re.fullmatch(r"(?:○|O|-|△|\d+\s*개|절반(?:,)?|전부|모두)", value, re.I))


def load_doc_tables(path: Path) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    schemas: dict[str, dict] = {}
    rows: dict[str, list[dict]] = defaultdict(list)
    seen_rows: set[tuple[str, int, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        table_id = str(record.get("table_id") or "")
        if not table_id:
            continue
        ctype = str(record.get("chunk_type") or "")
        if ctype == "table_schema":
            schema = dict(record.get("schema_json") or {})
            schema["_record"] = record
            prev = schemas.get(table_id)
            if prev is None or float(schema.get("parse_quality") or 0) > float(prev.get("parse_quality") or 0):
                schemas[table_id] = schema
        elif ctype == "table_row" and isinstance(record.get("row_data"), dict):
            key = (table_id, int(record.get("row_index") or 0), json.dumps(record["row_data"], ensure_ascii=False))
            if key not in seen_rows:
                seen_rows.add(key)
                rows[table_id].append(record)
    return schemas, rows


def cell_candidates(doc_id: str, schemas: dict[str, dict], rows: dict[str, list[dict]]) -> list[dict]:
    out: list[dict] = []
    for table_id, table_rows in rows.items():
        schema = schemas.get(table_id) or {}
        parse_quality = float(schema.get("parse_quality") or 0.0)
        if schema and parse_quality < 0.65:
            continue
        schema_record = schema.get("_record") or {}
        caption = compact(schema.get("caption") or schema.get("table_title") or schema_record.get("caption"))
        topics = [compact(v) for v in (schema.get("table_topics") or []) if compact(v)]
        first_value_counts: Counter[str] = Counter()
        for table_row in table_rows:
            values = [(compact(k), compact(v)) for k, v in table_row["row_data"].items() if compact(v)]
            if values:
                first_value_counts[values[0][1]] += 1
        for row in table_rows:
            row_data = [(compact(k), compact(v)) for k, v in row["row_data"].items()]
            nonempty = [(k, v) for k, v in row_data if v]
            if len(nonempty) < 2:
                continue
            base_row_column, base_row_key = nonempty[0]
            if not meaningful_column(base_row_column) or not meaningful_row_key(base_row_key):
                continue
            for target_pos, (column, answer) in enumerate(nonempty[1:], start=1):
                row_column, row_key = base_row_column, base_row_key
                if first_value_counts[base_row_key] > 1:
                    dimensions = nonempty[:target_pos]
                    if len(dimensions) < 2 or not meaningful_row_key(dimensions[1][1]):
                        continue
                    row_column = f"{dimensions[0][0]} / {dimensions[1][0]}"
                    row_key = f"{dimensions[0][1]} / {dimensions[1][1]}"
                    if len(row_key) > 90:
                        continue
                if not meaningful_column(column) or not usable_answer(answer):
                    continue
                if "정기검사" in column:
                    inspection_values = [
                        value
                        for key, value in nonempty[1:]
                        if "정기검사" in key and valid_inspection_value(value)
                    ]
                    if not valid_inspection_value(answer) or len(inspection_values) < 2:
                        continue
                if column == row_column or answer == row_key:
                    continue
                file_name = compact(row.get("file_name") or schema_record.get("file_name"))
                page = int(row.get("page_number") or row.get("page") or 0)
                if not file_name or page <= 0:
                    continue
                density = len(nonempty) / max(1, len(row_data))
                specificity = min(1.0, (len(row_key) + len(column)) / 45)
                answer_bonus = 0.2 if len(answer) <= 20 else 0.0
                clean_caption = bool(caption and caption not in {"표", "table"} and not has_bad_chars(caption))
                caption_bonus = 0.15 if clean_caption else 0.0
                score = parse_quality + density * 0.45 + specificity * 0.25 + answer_bonus + caption_bonus
                out.append(
                    {
                        "kind": "cell_lookup",
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "page": page,
                        "table_id": table_id,
                        "row_index": row.get("row_index"),
                        "row_column": row_column,
                        "row_key": row_key,
                        "column": column,
                        "answer": answer,
                        "caption": caption,
                        "topics": topics,
                        "parse_quality": parse_quality,
                        "row_chunk_id": row.get("chunk_id"),
                        "score": round(score, 4),
                    }
                )
    out.sort(key=lambda x: (-x["score"], x["page"], x["table_id"], str(x["row_index"])))
    return out


def schema_candidates(doc_id: str, schemas: dict[str, dict]) -> list[dict]:
    out: list[dict] = []
    for table_id, schema in schemas.items():
        record = schema.get("_record") or {}
        file_name = compact(record.get("file_name") or schema.get("source_file"))
        page = int(record.get("page_number") or record.get("page") or schema.get("page") or 0)
        if not file_name or page <= 0:
            continue
        caption = compact(schema.get("caption") or schema.get("table_title") or record.get("caption"))
        parse_quality = float(schema.get("parse_quality") or 0.0)
        columns = [compact(v) for v in (schema.get("column_names") or record.get("column_names") or [])]
        meaningful_columns = [v for v in columns if meaningful_column(v)]
        explicit_caption = bool(
            re.match(r"^(?:표|그림|table)\s*\d", caption, re.I)
            or (len(caption) >= 8 and caption not in {"정기검사", "화학성분", "기계적 성질"})
        )
        if (
            caption
            and len(caption) <= 80
            and not has_bad_chars(caption)
            and caption not in {"표", "해당 표"}
            and explicit_caption
        ):
            out.append(
                {
                    "kind": "schema_caption",
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "page": page,
                    "table_id": table_id,
                    "row_key": "",
                    "row_column": "",
                    "column": "caption",
                    "answer": caption,
                    "caption": caption,
                    "topics": [compact(v) for v in (schema.get("table_topics") or []) if compact(v)],
                    "parse_quality": parse_quality,
                    "row_chunk_id": record.get("chunk_id"),
                    "score": round(1.0 + parse_quality + min(0.35, len(meaningful_columns) * 0.07), 4),
                }
            )
        if meaningful_columns:
            answer = meaningful_columns[0]
            out.append(
                {
                    "kind": "schema_column",
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "page": page,
                    "table_id": table_id,
                    "row_key": "",
                    "row_column": "",
                    "column": "columns",
                    "answer": answer,
                    "caption": caption,
                    "topics": [compact(v) for v in (schema.get("table_topics") or []) if compact(v)],
                    "parse_quality": parse_quality,
                    "row_chunk_id": record.get("chunk_id"),
                    "score": round(0.8 + parse_quality + min(0.3, len(meaningful_columns) * 0.06), 4),
                }
            )
    out.sort(key=lambda x: (-x["score"], x["page"], x["table_id"], x["kind"]))
    return out


def choose_balanced(candidates: list[dict], count: int) -> list[dict]:
    selected: list[dict] = []
    used_tables: set[str] = set()
    used_pages: set[int] = set()
    used_pairs: set[tuple[str, str]] = set()

    def take(require_new_table: bool, require_new_page: bool) -> None:
        for candidate in candidates:
            if len(selected) >= count:
                return
            pair = (candidate["row_key"], candidate["column"])
            if pair in used_pairs:
                continue
            if require_new_table and candidate["table_id"] in used_tables:
                continue
            if require_new_page and candidate["page"] in used_pages:
                continue
            selected.append(candidate)
            used_tables.add(candidate["table_id"])
            used_pages.add(candidate["page"])
            used_pairs.add(pair)

    take(True, True)
    take(True, False)
    take(False, True)
    take(False, False)
    return selected[:count]


def choose_mixed(cells: list[dict], schemas: list[dict], count: int) -> list[dict]:
    selected = choose_balanced(cells, min(2, count))
    used_tables = {c["table_id"] for c in selected}
    used_pages = {c["page"] for c in selected}
    for candidate in schemas:
        if len(selected) >= count:
            break
        if candidate["table_id"] in used_tables:
            continue
        selected.append(candidate)
        used_tables.add(candidate["table_id"])
        used_pages.add(candidate["page"])
    for candidate in schemas:
        if len(selected) >= count:
            break
        if any(
            c["table_id"] == candidate["table_id"] and c["kind"] == candidate["kind"]
            for c in selected
        ):
            continue
        selected.append(candidate)
    return selected[:count]


def question_for(candidate: dict, scope: str) -> str:
    file_name = candidate["file_name"]
    page = candidate["page"]
    row = candidate["row_key"]
    column = candidate["column"]
    caption = candidate["caption"] or "해당 표"
    if candidate["kind"] == "schema_caption":
        return f"{file_name} {page}페이지에 있는 구조화 표의 제목은 무엇인가?"
    if candidate["kind"] == "schema_column":
        if not candidate["caption"]:
            return f"{file_name} {page}페이지의 구조화 표에서 주요 열 하나를 알려줘."
        if scope == "anchored":
            return f"{file_name} {page}페이지의 '{caption}' 표에서 주요 열 하나를 알려줘."
        if scope == "document_scoped":
            return f"{file_name}의 '{caption}' 표에 포함된 주요 열 하나는 무엇인가?"
        return f"KR '{caption}' 표에 포함된 주요 열 하나는 무엇인가?"
    if scope == "anchored":
        return f"{file_name} {page}페이지 표에서 '{row}' 행의 '{column}' 값은 무엇인가?"
    if scope == "document_scoped":
        return f"{file_name}의 표에서 '{row}' 항목에 해당하는 '{column}' 값은 무엇인가?"
    return f"KR {caption} 표에서 '{row}' 항목의 '{column}' 값은 무엇인가?"


def build_rows(manifest: dict, chunks_dir: Path, per_doc: int, generator_version: str) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    audit: list[dict] = []
    scopes = ("anchored", "document_scoped", "semantic_open")
    qnum = 1
    for doc_id in manifest.get("doc_ids") or []:
        path = chunks_dir / doc_id / "table_chunks.jsonl"
        if not path.exists():
            audit.append({"doc_id": doc_id, "candidate_count": 0, "selected_count": 0, "error": "missing table_chunks.jsonl"})
            continue
        schemas, table_rows = load_doc_tables(path)
        candidates = cell_candidates(doc_id, schemas, table_rows)
        schema_level = schema_candidates(doc_id, schemas)
        selected = choose_mixed(candidates, schema_level, per_doc)
        audit.append(
            {
                "doc_id": doc_id,
                "schema_count": len(schemas),
                "row_table_count": len(table_rows),
                "candidate_count": len(candidates),
                "schema_candidate_count": len(schema_level),
                "selected_count": len(selected),
                "selected_kinds": [c["kind"] for c in selected],
                "selected_pages": [c["page"] for c in selected],
                "selected_tables": [c["table_id"] for c in selected],
            }
        )
        for i, candidate in enumerate(selected):
            scope = scopes[i % len(scopes)]
            rows.append(
                {
                    "qid": f"TE22_{qnum:03d}",
                    "question": question_for(candidate, scope),
                    "gold_doc_id": candidate["doc_id"],
                    "gold_file_name": candidate["file_name"],
                    "gold_page": candidate["page"],
                    "gold_table_id": candidate["table_id"],
                    "gold_row_key": candidate["row_key"],
                    "gold_row_column": candidate["row_column"],
                    "gold_column": candidate["column"],
                    "gold_answer": candidate["answer"],
                    "question_type": candidate["kind"],
                    "eval_scope": scope,
                    "answer_focus": (
                        f"구조화 표에서 '{candidate['answer']}'을 직접 답하고 "
                        f"{candidate['file_name']} p.{candidate['page']}를 인용"
                    ),
                    "gold_row_chunk_id": candidate["row_chunk_id"],
                    "table_caption": candidate["caption"],
                    "table_topics": candidate["topics"],
                    "schema_parse_quality": candidate["parse_quality"],
                    "generator_version": generator_version,
                }
            )
            qnum += 1
    return rows, audit


def write_review(path: Path, rows: list[dict], audit: list[dict], corpus_label: str) -> None:
    lines = [
        f"# {corpus_label} — 22-document evaluation set",
        "",
        f"- Questions: {len(rows)}",
        f"- Documents covered: {len({r['gold_doc_id'] for r in rows})}",
        "- Scopes per document: anchored / document_scoped / semantic_open",
        "",
        "## Coverage audit",
        "",
        "| doc_id | candidates | selected | pages |",
        "|---|---:|---:|---|",
    ]
    for item in audit:
        pages = ", ".join(str(v) for v in item.get("selected_pages") or [])
        lines.append(
            f"| {item['doc_id']} | {item.get('candidate_count', 0)} | "
            f"{item.get('selected_count', 0)} | {pages} |"
        )
    lines.extend(["", "## Questions", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['qid']} · {row['eval_scope']}",
                "",
                f"- 문서: `{row['gold_doc_id']}` / `{row['gold_file_name']}` p.{row['gold_page']}",
                f"- 표: `{row['gold_table_id']}`",
                f"- 질문: {row['question']}",
                f"- Gold: `{row['gold_answer']}`",
                f"- 행/열: `{row['gold_row_key']}` × `{row['gold_column']}`",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--per-doc", type=int, default=3)
    parser.add_argument("--generator-version", default="table_eval_22_v1")
    parser.add_argument("--corpus-label", default="KR tables v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows, audit = build_rows(manifest, args.chunks_dir, args.per_doc, args.generator_version)
    for item in audit:
        print(
            f"{item['doc_id']}: cells={item.get('candidate_count', 0)} "
            f"schemas={item.get('schema_candidate_count', 0)} selected={item.get('selected_count', 0)} "
            f"kinds={item.get('selected_kinds', [])} pages={item.get('selected_pages', [])}"
        )
    covered = {row["gold_doc_id"] for row in rows}
    print(f"questions={len(rows)} covered_docs={len(covered)}/{len(manifest.get('doc_ids') or [])}")
    if len(covered) != len(manifest.get("doc_ids") or []):
        missing = sorted(set(manifest.get("doc_ids") or []) - covered)
        raise SystemExit(f"coverage failed; missing docs: {missing}")
    if any(item.get("selected_count", 0) < args.per_doc for item in audit):
        raise SystemExit("balance failed; one or more docs have too few verified cells")
    if not args.dry_run:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        write_review(args.review, rows, audit, args.corpus_label)
        print(f"wrote {args.out}")
        print(f"wrote {args.review}")


if __name__ == "__main__":
    main()
