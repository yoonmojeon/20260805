"""Create a human-readable report from the 22-document table QA benchmark artifacts."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data/eval/table_questions_22docs_v1.jsonl"
RETRIEVAL = ROOT / "data/processed/logs/table_eval_22docs_retrieval_v1.json"
ANSWERS = ROOT / "data/processed/logs/table_eval_22docs_answers_sample_v1.json"
OUT = ROOT / "docs/TABLE_QA_22DOC_EVALUATION_V1.md"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def compact(value: str, limit: int = 150) -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip().replace("|", "\\|")
    return value if len(value) <= limit else value[: limit - 1] + "…"


def main() -> None:
    questions = load_jsonl(QUESTIONS)
    retrieval = json.loads(RETRIEVAL.read_text(encoding="utf-8"))
    answers = json.loads(ANSWERS.read_text(encoding="utf-8"))
    r_overall = retrieval["summary"]["overall"]
    a_overall = answers["summary"]["overall"]
    grounded = sum(
        bool(item.get("answer_contains_gold"))
        and bool(item.get("table_recall@k"))
        and bool(item.get("citation_match"))
        for item in answers["results"]
    )
    by_doc_questions: dict[str, list[dict]] = defaultdict(list)
    for row in questions:
        by_doc_questions[str(row["gold_doc_id"])].append(row)
    no_cell_docs = [
        doc for doc, rows in by_doc_questions.items() if not any(r["question_type"] == "cell_lookup" for r in rows)
    ]
    qtypes = Counter(row["question_type"] for row in questions)
    scopes = Counter(row["eval_scope"] for row in questions)

    lines = [
        "# Table QA evaluation — kr_tables_v1 / 22 documents / v1",
        "",
        "## Verdict",
        "",
        "**FAIL — 현재 컬렉션은 22개 문서 범위의 표 QA에 사용할 수 있는 수준이 아니다.**",
        "",
        "| Metric | Required | Result | Pass |",
        "|---|---:|---:|:---:|",
        f"| Table ID recall@10 | 90% | {pct(r_overall.get('table_recall@k'))} | NO |",
        f"| Cell evidence@10 | 90% | {pct(r_overall.get('cell_exact_match'))} | NO |",
        f"| Grounded final answer | 85% | {grounded}/{len(answers['results'])} ({grounded / max(1, len(answers['results'])) * 100:.1f}%) | NO |",
        f"| Retrieval citation@10 | 90% | {pct(r_overall.get('citation_match'))} | NO |",
        "",
        "> `answer_contains_gold` alone is not a valid final-answer metric for one-character values such as O/○/-. "
        "The grounded metric requires answer string, gold table retrieval, and citation retrieval together.",
        "",
        "## Evaluation-set coverage",
        "",
        f"- Documents: {len(by_doc_questions)}/22",
        f"- Questions: {len(questions)} (3 per document)",
        f"- Question types: {dict(qtypes)}",
        f"- Retrieval scopes: {dict(scopes)}",
        f"- Documents with no trustworthy structured cell candidate: {len(no_cell_docs)}/22",
        "- Such documents use schema-caption/schema-column questions instead of fabricated cell gold answers.",
        "",
        "No-clean-cell documents:",
        "",
    ]
    lines.extend(f"- `{doc}`" for doc in sorted(no_cell_docs))
    lines.extend(
        [
            "",
            "## Retrieval results — all 66 questions, open corpus",
            "",
            "| Type | N | Table ID | Row | Cell | Citation |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key, value in retrieval["summary"]["by_question_type"].items():
        lines.append(
            f"| {key} | {value['n']} | {pct(value.get('table_recall@k'))} | "
            f"{pct(value.get('row_recall@k'))} | {pct(value.get('cell_exact_match'))} | "
            f"{pct(value.get('citation_match'))} |"
        )
    lines.extend(
        [
            f"| **Overall** | **{r_overall['n']}** | **{pct(r_overall.get('table_recall@k'))}** | "
            f"**{pct(r_overall.get('row_recall@k'))}** | **{pct(r_overall.get('cell_exact_match'))}** | "
            f"**{pct(r_overall.get('citation_match'))}** |",
            "",
            "## Final-answer sample — one question per document",
            "",
            f"- Sample size: {len(answers['results'])}",
            f"- Surface gold-string containment: {pct(a_overall.get('answer_contains_gold'))}",
            f"- Gold table retrieval: {pct(a_overall.get('table_recall@k'))}",
            f"- Grounded correct answer: {grounded}/{len(answers['results'])}",
            "",
            "| QID | Gold | Table hit | Actual answer |",
            "|---|---|:---:|---|",
        ]
    )
    for item in answers["results"]:
        lines.append(
            f"| {item['qid']} | {compact(item.get('gold_answer'), 55)} | "
            f"{'YES' if item.get('table_recall@k') else 'NO'} | {compact(item.get('answer'), 170)} |"
        )
    lines.extend(
        [
            "",
            "## Primary failure evidence",
            "",
            "1. Header/caption leakage assigns inspection-cycle columns or generic topics to unrelated engineering tables.",
            "2. Many documents expose only low-quality schema fields such as `content`, `col_2`, or parse quality around 0.248.",
            "3. Explicit file/page hints are not reliably enforced during schema routing; anchored questions achieved very low table recall.",
            "4. The answer model often repeats question terms or cites a requested page even when the gold table was not retrieved.",
            "",
            "## Decision",
            "",
            "Do not expand the current table-structuring pipeline to 715 PDFs yet. Add extraction quality gates, "
            "document/page routing constraints, and rebuild the 22-document table collection first. "
            "Then rerun this fixed evaluation set and require the acceptance thresholds above.",
            "",
            "## Artifacts",
            "",
            "- Questions: `data/eval/table_questions_22docs_v1.jsonl`",
            "- Question review: `data/eval/table_questions_22docs_v1_review.md`",
            "- Retrieval detail: `data/processed/logs/table_eval_22docs_retrieval_v1.json`",
            "- Answer sample detail: `data/processed/logs/table_eval_22docs_answers_sample_v1.json`",
        ]
    )
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
