"""Build the consolidated KR tables v2 construction and QA report."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUALITY = ROOT / "data/processed/logs/kr_tables_v2_quality.json"
INDEX = ROOT / "data/processed/index/unified_kr_tables_v2/index_manifest.json"
QUESTIONS = ROOT / "data/eval/table_questions_22docs_v2.jsonl"
RETRIEVAL = ROOT / "data/processed/logs/table_eval_22docs_retrieval_v2_gold.json"
ANSWERS = ROOT / "data/processed/logs/table_eval_22docs_answers_v2_sample.json"
PRACTICAL_QUESTIONS = ROOT / "data/eval/table_questions_22docs_practical_v1_curated.jsonl"
PRACTICAL_RETRIEVAL = ROOT / "data/processed/logs/table_eval_22docs_practical_v1_curated.json"
PRACTICAL_ANSWERS = ROOT / "data/processed/logs/table_eval_22docs_practical_v1_curated_answers_sample.json"
OUT = ROOT / "docs/TABLE_QA_22DOC_EVALUATION_V2.md"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    quality = json.loads(QUALITY.read_text(encoding="utf-8"))
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    retrieval = json.loads(RETRIEVAL.read_text(encoding="utf-8"))
    answers = json.loads(ANSWERS.read_text(encoding="utf-8"))
    practical_retrieval = json.loads(PRACTICAL_RETRIEVAL.read_text(encoding="utf-8"))
    practical_answers = json.loads(PRACTICAL_ANSWERS.read_text(encoding="utf-8"))
    questions = [json.loads(x) for x in QUESTIONS.read_text(encoding="utf-8").splitlines() if x.strip()]
    practical_questions = [
        json.loads(x) for x in PRACTICAL_QUESTIONS.read_text(encoding="utf-8").splitlines() if x.strip()
    ]
    r = retrieval["summary"]["overall"]
    a = answers["summary"]["overall"]
    p = practical_retrieval["summary"]["overall"]
    pa = practical_answers["summary"]["overall"]
    totals = quality["totals"]
    assessed = sum(int(v) for v in totals.values())
    passed = int(totals.get("pass", 0))

    lines = [
        "# Table QA evaluation — kr_tables_v2 / 22 documents",
        "",
        "## Verdict",
        "",
        "**NOT PASSED for practical table QA. The v2 structure-regression test passes, but the page-blind assistant-curated evaluation fails every practical gate.**",
        "",
        "### Assistant-curated practical evaluation (use for product direction)",
        "",
        "| Metric | Required | Result | Pass |",
        "|---|---:|---:|:---:|",
        f"| Table ID recall@10 | 90% | {pct(p['table_recall@k'])} | NO |",
        f"| Row evidence@10 | 90% | {pct(p['row_recall@k'])} | NO |",
        f"| Cell evidence@10 | 90% | {pct(p['cell_exact_match'])} | NO |",
        f"| Citation location@10 | 90% | {pct(p['citation_match'])} | NO |",
        f"| Final answer accuracy | 85% | {pct(pa['answer_contains_gold'])} ({pa['n']}-question sample) | NO |",
        f"| Final answer citation | 90% | {pct(pa['answer_cites_gold'])} ({pa['n']}-question sample) | NO |",
        "",
        "> The 66 questions were individually rewritten as simple single-cell work questions, but the gold cells "
        "have not yet been visually checked against every PDF. The result is sufficient to reject a production PASS.",
        "",
        "### Structure-regression test (diagnostic only)",
        "",
        "| Metric | Required | Result | Pass |",
        "|---|---:|---:|:---:|",
        f"| Table ID recall@10 | 90% | {pct(r['table_recall@k'])} | YES |",
        f"| Cell evidence@10 | 90% | {pct(r['cell_exact_match'])} | YES |",
        f"| Final answer sample | 85% | {pct(a['answer_contains_gold'])} ({a['n']} questions) | YES |",
        f"| Citation accuracy | 90% | {pct(a['answer_cites_gold'])} | YES |",
        "",
        "> 이 100% 결과는 66/66 문항에 정확한 PDF 파일명이 있고, 44/66 문항에 정답 페이지가 있으며, "
        "22/66 문항이 표 제목·열 이름 맞히기다. 구조화/라우팅 회귀시험으로만 사용하며 실무 QA 성능으로 인용하지 않는다.",
        "",
        "## Corpus construction",
        "",
        f"- Documents: {quality['documents']}",
        f"- Coordinate-grid tables assessed: {assessed:,}",
        f"- Passed and indexed: {passed:,} ({passed / max(1, assessed) * 100:.1f}%)",
        f"- Review quarantine: {int(totals.get('review', 0)):,}",
        f"- Rejected/quarantined: {int(totals.get('reject', 0)):,}",
        f"- Logical table chunks: {quality['output_chunks']:,}",
        f"- Embedded chunks after 420/60 splitting: {index['indexed_chunks']:,}",
        f"- Embedding model: `{index['embedding_model']}`",
        f"- Parser: `kr-table-v2-pymupdf-grid-1`",
        "- Search chunks contain no Unicode private-use glyphs; unmapped formula glyph runs are represented as `[수식기호]`.",
        "",
        "## What changed from v1",
        "",
        "1. PDF line/cell coordinates reconstruct columns before whitespace normalization.",
        "2. Multi-row/merged headers are flattened and duplicate columns receive stable suffixes.",
        "3. Parent schema/summary, row cell-facts, and bounded Markdown are indexed separately.",
        "4. File name and page hints are exact metadata filters, including file names containing spaces.",
        "5. Quoted row and column names are parsed as lookup slots; exact literal row hits outrank dense near-misses.",
        "6. Exact cell lookups use deterministic `column=value` answers instead of asking an LLM to rewrite the value.",
        "",
        "## Evaluation coverage",
        "",
        f"- Questions: {len(questions)} (3 per document)",
        "- Types: 44 cell lookup, 21 caption lookup, 1 column lookup",
        "- Scopes: 22 anchored, 22 document-scoped, 22 semantic/open",
        "- Retrieval is open-corpus; evaluation gold document IDs are not used as filters.",
        "- However, the regression question text itself leaks file names in all 66 questions and pages in 44 questions.",
        "",
        "| Scope | N | Table | Row | Cell | Citation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scope, values in retrieval["summary"]["by_scope"].items():
        lines.append(
            f"| {scope} | {values['n']} | {pct(values['table_recall@k'])} | "
            f"{pct(values['row_recall@k'])} | {pct(values['cell_exact_match'])} | "
            f"{pct(values['citation_match'])} |"
        )
    lines.extend([
        "",
        "## Practical curated coverage",
        "",
        f"- Questions: {len(practical_questions)} (3 per document)",
        "- Types: 66 simple single-cell lookups; comparison and multi-condition questions excluded",
        "- Exact page hints: 0; exact `.pdf` file names: 0; part-number prompts: 0",
        "- Human-verified gold cells: 0. PDF visual checking remains required.",
        "- Search hits: 16/66 tables, 15/66 rows, 16/66 cells",
        "- Nine of 22 documents scored zero on all three questions.",
        "- Answer sample: all 8/8 search-hit questions contained the gold value; 0/14 search-miss questions did.",
        "",
        "| Type | N | Table | Row | Cell | Citation |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for qtype, values in practical_retrieval["summary"]["by_question_type"].items():
        lines.append(
            f"| {qtype} | {values['n']} | {pct(values['table_recall@k'])} | "
            f"{pct(values['row_recall@k'])} | {pct(values['cell_exact_match'])} | "
            f"{pct(values['citation_match'])} |"
        )
    lines.extend([
        "",
        "## Remaining limitations",
        "",
        "- Borderless tables that cannot produce a reliable coordinate grid remain quarantined.",
        "- Formula-font glyph substitution preserves search context but does not recover the original mathematical symbol.",
        "- The practical final-answer result is a 22-question sample (one simple lookup per document), not all 66 answers.",
        "- The evaluation set is derived from the same v2 structured records; it proves pipeline consistency, not independent PDF ground truth.",
        "- Practical questions reveal weak page-blind semantic routing: the query parser missed a row entity in 52/66 questions.",
        "",
        "## Artifacts",
        "",
        "- `data/processed/tables_v2/<doc_id>/tables.jsonl`",
        "- `data/processed/chunks_v2/<doc_id>/table_chunks.jsonl`",
        "- `data/processed/logs/kr_tables_v2_quality.json`",
        "- `data/processed/index/unified_kr_tables_v2/index_manifest.json`",
        "- `data/eval/table_questions_22docs_v2.jsonl`",
        "- `data/eval/table_questions_22docs_v2_review.md`",
        "- `data/processed/logs/table_eval_22docs_retrieval_v2_gold.json`",
        "- `data/processed/logs/table_eval_22docs_answers_v2_sample.json`",
        "- `data/eval/table_questions_22docs_practical_v1_curated.jsonl`",
        "- `data/eval/table_questions_22docs_practical_v1_curated_review.md`",
        "- `data/processed/logs/table_eval_22docs_practical_v1_curated.json`",
        "- `data/processed/logs/table_eval_22docs_practical_v1_curated_answers_sample.json`",
        "",
        "## Rebuild commands",
        "",
        "```powershell",
        "python scripts/44_build_kr_tables_v2.py",
        "python scripts/45_reassess_kr_tables_v2.py",
        "$env:HF_HUB_OFFLINE='1'",
        "$env:TRANSFORMERS_OFFLINE='1'",
        "python scripts/10_build_unified_index.py --collection-id kr_tables_v2 --doc-list data/manifests/kr_table_top22.csv --manifest data/manifests/rag_corpus_457.csv --chunks-dir data/processed/chunks --table-chunks-dir data/processed/chunks_v2 --include-types table --structured-tables only --max-embedding-tokens 420 --embedding-overlap-tokens 60",
        "python scripts/48_build_table_eval_simple_curated.py",
        "$env:RAG_DEBUG_TRACE_STDERR='0'",
        "python scripts/42_table_eval_22_benchmark.py --questions data/eval/table_questions_22docs_practical_v1_curated.jsonl --chunks-dir data/processed/chunks_v2 --collection-id kr_tables_v2 --out data/processed/logs/table_eval_22docs_practical_v1_curated.json",
        "```",
    ])
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
