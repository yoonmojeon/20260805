# Table QA evaluation — kr_tables_v2 / 22 documents

## Verdict

**AUTOMATED PRACTICAL GATES PASSED.** Page-blind open-corpus retrieval passes all four 90% gates, and the 22-document answer sample passes the 85% answer and 90% citation gates. Independent PDF visual verification is still required before a production release.

### Assistant-curated practical evaluation (use for product direction)

| Metric | Required | Result | Pass |
|---|---:|---:|:---:|
| Table ID recall@10 | 90% | 90.9% (60/66) | YES |
| Row evidence@10 | 90% | 90.9% (60/66) | YES |
| Cell evidence@10 | 90% | 90.9% (60/66) | YES |
| Citation location@10 | 90% | 90.9% (60/66) | YES |
| Final answer accuracy | 85% | 95.5% (21/22 sample) | YES |
| Final answer citation | 90% | 90.9% (20/22 sample) | YES |

> The 66 questions were individually rewritten as simple single-cell work questions, but the gold cells have not yet been visually checked against every PDF. This is an automated gate pass, not an independent ground-truth audit.

### Structure-regression test (diagnostic only)

| Metric | Required | Result | Pass |
|---|---:|---:|:---:|
| Table ID recall@10 | 90% | 100.0% | YES |
| Cell evidence@10 | 90% | 100.0% | YES |
| Final answer sample | 85% | 100.0% (22 questions) | YES |
| Citation accuracy | 90% | 100.0% | YES |

> 이 100% 결과는 66/66 문항에 정확한 PDF 파일명이 있고, 44/66 문항에 정답 페이지가 있으며, 22/66 문항이 표 제목·열 이름 맞히기다. 구조화/라우팅 회귀시험으로만 사용하며 실무 QA 성능으로 인용하지 않는다.

## Corpus construction

- Documents: 22
- Coordinate-grid tables assessed: 3,399
- Passed and indexed: 3,206 (94.3%)
- Review quarantine: 9
- Rejected/quarantined: 184
- Logical table chunks: 31,204
- Embedded chunks after 420/60 splitting: 49,300
- Embedding model: `intfloat/multilingual-e5-base`
- Parser: `kr-table-v2-pymupdf-grid-1`
- Search chunks contain no Unicode private-use glyphs; unmapped formula glyph runs are represented as `[수식기호]`.

## What changed from v1

1. PDF line/cell coordinates reconstruct columns before whitespace normalization.
2. Multi-row/merged headers are flattened and duplicate columns receive stable suffixes.
3. Parent schema/summary, row cell-facts, and bounded Markdown are indexed separately.
4. File name and page hints are exact metadata filters, including file names containing spaces.
5. Quoted row and column names are parsed as lookup slots; exact literal row hits outrank dense near-misses.
6. Exact cell lookups use deterministic `column=value` answers instead of asking an LLM to rewrite the value.
7. Natural Korean subject/attribute slots and engineering code series are parsed without page or file hints.
8. A table-specific word + character-ngram BM25 index is fused with dense ranks using table-level RRF.
9. Candidate rows are reranked with dense rank, BM25 rank, parsed slots, numeric ranges, and cell-header relevance.
10. Matching structured cells are copied deterministically and cite exact `file p.page`; duplicate corroborating cells may cite multiple sources.

## Evaluation coverage

- Questions: 66 (3 per document)
- Types: 44 cell lookup, 21 caption lookup, 1 column lookup
- Scopes: 22 anchored, 22 document-scoped, 22 semantic/open
- Retrieval is open-corpus; evaluation gold document IDs are not used as filters.
- However, the regression question text itself leaks file names in all 66 questions and pages in 44 questions.

| Scope | N | Table | Row | Cell | Citation |
|---|---:|---:|---:|---:|---:|
| anchored | 22 | 100.0% | 100.0% | 100.0% | 100.0% |
| document_scoped | 22 | 100.0% | 100.0% | 100.0% | 100.0% |
| semantic_open | 22 | 100.0% | 100.0% | 100.0% | 100.0% |

## Practical curated coverage

- Questions: 66 (3 per document)
- Types: 66 simple single-cell lookups; comparison and multi-condition questions excluded
- Exact page hints: 0; exact `.pdf` file names: 0; part-number prompts: 0
- Human-verified gold cells: 0. PDF visual checking remains required.
- Search hits: 60/66 tables, rows, cells, and citation locations.
- Answer sample: 21/22 answers contained the gold value; 20/22 cited the exact gold file and page.
- Evaluation uses no gold document, file, page, table, row, or cell fields during retrieval/reranking.

| Type | N | Table | Row | Cell | Citation |
|---|---:|---:|---:|---:|---:|
| single_cell_lookup | 66 | 90.9% | 90.9% | 90.9% | 90.9% |

## Remaining limitations

- Borderless tables that cannot produce a reliable coordinate grid remain quarantined.
- Formula-font glyph substitution preserves search context but does not recover the original mathematical symbol.
- The practical final-answer result is a 22-question sample (one simple lookup per document), not all 66 answers.
- The evaluation set is derived from the same v2 structured records; it proves pipeline consistency, not independent PDF ground truth.
- Six of 66 retrieval questions still miss the strict gold table/cell at top 10; several concern duplicated or near-duplicated requirements across rule editions/documents.
- The answer sample covers one question per document (22), not all 66 generated answers.
- Table BM25 scores all 49,300 chunks per query and remains a latency optimization target.

## Artifacts

- `data/processed/tables_v2/<doc_id>/tables.jsonl`
- `data/processed/chunks_v2/<doc_id>/table_chunks.jsonl`
- `data/processed/logs/kr_tables_v2_quality.json`
- `data/processed/index/unified_kr_tables_v2/index_manifest.json`
- `data/eval/table_questions_22docs_v2.jsonl`
- `data/eval/table_questions_22docs_v2_review.md`
- `data/processed/logs/table_eval_22docs_retrieval_v2_gold.json`
- `data/processed/logs/table_eval_22docs_answers_v2_sample.json`
- `data/eval/table_questions_22docs_practical_v1_curated.jsonl`
- `data/eval/table_questions_22docs_practical_v1_curated_review.md`
- `data/processed/logs/table_eval_22docs_practical_v1_curated_hybrid_v2.json`
- `data/processed/logs/table_eval_22docs_practical_v1_curated_hybrid_answers_v3.json`

## Rebuild commands

```powershell
python scripts/44_build_kr_tables_v2.py
python scripts/45_reassess_kr_tables_v2.py
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
python scripts/10_build_unified_index.py --collection-id kr_tables_v2 --doc-list data/manifests/kr_table_top22.csv --manifest data/manifests/rag_corpus_457.csv --chunks-dir data/processed/chunks --table-chunks-dir data/processed/chunks_v2 --include-types table --structured-tables only --max-embedding-tokens 420 --embedding-overlap-tokens 60
python scripts/48_build_table_eval_simple_curated.py
python scripts/35_build_bm25_index.py --unified kr_tables_v2 --table --rebuild
$env:RAG_DEBUG_TRACE_STDERR='0'
python scripts/42_table_eval_22_benchmark.py --questions data/eval/table_questions_22docs_practical_v1_curated.jsonl --chunks-dir data/processed/chunks_v2 --collection-id kr_tables_v2 --out data/processed/logs/table_eval_22docs_practical_v1_curated_hybrid_v2.json
python scripts/42_table_eval_22_benchmark.py --questions data/eval/table_questions_22docs_practical_v1_curated.jsonl --chunks-dir data/processed/chunks_v2 --collection-id kr_tables_v2 --with-llm --one-per-doc --out data/processed/logs/table_eval_22docs_practical_v1_curated_hybrid_answers_v3.json
```
