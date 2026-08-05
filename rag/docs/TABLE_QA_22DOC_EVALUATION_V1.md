# Table QA evaluation — kr_tables_v1 / 22 documents / v1

## Verdict

**FAIL — 현재 컬렉션은 22개 문서 범위의 표 QA에 사용할 수 있는 수준이 아니다.**

| Metric | Required | Result | Pass |
|---|---:|---:|:---:|
| Table ID recall@10 | 90% | 10.6% | NO |
| Cell evidence@10 | 90% | 18.2% | NO |
| Grounded final answer | 85% | 0/22 (0.0%) | NO |
| Retrieval citation@10 | 90% | 10.6% | NO |

> `answer_contains_gold` alone is not a valid final-answer metric for one-character values such as O/○/-. The grounded metric requires answer string, gold table retrieval, and citation retrieval together.

## Evaluation-set coverage

- Documents: 22/22
- Questions: 66 (3 per document)
- Question types: {'schema_caption': 16, 'schema_column': 24, 'cell_lookup': 26}
- Retrieval scopes: {'anchored': 22, 'document_scoped': 22, 'semantic_open': 22}
- Documents with no trustworthy structured cell candidate: 9/22
- Such documents use schema-caption/schema-column questions instead of fabricated cell gold answers.

No-clean-cell documents:

- `kr_kr_rules_11_2014_1a998cbe`
- `kr_kr_rules_13_2025_50b24c68`
- `kr_kr_rules_2014_5be2e49c`
- `kr_kr_rules_2025_41f5a260`
- `kr_kr_rules_2025_57bac298`
- `kr_kr_rules_2025_6b416cae`
- `kr_kr_rules_2025_6f9f069b`
- `kr_kr_rules_2_2025_6bdd7355`
- `kr_kr_rules_3_2025_d82903a5`

## Retrieval results — all 66 questions, open corpus

| Type | N | Table ID | Row | Cell | Citation |
|---|---:|---:|---:|---:|---:|
| cell_lookup | 26 | 11.5% | 3.8% | 11.5% | 11.5% |
| schema_caption | 16 | 6.2% | 6.2% | 6.2% | 6.2% |
| schema_column | 24 | 12.5% | 12.5% | 33.3% | 12.5% |
| **Overall** | **66** | **10.6%** | **7.6%** | **18.2%** | **10.6%** |

## Final-answer sample — one question per document

- Sample size: 22
- Surface gold-string containment: 27.3%
- Gold table retrieval: 4.5%
- Grounded correct answer: 0/22

| QID | Gold | Table hit | Actual answer |
|---|---|:---:|---|
| TE22_001 | 표 2.2.71 가스의 종류 | NO | 결론: 2편_2025.pdf 95페이지의 화학성분 표의 제목은 '화학성분'입니다. 근거: [6] table_id=kr_kr_rules_2_2025_6bdd7355_p0095_t100 type=table_markdown source=KR file=2편_2025.pdf page=95 |
| TE22_004 | 표 1 : 선체 강재의 기계적 성질 | NO | 결론: 13편_2025.pdf 70페이지의 구조화 표 제목은 '화학성분'입니다. 근거: 1, 2 |
| TE22_007 | O | NO | 결론: 'N.A.' 행의 '제1차 정기검사' 값은 O입니다. 근거: 14편_2025.pdf p.494 |
| TE22_010 | - | NO | 결론: '- RPM' 행의 '제1차 정기검사' 값은 O입니다. 근거: Circular (K) Total_2026.pdf p.23 |
| TE22_013 | - | NO | 결론: '(LEL 50%) 제조사의 사양에 따름 13 제어 및 감시장치 고장' 행의 '제1차 정기검사' 값은 ○입니다. 근거: 제조법 및 형식승인 등에 관한 지침_2025.pdf p.59 |
| TE22_016 | 구역 | NO | 결론: '정기검사' 열은 제1차 정기검사, 제2차 정기검사, 제3차 정기검사, 제4차 및 이후 정기검사 순으로 나열되어 있습니다. 근거: 선급부호 안내서_2025.pdf p.10 |
| TE22_019 | - | NO | 결론: 'External deck for distributed loads' 행의 '제4차 및 이후 정기검사' 값은 ○입니다. 근거: 15편_2025.pdf p.194 |
| TE22_022 | ○ | NO | 결론: '펌프실, 파이프터널, 덕트킬, 기관구역, 드라이스페이스, 코퍼댐, 보이드 스페이스' 행의 '제1차 정기검사' 값은 ○입니다. 근거: 제조법 및 형식승인 등에 관한 지침_2025.pdf p.59 |
| TE22_025 | 적하조건 | NO | 결론: 화학성분의 주요 열은 '구역'입니다. 근거: 2025.pdf p.59 |
| TE22_028 | 범주 | NO | 결론: 'P' 열은 존재하지 않습니다. 근거: 11편_2014.pdf p.149 |
| TE22_031 | ○ | NO | 결론: '선저종늑골, 내저종늑골 및 빌지부의 종늑골' 행의 '제1차 정기검사' 값은 ○입니다. 근거: 7편_2025.pdf p.233 |
| TE22_034 | 표 4.6 의장품 및 구성품 목록(계속) | NO | 결론: 빙해운항선박 지침_2025.pdf 159페이지의 구조화 표 제목은 '표 4.4 재료구분 및 등급 (2017)'입니다. 근거: [2] table_id=kr_kr_rules_2025_41f5a260_p0156_t117 type=table_schema source=KR file=빙해운항선박 지침_20… |
| TE22_037 | - | NO | 결론: '종통 창구코밍 (11) III II (0.6 L 외부는 I)' 행의 '제4차 및 이후 정기검사' 값은 ○입니다. 근거: 12편_2014.pdf p.113 |
| TE22_040 | O | NO | 결론: '31. 저압 보조증기관 (≤ 0.7 MPa) L2W L2W O9 O9 O9' 행의 '제1차 정기검사' 값은 '-'. 근거: 5편_2025.pdf p.321 |
| TE22_043 | O | NO | 결론: '예비팬 자동시동' 행의 '제4차 및 이후 정기검사' 값은 ○입니다. 근거: 7편(5,6장)_2025.pdf p.212 |
| TE22_046 | ○ | NO | 결론: '- 잠수사온수공급장치' 행의 '제4차 및 이후 정기검사' 값은 ○입니다. 근거: 9편_2025.pdf p.232 |
| TE22_049 | O | NO | 결론: 'X X 시험방안 B(1) O(2)' 행의 '제1차 정기검사' 값은 O입니다. 근거: 4편_2025.pdf p.65 |
| TE22_052 | 표 3.3.15 트리핑브래킷 사이의 거리 | NO | 결론: 고속경구조선 규칙_2025.pdf 51페이지의 구조화 표 제목은 '화학성분'입니다. 근거: [8] table_id=kr_kr_rules_2025_57bac298_p0166_t082 type=table_schema source=KR file=고속경구조선 규칙_2025.pdf page=166 |
| TE22_055 | O | NO | 결론: '12 시스템을 다른 시스템과 통합하고 선박용 복합시스템을 시험하는 절차' 행의 '제1차 정기검사' 값은 ○입니다. 근거: 6편_2025.pdf p.93 |
| TE22_058 | - | NO | 결론: '제4차 및 이후 정기검사는 ○입니다.' 근거: 해양레저선박 지침_2018.pdf p.71 |
| TE22_061 | 거더의 종류 | NO | 결론: 대형요트 지침_2014.pdf 69페이지의 구조화 표에서 주요 열 하나는 'P'입니다. 근거: 대형요트 지침_2014.pdf p.69 |
| TE22_064 | 1. 화학성분 및 열처리의 규정 | YES | 결론: 화학성분의 주요 열은 'content'입니다. 근거: 저인화점연료선박 규칙_2025.pdf p.58 |

## Primary failure evidence

1. Header/caption leakage assigns inspection-cycle columns or generic topics to unrelated engineering tables.
2. Many documents expose only low-quality schema fields such as `content`, `col_2`, or parse quality around 0.248.
3. Explicit file/page hints are not reliably enforced during schema routing; anchored questions achieved very low table recall.
4. The answer model often repeats question terms or cites a requested page even when the gold table was not retrieved.

## Decision

Do not expand the current table-structuring pipeline to 715 PDFs yet. Add extraction quality gates, document/page routing constraints, and rebuild the 22-document table collection first. Then rerun this fixed evaluation set and require the acceptance thresholds above.

## Artifacts

- Questions: `data/eval/table_questions_22docs_v1.jsonl`
- Question review: `data/eval/table_questions_22docs_v1_review.md`
- Retrieval detail: `data/processed/logs/table_eval_22docs_retrieval_v1.json`
- Answer sample detail: `data/processed/logs/table_eval_22docs_answers_sample_v1.json`
