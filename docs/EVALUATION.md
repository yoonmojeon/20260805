# TEXT RAG 평가셋과 검증 결과

## 결론

TEXT RAG v3는 사용자가 제공한 약 400개 질문을 그대로 gold로 간주하지 않고, 9개 해사 시나리오의 PDF 문서·청크·필수 답변을 다시 연결한 평가셋입니다.

- 제공 원본: 유효 JSONL **403문항**
- 최종 평가셋: **405문항**
- 답변 가능: 360문항
- 근거가 없어 거절해야 함: 45문항
- 1-hop / 2-hop: 345 / 60문항
- 난이도 easy / medium / hard: 81 / 181 / 143문항

단일 문서 식별은 비교적 안정적입니다. 현재 어려운 부분은 긴 문서의 여러 조항을 빠짐없이 모으는 것, 여러 문서의 모든 근거를 유지하는 것, 검색 근거를 모델이 답변에 누락 없이 반영하는 것입니다.

## 9개 시나리오

| ID | 카테고리 | 기준 문서·주제 |
|---|---|---|
| V01 | 최신 동향 | MEPC 84/7/14, ISWG-GHG 20차 회의 |
| V02 | 회의 결과 | MSC 111-WP.1 본회의 보고서 초안 |
| V03 | 환경규제 | MEPC 84/6/2, 2024 선대 탄소집약도 |
| V04 | 환경규제 | MSC 111-WP.1·MSC 111/12, 대체연료·GHG 안전 |
| V05 | 자율운항 | MSC 111/5, MASS Code·원격운항자·2030 일정 |
| V06 | 선급 Rule | DNV-CG-0264 자율·원격운항 선박 |
| V07 | 선급 Rule | LR Notice No.1 Section 15 저인화점 연료 |
| V08 | 선급 Rule | ABS Smart Functions Guide v8 |
| V09 | 선급 Rule | ABS Autonomous/Remote Requirements v4 |

각 시나리오에는 다음 계약을 사람이 확인해 기록했습니다.

- 정확한 `doc_id`, 허용 가능한 대체 문서와 문서 label
- 4개 안팎의 핵심 주장과 허용 동의어
- 주장별 `gold_chunk_ids`, 페이지와 gold answer
- 검색에 섞일 수 있는 hard-negative 청크
- 답변에 나오면 안 되는 forbidden claims
- 정밀질문, 근거 없는 대상, 잘못된 전제, 교차문서 연결 대상

파일: `data/eval/text_rag_scenarios_v3.json`

## 403문항에서 405문항으로 재구성

원본 403문항의 구성은 seed 9, paraphrase 272, boundary 18, format 27, scope 28, integration 13, counterfactual 18, hard-negative 18개였습니다.

최종셋은 다음과 같이 만들었습니다.

| 단계 | 문항 수 | 처리 |
|---|---:|---|
| seed·boundary·format·scope·기존 integration 유지 | 95 | 질문 표현은 유지하고 gold를 시나리오 계약으로 교체 |
| 다양한 paraphrase 선택 | 135 | 시나리오별 15개, 3-gram Jaccard max-min 선택 |
| evidence precision 생성 | 45 | 시나리오별 지정 문서·핵심 근거 질문 5개 |
| noise robustness 생성 | 45 | seed 질문에 실제 hard-negative 주입 계약 5종 |
| negative rejection 생성 | 45 | 시나리오별 문서에서 확인할 수 없는 대상 5개 |
| counterfactual robustness 재생성 | 36 | 시나리오별 틀린 전제 4개와 교정 근거만 채점 |
| 교차문서 integration 추가 | 4 | MEPC·MSC·DNV·ABS 연결 질문 |
| 합계 | **405** | 정규화 중복 없음, 품질 gate 전부 통과 |

원본의 counterfactual과 hard-negative label은 부모 질문의 모든 gold를 무조건 상속해 짧고 정확한 교정 답변도 오답 처리할 수 있었습니다. v3에서는 틀린 전제를 바로잡는 데 필요한 point만 연결하고, negative rejection은 retrieval recall과 분리해 거절 행동으로 채점합니다.

최종 유형 분포:

| 유형 | 문항 수 | 확인 목적 |
|---|---:|---|
| paraphrase | 135 | 표현 변화에 대한 검색 안정성 |
| evidence precision | 45 | 지정 문서·개념의 정확 근거 회수 |
| noise robustness | 45 | 무관한 근거가 섞여도 답 유지 |
| negative rejection | 45 | 근거가 없을 때 추측하지 않고 거절 |
| counterfactual robustness | 36 | 잘못된 전제 교정 |
| scope | 28 | 적용범위·제외조건 |
| format | 27 | 요구한 출력 형식 준수 |
| boundary | 18 | 유사 문서·기관·범주 구분 |
| integration | 17 | 여러 문서·조항 통합 |
| seed | 9 | 시나리오 기준 질문 |

## 논문에서 가져온 설계 원칙

논문을 참고해 프로젝트용 결정적 생성·검수 파이프라인을 구현했습니다. 논문의 원 데이터나 전체 benchmark 구현을 그대로 사용한 것은 아닙니다.

### 실제 데이터 구성에 반영

- [RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework, ACL 2025](https://aclanthology.org/2025.acl-long.418/): 시나리오 스키마에서 question·answer·reference를 함께 만들고 Completeness·Hallucination·Irrelevance를 분리하는 발상.
- [MIRAGE: A Metric-Intensive Benchmark for RAG Evaluation, NAACL Findings 2025](https://aclanthology.org/2025.findings-naacl.157/): retrieval과 generation을 따로 진단하고 positive evidence와 유사 negative를 함께 관리하는 발상.
- [Benchmarking Large Language Models in RAG, RGB, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29728): noise robustness, negative rejection, information integration, counterfactual robustness의 네 능력 축.
- [GRADE, EMNLP Findings 2025](https://aclanthology.org/2025.findings-emnlp.236/): reasoning hop과 retrieval difficulty를 별도 축으로 기록하는 난이도 행렬.
- [AURA-QG, IJCNLP-AACL 2025](https://aclanthology.org/2025.ijcnlp-long.159/): answerability, non-redundancy, coverage를 질문셋 품질 gate로 보는 원칙.
- [RQUGE, ACL Findings 2023](https://aclanthology.org/2023.findings-acl.428/)와 [QGEval, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.658/): reference와의 표면 유사도보다 answerability·clarity·consistency를 중시하는 검수 원칙.

### 평가 설계만 참고

- [ARES, NAACL 2024](https://aclanthology.org/2024.naacl-long.20/): context relevance, answer faithfulness, answer relevance 및 소량 인간 라벨을 이용한 PPI 보정.
- [RAGAS, EACL 2024](https://aclanthology.org/2024.eacl-demo.16/): reference-free RAG 평가 관점.

현재 코드의 자동 점수는 keypoint 동의어, 인용, forbidden claim, 거절/반박 behavior, irrelevance와 형식 계약을 사용합니다. ARES judge를 학습하거나 PPI를 수행하지 않았고, RAGAS 패키지 점수를 공식 지표로 보고하지도 않습니다. 따라서 아래 수치는 재현 가능한 내부 회귀 지표이며 인간 평가를 완전히 대체하지 않습니다.

## 품질 gate

`rag/scripts/build_text_eval_v3.py`가 다음을 검증합니다.

- 질문 ID와 정규화 질문 중복 없음
- answerability와 retrieval target의 논리 일치
- 답변 가능 문항의 gold evidence·chunk 계약 존재
- paraphrase 간 최근접 3-gram Jaccard `< 0.75`
- evidence precision 질문에 문서 맥락이 독립적으로 포함됨
- scenario·test type·hop·difficulty 축 존재
- noise 문항에 실제 hard-negative chunk가 연결됨
- negative rejection은 gold 검색이 아닌 거절 행동으로 설정됨

최종 405문항은 이 gate를 모두 통과했습니다.

## 평가 결과

### 전체 405문항 기준선 — 2026-08-12

`full_corpus_715_v1`, accurate mode, `gemma4:12b` 기준입니다. 이는 최종 답변 품질 게이트를 추가하기 전 전체 기준선입니다.

검색:

| 지표 | 결과 |
|---|---:|
| Candidate document any / all | 97.22% / 93.89% |
| Final document any / all | 96.94% / 90.28% |
| Candidate semantic point recall | 61.46% |
| Final semantic point recall | 46.18% |
| 평균 / 중앙 검색시간 | 1.61초 / 1.14초 |

Gemma 답변:

| 지표 | 결과 |
|---|---:|
| 완료 / 오류 | 405 / 0 |
| Must-cover completeness | 53.68% |
| Behavior pass | 99.75% |
| Answerable citation | 99.72% |
| 금지 주장 clean | 100% |
| Negative irrelevance clean | 100% |
| Quality proxy | 77.33% |
| 평균 검색 / 생성 / 전체 | 1.99초 / 6.16초 / 8.15초 |

### 최근 개선 집중 검증 — 2026-08-14

9개 시나리오에서 seed, evidence precision, negative rejection, counterfactual, integration을 하나씩 뽑은 45문항 서비스 경로 검증입니다.

| 지표 | 개선 전 | 개선 후 |
|---|---:|---:|
| 평균 품질 | 67.40% | 80.64% |
| 평균 completeness | 51.10% | 75.69% |
| 요구 행동 준수 | 71.10% | 86.67% |
| 평균 E2E | 6.80초 | 6.55초 |

이후 전제 교정 9문항은 개별 재검증에서 9/9 통과했습니다. 다만 결함 답변만 Gemma로 한 번 재작성하므로 전제 교정 표본의 평균은 약 21초로 늘었습니다. 일반 문서 회귀 10문항 평균은 약 5초입니다.

추가 회귀:

- UI 문서 질문 10개: 10/10 통과
- 이전에 실패하던 질문 10개: 최종 개별 재검증 10/10 통과
- 당시 전체 저장소: 456 tests passed 및 하위 테스트 5개 통과
- 실제 UI: ABS 위험범주, 근거 없는 MASS 발효일 거절, ABS 두 문서 비교를 브라우저에서 확인

전체 기준선과 집중 검증은 문항 구성이 다르므로 숫자를 직접적인 단일 전후 실험으로 해석하면 안 됩니다. 최종 품질 게이트를 포함한 405문항 전체 재실행은 별도 장시간 검증 과제입니다.

### 고정 150문항 Accurate — 2026-08-22

PPT 및 제품 회귀용으로 405문항에서 시나리오·질문행위·난이도를 고르게 고정한 `accurate_eval_150.jsonl`을 사용했습니다. 150문항 중 답변 가능 133, 근거 부재를 거절해야 하는 문항 17개입니다. 카테고리는 Rule 65, 환경규제 34, 회의결과 17, 최신동향 17, 자율운항 17개이며, 단순 paraphrase뿐 아니라 evidence precision·noise·negative rejection·counterfactual·integration을 포함합니다.

| 지표 | Accurate 최종 |
|---|---:|
| 전체 품질 proxy | 82.26% |
| 필수내용 충족도 | 64.10% |
| 답변/행동 성공률 | 100.00% |
| 후보 / 최종 문서 hit | 96.24% / 92.48% |
| 페이지 hit | 83.46% |
| 평균 E2E | 10.30초 |

전역 Python BM25를 추가한 보호형 Dense+BM25+RRF A/B는 최종 semantic point recall 78.57%로 Dense 기준선과 같았고 평균 검색시간만 2.03→2.88초로 늘어 Accurate 기본값에 넣지 않았습니다.

### Advanced 검증 — 2026-08-22

Advanced는 로컬 FTS5/BM25·RRF candidate union, MiniLM 보조점수, Gemma listwise와 최종 감사를 추가한 고품질 모드입니다. 최종 150문항 검색 재실행은 오류 0건이었고, 후보/최종 문서 any hit 96.99%/95.49%, 최종 exact point recall 76.69%, 최종 semantic point recall 79.70%, 평균/중앙 검색 13.65초/8.66초였습니다. 상세 비교는 `reports/advanced_rag_final_20260822.md`에 고정합니다.

최종 코드 회귀는 650 tests passed 및 8개 하위 테스트 통과입니다.

개발 중 첫 40문항 검색 표본은 오류 0건, 후보/최종 문서 any hit 97.14%, 최종 exact point recall 78.57%, 최종 semantic point recall 82.14%, 평균 검색 12.70초였습니다. 이 실행 뒤 발견한 `MEPC 제외` 괄호 파서와 MSC 연료결과 근거 보호를 추가했으므로 최종 보고서 수치를 우선 사용해야 합니다.

실제 Gemma 생성 10문항 최종 합산은 8건 일괄 결과에 마지막 형식 결함 2건을 동일 평가기로 재실행해 교체한 값입니다. 오류 0건, 행동 준수·금지주장·irrelevance clean 100%, gold 문서/page hit 100%/88.89%, 평균 품질 proxy 87.63%, raw keypoint 충족 77.50%, groundedness proxy 77.94%, 평균 E2E 46.98초였습니다. 10개 답변을 직접 읽은 판정은 8건 OK, 2건 부분 보완, 최종 실패 0건입니다.

부분 보완은 MSC 연료 결과 답변이 요구 bullet 예산을 약간 넘긴 점과, ABS Smart Function의 “제안문/최종 결정문” 질문이 일반 선급 Guide에 회의 문서식 결정상태 표현을 과하게 적용한 점입니다. 또한 `T3-V06_p02`는 실제 질문이 DNV Smart Vessel·자율운항 CG의 “명칭과 범위”를 요구하지만 gold keypoint는 PRA·showstopper까지 요구합니다. 따라서 이 문항의 자동 completeness 25%는 실제 질문 적합성을 과소평가하며, 수작업 검수에서는 DNV-CG-0508과 DNV-CG-0264의 명칭·범위를 모두 확인해 OK로 분류했습니다.

초기 실행 이후 대표 결함은 다음처럼 다시 확인했습니다.

- `DNV-CG-0264` 목적 질문: 문서카드 경로 대신 명시 문서 사실 경로로 수정 후 품질 45%→100%.
- `MSC 111 연료 안전·위험평가`: 공식 결과 4축 exact recovery와 결과문서 보호 후 품질 45%→86.25%, completeness 0%→75%, gold page hit 100%.
- 암모니아 연료선 개념승인 복합 질문: UI 동일 경로에서 MSC 결과 + DNV/KR 설계 체크리스트 + 미확정 범위를 함께 생성했으며 최종 재실행은 118.51초였습니다.
- 표 질문 수작업 확인: RSTH 확관 허용 바깥지름 1.14배(28.32초), 10만 초과~15만 DWT 안전사용하중 250t(25.75초)을 정확 셀과 원문 crop으로 답했습니다.

자동 품질 proxy는 gold keypoint 동의어 기반이며 전문가 점수가 아닙니다. 특히 특정 gold가 질문 범위보다 넓거나, 최종 감사자가 사용하는 인용 행을 평가기가 반영하지 않던 과거 로그가 있으므로, 서비스 배포 판단에는 실제 답변·Evidence Table 수작업 검토를 함께 사용합니다. 최종 평가기는 감사 후 cited rows까지 반영하도록 수정했습니다.

## 재현

평가셋 재생성:

```powershell
.\.venv\Scripts\python.exe rag\scripts\build_text_eval_v3.py
```

검색 평가:

```powershell
.\.venv\Scripts\python.exe rag\scripts\eval_text_retrieval_v3.py `
  --out-dir data\processed\logs\text_rag_eval_v3\retrieval_current
```

Gemma 답변 평가:

```powershell
.\.venv\Scripts\python.exe rag\scripts\eval_text_answers_v3.py `
  --llm-model gemma4:12b `
  --out-dir data\processed\logs\text_rag_eval_v3\gemma_current
```

UI 문서 회귀:

```powershell
.\.venv\Scripts\python.exe scripts\run_ui_document_regression.py
```

관련 파일:

- 제공 원본: `data/eval/source/pilot_validation_questions_augmented_v2_with_gold.jsonl`
- 시나리오·gold 계약: `data/eval/text_rag_scenarios_v3.json`
- 최종 405문항: `data/eval/pilot_validation_text_v3.jsonl`
- 생성기: `rag/scripts/build_text_eval_v3.py`
- 검색 평가기: `rag/scripts/eval_text_retrieval_v3.py`
- 답변 평가기: `rag/scripts/eval_text_answers_v3.py`

평가 로그와 모델 출력은 `data/processed/`에 로컬 보관하며 대용량이므로 GitHub에 올리지 않습니다.

## 남은 개선 순서

1. 최종 품질 게이트를 포함한 405문항 전체 재실행과 사람 검수 표본 추가
2. 긴 문서에서 누락 evidence slot만 저비용으로 재검색
3. 다문서 질문의 source별 최소 근거 quota 강화
4. 자동 judge 도입 시 사람 라벨로 faithfulness·relevance 점수 보정
5. 전제 교정 repair prompt와 출력 길이를 줄여 20초대 지연 개선
