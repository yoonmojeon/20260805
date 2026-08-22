# MaritimeOpsRAG

선박 운항 데이터와 IMO·선급 PDF를 한 화면에서 질의하는 Windows 온프레미스 RAG 애플리케이션입니다. 기본 답변 모델은 Ollama의 `gemma4:12b`이며, 검색·재순위·답변·검수와 사용자 피드백은 외부 API 없이 로컬에서 처리됩니다.

## 현재 기능

| UI 탭 | 처리 경로 | 용도 |
|---|---|---|
| 통합 질문 | LLM 라우터 → OPS / RAG / HYBRID / CHAT | 정보원을 자동 판단하고 필요하면 운항 DB와 문서를 함께 조회 |
| 문서 검색 | RAG 고정 | IMO·선급 본문, 조항, 수치와 표 검색 |
| 운항 정보 | OPS 고정 | `maritime.db`의 항차·속력·연료·배출량·CII 조회와 보고서 생성 |
| 보고서 관리 | 로컬 파일 조회 | 생성된 Word 보고서 검색·검토·다운로드 |

- 선택 모델: `gemma4:12b`(기본), `llama3.1:8b`
- 규칙 라우터 선택 UI는 제거했습니다. 통합 탭은 LLM 라우팅을 사용하고, 호출 실패·저신뢰 때만 결정적 안전장치가 동작합니다.
- 통합 탭의 `운항 DB 강제`와 `문서 RAG 강제`는 테스트용 정보원 override이며 과거 rule router가 아닙니다.
- 문서 검색은 선택한 Fast / Accurate / Advanced 모드에 따라 Dense, 정확 식별자, 로컬 FTS/BM25, RRF, 재순위와 문서 내부 evidence-slot 검색을 조합합니다.
- 표 검색은 별도 precise-table collection을 사용하며 원문 파일·페이지·표 crop을 함께 표시합니다.

## Fast / Accurate / Advanced

UI의 `문서 답변 모드`에서 하나만 선택하면 검색부터 답변까지 같은 모드가 끝까지 적용됩니다. Advanced는 항상 로컬 `gemma4:12b`를 사용합니다.

| 구분 | Fast | Accurate | Advanced |
|---|---|---|---|
| 권장 용도 | 빠른 1차 조회·간단한 사실 질문 | 일반 시연·규정/회의/표 질문 | 복합 설계 검토·다문서 통합·중요 답변 |
| 기본 후보 폭 | 일반 본문 top 3 / pool 18; 질문 유형별 최대 top 10 | UI top 14, fetch 150; 회의 top 12 / pool 80 | 일반 본문 Dense 80 + 로컬 FTS/BM25 80 → RRF 80, 최대 union 120; 회의/표는 전용 검색기 유지 |
| 정확일치 | 문서코드·희소 복합명사 실패 보강 | Fast 기능 + 정확 문서 강제 선택·문서 내부 sparse 보강 | 질문 분해 후 누락 축별 최대 4회 bounded exact lookup |
| 재순위 | typed evidence slot, 별도 모델 없음 | 메타데이터·문서 권위·다양성·문서 내부 점수 | 후보 36개에 로컬 MiniLM cross-encoder 보조점수 + Gemma listwise, 최종 18개 |
| 문맥 보강 | 짧은 직접 근거 | 인접 조건·예외·목록 tail과 다문서 quota | Accurate 문맥 + parent/sibling 청크 및 누락 facet 재검색 |
| 답변 생성 | 일반 질문은 LLM 사용; 검증된 단순 표/회의/문서목록은 구조화 답변 가능 | Gemma/Llama 근거 생성 + 답변 계약·인용 검증 | Gemma 생성 + 근거 신뢰도 gate + 별도 Gemma 최종 감사/안전한 1회 repair |
| 표 질문 | precise-table 검색·셀 검증·원문 crop | 같은 검증 경로에 넓은 후보 예산 | 표는 검증된 결정적 셀 경로 유지; 불필요한 일반 재순위 미적용 |
| 체감 시간 | 목표 10초 이내, 첫 호출은 더 느릴 수 있음 | 보통 10~25초 | 단일 사실 약 30초, 복합 다문서 약 60~130초 가능 |

Fast도 단순 문자열 템플릿만 반환하는 모드가 아닙니다. 질문 유형을 분류하고 직접 근거를 압축한 뒤 LLM이 답을 작성하는 것이 기본이며, 정확 셀·검증된 회의 결과처럼 결정적 renderer가 더 안전한 경우에만 LLM 생성을 대체합니다.

### BM25·RRF·reranker가 적용되는 위치

- 과거 272k 청크 전역 Python BM25는 첫/반복 질의가 약 10~125초이고 고정 150문항 A/B에서 recall 이득이 없어서 Fast와 Accurate 기본값에서는 사용하지 않습니다.
- Advanced는 전역 메모리 스캔 대신 기존 Chroma 옆의 로컬 SQLite FTS5/BM25 sidecar를 사용합니다. Dense 상위 24개를 보호한 채 RRF로 합치므로 sparse 결과가 좋은 dense 후보를 제거할 수 없습니다.
- 이후 Apache-2.0 `cross-encoder/ms-marco-MiniLM-L4-v2`를 CPU 보조 점수로 사용하고, 최종 선택은 로컬 Gemma listwise reranker가 담당합니다. 모델 폴더가 없거나 로드에 실패하면 Accurate 후보를 그대로 보존하는 fail-closed 구조입니다.
- 검색 결과에 공식 회의 결과와 제안서가 함께 있으면 WP.1/Report/Resolution의 `approved`, `adopted`, `agreed` 문장을 보호합니다. 질문이 제안 자체를 물을 때는 이 보호를 적용하지 않습니다.

## 빠른 실행

```powershell
cd C:\Users\user\llmagent
ollama serve
```

새 PowerShell 창에서:

```powershell
cd C:\Users\user\llmagent
.\start.cmd
```

브라우저에서 `http://127.0.0.1:7860`을 엽니다. 7860 포트가 사용 중이면 기존 UI를 먼저 확인하거나 다른 포트를 지정합니다.

```powershell
$env:GRADIO_SERVER_PORT="7861"
.\.venv\Scripts\python.exe app.py
```

처음 설치하는 경우는 [설치·실행 안내](docs/USAGE.md)를 확인하세요.

## 로컬 데이터

전체 기능에는 다음 로컬 자산이 필요합니다.

- `data/maritime.db`: 운항 데이터
- `data/raw_pdfs/`: 원문 PDF
- `data/processed/index/unified_full_corpus_715_v1/`: 본문 Chroma·BM25 인덱스
- `data/processed/index/unified_full_corpus_715_tables_precise_v1/`: 표 Chroma·BM25 인덱스
- `data/processed/index/unified_full_corpus_715_v1/accurate_sparse_fts5_v2.sqlite3`: Advanced 로컬 sparse sidecar
- `models/cross-encoder-ms-marco-MiniLM-L4-v2/`: Advanced 로컬 cross-encoder
- `data/processed/`: 추출 본문·표·crop·평가 로그

대용량 PDF·DB·인덱스·모델 가중치는 GitHub에서 제외됩니다. 현재 로컬 폴더는 단독 실행이 가능하지만 새 PC에서 GitHub만 clone하면 데이터 자산과 reranker 가중치를 별도로 준비해야 합니다. 설치 방법은 [설치·실행 안내](docs/USAGE.md)에 있습니다.

## 라우팅·검색·답변 흐름

```text
UI 탭 또는 통합 LLM 라우터
  → OPS / RAG / HYBRID / CHAT
  → RAG이면 TEXT / TABLE / BOTH 판단
  → 문서코드·회의차수·선급·질문 요구사항 분석
  → Fast / Accurate / Advanced 검색 정책
  → 정확 문서·선급 포함/제외 조건을 끝까지 유지
  → 문서 내부 조항·희소 복합명사·표 셀·evidence-slot 보강
  → 근거 선택·다문서 quota·회의 결과 권위·표 원본 연결
  → Gemma/Llama 또는 검증된 근거 기반 구조화 답변
  → 답변 계약·인용 검증; Advanced는 별도 최종 감사
  → 한국어 답변 + [n] 인용 + 문서·페이지 Evidence Table
```

기존 임베딩은 다시 만들지 않았습니다. 검색 이후 단계에서 정확 문서 진입, 문서 내부 근거 회수, 다중 문서 비교와 최종 답변 검증을 개선했습니다. 자세한 구현은 [아키텍처·검색](docs/ARCHITECTURE.md)에 있습니다.

## 약 400개 TEXT RAG 질문셋을 만든 방법

사용자가 제공한 증강 v2는 실제 유효 JSONL 기준 **403문항**입니다. 이를 그대로 정답셋으로 쓰지 않고 9개 시나리오의 PDF 근거를 다시 확인해 최종 **405문항**의 `TEXT RAG v3` 평가셋을 만들었습니다.

| ID | 시나리오 | 주 근거 |
|---|---|---|
| V01 | MEPC 84 최신 환경규제 동향 | MEPC 84/7/14 |
| V02 | MSC 111 본회의 결과 | MSC 111-WP.1 |
| V03 | 2024 선대 CII 보고 | MEPC 84/6/2 |
| V04 | 대체연료·GHG 안전 | MSC 111-WP.1, MSC 111/12 |
| V05 | MASS Code·원격운항자 | MSC 111/5 |
| V06 | DNV 자율·원격운항 | DNV-CG-0264 |
| V07 | LR 저인화점 연료 | LR Notice No.1 Section 15 |
| V08 | ABS Smart Functions | ABS Smart Functions Guide v8 |
| V09 | ABS 자율·원격제어 | ABS Autonomous/Remote Requirements v4 |

재구성 절차는 다음과 같습니다.

1. 시나리오마다 정답 문서, keypoint, 허용 동의어, `gold_chunk_ids`, hard-negative와 금지 주장을 사람이 확인해 `text_rag_scenarios_v3.json`에 고정했습니다.
2. 원본 272개 paraphrase 중 시나리오별 15개를 3-gram Jaccard 기반 max-min 방식으로 골라 쉬운 중복 질문의 편중을 줄였습니다.
3. 원본의 seed·boundary·format·scope·integration 질문은 유지하되, 불명확했던 counterfactual·hard-negative 라벨은 상속하지 않고 근거 계약으로 다시 생성했습니다.
4. 각 시나리오에 단일 근거 정밀검색 5개, noise 5개, 근거 없는 요청 5개, 잘못된 전제 4개를 추가하고 4개의 교차문서 integration 질문을 보강했습니다.
5. 최종 405문항에 answerability, expected behavior, gold answer points, 문서·청크·페이지, hard-negative, forbidden claims, hop 수와 검색 난이도를 기록했습니다.
6. 답변가능성·근거 계약·중복·독립 질문 여부·난이도 축을 검사하는 품질 gate를 적용했으며 405개 모두 통과했습니다.

최종 유형은 paraphrase 135, evidence precision 45, noise 45, negative rejection 45, counterfactual 36, scope 28, format 27, boundary 18, integration 17, seed 9개입니다. 답변 가능 문항은 360개, 근거가 없어 거절해야 하는 문항은 45개입니다.

### 참고한 연구와 실제 반영 범위

아래 연구의 설계를 이 프로젝트 데이터에 맞게 적용했습니다. 해당 benchmark 코드를 그대로 복제한 것은 아닙니다.

| 연구 | 이 프로젝트에 반영한 항목 |
|---|---|
| [RAGEval, ACL 2025](https://aclanthology.org/2025.acl-long.418/) | 시나리오 → keypoint/gold answer/reference 계약, Completeness·Hallucination·Irrelevance 관점 |
| [MIRAGE, NAACL Findings 2025](https://aclanthology.org/2025.findings-naacl.157/) | retrieval과 generation 분리 진단, positive gold chunk와 유사 hard-negative |
| [RGB, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29728) | noise robustness, negative rejection, information integration, counterfactual robustness 4축 |
| [GRADE, EMNLP Findings 2025](https://aclanthology.org/2025.findings-emnlp.236/) | 1/2-hop과 easy/medium/hard 검색 난이도 기록 |
| [AURA-QG, IJCNLP-AACL 2025](https://aclanthology.org/2025.ijcnlp-long.159/) | 답변가능성·비중복·coverage 품질 gate |
| [RQUGE, ACL Findings 2023](https://aclanthology.org/2023.findings-acl.428/), [QGEval, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.658/) | 표면 문장 유사도보다 answerability·clarity·consistency를 우선하는 질문 검수 원칙 |
| [ARES, NAACL 2024](https://aclanthology.org/2024.naacl-long.20/), [RAGAS, EACL 2024](https://aclanthology.org/2024.eacl-demo.16/) | context relevance·faithfulness·answer relevance 평가 설계 참고 |

현재 자동 평가는 keypoint·인용·금지주장·거절·형식 계약으로 계산합니다. ARES의 PPI나 별도 학습 judge를 구현한 수치가 아니므로, 사람 검수로 보정된 학술 benchmark 점수처럼 해석하면 안 됩니다.

## 검증 현황

- 2026-08-12 전체 405문항 기준: 후보 문서 Any Recall 97.22%, 최종 문서 Any Recall 96.94%, Gemma must-cover 53.68%, 평균 E2E 8.15초
- 2026-08-14 개선 후 계층 표본 45문항: 품질 80.64%, completeness 75.69%, 요구 행동 준수 86.67%, 평균 6.55초
- 2026-08-22 Accurate 고정 150문항: 품질 82.26%, completeness 64.10%, 행동 준수 100%, 후보/최종 문서 hit 96.24%/92.48%, 평균 E2E 10.30초
- 2026-08-22 Advanced 고정 150문항 검색: 오류 0, 후보/최종 문서 hit 96.99%/95.49%, exact/semantic evidence recall 76.69%/79.70%, 평균 검색 13.65초
- Advanced 실제 생성 10문항 최종 합산(8건 일괄 + 결함 2건 재실행): 오류 0, 행동 준수·금지주장 clean 100%, 문서/page hit 100%/88.89%, 품질 proxy 87.63%, raw keypoint 충족 77.50%, 평균 E2E 46.98초
- 같은 10개 답변 수작업 검수: 8건 OK, 2건 부분 보완(긴 연료 결과의 bullet 예산, ABS 결정상태 표현), 최종 실패 0건. 자동 점수는 전문가 정답률이 아님
- UI 문서 회귀 10문항: 10/10 통과
- 이전 실패 유형 10문항: 최종 개별 재검증 10/10 통과
- 저장소 전체 회귀: 650 tests passed 및 하위 테스트 8개 통과

두 평가는 문항 구성이 달라 직접적인 전후 비교가 아닙니다. 전체 405문항 결과는 기준선이고, 45문항은 최근 결함 유형을 고르게 뽑은 집중 검증입니다. 자세한 정의와 제한은 [평가 방법과 결과](docs/EVALUATION.md)에 있습니다.

## 문서

- [아키텍처·라우팅·임베딩·검색](docs/ARCHITECTURE.md)
- [평가셋 생성·논문 근거·결과·한계](docs/EVALUATION.md)
- [설치·실행·문제 해결](docs/USAGE.md)
- [Advanced 최종 구현·검증 보고서](reports/advanced_rag_final_20260822.md)

## 릴리스에 포함한 검증 산출물

- `data/eval/accurate_eval_150.jsonl`: 현재 Accurate/Advanced 비교에 사용하는 고정 150문항
- `data/eval/accurate_eval_selection_manifest.json`: 405문항 원본에서 150문항을 선정한 기준과 ID
- `data/eval/pilot_validation_text_v3.jsonl`: 715개 PDF 범위를 대상으로 구성한 405문항 기준 평가셋
- `data/eval/table_questions_22docs_practical_v1_curated.jsonl`: 표 검색·답변 회귀 평가셋
- [고객 요구사항 최종 검증 보고서](reports/accurate_hanwha_feedback_final_20260822.md)
- [검증된 질문 50개 Word](reports/RAG_통과질문_50개.docx)
- [UI 시연용 질문 목록](docs/doc/질문리스트_시연용.md)

날짜별 중간 결과, 폐기된 A/B/C 비교 파일, 임시 broad 평가셋과 해당 파일만 만들던 일회성 스크립트는 릴리스에서 제외했습니다. 실행 시 생성되는 인덱스·모델·로그·운항 보고서는 로컬에 유지되며 Git에는 포함하지 않습니다.
