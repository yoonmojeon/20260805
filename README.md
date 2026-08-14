# MaritimeOpsRAG

선박 운항 데이터와 IMO·선급 문서를 한 화면에서 질의하는 Windows 로컬 RAG 애플리케이션입니다. 기본 답변 모델은 Ollama의 `gemma4:12b`이며, 모든 실행 경로는 `C:\Users\user\llmagent`를 기준으로 합니다.

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
- 문서 검색은 Dense + BM25 + 정확 식별자 + 문서 내부 evidence-slot 검색을 결합합니다.
- 표 검색은 별도 precise-table collection을 사용하며 원문 파일·페이지·표 crop을 함께 표시합니다.

## 빠른 실행

```powershell
cd C:\Users\user\llmagent
ollama serve
```

새 PowerShell 창에서:

```powershell
cd C:\Users\user\llmagent
.\.venv\Scripts\python.exe app.py
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
- `data/processed/`: 추출 본문·표·crop·평가 로그

대용량 PDF·DB·인덱스는 GitHub에서 제외됩니다. 현재 로컬 폴더는 단독 실행이 가능하지만 새 PC에서 GitHub만 clone하면 데이터 자산을 별도로 복사해야 합니다.

## 라우팅·검색·답변 흐름

```text
UI 탭 또는 통합 LLM 라우터
  → OPS / RAG / HYBRID / CHAT
  → RAG이면 TEXT / TABLE / BOTH 판단
  → 문서코드·회의차수·선급·질문 요구사항 분석
  → Dense + BM25 + 정확 식별자 후보 검색
  → 문서 내부 조항 검색과 evidence-slot 보강
  → 근거 선택·다문서 quota·표 연결
  → Gemma/Llama 또는 근거 기반 구조화 답변
  → 최종 품질 게이트: 전제 교정·근거 없는 요청 거절·빈 섹션 방지
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
- UI 문서 회귀 10문항: 10/10 통과
- 이전 실패 유형 10문항: 최종 개별 재검증 10/10 통과
- 저장소 전체 회귀: 456 tests passed 및 하위 테스트 5개 통과

두 평가는 문항 구성이 달라 직접적인 전후 비교가 아닙니다. 전체 405문항 결과는 기준선이고, 45문항은 최근 결함 유형을 고르게 뽑은 집중 검증입니다. 자세한 정의와 제한은 [평가 방법과 결과](docs/EVALUATION.md)에 있습니다.

## 문서

- [아키텍처·라우팅·임베딩·검색](docs/ARCHITECTURE.md)
- [평가셋 생성·논문 근거·결과·한계](docs/EVALUATION.md)
- [설치·실행·문제 해결](docs/USAGE.md)
