# MaritimeOpsRAG

운항 SQLite(`ship-data`)와 선급·IMO 문서 RAG(`MaritimeRAG`)를 한 Gradio UI에서 쓰는 프로젝트입니다.
질문이 오면 UI에서 선택한 Ollama 모델이 질문 전체의 의미를 보고 **안내 / 운항 DB / 문서 인덱스 / 둘 다** 중 필요한 소스를 고릅니다. 같은 모델이 검색 이후의 답변도 생성하지만, routing과 answer는 서로 다른 호출입니다.

```
질문 → minimal hard guard → selected model semantic router
     → chat | ops | rag | hybrid → DB/RAG 실행 → same selected model answer
```

저장소: [github.com/yoonmojeon/20260805](https://github.com/yoonmojeon/20260805)

## 라우팅 구조

### UI에서 라우팅 방식과 모델 선택

채팅 UI에는 라우팅 방식과 모델 선택이 각각 있습니다.

| UI 항목 | 선택값 | 설명 |
|---------|--------|------|
| 라우팅 방식 | **LLM-primary (기본·권장)** | 선택 모델이 질문의 의미를 보고 필요한 소스를 결정합니다. |
| 라우팅 방식 | Rules-only (비교용) | 기존 점수·패턴·prototype만으로 경로를 고릅니다. Ollama는 답변 생성에만 사용합니다. |
| 라우터·답변 모델 | **Gemma 4 12B (기본)** | 신규 30문항에서 라우팅과 최종 QA가 가장 좋았습니다. |
| 라우터·답변 모델 | Llama 3.1 8B | Gemma보다 빠르지만 정확도는 낮았습니다. |
| 라우터·답변 모델 | Mistral Nemo 12B | 비교 실험용으로 유지합니다. |

`운항 DB 강제`, `문서 RAG 강제`를 선택하면 라우팅 방식을 건너뛰고 해당 경로만 실행합니다. 답변 아래의 경로 배너에서 실제 경로, 라우팅 방식, 신뢰도, 모델을 확인할 수 있습니다.

### LLM-primary가 질문을 처리하는 순서

```text
사용자 질문
  → 수동 강제 경로 또는 확실한 인사·감사·정체성 질문만 hard guard
  → UI에서 선택한 Ollama 모델에 source-needs JSON 요청
       need_ops: 운항 DB가 필요한가?
       need_documents: 규정·회의·표 검색이 필요한가?
  → Python이 두 boolean을 경로로 변환
       false / false → chat
       true  / false → ops
       false / true  → rag
       true  / true  → hybrid
  → SQLite 및 Chroma 검색
  → 같은 선택 모델이 근거 기반 답변 생성
```

라우터와 답변 생성은 같은 모델을 사용하지만 **서로 다른 호출**입니다. 라우터는 답변 문장을 만들지 않고 `need_ops`, `need_documents`, confidence, 소스별 검색문만 JSON으로 반환합니다.

### 규칙 라우터는 삭제하지 않음

`OPS_PATTERNS`, `RAG_PATTERNS`, prototype, technical shape은 그대로 남아 있습니다.

- UI에서 `Rules-only`를 선택하면 기존 구조를 직접 시험할 수 있습니다.
- `LLM-primary`에서는 Ollama timeout, 빈 JSON, schema/boolean 오류, confidence 0.65 미만일 때 규칙 라우터가 fallback으로 동작합니다.
- RAG 안에서 `TEXT / TABLE / BOTH`를 고르는 것은 최상위 source 라우팅과 다른 문제이므로 `services/retrieval_mode.py`의 parser/rule 방식을 유지합니다.
- hybrid는 OPS 질문과 RAG 질문을 분리해 각각 실행한 뒤 선택 모델이 두 결과를 합칩니다.
- 짧은 후속 질문은 이전 경로·주제를 포함한 expanded question을 만들어 다시 판단합니다.

### 신규 30문항 × 3모델 실제 비교 (2026-08-11)

같은 30문항을 `Rules-only × 3모델`, `LLM-primary × 3모델`로 총 180회 실행했습니다. 엄격 QA는 **경로가 맞고 모범답안의 필수 수치·용어를 모두 포함한 경우만** 통과시켰습니다.

| 모델 | 라우팅 방식 | 경로 정확도 | 엄격 QA | 평균 라우터 | 평균 전체 응답 |
|------|-------------|-------------|---------|-------------|----------------|
| Llama 3.1 8B | Rules-only | 26/30 (86.7%) | 15/30 (50.0%) | 0 ms | 6.88초 |
| Llama 3.1 8B | LLM-primary | 28/30 (93.3%) | 16/30 (53.3%) | 802 ms | **6.84초** |
| Gemma 4 12B | Rules-only | 26/30 (86.7%) | 15/30 (50.0%) | 0 ms | 10.53초 |
| **Gemma 4 12B** | **LLM-primary** | **30/30 (100%)** | **20/30 (66.7%)** | 1,880 ms | 11.17초 |
| Mistral Nemo 12B | Rules-only | 26/30 (86.7%) | 12/30 (40.0%) | 0 ms | 7.89초 |
| Mistral Nemo 12B | LLM-primary | 29/30 (96.7%) | 15/30 (50.0%) | 1,023 ms | 7.58초 |

세 모델 모두 LLM-primary에서 경로 정확도와 QA가 개선됐습니다. Gemma를 기본값으로 정한 이유는 경로를 30/30 맞히고 OPS 6/6, hybrid 3/3을 통과했기 때문입니다. 다만 RAG는 Gemma도 7/16이어서 남은 병목은 주로 라우팅이 아니라 문서·표 검색입니다.

원본 질문과 전체 답변:

- `data/eval/quality_30_fresh_all_types.jsonl`
- `data/eval/quality_30_fresh_comparison.json`
- `data/eval/fresh_30_*.json`
- [docs/FRESH_30_EVALUATION.md](docs/FRESH_30_EVALUATION.md)

```powershell
# Rules-only와 LLM-primary를 세 모델 모두로 다시 비교
.\.venv\Scripts\python.exe scripts\compare_fresh_30.py

# 라우터 golden/held-out 비교
.\.venv\Scripts\python.exe tests\run_router_eval.py
```

### 균형형 100문항 × 3모델 최종 비교 (2026-08-11)

범위를 넓혀 **운항 DB 25, 텍스트 50, 표 15, 운항+문서 혼합 10**을 같은 조건으로 모델별 단독 실행했습니다. Gemma는 표 세트 검수에서 발견한 비표 문항 2개를 실제 표 문항으로 교체해 해당 2개만 재측정했으며, 최종 집계에는 질문별 결과를 한 번씩만 포함했습니다. 측정 PC는 RTX 5080 16GB, Core Ultra 7 265KF, RAM 96GB이며, 아래 응답시간은 이 환경 기준입니다.

| 모델 | 전체 엄격 QA | 운항 | 텍스트 | 표 | 혼합 | 평균 | P95 |
|------|-------------:|-----:|-------:|---:|-----:|-----:|----:|
| **Gemma 4 12B** | **63/100** | **21/25** | **25/50** | 7/15 | **10/10** | 10.89초 | 22.81초 |
| Llama 3.1 8B | 51/100 | 20/25 | 17/50 | **8/15** | 6/10 | **5.84초** | 14.15초 |
| Mistral Nemo 12B | 44/100 | 11/25 | 24/50 | 6/15 | 3/10 | 6.52초 | **13.58초** |

Gemma는 가장 정확하고 특히 혼합 질문에 강했습니다. Llama는 Gemma보다 약 46% 빠르지만 전체 엄격 QA가 12%p 낮았습니다. 텍스트는 대표 회의·정의 질문은 잘하지만 KR 제1편 세부 정의·조항이 약했고, 세 모델 모두 파일·페이지 단서가 없는 표 전체검색에서 반복 실패했습니다. 따라서 기본 모델은 Gemma를 유지하고, 다음 개선 우선순위는 모델 교체보다 TEXT/TABLE 하위 분류와 문서·페이지 메타데이터 필터입니다.

```powershell
# 100문항 재생성 및 모델별 재실행
.\.venv\Scripts\python.exe scripts\build_balanced_quality_100.py
.\.venv\Scripts\python.exe scripts\run_quality_30.py --questions data\eval\balanced_quality_100.jsonl --model gemma4:12b --output data\eval\balanced_quality_100_gemma4_12b.json
.\.venv\Scripts\python.exe scripts\run_quality_30.py --questions data\eval\balanced_quality_100.jsonl --model llama3.1:8b --output data\eval\balanced_quality_100_llama3.1_8b.json
.\.venv\Scripts\python.exe scripts\run_quality_30.py --questions data\eval\balanced_quality_100.jsonl --model mistral-nemo:12b --output data\eval\balanced_quality_100_mistral-nemo_12b.json
.\.venv\Scripts\python.exe scripts\summarize_balanced_quality_100.py
```

전체 문항·답변·시간과 3개 실제 답변 예시는 `data/eval/balanced_quality_100_report.md` 및 모델별 JSON에 있습니다. 자동 채점은 정답 핵심어 기반이므로 완전한 의미 정확도가 아닌 참고용 상한으로 해석합니다.

### 검색 보강 후 Gemma 단일 재검증 (2026-08-11)

임베딩과 Chroma 컬렉션은 다시 만들지 않고 검색 단계만 보강한 뒤, Gemma 4 12B 한 모델로 같은 균형형 100문항을 다시 실행했습니다. 정답 문서 강제 필터와 답변 캐시는 사용하지 않았고 LLM-primary 라우터부터 최종 답변까지 실제 UI와 같은 종단 경로로 측정했습니다.

| 구분 | 기존 | 검색 보강 후 | 변화 |
|------|-----:|-------------:|-----:|
| 전체 엄격 QA | 63/100 | **84/100** | **+21%p** |
| 운항 | 21/25 | **22/25** | +1 |
| 텍스트 | 25/50 | **41/50** | **+16** |
| 표 | 7/15 | **12/15** | **+5** |
| 혼합 | 10/10 | 9/10 | -1 |
| 평균 전체 응답 | 10.89초 | **9.67초** | **-1.22초** |
| P95 | 22.78초 | **22.64초** | -0.14초 |

최종 결과 파일은 `data/eval/balanced_quality_100_gemma4_12b_search_final.json`입니다. 전체 실행에서 일시적으로 흔들린 KR 직접 조항 2건은 명시 용어 우선순위를 고정한 뒤 별도 재검증 2/2를 통과했지만, 위 84/100 집계는 유리하게 수치를 덧붙이지 않고 실제 100회 실행 파일 그대로 유지했습니다.

보강 내용은 다음과 같습니다.

- 선택 문서 내부에서 조항 번호, 문서-local IDF, 질문 핵심 구문을 함께 사용해 희소 재정렬합니다.
- KR 제1편 고유 용어는 같은 조항 번호가 반복되는 타 편·적용지침보다 `kr_1_2025` 범위를 우선합니다.
- 정의·적용일·신청 주체·절차처럼 원문 한 조항으로 답할 수 있는 질문은 최상위 근거를 직접 인용해 불필요한 LLM 생성을 생략합니다.
- 조항 제목과 본문이 별도 청크이면 같은 문서·페이지의 관련 본문을 결합합니다.
- 표는 한글/영문 별칭과 literal 행 후보를 먼저 대조하고, 선택한 `table_id` 안에서 셀을 검증합니다.
- `510 ... 신청할 수 있는가?` 같은 질문의 `할 수 있다`를 봇 기능 안내로 오인하지 않도록 LLM chat 오분류 방어를 추가했습니다.

따라서 현재 기본값은 **Gemma + LLM-primary + 보강 검색**으로 유지합니다. 남은 병목은 회의 제출문서의 제목·의제 문구 보존, 일부 범용 표의 정확 셀 확정, OPS 숫자 반올림입니다.

## 기존 RAG 품질 개선

질문 하나하나를 임시로 고친 게 아니라, **자주 틀리던 패턴**을 막았습니다.

| 쉽게 말하면 | 자세한 내용 |
|-------------|-------------|
| **짧은 질문도 문서로** | `tcorr가 뭐야?`처럼 짧은 말도 안내만 하지 않고 규정·표 쪽에서 답합니다. |
| **표 숫자 함부로 안 말함** | 질문과 안 맞는 칸을 찍지 않습니다. 애매하면 “표를 확정하지 못했다”고 말합니다. |
| **파일·쪽 없는 표 질문** | `화물창 용접 다리 길이는?`처럼 PDF 이름을 안 적어도, 한글·영어 표기를 맞춰 표를 찾습니다. |
| **회의 주제 섞임 줄임** | IGC를 물었는데 MASS 이야기만 나오는 경우를 줄였습니다. |
| **표 제목 질문** | “이 표 제목이 뭐야?”는 표 머리글에서 바로 답합니다. |

### 임베딩을 다시 만들지 않는 계층 검색

본문·표 Chroma 컬렉션과 E5 임베딩은 그대로 두고, 질의 시 후보를 고르는 순서만 보강했습니다.

```text
TEXT: 질문 → source → 문서 집계·선택 → 선택 문서 안의 조항/쪽 → 문단
       └─ tcorr, MEPC 84/7/14, DNV-CG-0264 같은 식별자는 literal 후보도 함께 주입
       └─ 희소 검색은 전체 27만 청크가 아니라 선택 문서 안에서만 수행

TABLE: 질문 → table_id 고정 → 행 anchor → 라벨 행/다단 header path → 열 → 실제 cell 교차검증
       └─ 표·행·열이 한 셀에서 만나지 않으면 추측하지 않고 확인 불가로 답변
```

- 짧은 Rule 기호 정의는 전역 BM25를 거치지 않고 문서→조항 검색을 사용합니다. 비교가 필요하면 `MARITIME_RULE_GLOBAL_BM25=1`로 예전 전역 BM25를 잠시 켤 수 있습니다.
- 계층 본문 검색은 기본 ON이며, 비교 실험에서만 `MARITIME_TEXT_HIERARCHICAL=0`으로 끌 수 있습니다.
- `IMO 탄소집약도/CII`처럼 출처가 사실상 정해진 질문은 MEPC로 먼저 범위를 좁힙니다.
- 병합 헤더 표는 `판 및 국부 지지부재 → 항복/허용응력 → 값`처럼 2단 헤더를 복원합니다. 서로 다른 `table_id`의 행과 열은 섞지 않습니다.

따라서 이 변경에는 PDF 재처리, E5 재임베딩, Chroma 재빌드가 필요하지 않습니다. 기존 `full_corpus_715_v1`과 `full_corpus_715_tables_precise_v1`을 그대로 사용합니다.

실제 회귀 확인 결과:

| 질문 | 보강 후 결과 |
|------|--------------|
| `구조 규칙에서 쓰는 tcorr 기호는 어떤 두께를 뜻하지?` | 12편 117쪽을 찾아 `부식추가(corrosion addition), mm`로 답변 |
| `IMO 문서 기준으로 탄소집약도 등급 관리 요구사항을 요약해줘` | MEPC 84/6/1·84/6/2·84/6/21 범위에서 보고·등급 관리 근거 검색 |
| `14편 19쪽, 판과 국부 지지부재 허용응력은?` | 옆 열의 `5장 1절`이 아니라 `6장 4절 및 6장 5절` 선택 |

추가한 라우팅·검색 집중 회귀테스트는 **302개(추가 subtest 5개 포함) 통과**했습니다. 전체 회귀는 Windows 시스템 임시 폴더 대신 프로젝트 작업용 임시 폴더를 사용해 **416개 통과, 정책 테스트 2개 제외, 추가 subtest 5개 통과**를 확인했습니다. 제외한 2개는 현재 변경과 무관한 구조화 표 include 정책과 무조건 4-section 보정의 레거시 기대값이며, 누락된 선택형 보조 스크립트 수집 1건은 아래 제한사항에 남겨 두었습니다.

새 20문항 회귀셋은 chat 2, OPS 4, 회의 3, 정의 2, 규정 2, 표 5, hybrid 1, 범위 밖 1개로 구성했습니다. Gemma LLM-primary의 엄격 QA는 보강 전 **13/20(65%)**에서 보강 후 **20/20(100%)**로 올라갔고, route mismatch·빈 답변·실행 실패는 모두 0건이었습니다. 평균 전체 응답은 8.98초, 평균 router 호출은 1.86초였습니다. 이 수치는 해당 20문항 회귀셋의 결과이며 모든 임의 질문에 대한 보장은 아닙니다.

범위를 넓힌 50문항 실답변 스트레스 테스트에서는 route가 **50/50**으로 맞았고 빈 답변·실행 실패는 0건이었지만, 엄격 QA는 **25/50(50%)**였습니다. 채팅·OPS·본문·회의·hybrid와 파일/페이지가 특정된 표 질문은 **17/17**이었고, 파일/페이지를 생략한 `table_open`은 **8 PASS / 3 WEAK / 21 FAIL**, `table_reporting`은 0/1이었습니다. 평균 전체 응답은 10.76초, 평균 router 호출은 1.84초였습니다. 따라서 현재 가장 큰 병목은 최상위 라우팅이 아니라 **단서가 적은 범용 표 질문에서 올바른 table_id와 행·열을 찾는 단계**입니다.

별도의 100문항 경로 분류 세트에서는 Gemma LLM-primary가 **100/100**(fallback 0건, 평균 1.42초), Rules-only가 **99/100**(평균 0.24ms)이었습니다. 이 평가는 최종 답변이 아니라 `chat / ops / rag / hybrid` 경로만 채점한 결과입니다. Rules-only의 오분류는 일반 소개 질문 `소개해줘`를 `rag`로 보낸 1건이었습니다.

```powershell
.\.venv\Scripts\python.exe scripts/run_quality_30.py --model gemma4:12b
.\.venv\Scripts\python.exe scripts/run_quality_30.py --questions data\eval\hierarchical_retrieval_20.jsonl --model gemma4:12b --output data\eval\hierarchical_retrieval_20_gemma4_12b_after.json
.\.venv\Scripts\python.exe scripts/run_quality_30.py --questions data\eval\quality_50_open_mix.jsonl --model gemma4:12b --output data\eval\quality_50_open_mix_gemma4_12b_current.json
.\.venv\Scripts\python.exe scripts/run_route_suite.py --questions data\eval\suite_100_mixed.json --model gemma4:12b --output data\eval\suite_100_mixed_gemma4_12b_current.json
.\.venv\Scripts\python.exe scripts/run_route_suite.py --questions data\eval\suite_100_mixed.json --model gemma4:12b --rules-only --output data\eval\suite_100_mixed_rules_current.json
```

## 질문이 실제로 어떻게 흐르나

| 질문 | 경로 | 안에서 하는 일 |
|------|------|----------------|
| 현재 CII 알려줘 | ops | Tool → SQLite (`maritime.db`) |
| MSC 111 주요 결과 알려줘 | rag | Text Chroma (`full_corpus_715_v1`) |
| 선령별 탱크 검사 범위 알려줘 | rag | Table Chroma → 원본 표 crop |
| 우리 CII와 관련 규정 같이 알려줘 | hybrid | ops + rag를 나눠 실행한 뒤 합침 |

표 질문은 최상위 경로가 따로 있지 않습니다. 일단 `rag`로 간 다음, 안에서 본문/표 인덱스를 고릅니다.
답변의 `[n]`은 Evidence Table 행 번호이고, 아래쪽에 PDF에서 잘라 둔 표 crop이 붙습니다.

## 문서

| 문서 | 내용 |
|------|------|
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | 운항·문서 처리, 빌드/질의, 라우팅 |
| [docs/사용매뉴얼.md](docs/사용매뉴얼.md) | UI 실행·확인 절차 |
| [docs/TABLE_EMBEDDING_PIPELINE.md](docs/TABLE_EMBEDDING_PIPELINE.md) | 정밀 표 인덱스 만드는 법 |

## 코퍼스 규모 (현재 인덱스 기준)

| 구분 | 값 | 메모 |
|------|-----|------|
| PDF 전체 | **715** | `data/raw_pdfs`, `data/manifests/full_corpus_715.csv` |
| 본문 인덱스 | **714** | `full_corpus_715_v1` |
| 본문 제외 | **1** | `MSC 111-15-1 - WITHDRAWN` — 페이지 텍스트 0이라 청크가 안 나옴 |
| 표 인덱스 | **529** | `full_corpus_715_tables_precise_v1` |
| 쓸 만한 표 없음 | **179** | |
| TOC만/격리 | **7** | KR TOC 6 + ABS quarantine 1 |
| 표 추출 실패 | **0** | |

원본 PDF와 표 crop은 Git에는 포함되지 않지만, 이 PC의 `C:\Users\user\llmagent\data` 아래에는 PDF·중간 산출물·표 crop·본문/표 인덱스·운항 DB를 모두 로컬 복사했습니다. 검색과 재처리 모두 예전 바탕화면 폴더 없이 실행되며, 저장된 과거 절대경로는 실행 시 현재 프로젝트의 `data` 경로로 먼저 재해석합니다. 외부 절대경로 fallback은 기본적으로 차단되고, 호환이 꼭 필요할 때만 `MARITIME_ALLOW_EXTERNAL_DATA_PATHS=1`로 허용합니다.

```powershell
python scripts/audit_text_coverage.py
python scripts/audit_table_coverage.py
python scripts/inspect_rag_indexes.py --full
```

715개 PDF를 표 QA까지 전부 검증했다는 뜻은 아닙니다. 표가 잡힌 건 529개입니다.

본문+표를 같이 볼지(`BOTH`)는 기본 ON입니다. 끄려면 `MARITIME_RAG_DUAL=0`.

## 표 QA에서 막히던 것

| 증상 | 원인 | 고친 방식 |
|------|------|-----------|
| 표 질문만 느림 | 구버전 full-row Table BM25(~1.3GB) | 기본은 schema-only 슬림 BM25(~240MB). 끄려면 `MARITIME_TABLE_BM25=0` |
| 답은 있는데 crop 칸이 빔 | 표 질문이 meeting 경로로 새어 나감 | `table_qa`는 meeting 구조화 답변에서 제외 |
| MEPC 표 질문에서 「오류」만 | `analyze_query` UnboundLocalError | 지역 import 제거 |
| `[n]`만 보이고 근거 표가 없음 | Evidence Table / crop UI 없음 | Gradio에 Evidence + crop 갤러리 |
| 파일·쪽 없이 표만 물을 때 옆칸을 말함 | 질문과 안 맞는 칸을 그대로 확정 | 맞는 행인지 확인한 뒤, 애매하면 “확정 못함” |
| 한글 질문인데 표는 영어 | 같은 뜻 다른 표기(화물창 ↔ cargo hold 등) | 한글·영어를 같은 말로 묶어 검색 |

관련 코드: `services/rag_service.py`, `services/answer_ui.py`, `services/table_render.py`, `rag/scripts/rag_fast_mode.py`, `rag/scripts/table_qa_answer.py`, `rag/scripts/table_query_parser.py`, `rag/scripts/table_normalize_lib.py`, `rag/scripts/meeting_structured_answer.py`, `rag/scripts/meeting_category_profile.py`, `rag/scripts/bm25_index.py`.

## 현재 잘 안되는 부분과 해결 방향

신규 30문항과 계층 검색 20문항, 범용 표 중심 50문항은 라우팅과 최종 답변을 함께 채점했습니다. 20문항 회귀셋은 20/20이지만 50문항 스트레스 테스트는 25/50이므로, 아래 항목은 계속 관리해야 합니다.

| 현재 문제 | 확인된 예 | 해결 방향 |
|-----------|-----------|-----------|
| 파일·페이지 없는 범용 표 검색 | 정확한 파일·페이지가 있으면 3/3, 생략하면 `table_open` 8/32 | 전역 table catalog에서 질문의 행·열 literal/별칭을 먼저 찾고, table_id를 고정한 뒤 그 표 안에서 cell을 재검색합니다. 이 개선도 기존 table embedding을 재사용할 수 있습니다. |
| 회귀셋 밖 회의·규칙 표현 | 다른 회의차수, 긴 기호 목록, 서로 비슷한 KR 편 다수 | 질문별 기대 문서·페이지를 gold로 계속 추가하고 낮은 문서 신뢰도에서는 범위를 확인합니다. |
| 새로운 복합 표 형태 | 다중 행 요약, 병합 헤더, 정의형 위치 셀 외의 미확인 레이아웃 | table_id·라벨 행·다단 header·cell 교차검증을 새 표 유형으로 확대합니다. |
| OPS 모델이 도구 수치를 바꿈 | Mistral이 직전 항차 거리·CO2를 다른 값으로 생성 | 숫자 답변은 tool JSON을 검증한 뒤 템플릿으로 출력하거나 deterministic shortcut을 사용합니다. |
| 선택형 정밀 표 corpus 보조 스크립트 누락 | `49_vlm_table_pilot.py`, `53_snap_tatr_to_pdf.py` | 과거 원본을 복구하거나 `70_build_precise_table_corpus.py`가 현재 파이프라인만 사용하도록 정리합니다. |
| 레거시 전체테스트 정책 불일치 | table include 정책 1건, 4-section answer contract 1건 | 현재 인덱싱·답변 정책을 기준으로 테스트 기대값을 합의한 뒤 정리합니다. |

권장 개선 순서는 **범용 table catalog 검색 → 문서/페이지 gold 회귀셋 확대 → 새 표 레이아웃 추가 → OPS 숫자 검증 → 보조 corpus 스크립트 정리**입니다.

## 응답속도를 빠르게 하는 방법

### 바로 적용할 수 있는 방법

1. **속도가 우선이면 Llama 3.1 8B를 선택합니다.** 기존 30문항 비교 평균은 Llama 6.84초, Mistral 7.58초, Gemma 11.17초였습니다. 최신 검색 보강 후 Gemma 균형형 100문항 평균은 9.67초입니다(질문 구성이 달라 직접 모델 속도 비교값은 아닙니다).
2. **대화 중 모델을 자주 바꾸지 않습니다.** 다른 모델을 선택하면 Ollama가 GPU/메모리에 모델을 다시 올리는 첫 요청이 특히 느립니다.
3. **첫 질문은 warm-up으로 생각합니다.** UI는 RAG fast mode와 프로세스 내부 Chroma/e5 cache를 사용하므로 같은 프로세스의 두 번째 질문부터 대체로 빨라집니다.
4. **확실한 기존 표현만 시험할 때는 Rules-only를 사용할 수 있습니다.** 라우터 호출은 0ms지만 신규 표현과 hybrid 정확도는 낮아집니다.
5. **단순 운항 수치는 deterministic shortcut을 켤 수 있습니다.** 현재 상태·위치·속력·CII 같은 질문에서 답변 생성 호출을 줄입니다.

```powershell
cd C:\Users\user\llmagent
$env:MARITIME_OPS_DETERMINISTIC_SHORTCUTS="1"
.\.venv\Scripts\python.exe app.py
```

이 환경변수는 현재 PowerShell 창에만 적용됩니다. 모델별 순수 비교를 할 때는 다시 `0`으로 두어야 합니다.

본문만 주로 검색하고 표 recall 감소를 감수할 수 있으면 dual retrieval을 끌 수 있습니다.

```powershell
$env:MARITIME_RAG_DUAL="0"
.\.venv\Scripts\python.exe app.py
```

표 질문이나 `BOTH` 질문을 자주 쓴다면 기본값 `1`을 유지하는 편이 낫습니다.

### 코드에서 추가로 개선할 항목

- **스트리밍 답변:** 전체 시간이 같아도 첫 토큰부터 UI에 보여 체감 대기시간을 줄입니다.
- **hybrid 실행 최적화:** OPS를 deterministic tool 결과로 만들고 RAG와 병렬 실행한 뒤, 단순한 경우 두 번째 합성 호출을 생략합니다.
- **라우팅 cache:** 대화 상태와 무관한 동일 질문만 짧게 cache하면 반복 질문의 0.8~1.9초 router 비용을 줄일 수 있습니다.
- **작은 전용 router 실험:** 답변은 Gemma를 유지하고 더 작은 모델로 source-needs만 판단하는 A/B 평가를 추가합니다.
- **표 검색 후보 축소:** 파일·쪽·선급이 명시된 질문은 전체 벡터 검색 전에 metadata filter로 후보를 줄입니다.

현재 Rule 기호·문서코드 질문은 선택 문서 내부의 희소 재정렬을 사용하므로 예전 전역 BM25보다 검색 대기시간이 작습니다. 다만 새 Python 프로세스의 첫 질문은 E5/Chroma 로드 때문에 느릴 수 있고, 같은 UI 프로세스의 이후 질문부터 캐시 효과가 적용됩니다.

LLM-primary 자체가 항상 전체 응답을 크게 늦춘 것은 아닙니다. 같은 모델끼리 비교하면 Llama는 6.88→6.84초, Mistral은 7.89→7.58초였고 Gemma만 10.53→11.17초로 약 0.64초 늘었습니다. 실제 병목은 답변 모델 생성과 RAG/표 검색 쪽이 더 큽니다.

## 구조

```text
MaritimeOpsRAG/
├── app.py                 # Gradio UI
├── router/                # chat / ops / rag / hybrid
├── prompts/               # 경로별 프롬프트
├── services/              # orchestrator + 브리지
├── ops/                   # 운항 에이전트
├── rag/                   # 문서 RAG 스크립트
│   └── data/ → ../data
├── data/
│   ├── ho_data/           # 운항 Excel
│   ├── maritime.db        # load_hodata 후 생성
│   ├── raw_pdfs/          # PDF (이 PC에서는 로컬 복사본)
│   ├── manifests/
│   ├── eval/
│   └── processed/         # Chroma·청크·표 crop·중간 산출물 (로컬)
└── reports/output/
```

## `C:\Users\user\llmagent`에서 UI 실행

### 이 PC에서 바로 실행

현재 작업공간에는 `.venv`, 운항 DB, RAG 인덱스가 있으므로 보통 아래 순서만 실행하면 됩니다.

먼저 Ollama가 실행 중이고 모델이 설치돼 있는지 확인합니다.

```powershell
ollama list
```

연결 오류가 날 때만 별도 PowerShell 창에서 Ollama 서버를 실행합니다.

```powershell
ollama serve
```

새 PowerShell 창에서 UI를 실행합니다.

```powershell
cd C:\Users\user\llmagent
.\.venv\Scripts\python.exe app.py
```

브라우저에서 [http://127.0.0.1:7860](http://127.0.0.1:7860)을 엽니다. 콘솔에 표시되는 `0.0.0.0`은 서버 bind 주소이고 브라우저 접속 주소가 아닙니다.

이미 7860 포트에서 예전 UI가 실행 중이면 이전 터미널에서 `Ctrl+C`로 종료한 다음 위 명령을 다시 실행해야 변경된 모델·라우팅 선택지가 나타납니다.

```powershell
Get-NetTCPConnection -LocalPort 7860 -State Listen
```

정상 화면에서 확인할 항목:

- 라우터·답변 모델 기본값: `Gemma 4 12B (기본·권장)`
- 라우팅 방식 기본값: `LLM-primary (기본·권장)`
- 비교용 선택값: `Rules-only (비교용)`
- 상단 상태: 운항 DB `OK`, 문서 인덱스 `OK`
- 답변 경로 배너: `chat / ops / rag / hybrid`, 모델, 방식, 신뢰도

### 처음 설치할 때만

```powershell
git clone https://github.com/yoonmojeon/20260805.git
cd 20260805
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python ops/scripts/load_hodata.py

ollama pull gemma4:12b
ollama pull llama3.1:8b
ollama pull mistral-nemo:12b
ollama serve

python app.py
```

기존 `llmagent` 폴더에서는 clone과 전체 RAG 인덱스 재구축을 반복할 필요가 없습니다. DB가 없을 때만 `python ops/scripts/load_hodata.py`를 실행합니다.

확인 절차는 [docs/사용매뉴얼.md](docs/사용매뉴얼.md).

## 라우팅 세부 규칙

| 경로 | 예 | 데이터 |
|------|----|--------|
| chat | 너 누구야?, 뭐 할 수 있어 | 안내 템플릿 (`prompts/chat.py`) |
| ops | 운항 상태, CII, Noon/MRV | `maritime.db` + `prompts/ops.py` |
| rag | MEPC/MSC, 선급 Rule, 표 | Chroma + `prompts/rag.py` |
| hybrid | 우리 CII랑 MEPC 규제 같이 | ops·rag를 나눠 돌린 뒤 합침 |

- UI 강제 경로와 확실한 인사·감사·정체성·기능·메타 질문만 hard guard로 처리한다.
- 나머지 일반 질문은 선택 모델이 `need_ops`, `need_documents`를 의미적으로 판단한다.
- LLM이 문서 단서 없는 명확한 OPS 질문을 과도하게 hybrid로 분류하면 불필요한 RAG 호출만 제거한다. `attained와 required도 같이`처럼 한 소스의 두 값을 요청하는 표현은 dual-source 표지가 아니다.
- 두 source가 모두 필요하면 hybrid이며, 생성 query가 비거나 부정확하면 `split_hybrid_queries()`로 복구한다.
- 짧은 후속 질문은 이전 경로·주제를 펼친 `Expanded question`과 대화 상태를 모델에 전달한다.
- 기존 rule/prototype은 LLM 실패·저신뢰 fallback 및 진단용이다.
- UI에서 경로 강제, `운항만/문서만/둘 다로 다시` 가능.
- UI에서 선택한 Ollama 모델이 별도 호출로 routing과 answer를 모두 담당한다. 별도 `ROUTER_MODEL`은 없다.
- 라우터 평가: `python tests/run_router_eval.py`
- 전체 모델 비교: `python scripts/compare_models.py`

## 데이터

| 항목 | 위치 |
|------|------|
| 운항 Excel | `data/ho_data/*.xlsx` |
| 규정/회의 PDF | `data/raw_pdfs` |
| 문서 인덱스 | `data/processed/index/` (없으면 RAG는 안내만) |

PDF를 그대로 검색하지 않습니다. 대략:

1. PDF → 청크 (`data/processed/chunks/`)
2. 본문 임베딩 → `full_corpus_715_v1`
3. 표 파이프라인 → `full_corpus_715_tables_precise_v1`
4. (선택) BM25

715개 전처리는 수 시간 걸릴 수 있습니다. 표 쪽은 [docs/TABLE_EMBEDDING_PIPELINE.md](docs/TABLE_EMBEDDING_PIPELINE.md).

```powershell
(Get-ChildItem data\processed\chunks -Directory | Where-Object { Test-Path "$_\chunks.jsonl" }).Count
Get-Content data\processed\logs\preprocess_715.log -Tail 40
```

이어하기:

```powershell
cd rag
..\.venv\Scripts\Activate.ps1
powershell -ExecutionPolicy Bypass -File ..\scripts\build_rag_index.ps1
```

## 개별 앱

- 통합 UI: `python app.py` (권장)
- 운항만: `python ops/app_standalone.py`
