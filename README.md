# MaritimeOpsRAG

운항 SQLite와 선급·IMO 문서 RAG를 하나의 Gradio 채팅 UI에서 사용하는 프로젝트입니다. 질문에 따라 `chat`, `ops`, `rag`, `hybrid` 중 필요한 경로를 고르고, 문서 질문은 다시 `TEXT`, `TABLE`, `BOTH` 검색으로 나눕니다.

현재 권장 설정은 다음과 같습니다.

- 모델: **Gemma 4 12B** (`gemma4:12b`)
- 라우팅: **LLM-primary**
- 검색: **기존 Chroma 인덱스 + 계층 검색 + 문서 내부 희소검색**
- 비교용: **Rules-only 라우터 유지**
- 로컬 작업 경로: `C:\Users\user\llmagent`

## 빠른 실행

Ollama가 실행 중이고 필요한 모델이 설치되어 있는지 확인합니다.

```powershell
ollama list
ollama serve
```

다른 PowerShell에서 UI를 실행합니다.

```powershell
cd C:\Users\user\llmagent
.\.venv\Scripts\python.exe app.py
```

브라우저에서 [http://127.0.0.1:7860](http://127.0.0.1:7860)을 엽니다. `0.0.0.0`은 서버 바인드 주소이며 브라우저 접속 주소가 아닙니다.

7860 포트가 이미 사용 중이라는 오류가 나오면 이전에 실행한 UI 터미널에서 `Ctrl+C`로 종료한 뒤 다시 실행합니다.

```powershell
Get-NetTCPConnection -LocalPort 7860 -State Listen
```

처음 설치하는 환경에서는 다음 순서로 준비합니다. GitHub에는 대용량 원본 PDF와 로컬 인덱스 전체가 포함되지 않을 수 있으므로 `data/` 자산은 별도로 준비해야 합니다.

```powershell
git clone https://github.com/yoonmojeon/20260805.git
cd 20260805
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python ops/scripts/load_hodata.py
ollama pull gemma4:12b
python app.py
```

## 전체 처리 흐름

```text
사용자 질문
  → 데이터 경로 강제 여부 확인
  → LLM-primary 또는 Rules-only 상위 라우터
  → chat / ops / rag / hybrid
       ops    → maritime.db 조회
       rag    → TEXT / TABLE / BOTH 검색
       hybrid → ops와 rag를 각각 실행 후 통합
  → 검색 근거 검증
  → 선택한 Ollama 모델로 최종 답변
```

## 라우팅

### UI 선택 항목

| 항목 | 선택값 | 설명 |
|---|---|---|
| 데이터 경로 | 자동 라우팅 | 질문에 맞는 경로를 자동 선택합니다. |
| 데이터 경로 | 운항 DB 강제 | 상위 라우터를 건너뛰고 `ops`만 실행합니다. |
| 데이터 경로 | 문서 RAG 강제 | 상위 라우터를 건너뛰고 `rag`만 실행합니다. |
| 라우팅 방식 | **LLM-primary** | 선택 모델이 질문 의미를 보고 필요한 소스를 판단합니다. |
| 라우팅 방식 | Rules-only | 기존 패턴·점수·prototype으로 경로를 정합니다. |
| 모델 | **Gemma 4 12B** | 기본 모델입니다. |
| 모델 | Llama 3.1 8B / Mistral Nemo 12B | 속도·품질 비교용입니다. |

라우팅과 답변은 같은 선택 모델을 사용하지만 서로 다른 호출입니다. 라우터는 답변을 생성하지 않고 다음과 같은 소스 요구만 JSON으로 반환합니다.

```text
need_ops=false, need_documents=false → chat
need_ops=true,  need_documents=false → ops
need_ops=false, need_documents=true  → rag
need_ops=true,  need_documents=true  → hybrid
```

### LLM-primary 보호 장치

LLM 판단만 그대로 사용하지 않고 다음 보호 장치를 함께 적용합니다.

- 사용자가 UI에서 강제한 경로를 최우선으로 적용합니다.
- 명확한 인사·감사·정체성 질문은 간단한 hard guard로 처리합니다.
- 조항 번호, 선급 규정, 검사, 표 번호 등 강한 기술 단서가 있으면 일반 채팅 오분류를 막습니다.
- timeout, 빈 JSON, schema 오류, 낮은 confidence에서는 Rules-only 결과로 fallback합니다.
- 짧은 후속 질문은 이전 질문의 경로와 주제를 포함한 확장 질문으로 다시 판별합니다.
- `hybrid`는 운항 질문과 문서 질문을 분리해서 검색한 뒤 두 근거를 합칩니다.

Rules-only 라우터는 삭제하지 않았습니다. UI에서 기존 방식과 새 방식을 같은 질문으로 비교할 수 있습니다.

관련 코드:

- `router/intent_router.py`: LLM-primary 판단, fallback, 기술 질문 보호
- `router/cues.py`: 운항·문서 단서
- `services/routing_options.py`: UI 라우팅 방식
- `services/hybrid_service.py`: 운항·문서 결과 통합

## 검색

### TEXT / TABLE / BOTH 분류

상위 라우터가 `rag` 또는 `hybrid`를 선택하면 질문 형태를 한 번 더 분석합니다.

| 검색 모드 | 주요 질문 | 검색 대상 |
|---|---|---|
| TEXT | 정의, 취지, 조항, 회의 결과, 규정 설명 | 본문 Chroma `full_corpus_715_v1` |
| TABLE | 행·열·셀, 검사주기, 두께, 허용값, 단위 조건 | 표 Chroma `full_corpus_715_tables_precise_v1` |
| BOTH | 본문 설명과 표의 정확한 값이 모두 필요한 질문 | TEXT와 TABLE을 함께 검색 |

단순히 질문에 `행`이나 숫자가 있다고 TABLE로 보내지 않습니다. 실제 행·열·표·단위 조건이 있는지 파싱하고, 조항 설명이나 규정 정의는 TEXT를 우선합니다.

### TEXT 검색 순서

```text
질문 분석
  → 문서 코드·조항 번호·핵심 구문 추출
  → 기존 임베딩으로 넓은 후보 검색
  → 관련 문서 4~6개로 범위 축소
  → 선택 문서 내부에서 희소검색
  → 조항·문서·구문 일치 점수로 재정렬
  → 직접 규정 답변 또는 Gemma 근거 답변
```

주요 보강 내용은 다음과 같습니다.

- `MEPC 84/6/1`, `DNV-CG-0264`, `510절`, `tcorr` 같은 식별자는 의미 유사도뿐 아니라 문자열 일치도도 사용합니다.
- KR 규정 용어와 편·연도·조항 단서를 이용해 우선 문서를 정합니다.
- 전체 코퍼스를 다시 훑기보다 선택 문서 안에서 문서-local IDF와 희소검색을 수행합니다.
- 한국어 조사·어미를 정리해 `건조계약일의`와 `건조계약일`이 같은 핵심어로 검색되도록 합니다.
- 조항 제목과 본문이 분리된 경우 같은 문서·페이지의 인접 근거를 결합합니다.
- 정의·주체·절차·수치처럼 답이 명확한 질문은 검색 근거에서 직접 답해 불필요한 LLM 생성을 줄입니다.

이 검색 보강은 기존 임베딩과 Chroma 컬렉션을 그대로 사용하므로 **재임베딩이 필요하지 않습니다.**

### TABLE 검색 순서

```text
질문에서 표 주제·행·열·재료·조건·단위 추출
  → 관련 table_id 후보 선택
  → header path와 행 이름으로 재정렬
  → 선택한 표 내부에서 실제 셀 교차 검증
  → 근거 표와 crop을 답변에 연결
```

표 검색은 한국어·영어 별칭, ASCII 단위, 재료명, 시험 종류를 정규화합니다. 질문에 행 이름이 그대로 있으면 벡터 점수보다 literal 행 일치를 우선합니다. 조건에 맞는 셀을 찾지 못하면 비슷한 값을 추측하지 않고 확인 불가로 답하도록 제한합니다.

관련 코드:

- `services/retrieval_mode.py`: TEXT / TABLE / BOTH 판별
- `rag/scripts/retrieval_query_analysis.py`: 문서·조항·주제 분석
- `rag/scripts/retrieval_search.py`: 계층 검색과 재정렬
- `rag/scripts/rule_lookup_answer.py`: 직접 규정 근거 답변
- `rag/scripts/table_query_parser.py`: 표 질문 슬롯 분석
- `rag/scripts/table_schema_retrieval.py`: table_id·행·열 검색
- `rag/scripts/evidence_planner.py`: 근거 선택과 안전한 캐시

### 주요 검색 설정

| 환경변수 | 기본값 | 용도 |
|---|---:|---|
| `MARITIME_TEXT_HIERARCHICAL` | `1` | 문서 우선 계층 검색 |
| `MARITIME_RAG_DUAL` | `1` | TEXT와 TABLE 결합 검색 |
| `MARITIME_RULE_GLOBAL_BM25` | `0` | 비교용 전역 Rule BM25 |
| `MARITIME_OPS_DETERMINISTIC_SHORTCUTS` | `0` | 단순 운항 수치 답변에서 LLM 생략 |

## 데이터와 인덱스

| 항목 | 위치 또는 이름 |
|---|---|
| 운항 DB | `data/maritime.db` |
| 운항 원본 | `data/ho_data/*.xlsx` |
| 원본 PDF | `data/raw_pdfs/` |
| 본문 인덱스 | `full_corpus_715_v1` |
| 표 인덱스 | `full_corpus_715_tables_precise_v1` |
| 평가셋 | `data/eval/` |

현재 `C:\Users\user\llmagent`에는 실행에 필요한 DB·인덱스·PDF가 로컬로 연결되어 있습니다. 기본 설정에서는 다른 프로젝트 폴더를 참조하지 않습니다. 외부 데이터 경로 fallback은 `MARITIME_ALLOW_EXTERNAL_DATA_PATHS=1`을 명시한 경우에만 허용합니다.

## 현재 검증 결과

검색 보강 후 `gemma4:12b`, LLM-primary, 100문항으로 실제 실행한 결과입니다. 정답 문서 강제 필터와 답변 캐시는 사용하지 않았습니다.

| 유형 | 기존 | 검색 보강 후 | 평균 응답 |
|---|---:|---:|---:|
| 운항 25문항 | 21 | 22 | 11.10초 |
| 텍스트 50문항 | 25 | 41 | 3.92초 |
| 표 15문항 | 7 | 12 | 16.08초 |
| 혼합 10문항 | 10 | 9 | 25.21초 |
| **전체 100문항** | **63** | **84** | **9.67초** |

- 전체 정확도: 63% → **84%**
- 평균 응답: 10.89초 → **9.67초**
- 라우팅 mismatch: **0건**
- 결과 파일: `data/eval/balanced_quality_100_gemma4_12b_search_final.json`
- 전체 회귀: **416 passed**, 기존 정책 비교 2건 deselected

재실행 예시:

```powershell
.\.venv\Scripts\python.exe scripts\run_quality_30.py `
  --questions data\eval\balanced_quality_100.jsonl `
  --model gemma4:12b `
  --output data\eval\balanced_quality_100_gemma4_12b_search_final.json
```

## 남은 한계

- 회의 문서 제목·세션 번호·원문 파일명 보존이 약한 질문
- 복합 재료·행·열 조건이 동시에 들어가는 일부 표 셀
- 날짜 형식, 소수점 자릿수, CII 반올림 같은 정밀 출력
- 현재 항차와 이전 항차를 함께 묻는 혼합 질문
- 표와 혼합 질문의 평균 응답시간

다음 우선순위는 문서 메타데이터 전용 검색, 표 셀 좌표 인덱스, 항차 시간축 구분입니다.

## 프로젝트 구조와 상세 문서

```text
app.py        Gradio 통합 UI
router/       chat / ops / rag / hybrid 상위 라우터
services/     경로별 실행과 결과 통합
ops/          운항 DB 질의
rag/          본문·표 검색과 근거 답변
data/         운항 DB, PDF, 인덱스, 평가셋
docs/         상세 설계와 사용 문서
```

- [상세 아키텍처](docs/ARCHITECTURE.md)
- [사용 매뉴얼](docs/사용매뉴얼.md)
- [표 임베딩 파이프라인](docs/TABLE_EMBEDDING_PIPELINE.md)
- [평가 상세](data/eval/balanced_quality_100_report.md)
