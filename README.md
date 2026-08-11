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

## 기존 RAG 품질 개선

질문 하나하나를 임시로 고친 게 아니라, **자주 틀리던 패턴**을 막았습니다.

| 쉽게 말하면 | 자세한 내용 |
|-------------|-------------|
| **짧은 질문도 문서로** | `tcorr가 뭐야?`처럼 짧은 말도 안내만 하지 않고 규정·표 쪽에서 답합니다. |
| **표 숫자 함부로 안 말함** | 질문과 안 맞는 칸을 찍지 않습니다. 애매하면 “표를 확정하지 못했다”고 말합니다. |
| **파일·쪽 없는 표 질문** | `화물창 용접 다리 길이는?`처럼 PDF 이름을 안 적어도, 한글·영어 표기를 맞춰 표를 찾습니다. |
| **회의 주제 섞임 줄임** | IGC를 물었는데 MASS 이야기만 나오는 경우를 줄였습니다. |
| **표 제목 질문** | “이 표 제목이 뭐야?”는 표 머리글에서 바로 답합니다. |

최종 quality-30에서 공통으로 남은 실패는 주로 **파일·쪽 정보가 없는 표 질문**입니다. 특히 첫 정기검사 reporting 범위와 용접 각장 4.5mm 질의는 세 모델 모두 검색 근거를 확정하지 못했습니다.

```powershell
.\.venv\Scripts\python.exe scripts/run_quality_30.py --model gemma4:12b
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

## 코퍼스 규모 (이 PC 기준)

| 구분 | 값 | 메모 |
|------|-----|------|
| PDF 전체 | **715** | `data/raw_pdfs`, `data/manifests/full_corpus_715.csv` |
| 본문 인덱스 | **714** | `full_corpus_715_v1` |
| 본문 제외 | **1** | `MSC 111-15-1 - WITHDRAWN` — 페이지 텍스트 0이라 청크가 안 나옴 |
| 표 인덱스 | **529** | `full_corpus_715_tables_precise_v1` |
| 쓸 만한 표 없음 | **179** | |
| TOC만/격리 | **7** | KR TOC 6 + ABS quarantine 1 |
| 표 추출 실패 | **0** | |

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

신규 30문항은 라우팅과 최종 답변을 함께 채점했습니다. 현재 Gemma의 라우팅은 30/30이지만 최종 QA는 20/30이므로, 앞으로는 라우터보다 아래 항목을 먼저 고치는 편이 효과가 큽니다.

| 현재 문제 | 확인된 예 | 해결 방향 |
|-----------|-----------|-----------|
| 회의 문서의 정확한 안건·주제 검색 | MEPC 84 의제, MSC 111 IGC 개정, 암모니아 연료 논의 | 질문별 기대 문서·페이지를 gold로 만들고 문서명/agenda item metadata boost를 추가합니다. |
| 규칙·정의 검색이 다른 문서로 샘 | `tcorr`, IMO CII 요구, DNV 자율운항 guidance | 선급·IMO·문서종류를 hard filter하고 direct-clause가 없으면 답을 확정하지 않도록 합니다. |
| 표에서 옆 행·열을 선택 | 시험편 2개, 기관실 격벽 위치, 허용응력 절 번호 | 파일·쪽·table_id를 먼저 고정하고 header path + row key가 동시에 맞는 셀만 답하도록 합니다. |
| OPS 모델이 도구 수치를 바꿈 | Mistral이 직전 항차 거리·CO2를 다른 값으로 생성 | 숫자 답변은 tool JSON을 검증한 뒤 템플릿으로 출력하거나 deterministic shortcut을 사용합니다. |
| 기능 질문 답변이 간접적임 | “운항과 규정 문서를 둘 다 찾나?”에 되묻기 | capability chat mode에 직접적인 예/아니오 답변 템플릿을 추가합니다. |
| 선택형 정밀 표 corpus 보조 스크립트 누락 | `49_vlm_table_pilot.py`, `53_snap_tatr_to_pdf.py` | 과거 원본을 복구하거나 `70_build_precise_table_corpus.py`가 현재 파이프라인만 사용하도록 정리합니다. |

권장 개선 순서는 **표 셀 정합 → 문서/페이지 gold 회귀셋 → OPS 숫자 검증 → capability 답변 → 보조 corpus 스크립트 정리**입니다. 라우팅 프롬프트를 더 복잡하게 만드는 것보다 이 순서가 최종 정답률에 직접적인 효과가 있습니다.

## 응답속도를 빠르게 하는 방법

### 바로 적용할 수 있는 방법

1. **속도가 우선이면 Llama 3.1 8B를 선택합니다.** 신규 평가 평균은 Llama 6.84초, Mistral 7.58초, Gemma 11.17초였습니다.
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
│   ├── raw_pdfs/          # PDF (보통 junction)
│   ├── manifests/
│   ├── eval/
│   └── processed/         # Chroma·청크 (로컬)
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
