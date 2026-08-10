# MaritimeOpsRAG

운항 SQLite(`ship-data`)와 선급·IMO 문서 RAG(`MaritimeRAG`)를 한 Gradio UI에서 쓰는 프로젝트입니다.
질문이 오면 라우터가 **안내 / 운항 DB / 문서 인덱스 / 둘 다** 중 어디로 보낼지 고릅니다.

```
질문 → intent_router → chat | ops | rag | hybrid
                   ↘ Gradio UI (app.py)
```

저장소: [github.com/yoonmojeon/20260805](https://github.com/yoonmojeon/20260805)

## 최근 업데이트 (무엇이 나아졌나)

질문 하나하나를 임시로 고친 게 아니라, **자주 틀리던 패턴**을 막았습니다.

| 쉽게 말하면 | 자세한 내용 |
|-------------|-------------|
| **짧은 질문도 문서로** | `tcorr가 뭐야?`처럼 짧은 말도 안내만 하지 않고 규정·표 쪽에서 답합니다. |
| **표 숫자 함부로 안 말함** | 질문과 안 맞는 칸을 찍지 않습니다. 애매하면 “표를 확정하지 못했다”고 말합니다. |
| **파일·쪽 없는 표 질문** | `화물창 용접 다리 길이는?`처럼 PDF 이름을 안 적어도, 한글·영어 표기를 맞춰 표를 찾습니다. |
| **회의 주제 섞임 줄임** | IGC를 물었는데 MASS 이야기만 나오는 경우를 줄였습니다. |
| **표 제목 질문** | “이 표 제목이 뭐야?”는 표 머리글에서 바로 답합니다. |

30개 샘플 질문으로 맞춰 본 결과(키워드가 답에 들어갔는지):

| 모델 | 맞음 | 틀림 | 메모 |
|------|------|------|------|
| `llama3.1:8b` | **27**/30 | 3 | 기본으로 쓰기 좋음(가장 빠름) |
| `gemma4:12b` | **27**/30 | 3 | 문장이 조금 더 매끄러운 편 |
| `mistral-nemo:12b` | **26**/30 | 4 | 비슷하고 조금 더 느림 |

아직 틀리는 건 주로 **어느 파일인지 안 알려 준 표 질문** 몇 개입니다. (예: 평가 방법 SP-A, 용접 각장 4.5mm)

```powershell
.\.venv\Scripts\python.exe scripts/run_quality_30.py --model llama3.1:8b
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

## 빠른 시작

```powershell
git clone https://github.com/yoonmojeon/20260805.git
cd 20260805
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python ops/scripts/load_hodata.py

ollama pull llama3.1:8b
ollama serve

python app.py
```

브라우저: http://127.0.0.1:7860  
UI에서 답변 모델(`llama3.1:8b` / `gemma4:12b` / `mistral-nemo:12b`)을 고를 수 있습니다.  
확인 절차는 [docs/사용매뉴얼.md](docs/사용매뉴얼.md).

## 라우팅

| 경로 | 예 | 데이터 |
|------|----|--------|
| chat | 너 누구야?, 뭐 할 수 있어 | 안내 템플릿 (`prompts/chat.py`) |
| ops | 운항 상태, CII, Noon/MRV | `maritime.db` + `prompts/ops.py` |
| rag | MEPC/MSC, 선급 Rule, 표 | Chroma + `prompts/rag.py` |
| hybrid | 우리 CII랑 MEPC 규제 같이 | ops·rag를 나눠 돌린 뒤 합침 |

- 문장 전체를 맞추지 않고 단서·슬롯만 본다.
- 단서 없으면 chat에서 되묻는다. 문서 RAG로 추측 전송하지 않는다.
- hybrid는 “둘 다 / 비교”처럼 명시될 때만.
- 짧은 후속 질문은 이전 경로·주제를 이어 붙인 뒤 다시 분류한다.
- UI에서 경로 강제, `운항만/문서만/둘 다로 다시` 가능.
- UI에서 Ollama 답변 모델 3종 선택 가능.
- 라우터 평가: `python tests/run_router_eval.py`

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
