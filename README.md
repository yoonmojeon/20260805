# MaritimeOpsRAG

`ship-data`(운항 SQLite + CII/MRV)와 `MaritimeRAG`(선급·IMO 문서 RAG)를 하나의 질의 UI로 결합한 프로젝트입니다.

질문에 따라 **안내 / 운항 DB / 문서 벡터 인덱스**로 자동 라우팅합니다.

```
질문 → intent_router → chat | ops | rag | hybrid
                   ↘ Gradio UI (app.py)
```

저장소: [github.com/yoonmojeon/20260805](https://github.com/yoonmojeon/20260805)

## 문서 (공부·인수인계)

| 문서 | 내용 |
|------|------|
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | 운항·비정형 처리, 빌드/질의 타임, 라우팅 전체 설명 (상세) |
| [docs/사용매뉴얼.md](docs/사용매뉴얼.md) | UI 실행·통합 확인 절차 |
| [docs/TABLE_EMBEDDING_PIPELINE.md](docs/TABLE_EMBEDDING_PIPELINE.md) | 정밀 표 임베딩 파이프라인 재현 |

핵심만 요약하면:

## Corpus coverage (local runtime, measured)

| 구분 | 값 | 근거 |
|------|-----|------|
| Corpus PDFs | **715** | `data/raw_pdfs` rglob + `data/manifests/full_corpus_715.csv` |
| Text-indexed PDFs | **714** | `full_corpus_715_v1` manifest `doc_ids` |
| Text missing | **1** | `MSC 111-15-1 - WITHDRAWN (Italy).pdf` — 1,759B stub, page text 0, chunks.jsonl empty. Index intentionally skipped (`missing_chunks_doc_ids`). Not a silent bug; re-embed yields 0 chunks. |
| Table-indexed PDFs | **529** | `full_corpus_715_tables_precise_v1` |
| No usable table detected | **179** | precise `missing_table_documents` + empty `tables.jsonl` |
| Filtered / empty (pseudo TOC or quarantine) | **7** | 6 KR TOC-only + 1 ABS quarantined |
| Table extraction failed | **0** | audit classification |
| Coverage | **~74%** of 715 have table chunks |

진단:

```powershell
python scripts/audit_text_coverage.py
python scripts/audit_table_coverage.py
python scripts/inspect_rag_indexes.py --full
```

`715 PDFs 전체를 table QA로 완전히 검증했다`는 사실이 아닙니다. 표 인덱스는 529문서입니다.

BOTH retrieval 기본값: `MARITIME_RAG_DUAL` unset → **ON** (`services/rag_service.dual_retrieval_enabled`). `=0`이면 single-index fallback.

## 최근 업데이트 (표 QA · UI)

통합 Gradio(`app.py`)에서 표 질문이 느려지거나 crop/오류만 보이던 문제를 고쳤습니다.

| 증상 | 원인 | 조치 |
|------|------|------|
| 표 질문만 1분+ / 멈춤 | Table BM25 pickle(~1.3GB) 디스크 로드·재빌드 | 대화형 기본은 dense-only. 필요 시 `MARITIME_TABLE_BM25=1` |
| 답·Evidence는 나오는데 표 crop 빈칸 | `table_qa`가 meeting 경로로 잘못 들어가 `table_id`/`crop_path` 유실 | `uses_structured_meeting_answer`가 table_qa를 제외 |
| MEPC 등 「오류」만 표시 | `analyze_query` 지역 import → `UnboundLocalError` | 모듈 import만 사용, 표 질의는 meeting merge 스킵 |
| `[n]` 의미 불명확 | 인용 번호만 있고 Evidence Table 미표시 | Gradio에 Evidence Table + PDF crop gallery |

UI는 MaritimeRAG Streamlit과 같이 **원본 표 crop 이미지**를 우선 보여 줍니다(Markdown 표 재구성이 아님).

관련 코드: `services/rag_service.py`, `services/answer_ui.py`, `services/table_render.py`, `rag/scripts/rag_fast_mode.py`, `rag/scripts/meeting_category_profile.py`, `rag/scripts/bm25_index.py`.

## 구조

```text
MaritimeOpsRAG/
├── app.py                 # 통합 Gradio UI
├── router/                # chat / ops / rag 의도 분류
├── prompts/               # 경로별 시스템 프롬프트 (chat/ops/rag/router)
├── services/              # chat/ops/rag 브리지 + orchestrator
├── ops/                   # ship-data 에이전트
├── rag/                   # MaritimeRAG 스크립트/설정
│   └── data/ → ../data    # (junction)
├── data/
│   ├── ho_data/           # 운항 Excel
│   ├── maritime.db        # load_hodata 후 생성
│   ├── raw_pdfs/          # 바탕화면 '자료' junction
│   ├── manifests/         # 715-PDF corpus 목록
│   ├── eval/
│   └── processed/         # Chroma/청크 (로컬 구축)
└── reports/output/        # Noon/MRV docx
```

## 빠른 시작

```powershell
git clone https://github.com/yoonmojeon/20260805.git
cd 20260805
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 운항 DB
python ops/scripts/load_hodata.py

# Ollama
ollama pull llama3.1:8b
ollama serve

# 통합 UI
python app.py
```

브라우저: http://127.0.0.1:7860

UI 검증 절차는 [docs/사용매뉴얼.md](docs/사용매뉴얼.md)를 참고하세요.

## 라우팅 규칙

| 경로 | 예시 질문 | 데이터 / 프롬프트 |
|------|-----------|-------------------|
| chat | 너 누구야?, 안녕, 뭐 할 수 있어 | 고정 안내 (`prompts/chat.py`) |
| ops | 현재 운항 상태, CII 등급, Noon/MRV 보고서 | `data/maritime.db` + `prompts/ops.py` |
| rag | MEPC/MSC 동향, DNV·KR Rule, 표 질의 | Chroma + `prompts/rag.py` 정체성 |
| hybrid | 우리 CII랑 MEPC 규제 같이 | ops+rag 결과를 출처별로 합침 |

- 1차: 화행/슬롯 + 키워드 점수 (`router/`)
- 점수 0/0 또는 잡담 → **chat** (문서 RAG로 보내지 않음)
- 명시적 dual/비교 프레임만 **hybrid**. 근접 점수는 되묻기
- hybrid는 ops/rag 질의를 나눠 실행
- 짧은 후속 질문은 이전 주제·경로를 이어 질문을 펼침
- 애매하면 프로토타입 투표, 그다음 Ollama JSON (UI에서 끄기 가능)
- UI에서 경로 강제 및 `운항만/문서만/둘 다로 다시` 가능
- 라우터 평가: `python tests/run_router_eval.py`

## 데이터 배치

| 항목 | 위치 |
|------|------|
| 운항 Excel | `data/ho_data/*.xlsx` (ship-data에서 복사됨) |
| 규정/회의 PDF | `data/raw_pdfs` → 바탕화면 `자료` 폴더 junction |
| 문서 인덱스 | `data/processed/index/` (아직 없으면 RAG 답변 대신 안내 메시지) |

### 문서 RAG 인덱스란?

PDF 파일을 바로 검색하는 게 아니라, 다음 순서로 **검색용 DB**를 만듭니다.

1. PDF → 레이아웃 탐지 → 본문 청크(`data/processed/chunks/`)
2. 본문 임베딩 → Chroma `full_corpus_715_v1`
3. (별도) 표 정밀 파이프라인 → `full_corpus_715_tables_precise_v1`
4. (선택) BM25 희소 인덱스

이 작업이 끝나야 문서 질문에 답할 수 있습니다.  
715개 PDF 전처리는 환경에 따라 수 시간 이상 걸릴 수 있습니다.
자세한 표 파이프라인은 [docs/TABLE_EMBEDDING_PIPELINE.md](docs/TABLE_EMBEDDING_PIPELINE.md)를 보세요.

진행 로그:

```text
data/processed/logs/preprocess_715.log
```

진행 확인:

```powershell
(Get-ChildItem data\processed\chunks -Directory | Where-Object { Test-Path "$_\chunks.jsonl" }).Count
Get-Content data\processed\logs\preprocess_715.log -Tail 40
```

재실행(중단 후 이어하기):

```powershell
cd rag
..\.venv\Scripts\Activate.ps1
powershell -ExecutionPolicy Bypass -File ..\scripts\build_rag_index.ps1
```

## 개별 앱

- 운항만: `python ops/app_standalone.py` (경로 패치된 config/DB 사용)
- 문서 UI만: `cd rag` 후 `streamlit run scripts/15_rag_ui.py`
