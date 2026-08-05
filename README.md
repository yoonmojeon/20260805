# MaritimeOpsRAG

`ship-data`(운항 SQLite + CII/MRV)와 `MaritimeRAG`(선급·IMO 문서 RAG)를 하나의 질의 UI로 결합한 프로젝트입니다.

질문에 따라 **운항 DB** 또는 **문서 벡터 인덱스**로 자동 라우팅합니다.

```
질문 → intent_router → ops (SQLite) 또는 rag (Chroma)
                   ↘ Gradio UI (app.py)
```

## 구조

```text
MaritimeOpsRAG/
├── app.py                 # 통합 Gradio UI
├── router/                # ops vs rag 의도 분류
├── services/              # ops/rag 브리지 + orchestrator
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

## 사용 매뉴얼

통합이 잘 되었는지 확인하는 절차·검증 질문은 [docs/사용매뉴얼.md](docs/사용매뉴얼.md)를 참고하세요.

## 빠른 시작

```powershell
cd C:\Users\user\MaritimeOpsRAG
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

## 라우팅 규칙

| 경로 | 예시 질문 | 데이터 |
|------|-----------|--------|
| ops | 현재 운항 상태, CII 등급, Noon/MRV 보고서 | `data/maritime.db` |
| rag | MEPC/MSC 동향, DNV·KR Rule, 표 질의 | Chroma `full_corpus_715_v1` / `kr_tables_v2` |

- 1차: 키워드 점수 (`router/intent_router.py`)
- 애매하면: Ollama JSON 분류 (UI에서 끄기 가능)
- UI에서 **운항 DB 강제 / 문서 RAG 강제** 가능

## 데이터 배치

| 항목 | 위치 |
|------|------|
| 운항 Excel | `data/ho_data/*.xlsx` (ship-data에서 복사됨) |
| 규정/회의 PDF | `data/raw_pdfs` → 바탕화면 `자료` 폴더 junction |
| 문서 인덱스 | `data/processed/index/` (아직 없으면 RAG 답변 대신 안내 메시지) |

### 문서 RAG 인덱스란?

PDF 파일을 바로 검색하는 게 아니라, 다음 순서로 **검색용 DB**를 만듭니다.

1. PDF → 페이지 이미지 → 레이아웃 탐지 → 청크(`data/processed/chunks/`)
2. 청크를 임베딩해 Chroma collection `full_corpus_715_v1` 생성
3. BM25 희소 인덱스 생성

이 작업이 끝나야 문서 질문에 답할 수 있습니다.  
현재 PC는 **CPU 전용**이라 715개 PDF 전처리는 수 시간~하루 이상 걸릴 수 있습니다.

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
