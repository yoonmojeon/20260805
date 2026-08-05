# MaritimeOpsRAG

`ship-data`(운항 SQLite + CII/MRV)와 `MaritimeRAG`(선급·IMO 문서 RAG)를 하나의 질의 UI로 결합한 프로젝트입니다.

질문에 따라 **안내 / 운항 DB / 문서 벡터 인덱스**로 자동 라우팅합니다.

```
질문 → intent_router → chat | ops | rag | hybrid
                   ↘ Gradio UI (app.py)
```

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
