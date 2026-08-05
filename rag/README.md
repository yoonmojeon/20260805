# MaritimeRAG

MaritimeRAG는 선급 규칙과 IMO 회의 문서를 로컬에서 전처리하고 검색·요약하는 RAG 프로젝트입니다. PDF 레이아웃 분석, 본문·표 청크 생성, 다국어 벡터 검색과 BM25 하이브리드 검색, 근거 기반 답변, Streamlit UI를 제공합니다.

> 원본 선급 규칙 PDF와 일부 IMO 자료에는 저작권 또는 배포 제한이 있을 수 있습니다. 원본 문서, 모델 가중치, 생성된 인덱스는 Git 저장소에 포함하지 않습니다.

## 현재 구현 범위

- PDF 페이지 렌더링과 YOLOv10 기반 문서 레이아웃 탐지
- 텍스트 블록 병합, 표·그림 crop, 본문·표·그림 청크 생성
- multilingual E5 계열 임베딩과 Chroma 벡터 인덱스
- BM25 + 벡터 검색, 문서 메타데이터 재정렬, 선급사 필터
- IMO MSC/MEPC 회의 결과 및 동향 요약
- DNV/KR/ABS/LR Rule·Guidance 검색
- 표 schema·summary·markdown·row 기반 표 질의
- 빠른 모드와 정확 모드, Ollama 기반 로컬 답변 생성
- Recall@k, 표 검색, 응답 지연시간 및 회귀 평가
- 리소스 캐시와 스트리밍 답변을 적용한 Streamlit UI

## Corpus 현황

`data/manifests/full_corpus_715.csv` 기준 실제 PDF 구성은 다음과 같습니다.

| 출처 | 문서 수 |
|---|---:|
| MEPC | 203 |
| MSC | 106 |
| KR | 73 |
| DNV | 318 |
| ABS | 14 |
| LR | 1 |
| 합계 | 715 |

주요 collection ID:

- `full_corpus_715_v1`: 715개 PDF의 IMO 회의 자료와 선급 Rule/Guidance 본문 통합 검색
- `kr_tables_v2`: KR 22개 문서의 좌표 기반 구조화 표 전용 인덱스(현재 표 QA 기본값)
- `kr_tables_v1`: 기존 공백 기반 표 구조화 인덱스(비교·복구용)
- `pilot_100`: 100개 문서 파일럿 인덱스

기존 `full_corpus_v1`, `full_corpus`, `kr_tables`, `kr_tables_v1`은 비교·복구용으로 로컬에 보존합니다. UI 기본값은 일반 검색 `full_corpus_715_v1`, 표 검색 `kr_tables_v2`입니다.

Manifest와 평가 데이터는 Git에 포함되지만 PDF, 전처리 산출물, Chroma/BM25 인덱스는 로컬에서 준비하거나 별도 저장소에서 받아야 합니다.

## 권장 환경

- Windows 10/11 또는 Linux
- Python 3.11 권장(현재 개발 환경: Python 3.11.9)
- NVIDIA GPU 권장
  - 레이아웃 탐지와 대규모 임베딩은 CPU에서도 가능하지만 오래 걸립니다.
- Ollama
  - 기본 답변 모델: `llama3.1:8b`
  - 기본 API 주소: `http://localhost:11434`
- 레이아웃 모델
  - `models/layout/yolov10m_doclaynet.pt`
  - Hugging Face의 DocLayNet 호환 YOLOv10 가중치를 사용합니다.

PDF 렌더링은 PyMuPDF를 사용하므로 현재 파이프라인에는 별도 Poppler 설치가 필요하지 않습니다.

## 설치

PowerShell 예시:

```powershell
git clone https://github.com/minkchoii/MaritimeRAG.git
cd MaritimeRAG

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ollama 설치 후 기본 모델을 준비합니다.

```powershell
ollama pull llama3.1:8b
ollama serve
```

Ollama가 이미 백그라운드에서 실행 중이면 `ollama serve`를 다시 실행할 필요가 없습니다. UI에서 다른 설치된 모델을 지정할 수도 있습니다.

## 데이터 배치

원본 문서를 다음 구조로 배치합니다.

```text
data/raw_pdfs/
├─ ABS rules/
├─ dnv-class-2026-04/
├─ KR Rules/
├─ LR Rules/
├─ MEPC/
└─ MSC/
```

레이아웃 모델은 다음 위치에 둡니다.

```text
models/layout/yolov10m_doclaynet.pt
```

원본 PDF와 모델 파일은 `.gitignore`로 제외되어 있습니다. 공개 저장소나 공개 공유 링크에 업로드하지 마십시오.

## 빠른 실행

이미 `full_corpus_715_v1`과 `kr_tables_v2` 인덱스가 준비되어 있다면 다음 명령으로 UI를 실행합니다.

```powershell
streamlit run scripts/15_rag_ui.py
```

UI는 기본적으로 다음 리소스를 사용합니다.

- Chroma 인덱스: `data/processed/index/`
- 청크: `data/processed/chunks/`
- 일반 검색 collection: `full_corpus_715_v1`
- 표 검색 collection: `kr_tables_v2`
- 표 원문 청크: `data/processed/chunks_v2/`
- Ollama 모델: `llama3.1:8b`

## 전처리와 인덱스 구축

전건 인덱스를 만들기 전에 [Embedding Policy v1](docs/EMBEDDING_POLICY_V1.md)을 확인하십시오. v1은 `multilingual-e5-base`, 최대 420토큰, 60토큰 overlap과 본문/표 collection 분리를 재현 기준으로 사용합니다.

### 1. PDF manifest 생성

```powershell
python scripts/00_build_manifest.py `
  --input-dir data/raw_pdfs `
  --output data/manifests/pdf_manifest.csv
```

### 2. 문서별 전처리

한 문서를 전체 단계로 처리하는 예시입니다.

```powershell
python scripts/run_rag_batch.py `
  --doc-id kr_1_2025 `
  --steps pdf,layout,merge,crop,chunks,index
```

주요 산출물:

```text
data/processed/pages/<doc_id>/
data/processed/layout_json/<doc_id>/
data/processed/layout_json_merged/<doc_id>/
data/processed/crops_merged/<doc_id>/
data/processed/chunks/<doc_id>/chunks.jsonl
data/processed/index/<doc_id>/
```

### 3. 통합 벡터 인덱스 구축

실제 PDF 715개 corpus manifest를 이용하는 예시입니다.

```powershell
python scripts/10_build_unified_index.py `
  --doc-list data/manifests/full_corpus_715.csv `
  --manifest data/manifests/full_corpus_715.csv `
  --collection-id full_corpus_715_v1 `
  --embedding-preset e5-base `
  --include-types text,picture `
  --structured-tables exclude `
  --max-embedding-tokens 420 `
  --embedding-overlap-tokens 60
```

기본 임베딩 preset은 `e5-base`이며 `intfloat/multilingual-e5-base`를 사용합니다. 임베딩 모델이나 청크 표현을 변경하면 기존 인덱스를 재사용하지 말고 다시 구축해야 합니다.

### 4. BM25 인덱스 구축

```powershell
python scripts/35_build_bm25_index.py --unified full_corpus_715_v1 --rebuild
```

### 5. 표 인덱스

v2는 PDF의 선·셀 좌표로 열 경계를 먼저 복원한 뒤 다중 헤더를 평탄화합니다. 품질검사를 통과한 표만 `table_schema`, `table_summary`, `table_markdown`, `table_row`로 만들며 review/reject 표는 인덱스에서 격리합니다.

```powershell
python scripts/44_build_kr_tables_v2.py
python scripts/45_reassess_kr_tables_v2.py

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

python scripts/10_build_unified_index.py `
  --collection-id kr_tables_v2 `
  --doc-list data/manifests/kr_table_top22.csv `
  --manifest data/manifests/rag_corpus_457.csv `
  --chunks-dir data/processed/chunks `
  --table-chunks-dir data/processed/chunks_v2 `
  --embedding-preset e5-base `
  --include-types table `
  --structured-tables only `
  --max-embedding-tokens 420 `
  --embedding-overlap-tokens 60
```

표 인덱스는 `kr_tables_v2` collection으로 분리하여 운영하고 일반 본문은 제외합니다. 자동 평가 결과와 제한사항은 [Table QA v2 평가 보고서](docs/TABLE_QA_22DOC_EVALUATION_V2.md)에 기록되어 있습니다.

## CLI 검색과 평가

대화형 검색:

```powershell
python scripts/rag_query.py --unified full_corpus_715_v1 -i --full-text --top-k 5
```

파일럿 검색 평가:

```powershell
python scripts/13_rag_pilot_validation.py `
  --unified full_corpus_715_v1 `
  --skip-llm `
  --top-k 8
```

표 검색 평가:

```powershell
# 구조화·라우팅 회귀시험(질문에 파일명/페이지 힌트가 있어 실무 성능으로 사용 금지)
python scripts/42_table_eval_22_benchmark.py

# 페이지를 숨긴 단순 실무형 66문항 생성 및 검색 평가
python scripts/48_build_table_eval_simple_curated.py
python scripts/35_build_bm25_index.py --unified kr_tables_v2 --table --rebuild
$env:RAG_DEBUG_TRACE_STDERR = "0"
python scripts/42_table_eval_22_benchmark.py `
  --questions data/eval/table_questions_22docs_practical_v1_curated.jsonl `
  --chunks-dir data/processed/chunks_v2 `
  --collection-id kr_tables_v2 `
  --out data/processed/logs/table_eval_22docs_practical_v1_curated_hybrid_v2.json

# 문서당 1문항(22문항) 최종 답변·인용 평가
python scripts/42_table_eval_22_benchmark.py `
  --questions data/eval/table_questions_22docs_practical_v1_curated.jsonl `
  --chunks-dir data/processed/chunks_v2 `
  --collection-id kr_tables_v2 `
  --with-llm `
  --one-per-doc `
  --out data/processed/logs/table_eval_22docs_practical_v1_curated_hybrid_answers_v3.json
```

기존 v2 66문항의 100% 결과는 구조화 일관성 회귀시험일 뿐입니다. 현재 평가는 비교·다중조건을 제외하고 66개 문장을 개별 재작성한 `table_questions_22docs_practical_v1_curated.jsonl`을 사용합니다. 2026-07-20 open-corpus 측정 결과는 표 ID·행 근거·셀 근거·인용 위치가 모두 90.9%(60/66)입니다. 문서당 1문항의 최종 답변 표본은 정답 95.5%(21/22), 정확 파일·페이지 인용 90.9%(20/22)입니다. 검색에는 평가 정답 필드를 사용하지 않습니다. 다만 gold cell은 아직 PDF 전건 육안 검증 전이므로 모범답안은 `table_questions_22docs_practical_v1_curated_review.md`에서 PDF와 대조해야 합니다.

기본 단위 테스트:

```powershell
python scripts/test_hybrid_retrieval.py
python scripts/test_rule_lookup_answer.py
```

## Ollama와 답변 모드

기본 설정은 `scripts/rag_answer_lib.py`에 정의되어 있습니다.

| 항목 | 기본값 |
|---|---|
| provider | Ollama |
| model | `llama3.1:8b` |
| base URL | `http://localhost:11434` |
| 빠른 모드 | 검색 및 응답 지연 최소화 |
| 정확 모드 | 더 넓은 문맥과 근거 기반 LLM 합성 |

일부 평가 스크립트는 OpenAI provider도 지원하며, 이 경우 키를 코드나 설정 파일에 저장하지 말고 환경변수로 전달합니다.

```powershell
$env:OPENAI_API_KEY = "..."
```

## 인코딩

저장소의 README, Python, CSV, JSONL 파일은 UTF-8을 기준으로 합니다. Windows PowerShell의 출력 코드페이지 때문에 정상적인 한글이 깨져 보일 수 있습니다.

PowerShell에서 다음 명령을 먼저 실행하면 UTF-8 출력 문제를 줄일 수 있습니다.

```powershell
chcp 65001
$OutputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"
```

Python 파일 인코딩 확인 예시:

```powershell
python -c "from pathlib import Path; Path('README.md').read_text(encoding='utf-8'); print('UTF-8 OK')"
```

## Git에 포함하지 않는 파일

- 원본 PDF와 배포 제한 자료: `data/raw_pdfs/`
- 페이지, crop, chunk, 로그, 인덱스: `data/processed/`
- 레이아웃 및 임베딩 모델 가중치: `models/`
- UI·진단 출력: `outputs/`
- Python 가상환경과 캐시
- API 키, `.env`, 인증 파일

대용량 인덱스를 공유할 때는 별도 스토리지를 사용하고, Git에는 생성 명령, embedding preset, corpus manifest, 평가 결과와 체크섬만 기록하는 방식을 권장합니다.

## 프로젝트 구조

```text
MaritimeRAG/
├─ scripts/                  # 전처리·인덱스·RAG·표 QA·UI·단위 테스트
├─ data/
│  ├─ eval/                  # 고정 평가 질문과 검토 메모
│  ├─ manifests/             # 재현 가능한 715-PDF 코퍼스 목록
│  ├─ raw_pdfs/              # 사용자가 준비하는 원본 문서(Git 제외)
│  └─ processed/             # 생성 청크·인덱스·실행 로그(Git 제외)
├─ docs/                     # 임베딩·표 QA 정책과 평가 보고서
├─ config/                   # 임베딩 정책 설정
├─ requirements.txt
└─ README.md
```

`scripts/_tmp_*.py`, 구 버전 비교용 벤치마크, 로컬 재개 도구는 공개 실행 경로에서 제외했습니다. 기본 확인은 `python -m unittest discover -s scripts -p "test_*.py"`로 실행할 수 있습니다.
