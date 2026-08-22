# 설치·실행·문제 해결

## 준비

- Windows PowerShell
- Python 가상환경 `.venv`
- [Ollama](https://ollama.com/)
- 로컬 PDF·운항 DB·본문/표 인덱스

처음 설치하는 경우:

```powershell
cd C:\Users\user\llmagent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull gemma4:12b
ollama pull llama3.1:8b
```

Advanced의 비중국계 로컬 reranker도 준비합니다. 인터넷이 허용된 설치 PC에서 한 번만 내려받거나, 아래 완성 폴더를 운영 PC로 복사합니다. 서비스 실행 중에는 다운로드하지 않습니다.

```powershell
.\.venv\Scripts\huggingface-cli.exe download `
  cross-encoder/ms-marco-MiniLM-L4-v2 `
  --local-dir .\models\cross-encoder-ms-marco-MiniLM-L4-v2
```

- 모델: Microsoft MiniLM 기반 SentenceTransformers/UKP cross-encoder
- 라이선스: Apache-2.0
- 런타임 정책: `local_files_only=True`, `trust_remote_code=False`, `HF_HUB_OFFLINE=1`
- 가중치가 없으면 Advanced는 오류를 내지 않고 Accurate 후보 + Gemma listwise 경로로 안전하게 계속 실행합니다.

기존 Chroma가 있고 Advanced sparse sidecar만 없는 경우 한 번 생성합니다. 현재 272k 청크 기준 약 758MB이며 재임베딩은 하지 않습니다.

```powershell
.\.venv\Scripts\python.exe rag\scripts\36_build_sparse_fts_index.py `
  --unified full_corpus_715_v1 `
  --index-dir data\processed\index
```

GitHub에는 대용량 PDF·DB·인덱스가 포함되지 않습니다. 새 PC에서는 기존 `C:\Users\user\llmagent\data`의 다음 자산을 별도로 복사해야 합니다.

```text
data/maritime.db
data/raw_pdfs/
data/processed/index/unified_full_corpus_715_v1/
data/processed/index/unified_full_corpus_715_tables_precise_v1/
data/processed/index/unified_full_corpus_715_v1/accurate_sparse_fts5_v2.sqlite3
models/cross-encoder-ms-marco-MiniLM-L4-v2/
```

## UI 실행

Ollama를 시작합니다.

```powershell
ollama serve
```

새 PowerShell 창에서 UI를 실행합니다.

```powershell
cd C:\Users\user\llmagent
.\start.cmd
```

브라우저 주소는 `http://127.0.0.1:7860`입니다. 콘솔에 운항 DB·문서 인덱스·raw PDF 상태가 모두 준비됐는지 확인합니다.

## 탭과 모델

- **통합 질문**: LLM이 운항·문서·혼합·안내 경로를 자동 판단합니다. 디버깅할 때만 정보원을 강제할 수 있습니다.
- **문서 검색**: IMO·선급 규정, 회의 결과, 본문과 표를 묻습니다. 항상 RAG로 처리됩니다.
- **운항 정보**: 항차·속력·연료·배출량·CII를 조회하고 Word 보고서를 생성합니다. 항상 OPS로 처리됩니다.
- **보고서 관리**: `reports/output`의 실제 생성 보고서를 검색하고 내려받습니다.

모델은 Gemma 4 12B가 기본입니다. Llama 3.1 8B는 빠른 응답이 필요할 때 선택합니다. Mistral은 현재 UI와 모델 목록에 없습니다.

### 문서 답변 모드 선택

- **Fast**: 단순 조회와 빠른 시연. 일반 질문에도 LLM이 답을 작성하지만 검색 문맥과 출력이 짧습니다.
- **Accurate**: 기본 시연 권장. 규정 조항, 회의 결과, 표 질문을 균형 있게 처리합니다.
- **Advanced**: 복합 체크리스트·여러 문서 비교·누락 위험이 큰 질문. 로컬 FTS/BM25, RRF, MiniLM, Gemma listwise와 최종 감사를 모두 사용해 가장 느립니다. UI에서 선택하면 별도 버튼 없이 전체 경로가 자동 적용되고 모델은 Gemma 4 12B로 고정됩니다.

라우팅이 표/텍스트를 잘못 선택했다고 판단될 때만 `텍스트 인덱스로 다시 검색` 또는 `표 인덱스로 다시 검색`을 누릅니다. 이 버튼은 문서 경로와 운항 경로를 바꾸는 기능이 아니라 RAG 인덱스만 명시적으로 다시 선택합니다.

## 자주 생기는 문제

### 7860 포트가 이미 사용 중

대부분 기존 UI가 실행 중이라는 뜻입니다. 먼저 `http://127.0.0.1:7860`을 엽니다. 별도 인스턴스가 필요하면:

```powershell
$env:GRADIO_SERVER_PORT="7861"
.\.venv\Scripts\python.exe app.py
```

7860 사용 프로세스 확인:

```powershell
Get-NetTCPConnection -LocalPort 7860 -State Listen |
  Select-Object LocalAddress,LocalPort,OwningProcess
```

운영 직전 기존 UI만 안전하게 종료하려면 PID를 확인한 뒤 그 PID만 종료합니다.

```powershell
$conn = Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue
$conn | Select-Object LocalAddress,LocalPort,OwningProcess
if ($conn) { Stop-Process -Id $conn.OwningProcess }
```

그 다음 `start.cmd`를 한 번 실행하고 `http://127.0.0.1:7860`에서 화면을 확인합니다. PC를 계속 켜 두더라도 시연 10~20분 전에 한 번 재시작하면 최신 코드가 반영되고 로그 확인도 쉽습니다.

### Ollama 연결 실패 또는 모델 없음

```powershell
ollama list
ollama serve
ollama pull gemma4:12b
```

`ollama serve`는 별도 PowerShell 창에서 계속 실행합니다.

### 문서 인덱스가 없음

```powershell
Test-Path .\data\processed\index\unified_full_corpus_715_v1
Test-Path .\data\processed\index\unified_full_corpus_715_tables_precise_v1
```

하나라도 `False`면 GitHub clone만 받은 상태일 가능성이 큽니다. 기존 로컬 데이터에서 해당 폴더를 복사해야 합니다.

### 첫 질문만 느림

임베딩 모델과 Ollama 모델을 메모리에 처음 올리는 시간입니다. 이후 호출은 keep-alive와 캐시를 사용합니다. 계속 느리면:

1. Llama 모델을 선택합니다.
2. 통합 탭 대신 문서 검색 또는 운항 정보 고정 탭을 사용합니다.
3. 질문에서 요구하는 항목 수와 출력 범위를 줄입니다.
4. Ollama가 CPU가 아니라 GPU를 사용하는지 확인합니다.

잘못된 전제 검증이나 결함 답변은 품질 게이트가 Gemma를 한 번 더 호출할 수 있어 일반 질문보다 오래 걸립니다.

### 표가 답변에 나오지 않음

표 수치·시험 횟수·행/열을 명확히 질문합니다. 문서 검색 탭 하단 진단에서 표 인덱스가 준비됐는지 확인하고, `data/processed/`의 원문 crop 자산도 유지해야 합니다.

## 검증

전체 코드 회귀:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Windows 전역 pytest 임시 폴더 권한 오류가 발생하면 프로젝트 내부의 새 경로를 지정합니다.

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp data\pytest_local_verify
```

UI 문서 회귀:

```powershell
.\.venv\Scripts\python.exe scripts\run_ui_document_regression.py
```

405개 TEXT RAG 평가셋의 생성·검색·답변 평가 명령은 [평가셋과 검증 결과](EVALUATION.md)에 있습니다.
