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

GitHub에는 대용량 PDF·DB·인덱스가 포함되지 않습니다. 새 PC에서는 기존 `C:\Users\user\llmagent\data`의 다음 자산을 별도로 복사해야 합니다.

```text
data/maritime.db
data/raw_pdfs/
data/processed/index/unified_full_corpus_715_v1/
data/processed/index/unified_full_corpus_715_tables_precise_v1/
```

## UI 실행

Ollama를 시작합니다.

```powershell
ollama serve
```

새 PowerShell 창에서 UI를 실행합니다.

```powershell
cd C:\Users\user\llmagent
.\.venv\Scripts\python.exe app.py
```

브라우저 주소는 `http://127.0.0.1:7860`입니다. 콘솔에 운항 DB·문서 인덱스·raw PDF 상태가 모두 준비됐는지 확인합니다.

## 탭과 모델

- **통합 질문**: LLM이 운항·문서·혼합·안내 경로를 자동 판단합니다. 디버깅할 때만 정보원을 강제할 수 있습니다.
- **문서 검색**: IMO·선급 규정, 회의 결과, 본문과 표를 묻습니다. 항상 RAG로 처리됩니다.
- **운항 정보**: 항차·속력·연료·배출량·CII를 조회하고 Word 보고서를 생성합니다. 항상 OPS로 처리됩니다.
- **보고서 관리**: `reports/output`의 실제 생성 보고서를 검색하고 내려받습니다.

모델은 Gemma 4 12B가 기본입니다. Llama 3.1 8B는 빠른 응답이 필요할 때 선택합니다. Mistral은 현재 UI와 모델 목록에 없습니다.

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
