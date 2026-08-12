# 설치·실행·문제 해결

## 1. 준비

- Windows PowerShell
- Python 가상환경 `.venv`
- [Ollama](https://ollama.com/)
- 로컬 데이터와 인덱스: README의 “로컬 데이터” 목록 참조

처음 설치하는 경우:

```powershell
cd C:\Users\user\llmagent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull gemma4:12b
ollama pull llama3.1:8b
```

## 2. UI 실행

Ollama가 실행 중인지 확인합니다.

```powershell
ollama list
```

그다음 UI를 시작합니다.

```powershell
cd C:\Users\user\llmagent
.\.venv\Scripts\python.exe app.py
```

브라우저 주소는 `http://127.0.0.1:7860`입니다. 콘솔에 `운항DB: OK`, `문서인덱스: OK`, `raw_pdfs: 연결됨`이 표시되는지 확인하세요.

## 3. 탭 사용법

- **통합 질문**: 정보원이 불분명하거나 운항+규정을 함께 물을 때 사용합니다.
- **문서 검색**: IMO·선급 규정, 회의 결과, 표를 물을 때 사용합니다. 자동으로 RAG에 고정됩니다.
- **운항 정보**: 우리 선박 센서, 현재 상태, 항차 데이터를 물을 때 사용합니다. 자동으로 OPS에 고정됩니다.

모델은 Gemma가 기본입니다. 응답속도를 우선하면 Llama를 선택할 수 있습니다.

## 4. 자주 생기는 문제

### 7860 포트를 이미 사용 중

이미 켜진 UI를 `http://127.0.0.1:7860`에서 먼저 확인합니다. 별도 인스턴스를 띄우려면:

```powershell
$env:GRADIO_SERVER_PORT="7861"
.\.venv\Scripts\python.exe app.py
```

### Ollama 연결 실패 또는 모델 없음

```powershell
ollama list
ollama serve
ollama pull gemma4:12b
```

`ollama serve`는 별도 PowerShell 창에서 유지합니다.

### 문서 인덱스가 없음

다음 두 폴더를 확인합니다.

```powershell
Test-Path .\data\processed\index\unified_full_corpus_715_v1
Test-Path .\data\processed\index\unified_full_corpus_715_tables_precise_v1
```

둘 중 하나가 `False`면 GitHub clone만 받은 상태일 가능성이 큽니다. 인덱스는 용량 때문에 GitHub에 포함되지 않습니다.

### 첫 질문만 느림

임베딩 모델과 Ollama 모델을 처음 메모리에 올리는 시간입니다. 이후 호출은 keep-alive와 캐시를 사용합니다. 계속 느리면 Llama 선택, 출력 범위 축소, 문서 탭/운항 탭의 고정 라우팅 사용 순으로 확인하세요.

## 5. 검증

코드 회귀 테스트:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

TEXT RAG 평가셋 재생성과 검색 평가는 [평가 방법과 결과](EVALUATION.md)를 따릅니다.
