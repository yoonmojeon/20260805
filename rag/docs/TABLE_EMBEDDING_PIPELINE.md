# 표 임베딩 파이프라인

715개 문서용 표 인덱스와 KR 7편 고정밀 복원 절차를 구분해서 적습니다.

원본 PDF, 모델 가중치, crop, 청크, Chroma DB는 Git에 넣지 않는다.

## 1. clone 후 다시 만들 수 있는 것

저장소에 있는 것:

- 715 문서 매니페스트
- 레이아웃·표 좌표·품질검사·청킹·임베딩·Chroma 코드
- KR 7편 TATR + PyMuPDF + HancomEQN 구현·PUA 매핑
- 표 검색 회귀 질문·단위 테스트

없는 것:

- 원본 PDF (배포 제한)
- YOLO / Table Transformer / 임베딩 / VLM / LLM 가중치
- 페이지·표 crop 이미지
- 생성된 JSONL·로그
- Chroma·BM25 인덱스
- `.venv`, 모델 캐시, API 키

## 2. 저장소 받기

기본 브랜치는 `main`이다. 기능 변경은 브랜치/PR로 보고, 쓸 때는 병합된 `main`을 clone한다.

```powershell
git clone --branch main https://github.com/yoonmojeon/20260805.git
cd 20260805
```

이미 저장소를 clone했다면 다음과 같이 `main`을 최신 상태로 맞춥니다.

```powershell
git fetch origin
git switch main
git pull
```

## 3. Python 환경 구성

개발 및 검증 환경은 Python 3.11입니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

GPU를 사용할 경우 대상 PC의 CUDA 환경에 맞는 PyTorch를 먼저 설치합니다.
설치 명령은 [PyTorch 공식 설치 페이지](https://pytorch.org/get-started/locally/)에서
확인하는 것이 가장 안전합니다.

그다음 표 파이프라인 의존성을 설치합니다.

```powershell
pip install -r requirements-table.txt
```

설치 확인:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

`True`가 출력되면 CUDA GPU를 사용할 수 있습니다. `False`여도 일부 단계는
CPU로 실행할 수 있지만, 715개 문서 전체 레이아웃 분석과 임베딩에는 매우 오랜
시간이 걸릴 수 있습니다.

## 4. 원본 PDF와 모델 배치

원본 PDF는 `data/raw_pdfs/` 아래에 배치합니다. 기본 폴더 구조는 다음과
같습니다.

```text
data/raw_pdfs/
├─ ABS rules/
├─ dnv-class-2026-04/
├─ KR Rules/
├─ LR Rules/
├─ MEPC/
└─ MSC/
```

DocLayNet 레이아웃 검출 모델은 다음 위치에 둡니다.

```text
models/layout/yolov10m_doclaynet.pt
```

체크인된 `data/manifests/full_corpus_715.csv`에는 715개 문서의 기준 목록이
들어 있습니다. 다른 컴퓨터에서 PDF 파일명이나 경로가 달라졌다면 로컬
매니페스트를 새로 생성합니다.

```powershell
python scripts/00_build_manifest.py `
  --input-dir data/raw_pdfs `
  --output data/manifests/pdf_manifest.local.csv
```

실행 전에 매니페스트 문서 수와 누락 PDF를 반드시 확인해야 합니다. 로컬
매니페스트를 사용할 때는 이후 모든 명령에서 같은 파일을 `--manifest`로
지정해야 합니다.

## 5. 715개 문서 정밀 표 인덱스 구축

KR 7편과 같은 정밀 경로의 전체 실행 진입점은
`scripts/70_build_precise_table_corpus.py`입니다. 715는 표 개수가 아니라 PDF
문서 개수이며, 실제 표 수는 전처리 후 생성되는 매니페스트에서 확인합니다.

```powershell
$env:PYTHONUTF8 = "1"

python scripts/70_build_precise_table_corpus.py `
  --doc-list data/manifests/full_corpus_715.csv `
  --pdf-manifest data/manifests/full_corpus_715.csv `
  --collection-id full_corpus_715_tables_precise_v1 `
  --resume
```

이 명령은 전처리, 전체 표 crop, TATR, PDF 벡터선 스냅, PyMuPDF 텍스트 배치,
검증된 HancomEQN 복원, 복합 표 분할, 영역별 TATR, 구조화 청크, E5 임베딩과
Chroma 생성을 순서대로 수행합니다. `--resume`은 단계별 결과를 재사용합니다.

KR 문서는 7편에서 검증한 HancomEQN 공통 코드 매핑을 기본 적용하여 7편과
동일한 복원 경로를 사용합니다. 문서별 매핑이 registry에 있으면 그것을 우선
사용합니다. 매핑 범위 밖 PUA를 임의로 추정하지 않으며, 미복원 문자가 남은
표만 격리하고 사유를
`data/processed/logs/full_corpus_715_precise_audit.json`에 기록합니다.

로컬 매니페스트를 쓰는 경우:

```powershell
python scripts/70_build_precise_table_corpus.py `
  --doc-list data/manifests/pdf_manifest.local.csv `
  --pdf-manifest data/manifests/pdf_manifest.local.csv `
  --collection-id full_corpus_715_tables_precise_v1 `
  --resume
```

### 좌표 기반 기준선

`scripts/69_build_table_corpus.py`는 TATR/PUA 복원 전의 범용 좌표 기반
기준선입니다. 빠른 비교 실험과 정밀 파이프라인의 전처리 단계에서 사용합니다.

이 스크립트는 다음 단계를 순서대로 실행합니다.

1. PDF 페이지 렌더링
2. 문서 레이아웃 검출 및 병합
3. 표 후보 영역 추출
4. PyMuPDF 기반 셀 및 좌표 복원
5. 표 품질검사와 review/reject 격리
6. 표 전용 구조화 청크 생성
7. `multilingual-e5-base` 임베딩
8. 표 전용 Chroma 컬렉션 생성

기준선만 실행하는 명령:

```powershell
$env:PYTHONUTF8 = "1"

python scripts/69_build_table_corpus.py `
  --doc-list data/manifests/full_corpus_715.csv `
  --manifest data/manifests/full_corpus_715.csv `
  --collection-id full_corpus_715_tables_v1 `
  --resume-completed
```

로컬에서 새로 만든 매니페스트를 사용하는 예:

```powershell
python scripts/69_build_table_corpus.py `
  --doc-list data/manifests/pdf_manifest.local.csv `
  --manifest data/manifests/pdf_manifest.local.csv `
  --collection-id full_corpus_715_tables_v1 `
  --resume-completed
```

`--resume-completed`는 이미 생성된 문서별 결과를 재사용합니다. 장시간 실행이
중단되었을 때 처음부터 다시 처리하지 않도록 전체 실행에서 사용하는 것을
권장합니다.

## 6. 단계별 재실행 방법

정밀 파이프라인은 `--stages`로 일부 단계만 재실행할 수 있습니다.

```powershell
python scripts/70_build_precise_table_corpus.py `
  --doc-list data/manifests/full_corpus_715.csv `
  --pdf-manifest data/manifests/full_corpus_715.csv `
  --stages chunks,index `
  --resume
```

단계 이름은 `preprocess,prepare,tatr,snap,restore,segment,region-tatr,chunks,index`
입니다. 앞 단계 산출물이 없는 상태에서 뒤 단계만 실행하면 실패합니다.

PDF 렌더링과 레이아웃 전처리가 이미 끝났다면 다음과 같이 시작할 수 있습니다.

```powershell
python scripts/69_build_table_corpus.py `
  --doc-list data/manifests/full_corpus_715.csv `
  --manifest data/manifests/full_corpus_715.csv `
  --collection-id full_corpus_715_tables_v1 `
  --skip-preprocess `
  --resume-completed
```

주요 생략 옵션은 다음과 같습니다.

| 옵션 | 생략하는 단계 |
|---|---|
| `--skip-preprocess` | PDF 렌더링, 레이아웃, 본문 청크 전처리 |
| `--skip-v1-extraction` | 1차 표 후보와 기본 표 청크 생성 |
| `--skip-v2-reconstruction` | PyMuPDF 좌표 기반 v2 표 복원 |
| `--skip-index` | E5 임베딩과 Chroma 인덱스 구축 |
| `--resume-completed` | 이미 생성된 문서별 결과 재사용 |

예를 들어 표 청크는 완성됐고 벡터 DB만 다시 만들려면 다음처럼 실행합니다.

```powershell
python scripts/69_build_table_corpus.py `
  --doc-list data/manifests/full_corpus_715.csv `
  --manifest data/manifests/full_corpus_715.csv `
  --collection-id full_corpus_715_tables_v2 `
  --skip-preprocess `
  --skip-v1-extraction `
  --skip-v2-reconstruction
```

새 인덱스를 만들 때는 기존 collection ID를 덮어쓰지 말고
`full_corpus_715_tables_v2`처럼 새 버전을 사용하는 것이 안전합니다.

## 7. 생성되는 결과

주요 산출물은 다음 위치에 생성됩니다.

```text
data/processed/tables/<doc_id>/tables.jsonl
data/processed/tables_v2/<doc_id>/tables.jsonl
data/processed/chunks_v2/<doc_id>/table_chunks.jsonl
data/processed/logs/full_corpus_715_tables_v2_quality.json
data/processed/index/unified_full_corpus_715_tables_v1/
data/processed/precise_tables/<doc_id>/<table_id>/
data/processed/chunks_tables_precise/<doc_id>/table_chunks.jsonl
data/processed/logs/full_corpus_715_precise_audit.json
data/processed/index/unified_full_corpus_715_tables_precise_v1/
```

`quality_status=pass`인 구조화 청크만 운영 인덱스에 포함해야 합니다.
`review` 또는 `reject` 표는 감사 대상으로 보존하되 자동으로 운영 인덱스에
승격하면 안 됩니다.

## 8. KR 7편 고정밀 표 복원 파이프라인

KR 7편에서 검증한 고정밀 경로는 다음과 같습니다.

```text
표 crop
  → Table Transformer 행·열·병합 셀 인식
  → PDF 벡터선과 TATR 좌표 스냅
  → PyMuPDF 네이티브 텍스트 배치
  → HancomEQN PUA 수식 복원
  → 복합 표 영역 분할과 영역별 TATR
  → 영역·행·요약 RAG 레코드 생성
  → multilingual-e5-base 임베딩
  → Chroma 표 전용 인덱스
```

Table Transformer 모델을 Hugging Face 캐시에 내려받습니다.

```powershell
huggingface-cli download microsoft/table-transformer-structure-recognition-v1.1-all
```

인터넷 연결 없이 실행하려면 다운로드가 끝난 뒤 다음 환경변수를 사용합니다.

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

관련 구현 파일:

- `scripts/52_tatr_structure_pilot.py`
- `scripts/53_snap_tatr_to_pdf.py`
- `scripts/54_inventory_hancomeqn.py`
- `scripts/55_restore_hancomeqn.py`
- `scripts/59_build_kr7_expanded_pilot.py`부터
  `scripts/68_eval_kr7_expanded_qa.py`까지
- `scripts/hancomeqn_restore.py`
- `data/config/hancomeqn_maps/7_2025_bdc15136d686.json`
- `scripts/70_build_precise_table_corpus.py`
- `data/config/hancomeqn_maps/registry.json`

### 중요한 제한

고정밀 코드 경로는 이제 715개 문서 목록 전체를 입력받을 수 있지만, 실제
정답 기반 QA 검증은 현재 `7편_2025.pdf`에서 완료된 상태입니다. "전건 실행
가능"과 "715개 문서 모두 정확도 검증 완료"는 다르므로, 전건 실행 후
출처·표 형태별 블라인드 QA가 필요합니다.

다른 PDF는 다음 차이가 있을 수 있습니다.

- HancomEQN 이외의 수식 글꼴 사용
- 문서별로 다른 PUA 코드와 glyph 매핑
- 테두리 없는 표 또는 부분 테두리 표
- 여러 하위 표가 한 표 영역에 포함된 복합 레이아웃
- 세로 병합 셀과 이어지는 다음 페이지 표
- 스캔 이미지로만 구성된 표

따라서 다른 문서에 고정밀 경로를 적용할 때는 문서별 PUA 인벤토리, 매핑
커버리지, 구조 불일치 감사, 블라인드 QA 평가를 거쳐야 합니다.

## 9. 테스트와 검증

전체 단위 테스트:

```powershell
$env:PYTHONPATH = "scripts"
python -m unittest discover -s scripts -p "test_*.py" -q
```

KR 7편 28문항 회귀평가:

```powershell
python scripts/68_eval_kr7_expanded_qa.py `
  --questions data/eval/kr7_expanded_table_qa_28.json `
  --top-k 10 `
  --answers
```

신규 16문항과 블라인드 문항도 같은 방식으로 실행할 수 있습니다.

```powershell
python scripts/68_eval_kr7_expanded_qa.py `
  --questions data/eval/kr7_expanded_v2_new_qa_16.json `
  --top-k 10 `
  --answers

python scripts/68_eval_kr7_expanded_qa.py `
  --questions data/eval/kr7_blind_regression_4.json `
  --top-k 10 `
  --answers
```

## 10. 점진적 확장 권장 순서

715개 문서를 한 번에 처리하더라도 모든 결과를 즉시 운영 인덱스로 승격하지
않는 것이 좋습니다.

권장 순서:

1. 전체 문서에서 표 후보와 품질 통계를 생성합니다.
2. 문서 출처와 표 형태별로 대표 표본을 선정합니다.
3. 수식, 병합 헤더, 범위 조건, 다음 페이지 계속 표를 포함해 QA를 만듭니다.
4. 미복원 PUA와 구조 불일치 표는 감사 큐로 격리합니다.
5. 블라인드 QA 기준을 통과한 문서군만 새 collection에 포함합니다.
6. 기존 인덱스를 보존한 채 UI에서 새 collection을 시험합니다.
7. 품질이 확인되면 다음 문서군으로 확장합니다.

최소 품질 게이트:

- 선택한 모든 문서의 처리 성공 여부
- 누락 PDF 및 누락 레이아웃 페이지 수
- `pass`, `review`, `reject` 표 개수
- 미복원 PUA 문자와 표 개수
- TATR, PDF 벡터선, 기존 검출기의 구조 일치율
- 표 검색 Recall@k
- 최종 답변 정확도와 근거 페이지 정확도

## 11. Git에 올릴 파일과 제외할 파일

Git에 포함할 파일:

- Python 소스 코드
- 소형 설정 및 PUA 매핑 파일
- 단위 테스트
- 소형 평가 질문
- 실행 문서와 README
- 코퍼스 문서 목록 매니페스트

Git에서 제외할 파일:

- 원본 PDF
- 모델 가중치
- 렌더링 페이지 및 표 crop
- 생성된 표 JSONL과 임베딩 청크
- Chroma/BM25 인덱스
- 실행 로그 및 감사 이미지
- 로컬 절대경로가 포함된 생성 매니페스트
- `.venv`, Hugging Face 캐시
- `.env`, API 키, 인증서 및 credentials 파일

대용량 인덱스를 다른 컴퓨터로 옮겨야 한다면 Git에 넣지 말고 압축 파일,
오브젝트 스토리지 또는 별도 내부 저장소를 사용합니다. Git에는 인덱스를 다시
만들 수 있는 명령, embedding preset, 문서 목록, 코드 버전만 기록하는 것이
좋습니다.
