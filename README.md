# MaritimeOpsRAG

선박 운항 데이터와 IMO·선급 문서를 한 UI에서 질의하는 로컬 RAG 애플리케이션입니다. 실행에 필요한 코드·데이터·인덱스 경로는 모두 이 저장소의 `llmagent` 폴더를 기준으로 합니다.

## 현재 구성

| UI 탭 | 처리 경로 | 용도 |
|---|---|---|
| 통합 질문 | LLM 라우터 → OPS / RAG / HYBRID / CHAT | 질문의 정보원을 자동 판단 |
| 문서 검색 | RAG 고정 | IMO·선급 본문과 표 검색 |
| 운항 정보 | OPS 고정 | `maritime.db` 기반 선박 상태·센서·항차 질의 |

- 답변 모델: `gemma4:12b`(기본), `llama3.1:8b`
- 라우팅: 사용자가 선택하는 규칙 라우터는 제거했습니다. 통합 탭은 LLM 판단을 기본으로 하고, 모델 실패나 저신뢰 때만 안전한 결정적 fallback을 사용합니다.
- 문서 검색: Dense + BM25 + 메타데이터/문서코드 + 증거 슬롯 보강
- 표 검색: 본문과 분리된 precise-table collection을 사용하고 관련 표와 crop을 함께 표시합니다.

## 빠른 실행

```powershell
cd C:\Users\user\llmagent
.\.venv\Scripts\Activate.ps1
ollama serve
```

새 PowerShell 창에서 다음을 실행합니다.

```powershell
cd C:\Users\user\llmagent
.\.venv\Scripts\python.exe app.py
```

브라우저에서 `http://127.0.0.1:7860`을 엽니다. 7860 포트가 이미 사용 중이면:

```powershell
$env:GRADIO_SERVER_PORT="7861"
.\.venv\Scripts\python.exe app.py
```

처음 설치하거나 모델이 없으면 [실행 안내](docs/USAGE.md)를 확인하세요.

## 로컬 데이터

로컬 `C:\Users\user\llmagent`에는 다음 자산이 있어야 전체 기능을 사용할 수 있습니다.

- `data/maritime.db`: 운항 데이터
- `data/raw_pdfs/`: 원문 PDF
- `data/processed/index/unified_full_corpus_715_v1/`: 본문 Chroma·BM25 인덱스
- `data/processed/index/unified_full_corpus_715_tables_precise_v1/`: 표 Chroma·BM25 인덱스
- `data/processed/`: 추출 본문, 표, crop, 평가 로그

이 대용량 자산은 `.gitignore`로 GitHub 업로드에서 제외됩니다. 따라서 현재 로컬 폴더에서는 단독 실험이 가능하지만, GitHub를 새 PC에 clone한 것만으로는 PDF·DB·인덱스가 생기지 않습니다.

## 검색 흐름

```text
질문 분석
  → 문서코드·회의차수·선급·표현식·요구 개수 추출
  → 본문/표/혼합 검색 모드 결정
  → Dense + BM25 후보 검색
  → 정확 식별자·문서 registry·메타데이터 보정
  → 문서 내부 조항 검색과 evidence-slot 보강
  → 최종 근거 선택
  → Gemma/Llama 답변 + 문서·페이지 인용
```

기존 임베딩은 다시 만들지 않았습니다. 검색 단계에서 후보 문서 진입, 정확 식별자, 문서 내부 조항 회수, 다중 근거 선택을 개선했습니다. 자세한 내용은 [아키텍처와 검색](docs/ARCHITECTURE.md)에 정리했습니다.

## TEXT RAG v3 평가 요약

증강 원본을 그대로 정답으로 간주하지 않고, 9개 시나리오의 문서·청크·필수 답변 요소를 재검토해 405문항으로 구성했습니다.

| 지표 | 결과 |
|---|---:|
| 검색 대상 360문항의 후보 문서 Any Recall | 97.22% |
| 검색 대상 360문항의 최종 문서 Any Recall | 96.94% |
| 단일 근거 정밀검색 45문항의 최종 문서 Recall | 100% |
| 단일 근거 정밀검색 45문항의 의미 기반 근거 recall | 71.11% |
| Gemma 전체 must-cover completeness | 53.68% |
| Gemma 답변 형식·거절·반박 behavior pass | 99.75% |
| Gemma 답변 가능 문항의 인용률 | 99.72% |
| Gemma 평균 검색 / 생성 / 전체 시간 | 1.99초 / 6.16초 / 8.15초 |

즉, 단일 문서의 식별과 진입은 안정적입니다. 다만 긴 문서에서 여러 조항을 한 번에 요구하거나 여러 문서를 통합하는 질문은 필수 항목 누락이 남아 있습니다. 수치의 정의와 테스트셋 생성법은 [평가 방법과 결과](docs/EVALUATION.md)를 확인하세요.

## 문서

- [아키텍처·라우팅·임베딩·검색](docs/ARCHITECTURE.md)
- [설치·실행·문제 해결](docs/USAGE.md)
- [평가셋 생성·현재 결과·한계](docs/EVALUATION.md)
