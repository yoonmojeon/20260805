# Embedding Policy v1

이 문서는 MaritimeRAG 전건 인덱스 v1의 재현 기준을 고정한다. 아래 항목을 변경하면 해당 collection을 전건 재생성해야 한다.

## 모델

- preset: `e5-base`
- model: `intfloat/multilingual-e5-base`
- Hugging Face revision: `d128750597153bb5987e10b1c3493a34e5a4502a`
- provider: `sentence-transformers` 로컬 실행
- 언어: 한국어·영어
- query prefix: `query: `
- passage prefix: `passage: `
- normalized embeddings: 활성화
- Chroma distance: cosine

## Chunk 정책

- 기존 조문·레이아웃 element가 420토큰 이하이면 그대로 유지한다.
- 최종 임베딩 문자열이 420토큰을 초과할 때만 분할한다.
- 분할 우선순위: 기존 조문/element → 문단 → 문장 → token 경계
- 인접 분할 조각은 60토큰을 겹친다.
- 토큰 수에는 `passage:` prefix와 임베딩 헤더를 포함한다.
- 최종 입력은 반드시 420토큰 이하여야 한다.
- 분할 chunk ID는 `<원본 chunk_id>__eNNN` 형식이다.
- 원본 ID, 분할 순번, 최종 토큰 수는 Chroma metadata에 저장한다.

## 임베딩 문자열과 metadata

임베딩 문자열에 포함:

- source와 사람이 읽을 수 있는 파일명
- folder와 document type
- section 제목과 조문 번호
- 표 caption, column 이름, 실제 본문/표 내용

Chroma metadata에만 저장:

- chunk/document/element/table 내부 ID
- page와 row 번호
- crop 및 원본 페이지 이미지 경로
- 분할 원본 ID와 분할 순번

내부 ID와 페이지 번호는 의미 검색의 노이즈를 줄이기 위해 임베딩 문자열에서 제외한다.

## Collection 분리

### `full_corpus_715_v1`

- manifest: `data/manifests/full_corpus_715.csv`
- 일반 본문과 caption이 있는 그림만 포함
- 구조화 표 chunk와 legacy table 문자열 제외
- 실행 옵션: `--include-types text,picture --structured-tables exclude`

### `kr_tables_v1`

- manifest: `data/manifests/kr_table_top22.csv`
- `table_schema`, `table_summary`, `table_markdown`, `table_row`만 포함
- 일반 본문과 그림 제외
- 실행 옵션: `--include-types table --structured-tables only`

### `kr_tables_v2` (현재 표 QA 기본값)

- 문서 범위와 임베딩 모델, 420/60 분할 정책은 v1과 동일
- 원본 PDF 좌표로 재구성한 `data/processed/chunks_v2`를 사용
- 품질검사를 통과한 `table_schema`, `table_summary`, `table_markdown`, `table_row`만 포함
- review/reject 표는 검색 인덱스에서 격리
- 표 표현과 embedding metadata가 v1과 달라 별도 collection으로 운영
- 구축·평가 결과: [Table QA v2 평가 보고서](TABLE_QA_22DOC_EVALUATION_V2.md)

## 전수 분석 결과

2026-07-15 기준:

| collection | 분할 전 | 420토큰 초과 | 분할 후 | 최장 입력 |
|---|---:|---:|---:|---:|
| `full_corpus_715_v1` | 266,721 | 4,023 | 271,849 | 420 |
| `full_corpus_v1` (legacy) | 145,915 | 3,155 | 150,113 | 420 |
| `kr_tables_v1` | 117,650 | 3,691 | 123,803 | 420 |

파일럿 및 전건 검증:

- 일반 검색 7문항 baseline: v0와 v1 모두 gold document 6/7, gold page 2/7
- 표 schema 검색 8문항: v0 50%에서 전건 v1 62.5%로 개선
- 전건 Chroma count: `full_corpus_715_v1` 271,849, `kr_tables_v1` 123,803
- `full_corpus_715_v1`은 실제 PDF 715건을 모두 전처리했으며, 본문이 비어 있는 MSC 철회문서 1건을 제외한 714건이 검색 가능하다.
- DNV 문서 318건 전체가 포함되며 DNV 청크는 141,913개다.
- 각 manifest count와 Chroma count 일치, 최종 token 최대값 420 확인

## 재생성이 필요한 변경

- embedding 모델 또는 revision
- query/passsage prefix나 normalization
- 최대 토큰 수 또는 overlap
- 분할 경계 규칙
- 임베딩 문자열에 포함하는 metadata
- 표 schema/summary/row/markdown 표현
- 포함·제외 chunk 유형

답변 prompt, Ollama 답변 모델, top-k, reranking 가중치, UI 변경은 기존 metadata가 충분하면 재임베딩이 필요하지 않다.

## 구축 명령

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

python scripts/10_build_unified_index.py `
  --doc-list data/manifests/full_corpus_715.csv `
  --manifest data/manifests/full_corpus_715.csv `
  --collection-id full_corpus_715_v1 `
  --embedding-preset e5-base `
  --include-types text,picture `
  --structured-tables exclude `
  --max-embedding-tokens 420 `
  --embedding-overlap-tokens 60

python scripts/10_build_unified_index.py `
  --doc-list data/manifests/kr_table_top22.csv `
  --collection-id kr_tables_v1 `
  --embedding-preset e5-base `
  --include-types table `
  --structured-tables only `
  --max-embedding-tokens 420 `
  --embedding-overlap-tokens 60

python scripts/44_build_kr_tables_v2.py
python scripts/45_reassess_kr_tables_v2.py

python scripts/10_build_unified_index.py `
  --doc-list data/manifests/kr_table_top22.csv `
  --manifest data/manifests/rag_corpus_457.csv `
  --collection-id kr_tables_v2 `
  --chunks-dir data/processed/chunks `
  --table-chunks-dir data/processed/chunks_v2 `
  --embedding-preset e5-base `
  --include-types table `
  --structured-tables only `
  --max-embedding-tokens 420 `
  --embedding-overlap-tokens 60
```
