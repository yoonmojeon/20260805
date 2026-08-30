# MaritimeOpsRAG Advanced 최종 구현·검증 보고서

- 기준일: 2026-08-22
- 실행 환경: Windows, Chroma, multilingual-e5-base, Ollama `gemma4:12b`
- 운영 조건: 검색·재순위·생성·검수 모두 온프레미스
- 평가셋: `data/eval/accurate_eval_150.jsonl` 150문항(답변 가능 133 / 거절 17)

## 결론

기존 임베딩과 714개 PDF 인덱스를 다시 만들지 않고 Fast·Accurate를 보존한 채
Advanced 모드를 추가했다. Advanced는 Accurate의 정답 후보를 보호하면서 로컬
FTS5/BM25·RRF, 질문 분해, 정확일치 누락근거 회수, parent/sibling 문맥,
비중국계 MiniLM cross-encoder, Gemma listwise reranking과 최종 답변 감사를 수행한다.

중요한 결론은 다음과 같다.

1. 초기 진단에서 보고됐던 “KR 규정 질문이 회의 요약으로 간다”는 버그는 재현되지
   않았고, 이 항목을 수정 대상으로 삼지 않았다.
2. 실제 결함은 희소 한국어 복합명사와 다중 회의결과가 최종 evidence selection에서
   빠지는 경우, 그리고 정답 근거가 있어도 소형 모델이 답변에서 일부를 누락하는 경우였다.
3. `방식조치` 같은 희소 개념은 zero-candidate일 때만 Chroma exact lookup을 붙였고,
   Advanced의 복합 질문은 독립 근거 축마다 bounded lookup을 수행한다.
4. 전역 Python BM25는 Accurate A/B에서 recall 개선 없이 지연만 늘어 기본값으로
   사용하지 않는다. Advanced는 전역 스캔 대신 로컬 SQLite FTS5 sidecar를 사용한다.
5. 자동 점수는 전문가 합의 정답률이 아니라 gold keypoint 기반 회귀 proxy다. 최종
   배포 판단에는 실제 답변과 Evidence Table 수작업 검토가 필요하다.

## 구현 내용

### 검색

- 명시 문서번호는 일반 Accurate 경로에서 해당 PDF를 강제 선택한다.
- 선급 포함/제외 조건은 후보·pool·Evidence Table까지 유지한다.
- 희소 한국어 복합명사와 영문 원문 표현을 bounded exact lookup으로 연결한다.
- Advanced는 Dense 80 + FTS/BM25 80을 RRF로 결합하고 Dense 상위 24개를 보호한다.
- 질문을 facet으로 분해하고 누락 축만 최대 2개 query로 재검색한다.
- 한영 exact facet은 최대 4개까지 독립 검색한다.
- 한 PDF의 반복 청크가 후보를 독점하지 않도록 follow-up 결과를 round-robin merge한다.
- 분리된 조항의 parent/sibling과 인접 페이지 문맥을 최대 8개 보강한다.
- 공식 결과 질문에서 Proposal/Comments보다 WP.1/Report/Resolution의
  approved/adopted/agreed 문장을 보호한다.

### 재순위

- 36개 후보를 `cross-encoder/ms-marco-MiniLM-L4-v2`로 로컬 CPU 보조 채점한다.
- 해당 모델은 Microsoft MiniLM 기반, Apache-2.0, 비중국계 모델이다.
- 실제 최종 18개 선택은 로컬 Gemma 4 12B listwise reranker가 담당한다.
- cross-encoder 실패·미설치·Gemma JSON 불량 시 Accurate 순서를 유지하는 fail-closed다.

### 답변

- 기대 형식 1) 핵심 요약, 2) 선박 운항/업무 영향, 3) 추후 확인 필요사항,
  4) 관련 선급 Rule/Guidance를 유지한다.
- 최신 동향은 전체 7~10 bullet, 환경규제·자율운항 5~7 bullet, 단순 Rule은
  2~3개의 직접 답을 우선한다. 근거가 더 필요한 복합 질문은 체크리스트를 생략하지 않는다.
- 문서차수·문서명·페이지·조항/결의/가이드 명칭과 `[n]`을 연결한다.
- Proposal/초안/승인/채택/발효 목표를 구분하고 근거보다 강한 의무로 바꾸지 않는다.
- Advanced는 근거 confidence를 노출하고 별도 Gemma 감사자가 누락·왜곡·인용을 검토한다.
- 감사 수정안은 허용 인용, 섹션 1~4, 필수어, 길이 검사를 통과할 때만 채택한다.
- Gemma가 만든 `[1, 11] [1]` 형태는 중복 없는 atomic `[1][11]`로 정규화해
  모든 번호가 화면 Evidence Table의 한 행과 연결되게 한다.
- UI의 `답변 도움됨` / `답변 개선 필요`는 외부 전송 없이 로컬 JSONL에 저장한다.

### 표

- 표는 별도 precise-table collection을 사용한다.
- 질문의 행·열 조건을 모두 만족하는 셀을 확정하고, 원문 PDF·페이지·crop을 표시한다.
- Advanced에서도 검증된 표 셀 경로를 일반 text listwise로 대체하지 않는다.
- 라우팅이 잘못됐다고 판단할 때 UI의 `텍스트 인덱스로 다시 검색` 또는
  `표 인덱스로 다시 검색`으로 명시적으로 재시도할 수 있다.

## 평가 결과

### Accurate 고정 150문항 기준선

| 지표 | 결과 |
|---|---:|
| 품질 proxy | 82.26% |
| 필수내용 충족도 | 64.10% |
| 행동 성공률 | 100.00% |
| 후보 문서 hit | 96.24% |
| 최종 문서 hit | 92.48% |
| 페이지 hit | 83.46% |
| 평균 E2E | 10.30초 |

### Advanced 고정 150문항 검색

| 지표 | Accurate 검색 기준선 | Advanced 최종 | 변화 |
|---|---:|---:|---:|
| 후보 문서 any hit | 96.24% | 96.99% | +0.75%p |
| 최종 문서 any hit | 94.74% | 95.49% | +0.75%p |
| 후보 exact point recall | 81.20% | 84.40% | +3.20%p |
| 최종 exact point recall | 75.00% | 76.69% | +1.69%p |
| 최종 semantic point recall | 78.57% | 79.70% | +1.13%p |
| 평균 검색시간 | 2.03초 | 13.65초 | +11.62초 |
| 중앙 검색시간 | - | 8.66초 | - |

Advanced 150문항은 오류 0건이며, 최종 문서 hit 90% 이상·최종 evidence
recall 70% 이상이라는 목표를 충족했다. 단, BM25·질문 계획·Gemma listwise 비용으로
Accurate보다 평균 검색이 11.62초 느리다. 이 수치는 답변 생성 전 검색 단계만의 시간이다.

### Advanced 실제 생성·수작업 확인

- 최종 10문항 합산(8건 일괄 + 마지막 결함 2건 동일 평가기 재실행): 오류 0,
  행동 준수·금지주장·irrelevance clean 100%, gold 문서/page hit 100%/88.89%,
  품질 proxy 87.63%, raw keypoint 충족 77.50%, groundedness proxy 77.94%,
  평균 E2E 46.98초.
- 10개 답변을 직접 읽은 판정은 8건 OK, 2건 부분 보완, 최종 실패 0건이다.
  부분 보완은 긴 MSC 연료결과의 bullet 예산과 ABS Guide에 대한 결정상태 표현이다.
- `T3-V06_p02`는 질문이 CG “명칭과 범위”를 요구하지만 gold가 PRA·showstopper까지
  요구하는 평가계약 불일치가 있어 자동 completeness가 25%다. 실제 답변은
  DNV-CG-0508과 DNV-CG-0264의 명칭·범위를 모두 포함해 수작업 OK로 판정했다.
- MSC 111 연료 안전·위험평가: 품질 45%→86.25%, completeness 0%→75%,
  gold document/page hit 100%. 수소 잠정지침 승인, 액화수소 결의 채택,
  대체연료 안전규제 작업계획을 답변에 복원했다.
- DNV-CG-0264 PRA 목적 질문: 문서카드가 아니라 명시 문서 사실 경로로 보내 품질
  45%→100%를 확인했다.
- 해양 플라스틱 2026 자료 목록: 공통 연도 범위가 적용되는 세 항목을 모두 답했다.
- MASS Code: 비강제 Code 결정, mandatory Code 2030 채택·2032 발효 목표와
  일부 대표단의 2036 의견을 구분했다.
- DNV-CP-0399: 1 kV·3 kV 전력 케이블과 150/250 V (300 V) 제어·계측
  케이블 두 대상을 중복 없이 답했다.
- 암모니아 개념승인 복합 질문: MSC 111 결과, DNV Fuel ready, KR Ammonia Ready,
  탱크·연료계통·환기/가스검지·가스확산 분석과 미확정 적용범위를 함께 생성했다.
  UI 동일 경로 최종 E2E는 118.51초였다.
- 표: RSTH 12·22·23·24 확관 후 1.14배, 10만 초과~15만 DWT 안전사용하중
  250t을 정확 셀과 원문 crop으로 확인했다. 각각 28.32초, 25.75초였다.

### 최종 답변 계약 보강

- Advanced에서는 단문/문서카드용 레거시 UI 축약을 건너뛰고 네 섹션을 끝까지 보존한다.
- 단순 Rule/Guidance 찾기는 직접 문서·범위·대표 근거 2~3개 사실로 제한한다.
- “명칭과 범위” 질문은 문서별 한 bullet 안에 두 요소를 함께 쓰며 주변 CG를 제거한다.
- 전제 검증은 첫 문장에 명시적 판정을 두고, 근거 없는 감사자 추가 bullet은 제거한다.
- 일반 선급 Rule/Guide/Requirements에는 IMO 회의식 제안·최종결정 상태를 적용하지
  않고, 근거가 draft/proposal이라고 직접 밝힌 경우에만 미확정 상태를 표시한다.
- 근거 부재 거절도 네 섹션 shell로 표시하고 값을 추정하지 않는다.
- 최종 전체 회귀는 650 tests passed 및 8개 하위 테스트 통과다.

## 모드별 운영 권고

| 모드 | 운영 권고 |
|---|---|
| Fast | 간단한 사실·초기 탐색. warm 목표 10초 이내 |
| Accurate | 기본 시연. 규정·회의·표의 속도/정확도 균형 |
| Advanced | 중요한 복합 질문·다문서 체크리스트. 30초 이상, 복합 질문 1~2분 허용 |

## 한계와 다음 개선

- 현재 코퍼스에서 실제 확인되는 회의 세션은 MEPC 84와 MSC 111 중심이다. 없는
  MEPC 80~83, MSC 107~110의 시계열 사실은 검색/프롬프트만으로 만들 수 없다.
- PDF 715개 목표 중 인덱스 문서 프로필은 714개다. 누락 1개는 원본/전처리 상태를
  확인해 다음 재인덱싱 때 보완해야 한다.
- MiniLM cross-encoder는 영어 passage 모델이다. bilingual exact query와 Gemma
  listwise의 보조 신호로만 사용하므로 한국어-영어 최종 판단을 단독으로 맡기지 않는다.
- Advanced는 품질을 우선해 LLM을 계획·재순위·생성·감사에 사용하므로 느리다.
  이후에는 facet 결과 캐시 영속화와 감사 trigger 축소로 지연을 줄일 수 있다.
- 150문항 전체 답변의 전문가 이중평가가 없으므로 “완벽”을 보장할 수 없다.
  UI 피드백을 쌓아 실패 질문을 gold keypoint와 함께 지속적으로 회귀셋에 추가해야 한다.

## 배포 체크

1. `ollama serve`와 `ollama list`에서 `gemma4:12b`를 확인한다.
2. 본문/표 Chroma, sparse sidecar, cross-encoder 로컬 폴더를 확인한다.
3. 7860의 기존 UI PID만 종료한다.
4. 프로젝트 루트의 `start.cmd`를 실행한다.
5. `http://127.0.0.1:7860`에서 Fast/Accurate/Advanced와 텍스트 3개·표 2개 예시를 확인한다.
