# Accurate RAG 한화오션 피드백 반영 최종 보고

- 기준일: 2026-08-22
- 평가 경로: UI Accurate와 동일한 in-process 검색 → Gemma 4 12B 생성 → 답변 계약/인용 검증
- 고정 평가셋: `data/eval/accurate_eval_150.jsonl` (150문항, 답변 가능 133 / 부재·거절 17)
- 최종 전체 로그: `data/processed/logs/text_rag_eval_v3/fixed150_e2e_hanwha_stable_20260822`
- 기존 기준선: `data/processed/logs/text_rag_eval_v3/ppt_fixed150_e2e_final_20260821`

## 결론

현재 인덱스를 다시 임베딩하지 않고도 Accurate 답변 품질이 기존보다 상승했다.
검색 recall은 그대로 보존했고, 질문 행위 분류·문서 프로필·권위/상태 판별·가이드형
답변·전제/부재 검증을 개선했다. 보호형 Dense+BM25+RRF도 구현하고 같은 150문항으로
검증했으나 품질 이득 없이 검색시간만 증가하여 기본값으로 켜지 않았다.

## 150문항 E2E 전후 비교

| 지표 | 기존 | 최종 | 변화 |
|---|---:|---:|---:|
| 전체 품질 점수 | 79.59% | 82.26% | +2.67%p |
| 필수내용 충족도 | 60.34% | 64.10% | +3.76%p |
| 답변/행동 성공률 | 97.33% | 100.00% | +2.67%p |
| 근거성 | 86.00% | 86.00% | 유지 |
| 인용 페이지 정확도 | 38.97% | 39.00% | 유지 |
| 답변 가능 문항 인용률 | 100.00% | 99.25% | -0.75%p |
| 금지 주장 미발생률 | 100.00% | 100.00% | 유지 |
| 최종 문서 hit | 92.48% | 92.48% | 유지 |
| 후보 문서 hit | 96.24% | 96.24% | 유지 |
| 페이지 hit | 83.46% | 83.46% | 유지 |
| 평균 검색시간 | 2.01초 | 2.09초 | +0.08초 |
| 평균 생성시간 | 7.96초 | 8.21초 | +0.25초 |
| 평균 E2E | 9.97초 | 10.30초 | +0.32초 |

최종 전체 실행 후 정규식 범위를 더 좁힌 영향 문항 4개를 별도로 재검증했고, 모두
추가 상승했다(품질 0.5875→0.725, 0.25→0.725, 0.5875→0.8625,
0.45→0.5875). 이 추가 상승은 위 보수적 헤드라인 수치에는 합산하지 않았다.

## 검색 A/B

| 검색 구성 | 후보 문서 hit | 최종 문서 hit | 후보 근거 point recall | 최종 point recall | 최종 semantic point recall | 평균 검색시간 |
|---|---:|---:|---:|---:|---:|---:|
| 현재 Dense 중심 기본값 | 96.24% | 94.74% | 81.20% | 75.00% | 78.57% | 2.03초 |
| 보호형 Dense+BM25+RRF | 96.24% | 94.74% | 81.20% | 75.00% | 78.57% | 2.88초 |

BM25/RRF는 Dense 상위 24개를 보호하고 결과를 최대 80개 union하는 방식으로
구현되어 있다. 그러나 고정셋에서 recall이 전혀 오르지 않고 평균 +0.85초가 들어
기본값은 OFF로 유지했다. 목적형 MSC/MEPC 회의 검색기도 일반 hybrid로 대체하지
않도록 보호했다.

## 반영한 개선

1. 질문을 UI 4분류 외에 `행위/답변형식/시간범위/권위요구` 축으로 추가 분류했다.
   정확 수치·조항, 문서 가이드, 시계열, 영향 분석, 전제 검증을 분리한다.
2. 기존 FTS/청크 인덱스에서 714개 PDF 문서 프로필을 자동 생성했다. publisher,
   source type, document id, session, 문서 성격, 활용 시점, Rev/Add 관계를 사용한다.
3. 단순 Rule 안내는 최신 피드백대로 2~3 bullet 강제 제한을 제거했다. 문서별로
   문서명·성격·적용범위·활용 시점·관련 참조문서를 안내한다.
4. `DNV-CG-0264`, `ABS Smart Functions`, `Requirements`처럼 제목이 가까운 문서는
   직접 문서와 주변 문서를 구분하고, 제외 조건을 검색 끝까지 유지한다.
5. Proposal/Report/Outcome/Amendments/Resolution/INF/J 상태를 분리했다. J 운영
   문서와 administrative 문서는 최종 결정 근거가 될 수 없다.
6. “이 선급 문서는 IMO 협약이다” 같은 전제 검증은 모든 Accurate 카테고리에서
   전용 판정 경로를 사용한다. 첫 bullet의 명시적 판정이 후처리에서 사라지지 않는다.
7. “지정 문서에서 없는 인증번호/타 선급 번호를 찾아줘”는 일반 설명으로 넘어가지
   않고 확인 불가·추정 금지로 종료한다.
8. 특정 세션 범위/최신 질문은 현재 코퍼스 범위를 확인하고, 미수록 회차를 답변에
   명시하는 coverage guard를 적용했다.
9. Rule 문서 안내에 필요한 대표 청크를 문서별로 선발해 한 PDF의 반복 청크가 다른
   핵심 문서를 밀어내지 않도록 했다.

## 현재 코퍼스의 객관적 범위와 남은 한계

문서 프로필 기준 714개 PDF는 DNV 318, IMO 308, KR 73, ABS 14, LR 1개다.
문서 유형은 class rule 406, meeting record 286, amendments 9, resolution 4,
administrative 9개다. 회의 세션은 현재 MEPC 84(203개)와 MSC 111(105개)만
확인된다.

따라서 피드백이 요구한 MEPC 80~84, MSC 107~111의 실제 시계열 답변과 MARPOL,
SOLAS, COLREG, STCW, MASS Code 본체 기반 답변을 완성하려면 해당 원문 PDF 추가와
인덱스 갱신이 필요하다. 현재 코드는 이 부재를 숨기거나 추정하지 않고 범위 경고로
표시한다. 이 부분은 검색 알고리즘이나 프롬프트만으로 해결할 수 없다.

## 검증 및 보존 파일

- 전체 자동 테스트: 435 passed
- UI: `http://127.0.0.1:7860` 재기동 후 HTTP 200 확인
- UI 서버 설정: Fast/Accurate 선택 및 텍스트/표 인덱스 재검색 버튼 4개 확인
- 최종 E2E: `data/processed/logs/text_rag_eval_v3/fixed150_e2e_hanwha_stable_20260822`
- 기존 E2E: `data/processed/logs/text_rag_eval_v3/ppt_fixed150_e2e_final_20260821`
- 검색 기준선: `data/processed/logs/text_rag_eval_v3/ppt_fixed150_retrieval_final_20260821`
- BM25/RRF A/B: `data/processed/logs/text_rag_eval_v3/fixed150_protected_hybrid_final_20260821`
- 문서 프로필: `data/processed/index/unified_full_corpus_715_v1/document_profiles_v1.json`
- 최종 후 영향 문항 검증: `data/processed/logs/text_rag_eval_v3/smoke_guide_scope_stability_20260822`

중간 실패/스모크 평가 폴더 23개와 테스트 임시 폴더는 삭제했다. 위 기준선·최종값·
A/B 근거와 기존 405문항 기록은 보존했다.
