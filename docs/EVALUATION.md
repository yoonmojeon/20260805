# TEXT RAG 평가 방법과 결과

## 1. 결론

단일 문서 검색은 “문서 식별” 관점에서 괜찮습니다. 특히 문서나 개념이 지정된 단일 근거 정밀검색 45문항은 후보·최종 문서 recall이 모두 100%였습니다. 다만 의미상 필요한 세부 근거까지 최종 컨텍스트에 들어온 비율은 71.11%, Gemma must-cover completeness는 61.11%였습니다.

따라서 “단일 문서를 못 찾는다”가 주 문제는 아닙니다. 현재 병목은 긴 문서 내부에서 필요한 조항을 빠짐없이 고르는 것과, 여러 문서의 근거를 한 답변에 통합하는 것입니다.

## 2. 평가셋을 만든 방법

업로드된 증강 v2 399문항은 질문 후보로 사용하되, 기존 gold를 그대로 신뢰하지 않았습니다.

1. 9개 도메인 시나리오를 정의했습니다: 최신동향, 회의결과, 환경규제 2종, 자율운항, 선급 Rule 4종.
2. 시나리오마다 답변 가능한 문서, gold chunk, 필수 답변 point, 허용 동의어를 `text_rag_scenarios_v3.json`에 기록했습니다.
3. v2에서 시나리오별로 lexical diversity가 있는 paraphrase 15개씩 선택했습니다.
4. RAGEval 방식의 completeness 계약과 MIRAGE 방식의 positive/hard-negative chunk를 붙였습니다.
5. RGB 축을 반영해 noise, negative rejection, integration, counterfactual 문항을 구성했습니다.
6. hop 수와 retrieval difficulty를 기록하고, answerability·evidence contract·중복·coverage 품질 gate를 통과시켰습니다.
7. counterfactual은 부모 질문의 모든 keypoint를 요구하지 않고, 잘못된 전제를 교정하는 데 필요한 근거만 채점하도록 고쳤습니다.

최종 구성은 405문항입니다.

| 유형 | 문항 수 | 확인 목적 |
|---|---:|---|
| paraphrase | 135 | 표현 변화에 대한 검색 안정성 |
| evidence precision | 45 | 지정 문서·개념의 정확 근거 회수 |
| noise robustness | 45 | 무관한 지시가 섞여도 답 유지 |
| negative rejection | 45 | 근거가 없을 때 거절 |
| counterfactual robustness | 36 | 잘못된 전제 교정 |
| scope | 28 | 적용범위·제외조건 |
| format | 27 | 요구 출력 형식 |
| boundary | 18 | 유사 문서·범주 경계 |
| integration | 17 | 여러 문서 통합 |
| seed | 9 | 시나리오 기준 문항 |

재현 파일:

- 원본: `data/eval/source/pilot_validation_questions_augmented_v2_with_gold.jsonl`
- 시나리오와 gold 계약: `data/eval/text_rag_scenarios_v3.json`
- 최종 405문항: `data/eval/pilot_validation_text_v3.jsonl`
- 생성기: `rag/scripts/build_text_eval_v3.py`

생성 명령:

```powershell
.\.venv\Scripts\python.exe rag\scripts\build_text_eval_v3.py
```

## 3. 검색 평가 결과

2026-08-12, `full_corpus_715_v1`, accurate mode 기준입니다. 전체 405문항 중 answerable 360문항의 검색 recall을 계산했습니다.

| 지표 | 결과 |
|---|---:|
| Candidate document any / all | 97.22% / 93.89% |
| Final document any / all | 96.94% / 90.28% |
| Candidate semantic point recall | 61.46% |
| Final semantic point recall | 46.18% |
| 평균 / 중앙 검색시간 | 1.61초 / 1.14초 |

`any`는 gold 문서 중 하나 이상, `all`은 요구한 모든 gold 문서를 회수했는지를 뜻합니다. semantic point recall은 정확히 같은 chunk ID가 아니어도 같은 문서에서 허용 동의어로 동일 사실을 담은 근거를 찾으면 인정합니다.

검색만 재실행:

```powershell
.\.venv\Scripts\python.exe rag\scripts\eval_text_retrieval_v3.py `
  --out-dir data\processed\logs\text_rag_eval_v3\retrieval_current
```

## 4. Gemma 답변 평가 결과

`gemma4:12b`로 405문항을 모두 실행했고 오류는 0건이었습니다.

| 지표 | 전체 결과 |
|---|---:|
| Must-cover completeness | 53.68% |
| Behavior pass | 99.75% |
| Answerable citation | 99.72% |
| 금지 주장 clean | 100% |
| Negative irrelevance clean | 100% |
| Quality proxy | 77.33% |
| 평균 검색 / 생성 / 전체 | 1.99초 / 6.16초 / 8.15초 |
| 전체시간 중앙값 | 7.97초 |

주요 유형별 결과:

| 유형 | 문서 검색 | Completeness | Quality | 평균 전체시간 |
|---|---:|---:|---:|---:|
| evidence precision 45 | final any/all 100% | 61.11% | 78.61% | 8.34초 |
| counterfactual 36 | final any 100% | 83.33% | 91.53% | 5.34초 |
| negative rejection 45 | 해당 없음 | 거절 100% | 100% | 1.98초 |
| integration 17 | final any/all 100%/70.59% | 39.71% | 66.84% | 9.88초 |
| scope 28 | final any/all 92.86%/82.14% | 33.93% | 63.66% | 7초대 |

답변 평가 실행:

```powershell
.\.venv\Scripts\python.exe rag\scripts\eval_text_answers_v3.py `
  --llm-model gemma4:12b `
  --out-dir data\processed\logs\text_rag_eval_v3\gemma_current
```

평가 로그는 크기와 모델 출력 때문에 `data/processed/` 아래에 로컬로 보관하고 GitHub에는 넣지 않습니다. GitHub에는 질문셋, gold 계약, 평가 코드, 위 집계 결과만 올립니다.

## 5. 해석할 때 주의할 점

- completeness는 사람이 읽는 전반적 유용성이 아니라, 미리 정의한 must-cover point가 문자열·동의어로 답에 나타났는지를 보는 보수적 지표입니다.
- quality proxy는 completeness, 인용, 거절/반박 행동을 합친 자동 점수이며 사람 평가를 완전히 대체하지 않습니다.
- 같은 사실을 담은 다른 청크를 찾는 경우 exact chunk recall은 과소평가할 수 있어 semantic point recall을 함께 봅니다.
- 현재 9개 시나리오 중심이므로 전체 715문서의 모든 주제를 대표하지 않습니다. 다음 평가는 시나리오와 사람 검수 gold를 더 넓혀야 합니다.

## 6. 다음 개선 순서

1. **문서 내부 coverage 재검색**: 답변 slot마다 문서 내 top-k를 따로 확보하고, 누락 slot만 2차 검색합니다.
2. **다문서 quota**: integration 질문은 문서별 최소 근거 수를 보장한 뒤 중복을 제거합니다.
3. **claim-evidence 검증**: 생성 전/후 각 핵심 문장을 실제 청크와 매칭해 누락과 과장을 탐지합니다.
4. **gold 확대**: 시나리오 수와 문서 범위를 늘리고, 최소 표본은 PDF 원문과 사람이 대조합니다.
5. **속도 분리**: 정확 식별자 질문은 문서 직접 진입, 단순 질문은 짧은 출력, 복잡 질문만 coverage 재검색을 수행합니다.
