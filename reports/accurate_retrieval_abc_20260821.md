# Accurate Retrieval A/B/C 평가

- 평가셋: 기존 405문항에서 seed 260821로 고정 추출한 150문항(질문 수정·생성 없음)
- 공식 retrieval 범위: 실제 생성 컨텍스트 Top-10

| 변형 | Document Hit@10 | Evidence Recall@10 | 의미동등 Recall@10 | 평균 검색 | 판정 |
|---|---:|---:|---:|---:|---|
| A | 94.7% | 38.5% | 51.9% | 1.98s | GO/default |
| B | 94.0% | 42.7% | 55.3% | 3.88s | MIXED |
| C | 94.0% | 42.7% | 55.3% | 4.30s | NO-GO |

## 카테고리별

| 카테고리 | 변형 | n | 문서 Hit@10 | 근거 Recall@10 |
|---|---|---:|---:|---:|
| autonomous | A | 17 | 93.3% | 25.0% |
| autonomous | B | 17 | 93.3% | 61.7% |
| autonomous | C | 17 | 93.3% | 61.7% |
| env_regulation | A | 34 | 93.3% | 47.5% |
| env_regulation | B | 34 | 93.3% | 46.7% |
| env_regulation | C | 34 | 93.3% | 46.7% |
| meeting_outcome | A | 17 | 100.0% | 31.7% |
| meeting_outcome | B | 17 | 93.3% | 33.3% |
| meeting_outcome | C | 17 | 93.3% | 33.3% |
| rule_lookup | A | 65 | 98.3% | 36.6% |
| rule_lookup | B | 65 | 98.3% | 36.6% |
| rule_lookup | C | 65 | 98.3% | 36.6% |
| trend_summary | A | 17 | 80.0% | 48.3% |
| trend_summary | B | 17 | 80.0% | 48.3% |
| trend_summary | C | 17 | 80.0% | 48.3% |

## 회귀 판정

- A→B 신규 문서 성공: 0개 — 없음
- A→B 문서 회귀: 1개 — T3-V02_b01
- A→C 신규 문서 성공: 0개 — 없음
- A→C 문서 회귀: 1개 — T3-V02_b01

## 결론

B는 희소 용어·정확 식별자 근거 Recall을 높였지만 문서 Hit 회귀가 있어 플래그 뒤에 유지합니다. C는 B 대비 품질 이득 없이 지연만 늘어 사용하지 않습니다. 기본 UI는 기존 A로 유지합니다.
