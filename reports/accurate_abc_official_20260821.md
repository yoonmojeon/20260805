# Accurate A/B/C 공식 6개 지표

고정 150문항, 동일 Gemma 4 12B·temperature 0.1·Top-10 생성 컨텍스트 기준입니다.

| Metric | A Legacy | B Hybrid RRF | C Hybrid RRF + Reranker |
|---|---:|---:|---:|
| Document Hit@10 | 92.5% | 91.7% | 91.7% |
| Evidence Recall@10 | 34.0% | 38.3% | 38.3% |
| Citation Page Accuracy | 28.3% | 28.3% | 28.3% |
| Answer Pass Rate | 96.0% | 96.0% | 96.0% |
| Groundedness | 74.7% | 74.7% | 74.7% |
| Avg E2E Latency | 9.96s | 12.45s | 12.45s |

## 판정

- A: **GO / 기본 유지**
- B: **NO-GO** — 근거 recall debug는 상승했지만 최종 문서 hit·완전성은 하락하고 E2E가 증가했습니다.
- C: **NO-GO** — B 대비 공식 품질 이득 없이 재랭커 비용만 추가됐습니다.

> Groundedness는 citation contract 기반 자동 proxy이며 전문가 entailment 판정이 아닙니다.
