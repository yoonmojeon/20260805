# Workspace cleanup — 2026-08-21

## 삭제 결과

- 8월 17일 이전 `text_rag_eval_v3` 실행 폴더: 삭제
- 이번 고정 150문항 튜닝 중간 실행 폴더: 삭제
- `pytest_tmp*` 및 workspace `tmp` 산출물: 삭제
- Word 문서 시각검수용 PNG 렌더링 캐시: 삭제(원본 DOCX 유지)
- 8월 17일 이전 모델 비교·품질평가 결과 JSON/리뷰 Markdown: 삭제
- 총 삭제 용량: 약 **153.62MB**

## 유지한 최종 text RAG 평가

- `answers_seed9_accurate_gemma_final_20260821`
- `full405_accurate_target_final_20260821`
- `ppt_fixed150_retrieval_final_20260821`
- `ppt_fixed150_e2e_final_20260821`

## 의도적으로 유지한 항목

- `pilot_validation_text_v3.jsonl`: 고정 150문항의 원본 405문항 평가셋
- `accurate_eval_150.jsonl` 및 선택 매니페스트
- 오래된 날짜라도 현재 코드·테스트에서 참조하는 질문/정답 입력 데이터셋
- 현재 UI/Accurate 경로에서 import하는 `accurate_hybrid_v2.py`, `dynamic_evidence.py`, `compound_regulatory.py` 등 런타임 모듈
- 요청받아 생성한 Word 원본 문서

소스 Python 파일은 exact duplicate가 없었고, 오래돼 보이는 이름의 모듈도 현재 실행 경로에서 사용되고 있어 삭제하지 않았다.
