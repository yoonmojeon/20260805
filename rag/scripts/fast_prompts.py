"""Type-specific Fast mode prompts."""
from __future__ import annotations

from fast_question_classifier import FastQuestionType, classify_fast_question_type
from table_retrieval import classify_table_query_mode

from fast_imo_terms import IMO_GLOSSARY_PROMPT

FAST_SYSTEM_BASE = (
    "너는 IMO/해사 도메인 문서를 기반으로 답변하는 MaritimeRAG Assistant다. "
    "검색 근거의 제목·문서번호·agenda item을 확인해 문서 성격과 범위를 판단한다. "
    "근거에 없는 내용은 추측하지 않는다. 모든 사실 문장 끝에 [N]을 붙인다. "
    "사용자가 요구하지 않은 실무 영향이나 대응 조언은 만들지 않는다. 답변은 간결하게 작성한다."
)

LOW_CONFIDENCE_NOTE = (
    "현재 Fast 검색 결과 기준으로 확인 가능한 범위는 다음과 같으며, "
    "상세 검증은 Accurate mode에서 추가 확인이 필요합니다."
)

TABLE_CELL_LEGEND = """표 기호·값 해석 (근거에 있는 내용만 사용):
- ○ : 해당 차수 정기검사에서 reporting(검사 보고) 대상
- - : 해당 차수에는 별도 요건 없음
- 숫자(1개, 2개, 절반 등) : 검사·보고할 탱크/구역 개수 또는 비율
- '대표적인' : 대표적인 평형수탱크를 선정하여 검사
- '임의로 선정' : 규정 범위 내에서 임의로 탱크/구역을 선정
- 긴 문장 셀 : 선종·적재 방식별 선정 방법을 그대로 반영해 설명"""

MATERIAL_TABLE_LEGEND = """재료·화학성분 표 해석:
- 0.035 이하, 0.18 이하 등 : 화학 **함량(%)** 상·하한 (C, Mn, P, S 등)
- 235이상, 315이상, 490 등 + N/mm² : **항복·인장강도** — 화학 함량(S, P)과 혼동 금지
- 질문이 황(S)·탄소(C) 등 **원소 함량**이면 근거 표에 해당 열/함량이 있을 때만 답한다
- 근거에 원소 함량이 없고 인장강도·용접 시험재·재료계수만 있으면 「해당 표에서 화학 함량 확인 불가」라고 한다"""


def _table_practical_body(question: str, compact_context: str, table_mode: str, row: dict) -> str:
    focus = str(row.get("answer_focus") or "").strip()
    focus_line = f"\n답변 시 특히 다룰 점: {focus}" if focus else ""

    mode_hints = {
        "row_lookup": "질문 구역의 **모든 검사 차수(열)**를 빠짐없이 다룬다.",
        "table_summary": "표 전체 구조(구역×검사차수)와 대표 구역 예시를 설명한다.",
        "column_comparison": "비교 대상 선령/차수 구간을 나란히 설명하고 차이·공통점을 문장으로 정리한다.",
        "cell_lookup": "질문이 묻는 선령·차수·구역에 **직접 답하는 문장**을 먼저 쓴다.",
    }
    mode_extra = mode_hints.get(table_mode, "질문 의도에 맞게 표 내용을 실무 관점에서 설명한다.")

    legend = TABLE_CELL_LEGEND
    if any(k in question for k in ("황", "탄소", "망간", "함량", "화학", "AH", "DH", "연강", "고장력")):
        legend = f"{TABLE_CELL_LEGEND}\n\n{MATERIAL_TABLE_LEGEND}"

    return (
        f"질문: {question}\n\n근거:\n{compact_context}\n\n"
        f"{legend}{focus_line}\n\n"
        "위 근거만 사용해 **현업 담당자가 바로 이해할 수 있는** 답변을 작성한다.\n"
        "금지: 셀 값·기호 하나만 출력, '정답: ○' 형식, 페이지·표 번호만 나열.\n"
        "금지: 질문 원소(예: 황 S)와 무관한 인장강도·용접·재료계수 수치를 화학 함량 답으로 쓰지 말 것.\n\n"
        "형식:\n"
        "- **결론:** (2~3문장. 질문에 직접 답하고 표 기호를 실무 의미로 풀어 설명)\n"
        "- **검사·reporting 요건:** (해당 구역·선령·차수별 bullet, 각 bullet 1문장 이상)\n"
        "- **실무 포인트:** (검사 계획·선정 시 유의사항 1~2 bullet, 근거 있을 때만)\n"
        "- **근거:** 문서·페이지·표 [N]\n\n"
        f"추가 지침: {mode_extra}"
    )


def resolve_table_prompt_mode(row: dict, question: str) -> str:
    """Map eval question_type or heuristics to table prompt sub-mode."""
    explicit = str(row.get("question_type") or "").strip().lower()
    if explicit in {"cell_lookup", "row_lookup", "table_summary", "column_comparison", "max/min"}:
        if explicit == "max/min":
            return "cell_lookup"
        return explicit
    return classify_table_query_mode(question)


def build_fast_system_prompt(fast_type: FastQuestionType) -> str:
    hints = {
        "meeting_summary": (
            "회의 Draft Report(WP.1 등)에서 adopted/approved 핵심 결정만 요약한다. "
            "다른 body outcome·FAL 검토·절차 항목은 제외한다."
        ),
        "meeting_outcome_question": "회의 outcome·adopted/approved 결정을 우선 반영한다.",
        "table_question": (
            "표·수치 규정을 검사·reporting 실무 관점에서 문장으로 설명한다. "
            "셀 기호·단어를 그대로만 답하지 않는다."
        ),
        "rule_question": "조항·적용범위·실무 영향을 구분한다.",
        "broad_summary_question": "서로 다른 문서 근거를 bullet마다 구분한다.",
        "figure_or_diagram_question": "그림/캡션 설명만 사용한다.",
    }
    extra = hints.get(fast_type, "")
    return f"{FAST_SYSTEM_BASE} {extra}\n{IMO_GLOSSARY_PROMPT}".strip()


def build_fast_user_prompt(
    row: dict,
    compact_context: str,
    *,
    fast_type: FastQuestionType | None = None,
    low_confidence: bool = False,
) -> str:
    question = str(row.get("question", ""))
    qtype = fast_type or classify_fast_question_type(question, row)
    conf_block = f"\n\n**참고:** {LOW_CONFIDENCE_NOTE}" if low_confidence else ""

    if qtype == "meeting_summary":
        n = int(row.get("outcome_item_count") or 3)
        m = __import__("re").search(r"(\d+)\s*개", question)
        if m:
            n = int(m.group(1))
        body = (
            f"질문: {question}\n\n근거:\n{compact_context}\n\n"
            f"위 근거(WP.1 Draft Report 우선)만 사용해 **핵심 회의 결과 {n}개**를 번호 목록으로 답변해줘.\n"
            "형식: 각 항목 **핵심 결과명** / 문서에 명시된 주요 내용 / 근거 [N].\n"
            "1번은 비강제 MASS Code 채택(근거 있을 때). GHG·대체연료·LRIT·VDES·호르무즈 등 우선.\n"
            "금지: 적용 대상/발효/영향 없음, Strategic Plan, FAL 50, A 34/C 135, 타 body outcome."
        )
    elif qtype == "meeting_outcome_question":
        n = int(row.get("outcome_item_count") or 3)
        body = (
            f"질문: {question}\n\n근거:\n{compact_context}\n\n"
            f"위 근거만 사용해 **주요 결과 {n}~5개**를 번호 목록으로 답변해줘.\n"
            "각 항목마다 문서에 명시된 **결정·논의 내용 한 문장**과 **근거 [N]**만 쓴다.\n"
            "사용자가 묻지 않은 영향은 쓰지 않는다. citation [N] 필수."
        )
    elif qtype == "table_question":
        table_mode = resolve_table_prompt_mode(row, question)
        body = _table_practical_body(question, compact_context, table_mode, row)
    elif qtype == "rule_question":
        body = (
            f"질문: {question}\n\n근거:\n{compact_context}\n\n"
            "형식:\n"
            "- **결론:** (1~2문장)\n"
            "- **근거 조항:** 문서명·조항 + [N]\n"
            "- **실무 영향:** (1문장)\n"
            "- **추가 확인:** (context에 없을 때만)\n"
        )
    elif qtype == "broad_summary_question":
        body = (
            f"질문: {question}\n\n근거:\n{compact_context}\n\n"
            "핵심 **3~5 bullet**. 각 bullet은 직접 근거가 있는 한 문장 + citation [N]. "
            "서로 다른 근거 문서를 우선 사용."
        )
    elif qtype == "figure_or_diagram_question":
        body = (
            f"질문: {question}\n\n근거:\n{compact_context}\n\n"
            "그림/캡션 근거로 2~4 bullet. 각 bullet 끝 [N]. "
            "시각 정보가 없으면 '그림 근거 부족' 명시."
        )
    else:
        body = (
            f"질문: {question}\n\n근거:\n{compact_context}\n\n"
            "핵심만 3~5 bullet. 각 1~2문장. citation [N] 가능하면 포함."
        )

    if not low_confidence and qtype != "table_question":
        body += "\n마지막 줄: '상세 분석은 Accurate mode에서 수행 가능합니다.'"
    return body + conf_block
