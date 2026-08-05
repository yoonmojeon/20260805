"""Answer template for meeting outcome questions."""
from __future__ import annotations

from meeting_outcome_retrieval import parse_outcome_item_count
from meeting_summary_context import get_summary_context
from meeting_summary_intent import is_meeting_summary_intent, validate_meeting_summary_answer
from retrieval_query_analysis import analyze_query


def session_title_from_question(question: str, row: dict | None = None) -> str:
    sig = analyze_query(question)
    if sig.session_codes:
        body, num = sig.session_codes[0]
        return f"{body} {num}"
    return "회의"


def build_meeting_summary_system_prompt(row: dict) -> str:
    question = str(row.get("question", ""))
    ctx = get_summary_context(question, row)
    n_items = parse_outcome_item_count(question, row)
    session = session_title_from_question(question, row)

    return f"""당신은 IMO 회의 결과(outcome) 요약 전문 조력자입니다.
반드시 한국어로 답변합니다. 제공된 검색 근거(context)에 없는 결의·날짜·수치·문서번호는 추측하지 마세요.

**우선 근거:** IMO 공식 meeting highlights / Secretary-General closing remarks → {session} Draft Report(WP.1) / Session Report.
**후순위·제외:** MSC 111/2, Outcome of C 135/A 34/TC 75, Strategic Plan, FAL 50 검토, 타 IMO body 결과 보고.

**출력 형식 (변경 금지):**

## {session} 주요 결과 {n_items}개

1. [핵심 결과명]
- 주요 내용: (채택/승인 결정 1~2문장)
- 실무 의미: (선박 설계·운항·규제 관점, 근거 있을 때만 1문장)
- 근거: [N] 또는 p.N

(2~{n_items}번 동일 형식)

**작성 규칙:**
- 정확히 {n_items}개의 서로 다른 **MSC 자체 핵심 결정**만 작성.
- **1번 필수:** 비강제 MASS Code 채택(근거에 있을 때, MSC 111 전체 요약 시).
- **우선 후보:** GHG safety, 대체연료/신기술 안전, LRIT, VDES, 수소·암모니아 연료 지침, 호르무즈 결의.
- **제외:** Strategic Plan, FAL 50, A 34/C 135, 타 IMO body outcome 보고, noted/invited만 있는 절차 항목.
- **금지 필드:** 적용 대상 없음, 시행/발효 시점 없음, 선박 영향 없음, 검색 결과 내 확인 불가 — 해당 줄 자체를 쓰지 말 것.
- resolution→결의안, adopted→채택, approved→승인.
"""


def build_meeting_summary_user_prompt(row: dict, context: str) -> str:
    question = str(row.get("question", ""))
    ctx = get_summary_context(question, row)
    n_items = parse_outcome_item_count(question, row)
    session = session_title_from_question(question, row)
    mass_hint = ""
    if ctx.require_topics and "MASS Code" in ctx.require_topics:
        mass_hint = "**1번은 비강제 MASS Code 채택**(근거에 있을 때). "
    return f"""질문: {question}

검색 근거 (context) — **{session} Draft Report / 회의 최종 보고서 본문을 우선 사용**:
{context}

위 context만 사용해 **{n_items}개 핵심 회의 결과**를 번호 목록으로 작성하세요.
{mass_hint}나머지는 GHG safety, 대체연료/신기술, LRIT, VDES, 호르무즈 등 핵심 결정 우선(근거 있을 때).
규정 변경 분석 형식(적용 대상/발효/영향 없음, 확인 불가)은 사용하지 마세요.
"""


def build_meeting_outcome_system_prompt(row: dict) -> str:
    if is_meeting_summary_intent(str(row.get("question", "")), row):
        return build_meeting_summary_system_prompt(row)
    question = str(row.get("question", ""))
    n_items = parse_outcome_item_count(question, row)
    session = session_title_from_question(question, row)

    return f"""당신은 IMO 회의 결과(outcome) 분석 전문 조력자입니다.
반드시 한국어로 답변합니다. 제공된 검색 근거(context)에 없는 결의·날짜·수치·문서번호는 추측하지 마세요.

**출력 형식 (변경 금지):**

## {session} 주요 결과 {n_items}개

1. [결과명]
- 결정 내용:
- 적용 대상:
- 시행/발효 시점:
- 선박 운항/설계/업무 영향:
- 근거 문서: [N]

2. [결과명]
- 결정 내용:
- 적용 대상:
- 시행/발효 시점:
- 선박 운항/설계/업무 영향:
- 근거 문서: [N]

(필요 시 3~{n_items}번까지 동일 형식 반복)

**작성 규칙:**
- 정확히 {n_items}개의 numbered outcome 항목을 작성한다.
- 각 항목은 서로 다른 결정·안건·문서를 다룬다 (중복 금지).
- context 청크는 **[1], [2], …** 번호로 표시되어 있다. **근거 문서** 줄에 해당 [N]을 반드시 표기한다.
- 결정 내용·적용 대상·시행 시점·업무 영향은 context에 있는 정보만 작성한다.
- context에 없는 항목은 해당 필드에 "검색 결과 내 확인 불가"라고 명시한다.
- MEPC/TC/선급 Guidance 등 질문 회의와 무관한 주변 문서 내용은 outcome 항목에 넣지 말 것.
- Summary Report·Outcome·Adopted/Approved 결정을 우선 반영한다.
"""


def build_meeting_outcome_user_prompt(row: dict, context: str) -> str:
    if is_meeting_summary_intent(str(row.get("question", "")), row):
        return build_meeting_summary_user_prompt(row, context)
    question = str(row.get("question", ""))
    n_items = parse_outcome_item_count(question, row)
    must = row.get("must_cover") or []
    must_block = ""
    if must:
        must_block = (
            f"\n[가능하면 반영할 필수 주제 — context에 있을 때만] {', '.join(must)}\n"
        )
    keywords = row.get("expected_keywords") or []
    kw_block = ""
    if keywords:
        kw_block = f"\n[키워드 — context에 있을 때] {', '.join(keywords)}\n"

    return f"""질문: {question}

검색 근거 (context):
{context}
{must_block}{kw_block}
위 context만 사용해 **{n_items}개 outcome 항목** 형식으로 답변하세요.
각 항목의 **근거 문서** 줄에 citation [N]을 포함하세요.
"""


def answer_depth_score(answer: str, *, min_items: int = 3, meeting_summary: bool = False) -> float:
    """Heuristic 0~1 score for meeting outcome answer structure."""
    if not answer:
        return 0.0
    if meeting_summary:
        passed, _ = validate_meeting_summary_answer(answer)
        return 1.0 if passed else 0.35
    score = 0.0
    if "결정 내용" in answer:
        score += 0.2
    if "적용 대상" in answer:
        score += 0.15
    if "시행" in answer or "발효" in answer:
        score += 0.15
    if "운항" in answer or "설계" in answer or "업무 영향" in answer:
        score += 0.15
    if "근거 문서" in answer:
        score += 0.15
    import re

    numbered = len(re.findall(r"^\s*\d+\.\s*\[", answer, re.MULTILINE))
    if numbered == 0:
        numbered = len(re.findall(r"^\s*\d+\.\s+", answer, re.MULTILINE))
    score += min(0.2, 0.2 * (numbered / max(min_items, 1)))
    return min(1.0, score)
