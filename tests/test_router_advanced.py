from router.intent_router import route_question
from router.rewrite import split_hybrid_queries
from tests.router_heldout_cases import ADVANCED_SINGLE, HELDOUT_SINGLE, MULTITURN_SCENARIOS
from services.rag_service import _ensure_rag_path

_ensure_rag_path()
from retrieval_query_analysis import analyze_query


def test_seemp_ship_health_is_ops():
    d = route_question("올해 SEEMP 잘 되고 있어?", use_llm_fallback=False)
    assert d.route == "ops"


def test_meeting_report_is_rag_not_ops():
    d = route_question("회의 내용으로 보고서 만들어줘", use_llm_fallback=False)
    assert d.route == "rag"


def test_document_cii_is_rag():
    d = route_question("문서에서 CII 규제 찾아줘", use_llm_fallback=False)
    assert d.route == "rag"


def test_compare_frame_is_hybrid():
    d = route_question("그 규정 기준으로 우리 배는 어때", use_llm_fallback=False)
    assert d.route == "hybrid"
    assert d.ops_query
    assert d.rag_query
    assert d.ops_query != d.rag_query


def test_close_scores_without_dual_do_not_auto_hybrid():
    d = route_question("CII 규정", use_llm_fallback=False)
    assert d.route in {"rag", "chat"}
    assert d.route != "hybrid"


def test_thanks_is_chat():
    d = route_question("감사합니다", use_llm_fallback=False)
    assert d.route == "chat"
    assert d.chat_mode in {"thanks", "clarify", "identity"}


def test_hybrid_split_uses_each_side():
    ops_q, rag_q = split_hybrid_queries("우리 CII랑 MEPC 규제 같이 알려줘")
    assert "CII" in ops_q.upper()
    assert "MEPC" in rag_q.upper()


def test_named_cii_report_expands_result_and_method_terms_first():
    signals = analyze_query("MEPC 84/6/2 CII 선대 보고서를 설명해줘")
    joined = " ".join(signals.expanded_terms[:8])
    assert "10.8%" in joined
    assert "AER" in joined
    assert "EEOI" in joined


def test_abs_two_document_comparison_keeps_both_document_hints():
    signals = analyze_query(
        "ABS Guide for Smart Functions와 Requirements for Autonomous and "
        "Remote Control Functions를 위험범주와 검증 관점에서 비교해줘"
    )
    assert "ABS-Smart-Functions-Guide" in signals.rule_doc_hints
    assert "ABS-Autonomous-Remote-Requirements" in signals.rule_doc_hints
    joined = " ".join(signals.expanded_terms[:8]).lower()
    assert "optional class notation" in joined
    assert "operations supervision" in joined


def test_dialogue_state_keeps_topic():
    first = route_question("올해 CII 등급을 알려줘", use_llm_fallback=False)
    assert first.dialogue_state
    assert first.dialogue_state["last_route"] == "ops"
    second = route_question(
        "그럼 더 자세히",
        use_llm_fallback=False,
        dialogue_state=first.dialogue_state,
    )
    assert second.route == "ops"
    assert second.expanded_question
    assert "CII" in (second.expanded_question or "").upper()


def test_advanced_single_cases():
    for expected, question in ADVANCED_SINGLE:
        d = route_question(question, use_llm_fallback=False)
        assert d.route == expected, f"{question}: expected {expected}, got {d.route}"


def test_heldout_single_cases():
    misses = []
    for expected, question in HELDOUT_SINGLE:
        d = route_question(question, use_llm_fallback=False)
        if d.route != expected:
            misses.append((expected, d.route, question, d.reason))
    assert not misses, misses


def test_multiturn_scenarios():
    for scenario in MULTITURN_SCENARIOS:
        state = None
        for question, expected in scenario["turns"]:
            d = route_question(
                question, use_llm_fallback=False, dialogue_state=state
            )
            assert d.route == expected, (
                f"{scenario['id']} | {question}: expected {expected}, "
                f"got {d.route} ({d.reason})"
            )
            state = d.dialogue_state
