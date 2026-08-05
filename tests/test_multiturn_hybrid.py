from services.hybrid_service import merge_hybrid_answers
from services.orchestrator import handle_question


def test_merge_hybrid_answers_labels_sources():
    text = merge_hybrid_answers("운항 답", "문서 답")
    assert "운항 데이터 (ops)" in text
    assert "규정·회의 문서 (rag)" in text
    assert "운항 답" in text
    assert "문서 답" in text


def test_handle_question_tracks_last_route():
    from router.intent_router import route_question

    first = route_question("현재 운항 상태 알려줘", use_llm_fallback=False)
    assert first.route == "ops"
    second = route_question(
        "그럼 더 자세히", use_llm_fallback=False, last_route="ops"
    )
    assert second.route == "ops"
    assert second.method == "multiturn"

    tracked = handle_question("질문", use_llm_router=False, force_route="chat")
    assert "last_route" in tracked
    assert "dialogue_state" in tracked


def test_handle_question_hybrid_route_only():
    from router.intent_router import route_question

    d = route_question("올해 배출량이랑 환경규제 동시에 설명해줘", use_llm_fallback=False)
    assert d.route == "hybrid"
