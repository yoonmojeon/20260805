from prompts.chat import render_chat_answer
from services.orchestrator import handle_question


def test_render_identity_mentions_product():
    text = render_chat_answer("너 누구야?", chat_mode="identity")
    assert "MaritimeOpsRAG" in text
    assert "운항" in text
    assert "문서" in text


def test_unknown_clarifies_instead_of_rejecting():
    result = handle_question("아무 말", use_llm_router=False)
    assert result["route"]["route"] == "chat"
    assert result["route"]["chat_mode"] == "clarify"
    assert "범위가 아닙니다" not in result["answer"]
    assert "운항" in result["answer"]
    assert "문서" in result["answer"]


def test_clarify_directly_states_both_capabilities_are_available():
    text = render_chat_answer(
        "운항 정보와 규정 문서를 둘 다 찾아볼 수 있는 서비스야?",
        chat_mode="clarify",
    )
    assert "둘 다 찾아볼 수 있습니다" in text


def test_router_question_explains_paths():
    result = handle_question("라우터가 뭘로 구분되어있어?", use_llm_router=False)
    assert result["route"]["route"] == "chat"
    assert "ops" in result["answer"].lower()
    assert "rag" in result["answer"].lower()


def test_handle_who_are_you_does_not_use_rag():
    result = handle_question("너 누구야?", use_llm_router=False)
    assert result["route"]["route"] == "chat"
    assert result["source"] == "chat"
    assert "MaritimeOpsRAG" in result["answer"]
