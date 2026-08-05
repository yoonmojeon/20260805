from router.intent_router import route_question


def test_ops_voyage_status():
    d = route_question("현재 운항 상태 알려줘", use_llm_fallback=False)
    assert d.route == "ops"


def test_ops_cii():
    d = route_question("올해 CII 등급을 알려줘", use_llm_fallback=False)
    assert d.route == "ops"


def test_rag_mepc():
    d = route_question("최신 MEPC 회의 주요 내용을 정리해줘", use_llm_fallback=False)
    assert d.route == "rag"


def test_rag_dnv_rule():
    d = route_question("DNV에서 자율운항 관련 Rule/Guidance를 찾아줘", use_llm_fallback=False)
    assert d.route == "rag"


def test_force_route():
    d = route_question("아무 말", force_route="ops", use_llm_fallback=False)
    assert d.route == "ops"
    assert d.method == "manual"
