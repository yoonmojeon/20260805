from router.intent_router import route_question


def test_ops_voyage_status():
    d = route_question("현재 운항 상태 알려줘", use_llm_fallback=False)
    assert d.route == "ops"


def test_ops_colloquial_location():
    d = route_question("지금 배 어디야?", use_llm_fallback=False)
    assert d.route == "ops"


def test_ops_colloquial_fuel():
    d = route_question("기름 얼마나 썼어", use_llm_fallback=False)
    assert d.route == "ops"


def test_ops_colloquial_speed():
    d = route_question("지금 스피드 얼마야", use_llm_fallback=False)
    assert d.route == "ops"


def test_ops_cii():
    d = route_question("올해 CII 등급을 알려줘", use_llm_fallback=False)
    assert d.route == "ops"


def test_rag_mepc():
    d = route_question("최신 MEPC 회의 주요 내용을 정리해줘", use_llm_fallback=False)
    assert d.route == "rag"


def test_rag_colloquial_meeting():
    d = route_question("작년에 회의에서 CII 어떻게 됐대", use_llm_fallback=False)
    assert d.route == "rag"


def test_rag_dnv_rule():
    d = route_question("DNV에서 자율운항 관련 Rule/Guidance를 찾아줘", use_llm_fallback=False)
    assert d.route == "rag"


def test_rag_cii_regulation_not_ops():
    d = route_question("MEPC에서 CII 규제 어떻게 바뀌었나", use_llm_fallback=False)
    assert d.route == "rag"


def test_rag_inspection_table():
    d = route_question("검사 주기 표 어디 있어", use_llm_fallback=False)
    assert d.route == "rag"


def test_chat_who_are_you():
    d = route_question("너 누구야?", use_llm_fallback=False)
    assert d.route == "chat"
    assert d.chat_mode == "identity"


def test_chat_greeting():
    d = route_question("안녕", use_llm_fallback=False)
    assert d.route == "chat"
    assert d.chat_mode == "greeting"


def test_chat_capabilities():
    d = route_question("뭐 할 수 있어?", use_llm_fallback=False)
    assert d.route == "chat"
    assert d.chat_mode == "identity"


def test_chat_router_meta_question():
    d = route_question("라우터가 뭘로 구분되어있어?", use_llm_fallback=False)
    assert d.route == "chat"
    assert d.chat_mode == "meta"


def test_unknown_goes_to_clarify_not_rag():
    d = route_question("아무 말", use_llm_fallback=False)
    assert d.route == "chat"
    assert d.chat_mode == "clarify"


def test_oos_weather():
    d = route_question("오늘 날씨 어때", use_llm_fallback=False)
    assert d.route == "chat"
    assert d.chat_mode == "oos"


def test_hybrid_cii_and_mepc():
    d = route_question("우리 CII랑 MEPC 규제 같이 알려줘", use_llm_fallback=False)
    assert d.route == "hybrid"


def test_multiturn_keeps_ops_on_followup():
    d = route_question("그럼 더 자세히", use_llm_fallback=False, last_route="ops")
    assert d.route == "ops"
    assert d.method == "multiturn"


def test_multiturn_switch_to_rag():
    d = route_question("문서로 봐줘", use_llm_fallback=False, last_route="ops")
    assert d.route == "rag"
    assert d.method == "multiturn"


def test_multiturn_switch_to_ops():
    d = route_question("운항 쪽으로", use_llm_fallback=False, last_route="rag")
    assert d.route == "ops"
    assert d.method == "multiturn"


def test_force_route():
    d = route_question("아무 말", force_route="ops", use_llm_fallback=False)
    assert d.route == "ops"
    assert d.method == "manual"
