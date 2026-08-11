from router.intent_router import route_question


def test_ops_voyage_status():
    d = route_question("현재 운항 상태 알려줘", use_llm_fallback=False)
    assert d.route == "ops"


def test_ops_colloquial_current_voyage():
    d = route_question("지금 항차 상태 요약해줘", use_llm_fallback=False)
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


def test_rag_structural_min_thickness_table():
    d = route_question(
        "선박 길이 L이 170m 미만일 때 요구되는 최소 두께는 얼마인가?",
        use_llm_fallback=False,
    )
    assert d.route == "rag"
    assert d.rag_score > 0


def test_rag_corrosion_tcorr():
    d = route_question(
        "화물탱크 내 구조부재 부식추가 tcorr 표에서 범주별 값은 어떻게 되나?",
        use_llm_fallback=False,
    )
    assert d.route == "rag"


def test_rag_short_term_tcorr_colloquial():
    """Hangul after Latin term must not break tcorr boundary (\\btcorr\\b fails)."""
    d = route_question("tcorr가 뭐야?", use_llm_fallback=False)
    assert d.route == "rag"
    assert d.rag_score > 0


def test_rag_open_table_cell_shapes():
    for q in (
        "재화중량이 10만 톤 초과 15만 톤 이하인 선박의 안전사용하중은 몇 톤인가?",
        "넘침식 또는 순차식 평형수 교환에는 어떤 설계하중 시나리오를 적용하는가?",
        "호퍼탱크 경사판과 연결된 이중선측 수평거더 웨브는 어떤 방법으로 평가하는가?",
    ):
        d = route_question(q, use_llm_fallback=False)
        assert d.route == "rag", (q, d.route, d.reason)


def test_rag_definition_substantial_corrosion():
    d = route_question(
        "과도한 부식(substantial corrosion)의 정의는?",
        use_llm_fallback=False,
    )
    assert d.route == "rag"
    assert d.rag_score > 0


def test_rag_definition_shape_without_society():
    d = route_question("허용 부식여유의 정의는 무엇인가?", use_llm_fallback=False)
    assert d.route == "rag"


def test_rag_table_number_chemical():
    d = route_question("표 2.1.65 종류 및 화학성분 표의 주요 열 구성은?", use_llm_fallback=False)
    assert d.route == "rag"


def test_rag_ah32_welding():
    d = route_question("AH32 용접강 관련 표에서 확인해야 할 주요 항목은 무엇인가?", use_llm_fallback=False)
    assert d.route == "rag"


def test_rag_age_cargo_hold_reporting():
    d = route_question(
        "선령 5~10년 구간 선박의 화물창은 정기검사에서 어떤 reporting 요건이 있나?",
        use_llm_fallback=False,
    )
    assert d.route == "rag"


def test_shape_overrides_llm_chat_to_rag():
    """If LLM says chat on a technical table question, shape policy must force rag."""
    from router import intent_router as ir

    def fake_llm(*_a, **_k):
        return {
            "need_ops": False,
            "need_documents": False,
            "ops_query": "",
            "rag_query": "",
            "reason": "검색 불필요",
            "confidence": 0.4,
        }

    q = "요구되는 최소 판두께 값은?"
    original_score = ir.score_question
    original_llm = ir._llm_classify

    def zero_score(_q):
        return 0.0, 0.0

    ir.score_question = zero_score  # type: ignore[assignment]
    ir._llm_classify = fake_llm  # type: ignore[assignment]
    try:
        d = ir.route_question(q, use_llm_fallback=True)
        assert d.route == "rag", (d.route, d.method, d.reason)
        assert d.fallback_used is True
        assert "shape" in str((d.slots or {}).get("fallback_method") or "")
    finally:
        ir.score_question = original_score  # type: ignore[assignment]
        ir._llm_classify = original_llm  # type: ignore[assignment]



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


def test_llm_primary_derives_route_from_source_needs():
    from router import intent_router as ir

    seen = {}

    def fake_llm(*_a, **kwargs):
        seen["model"] = kwargs["model"]
        return {
            "need_ops": True,
            "need_documents": True,
            "confidence": 0.93,
            "reason": "선박 수치와 규정이 모두 필요함",
            "ops_query": "현재 우리 선박 탄소 성적 조회",
            "rag_query": "국제 탄소집약도 기준 검색",
        }

    original = ir._llm_classify
    ir._llm_classify = fake_llm  # type: ignore[assignment]
    try:
        d = ir.route_question(
            "우리 배 탄소 성적이 국제 기준에 맞는지 봐줘",
            use_llm_fallback=True,
            active_model="gemma4:12b",
        )
    finally:
        ir._llm_classify = original  # type: ignore[assignment]
    assert d.route == "hybrid"
    assert d.method == "llm"
    assert d.need_ops is True and d.need_documents is True
    assert d.model == d.router_model == "gemma4:12b"
    assert seen["model"] == "gemma4:12b"


def test_llm_hybrid_is_guarded_when_plain_ops_question_has_no_document_cue():
    from router import intent_router as ir

    def fake_llm(*_a, **_kwargs):
        return {
            "need_ops": True,
            "need_documents": True,
            "confidence": 0.93,
            "reason": "CII 용어가 있어 두 소스가 필요하다고 추정",
            "ops_query": "2026년 누적 CII attained required",
            "rag_query": "CII 규정 검색",
        }

    original = ir._llm_classify
    ir._llm_classify = fake_llm  # type: ignore[assignment]
    try:
        d = ir.route_question(
            "2026년 누적 탄소집약도 성적표를 보여줘. attained와 required도 같이.",
            use_llm_fallback=True,
        )
    finally:
        ir._llm_classify = original  # type: ignore[assignment]
    assert d.route == "ops"
    assert d.method == "llm_guarded"
    assert d.need_ops is True and d.need_documents is False
    assert d.rag_query is None


def test_llm_chat_is_guarded_for_explicit_rule_clause_question():
    from router import intent_router as ir

    def fake_llm(*_a, **_kwargs):
        return {
            "need_ops": False,
            "need_documents": False,
            "confidence": 0.96,
            "reason": "일반 대화로 잘못 추정",
            "ops_query": "",
            "rag_query": "",
        }

    original = ir._llm_classify
    ir._llm_classify = fake_llm  # type: ignore[assignment]
    try:
        d = ir.route_question(
            "510 문서준수확인서는 누가 신청할 수 있는가?",
            use_llm_fallback=True,
        )
    finally:
        ir._llm_classify = original  # type: ignore[assignment]
    assert d.route == "rag"
    assert d.method == "llm_guarded"
    assert d.need_documents is True


def test_invalid_boolean_uses_deterministic_fallback():
    from router import intent_router as ir

    def fake_llm(*_a, **_kwargs):
        return {
            "need_ops": "true",
            "need_documents": False,
            "confidence": 0.99,
            "reason": "invalid boolean",
            "ops_query": "",
            "rag_query": "",
        }

    original = ir._llm_classify
    ir._llm_classify = fake_llm  # type: ignore[assignment]
    try:
        d = ir.route_question("지금 배 어디야?", use_llm_fallback=True)
    finally:
        ir._llm_classify = original  # type: ignore[assignment]
    assert d.route == "ops"
    assert d.method == "llm_fallback_rules"
    assert d.fallback_used is True
