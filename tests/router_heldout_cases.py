"""Held-out paraphrases, typos, and multi-turn scenarios. Not used to tune cues first."""

HELDOUT_SINGLE: list[tuple[str, str]] = [
    ("ops", "배 지금 어디 떠 있어"),
    ("ops", "이번 항차 기름값 얼마나 나왔어"),
    ("ops", "씨아이아이 올해 등급 뭐야"),
    ("ops", "눈리포트 하나 뽑아줘"),
    ("rag", "멥시에서 뭐 바뀌었대"),
    ("rag", "디엔브이 자율운항 가이드 찾아줘"),
    ("rag", "검사주기 표 좀"),
    ("rag", "작년에 아이모 회의 CII 결론"),
    ("chat", "너는 뭐하는 프로그램이야"),
    ("chat", "고마워요"),
    ("hybrid", "우리 배출량이랑 마르폴 요건 같이"),
    ("hybrid", "CII랑 규정 둘다 봐줘"),
]

MULTITURN_SCENARIOS: list[dict] = [
    {
        "id": "keep_ops_detail",
        "turns": [
            ("올해 CII 등급을 알려줘", "ops"),
            ("그럼 더 자세히", "ops"),
        ],
    },
    {
        "id": "switch_to_docs",
        "turns": [
            ("올해 CII 등급을 알려줘", "ops"),
            ("문서로 봐줘", "rag"),
        ],
    },
    {
        "id": "doc_then_our_ship",
        "turns": [
            ("최신 MEPC 회의 주요 내용을 정리해줘", "rag"),
            ("그럼 우리 배는 괜찮아?", "hybrid"),
        ],
    },
    {
        "id": "compare_rule_to_ship",
        "turns": [
            ("MARPOL Annex VI 요건", "rag"),
            ("그 규정 기준으로 우리 배는?", "hybrid"),
        ],
    },
    {
        "id": "new_ops_after_rag",
        "turns": [
            ("최신 MEPC 회의 주요 내용을 정리해줘", "rag"),
            ("우리 배 CII 얼마야", "ops"),
        ],
    },
]

ADVANCED_SINGLE: list[tuple[str, str]] = [
    ("ops", "올해 SEEMP 잘 되고 있어?"),
    ("rag", "회의 내용으로 보고서 만들어줘"),
    ("rag", "문서에서 CII 규제 찾아줘"),
    ("hybrid", "그 규정 기준으로 우리 배는 어때"),
    ("chat", "이름이 뭐야"),
    ("chat", "감사합니다"),
]
