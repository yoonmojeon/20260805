from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "rag" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from grounded_answer_policy import classify_document_status
from imo_doc_classify import classify_imo_filename, tier_for_query
from question_profile import build_question_profile
from compound_regulatory import is_compound_regulatory_class_question
from rule_lookup_document_analysis import summarize_scope_ko


class Chunk:
    def __init__(self, file_name: str, text: str = ""):
        self.file_name = file_name
        self.source = "MEPC"
        self.text = text


def test_rule_discovery_and_exact_fact_are_separate_internal_intents():
    guide = build_question_profile(
        "DNV에서 자율운항 관련 Rule/Guidance를 찾아줘.",
        {"category": "rule_lookup"},
    )
    fact = build_question_profile(
        "DNV-CP-0399는 어떤 정격 전압에 적용되나요?",
        {"category": "rule_lookup"},
    )
    assert guide.primary_intent == "rule_document_guide"
    assert guide.answer_style == "document_cards"
    assert fact.primary_intent == "exact_fact"
    assert fact.answer_style == "short_fact"


def test_session_range_is_an_independent_time_axis():
    profile = build_question_profile("MEPC 80~84 규제 동향을 회차별로 정리해줘.")
    assert profile.time_scope == "session_range"
    assert profile.session_range == ("MEPC", 80, 84)


def test_j_documents_are_administrative_and_cannot_win_outcome_search():
    name = "MEPC 84-J-2 - Provisional List Of Documents (Secretariat).pdf"
    assert classify_imo_filename(name) == "administrative"
    assert tier_for_query(
        "administrative", wants_summary=True, wants_outcome=True, wants_agenda=False
    ) <= -2.0
    assert classify_document_status(Chunk(name)).code == "administrative"


def test_named_class_document_premise_check_is_not_compound():
    question = (
        "DNV-CG-0264를 기준으로 'DNV-CG-0264는 IMO 강제협약이다'라는 "
        "전제가 맞는지 검증하고, 틀리면 문서 근거로 바로잡아줘."
    )
    assert not is_compound_regulatory_class_question(question)


def test_real_meeting_and_class_checklist_remains_compound():
    question = (
        "MSC 111 논의와 DNV 선급 규정을 함께 근거로 암모니아 연료선 "
        "설계 검토 체크리스트를 작성해줘."
    )
    assert is_compound_regulatory_class_question(question)


def test_premise_verification_has_short_answer_profile():
    profile = build_question_profile(
        "ABS Requirements는 IMO 협약이라는 전제가 맞는지 검증하고 틀리면 바로잡아줘.",
        {"category": "rule_lookup"},
    )
    assert profile.primary_intent == "premise_verification"
    assert profile.answer_style == "short_fact"


def test_document_guide_paraphrases_without_find_word_are_detected():
    for question in (
        "ABS Smart Functions 문서를 rule lookup 형식으로 알려줘.",
        "자율운항 프로젝트에 필요한 DNV Guidance 명칭을 알려줘.",
        "문서명·핵심 요건만 bullet로 ABS 자율/원격제어 Requirements를 정리해줘.",
    ):
        profile = build_question_profile(question, {"category": "rule_lookup"})
        assert profile.answer_style == "document_cards"


def test_named_document_absent_fact_lookup_is_not_a_document_guide():
    profile = build_question_profile(
        "DNV-CG-0264에서 KR 선급의 동일 문서번호를 찾아 문서 근거와 함께 알려줘.",
        {"category": "rule_lookup"},
    )
    assert profile.answer_style == "short_fact"


def test_society_wide_guide_request_is_not_misread_as_specific_lookup():
    profile = build_question_profile(
        "DNV에서 자율운항 Rule/Guidance를 찾아줘. 확정 근거만 사용해줘.",
        {"category": "rule_lookup"},
    )
    assert profile.answer_style == "document_cards"


def test_abs_smart_guide_scope_uses_file_identity_before_exclusion_sentence():
    scope = summarize_scope_ko(
        "Semi-Autonomy and Full Autonomy functions are excluded from this Guide.",
        "GuideforSmartFunctionsforMarineVesselsandOffshoreUnits-v8.pdf",
        "class_guideline",
    )
    assert "SMART(INF/SHM/MHM)" in scope
    assert "자율운항(autonomous) 관련" not in scope
