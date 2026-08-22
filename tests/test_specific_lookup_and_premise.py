from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "rag" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from rag_answer_lib import _specific_lookup_target_terms  # noqa: E402
from rag_query_router import is_rule_guidance_lookup  # noqa: E402


def test_extracts_requested_object_after_named_document_scope():
    question = (
        "MSC 111-WP.1 본회의 보고서 초안에서 MSC 111이 승인한 "
        "KR 선급부호 목록을 찾아 문서 근거와 함께 알려줘."
    )
    terms = _specific_lookup_target_terms(question)
    assert "kr" in terms
    assert "선급부호" in terms
    assert "목록" not in terms


def test_specific_lookup_terms_preserve_rare_english_identifier():
    terms = _specific_lookup_target_terms(
        "DNV-CG-0264에서 Remote Operation Centre 요구사항을 찾아 문서 근거로 알려줘."
    )
    assert {"remote", "operation", "centre"}.issubset(set(terms))


def test_named_document_fact_does_not_use_document_card_route():
    question = "DNV-CG-0264의 초기 위험평가(PRA) 목적을 정리해줘."
    assert not is_rule_guidance_lookup(question, {"category": "rule_lookup"})
