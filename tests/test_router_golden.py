import pytest

from router.intent_router import route_question
from tests.router_golden_cases import CASES_BY_ROUTE


def _cases():
    rows = []
    for expected, questions in CASES_BY_ROUTE.items():
        for q in questions:
            rows.append((expected, q))
    return rows


@pytest.mark.parametrize("expected,question", _cases())
def test_golden_route(expected: str, question: str):
    decision = route_question(question, use_llm_fallback=False)
    assert decision.route == expected, (
        f"[{expected}] got={decision.route} "
        f"ops={decision.ops_score:.1f} rag={decision.rag_score:.1f} "
        f"mode={decision.chat_mode} | {question}"
    )


@pytest.mark.parametrize("route,questions", CASES_BY_ROUTE.items())
def test_golden_set_size(route: str, questions: list[str]):
    minimum = 12 if route == "hybrid" else 50
    assert len(questions) >= minimum
    assert len(set(questions)) == len(questions)
