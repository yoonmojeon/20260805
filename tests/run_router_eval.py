"""Print router accuracy for golden, held-out, and multi-turn cases."""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router.intent_router import route_question
from tests.router_golden_cases import CASES_BY_ROUTE
from tests.router_heldout_cases import HELDOUT_SINGLE, MULTITURN_SCENARIOS


def _print_confusion(title: str, confusion: dict, labels: tuple[str, ...]) -> None:
    print(title)
    header = f"{'expect':<8}" + "".join(f"{lab[:5]:>6}" for lab in labels)
    print(header)
    for expected in labels:
        row = confusion[expected]
        print(f"{expected:<8}" + "".join(f"{row[lab]:6d}" for lab in labels))


def eval_single(cases_by_route: dict[str, list[str]]) -> tuple[int, int, list]:
    misses = []
    confusion: dict = defaultdict(Counter)
    total = ok = 0
    for expected, questions in cases_by_route.items():
        for q in questions:
            total += 1
            d = route_question(q, use_llm_fallback=False)
            confusion[expected][d.route] += 1
            if d.route == expected:
                ok += 1
            else:
                misses.append((expected, d.route, q, d.ops_score, d.rag_score, d.reason))
    return ok, total, misses, confusion


def eval_pairs(pairs: list[tuple[str, str]]) -> tuple[int, int, list]:
    misses = []
    total = ok = 0
    for expected, q in pairs:
        total += 1
        d = route_question(q, use_llm_fallback=False)
        if d.route == expected:
            ok += 1
        else:
            misses.append((expected, d.route, q, d.ops_score, d.rag_score, d.reason))
    return ok, total, misses


def eval_multiturn() -> tuple[int, int, list]:
    misses = []
    total = ok = 0
    for scenario in MULTITURN_SCENARIOS:
        state = None
        for q, expected in scenario["turns"]:
            total += 1
            d = route_question(q, use_llm_fallback=False, dialogue_state=state)
            if d.route == expected:
                ok += 1
            else:
                misses.append(
                    (scenario["id"], expected, d.route, q, d.reason)
                )
            state = d.dialogue_state
    return ok, total, misses


def main() -> int:
    ok, total, misses, confusion = eval_single(CASES_BY_ROUTE)
    print(f"router golden eval: {ok}/{total} = {ok / total:.1%}")
    print()
    _print_confusion("golden confusion", confusion, ("chat", "ops", "rag", "hybrid"))
    failed = False
    if misses:
        failed = True
        print()
        print(f"golden misses ({len(misses)}):")
        for expected, got, q, ops, rag, reason in misses:
            print(f"  exp={expected:6} got={got:6} ops={ops:.1f} rag={rag:.1f} | {q}")
            print(f"           {reason}")
    else:
        print()
        print("all golden cases passed")

    h_ok, h_total, h_misses = eval_pairs(HELDOUT_SINGLE)
    print()
    print(f"held-out single: {h_ok}/{h_total} = {h_ok / max(h_total, 1):.1%}")
    if h_misses:
        failed = True
        for expected, got, q, ops, rag, reason in h_misses:
            print(f"  exp={expected:6} got={got:6} ops={ops:.1f} rag={rag:.1f} | {q}")
            print(f"           {reason}")

    m_ok, m_total, m_misses = eval_multiturn()
    print()
    print(f"multi-turn scenarios: {m_ok}/{m_total} = {m_ok / max(m_total, 1):.1%}")
    if m_misses:
        failed = True
        for sid, expected, got, q, reason in m_misses:
            print(f"  {sid}: exp={expected} got={got} | {q}")
            print(f"           {reason}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
