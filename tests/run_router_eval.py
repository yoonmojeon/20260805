"""Print router accuracy for the 150 golden questions."""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router.intent_router import route_question
from tests.router_golden_cases import CASES_BY_ROUTE


def main() -> int:
    misses: list[tuple[str, str, str, float, float, str]] = []
    confusion = defaultdict(Counter)
    total = 0
    ok = 0

    for expected, questions in CASES_BY_ROUTE.items():
        for q in questions:
            total += 1
            d = route_question(q, use_llm_fallback=False)
            confusion[expected][d.route] += 1
            if d.route == expected:
                ok += 1
            else:
                misses.append(
                    (expected, d.route, q, d.ops_score, d.rag_score, d.reason)
                )

    print(f"router golden eval: {ok}/{total} = {ok / total:.1%}")
    print()
    print(f"{'expect':<8} {'chat':>5} {'ops':>5} {'rag':>5} {'hyb':>5}")
    for expected in ("chat", "ops", "rag", "hybrid"):
        row = confusion[expected]
        print(
            f"{expected:<8} {row['chat']:5d} {row['ops']:5d} {row['rag']:5d} {row['hybrid']:5d}"
        )

    if misses:
        print()
        print(f"misses ({len(misses)}):")
        for expected, got, q, ops, rag, reason in misses:
            print(f"  exp={expected:4} got={got:4} ops={ops:.1f} rag={rag:.1f} | {q}")
            print(f"           {reason}")
        return 1

    print()
    print("all golden cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
