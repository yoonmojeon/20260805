"""Dynamic evidence budget for Fast mode slot selection.

Fast mode used one fixed chunk count per question type (3 for general, 4 for
rule).  A definition needs one or two chunks, while "적용 대상 + 예외 + 검사 주기
+ 조건" needs one chunk per facet.  This module sizes the budget from two
signals: how many retrieved chunks still score close to the best one, and how
many distinct facets the question asks about.
"""
from __future__ import annotations

import os
import re
import statistics
from dataclasses import dataclass

FACET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("scope", re.compile(r"적용\s*(?:대상|범위)|적용되는|해당\s*선박|applicab|\bscope\b", re.I)),
    ("exception", re.compile(r"예외|제외|면제|다만|except|exempt|unless", re.I)),
    (
        "interval",
        re.compile(r"주기|간격|기한|몇\s*년|몇\s*개월|interval|periodic|frequenc|renewal", re.I),
    ),
    ("condition", re.compile(r"조건|요건|기준|criteri|condition|requirement", re.I)),
    ("procedure", re.compile(r"절차|방법|순서|procedure|process", re.I)),
    ("value", re.compile(r"몇\s*(?:mm|톤|퍼센트|%)|얼마|수치|두께|비율|이상|이하|최소|최대", re.I)),
    ("definition", re.compile(r"정의|무슨\s*뜻|무엇을\s*뜻|의미(?:는|가)|definition|\bmeans\b", re.I)),
)

# A knee only counts when the score drop clearly exceeds the routine spacing
# between neighbouring chunks; E5 similarities sit within a few thousandths.
KNEE_MIN_GAP = 0.01
KNEE_GAP_RATIO = 2.0


@dataclass(frozen=True)
class EvidenceBudget:
    count: int
    max_docs: int
    facets: tuple[str, ...]
    basis: str

    def as_meta(self) -> dict[str, object]:
        return {
            "count": self.count,
            "max_docs": self.max_docs,
            "facets": list(self.facets),
            "basis": self.basis,
        }


def dynamic_evidence_enabled() -> bool:
    return os.environ.get("MARITIME_DYNAMIC_EVIDENCE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def detect_facets(question: str) -> tuple[str, ...]:
    text = str(question or "")
    found = [name for name, pattern in FACET_PATTERNS if pattern.search(text)]
    # An enumerated ask ("세 가지", "3개") is itself a demand for more evidence.
    if re.search(r"(\d+)\s*(?:가지|개)\s*(?:를|을)?\s*(?:알려|설명|나열|말해)", text):
        found.append("enumerated")
    return tuple(dict.fromkeys(found))


def knee_cut(distances: list[float], *, floor: int, ceiling: int) -> int | None:
    """Index where retrieval quality drops off, or None when the pool is flat."""
    values = [float(d) for d in distances[: max(ceiling + 1, floor + 1)]]
    if len(values) < floor + 1:
        return None
    gaps = [values[i] - values[i - 1] for i in range(1, len(values))]
    gaps = [gap for gap in gaps if gap >= 0]
    if len(gaps) < 2:
        return None
    typical = statistics.median(gaps) or 0.0
    best_cut: int | None = None
    best_gap = 0.0
    for i in range(1, len(values)):
        cut = i  # keep values[:cut]
        if cut < floor or cut > ceiling:
            continue
        gap = values[i] - values[i - 1]
        if gap < max(KNEE_MIN_GAP, typical * KNEE_GAP_RATIO):
            continue
        if gap > best_gap:
            best_gap = gap
            best_cut = cut
    return best_cut


def plan_evidence_budget(
    distances: list[float],
    question: str,
    *,
    base_count: int,
    base_max_docs: int,
    floor: int = 2,
    ceiling: int = 6,
) -> EvidenceBudget:
    facets = detect_facets(question)
    facet_count = len(facets)

    if facets == ("definition",):
        demand = min(base_count, max(floor, 2))
    elif facet_count <= 1:
        demand = base_count
    else:
        demand = min(ceiling, base_count + facet_count - 1)

    knee = knee_cut(distances, floor=floor, ceiling=max(demand, floor))
    if knee is not None and knee < demand:
        count, basis = knee, "score_knee"
    else:
        count, basis = demand, "facet_demand" if facet_count > 1 else "base"

    count = max(floor, min(count, ceiling))
    if distances:
        count = min(count, len(distances))
    max_docs = base_max_docs + (1 if facet_count >= 3 else 0)
    return EvidenceBudget(count=count, max_docs=max_docs, facets=facets, basis=basis)
