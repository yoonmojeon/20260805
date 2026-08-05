"""IMO document terminology normalization for Fast answers."""
from __future__ import annotations

import re

# Post-process Korean output: wrong translations → correct
KO_FIXES = [
    (re.compile(r"(\d+)\s*개의\s*결정(?!사항)"), r"\1건의 결의안"),
    (re.compile(r"(\d+)\s*건의\s*결정(?!사항)"), r"\1건의 결의안"),
    (re.compile(r"(\d+)\s*decisions\b", re.I), r"\1건의 결의안"),
    (re.compile(r"(\d+)\s*resolutions\b", re.I), r"\1건의 결의안"),
    (re.compile(r"결의안안"), "결의안"),
]

IMO_GLOSSARY_PROMPT = """
## IMO 문서 용어 (번역 시 반드시 구분)
- resolution = 결의안 (복수: N건의 결의안). **결정과 혼동 금지**
- decision = 결정
- adopted = 채택했다
- approved = 승인했다
- endorsed = 승인/지지했다 (문맥상 승인했다)
- noted = 주목했다 / 확인했다
- invited = 요청/권고/초청했다 (문맥)
- requested = 요청했다
- Committee = 위원회
- Assembly = 총회
- Council = 이사회
- plenary meeting = 본회의
- public release = 공개 배포
- action requested = 위원회에 요청된 조치

예: "22 resolutions" → "22건의 결의안" (절대 "22개의 결정" 아님)
"""


def normalize_imo_terms(text: str) -> str:
    if not text:
        return text
    out = text
    for pat, repl in KO_FIXES:
        out = pat.sub(repl, out)
    return out


def detect_terminology_violations(text: str) -> list[str]:
    violations: list[str] = []
    if re.search(r"\d+\s*개의\s*결정", text) and "결의안" not in text:
        violations.append("resolution_translated_as_decision")
    if re.search(r"\d+\s*decisions\b", text, re.I):
        violations.append("english_decisions_for_resolutions")
    if re.search(r"MSC\s*111의\s*주요\s*결과는", text) and "자체의 최종 결과가 아니라" not in text:
        violations.append("missing_scope_correction")
    return violations
