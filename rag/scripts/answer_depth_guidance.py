"""Shared Accurate-mode answer depth guidance (format unchanged, report-style bullets)."""

import re

CITATION_GUIDANCE = """
## Citation (필수 — §1·§2 모든 bullet)
- context 청크는 **[1], [2], …** 로 번호가 매겨져 있다 (헤더 `[N] source=…` 참조).
- **§1·§2의 모든 bullet** 문장 끝에 해당 근거 번호를 **반드시** `[N]` 또는 `[2][3]` 형식으로 붙인다.
- citation **없는 bullet은 작성하지 말 것** (근거 없는 문장 금지).
- `[근거]`, `[근거 MEPC…]` placeholder **금지** — 숫자만 사용.
- 존재하지 않는 번호 인용 금지 (context에 없는 [99] 등).
- §3 추후 확인 항목은 context에 없을 때 citation 생략 가능.
"""

EVIDENCE_DISPLAY_GUIDANCE = """
## 근거 표시 (§1·§2 필수)
각 bullet에 가능한 범위에서 **회의차수·문서명·조항/결의·가이드 명칭**을 명시한다.
- 회의: MEPC 84, MSC 111, ISWG-GHG 20차 등
- 문서: MEPC 84/7/14, DNV-CG-0264, LR Notice No.1 등
- 조항·결의·가이드: MARPOL Annex VI reg.36·37, Section 15, MASS Code, SEEMP Part III 등
"""

CATEGORY_BULLET_DEFAULTS: dict[str, tuple[int, int, int]] = {
    "trend_summary": (4, 6, 3),
    "env_regulation": (3, 5, 3),
    "autonomous": (3, 5, 3),
    # Exact Rule facts are compacted separately.  Document-guide questions
    # follow the revised business feedback: one card per relevant document,
    # with no arbitrary 2–3 bullet ceiling.
    "rule_lookup": (2, 6, 2),
}

# The 1.2 answer-length contract is for the whole answer, not for every
# section independently.  Section 1 still uses ``CATEGORY_BULLET_DEFAULTS`` as
# its drafting target; these ceilings prevent the four-section renderer from
# multiplying a short Rule lookup into an eight-bullet report.
CATEGORY_TOTAL_BULLET_LIMITS: dict[str, tuple[int, int]] = {
    "trend_summary": (7, 10),
    "meeting_outcome": (7, 10),
    "env_regulation": (5, 7),
    "autonomous": (5, 7),
    # Ordinary Rule facts stay within the agreed 2-3 bullet UI contract.
    # Document-card questions explicitly lift this ceiling below so multiple
    # relevant instruments are not discarded.
    "rule_lookup": (2, 3),
}

_PLACEHOLDER_BULLET_MARKERS = (
    "검색 근거에서 질문에 직접 답할 내용을 확인하지 못했습니다",
    "검색 근거에서 직접 확인되는 별도 운항·업무 영향이 없습니다",
    "추가 확인 필요사항이 별도로 식별되지 않았습니다",
    "관련 선급 Rule / Guidance가 검색 근거에 없거나 해당하지 않습니다",
)


def _fact_line_key(line: str) -> str:
    prose = re.sub(r"\[\d+\]", "", str(line or ""))
    prose = re.sub(r"[*_`#>]", "", prose).lower()
    return re.sub(r"[^0-9a-z가-힣]+", "", prose)


def _fact_line_tokens(line: str) -> set[str]:
    prose = re.sub(r"\[\d+\]", "", str(line or "")).lower()
    stop = {
        "해당", "문서", "프로그램", "기반", "대한", "위한", "통해",
        "합니다", "됩니다", "적용", "형식승인", "요구사항",
    }
    tokens: set[str] = set()
    korean_suffixes = (
        "입니다", "합니다", "됩니다", "인가요", "한가요", "이어야", "하여야",
        "으로서", "로서", "에서는", "에게서", "이며", "하며", "에서", "으로",
        "에는", "에게", "까지", "부터", "은", "는", "이", "가", "을", "를", "의",
    )
    for token in re.findall(r"\d+(?:[./]\d+)*|[a-z]{2,}|[가-힣]{2,}", prose):
        if re.fullmatch(r"[가-힣]{2,}", token):
            for suffix in korean_suffixes:
                if token.endswith(suffix) and len(token) - len(suffix) >= 2:
                    token = token[: -len(suffix)]
                    break
        if token not in stop:
            tokens.add(token)
    return tokens


def _fact_line_numbers(line: str) -> set[str]:
    prose = re.sub(r"\[\d+\]", "", str(line or ""))
    return set(re.findall(r"\d+(?:[./]\d+)*", prose))


def _strip_redundant_english_parentheticals(line: str) -> str:
    """Drop long English glosses when the same fact is already stated in Korean.

    Short identifiers and units such as ``TA``, ``SWL`` and ``(300 V)`` remain.
    This is limited to the compact fact profile so analytical answers keep
    official English titles when those titles are useful evidence labels.
    """

    source = str(line or "")
    out: list[str] = []
    index = 0
    while index < len(source):
        if source[index] != "(":
            out.append(source[index])
            index += 1
            continue
        depth = 1
        end = index + 1
        while end < len(source) and depth:
            if source[end] == "(":
                depth += 1
            elif source[end] == ")":
                depth -= 1
            end += 1
        if depth:
            out.append(source[index:])
            break
        segment = source[index:end]
        english_words = re.findall(r"[A-Za-z]{2,}", segment)
        if len(english_words) < 4:
            out.append(segment)
        index = end
    cleaned = "".join(out)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return cleaned.strip()


def _question_relevance_score(line: str, question: str) -> float:
    """Prefer the direct condition/value bullet when a small model over-drafts."""
    q_tokens = _fact_line_tokens(question)
    line_tokens = _fact_line_tokens(line)
    overlap = q_tokens.intersection(line_tokens)
    score = float(sum(min(len(token), 10) for token in overlap))
    for value in _fact_line_numbers(question):
        if value in _fact_line_numbers(line):
            score += 4.0
    # Parenthesized English terms are often the most discriminative regulatory
    # anchor (for example acknowledgement, transfer, WtT or SWL).
    q_low = question.lower()
    line_low = line.lower()
    for token in re.findall(r"[a-z][a-z0-9_-]{2,}", q_low):
        if token in line_low:
            score += 3.0
    return score


def _select_exact_fact_lines(
    lines: list[str], question: str, slots: int
) -> list[str]:
    if len(lines) <= slots:
        return lines
    # The model normally puts direct values and enumerated facts first.  A
    # relevance re-rank is reserved for yes/no questions, where a generic
    # first bullet can otherwise displace a later explicit exception (for
    # example "acknowledgment does not apply").
    if not re.search(
        r"필요(?:한가|합니까|한가요)|여부|해당(?:하는가|합니까)|"
        r"맞(?:는가|습니까)|아닌가|acknowledg",
        question,
        re.I,
    ):
        return lines[:slots]
    scored = [
        (_question_relevance_score(line, question), index, line)
        for index, line in enumerate(lines)
    ]
    chosen = sorted(
        sorted(scored, key=lambda item: (-item[0], item[1]))[:slots],
        key=lambda item: item[1],
    )
    return [line for _score, _index, line in chosen]


def _dedupe_subsumed_lines(lines: list[str]) -> tuple[list[str], int]:
    """Remove a later recap bullet when it adds no distinct fact."""
    kept: list[str] = []
    kept_terms: list[set[str]] = []
    kept_number_sets: list[set[str]] = []
    known_terms: set[str] = set()
    known_numbers: set[str] = set()
    removed = 0
    for line in lines:
        terms = _fact_line_tokens(line)
        numbers = _fact_line_numbers(line)
        if len(kept) == 1 and kept_terms[0]:
            prior_coverage = len(kept_terms[0].intersection(terms)) / len(kept_terms[0])
            if (
                prior_coverage >= 0.78
                and kept_number_sets[0] <= numbers
                and len(terms) > len(kept_terms[0])
            ):
                kept[0] = line
                kept_terms[0] = terms
                kept_number_sets[0] = numbers
                known_terms = set(terms)
                known_numbers = set(numbers)
                removed += 1
                continue
        coverage = len(terms.intersection(known_terms)) / len(terms) if terms else 0.0
        if len(kept) >= 1 and len(terms) >= 5 and coverage >= 0.78 and numbers <= known_numbers:
            removed += 1
            continue
        kept.append(line)
        kept_terms.append(terms)
        kept_number_sets.append(numbers)
        known_terms.update(terms)
        known_numbers.update(numbers)
    return kept, removed


def _dedupe_exact_fact_lines(lines: list[str]) -> tuple[list[str], int]:
    """Remove same-value paraphrases from a compact factual lookup."""
    kept: list[str] = []
    removed = 0
    for line in lines:
        key = _fact_line_key(line)
        tokens = _fact_line_tokens(line)
        numbers = _fact_line_numbers(line)
        duplicate = False
        for prior in kept:
            prior_key = _fact_line_key(prior)
            prior_tokens = _fact_line_tokens(prior)
            prior_numbers = _fact_line_numbers(prior)
            containment = (
                min(len(key), len(prior_key)) >= 16
                and (key in prior_key or prior_key in key)
            )
            union = tokens | prior_tokens
            similarity = len(tokens & prior_tokens) / len(union) if union else 0.0
            same_value = bool(numbers and numbers == prior_numbers)
            semantic_duplicate = (
                len(tokens.intersection(prior_tokens)) >= 4
                and similarity >= 0.52
            )
            if containment or semantic_duplicate or (same_value and similarity >= 0.42):
                duplicate = True
                break
        if duplicate:
            removed += 1
        else:
            kept.append(line)
    return kept, removed


def category_bullet_budget(category: str, row: dict | None = None) -> tuple[int, int, int]:
    """Section 1 bullet (min, max, priority_top_n). Eval row may override."""
    row = row or {}
    dmin, dmax, dpriority = CATEGORY_BULLET_DEFAULTS.get(category, (5, 7, 3))
    bmin = int(row.get("answer_bullets_min") or dmin)
    bmax = int(row.get("answer_bullets_max") or dmax)
    priority = int(row.get("summary_priority_bullets") or dpriority)
    return bmin, bmax, priority


def category_total_bullet_budget(
    category: str, row: dict | None = None
) -> tuple[int, int]:
    """Return the agreed whole-answer bullet range for a UI category."""
    row = row or {}
    from compound_regulatory import is_compound_regulatory_class_question

    if is_compound_regulatory_class_question(str(row.get("question") or "")):
        # A two-lane design-review answer needs four checklist bullets plus one
        # decision, one uncertainty and one class instrument.  The ordinary
        # 5–7 environment cap deletes required evidence after validation.
        low, high = (7, 9)
    else:
        low, high = CATEGORY_TOTAL_BULLET_LIMITS.get(category, (2, 7))
    low = int(row.get("answer_total_bullets_min") or low)
    high = int(row.get("answer_total_bullets_max") or high)
    return low, max(low, high)


def apply_category_total_bullet_limit(
    answer: str,
    category: str,
    row: dict | None = None,
) -> tuple[str, dict]:
    """Apply the whole-answer length contract without inventing or merging facts.

    The function only removes lower-priority bullets and turns legacy
    hyphen-prefixed empty-section notices into blockquotes.  Citations and
    factual wording are otherwise unchanged.  Four section headings remain in
    place so every RAG route presents the same UI shape.
    """
    row = row or {}
    low, high = category_total_bullet_budget(category, row)
    compact_rule_fact = bool(
        row.get("_answer_profile") == "exact_rule_fact"
    )
    document_cards = str(
        (row.get("_question_profile") or {}).get("answer_style") or ""
    ) == "document_cards"
    headings = tuple(SECTION_TITLES.values())
    bodies: dict[str, list[str]] = {heading: [] for heading in headings}
    current = headings[0]
    for raw in str(answer or "").splitlines():
        stripped = raw.strip()
        matched = next(
            (
                heading
                for heading in headings
                if re.match(
                    rf"^#+\s*{re.escape(heading.removeprefix('## ').strip())}\s*$",
                    stripped,
                    re.I,
                )
            ),
            None,
        )
        if matched:
            current = matched
            continue
        if stripped:
            bodies[current].append(stripped)

    factual: dict[str, list[str]] = {heading: [] for heading in headings}
    nonfacts: dict[str, list[str]] = {heading: [] for heading in headings}
    for heading in headings:
        for line in bodies[heading]:
            if line.startswith("- ") and any(
                marker in line for marker in _PLACEHOLDER_BULLET_MARKERS
            ):
                nonfacts[heading].append("> " + line[2:].strip())
            elif line.startswith("- "):
                factual[heading].append(line)
            else:
                nonfacts[heading].append(line)

    recap_duplicates_removed = 0
    if compact_rule_fact:
        factual[headings[0]] = [
            _strip_redundant_english_parentheticals(line)
            for line in factual[headings[0]]
        ]
    for heading in headings:
        if document_cards and heading == headings[0]:
            removed = 0
        else:
            factual[heading], removed = _dedupe_subsumed_lines(factual[heading])
        recap_duplicates_removed += removed

    before = sum(len(lines) for lines in factual.values())
    duplicate_facts_removed = 0
    if compact_rule_fact:
        factual[headings[0]], duplicate_facts_removed = _dedupe_exact_fact_lines(
            factual[headings[0]]
        )
        fact_slots = max(1, min(3, int(row.get("_answer_fact_slots") or 1)))
        factual[headings[0]] = _select_exact_fact_lines(
            factual[headings[0]], str(row.get("question") or ""), fact_slots
        )
        factual[headings[1]] = []
        factual[headings[2]] = []
        factual[headings[3]] = factual[headings[3]][:1]
        before = sum(len(lines) for lines in factual.values()) + duplicate_facts_removed
    if document_cards:
        # The revised Rule-guide contract is document-count driven.  Preserve
        # every already-grounded card/usage/caveat bullet produced upstream;
        # exact clause/value questions never enter this branch.
        high = max(high, before)
    if (
        category == "rule_lookup"
        and not document_cards
        and not factual[headings[3]]
        and factual[headings[0]]
    ):
        # A verified Rule can be described in section 1 even when an upstream
        # builder failed to populate section 4.  Re-express that same cited
        # document identity as the requested catalog pointer; the total-budget
        # pass below drops lower-priority follow-up prose so the answer remains
        # within three bullets.
        source_line = factual[headings[0]][0][2:].strip()
        factual[headings[3]] = [
            f"- **관련 선급 Rule / Guidance**: {source_line}"
        ]
        before += 1
    if before > high:
        if category == "rule_lookup":
            # A lookup should identify the best two documents/requirements and
            # retain one distinct Rule/Guidance or limitation item.
            keep = {
                headings[0]: factual[headings[0]][:2],
                headings[1]: [],
                headings[2]: [],
                headings[3]: factual[headings[3]][:1],
            }
            if sum(len(lines) for lines in keep.values()) < high:
                for section in (headings[2], headings[1]):
                    if factual[section]:
                        keep[section] = factual[section][:1]
                        if sum(len(lines) for lines in keep.values()) >= high:
                            break
        else:
            # Categories 1–3: the top three core facts come first, followed by
            # at most two work impacts, one caveat and one class-rule pointer.
            from compound_regulatory import is_compound_regulatory_class_question

            compound_regulatory_class = is_compound_regulatory_class_question(
                str(row.get("question") or "")
            )
            section_caps = (
                (2, 4, 1, 2)
                if compound_regulatory_class
                else (5, 2, 2, 1)
                if category in {"trend_summary", "meeting_outcome"}
                else (3, 2, 1, 1)
            )
            keep = {
                heading: factual[heading][:cap]
                for heading, cap in zip(headings, section_caps)
            }
            while sum(len(lines) for lines in keep.values()) > high:
                for heading in (headings[2], headings[1], headings[3], headings[0]):
                    minimum = 3 if heading == headings[0] else 0
                    if len(keep[heading]) > minimum:
                        keep[heading].pop()
                        break
                else:
                    break
        factual = keep

    rendered: list[str] = []
    rendered_headings = (
        (headings[0], headings[3]) if compact_rule_fact else headings
    )
    for heading in rendered_headings:
        rendered.extend([heading, ""])
        lines = factual.get(heading, [])
        if lines:
            rendered.extend(lines)
        else:
            notes = nonfacts.get(heading, [])
            if notes:
                rendered.extend(notes)
            else:
                defaults = {
                    headings[0]: "> 검색 근거에서 질문에 직접 답할 내용을 확인하지 못했습니다.",
                    headings[1]: "> 검색 근거에서 직접 확인되는 별도 운항·업무 영향이 없습니다.",
                    headings[2]: "> 추가 확인 필요사항이 별도로 식별되지 않았습니다.",
                    headings[3]: "> 관련 선급 Rule / Guidance가 검색 근거에 없거나 해당하지 않습니다.",
                }
                rendered.append(defaults[heading])
        rendered.append("")
    after = sum(len(lines) for lines in factual.values())
    return "\n".join(rendered).strip(), {
        "category": category,
        "expected_min": low,
        "max": high,
        "before": before,
        "after": after,
        "trimmed": max(0, before - after),
        "below_target": after < low,
        "answer_profile": "exact_rule_fact" if compact_rule_fact else "standard",
        "duplicate_facts_removed": duplicate_facts_removed,
        "recap_duplicates_removed": recap_duplicates_removed,
    }


SECTION_TITLES = {
    "1": "## 1) 핵심 요약",
    "2": "## 2) 선박 운항/업무 영향",
    "3": "## 3) 추후 확인 필요사항",
    "4": "## 4) 관련 선급 Rule / Guidance",
}


def join_four_sections(parts: dict[str, str]) -> str:
    out: list[str] = []
    for key in ("1", "2", "3", "4"):
        body = (parts.get(key) or "").strip()
        if not body:
            continue
        out.append(SECTION_TITLES[key])
        out.append(body)
    return "\n\n".join(out).strip()

ANSWER_DENSITY_GUIDANCE = """
## bullet 작성 원칙
각 bullet은 질문에 직접 답하는 **하나의 사실 문장**으로 작성하고 문장 끝에 citation [N]을 붙인다.
원문에 명시된 논의·수치·조치 요청·규정 문구만 사용한다.
"따라서 선사/기술부는 ~해야 한다"와 같은 실무 조언은 사용자가 명시적으로 요청하고,
인용 문장 자체에 shall/must/required 등 의무 표현이 있을 때만 작성한다.

**금지 패턴:**
- `A ↔ B`, `A, B, C`만 나열하고 **문장을 완성하지 않는 bullet**
- bullet이 **연결 기호(↔, /, ·)** 또는 **쉼표 구분 키워드**로 끝나는 것
- "논의되었습니다", "정리되었습니다", "필요합니다", "필요함"으로 **끝내는 것** (영향 문장 없이)
- 영문 고유명사·약어만 나열 (GFI, SEEMP, CII 등은 **문장 안에서** 풀어 쓸 것)

**전문 용어는 유지**하되 보고서 문장으로 다듬는다:
GFI, MARPOL Annex VI reg.36·37, SEEMP Part III, CII fleet carbon intensity,
well-to-tank emission factor, LCA/sustainability themes 등

**bullet 템플릿:**
`- **[주제]**: [문서에 직접 명시된 핵심 사실 한 문장]. [N]`
"""

SECTION2_OPERATIONAL_GUIDANCE = """
## 2) 선박 운항/업무 영향
- 사용자가 실무 영향·선사 대응을 명시적으로 물었을 때만 작성한다.
- 인용 원문에 직접 명시된 의무·보고·검증·적용 요건만 한 문장씩 작성한다.
- 원문의 논의나 분석 결과를 선사·기술부의 의무로 바꾸어 추론하지 않는다.
"""

SECTION2_RULE_LOOKUP_GUIDANCE = """
## 2) 선박 운항/업무 영향
- bullet **1~2개** (§1 문서별 요약·동일 결론 문장 반복 금지)
- 검색된 Rule/Guidance 전체에 대한 **통합 실무 조치**만 (notation 검토, 적용 범위·class 승인 절차 등)
- citation [N]
"""

SECTION3_FOLLOWUP_GUIDANCE = """
## 3) 추후 확인 필요사항
- bullet **2~4개**, §1·§2에 없는 항목만
- 각 bullet은 아래 태그 **하나로 시작**:
  - `- [미확정 규제] …`
  - `- [해석 논란] …` 또는 `- [해석 근거] …`
  - `- [선급별 상이 요구] …`
- 무엇을·왜·어느 문서에서 확인할지 **완전한 문장**으로
- context 없음: `- [해석 근거] 검색 결과 내 확인 불가 — (이유)`
"""

SECTION4_GUIDANCE = """
## 4) 관련 선급 Rule / Guidance
- context에 DNV/LR/KR/ABS Rule·Guidance가 있으면 문서명·scope bullet (§1과 **완전 중복 나열은 피함**)
- catalog_table·cross-ref 표에만 있는 문서는 **후보**로 표시
- IMO 회의 자료만 검색된 경우: `- 본 검색은 IMO 회의 자료 중심. 선급 Rule/Guidance는 별도 검색 필요.`
- 해당 없으면: `- 해당 없음`
"""

FORMAT_RULES = """
**출력 형식:**
1) `## 1) 핵심 요약`은 필수다.
2) `## 2) 선박 운항/업무 영향`은 사용자가 이를 명시적으로 요청한 경우만 작성한다.
3) `## 3) 추후 확인 필요사항`은 문서 상태·미확정 사항이 실제로 있을 때만 작성한다.
4) `## 4) 관련 선급 Rule / Guidance`는 Rule/Guidance 질문일 때만 작성한다.

- 섹션 제목은 `## N) …` (`###` 금지)
- 모든 bullet = `- ` 한 줄 (sub-bullet·들여쓰기 중첩 금지)
- §1에 §2(실무 영향) 내용 넣지 말 것
- §1·§2의 사실 문장은 한 문장씩 분리하고 각 문장 끝에 citation [N]을 붙인다.
"""

ANTI_REPETITION_GUIDANCE = """
## 반복 금지 (최우선 — 모든 섹션·카테고리)
- **동일·유사 문장을 2회 이상 쓰지 말 것** (§1 bullet 간, §1↔§2, 문서별 bullet 간).
- 문서별 bullet마다 **고유한 사실**(문서번호, scope, notation, 적용대상, 핵심 요건)만 다르게 쓸 것.
- **고정 결론 문구를 모든 bullet에 붙이지 말 것** — 예: "이는 Smart/autonomous… class compliance 범위를 명시", "따라서 설계·운항 부서는 fleet별로 검토해야 한다"를 CG/RP마다 복사 금지.
- "이는 ~와 연결된다", "따라서 ~해야 한다" 문형을 **bullet마다 같은 말로** 반복하지 말고, context에 맞는 **다른 표현** 또는 **한 번만** 통합 서술.
- §2는 §1 문장·문서 나열을 **다시 쓰지 말 것** — 실무 조치를 **1~2 bullet로 통합** (rule_lookup은 특히 엄격).
- 유사한 선급 RP/CG가 여러 개면: 문서별 bullet **또는** 주제 통합 bullet 중 하나만 — **둘 다 같은 영향 문장으로 끝내지 말 것**.
"""

RULE_LOOKUP_GUIDANCE = """
## Rule/Guidance 조회 (카테고리 rule_lookup)
- 문서 안내형 질문은 관련 file_name별로 한 카드씩 작성하며, 문서 수에 따라 bullet 수를 유연하게 정한다.
- 각 문서 카드는 문서 번호·제목, 문서 성격, scope·적용대상, 활용 시점, 핵심 요건을 서로 다른 내용으로 정리한다.
- **문서명·번호는 user prompt 「인용 허용 문서」·「Citation 매핑」의 file_name만** 사용.
- citation [N]과 bullet 문서명이 **일치**해야 함.
- placeholder `(context의 …)` 출력 금지.
- §2는 **1~2 bullet** 통합 실무 조치만.
- §4는 §1에 없는 추가 선급 Rule·catalog 후보를 정리 (있을 때만).
"""

RULE_LOOKUP_OUTPUT_SCOPE = ""

RULE_LOOKUP_EVIDENCE_GUIDANCE = """
## Rule/Guidance 근거 (최우선)
- 답변의 모든 문서명은 context 헤더 `doc=파일명` 또는 user prompt 인용 목록과 **완전히 일치**해야 한다.
- context에 없는 DNV-RP-*, DNV-RU-SHIP Pt.* 등은 **작성 금지** (코퍼스 미수록 가능).
- cross-reference 표( Document code / Title 목록만 있는 청크)는 주제 설명 근거로 쓰지 말고, **해당 표에 등장하는 문서가 별도 [N] 본문 청크로 있을 때만** 언급.
"""

ENV_REGULATION_V01_HINT = """
[V01 — §1 최소 7개 bullet, context에 있을 때 주제별 1 bullet]

[나쁨 — 키워드 나열·↔ 종결]
- "GFI compliance/reporting/verification ↔ MARPOL Annex VI reg.36·37"
- "SEEMP Guidelines 개정, Fifth IMO GHG Study, LCA/sustainability themes"
- "MEPC 84에서 CII 관련 검토가 논의되었습니다."

[좋음 — 보고서형 2~3문장]
- "**GFI·MARPOL Annex VI reg.36·37**: ISWG-GHG 20은 GFI compliance·reporting·verification을 MARPOL Annex VI regulation 36·37과 정합되게 다루었다. 이는 선박 연료·에너지 사용에 대한 국제 보고·검증 의무 체계를 확장하는 쪽이다. 따라서 선사는 GFI 산정·제출 데이터와 기존 DCS/연료 보고 필드 간 매핑을 점검해야 한다 [1][8]"
- "**CII fleet carbon intensity**: MEPC 84-6-2는 2024 fleet CII 결과 제출을 보고했다. 이는 fleet carbon intensity 추세를 공개하는 자료이다. 따라서 운항부는 SEEMP Part III 갱신 주기와 연료·속도 데이터 수집 체계를 재점검해야 한다 [4]"
- "**GESAMP-LCA WG**: 2차 회의는 well-to-tank emission factor의 representativeness 기준을 정리했다. 이는 대체연료 GFI·LCA/sustainability themes 산정 시 데이터 품질 기준을 구체화한다. 그 결과 연료 공급망 LCA 증빙·내부 사전 수집 범위를 확대해야 할 수 있다 [13]"
"""

GOOD_BAD_EXAMPLES = """
[Citation 예시 — 모든 문항 공통]
- 나쁨: "DNV-CG-0264는 자율운항 및 원격운항 선박에 대한 guidance를 제공한다." (citation 없음)
- 좋음: "DNV-CG-0264는 autonomous·remotely operated vessels의 scope와 notation(AUTO, REMO 등) 요건을 정의한다 [2]"

[반복 금지 — Rule/Guidance 나열]
- 나쁨: 검색되지 않은 RP-C205·RU-SHIP 이름을 붙이거나, 모든 bullet 끝에 동일한 "fleet별 검토" 문장 반복
- 나쁨: "(context의 고유 주제)" 같은 placeholder를 그대로 출력
- 좋음: **context에 있는 file_name만** — "DNV-CG-0264.pdf: autoremote·remote link 요건 [2]", "DNV-RU-OU-0103.pdf: Smart notation 별도 적용 [6]" — 공통 실무 조치는 §2에 **1회만**

[문장 품질 — GFI 예시]
- 나쁨: "GFI compliance/reporting/verification ↔ MARPOL Annex VI reg.36·37"
- 좋음: "ISWG-GHG 20은 GFI 준수·보고·검증(verification)을 MARPOL Annex VI regulation 36·37과 연동해 정비하고 있다. 이는 선박 연료 GHG 배출에 대한 국제 보고·검증 프레임과 직결된다. 따라서 선사는 GFI 관련 데이터 필드와 내부 연료·배출 ledger를 대조해야 한다 [1]"

[문장 품질 — SEEMP/CII 예시]
- 나쁨: "SEEMP Part III 갱신 주기 재점검 필요"
- 좋음: "2024 fleet CII 결과 보고는 fleet carbon intensity 추세를 공개한다. 이는 SEEMP Part III 갱신·연료 효율 조치와 연동된다. 따라서 운항·기술 부서는 SEEMP Part III 갱신 일정과 CII rating 목표를 함께 재설정해야 한다 [4]"
"""
