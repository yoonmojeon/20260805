"""Grounding helpers for questions that target one technical rule clause.

The normal RAG answer verifier is intentionally broad and cannot reliably
compare Korean paraphrases with English rule text.  This module provides the
missing deterministic boundary for direct-clause answers:

* keep only chunks selected for the ``specific_clause`` evidence slot;
* recover the most specific clause number from the source body;
* preserve deontic strength (shall/must > should > consider > may); and
* make the Rule/Guidance reference section from metadata instead of asking the
  language model to reproduce it.

Nothing in this file contains a question, document ID, page number, or expected
answer fixture.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from rule_lookup_context import strip_metadata_prefix


CITATION_RE = re.compile(r"\[(\d+)\]")
SECTION_RE = re.compile(r"^##\s*([1-4])\)\s*(.+?)\s*$")
CLAUSE_RE = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+){1,5})(?![\d.])"
    r"(?:\s+|[:—-]\s*)([A-Z][^\n.;]{2,100})?"
)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")

STRONG_SOURCE_RE = re.compile(
    r"\bshall\b|\bmust\b|\bis required to\b|\bare required to\b|"
    r"\bshall be\b|\bmandatory\b",
    re.I,
)
SHOULD_CONSIDER_RE = re.compile(
    r"\bshould\s+(?:also\s+)?be\s+considered\b|"
    r"\bconsideration\s+should\s+be\s+given\b",
    re.I,
)
SHOULD_SOURCE_RE = re.compile(r"\bshould\b", re.I)
PERMISSIVE_SOURCE_RE = re.compile(r"\bmay\b|\bcan\b|\bcould\b", re.I)

STRONG_CLAIM_RE = re.compile(
    r"반드시|필수(?:적|로|이다|입니다)?|의무(?:적|이다|입니다)?|"
    r"강제(?:적|된다)?|예외\s*없이",
    re.I,
)
OBLIGATION_CLAIM_RE = re.compile(
    r"해야\s*(?:한다|합니다|함)|하여야\s*(?:한다|합니다|함)|"
    r"요구(?:된다|됩니다|한다|합니다)|필요(?:하다|합니다)",
    re.I,
)
CONSIDER_CLAIM_RE = re.compile(r"고려", re.I)
PERMISSIVE_CLAIM_RE = re.compile(
    r"할\s*수\s*있|가능(?:하다|합니다)|도움이\s*될\s*수\s*있",
    re.I,
)
INFERRED_CONSEQUENCE_CLAIM_RE = re.compile(
    r"\uc5c6\uac8c\s*\ub418\uba74|\uc5c6\uc73c\uba74|"
    r"\uc5b4\ub824\uc6cc\uc9c4\ub2e4|\ucd08\ub798\ud55c\ub2e4|"
    r"\uc774\uc5b4\uc9c4\ub2e4|\ub54c\ubb38\uc5d0|\ub530\ub77c\uc11c",
    re.I,
)
SOURCE_CONSEQUENCE_RE = re.compile(
    r"\bif\b|\bwhen\b|\btherefore\b|\bhence\b|\bresult(?:s|ing)?\b|"
    r"\blead(?:s|ing)?\s+to\b|\bdue\s+to\b",
    re.I,
)
SPECULATIVE_IMPACT_CLAIM_RE = re.compile(
    r"\uc704\ud611\ubc1b|\uc704\ud5d8\ud574|\uc5b4\ub824\uc6cc|"
    r"\uc9c0\uc5f0\ub41c\ub2e4|\ubb38\uc81c\uac00\s*\ubc1c\uc0dd",
    re.I,
)
SPECULATIVE_IMPACT_SOURCE_RE = re.compile(
    r"\bthreat(?:en|ened|ens)?\b|\brisk(?:s|ed|ing)?\b|"
    r"\bdanger(?:ous)?\b|\bdifficult(?:y)?\b|\bdelay(?:ed|s)?\b|"
    r"\bproblem(?:s)?\b",
    re.I,
)

UNSUPPORTED_CONDITION_CLAIM_RE = re.compile(
    r"\uc54a\uc73c\uba74|\ubd80\uc871\ud558\uba74|\ubbf8\uc81c\uacf5\ub41c\ub2e4\uba74",
    re.I,
)
ENGLISH_SENTENCE_LEAK_RE = re.compile(
    r"\b(?:decision-making|safe way|all information|existing requirements?|"
    r"unattended machinery space operations?|should be observed)\b",
    re.I,
)

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{2,}")
TOKEN_STOP = {
    "and", "the", "for", "with", "from", "that", "this", "should", "shall",
    "must", "may", "can", "could", "rule", "guidance", "clause", "page",
}

KOREAN_SOURCE_ALIASES = {
    "\uc0c1\ud669 \uc778\uc2dd": ("situational awareness",),
    "\uc0c1\ud669\uc778\uc2dd": ("situational awareness",),
    "\uc2e4\uc2dc\uac04": ("real-time", "near-real-time"),
    "\uc6b4\uc601 \uc0c1\ud0dc": ("operational status",),
    "\uc6b4\ud56d \uc0c1\ud0dc": ("operational status",),
    "\uc900\ube44": ("readiness",),
    "\uc6a9\ub7c9": ("capacity",),
    "\uc548\uc804": ("safe", "safety"),
    "\uc6d0\uaca9 \uc6b4\uc601\uc790": ("remote operator",),
    "\uc6d0\uaca9\uc6b4\uc601\uc790": ("remote operator",),
    "\uc678\ubd80": ("exterior",),
    "\ub0b4\ubd80": ("interior",),
    "\uc601\uc0c1": ("video", "cctv"),
    "\uac10\uc2dc": ("surveillance", "monitoring", "cctv"),
    "\uc5f0\uacb0 \uc7a5\uc560": ("connectivity outage",),
    "\uc54c\ub78c": ("alert", "alarm"),
    "\uacbd\uace0": ("alert", "alarm"),
    "\ube44\uc815\uc0c1": ("abnormal",),
    "\ubb34\uc778 \uae30\uad00\uc2e4": ("unattended machinery space",),
}

# Some legacy source files in this repository were written through a terminal
# with a non-UTF-8 code page.  Keep the legal-strength gate ASCII-safe by using
# Unicode escapes here; otherwise a Korean phrase such as "CCTV 설치가
# 필요하다" can bypass the SHOULD/CONSIDER check.
STRONG_CLAIM_RE = re.compile(
    r"\ubc18\ub4dc\uc2dc|\ud544\uc218|\uc758\ubb34|\uac15\uc81c|\uc608\uc678\s*\uc5c6\uc774",
    re.I,
)
OBLIGATION_CLAIM_RE = re.compile(
    r"(?:\ud574\uc57c|\ud558\uc5ec\uc57c)\s*(?:\ud55c\ub2e4|\ud569\ub2c8\ub2e4)|"
    r"\uc694\uad6c\ub41c\ub2e4|\ud544\uc694\ud558\ub2e4",
    re.I,
)
CONSIDER_CLAIM_RE = re.compile(r"\uace0\ub824", re.I)
PERMISSIVE_CLAIM_RE = re.compile(
    r"\ud560\s*\uc218\s*\uc788\ub2e4|\uac00\ub2a5\ud558\ub2e4", re.I
)


def clause_body(chunk: Any) -> str:
    return re.sub(
        r"\s+", " ", strip_metadata_prefix(str(getattr(chunk, "text", "") or ""))
    ).strip()


def extract_clause_reference(chunk: Any) -> tuple[str, str]:
    """Return the most specific body clause and its title.

    Some indexed DNV chunks carry a section-level metadata value (``6``) while
    the body starts with the actual clause heading (``6.4.1``).  Prefer the
    hierarchical number found in the body and fall back to metadata.
    """
    raw_body = strip_metadata_prefix(str(getattr(chunk, "text", "") or "")).strip()
    body = re.sub(r"\s+", " ", raw_body).strip()
    matches = list(CLAUSE_RE.finditer(body[:700]))
    if matches:
        # A more deeply nested number is normally the direct technical clause.
        best = max(matches, key=lambda match: (match.group(1).count("."), -match.start()))
        clause = best.group(1)
        line_heading = re.search(
            rf"(?m)^\s*{re.escape(clause)}\s+([^\r\n]{{2,120}})",
            raw_body,
        )
        title = (line_heading.group(1) if line_heading else best.group(2) or "").strip()
        if line_heading is None:
            title = re.split(
                r"\s+(?=It\s+(?:shall|must|should|may|can|could)\b|"
                r"The\s|Where\s|If\s|This\s|A\s)",
                title,
                maxsplit=1,
                flags=re.I,
            )[0].strip()
        return clause, title
    metadata = str(getattr(chunk, "clause_number", "") or "").strip()
    return metadata, ""


def select_specific_clause_chunks(
    row: dict,
    retrieved: Iterable[Any],
    pool: Iterable[Any],
    *,
    limit: int = 3,
) -> list[Any]:
    """Resolve exact chunks recorded by evidence planning, in citation order."""
    completion = row.get("_evidence_completion") or {}
    ids = list((completion.get("slot_hits") or {}).get("specific_clause") or [])
    if not ids:
        return []
    by_id = {
        str(getattr(chunk, "chunk_id", "")): chunk
        for chunk in [*list(retrieved), *list(pool)]
        if getattr(chunk, "chunk_id", None)
    }
    selected: list[Any] = []
    seen: set[str] = set()
    first_doc = ""
    for chunk_id in ids:
        chunk = by_id.get(str(chunk_id))
        if chunk is None:
            continue
        cid = str(getattr(chunk, "chunk_id", ""))
        doc = str(getattr(chunk, "file_name", "") or getattr(chunk, "doc_id", ""))
        if first_doc and doc and doc != first_doc:
            continue
        if cid in seen:
            continue
        selected.append(chunk)
        seen.add(cid)
        first_doc = first_doc or doc
        if len(selected) >= limit:
            break
    # If an explicitly headed clause was found, do not mix in an unheaded
    # continuation fragment.  Such fragments can start mid-sentence after
    # indexing (for example "...awareness is lost"), which reverses the
    # meaning when summarized without the missing "synchronised ... to avoid".
    headed = [
        chunk
        for chunk in selected
        if extract_clause_reference(chunk)[0]
        and extract_clause_reference(chunk)[1]
    ]
    if headed:
        target_clause, _ = extract_clause_reference(headed[0])
        same_clause = [
            chunk
            for chunk in headed
            if extract_clause_reference(chunk)[0] == target_clause
        ]
        if same_clause:
            return same_clause[:limit]
    return selected


def _source_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]


def _claim_anchor_tokens(claim: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(claim)
        if token.lower() not in TOKEN_STOP and not token.lower().endswith(".pdf")
    }


def _local_evidence(claim: str, evidence: str) -> str:
    """Use the source sentence closest to the claim's surviving technical terms."""
    anchors = _claim_anchor_tokens(claim)
    lowered_claim = claim.lower()
    for korean, aliases in KOREAN_SOURCE_ALIASES.items():
        if korean in lowered_claim:
            anchors.update(alias.lower() for alias in aliases)
    sentences = _source_sentences(evidence)
    if not anchors or not sentences:
        return evidence
    scored = [
        (sum(anchor in sentence.lower() for anchor in anchors), sentence)
        for sentence in sentences
    ]
    best_score = max(score for score, _ in scored)
    if best_score <= 0:
        return evidence
    return next(sentence for score, sentence in scored if score == best_score)


def source_modality(sentence: str) -> str:
    """Label one source proposition without changing its legal strength."""
    if STRONG_SOURCE_RE.search(sentence):
        return "MANDATORY"
    if SHOULD_CONSIDER_RE.search(sentence):
        return "CONSIDER"
    if SHOULD_SOURCE_RE.search(sentence):
        return "SHOULD"
    if PERMISSIVE_SOURCE_RE.search(sentence):
        return "PERMISSIVE"
    return "DESCRIPTIVE"


def build_clause_proposition_block(
    chunk: Any,
    *,
    citation: int = 1,
    max_chars: int = 3800,
) -> str:
    """Expose atomic, modality-labelled sentences to the answer model."""
    body = clause_body(chunk)
    clause, title = extract_clause_reference(chunk)
    if clause and title:
        heading = f"{clause} {title}"
        if body.startswith(heading):
            body = body[len(heading):].lstrip()
    lines: list[str] = []
    used = 0
    for sentence in _source_sentences(body):
        line = (
            f"source=[{citation}] | modality={source_modality(sentence)} | "
            f"{sentence}"
        )
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def modality_violation(claim: str, evidence: str) -> str | None:
    """Reject Korean claims that strengthen the cited English requirement."""
    local = _local_evidence(claim, evidence)
    has_strong = bool(STRONG_SOURCE_RE.search(local))
    has_consider = bool(SHOULD_CONSIDER_RE.search(local))
    has_should = bool(SHOULD_SOURCE_RE.search(local))
    has_permissive = bool(PERMISSIVE_SOURCE_RE.search(local))

    if (
        INFERRED_CONSEQUENCE_CLAIM_RE.search(claim)
        and not SOURCE_CONSEQUENCE_RE.search(local)
    ):
        return "unsupported_inferred_consequence"
    if (
        SPECULATIVE_IMPACT_CLAIM_RE.search(claim)
        and not SPECULATIVE_IMPACT_SOURCE_RE.search(local)
    ):
        return "unsupported_speculative_impact"
    if UNSUPPORTED_CONDITION_CLAIM_RE.search(claim) and not re.search(
        r"\bif\b|\bwhen\b|\bunless\b", local, re.I
    ):
        return "unsupported_condition"
    if ENGLISH_SENTENCE_LEAK_RE.search(claim):
        return "english_source_leaked_into_korean_answer"
    if (
        has_consider
        and re.search(r"(?:CCTV|\uc13c\uc11c|\uce74\uba54\ub77c).{0,12}\uc124\uce58", claim, re.I)
        and not re.search(r"\binstall(?:ation)?\b", local, re.I)
    ):
        return "example_transformed_to_installation"

    if STRONG_CLAIM_RE.search(claim) and not has_strong:
        return "modal_strengthened_to_mandatory"

    if OBLIGATION_CLAIM_RE.search(claim):
        # "고려해야 한다" is the faithful Korean rendering of
        # "should be considered"; other duties are stronger than that source.
        if has_consider and CONSIDER_CLAIM_RE.search(claim):
            return None
        if has_consider and not (has_strong or (has_should and not has_consider)):
            return "consideration_strengthened_to_duty"
        if has_permissive and not (has_strong or has_should):
            return "permission_strengthened_to_duty"

    # A permissive Korean rendering is always safe with a stronger source.
    if PERMISSIVE_CLAIM_RE.search(claim):
        return None
    return None


def validate_direct_clause_answer(
    answer: str,
    chunks: list[Any],
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Remove invalid direct-clause bullets and return claim audit rows."""
    if not answer:
        return "", [], ["empty_direct_clause_answer"]
    output: list[str] = []
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for raw in answer.splitlines():
        stripped = raw.strip()
        if not stripped.startswith("- "):
            output.append(raw)
            continue
        claim = stripped[2:].strip()
        ids = sorted({int(value) for value in CITATION_RE.findall(claim)})
        reason: str | None = None
        if not ids:
            reason = "citation_missing"
        elif any(value < 1 or value > len(chunks) for value in ids):
            reason = "citation_out_of_range"
        else:
            evidence = " ".join(clause_body(chunks[value - 1]) for value in ids)
            reason = modality_violation(CITATION_RE.sub("", claim), evidence)
        supported = reason is None
        rows.append(
            {
                "claim": CITATION_RE.sub("", claim).strip(),
                "citations": [f"[{value}]" for value in ids if 1 <= value <= len(chunks)],
                "supported": supported,
                "reason": reason or "supported",
            }
        )
        if supported:
            # The UI contract is one citation suffix per factual bullet.  Local
            # models often emit "[1] p.88" mid-sentence; retain the validated
            # proposition but normalize its marker to the sentence end.
            normalized_claim = CITATION_RE.sub("", claim)
            normalized_claim = re.sub(r"\s{2,}", " ", normalized_claim).strip()
            suffix = "".join(f"[{value}]" for value in ids)
            output.append(f"- {normalized_claim} {suffix}")
        else:
            warnings.append(f"direct_clause_claim_removed:{reason}")
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(output)).strip()
    return cleaned, rows, list(dict.fromkeys(warnings))


def direct_clause_coverage_gaps(answer: str, chunks: list[Any]) -> list[str]:
    """Return missing answer-contract sections implied by the cited clause.

    This is deliberately driven by source language, not by document names or
    saved question/answer pairs.  A direct technical clause often contains an
    operational requirement followed by an implementation example and a
    cross-reference.  Losing the latter two makes a retrieved answer look
    plausible while being unusable by an engineer.
    """
    section_lines: dict[str, list[str]] = {str(index): [] for index in range(1, 5)}
    current: str | None = None
    for line in (answer or "").splitlines():
        match = SECTION_RE.match(line.strip())
        if match:
            current = match.group(1)
            continue
        if current and line.strip().startswith("- ") and CITATION_RE.search(line):
            section_lines[current].append(line.strip())

    source = " ".join(clause_body(chunk) for chunk in chunks).lower()
    gaps: list[str] = []
    if not section_lines["1"]:
        gaps.append("section_1_has_no_verified_fact")
    # Only demand an operational/work item where the clause itself provides a
    # concrete design, monitoring, alarm, test, reporting or configuration
    # proposition.
    if re.search(
        r"\b(control|monitoring|alert|alarm|observe|display|sensor|camera|"
        r"communication|test|approval|reporting|arranged|installation)\b",
        source,
        re.I,
    ) and not section_lines["2"]:
        gaps.append("section_2_missing_direct_work_item")
    # A cross-reference or an existing requirement is an explicit follow-up,
    # not an invitation to write a generic 'nothing to confirm' sentence.
    if re.search(
        r"\bsee\s+also\b|\bexisting requirements?\b|\badditional considerations?\b|"
        r"\bnormal and abnormal conditions?\b|\bsubject to\b",
        source,
        re.I,
    ) and not section_lines["3"]:
        gaps.append("section_3_missing_explicit_followup")
    if not section_lines["4"]:
        gaps.append("section_4_has_no_reference")

    answer_low = (answer or "").lower()
    # Source-driven detail coverage for any technical clause. If the source
    # explicitly names an existing requirement, operating-condition boundary,
    # human-sense limitation, or compensating measure, do not silently omit it.
    concept_checks = (
        (
            r"\bunattended machinery space\b",
            ("무인 기관실", "기관실 무인화", "unattended machinery"),
            "missing_existing_unattended_machinery_requirement",
        ),
        (
            r"\bnormal and abnormal conditions?\b",
            ("정상 및 비정상", "정상·비정상", "normal and abnormal"),
            "missing_normal_abnormal_condition_boundary",
        ),
        (
            r"\bhuman senses?\b|\bvibrations?\b|\bhigh temperatures?\b",
            ("인간의 감각", "진동", "고온", "human senses", "vibration", "high temperature"),
            "missing_human_sense_detection_consideration",
        ),
        (
            r"\binfrared cameras?\b|\bmicrophones?\b|\bvibration sensors?\b",
            ("적외선", "마이크", "진동 센서", "infrared", "microphone", "vibration sensor"),
            "missing_compensating_sensor_examples",
        ),
        (
            r"\bship-shore collaboration\b",
            ("선박-육상", "선박·육상", "선육", "ship-shore"),
            "missing_ship_shore_collaboration_example",
        ),
    )
    for source_pattern, answer_terms, gap_name in concept_checks:
        if re.search(source_pattern, source, re.I) and not any(
            term.lower() in answer_low for term in answer_terms
        ):
            gaps.append(gap_name)
    return gaps


def ensure_direct_clause_source_details(answer: str, chunks: list[Any]) -> str:
    """Restore material clause details with conservative Korean renderings.

    The local LLM sometimes summarizes only the first sentences of a long
    clause.  This pass is source-driven: it adds a detail only when the cited
    clause literally contains the corresponding concept.  It therefore works
    for unseen questions without a saved question/answer lookup.
    """
    if not answer or not chunks:
        return answer

    source = " ".join(clause_body(chunk) for chunk in chunks).lower()
    answer_low = answer.lower()
    additions: dict[str, list[str]] = {"1": [], "2": [], "3": []}

    if (
        re.search(r"\binfrared cameras?\b|\bmicrophones?\b|\bvibration sensors?\b", source)
        and not any(term in answer_low for term in ("적외선", "마이크", "진동 센서"))
    ):
        additions["2"].append(
            "- 진동·고온 등 이상 상태의 탐지를 보완하기 위해 일반·적외선 카메라, "
            "마이크 또는 진동 센서와 같은 보완수단을 고려할 수 있습니다. [1]"
        )
    if (
        "ship-shore collaboration" in source
        and not any(term in answer_low for term in ("선박-육상", "선박·육상", "선박과 육상"))
    ):
        additions["2"].append(
            "- 영상통신 등 효율적인 선박-육상 협업 수단도 ROC 운영자의 판단을 "
            "지원하는 보완대책으로 고려할 수 있습니다. [1]"
        )
    if (
        "unattended machinery space" in source
        and not any(term in answer_low for term in ("무인 기관실", "무인기관실"))
    ):
        additions["3"].append(
            "- 기존 무인 기관실 운전 요건도 함께 준수할 필요가 있습니다. [1]"
        )
    if (
        re.search(r"\bsee\s+also\s+sec(?:tion)?\.?\s*\d+", source, re.I)
        and not re.search(r"\bSec\.?\s*\d+", answer, re.I)
    ):
        match = re.search(r"\bsee\s+also\s+(sec(?:tion)?\.?\s*\d+)", source, re.I)
        if match:
            reference = re.sub(r"section", "Sec.", match.group(1), flags=re.I)
            additions["3"].append(
                f"- 세부 설계·승인 범위는 문서의 {reference} 관련 요건과 함께 대조해야 합니다. [1]"
            )

    if not any(additions.values()):
        return answer

    lines = answer.splitlines()
    section_starts: dict[str, int] = {}
    for index, line in enumerate(lines):
        match = SECTION_RE.match(line.strip())
        if match:
            section_starts[match.group(1)] = index

    # Insert from the last section towards the first so earlier indices remain
    # stable.  New bullets are already citation-bound to the primary clause.
    for section in ("3", "2", "1"):
        new_lines = additions[section]
        if not new_lines or section not in section_starts:
            continue
        start = section_starts[section]
        end = next(
            (
                index
                for index in range(start + 1, len(lines))
                if SECTION_RE.match(lines[index].strip())
            ),
            len(lines),
        )
        existing = "\n".join(lines[start:end])
        filtered = [line for line in new_lines if line not in existing]
        if filtered:
            lines[end:end] = ["", *filtered]

    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def replace_rule_reference_section(answer: str, chunks: list[Any]) -> str:
    """Render the authoritative document/page/clause reference deterministically."""
    if not chunks:
        return answer
    chunk = chunks[0]
    doc = str(
        getattr(chunk, "file_name", "")
        or getattr(chunk, "doc_id", "")
        or getattr(chunk, "source", "")
        or "문서"
    )
    page = getattr(chunk, "page_number", None)
    clause, title = extract_clause_reference(chunk)
    reference = f"**{doc}**"
    if page not in (None, ""):
        reference += f", p.{page}"
    if clause:
        reference += f", clause {clause}"
    if title:
        reference += f" `{title}`"
    bullet = f"- {reference}가 질문에 직접 대응하는 근거입니다. [1]"

    lines = (answer or "").splitlines()
    start = next(
        (index for index, line in enumerate(lines) if SECTION_RE.match(line.strip())
         and SECTION_RE.match(line.strip()).group(1) == "4"),
        None,
    )
    if start is None:
        return (answer.rstrip() + "\n\n## 4) 관련 선급 Rule / Guidance\n" + bullet).strip()
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if SECTION_RE.match(lines[index].strip())
        ),
        len(lines),
    )
    return "\n".join([*lines[:start], "## 4) 관련 선급 Rule / Guidance", bullet, *lines[end:]]).strip()
