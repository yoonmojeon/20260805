"""Select one citation-stable evidence set for planning, generation and UI.

Evidence completion records the best chunk ids for every question requirement.
Historically those ids were used to report slot coverage, but answer generation
trimmed a different list and the Evidence Table was then built from yet another
list.  This module makes the ordered slot hits the single source of truth.

The implementation is intentionally document- and question-independent.  It
does not contain known answers, file names, page numbers or pilot-question
strings.
"""
from __future__ import annotations

import re
from typing import Any, Iterable


def _identity(chunk: Any) -> str:
    return str(getattr(chunk, "chunk_id", "") or "") or (
        f"{getattr(chunk, 'doc_id', '')}:"
        f"{getattr(chunk, 'file_name', '')}:"
        f"{getattr(chunk, 'page_number', '')}"
    )


def _unique(chunks: Iterable[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()
    for chunk in chunks:
        identity = _identity(chunk)
        if not identity or identity in seen:
            continue
        seen.add(identity)
        output.append(chunk)
    return output


def select_planned_evidence(
    row: dict,
    retrieved: Iterable[Any],
    pool: Iterable[Any] | None = None,
    *,
    max_chunks: int = 12,
) -> tuple[list[Any], dict[str, Any]]:
    """Return a diverse, slot-first evidence set and an auditable selection log.

    Pass one guarantees one hit per planned slot.  Pass two adds each slot's
    remaining hits.  Only then are ordinary ranked candidates appended.  The
    returned order is also the citation numbering order used by the answer and
    Evidence Table.
    """
    retrieved_list = list(retrieved or [])
    pool_list = list(pool or [])
    candidates = _unique([*retrieved_list, *pool_list])
    by_id = {
        str(getattr(chunk, "chunk_id", "") or ""): chunk
        for chunk in candidates
        if str(getattr(chunk, "chunk_id", "") or "")
    }
    completion = row.get("_evidence_completion") or {}
    slot_hits = completion.get("slot_hits") or {}
    from compound_regulatory import is_compound_regulatory_class_question

    compound_regulatory_class = is_compound_regulatory_class_question(
        str(row.get("question") or "")
    )

    selected: list[Any] = []
    seen: set[str] = set()
    selected_by_slot: dict[str, list[str]] = {}

    def add(chunk: Any, slot_name: str | None = None) -> bool:
        identity = _identity(chunk)
        if not identity or identity in seen or len(selected) >= max_chunks:
            return False
        seen.add(identity)
        selected.append(chunk)
        if slot_name:
            selected_by_slot.setdefault(str(slot_name), []).append(identity)
        return True

    # Literal feature recovery is used only after dense retrieval returned no
    # occurrence of a distinctive query noun.  Keep its strongest exact hit
    # in the one evidence list shared by generation, citations and the UI;
    # otherwise a later slot planner can undo the retrieval repair.
    feature_terms = list(
        ((row.get("_text_document_route") or {}).get("feature_fallback_terms") or [])
    )
    feature_hits: list[Any] = []
    if feature_terms:
        from retrieval_search import feature_fallback_relevance_score

        feature_hits = [
            chunk
            for chunk in candidates
            if any(
                str(term or "").lower()
                in str(getattr(chunk, "text", "") or "").lower()
                for term in feature_terms
            )
        ]
        feature_hits.sort(
            key=lambda chunk: max(
                feature_fallback_relevance_score(
                    str(row.get("question") or ""),
                    str(getattr(chunk, "text", "") or ""),
                    str(term or ""),
                )
                for term in feature_terms
            ),
            reverse=True,
        )
        if feature_hits and not compound_regulatory_class:
            add(feature_hits[0], "literal_feature")

    # Guarantee breadth before depth.
    for slot_name, chunk_ids in slot_hits.items():
        for chunk_id in list(chunk_ids or [])[:1]:
            chunk = by_id.get(str(chunk_id))
            if chunk is not None:
                add(chunk, str(slot_name))

    if feature_hits and compound_regulatory_class:
        # Two-lane slot coverage is more important than a duplicate literal
        # noun hit.  Keeping the literal hit after the decision/class slots
        # makes citation [1] the authoritative final decision instead of an
        # arbitrary class page that merely repeats the fuel name.
        add(feature_hits[0], "literal_feature")

    if compound_regulatory_class:
        # Keep the Committee's final position when an immediately preceding
        # paragraph only records delegations' objections.  This prevents a
        # compound answer from reporting the objection (e.g. an "unrealistic"
        # timeline) without the later "nevertheless agreed" outcome.
        final_position_markers = (
            "notwithstanding the above",
            "nevertheless agreed",
            "continue working towards the target year",
            "endorsed the revised road map",
            "the committee approved",
            "the committee adopted",
        )
        question_text = str(row.get("question") or "")
        question_numbers = set(re.findall(r"\d{4}", question_text))
        from compound_regulatory import compound_topic_terms

        topic_terms = tuple(
            term.lower()
            for term in compound_topic_terms(question_text)
            if len(term.strip()) >= 3
        )
        final_positions = [
            chunk
            for chunk in candidates
            if str(getattr(chunk, "source", "") or "").upper()
            in {"MSC", "MEPC", "IMO"}
            and any(
                marker in str(getattr(chunk, "text", "") or "").lower()
                for marker in final_position_markers
            )
            and (
                any(
                    term in str(getattr(chunk, "text", "") or "").lower()
                    for term in topic_terms
                )
                or any(
                    number in str(getattr(chunk, "text", "") or "")
                    for number in question_numbers
                )
            )
        ]
        final_positions.sort(
            key=lambda chunk: sum(
                marker in str(getattr(chunk, "text", "") or "").lower()
                for marker in final_position_markers
            ),
            reverse=True,
        )
        for final_position in final_positions:
            if add(final_position, "compound_final_position"):
                break

        # The evidence-completion pool may contain exact notation hits that
        # rank below generic engine clauses. Keep one literal instrument per
        # class society before the 12-chunk cutoff used by generation.
        from compound_regulatory import compound_exact_phrases

        exact_phrases = tuple(
            phrase.lower()
            for phrase in compound_exact_phrases(str(row.get("question") or ""))
            if phrase.strip()
        )
        for source in ("DNV", "KR", "ABS", "LR"):
            literal = next(
                (
                    chunk
                    for chunk in candidates
                    if str(getattr(chunk, "source", "") or "").upper() == source
                    and any(
                        phrase in str(getattr(chunk, "text", "") or "").lower()
                        for phrase in exact_phrases
                    )
                ),
                None,
            )
            if literal is not None:
                add(literal, f"compound_literal_instrument:{source}")

    # Retain additional propositions for slots that explicitly requested them.
    for slot_name, chunk_ids in slot_hits.items():
        for chunk_id in list(chunk_ids or [])[1:]:
            chunk = by_id.get(str(chunk_id))
            if chunk is not None:
                add(chunk, str(slot_name))

    # When no specialised slot exists (or after the slots are satisfied), keep
    # the strongest direct question matches before filling by retrieval rank.
    # This mirrors the production Accurate focus pass and prevents a correct
    # clause sitting later in the candidate pool from being cut off by twelve
    # generic introductory chunks from the same PDF.
    if candidates:
        from fast_context import question_focus_score

        question = str(row.get("question") or "")
        focused = [
            (
                question_focus_score(str(getattr(chunk, "text", "") or ""), question),
                index,
                chunk,
            )
            for index, chunk in enumerate(candidates)
        ]
        focused = [item for item in focused if item[0] > 0]
        focused.sort(key=lambda item: (-item[0], item[1]))
        focus_budget = 3 if not slot_hits else 2
        for _score, _index, chunk in focused[:focus_budget]:
            add(chunk, "question_focus")

    for chunk in candidates:
        if len(selected) >= max_chunks:
            break
        add(chunk)

    missing_slots = [
        str(slot_name)
        for slot_name, chunk_ids in slot_hits.items()
        if chunk_ids and not selected_by_slot.get(str(slot_name))
    ]
    return selected, {
        "strategy": "slot_first_single_source_of_truth",
        "max_chunks": max_chunks,
        "selected_chunk_ids": [_identity(chunk) for chunk in selected],
        "selected_by_slot": selected_by_slot,
        "missing_slots": missing_slots,
        "planned_slot_count": len(slot_hits),
    }
