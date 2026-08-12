"""Lookup doc_ids from corpus manifest by IMO filename patterns."""
from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

from imo_doc_classify import classify_imo_filename, session_agenda_key
from retrieval_query_analysis import QuerySignals

DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "data/manifests/rag_corpus_407.csv"


@lru_cache(maxsize=4)
def load_corpus_rows(corpus_path: str) -> tuple[dict, ...]:
    path = Path(corpus_path)
    if not path.exists():
        return ()
    rows: list[dict] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return tuple(rows)


def clear_corpus_rows_cache() -> None:
    load_corpus_rows.cache_clear()


def _session_in_name(file_name: str, body: str, num: int) -> bool:
    fn = (file_name or "").lower()
    body_l = body.lower()
    return (
        f"{body_l} {num}-" in fn
        or f"{body_l}-{num}" in fn
        or f"{body_l}/{num}" in fn
        or re.search(rf"\b{body_l}\s*{num}\b", fn) is not None
    )


def priority_doc_ids(
    signals: QuerySignals,
    *,
    corpus_path: Path = DEFAULT_CORPUS,
    preferred_types: tuple[str, ...] | None = None,
    agenda_items: tuple[int, ...] | None = None,
    limit: int = 24,
) -> list[str]:
    """Return doc_ids whose filenames match session + document role."""
    rows = load_corpus_rows(str(corpus_path))
    if not rows or not signals.session_codes:
        return []

    sources = {body for body, _ in signals.session_codes}
    out: list[tuple[int, str]] = []

    for row in rows:
        source = str(row.get("source", "")).upper()
        if source not in sources:
            continue
        file_name = str(row.get("file_name", ""))
        doc_id = str(row.get("doc_id", ""))
        if not doc_id or not file_name:
            continue

        matched_session = False
        matched_agenda = agenda_items is None
        for body, num in signals.session_codes:
            if _session_in_name(file_name, body, num):
                matched_session = True
            if agenda_items:
                key = session_agenda_key(file_name)
                if key and key[0] == body and key[1] == num and key[2] in agenda_items:
                    matched_agenda = True

        if not matched_session or not matched_agenda:
            continue

        doc_type = classify_imo_filename(file_name)
        if preferred_types and doc_type not in preferred_types:
            continue

        priority = 0
        if doc_type == "session_outcome":
            priority += 100
        elif doc_type == "working_group_report":
            priority += 80
        elif doc_type == "session_report":
            priority += 70
        elif doc_type == "agenda_report":
            priority += 60
        elif doc_type == "subcommittee_report":
            priority += 30
        if "secretariat" in file_name.lower():
            priority += 5
        out.append((priority, doc_id))

    out.sort(key=lambda x: (-x[0], x[1]))
    seen: set[str] = set()
    ids: list[str] = []
    for _, doc_id in out:
        if doc_id not in seen:
            seen.add(doc_id)
            ids.append(doc_id)
        if len(ids) >= limit:
            break
    return ids


def priority_doc_ids_for_signals(signals: QuerySignals) -> list[str]:
    """Map query intent to priority IMO documents."""
    if signals.wants_rule_lookup or not signals.session_codes:
        return []

    preferred: tuple[str, ...] | None = None
    agenda: tuple[int, ...] | None = None

    latest_mepc_environment = (
        any(body == "MEPC" and num == 84 for body, num in signals.session_codes)
        and bool(signals.topics.intersection({"ghg", "marpol", "cii"}))
    )

    if latest_mepc_environment:
        agenda = (3, 6, 7, 10)
        preferred = (
            "working_group_report",
            "agenda_report",
            "session_report",
            "session_outcome",
            "reference_outcome",
        )
    elif "mass" in signals.topics:
        agenda = (5,)
        preferred = (
            "working_group_report",
            "session_outcome",
            "session_report",
            "agenda_report",
        )
    elif "ghg" in signals.topics and any(b == "MEPC" for b, _ in signals.session_codes):
        agenda = (7,)
        preferred = ("working_group_report", "agenda_report", "session_outcome")
    elif "alt_fuel" in signals.topics and any(b == "MSC" for b, _ in signals.session_codes):
        agenda = (12, 14)
        preferred = (
            "subcommittee_report",
            "working_group_report",
            "agenda_report",
            "session_outcome",
            "session_report",
        )
    elif "cii" in signals.topics or "marpol" in signals.topics:
        agenda = (6,)
        preferred = ("agenda_report", "working_group_report", "session_outcome")
    elif signals.wants_agenda:
        preferred = ("agenda",)
    elif signals.wants_summary or signals.wants_outcome:
        preferred = (
            "session_outcome",
            "session_report",
            "working_group_report",
            "agenda_report",
        )

    return priority_doc_ids(signals, preferred_types=preferred, agenda_items=agenda)
