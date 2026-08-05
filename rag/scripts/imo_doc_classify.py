"""IMO document filename classification for retrieval routing."""
from __future__ import annotations

import re

# Agenda item at session level: "MSC 111-2 -", "MEPC 84-7-14 -"
SESSION_AGENDA_RE = re.compile(
    r"\b(msc|mepc)\s*(\d{1,3})-(\d{1,2})(?:-\d+)?\s*-",
    re.IGNORECASE,
)

REFERENCE_OUTCOME_RE = re.compile(r"outcome\s+of\s+", re.I)
SESSION_FINAL_REPORT_RE = re.compile(
    r"(?:"
    r"\bwp\.?\s*1\b.*draft\s+report|"
    r"draft\s+report\s+of\s+the\s+(?:maritime\s+safety|marine\s+environment)\s+committee|"
    r"report\s+of\s+the\s+(?:maritime\s+safety|marine\s+environment)\s+committee\s+on\s+its"
    r")",
    re.I,
)

OFFICIAL_MEETING_SUMMARY_RE = re.compile(
    r"(?:"
    r"meeting\s+highlights|"
    r"secretary.?general.*closing|"
    r"closing\s+remarks|"
    r"press\s+briefing|"
    r"meeting\s+summary"
    r")",
    re.I,
)

MSC_SESSION_REPORT_RE = re.compile(
    r"\b(?:msc|mepc)\s*(\d{1,3})[-/]22\b.*report\s+of\s+the",
    re.I,
)

SUMMARY_PENALTY_FILENAME_RE = re.compile(
    r"(?:"
    r"outcome\s+of\s+|"
    r"decisions?\s+of\s+other\s+imo\s+bodies?|"
    r"strategic\s+plan|"
    r"\bfal\s*50\b|"
    r"\b(?:msc|mepc)\s*\d{1,3}[-/]2(?:[-/]|$)"
    r")",
    re.I,
)

BROAD_SESSION_OUTCOME_MARKERS = (
    "주요 결과",
    "회의 결과",
    "최종 결과",
    "세션 결과",
    "전체 결과",
    "핵심 성과",
    "핵심 outcome",
    "결정된 주요 내용",
    "key outcomes",
    "key outcome",
    "overall outcome",
    "session outcome",
    "main outcomes",
    "outcome summary",
)


def meeting_summary_source_tier(file_name: str, *, doc_id: str = "") -> int:
    """Source priority for meeting_summary: 0=official summary, 1=session report, 2=neutral, 3=penalty."""
    fn = (file_name or "").lower()
    did = (doc_id or "").lower()
    blob = f"{fn} {did}"
    if not blob.strip():
        return 2
    if OFFICIAL_MEETING_SUMMARY_RE.search(blob):
        return 0
    if MSC_SESSION_REPORT_RE.search(blob):
        return 1
    if SESSION_FINAL_REPORT_RE.search(blob):
        return 1
    if "summary report" in blob and re.search(r"\b(msc|mepc)\s*\d", blob):
        return 1
    if SUMMARY_PENALTY_FILENAME_RE.search(blob):
        return 3
    if REFERENCE_OUTCOME_RE.search(blob):
        return 3
    return 2


def meeting_outcome_scope(file_name: str) -> str:
    """Finer scope for meeting-outcome retrieval (filename heuristics)."""
    fn = (file_name or "").lower()
    if not fn:
        return "unknown"
    if SESSION_FINAL_REPORT_RE.search(fn):
        return "session_final_report"
    if "summary report" in fn and re.search(r"\b(msc|mepc)\s*\d", fn):
        return "session_final_report"
    if REFERENCE_OUTCOME_RE.search(fn):
        return "reference_body_outcome"
    if "report of the" in fn and (
        "working group" in fn or "sub-committee" in fn or "sub committee" in fn or "intersessional" in fn
    ):
        return "working_group_report"
    if " outcome" in fn or "-outcome" in fn:
        return "agenda_outcome"
    return "unknown"


def asks_broad_session_outcome(question: str, *, topics: tuple[str, ...] | None = None) -> bool:
    """True when the user asks for the whole session's outcomes, not a single agenda topic."""
    if topics:
        return False
    q = (question or "").lower()
    if any(marker.lower() in q for marker in BROAD_SESSION_OUTCOME_MARKERS):
        return True
    if re.search(r"\d+\s*개\s*(?:항목|개)", question or ""):
        return True
    if re.search(r"\d+\s*(?:items?|points?|bullets?)", q):
        return True
    return False


def classify_imo_filename(file_name: str) -> str:
    """Return document role label from IMO-style filename."""
    fn = (file_name or "").lower()
    if not fn:
        return "unknown"

    if "comments on" in fn or "comment on" in fn:
        return "comments"
    if "proposal for" in fn:
        return "proposal"
    if "-inf." in fn or re.search(r"\binf\.\d", fn):
        return "inf"
    if REFERENCE_OUTCOME_RE.search(fn):
        return "reference_outcome"
    if SESSION_FINAL_REPORT_RE.search(fn):
        return "session_report"
    if " outcome" in fn or "-outcome" in fn:
        return "session_outcome"
    if "annotations to the provisional agenda" in fn or re.search(r"-1-1\b|-1-rev", fn):
        return "agenda"
    if "report of the" in fn and ("sub-committee" in fn or "sub committee" in fn or "thesub-committee" in fn):
        return "subcommittee_report"
    if "intersessional" in fn and "report" in fn:
        return "working_group_report"
    if "correspondence group" in fn and "report" in fn:
        return "working_group_report"
    if "report of the" in fn:
        return "session_report"
    if re.search(r"-\d{1,2}\s*-", fn) and "report" in fn:
        return "agenda_report"
    return "other"


def session_agenda_key(file_name: str) -> tuple[str, int, int] | None:
    """Parse (body, session, agenda) from filename, e.g. MSC 111-5 -> MSC,111,5."""
    m = SESSION_AGENDA_RE.search(file_name or "")
    if not m:
        return None
    return m.group(1).upper(), int(m.group(2)), int(m.group(3))


def tier_for_query(doc_type: str, *, wants_summary: bool, wants_outcome: bool, wants_agenda: bool) -> float:
    """Signed boost tier: positive = prefer, negative = penalize."""
    if wants_agenda:
        if doc_type == "agenda":
            return 0.35
        if doc_type in ("proposal", "comments", "inf"):
            return -0.15
        return 0.0

    if wants_summary or wants_outcome:
        scores = {
            "session_report": 0.34,
            "session_outcome": 0.22,
            "reference_outcome": -0.18,
            "working_group_report": 0.14,
            "agenda_report": 0.12,
            "subcommittee_report": -0.12,
            "proposal": -0.20,
            "comments": -0.25,
            "inf": -0.18,
            "agenda": 0.05,
            "other": 0.0,
            "unknown": 0.0,
        }
        return scores.get(doc_type, 0.0)

    return 0.0
