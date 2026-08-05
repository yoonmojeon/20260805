"""Topic clustering and deduplication for meeting summary bullets."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

TOPIC_RULES: list[tuple[str, list[str]]] = [
    ("mass_code", ["mass code", "maritime autonomous", "autonomous surface ship", "remotely operated", "degree of autonomy"]),
    ("ghg_safety", ["ghg safety", "alternative fuel", "ammonia", "hydrogen", "methanol", "lng", "low-flashpoint", "fuel safety", "new technolog"]),
    ("ghg_framework", ["net-zero", "gfi", "seemp", "marpol annex vi", "lifecycle", "lca", "ghg study"]),
    ("cii_reporting", ["cii", "carbon intensity", "fleet report", "reporting year", "data collection", "verification"]),
    ("lrit_vdes", ["lrit", "vdes", "long-range identification", "gmdss"]),
    ("maritime_safety", ["solas", "maritime safety", "navigation", "fire safety code"]),
    ("hormuz", ["hormuz", "strait"]),
    ("general_outcome", ["adopted", "approved", "resolution", "outcome"]),
]


@dataclass
class TopicCluster:
    topic_id: str
    label_ko: str
    members: list[Any] = field(default_factory=list)
    best_score: float = 0.0

    @property
    def representative(self) -> Any | None:
        return self.members[0] if self.members else None


def _topic_id_for_text(text: str) -> str:
    low = text.lower()
    for tid, keys in TOPIC_RULES:
        if any(k in low for k in keys):
            return tid
    return "general_outcome"


def outcome_topic_id(text: str) -> str:
    """Finer topic labels for MSC meeting-outcome bullet selection."""
    low = text.lower()
    if any(k in low for k in ("lrit", "vdes", "gmdss", "long-range identification")):
        if "mass code" not in low:
            return "lrit_vdes"
    if any(k in low for k in ("hormuz", "strait of hormuz")):
        return "hormuz"
    if any(
        k in low
        for k in (
            "alternative fuel",
            "ghg safety",
            "ammonia",
            "hydrogen",
            "methanol",
            "low-flashpoint",
            "new technolog",
            "ise ",
            "interim guideline",
        )
    ):
        if "mass code" not in low:
            return "ghg_safety"
    if any(k in low for k in ("solas", "fire safety code")) and "mass code" not in low:
        return "maritime_safety"
    if "mass code" in low or "maritime autonomous" in low:
        return "mass_code"
    return _topic_id_for_text(text)


def _topic_label_ko(topic_id: str) -> str:
    return {
        "mass_code": "MASS Code / 자율운항",
        "ghg_safety": "GHG·대체연료 안전",
        "ghg_framework": "GHG·Net-Zero·SEEMP/GFI",
        "cii_reporting": "CII·규제 보고",
        "lrit_vdes": "LRIT·VDES·GMDSS",
        "maritime_safety": "SOLAS·해상안전",
        "hormuz": "호르무즈 해협 관련",
        "general_outcome": "회의 결정·결과",
    }.get(topic_id, "기타 회의 결과")


def cluster_chunks(
    scored_chunks: list[tuple[float, Any]],
    *,
    max_clusters: int = 10,
) -> list[TopicCluster]:
    buckets: dict[str, TopicCluster] = {}
    for score, chunk in scored_chunks:
        text = str(getattr(chunk, "text", "") or "")
        tid = _topic_id_for_text(text)
        if tid not in buckets:
            buckets[tid] = TopicCluster(topic_id=tid, label_ko=_topic_label_ko(tid))
        buckets[tid].members.append(chunk)
        buckets[tid].best_score = max(buckets[tid].best_score, score)

    clusters = sorted(buckets.values(), key=lambda c: -c.best_score)
    for c in clusters:
        c.members.sort(
            key=lambda ch: -float(getattr(ch, "_meeting_score", 0) or 0),
        )
    return clusters[:max_clusters]


MSC_OUTCOME_TOPIC_PRIORITY: tuple[str, ...] = (
    "mass_code",
    "ghg_safety",
    "lrit_vdes",
    "maritime_safety",
    "hormuz",
    "ghg_framework",
    "cii_reporting",
    "general_outcome",
)


def pick_diverse_topic_chunks(
    scored_chunks: list[tuple[float, Any]],
    n: int,
    *,
    used_chunk_ids: set[str] | None = None,
    topic_priority: tuple[str, ...] | None = None,
    topic_caps: dict[str, int] | None = None,
    exclude_topics: set[str] | None = None,
    allowed_topics: set[str] | None = None,
    topic_fn=None,
) -> list[Any]:
    """Pick up to n chunks with distinct topic_ids, highest score first per topic."""
    topic_fn = topic_fn or _topic_id_for_text
    used = set(used_chunk_ids or ())
    caps = dict(topic_caps or {})
    cap_counts: dict[str, int] = {}
    by_topic: dict[str, tuple[float, Any]] = {}
    for score, chunk in scored_chunks:
        cid = str(getattr(chunk, "chunk_id", "") or "")
        if cid in used:
            continue
        text = str(getattr(chunk, "text", "") or "")
        tid = topic_fn(text)
        if exclude_topics and tid in exclude_topics:
            continue
        if allowed_topics and tid not in allowed_topics:
            continue
        prev = by_topic.get(tid)
        if prev is None or score > prev[0]:
            by_topic[tid] = (score, chunk)

    picked: list[Any] = []
    order = list(topic_priority or ()) + [t for t in by_topic if t not in (topic_priority or ())]
    for tid in order:
        if tid not in by_topic:
            continue
        if caps.get(tid, 999) <= cap_counts.get(tid, 0):
            continue
        _, chunk = by_topic[tid]
        cid = str(getattr(chunk, "chunk_id", "") or "")
        if cid in used:
            continue
        used.add(cid)
        picked.append(chunk)
        cap_counts[tid] = cap_counts.get(tid, 0) + 1
        if len(picked) >= n:
            break

    if len(picked) < n:
        for _score, chunk in scored_chunks:
            cid = str(getattr(chunk, "chunk_id", "") or "")
            if cid in used:
                continue
            text = str(getattr(chunk, "text", "") or "")
            tid = topic_fn(text)
            if exclude_topics and tid in exclude_topics:
                continue
            if allowed_topics and tid not in allowed_topics:
                continue
            if caps.get(tid, 999) <= cap_counts.get(tid, 0):
                continue
            used.add(cid)
            picked.append(chunk)
            cap_counts[tid] = cap_counts.get(tid, 0) + 1
            if len(picked) >= n:
                break
    return picked


def dedupe_page_chunks(chunks: list[Any]) -> list[Any]:
    seen: set[tuple[str, int | None]] = set()
    out: list[Any] = []
    for c in chunks:
        key = (str(getattr(c, "file_name", "")), getattr(c, "page_number", None))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
