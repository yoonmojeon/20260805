"""Answer-level deduplication for meeting structured answers."""
from __future__ import annotations

import re

from meeting_topic_cluster import _topic_id_for_text

CITATION_RE = re.compile(r"\[(\d+)\]")
BULLET_RE = re.compile(r"^-\s+(.+)$", re.M)


def _normalize_summary(text: str) -> str:
    t = re.sub(r"\[\d+\]", "", text)
    t = re.sub(r"\([^)]*근거[^)]*\)", "", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t[:120]


def detect_section1_topics(section1: str) -> list[str]:
    from meeting_topic_cluster import outcome_topic_id

    topics: list[str] = []
    for m in BULLET_RE.finditer(section1):
        body = m.group(1)
        topics.append(outcome_topic_id(body))
    return topics


def dedup_section1_bullets(
    bullets: list[str],
    *,
    max_count: int | None = None,
    topic_caps: dict[str, int] | None = None,
    forbidden_substrings: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    caps = dict(topic_caps or {})
    cap_counts: dict[str, int] = {}
    seen_norm: set[str] = set()
    seen_cites: set[str] = set()
    seen_topics: set[str] = set()
    out: list[str] = []

    for bullet in bullets:
        body = bullet.lstrip("- ").strip()
        norm = _normalize_summary(body)
        cites = set(CITATION_RE.findall(body))
        topic = _topic_id_for_text(body)

        if forbidden_substrings and any(s.lower() in body.lower() for s in forbidden_substrings):
            warnings.append("wrong_topic_in_answer")
            continue

        if norm in seen_norm:
            warnings.append("duplicate_topic")
            continue
        if cites and cites <= seen_cites and len(cites) == 1:
            warnings.append("duplicate_citation")
            continue
        cap = caps.get(topic)
        if cap is not None and cap_counts.get(topic, 0) >= cap:
            warnings.append("duplicate_topic")
            continue
        if topic in seen_topics and topic not in ("general_outcome",):
            warnings.append("duplicate_topic")
            continue

        seen_norm.add(norm)
        seen_cites.update(cites)
        seen_topics.add(topic)
        cap_counts[topic] = cap_counts.get(topic, 0) + 1
        out.append(bullet if bullet.startswith("- ") else f"- {body}")

    if max_count is not None and len(out) > max_count:
        warnings.append("answer_count_mismatch")
        out = out[:max_count]

    return out, list(dict.fromkeys(warnings))


def split_four_sections(answer: str) -> dict[str, str]:
    parts: dict[str, str] = {}
    current = "1"
    buf: list[str] = []
    for line in answer.splitlines():
        if line.startswith("## 1)"):
            current, buf = "1", []
            continue
        if line.startswith("## 2)"):
            parts["1"] = "\n".join(buf).strip()
            current, buf = "2", []
            continue
        if line.startswith("## 3)"):
            parts["2"] = "\n".join(buf).strip()
            current, buf = "3", []
            continue
        if line.startswith("## 4)"):
            parts["3"] = "\n".join(buf).strip()
            current, buf = "4", []
            continue
        buf.append(line)
    if buf:
        parts[current] = "\n".join(buf).strip()
    return parts


def apply_answer_dedup(
    answer: str,
    *,
    profile_intent: str,
    requested_count: int | None = None,
) -> tuple[str, list[str]]:
    from meeting_topic_scoring import topic_caps_for_intent

    parts = split_four_sections(answer)
    s1 = parts.get("1", "")
    bullets = [f"- {m.group(1).strip()}" for m in BULLET_RE.finditer(s1)]
    if not bullets:
        return answer, []

    forbidden: list[str] = []
    if profile_intent == "mass_code_timeline":
        return answer, []

    if profile_intent == "meeting_outcome":
        return answer, []

    if profile_intent == "altfuel_ghg_safety":
        forbidden = ["mass code", "자율운항", "maritime autonomous"]

    deduped, warnings = dedup_section1_bullets(
        bullets,
        max_count=requested_count,
        topic_caps=topic_caps_for_intent(profile_intent),
        forbidden_substrings=forbidden,
    )
    parts["1"] = "\n".join(deduped)

    from answer_depth_guidance import join_four_sections

    return join_four_sections(parts), warnings
