"""Heuristic evidence claim extraction for Fast mode (pre-LLM)."""
from __future__ import annotations

import re
from typing import Any

from imo_doc_classify import meeting_summary_source_tier
from meeting_summary_context import (
    get_summary_context,
    is_penalty_summary_source,
    score_summary_claim_text,
    topic_priority_for_context,
)
from meeting_summary_intent import is_meeting_summary_intent
from rag_answer_lib import RetrievedChunk

IMO_ACTION_RE = re.compile(
    r"\b(adopted|approved|endorsed|finalized|finalised|mandatory|non-mandatory|"
    r"entry into force|entered into force|agreed|recalled)\b",
    re.I,
)
RESOLUTION_COUNT_RE = re.compile(
    r"(\d+)\s+(resolutions?|decisions?)\b",
    re.I,
)


def _snippet(text: str, max_len: int = 220) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t[: max_len - 1] + "…" if len(t) > max_len else t


def _confidence_for(
    sent: str,
    chunk: RetrievedChunk,
    *,
    meeting_summary: bool = False,
    ctx=None,
) -> str:
    from meeting_summary_context import score_summary_claim_text

    source = chunk.file_name or chunk.doc_id or ""
    if meeting_summary:
        score = score_summary_claim_text(
            sent, source_name=source, doc_id=chunk.doc_id or "", ctx=ctx
        )
        if score >= 5.0:
            return "high"
        if score >= 2.0:
            return "medium"
        if score < -2.0:
            return "low"
    if RESOLUTION_COUNT_RE.search(sent) or IMO_ACTION_RE.search(sent):
        return "high"
    if len(sent) > 40 and chunk.page_number is not None:
        return "medium"
    return "low"


def _term_check(sent: str) -> dict[str, str]:
    out: dict[str, str] = {}
    lower = sent.lower()
    for term in (
        "resolution",
        "decisions",
        "decision",
        "adopted",
        "approved",
        "endorsed",
        "mandatory",
        "guidelines",
        "work plan",
    ):
        if term in lower or (term == "resolution" and "resolutions" in lower):
            out[term.rstrip("s") if term.endswith("s") else term] = "present"
    m = RESOLUTION_COUNT_RE.search(sent)
    if m:
        out["resolution_count"] = m.group(1)
        out[m.group(2).lower().rstrip("s")] = m.group(1)
    return out


def _chunks_for_claims(chunks: list[RetrievedChunk], question: str, row: dict | None) -> list[RetrievedChunk]:
    from meeting_summary_context import is_penalty_summary_source, meeting_summary_source_tier

    ctx = get_summary_context(question, row)
    if not is_meeting_summary_intent(question, row) and not ctx.apply_session_final_priority:
        return chunks
    if ctx.apply_session_final_priority:
        preferred = [
            c
            for c in chunks
            if meeting_summary_source_tier(c.file_name or "", doc_id=c.doc_id or "", ctx=ctx) <= 1
        ]
        if preferred:
            return preferred
    if ctx.preferred_doc_hints:
        hinted = [
            c
            for c in chunks
            if any(h in (c.file_name or "").lower() or h in (c.doc_id or "").lower() for h in ctx.preferred_doc_hints)
        ]
        if hinted:
            return hinted
    fallback = [
        c
        for c in chunks
        if not is_penalty_summary_source(c.file_name or "", c.doc_id or "", ctx=ctx)
    ]
    return fallback if fallback else chunks


def extract_claim_candidates(
    chunks: list[RetrievedChunk],
    *,
    max_claims: int = 24,
    question: str = "",
    row: dict | None = None,
) -> list[dict[str, Any]]:
    meeting_summary = is_meeting_summary_intent(question, row)
    ctx = get_summary_context(question, row)
    chunks = _chunks_for_claims(chunks, question, row)
    claims: list[dict[str, Any]] = []
    seen: set[str] = set()

    for cite_id, chunk in enumerate(chunks, start=1):
        source = chunk.file_name or chunk.doc_id or ""
        if meeting_summary and is_penalty_summary_source(source, chunk.doc_id or "", ctx=ctx):
            continue
        text = chunk.text or ""
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 25:
                continue
            summary_score = score_summary_claim_text(
                sent, source_name=source, doc_id=chunk.doc_id or "", ctx=ctx
            )
            if meeting_summary or ctx.apply_session_final_priority:
                if summary_score < 0.5 and not IMO_ACTION_RE.search(sent):
                    continue
            elif not IMO_ACTION_RE.search(sent) and not RESOLUTION_COUNT_RE.search(sent):
                continue
            key = _snippet(sent, 120).lower()
            if key in seen:
                continue
            seen.add(key)
            conf = _confidence_for(sent, chunk, meeting_summary=meeting_summary, ctx=ctx)
            if (meeting_summary or ctx.apply_session_final_priority) and conf == "low":
                continue
            claims.append(
                {
                    "claim": _snippet(sent, 180),
                    "evidence": _snippet(sent, 220),
                    "page": chunk.page_number,
                    "chunk_id": chunk.chunk_id,
                    "cite_id": cite_id,
                    "source": source,
                    "confidence": conf,
                    "term_check": _term_check(sent),
                    "summary_score": round(summary_score, 2),
                }
            )

        if not meeting_summary:
            for m in RESOLUTION_COUNT_RE.finditer(text):
                n, kind = m.group(1), m.group(2).lower()
                claim_txt = f"{n} {kind} mentioned in document"
                key = claim_txt.lower()
                if key not in seen:
                    seen.add(key)
                    claims.append(
                        {
                            "claim": f"문서에 {n}건의 {kind} 언급",
                            "evidence": _snippet(text[max(0, m.start() - 40) : m.end() + 80], 200),
                            "page": chunk.page_number,
                            "chunk_id": chunk.chunk_id,
                            "cite_id": cite_id,
                            "source": source,
                            "confidence": "high",
                            "term_check": {kind.rstrip("s"): n},
                            "summary_score": 5.0,
                        }
                    )

    if meeting_summary:
        claims.sort(
            key=lambda c: (
                -float(c.get("summary_score", 0)),
                {"high": 0, "medium": 1, "low": 2}[c.get("confidence", "low")],
            )
        )
    else:
        claims.sort(key=lambda c: {"high": 0, "medium": 1, "low": 2}[c["confidence"]])
    return claims[:max_claims]


def filter_claims_for_answer(claims: list[dict[str, Any]], *, min_confidence: str = "medium") -> list[dict[str, Any]]:
    order = {"high": 0, "medium": 1, "low": 2}
    threshold = order[min_confidence]
    return [c for c in claims if order.get(c.get("confidence", "low"), 2) <= threshold]


def _topic_key(text: str, keywords: tuple[str, ...]) -> bool:
    lower = (text or "").lower()
    return any(kw in lower for kw in keywords)


def _too_similar_claim(a: str, b: str) -> bool:
    a_norm = re.sub(r"\s+", " ", (a or "").lower())[:90]
    b_norm = re.sub(r"\s+", " ", (b or "").lower())[:90]
    if a_norm[:70] == b_norm[:70]:
        return True
    if a_norm[:45] and a_norm[:45] == b_norm[:45]:
        return True
    return False


def _claim_topic_signature(text: str) -> str | None:
    lower = (text or "").lower()
    if "mass code" in lower or "maritime autonomous" in lower:
        return "mass"
    if "in adopting resolution" in lower and "solas" in lower:
        return "solas_amend"
    if "lrit" in lower:
        return "lrit"
    if "vdes" in lower:
        return "vdes"
    if "alternative fuel" in lower or "ghg" in lower:
        return "ghg_alt_fuel"
    if "hormuz" in lower:
        return "hormuz"
    if "ammonia" in lower or "hydrogen" in lower:
        return "alt_fuel_guidelines"
    return None


def _select_summary_claims(
    claims: list[dict[str, Any]], target_n: int, *, ctx=None
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    used_signatures: set[str] = set()

    def add(c: dict[str, Any]) -> None:
        claim = c.get("claim") or ""
        key = claim[:80]
        if key in used_keys:
            return
        sig = _claim_topic_signature(claim)
        if sig and sig in used_signatures:
            return
        for prev in selected:
            if _too_similar_claim(claim, prev.get("claim") or ""):
                return
        used_keys.add(key)
        if sig:
            used_signatures.add(sig)
        selected.append(c)

    ranked = sorted(claims, key=lambda c: -float(c.get("summary_score", 0)))

    topic_priority = topic_priority_for_context(ctx) if ctx else ()

    for _label, kws, required in topic_priority:
        if not required and len(selected) >= target_n:
            break
        for c in ranked:
            ev = f"{c.get('evidence', '')} {c.get('claim', '')}".lower()
            if _topic_key(ev, kws):
                add(c)
                break
        if required and not any(
            _topic_key(f"{c.get('evidence', '')} {c.get('claim', '')}", kws) for c in selected
        ):
            for c in ranked:
                if _topic_key(f"{c.get('evidence', '')} {c.get('claim', '')}", kws):
                    add(c)
                    break

    for c in ranked:
        if len(selected) >= target_n:
            break
        add(c)
    return selected[:target_n]


def select_claims_for_question(
    claims: list[dict[str, Any]],
    question: str,
    *,
    target_n: int = 3,
    row: dict | None = None,
) -> list[dict[str, Any]]:
    if not claims:
        return []
    n = target_n
    m = re.search(r"(\d+)\s*개", question)
    if m:
        n = int(m.group(1))

    if is_meeting_summary_intent(question, row):
        return _select_summary_claims(claims, n, ctx=get_summary_context(question, row))

    selected: list[dict[str, Any]] = []
    used: set[str] = set()

    def add(c: dict) -> None:
        key = (c.get("claim") or "")[:80]
        if key not in used:
            selected.append(c)
            used.add(key)

    for c in claims:
        if len(selected) >= n:
            break
        add(c)
    return selected[:n]
