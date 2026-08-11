"""Clause-aware hybrid retrieval (dense + lexical + metadata boost)."""
from __future__ import annotations

import math
import os
import re
from typing import Any

from clause_parse import is_article_clause_number
from imo_doc_classify import classify_imo_filename, tier_for_query
from imo_doc_registry import priority_doc_ids_for_signals
from retrieval_query_analysis import (
    CLASS_RULE_SOURCES,
    QuerySignals,
    analyze_query,
    session_file_prefixes,
    topic_agenda_prefixes,
)

CLAUSE_IN_QUERY_RE = re.compile(
    r"(?:제?\s*)?(\d{3,4})\s*절|(\d{3,4})절|(?:^|\s)(\d{3,4})(?:\s*절|\s|$)",
    re.IGNORECASE,
)

TOKEN_RE = re.compile(r"[\w가-힣]+", re.UNICODE)

CLAUSE_EXACT_BOOST = 0.22
CLAUSE_IN_TEXT_BOOST = 0.08
LEXICAL_BOOST_SCALE = 0.18
REFERENCE_LINE_BOOST = 0.06
SESSION_FILE_BOOST = 0.14
TOPIC_PREFIX_BOOST = 0.20
RULE_FILENAME_BOOST = 0.24
PRIORITY_DOC_BOOST = 0.28
EXPANDED_TERM_BOOST = 0.06
SUBCOMM_PENALTY = 0.14
DOC_CODE_CROSSREF_RE = re.compile(r"document\s+code.*title", re.I)
DNV_RULE_CODE_RE = re.compile(r"^DNV-(?:CG|RP|RU)-[A-Z0-9-]+$", re.I)
KR_RULE_PART_RE = re.compile(
    r"(?:제\s*)?(\d{1,2})\s*편(?:\s*[_-]?\s*(20\d{2})|\s*년판)?",
    re.IGNORECASE,
)
KR_PART1_CONTEXT_RE = re.compile(
    r"선급(?:등록|부호|검사|기술규칙)|공동선급선|중복선급선|동형선|"
    r"선박소유자|지적사항|불가항력|풍우밀|과도한\s*부식|쇠모한도|"
    r"건조계약일|문서준수확인서|탈급|양자\s*협정|등록된\s*선박|"
    r"시험\s*및\s*검사|제조중등록검사|"
    r"dual\s+class\s+vessel|double\s+class\s+vessel|sister\s+ship|"
    r"condition\s+of\s+class|force\s+majeure|weathertight|"
    r"substantial\s+corrosion",
    re.IGNORECASE,
)
EXACT_IDENTIFIER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"t(?:corr|c\s*[12])|"
    r"AC-[A-Z0-9+.-]+|"
    r"CII|EEXI|SEEMP|MARPOL|"
    r"DNV-(?:CG|RP|RU|OS|CP|SI)-[A-Z0-9-]+|"
    r"(?:MEPC|MSC)\s*\d{1,3}(?:\s*[/.-]\s*[A-Z0-9]+)+"
    r")(?![A-Za-z0-9])|\d+\s*장\s*\d+\s*절",
    re.I,
)
SPARSE_TOKEN_RE = re.compile(
    r"[A-Za-z]+(?:[A-Za-z0-9]*)(?:[-./][A-Za-z0-9]+)*|\d+(?:\.\d+)?|[가-힣]{2,}",
    re.UNICODE,
)
SPARSE_STOPWORDS = {
    "관련", "기준", "문서", "규정", "주요", "어떤", "무엇", "알려줘", "찾아줘",
    "에서", "으로", "대한", "the", "and", "for", "with", "what", "which",
}
KOREAN_QUERY_SUFFIXES = (
    "으로부터", "에서는", "에게서", "이라고", "라는", "하는가", "되는가",
    "해야", "하고", "에서", "으로", "에게", "에는", "은", "는", "이", "가",
    "을", "를", "의", "와", "과", "에",
)
DOCUMENT_ROUTE_MAX_DOCS = 4
DOCUMENT_ROUTE_MAX_DOCS_BROAD = 6
DOCUMENT_ROUTE_STAGE2_FETCH = 72
DOCUMENT_ROUTE_BOOSTS = (0.20, 0.13, 0.08, 0.04, 0.02, 0.01)
SCOPED_SPARSE_MAX_ROWS = 6000


def _priority_rule_file_names(signals: QuerySignals, query: str = "") -> list[str]:
    """Return exact class-rule filenames worth querying alongside dense hits.

    The global dense search can miss a short document code when the question is
    phrased by topic.  Exact metadata routing is cheap and keeps interactive
    retrieval fast without scanning the full BM25 corpus.
    """
    names: list[str] = []
    for hint in signals.rule_doc_hints:
        code = str(hint or "").strip()
        if DNV_RULE_CODE_RE.fullmatch(code):
            names.append(f"{code.upper()}.pdf")

    autonomous_hint = any(
        str(hint or "").lower() == "autonomous" for hint in signals.rule_doc_hints
    )
    if (
        signals.wants_rule_lookup
        and str(signals.class_society_hint or "").upper() == "DNV"
        and (autonomous_hint or "mass" in signals.topics)
    ):
        names.append("DNV-CG-0264.pdf")
    part = KR_RULE_PART_RE.search(query or "")
    if part:
        volume = int(part.group(1))
        year = part.group(2) or "2025"
        names.append(f"{volume}편_{year}.pdf")
    elif KR_PART1_CONTEXT_RE.search(query or ""):
        names.append("1편_2025.pdf")
    return list(dict.fromkeys(names))


def _direct_priority_rule_doc_ids(query: str, signals: QuerySignals) -> list[str]:
    """Known corpus identities for strongly named rule documents."""
    out: list[str] = []
    q = query or ""
    part = KR_RULE_PART_RE.search(q)
    unnamed_korean_clause = bool(
        not part
        and not signals.class_society_hint
        and re.search(r"[가-힣]", q)
        and re.search(r"(?:^|\s)\d{3,4}\s*(?:절|조|항)(?:\s|에서|의|$)", q)
    )
    if (part and int(part.group(1)) == 1 and (part.group(2) or "2025") == "2025") or (
        not part and KR_PART1_CONTEXT_RE.search(q)
    ) or unnamed_korean_clause:
        out.append("kr_1_2025")
    if "Notice No.1" in signals.rule_doc_hints:
        out.append("lr_notice_no_1_2025")
    if (
        str(signals.class_society_hint or "").upper() == "ABS"
        and any(str(h).lower() == "autonomous" for h in signals.rule_doc_hints)
    ):
        out.append(
            "abs_abs_rules_requirementsforautonomousandremotecontrolfunctions_v4_1d89b7bb"
        )
    return list(dict.fromkeys(out))


def infer_query_narrow_doc_id(query: str, signals: QuerySignals) -> str | None:
    """Return a single high-confidence document identity stated by the query."""
    direct = _direct_priority_rule_doc_ids(query, signals)
    return direct[0] if len(direct) == 1 else None


def _resolve_priority_rule_doc_ids(
    collection, signals: QuerySignals, query: str = ""
) -> list[str]:
    """Resolve exact rule filenames to Chroma document ids."""
    direct = _direct_priority_rule_doc_ids(query, signals)
    names = _priority_rule_file_names(signals, query)
    if not names:
        return direct
    where = {"file_name": names[0]} if len(names) == 1 else {"file_name": {"$in": names}}
    try:
        raw = collection.get(where=where, include=["metadatas"], limit=100)
    except Exception:
        return direct
    doc_ids: list[str] = list(direct)
    for meta in raw.get("metadatas") or []:
        doc_id = str((meta or {}).get("doc_id") or "").strip()
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)
    return doc_ids


def _hierarchical_text_enabled() -> bool:
    return os.environ.get("MARITIME_TEXT_HIERARCHICAL", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def extract_exact_identifiers(query: str) -> list[str]:
    """Extract terms for which literal retrieval is safer than embeddings."""
    out: list[str] = []
    for match in EXACT_IDENTIFIER_RE.finditer(query or ""):
        value = re.sub(r"\s*([/.-])\s*", r"\1", match.group(0).strip())
        value = re.sub(r"\s+", " ", value)
        if value and value.lower() not in {item.lower() for item in out}:
            out.append(value)
    return out[:4]


def _literal_variants(identifier: str) -> list[str]:
    variants = [identifier]
    if re.match(r"^(?:MEPC|MSC)\s*\d", identifier, re.I):
        variants.extend(
            [
                identifier.replace("/", "-"),
                identifier.replace("-", "/"),
                re.sub(r"\s+", " ", identifier),
            ]
        )
    if re.match(r"^t(?:corr|c\s*[12])$", identifier, re.I):
        variants.extend([identifier.lower(), identifier.upper()])
    return list(dict.fromkeys(v for v in variants if v))[:4]


def _identifier_matches_filename(identifier: str, file_name: str) -> bool:
    """Match document codes while tolerating IMO slash/hyphen filename forms."""
    imo_identifier = re.fullmatch(
        r"\s*(MEPC|MSC)\s*(\d{1,3})(?:\s*[/.-]\s*([A-Z0-9]+))+(?:\s*)",
        identifier or "",
        re.I,
    )
    if imo_identifier:
        ident_parts = re.findall(r"[A-Za-z]+|\d+", identifier or "")
        if not re.match(r"\s*(MEPC|MSC)\b", file_name or "", re.I):
            return False
        file_parts = re.findall(r"[A-Za-z]+|\d+", (file_name or "")[:100])
        return [part.lower() for part in ident_parts] == [
            part.lower() for part in file_parts[: len(ident_parts)]
        ]
    dnv_identifier = re.fullmatch(
        r"DNV-(?:CG|RP|RU|OS|CP|SI)-[A-Z0-9-]+", identifier or "", re.I
    )
    if dnv_identifier:
        prefix = re.match(
            r"\s*(DNV-(?:CG|RP|RU|OS|CP|SI)-[A-Z0-9-]+)", file_name or "", re.I
        )
        return bool(prefix and prefix.group(1).lower() == identifier.lower())
    ident = re.sub(r"[^a-z0-9]+", "", (identifier or "").lower())
    name = re.sub(r"[^a-z0-9]+", "", (file_name or "").lower())
    return bool(ident and len(ident) >= 3 and ident in name)


def _query_exact_identifier_hits(
    collection,
    query: str,
    *,
    where: dict | None,
    limit_per_term: int = 40,
) -> tuple[dict[str, Any], dict[str, float], list[str]]:
    """Use Chroma's existing documents for cheap exact-code recall.

    This is deliberately limited to distinctive identifiers.  It avoids the
    271k-row global BM25 path and does not create or modify any embedding.
    """
    identifiers = extract_exact_identifiers(query)
    out: dict[str, list[list]] = {
        "ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]
    }
    scores: dict[str, float] = {}
    seen: set[str] = set()
    positions: dict[str, int] = {}
    for identifier in identifiers:
        variants = _literal_variants(identifier)
        if re.fullmatch(r"tc\s*orr", identifier, re.I) and re.search(
            r"(?:기호.{0,12}(?:뜻|의미)|무엇을\s*뜻|무슨\s*뜻|정의)",
            query,
            re.I,
        ):
            variants = list(
                dict.fromkeys(
                    variants
                    + [
                        "국부 부식추가 tcorr",
                        "국부 부식추가",
                        "tcorr :",
                    ]
                )
            )
        for variant in variants:
            try:
                kwargs: dict[str, Any] = {
                    "where_document": {"$contains": variant},
                    "limit": limit_per_term,
                    "include": ["metadatas", "documents"],
                }
                if where:
                    kwargs["where"] = where
                raw = collection.get(**kwargs)
            except Exception:
                continue
            for cid, meta, document in zip(
                raw.get("ids") or [],
                raw.get("metadatas") or [],
                raw.get("documents") or [],
            ):
                text = str(document or "")
                exact = variant.lower() in text.lower()
                file_match = _identifier_matches_filename(
                    identifier, str((meta or {}).get("file_name") or "")
                )
                literal_score = 1.8 if file_match else (1.0 if exact else 0.7)
                scores[cid] = max(scores.get(cid, 0.0), literal_score)
                literal_distance = 0.08 if file_match else (0.22 if exact else 0.35)
                if cid in seen:
                    pos = positions[cid]
                    out["distances"][0][pos] = min(
                        float(out["distances"][0][pos]), literal_distance
                    )
                    continue
                seen.add(cid)
                positions[cid] = len(out["ids"][0])
                out["ids"][0].append(cid)
                # Synthetic distance only seeds the candidate pool; final order
                # is recalculated with metadata, document and sparse scores.
                out["distances"][0].append(literal_distance)
                out["metadatas"][0].append(meta or {})
                out["documents"][0].append(text)
    return out, scores, identifiers


def _document_route_candidates(
    *,
    query: str,
    ids: list[str],
    distances: dict[str, float],
    metadatas: dict[str, dict],
    documents: dict[str, str],
    signals: QuerySignals,
    clause_hints: list[str],
    priority_doc_ids: set[str],
    exact_scores: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate chunk evidence into a ranked document shortlist."""
    grouped: dict[str, dict[str, Any]] = {}
    for global_rank, cid in enumerate(ids, 1):
        meta = metadatas.get(cid) or {}
        doc_id = str(meta.get("doc_id") or "").strip()
        if not doc_id:
            continue
        adjusted = adjusted_distance(
            distances[cid],
            query=query,
            document=documents.get(cid, ""),
            meta=meta,
            clause_hints=clause_hints,
            signals=signals,
            priority_doc_ids=priority_doc_ids,
        )
        item = grouped.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "file_name": str(meta.get("file_name") or ""),
                "source": str(meta.get("source") or ""),
                "best_distance": adjusted,
                "raw_best_distance": float(distances[cid]),
                "first_rank": global_rank,
                "hit_count": 0,
                "exact_hit_score": 0.0,
            },
        )
        item["hit_count"] += 1
        item["exact_hit_score"] = max(
            float(item["exact_hit_score"]),
            float((exact_scores or {}).get(cid, 0.0)),
        )
        item["best_distance"] = min(float(item["best_distance"]), adjusted)
        item["raw_best_distance"] = min(
            float(item["raw_best_distance"]), float(distances[cid])
        )
        item["first_rank"] = min(int(item["first_rank"]), global_rank)

    ranked: list[dict[str, Any]] = []
    for item in grouped.values():
        file_overlap = lexical_overlap(
            query,
            f"{item['file_name']} {item['doc_id']}",
            signals.expanded_terms,
        )
        support = min(0.10, math.log1p(int(item["hit_count"])) * 0.025)
        priority = item["doc_id"] in priority_doc_ids
        identifier_file_match = any(
            _identifier_matches_filename(identifier, str(item["file_name"]))
            for identifier in extract_exact_identifiers(query)
        )
        route_score = (
            float(item["best_distance"])
            - support
            - file_overlap * 0.28
            - (0.22 if priority else 0.0)
            - min(0.75, float(item["exact_hit_score"]) * 0.70)
            - (0.45 if identifier_file_match else 0.0)
        )
        item.update(
            {
                "score": route_score,
                "file_overlap": file_overlap,
                "priority": priority,
                "identifier_file_match": identifier_file_match,
            }
        )
        ranked.append(item)
    ranked.sort(key=lambda item: (float(item["score"]), int(item["first_rank"])))
    return ranked


def _sparse_query_terms(query: str, signals: QuerySignals) -> list[tuple[str, float]]:
    weighted: dict[str, float] = {}
    exact = {value.lower() for value in extract_exact_identifiers(query)}
    for raw in SPARSE_TOKEN_RE.findall(query or ""):
        term = raw.lower().strip()
        if len(term) < 2 or term in SPARSE_STOPWORDS:
            continue
        weight = 8.0 if term in exact else (3.2 if re.search(r"\d|[-./]", term) else 1.0)
        if raw.isupper() and len(raw) >= 3:
            weight = max(weight, 2.0)
        elif len(term) >= 5:
            weight = max(weight, 1.25)
        weighted[term] = max(weighted.get(term, 0.0), weight)
        if re.fullmatch(r"[가-힣]{3,}", raw):
            for suffix in KOREAN_QUERY_SUFFIXES:
                if term.endswith(suffix) and len(term) - len(suffix) >= 2:
                    stem = term[: -len(suffix)]
                    weighted[stem] = max(weighted.get(stem, 0.0), weight * 1.1)
                    break
    for raw in signals.expanded_terms[:16]:
        term = str(raw or "").lower().strip()
        if len(term) >= 3:
            weighted[term] = max(weighted.get(term, 0.0), 0.55)
    return list(weighted.items())[:32]


def rank_scoped_sparse_rows(
    query: str,
    signals: QuerySignals,
    ids: list[str],
    metadatas: list[dict],
    documents: list[str],
    *,
    top_k: int = 18,
) -> list[tuple[float, str, dict, str]]:
    """Rank chunks inside routed documents using lightweight exact terms.

    The same section number can occur in several chapters of a rule book.  A
    plain term count therefore overvalues repeated identifiers (for example
    ``902``) and generic words such as "procedure".  Document-local IDF,
    heading matches, and short query phrases make the topic next to the section
    number decisive without requiring a new embedding index.
    """
    terms = _sparse_query_terms(query, signals)
    if not terms:
        return []
    row_texts = [
        f"{(meta or {}).get('file_name', '')} {document or ''}".lower()
        for meta, document in zip(metadatas, documents)
    ]
    row_count = max(1, len(row_texts))
    doc_frequency = {
        term: sum(1 for text in row_texts if term in text)
        for term, _weight in terms
    }

    # Phrase matching keeps one-syllable connectors such as Korean "및" even
    # though they are intentionally excluded from unigram scoring.
    raw_tokens = [
        token.lower()
        for token in re.findall(
            r"[A-Za-z]+(?:[A-Za-z0-9]*)(?:[-./][A-Za-z0-9]+)*|\d+(?:\.\d+)?|[가-힣]+",
            query or "",
        )
    ]
    phrases: list[tuple[str, float]] = []
    phrase_values: set[str] = set()
    for size, weight in ((3, 10.0), (2, 2.5)):
        for start in range(max(0, len(raw_tokens) - size + 1)):
            phrase = " ".join(raw_tokens[start : start + size]).strip()
            if len(phrase) >= 5 and phrase not in phrase_values:
                phrases.append((phrase, weight))
                phrase_values.add(phrase)

    topic_anchors = []
    for match in re.finditer(
        r"\d{3,4}\s*(?:section|clause|절|조)\s*([A-Za-z가-힣][A-Za-z가-힣0-9_-]{1,24})",
        query or "",
        re.IGNORECASE,
    ):
        anchor = match.group(1).lower().strip()
        if anchor not in {"검사", "inspection", "requirements", "요건"}:
            topic_anchors.append(anchor)

    scored: list[tuple[float, str, dict, str]] = []
    for cid, meta, document, text in zip(ids, metadatas, documents, row_texts):
        heading = text[:220]
        score = 0.0
        hits = 0
        for term, weight in terms:
            if term in text:
                hits += 1
                frequency = doc_frequency.get(term, row_count)
                idf = math.log(1.0 + (row_count + 1.0) / (frequency + 1.0))
                effective_weight = weight * min(2.8, 0.75 + 0.38 * idf)
                score += effective_weight
                if " " in term or re.search(r"\d|[-./]", term):
                    score += effective_weight * 0.35
                if term in heading:
                    score += effective_weight * 0.45
        for phrase, phrase_weight in phrases:
            if phrase in text:
                score += phrase_weight * (1.45 if phrase in heading else 1.0)
        for anchor in topic_anchors:
            if anchor in heading:
                score += 12.0
            elif anchor in text:
                score += 4.0
        if hits:
            score += min(0.8, hits * 0.08)
            scored.append((score, cid, meta or {}, document or ""))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def _load_scoped_sparse_rows(
    collection,
    doc_ids: list[str],
    *,
    base_where: dict | None,
) -> tuple[list[str], list[dict], list[str]]:
    if not doc_ids:
        return [], [], []
    ids: list[str] = []
    metadatas: list[dict] = []
    documents: list[str] = []
    per_doc_limit = max(700, SCOPED_SPARSE_MAX_ROWS // max(1, len(doc_ids)))
    for doc_id in doc_ids:
        where = _merge_where(base_where, {"doc_id": doc_id})
        try:
            raw = collection.get(
                where=where,
                limit=per_doc_limit,
                include=["metadatas", "documents"],
            )
        except Exception:
            continue
        ids.extend(raw.get("ids") or [])
        metadatas.extend(raw.get("metadatas") or [])
        documents.extend(raw.get("documents") or [])
    return ids, metadatas, documents


def extract_clause_hints(query: str) -> list[str]:
    hints: list[str] = []
    for groups in CLAUSE_IN_QUERY_RE.finditer(query):
        for g in groups.groups():
            if g and g.isdigit() and len(g) >= 3:
                hints.append(g)
    seen: set[str] = set()
    out: list[str] = []
    for h in hints:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def _meta_clause(meta: dict) -> str:
    return str(meta.get("clause_number") or meta.get("article_number") or "").strip()


def lexical_overlap(query: str, document: str, extra_terms: list[str] | None = None) -> float:
    q_tokens = {t for t in TOKEN_RE.findall(query.lower()) if len(t) > 1}
    if extra_terms:
        for term in extra_terms:
            q_tokens.update(t for t in TOKEN_RE.findall(term.lower()) if len(t) > 1)
    if not q_tokens:
        return 0.0
    d_tokens = {t for t in TOKEN_RE.findall(document.lower()) if len(t) > 1}
    return len(q_tokens & d_tokens) / len(q_tokens)


def _file_name(meta: dict) -> str:
    return str(meta.get("file_name") or meta.get("doc_id") or "").lower()


def _doc_type(meta: dict) -> str:
    dt = str(meta.get("doc_type") or "").strip()
    if dt:
        return dt
    return classify_imo_filename(str(meta.get("file_name") or ""))


def _metadata_boosts(
    *,
    meta: dict,
    document: str,
    signals: QuerySignals,
    priority_doc_ids: set[str],
    query: str = "",
) -> tuple[float, float]:
    boost = 0.0
    penalty = 0.0
    fname = _file_name(meta)
    doc_id = str(meta.get("doc_id") or "")
    doc_head = (document or "")[:800].lower()
    combined = f"{fname} {doc_head}"
    doc_type = _doc_type(meta)

    if doc_id and doc_id in priority_doc_ids:
        boost += PRIORITY_DOC_BOOST

    tier = tier_for_query(
        doc_type,
        wants_summary=signals.wants_summary,
        wants_outcome=signals.wants_outcome,
        wants_agenda=signals.wants_agenda,
    )
    if tier >= 0:
        boost += tier
    else:
        penalty += abs(tier)

    for prefix in session_file_prefixes(signals):
        if prefix.replace("/", "-") in fname.replace("/", "-") or prefix in fname:
            boost += SESSION_FILE_BOOST
            break

    for prefix in topic_agenda_prefixes(signals):
        if prefix in fname:
            boost += TOPIC_PREFIX_BOOST
            break

    if doc_type == "subcommittee_report" and (signals.wants_summary or signals.wants_outcome):
        penalty += SUBCOMM_PENALTY

    if signals.wants_agenda:
        if doc_type == "agenda":
            boost += 0.12

    if signals.wants_rule_lookup:
        ql = (query or "").lower()
        if "dnv-cg-0264" in fname or "cg-0264" in fname or "cg 0264" in fname:
            boost += RULE_FILENAME_BOOST + 0.12
        if "notice no.1" in fname or "notice no. 1" in fname:
            boost += RULE_FILENAME_BOOST
        if any(k in fname for k in ("autonomous", "remotely-operated", "remotely operated", "smart-vessel", "smart vessel")):
            boost += 0.14
        if any(k in combined for k in ("autonomous", "remotely operated", "smart vessel", "notation")):
            boost += 0.08
        if re.search(r"rp-c\d", fname) and "cg-0264" not in fname:
            if any(k in ql for k in ("smart", "autonomous", "자율", "vessel", "mass")):
                penalty += 0.10
        if DOC_CODE_CROSSREF_RE.search(doc_head):
            penalty += 0.16
        society = str(signals.class_society_hint or "").upper()
        chunk_source = str(meta.get("source") or "").upper()
        if society and chunk_source in CLASS_RULE_SOURCES:
            if chunk_source == society:
                boost += 0.20
            else:
                penalty += 0.22

    for hint in signals.rule_doc_hints:
        if hint.lower() in combined:
            boost += EXPANDED_TERM_BOOST

    for term in signals.expanded_terms:
        t = term.lower()
        if len(t) >= 4 and t in combined:
            boost += EXPANDED_TERM_BOOST

    if "mass" in signals.topics and "mass" in combined:
        boost += EXPANDED_TERM_BOOST
    if "ghg" in signals.topics and "ghg" in combined:
        boost += EXPANDED_TERM_BOOST
    if "alt_fuel" in signals.topics:
        if any(k in combined for k in ("low-flashpoint", "low flashpoint", "section 15", "alternative fuel", "igf")):
            boost += TOPIC_PREFIX_BOOST
    if "igc" in signals.topics and "igc" in combined:
        boost += EXPANDED_TERM_BOOST

    if signals.meeting_outcome_question:
        from meeting_outcome_retrieval import is_comparison_question, meeting_outcome_metadata_adjustment

        mo_boost, mo_penalty = meeting_outcome_metadata_adjustment(
            meta=meta,
            document=document,
            signals=signals,
            question=query,
            is_comparison=is_comparison_question(query),
        )
        boost += mo_boost
        penalty += mo_penalty

    return boost, penalty


def adjusted_distance(
    distance: float,
    *,
    query: str,
    document: str,
    meta: dict,
    clause_hints: list[str],
    signals: QuerySignals | None = None,
    priority_doc_ids: set[str] | None = None,
) -> float:
    score = float(distance)
    meta_clause = _meta_clause(meta)
    doc_head = (document or "")[:400]
    sig = signals or analyze_query(query)
    prio = priority_doc_ids or set()

    for hint in clause_hints:
        if meta_clause == hint:
            score -= CLAUSE_EXACT_BOOST
        elif hint in doc_head and is_article_clause_number(hint):
            score -= CLAUSE_IN_TEXT_BOOST
        if f"{hint}." in doc_head or f"{hint}절" in doc_head:
            score -= REFERENCE_LINE_BOOST

    meta_boost, meta_penalty = _metadata_boosts(
        meta=meta, document=document, signals=sig, priority_doc_ids=prio, query=query
    )
    score -= meta_boost
    score += meta_penalty
    score -= LEXICAL_BOOST_SCALE * lexical_overlap(query, document, sig.expanded_terms)
    score -= _definition_clause_boost(query, document)
    return score


def _definition_clause_boost(query: str, document: str) -> float:
    """Prefer defining clauses over later formula references for symbol asks."""
    q = str(query or "")
    if not re.search(
        r"(?:기호.{0,12}(?:뜻|의미)|무엇을\s*뜻|무슨\s*뜻|정의|what.{0,12}mean)",
        q,
        re.I,
    ):
        return 0.0
    body = str(document or "")
    boost = 0.0
    if re.search(
        r"(?:정의(?:된|한다|는)|이라\s*함은|을\s*말한다|means|is\s+defined\s+as)",
        body,
        re.I,
    ):
        boost += 0.10
    if re.search(r"\btc\s*orr\b|\bt_corr\b", q, re.I):
        if re.search(r"국부\s*부식추가.{0,40}tc\s*orr|tc\s*orr.{0,40}국부\s*부식추가", body, re.I):
            boost += 0.30
        elif re.search(r"tc\s*orr\s*[:=：]", body, re.I):
            boost += 0.20
    return min(0.40, boost)


def _merge_where(*clauses: dict | None) -> dict | None:
    parts = [c for c in clauses if c]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}


def _meta_matches_where(meta: dict, where: dict | None) -> bool:
    if not where:
        return True
    meta = meta or {}
    if "$and" in where:
        return all(_meta_matches_where(meta, clause) for clause in where["$and"])
    if "$or" in where:
        return any(_meta_matches_where(meta, clause) for clause in where["$or"])
    for key, expected in where.items():
        actual = meta.get(key)
        if isinstance(expected, dict):
            for op, val in expected.items():
                if op == "$in":
                    if actual not in val:
                        return False
                elif op == "$eq":
                    if actual != val:
                        return False
                elif op == "$ne":
                    if actual == val:
                        return False
                else:
                    return False
        elif actual != expected:
            return False
    return True


def _filter_chroma_raw(raw: dict, where: dict | None) -> dict:
    if not where or not raw.get("ids") or not raw["ids"][0]:
        return raw
    out: dict[str, list[list]] = {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}
    for cid, dist, meta, doc in zip(
        raw["ids"][0],
        raw["distances"][0],
        raw["metadatas"][0],
        raw["documents"][0],
    ):
        if _meta_matches_where(meta or {}, where):
            out["ids"][0].append(cid)
            out["distances"][0].append(dist)
            out["metadatas"][0].append(meta)
            out["documents"][0].append(doc)
    return out


def safe_chroma_query(
    collection,
    *,
    query_embeddings: list[list[float]],
    n_results: int,
    where: dict | None = None,
) -> dict[str, Any]:
    """Chroma query with retries when the vector index returns stale/missing ids."""
    attempts: list[tuple[int, dict | None]] = [
        (n_results, where),
        (min(n_results, 40), where),
        (min(n_results, 40), None),
    ]
    last_exc: Exception | None = None
    for n, clause in attempts:
        try:
            kwargs: dict[str, Any] = {
                "query_embeddings": query_embeddings,
                "n_results": max(1, n),
            }
            if clause:
                kwargs["where"] = clause
            raw = collection.query(**kwargs)
            if clause is None and where is not None:
                return _filter_chroma_raw(raw, where)
            return raw
        except Exception as exc:
            last_exc = exc
            if "error finding id" not in str(exc).lower():
                raise
    if last_exc is not None:
        raise last_exc
    return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}


def query_with_hybrid_ranking(
    collection,
    query: str,
    query_vector: list[float],
    *,
    top_k: int = 5,
    fetch_k: int | None = None,
    source: str | None = None,
    doc_id: str | None = None,
    timing=None,
) -> dict[str, Any]:
    """Hierarchical text retrieval over the existing Chroma embeddings.

    Stage 1 retrieves broad chunks and aggregates them to documents. Stage 2
    searches only the routed documents and optionally injects exact identifier
    hits plus a lightweight document-scoped sparse rerank. No index rebuild is
    performed.
    """
    clause_hints = extract_clause_hints(query)
    signals = analyze_query(query)
    priority_ids = priority_doc_ids_for_signals(signals)
    if doc_id is None:
        for priority_id in _resolve_priority_rule_doc_ids(collection, signals, query):
            if priority_id not in priority_ids:
                priority_ids.append(priority_id)
    priority_set = set(priority_ids)
    n_fetch = fetch_k or max(top_k * 15, 80)
    base_where = _merge_where(
        {"source": source.upper()} if source else None,
        {"doc_id": doc_id} if doc_id else None,
    )

    merged_ids: list[str] = []
    merged_dist: dict[str, float] = {}
    merged_meta: dict[str, dict] = {}
    merged_doc: dict[str, str] = {}
    exact_scores: dict[str, float] = {}
    scoped_sparse_ratios: dict[str, float] = {}

    def absorb(raw: dict) -> None:
        if not raw.get("ids") or not raw["ids"][0]:
            return
        for cid, dist, meta, doc in zip(
            raw["ids"][0],
            raw["distances"][0],
            raw["metadatas"][0],
            raw["documents"][0],
        ):
            if cid in merged_dist:
                merged_dist[cid] = min(merged_dist[cid], float(dist))
            else:
                merged_ids.append(cid)
                merged_dist[cid] = float(dist)
                merged_meta[cid] = meta or {}
                merged_doc[cid] = doc or ""

    if timing is not None and hasattr(timing, "mark"):
        timing.mark("t_vector_search_start")

    raw_vector = safe_chroma_query(
        collection,
        query_embeddings=[query_vector],
        n_results=min(n_fetch, 150),
        where=base_where,
    )
    absorb(raw_vector)

    hierarchy_enabled = _hierarchical_text_enabled()
    if hierarchy_enabled:
        exact_raw, exact_scores, exact_identifiers = _query_exact_identifier_hits(
            collection,
            query,
            where=base_where,
        )
        absorb(exact_raw)
    else:
        exact_identifiers = []

    if timing is not None and hasattr(timing, "mark"):
        timing.mark("t_vector_search_end")
        timing.mark("t_metadata_filter_start")

    if priority_ids:
        batch = priority_ids[:20]
        try:
            routed = safe_chroma_query(
                collection,
                query_embeddings=[query_vector],
                n_results=min(20, n_fetch),
                where=_merge_where(base_where, {"doc_id": {"$in": batch}}),
            )
            absorb(routed)
        except Exception:
            for pid in batch[:8]:
                try:
                    one = safe_chroma_query(
                        collection,
                        query_embeddings=[query_vector],
                        n_results=8,
                        where=_merge_where(base_where, {"doc_id": pid}),
                    )
                    absorb(one)
                except Exception:
                    pass

    if clause_hints:
        for hint in clause_hints[:3]:
            try:
                clause_where = {
                    "$or": [
                        {"clause_number": hint},
                        {"article_number": hint},
                    ]
                }
                filtered = safe_chroma_query(
                    collection,
                    query_embeddings=[query_vector],
                    n_results=min(15, n_fetch),
                    where=_merge_where(base_where, clause_where),
                )
                absorb(filtered)
            except Exception:
                pass

    document_route: dict[str, Any] = {
        "enabled": False,
        "selected_doc_ids": [],
        "candidates": [],
        "confidence": 0.0,
        "exact_identifiers": exact_identifiers,
        "scoped_sparse_used": False,
    }
    selected_doc_ids: list[str] = []
    if hierarchy_enabled and doc_id and (clause_hints or signals.wants_rule_lookup):
        sparse_ids, sparse_metas, sparse_docs = _load_scoped_sparse_rows(
            collection,
            [doc_id],
            base_where=base_where,
        )
        sparse_ranked = rank_scoped_sparse_rows(
            query,
            signals,
            sparse_ids,
            sparse_metas,
            sparse_docs,
            top_k=max(24, top_k * 4),
        )
        best_sparse_score = max((item[0] for item in sparse_ranked), default=1.0)
        doc_best = min(merged_dist.values(), default=0.58)
        for sparse_score, cid, meta, document in sparse_ranked:
            synthetic_distance = max(
                0.0,
                doc_best + 0.05 - min(0.28, sparse_score * 0.03),
            )
            scoped_sparse_ratios[cid] = max(
                scoped_sparse_ratios.get(cid, 0.0),
                sparse_score / max(best_sparse_score, 1e-9),
            )
            absorb(
                {
                    "ids": [[cid]],
                    "distances": [[synthetic_distance]],
                    "metadatas": [[meta]],
                    "documents": [[document]],
                }
            )
        document_route.update(
            {
                "enabled": True,
                "selected_doc_ids": [doc_id],
                "confidence": 1.0,
                "scoped_sparse_used": bool(sparse_ranked),
            }
        )
    if hierarchy_enabled and not doc_id and merged_ids:
        doc_candidates = _document_route_candidates(
            query=query,
            ids=merged_ids,
            distances=merged_dist,
            metadatas=merged_meta,
            documents=merged_doc,
            signals=signals,
            clause_hints=clause_hints,
            priority_doc_ids=priority_set,
            exact_scores=exact_scores,
        )
        max_docs = (
            DOCUMENT_ROUTE_MAX_DOCS_BROAD
            if signals.wants_summary or signals.wants_outcome or signals.meeting_outcome_question
            else DOCUMENT_ROUTE_MAX_DOCS
        )
        selected_doc_ids = [str(item["doc_id"]) for item in doc_candidates[:max_docs]]
        if selected_doc_ids:
            stage2 = safe_chroma_query(
                collection,
                query_embeddings=[query_vector],
                n_results=min(
                    max(DOCUMENT_ROUTE_STAGE2_FETCH, top_k * 10),
                    120,
                ),
                where=_merge_where(
                    base_where,
                    {"doc_id": {"$in": selected_doc_ids}},
                ),
            )
            absorb(stage2)

            # Only pay the scoped lexical cost for exact-code/clause or rule
            # questions, where dense embeddings are known to miss literal terms.
            use_scoped_sparse = bool(
                exact_identifiers or clause_hints or signals.wants_rule_lookup
            )
            if use_scoped_sparse:
                sparse_ids, sparse_metas, sparse_docs = _load_scoped_sparse_rows(
                    collection,
                    selected_doc_ids,
                    base_where=base_where,
                )
                sparse_ranked = rank_scoped_sparse_rows(
                    query,
                    signals,
                    sparse_ids,
                    sparse_metas,
                    sparse_docs,
                    top_k=max(18, top_k * 3),
                )
                best_sparse_score = max((item[0] for item in sparse_ranked), default=1.0)
                best_by_doc = {
                    str(item["doc_id"]): float(item["raw_best_distance"])
                    for item in doc_candidates
                }
                for sparse_score, cid, meta, document in sparse_ranked:
                    routed_doc = str((meta or {}).get("doc_id") or "")
                    synthetic_distance = max(
                        0.0,
                        best_by_doc.get(routed_doc, 0.58)
                        + 0.05
                        - min(0.20, sparse_score * 0.025),
                    )
                    scoped_sparse_ratios[cid] = max(
                        scoped_sparse_ratios.get(cid, 0.0),
                        sparse_score / max(best_sparse_score, 1e-9),
                    )
                    # A sparse-exact hit may already be present in the dense
                    # pool with a poor distance.  Re-absorb it so ``min`` can
                    # lower that distance; otherwise the lexical signal is
                    # recorded but cannot overcome the stale dense score.
                    absorb(
                        {
                            "ids": [[cid]],
                            "distances": [[synthetic_distance]],
                            "metadatas": [[meta]],
                            "documents": [[document]],
                        }
                    )
                document_route["scoped_sparse_used"] = bool(sparse_ranked)

        margin = (
            float(doc_candidates[1]["score"]) - float(doc_candidates[0]["score"])
            if len(doc_candidates) > 1
            else 0.25
        )
        first_priority = bool(doc_candidates and doc_candidates[0].get("priority"))
        confidence = min(1.0, 0.42 + max(0.0, min(0.30, margin)) + (0.22 if first_priority else 0.0))
        document_route.update(
            {
                "enabled": True,
                "selected_doc_ids": selected_doc_ids,
                "confidence": round(confidence, 3),
                "candidates": [
                    {
                        "doc_id": item["doc_id"],
                        "file_name": item["file_name"],
                        "source": item["source"],
                        "score": round(float(item["score"]), 4),
                        "hit_count": int(item["hit_count"]),
                        "priority": bool(item["priority"]),
                        "exact_hit_score": round(float(item["exact_hit_score"]), 3),
                        "identifier_file_match": bool(item["identifier_file_match"]),
                    }
                    for item in doc_candidates[:max_docs]
                ],
            }
        )

    if timing is not None and hasattr(timing, "mark"):
        timing.mark("t_metadata_filter_end")
        timing.mark("t_rerank_start")

    doc_route_rank = {value: rank for rank, value in enumerate(selected_doc_ids)}

    def final_distance(cid: str) -> float:
        score = adjusted_distance(
            merged_dist[cid],
            query=query,
            document=merged_doc[cid],
            meta=merged_meta[cid],
            clause_hints=clause_hints,
            signals=signals,
            priority_doc_ids=priority_set,
        )
        routed_doc = str((merged_meta.get(cid) or {}).get("doc_id") or "")
        rank = doc_route_rank.get(routed_doc)
        if rank is not None and rank < len(DOCUMENT_ROUTE_BOOSTS):
            score -= DOCUMENT_ROUTE_BOOSTS[rank]
        score -= min(0.60, exact_scores.get(cid, 0.0) * 0.38)
        # IDF and phrase matches can make several lexical scores exceed a
        # fixed cap.  A within-document ratio preserves the decisive gap
        # between the strongest clause and a generic introduction.
        score -= scoped_sparse_ratios.get(cid, 0.0) * 0.32
        return score

    ranked = sorted(merged_ids, key=final_distance)[:top_k]

    if timing is not None and hasattr(timing, "mark"):
        timing.mark("t_rerank_end")

    return {
        "ids": [ranked],
        "distances": [[merged_dist[cid] for cid in ranked]],
        "metadatas": [[merged_meta[cid] for cid in ranked]],
        "documents": [[merged_doc[cid] for cid in ranked]],
        "clause_hints": clause_hints,
        "query_signals": signals,
        "priority_doc_ids": priority_ids,
        "document_route": document_route,
        "final_scores": [[final_distance(cid) for cid in ranked]],
    }


def _enrich_query_standard(query: str) -> str:
    """Base query enrichment without meeting-outcome branch."""
    hints = extract_clause_hints(query)
    signals = analyze_query(query)
    parts: list[str] = []
    if hints:
        parts.extend(f"{h}절 {h}." for h in hints[:2])
    if signals.expanded_terms:
        parts.extend(signals.expanded_terms[:12])
    if signals.wants_summary or signals.wants_outcome:
        parts.extend(["outcome", "executive summary", "report", "resolution", "decision"])
    if not parts:
        return query
    return f"{' '.join(parts)} {query}".strip()


def enrich_query_for_embedding(query: str, model_name: str) -> str:
    """Prepend clause hints and cross-lingual/session terms for E5."""
    signals = analyze_query(query)
    if signals.meeting_outcome_question:
        from meeting_outcome_retrieval import enrich_meeting_outcome_query

        return enrich_meeting_outcome_query(query, model_name)
    from table_retrieval import enrich_table_query_for_embedding, is_table_question

    if is_table_question(query):
        return enrich_table_query_for_embedding(_enrich_query_standard(query))
    return _enrich_query_standard(query)
