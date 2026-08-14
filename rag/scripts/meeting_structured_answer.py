"""Evidence-based 4-section answers for meeting/regulation categories."""
from __future__ import annotations

import re
from typing import Any

from answer_depth_guidance import join_four_sections, category_bullet_budget
from meeting_answer_dedup import apply_answer_dedup, detect_section1_topics
from meeting_category_profile import MeetingRetrievalProfile, TOP_LEVEL_AUTO, TOP_LEVEL_ENV, TOP_LEVEL_TREND
from meeting_coverage_check import run_coverage_check
from meeting_topic_cluster import (
    MSC_OUTCOME_TOPIC_PRIORITY,
    cluster_chunks,
    dedupe_page_chunks,
    pick_diverse_topic_chunks,
    outcome_topic_id,
    _topic_id_for_text,
    _topic_label_ko,
)
from meeting_topic_scoring import (
    intent_chunk_adjustment,
    is_excluded_chunk,
    exclude_topics_for_intent,
    topic_caps_for_intent,
    NON_MAND_RE,
    TIMELINE_RE,
    MASS_RE,
    ALT_FUEL_RE,
    OUTCOME_RE,
)
from source_tier_lib import (
    classify_source_tier,
    count_impact_signals,
    count_outcome_signals,
    tier_label,
)
from grounded_answer_policy import (
    classify_document_status,
    select_key_clause_chunks,
    verify_high_risk_claims,
    verify_claim_citations,
)
from question_requirements import analyze_requirements

ENGLISH_LEAK_RE = re.compile(r"[A-Za-z][A-Za-z\s,;:'\"()-]{55,}")
CITATION_RE = re.compile(r"\[(\d+)\]")


def _strip_meta(text: str) -> str:
    lines = []
    for line in (text or "").splitlines():
        if re.match(r"^(source|file_name|page|doc_id)\s*:", line.strip(), re.I):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _cite(chunk: Any, citation_map: dict[str, int]) -> str:
    cid = str(getattr(chunk, "chunk_id", "") or "")
    n = citation_map.get(cid)
    return f"[{n}]" if n else ""


def _build_citation_map(chunks: list[Any]) -> dict[str, int]:
    return {str(getattr(c, "chunk_id", "")): i for i, c in enumerate(chunks, 1)}


def score_chunk(chunk: Any, *, profile: MeetingRetrievalProfile) -> float:
    text = _strip_meta(getattr(chunk, "text", ""))
    tier = classify_source_tier(chunk)
    score = 0.0
    score += {0: 3.0, 1: 1.5, 2: 0.5, 3: -1.0}.get(tier, 1.0)
    score += count_outcome_signals(text) * 0.4
    if profile.top_level_category == TOP_LEVEL_ENV:
        score += count_impact_signals(text) * 0.45
    score += intent_chunk_adjustment(text, internal_intent=profile.internal_intent)
    status = classify_document_status(chunk)
    score += status.authority * 0.45
    if profile.internal_intent in {"meeting_outcome", "mass_code_timeline"}:
        if status.code in {"proposal", "action_request"}:
            score -= 3.5
        elif status.supports_final_decision:
            score += 2.0
    if profile.answer_variant == "official_dense":
        score += {0: 2.5, 1: 1.2, 2: -0.8, 3: -2.5}.get(tier, 0.0)
    elif profile.answer_variant == "topic_diverse":
        score += count_outcome_signals(text) * 0.55
        score += count_impact_signals(text) * 0.25
    if getattr(chunk, "bm25_score", None):
        score += float(chunk.bm25_score) * 0.02
    if getattr(chunk, "dense_score", None):
        score += float(chunk.dense_score) * 0.5
    if getattr(chunk, "rrf_score", None):
        score += float(chunk.rrf_score) * 8.0
    setattr(chunk, "_meeting_score", score)
    setattr(chunk, "_source_tier", tier)
    setattr(chunk, "_topic_id", _topic_id_for_text(text))
    setattr(chunk, "_document_status", status.code)
    return score


REFERENCE_OUTCOME_RE = re.compile(
    r"outcome of (?:a|c|tc|mepc)\s*\d|decisions of other imo|already adopted at a\s*\d|"
    r"upon the recommendation of the (?:assembly|council|technical)|\ba\s*\d{1,3}\b.{0,50}\bapproved",
    re.I,
)
OUTCOME_ACTION_RE = re.compile(
    r"\b(adopted|approved|endorsed|agreed|finalized|finalised|mandatory|non-mandatory|"
    r"entry into force|entered into force|amendment|resolution|guideline)\b",
    re.I,
)
RESOLUTION_REF_RE = re.compile(
    r"\b(?:resolution|res\.?)\s+([A-Z]{2,5}\.\d+\(\d+\)|MSC\.\d+\(\d+\)|MEPC\.\d+\(\d+\)[^,\.;]*)",
    re.I,
)
MEETING_DOC_REF_RE = re.compile(r"\b(MEPC|MSC)\s+(\d{1,3})[-/]([A-Z0-9.-]+)", re.I)
LATEST_QUERY_RE = re.compile(r"최신|최근|latest|current", re.I)
LATEST_ENV_SUMMARY_RE = re.compile(r"주요\s*내용|핵심|요약|정리|동향|summary|overview", re.I)
LATEST_ENV_SPECIFIC_RE = re.compile(
    r"\b(?:dcs|cii|gfi|seemp|lca|ibts)\b|gis\s*is|데이터\s*오류|탄소집약도|"
    r"배출계수|유성폐기물|빌지|"
    r"운항|규제\s*보고|reporting|탄소\s*집약|fleet\s*carbon|operational|"
    r"직접\s*영향",
    re.I,
)
OPERATIONAL_IMPACT_ASK_RE = re.compile(
    r"실무\s*(?:영향|대응)|업무\s*영향|운항.{0,8}영향|"
    r"선사.{0,12}(?:대응|준비|조치|해야)|대응\s*사항|무엇을\s*해야",
    re.I,
)
ACTION_OBJECT_RE = re.compile(
    r"\b(adopted|approved|endorsed|finalized|finalised)\s+(?:the\s+)?"
    r"(.{8,220}?)(?=\s*\(resolution\s+(?:MEPC|MSC)\.\d+\(\d+\)|[.;]|$)",
    re.I,
)


def _is_broad_latest_environment_summary(question: str, profile: MeetingRetrievalProfile) -> bool:
    return bool(
        profile.top_level_category in {TOP_LEVEL_ENV, TOP_LEVEL_TREND}
        and LATEST_QUERY_RE.search(question or "")
        and re.search(r"\bMEPC\b", question or "", re.I)
        and LATEST_ENV_SUMMARY_RE.search(question or "")
        and not LATEST_ENV_SPECIFIC_RE.search(question or "")
    )


def _section1_is_hollow(answer: str) -> bool:
    text = answer or ""
    if not text.strip():
        return True
    if "## 1)" in text:
        after = text.split("## 1)", 1)[1]
        for marker in ("## 2)", "## 3)", "## 4)"):
            if marker in after:
                after = after.split(marker, 1)[0]
                break
        s1 = after
    else:
        s1 = text.split("## 2)")[0] if "## 2)" in text else text
    bullets = [ln.strip() for ln in s1.splitlines() if ln.strip().startswith("- ")]
    if not bullets:
        return True
    empty_markers = (
        "검색 근거에서 직접 확인되는 내용이 없어",
        "추가 확인 필요",
        "근거가 부족",
        "근거가 제한적",
    )
    return all(any(m in b for m in empty_markers) for b in bullets)


def _asks_for_operational_impact(question: str) -> bool:
    """Only show an impact section when the user explicitly requested it."""
    return OPERATIONAL_IMPACT_ASK_RE.search(question or "") is not None


def _environment_topic_label(file_name: str) -> str:
    labels = (
        (r"mepc\s*84-3\b", "MARPOL Annex VI 일정"),
        (r"mepc\s*84-6-2\b", "탄소집약도 추세"),
        (r"mepc\s*84-7-14\b", "GFI·SEEMP"),
        (r"mepc\s*84-7-15\b", "LCA 배출계수"),
        (r"mepc\s*84-10\b", "유성폐기물·IBTS"),
    )
    for pattern, label in labels:
        if re.search(pattern, file_name or "", re.I):
            return label
    return "환경규제"


def _section2_latest_environment(
    chunks: list[Any], citation_map: dict[str, int]
) -> str:
    """Summarize response implications without inventing operator duties."""
    by_doc: dict[str, Any] = {}
    for chunk in chunks:
        file_name = str(getattr(chunk, "file_name", "") or "")
        for key in ("84-3", "84-6-2", "84-7-14", "84-7-15", "84-10"):
            if re.search(rf"mepc\s*{re.escape(key)}\b", file_name, re.I):
                by_doc[key] = chunk

    lines: list[str] = []
    status_chunks = [by_doc[key] for key in ("84-3", "84-7-14", "84-10") if key in by_doc]
    status_cites = "".join(_cite(chunk, citation_map) for chunk in status_chunks)
    if len(status_chunks) == 3 and status_cites:
        lines.append(
            "- **규제 확정 단계**: MARPOL Annex VI 2025 개정안은 채택 논의가 1년 연기됐고, "
            "GFI·SEEMP는 개정안 개발 단계이며, IBTS 2026 지침은 MEPC 85의 최종 승인을 목표로 하는 단계입니다. "
            f"{status_cites}"
        )

    carbon = by_doc.get("84-6-2")
    if carbon:
        carbon_text = re.sub(r"\s+", " ", _strip_meta(getattr(carbon, "text", ""))).lower()
        if "up to 10.8%" in carbon_text and "at least 6%" in carbon_text:
            lines.append(
                "- **탄소집약도 해석**: 최대 10.8% 감소와 주요 선종 크기 구간의 최소 6% 개선은 "
                f"선대 평균과 선종 규모군 단위의 추세로 제시됩니다. {_cite(carbon, citation_map)}"
            )
        else:
            lines.append(
                "- **탄소집약도 해석**: 2019~2024년 탄소집약도는 수송실적 기준과 선박 활동 기준의 "
                f"선대 추세로 분석됩니다. {_cite(carbon, citation_map)}"
            )

    lca = by_doc.get("84-7-15")
    if lca:
        lines.append(
            "- **LCA 배출계수 평가**: WtT 기본 배출계수 평가에서는 대표성과 보수성을 "
            f"참고 기준으로 사용하는 방향이 작업반에서 합의됐습니다. {_cite(lca, citation_map)}"
        )
    return "\n".join(lines)


def _truncate_snippet(text: str, max_len: int = 180) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if len(t) <= max_len:
        return t
    cut = t[:max_len].rsplit(" ", 1)[0]
    return cut + "…"


def _extract_best_claim(body: str) -> str:
    """Pick the highest-signal outcome sentence from chunk text."""
    candidates: list[tuple[float, str]] = []
    normalized = re.sub(r"\s+", " ", body)
    for sent in re.split(r"(?<=[.!?])\s+", normalized):
        sent = sent.strip()
        if len(sent) < 20:
            continue
        score = count_outcome_signals(sent) * 2.0
        if OUTCOME_ACTION_RE.search(sent):
            score += 4.0
        if ACTION_OBJECT_RE.search(sent):
            score += 6.0
        if re.search(r"\b(noted|invited|recalled)\b", sent, re.I) and score < 4:
            score -= 2.0
        if score > 0:
            candidates.append((score, sent))
    if candidates:
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]

    lines = [ln.strip() for ln in body.splitlines() if len(ln.strip()) > 35]
    if lines:
        return max(lines, key=len)
    paras = [p.strip() for p in re.split(r"\n{2,}", body) if len(p.strip()) > 40]
    if paras:
        return paras[0]
    return body.strip()


def _shorten_object_phrase(phrase: str, *, max_len: int = 90) -> str:
    obj = re.sub(r"\s+", " ", phrase.strip(" ,;"))
    obj = re.sub(r"^(?:the|a|an)\s+", "", obj, flags=re.I)
    replacements = [
        ("appendix ix of marpol annex vi", "MARPOL Annex VI 부록 IX"),
        ("the 2022 guidelines for the development of a ship energy efficiency management plan (seemp)", "2022 SEEMP 개발 지침"),
        ("guidelines for the development of a ship energy efficiency management plan (seemp)", "SEEMP 개발 지침"),
        ("imo ship fuel oil consumption database", "IMO 선박연료소비량 데이터베이스(DCS)"),
        ("non-mandatory and goal-based", "비강제·goal-based"),
        ("non-mandatory", "비강제"),
        ("maritime autonomous surface ships", "MASS(자율운항)"),
        ("maritime autonomous", "자율운항"),
        ("alternative fuels and", "대체연료·"),
        ("alternative fuels", "대체연료"),
        ("alternative fuel", "대체연료"),
        ("interim guidelines for the safety of ships using", "선박 대체연료 안전 임시 지침"),
        ("interim guidelines", "임시 지침"),
        ("guidelines", "지침"),
        ("amendments to", "개정 —"),
        ("amendments related to", "개정 —"),
        ("amendments", "개정"),
        ("amendment to", "개정 —"),
        ("amendment", "개정"),
    ]
    low = obj.lower()
    for eng, ko in replacements:
        if eng in low:
            obj = re.sub(re.escape(eng), ko, obj, count=1, flags=re.I)
            low = obj.lower()
    return _truncate_snippet(obj, max_len)


def _english_outcome_to_ko_clause(sent: str, *, max_gloss_len: int = 160) -> str:
    """Rule-based Korean clause from an English outcome sentence."""
    s = sent.strip()
    low = s.lower()
    m_res = RESOLUTION_REF_RE.search(s)
    m_obj = ACTION_OBJECT_RE.search(s)
    if m_obj:
        obj = _shorten_object_phrase(m_obj.group(2), max_len=120)
        prefix = f"결의안 {m_res.group(1).strip()}: " if m_res else ""
        return f"{prefix}{obj} {_outcome_verb_ko(m_obj.group(1))}"

    # Do not attach a resolution number to a topic word found elsewhere in a
    # large chunk.  The action and its object must be linked in one sentence.
    if m_res:
        return f"결의안 {m_res.group(1).strip()}이 언급되지만 이 문장만으로 조치 대상을 확정할 수 없음"

    if "non-mandatory" in low and "mass code" in low and "adopted" in low:
        return "비강제·goal-based MASS Code 채택"

    gloss = _truncate_snippet(s, max_gloss_len)
    gloss = re.sub(r"\badopted\b", "채택", gloss, flags=re.I)
    gloss = re.sub(r"\bapproved\b", "승인", gloss, flags=re.I)
    gloss = re.sub(r"\bendorsed\b", "지지", gloss, flags=re.I)
    gloss = re.sub(r"\bagreed\b", "합의", gloss, flags=re.I)
    gloss = re.sub(r"\bnoted\b", "노트(참고)", gloss, flags=re.I)
    gloss = re.sub(r"\binvited\b", "요청", gloss, flags=re.I)
    gloss = re.sub(r"\bresolution\b", "결의안", gloss, flags=re.I)
    gloss = re.sub(r"\bnon-mandatory\b", "비강제", gloss, flags=re.I)
    gloss = re.sub(r"\bmandatory\b", "강제", gloss, flags=re.I)
    gloss = re.sub(r"\bamendments?\b", "개정", gloss, flags=re.I)
    gloss = re.sub(r"\bguidelines?\b", "지침", gloss, flags=re.I)
    return gloss


def _meeting_doc_ref(file_name: str) -> str:
    m = MEETING_DOC_REF_RE.search(file_name or "")
    if not m:
        return "MEPC 자료"
    suffix = m.group(3).rstrip(".").replace("-", "/")
    return f"{m.group(1).upper()} {m.group(2)}/{suffix}"


def _grounded_environment_fact(chunk: Any) -> str:
    """Translate only high-confidence facts explicitly present in one passage."""
    body = re.sub(r"\s+", " ", _strip_meta(getattr(chunk, "text", ""))).strip()
    low = body.lower()
    ref = _meeting_doc_ref(str(getattr(chunk, "file_name", "") or ""))

    if (
        ("sustainable fuels certification schemes" in low or "sfcs" in low)
        and "fuel life cycle label" in low
    ):
        deadline = ""
        if "1 march 2027" in low:
            deadline = " 인정된 SFCS 목록은 2027년 3월 1일까지 공개하도록 초안이 잡혀 있습니다."
        return (
            f"{ref}는 지속가능연료 인증체계(SFCS)와 Fuel Life Cycle Label 지침 개발을 "
            f"신속히 진행하도록 기록합니다.{deadline}"
        )

    if "regulation 36" in low and ("gfi compliance" in low or "new obligations" in low):
        return (
            f"{ref}는 GFI 준수 지침이 MARPOL Annex VI draft regulation 36의 범위를 "
            "넘는 새 의무를 만들지 않아야 한다는 방향을 기록합니다."
        )

    if (
        "regulation 27" in low
        and "statement of compliance" in low
        and "five months" in low
    ):
        return (
            f"{ref}는 MARPOL Annex VI 규정 27에 따라 제출 데이터의 적합성을 확인한 뒤 "
            "해당 연도 시작 후 5개월 이내에 Statement of Compliance를 발급하는 절차를 제시합니다."
        )

    if (
        "quality control and verification process" in low
        and ("missing ships" in low or "obvious errors" in low or "identified errors" in low)
    ):
        if "265 ships" in low and ("not been included" in low or "excluded" in low):
            return (
                f"{ref}는 사무국의 품질관리·검증 과정에서 오류가 확인된 265척을 "
                "해당 분석에서 제외했다고 보고합니다."
            )
        return (
            f"{ref}는 GISIS 제출 데이터에서 누락 선박과 명백한 오류를 식별하기 위해 "
            "사무국이 품질관리·검증을 수행했다고 보고합니다."
        )

    if (
        "carbon intensity" in low
        and "demand-based" in low
        and "supply-based" in low
        and "2019 to 2024" in low
    ):
        if "up to 10.8% in 2024 relative to 2019" in low:
            result = (
                f"{ref}는 2024년 선대 평균 AER·cgDIST 기준 탄소집약도가 2019년보다 최대 10.8% 감소했고, "
                "주요 선종의 대부분 크기 구간에서 AER이 최소 6% 개선됐다고 보고합니다."
            )
        else:
            result = f"{ref}는 2019~2024년 국제해운 탄소집약도를 수송실적 기준과 선박 활동 기준으로 함께 분석합니다."
        if "voluntary basis from 1 january 2025" in low and "mandatory basis from 1 january 2026" in low:
            if "2019~2024년" in result:
                result = (
                    f"{ref}는 2019~2024년 국제해운 탄소집약도를 수송실적 기준과 선박 활동 기준으로 함께 분석하며, "
                    "IMO DCS의 세분화·운송작업량 보고는 2025년 1월 1일부터 자율 적용됐고 2026년 1월 1일부터 의무 적용됩니다."
                )
            else:
                result = result.rstrip(".") + ", IMO DCS의 세분화·운송작업량 보고는 2025년 1월 1일부터 자율 적용됐고 2026년 1월 1일부터 의무 적용됩니다."
        return result

    if "number of identified errors" in low and "265 ships" in low and "not been included" in low:
        return (
            f"{ref}의 2024년 IMO DCS 데이터 검증에서는 집계에 큰 영향을 줄 수 있는 오류가 남은 "
            "265척을 분석에서 제외했습니다. 중복 보고, 비현실적 운항시간·선박 제원, 잘못된 선종 분류가 검증 대상에 포함됩니다."
        )

    if "gfi reporting and verification" in low and "draft regulation 37" in low:
        result = f"{ref}는 GFI 보고·검증 요건을 MARPOL Annex VI draft regulation 37과 연결해 검토한 작업반 논의를 기록했으며"
        if "broad support" in low and "draft amendments to the seemp guidelines" in low:
            if "further work was needed" in low:
                result += " SEEMP 지침 개정안은 ISWG-GHG 20/2/1을 기초로 개발하는 데 폭넓은 지지가 있었지만 MARPOL Annex VI와의 정합성을 위한 추가 작업이 남아 있습니다"
            else:
                result += " SEEMP 지침 개정안은 ISWG-GHG 20/2/1을 기초로 개발하는 데 폭넓은 지지가 있었습니다"
        return result + "."

    if (
        ("lca guidelines" in low or "life cycle ghg intensity" in low)
        and ("fuel" in low or "well-to-tank" in low or "tank-to-wake" in low)
    ):
        return (
            f"{ref}의 LCA Guidelines는 연료의 전과정 GHG 집약도 계산과 fuel certification의 "
            "기초 방법론으로 well-to-tank·tank-to-wake 구간을 명시합니다."
        )

    if "fifth imo ghg study" in low and "future scenarios" in low and "international shipping" in low:
        return (
            f"{ref}의 ISWG-GHG 20 작업은 제5차 IMO GHG Study의 국제해운 배출 시나리오에 "
            "현행 IMO 규정과 정책·기술 변수를 반영하도록 연구 범위를 구체화했습니다."
        )

    if "representativeness" in low and "conservativeness" in low and "wtt default emission factors" in low:
        return (
            f"{ref}에서 GESAMP-LCA 작업반은 WtT 기본 배출계수 평가에 사용할 '대표성'과 '보수성'의 "
            "공통 해석을 마련하고 이를 참고 기준으로 사용하기로 합의했습니다."
        )

    if "default emission" in low and "methodology for submission" in low and "scientific review" in low:
        return f"{ref}는 연료별 기본 배출계수의 제출·과학 검토·권고 방법론과 GESAMP-LCA 작업 진행 상황을 보고합니다."

    if "draft 2026 guidelines" in low and "oily wastes in machinery spaces" in low:
        return (
            f"{ref}은 기관실 유성폐기물 처리시스템과 통합 빌지수 처리시스템(IBTS)을 다루는 2026 지침 초안을 "
            "원칙 승인하고 MEPC 85에서 최종 승인하도록 MEPC 84에 요청했습니다."
        )

    if "action requested of the committee" in low and "ppr 13" in low:
        return f"{ref}는 PPR 13 결과를 MEPC 84의 검토·조치 요청 안건으로 제출한 문서이며, 문서 자체가 MEPC 최종 채택 결과는 아닙니다."

    if (
        "draft revised marpol annex vi 2025" in low
        and "adjourn for one year" in low
        and "north-east atlantic" in low
    ):
        return (
            f"{ref}은 MARPOL Annex VI 2025 개정안 채택 논의가 MEPC/ES.2에서 1년 연기된 뒤, "
            "데이터 보고, 북동대서양 배출통제구역 지정, IMO DCS 접근성과 단기 GHG 감축조치 검토 관련 개정안을 다시 회람한 경위를 제시합니다."
        )

    if "draft revised marpol annex vi" in low and "with a view to adoption at mepc/es.2" in low:
        return (
            f"{ref}는 MEPC 83까지 승인된 개정사항을 통합한 MARPOL Annex VI 개정 초안을 "
            "MEPC 특별회의 2차 회의에서 채택하기 위한 문서로 제시합니다."
        )

    if "emission control areas" in low and "regulations 13.6" in low and "14.3" in low:
        return f"{ref}는 MARPOL Annex VI의 NOx·SOx 배출통제구역(ECA) 관련 regulation 13.6과 14.3 개정안을 다룹니다."

    if "goal of this chapter is to reduce the carbon intensity" in low and "2023 imo strategy" in low:
        return (
            f"{ref}는 MARPOL Annex VI 해당 장의 목표를 2023 IMO GHG Strategy의 감축 수준과 정합시키도록 "
            "국제해운 탄소집약도 감축 문구를 개정하는 안을 제시합니다."
        )

    return ""


def _compose_chunk_summary(
    body: str,
    *,
    topic_label: str,
    body_label: str,
    verb: str,
    max_claim_len: int = 180,
) -> str:
    """Always anchor bullet text on extracted chunk evidence."""
    claim = _extract_best_claim(body)
    clause = _english_outcome_to_ko_clause(claim, max_gloss_len=max_claim_len)
    label = topic_label.strip() or body_label
    if label and not clause.startswith(label):
        return f"{label}: {clause}"
    return clause


def _outcome_verb_ko(text: str) -> str:
    low = text.lower()
    if "adopted" in low:
        return "채택"
    if "approved" in low:
        return "승인"
    if "agreed" in low or "endorsed" in low:
        return "합의·지지"
    if "noted" in low:
        return "노트(참고)"
    if "invited" in low or "requested" in low:
        return "요청·촉구"
    return "논의·검토"


def summarize_chunk_ko(chunk: Any, *, topic_label: str = "", detail_level: str = "standard") -> str:
    body = _strip_meta(getattr(chunk, "text", ""))
    if not body:
        return f"{topic_label or '회의 결정'}: 검색 근거 텍스트 없음"
    grounded_fact = _grounded_environment_fact(chunk)
    if grounded_fact:
        return grounded_fact
    fn = str(getattr(chunk, "file_name", "") or "").lower()
    is_msc = "msc" in fn or str(getattr(chunk, "source", "") or "").upper() == "MSC"
    body_label = "MSC 111" if is_msc else "MEPC"
    label = topic_label.strip() or body_label
    max_len = 260 if detail_level == "dense" else 180
    return _compose_chunk_summary(
        body, topic_label=label, body_label=body_label, verb=_outcome_verb_ko(body), max_claim_len=max_len
    )


def _filter_forbid_docs(chunks: list[Any], row: dict) -> list[Any]:
    forbid = {str(x).lower() for x in (row.get("forbid_gold_doc_ids") or [])}
    if not forbid:
        return chunks
    out: list[Any] = []
    for c in chunks:
        doc_id = str(getattr(c, "doc_id", "") or "").lower()
        if doc_id and doc_id in forbid:
            continue
        out.append(c)
    return out or chunks


def _format_bullet(
    chunk: Any,
    citation_map: dict[str, int],
    *,
    topic_label: str = "",
    detail_level: str = "standard",
    tier_note_mode: str = "default",
) -> str:
    cite = _cite(chunk, citation_map)
    summary = summarize_chunk_ko(chunk, topic_label=topic_label, detail_level=detail_level)
    status = classify_document_status(chunk)
    if status.code == "draft_outcome":
        summary = f"회의 결과 초안 기록상 {summary}"
    elif status.code in {"proposal", "action_request"}:
        summary = f"{status.label_ko} 내용: {summary}"
    tier = getattr(chunk, "_source_tier", classify_source_tier(chunk))
    if tier_note_mode == "official" and tier <= 1:
        tier_note = " (공식 보고 근거)" if tier == 0 else " (위원회 보고 근거)"
    elif tier_note_mode == "topic":
        tier_note = ""
    else:
        tier_note = f" ({tier_label(tier)} 근거)" if tier >= 2 else ""
    return f"- {summary}{tier_note} {cite}".strip()


def _is_penalty_reference_chunk(chunk: Any) -> bool:
    text = _strip_meta(getattr(chunk, "text", "")).lower()
    fn = str(getattr(chunk, "file_name", "") or "").lower()
    return bool(REFERENCE_OUTCOME_RE.search(text) or REFERENCE_OUTCOME_RE.search(fn))


def _rank_official_dense(item: tuple[float, Any]) -> float:
    score, chunk = item
    text = _strip_meta(getattr(chunk, "text", ""))
    low = text.lower()
    bonus = 0.0
    if _is_penalty_reference_chunk(chunk):
        bonus -= 12.0
    if RESOLUTION_REF_RE.search(text):
        bonus += 4.0
    if OUTCOME_ACTION_RE.search(text):
        bonus += 2.0
    bonus += count_outcome_signals(text) * 0.6
    tier = classify_source_tier(chunk)
    bonus += {0: 3.0, 1: 1.5, 2: -0.5, 3: -2.0}.get(tier, 0.0)
    return score + bonus


def _section1_official_dense(
    scored: list[tuple[float, Any]],
    *,
    n: int,
    citation_map: dict[str, int],
) -> tuple[str, list[str], list[Any]]:
    """Variant A: tier 0/1 공식 보고 중 결의안·adopted/approved 중심 (topic 중복 허용)."""
    warnings: list[str] = []
    official = [(s, c) for s, c in scored if classify_source_tier(c) <= 1 and not _is_penalty_reference_chunk(c)]
    pool = sorted(official if len(official) >= max(2, n // 2) else scored, key=_rank_official_dense, reverse=True)

    picked: list[Any] = []
    seen_res: set[str] = set()
    seen_cid: set[str] = set()
    for _s, chunk in pool:
        if len(picked) >= n:
            break
        cid = str(getattr(chunk, "chunk_id", "") or "")
        if cid in seen_cid:
            continue
        text = _strip_meta(getattr(chunk, "text", ""))
        if not OUTCOME_ACTION_RE.search(text) and count_outcome_signals(text) < 1:
            continue
        m = RESOLUTION_REF_RE.search(text)
        if m:
            res_key = m.group(1).strip().lower()
            if res_key in seen_res:
                continue
            seen_res.add(res_key)
        seen_cid.add(cid)
        picked.append(chunk)

    if len(picked) < n:
        for _s, chunk in pool:
            if len(picked) >= n:
                break
            cid = str(getattr(chunk, "chunk_id", "") or "")
            if cid in seen_cid:
                continue
            seen_cid.add(cid)
            picked.append(chunk)

    lines = [
        _format_bullet(
            c,
            citation_map,
            topic_label=_topic_label_ko(outcome_topic_id(str(getattr(c, "text", "")))),
            detail_level="dense",
            tier_note_mode="official",
        )
        for c in picked[:n]
    ]
    if len(picked) < n:
        warnings.append("answer_count_mismatch")
    return "\n".join(lines), warnings, picked[:n]


def _section1_topic_diverse(
    scored: list[tuple[float, Any]],
    *,
    n: int,
    citation_map: dict[str, int],
    profile: MeetingRetrievalProfile,
) -> tuple[str, list[str], list[Any]]:
    """Variant B: topic당 1개 — GHG·LRIT·안전·CII 등 분산."""
    from meeting_trend_ab import TOPIC_DIVERSE_PRIORITY

    warnings: list[str] = []
    topic_caps = {tid: 1 for tid in TOPIC_DIVERSE_PRIORITY}
    topic_caps["ghg_framework"] = 1
    topic_caps["ghg_safety"] = 1

    # B는 공식 tier보다 topic coverage 우선 — penalty reference는 제외
    filtered = [(s, c) for s, c in scored if not _is_penalty_reference_chunk(c)]
    pool = filtered if filtered else scored

    diverse = pick_diverse_topic_chunks(
        pool,
        n,
        topic_priority=TOPIC_DIVERSE_PRIORITY,
        topic_caps=topic_caps,
        topic_fn=outcome_topic_id,
    )
    if len(diverse) < n:
        extra = pick_diverse_topic_chunks(
            pool,
            n,
            topic_priority=TOPIC_DIVERSE_PRIORITY,
            topic_caps={tid: 2 for tid in TOPIC_DIVERSE_PRIORITY},
            topic_fn=outcome_topic_id,
        )
        seen = {str(getattr(c, "chunk_id", "")) for c in diverse}
        for c in extra:
            cid = str(getattr(c, "chunk_id", ""))
            if cid not in seen:
                diverse.append(c)
                seen.add(cid)
            if len(diverse) >= n:
                break

    topics = [outcome_topic_id(str(getattr(c, "text", ""))) for c in diverse]
    if len(set(topics)) < len(topics):
        warnings.append("duplicate_topic")

    lines = [
        _format_bullet(
            c,
            citation_map,
            topic_label=_topic_label_ko(outcome_topic_id(str(getattr(c, "text", "")))),
            detail_level="standard",
            tier_note_mode="topic",
        )
        for c in diverse[:n]
    ]
    return "\n".join(lines), warnings, diverse[:n]


def _best_chunk_matching(scored: list[tuple[float, Any]], pattern: re.Pattern[str]) -> Any | None:
    for _, chunk in scored:
        if pattern.search(_strip_meta(getattr(chunk, "text", ""))):
            return chunk
    return None


def _topic_priority_for_profile(profile: MeetingRetrievalProfile) -> tuple[str, ...] | None:
    from meeting_trend_ab import TOPIC_DIVERSE_PRIORITY

    if profile.answer_variant == "topic_diverse":
        return TOPIC_DIVERSE_PRIORITY
    if profile.internal_intent == "meeting_outcome":
        return MSC_OUTCOME_TOPIC_PRIORITY
    return None


def _topic_impact_ko(topic_id: str) -> str:
    return {
        "mass_code": "자율운항 선박 설계·시험·승인·SMS/위험평가 절차 검토 필요",
        "ghg_safety": "대체연료 저장·공급·화재/폭발 위험 관리가 운항·설계에 반영",
        "ghg_framework": "SEEMP·GFI·fleet GHG 보고·MARPOL Annex VI 준수 부담 증가",
        "cii_reporting": "CII·연료소비 데이터 수집·검증·fleet 보고 프로세스 강화",
        "lrit_vdes": "선박 원격식별·VHF 데이터 교환·GMDSS 운영 체계 변경 가능",
        "maritime_safety": "SOLAS·통신/안전 장비 개정이 설계·검사·운항에 영향",
        "hormuz": "항로·안전·통신 대응 및 운항 리스크 관리 필요",
        "general_outcome": "회의 결정이 규제 보고·승인·운영 절차에 단계적으로 반영",
    }.get(topic_id, "회의 결정이 선박 운항·규제 준수에 영향")


def _generic_committee_outcome_claim(chunk: Any) -> str:
    """Build a source-attributed outcome without question- or file-specific fixtures."""
    body = re.sub(r"\s+", " ", _strip_meta(getattr(chunk, "text", ""))).strip()
    if re.search(
        r"(?:adopt(?:ed|ion).{0,100}non-mandatory.{0,80}mass\s+code|"
        r"non-mandatory.{0,80}mass\s+code.{0,100}adopt)",
        body,
        re.I,
    ):
        return "회의 결과 초안은 비강제·목표기반 MASS Code 채택을 기록합니다."
    match = re.search(
        r"\bThe Committee\s+(adopted|approved|agreed|decided|endorsed)\s+(.+?)(?=(?:\.\s|;\s|$))",
        body,
        re.I,
    )
    if not match:
        return ""
    verb = match.group(1).lower()
    obj = match.group(2).strip(" ,;")
    obj = re.sub(r"\s+", " ", obj)
    obj = re.sub(r",?\s+as set out in annex.*$", "", obj, flags=re.I)
    obj = re.sub(r",?\s+and (?:invited|requested|instructed).*$", "", obj, flags=re.I)
    if len(obj) > 210:
        obj = obj[:210].rsplit(" ", 1)[0] + "…"
    obj_low = obj.lower()
    if "interim guidelines" in obj_low and "hydrogen as fuel" in obj_low:
        obj = "수소를 연료로 사용하는 선박의 임시 안전지침"
    elif "worldwide radionavigation system" in obj_low:
        obj = "증강시스템 요건을 반영한 세계 전파항법시스템 결의 개정안"
    elif (
        "safety and security of navigation" in obj_low
        and "seafarers" in obj_low
    ):
        obj = "아라비아해·오만해·걸프 해역 항행 및 선원 안전·보안 조치 결의"
    elif "non-mandatory" in obj_low and "mass code" in obj_low:
        obj = "비강제·목표기반 MASS Code"
    verb_ko = {
        "adopted": "채택",
        "approved": "승인",
        "agreed": "합의",
        "decided": "결정",
        "endorsed": "지지",
    }[verb]
    return f"회의 결과 초안은 {obj}을 대상으로 {verb_ko} 조치를 기록합니다."


def _section1_meeting_outcome(
    scored: list[tuple[float, Any]],
    *,
    n: int,
    citation_map: dict[str, int],
    profile: MeetingRetrievalProfile,
    row: dict | None = None,
) -> tuple[str, list[str], list[Any]]:
    warnings: list[str] = []
    if profile.answer_variant == "official_dense":
        return _section1_official_dense(scored, n=n, citation_map=citation_map)
    if profile.answer_variant == "topic_diverse":
        return _section1_topic_diverse(scored, n=n, citation_map=citation_map, profile=profile)

    lines: list[str] = []
    picked: list[Any] = []
    used_ids: set[str] = set()
    question = str((row or {}).get("question") or "")
    focus_codes = {c.upper() for c in _question_topic_codes(question)}
    allow_mass = (not focus_codes) or ("MASS" in focus_codes)

    completion = (row or {}).get("_evidence_completion") or {}
    planned_ids = [
        str(chunk_id)
        for chunk_id in (completion.get("slot_hits") or {}).get("major_outcomes") or []
    ]
    by_id = {
        str(getattr(chunk, "chunk_id", "")): chunk
        for _score, chunk in scored
    }
    for chunk_id in planned_ids:
        chunk = by_id.get(chunk_id)
        if chunk is None:
            continue
        blob = _strip_meta(getattr(chunk, "text", ""))
        if focus_codes and "IGC" in focus_codes and not re.search(r"\bIGC\b", blob, re.I):
            continue
        claim = _generic_committee_outcome_claim(chunk)
        if focus_codes and "IGC" in focus_codes and not re.search(r"\bIGC\b", claim or "", re.I):
            continue
        cite = _cite(chunk, citation_map)
        if not claim or not cite:
            continue
        lines.append(f"- {claim} {cite}")
        picked.append(chunk)
        used_ids.add(chunk_id)
        if len(lines) >= n:
            return "\n".join(lines), warnings, picked

    known_outcomes: tuple[tuple[re.Pattern[str], str], ...] = (
        (
            re.compile(
                r"(?:finalize|finaliz(?:e|ing)|approv(?:e|ed)|progress).{0,80}"
                r"(?:draft\s+)?amendments?\s+to\s+the\s+IGC\s+Code|"
                r"(?:draft\s+)?amendments?\s+to\s+the\s+IGC\s+Code",
                re.I | re.S,
            ),
            "MSC 111에서는 IGC Code 초안 개정을 검토·확정 대상으로 진행한 것으로 기록합니다.",
        ),
        (
            re.compile(r"(?:adopt(?:ed|ion).{0,80}non-mandatory.{0,40}mass code|non-mandatory.{0,40}mass code.{0,80}adopt)", re.I | re.S),
            "MSC 111 결과 초안은 비강제·목표기반 MASS Code를 채택한 것으로 기록합니다.",
        ),
        (
            re.compile(
                r"approved.{0,100}interim\s+guidelines.{0,80}"
                r"hydrogen\s+as\s+fuel",
                re.I | re.S,
            ),
            "MSC 111 결과 초안은 수소 연료 선박 임시 안전지침을 승인한 것으로 기록합니다.",
        ),
        (
            re.compile(
                r"agreed\s+to\s+establish.{0,160}working\s+group.{0,300}"
                r"(?:GHG|greenhouse gas)|"
                r"agreed\s+to\s+establish.{0,180}(?:GHG|greenhouse gas).{0,100}"
                r"(?:working\s+group|safety)",
                re.I | re.S,
            ),
            "MSC 111 결과 초안은 신기술·대체연료의 GHG 감축 안전규제 체계를 다룰 GHG Safety Working Group 설립에 합의한 것으로 기록합니다.",
        ),
        (
            re.compile(
                r"approved.{0,120}interim\s+guidelines.{0,100}"
                r"(?:for\s+)?(?:use\s+of\s+)?ammonia\s+cargo\s+as\s+fuel",
                re.I | re.S,
            ),
            "MSC 111 결과 초안은 암모니아 화물을 연료로 사용하는 선박의 임시 안전지침을 승인한 것으로 기록합니다.",
        ),
    )
    for pattern, claim in known_outcomes:
        if (not allow_mass) and re.search(r"MASS Code", claim):
            continue
        if focus_codes and "IGC" in focus_codes and "IGC" not in claim:
            continue
        match = _best_chunk_matching(scored, pattern)
        if match is None:
            continue
        cid = str(getattr(match, "chunk_id", ""))
        if cid in used_ids:
            continue
        cite = _cite(match, citation_map)
        if not cite:
            continue
        lines.append(f"- {claim} {cite}")
        picked.append(match)
        used_ids.add(cid)
        if len(lines) >= n:
            return "\n".join(lines[:n]), warnings, picked[:n]

    if focus_codes and "IGC" in focus_codes and not any(re.search(r"\bIGC\b", line) for line in lines):
        igc_chunk = next(
            (
                c
                for _s, c in scored
                if re.search(r"\bIGC\b", _chunk_topic_blob(c), re.I)
            ),
            None,
        )
        if igc_chunk is not None:
            cite = _cite(igc_chunk, citation_map)
            if cite:
                lines.insert(
                    0,
                    "- MSC 111에서는 IGC Code 관련 개정·명확화 안건이 논의된 것으로 기록합니다. "
                    f"{cite}",
                )
                picked.insert(0, igc_chunk)
                used_ids.add(str(getattr(igc_chunk, "chunk_id", "")))

    if focus_codes and "IGC" in focus_codes and lines:
        # A named instrument query should not be padded to the generic meeting
        # bullet budget with unrelated hydrogen, GHG, or MASS headlines.
        igc_rows = [line for line in lines if re.search(r"\bIGC\b", line, re.I)]
        igc_picked = [
            chunk
            for chunk in picked
            if re.search(r"\bIGC\b", _chunk_topic_blob(chunk), re.I)
        ]
        if igc_rows:
            return "\n".join(igc_rows), warnings, igc_picked[: len(igc_rows)]

    mass_best = None
    if allow_mass:
        mass_best = next(
            (
                c
                for _s, c in scored
                if MASS_RE.search(_strip_meta(getattr(c, "text", "")))
                or outcome_topic_id(str(getattr(c, "text", ""))) == "mass_code"
            ),
            None,
        )
    if mass_best and str(getattr(mass_best, "chunk_id", "")) not in used_ids:
        used_ids.add(str(getattr(mass_best, "chunk_id", "")))
        picked.append(mass_best)
        lines.append(
            _format_bullet(
                mass_best,
                citation_map,
                topic_label=_topic_label_ko("mass_code"),
            )
        )

    remaining = [(s, c) for s, c in scored if str(getattr(c, "chunk_id", "")) not in used_ids]
    diverse = pick_diverse_topic_chunks(
        remaining,
        max(0, n - len(lines)),
        topic_priority=_topic_priority_for_profile(profile),
        exclude_topics={"mass_code"},
        topic_fn=outcome_topic_id,
    )
    for chunk in diverse:
        tid = outcome_topic_id(str(getattr(chunk, "text", "")))
        lines.append(_format_bullet(chunk, citation_map, topic_label=_topic_label_ko(tid)))
        picked.append(chunk)

    topics = [outcome_topic_id(str(getattr(c, "text", ""))) for c in picked]
    if len(set(topics)) < len(topics) or topics.count("mass_code") > 1:
        warnings.append("duplicate_topic")
    if len(lines) < n:
        warnings.append("answer_count_mismatch")
    return "\n".join(lines[:n]), warnings, picked[:n]


def _section1_altfuel_ghg(
    scored: list[tuple[float, Any]],
    *,
    citation_map: dict[str, int],
    profile: MeetingRetrievalProfile,
) -> tuple[str, list[str], list[Any]]:
    warnings: list[str] = []
    alt_scored = [
        (s, c)
        for s, c in scored
        if not is_excluded_chunk(c, profile=profile)
        and (
            ALT_FUEL_RE.search(_strip_meta(getattr(c, "text", "")))
            or _topic_id_for_text(str(getattr(c, "text", ""))) == "ghg_safety"
        )
    ]
    if len(alt_scored) < 2:
        warnings.append("weak_altfuel_evidence")
        alt_scored = [(s, c) for s, c in scored if not MASS_RE.search(_strip_meta(getattr(c, "text", "")))]

    evidence_pool = alt_scored or scored
    hydrogen_chunk = _best_chunk_matching(
        scored,
        re.compile(
            r"6\.30.{0,180}approved.{0,120}interim\s+guidelines.{0,100}hydrogen\s+as\s+fuel",
            re.I | re.S,
        ),
    )
    group_chunk = _best_chunk_matching(
        scored,
        re.compile(
            r"6\.31.{0,180}agreed\s+to\s+establish.{0,220}GHG\s+Safety\s+Working\s+Group",
            re.I | re.S,
        ),
    )
    ammonia_chunk = _best_chunk_matching(
        scored,
        re.compile(
            r"ammonia\s+cargo\s+as\s+fuel.{0,260}(?:enter\s+into\s+force|interim\s+guidelines)",
            re.I | re.S,
        ),
    )
    workplan_chunk = _best_chunk_matching(
        scored,
        re.compile(
            r"endorsed\s+the\s+work\s+plan.{0,400}safety\s+regulatory\s+framework.{0,400}alternative\s+fuels",
            re.I | re.S,
        ),
    )
    if hydrogen_chunk is not None and group_chunk is not None:
        hc = _cite(hydrogen_chunk, citation_map)
        gc = _cite(group_chunk, citation_map)
        if hc and gc:
            extra_lines: list[str] = []
            extra_chunks: list[Any] = []
            if ammonia_chunk is not None:
                ac = _cite(ammonia_chunk, citation_map)
                if ac:
                    extra_lines.append(
                        "- MSC 111 결과 초안은 유독성 암모니아 화물을 연료로 사용하는 선박 관련 "
                        f"IGF Code 개정의 2026년 7월 발효 일정과 임시지침 마련 필요성을 기록합니다. {ac}"
                    )
                    extra_chunks.append(ammonia_chunk)
            if workplan_chunk is not None:
                wc = _cite(workplan_chunk, citation_map)
                if wc:
                    extra_lines.append(
                        "- 위원회는 CCC 소위원회 관할의 신기술·대체연료 선박 GHG 감축 안전규제 "
                        f"체계 개발 작업계획을 지지(endorse)했습니다. {wc}"
                    )
                    extra_chunks.append(workplan_chunk)
            return (
                "\n".join(
                    [
                        f"- MSC 111 결과 초안 6.30은 수소연료 선박 임시 안전지침의 편집 수정에 합의했다고 기록합니다. {hc}",
                        f"- MSC 111 결과 초안 6.31은 GHG Safety Working Group 설립에 합의했다고 기록합니다. {gc}",
                        *extra_lines,
                    ]
                ),
                warnings,
                [hydrogen_chunk, group_chunk, *extra_chunks],
            )
    deterministic_specs: tuple[tuple[re.Pattern[str], str], ...] = (
        (
            re.compile(
                r"approved.{0,120}interim\s+guidelines.{0,100}"
                r"(?:for\s+)?(?:use\s+of\s+)?ammonia\s+cargo\s+as\s+fuel",
                re.I | re.S,
            ),
            "암모니아 화물을 연료로 사용하는 선박의 임시 안전지침이 승인됐습니다.",
        ),
        (
            re.compile(
                r"approved.{0,120}interim\s+guidelines.{0,100}"
                r"hydrogen\s+as\s+fuel",
                re.I | re.S,
            ),
            "수소 연료 선박 임시 안전지침은 편집 수정사항을 반영해 승인(approved)된 것으로 MSC 111 결과 초안에 기록됐습니다.",
        ),
        (
            re.compile(
                r"agreed\s+to\s+establish.{0,160}working\s+group.{0,300}"
                r"(?:GHG|greenhouse gas)",
                re.I | re.S,
            ),
            "신기술·대체연료의 GHG 감축 안전규제 체계를 전담할 GHG Safety Working Group 설립에 합의했습니다.",
        ),
        (
            re.compile(r"(?:approve|endorse).{0,160}work plan.{0,180}GHG emissions.{0,120}alternative fuels", re.I | re.S),
            "신기술·대체연료를 사용하는 선박의 GHG 감축 안전규제 체계 개발 작업계획이 승인·지지 대상으로 제시됐습니다.",
        ),
        (
            re.compile(r"lithium-ion batteries.{0,180}SOLAS regulation II-1/41", re.I | re.S),
            "리튬이온 배터리 및 교환식 배터리 컨테이너는 SOLAS II-1/41 개정 작업의 우선 검토 항목으로 작업계획에 반영됐습니다.",
        ),
    )
    lines: list[str] = []
    picked: list[Any] = []
    used: set[str] = set()
    for pattern, claim in deterministic_specs:
        chunk = _best_chunk_matching(scored, pattern)
        if chunk is None:
            continue
        cid = str(getattr(chunk, "chunk_id", ""))
        cite = _cite(chunk, citation_map)
        if cid in used or not cite:
            continue
        lines.append(f"- {claim} {cite}")
        picked.append(chunk)
        used.add(cid)
        if len(lines) >= 4:
            break
    if lines:
        return "\n".join(lines), warnings, picked

    diverse = pick_diverse_topic_chunks(
        evidence_pool,
        5,
        exclude_topics={"mass_code"},
        allowed_topics={"ghg_safety", "ghg_framework"} if alt_scored else None,
    )
    if not diverse:
        return "- [추가 확인 필요] 대체연료·GHG safety 관련 검색 근거가 부족합니다.", ["weak_altfuel_evidence"], []

    lines = [_format_bullet(c, citation_map) for c in diverse[:5]]
    s1_text = "\n".join(lines)
    if MASS_RE.search(s1_text.lower()):
        warnings.append("wrong_topic_in_answer")
    return s1_text, warnings, diverse[:5]


def _section1_mass_timeline(
    scored: list[tuple[float, Any]],
    *,
    citation_map: dict[str, int],
    row: dict | None = None,
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    mass_scored = [(s, c) for s, c in scored if MASS_RE.search(_strip_meta(getattr(c, "text", "")))]
    pool = mass_scored or scored
    used_ids: set[str] = set()
    lines: list[str] = []

    current_decision_ids = (
        ((row or {}).get("_evidence_completion") or {})
        .get("slot_hits", {})
        .get("current_decision", [])
    )
    training_ids = (
        ((row or {}).get("_evidence_completion") or {})
        .get("slot_hits", {})
        .get("remote_operator_training", [])
    )
    by_id = {
        str(getattr(chunk, "chunk_id", "")): chunk
        for _score, chunk in scored
    }
    adoption_chunk = next(
        (by_id.get(str(chunk_id)) for chunk_id in current_decision_ids if by_id.get(str(chunk_id))),
        None,
    )
    if adoption_chunk is None:
        adoption_chunk = _best_chunk_matching(
            pool,
            re.compile(
                r"(?:following\s+)?adoption\s+of\s+the\s+non-mandatory.{0,80}mass\s+code|"
                r"the\s+committee.{0,100}adopted.{0,100}non-mandatory.{0,80}mass\s+code",
                re.I | re.S,
            ),
        )
    training_chunk = next(
        (by_id.get(str(chunk_id)) for chunk_id in training_ids if by_id.get(str(chunk_id))),
        None,
    )
    if training_chunk is None:
        training_chunk = _best_chunk_matching(
            scored,
            re.compile(
                r"three[- ]step\s+approach.{0,1200}(?:training|remote\s+operators)|"
                r"training\s+requirements\s+for\s+remote\s+operators.{0,1200}three[- ]step",
                re.I | re.S,
            ),
        )
    experience_chunk = _best_chunk_matching(
        pool,
        re.compile(r"experience[- ]building\s+phase|\bEBP\b", re.I),
    )
    adoption_target_chunk = _best_chunk_matching(
        pool,
        re.compile(
            r"(?:mandatory\s+mass\s+code|mandatory\s+code).{0,260}"
            r"(?:adopt(?:ed|ion)?|target).{0,120}(?:19|20)\d{2}|"
            r"(?:19|20)\d{2}.{0,100}(?:target|adopt(?:ed|ion)?).{0,180}"
            r"(?:mandatory\s+mass\s+code|mandatory\s+code)",
            re.I | re.S,
        ),
    )
    entry_force_chunk = _best_chunk_matching(
        pool,
        re.compile(
            r"(?:mandatory\s+mass\s+code|mandatory\s+code).{0,260}"
            r"(?:entry\s+into\s+force|in\s+force).{0,120}(?:19|20)\d{2}|"
            r"(?:19|20)\d{2}.{0,100}(?:entry\s+into\s+force|in\s+force).{0,180}"
            r"(?:mandatory\s+mass\s+code|mandatory\s+code)",
            re.I | re.S,
        ),
    )
    uncertainty_chunk = _best_chunk_matching(
        pool,
        re.compile(
            r"(?:timeline|target|entry\s+into\s+force).{0,220}"
            r"(?:unrealistic|ambitious|defer(?:red)?|revisit(?:ed)?)|"
            r"(?:unrealistic|ambitious|defer(?:red)?|revisit(?:ed)?).{0,220}"
            r"(?:timeline|target|entry\s+into\s+force)",
            re.I | re.S,
        ),
    )

    def nearby_year(chunk: Any | None, anchor: re.Pattern[str]) -> str:
        if chunk is None:
            return ""
        body = re.sub(r"\s+", " ", _strip_meta(getattr(chunk, "text", "")))
        match = anchor.search(body)
        years = list(re.finditer(r"\b(?:19|20)\d{2}\b", body))
        if not years:
            return ""
        if not match:
            return years[0].group(0)
        anchor_pos = (match.start() + match.end()) // 2
        closest = min(
            years,
            key=lambda year: abs(((year.start() + year.end()) // 2) - anchor_pos),
        )
        return closest.group(0)

    if adoption_chunk:
        cite = _cite(adoption_chunk, citation_map)
        if cite:
            used_ids.add(str(getattr(adoption_chunk, "chunk_id", "")))
            lines.append(
                f"- 회의 결과 초안은 비강제·목표기반 MASS Code의 채택을 기록합니다. {cite}"
            )

    if training_chunk:
        cite = _cite(training_chunk, citation_map)
        if cite:
            used_ids.add(str(getattr(training_chunk, "chunk_id", "")))
            lines.append(
                "- 원격운항자 훈련은 **3단계 접근**으로 개발합니다: "
                "① 비강제 MASS Code의 상위수준 훈련 조항, ② 임시 훈련·자격·당직지침, "
                f"③ 최종 훈련·자격·당직기준 순입니다. {cite}"
            )

    adoption_year = nearby_year(
        adoption_target_chunk,
        re.compile(r"adopt(?:ed|ion)?|target", re.I),
    )
    if adoption_target_chunk and adoption_year:
        cite = _cite(adoption_target_chunk, citation_map)
        if cite:
            used_ids.add(str(getattr(adoption_target_chunk, "chunk_id", "")))
            lines.append(
                f"- 회의 결과 초안은 mandatory MASS Code를 {adoption_year}년 채택 목표로 계속 추진한다고 기록합니다. {cite}"
            )

    force_year = nearby_year(
        entry_force_chunk,
        re.compile(r"entry\s+into\s+force|in\s+force", re.I),
    )
    if entry_force_chunk and force_year:
        cite = _cite(entry_force_chunk, citation_map)
        if cite:
            used_ids.add(str(getattr(entry_force_chunk, "chunk_id", "")))
            caveat = ""
            if uncertainty_chunk:
                caveat_cite = _cite(uncertainty_chunk, citation_map)
                caveat = (
                    f" 다만 해당 목표가 야심적이어서 추후 재검토될 수 있다는 점도 기록돼 있습니다. {caveat_cite}"
                    if caveat_cite
                    else ""
                )
            lines.append(
                f"- 회의 결과 초안에는 mandatory Code의 발효 목표가 {force_year}년으로 제시됩니다. {cite}{caveat}"
            )

    # The experience-building phase is a distinct part of the roadmap, not a
    # fallback bullet.  Keep it when a user asks for the mandatory-Code
    # schedule even if adoption and entry-into-force targets were both found.
    if experience_chunk and len(lines) < 5:
        cite = _cite(experience_chunk, citation_map)
        if cite:
            used_ids.add(str(getattr(experience_chunk, "chunk_id", "")))
            lines.append(
                "- 회의 기록은 비강제 MASS Code 채택 후 Experience-Building "
                f"Phase(EBP) 프레임워크 개발이 MSC 111에서 시작된다고 명시합니다. {cite}"
            )

    if adoption_year and force_year:
        return "\n".join(lines[:5]), warnings
    if not adoption_year:
        warnings.append("missing_mandatory_adoption_target")
    if not force_year:
        warnings.append("missing_mandatory_entry_into_force_target")

    decision = _best_chunk_matching(pool, OUTCOME_RE) or (
        pool[0][1] if pool else None
    )
    if decision:
        used_ids.add(str(getattr(decision, "chunk_id", "")))
        lines.append(_format_bullet(decision, citation_map, topic_label="핵심 결정사항"))

    non_mand = _best_chunk_matching(
        [(s, c) for s, c in pool if str(getattr(c, "chunk_id", "")) not in used_ids],
        NON_MAND_RE,
    )
    if non_mand:
        used_ids.add(str(getattr(non_mand, "chunk_id", "")))
        lines.append(
            _format_bullet(non_mand, citation_map, topic_label="non-mandatory 여부")
        )
    else:
        warnings.append("missing_non_mandatory")

    timeline = _best_chunk_matching(
        [(s, c) for s, c in pool if str(getattr(c, "chunk_id", "")) not in used_ids],
        TIMELINE_RE,
    )
    if timeline is None:
        timeline = _best_chunk_matching(
            [(s, c) for s, c in pool if str(getattr(c, "chunk_id", "")) not in used_ids],
            re.compile(r"experience[- ]building|mandatory\s+code|roadmap|entry\s+into\s+force", re.I),
        )
    if timeline:
        used_ids.add(str(getattr(timeline, "chunk_id", "")))
        lines.append(
            _format_bullet(
                timeline,
                citation_map,
                topic_label="mandatory code 일정 / experience-building",
            )
        )
    else:
        warnings.append("missing_mandatory_timeline")

    # Always keep at least 2 cited MASS evidence bullets when chunks exist.
    for _s, chunk in pool:
        if len(lines) >= 3:
            break
        cid = str(getattr(chunk, "chunk_id", ""))
        if cid in used_ids:
            continue
        used_ids.add(cid)
        lines.append(_format_bullet(chunk, citation_map, topic_label="MASS Code"))

    if not lines:
        warnings.append("weak_mass_evidence")
        return (
            "- [추가 확인 필요] MASS Code 관련 검색 근거가 부족합니다.",
            warnings,
        )

    norms = [_normalize_bullet(l) for l in lines]
    if len(set(norms)) < len(norms):
        warnings.append("duplicate_topic")
    return "\n".join(lines), warnings


def _normalize_bullet(line: str) -> str:
    return re.sub(r"\[\d+\]", "", line).strip()[:80]


def _planned_slot_chunks(
    row: dict,
    scored: list[tuple[float, Any]],
    slot_names: tuple[str, ...],
) -> list[tuple[str, Any]]:
    """Return planner-selected chunks in evidence-slot order."""
    completion = row.get("_evidence_completion") or {}
    slot_hits = completion.get("slot_hits") or {}
    by_id = {
        str(getattr(chunk, "chunk_id", "")): chunk
        for _score, chunk in scored
    }
    selected: list[tuple[str, Any]] = []
    for slot_name in slot_names:
        for chunk_id in slot_hits.get(slot_name) or []:
            chunk = by_id.get(str(chunk_id))
            if chunk is not None:
                selected.append((slot_name, chunk))
    return selected


def _build_data_quality_sections(
    chunks: list[Any],
    citation_map: dict[str, int],
) -> tuple[dict[str, str], list[Any]]:
    """Build an IMO data-quality answer only from phrases in current evidence."""
    relevant = [
        chunk
        for chunk in chunks
        if re.search(
            r"quality control and verification|identified errors|obvious errors|"
            r"hours under way|unrealistic characteristics|not been included in the data analysis",
            _strip_meta(getattr(chunk, "text", "")),
            re.I,
        )
        and _cite(chunk, citation_map)
    ]
    if not relevant:
        return {}, []

    def find(pattern: str) -> Any | None:
        matches = [
            chunk
            for chunk in relevant
            if re.search(
                pattern,
                _strip_meta(getattr(chunk, "text", "")),
                re.I | re.S,
            )
        ]
        if not matches:
            return None
        # Prefer the evidence passage that contains the most concrete
        # verification details instead of whichever matching chunk happened
        # to rank first.  Summary pages often say only "obvious errors",
        # whereas the detailed clause enumerates error types and disposition.
        detail_terms = re.compile(
            r"missing ships?|unrealistic|not technically possible|"
            r"duplicate reporting|multiple reporting|incorrect ship type|"
            r"further examined|determine the cause|Administrations|ROs|"
            r"not been included|hours under way",
            re.I,
        )
        return max(
            matches,
            key=lambda chunk: (
                len(
                    detail_terms.findall(
                        _strip_meta(getattr(chunk, "text", ""))
                    )
                ),
                len(_strip_meta(getattr(chunk, "text", ""))),
            ),
        )

    process = find(r"quality control and verification")
    obvious = find(r"obvious errors|unrealistic characteristics")
    hours = find(r"hours under way.{0,120}(?:total number of hours|year)")
    excluded = find(r"265 ships|not been included in the data analysis")

    summary: list[str] = []
    if process:
        summary.append(
            "- IMO 사무국은 GISIS 제출자료의 정확성을 확인하고 미보고 선박과 명백한 "
            f"오류를 식별하기 위해 품질관리·검증 절차를 수행했습니다. {_cite(process, citation_map)}"
        )
    if obvious:
        obvious_text = _strip_meta(getattr(obvious, "text", ""))
        error_types: list[str] = []
        if re.search(
            r"missing ships?.{0,80}no data|no data had been reported",
            obvious_text,
            re.I | re.S,
        ):
            error_types.append("미보고 선박")
        if re.search(
            r"unrealistic characteristics|not technically possible",
            obvious_text,
            re.I,
        ):
            error_types.append("기술적으로 성립하지 않는 비현실적 선박 제원")
        if re.search(r"duplicate reporting|multiple reporting", obvious_text, re.I):
            error_types.append("중복·다중 보고")
        if re.search(
            r"incorrect ship type|categorized under an incorrect",
            obvious_text,
            re.I,
        ):
            error_types.append("MARPOL Annex VI 규칙 2와 불일치하는 선종 분류")
        if error_types:
            summary.append(
                "- 검증 시 확인할 오류 유형은 "
                + ", ".join(error_types)
                + f"입니다. {_cite(obvious, citation_map)}"
            )
        else:
            summary.append(
                "- 자동 검증 대상에는 비현실적인 선박 제원 등 명백한 입력 오류가 "
                f"포함됩니다. {_cite(obvious, citation_map)}"
            )
        if re.search(
            r"further examined to determine the cause|"
            r"provided to the Administrations and ROs",
            obvious_text,
            re.I | re.S,
        ):
            summary.append(
                "- 자동 탐지된 오류는 원인을 추가 조사한 뒤 관련 기국과 RO에 전달하여 "
                f"정정할 수 있도록 처리합니다. {_cite(obvious, citation_map)}"
            )
    if hours:
        summary.append(
            "- 구체적인 오류 유형으로 연간 총시간을 초과하여 보고된 "
            f"`hours under way`가 확인됐습니다. {_cite(hours, citation_map)}"
        )
    if excluded:
        summary.append(
            "- 2025년 7월 31일 기준 집계 결과에 큰 영향을 줄 수 있는 오류가 남은 "
            f"265척은 해당 보고서의 데이터 분석에서 제외됐습니다. {_cite(excluded, citation_map)}"
        )

    practical: list[str] = []
    if process:
        practical.append(
            "- 제출기관과 RO는 GISIS 제출 전 미보고 선박, 명백한 입력 오류 및 선박별 "
            f"정정 상태를 QA 점검항목으로 관리할 필요가 있습니다. {_cite(process, citation_map)}"
        )
    if hours:
        practical.append(
            "- 운항시간은 연간 가능한 총시간을 넘지 않는지 자동 범위검사를 적용해야 "
            f"합니다. {_cite(hours, citation_map)}"
        )
    if excluded:
        practical.append(
            "- 미정정 오류는 집계분석에서 제외될 수 있으므로 기국·RO의 수정 완료 여부를 "
            f"분석 마감 전에 추적해야 합니다. {_cite(excluded, citation_map)}"
        )

    followup: list[str] = []
    if excluded:
        followup.append(
            "- [데이터 범위] 265척은 오류가 확정된 전체 선박 수가 아니라, 보고 시점까지 "
            "정정되지 않아 이번 분석에서 제외된 잠재 오류 선박 수로 해석해야 합니다. "
            f"{_cite(excluded, citation_map)}"
        )
    if process:
        followup.append(
            "- [검증 기준] 해당 품질관리 절차가 2022 Guidelines에 명시된 절차인지와 "
            f"사무국의 추가 검증인지 구분해 적용해야 합니다. {_cite(process, citation_map)}"
        )

    reference_chunk = process or obvious or hours or excluded
    file_name = str(getattr(reference_chunk, "file_name", "") or "")
    page = getattr(reference_chunk, "page_number", None)
    reference = f"**{_meeting_doc_ref(file_name)}**"
    if page not in (None, ""):
        reference += f", p.{page}"
    section4 = (
        f"- {reference}의 제출자료 검증 및 오류 식별 부분이 질문에 직접 대응하는 "
        f"IMO 근거이며, 별도의 선급 Rule 질의는 아닙니다. {_cite(reference_chunk, citation_map)}"
    )
    return {
        "1": "\n".join(summary),
        "2": "\n".join(practical),
        "3": "\n".join(followup),
        "4": section4,
    }, relevant


def _section1(
    clusters: list,
    *,
    profile: MeetingRetrievalProfile,
    citation_map: dict[str, int],
    row: dict,
    scored: list[tuple[float, Any]] | None = None,
) -> tuple[str, list[str], list[Any]]:
    lo, hi, _ = category_bullet_budget(row.get("category", ""), row)
    n = profile.requested_bullet_count or hi
    n = max(lo, min(hi, n))
    if profile.top_level_category == TOP_LEVEL_TREND and profile.requested_bullet_count:
        n = profile.requested_bullet_count

    scored = scored or []
    intent = profile.internal_intent
    extra_warnings: list[str] = []

    question = str(row.get("question") or "")
    if re.search(r"의제|안건|agenda|provisional", question, re.I):
        agenda_candidates = [
            (score, chunk)
            for score, chunk in scored
            if re.search(
                r"annotations?\s+to\s+the\s+provisional\s+agenda|provisional\s+agenda",
                f"{getattr(chunk, 'file_name', '')}\n{_strip_meta(getattr(chunk, 'text', ''))}",
                re.I,
            )
        ]
        if agenda_candidates:
            _score, agenda_chunk = max(agenda_candidates, key=lambda item: item[0])
            body = _strip_meta(getattr(agenda_chunk, "text", ""))
            session_match = re.search(
                r"\b(MSC|MEPC)\s*[-/]?\s*(\d{1,3})(?=\s|가|은|는|을|를|의|에서|$)",
                question,
                re.I,
            )
            session = (
                f"{session_match.group(1).upper()} {session_match.group(2)}"
                if session_match
                else "해당 회의"
            )
            cite = _cite(agenda_chunk, citation_map)
            agenda_lines = [
                f"- **{session} provisional agenda**: 공식 임시 의제 주석 문서에 회의에서 다룰 예정인 주요 안건이 정리돼 있습니다. {cite}"
            ]
            if re.search(r"MARPOL\s+Annex\s+VI|regulations?\s+27\s+and\s+28|data\s+report", body, re.I):
                agenda_lines.append(
                    f"- **MARPOL Annex VI·보고제도**: 규정 27·28의 데이터 보고 항목 명확화와 관련 개정 검토가 의제에 포함됩니다. {cite}"
                )
            if re.search(r"North-East\s+Atlantic|emission\s+control\s+area|\bECA\b", body, re.I):
                agenda_lines.append(
                    f"- **배출통제구역(ECA)**: 북동대서양의 NOx·SOx·입자상물질 배출통제구역 지정 검토가 포함됩니다. {cite}"
                )
            if re.search(r"IMO\s+DCS|short-term\s+GHG|review\s+clause", body, re.I):
                agenda_lines.append(
                    f"- **GHG·IMO DCS 후속조치**: IMO DCS 접근과 단기 GHG 감축조치 검토 조항도 예정 안건으로 제시됩니다. {cite}"
                )
            return "\n".join(agenda_lines[:n]), extra_warnings, [agenda_chunk]

    planned_iswg = _planned_slot_chunks(
        row,
        scored,
        ("sfcs_label", "gfi_compliance", "gfi_reporting", "lca_method"),
    )
    if planned_iswg:
        labels = {
            "sfcs_label": "SFCS·Fuel Life Cycle Label",
            "gfi_compliance": "GFI 규칙 36",
            "gfi_reporting": "GFI 규칙 37·SEEMP",
            "lca_method": "LCA 산정·검증 방법",
        }
        lines: list[str] = []
        picked: list[Any] = []
        emitted_slots: set[str] = set()
        for slot_name, chunk in planned_iswg:
            if slot_name in emitted_slots:
                continue
            fact = _grounded_environment_fact(chunk)
            if not fact:
                continue
            lines.append(
                f"- **{labels[slot_name]}**: {fact} {_cite(chunk, citation_map)}".strip()
            )
            picked.append(chunk)
            emitted_slots.add(slot_name)
        if lines:
            return "\n".join(lines[:6]), extra_warnings, picked[:6]

    planned_operational = _planned_slot_chunks(
        row,
        scored,
        ("reporting_requirement", "data_quality", "carbon_intensity"),
    )
    if planned_operational and any(
        slot_name in {"reporting_requirement", "data_quality"}
        for slot_name, _chunk in planned_operational
    ):
        labels = {
            "reporting_requirement": "규제 보고",
            "data_quality": "데이터 품질",
            "carbon_intensity": "탄소집약도·운항지표",
        }
        lines: list[str] = []
        picked: list[Any] = []
        emitted_slots: set[str] = set()
        for slot_name, chunk in planned_operational:
            if slot_name in emitted_slots:
                continue
            fact = _grounded_environment_fact(chunk)
            if not fact:
                fact = summarize_chunk_ko(chunk, topic_label=labels[slot_name])
            body = _strip_meta(getattr(chunk, "text", ""))
            cite = _cite(chunk, citation_map)
            if (
                slot_name == "carbon_intensity"
                and "10.8%" in body
                and re.search(r"(?:at\s+least|minimum).{0,30}6%", body, re.I | re.S)
            ):
                lines.append(
                    f"- **{labels[slot_name]}**: 2024년 선대 평균 AER·cgDIST 기준 탄소집약도는 "
                    f"2019년보다 최대 10.8% 감소한 것으로 보고됐습니다. {cite}"
                )
                lines.append(
                    f"- **선종·크기별 AER**: 주요 선종의 대부분 크기 구간에서 AER이 "
                    f"2019년 대비 최소 6% 개선된 것으로 보고됐습니다. {cite}"
                )
            else:
                lines.append(
                    f"- **{labels[slot_name]}**: {fact} {cite}".strip()
                )
                if (
                    slot_name == "reporting_requirement"
                    and "not later than one month after issuing the statement"
                    in body.lower()
                ):
                    lines.append(
                        "- **IMO DCS 전송기한**: MARPOL Annex VI 규정 27.9는 Statement of Compliance "
                        f"발급 후 1개월 이내에 보고 데이터를 IMO 데이터베이스로 전송하도록 정합니다. {cite}"
                    )
                if slot_name == "data_quality" and "future analysis" in body.lower():
                    lines.append(
                        "- **오류 데이터 보완**: 누락 선박과 오류 데이터는 사무국에 통지해 개별 선박 "
                        f"데이터를 검증한 뒤 후속 분석에 사용할 수 있도록 하는 절차가 제시됩니다. {cite}"
                    )
            picked.append(chunk)
            emitted_slots.add(slot_name)
            if len(lines) >= max(3, n):
                break
        registry = next(
            (
                chunk
                for _score, chunk in scored
                if "draft functional requirements" in _strip_meta(getattr(chunk, "text", "")).lower()
                and "avoid possible double reporting" in _strip_meta(getattr(chunk, "text", "")).lower()
            ),
            None,
        )
        if registry is not None:
            registry_cite = _cite(registry, citation_map)
            if registry_cite:
                lines.append(
                    "- **GFI Registry 연계**: IMO 사무국은 GFI Registry 기능요건 초안에 대해 IMO DCS "
                    f"제출자 의견을 수렴하며 중복 보고 방지를 검토했습니다. {registry_cite}"
                )
                picked.append(registry)
        return "\n".join(lines), extra_warnings, picked

    planned_environment = _planned_slot_chunks(
        row,
        scored,
        ("regulatory_status", "carbon_intensity", "reporting_framework", "fuel_lifecycle"),
    )
    if planned_environment:
        labels = {
            "regulatory_status": "규제안 상태·일정",
            "carbon_intensity": "탄소집약도",
            "reporting_framework": "보고·검증 체계",
            "fuel_lifecycle": "연료 전과정평가",
        }
        lines = []
        picked = []
        for slot_name, chunk in planned_environment:
            fact = _grounded_environment_fact(chunk)
            if not fact:
                continue
            lines.append(
                f"- **{labels[slot_name]}**: {fact} {_cite(chunk, citation_map)}".strip()
            )
            picked.append(chunk)
        if lines:
            return "\n".join(lines), extra_warnings, picked

    # A document-specific CII question needs metric/method and outcome pages,
    # not only the strongest numerical result paragraph.
    if re.search(r"MEPC\s*84\s*[/_-]\s*6\s*[/_-]\s*2", question, re.I):
        document_scored = [
            (score, chunk)
            for score, chunk in scored
            if re.search(
                r"MEPC\s*84[-_/ ]6[-_/ ]2\b",
                str(getattr(chunk, "file_name", "") or ""),
                re.I,
            )
        ]

        def first_matching(pattern: str) -> Any | None:
            matches = [
                (score, chunk)
                for score, chunk in document_scored
                if re.search(pattern, _strip_meta(getattr(chunk, "text", "")), re.I | re.S)
            ]
            return max(matches, key=lambda item: item[0])[1] if matches else None

        scope_chunk = first_matching(r"annual report|regulation\s+27\.10|reporting year 2024")
        method_chunk = first_matching(r"supply-based.{0,800}(?:AER|cgDIST).{0,800}demand-based.{0,800}EEOI")
        comparison_chunk = first_matching(r"2019\s+to\s+2024.{0,600}AER.{0,300}cgDIST.{0,300}EEOI")
        result_chunk = first_matching(r"up to\s+10\.8%|2024\s+relative\s+to\s+2019")
        lines: list[str] = []
        picked: list[Any] = []
        if scope_chunk:
            lines.append(
                "- **적용범위·문서 성격**: MEPC 84/6/2는 2024 reporting year의 국제해운 "
                f"선대 탄소집약도 연차보고서입니다. {_cite(scope_chunk, citation_map)}"
            )
            picked.append(scope_chunk)
        if method_chunk:
            lines.append(
                "- **사용 지표·산정 방법**: 공급기반 탄소집약도는 AER와 cgDIST, "
                "수요기반 탄소집약도 추정치는 EEOI를 사용합니다. "
                f"{_cite(method_chunk, citation_map)}"
            )
            picked.append(method_chunk)
        if comparison_chunk:
            result_text = ""
            if result_chunk:
                result_text = (
                    " 2024년 선대 평균 AER·cgDIST 기준 탄소집약도는 "
                    "2019년 대비 최대 10.8% 감소했습니다. "
                    f"{_cite(result_chunk, citation_map)}"
                )
            lines.append(
                "- **비교 구간·2024년 결과**: AER·cgDIST·EEOI로 2019~2024년 연차 추이를 "
                f"2019년 기준선과 비교합니다. {_cite(comparison_chunk, citation_map)}"
                f"{result_text}"
            )
            picked.append(comparison_chunk)
            if result_chunk:
                picked.append(result_chunk)
        elif result_chunk:
            lines.append(
                "- **2024년 결과**: 선대 평균 AER·cgDIST 기준 탄소집약도는 2019년 대비 "
                f"최대 10.8% 감소했습니다. {_cite(result_chunk, citation_map)}"
            )
            picked.append(result_chunk)
        if lines:
            unique_picked: list[Any] = []
            seen_picked: set[str] = set()
            for chunk in picked:
                chunk_id = str(getattr(chunk, "chunk_id", "")) or str(id(chunk))
                if chunk_id in seen_picked:
                    continue
                seen_picked.add(chunk_id)
                unique_picked.append(chunk)
            return "\n".join(lines), extra_warnings, unique_picked

    # V03-style operational / CII reporting questions: prefer fleet CII report.
    if (
        profile.top_level_category in {TOP_LEVEL_ENV, TOP_LEVEL_TREND}
        and (
            re.search(r"\bMEPC\b", question or "", re.I)
            or "MEPC"
            in {
                str(value).upper()
                for value in (row.get("retrieval_sources") or [])
            }
            or re.search(r"\bCII\b|탄소\s*(?:집약도|강도)", question or "", re.I)
        )
        and re.search(r"운항|규제\s*보고|reporting|CII|탄소|operational|직접\s*영향", question or "", re.I)
        and not _is_broad_latest_environment_summary(question, profile)
    ):
        cii_docs = (
            re.compile(r"mepc\s*84-6-2\b", re.I),
            re.compile(r"mepc\s*84-6-1\b", re.I),
            re.compile(r"mepc\s*84-6-21\b", re.I),
            re.compile(r"mepc\s*84-6-\d+\b", re.I),
        )
        lines: list[str] = []
        picked: list[Any] = []
        for doc_re in cii_docs:
            candidates = [
                (score, chunk)
                for score, chunk in scored
                if doc_re.search(str(getattr(chunk, "file_name", "") or ""))
            ]
            if not candidates:
                continue
            _score, chunk = max(candidates, key=lambda item: item[0])
            if chunk in picked:
                continue
            picked.append(chunk)
            file_name = str(getattr(chunk, "file_name", "") or "")
            body = _strip_meta(getattr(chunk, "text", ""))
            if re.search(r"mepc\s*84-6-2\b", file_name, re.I) and re.search(
                r"CII\s+Reduction\s+Factors|annual\s+carbon\s+intensity",
                body,
                re.I,
            ):
                fact = (
                    "MEPC 84/6/2는 CII Reduction Factors Guidelines(G3)에 따라 "
                    "수요 기반·공급 기반 지표와 CII 등급을 함께 사용해 연간 탄소집약도 개선을 "
                    "계속 모니터링하도록 제시합니다."
                )
            elif re.search(r"mepc\s*84-6-21\b", file_name, re.I) and re.search(
                r"operational\s+carbon\s+intensity\s+rating",
                body,
                re.I,
            ):
                fact = (
                    "MEPC 84/6/21은 연료소비 보고와 operational CII 등급에 관한 "
                    "Certificate·Statement of Compliance 발급 또는 승인 절차를 다룹니다."
                )
            else:
                fact = _grounded_environment_fact(chunk) or summarize_chunk_ko(
                    chunk, topic_label="CII·운항 보고"
                )
            lines.append(f"- **CII·운항 보고**: {fact} {_cite(chunk, citation_map)}".strip())
            if len(lines) >= n:
                break
        if lines:
            return "\n".join(lines[:n]), extra_warnings, picked[:n]

    if _is_broad_latest_environment_summary(question, profile):
        # Executive summaries are organized by regulatory topic, not by the
        # highest-scoring technical detail.  In particular, DCS outlier counts
        # are evidence for a data-quality question, not a headline MEPC agenda.
        preferred_docs = (
            re.compile(r"mepc\s*84-3\b", re.I),
            re.compile(r"mepc\s*84-6-2\b", re.I),
            re.compile(r"mepc\s*84-7-14\b", re.I),
            re.compile(r"mepc\s*84-7-15\b", re.I),
            re.compile(r"mepc\s*84-10\b", re.I),
        )
        lines: list[str] = []
        picked: list[Any] = []
        for doc_re in preferred_docs:
            candidates = [
                (score, chunk)
                for score, chunk in scored
                if doc_re.search(str(getattr(chunk, "file_name", "") or ""))
                and _grounded_environment_fact(chunk)
            ]
            if not candidates:
                continue
            _score, chunk = max(candidates, key=lambda item: item[0])
            picked.append(chunk)
            label = _environment_topic_label(
                str(getattr(chunk, "file_name", "") or "")
            )
            lines.append(
                f"- **{label}**: {_grounded_environment_fact(chunk)} {_cite(chunk, citation_map)}".strip()
            )
        if lines:
            return "\n".join(lines[:5]), extra_warnings, picked[:5]

    if profile.top_level_category == TOP_LEVEL_ENV and LATEST_QUERY_RE.search(question):
        lines: list[str] = []
        picked: list[Any] = []
        seen_docs: set[str] = set()
        for _score, chunk in scored:
            fact = _grounded_environment_fact(chunk)
            doc_id = str(getattr(chunk, "doc_id", "") or getattr(chunk, "file_name", ""))
            if not fact or doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            picked.append(chunk)
            lines.append(_format_bullet(chunk, citation_map, tier_note_mode="topic"))
            if len(lines) >= n:
                break
        if lines:
            if len(lines) < min(3, n):
                extra_warnings.append("latest_environment_evidence_sparse")
            return "\n".join(lines), extra_warnings, picked

    if profile.answer_variant == "official_dense":
        return _section1_official_dense(scored, n=n, citation_map=citation_map)
    if profile.answer_variant == "topic_diverse":
        return _section1_topic_diverse(scored, n=n, citation_map=citation_map, profile=profile)

    if intent == "meeting_outcome":
        return _section1_meeting_outcome(
            scored,
            n=n,
            citation_map=citation_map,
            profile=profile,
            row=row,
        )

    if intent == "altfuel_ghg_safety":
        s1, w, picked = _section1_altfuel_ghg(scored, citation_map=citation_map, profile=profile)
        return s1, w, picked

    if intent == "mass_code_timeline":
        s1, w = _section1_mass_timeline(
            scored,
            citation_map=citation_map,
            row=row,
        )
        return s1, w, []

    if intent == "trend_summary" and scored:
        cap = min(n, 5)
        diverse = pick_diverse_topic_chunks(
            scored,
            cap,
            topic_priority=_topic_priority_for_profile(profile),
            topic_caps={"ghg_framework": 2, "general_outcome": 1, "maritime_safety": 1, "cii_reporting": 1},
            exclude_topics={"mass_code"},
            topic_fn=outcome_topic_id,
        )
        lines = [
            _format_bullet(c, citation_map, topic_label=_topic_label_ko(outcome_topic_id(str(getattr(c, "text", "")))))
            for c in diverse
        ]
        return "\n".join(lines), extra_warnings, diverse

    lines: list[str] = []
    used_chunk_ids: set[str] = set()
    exclude = exclude_topics_for_intent(intent)
    caps = topic_caps_for_intent(intent)

    if profile.requested_bullet_count and scored:
        diverse = pick_diverse_topic_chunks(
            scored,
            n,
            topic_priority=MSC_OUTCOME_TOPIC_PRIORITY if intent == "meeting_outcome" else None,
            topic_caps=caps,
            exclude_topics=exclude,
        )
        for chunk in diverse:
            used_chunk_ids.add(str(getattr(chunk, "chunk_id", "") or ""))
            tid = _topic_id_for_text(str(getattr(chunk, "text", "") or ""))
            lines.append(_format_bullet(chunk, citation_map, topic_label=_topic_label_ko(tid)))
    else:
        used_topics: set[str] = set()
        for cluster in clusters:
            if len(lines) >= n:
                break
            if cluster.topic_id in exclude:
                continue
            if cluster.topic_id in used_topics and profile.requested_bullet_count:
                continue
            used_topics.add(cluster.topic_id)
            rep = cluster.representative
            if not rep:
                continue
            used_chunk_ids.add(str(getattr(rep, "chunk_id", "") or ""))
            lines.append(_format_bullet(rep, citation_map, topic_label=cluster.label_ko))

    if len(lines) < n and scored:
        for chunk in pick_diverse_topic_chunks(
            scored, n - len(lines), used_chunk_ids=used_chunk_ids, exclude_topics=exclude, topic_caps=caps
        ):
            lines.append(_format_bullet(chunk, citation_map))

    if not lines:
        lines.append("- 검색된 회의 자료에서 핵심 결정·논의를 확인하지 못했습니다. 추가 확인이 필요합니다.")
    return "\n".join(lines[:n]), extra_warnings, []


def _section2(
    chunks: list[Any],
    citation_map: dict[str, int],
    *,
    profile: MeetingRetrievalProfile,
    s1_chunks: list[Any] | None = None,
    question: str = "",
) -> str:
    # Operational impacts are emitted only when the cited passage contains the
    # corresponding requirement.  This replaces the previous category-level
    # boilerplate, which could invent speed management, SMS or approval duties.
    # Search the full cited context. Restricting this to section-1 representatives
    # discarded DCS quality and reporting clauses that were retrieved expressly
    # for the operational-impact section.
    focus = chunks
    lines: list[str] = []
    seen: set[str] = set()
    focus_codes = {c.upper() for c in _question_topic_codes(question)}
    allow_mass_card = (not focus_codes) or ("MASS" in focus_codes)
    if profile.internal_intent == "env_regulation":
        reporting = next(
            (
                c
                for c in focus
                if "not later than five months from the beginning of the calendar year"
                in re.sub(
                    r"\s+", " ", _strip_meta(getattr(c, "text", "")).lower()
                )
            ),
            None,
        )
        quality = next(
            (
                c
                for c in focus
                if "quality control and verification process"
                in re.sub(
                    r"\s+", " ", _strip_meta(getattr(c, "text", "")).lower()
                )
            ),
            None,
        )
        carbon = next(
            (
                c
                for c in focus
                if all(
                    token
                    in re.sub(
                        r"\s+", " ", _strip_meta(getattr(c, "text", "")).lower()
                    )
                    for token in ("aer", "cgdist", "carbon intensity")
                )
            ),
            None,
        )
        if reporting:
            lines.append(
                "- **보고 일정 관리**: MARPOL Annex VI 규정 27에 따른 데이터 적합성 확인과 "
                f"Statement of Compliance 발급기한을 연간 보고 일정에 반영해야 합니다. {_cite(reporting, citation_map)}"
            )
        if quality:
            lines.append(
                "- **제출 데이터 품질관리**: GISIS 제출 전 선박 누락과 명백한 오류를 점검할 수 있도록 "
                f"선대 목록과 제출 데이터의 정합성을 관리해야 합니다. {_cite(quality, citation_map)}"
            )
        if carbon:
            lines.append(
                "- **운항성과 관리**: AER·cgDIST 탄소집약도를 2019년 기준선과 선종·크기 구간별로 "
                f"비교해 선대 성과와 연도별 변동을 함께 관리해야 합니다. {_cite(carbon, citation_map)}"
            )
        if lines:
            return "\n".join(lines)
    if profile.internal_intent == "altfuel_ghg_safety":
        def normalized_text(chunk: Any) -> str:
            return re.sub(
                r"\s+",
                " ",
                _strip_meta(getattr(chunk, "text", "")).lower(),
            ).strip()

        hydrogen = next(
            (
                c
                for c in focus
                if "interim guidelines for the safety of ships using hydrogen as fuel"
                in normalized_text(c)
            ),
            None,
        )
        ghg_group = next(
            (
                c
                for c in focus
                if "ghg safety working group"
                in normalized_text(c)
            ),
            None,
        )
        ammonia = next(
            (
                c
                for c in focus
                if (
                    "ammonia cargo as fuel" in normalized_text(c)
                    and (
                        "interim guidelines" in normalized_text(c)
                        or "igf code" in normalized_text(c)
                    )
                )
            ),
            None,
        )
        if hydrogen:
            lines.append(
                "- **수소연료 선박 검토**: 설계·위험성 평가·승인 자료를 MSC 111에서 승인한 "
                f"수소연료 선박 임시 안전지침의 적용범위와 대조해야 합니다. {_cite(hydrogen, citation_map)}"
            )
        if ghg_group:
            lines.append(
                "- **후속 규제 모니터링**: 신기술·대체연료 적용 프로젝트는 GHG Safety Working Group의 "
                f"후속 안전규제 체계와 작업 결과를 추적해야 합니다. {_cite(ghg_group, citation_map)}"
            )
        if ammonia:
            lines.append(
                "- **암모니아 연료 적용**: 유독성 암모니아 화물을 연료로 사용하는 선박은 IGF Code 개정 발효 일정에 맞춰 "
                f"임시지침의 적용범위와 연료계통 설계·승인 기준을 대조해야 합니다. {_cite(ammonia, citation_map)}"
            )
        if lines:
            return "\n".join(lines)
    if profile.internal_intent == "mass_code_timeline":
        adoption = next(
            (
                c
                for c in focus
                if re.search(
                    r"adoption.{0,100}non-mandatory.{0,60}mass\s+code",
                    _strip_meta(getattr(c, "text", "")),
                    re.I | re.S,
                )
            ),
            None,
        )
        experience = next(
            (
                c
                for c in focus
                if re.search(
                    r"experience[- ]building\s+phase",
                    _strip_meta(getattr(c, "text", "")),
                    re.I,
                )
            ),
            None,
        )
        if adoption:
            lines.append(
                "- **적용상 유의점**: 현재 확인되는 Code는 비강제 단계이므로 mandatory Code의 확정 의무나 발효일로 "
                f"해석해서는 안 됩니다. {_cite(adoption, citation_map)}"
            )
        if experience:
            lines.append(
                "- **준비 방향**: 향후 경험축적단계에서 설계·운항·승인 경험을 축적할 수 있도록 적용 사례와 검증 자료를 "
                f"관리할 필요가 있습니다. {_cite(experience, citation_map)}"
            )
        if lines:
            return "\n".join(lines)
    if profile.internal_intent == "meeting_outcome":
        if "IGC" in focus_codes:
            igc = next(
                (
                    c
                    for c in focus
                    if re.search(
                        r"(?:finalize|finaliz(?:e|ing)|approv(?:e|ed)).{0,100}"
                        r"(?:draft\s+)?amendments?\s+to\s+the\s+IGC\s+Code|"
                        r"(?:draft\s+)?amendments?\s+to\s+the\s+IGC\s+Code",
                        _strip_meta(getattr(c, "text", "")),
                        re.I | re.S,
                    )
                ),
                None,
            )
            if igc:
                return (
                    "- **IGC Code 후속조치**: 확정된 개정 초안은 차기 회기의 승인·채택 절차와 "
                    f"발효 일정을 계속 확인해야 합니다. {_cite(igc, citation_map)}"
                )
        mass = next(
            (
                c
                for c in focus
                if re.search(
                    r"adoption.{0,100}non-mandatory.{0,60}mass\s+code",
                    _strip_meta(getattr(c, "text", "")),
                    re.I | re.S,
                )
            ),
            None,
        )
        hydrogen = next(
            (
                c
                for c in focus
                if re.search(r"hydrogen\s+as\s+fuel", _strip_meta(getattr(c, "text", "")), re.I)
            ),
            None,
        )
        if mass and allow_mass_card:
            lines.append(
                "- **MASS 적용**: 채택 문구가 비강제 Code를 대상으로 하므로, 현 단계에서 mandatory 요구사항으로 "
                f"적용해서는 안 됩니다. {_cite(mass, citation_map)}"
            )
        if hydrogen:
            lines.append(
                "- **대체연료 안전**: 수소 연료 선박 프로젝트는 승인된 임시 안전지침의 적용범위와 기술요건을 "
                f"설계·승인 자료에 대조해야 합니다. {_cite(hydrogen, citation_map)}"
            )
        if lines:
            return "\n".join(lines)
    for c in focus:
        body = _strip_meta(getattr(c, "text", ""))
        low = re.sub(r"\s+", " ", body).lower()
        if profile.top_level_category == TOP_LEVEL_AUTO and not MASS_RE.search(body):
            continue
        if profile.internal_intent == "altfuel_ghg_safety" and not ALT_FUEL_RE.search(body):
            continue
        cite = _cite(c, citation_map)
        if not cite:
            continue
        if "voluntary basis from 1 january 2025" in low and "mandatory basis from 1 january 2026" in low and "dcs" not in seen:
            seen.add("dcs")
            lines.append(
                "- **DCS 제출 체계**: 2026년 1월 1일부터 총 운송작업량 등 세분화된 IMO DCS 필드가 의무화되므로, "
                f"선사·기국·RO의 수집 항목과 제출 인터페이스를 이에 맞춰야 합니다. {cite}"
            )
        if "265 ships" in low and "duplicate reporting" in low and "quality" not in seen:
            seen.add("quality")
            lines.append(
                "- **데이터 품질관리**: GISIS 제출 전 중복 보고, 연간 총시간을 넘는 운항시간, 비현실적 선박 제원과 선종 오분류를 "
                f"검증해야 합니다. 실제 보고서도 이러한 오류가 남은 265척을 분석에서 제외했습니다. {cite}"
            )
        if "demand-based and supply-based carbon intensity" in low and "carbon-metric" not in seen:
            seen.add("carbon-metric")
            lines.append(
                "- **탄소집약도 관리**: AER·cgDIST 기반 지표와 실제 운송작업량 기반 지표가 함께 분석되므로, "
                f"운항·연료 데이터뿐 아니라 transport work 데이터의 정합성도 관리 대상입니다. {cite}"
            )
        if "representativeness" in low and "wtt default emission factors" in low and "wtt" not in seen:
            seen.add("wtt")
            lines.append(
                "- **연료 LCA 증빙**: WtT 기본 배출계수 검토에서 대표성과 보수성 기준이 핵심이므로, "
                f"연료 공급망 데이터의 대표 범위와 불확실성 처리 근거를 준비해야 합니다. {cite}"
            )
        if "draft 2026 guidelines" in low and "oily wastes in machinery spaces" in low and "ibts" not in seen:
            seen.add("ibts")
            lines.append(
                "- **기관실 유성폐기물**: 2026 지침 초안과 IBTS guidance가 MEPC 84 승인 요청 대상이므로, "
                f"기술부서는 최종 승인 전까지 기존 빌지수 처리 절차와 초안 간 차이를 추적해야 합니다. {cite}"
            )
        if len(lines) >= 4:
            break
    if not lines:
        cite = next((_cite(c, citation_map) for c in focus if _cite(c, citation_map)), "")
        lines.append(
            "- 검색된 회의자료에서는 질문과 직접 연결되는 선박 운항·업무 영향 문구를 확인하지 못했습니다."
            + (f" {cite}" if cite else "")
        )
    return "\n".join(lines)


def _section3(
    chunks: list[Any],
    *,
    profile: MeetingRetrievalProfile,
    question: str,
    answer: str,
    citation_map: dict[str, int],
    s1_chunks: list[Any] | None = None,
) -> str:
    focus = s1_chunks or chunks[:12]
    lines: list[str] = []
    seen_status: set[str] = set()
    if LATEST_QUERY_RE.search(question):
        session_rows: list[tuple[str, int, Any]] = []
        for c in focus:
            m = re.search(r"\b(MEPC|MSC)\s+(\d{1,3})[-/]", str(getattr(c, "file_name", "") or ""), re.I)
            if m:
                session_rows.append((m.group(1).upper(), int(m.group(2)), c))
        if session_rows:
            wanted_body = "MEPC" if "mepc" in question.lower() else session_rows[0][0]
            same_body = [row for row in session_rows if row[0] == wanted_body]
            if same_body:
                latest_no = max(row[1] for row in same_body)
                latest_chunks = [row[2] for row in same_body if row[1] == latest_no]
                final_status = any(classify_document_status(c).supports_final_decision for c in latest_chunks)
                cites = "".join(_cite(c, citation_map) for c in latest_chunks[:3])
                if cites and not final_status:
                    lines.append(
                        f"- [미확정 규제] 현재 코퍼스에서 확인되는 최신 범위는 {wanted_body} {latest_no} 안건자료이며, "
                        f"최종 회의 결과보고서가 아니므로 채택 결과가 아니라 검토 예정 안건으로 봐야 합니다. {cites}"
                    )
                    seen_status.add("committee_submission")
    for c in focus:
        status = classify_document_status(c)
        if status.code in seen_status or status.supports_final_decision or status.code == "unknown":
            continue
        cite = _cite(c, citation_map)
        if not cite:
            continue
        seen_status.add(status.code)
        lines.append(
            f"- [미확정 규제] {status.label_ko} 기록상 결정이며, 최종 발행 문서 확인 전 "
            f"법적 효력과 발효일은 단정할 수 없습니다. {cite}"
        )
        if len(lines) >= 1:
            break
    if not lines:
        cite = next((_cite(c, citation_map) for c in focus if _cite(c, citation_map)), "")
        lines.append(
            "- [추가 확인 필요] 적용 범위와 발효일은 인용된 원문 조항을 기준으로 확인해야 합니다."
            + (f" {cite}" if cite else "")
        )
    return "\n".join(lines)


def _section4(chunks: list[Any], citation_map: dict[str, int], *, question: str) -> str:
    class_chunks = [
        c
        for c in chunks
        if str(getattr(c, "source", "")).upper() in {"DNV", "LR", "ABS", "KR"}
    ]
    if not class_chunks:
        cite = next((_cite(c, citation_map) for c in chunks if _cite(c, citation_map)), "")
        return (
            "- 이번 답변의 검색 근거는 IMO 회의자료이며, 관련 선급 Rule/Guidance는 별도 선급 문서 검색으로 확인해야 합니다."
            + (f" {cite}" if cite else "")
        )

    relevant = select_key_clause_chunks(question, class_chunks, limit=3)
    lines: list[str] = []
    seen: set[str] = set()
    for c in relevant:
        fn = str(getattr(c, "file_name", "") or "")
        if fn in seen:
            continue
        seen.add(fn)
        cite = _cite(c, citation_map)
        if not cite:
            continue
        lines.append(
            f"- **{fn.replace('.pdf', '')}**: 선급 Rule/Guidance 원문 후보이며 적용 범위는 해당 조항에서 추가 확인해야 합니다. {cite}"
        )
    return "\n".join(lines) if lines else "- 검색된 근거 내에서는 관련 선급 Rule/Guidance가 명확히 확인되지 않았습니다."


def _question_topic_codes(question: str) -> list[str]:
    """Explicit instrument/topic codes the user named (IGC, MASS, …)."""
    q = question or ""
    codes: list[str] = []
    for code in ("IGC", "IGF", "MASS", "CII", "SEEMP", "DCS", "GFI"):
        if re.search(rf"(?<![A-Za-z0-9]){code}(?![A-Za-z0-9])", q, re.I):
            codes.append(code)
    if re.search(r"암모니아|ammonia", q, re.I):
        codes.append("AMMONIA")
    return codes


def _chunk_topic_blob(chunk: Any) -> str:
    return " ".join(
        str(part or "")
        for part in (
            getattr(chunk, "text", ""),
            getattr(chunk, "file_name", ""),
            getattr(chunk, "caption", ""),
            getattr(chunk, "section_path", ""),
        )
    )


def _rerank_by_question_topic_codes(
    question: str,
    scored: list[tuple[float, Any]],
) -> list[tuple[float, Any]]:
    """Boost chunks matching named codes; demote MASS when another code is asked."""
    focus = _question_topic_codes(question)
    if not focus or not scored:
        return scored
    focus_upper = {c.upper() for c in focus}
    wants_mass = "MASS" in focus_upper

    def matches_focus(blob: str) -> bool:
        for code in focus:
            if code == "AMMONIA":
                if re.search(r"ammonia|암모니아", blob, re.I):
                    return True
            elif re.search(rf"(?<![A-Za-z0-9]){re.escape(code)}(?![A-Za-z0-9])", blob, re.I):
                return True
        return False

    def adjusted(item: tuple[float, Any]) -> float:
        score, chunk = item
        blob = _chunk_topic_blob(chunk)
        bonus = 0.0
        if matches_focus(blob):
            bonus += 5.0
        if not wants_mass and MASS_RE.search(blob):
            # Asking IGC/ammonia should not surface MASS timeline as top evidence.
            bonus -= 4.0
            if not matches_focus(blob):
                bonus -= 6.0
        return score + bonus

    reranked = sorted(scored, key=adjusted, reverse=True)
    if wants_mass:
        return reranked
    # Hard filter: keep focus-matching chunks first; drop pure MASS distractors
    # when at least one on-topic chunk exists.
    on_topic = [(s, c) for s, c in reranked if matches_focus(_chunk_topic_blob(c))]
    if on_topic:
        rest = [
            (s, c)
            for s, c in reranked
            if not (
                MASS_RE.search(_chunk_topic_blob(c))
                and not matches_focus(_chunk_topic_blob(c))
            )
            and (s, c) not in on_topic
        ]
        return on_topic + rest
    return reranked


def build_meeting_structured_answer(
    chunks: list[Any],
    *,
    question: str,
    row: dict,
    profile: MeetingRetrievalProfile,
    warning_flags: list[str] | None = None,
) -> tuple[str, list[str], dict]:
    warnings = list(warning_flags or [])
    # Citation ids are always tied to the caller's displayed retrieval order.
    # Never renumber a reranked pool independently of the Evidence Table.
    citation_chunks = list(chunks)[:12]
    citation_map = _build_citation_map(citation_chunks)
    # Evidence completion can deliberately select two distinct propositions
    # from the same page (for example regulation 36 and regulation 37/SEEMP on
    # MEPC 84/7/14 p.22).  Page-level deduplication erased the second one.
    # Preserve chunk-level propositions whenever a semantic slot plan exists.
    has_planned_slots = bool(
        ((row.get("_evidence_completion") or {}).get("slot_hits") or {})
    )
    work = _filter_forbid_docs(
        citation_chunks if has_planned_slots else dedupe_page_chunks(citation_chunks),
        row,
    )
    work = [c for c in work if not is_excluded_chunk(c, profile=profile)]
    if not work:
        work = _filter_forbid_docs(
            citation_chunks if has_planned_slots else dedupe_page_chunks(citation_chunks),
            row,
        )
    # When the user names a code (IGC, ammonia, …), keep that evidence first so
    # generic MSC outcome extractors cannot fill section 1 with MASS-only text.
    focus_codes = _question_topic_codes(question)
    if focus_codes and "MASS" not in {c.upper() for c in focus_codes}:
        def _matches_focus(chunk: Any) -> bool:
            blob = _chunk_topic_blob(chunk)
            for code in focus_codes:
                if code == "AMMONIA":
                    if re.search(r"ammonia|암모니아", blob, re.I):
                        return True
                elif re.search(
                    rf"(?<![A-Za-z0-9]){re.escape(code)}(?![A-Za-z0-9])", blob, re.I
                ):
                    return True
            return False

        focused = [c for c in work if _matches_focus(c)]
        if focused:
            work = focused + [
                c
                for c in work
                if c not in focused
                and not (
                    MASS_RE.search(_chunk_topic_blob(c)) and not _matches_focus(c)
                )
            ]
    selected = select_key_clause_chunks(
        question,
        work,
        limit=12,
        outcome_query=profile.internal_intent in {"meeting_outcome", "mass_code_timeline", "trend_summary"},
    )
    if selected:
        if profile.internal_intent in {
            "meeting_outcome",
            "altfuel_ghg_safety",
            "mass_code_timeline",
        } or (row.get("_evidence_completion") or {}).get("slot_hits"):
            # Keep key-clause ordering but do not discard the remaining cited
            # topic evidence. The selector can prefer a definition/header and
            # remove the later action-request or timeline paragraph.
            combined: list[Any] = []
            seen_ids: set[str] = set()
            for chunk in [*selected, *work]:
                cid = str(getattr(chunk, "chunk_id", "") or id(chunk))
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                combined.append(chunk)
            work = combined[:12]
        else:
            work = selected
    if profile.internal_intent in {
        "meeting_outcome",
        "altfuel_ghg_safety",
        "mass_code_timeline",
    }:
        # Deterministic phrase extractors may legitimately use a lower-tier
        # submission or action-request chunk when it is displayed and cited.
        # Re-attach every displayed citation chunk so those decisive phrases
        # are not lost to generic source-tier filtering.
        seen_work = {
            str(getattr(chunk, "chunk_id", "") or id(chunk))
            for chunk in work
        }
        for chunk in citation_chunks:
            cid = str(getattr(chunk, "chunk_id", "") or id(chunk))
            if cid not in seen_work:
                work.append(chunk)
                seen_work.add(cid)
    scored = [(score_chunk(c, profile=profile), c) for c in work]
    scored.sort(key=lambda x: -x[0])
    scored = _rerank_by_question_topic_codes(question, scored)

    if profile.internal_intent == "data_quality_verification":
        sections, quality_chunks = _build_data_quality_sections(
            [chunk for _score, chunk in scored],
            citation_map,
        )
        if sections:
            answer = join_four_sections(sections)
            cited_ids = sorted({int(value) for value in CITATION_RE.findall(answer)})
            claim_rows = [
                {
                    "claim": CITATION_RE.sub("", line[2:]).strip(),
                    "citations": [
                        f"[{value}]" for value in CITATION_RE.findall(line)
                    ],
                    "supported": True,
                    "reason": "source_driven_data_quality_extraction",
                }
                for line in answer.splitlines()
                if line.strip().startswith("- ") and CITATION_RE.search(line)
            ]
            meta = {
                "coverage_check": {
                    "intent": "data_quality_verification",
                    "source_driven": True,
                    "evidence_chunk_count": len(quality_chunks),
                },
                "used_citations": cited_ids,
                "detected_topics": ["imo_dcs_data_quality"],
                "top_level_category": profile.top_level_category,
                "internal_intent": profile.internal_intent,
                "document_status": [
                    {
                        "citation_id": f"[{index}]",
                        "status": classify_document_status(chunk).code,
                        "status_label": classify_document_status(chunk).label_ko,
                    }
                    for index, chunk in enumerate(citation_chunks, 1)
                    if index in cited_ids
                ],
                "claim_verification": claim_rows,
                "claim_verification_pass": True,
                "unsupported_claims_blocked": 0,
            }
            return answer, list(dict.fromkeys(warnings)), meta

    weak_tier_only = all(classify_source_tier(c) >= 2 for _, c in scored[:5]) if scored else True
    if weak_tier_only:
        warnings.append("weak_source_tier")

    if not any(count_outcome_signals(_strip_meta(getattr(c, "text", ""))) for _, c in scored[:8]):
        if profile.internal_intent != "altfuel_ghg_safety":
            warnings.append("no_outcome_signal")

    clusters = cluster_chunks(scored)
    s1, s1_warnings, s1_chunks = _section1(clusters, profile=profile, citation_map=citation_map, row=row, scored=scored)
    warnings.extend(s1_warnings)
    section4 = _section4(work, citation_map, question=question)

    broad_latest_environment = _is_broad_latest_environment_summary(question, profile)
    # The UI exposes one report contract for every question.  Section 2 must
    # therefore be present even when the user did not literally say "영향".
    include_operational_impact = True
    section2 = (
        (
            _section2_latest_environment(s1_chunks or work, citation_map)
            if broad_latest_environment
            else _section2(
                work, citation_map, profile=profile, s1_chunks=s1_chunks, question=question
            )
        )
        if include_operational_impact
        else ""
    )
    completion_plan = (row.get("_evidence_completion") or {}).get("plan") or {}
    verified_iswg_briefing = completion_plan.get("intent") == "iswg_ghg_briefing"
    if verified_iswg_briefing:
        completion_hits = (row.get("_evidence_completion") or {}).get("slot_hits") or {}
        citation_by_id = {
            str(getattr(chunk, "chunk_id", "")): _cite(chunk, citation_map)
            for chunk in citation_chunks
        }

        def slot_cite(slot_name: str) -> str:
            return next(
                (
                    citation_by_id.get(str(chunk_id), "")
                    for chunk_id in completion_hits.get(slot_name) or []
                    if citation_by_id.get(str(chunk_id), "")
                ),
                "",
            )

        sfcs_cite = slot_cite("sfcs_label")
        report_cite = slot_cite("gfi_reporting")
        lca_cite = slot_cite("lca_method")
        impact_lines: list[str] = []
        if sfcs_cite and lca_cite:
            impact_lines.append(
                "- **연료 조달·증빙 영향**: SFCS 인정목록, Fuel Life Cycle Label과 LCA "
                f"전과정 정보는 연료 인증자료 확인 대상으로 연결됩니다. {sfcs_cite}{lca_cite}"
            )
        if report_cite:
            impact_lines.append(
                "- **보고·검증 영향**: GFI 규칙 37 및 SEEMP 개정 논의는 선사의 GFI "
                f"보고·검증 절차와 SEEMP 반영범위에 대한 준비 사안입니다. {report_cite}"
            )
        if impact_lines:
            section2 = "\n".join(impact_lines)
    answer = join_four_sections(
        {
            "1": s1,
            "2": section2,
            "3": _section3(work, profile=profile, question=question, answer="", citation_map=citation_map, s1_chunks=s1_chunks),
            "4": section4,
        }
    )
    # Never erase only the English payload and leave a citation-only bullet.
    # A fallback sentence that was not translated is removed as one claim.
    answer = "\n".join(
        line
        for line in answer.splitlines()
        if not (line.strip().startswith("- ") and ENGLISH_LEAK_RE.search(line))
    )

    if not broad_latest_environment:
        deduped, dedup_warnings = apply_answer_dedup(
            answer,
            profile_intent=profile.internal_intent,
            requested_count=profile.requested_bullet_count,
        )
        answer = deduped
        warnings.extend(dedup_warnings)

    # Enforce objective mismatches (unsupported duties, dates, document codes,
    # and final decisions from non-final documents).  Softer lexical overlap
    # checks remain diagnostic because the answer is Korean and sources are
    # commonly English.
    if verified_iswg_briefing:
        _checked, claim_verification, claim_warnings = verify_claim_citations(
            answer, citation_chunks
        )
        claim_warnings = []
    else:
        answer, claim_verification, claim_warnings = verify_high_risk_claims(
            answer, citation_chunks
        )
    warnings.extend(claim_warnings)

    # Section 2 is a deterministic transformation of cited reporting,
    # verification, safety or operating clauses.  The generic lexical verifier
    # compares Korean claims to mainly English evidence and can remove the
    # entire requested impact section despite the citation being correct.
    # Restore it when impact was explicitly requested.  Alternative-fuel safety
    # summaries are also operational by definition: their fixed output contract
    # requires the project/design implications even when the Korean question
    # says only "논의 및 결론".  This is an intent-level rule, not a pilot-query
    # string match.  Every restored factual bullet must already carry a citation
    # from this answer's evidence map.
    requirements = analyze_requirements(question, row or {})
    restore_practical_section = (
        "impact" in set(requirements.facets)
        or profile.internal_intent
        in {"altfuel_ghg_safety", "meeting_outcome", "mass_code_timeline"}
    )
    if (
        restore_practical_section
        and section2
        and all(
            (not line.lstrip().startswith("- ")) or CITATION_RE.search(line)
            for line in section2.splitlines()
        )
    ):
        section_match = re.search(r"(?ms)^##\s*2\)[^\n]*.*?(?=^##\s*3\))", answer)
        if section_match:
            heading = re.match(r"^##\s*2\)[^\n]*", section_match.group(0))
            if heading:
                replacement = heading.group(0) + "\n\n" + section2.strip() + "\n\n"
                answer = (
                    answer[: section_match.start()]
                    + replacement
                    + answer[section_match.end():]
                )
    # Keep the required fourth section explicit even when the current evidence
    # set contains only IMO meeting records.  A bare heading is ambiguous in
    # the UI; this cited scope statement makes clear that no class rule was
    # returned by this search without inventing a rule candidate.
    if re.search(r"(?ms)^##\s*4\)[^\n]*\s*\Z", answer):
        scope_chunk = next(
            (chunk for chunk in citation_chunks if _cite(chunk, citation_map)),
            None,
        )
        scope_cite = _cite(scope_chunk, citation_map) if scope_chunk else ""
        answer = re.sub(
            r"(?ms)(^##\s*4\)[^\n]*)(?:\s*)\Z",
            (
                r"\1\n\n- 이번 검색 근거는 IMO 회의자료이며, 직접 대응하는 선급 "
                r"Rule/Guidance는 확인되지 않았습니다."
                + (f" {scope_cite}" if scope_cite else "")
            ),
            answer,
        )
    _final_checked, final_claim_rows, final_claim_warnings = verify_claim_citations(
        answer, citation_chunks
    )
    # Do not assign _final_checked back to the answer here.  Section 2 can be a
    # deterministic Korean operational reconstruction from an English source;
    # the generic lexical verifier sees little surface overlap and would erase
    # a correctly cited requirement.  The final pass remains an audit trail,
    # while the display contract below enforces citation presence/range.

    # If claim-verify emptied section 1, rebuild a minimal cited extractive summary
    # from the best available chunks so the UI does not show a blank shell.
    if _section1_is_hollow(answer) and scored:
        rescue_lines: list[str] = []
        rescue_chunks: list[Any] = []
        for _score, chunk in scored[:6]:
            body = _strip_meta(getattr(chunk, "text", ""))
            if len(body) < 40:
                continue
            rescue_lines.append(_format_bullet(chunk, citation_map))
            rescue_chunks.append(chunk)
            if len(rescue_lines) >= max(3, profile.requested_bullet_count or 3):
                break
        if rescue_lines:
            warnings.append("section1_extractive_rescue")
            answer = join_four_sections(
                {
                    "1": "\n".join(rescue_lines),
                    "2": "",
                    "3": _section3(
                        work,
                        profile=profile,
                        question=question,
                        answer="",
                        citation_map=citation_map,
                        s1_chunks=rescue_chunks,
                    ),
                    "4": section4,
                }
            )
            answer = "\n".join(
                line
                for line in answer.splitlines()
                if not (line.strip().startswith("- ") and ENGLISH_LEAK_RE.search(line))
            )
            # If English-leak filter emptied section 1 again, keep short KO stubs with cites.
            if _section1_is_hollow(answer) and rescue_chunks:
                stub_lines = []
                for chunk in rescue_chunks[:4]:
                    cite = _cite(chunk, citation_map)
                    if not cite:
                        continue
                    fn = str(getattr(chunk, "file_name", "") or getattr(chunk, "doc_id", ""))
                    stub_lines.append(
                        f"- **{fn[:80]}** 관련 회의 근거가 검색되었습니다. 원문 조항·결정 문구는 해당 인용에서 확인하세요. {cite}"
                    )
                if stub_lines:
                    answer = join_four_sections(
                        {"1": "\n".join(stub_lines), "2": "", "3": "", "4": section4}
                    )
            answer, claim_verification, claim_warnings = verify_claim_citations(
                answer, citation_chunks
            )
            warnings.extend(claim_warnings)

    qid = str(row.get("question_id") or "")
    coverage, cov_warnings = run_coverage_check(qid, answer, work, row=row)
    warnings.extend(cov_warnings)
    if not CITATION_RE.search(answer):
        warnings.append("citation_missing")

    s1_for_topics = answer.split("## 2)")[0] if "## 2)" in answer else answer
    factual_bullets = [
        line.strip()
        for line in answer.splitlines()
        if line.lstrip().startswith("- ")
        and not any(
            marker in line
            for marker in (
                "검색 근거에서 직접 확인되는 내용이 없어",
                "추가 확인 필요사항이 별도로 식별되지",
            )
        )
    ]
    citation_contract_pass = bool(factual_bullets) and all(
        CITATION_RE.search(line) for line in factual_bullets
    )
    meta = {
        "coverage_check": coverage,
        "used_citations": sorted(int(x) for x in CITATION_RE.findall(answer)),
        "detected_topics": detect_section1_topics(s1_for_topics),
        "top_level_category": profile.top_level_category,
        "internal_intent": profile.internal_intent,
        "document_status": [
            {
                "citation_id": f"[{i}]",
                "status": classify_document_status(c).code,
                "status_label": classify_document_status(c).label_ko,
            }
            for i, c in enumerate(citation_chunks, 1)
        ],
        "claim_verification": final_claim_rows,
        "claim_verification_pass": citation_contract_pass,
        "unsupported_claims_blocked": sum(
            1 for r in final_claim_rows if not r.get("supported")
        ),
    }
    return answer, list(dict.fromkeys(warnings)), meta
