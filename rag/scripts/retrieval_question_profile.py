"""Question-type retrieval profiles: broad summary vs narrow Rule lookup."""
from __future__ import annotations

from dataclasses import dataclass, field

from meeting_outcome_retrieval import is_meeting_outcome_question
from question_classifier import (
    CATEGORY_LABELS_KO,
    classify_question_category,
    detect_broad_summary_mode,
)
from retrieval_verification import classify_question_mode, detect_narrow_doc_id


@dataclass
class RetrievalQuestionProfile:
    """Resolved retrieval strategy for a question (UI sliders are capped/tuned by profile)."""

    question_category: str
    question_mode: str  # broad | narrow
    broad_summary: bool
    answer_mode: str  # standard_rag | multi_doc_summary | meeting_outcome
    profile_id: str
    label_ko: str
    fetch_k: int
    top_k: int
    max_docs: int
    max_chunks_per_doc: int
    max_chunks_per_page: int
    use_diversity_rerank: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "label_ko": self.label_ko,
            "question_category": self.question_category,
            "question_mode": self.question_mode,
            "broad_summary": self.broad_summary,
            "answer_mode": self.answer_mode,
            "fetch_k": self.fetch_k,
            "top_k": self.top_k,
            "max_docs": self.max_docs,
            "max_chunks_per_doc": self.max_chunks_per_doc,
            "max_chunks_per_page": self.max_chunks_per_page,
            "use_diversity_rerank": self.use_diversity_rerank,
            "notes": list(self.notes),
        }


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def build_retrieval_profile(
    question: str,
    row: dict,
    *,
    ui_top_k: int = 10,
    ui_fetch_k: int = 120,
    ui_max_docs: int = 10,
    ui_max_chunks_per_doc: int = 3,
    ui_max_chunks_per_page: int = 1,
    ui_use_rerank: bool = True,
    eval_constrained: bool = False,
) -> RetrievalQuestionProfile:
    """
    Map question → retrieval profile.

    UI sliders set upper bounds; profile tightens fetch_k / max_docs for narrow lookups.
    """
    explicit = str(row.get("category") or "").strip()
    category = explicit if explicit in CATEGORY_LABELS_KO else classify_question_category(question, row)
    enriched = {**row, "category": category}

    if row.get("_table_qa") or category == "table_qa":
        return RetrievalQuestionProfile(
            question_category="table_qa",
            question_mode="narrow",
            broad_summary=False,
            answer_mode="table_qa",
            profile_id="table_schema_two_stage",
            label_ko="표 스키마→행·셀 조회",
            fetch_k=_clamp(ui_fetch_k, 24, 40),
            top_k=_clamp(ui_top_k, 10, 16),
            max_docs=min(ui_max_docs, 3),
            max_chunks_per_doc=max(ui_max_chunks_per_doc, 8),
            max_chunks_per_page=max(ui_max_chunks_per_page, 8),
            use_diversity_rerank=False,
            notes=["표 ID를 먼저 고른 뒤 schema·summary·row·markdown 근거를 결합"],
        )

    from meeting_category_profile import uses_structured_meeting_answer

    if (
        is_meeting_outcome_question(question, enriched)
        and not eval_constrained
        and uses_structured_meeting_answer(enriched, legacy_category=category)
    ):
        return RetrievalQuestionProfile(
            question_category=category,
            question_mode="broad",
            broad_summary=False,
            answer_mode="meeting_outcome",
            profile_id="meeting_outcome",
            label_ko="회의 결과·비교 (넓은 pool)",
            fetch_k=max(ui_fetch_k, 100),
            top_k=max(ui_top_k, 12),
            max_docs=max(ui_max_docs, 6),
            max_chunks_per_doc=max(ui_max_chunks_per_doc, 4),
            max_chunks_per_page=ui_max_chunks_per_page,
            use_diversity_rerank=ui_use_rerank,
            notes=["회의 outcome 파이프라인 — fetch pool 확대"],
        )

    broad_summary = (
        detect_broad_summary_mode(question, enriched, category)
        and not eval_constrained
        and not detect_narrow_doc_id(question, enriched)
    )
    if broad_summary:
        return RetrievalQuestionProfile(
            question_category=category,
            question_mode="broad",
            broad_summary=True,
            answer_mode="multi_doc_summary",
            profile_id="broad_summary",
            label_ko="요약형 (다문서 통합)",
            fetch_k=max(ui_fetch_k, 120),
            top_k=max(ui_top_k, 10),
            max_docs=max(ui_max_docs, 8),
            max_chunks_per_doc=max(ui_max_chunks_per_doc, 3),
            max_chunks_per_page=ui_max_chunks_per_page,
            use_diversity_rerank=ui_use_rerank,
            notes=[
                "MEPC/MSC 등 다문서 요약 — fetch_k·max_docs 넓게",
                f"category={category}",
            ],
        )

    qmode = classify_question_mode(question, enriched)

    if category == "rule_lookup":
        return RetrievalQuestionProfile(
            question_category=category,
            question_mode="narrow",
            broad_summary=False,
            answer_mode="rule_guidance_lookup",
            profile_id="rule_guidance_lookup",
            label_ko="Rule/Guidance 조회",
            fetch_k=_clamp(ui_fetch_k, 24, 56),
            top_k=_clamp(ui_top_k, 8, 12),
            max_docs=min(ui_max_docs, 3),
            max_chunks_per_doc=max(ui_max_chunks_per_doc, 4),
            max_chunks_per_page=ui_max_chunks_per_page,
            use_diversity_rerank=ui_use_rerank,
            notes=[
                "선급 Rule 찾기 — max_docs≤3, 문서당 청크 깊이 우선",
                "유사 RP 다수 편입 방지",
            ],
        )

    if category == "autonomous" and qmode == "narrow":
        return RetrievalQuestionProfile(
            question_category=category,
            question_mode="narrow",
            broad_summary=False,
            answer_mode="standard_rag",
            profile_id="autonomous_narrow",
            label_ko="자율운항 포인트 조회",
            fetch_k=_clamp(ui_fetch_k, 40, 80),
            top_k=_clamp(ui_top_k, 8, 12),
            max_docs=min(ui_max_docs, 4),
            max_chunks_per_doc=max(ui_max_chunks_per_doc, 3),
            max_chunks_per_page=ui_max_chunks_per_page,
            use_diversity_rerank=ui_use_rerank,
            notes=["MASS/자율운항 특정 조회 — 중간 범위"],
        )

    if category in {"trend_summary", "env_regulation"}:
        return RetrievalQuestionProfile(
            question_category=category,
            question_mode="broad",
            broad_summary=False,
            answer_mode="standard_rag",
            profile_id="regulation_standard",
            label_ko="규제·동향 (표준 RAG)",
            fetch_k=max(ui_fetch_k, 60),
            top_k=max(ui_top_k, 8),
            max_docs=min(ui_max_docs, 6),
            max_chunks_per_doc=max(ui_max_chunks_per_doc, 2),
            max_chunks_per_page=ui_max_chunks_per_page,
            use_diversity_rerank=ui_use_rerank,
            notes=[f"category={category} — 다문서 요약 미적용, diversity 유지"],
        )

    return RetrievalQuestionProfile(
        question_category=category,
        question_mode=qmode,
        broad_summary=False,
        answer_mode="standard_rag",
        profile_id="default",
        label_ko="기본",
        fetch_k=ui_fetch_k,
        top_k=ui_top_k,
        max_docs=ui_max_docs if qmode == "broad" else min(ui_max_docs, 4),
        max_chunks_per_doc=ui_max_chunks_per_doc,
        max_chunks_per_page=ui_max_chunks_per_page,
        use_diversity_rerank=ui_use_rerank,
        notes=[f"mode={qmode}, category={category}"],
    )


def profile_to_run_config_kwargs(profile: RetrievalQuestionProfile) -> dict:
    return {
        "question_mode": profile.question_mode,
        "top_k": profile.top_k,
        "fetch_k": profile.fetch_k,
        "max_docs": profile.max_docs,
        "max_chunks_per_doc": profile.max_chunks_per_doc,
        "max_chunks_per_page": profile.max_chunks_per_page,
        "use_diversity_rerank": profile.use_diversity_rerank,
    }
