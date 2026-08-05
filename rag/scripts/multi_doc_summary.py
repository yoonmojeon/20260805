"""Multi-document summarization pipeline for MEPC/MSC broad questions."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from answer_depth_guidance import (
    ANSWER_DENSITY_GUIDANCE,
    ANTI_REPETITION_GUIDANCE,
    CITATION_GUIDANCE,
    ENV_REGULATION_V01_HINT,
    FORMAT_RULES,
    GOOD_BAD_EXAMPLES,
    RULE_LOOKUP_GUIDANCE,
    SECTION2_OPERATIONAL_GUIDANCE,
    SECTION2_RULE_LOOKUP_GUIDANCE,
    SECTION3_FOLLOWUP_GUIDANCE,
)

MEETING_RE = re.compile(r"\b(MEPC|MSC)\s*(\d{2,3})\b", re.I)
FILE_MEETING_RE = re.compile(r"(?:MEPC|MSC)\s*(\d{2,3})[-_/]", re.I)

INTEGRATION_TOPICS_MEPC = [
    "IMO Net-Zero Framework",
    "GHG 감축 및 연료 규제",
    "GFI / ZNZs / remedial unit / registry",
    "CII / SEEMP / EEXI",
    "MARPOL Annex VI 보고·검증 체계",
    "LCA Guidelines / 연료 전주기 배출",
    "선박 운항 및 선사 업무 영향",
    "추가 확인이 필요한 선급 Rule / Guidance",
]

INTEGRATION_TOPICS_MSC = [
    "MASS Code / 자율운항 규제",
    "mandatory / non-mandatory code 일정",
    "goal-based safety requirements",
    "degree of autonomy / functional requirements",
    "항해 안전 / 통신 / 장비",
    "선박 운항 및 선사 업무 영향",
    "추가 확인이 필요한 선급 Rule / Guidance",
]


@dataclass
class DocGroup:
    doc_id: str
    file_name: str
    source: str
    meeting: int | None
    chunks: list[Any] = field(default_factory=list)
    citation_ids: list[int] = field(default_factory=list)

    @property
    def pages(self) -> list[int]:
        pages = []
        for c in self.chunks:
            p = getattr(c, "page_number", None)
            if p is not None:
                pages.append(int(p))
        return sorted(set(pages))


@dataclass
class MultiDocResult:
    retrieved: list[Any]
    doc_groups: list[DocGroup]
    mini_summaries: list[str]
    answer: str
    warnings: list[str] = field(default_factory=list)
    source_doc_list: str = ""


def parse_meeting_number(file_name: str, source: str = "") -> int | None:
    text = f"{file_name} {source}"
    m = MEETING_RE.search(text) or FILE_MEETING_RE.search(text)
    if m:
        try:
            return int(m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1))
        except (TypeError, ValueError):
            pass
    m2 = FILE_MEETING_RE.search(file_name or "")
    if m2:
        try:
            return int(m2.group(1))
        except (TypeError, ValueError):
            pass
    return None


def _doc_score(chunks: list[Any], file_name: str) -> float:
    meeting = parse_meeting_number(file_name, str(getattr(chunks[0], "source", "")))
    min_dist = min(float(getattr(c, "distance", 1.0)) for c in chunks)
    meeting_bonus = (meeting or 0) * 0.003
    return min_dist - meeting_bonus


def discover_document_candidates(
    pool: list[Any],
    *,
    top_docs: int = 15,
    max_chunks_per_doc: int = 3,
    min_unique_docs: int = 3,
) -> tuple[list[DocGroup], list[str]]:
    """Step 1: group pool by doc, rank by relevance + meeting recency."""
    by_doc: dict[str, list[Any]] = defaultdict(list)
    for c in pool:
        by_doc[str(getattr(c, "doc_id", ""))].append(c)

    ranked: list[tuple[str, list[Any], float]] = []
    for doc_id, chunks in by_doc.items():
        if not doc_id:
            continue
        chunks_sorted = sorted(chunks, key=lambda x: float(getattr(x, "distance", 1.0)))
        fname = str(getattr(chunks_sorted[0], "file_name", "") or doc_id)
        ranked.append((doc_id, chunks_sorted, _doc_score(chunks_sorted, fname)))

    ranked.sort(key=lambda x: x[2])
    warnings: list[str] = []
    selected: list[DocGroup] = []
    for doc_id, chunks_sorted, _ in ranked[:top_docs]:
        top_chunks = chunks_sorted[:max_chunks_per_doc]
        fname = str(getattr(top_chunks[0], "file_name", "") or doc_id)
        source = str(getattr(top_chunks[0], "source", ""))
        meeting = parse_meeting_number(fname, source)
        selected.append(
            DocGroup(
                doc_id=doc_id,
                file_name=fname,
                source=source,
                meeting=meeting,
                chunks=top_chunks,
            )
        )

    if len(selected) < min_unique_docs and len(ranked) >= min_unique_docs:
        warnings.append(
            f"문서 후보 {len(selected)}개만 선정됨 (목표 최소 {min_unique_docs}개)"
        )
    if len(selected) == 1:
        warnings.append("검색 결과가 특정 문서에 편중됨 — 추가 MEPC/MSC 문서 검색 권장")

    return selected, warnings


def assign_global_citations(doc_groups: list[DocGroup]) -> list[Any]:
    """Flatten chunks with global [1]..[N] citation ids."""
    flat: list[Any] = []
    cid = 1
    for dg in doc_groups:
        dg.citation_ids = []
        for c in dg.chunks:
            dg.citation_ids.append(cid)
            flat.append(c)
            cid += 1
    return flat


def _chunk_block(c: Any, cite_id: int) -> str:
    fname = getattr(c, "file_name", "") or getattr(c, "doc_id", "")
    pn = getattr(c, "page_number", "?")
    return (
        f"[{cite_id}] source={getattr(c, 'source', '')} | doc={fname} | p{pn}\n"
        f"{getattr(c, 'text', '')}"
    )


def build_doc_context_block(dg: DocGroup) -> str:
    parts = []
    for c, cid in zip(dg.chunks, dg.citation_ids):
        parts.append(_chunk_block(c, cid))
    return "\n\n---\n\n".join(parts)


def build_mini_summary_system(category: str, source: str) -> str:
    if source.upper() == "MSC" or category == "autonomous":
        return (
            "당신은 MSC 해상안전·자율운항(MASS) 문서 분석 보조자입니다. "
            "한국어로 작성합니다. 검색 청크에 없는 내용은 추측하지 마세요."
        )
    return (
        "당신은 MEPC 환경규제(GHG, CII, SEEMP, GFI, MARPOL Annex VI, Net-Zero) 문서 분석 보조자입니다. "
        "한국어로 작성합니다. 검색 청크에 없는 내용은 추측하지 마세요."
    )


def build_mini_summary_user(dg: DocGroup, category: str) -> str:
    meeting = f"MEPC/MSC {dg.meeting}" if dg.meeting else "회의차수 미상"
    ctx = build_doc_context_block(dg)
    impact_axis = (
        "선박 운항·안전·MASS Code·mandatory code 일정·검사"
        if dg.source.upper() == "MSC" or category == "autonomous"
        else "선박 운항·연료·배출 보고·CII·SEEMP·GFI·규제 대응"
    )
    return f"""다음은 단일 문서의 검색 청크입니다.

문서명: {dg.file_name}
source: {dg.source}
회의차수: {meeting}
페이지: {', '.join(f'p{p}' for p in dg.pages) or '—'}

청크:
{ctx}

위 청크만 사용하여 아래 항목을 **각 2~4문장**으로 작성하세요 (단순 키워드 나열 금지).
인용은 청크 번호 [N] 형식으로 표기하세요.

## 문서 핵심 요약
- 환경/안전 규제 이슈 (문서가 다루는 핵심 안건)

## 결정사항 또는 쟁점
- (context에 있을 때만)

## {impact_axis} 영향
- (context에 있을 때만)

## 후속 확인 필요
- (미확정·워킹그룹 진행 중 등)

## 사용 citation
- [번호] 목록"""


def build_synthesis_system(category: str, sources: list[str]) -> str:
    src = ", ".join(sources) if sources else "IMO"
    if category == "rule_lookup":
        role = "선급 Rule/Guidance 조회 관점으로 답합니다."
        section2 = SECTION2_RULE_LOOKUP_GUIDANCE
        rule_block = RULE_LOOKUP_GUIDANCE
        section1_extra = "- 문서별 **고유 사실**만 또는 주제 통합 (동일 결론 문장 반복 금지)"
    elif category == "autonomous" or (sources and sources[0].upper() == "MSC"):
        role = (
            "MSC 안전운항·자율운항(MASS Code)·항해안전·통신·장비·규정 이행 관점으로 답합니다."
        )
        section2 = SECTION2_OPERATIONAL_GUIDANCE
        rule_block = ""
        section1_extra = "- 여러 문서를 **주제별 통합** (중복·동일 문장 반복 금지)"
    else:
        role = (
            "MEPC 배출·GHG·에너지효율·MARPOL Annex VI·CII·SEEMP·GFI·Net-Zero Framework "
            "환경규제 대응 관점으로 답합니다."
        )
        section2 = SECTION2_OPERATIONAL_GUIDANCE
        rule_block = ""
        section1_extra = "- 여러 문서를 **주제별 통합** (중복·동일 문장 반복 금지)"

    return f"""당신은 해사 규제 문서 분석 보조자입니다. 반드시 한국어로 답변합니다.
{role}
검색 청크·mini-summary에 없는 내용은 추정하지 마세요. 단순 키워드 나열 금지.
source: {src}
{FORMAT_RULES}
{ANTI_REPETITION_GUIDANCE}
{CITATION_GUIDANCE}
{ANSWER_DENSITY_GUIDANCE}
{rule_block}

## 1) 핵심 요약
- bullet 7~10개 이내, **상단 3개 bullet 최우선 — 굵게**
- 각 bullet **최소 2문장**, 3단 구조(논의·규제의미·업무영향), citation [N]
{section1_extra}

{section2}

{SECTION3_FOLLOWUP_GUIDANCE}

## 4) 관련 선급 Rule / Guidance
- 선급 Rule/Guidance 청크가 있으면 문서명·핵심 (각 1~2문장, §1 중복 금지)
- 없으면 정확히: "본 검색 결과는 IMO/MEPC 회의 자료 중심이며, DNV/LR/ABS/KR 등 선급 Rule 또는 Guidance는 별도 검색이 필요하다."

규칙:
- [근거] placeholder 금지, [1][2] 숫자 citation만
- 검색이 특정 문서에 편중되었으면 ## 3) [해석 근거] 태그로 명시
- 추론 서두 없이 바로 ## 1)부터 출력
{GOOD_BAD_EXAMPLES}
"""


def build_batched_mini_summary_user(doc_groups: list[DocGroup], category: str) -> str:
    """One LLM call summarizing up to a few documents."""
    blocks = []
    for dg in doc_groups:
        meeting = f"MEPC/MSC {dg.meeting}" if dg.meeting else "회의차수 미상"
        blocks.append(
            f"### {dg.file_name}\n"
            f"source: {dg.source} | {meeting} | pages: {', '.join(f'p{p}' for p in dg.pages) or '—'}\n\n"
            f"{build_doc_context_block(dg)}"
        )
    docs = "\n\n---DOC---\n\n".join(blocks)
    impact = (
        "선박 운항·안전·MASS Code"
        if any(dg.source.upper() == "MSC" for dg in doc_groups) or category == "autonomous"
        else "선박 운항·연료·배출·CII·SEEMP·GFI"
    )
    return f"""다음 {len(doc_groups)}개 문서의 검색 청크입니다. 문서별로 구분해 요약하세요.

{docs}

각 문서마다 아래 형식으로 **2~3문장**씩 작성 (키워드 나열 금지, citation [N] 포함):

## {{문서 파일명}}
- 핵심 이슈:
- 결정/쟁점 (있을 때):
- {impact} 영향 (있을 때):
- citation: [N] ..."""


def build_direct_synthesis_user(
    row: dict,
    category: str,
    doc_groups: list[DocGroup],
    topics: list[str],
    warnings: list[str],
) -> str:
    """Single-pass synthesis prompt — doc-grouped chunks, no mini-summary step."""
    doc_blocks = []
    for dg in doc_groups:
        meeting = f"MEPC/MSC {dg.meeting}" if dg.meeting else "?"
        doc_blocks.append(
            f"### {dg.file_name} ({dg.source} {meeting})\n{build_doc_context_block(dg)}"
        )
    warn = "\n".join(f"- ⚠ {w}" for w in warnings) if warnings else "없음"
    qid = str(row.get("question_id", ""))
    v01_block = ENV_REGULATION_V01_HINT if qid == "V01" else ""
    return f"""질문: {row['question']}
카테고리: {category}
{v01_block}
통합 주제 (청크에 있을 때만 — 없으면 §3에 "검색 결과 내 확인 불가"):
{chr(10).join(f'- {t}' for t in topics)}

경고:
{warn}

문서별 검색 청크:
{chr(10).join(doc_blocks)}

위 청크만 사용하여 최종 4섹션 답변을 작성하세요.

**작성 체크 (각 bullet마다):**
- **§1·§2 bullet 끝 citation [N] 필수** (context `[1]`, `[2]` … 와 일치)
- 키워드 나열·`↔` 종결 금지 → 2~3문장 보고서형으로 풀어 쓸 것
- 3요소(논의·개정·요구 / 규제 의미 / 업무 영향) 중 **최소 2개** 포함
- 마지막 문장은 "따라서 …" 또는 "이는 …와 연결된다" 형태
- §1: context에 있는 **주제마다 1 bullet** (보고서형 2~3문장, 억지로 bullet 수만 늘리지 말 것)
- §2: 실무 조치만 (§1과 다른 문장)
- §3: `[미확정 규제]` / `[해석 근거]` / `[선급별 상이 요구]` 태그
- sub-bullet·### 금지, 바로 ## 1)부터 출력"""


def build_synthesis_user(
    row: dict,
    category: str,
    mini_summaries: list[str],
    doc_groups: list[DocGroup],
    topics: list[str],
    warnings: list[str],
    *,
    batched: bool = False,
) -> str:
    doc_index = "\n".join(
        f"- {dg.file_name} | {dg.source} {dg.meeting or '?'} | pages {dg.pages} | cites {dg.citation_ids}"
        for dg in doc_groups
    )
    if batched:
        mini_block = "\n\n---BATCH---\n\n".join(mini_summaries)
    else:
        mini_block = "\n\n---DOC---\n\n".join(
            f"### {dg.file_name}\n{ms}" for dg, ms in zip(doc_groups, mini_summaries)
        )
    warn = "\n".join(f"- ⚠ {w}" for w in warnings) if warnings else "없음"
    return f"""질문: {row['question']}
카테고리: {category}

통합 주제 (context/mini-summary에 있을 때만 다룸):
{chr(10).join(f'- {t}' for t in topics)}

경고:
{warn}

문서 인덱스:
{doc_index}

문서별 mini-summary:
{mini_block}

위 mini-summary만 사용하여 최종 4섹션 답변을 작성하세요.
**§1·§2 bullet마다 문장 끝 citation [N] 필수.** 각 핵심 bullet은 3문장 이상, citation [N]을 포함하세요."""


def build_source_document_list(doc_groups: list[DocGroup], mini_summaries: list[str]) -> str:
    lines = ["", "## 근거 문서 목록", ""]
    per_doc_ms = mini_summaries
    if len(mini_summaries) != len(doc_groups):
        per_doc_ms = [""] * len(doc_groups)
    for dg, ms in zip(doc_groups, per_doc_ms):
        meeting = f"MEPC/MSC {dg.meeting}" if dg.meeting else "—"
        cites = ", ".join(f"[{c}]" for c in dg.citation_ids)
        if ms and not ms.startswith("(single-pass"):
            snippet = re.sub(r"\s+", " ", ms)[:280]
        elif dg.chunks:
            snippet = re.sub(r"\s+", " ", str(getattr(dg.chunks[0], "text", "")))[:280]
        else:
            snippet = "—"
        lines.append(f"### {dg.file_name}")
        lines.append(f"- **회의차수:** {meeting}")
        lines.append(f"- **페이지:** {', '.join(f'p{p}' for p in dg.pages) or '—'}")
        lines.append(f"- **citation:** {cites}")
        lines.append(f"- **핵심 근거:** {snippet}…")
        lines.append("")
    return "\n".join(lines)


def integration_topics(category: str, sources: list[str]) -> list[str]:
    if category == "autonomous" or ("MSC" in {s.upper() for s in sources} and "MEPC" not in sources):
        return INTEGRATION_TOPICS_MSC
    return INTEGRATION_TOPICS_MEPC


def _cap_doc_groups(doc_groups: list[DocGroup], max_llm_docs: int) -> tuple[list[DocGroup], list[str]]:
    if max_llm_docs <= 0 or len(doc_groups) <= max_llm_docs:
        return doc_groups, []
    capped = doc_groups[:max_llm_docs]
    return capped, [
        f"LLM 단계는 상위 {max_llm_docs}개 문서만 사용 (검색 evidence는 전체 유지)"
    ]


def run_multi_doc_summary(
    row: dict,
    pool: list[Any],
    *,
    category: str,
    call_llm,
    doc_groups: list[DocGroup] | None = None,
    top_docs: int = 15,
    max_chunks_per_doc: int = 3,
    min_unique_docs: int = 3,
    strategy: str = "single_pass",
    max_llm_docs: int = 6,
    batch_size: int = 3,
    on_progress=None,
) -> MultiDocResult:
    """Steps 1–3: discover docs → (optional) mini-summaries → synthesis.

    strategy:
      - single_pass: one LLM call with doc-grouped chunks (~1–2 min)
      - batched: mini-summary per batch of docs + synthesis (~2–4 min, default)
      - full: one mini-summary per document + synthesis (slow, highest detail)
    """
    warnings: list[str] = []
    if doc_groups is None:
        doc_groups, warnings = discover_document_candidates(
            pool,
            top_docs=top_docs,
            max_chunks_per_doc=max_chunks_per_doc,
            min_unique_docs=min_unique_docs,
        )
    else:
        assign_global_citations(doc_groups)

    doc_groups, cap_warnings = _cap_doc_groups(doc_groups, max_llm_docs)
    warnings.extend(cap_warnings)

    retrieved = [c for dg in doc_groups for c in dg.chunks]
    sources = list(row.get("retrieval_sources") or [])
    topics = integration_topics(category, sources)
    synth_system = build_synthesis_system(category, sources)

    def _progress(step: str) -> None:
        if on_progress:
            on_progress(step)

    mini_summaries: list[str] = []
    strategy = (strategy or "batched").lower()

    if strategy == "single_pass":
        _progress("synthesis (single-pass)")
        synth_user = build_direct_synthesis_user(row, category, doc_groups, topics, warnings)
        answer = call_llm(synth_system, synth_user, num_predict=2800)
        mini_summaries = ["(single-pass — mini-summary 생략)"] * len(doc_groups)
    elif strategy == "batched":
        batches = [
            doc_groups[i : i + batch_size] for i in range(0, len(doc_groups), batch_size)
        ]
        for i, batch in enumerate(batches, start=1):
            _progress(f"batch mini-summary {i}/{len(batches)}")
            src = batch[0].source if batch else "MEPC"
            system = build_mini_summary_system(category, src)
            user = build_batched_mini_summary_user(batch, category)
            mini_summaries.append(call_llm(system, user, num_predict=900))
        _progress("synthesis")
        synth_user = build_synthesis_user(
            row, category, mini_summaries, doc_groups, topics, warnings, batched=True
        )
        answer = call_llm(synth_system, synth_user, num_predict=2800)
    else:
        for i, dg in enumerate(doc_groups, start=1):
            _progress(f"doc mini-summary {i}/{len(doc_groups)}")
            system = build_mini_summary_system(category, dg.source)
            user = build_mini_summary_user(dg, category)
            mini_summaries.append(call_llm(system, user, num_predict=550))
        _progress("synthesis")
        synth_user = build_synthesis_user(row, category, mini_summaries, doc_groups, topics, warnings)
        answer = call_llm(synth_system, synth_user, num_predict=2800)

    source_list = build_source_document_list(doc_groups, mini_summaries)
    if "## 근거 문서 목록" not in answer:
        answer = answer.rstrip() + source_list

    unique = len(doc_groups)
    if unique < min_unique_docs:
        warnings.append(f"unique_doc_count={unique} (목표 ≥{min_unique_docs})")

    return MultiDocResult(
        retrieved=retrieved,
        doc_groups=doc_groups,
        mini_summaries=mini_summaries,
        answer=answer,
        warnings=warnings,
        source_doc_list=source_list,
    )
