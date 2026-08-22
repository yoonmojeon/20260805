from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAG_SCRIPTS = ROOT / "rag" / "scripts"
for path in (ROOT, RAG_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fast_context import (  # noqa: E402
    _question_focused_excerpt,
    korean_question_focus_score,
    question_focus_score,
)
from grounded_dynamic_answer import validate_answer_requirements  # noqa: E402
from question_requirements import analyze_requirements  # noqa: E402
from rag_answer_lib import RetrievedChunk  # noqa: E402
from rag_fast_mode import (  # noqa: E402
    _document_local_query_hits,
    _source_local_query_hits,
    _normalize_fast_page_citations,
    build_fast_context_and_chunks,
)
from table_retrieval import is_table_question  # noqa: E402


def test_focus_excerpt_keeps_matching_parallel_rule_item():
    source = (
        "The specimens shall be positioned as follows:\n"
        "— for multi-run technique, at mid thickness of the weld\n"
        "— for two-run welded test assemblies, on the 2nd run side, "
        "2 mm below the surface\n"
        "— for electroslag welding, 2 mm below the surface"
    )
    excerpt = _question_focused_excerpt(
        source,
        "2회 용접(two-run welded) 시험편은 어디에서 절단합니까?",
    )
    assert "two-run welded" in excerpt
    assert "2nd run side" in excerpt
    assert "mid thickness" not in excerpt


def test_focus_excerpt_uses_short_uppercase_technical_anchor():
    source = (
        "A generic generator may be tested at 60% load.\n"
        "— a CGS considered one of two independent power sources shall be "
        "tested at normal seagoing load\n"
        "— an emergency unit follows the approved load balance"
    )
    excerpt = _question_focused_excerpt(source, "CGS는 어떤 부하에서 시험합니까?")
    assert "normal seagoing load" in excerpt
    assert "60% load" not in excerpt


def test_focus_excerpt_anchors_full_literal_phrase_after_pdf_line_wraps():
    source = (
        "[msc] file=MSC 111-5-7 Development of the experience-building phase.pdf\n"
        ".2 evidence analysis; and, unrelated introduction.\n"
        "The term evidence is proposed to cover data, research and experience from\n"
        "all stakeholders. It is suggested that evidence collection may be derived "
        "from operational practice, tests, simulations and academic studies."
    )
    excerpt = _question_focused_excerpt(
        source, "is proposed to cover data, research and experience"
    )
    assert "all stakeholders" in excerpt
    assert "operational practice" in excerpt


def test_focus_excerpt_uses_korean_query_cluster_not_page_heading():
    source = (
        "부유식 해상구조물 규칙 일반사항과 정의. "
        "비상전원은 조타기, 비상조명, 화재탐지장치 및 통화장치에 "
        "적어도 6시간 이상 급전할 수 있어야 한다. "
        "별도의 특수 통신설비는 18시간 급전하며 특정 보급품은 4일을 적용한다."
    )
    excerpt = _question_focused_excerpt(
        source,
        "부유식 해상구조물의 비상전원은 어느 장치들에 몇 시간 급전해야 합니까?",
    )
    assert "비상전원" in excerpt
    assert "6시간" in excerpt


def test_focus_score_prefers_chunk_with_named_technical_term():
    question = "2회 용접(two-run welded) 시험편은 어디에서 절단합니까?"
    matching = "for two-run welded test assemblies, on the 2nd run side"
    neighbouring = "for multi-run technique, at mid thickness"
    assert question_focus_score(matching, question) > 0
    assert question_focus_score(neighbouring, question) == 0


def test_korean_focus_score_strips_particles_from_compound_rule_term():
    question = "방식조치의 요건과 예외를 알려줘"
    direct = korean_question_focus_score("3. 방식조치의 적용 범위", question)
    incidental = korean_question_focus_score("x" * 900 + " 방식조치 참조", question)
    assert direct >= 7
    assert direct > incidental
    assert korean_question_focus_score("프로펠러 부착과 고온 가열", question) == 0


def test_document_local_rerank_recovers_underlying_study(tmp_path):
    doc_id = "mepc_named_paper"
    folder = tmp_path / doc_id
    folder.mkdir()
    rows = [
        {
            "chunk_id": f"{doc_id}_action",
            "doc_id": doc_id,
            "page_number": 5,
            "text": "The Committee is invited to consider this document and take action as appropriate.",
        },
        {
            "chunk_id": f"{doc_id}_study",
            "doc_id": doc_id,
            "page_number": 2,
            "text": (
                "The values are derived from findings of the FUMES study, based on "
                "drone, helicopter and onboard measurements of LNG methane slip."
            ),
        },
    ]
    (folder / "chunks.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    hits = _document_local_query_hits(
        tmp_path,
        doc_id,
        "LNG Cslip 개정은 어떤 연구 결과에 기반합니까?",
        existing=[],
    )
    assert hits
    assert hits[0].chunk_id.endswith("_study")


def test_document_local_rerank_prioritizes_named_definition_and_relative_value(tmp_path):
    doc_id = "mepc_named_fact"
    folder = tmp_path / doc_id
    folder.mkdir()
    rows = [
        {
            "chunk_id": f"{doc_id}_generic",
            "doc_id": doc_id,
            "page_number": 2,
            "text": "Annual supply-based carbon intensity monitoring covers 2019 to 2024.",
        },
        {
            "chunk_id": f"{doc_id}_answer",
            "doc_id": doc_id,
            "page_number": 5,
            "text": (
                "Supply-based carbon intensity showed a decrease of up to 10.8% "
                "in 2024 relative to 2019."
            ),
        },
    ]
    (folder / "chunks.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    hits = _document_local_query_hits(
        tmp_path,
        doc_id,
        "2019년 대비 2024년 supply-based carbon intensity는 어느 정도 감소했습니까?",
        existing=[],
    )
    assert hits[0].chunk_id.endswith("_answer")


def test_document_local_rerank_keeps_requested_fuel_and_committee_outcome(tmp_path):
    doc_id = "msc_111_wp1"
    folder = tmp_path / doc_id
    folder.mkdir()
    rows = [
        {
            "chunk_id": f"{doc_id}_p0063_m001",
            "doc_id": doc_id,
            "page_number": 63,
            "text": (
                "The Committee discussed draft interim guidelines for ships "
                "using ammonia cargo as fuel."
            ),
        },
        {
            "chunk_id": f"{doc_id}_p0064_m003",
            "doc_id": doc_id,
            "page_number": 64,
            "text": (
                "Following editorial modifications, the Committee approved the "
                "Interim guidelines for the safety of ships using hydrogen as fuel."
            ),
        },
    ]
    (folder / "chunks.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    hits = _document_local_query_hits(
        tmp_path,
        doc_id,
        "MSC 111에서 수소 연료 잠정 안전지침이 부결됐다는 전제가 맞는지 검증해줘",
        existing=[],
    )
    assert hits
    assert hits[0].chunk_id.endswith("_p0064_m003")
    assert "approved" in hits[0].text


def test_source_local_rerank_hard_scopes_explicit_dnv_cp(tmp_path):
    for doc_id, code in (
        ("dnv_dnv_cp_0099_deadbeef", "DNV-CP-0099"),
        ("dnv_dnv_cp_0293_deadbeef", "DNV-CP-0293"),
    ):
        folder = tmp_path / doc_id
        folder.mkdir()
        (folder / "chunks.jsonl").write_text(
            json.dumps(
                {
                    "chunk_id": f"{doc_id}_p0005_m001",
                    "doc_id": doc_id,
                    "source_pdf": f"{code}.pdf",
                    "page_number": 5,
                    "text": (
                        "The type approval certificate covers one grade and may "
                        "include colour variants and thinned variants."
                    ),
                }
            ),
            encoding="utf-8",
        )
    hits = _source_local_query_hits(
        tmp_path,
        "DNV",
        "DNV-CP-0099에서 grade와 variants는 어떻게 정의합니까?",
        existing=[],
    )
    assert hits
    assert all("cp_0099" in hit.doc_id for hit in hits)


def test_source_local_rerank_retains_dense_favoured_document(tmp_path):
    dense_doc = "dnv_dnv_cp_0100_dense"
    lexical_doc = "dnv_dnv_cp_0200_lexical"
    rows = {
        dense_doc: "A special approval report shall document the starting material.",
        lexical_doc: (
            "The approval report shall document material material material and "
            "the general approval procedure."
        ),
    }
    for doc_id, body in rows.items():
        folder = tmp_path / doc_id
        folder.mkdir()
        (folder / "chunks.jsonl").write_text(
            json.dumps(
                {
                    "chunk_id": f"{doc_id}_p0005_m001",
                    "doc_id": doc_id,
                    "source_pdf": f"{doc_id}.pdf",
                    "page_number": 5,
                    "text": body,
                }
            ),
            encoding="utf-8",
        )
    dense_seed = RetrievedChunk(
        chunk_id="dense-seed",
        doc_id=dense_doc,
        source="DNV",
        file_name="DNV-CP-0100.pdf",
        page_number=1,
        clause_number="",
        element_type="text",
        distance=0.01,
        text="semantic seed",
    )
    hits = _source_local_query_hits(
        tmp_path,
        "DNV",
        "starting material approval report에는 무엇이 필요합니까?",
        existing=[dense_seed],
        limit=2,
    )
    assert hits
    assert hits[0].doc_id == dense_doc


def test_feature_fallback_chunk_survives_typed_slots_and_drives_context():
    def chunk(chunk_id: str, text: str, distance: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk_id,
            doc_id=chunk_id.split("_")[0],
            source="MEPC",
            file_name=f"{chunk_id}.pdf",
            page_number=2,
            clause_number="",
            element_type="text",
            distance=distance,
            text=text,
        )

    pool = [
        chunk(
            "generic_intro",
            "This paper reviews LNG traffic and provides modelling background.",
            0.04,
        ),
        chunk(
            "requested_fact",
            "Increased traffic introduces underwater radiated noise and the risk "
            "of whale strikes, while invasive species disrupt food webs.",
            0.20,
        ),
    ]
    row = {
        "question": "GoC LNG 운반선 통행 증가의 해양 환경 영향은?",
        "_text_document_route": {
            "feature_fallback_terms": ["underwater radiated noise", "whale strike"]
        },
    }
    selected, compact, meta = build_fast_context_and_chunks(
        pool, row, chunks_dir=None
    )

    assert selected[0].chunk_id == "requested_fact"
    assert "underwater radiated noise" in compact
    assert meta["fast_evidence_slots"][0] == "feature_fallback"


def test_feature_fallback_prefers_atomic_clause_over_parent_page():
    def chunk(chunk_id: str, text: str) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk_id,
            doc_id="mepc_seemp",
            source="MEPC",
            file_name="MEPC 84-6-21.pdf",
            page_number=3,
            clause_number="",
            element_type="text",
            distance=0.2,
            text=text,
        )

    target = (
        "A ship rated as D for three consecutive years or rated as E shall not "
        "be issued a Statement of Compliance unless corrective actions are "
        "developed, reflected in the SEEMP and the SEEMP is updated and verified."
    )
    parent = (
        "The Administration shall determine whether data are reported. "
        "Several unrelated verification paragraphs follow. " + target
    )
    row = {
        "question": "D 3년 연속 또는 E 등급 선박의 Statement of Compliance 발급 조건은?",
        "_text_document_route": {
            "feature_fallback_terms": ["shall not be issued a"]
        },
    }
    selected, compact, meta = build_fast_context_and_chunks(
        [chunk("parent", parent), chunk("atomic", target)], row, chunks_dir=None
    )

    assert selected[0].chunk_id == "atomic"
    assert "corrective actions" in compact
    assert "determine whether data" not in compact
    assert meta["fast_evidence_slots"]
    assert all(slot == "feature_fallback" for slot in meta["fast_evidence_slots"])


def test_table_detection_does_not_treat_korean_word_substrings_as_rows():
    assert not is_table_question("LNG 운반선 통행 증가의 해양 환경 영향은?")
    assert not is_table_question("MSC 111 발표 결과와 규제 reporting 영향을 요약해줘")


def test_table_detection_keeps_explicit_table_and_domain_fields():
    assert is_table_question("표에서 제4차 정기검사 열의 값을 알려줘")
    assert is_table_question("tcorr와 최소 두께를 알려줘")


def test_page_only_fast_citations_are_normalized_to_evidence_ids():
    chunk = RetrievedChunk(
        chunk_id="fact",
        doc_id="doc",
        source="MSC",
        file_name="MSC 111-5-7.pdf",
        page_number=5,
        clause_number="",
        element_type="text",
        distance=0.1,
        text="evidence text",
    )
    answer = "- 운항 시험과 시뮬레이션을 포함합니다 [p.5].\n- 연구도 포함합니다 [1, p.5]."
    assert _normalize_fast_page_citations(answer, [chunk]) == (
        "- 운항 시험과 시뮬레이션을 포함합니다 [1].\n- 연구도 포함합니다 [1]."
    )


def test_numbered_source_list_contract_detects_omitted_items():
    source = "Documentation: " + " ".join(
        f"{number}) item {number}" for number in range(1, 7)
    )
    evidence = RetrievedChunk(
        chunk_id="list",
        doc_id="doc",
        source="DNV",
        file_name="DNV-CP-0001.pdf",
        page_number=6,
        clause_number="3",
        element_type="text",
        distance=0.1,
        text=source,
    )
    question = "제출해야 하는 필수 문서 목록에는 어떤 항목들이 포함됩니까?"
    requirements = analyze_requirements(question, {})
    incomplete = (
        "## 1) 핵심 요약\n- 1) 첫째, 2) 둘째 항목입니다. [1]\n"
        "## 2) 선박 운항/업무 영향\n- 검색 근거에서 확인되지 않음\n"
        "## 3) 추후 확인 필요사항\n- 검색 근거에서 확인되지 않음\n"
        "## 4) 관련 선급 Rule / Guidance\n- DNV-CP-0001 [1]"
    )
    _, warnings = validate_answer_requirements(
        incomplete, requirements, [evidence]
    )
    assert "requested_numbered_list_incomplete" in warnings

    complete = incomplete.replace(
        "1) 첫째, 2) 둘째",
        "1) 첫째, 2) 둘째, 3) 셋째, 4) 넷째, 5) 다섯째, 6) 여섯째",
    )
    _, warnings = validate_answer_requirements(complete, requirements, [evidence])
    assert "requested_numbered_list_incomplete" not in warnings


def test_clean_korean_value_definition_and_list_intents_are_detected():
    value = analyze_requirements("2019년 대비 2024년 탄소 집약도는 어느 정도 감소했습니까?")
    definition = analyze_requirements("administrative requirement의 정의는 무엇인가요?")
    list_request = analyze_requirements("형식승인을 위해 제출해야 하는 서류 목록은 무엇인가요?")
    assert "value" in value.facets
    assert "definition" in definition.facets
    assert "list" in list_request.facets


def test_technical_quantity_questions_are_value_intents():
    for question in (
        "전압 조건은 어떻게 됩니까?",
        "시험 온도는 무엇입니까?",
        "요구 두께와 압력은 얼마입니까?",
        "샤프트의 원주 속도 조건은 어떻게 됩니까?",
    ):
        assert "value" in analyze_requirements(question).facets


def test_korean_word_count_and_device_list_are_preserved():
    two_issues = analyze_requirements("최근 논의의 두 가지 주요 이슈는 무엇인가요?")
    device_list = analyze_requirements(
        "어느 장치들에 대해 최소 몇 시간 이상 급전해야 합니까?"
    )
    assert two_issues.requested_count == 2
    assert "list" in two_issues.facets
    assert "list" in device_list.facets


def test_dash_source_list_contract_detects_hollow_list_answer():
    evidence = RetrievedChunk(
        chunk_id="dash-list",
        doc_id="doc",
        source="DNV",
        file_name="DNV-CP-0001.pdf",
        page_number=6,
        clause_number="2",
        element_type="text",
        distance=0.1,
        text="Documentation:\n— application\n— drawing\n— material\n— test report",
    )
    requirements = analyze_requirements("제출 서류 목록은 무엇인가요?", {})
    hollow = (
        "## 1) 핵심 요약\n- 제출 서류는 다음과 같습니다. [1]\n"
        "## 2) 선박 운항/업무 영향\n- 확인되지 않음\n"
        "## 3) 추후 확인 필요사항\n- 확인되지 않음\n"
        "## 4) 관련 선급 Rule / Guidance\n- DNV-CP-0001 [1]"
    )
    _, warnings = validate_answer_requirements(hollow, requirements, [evidence])
    assert "requested_source_list_incomplete" in warnings


def test_parallel_condition_values_must_both_survive_generation():
    evidence = RetrievedChunk(
        chunk_id="parallel-values",
        doc_id="doc",
        source="DNV",
        file_name="DNV-CP-0001.pdf",
        page_number=11,
        clause_number="3.3",
        element_type="text",
        distance=0.1,
        text=(
            "Circumferential velocity should be 6 m/s for oil or water "
            "lubrication and should be 3 m/s for grease lubrication."
        ),
    )
    requirements = analyze_requirements(
        "윤활 방식에 따른 원주 속도는 얼마인가요?", {}
    )
    incomplete = (
        "## 1) 핵심 요약\n- 오일 또는 수분 윤활은 6 m/s이다. [1]\n"
        "## 2) 선박 운항/업무 영향\n- 확인되지 않음\n"
        "## 3) 추후 확인 필요사항\n- 확인되지 않음\n"
        "## 4) 관련 선급 Rule / Guidance\n- DNV-CP-0001 [1]"
    )
    _, warnings = validate_answer_requirements(incomplete, requirements, [evidence])
    assert "requested_parallel_values_incomplete" in warnings


def test_parallel_subjects_deadline_and_exception_are_contract_items():
    parallel = RetrievedChunk(
        chunk_id="parallel-subjects",
        doc_id="doc",
        source="KR",
        file_name="rules.pdf",
        page_number=1,
        clause_number="1",
        element_type="text",
        distance=0.1,
        text=(
            "액체관의 경우 설계압력의 1.5 배로 시험한다. "
            "화물증기관의 경우 최대사용압력의 1.5 배로 시험한다."
        ),
    )
    req = analyze_requirements(
        "액체관과 화물증기관의 압력시험 기준은 각각 어떻게 되나요?"
    )
    answer = (
        "## 1) 핵심 요약\n- 화물증기관은 최대사용압력의 1.5 배이다. [1]\n"
        "## 2) 선박 운항/업무 영향\n- 확인되지 않음\n"
        "## 3) 추후 확인 필요사항\n- 확인되지 않음\n"
        "## 4) 관련 선급 Rule / Guidance\n- rules.pdf [1]"
    )
    _, warnings = validate_answer_requirements(answer, req, [parallel])
    assert "requested_parallel_subjects_incomplete" in warnings

    deadline = RetrievedChunk(
        chunk_id="deadline",
        doc_id="doc",
        source="DNV",
        file_name="rule.pdf",
        page_number=2,
        clause_number="2",
        element_type="text",
        distance=0.1,
        text="Final reporting shall be presented within two (2) weeks after termination.",
    )
    req = analyze_requirements("최종 보고서는 언제까지 제출해야 합니까?")
    _, warnings = validate_answer_requirements(answer, req, [deadline])
    assert "requested_deadline_value_missing" in warnings

    exception = RetrievedChunk(
        chunk_id="exception",
        doc_id="doc",
        source="KR",
        file_name="rule.pdf",
        page_number=3,
        clause_number="3",
        element_type="text",
        distance=0.1,
        text=(
            "밸브는 용융점이 925 °C를 넘는 재료로 제작되어야 한다. "
            "Fail-safe operation is not compromised, internal parts may be of a lower melting point."
        ),
    )
    req = analyze_requirements("밸브 재료는 어떤 온도 요건을 충족해야 합니까?")
    basic_only = answer.replace(
        "화물증기관은 최대사용압력의 1.5 배이다.",
        "밸브는 용융점이 925 °C를 넘는 재료여야 한다.",
    )
    _, warnings = validate_answer_requirements(basic_only, req, [exception])
    assert "requested_exception_missing" in warnings


def test_direct_location_evidence_rejects_false_negative_answer():
    evidence = RetrievedChunk(
        chunk_id="test-sites",
        doc_id="doc",
        source="DNV",
        file_name="DNV-CP-0097.pdf",
        page_number=5,
        clause_number="1.2",
        element_type="text",
        distance=0.1,
        text=(
            "Type tests shall be carried out at one of the Society's laboratories, "
            "at a recognized independent laboratory, or at the manufacturer's "
            "premises in the presence of a surveyor."
        ),
    )
    requirements = analyze_requirements(
        "형식 승인을 위한 유형 시험은 어떤 장소에서 실시될 수 있습니까?", {}
    )
    negative = (
        "## 1) 핵심 요약\n- 시험 장소는 검색 근거에서 확인되지 않음. [1]\n"
        "## 2) 선박 운항/업무 영향\n- 확인되지 않음\n"
        "## 3) 추후 확인 필요사항\n- 확인되지 않음\n"
        "## 4) 관련 선급 Rule / Guidance\n- DNV-CP-0097 [1]"
    )
    _, warnings = validate_answer_requirements(negative, requirements, [evidence])
    assert "false_negative_despite_direct_evidence" in warnings
