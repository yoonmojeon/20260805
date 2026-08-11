"""Unit tests for retrieval mode, table markdown rebuild, and BOTH fuse helpers."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.rag_service import fuse_evidence
from services.retrieval_mode import RetrievalMode, classify_retrieval_mode
from services.table_render import parse_row_cells, rows_to_markdown, strip_embed_header


def test_retrieval_mode_text_meeting():
    mode = classify_retrieval_mode("MSC 111에서 MASS Code 관련 핵심 결정은?")
    assert mode == RetrievalMode.TEXT


def test_retrieval_mode_table_age_tank():
    mode = classify_retrieval_mode("선령 15년 초과 평형수탱크 검사 범위는?")
    assert mode == RetrievalMode.TABLE


def test_retrieval_mode_table_min_thickness_cell_lookup():
    mode = classify_retrieval_mode(
        "선박 길이 L이 170m 미만일 때 요구되는 최소 두께는 얼마인가?"
    )
    assert mode == RetrievalMode.TABLE


def test_retrieval_mode_table_tcorr():
    mode = classify_retrieval_mode(
        "화물탱크 내 구조부재 부식추가 tcorr 표에서 범주별 값은 어떻게 되나?"
    )
    assert mode in {RetrievalMode.TABLE, RetrievalMode.BOTH}


def test_retrieval_mode_both_intent_and_scope():
    mode = classify_retrieval_mode(
        "평형수탱크 검사 규정의 취지와 선령별 검사 범위를 알려줘"
    )
    assert mode == RetrievalMode.BOTH


def test_retrieval_mode_parser_scores_exposed():
    from services.retrieval_mode import table_shape_score

    score, detail = table_shape_score(
        "선박 길이 L이 170m 미만일 때 요구되는 최소 두께는 얼마인가?"
    )
    assert score >= 0.55
    assert detail.get("numeric_range") is True or detail.get("parser")


def test_retrieval_mode_file_page_forces_table():
    mode = classify_retrieval_mode("2편_2025.pdf 10페이지 구조화 표 제목은?")
    assert mode == RetrievalMode.TABLE


def test_retrieval_mode_definition_stays_text():
    mode = classify_retrieval_mode("과도한 부식(substantial corrosion)의 정의는?")
    assert mode == RetrievalMode.TEXT


def test_retrieval_mode_rule_symbol_definition_stays_text():
    mode = classify_retrieval_mode("구조 규칙에서 쓰는 tcorr 기호는 어떤 두께를 뜻하지?")
    assert mode == RetrievalMode.TEXT


def test_retrieval_mode_cii_requirements_summary_stays_text():
    mode = classify_retrieval_mode(
        "IMO 문서 기준으로 선박 탄소집약도 등급을 관리하는 요구사항을 요약해줘."
    )
    assert mode == RetrievalMode.TEXT


def test_retrieval_mode_named_society_guidance_summary_stays_text():
    mode = classify_retrieval_mode(
        "DNV의 자율운항선박 관련 지침이 강조하는 핵심 안전 원칙을 찾아줘."
    )
    assert mode == RetrievalMode.TEXT


def test_retrieval_mode_clause_procedure_stays_text():
    assert classify_retrieval_mode(
        "902절 탈급(선급등록 취소)의 적용 대상과 절차는?"
    ) == RetrievalMode.TEXT


def test_retrieval_mode_numbered_clause_with_inspection_words_stays_text():
    assert classify_retrieval_mode(
        "801절에서 검사 준비가 안 되었거나 입회자가 없을 때 검사원은 어떻게 할 수 있는가?"
    ) == RetrievalMode.TEXT


def test_retrieval_mode_rule_effective_date_stays_text():
    assert classify_retrieval_mode(
        "2025년판 제1편 규칙은 언제부터 검사 신청 선박에 적용되는가?"
    ) == RetrievalMode.TEXT


def test_retrieval_mode_rule_comparison_stays_text():
    assert classify_retrieval_mode(
        "쇠모한도를 초과한 부식과 과도한 부식의 차이는?"
    ) == RetrievalMode.TEXT


def test_retrieval_mode_word_containing_row_syllable_is_not_table_frame():
    assert classify_retrieval_mode(
        "시험 및 검사는 원칙적으로 어떻게 시행해야 하는가?"
    ) == RetrievalMode.TEXT


def test_table_rows_ordered_and_columns_preserved():
    pairs = [
        (
            "[table_row]\n열1=Age | 열2=Tank | 열3=Survey",
            {"element_id": "t1:ROW000", "chunk_type": "table_row", "table_id": "t1"},
        ),
        (
            "[table_row]\n열1=Age: <=10 | 열2=Tank: Selected | 열3=Survey: Annual",
            {"element_id": "t1:ROW001", "chunk_type": "table_row", "table_id": "t1"},
        ),
        (
            "[table_row]\n열1=Age: >15 | 열2=Tank: All tanks | 열3=Survey: Special",
            {"element_id": "t1:ROW002", "chunk_type": "table_row", "table_id": "t1"},
        ),
    ]
    # Shuffle fetch order; ROW ids must restore order
    shuffled = [pairs[0], pairs[2], pairs[1]]
    docs_shuf = [p[0] for p in shuffled]
    metas_shuf = [p[1] for p in shuffled]
    md, hits = rows_to_markdown(
        docs_shuf, metas_shuf, highlight_question=">15 All tanks", max_rows=40
    )
    lines = [ln for ln in md.splitlines() if ln.startswith("|")]
    assert "Age" in lines[0] and "Tank" in lines[0] and "Survey" in lines[0]
    assert "<=10" in lines[2]
    assert "All tanks" in lines[3] or "**All tanks**" in lines[3]
    assert isinstance(hits, list)


def test_parse_row_cells_keeps_column_order_keys():
    text = "열1=Quantity in Operations: Frequency | 열2=Permanent Variation: ±5% | 열3=Transient Variation: ±10%"
    cells = parse_row_cells(text)
    assert list(cells.keys()) == [1, 2, 3]
    assert "Frequency" in cells[1] or cells[1].endswith("Frequency")


def test_strip_embed_header():
    raw = (
        "[table_row] source=ABS file=x.pdf\n"
        "표: tid\n"
        "문서: x.pdf, 13쪽\n"
        "열1=A | 열2=B"
    )
    assert strip_embed_header(raw) == "열1=A | 열2=B"


def test_fuse_evidence_keeps_text_and_table_with_caps():
    text_hits = [
        SimpleNamespace(chunk_id=f"tx{i}", doc_id="d1", page_number=i, text=f"text-{i}")
        for i in range(10)
    ]
    table_hits = [
        SimpleNamespace(
            chunk_id=f"tb{i}",
            doc_id="d2",
            page_number=1,
            table_id="tid",
            text=f"열1=v{i}",
        )
        for i in range(10)
    ]
    fused = fuse_evidence(
        text_hits=text_hits,
        table_hits=table_hits,
        text_top_k=4,
        table_top_k=5,
        prefer="balanced",
    )
    sources = [getattr(c, "evidence_source", "") for c in fused]
    assert "text" in sources and "table" in sources
    assert sources.count("text") <= 4
    assert sources.count("table") <= 5


def test_dual_retrieval_defaults_on_when_env_unset():
    import os

    from services.rag_service import dual_retrieval_enabled

    os.environ.pop("MARITIME_RAG_DUAL", None)
    assert dual_retrieval_enabled() is True
    os.environ["MARITIME_RAG_DUAL"] = "0"
    assert dual_retrieval_enabled() is False
    os.environ["MARITIME_RAG_DUAL"] = "1"
    assert dual_retrieval_enabled() is True
    os.environ.pop("MARITIME_RAG_DUAL", None)


def test_crop_path_rebases_old_absolute_path_to_local():
    import os
    from unittest.mock import patch

    import services.rag_service as rs

    table_id = "kr_rules_abcdef12_p0042_t007"
    local_root = ROOT / "local-test-data" / "processed" / "precise_tables"
    local_crop = local_root / "abcdef12" / "p0042_t007" / "crop.png"
    old_crop = (
        r"C:\Users\user\Desktop\20260805\data\processed\precise_tables"
        r"\abcdef12\p0042_t007\crop.png"
    )

    with patch.object(rs, "PRECISE_TABLES_DIR", local_root), patch.dict(
        os.environ, {"MARITIME_ALLOW_EXTERNAL_DATA_PATHS": "0"}
    ), patch(
        "services.rag_service.Path.is_file",
        autospec=True,
        side_effect=lambda path: path == local_crop,
    ):
        resolved = rs._resolve_crop_path({"crop_path": old_crop}, table_id)

    assert Path(resolved) == local_crop


def test_crop_path_does_not_fall_back_to_external_by_default():
    import os
    from unittest.mock import patch

    import services.rag_service as rs

    external = ROOT / "external-test-data" / "crop.png"
    local_root = ROOT / "local-test-data" / "processed" / "precise_tables"
    with patch.object(rs, "PRECISE_TABLES_DIR", local_root), patch.dict(
        os.environ, {"MARITIME_ALLOW_EXTERNAL_DATA_PATHS": "0"}
    ), patch(
        "services.rag_service.Path.is_file",
        autospec=True,
        side_effect=lambda path: path == external,
    ):
        resolved = rs._resolve_crop_path({"crop_path": str(external)}, "unknown")

    assert resolved == ""


def test_both_queries_text_and_table_collections():
    """BOTH must invoke both retrieval sides and fuse hits."""
    import os
    from unittest.mock import patch

    import services.rag_service as rs
    from services.retrieval_mode import RetrievalMode

    os.environ.pop("MARITIME_RAG_DUAL", None)
    calls: list[bool] = []

    def fake_search(question, *, latency_mode, table_side):
        calls.append(bool(table_side))
        kind = "table" if table_side else "text"
        chunks = [
            SimpleNamespace(
                chunk_id=f"{kind}-{i}",
                doc_id=f"doc-{kind}",
                page_number=i + 1,
                table_id=("tid" if table_side else ""),
                text=("열1=15년" if table_side else "검사 취지는 안전 확보"),
            )
            for i in range(3)
        ]
        return {
            "row": {"question": question, "category": "table_qa" if table_side else "rule_lookup"},
            "unified_id": "table" if table_side else "text",
            "search_out": {
                "retrieved": chunks,
                "answer_mode": "table_qa" if table_side else "rag",
            },
            "retrieved": chunks,
            "timing_metrics": {},
            "evidence_source": kind,
        }

    def fake_answer(**kwargs):
        return {"answer": "테스트 답변 (근거 기반)", "timing_metrics": {}}

    class FakeRagInprocess:
        @staticmethod
        def run_answer_inprocess(**kwargs):
            return fake_answer(**kwargs)

    with patch.object(rs, "rag_index_ready", return_value=True), patch.object(
        rs, "_run_search_only", side_effect=fake_search
    ), patch.object(rs, "_related_tables_from_hits", return_value=("", [], [])), patch.object(
        rs, "_ensure_rag_path"
    ), patch("os.chdir"), patch.dict(
        "sys.modules", {"rag_inprocess": FakeRagInprocess()}
    ):
        out = rs.run_rag_query(
            "평형수탱크 검사 규정의 취지와 선령별 검사 범위를 알려줘",
            retrieval_mode=RetrievalMode.BOTH,
        )

    assert True in calls and False in calls
    meta = out.get("meta") or {}
    assert meta.get("retrieval_mode") == "both"
    assert meta.get("dual_retrieval_enabled") is True
    assert meta.get("n_text_chunks", 0) > 0
    assert meta.get("n_table_chunks", 0) > 0
    assert meta.get("n_merged_chunks", 0) > 0
    debug = meta.get("debug") or {}
    assert debug.get("mode") == "BOTH"
    assert debug.get("text_hits", 0) > 0
    assert debug.get("table_hits", 0) > 0
    assert "테스트 답변" in (out.get("answer") or "")
    assert "text_hits=" not in (out.get("answer") or "")


if __name__ == "__main__":
    test_retrieval_mode_text_meeting()
    test_retrieval_mode_table_age_tank()
    test_retrieval_mode_both_intent_and_scope()
    test_table_rows_ordered_and_columns_preserved()
    test_parse_row_cells_keeps_column_order_keys()
    test_strip_embed_header()
    test_fuse_evidence_keeps_text_and_table_with_caps()
    test_dual_retrieval_defaults_on_when_env_unset()
    test_both_queries_text_and_table_collections()
    print("ok")
