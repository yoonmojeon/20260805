from __future__ import annotations

import app


def _fake_result(question: str, route: str, model: str) -> dict:
    return {
        "answer": f"{route}:{question}",
        "route": {
            "route": route,
            "confidence": 1.0,
            "method": "forced",
            "reason": "test",
        },
        "history": [],
        "dialogue_state": {},
        "files": [],
        "evidence_table": [],
        "related_tables": [],
        "llm_model": model,
    }


def test_gradio_has_four_workflow_tabs() -> None:
    config = app.demo.get_config_file()
    components = config.get("components") or []
    tabs = [
        component.get("props", {}).get("label")
        for component in components
        if component.get("type") == "tabitem"
    ]
    assert tabs == ["통합 질문", "문서 검색", "운항 정보", "보고서 관리"]
    assert sum(1 for component in components if component.get("type") == "state") == 9


def test_ui_has_only_gemma_and_llama_without_routing_strategy() -> None:
    config = app.demo.get_config_file()
    components = config.get("components") or []
    dropdowns = [
        component for component in components if component.get("type") == "dropdown"
    ]
    assert len(dropdowns) == 1
    assert dropdowns[0]["props"]["value"] == "gemma4:12b"
    assert [value for _label, value in dropdowns[0]["props"]["choices"]] == [
        "gemma4:12b",
        "llama3.1:8b",
    ]
    radio_labels = [
        component.get("props", {}).get("label")
        for component in components
        if component.get("type") == "radio"
    ]
    assert "라우팅 방식" not in radio_labels


def test_document_tab_forces_rag(monkeypatch) -> None:
    captured: dict = {}

    def fake_handle(question, history, **kwargs):
        captured.update(kwargs)
        return _fake_result(question, "rag", kwargs["llm_model"])

    monkeypatch.setattr(app, "handle_question", fake_handle)
    app.document_chat_fn("문서 질문", [], {}, "gemma4:12b")
    assert captured["force_route"] == "rag"
    assert captured["use_llm_router"] is True


def test_ops_tab_forces_ops(monkeypatch) -> None:
    captured: dict = {}

    def fake_handle(question, history, **kwargs):
        captured.update(kwargs)
        return _fake_result(question, "ops", kwargs["llm_model"])

    monkeypatch.setattr(app, "handle_question", fake_handle)
    app.ops_chat_fn("운항 질문", [], {}, "llama3.1:8b")
    assert captured["force_route"] == "ops"
    assert captured["use_llm_router"] is True


def test_route_banner_is_korean_and_hides_raw_router_reason() -> None:
    banner = app._route_banner(
        {
            "route": "rag",
            "confidence": 0.98,
            "method": "llm",
            "reason": "The question asks for an English technical guideline.",
        },
        "gemma4:12b",
        {"end_to_end_latency_ms": 1234},
    )
    assert "확인 자료: 문서 검색" in banner
    assert "LLM 자동 분류" in banner
    assert "응답 1.2초" in banner
    assert "The question asks" not in banner


def test_report_manager_reads_only_real_generated_files(tmp_path, monkeypatch) -> None:
    report = tmp_path / "NoonReport_TEST.docx"
    report.write_bytes(b"docx")
    (tmp_path / "debug.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(app, "REPORTS_DIR", tmp_path)

    rows = app._report_inventory()
    assert [row["name"] for row in rows] == ["NoonReport_TEST.docx"]
    assert rows[0]["status"] == "AI 초안 · 검토 필요"
    assert "NoonReport TEST" in app.build_report_manager_html()
