from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
RAG_SCRIPTS = ROOT / "rag" / "scripts"
if str(RAG_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RAG_SCRIPTS))

from table_qa_answer import build_deterministic_table_answer  # noqa: E402
from table_query_parser import parse_table_query  # noqa: E402


def _chunk(chunk_id: str, text: str, table_id: str = "t1") -> SimpleNamespace:
    return SimpleNamespace(
        chunk_id=chunk_id,
        chunk_type="table_row",
        table_id=table_id,
        text=text,
        file_name="rule.pdf",
        page_number=1,
    )


def _answer(question: str, chunks: list[SimpleNamespace]) -> str | None:
    parsed = parse_table_query(question).to_dict()
    row = {"question": question}
    return build_deterministic_table_answer(
        row,
        chunks,
        debug={"parsed_query": parsed, "selected_table_id": chunks[0].table_id},
    )


def _answer_with_selected(
    question: str, chunks: list[SimpleNamespace], selected_table: str
) -> str | None:
    parsed = parse_table_query(question).to_dict()
    return build_deterministic_table_answer(
        {"question": question},
        chunks,
        debug={"parsed_query": parsed, "selected_table_id": selected_table},
    )


def test_same_file_inspection_scope_conflict_reports_both_tables() -> None:
    question = "빌지저장탱크는 제4차 이후 정기검사에서 내부검사 대상인가?"
    chunks = [
        SimpleNamespace(
            chunk_id="p59",
            chunk_type="table_row",
            table_id="table_p59",
            text=(
                "열1=구역: 빌지저장탱크 (Bilge Holding Tank) | "
                "열2=제1차 정기검사: ○ | 열3=제2차 정기검사: ○ | "
                "열4=제3차 정기검사: ○ | 열5=제4차 및 이후 정기검사: ○"
            ),
            file_name="1편_2025.pdf",
            page_number=59,
        ),
        SimpleNamespace(
            chunk_id="p63",
            chunk_type="table_row",
            table_id="table_p63",
            text=(
                "열1=정기검사 구분 탱크: 연료유탱크, 윤활유탱크, 청수탱크, "
                "빌지저장탱크 (Bilge Holding Tank) | 열2=제1차 정기검사: △ | "
                "열3=제2차 정기검사: △ | 열4=제3차 정기검사: △ | "
                "열5=제4차 및 이후 정기검사: △"
            ),
            file_name="1편_2025.pdf",
            page_number=63,
        ),
    ]
    answer = _answer_with_selected(question, chunks, selected_table="table_p63")
    assert answer and "하나의 값으로 단정할 수 없습니다" in answer
    assert "**○**" in answer and "**△**" in answer
    assert "p.59" in answer and "p.63" in answer


def test_hatch_manholes_select_secondary_barrier_steel_column() -> None:
    question = (
        "창구와 맨홀을 2차방벽 또는 방벽간 구역에 설치할 때 "
        "어떤 강재를 사용해야 하는가?"
    )
    answer = _answer(
        question,
        [
            _chunk(
                "row4",
                "표: 저온구역 의장품\n"
                "열1=~와 해당 의장품: 창구, 맨홀 (덮개, 코밍 포함, 피팅류 제외) | "
                "열3=1차 방벽: 저온강 | "
                "열4=2차방벽 및 방벽간 구역: 저온강 | "
                "열5=2차 방벽 이면구역: NA",
            )
        ],
    )
    assert answer and "저온강" in answer


def test_definition_selects_same_atomic_row_not_sibling_summary_value() -> None:
    question = "좌굴패널은 어떤 판 패널을 의미하는가?"
    chunks = [
        _chunk(
            "row21",
            "표: 용어의 정의\n열1=용어: 좌굴패널(buckling panel) | "
            "열2=정의: 좌굴해석을 고려하는 요소 판 패널",
        ),
        _chunk(
            "row10",
            "표: 용어의 정의\n열1=용어: 만곡부 외판 | "
            "열2=정의: 선저외판과 선측외판 사이의 굽은 판",
        ),
    ]
    answer = _answer(question, chunks)
    assert answer and "좌굴해석을 고려하는 요소 판 패널" in answer
    assert "선저외판" not in answer


def test_test_standard_maps_to_test_method_column() -> None:
    question = "단열재료의 화재 및 화염전파 저항 시험에는 어떤 시험규격을 적용하는가?"
    answer = _answer(
        question,
        [
            _chunk(
                "row13",
                "표: 기계적 성질\n열1=시험항목: 13. 화재 및 화염전파에 대한 저항 | "
                "열2=시험방법: DIN 4102",
            )
        ],
    )
    assert answer and "DIN 4102" in answer


def test_pump_count_prefers_powered_pump_over_empty_manual_cell() -> None:
    question = "길이 25m 이상 30m 미만인 선박에는 동력 빌지펌프가 몇 대 필요한가?"
    answer = _answer(
        question,
        [
            _chunk(
                "row3",
                "표: 빌지펌프\n열1=선박의 길이: 25 m 이상 30 m 미만 | "
                "열2=펌프 독립동력펌프: 1대 | 열3=수동펌프: ─",
            )
        ],
    )
    assert answer and "1대" in answer


def test_verified_table_answer_exposes_cell_lineage_object() -> None:
    question = "길이 25m 이상 30m 미만인 선박에는 동력 빌지펌프가 몇 대 필요한가?"
    chunk = _chunk(
        "row3-lineage",
        "표: 빌지펌프\n열1=선박의 길이: 25 m 이상 30 m 미만 | "
        "열2=펌프 독립동력펌프: 1대 | 열3=수동펌프: ─",
    )
    row = {"question": question}
    answer = build_deterministic_table_answer(
        row,
        [chunk],
        debug={
            "parsed_query": parse_table_query(question).to_dict(),
            "selected_table_id": chunk.table_id,
        },
    )
    lineage = row.get("_table_evidence_object") or {}
    assert answer and lineage["cell_value"] == "1대"
    assert lineage["table_id"] == "t1"
    assert lineage["support_chunk_ids"] == ["row3-lineage"]


def test_temperature_unit_c_is_not_carbon_column_or_chemistry_topic() -> None:
    parsed = parse_table_query(
        "무할로겐 고등급 에틸렌 프로필렌 고무 절연물의 최고 허용 도체온도는 몇 °C인가?"
    )
    assert "C" not in parsed.column_entities
    assert "chemical_composition" not in parsed.table_topic_candidates


def test_normal_conductor_temperature_beats_short_circuit_temperature() -> None:
    question = (
        "무할로겐 고등급 에틸렌 프로필렌 고무 절연물의 "
        "최고 허용 도체온도는 몇 °C인가?"
    )
    answer = _answer(
        question,
        [
            _chunk(
                "row6",
                "표: 절연재료\n열1=절연재료: 무 할로겐 고등급 에틸렌 프로필렌 고무 | "
                "열2=도체 최고 > 정상 운전: 90 | 열3=허용온도 (℃) > 단락: 250",
            )
        ],
    )
    assert answer and "90" in answer
    assert "250" not in answer


def test_korean_primary_support_member_maps_to_english_structural_row() -> None:
    question = "주요 지지부재는 최종강도 검토 대상인가?"
    answer = _answer(
        question,
        [
            _chunk(
                "row3",
                "표: Table 2 Structural assessment\n"
                "열1=Structural: Primary supporting members | "
                "열3=Yielding check: Y | 열4=Buckling check: Y | "
                "열5=Ultimate strength check: Y(2)",
            )
        ],
    )
    assert answer and "Y(2)" in answer


def test_compact_wire_diameter_question_selects_range_difference() -> None:
    question = (
        "소선지름이 0.20mm 이상 1.00mm 이하일 때 "
        "최대지름과 최소지름의 허용 차이는 몇 mm인가?"
    )
    answer = _answer(
        question,
        [
            _chunk(
                "row1",
                "표: 소선 허용차\n"
                "열1=소선의: 0.20 | 열2=공칭지름: 이상 1.00 | "
                "열3=(mm): 이하 | 열4=최대인 것과 최소인 것의 차 (mm): 0.06",
            )
        ],
    )
    assert answer and "0.06" in answer


def test_mineral_content_maps_to_nominal_value_tolerance() -> None:
    question = "적층용 액상 수지의 광물 함유량은 제조자 공칭값에서 몇 %까지 벗어날 수 있는가?"
    answer = _answer(
        question,
        [
            _chunk(
                "row5",
                "표: 수지 특성\n"
                "열1=특성: 광물 함유 (적층용 수지에 대해서만) | "
                "열2=시험방법: DIN 16945 | "
                "열3=요건 (제조자에 의해 규정된 공칭값 오차%): ±5",
            )
        ],
    )
    assert answer and "±5" in answer


def test_non_end_bulkhead_selects_shell_distance() -> None:
    question = "1·2구역의 비선수미 격벽에는 외판에서 몇 m 이내까지 판 구조요건을 적용하는가?"
    answer = _answer(
        question,
        [
            _chunk(
                "row2",
                "표: 표 3.13 판 구조 요건 적용구역\n"
                "열1=구역: 선수미 격벽이외의 격벽 중 1 구역 및 2 구역 내의 격벽 | "
                "열2=선박등급: 쇄빙선, Arctic4 ∼ Arctic9 | "
                "열3=외판으로부터 거리: 1.2m 이내",
            )
        ],
    )
    assert answer and "1.2m 이내" in answer


def test_alarm_item_selects_display_location() -> None:
    question = "선급이 추가로 요구하는 기관 표시·경보항목은 어디에 표시해야 하는가?"
    answer = _answer(
        question,
        [
            _chunk(
                "row1",
                "표: 기관 표시 및 경보\n"
                "열1=기관에 따라 우리 | 열2=항: 선급이 | "
                "열3=목: 필요하다고 인정하는 항목 | "
                "열4=표 시 장 소: 우리 선급이 요구하는 개소",
            )
        ],
    )
    assert answer and "우리 선급이 요구하는 개소" in answer


def test_exact_atomic_intersection_can_rescue_wrong_routed_table() -> None:
    question = "주요 지지부재는 최종강도 검토 대상인가?"
    answer = _answer_with_selected(
        question,
        [
            _chunk(
                "noise",
                "표: 부식 허용치\n열1=구조부재: PSM의 면재 | 열2=허용치: 1.5",
                table_id="wrong",
            ),
            _chunk(
                "right",
                "표: Structural assessment\n"
                "열1=Structural: Primary supporting members | "
                "열4=Buckling check: Y | 열5=Ultimate strength check: Y(2)",
                table_id="right",
            ),
        ],
        selected_table="wrong",
    )
    assert answer and "Y(2)" in answer
    assert "1.5" not in answer


def test_underspecified_ratio_reports_all_category_cells() -> None:
    question = "주갑판 아래의 안덮개 설치비율은 얼마인가?"
    answer = _answer_with_selected(
        question,
        [
            _chunk(
                "wrong",
                "표: 안덮개의 수\n열1=위치: 선루 전단벽 | "
                "열2=SA 0: 100 % | 열3=SA 1: 100 % | "
                "열4=SA 2: 50 % | 열5=SA3: 25 %",
                table_id="wrong",
            ),
            _chunk(
                "row2",
                "표: 안덮개의 수\n열1=창의 위치: 주 갑판 아래 | "
                "열2=SA 0: 100 % | 열3=SA 1: 종류마다 1개 | "
                "열4=SA 2: 0 % | 열5=SA3: 0 %",
                table_id="right",
            )
        ],
        selected_table="wrong",
    )
    assert answer and "SA 0 → 100 %" in answer
    assert "SA 1 → 종류마다 1개" in answer
    assert "SA3 → 0 %" in answer
