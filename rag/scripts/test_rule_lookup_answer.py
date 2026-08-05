"""Unit checks for rule_lookup_answer repair pipeline."""
from __future__ import annotations

from dataclasses import dataclass

from rule_lookup_answer import finalize_rule_lookup_answer


@dataclass
class _Chunk:
    chunk_id: str
    file_name: str
    text: str
    page_number: int = 1


def _sample_chunks() -> list[_Chunk]:
    return [
        _Chunk("1", "DNV-CG-0264.pdf", "2 Objective guidance for safe implementation of autoremote vessel functions and AROS notations.", 9),
        _Chunk("2", "DNV-CG-0264.pdf", "8.1 General principles for autoremote vessel design fault tolerance.", 24),
        _Chunk("3", "DNV-RU-OU-0103.pdf", "23.4 Application Smart notation may be applied to units in operation.", 106),
        _Chunk("4", "DNV-RU-UWT-Pt5.pdf", "Supporting underwater technology rules reference.", 172),
    ]


def test_strips_hallucination_and_rebuilds_section4() -> None:
    raw = """## 1) 핵심 요약
- **DNV-CG-0264.pdf**: autoremote guidance [1]
- **DNV-RU-OU-0103.pdf**: Smart notation [3]

## 2) 선박 운항/업무 영향
- **DNV-CG-0264.pdf**: autoremote guidance work process [1]
- AROS notation 정의 필요 [2]

## 3) 추후 확인 필요사항
- DNV-RU-SHIP Pt.6 Ch.12 Sec.2 확인 필요

## 4) 관련 선급 Rule / Guidance
- **DNV-RU-SHIP Pt.6 Ch.12**: autoremote [9]
- **DNV-CG-0508**: Smart notation [6]
"""
    chunks = _sample_chunks()
    out, notes = finalize_rule_lookup_answer(raw, chunks)

    assert "DNV-RU-SHIP" not in out
    assert "DNV-CG-0508" not in out
    assert "DNV-RU-UWT-Pt5.pdf" in out
    assert "## 4)" in out
    assert any("duplicate" in n.lower() or "dedupe" in n.lower() or "§2" in n for n in notes)
    assert "DNV-CG-0264.pdf" not in out.split("## 4)")[1] or "본 검색 context" in out


if __name__ == "__main__":
    test_strips_hallucination_and_rebuilds_section4()
    print("ok")
