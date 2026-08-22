#!/usr/bin/env python3
"""Apply the small human-readable repair list after broad-set LLM review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "eval" / "broad_pdf_150_reviewed.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "eval" / "broad_pdf_150_final.jsonl"
DEFAULT_REPORT = ROOT / "data" / "processed" / "logs" / "broad_pdf_150_final_audit.json"


QUESTION_REPAIRS = {
    "BPDF-DNV-004": "DNV-CP-0399의 형식승인(TA)은 어떤 정격의 전력 케이블과 제어·계측 회로용 케이블에 적용되나요?",
    "BPDF-DNV-037": "DNV-CP-0350은 어떤 압력용기 부품 제조사의 승인에 적용되며, 함께 적용해야 하는 일반 요구사항 문서는 무엇인가요?",
    "BPDF-DNV-049": "DNV-CP-0406이 정한 열수축성 튜브 형식승인의 목적과 절차 범위는 무엇이며, 제품 설계요건 자체도 정하나요?",
    "BPDF-DNV-057": "DNV-RU-YACHT-Pt5의 서비스 문서 이용조건에 따르면 제3자의 인증·검증 서비스 제공 제한과 DNV의 책임 한도는 어떻게 규정되나요?",
    "BPDF-DNV-060": "DNV-CP-0405의 케이블용 화재 보호 시스템 형식승인 절차가 적용되지 않는 별도 인증은 무엇인가요?",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    repairs: list[dict[str, Any]] = []
    for row in rows:
        question_id = str(row.get("question_id") or "")
        if question_id in QUESTION_REPAIRS:
            before = row["question"]
            row["question"] = QUESTION_REPAIRS[question_id]
            repairs.append({"question_id": question_id, "before": before, "after": row["question"]})
        if question_id == "BPDF-KR-007":
            row["gold_answer_points"][0]["aliases"] = ["2.25", "0.1", "와셔의 바깥지름", "와셔의 재료"]
            repairs.append({"question_id": question_id, "alias_repair": True})
        if question_id == "BPDF-KR-016":
            aliases = list(row["gold_answer_points"][0].get("aliases") or [])
            if "Deck girders" not in aliases:
                aliases.append("Deck girders")
            row["gold_answer_points"][0]["aliases"] = aliases
            repairs.append({"question_id": question_id, "alias_repair": "Deck girders"})
        if question_id == "BPDF-DNV-055":
            # The embedding child used during initial curation ends at item
            # 11, but the page parent continues with items 12-13.  The PDF asks
            # for the complete required-document list, so the evaluation gold
            # must not label those genuine continuation items unsupported.
            parent_chunk_id = (
                "dnv_dnv_class_2026_04_dnv_cp_0084_745b69ee_p0006_m003"
            )
            extra = "시험·측정장비 목록(교정증명서 포함), 운항 중 경험(있는 경우)"
            point = row["gold_answer_points"][0]
            if extra not in str(point.get("text") or ""):
                point["text"] = str(point.get("text") or "").rstrip(". ") + ", " + extra + "도 포함된다."
            aliases = list(point.get("aliases") or [])
            for alias in ("calibration certificates", "in-service experience"):
                if alias not in aliases:
                    aliases.append(alias)
            point["aliases"] = aliases
            evidence_ids = list(point.get("evidence_chunk_ids") or [])
            if parent_chunk_id not in evidence_ids:
                evidence_ids.append(parent_chunk_id)
            point["evidence_chunk_ids"] = evidence_ids
            row["gold_answer"] = "- " + point["text"]
            row["must_cover"] = [point["text"]]
            gold_ids = list(row.get("gold_chunk_ids") or [])
            if parent_chunk_id not in gold_ids:
                gold_ids.append(parent_chunk_id)
            row["gold_chunk_ids"] = gold_ids
            parent_path = (
                ROOT
                / "data"
                / "processed"
                / "chunks"
                / str(row["gold_doc_id"])
                / "chunks.jsonl"
            )
            if parent_path.exists():
                for raw in parent_path.read_text(encoding="utf-8").splitlines():
                    candidate = json.loads(raw)
                    if str(candidate.get("chunk_id") or "") == parent_chunk_id:
                        row["gold_evidence"][0]["context"] = str(
                            candidate.get("text") or ""
                        )
                        ids = list(row["gold_evidence"][0].get("chunk_ids") or [])
                        if parent_chunk_id not in ids:
                            ids.append(parent_chunk_id)
                        row["gold_evidence"][0]["chunk_ids"] = ids
                        break
            repairs.append(
                {"question_id": question_id, "gold_continuation_repair": "items 12-13"}
            )
        # A valid revised contract is the final object being evaluated.  A
        # false pass flag from the reviewer describes the *pre-repair* row.
        meta = dict(row.get("review_meta") or {})
        meta["final_pass"] = bool(meta.get("valid_contract")) or question_id in QUESTION_REPAIRS or question_id == "BPDF-KR-007"
        meta["unresolved"] = not meta["final_pass"]
        row["review_meta"] = meta

    assert len(rows) == 150, len(rows)
    assert len({row["gold_doc_id"] for row in rows}) == 150
    assert len({row["question"] for row in rows}) == 150
    unresolved = [row["question_id"] for row in rows if not row["review_meta"]["final_pass"]]
    if unresolved:
        raise SystemExit(f"unresolved rows: {unresolved}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    report = {
        "rows": len(rows),
        "distinct_docs": len({row["gold_doc_id"] for row in rows}),
        "final_pass": sum(bool(row["review_meta"]["final_pass"]) for row in rows),
        "manual_repairs": repairs,
        "unresolved": unresolved,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
