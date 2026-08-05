"""Audit generated answers against the UI answer/evidence contract.

This does not judge a saved question against a hard-coded answer.  It checks
properties that must hold for every question: four-section structure, Korean
rendering, citation placement, and exact citation-to-Evidence-Table mapping.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SECTION_RE = re.compile(r"^##\s*([1-4])\)", re.M)
CITATION_RE = re.compile(r"\[(\d+)\]")
KOREAN_RE = re.compile(r"[가-힣]")
LATIN_RE = re.compile(r"[A-Za-z]")


def _records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def audit_record(record: dict[str, Any]) -> dict[str, Any]:
    answer = str(record.get("answer") or "")
    evidence = record.get("evidence_table") or []
    evidence_ids = {
        int(match.group(1))
        for row in evidence
        if isinstance(row, dict)
        for match in [CITATION_RE.search(str(row.get("citation_id") or ""))]
        if match
    }
    cited_ids = {int(value) for value in CITATION_RE.findall(answer)}
    headings = SECTION_RE.findall(answer)

    failures: list[str] = []
    if headings != ["1", "2", "3", "4"]:
        failures.append("four_section_contract")
    if not answer.strip():
        failures.append("empty_answer")
    if cited_ids - evidence_ids:
        failures.append(
            "citation_without_evidence:"
            + ",".join(map(str, sorted(cited_ids - evidence_ids)))
        )
    if evidence_ids - cited_ids:
        failures.append(
            "unused_evidence:"
            + ",".join(map(str, sorted(evidence_ids - cited_ids)))
        )

    bullet_lines = [
        line.strip()
        for line in answer.splitlines()
        if re.match(r"^[-*]\s+", line.strip())
    ]
    factual_bullets = [
        line
        for line in bullet_lines
        if not any(
            marker in line
            for marker in (
                "검색 근거에서 직접 확인되는",
                "추가 확인 필요사항이 별도로",
                "검색 근거에 없거나 해당하지",
                "검색 근거에서 확인되지 않음",
            )
        )
    ]
    uncited = [line for line in factual_bullets if not CITATION_RE.search(line)]
    if uncited:
        failures.append(f"uncited_factual_bullets:{len(uncited)}")

    # Natural Korean answers may retain standard names and clause titles in
    # English.  Require Korean to remain the dominant narrative language.
    korean_chars = len(KOREAN_RE.findall(answer))
    latin_chars = len(LATIN_RE.findall(answer))
    if korean_chars == 0 or korean_chars < latin_chars * 0.35:
        failures.append("korean_narrative_insufficient")

    malformed_evidence = [
        row
        for row in evidence
        if not isinstance(row, dict)
        or not str(row.get("file_name") or "").strip()
        or row.get("page") in (None, "")
        or not str(row.get("chunk_id") or "").strip()
    ]
    if malformed_evidence:
        failures.append(f"malformed_evidence_rows:{len(malformed_evidence)}")

    return {
        "question_id": record.get("question_id") or "unseen",
        "question": record.get("question"),
        "pass": not failures,
        "failures": failures,
        "section_count": len(headings),
        "factual_bullet_count": len(factual_bullets),
        "citation_ids": sorted(cited_ids),
        "evidence_ids": sorted(evidence_ids),
        "korean_chars": korean_chars,
        "latin_chars": latin_chars,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/logs/answer_quality_contract_report.json"),
    )
    args = parser.parse_args()

    audited: list[dict[str, Any]] = []
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in _records(payload):
            result = audit_record(record)
            result["input"] = str(path)
            audited.append(result)

    report = {
        "pass": all(item["pass"] for item in audited),
        "case_count": len(audited),
        "passed": sum(bool(item["pass"]) for item in audited),
        "failed": sum(not item["pass"] for item in audited),
        "results": audited,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
