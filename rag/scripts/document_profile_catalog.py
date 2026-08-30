"""Document-level metadata sidecar built without changing the vector index."""
from __future__ import annotations

import json
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any


PROFILE_FILE_NAME = "document_profiles_v1.json"
CLASS_SOURCES = {"DNV", "KR", "ABS", "LR"}
CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])((?:DNV|KR|ABS|LR)[-_ ](?:CG|CP|RP|RU|NV|SI|OS)[-_ ]?[A-Z0-9.-]+)",
    re.I,
)
REV_RE = re.compile(r"(?:[-_ ]|\b)Rev[._ -]?(\d+)", re.I)
ADD_RE = re.compile(r"(?:[-_ ]|\b)Add[._ -]?(\d+)", re.I)


def _display_code(file_name: str) -> str:
    match = CODE_RE.search(file_name or "")
    if match:
        return re.sub(
            r"\.PDF$", "", re.sub(r"[_ ]+", "-", match.group(1)).upper()
        )
    stem = Path(file_name or "").stem
    stem = re.sub(r"(?<=[a-z])for(?=[A-Z])", " for ", stem)
    stem = re.sub(r"(?<=[a-z])and(?=[A-Z])", " and ", stem)
    stem = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem)
    return re.sub(r"[_]+", " ", stem).strip()


def _document_family(file_name: str, source_type: str) -> str:
    blob = str(file_name or "").upper()
    if "-CG-" in blob or "GUIDELINE" in blob or "GUIDE" in blob or "GUIDANCE" in blob:
        return "class_guideline"
    if "-CP-" in blob:
        return "type_approval_programme"
    if "-RP-" in blob:
        return "recommended_practice"
    if "-RU-" in blob or "RULE" in blob:
        return "class_rule"
    return source_type or "document"


def _purpose_and_use(source: str, family: str, source_type: str) -> tuple[str, str]:
    if family == "class_guideline":
        return "설계·승인 방법과 적용 절차를 안내하는 선급 가이드", "개념설계, AIP/선급 승인 준비, 위험성평가 계획 시"
    if family == "type_approval_programme":
        return "제품·시스템의 형식승인 범위와 시험 기준을 정하는 프로그램", "TA 신청 범위와 제출 시험성적을 확인할 때"
    if family in {"class_rule", "class_rulebook"} or source in CLASS_SOURCES:
        return "선급 적용범위와 기술·검사 요구사항을 정하는 규칙", "설계 적합성, 도면승인, 검사·선급부호 요건을 확인할 때"
    if source_type == "resolution":
        return "IMO가 채택한 결의·규제 문서", "채택 상태, 의무 근거와 발효 조건을 확인할 때"
    if source_type == "meeting_record":
        return "IMO 위원회 회의 의제·제출·결과 문서", "논의 경과와 당시 결론·미확정 쟁점을 추적할 때"
    return "색인된 해사 규정·기술 문서", "질문 주제의 직접 근거와 적용범위를 확인할 때"


def _revision_fields(file_name: str) -> tuple[str, str, str]:
    rev = REV_RE.search(file_name or "")
    add = ADD_RE.search(file_name or "")
    base = REV_RE.sub("", ADD_RE.sub("", Path(file_name or "").stem))
    base = re.sub(r"[-_ ]+", "-", base).strip("-").lower()
    return base, rev.group(1) if rev else "", add.group(1) if add else ""


def build_document_profiles(sparse_db: Path, out_path: Path) -> dict[str, Any]:
    """Build one compact profile per PDF from the existing FTS sidecar."""
    connection = sqlite3.connect(sparse_db)
    connection.row_factory = sqlite3.Row
    try:
        rows = list(
            connection.execute(
                """SELECT doc_id, MIN(file_name) AS file_name,
                          MIN(source) AS source, MIN(publisher) AS publisher,
                          MIN(source_type) AS source_type,
                          MIN(session_org) AS session_org,
                          MIN(session_number) AS session_number,
                          MIN(document_status) AS document_status,
                          COUNT(*) AS chunk_count, MIN(page_number) AS first_page
                   FROM chunks
                   GROUP BY doc_id
                   ORDER BY MIN(source), MIN(file_name)"""
            )
        )
        meta = {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key, value FROM index_meta")
        }
    finally:
        connection.close()

    documents: list[dict[str, Any]] = []
    by_base: dict[str, list[str]] = {}
    for row in rows:
        source = str(row["source"] or "").upper()
        file_name = str(row["file_name"] or "")
        source_type = str(row["source_type"] or "unknown")
        from imo_doc_classify import classify_imo_filename

        role = classify_imo_filename(file_name)
        if role in {"administrative", "resolution", "amendments"}:
            source_type = role
        family = _document_family(file_name, source_type)
        purpose, when_to_use = _purpose_and_use(source, family, source_type)
        base, revision, addendum = _revision_fields(file_name)
        item = {
            "doc_id": str(row["doc_id"] or ""),
            "file_name": file_name,
            "display_code": _display_code(file_name),
            "publisher": str(row["publisher"] or source),
            "source": source,
            "source_type": source_type,
            "document_family": family,
            "document_status": str(row["document_status"] or "unknown"),
            "session_org": str(row["session_org"] or ""),
            "session_number": row["session_number"],
            "base_document_key": base,
            "revision": revision,
            "addendum": addendum,
            "purpose": purpose,
            "when_to_use": when_to_use,
            "chunk_count": int(row["chunk_count"] or 0),
        }
        documents.append(item)
        by_base.setdefault(base, []).append(item["doc_id"])
    for item in documents:
        item["related_doc_ids"] = [
            value for value in by_base.get(item["base_document_key"], [])
            if value != item["doc_id"]
        ]
    payload = {
        "schema_version": "document-profile-v1",
        "source_fingerprint": meta.get("fingerprint", ""),
        "document_count": len(documents),
        "documents": documents,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    load_document_profiles.cache_clear()
    return payload


@lru_cache(maxsize=4)
def load_document_profiles(path: str) -> dict[str, dict[str, Any]]:
    profile_path = Path(path)
    if not profile_path.exists():
        return {}
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    return {
        str(item.get("doc_id") or ""): item
        for item in payload.get("documents", [])
        if item.get("doc_id")
    }


def profile_for_chunk(chunk: Any, path: Path) -> dict[str, Any]:
    return load_document_profiles(str(path.resolve())).get(
        str(getattr(chunk, "doc_id", "") or ""), {}
    )
