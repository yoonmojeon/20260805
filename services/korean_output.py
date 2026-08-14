"""User-facing Korean answer guard for English source-copy leakage.

Most RAG prompts already require Korean.  A fast deterministic fallback can,
however, return a source sentence without invoking the answer model.  This
module repairs only that exceptional case and preserves citations/structure.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any


_CITATION_RE = re.compile(r"\[(\d+)\]")
_HEADING_RE = re.compile(r"^#{1,6}\s*(\d)\)")


def english_prose_leak_lines(answer: str) -> list[str]:
    """Return explanatory lines that are English prose rather than proper names."""
    leaks: list[str] = []
    section = 0
    for raw in str(answer or "").splitlines():
        line = raw.strip()
        heading = _HEADING_RE.match(line)
        if heading:
            section = int(heading.group(1))
            continue
        if not line or section == 4:
            continue
        prose = _CITATION_RE.sub("", line).strip("-*• ")
        latin_words = re.findall(r"[A-Za-z]{3,}", prose)
        hangul = re.findall(r"[가-힣]", prose)
        if len(latin_words) >= 10 and len(hangul) < 5:
            leaks.append(line)
    return leaks


def _safe_korean_fallback(answer: str) -> str:
    """Replace leaked prose with an explicit review note if Ollama is unavailable."""
    leaking = set(english_prose_leak_lines(answer))
    if not leaking:
        return answer
    out: list[str] = []
    for raw in str(answer or "").splitlines():
        if raw.strip() not in leaking:
            out.append(raw)
            continue
        citations = "".join(f"[{n}]" for n in _CITATION_RE.findall(raw))
        suffix = f" {citations}" if citations else ""
        out.append(
            "- 영문 원문 근거는 확인했지만 한국어 변환을 완료하지 못했습니다. "
            f"해당 인용 근거를 원문에서 확인해 주세요.{suffix}"
        )
    return "\n".join(out)


def ensure_korean_answer(
    question: str,
    answer: str,
    *,
    model: str,
) -> tuple[str, dict[str, Any]]:
    """Repair an English prose leak once; return unchanged Korean answers quickly."""
    leaks = english_prose_leak_lines(answer)
    if not leaks:
        return answer, {"triggered": False}

    base = (
        os.environ.get("MARITIME_OLLAMA_BASE")
        or os.environ.get("OLLAMA_HOST")
        or "http://127.0.0.1:11434"
    ).rstrip("/")
    started = time.perf_counter()

    def send(payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{base}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        translations: dict[str, str] = {}
        for leak in leaks:
            body: dict[str, Any] = {
                "model": model,
                "stream": False,
                "think": False,
                "options": {"temperature": 0.0, "num_predict": 600},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "너는 해사 규정 영문 한 문장을 한국어로 번역한다. 입력된 bullet의 "
                            "사실, 문서명, 약어, 숫자, 인용 [n]을 그대로 보존한다. 새로운 사실, "
                            "해석, 제목, 다른 bullet을 추가하지 않는다. class notation은 선급 부호로 "
                            "번역하고 SMART 계열 notation 표기는 보존한다. 번역한 bullet 한 줄만 출력한다."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"질문 맥락: {question}\n\n번역할 bullet:\n{leak}",
                    },
                ],
            }
            try:
                payload = send(body)
            except urllib.error.HTTPError as exc:
                if exc.code not in {400, 422}:
                    raise
                compatible = dict(body)
                compatible.pop("think", None)
                payload = send(compatible)
            raw_translation = str(
                (payload.get("message") or {}).get("content") or ""
            ).strip()
            bullet_lines = [
                line.strip()
                for line in raw_translation.splitlines()
                if line.strip().startswith(("-", "*", "•"))
            ]
            translated = bullet_lines[0] if bullet_lines else raw_translation
            if not translated.startswith(("-", "*", "•")):
                translated = "- " + translated.lstrip()
            valid = bool(translated)
            valid = valid and len(re.findall(r"[가-힣]", translated)) >= 5
            valid = valid and not english_prose_leak_lines(translated)
            valid = valid and set(_CITATION_RE.findall(translated)) == set(
                _CITATION_RE.findall(leak)
            )
            if not valid:
                raise ValueError("korean_repair_contract_failed")
            translations[leak] = translated

        repaired_lines = [
            translations.get(raw.strip(), raw) for raw in str(answer or "").splitlines()
        ]
        repaired = "\n".join(repaired_lines)
        return repaired, {
            "triggered": True,
            "success": True,
            "model": model,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "leak_lines": len(leaks),
        }
    except Exception as exc:
        return _safe_korean_fallback(answer), {
            "triggered": True,
            "success": False,
            "model": model,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "leak_lines": len(leaks),
            "error": f"{type(exc).__name__}: {exc}",
        }
