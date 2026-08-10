"""Utilities for restoring HancomEQN private-use glyphs from native PDF text."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import fitz

PUA_RE = re.compile(r"[\ue000-\uf8ff]")


@dataclass(frozen=True)
class Glyph:
    char: str
    bbox: tuple[float, float, float, float]
    origin: tuple[float, float]
    size: float
    font: str
    color: int = 0

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)


def is_hancomeqn(font: str) -> bool:
    return "HancomEQN" in font


def font_fingerprint(doc: fitz.Document, font_xref: int) -> str:
    """Fingerprint the embedded font plus its ToUnicode CMap."""
    chunks: list[bytes] = []
    try:
        extracted = doc.extract_font(font_xref)
        if extracted and len(extracted) >= 4 and extracted[3]:
            chunks.append(extracted[3])
    except Exception:
        pass
    try:
        _, descendant = doc.xref_get_key(font_xref, "DescendantFonts")
        chunks.append(descendant.encode("ascii", errors="ignore"))
        kind, value = doc.xref_get_key(font_xref, "ToUnicode")
        if kind == "xref":
            cmap_xref = int(re.search(r"\d+", value).group())
            chunks.append(doc.xref_stream(cmap_xref))
    except Exception:
        pass
    return hashlib.sha256(b"\0".join(chunks)).hexdigest()[:16]


def hancomeqn_fonts(doc: fitz.Document, page: fitz.Page) -> list[dict[str, Any]]:
    result = []
    for item in page.get_fonts(full=True):
        xref, _, _, name = int(item[0]), str(item[1]), str(item[2]), str(item[3])
        if "HancomEQN" not in name:
            continue
        result.append({"xref": xref, "name": name, "fingerprint": font_fingerprint(doc, xref)})
    return result


def load_mapping(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_page_glyphs(page: fitz.Page, clip: fitz.Rect | None = None) -> Iterable[Glyph]:
    raw = page.get_text("rawdict", clip=clip)
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font, size = str(span.get("font", "")), float(span.get("size", 0))
                for char in span.get("chars", []):
                    yield Glyph(
                        char=str(char["c"]), bbox=tuple(float(v) for v in char["bbox"]),
                        origin=tuple(float(v) for v in char.get("origin", (char["bbox"][0], char["bbox"][3]))),
                        size=size, font=font, color=int(span.get("color", 0)),
                    )


def map_token(char: str, mapping: dict[str, Any]) -> tuple[str, bool, str | None]:
    if not PUA_RE.fullmatch(char):
        return char, True, None
    value = mapping.get("glyphs", {}).get(f"U+{ord(char):04X}")
    if value is None:
        return f"[U+{ord(char):04X}]", False, None
    if isinstance(value, str):
        return value, True, None
    return str(value.get("text", "")), True, value.get("role")


def restore_inline(text: str, mapping: dict[str, Any]) -> tuple[str, list[str]]:
    """Conservative character substitution for prose containing isolated equation glyphs."""
    output: list[str] = []
    unknown: list[str] = []
    for char in text:
        token, known, role = map_token(char, mapping)
        if not known:
            unknown.append(f"U+{ord(char):04X}")
        if role not in {"radical_bar", "delimiter_piece"}:
            output.append(token)
    return "".join(output), sorted(set(unknown))


def _cluster_baselines(glyphs: list[Glyph]) -> list[tuple[float, list[Glyph]]]:
    eqn = [g for g in glyphs if is_hancomeqn(g.font) and g.char.strip()]
    if not eqn:
        return []
    ordinary = [g.size for g in eqn if g.size <= median(g.size for g in eqn) * 1.35]
    main_size = median(ordinary or [g.size for g in eqn])
    anchors = [g for g in eqn if g.size >= main_size * 0.86 and g.size <= main_size * 1.35]
    centers: list[list[float]] = []
    for y in sorted(g.origin[1] for g in anchors):
        if not centers or abs(y - sum(centers[-1]) / len(centers[-1])) > main_size * 0.48:
            centers.append([y])
        else:
            centers[-1].append(y)
    baselines = [sum(group) / len(group) for group in centers]
    groups: list[tuple[float, list[Glyph]]] = [(y, []) for y in baselines]
    for glyph in eqn:
        target = min(range(len(groups)), key=lambda i: abs(glyph.origin[1] - groups[i][0]))
        groups[target][1].append(glyph)
    return [(y, sorted(items, key=lambda g: (g.bbox[0], g.origin[1]))) for y, items in groups if items]


def _unicode_script(text: str, kind: str) -> str:
    tables = {
        "sub": str.maketrans("0123456789+-=()min", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₘᵢₙ"),
        "sup": str.maketrans("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"),
    }
    translated = text.translate(tables[kind])
    return translated if len(translated) == len(text) else (f"_({text})" if kind == "sub" else f"^({text})")


def restore_formula(glyphs: list[Glyph], mapping: dict[str, Any]) -> dict[str, Any]:
    """Restore formula lines, using font size/origin to preserve scripts."""
    lines = []
    unknown: set[str] = set()
    total_pua = known_pua = 0
    for baseline, chars in _cluster_baselines(glyphs):
        normal_sizes = [g.size for g in chars if abs(g.origin[1] - baseline) < max(g.size * .25, 1.5)]
        main_size = median(normal_sizes or [g.size for g in chars])
        parts: list[tuple[str, str, str | None]] = []
        for glyph in chars:
            token, known, role = map_token(glyph.char, mapping)
            if PUA_RE.fullmatch(glyph.char):
                total_pua += 1
                known_pua += int(known)
                if not known:
                    unknown.add(f"U+{ord(glyph.char):04X}")
            dy = glyph.origin[1] - baseline
            kind = "normal"
            if glyph.size < main_size * .82:
                kind = "sub" if dy > main_size * .13 else "sup" if dy < -main_size * .13 else "normal"
            parts.append((token, kind, role))

        def serialize_parts(selected: list[tuple[str, str, str | None]], mode: str, allow_sqrt: bool = True) -> str:
            out: list[str] = []
            index = 0
            while index < len(selected):
                token, kind, role = selected[index]
                if role in {"radical_bar", "delimiter_piece"}:
                    index += 1
                    continue
                if role == "sqrt" and allow_sqrt:
                    radicand = serialize_parts(selected[index + 1:], mode, allow_sqrt=False)
                    if mode == "latex": out.append(r"\sqrt{" + radicand + "}")
                    elif mode == "display": out.append("√" + radicand)
                    else: out.append("sqrt(" + radicand + ")")
                    break
                if kind == "normal":
                    out.append(token)
                else:
                    run = [token]
                    while index + 1 < len(selected) and selected[index + 1][1] == kind:
                        index += 1
                        run.append(selected[index][0])
                    value = "".join(run)
                    if mode == "latex": out.append(("_{" if kind == "sub" else "^{") + value + "}")
                    elif mode == "display": out.append(_unicode_script(value, kind))
                    else: out.append(("_(" if kind == "sub" else "^(") + value + ")")
                index += 1
            return "".join(out).strip()

        def serialize(mode: str) -> str:
            return serialize_parts(parts, mode)

        lines.append({"baseline": round(baseline, 3), "display": serialize("display"),
                      "latex": serialize("latex"), "normalized": serialize("normalized")})
    return {
        "display": "\n".join(line["display"] for line in lines),
        "latex": r" \\ ".join(line["latex"] for line in lines),
        "normalized": " ; ".join(line["normalized"] for line in lines),
        "lines": lines,
        "confidence": round(known_pua / total_pua, 4) if total_pua else 1.0,
        "unknown_glyphs": sorted(unknown),
        "needs_review": bool(unknown),
    }


def point_in_bbox(point: tuple[float, float], bbox: list[float]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]
