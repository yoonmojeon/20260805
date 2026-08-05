from __future__ import annotations

import argparse
import sys
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import fitz
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from clause_parse import (
    article_number_from_text,
    clause_number_from_text,
    should_split_merge_blocks,
)
from layout_paths import normalize_page_json_paths


FIGURE_BLOCKER_TYPES = frozenset({"figure", "picture"})
TABLE_BLOCKER_TYPES = frozenset({"table"})
BLOCKER_TYPES = TABLE_BLOCKER_TYPES | FIGURE_BLOCKER_TYPES

PRESERVE_TYPES = frozenset({"title", "caption", "table", "figure", "picture"})

VERTICAL_GAP_RATIO = 0.015
X_START_DIFF_RATIO = 0.05
X_OVERLAP_RATIO_MIN = 0.6
PAGE_TEXT_ELEMENTS_MAX = 20
BBOX_IOU_DEDUPE_THRESHOLD = 0.92

@dataclass
class MergeBlock:
    element_ids: list[str]
    source_types: list[str]
    confidence: float
    bbox: list[float]

    @classmethod
    def from_element(cls, element: dict) -> MergeBlock:
        return cls(
            element_ids=[str(element["element_id"])],
            source_types=[str(element.get("type", "text")).lower()],
            confidence=float(element.get("confidence", 0.0)),
            bbox=[float(v) for v in element["bbox"]],
        )

    def to_element(self, element_id: str) -> dict:
        out: dict = {
            "element_id": element_id,
            "type": "text",
            "confidence": round(self.confidence, 6),
            "bbox": [round(v, 2) for v in self.bbox],
        }
        if len(self.element_ids) > 1:
            out["merged_from"] = list(self.element_ids)
        return out


class PdfTextReader:
    def __init__(self, pdf_path: Path | None) -> None:
        self.pdf_path = pdf_path
        self._doc: fitz.Document | None = None
        if pdf_path is not None and pdf_path.exists():
            self._doc = fitz.open(pdf_path)

    @property
    def available(self) -> bool:
        return self._doc is not None

    def close(self) -> None:
        if self._doc is not None:
            self._doc.close()
            self._doc = None

    def extract_bbox_text(
        self,
        page_number: int,
        bbox: list[float],
        layout_page_width: int,
        layout_page_height: int,
    ) -> str:
        if self._doc is None:
            return ""
        page_index = page_number - 1
        if page_index < 0 or page_index >= len(self._doc):
            return ""

        page = self._doc[page_index]
        sx = page.rect.width / max(layout_page_width, 1)
        sy = page.rect.height / max(layout_page_height, 1)
        x1, y1, x2, y2 = bbox
        rect = fitz.Rect(x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        return page.get_text("text", clip=rect).strip()


def parse_merge_types(raw: str) -> frozenset[str]:
    types = {part.strip().lower() for part in raw.split(",") if part.strip()}
    if "list" in types:
        types.add("list-item")
    return frozenset(types)


def sort_key_bbox(bbox: list[float]) -> tuple[float, float]:
    return (bbox[1], bbox[0])


def bbox_iou(a: list[float], b: list[float]) -> float:
    x_left = max(a[0], b[0])
    y_top = max(a[1], b[1])
    x_right = min(a[2], b[2])
    y_bottom = min(a[3], b[3])
    inter_w = max(0.0, x_right - x_left)
    inter_h = max(0.0, y_bottom - y_top)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
    return inter / (area_a + area_b - inter)


def dedupe_mergeable_elements(elements: list[dict]) -> list[dict]:
    sorted_elements = sorted(elements, key=lambda el: -float(el.get("confidence", 0.0)))
    kept: list[dict] = []
    for element in sorted_elements:
        bbox = [float(v) for v in element["bbox"]]
        if any(bbox_iou(bbox, [float(v) for v in kept_el["bbox"]]) >= BBOX_IOU_DEDUPE_THRESHOLD for kept_el in kept):
            continue
        kept.append(element)
    return kept


def first_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def resolve_pdf_path(
    doc_id: str,
    manifest_path: Path,
    pdf_path_arg: Path | None,
    project_root: Path,
) -> Path | None:
    if pdf_path_arg is not None:
        if pdf_path_arg.exists():
            return pdf_path_arg.resolve()
        raise FileNotFoundError(f"PDF not found: {pdf_path_arg}")

    if not manifest_path.exists():
        return None

    df = pd.read_csv(manifest_path)
    matched = df[df["doc_id"] == doc_id]
    if matched.empty:
        return None

    row = matched.iloc[0]
    candidates = [Path(str(row["file_path"]))]
    file_name = str(row.get("file_name", ""))
    if file_name:
        candidates.append(project_root / "data" / "raw_pdfs" / file_name)
        for found in (project_root / "data" / "raw_pdfs").rglob(file_name):
            candidates.append(found)

    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.as_posix()
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()
    return None


def bbox_overlaps(a: list[float], b: list[float]) -> bool:
    x_left = max(a[0], b[0])
    y_top = max(a[1], b[1])
    x_right = min(a[2], b[2])
    y_bottom = min(a[3], b[3])
    return x_right > x_left and y_bottom > y_top


def vertical_gap(upper: list[float], lower: list[float]) -> float:
    if lower[1] >= upper[3]:
        return lower[1] - upper[3]
    return 0.0


def x_overlap_ratio(a: list[float], b: list[float]) -> float:
    overlap_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    if overlap_w <= 0:
        return 0.0
    w_a = max(a[2] - a[0], 1e-6)
    w_b = max(b[2] - b[0], 1e-6)
    return overlap_w / min(w_a, w_b)


def overlaps_blockers(bbox: list[float], blocker_bboxes: list[list[float]]) -> bool:
    return any(bbox_overlaps(bbox, blocker) for blocker in blocker_bboxes)


def is_article_body_continuation(current_text: str, candidate_text: str) -> bool:
    """Allow merge when the next block continues an article (indented body, new article line)."""
    article = article_number_from_text(current_text)
    if not article:
        return False
    candidate_article = article_number_from_text(candidate_text)
    if candidate_article and candidate_article != article:
        return False
    return True


def can_merge(
    current: MergeBlock,
    candidate: MergeBlock,
    page_width: int,
    page_height: int,
    blocker_bboxes: list[list[float]],
    current_text: str = "",
    candidate_text: str = "",
) -> bool:
    gap_limit = page_height * VERTICAL_GAP_RATIO
    gap = vertical_gap(current.bbox, candidate.bbox)
    if gap > gap_limit:
        if is_article_body_continuation(current_text, candidate_text):
            gap_limit = page_height * 0.02
        if gap > gap_limit:
            return False

    x_start_diff = abs(current.bbox[0] - candidate.bbox[0])
    x_aligned = x_start_diff <= page_width * X_START_DIFF_RATIO
    x_overlap_ok = x_overlap_ratio(current.bbox, candidate.bbox) >= X_OVERLAP_RATIO_MIN
    if not (x_aligned or x_overlap_ok):
        if not is_article_body_continuation(current_text, candidate_text):
            return False

    if overlaps_blockers(current.bbox, blocker_bboxes):
        return False
    if overlaps_blockers(candidate.bbox, blocker_bboxes):
        return False
    return True


def merge_two(a: MergeBlock, b: MergeBlock) -> MergeBlock:
    return MergeBlock(
        element_ids=a.element_ids + b.element_ids,
        source_types=a.source_types + b.source_types,
        confidence=max(a.confidence, b.confidence),
        bbox=[
            min(a.bbox[0], b.bbox[0]),
            min(a.bbox[1], b.bbox[1]),
            max(a.bbox[2], b.bbox[2]),
            max(a.bbox[3], b.bbox[3]),
        ],
    )


def merge_text_blocks_on_page(
    elements: list[dict],
    merge_types: frozenset[str],
    page_width: int,
    page_height: int,
    doc_id: str,
    page_number: int,
    pdf_reader: PdfTextReader,
) -> tuple[list[dict], int, int]:
    blocker_bboxes = [
        [float(v) for v in el["bbox"]]
        for el in elements
        if str(el.get("type", "")).lower() in BLOCKER_TYPES
    ]

    preserved: list[dict] = []
    mergeable: list[dict] = []
    for element in elements:
        element_type = str(element.get("type", "unknown")).lower()
        if element_type in PRESERVE_TYPES:
            preserved.append(dict(element))
            continue
        if element_type in merge_types:
            mergeable.append(element)
        elif element_type == "section-header":
            # "902. 탈급", "1201. 정부규정" 등 절 제목은 다음 본문과 병합
            mergeable.append(element)
        else:
            preserved.append(dict(element))

    mergeable = dedupe_mergeable_elements(mergeable)
    mergeable.sort(key=lambda el: sort_key_bbox([float(v) for v in el["bbox"]]))

    text_cache: dict[str, str] = {}

    def bbox_text_for(element: dict) -> str:
        element_id = str(element["element_id"])
        if element_id not in text_cache:
            text_cache[element_id] = pdf_reader.extract_bbox_text(
                page_number,
                [float(v) for v in element["bbox"]],
                page_width,
                page_height,
            )
        return text_cache[element_id]

    merged_blocks: list[MergeBlock] = []
    block_texts: list[str] = []
    merge_ops = 0
    clause_boundary_splits = 0
    current: MergeBlock | None = None
    current_text = ""
    current_clause_no: str | None = None

    for element in mergeable:
        candidate = MergeBlock.from_element(element)
        candidate_text = bbox_text_for(element)
        candidate_clause_no = clause_number_from_text(candidate_text)

        if current is None:
            current = candidate
            current_text = candidate_text
            current_clause_no = candidate_clause_no
            continue

        if should_split_merge_blocks(
            current_text,
            current_clause_no,
            candidate_text,
            candidate_clause_no,
        ):
            merged_blocks.append(current)
            block_texts.append(current_text)
            clause_boundary_splits += 1
            current = candidate
            current_text = candidate_text
            current_clause_no = candidate_clause_no
            continue

        if can_merge(
            current,
            candidate,
            page_width,
            page_height,
            blocker_bboxes,
            current_text=current_text,
            candidate_text=candidate_text,
        ):
            current = merge_two(current, candidate)
            if candidate_text:
                current_text = f"{current_text}\n{candidate_text}".strip()
            if current_clause_no is None and candidate_clause_no:
                current_clause_no = candidate_clause_no
            merge_ops += 1
        else:
            merged_blocks.append(current)
            block_texts.append(current_text)
            current = candidate
            current_text = candidate_text
            current_clause_no = candidate_clause_no

    if current is not None:
        merged_blocks.append(current)
        block_texts.append(current_text)

    merged_elements: list[dict] = []
    for idx, block in enumerate(merged_blocks):
        element_id = f"{doc_id}_p{page_number:04d}_m{idx:03d}"
        out_el = block.to_element(element_id)
        if idx < len(block_texts):
            block_text = block_texts[idx]
            clause_no = clause_number_from_text(block_text)
            if clause_no:
                out_el["clause_number"] = clause_no
            article_no = article_number_from_text(block_text)
            if article_no:
                out_el["article_number"] = article_no
        merged_elements.append(out_el)

    output = preserved + merged_elements
    output.sort(key=lambda el: sort_key_bbox([float(v) for v in el["bbox"]]))
    return output, merge_ops, clause_boundary_splits


def count_types(elements: list[dict]) -> Counter[str]:
    return Counter(str(el.get("type", "unknown")).lower() for el in elements)


def count_pages_with_text_gt(elements_by_page: dict[int, list[dict]], limit: int) -> list[int]:
    pages: list[int] = []
    for page_number, elements in sorted(elements_by_page.items()):
        text_count = count_types(elements).get("text", 0)
        if text_count > limit:
            pages.append(page_number)
    return pages


def load_pages(layout_doc_dir: Path) -> dict[int, dict]:
    pages: dict[int, dict] = {}
    for json_file in sorted(layout_doc_dir.glob("page_*.json")):
        page_data = json.loads(json_file.read_text(encoding="utf-8"))
        pages[int(page_data["page_number"])] = page_data
    return pages


def build_comparison_report(
    doc_id: str,
    before_pages: dict[int, dict],
    after_pages: dict[int, dict],
    total_merge_ops: int,
    total_clause_splits: int,
    pdf_path: Path | None,
) -> str:
    before_all: list[dict] = []
    after_all: list[dict] = []
    before_by_page: dict[int, list[dict]] = {}
    after_by_page: dict[int, list[dict]] = {}

    for page_number, page_data in before_pages.items():
        elements = page_data.get("elements", [])
        before_all.extend(elements)
        before_by_page[page_number] = elements
    for page_number, page_data in after_pages.items():
        elements = page_data.get("elements", [])
        after_all.extend(elements)
        after_by_page[page_number] = elements

    before_types = count_types(before_all)
    after_types = count_types(after_all)
    all_types = sorted(set(before_types) | set(after_types))

    before_text_pages = count_pages_with_text_gt(before_by_page, PAGE_TEXT_ELEMENTS_MAX)
    after_text_pages = count_pages_with_text_gt(after_by_page, PAGE_TEXT_ELEMENTS_MAX)

    lines = [
        f"Layout Merge Comparison: {doc_id}",
        "=" * 60,
        "",
        f"PDF text source: {pdf_path.as_posix() if pdf_path else '(not available)'}",
        f"Pages processed: {len(before_pages)}",
        f"Merge operations (pairwise): {total_merge_ops}",
        f"Clause boundary splits (decimal N.): {total_clause_splits}",
        "",
        f"Total elements: {len(before_all)} -> {len(after_all)} "
        f"({len(after_all) - len(before_all):+d})",
        "",
        "Element counts by type (before -> after):",
    ]
    for element_type in all_types:
        b = before_types.get(element_type, 0)
        a = after_types.get(element_type, 0)
        lines.append(f"  - {element_type}: {b} -> {a} ({a - b:+d})")

    lines.extend(
        [
            "",
            f"Pages with text count > {PAGE_TEXT_ELEMENTS_MAX}:",
            f"  before: {len(before_text_pages)} pages {before_text_pages}",
            f"  after:  {len(after_text_pages)} pages {after_text_pages}",
            "",
            "Per-page text count (before -> after):",
        ]
    )
    for page_number in sorted(before_by_page):
        b = count_types(before_by_page[page_number]).get("text", 0)
        a = count_types(after_by_page[page_number]).get("text", 0)
        if b != a:
            lines.append(f"  - page {page_number:04d}: {b} -> {a} ({a - b:+d})")

    lines.extend(
        [
            "",
            "Merge parameters:",
            f"  - vertical_gap <= page_height * {VERTICAL_GAP_RATIO}",
            f"  - x_start_diff <= page_width * {X_START_DIFF_RATIO} OR x_overlap >= {X_OVERLAP_RATIO_MIN}",
            f"  - no overlap with table/figure/picture bboxes",
            f"  - no merge across different decimal clause numbers (e.g. 13. vs 14.) from PDF text",
            f"  - dedupe overlapping mergeable bboxes (IoU >= {BBOX_IOU_DEDUPE_THRESHOLD})",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge adjacent text/list layout bboxes per page.")
    parser.add_argument("--doc-id", type=str, required=True, help="Document ID")
    parser.add_argument(
        "--layout-dir",
        type=Path,
        default=Path("data/processed/layout_json"),
        help="Input layout JSON root directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/layout_json_merged"),
        help="Output merged layout JSON root directory",
    )
    parser.add_argument(
        "--merge-types",
        type=str,
        default="text,list",
        help="Comma-separated element types to merge (list includes list-item)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/pdf_manifest.csv"),
        help="PDF manifest for resolving source PDF path",
    )
    parser.add_argument(
        "--pdf-path",
        type=Path,
        default=None,
        help="Optional explicit PDF path (overrides manifest lookup)",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("data/processed/logs"),
        help="Directory for merge comparison report",
    )
    args = parser.parse_args()

    project_root = Path.cwd()
    merge_types = parse_merge_types(args.merge_types)
    layout_doc_dir = args.layout_dir / args.doc_id
    out_doc_dir = args.out_dir / args.doc_id
    out_doc_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = resolve_pdf_path(args.doc_id, args.manifest, args.pdf_path, project_root)
    pdf_reader = PdfTextReader(pdf_path)
    if not pdf_reader.available:
        print(f"[WARN] PDF not found for {args.doc_id}; clause-number boundaries disabled.")

    before_pages = load_pages(layout_doc_dir)
    if not before_pages:
        pdf_reader.close()
        raise FileNotFoundError(f"No layout pages found in: {layout_doc_dir}")

    after_pages: dict[int, dict] = {}
    total_merge_ops = 0
    total_clause_splits = 0
    before_counter: Counter[str] = Counter()
    after_counter: Counter[str] = Counter()

    try:
        for page_number, page_data in sorted(before_pages.items()):
            elements = page_data.get("elements", [])
            before_counter.update(count_types(elements))

            merged_elements, merge_ops, clause_splits = merge_text_blocks_on_page(
                elements=elements,
                merge_types=merge_types,
                page_width=int(page_data.get("width", 0)),
                page_height=int(page_data.get("height", 0)),
                doc_id=str(page_data.get("doc_id", args.doc_id)),
                page_number=page_number,
                pdf_reader=pdf_reader,
            )
            total_merge_ops += merge_ops
            total_clause_splits += clause_splits
            after_counter.update(count_types(merged_elements))

            out_page = normalize_page_json_paths(dict(page_data))
            out_page["elements"] = merged_elements
            out_page["merge_meta"] = {
                "source_layout_dir": str(layout_doc_dir.as_posix()),
                "pdf_path": pdf_path.as_posix() if pdf_path else None,
                "merge_types": sorted(merge_types),
                "merge_operations": merge_ops,
                "clause_boundary_splits": clause_splits,
                "element_count_before": len(elements),
                "element_count_after": len(merged_elements),
            }
            after_pages[page_number] = out_page

            out_path = out_doc_dir / f"page_{page_number:04d}.json"
            out_path.write_text(json.dumps(out_page, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        pdf_reader.close()

    comparison = build_comparison_report(
        args.doc_id,
        before_pages,
        after_pages,
        total_merge_ops,
        total_clause_splits,
        pdf_path,
    )
    comparison_path = args.logs_dir / f"{args.doc_id}_merge_comparison.txt"
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(comparison, encoding="utf-8")

    print(f"Merged layout saved: {out_doc_dir}")
    print(f"PDF: {pdf_path if pdf_path else 'N/A'}")
    print(f"Comparison report: {comparison_path}")
    print("\nElement type counts (before -> after):")
    all_types = sorted(set(before_counter) | set(after_counter))
    for element_type in all_types:
        b = before_counter[element_type]
        a = after_counter[element_type]
        print(f"  - {element_type}: {b} -> {a} ({a - b:+d})")
    print(f"\nTotal elements: {sum(before_counter.values())} -> {sum(after_counter.values())}")
    print(f"Merge operations: {total_merge_ops}")
    print(f"Clause boundary splits: {total_clause_splits}")
    p28_before = count_types(before_pages[28].get("elements", [])).get("text", 0) if 28 in before_pages else 0
    p28_after = count_types(after_pages[28].get("elements", [])).get("text", 0) if 28 in after_pages else 0
    print(f"Page 28 text blocks: {p28_before} -> {p28_after}")
    print(comparison, end="")


if __name__ == "__main__":
    main()
