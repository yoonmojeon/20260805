from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from PIL import Image
from tqdm import tqdm


def clip_bbox(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[int, int, int, int]:
    cx1 = max(0, min(int(round(x1)), w))
    cy1 = max(0, min(int(round(y1)), h))
    cx2 = max(0, min(int(round(x2)), w))
    cy2 = max(0, min(int(round(y2)), h))
    return cx1, cy1, cx2, cy2


def parse_element_suffix(element_id: str, fallback: int) -> str:
    m = re.search(r"_p\d+_(.+)$", element_id)
    if m:
        return m.group(1)
    m = re.search(r"_(m|e)(\d+)$", element_id)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return f"e{fallback:03d}"


def parse_page_number_from_name(page_name: str, fallback: int) -> int:
    m = re.search(r"(\d+)", Path(page_name).stem)
    if m:
        return int(m.group(1))
    return fallback


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    doc_id = args.doc_id
    image_dir = args.image_dir or Path("data/processed/pages") / doc_id

    if args.merged:
        layout_dir = args.layout_dir or Path("data/processed/layout_json_merged") / doc_id
        out_dir = args.out_dir or Path("data/processed/crops_merged")
    else:
        layout_dir = args.layout_dir or Path("data/processed/layout_json") / doc_id
        out_dir = args.out_dir or Path("data/processed/crops")

    return image_dir, layout_dir, out_dir, out_dir / doc_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop detected layout elements from page images.")
    parser.add_argument("--doc-id", type=str, required=True, help="Document ID")
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Use merged layout JSON and write to crops_merged (overrides default dirs)",
    )
    parser.add_argument("--image-dir", type=Path, default=None, help="Page image directory")
    parser.add_argument("--layout-dir", type=Path, default=None, help="Layout JSON directory (doc subfolder)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Crop output root directory")
    parser.add_argument("--min-width", type=int, default=20)
    parser.add_argument("--min-height", type=int, default=20)
    args = parser.parse_args()

    image_dir, layout_dir, out_root, doc_crop_dir = resolve_paths(args)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not layout_dir.exists():
        raise FileNotFoundError(f"Layout directory not found: {layout_dir}")
    if args.min_width <= 0 or args.min_height <= 0:
        raise ValueError("--min-width and --min-height must be > 0")

    doc_crop_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = doc_crop_dir / "crops_manifest.jsonl"

    json_files = sorted(layout_dir.glob("page_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No page JSON files found in: {layout_dir}")

    type_counter: Counter[str] = Counter()
    saved_count = 0
    skipped_small = 0
    skipped_missing_image = 0
    merged_block_count = 0

    with manifest_path.open("w", encoding="utf-8") as manifest_f:
        for json_file in tqdm(json_files, desc="Crop Elements"):
            page_data = json.loads(json_file.read_text(encoding="utf-8"))

            page_name = page_data.get("page_name", f"{json_file.stem}.png")
            page_number = int(
                page_data.get(
                    "page_number",
                    parse_page_number_from_name(page_name, fallback=1),
                )
            )
            source_image_path = image_dir / page_name
            if not source_image_path.exists():
                source_image_path_alt = Path(page_data.get("image_path", ""))
                if source_image_path_alt.exists():
                    source_image_path = source_image_path_alt
                else:
                    skipped_missing_image += 1
                    print(f"[WARN] Missing page image for {json_file.name}: {source_image_path}")
                    continue

            with Image.open(source_image_path) as img:
                w, h = img.size

                for idx, element in enumerate(page_data.get("elements", [])):
                    element_id = str(element.get("element_id", f"{args.doc_id}_p{page_number:04d}_e{idx:03d}"))
                    element_type = str(element.get("type", "unknown")).lower()
                    confidence = float(element.get("confidence", 0.0))
                    merged_from = element.get("merged_from")
                    bbox = element.get("bbox", [0, 0, 0, 0])
                    if len(bbox) != 4:
                        continue

                    x1, y1, x2, y2 = bbox
                    cx1, cy1, cx2, cy2 = clip_bbox(x1, y1, x2, y2, w, h)
                    crop_w = cx2 - cx1
                    crop_h = cy2 - cy1

                    if crop_w < args.min_width or crop_h < args.min_height:
                        skipped_small += 1
                        continue

                    suffix = parse_element_suffix(element_id, fallback=idx)
                    element_dir = doc_crop_dir / element_type
                    element_dir.mkdir(parents=True, exist_ok=True)
                    crop_name = f"page_{page_number:04d}_{suffix}_{element_type}.png"
                    crop_path = element_dir / crop_name

                    crop = img.crop((cx1, cy1, cx2, cy2))
                    crop.save(crop_path)

                    meta: dict = {
                        "doc_id": args.doc_id,
                        "page_number": page_number,
                        "element_id": element_id,
                        "element_type": element_type,
                        "confidence": confidence,
                        "bbox": [cx1, cy1, cx2, cy2],
                        "crop_path": crop_path.resolve().as_posix(),
                        "source_image_path": source_image_path.resolve().as_posix(),
                        "layout_source": "merged" if args.merged else "original",
                    }
                    if merged_from:
                        meta["merged_from"] = list(merged_from)
                        merged_block_count += 1
                    manifest_f.write(json.dumps(meta, ensure_ascii=False) + "\n")

                    type_counter[element_type] += 1
                    saved_count += 1

    print(f"Layout source: {'merged' if args.merged else 'original'}")
    print(f"Layout dir: {layout_dir}")
    print(f"Cropped elements saved under: {doc_crop_dir}")
    print(f"Crops manifest saved: {manifest_path}")
    print(f"Saved crops: {saved_count}")
    print(f"Merged text blocks: {merged_block_count}")
    print(f"Skipped small crops: {skipped_small}")
    print(f"Skipped missing images: {skipped_missing_image}")
    print("Element type counts:")
    for element_type, count in sorted(type_counter.items()):
        print(f"- {element_type}: {count}")


if __name__ == "__main__":
    main()
