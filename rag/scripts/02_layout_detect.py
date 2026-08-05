from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import torch
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm


def parse_page_number(page_name: str, fallback: int) -> int:
    m = re.search(r"(\d+)", page_name)
    if m:
        return int(m.group(1))
    return fallback


def make_element_id(doc_id: str, page_number: int, element_index: int) -> str:
    return f"{doc_id}_p{page_number:04d}_e{element_index:03d}"


def detect_page(
    model: YOLO,
    image_path: Path,
    doc_id: str,
    page_number: int,
    conf: float,
    device: str | int = 0,
) -> tuple[dict, object]:
    result = model.predict(
        source=str(image_path),
        conf=conf,
        device=device,
        verbose=False,
    )[0]
    names = result.names if hasattr(result, "names") else model.names

    elements: list[dict] = []
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        for idx, (box, cls_id, score) in enumerate(zip(boxes, classes, scores)):
            class_id = int(cls_id)
            class_name = str(names.get(class_id, class_id)).lower()
            elements.append(
                {
                    "element_id": make_element_id(doc_id, page_number, idx),
                    "type": class_name,
                    "confidence": round(float(score), 6),
                    "bbox": [round(float(v), 2) for v in box],
                }
            )

    page_json = {
        "doc_id": doc_id,
        "page_number": page_number,
        "image_path": image_path.resolve().as_posix(),
        "page_name": image_path.name,
        "width": int(result.orig_shape[1]),
        "height": int(result.orig_shape[0]),
        "elements": elements,
    }
    return page_json, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO layout detection and save per-page JSON.")
    parser.add_argument("--image-dir", type=Path, required=True, help="Input image directory")
    parser.add_argument("--model", type=Path, required=True, help="YOLO weight path")
    parser.add_argument("--doc-id", type=str, required=True, help="Document ID")
    parser.add_argument("--out-dir", type=Path, required=True, help="JSON output root directory")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save-vis", action="store_true", help="Save visualization images")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    output_doc_dir = args.out_dir / args.doc_id
    output_doc_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = Path("outputs/visualized") / args.doc_id
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    device: str | int = 0 if torch.cuda.is_available() else "cpu"
    print(f"Layout device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == 0 else ""))
    image_paths = sorted(list(args.image_dir.glob("*.png")) + list(args.image_dir.glob("*.jpg")))
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {args.image_dir}")

    for idx, image_path in enumerate(tqdm(image_paths, desc="Layout Detect"), start=1):
        page_number = parse_page_number(image_path.stem, fallback=idx)
        page_data, result = detect_page(
            model=model,
            image_path=image_path,
            doc_id=args.doc_id,
            page_number=page_number,
            conf=args.conf,
            device=device,
        )

        json_path = output_doc_dir / f"page_{page_number:04d}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(page_data, f, ensure_ascii=False, indent=2)

        if args.save_vis:
            plotted = result.plot()
            vis_img = Image.fromarray(plotted[:, :, ::-1])
            vis_img.save(vis_dir / f"page_{page_number:04d}.png")

    print(f"Layout JSON saved under: {output_doc_dir}")
    if args.save_vis:
        print(f"Visualization images saved under: {vis_dir}")


if __name__ == "__main__":
    main()
