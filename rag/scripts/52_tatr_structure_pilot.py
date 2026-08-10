"""Run TATR-v1.1-All on table crops and save auditable structure detections.

TATR predicts geometric objects only. PDF text and formula recovery remain
separate authoritative inputs and can later be assigned to detected cells by
bbox containment.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "microsoft/table-transformer-structure-recognition-v1.1-all"

COLORS = {
    "table": "#7f8c8d",
    "table column": "#2471a3",
    "table row": "#c0392b",
    "table column header": "#239b56",
    "table projected row header": "#8e44ad",
    "table spanning cell": "#d68910",
}


def box_iou(left: list[float], right: list[float]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-9)


def suppress_duplicates(detections: list[dict[str, Any]], iou_threshold: float = 0.85) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for detection in sorted(detections, key=lambda item: float(item["score"]), reverse=True):
        if any(
            detection["label"] == other["label"]
            and box_iou(detection["bbox_padded"], other["bbox_padded"]) >= iou_threshold
            for other in kept
        ):
            continue
        kept.append(detection)
    return kept


def _sort_key(detection: dict[str, Any]) -> tuple[float, float]:
    box = detection["bbox_crop"]
    if detection["label"] == "table column":
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
    return ((box[1] + box[3]) / 2.0, (box[0] + box[2]) / 2.0)


def summarize(detections: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(item["label"] for item in detections)
    ordered: dict[str, list[dict[str, Any]]] = {}
    for label in ("table row", "table column", "table column header", "table projected row header", "table spanning cell"):
        items = sorted((item for item in detections if item["label"] == label), key=_sort_key)
        ordered[label] = [
            {"score": item["score"], "bbox_crop": item["bbox_crop"], "bbox_normalized": item["bbox_normalized"]}
            for item in items
        ]
    return {
        "row_count": counts["table row"],
        "column_count": counts["table column"],
        "column_header_count": counts["table column header"],
        "projected_row_header_count": counts["table projected row header"],
        "spanning_cell_count": counts["table spanning cell"],
        "objects": ordered,
    }


def annotate(image, detections: list[dict[str, Any]], output: Path) -> None:
    from PIL import ImageDraw, ImageFont

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for item in sorted(detections, key=lambda value: value["score"]):
        box = item["bbox_padded"]
        color = COLORS.get(item["label"], "#000000")
        draw.rectangle(box, outline=color, width=3)
        label = f"{item['label']} {item['score']:.2f}"
        text_box = draw.textbbox((box[0], box[1]), label, font=font)
        draw.rectangle(text_box, fill="white")
        draw.text((box[0], box[1]), label, fill=color, font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def run(crop: Path, model_id: str, threshold: float, padding: int, output_dir: Path) -> dict[str, Any]:
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image, ImageOps
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    model_path = snapshot_download(model_id, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
    model = AutoModelForObjectDetection.from_pretrained(model_path, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    original = Image.open(crop).convert("RGB")
    image = ImageOps.expand(original, border=padding, fill="white")
    # This checkpoint stores the legacy DETR setting as only
    # {"longest_edge": 800}; current processors require both keys.
    inputs = processor(
        images=image,
        return_tensors="pt",
        size={"shortest_edge": 800, "longest_edge": 800},
    ).to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    target_sizes = torch.tensor([[image.height, image.width]], device=device)
    processed = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]

    detections: list[dict[str, Any]] = []
    for score, label_id, box_tensor in zip(processed["scores"], processed["labels"], processed["boxes"]):
        box = [round(float(value), 2) for value in box_tensor.detach().cpu().tolist()]
        crop_box = [
            round(max(0.0, min(float(original.width), box[0] - padding)), 2),
            round(max(0.0, min(float(original.height), box[1] - padding)), 2),
            round(max(0.0, min(float(original.width), box[2] - padding)), 2),
            round(max(0.0, min(float(original.height), box[3] - padding)), 2),
        ]
        normalized = [
            round(crop_box[0] / original.width, 5), round(crop_box[1] / original.height, 5),
            round(crop_box[2] / original.width, 5), round(crop_box[3] / original.height, 5),
        ]
        detections.append({
            "label_id": int(label_id), "label": model.config.id2label[int(label_id)],
            "score": round(float(score), 6), "bbox_padded": box,
            "bbox_crop": crop_box, "bbox_normalized": normalized,
        })
    detections = suppress_duplicates(detections)
    summary = summarize(detections)
    result = {
        "model": model_id,
        "crop": str(crop.resolve()),
        "crop_size": [original.width, original.height],
        "padding": padding,
        "threshold": threshold,
        "device": str(device),
        "summary": summary,
        "detections": sorted(detections, key=lambda item: (item["label_id"], _sort_key(item))),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "structure.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    annotate(image, detections, output_dir / "structure_overlay.png")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--padding", type=int, default=20)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    crop = args.crop.resolve()
    output_dir = args.output_dir or crop.parent / "tatr_v1_1_all"
    result = run(crop, args.model, args.threshold, args.padding, output_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"saved: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
