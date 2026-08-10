"""Run one loaded TATR model across the KR Part 7 expanded pilot crops."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TATR = importlib.import_module("52_tatr_structure_pilot")


def load_runtime(model_id: str):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, AutoModelForObjectDetection, TableTransformerConfig

    model_path = snapshot_download(model_id, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
    # Newer transformers/huggingface_hub validates the DETR config strictly.
    # Build the config in memory after normalizing the checkpoint's legacy null.
    config_data = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    config_data["dilation"] = False
    config = TableTransformerConfig(**config_data)
    model = AutoModelForObjectDetection.from_pretrained(
        model_path, local_files_only=True, config=config
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return torch, processor, model, device


def infer(crop: Path, runtime, threshold: float, padding: int, output_dir: Path) -> dict:
    from PIL import Image, ImageOps

    torch, processor, model, device = runtime
    original = Image.open(crop).convert("RGB"); image = ImageOps.expand(original, border=padding, fill="white")
    inputs = processor(images=image, return_tensors="pt", size={"shortest_edge": 800, "longest_edge": 800}).to(device)
    with torch.inference_mode(): outputs = model(**inputs)
    sizes = torch.tensor([[image.height, image.width]], device=device)
    processed = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=sizes)[0]
    detections = []
    for score, label_id, box_tensor in zip(processed["scores"], processed["labels"], processed["boxes"]):
        box = [round(float(value), 2) for value in box_tensor.detach().cpu().tolist()]
        crop_box = [round(max(0.0, min(float(original.width), box[0] - padding)), 2),
                    round(max(0.0, min(float(original.height), box[1] - padding)), 2),
                    round(max(0.0, min(float(original.width), box[2] - padding)), 2),
                    round(max(0.0, min(float(original.height), box[3] - padding)), 2)]
        detections.append({"label_id": int(label_id), "label": model.config.id2label[int(label_id)],
                           "score": round(float(score), 6), "bbox_padded": box, "bbox_crop": crop_box,
                           "bbox_normalized": [round(crop_box[0]/original.width,5), round(crop_box[1]/original.height,5),
                                               round(crop_box[2]/original.width,5), round(crop_box[3]/original.height,5)]})
    detections = TATR.suppress_duplicates(detections); summary = TATR.summarize(detections)
    result = {"model": TATR.MODEL_ID, "crop": str(crop.resolve()), "crop_size": [original.width, original.height],
              "padding": padding, "threshold": threshold, "device": str(device), "summary": summary,
              "detections": sorted(detections, key=lambda item: (item["label_id"], TATR._sort_key(item)))}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/"structure.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    TATR.annotate(image, detections, output_dir/"structure_overlay.png")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT/"data/manifests/kr7_expanded_table_pilot.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5); parser.add_argument("--padding", type=int, default=20)
    args = parser.parse_args(); manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    runtime = load_runtime(TATR.MODEL_ID); rows = []
    for index, table in enumerate(manifest["tables"], 1):
        crop = Path(table["crop_path"]); result_path = crop.parent/"tatr_v1_1_all"/"structure.json"
        if args.reuse_existing and result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result = infer(crop, runtime, args.threshold, args.padding, crop.parent/"tatr_v1_1_all")
        summary = result["summary"]; rows.append({"table_id":table["table_id"], **{k:summary[k] for k in
            ("row_count","column_count","column_header_count","spanning_cell_count")}})
        print(f"[{index:02d}/{len(manifest['tables'])}] {table['table_id']} rows={summary['row_count']} cols={summary['column_count']}", flush=True)
    output = args.output or args.manifest.with_name(f"{args.manifest.stem}_tatr_results.json")
    output.write_text(json.dumps({"model":TATR.MODEL_ID,"tables":rows},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"saved: {output.resolve()}")


if __name__ == "__main__": main()
