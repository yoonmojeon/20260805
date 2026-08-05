from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(cmd: list[str]) -> None:
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full preprocessing pipeline.")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/pdf_manifest.csv"))
    parser.add_argument("--manifest-test", type=Path, default=Path("data/manifests/pdf_manifest_test.csv"))
    parser.add_argument("--poppler-path", type=str, default=None)
    args = parser.parse_args()

    py = args.python

    run_step(
        [
            py,
            "scripts/00_build_manifest.py",
            "--manifest",
            str(args.manifest),
            "--manifest-test",
            str(args.manifest_test),
        ]
    )

    cmd_pdf_to_img = [
        py,
        "scripts/01_pdf_to_images.py",
        "--manifest",
        str(args.manifest),
        "--dpi",
        str(args.dpi),
    ]
    if args.poppler_path:
        cmd_pdf_to_img.extend(["--poppler-path", args.poppler_path])
    run_step(cmd_pdf_to_img)

    run_step(
        [
            py,
            "scripts/02_layout_detect.py",
            "--conf",
            str(args.conf),
            "--iou",
            str(args.iou),
        ]
    )

    run_step([py, "scripts/03_crop_elements.py"])
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
