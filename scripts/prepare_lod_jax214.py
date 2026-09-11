#!/usr/bin/env python3
"""Lock the JAX_214 building/parking ROI from the actual train view 0 GT. Does not train."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

from lod.jax214 import RUN_DIR, ROI, VIEW_IMAGE, check_dataset, panel_roi, selection_payload
from lod.lineage import write_json


ROOT = Path(__file__).resolve().parents[1]


def _overlay(image: Image.Image, roi, path: Path, *, label: str) -> None:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    left = (roi.center_x - roi.width / 2.0) * w
    right = (roi.center_x + roi.width / 2.0) * w
    top = (roi.center_y - roi.height / 2.0) * h
    bottom = (roi.center_y + roi.height / 2.0) * h
    draw.rectangle([left, top, right, bottom], outline=(220, 40, 30), width=6)
    draw.ellipse(
        [roi.center_x * w - 5, roi.center_y * h - 5, roi.center_x * w + 5, roi.center_y * h + 5],
        fill=(255, 220, 0),
    )
    draw.text((left + 8, max(8, top - 28)), label, fill=(220, 40, 30))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=str, default=str(RUN_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()
    dataset = check_dataset()
    write_json(run_dir / "DATA_CHECK.json", dataset)
    if not dataset["ok"]:
        raise SystemExit(f"JAX_214 dataset check failed: {dataset}")

    selection_path = run_dir / "SELECTION.json"
    if selection_path.is_file() and not args.force:
        print(selection_path)
        return

    payload = selection_payload(
        locked_at=datetime.now(timezone.utc).isoformat(),
        dataset=dataset,
    )
    write_json(selection_path, payload)
    image = Image.open(ROOT / "skyfall-gs_exp/skyfall-gs_data/datasets_JAX/JAX_214/images" / f"{VIEW_IMAGE}.png").convert("RGB")
    roi = panel_roi()
    _overlay(image, roi, run_dir / "selection" / "overlay_view0.png", label=f"{ROI['id']} ({ROI['center_x']:.2f},{ROI['center_y']:.2f})")
    crop = image.crop((
        int((roi.center_x - roi.width / 2.0) * image.width),
        int((roi.center_y - roi.height / 2.0) * image.height),
        int((roi.center_x + roi.width / 2.0) * image.width),
        int((roi.center_y + roi.height / 2.0) * image.height),
    ))
    crop.save(run_dir / "selection" / "roi_gt_crop.png")
    print(selection_path)


if __name__ == "__main__":
    main()
