#!/usr/bin/env python3
"""Render the trained G_R+LoD model along a supplied camera trajectory.

Read the standard outputs of train_widefe_gszoom.py directly: the final native
G_R checkpoint, l1_final/l2_final.lod.pt, and its adjacent .appearance.pt file.
No private CPU-exported appearance package is needed. The native checkpoint is
also used to verify that the LoD checkpoint has the same frozen L0 parameters.

Example (run through wait_for_idle_gpu.py on a shared machine):
  python scripts/render_trained_lod_orbit.py \
    --start-checkpoint outputs/run/curriculum/e04_45/gs/chkpnt80000.pth \
    --lod-checkpoint outputs/run/l2_train/l2_final.lod.pt \
    --source-path /path/to/JAX_068 \
    --camera-path camera_paths/JAX/068/r328_e17_fov20.json \
    --output outputs/evaluation/JAX_068/current/r328_e17_fov20.mp4

Use --max-frames 3 --save-frames for a real rendering smoke check. This tool
honors the single GPU assigned in CUDA_VISIBLE_DEVICES; it does not pick or
reserve a device. The resulting unlabelled native video can be evaluated with
eval.py, using --frame_rate 30 --resolution 1024 for the released JAX protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-checkpoint", type=Path, required=True)
    parser.add_argument("--lod-checkpoint", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--camera-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--appearance-index", type=int, default=6)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--save-frames", action="store_true", help="Also retain exact native PNG frames beside the video")
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible in ("", "-1") or len(visible.split(",")) != 1:
        parser.error("Assign exactly one GPU through CUDA_VISIBLE_DEVICES (use wait_for_idle_gpu.py)")
    sidecar = Path(str(args.lod_checkpoint) + ".appearance.pt")
    paths = {"native_checkpoint": args.start_checkpoint, "lod_checkpoint": args.lod_checkpoint,
             "appearance_sidecar": sidecar, "camera_path": args.camera_path}
    for label, path in paths.items():
        if not path.is_file():
            parser.error(f"Missing {label}: {path}")
    if not args.source_path.is_dir():
        parser.error(f"Missing dataset: {args.source_path}")
    if args.output.suffix.lower() != ".mp4":
        parser.error("--output must be an .mp4 path")
    payload = json.loads(args.camera_path.read_text())
    width, height, fps = int(payload["render_width"]), int(payload["render_height"]), float(payload["fps"])
    if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
        parser.error("The trajectory must declare a positive raster and fps")
    identities = {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in paths.items()}

    import mediapy as media
    import numpy as np
    import torch
    from PIL import Image
    from lod.importer import import_skyfall_l0, load_lod_onto_bundle, assert_l0_tensors_match
    from lod.render import render_lod_appearance
    from refinement.scene_zoom import SceneZoomConfig, build_arg_namespace, load_scene_context
    from scripts.local_scene_comparison import minicams_from_path_json

    if not torch.cuda.is_available():
        raise RuntimeError("The assigned CUDA device is unavailable")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda:0")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_video = output.with_name(output.stem + ".tmp.mp4")
    frames_dir = output.parent / (output.stem + "_frames")
    if args.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=".lod-render-", dir=output.parent) as scratch:
        config = SceneZoomConfig(start_checkpoint=str(args.start_checkpoint.resolve()),
                                 source_path=str(args.source_path.resolve()), output_dir=scratch)
        namespace = build_arg_namespace(config)
        context = load_scene_context(config, namespace)
        bundle = import_skyfall_l0(context["gaussians"], context["train_cameras"], freeze=True)
        load_lod_onto_bundle(bundle, str(args.lod_checkpoint.resolve()), device=str(device))
        invariant = assert_l0_tensors_match(context["gaussians"], bundle)
        table = bundle.appearance.image_embeddings
        if not 0 <= args.appearance_index < int(table.shape[0]):
            raise ValueError(f"appearance index {args.appearance_index} is outside {int(table.shape[0])} rows")
        appearance = table[args.appearance_index].detach()
        cameras = minicams_from_path_json(payload, device)
        if args.max_frames is not None:
            cameras = cameras[:args.max_frames]
        if not cameras:
            raise ValueError("The camera trajectory is empty")
        for camera in cameras:
            camera.focal_x = camera.image_width / (2 * math.tan(float(camera.FoVx) / 2))
            camera.focal_y = camera.image_height / (2 * math.tan(float(camera.FoVy) / 2))
        image_hashes = []
        with torch.inference_mode(), media.VideoWriter(
            str(temporary_video), shape=(height, width), fps=fps, crf=18,
            ffmpeg_args=["-threads", str(args.cpu_threads)],
        ) as writer:
            for index, camera in enumerate(cameras):
                image = render_lod_appearance(bundle, camera, background=context["background"],
                    kernel_size=0.1, appearance_embedding=appearance,
                    lod=True, compact=True, require_depth=False)["render"]
                if tuple(image.shape) != (3, height, width) or not bool(torch.isfinite(image).all()):
                    raise ValueError(f"Invalid rendered raster at frame {index}")
                rgb = image.detach().float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                pixels = np.rint(rgb * 255).astype(np.uint8)
                image_hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
                writer.add_image(pixels)
                if args.save_frames:
                    Image.fromarray(pixels).save(frames_dir / f"frame_{index:05d}.png")
        temporary_video.replace(output)
        receipt = {"status": "complete", "inputs": identities, "output": str(output),
                   "output_sha256": sha256(output), "width": width, "height": height,
                   "fps": fps, "frames": len(cameras), "appearance_index": args.appearance_index,
                   "kernel_size": 0.1, "l0_invariant": invariant,
                   "level_points": [int(layer.xyz.shape[0]) for layer in bundle.lod.layers],
                   "native_rgb_sha256": image_hashes, "cuda_visible_devices": visible,
                   "seconds_including_loading_and_encoding": time.monotonic() - started}
    receipt_path = output.with_suffix(".receipt.json")
    temporary_receipt = receipt_path.with_suffix(".json.tmp")
    temporary_receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    temporary_receipt.replace(receipt_path)
    print(json.dumps(receipt, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
