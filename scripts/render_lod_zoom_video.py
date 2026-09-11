#!/usr/bin/env python3
"""ROI-centered continuous zoom video. Native 1x is a different crop and is not mixed in."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile

import torch

from arguments import ModelParams, PipelineParams
from lod.importer import import_skyfall_l0, load_lod_onto_bundle
from lod.inspect import layer_weight_means
from lod.path import DEFAULT_GZ_ROOT
from lod.rasterizer import require_rade_gs
from lod.render import render_lod_appearance
from scene import GaussianModel, Scene
from train_zoom_gen import apply_stage1_cfg_to_args, load_stage1_cfg, pick_base_camera
from utils.general_utils import safe_state
from utils.zoom_camera import NormalizedROI, make_zoom_camera
from utils.zoom_mvp_utils import save_tensor_image, select_appearance_embedding


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    ModelParams(parser)
    PipelineParams(parser)
    parser.add_argument("--start_checkpoint", type=str, required=True)
    parser.add_argument("--lod_checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--gz_root", type=str, default=str(DEFAULT_GZ_ROOT))
    parser.add_argument("--view_index", type=int, default=0)
    parser.add_argument("--view_name", type=str, default="", help="Lock the base camera by image stem. Preferred over view_index.")
    parser.add_argument("--roi_center_x", type=float, default=0.592)
    parser.add_argument("--roi_center_y", type=float, default=0.53)
    parser.add_argument("--roi_width", type=float, default=0.1)
    parser.add_argument("--roi_height", type=float, default=0.1)
    parser.add_argument("--step_scale", type=float, default=2.0)
    parser.add_argument("--start_factor", type=float, default=1.25)
    parser.add_argument("--end_factor", type=float, default=4.0)
    parser.add_argument("--n_frames", type=int, default=36)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    require_rade_gs()
    safe_state(args.quiet)
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    stage1_cfg = load_stage1_cfg(os.path.dirname(os.path.abspath(args.start_checkpoint)))
    apply_stage1_cfg_to_args(args, stage1_cfg)
    extract_parser = argparse.ArgumentParser(add_help=False)
    dataset = ModelParams(extract_parser).extract(args)
    dataset.model_path = os.path.dirname(output)

    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs,
        dataset.appearance_embedding_dim,
    )
    model_params, first_iter = torch.load(args.start_checkpoint, weights_only=False)
    scene = Scene(
        dataset, gaussians, load_iteration=first_iter, shuffle=False,
        ply_path=os.path.dirname(args.start_checkpoint),
    )
    gaussians.load_from_checkpoints(model_params)
    train_cameras = list(scene.getTrainCameras())
    gaussians.compute_3D_filter(cameras=train_cameras)
    bundle = import_skyfall_l0(
        gaussians, train_cameras, gz_root=args.gz_root, step_scale=args.step_scale, freeze=True,
    )
    load_lod_onto_bundle(bundle, os.path.abspath(args.lod_checkpoint), device=str(gaussians.get_xyz.device))
    roi = NormalizedROI(args.roi_center_x, args.roi_center_y, args.roi_width, args.roi_height)
    start_factor = max(float(args.start_factor), float(roi.min_zoom_factor()))
    n_frames = max(2, int(args.n_frames))
    factors = [
        float(start_factor + (args.end_factor - start_factor) * index / (n_frames - 1))
        for index in range(n_frames)
    ]
    base_camera, is_train_view = pick_base_camera(scene, args.view_index, False, image_name=args.view_name)
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device=gaussians.get_xyz.device,
    )
    embedding = select_appearance_embedding(gaussians, base_camera.uid, is_train_view)
    kernel = float(dataset.kernel_size)
    rows = []
    prev_mean = None
    with tempfile.TemporaryDirectory(prefix="lod-zoom-") as tmp:
        for index, factor in enumerate(factors):
            camera = make_zoom_camera(
                base_camera, roi, factor, uid=60000 + index,
                image_name=f"{base_camera.image_name}_zoom{factor:.3f}_video",
            )
            pkg = render_lod_appearance(
                bundle, camera, background=background, kernel_size=kernel,
                appearance_embedding=embedding, lod=True, compact=False,
            )
            rgb = pkg["render"].clamp(0, 1)
            mean_rgb = float(rgb.mean().item())
            row = {
                "frame": index,
                "factor": factor,
                "mean_rgb": mean_rgb,
                "weights": layer_weight_means(pkg),
            }
            if prev_mean is not None:
                row["mean_rgb_jump"] = abs(mean_rgb - prev_mean)
            rows.append(row)
            prev_mean = mean_rgb
            save_tensor_image(rgb, os.path.join(tmp, f"frame_{index:04d}.png"))
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FileNotFoundError("ffmpeg is required to write the zoom video")
        subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error",
                "-framerate", str(int(args.fps)),
                "-i", os.path.join(tmp, "frame_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                output,
            ],
            check=True,
        )
    jumps = [row["mean_rgb_jump"] for row in rows if "mean_rgb_jump" in row]
    _write_json(os.path.splitext(output)[0] + ".json", {
        "output": output,
        "start_factor": start_factor,
        "end_factor": float(args.end_factor),
        "fps": int(args.fps),
        "max_mean_rgb_jump": max(jumps) if jumps else 0.0,
        "rows": rows,
    })
    print(json.dumps({
        "output": output,
        "n_frames": len(rows),
        "max_mean_rgb_jump": max(jumps) if jumps else 0.0,
    }, indent=2))


if __name__ == "__main__":
    main()
