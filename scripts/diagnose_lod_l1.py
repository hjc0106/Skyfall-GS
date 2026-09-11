#!/usr/bin/env python3
"""Cross-view / scale / geometry inspect for a saved L1 bundle. Does not train."""

from __future__ import annotations

import argparse
import json
import os

import torch

from arguments import ModelParams, PipelineParams
from lod.importer import import_skyfall_l0, load_lod_onto_bundle
from lod.inspect import l1_geometry_report, neighbor_zoom_report, scale_sweep
from lod.path import DEFAULT_GZ_ROOT
from lod.rasterizer import require_rade_gs
from scene import GaussianModel, Scene
from train_zoom_gen import apply_stage1_cfg_to_args, load_stage1_cfg, pick_base_camera
from utils.general_utils import safe_state
from utils.zoom_camera import NormalizedROI
from utils.zoom_mvp_utils import select_appearance_embedding


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
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gz_root", type=str, default=str(DEFAULT_GZ_ROOT))
    parser.add_argument("--view_index", type=int, default=0)
    parser.add_argument("--roi_center_x", type=float, default=0.592)
    parser.add_argument("--roi_center_y", type=float, default=0.53)
    parser.add_argument("--roi_width", type=float, default=0.1)
    parser.add_argument("--roi_height", type=float, default=0.1)
    parser.add_argument("--zoom_factor", type=float, default=2.0)
    parser.add_argument("--step_scale", type=float, default=2.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    require_rade_gs()
    safe_state(args.quiet)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    stage1_cfg = load_stage1_cfg(os.path.dirname(os.path.abspath(args.start_checkpoint)))
    apply_stage1_cfg_to_args(args, stage1_cfg)
    extract_parser = argparse.ArgumentParser(add_help=False)
    dataset = ModelParams(extract_parser).extract(args)
    dataset.model_path = output_dir

    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs,
        dataset.appearance_embedding_dim,
    )
    model_params, first_iter = torch.load(args.start_checkpoint, weights_only=False)
    scene = Scene(
        dataset, gaussians, load_iteration=first_iter, shuffle=False,
        ply_path=os.path.dirname(os.path.abspath(args.start_checkpoint)),
    )
    gaussians.load_from_checkpoints(model_params)
    train_cameras = list(scene.getTrainCameras())
    test_cameras = list(scene.getTestCameras())
    gaussians.compute_3D_filter(cameras=train_cameras)
    bundle = import_skyfall_l0(
        gaussians, train_cameras, gz_root=args.gz_root, step_scale=args.step_scale, freeze=True,
    )
    load_lod_onto_bundle(bundle, os.path.abspath(args.lod_checkpoint), device=str(gaussians.get_xyz.device))
    roi = NormalizedROI(args.roi_center_x, args.roi_center_y, args.roi_width, args.roi_height)
    base_camera, is_train_view = pick_base_camera(scene, args.view_index, False)
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device=gaussians.get_xyz.device,
    )
    embedding = select_appearance_embedding(gaussians, base_camera.uid, is_train_view)
    cross_dir = os.path.join(output_dir, "cross_view")
    os.makedirs(cross_dir, exist_ok=True)
    payload = {
        "geometry": l1_geometry_report(bundle),
        "scale_sweep": scale_sweep(
            bundle, base_camera, roi, background=background, kernel=float(dataset.kernel_size),
            embedding=embedding, out_dir=cross_dir,
        ),
        "views": neighbor_zoom_report(
            bundle, base_camera, train_cameras, test_cameras, roi,
            background=background, kernel=float(dataset.kernel_size), gaussians=gaussians,
            zoom_factor=args.zoom_factor, out_dir=cross_dir,
        ),
    }
    _write_json(os.path.join(output_dir, "cross_view.json"), payload)
    print(json.dumps({
        "geometry": payload["geometry"],
        "max_mean_rgb_jump": payload["scale_sweep"]["max_mean_rgb_jump"],
        "neighbors": payload["views"]["neighbors"],
        "test_2x": payload["views"]["test_2x"],
    }, indent=2))


if __name__ == "__main__":
    main()
