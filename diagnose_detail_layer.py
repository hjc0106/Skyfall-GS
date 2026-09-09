#!/usr/bin/env python3
"""Contribution diagnostics for an existing zoom-MVP detail-layer run. No training."""

from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace

import torch

from arguments import IDUParams, ModelParams, PipelineParams
from scene import Scene, GaussianModel
from train_zoom_mvp import (
    apply_stage1_cfg_to_args,
    load_stage1_cfg,
    pick_base_camera,
)
from utils.zoom_camera import NormalizedROI, make_zoom_camera
from utils.zoom_mvp_utils import (
    crop_chw,
    diagnose_detail_contribution,
    parse_crop_box,
    save_gray_image,
    save_tensor_image,
    write_json,
)


def load_run_cfg(run_dir: str) -> Namespace:
    path = os.path.join(run_dir, "cfg_args")
    with open(path, "r", encoding="utf-8") as f:
        return eval(f.read(), {"IDUParams": IDUParams, "Namespace": Namespace})


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose detail-layer coverage vs composite contribution")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--level_dir", type=str, default=None, help="Defaults to run_dir/zoom_4x")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    run_cfg = load_run_cfg(run_dir)
    level_dir = os.path.abspath(args.level_dir) if args.level_dir else os.path.join(run_dir, "zoom_4x")
    ckpt_path = os.path.join(run_dir, "chkpnt_final.pth")
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(level_dir, "chkpnt_zoom4.pth")
    manifest = json.load(open(os.path.join(run_dir, "manifest.json"), encoding="utf-8"))
    n_detail = int(manifest.get("detail_count") or run_cfg.detail_count)
    crop_box = parse_crop_box(getattr(run_cfg, "crop_box", None))

    start_checkpoint = os.path.abspath(run_cfg.start_checkpoint)
    stage1_dir = os.path.dirname(start_checkpoint)
    apply_stage1_cfg_to_args(args, load_stage1_cfg(stage1_dir))
    args.source_path = run_cfg.source_path
    dataset = lp.extract(args)
    pipe = pp.extract(args)
    dataset.model_path = run_dir

    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs,
        dataset.appearance_embedding_dim,
    )
    model_params, first_iter = torch.load(start_checkpoint, weights_only=False)
    scene = Scene(dataset, gaussians, load_iteration=first_iter, shuffle=False, ply_path=stage1_dir)
    detail_params, _ = torch.load(ckpt_path, weights_only=False)
    gaussians.load_from_checkpoints(detail_params)

    n_coarse = int(gaussians._xyz.shape[0]) - n_detail
    print(f"Loaded {gaussians._xyz.shape[0]} Gaussians (coarse={n_coarse}, detail={n_detail})")

    roi = NormalizedROI(run_cfg.roi_center_x, run_cfg.roi_center_y, run_cfg.roi_width, run_cfg.roi_height)
    base_cam, _ = pick_base_camera(scene, run_cfg.view_index, run_cfg.use_test_view)
    zoom_cam = make_zoom_camera(
        base_cam,
        roi,
        float(manifest["zoom_factors"][0]),
        uid=10000,
        image_name=f"{base_cam.image_name}_zoom4",
    )
    train_cameras = scene.getTrainCameras()
    gaussians.compute_3D_filter(cameras=list(train_cameras) + [zoom_cam])

    diag = diagnose_detail_contribution(
        gaussians, zoom_cam, pipe, dataset.kernel_size, n_coarse
    )
    solo = diag["solo_alpha"]
    weight = diag["composite_weight"]
    blocked = diag["blocked"]
    stats = dict(diag["stats"])
    if crop_box is not None:
        solo_c = crop_chw(solo.unsqueeze(0), crop_box).squeeze(0)
        weight_c = crop_chw(weight.unsqueeze(0), crop_box).squeeze(0)
        blocked_c = crop_chw(blocked.unsqueeze(0), crop_box).squeeze(0)
        covered_c = solo_c > 0.05
        stats.update(
            {
                "crop_solo_alpha_mean": float(solo_c.mean().item()),
                "crop_solo_frac_gt_0.05": float((solo_c > 0.05).float().mean().item()),
                "crop_solo_frac_gt_0.2": float((solo_c > 0.2).float().mean().item()),
                "crop_composite_weight_mean": float(weight_c.mean().item()),
                "crop_composite_frac_gt_0.05": float((weight_c > 0.05).float().mean().item()),
                "crop_composite_frac_gt_0.2": float((weight_c > 0.2).float().mean().item()),
                "crop_survival_where_solo": float(
                    (weight_c[covered_c] / solo_c[covered_c].clamp(min=1e-6)).mean().item()
                )
                if covered_c.any()
                else None,
                "crop_mean_blocked_where_solo": float(blocked_c[covered_c].mean().item())
                if covered_c.any()
                else None,
            }
        )
        save_gray_image(solo_c, os.path.join(level_dir, "crop_detail_only_alpha.png"))
        save_gray_image(weight_c, os.path.join(level_dir, "crop_detail_composite_weight.png"))
        save_gray_image(blocked_c, os.path.join(level_dir, "crop_detail_blocked.png"))

    save_gray_image(solo, os.path.join(level_dir, "detail_only_alpha.png"))
    save_gray_image(weight, os.path.join(level_dir, "detail_composite_weight.png"))
    save_gray_image(blocked, os.path.join(level_dir, "detail_blocked.png"))
    # Red = solo coverage lost to occlusion; green = contribution that survives compositing.
    compare = torch.stack([blocked, weight, torch.zeros_like(weight)], dim=0).clamp(0.0, 1.0)
    save_tensor_image(compare, os.path.join(level_dir, "detail_coverage_vs_weight.png"))
    write_json(os.path.join(level_dir, "detail_contribution.json"), stats)
    print(json.dumps(stats, indent=2))
    print(f"Wrote diagnostics to {level_dir}")


if __name__ == "__main__":
    main()
