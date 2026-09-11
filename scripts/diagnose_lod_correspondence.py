#!/usr/bin/env python3
"""Depth-driven co-visible correspondence for saved L1 bundles. Does not train."""

from __future__ import annotations

import argparse
import json
import os

import torch

from arguments import ModelParams, PipelineParams
from lod.correspondence import neighbor_correspondence_report
from lod.importer import import_skyfall_l0, load_lod_onto_bundle
from lod.path import DEFAULT_GZ_ROOT
from lod.rasterizer import require_rade_gs
from scene import GaussianModel, Scene
from train_zoom_gen import apply_stage1_cfg_to_args, load_stage1_cfg, pick_base_camera
from utils.general_utils import safe_state
from utils.zoom_camera import NormalizedROI


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def _compact(payload: dict) -> dict:
    compact = {"target": payload.get("target"), "neighbors": {}}
    for name, rec in (payload.get("neighbors") or {}).items():
        l0 = rec.get("l0_depth") or {}
        trained = rec.get("trained_depth") or {}
        compact["neighbors"][name] = {
            "direct_change_rgb": (rec.get("neighbor_vs_l0_direct") or {}).get("rgb_l1"),
            "direct_change_hf": (rec.get("neighbor_vs_l0_direct") or {}).get("hf_l1"),
            "covisible_change_rgb": (rec.get("neighbor_vs_l0_direct") or {}).get("rgb_l1_covisible_l0depth"),
            "l0_depth_coverage": l0.get("coverage"),
            "floor_hf": (l0.get("floor_l0_rgb") or {}).get("hf_l1"),
            "matched_hf": (l0.get("l1_rgb") or {}).get("hf_l1"),
            "excess_hf": (l0.get("excess_over_floor") or {}).get("hf_l1"),
            "excess_rgb": (l0.get("excess_over_floor") or {}).get("rgb_l1"),
            "unexplained_ratio": (l0.get("change_agreement") or {}).get("unexplained_ratio"),
            "vehicles_excess_hf": ((l0.get("excess_over_floor") or {}).get("crops") or {}).get("vehicles", {}).get("hf_l1"),
            "building_excess_hf": ((l0.get("excess_over_floor") or {}).get("crops") or {}).get("building", {}).get("hf_l1"),
            "occlusion_excess_hf": ((l0.get("excess_over_floor") or {}).get("occlusion_boundary") or {}).get("hf_l1"),
            "shared_emb_excess_hf": ((l0.get("shared_target_embedding") or {}).get("excess_over_floor") or {}).get("hf_l1"),
            "trained_coverage": trained.get("coverage"),
            "trained_matched_hf": (trained.get("l1_rgb") or {}).get("hf_l1"),
            "newly_occluded": trained.get("frac_newly_occluded"),
            "outlier_concentration": ((rec.get("outliers_on_l0depth_l1_residual") or {}).get("scale_or_offset_gt_4x") or {}).get("concentration_vs_all_l1"),
        }
    return compact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    ModelParams(parser)
    PipelineParams(parser)
    parser.add_argument("--start_checkpoint", type=str, required=True)
    parser.add_argument("--root", type=str, required=True, help="Directory with mode subdirs that contain l1_final.lod.pt")
    parser.add_argument("--modes", type=str, default="target_only,spynet,geometry")
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
    root = os.path.abspath(args.root)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    missing = [mode for mode in modes if not os.path.isfile(os.path.join(root, mode, "l1_final.lod.pt"))]
    if missing:
        raise FileNotFoundError(f"missing l1_final.lod.pt for modes: {missing}")

    stage1_cfg = load_stage1_cfg(os.path.dirname(os.path.abspath(args.start_checkpoint)))
    apply_stage1_cfg_to_args(args, stage1_cfg)
    extract_parser = argparse.ArgumentParser(add_help=False)
    dataset = ModelParams(extract_parser).extract(args)
    dataset.model_path = os.path.join(root, "correspondence_scratch")

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
    gaussians.compute_3D_filter(cameras=train_cameras)
    bundle = import_skyfall_l0(
        gaussians, train_cameras, gz_root=args.gz_root, step_scale=args.step_scale, freeze=True,
    )
    roi = NormalizedROI(args.roi_center_x, args.roi_center_y, args.roi_width, args.roi_height)
    base_camera, _ = pick_base_camera(scene, args.view_index, False)
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device=gaussians.get_xyz.device,
    )
    summary = {"modes": {}, "question": "Are neighbor-view L1 edits corresponding details, or only larger changes?"}
    for mode in modes:
        lod_path = os.path.join(root, mode, "l1_final.lod.pt")
        load_lod_onto_bundle(bundle, lod_path, device=str(gaussians.get_xyz.device))
        out_dir = os.path.join(root, mode, "correspondence")
        os.makedirs(out_dir, exist_ok=True)
        payload = neighbor_correspondence_report(
            bundle, base_camera, train_cameras, roi,
            background=background, kernel=float(dataset.kernel_size), gaussians=gaussians,
            zoom_factor=args.zoom_factor, out_dir=out_dir,
        )
        _write_json(os.path.join(root, mode, "correspondence.json"), payload)
        compact = _compact(payload)
        summary["modes"][mode] = compact
        print(json.dumps({"mode": mode, **compact}, indent=2))
    _write_json(os.path.join(root, "lod_l1_correspondence_summary.json"), summary)


if __name__ == "__main__":
    main()
