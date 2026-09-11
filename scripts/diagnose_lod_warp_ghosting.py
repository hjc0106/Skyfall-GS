#!/usr/bin/env python3
"""Local geometry-warp diagnosis for tree / eave ghosting. Does not change thresholds."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from arguments import ModelParams, PipelineParams
from lod.correspondence import rade_pkg_camera_z, warp_source_into_destination
from lod.importer import import_skyfall_l0, load_lod_onto_bundle
from lod.path import DEFAULT_GZ_ROOT
from lod.rasterizer import require_rade_gs
from lod.render import render_lod_appearance
from lod.warp_diag import (
    GHOSTING_CROPS,
    classify_allowed_errors,
    class_id_map,
    class_overlay,
    colorize_scalar,
    contact_sheet,
    crop_image,
    depth_consistency_maps,
    fractional_crop,
    region_report,
    rgb_l1,
    sample_plane,
    tensor_to_image,
    upsample_mask,
)
from refinement.dloral_flows import downsample_image_flow, roundtrip_diagnostics
from refinement.geometry_warp import estimate_depth_scene_scale, estimate_spatial_target, rank_cameras_by_spatial_overlap
from scene import GaussianModel, Scene
from train_zoom_gen import apply_stage1_cfg_to_args, load_stage1_cfg, pick_base_camera
from utils.general_utils import safe_state
from utils.zoom_camera import NormalizedROI, make_zoom_camera
from utils.zoom_mvp_utils import embedding_for_train_camera, select_appearance_embedding


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def _spatial(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().float()
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    return value


def _render(bundle, camera, *, background, kernel: float, embedding):
    pkg = render_lod_appearance(
        bundle, camera, background=background, kernel_size=kernel,
        appearance_embedding=embedding, lod=True, compact=False,
    )
    return {
        "rgb": pkg["render"].clamp(0, 1),
        "alpha": _spatial(pkg["render_alpha"]),
        "camera_z": rade_pkg_camera_z(pkg, camera),
    }


def _roundtrip_map(forward, forward_valid, reverse, reverse_valid) -> torch.Tensor:
    height, width = forward_valid.shape
    device = forward.device
    dtype = forward.dtype
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    origin = torch.stack((xs, ys), dim=-1)
    source = origin + forward
    from refinement.dloral_flows import sample_flow_at

    reverse_at_source, reverse_hit = sample_flow_at(reverse, source, reverse_valid)
    error = (source + reverse_at_source - origin).norm(dim=-1)
    return torch.where(forward_valid & reverse_hit & torch.isfinite(error), error, torch.full_like(error, float("nan")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    ModelParams(parser)
    PipelineParams(parser)
    parser.add_argument("--start_checkpoint", type=str, required=True)
    parser.add_argument("--l1_checkpoint", type=str, required=True)
    parser.add_argument("--supervision_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--gz_root", type=str, default=str(DEFAULT_GZ_ROOT))
    parser.add_argument("--view_index", type=int, default=0)
    parser.add_argument("--roi_center_x", type=float, default=0.28)
    parser.add_argument("--roi_center_y", type=float, default=0.28)
    parser.add_argument("--roi_width", type=float, default=0.1)
    parser.add_argument("--roi_height", type=float, default=0.1)
    parser.add_argument("--zoom_factor", type=float, default=4.0)
    parser.add_argument("--step_scale", type=float, default=2.0)
    parser.add_argument("--alpha_threshold", type=float, default=0.05)
    parser.add_argument("--neighbor_name", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    require_rade_gs()
    safe_state(args.quiet)
    supervision = os.path.abspath(args.supervision_dir)
    output_dir = os.path.abspath(args.output_dir or os.path.join(supervision, "warp_ghosting"))
    os.makedirs(output_dir, exist_ok=True)
    geometry = json.loads(open(os.path.join(supervision, "geometry.json"), encoding="utf-8").read())
    neighbor_name = args.neighbor_name or (geometry.get("selected") or [{}])[0].get("name")
    if not neighbor_name:
        raise ValueError("no neighbor name in geometry.json")

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
    gaussians.compute_3D_filter(cameras=train_cameras)
    bundle = import_skyfall_l0(
        gaussians, train_cameras, gz_root=args.gz_root, step_scale=args.step_scale, freeze=True,
    )
    load_lod_onto_bundle(bundle, os.path.abspath(args.l1_checkpoint), device=str(gaussians.get_xyz.device))
    roi = NormalizedROI(args.roi_center_x, args.roi_center_y, args.roi_width, args.roi_height)
    base_camera, is_train_view = pick_base_camera(scene, args.view_index, False)
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device=gaussians.get_xyz.device,
    )
    kernel = float(dataset.kernel_size)
    embedding = select_appearance_embedding(gaussians, base_camera.uid, is_train_view)
    target_zoom = make_zoom_camera(
        base_camera, roi, args.zoom_factor, uid=50000,
        image_name=f"{base_camera.image_name}_zoom{args.zoom_factor:g}",
    )
    target = _render(bundle, target_zoom, background=background, kernel=kernel, embedding=embedding)
    spatial = estimate_spatial_target(
        target_zoom, target["camera_z"], target["alpha"], alpha_threshold=args.alpha_threshold,
    )
    scene_scale = float(geometry.get("depth_scene_scale") or estimate_depth_scene_scale(target_zoom, spatial.median_depth or 1.0))
    selected, _ = rank_cameras_by_spatial_overlap(
        spatial, train_cameras, zoom_factor=args.zoom_factor, k=2, pool_size=8,
        target_camera=target_zoom, exclude_uids=(int(base_camera.uid), int(target_zoom.uid)),
    )
    match = next((item for item in selected if item[0].image_name == neighbor_name), None)
    if match is None:
        raise ValueError(f"neighbor {neighbor_name} was not in the spatial ranking")
    base_neighbor, projected = match
    neighbor_roi = NormalizedROI(projected.center_x, projected.center_y, projected.width, projected.height)
    neighbor_zoom = make_zoom_camera(
        base_neighbor, neighbor_roi, args.zoom_factor, uid=51000,
        image_name=f"{base_neighbor.image_name}_zoom{args.zoom_factor:g}_spatial",
    )
    neigh = _render(
        bundle, neighbor_zoom, background=background, kernel=kernel,
        embedding=embedding_for_train_camera(gaussians, base_neighbor.uid),
    )
    warp, warped = warp_source_into_destination(
        dest_camera=target_zoom, source_camera=neighbor_zoom,
        dest_depth=target["camera_z"], source_depth=neigh["camera_z"],
        dest_alpha=target["alpha"], source_alpha=neigh["alpha"],
        source_rgb=neigh["rgb"], scene_scale=scene_scale, alpha_threshold=args.alpha_threshold,
    )
    reverse, _ = warp_source_into_destination(
        dest_camera=neighbor_zoom, source_camera=target_zoom,
        dest_depth=neigh["camera_z"], source_depth=target["camera_z"],
        dest_alpha=neigh["alpha"], source_alpha=target["alpha"],
        source_rgb=target["rgb"], scene_scale=scene_scale, alpha_threshold=args.alpha_threshold,
    )
    depth = depth_consistency_maps(
        target_zoom, target["camera_z"], neighbor_zoom, neigh["camera_z"], scene_scale=scene_scale,
    )
    sampled_alpha = sample_plane(neigh["alpha"], depth["grid"])
    residual = rgb_l1(warped, target["rgb"])
    rt_map = _roundtrip_map(warp.pixel_flow, warp.valid_mask, reverse.pixel_flow, reverse.valid_mask)
    valid = warp.valid_mask.detach().cpu()
    residual = residual.detach().cpu()
    rt_map = rt_map.detach().cpu()
    slack = depth["slack_ratio"].detach().cpu()
    dest_alpha = target["alpha"].detach().cpu()
    sampled_alpha = sampled_alpha.detach().cpu()
    warped_cpu = warped.detach().cpu()
    target_cpu = target["rgb"].detach().cpu()
    feat_flow, feat_valid = downsample_image_flow(warp.pixel_flow.detach().cpu(), valid=valid)
    feat_up = upsample_mask(feat_valid, int(valid.shape[0]), int(valid.shape[1]))
    saved_feat = os.path.join(supervision, "dloral", "flow_valid.npy")
    if os.path.isfile(saved_feat):
        saved = torch.from_numpy(np.load(saved_feat))
        feat_valid = saved.bool() if tuple(saved.shape) == tuple(feat_valid.shape) else feat_valid
        feat_up = upsample_mask(feat_valid, int(valid.shape[0]), int(valid.shape[1]))
    classes = classify_allowed_errors(valid, residual, rt_map, slack, dest_alpha, sampled_alpha)
    ids = class_id_map(classes, valid)

    height, width = int(valid.shape[0]), int(valid.shape[1])
    reverse_feat = downsample_image_flow(reverse.pixel_flow.detach().cpu(), valid=reverse.valid_mask.detach().cpu())
    latent_rt = roundtrip_diagnostics(feat_flow, feat_valid, *reverse_feat)
    report = {
        "neighbor": neighbor_name,
        "scene_scale": scene_scale,
        "thresholds": {
            "rgb_l1": 0.05,
            "roundtrip_px": 1.0,
            "mixed_alpha": [0.2, 0.8],
            "near_slack": 0.5,
            "note": "Reporting labels only. Do not tighten the global depth/coverage tests from this file.",
        },
        "full": region_report(
            valid=valid, feat_valid=feat_up, rgb_residual=residual,
            roundtrip=rt_map, slack_ratio=slack, dest_alpha=dest_alpha,
            classes=classes,
        ),
        "crops": {},
        "warp_metadata": warp.metadata,
        "feature_roundtrip": latent_rt,
        "do_not": ["tighten_global_threshold", "change_densify", "add_scale_offset_constraint", "go_to_8x"],
    }
    panels_full = [
        ("target", tensor_to_image(target_cpu)),
        ("warped", tensor_to_image(warped_cpu)),
        ("rgb residual", colorize_scalar(residual, vmax=0.15, mask=valid)),
        ("roundtrip px", colorize_scalar(torch.nan_to_num(rt_map, nan=0.0), vmax=4.0, mask=valid)),
        ("depth slack", colorize_scalar(slack.clamp(0, 2), vmax=1.0, mask=valid)),
        ("dest alpha", tensor_to_image(dest_alpha)),
        ("sampled alpha", tensor_to_image(sampled_alpha)),
        ("full-res valid", tensor_to_image(valid.float())),
        ("feature valid x8", tensor_to_image(feat_up.float())),
        ("allowed errors", class_overlay(ids)),
    ]
    overview = contact_sheet(panels_full, columns=5)
    overview.resize((overview.width // 4, overview.height // 4), getattr(Image, "BOX", Image.Resampling.BOX)).save(
        os.path.join(output_dir, "overview.png")
    )

    for name, frac in GHOSTING_CROPS.items():
        box = fractional_crop(width, height, frac)
        x0, y0, x1, y1 = box
        region = torch.zeros_like(valid)
        region[y0:y1, x0:x1] = True
        report["crops"][name] = {
            "box": list(box),
            **region_report(
                valid=valid, feat_valid=feat_up, rgb_residual=residual,
                roundtrip=rt_map, slack_ratio=slack, dest_alpha=dest_alpha,
                classes=classes, region=region,
            ),
        }
        crop_panels = [(title, crop_image(image, box)) for title, image in panels_full]
        contact_sheet(crop_panels, columns=5).save(os.path.join(output_dir, f"{name}_sheet.png"))

    _write_json(os.path.join(output_dir, "report.json"), report)
    print(json.dumps({
        "output_dir": output_dir,
        "neighbor": neighbor_name,
        "full_ghosted_frac_of_valid": report["full"]["ghosted_frac_of_valid"],
        "classes": report["full"]["classes_frac_of_valid"],
        "trees": report["crops"]["trees"]["classes_frac_of_valid"],
        "eaves": report["crops"]["eaves"]["classes_frac_of_valid"],
        "feature_leaked_trees": report["crops"]["trees"]["feature_leaked_frac"],
        "median_roundtrip": report["full"]["roundtrip_px"]["p50"],
        "median_slack": report["full"]["slack_ratio"]["p50"],
    }, indent=2))


if __name__ == "__main__":
    main()
