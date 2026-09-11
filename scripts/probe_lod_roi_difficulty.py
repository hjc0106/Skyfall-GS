#!/usr/bin/env python3
"""Score candidate ROIs for parallax and occlusion on frozen L0. Does not train."""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from arguments import ModelParams, PipelineParams
from lod.correspondence import rade_pkg_camera_z, warp_source_into_destination
from lod.importer import import_skyfall_l0
from lod.path import DEFAULT_GZ_ROOT
from lod.rasterizer import require_rade_gs
from lod.render import render_lod_appearance
from refinement.geometry_warp import estimate_depth_scene_scale, estimate_spatial_target, rank_cameras_by_spatial_overlap
from scene import GaussianModel, Scene
from train_zoom_gen import apply_stage1_cfg_to_args, load_stage1_cfg, pick_base_camera
from utils.general_utils import safe_state
from utils.zoom_camera import NormalizedROI, make_zoom_camera, save_roi_overlay
from utils.zoom_mvp_utils import embedding_for_train_camera, save_tensor_image, select_appearance_embedding


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


def _depth_stats(depth: torch.Tensor, alpha: torch.Tensor, threshold: float = 0.05) -> dict[str, float]:
    valid = (alpha >= threshold) & torch.isfinite(depth) & (depth > 0)
    if not bool(valid.any()):
        return {"valid_frac": 0.0, "spread": 0.0, "edge": 0.0, "median": 0.0}
    values = depth[valid]
    q = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9], device=values.device))
    spread = float(((q[2] - q[0]) / q[1].clamp_min(1e-6)).item())
    kernel_x = depth.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    kernel_y = depth.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    plane = depth.float()[None, None]
    grad = (F.conv2d(plane, kernel_x, padding=1).abs() + F.conv2d(plane, kernel_y, padding=1).abs())[0, 0]
    edge = float(grad[valid].mean().item() / float(q[1].clamp_min(1e-6).item()))
    return {
        "valid_frac": float(valid.float().mean().item()),
        "spread": spread,
        "edge": edge,
        "median": float(q[1].item()),
    }


def _score_roi(
    *,
    bundle,
    base_camera,
    train_cameras,
    roi: NormalizedROI,
    zoom_factor: float,
    background,
    kernel: float,
    embedding,
    gaussians,
    neighbor_count: int,
    pool_size: int,
    alpha_threshold: float,
) -> dict:
    try:
        roi.validate_for_zoom(4.0)
        roi.validate_for_zoom(zoom_factor)
    except ValueError as exc:
        return {"ok": False, "reason": str(exc), "roi": vars(roi)}
    target_zoom = make_zoom_camera(
        base_camera, roi, zoom_factor, uid=70000,
        image_name=f"{base_camera.image_name}_probe{zoom_factor:g}",
    )
    target = _render(bundle, target_zoom, background=background, kernel=kernel, embedding=embedding)
    spatial = estimate_spatial_target(
        target_zoom, target["camera_z"], target["alpha"], alpha_threshold=alpha_threshold,
    )
    scene_scale = estimate_depth_scene_scale(target_zoom, spatial.median_depth or 1.0)
    selected, _ = rank_cameras_by_spatial_overlap(
        spatial, train_cameras, zoom_factor=zoom_factor,
        k=neighbor_count, pool_size=pool_size, target_camera=target_zoom,
        exclude_uids=(int(base_camera.uid), int(target_zoom.uid)),
    )
    neighbors = []
    for index, (base_neighbor, projected) in enumerate(selected):
        neighbor_roi = NormalizedROI(projected.center_x, projected.center_y, projected.width, projected.height)
        try:
            neighbor_zoom = make_zoom_camera(
                base_neighbor, neighbor_roi, zoom_factor, uid=71000 + index,
                image_name=f"{base_neighbor.image_name}_probe_spatial",
            )
        except ValueError as exc:
            neighbors.append({"name": base_neighbor.image_name, "ok": False, "reason": str(exc)})
            continue
        neigh = _render(
            bundle, neighbor_zoom, background=background, kernel=kernel,
            embedding=embedding_for_train_camera(gaussians, base_neighbor.uid),
        )
        warp, _ = warp_source_into_destination(
            dest_camera=target_zoom, source_camera=neighbor_zoom,
            dest_depth=target["camera_z"], source_depth=neigh["camera_z"],
            dest_alpha=target["alpha"], source_alpha=neigh["alpha"],
            source_rgb=neigh["rgb"], scene_scale=scene_scale, alpha_threshold=alpha_threshold,
        )
        reverse, _ = warp_source_into_destination(
            dest_camera=neighbor_zoom, source_camera=target_zoom,
            dest_depth=neigh["camera_z"], source_depth=target["camera_z"],
            dest_alpha=neigh["alpha"], source_alpha=target["alpha"],
            source_rgb=target["rgb"], scene_scale=scene_scale, alpha_threshold=alpha_threshold,
        )
        shift = (
            (projected.center_x - roi.center_x) ** 2 + (projected.center_y - roi.center_y) ** 2
        ) ** 0.5
        neighbors.append({
            "ok": True,
            "name": base_neighbor.image_name,
            "coverage": float(warp.coverage),
            "reverse_coverage": float(reverse.coverage),
            "center_shift": float(shift),
            "projected_center": [projected.center_x, projected.center_y],
            "in_frustum_fraction": float(projected.in_frustum_fraction),
        })
    ok_neighbors = [row for row in neighbors if row.get("ok")]
    depth = _depth_stats(target["camera_z"], target["alpha"], alpha_threshold)
    mean_shift = sum(row["center_shift"] for row in ok_neighbors) / len(ok_neighbors) if ok_neighbors else 0.0
    mean_reverse = sum(row["reverse_coverage"] for row in ok_neighbors) / len(ok_neighbors) if ok_neighbors else 0.0
    mean_cover = sum(row["coverage"] for row in ok_neighbors) / len(ok_neighbors) if ok_neighbors else 0.0
    occlusion = 1.0 - mean_reverse
    return {
        "ok": True,
        "roi": {"center_x": roi.center_x, "center_y": roi.center_y, "width": roi.width, "height": roi.height},
        "depth": depth,
        "n_neighbors": len(ok_neighbors),
        "mean_coverage": mean_cover,
        "mean_reverse_coverage": mean_reverse,
        "mean_center_shift": mean_shift,
        "occlusion": occlusion,
        "difficulty": mean_shift * 4.0 + occlusion * 1.5 + float(depth["spread"]) * 0.25 + float(depth["edge"]) * 0.15,
        "neighbors": neighbors,
        "target_rgb": target["rgb"],
    }


def _candidates(width: float, height: float) -> list[NormalizedROI]:
    rois = [NormalizedROI(0.592, 0.53, width, height)]
    for cx in (0.28, 0.38, 0.48, 0.62, 0.72):
        for cy in (0.28, 0.40, 0.52, 0.64, 0.74):
            if abs(cx - 0.592) < 0.04 and abs(cy - 0.53) < 0.04:
                continue
            rois.append(NormalizedROI(cx, cy, width, height))
    return rois


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    ModelParams(parser)
    PipelineParams(parser)
    parser.add_argument("--start_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gz_root", type=str, default=str(DEFAULT_GZ_ROOT))
    parser.add_argument("--view_index", type=int, default=0)
    parser.add_argument("--roi_width", type=float, default=0.1)
    parser.add_argument("--roi_height", type=float, default=0.1)
    parser.add_argument("--zoom_factor", type=float, default=2.0)
    parser.add_argument("--step_scale", type=float, default=2.0)
    parser.add_argument("--geometry_neighbor_count", type=int, default=2)
    parser.add_argument("--geometry_pool_size", type=int, default=8)
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
        ply_path=os.path.dirname(args.start_checkpoint),
    )
    gaussians.load_from_checkpoints(model_params)
    train_cameras = list(scene.getTrainCameras())
    gaussians.compute_3D_filter(cameras=train_cameras)
    bundle = import_skyfall_l0(
        gaussians, train_cameras, gz_root=args.gz_root, step_scale=args.step_scale, freeze=True,
    )
    base_camera, is_train_view = pick_base_camera(scene, args.view_index, False)
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device=gaussians.get_xyz.device,
    )
    embedding = select_appearance_embedding(gaussians, base_camera.uid, is_train_view)
    kernel = float(dataset.kernel_size)
    save_roi_overlay(
        base_camera, NormalizedROI(0.592, 0.53, args.roi_width, args.roi_height),
        os.path.join(output_dir, "baseline_roi.png"),
    )
    scored = []
    for index, roi in enumerate(_candidates(args.roi_width, args.roi_height)):
        rec = _score_roi(
            bundle=bundle, base_camera=base_camera, train_cameras=train_cameras, roi=roi,
            zoom_factor=args.zoom_factor, background=background, kernel=kernel,
            embedding=embedding, gaussians=gaussians,
            neighbor_count=int(args.geometry_neighbor_count),
            pool_size=int(args.geometry_pool_size), alpha_threshold=0.05,
        )
        rgb = rec.pop("target_rgb", None)
        rec["index"] = index
        rec["is_baseline"] = abs(roi.center_x - 0.592) < 1e-6 and abs(roi.center_y - 0.53) < 1e-6
        if rec.get("ok") and rgb is not None:
            tag = f"{roi.center_x:.3f}_{roi.center_y:.3f}"
            save_tensor_image(rgb, os.path.join(output_dir, f"zoom2_{tag}.png"))
            save_roi_overlay(base_camera, roi, os.path.join(output_dir, f"overlay_{tag}.png"))
            del rgb
        scored.append(rec)
        print(json.dumps({k: rec[k] for k in rec if k not in ("neighbors", "target_rgb")}, default=str))

    runnable = [
        row for row in scored
        if row.get("ok") and row.get("mean_coverage", 0) >= 0.3 and row.get("n_neighbors", 0) >= 1
    ]
    baseline = next((row for row in scored if row.get("is_baseline")), None)
    harder = sorted(
        [row for row in runnable if not row.get("is_baseline")],
        key=lambda row: (
            row["difficulty"],
            row["mean_center_shift"],
            row["occlusion"],
        ),
        reverse=True,
    )
    chosen = None
    if baseline is not None:
        for row in harder:
            if row["mean_center_shift"] > baseline["mean_center_shift"] * 1.3 and row["occlusion"] > baseline["occlusion"]:
                chosen = row
                break
    if chosen is None and harder:
        chosen = harder[0]
    payload = {
        "baseline": baseline,
        "chosen": None if chosen is None else {k: v for k, v in chosen.items() if k != "target_rgb"},
        "ranked": [{k: v for k, v in row.items() if k != "target_rgb"} for row in harder[:8]],
        "all": [{k: v for k, v in row.items() if k != "target_rgb"} for row in scored],
    }
    _write_json(os.path.join(output_dir, "roi_probe.json"), payload)
    print(json.dumps({
        "baseline_shift": None if baseline is None else baseline.get("mean_center_shift"),
        "baseline_occlusion": None if baseline is None else baseline.get("occlusion"),
        "chosen": None if chosen is None else chosen["roi"],
        "chosen_shift": None if chosen is None else chosen.get("mean_center_shift"),
        "chosen_occlusion": None if chosen is None else chosen.get("occlusion"),
    }, indent=2))


if __name__ == "__main__":
    main()
