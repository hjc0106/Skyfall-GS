#!/usr/bin/env python3
"""L0 import acceptance: Skyfall Stage1 vs imported GaussianLoD, no new Gaussians.

Compares original training views and the current 2x focal zoom (still 2048 px).
Does not recompute Skyfall filter_3D after import and does not run super-resolution.
"""

from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace
from typing import Any

import torch

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from lod.camera import lod_camera_diagnostics, skyfall_camera_to_lod
from lod.importer import assert_l0_tensors_match, copy_l0_into_gaussian_model, import_skyfall_l0
from lod.path import DEFAULT_GZ_ROOT
from scene import GaussianModel, Scene
from train_zoom_gen import apply_stage1_cfg_to_args, load_stage1_cfg, pick_base_camera
from utils.general_utils import safe_state
from utils.zoom_camera import NormalizedROI, make_zoom_camera
from utils.zoom_mvp_utils import (
    embedding_for_train_camera,
    image_hf_l1,
    image_l1,
    save_gray_image,
    save_tensor_image,
    select_appearance_embedding,
)


def _spatial_map(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None or not torch.is_tensor(tensor):
        return None
    value = tensor.detach().float()
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or min(value.shape) < 8:
        return None
    return value


def _save_map(tensor: torch.Tensor | None, path: str) -> None:
    value = _spatial_map(tensor)
    if value is None:
        return
    denom = value.max().clamp_min(1e-6)
    save_gray_image(value / denom, path)


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def _render_pkg(camera, gaussians, pipe, background, kernel_size, embedding):
    return render(
        camera,
        gaussians,
        pipe,
        background,
        kernel_size=kernel_size,
        testing=False,
        appearance_embedding=embedding,
    )


def _pair_metrics(native: dict[str, torch.Tensor], imported: dict[str, torch.Tensor]) -> dict[str, float]:
    rgb_a, rgb_b = native["render"], imported["render"]
    metrics = {
        "rgb_l1": image_l1(rgb_a, rgb_b),
        "rgb_hf_l1": image_hf_l1(rgb_a, rgb_b),
        "rgb_max": float((rgb_a - rgb_b).abs().max().item()),
    }
    alpha_a = _spatial_map(native.get("render_alpha"))
    alpha_b = _spatial_map(imported.get("render_alpha"))
    if alpha_a is not None and alpha_b is not None:
        metrics["alpha_l1"] = image_l1(alpha_a.unsqueeze(0), alpha_b.unsqueeze(0))
        metrics["alpha_max"] = float((alpha_a - alpha_b).abs().max().item())
        metrics["alpha_shape"] = list(alpha_a.shape)
    depth_a = _spatial_map(native.get("render_depth"))
    depth_b = _spatial_map(imported.get("render_depth"))
    if depth_a is not None and depth_b is not None:
        metrics["depth_l1"] = image_l1(depth_a.unsqueeze(0), depth_b.unsqueeze(0))
        metrics["depth_max"] = float((depth_a - depth_b).abs().max().item())
        metrics["depth_shape"] = list(depth_a.shape)
        if alpha_a is not None:
            opaque = alpha_a > 0.05
            if bool(opaque.any().item()):
                diff = (depth_a - depth_b).abs()[opaque]
                metrics["depth_l1_opaque"] = float(diff.mean().item())
                metrics["depth_rel_opaque"] = float(
                    (diff / depth_a[opaque].abs().clamp_min(1e-3)).mean().item()
                )
    return metrics


def _mean(items: list[dict[str, float]], key: str) -> float | None:
    values = [item[key] for item in items if key in item]
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    ModelParams(parser)
    PipelineParams(parser)
    parser.add_argument("--start_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gz_root", type=str, default=str(DEFAULT_GZ_ROOT))
    parser.add_argument("--view_index", type=int, default=0)
    parser.add_argument("--view_name", type=str, default="", help="Lock the base camera by image stem. Preferred over view_index.")
    parser.add_argument("--roi_center_x", type=float, default=0.592)
    parser.add_argument("--roi_center_y", type=float, default=0.53)
    parser.add_argument("--roi_width", type=float, default=0.1)
    parser.add_argument("--roi_height", type=float, default=0.1)
    parser.add_argument("--zoom_factor", type=float, default=2.0)
    parser.add_argument("--step_scale", type=float, default=2.0, help="Focal LoD step, not an image-size ratio")
    parser.add_argument("--max_train_views", type=int, default=0, help="0 means every training view")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    safe_state(args.quiet)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    stage1_checkpoint = os.path.abspath(args.start_checkpoint)
    stage1_cfg = load_stage1_cfg(os.path.dirname(stage1_checkpoint))
    apply_stage1_cfg_to_args(args, stage1_cfg)
    extract_parser = argparse.ArgumentParser(add_help=False)
    dataset = ModelParams(extract_parser).extract(args)
    pipe = PipelineParams(extract_parser).extract(args)
    dataset.model_path = output_dir

    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs,
        dataset.appearance_embedding_dim,
    )
    model_params, first_iter = torch.load(stage1_checkpoint, weights_only=False)
    scene = Scene(dataset, gaussians, load_iteration=first_iter, shuffle=False, ply_path=os.path.dirname(stage1_checkpoint))
    gaussians.load_from_checkpoints(model_params)
    train_cameras = scene.getTrainCameras()
    gaussians.compute_3D_filter(cameras=list(train_cameras))

    roi = NormalizedROI(args.roi_center_x, args.roi_center_y, args.roi_width, args.roi_height)
    base_camera, is_train_view = pick_base_camera(scene, args.view_index, False, image_name=args.view_name)
    zoom_camera = make_zoom_camera(
        base_camera,
        roi,
        args.zoom_factor,
        uid=10000,
        image_name=f"{base_camera.image_name}_zoom{args.zoom_factor:g}",
    )

    bundle = import_skyfall_l0(
        gaussians,
        train_cameras,
        gz_root=args.gz_root,
        step_scale=args.step_scale,
        freeze=True,
    )
    tensor_report = assert_l0_tensors_match(gaussians, bundle)
    twin = GaussianModel(
        dataset.sh_degree,
        dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs,
        dataset.appearance_embedding_dim,
    )
    twin.active_sh_degree = gaussians.active_sh_degree
    twin.max_sh_degree = gaussians.max_sh_degree
    copy_l0_into_gaussian_model(bundle, twin)

    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device=gaussians.get_xyz.device,
    )
    views = list(train_cameras)
    if args.max_train_views > 0:
        views = views[: args.max_train_views]
    train_metrics = []
    for camera in views:
        embedding = embedding_for_train_camera(gaussians, camera.uid)
        native = _render_pkg(camera, gaussians, pipe, background, dataset.kernel_size, embedding)
        imported = _render_pkg(camera, twin, pipe, background, dataset.kernel_size, embedding)
        row = {"image_name": camera.image_name, **_pair_metrics(native, imported)}
        train_metrics.append(row)

    zoom_embedding = select_appearance_embedding(gaussians, base_camera.uid, is_train_view)
    native_zoom = _render_pkg(zoom_camera, gaussians, pipe, background, dataset.kernel_size, zoom_embedding)
    imported_zoom = _render_pkg(zoom_camera, twin, pipe, background, dataset.kernel_size, zoom_embedding)
    zoom_metrics = _pair_metrics(native_zoom, imported_zoom)
    save_tensor_image(native_zoom["render"], os.path.join(output_dir, "zoom_native.png"))
    save_tensor_image(imported_zoom["render"], os.path.join(output_dir, "zoom_imported.png"))
    save_tensor_image(
        (native_zoom["render"] - imported_zoom["render"]).abs() * 10.0,
        os.path.join(output_dir, "zoom_absdiff_x10.png"),
    )
    _save_map(native_zoom.get("render_depth"), os.path.join(output_dir, "zoom_depth_native.png"))
    _save_map(imported_zoom.get("render_depth"), os.path.join(output_dir, "zoom_depth_imported.png"))
    _save_map(native_zoom.get("render_alpha"), os.path.join(output_dir, "zoom_alpha_native.png"))
    _save_map(imported_zoom.get("render_alpha"), os.path.join(output_dir, "zoom_alpha_imported.png"))

    native_base = _render_pkg(
        base_camera,
        gaussians,
        pipe,
        background,
        dataset.kernel_size,
        embedding_for_train_camera(gaussians, base_camera.uid),
    )
    imported_base = _render_pkg(
        base_camera,
        twin,
        pipe,
        background,
        dataset.kernel_size,
        embedding_for_train_camera(gaussians, base_camera.uid),
    )
    save_tensor_image(native_base["render"], os.path.join(output_dir, "train_native.png"))
    save_tensor_image(imported_base["render"], os.path.join(output_dir, "train_imported.png"))

    lod_train = skyfall_camera_to_lod(base_camera, gz_root=args.gz_root)
    lod_zoom = skyfall_camera_to_lod(zoom_camera, gz_root=args.gz_root)
    filter_abs = (bundle.layer0().filter_3d - gaussians.filter_3D).abs()
    payload = {
        "accepted": True,
        "note": (
            "L0 import copies Skyfall parameters and filter_3D; appearance stays frozen. "
            "2x is a focal zoom at 2048 px. Rasterizer is Skyfall diff_gauss; RaDe-GS is not required for this check."
        ),
        "n_points": bundle.n_points,
        "sh_degree": bundle.sh_degree,
        "step_scale": bundle.step_scale,
        "stage0_scale": bundle.lod.stage_records[0]["scale"],
        "tensor_max_abs": tensor_report,
        "filter_copied_max_abs": float(filter_abs.max().item()),
        "l0_frozen": bool(bundle.layer0().frozen),
        "appearance_enabled": bool(bundle.appearance.enabled),
        "appearance_frozen": not any(p.requires_grad for p in bundle.appearance.mlp.parameters())
        if bundle.appearance.mlp is not None
        else True,
        "train_view_mean": {
            "rgb_l1": _mean(train_metrics, "rgb_l1"),
            "rgb_hf_l1": _mean(train_metrics, "rgb_hf_l1"),
            "depth_l1": _mean(train_metrics, "depth_l1"),
            "depth_l1_opaque": _mean(train_metrics, "depth_l1_opaque"),
            "depth_rel_opaque": _mean(train_metrics, "depth_rel_opaque"),
            "alpha_l1": _mean(train_metrics, "alpha_l1"),
        },
        "train_views": train_metrics,
        "zoom_2x": zoom_metrics,
        "base_camera": lod_camera_diagnostics(base_camera, lod_train),
        "zoom_camera": lod_camera_diagnostics(zoom_camera, lod_zoom),
        "zoom_fx_ratio": float(lod_zoom.fx / lod_train.fx),
        "zoom_width_ratio": float(lod_zoom.width / lod_train.width),
        "view_name": str(base_camera.image_name),
        "view_index_requested": int(args.view_index),
        "cameras_json_note": "stage1 cameras.json lists test cameras before train cameras; ROI identity is view_name.",
    }
    if abs(payload["zoom_width_ratio"] - 1.0) > 1e-9:
        payload["accepted"] = False
        payload["error"] = "2x zoom changed raster size; LoD scale must not be inferred from image size."
    if abs(payload["zoom_fx_ratio"] - args.zoom_factor) > 0.02:
        payload["accepted"] = False
        payload["error"] = (
            f"2x focal ratio is {payload['zoom_fx_ratio']:.6g}, expected ~{args.zoom_factor}."
        )
    rgb_ok = (payload["train_view_mean"]["rgb_l1"] or 0.0) < 1e-6 and zoom_metrics["rgb_l1"] < 1e-6
    if not rgb_ok:
        payload["accepted"] = False
        payload.setdefault("error", "Imported L0 RGB does not match Skyfall renders.")
    _write_json(os.path.join(output_dir, "l0_consistency.json"), payload)
    print(json.dumps({k: payload[k] for k in ("accepted", "train_view_mean", "zoom_2x", "zoom_fx_ratio", "zoom_width_ratio", "filter_copied_max_abs")}, indent=2))
    if not payload["accepted"]:
        raise SystemExit(f"L0 import check failed: {payload.get('error')}")


if __name__ == "__main__":
    main()
