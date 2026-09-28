#!/usr/bin/env python3
"""Smoke-test RaDe-GS against Skyfall's diff_gauss. Does not train LoD."""

from __future__ import annotations

import argparse
import json
import os

import torch

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render as render_skyfall
from lod.camera import lod_camera_diagnostics, skyfall_camera_to_lod
from lod.importer import import_skyfall_l0
from lod.path import DEFAULT_GZ_ROOT
from lod.rasterizer import render_lod, render_skyfall_rade, require_rade_gs
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


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def _spatial(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None or not torch.is_tensor(tensor):
        return None
    value = tensor.detach().float()
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        return None
    return value


def _pair(native: dict[str, torch.Tensor], rade: dict[str, torch.Tensor]) -> dict[str, float]:
    rgb_a, rgb_b = native["render"], rade["render"]
    metrics = {
        "rgb_l1": image_l1(rgb_a, rgb_b),
        "rgb_hf_l1": image_hf_l1(rgb_a, rgb_b),
        "rgb_max": float((rgb_a - rgb_b).abs().max().item()),
        "rade_image_mean": float(rgb_b.detach().float().mean().item()),
        "rade_image_max": float(rgb_b.detach().float().max().item()),
        "visible": int((rade["radii"] > 0).sum().item()),
        "finite": bool(torch.isfinite(rgb_b).all().item()),
    }
    alpha_a, alpha_b = _spatial(native.get("render_alpha")), _spatial(rade.get("render_alpha"))
    if alpha_a is not None and alpha_b is not None:
        metrics["alpha_l1"] = image_l1(alpha_a.unsqueeze(0), alpha_b.unsqueeze(0))
        metrics["alpha_max"] = float(alpha_b.max().item())
    depth_a, depth_b = _spatial(native.get("render_depth")), _spatial(rade.get("render_depth"))
    if depth_a is not None and depth_b is not None:
        metrics["depth_l1"] = image_l1(depth_a.unsqueeze(0), depth_b.unsqueeze(0))
        opaque = alpha_a > 0.05 if alpha_a is not None else None
        if opaque is not None and bool(opaque.any().item()):
            diff = (depth_a - depth_b).abs()[opaque]
            metrics["depth_rel_opaque"] = float((diff / depth_a[opaque].abs().clamp_min(1e-3)).mean().item())
    return metrics


def _lod_stats(name: str, pkg: dict[str, torch.Tensor]) -> dict[str, float]:
    image = pkg["image"].detach().float()
    alpha = pkg["alpha"].detach().float()
    return {
        "name": name,
        "image_mean": float(image.mean().item()),
        "image_max": float(image.max().item()),
        "alpha_mean": float(alpha.mean().item()),
        "visible": int((pkg["radii"] > 0).sum().item()),
        "finite": bool(torch.isfinite(image).all().item() and torch.isfinite(alpha).all().item()),
    }


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
    parser.add_argument("--step_scale", type=float, default=2.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    settings, rasterizer_cls = require_rade_gs()
    import diff_gauss  # Skyfall rasterizer must remain importable.

    safe_state(args.quiet)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    stage1_checkpoint = os.path.abspath(args.start_checkpoint)
    stage1_cfg = load_stage1_cfg(os.path.dirname(stage1_checkpoint))
    apply_stage1_cfg_to_args(args, stage1_cfg)
    extract_parser = argparse.ArgumentParser(add_help=False)
    dataset = ModelParams(extract_parser).extract(args)
    pipe = PipelineParams(extract_parser).extract(args)
    # The "native" reference render must stay on the original diff_gauss
    # kernel for the cross-rasterizer comparison, regardless of the Stage1
    # run's persisted backend.
    pipe.rasterizer_backend = "diff_gauss"
    dataset.model_path = output_dir

    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs,
        dataset.appearance_embedding_dim,
    )
    model_params, first_iter = torch.load(stage1_checkpoint, weights_only=False)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=first_iter,
        shuffle=False,
        ply_path=os.path.dirname(stage1_checkpoint),
    )
    gaussians.load_from_checkpoints(model_params)
    train_cameras = scene.getTrainCameras()
    gaussians.compute_3D_filter(cameras=list(train_cameras))

    bundle = import_skyfall_l0(
        gaussians,
        train_cameras,
        gz_root=args.gz_root,
        step_scale=args.step_scale,
        freeze=True,
    )
    roi = NormalizedROI(args.roi_center_x, args.roi_center_y, args.roi_width, args.roi_height)
    base_camera, is_train_view = pick_base_camera(scene, args.view_index, False, image_name=args.view_name)
    zoom_camera = make_zoom_camera(
        base_camera,
        roi,
        args.zoom_factor,
        uid=10000,
        image_name=f"{base_camera.image_name}_zoom{args.zoom_factor:g}",
    )
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device=gaussians.get_xyz.device,
    )
    train_embedding = embedding_for_train_camera(gaussians, base_camera.uid)
    zoom_embedding = select_appearance_embedding(gaussians, base_camera.uid, is_train_view)

    with torch.no_grad():
        native_train = render_skyfall(
            base_camera, gaussians, pipe, background, dataset.kernel_size, appearance_embedding=train_embedding
        )
        rade_train = render_skyfall_rade(
            base_camera, gaussians, background=background, kernel_size=dataset.kernel_size, appearance_embedding=train_embedding
        )
        native_zoom = render_skyfall(
            zoom_camera, gaussians, pipe, background, dataset.kernel_size, appearance_embedding=zoom_embedding
        )
        rade_zoom = render_skyfall_rade(
            zoom_camera, gaussians, background=background, kernel_size=dataset.kernel_size, appearance_embedding=zoom_embedding
        )
        lod_train = skyfall_camera_to_lod(
            base_camera, gz_root=args.gz_root, device=gaussians.get_xyz.device, use_skyfall_center=True
        )
        lod_zoom = skyfall_camera_to_lod(
            zoom_camera, gz_root=args.gz_root, device=gaussians.get_xyz.device, use_skyfall_center=True
        )
        gz_train = render_lod(
            bundle.lod, lod_train, gz_root=args.gz_root, lod=False, kernel_size=float(dataset.kernel_size)
        )
        gz_zoom = render_lod(
            bundle.lod, lod_zoom, gz_root=args.gz_root, lod=False, kernel_size=float(dataset.kernel_size)
        )

    train_metrics = _pair(native_train, rade_train)
    zoom_metrics = _pair(native_zoom, rade_zoom)
    save_tensor_image(native_train["render"], os.path.join(output_dir, "native_train.png"))
    save_tensor_image(rade_train["render"], os.path.join(output_dir, "rade_train.png"))
    save_tensor_image((native_train["render"] - rade_train["render"]).abs() * 10.0, os.path.join(output_dir, "rade_train_absdiff_x10.png"))
    save_tensor_image(native_zoom["render"], os.path.join(output_dir, "native_zoom2x.png"))
    save_tensor_image(rade_zoom["render"], os.path.join(output_dir, "rade_zoom2x.png"))
    save_tensor_image((native_zoom["render"] - rade_zoom["render"]).abs() * 10.0, os.path.join(output_dir, "rade_zoom2x_absdiff_x10.png"))
    save_tensor_image(gz_train["image"].clamp(0, 1), os.path.join(output_dir, "gz_sh_train.png"))
    save_tensor_image(gz_zoom["image"].clamp(0, 1), os.path.join(output_dir, "gz_sh_zoom2x.png"))
    alpha_map = _spatial(rade_train.get("render_alpha"))
    if alpha_map is not None:
        save_gray_image(alpha_map, os.path.join(output_dir, "rade_train_alpha.png"))

    payload = {
        "accepted": True,
        "note": (
            "RaDe-GS is installed beside diff_gauss. Appearance RGB compares Skyfall "
            "view/proj + colors_precomp. GaussianZoom render is SH-only and is not an RGB match."
        ),
        "n_points": bundle.n_points,
        "settings_fields": list(settings._fields),
        "rasterizer": rasterizer_cls.__module__,
        "diff_gauss": getattr(diff_gauss, "__name__", "diff_gauss"),
        "appearance_vs_rade": {"train": train_metrics, "zoom_2x": zoom_metrics},
        "gz_sh_only": {"train": _lod_stats("train", gz_train), "zoom_2x": _lod_stats("zoom_2x", gz_zoom)},
        "base_camera": lod_camera_diagnostics(base_camera, lod_train),
        "zoom_camera": lod_camera_diagnostics(zoom_camera, lod_zoom),
    }
    if not train_metrics["finite"] or not zoom_metrics["finite"]:
        payload["accepted"] = False
        payload["error"] = "RaDe-GS produced non-finite appearance RGB."
    if train_metrics["visible"] <= 0 or zoom_metrics["visible"] <= 0:
        payload["accepted"] = False
        payload["error"] = "RaDe-GS rendered an empty view."
    rgb_ok = train_metrics["rgb_l1"] < 5e-3 and zoom_metrics["rgb_l1"] < 5e-3
    if not rgb_ok:
        payload["accepted"] = False
        payload["error"] = (
            f"RaDe-GS appearance RGB diverges from diff_gauss "
            f"(train L1={train_metrics['rgb_l1']:.4g}, zoom L1={zoom_metrics['rgb_l1']:.4g})."
        )
    _write_json(os.path.join(output_dir, "rade_smoke.json"), payload)
    print(
        json.dumps(
            {
                "accepted": payload["accepted"],
                "appearance_vs_rade": payload["appearance_vs_rade"],
                "gz_sh_only": payload["gz_sh_only"],
                "error": payload.get("error"),
            },
            indent=2,
        )
    )
    if not payload["accepted"]:
        raise SystemExit(f"RaDe-GS smoke failed: {payload.get('error')}")


if __name__ == "__main__":
    main()
