#!/usr/bin/env python3
"""Stage1 reconstruction vs GT for a named camera. Does not generate supervision."""

from __future__ import annotations

import argparse
import json
import os

import torch

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from lod.camera import camera_image_stem, lod_camera_diagnostics, resolve_scene_camera, skyfall_camera_to_lod
from lod.jax214 import ROI, VIEW_IMAGE
from lod.lineage import write_json
from lod.path import DEFAULT_GZ_ROOT
from scene import GaussianModel, Scene
from train_zoom_gen import apply_stage1_cfg_to_args, load_stage1_cfg
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.zoom_camera import NormalizedROI, make_zoom_camera, save_roi_overlay
from utils.zoom_mvp_utils import embedding_for_train_camera, image_l1, save_gray_image, save_tensor_image


def _mask(camera) -> torch.Tensor | None:
    mask = getattr(camera, "original_mask", None)
    if mask is None or not torch.is_tensor(mask):
        return None
    value = mask.detach().float()
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    return value


def _metrics(render_rgb: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None) -> dict[str, float]:
    rgb = render_rgb.clamp(0, 1)
    target = gt.clamp(0, 1)
    if mask is not None:
        keep = mask.to(device=rgb.device)
        if keep.ndim == 2:
            keep = keep.unsqueeze(0)
        rgb = rgb * keep
        target = target * keep
    return {
        "l1": image_l1(rgb, target),
        "psnr": float(psnr(rgb, target).mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    ModelParams(parser)
    PipelineParams(parser)
    parser.add_argument("--start_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gz_root", type=str, default=str(DEFAULT_GZ_ROOT))
    parser.add_argument("--view_index", type=int, default=0)
    parser.add_argument("--view_name", type=str, default=VIEW_IMAGE)
    parser.add_argument("--roi_center_x", type=float, default=ROI["center_x"])
    parser.add_argument("--roi_center_y", type=float, default=ROI["center_y"])
    parser.add_argument("--roi_width", type=float, default=ROI["width"])
    parser.add_argument("--roi_height", type=float, default=ROI["height"])
    parser.add_argument("--zoom_factor", type=float, default=2.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    safe_state(args.quiet)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    stage1_checkpoint = os.path.abspath(args.start_checkpoint)
    stage1_cfg = load_stage1_cfg(os.path.dirname(stage1_checkpoint))
    apply_stage1_cfg_to_args(args, stage1_cfg)
    extract = argparse.ArgumentParser(add_help=False)
    dataset = ModelParams(extract).extract(args)
    pipe = PipelineParams(extract).extract(args)
    dataset.model_path = output_dir

    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs,
        dataset.appearance_embedding_dim,
    )
    model_params, first_iter = torch.load(stage1_checkpoint, weights_only=False)
    scene = Scene(
        dataset, gaussians, load_iteration=first_iter, shuffle=False,
        ply_path=os.path.dirname(stage1_checkpoint),
    )
    gaussians.load_from_checkpoints(model_params)
    train = list(scene.getTrainCameras())
    test = list(scene.getTestCameras())
    gaussians.compute_3D_filter(cameras=train)

    train_names = [camera_image_stem(cam.image_name) for cam in train]
    test_names = [camera_image_stem(cam.image_name) for cam in test]
    cameras_json = os.path.join(os.path.dirname(stage1_checkpoint), "cameras.json")
    json_first = None
    if os.path.isfile(cameras_json):
        with open(cameras_json, encoding="utf-8") as handle:
            json_first = json.load(handle)[0].get("img_name")

    base, resolved_index = resolve_scene_camera(
        train, view_index=args.view_index, image_name=args.view_name, split="train",
    )
    roi = NormalizedROI(args.roi_center_x, args.roi_center_y, args.roi_width, args.roi_height)
    save_roi_overlay(base, roi, os.path.join(output_dir, "overlay_locked_name.png"))

    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device=gaussians.get_xyz.device,
    )

    def render_row(camera, tag: str) -> dict:
        embedding = embedding_for_train_camera(gaussians, camera.uid)
        is_test = tag.startswith("test_")
        pkg = render(
            camera, gaussians, pipe, background, kernel_size=dataset.kernel_size,
            testing=is_test,
            appearance_embedding=None if is_test else embedding,
        )
        rgb = pkg["render"].clamp(0, 1)
        gt = camera.original_image.to(device=rgb.device).clamp(0, 1)
        mask = _mask(camera)
        save_tensor_image(rgb, os.path.join(output_dir, f"{tag}_render.png"))
        save_tensor_image(gt, os.path.join(output_dir, f"{tag}_gt.png"))
        save_tensor_image((rgb - gt).abs() * 4.0, os.path.join(output_dir, f"{tag}_absdiff_x4.png"))
        depth = pkg.get("render_depth")
        if torch.is_tensor(depth):
            depth_map = depth.detach().float()
            if depth_map.ndim == 3:
                depth_map = depth_map[0]
            valid = depth_map[depth_map > 0]
            denom = float(valid.median().item()) if valid.numel() else 1.0
            save_gray_image((depth_map / max(denom, 1e-6)).clamp(0, 1), os.path.join(output_dir, f"{tag}_depth.png"))
        alpha = pkg.get("render_alpha")
        if torch.is_tensor(alpha):
            save_gray_image(alpha.detach().float().squeeze(), os.path.join(output_dir, f"{tag}_alpha.png"))
        row = {"image_name": camera.image_name, "uid": int(camera.uid), **_metrics(rgb, gt, mask)}
        if mask is not None:
            row["mask_mean"] = float(mask.mean().item())
        return row

    locked = render_row(base, "locked")
    zoom = make_zoom_camera(base, roi, args.zoom_factor, uid=10000, image_name=f"{base.image_name}_zoom{args.zoom_factor:g}")
    embedding = embedding_for_train_camera(gaussians, base.uid)
    zoom_pkg = render(
        zoom, gaussians, pipe, background, kernel_size=dataset.kernel_size,
        testing=False, appearance_embedding=embedding,
    )
    save_tensor_image(zoom_pkg["render"].clamp(0, 1), os.path.join(output_dir, "locked_2x_render.png"))
    extra = []
    for camera in train[1:3]:
        extra.append(render_row(camera, f"train_{camera_image_stem(camera.image_name)}"))
    test_rows = [render_row(camera, f"test_{camera_image_stem(camera.image_name)}") for camera in test[:2]]

    lod = skyfall_camera_to_lod(base, gz_root=args.gz_root)
    payload = {
        "accepted": True,
        "view_name": camera_image_stem(base.image_name),
        "resolved_train_index": resolved_index,
        "requested_view_index": int(args.view_index),
        "train_names": train_names,
        "test_names": test_names,
        "cameras_json_first": json_first,
        "cameras_json_is_not_train_order": json_first not in (None, train_names[0] if train_names else None),
        "locked": locked,
        "other_train": extra,
        "test": test_rows,
        "camera": lod_camera_diagnostics(base, lod),
        "roi": {"center_x": roi.center_x, "center_y": roi.center_y, "width": roi.width, "height": roi.height},
        "n_gaussians": int(gaussians.get_xyz.shape[0]),
        "iteration": int(first_iter),
        "note": "Identity is view_name. stage1 cameras.json concatenates test then train; do not use that id 0.",
    }
    if camera_image_stem(base.image_name) != camera_image_stem(args.view_name):
        payload["accepted"] = False
        payload["error"] = "resolved camera name does not match lock"
    write_json(os.path.join(output_dir, "stage1_baseline.json"), payload)
    print(json.dumps({
        "accepted": payload["accepted"],
        "view_name": payload["view_name"],
        "resolved_train_index": resolved_index,
        "locked": locked,
        "test": test_rows,
        "cameras_json_first": json_first,
        "cx_pixel": payload["camera"]["cx_pixel"],
        "n_gaussians": payload["n_gaussians"],
    }, indent=2))
    if not payload["accepted"]:
        raise SystemExit("Stage1 named-camera check failed")


if __name__ == "__main__":
    main()
