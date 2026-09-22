#!/usr/bin/env python3
"""Official JAX_068 LoD eval on locked test windows, RaDe-GS + frozen appearance.

Compares L0 / L0+L1 / L0+L1+L2 on the same rasterizer. Test images are not
used for training. Overlapping windows are reported per row, not as a success
rate. Primary scores are valid-region MAE/PSNR and eroded SSIM/LPIPS.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from arguments import ModelParams, PipelineParams
from lod.camera import camera_image_stem, find_named_camera
from lod.eval_protocol import (
    crop_chw,
    downsample_render_to_native,
    native_crop_box,
    protocol_payload,
    reconstruction_metrics,
)
from lod.importer import import_skyfall_l0, load_lod_onto_bundle
from lod.jax068 import EVAL_DIR, SCENE_DIR, STAGE1_CHECKPOINT, TEST_VIEWS
from lod.lineage import write_json
from lod.path import DEFAULT_GZ_ROOT
from lod.rasterizer import require_rade_gs
from lod.render import render_lod_appearance
from scene import GaussianModel, Scene
from train_zoom_gen import apply_stage1_cfg_to_args, load_stage1_cfg
from utils.general_utils import safe_state
from utils.zoom_camera import NormalizedROI, make_zoom_camera
from utils.zoom_mvp_utils import save_tensor_image


def _mask(camera) -> torch.Tensor | None:
    mask = getattr(camera, "original_mask", None)
    if mask is None or not torch.is_tensor(mask):
        return None
    value = mask.detach().float()
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    return value


def _lpips(*, spatial: bool = False):
    import lpips

    net = lpips.LPIPS(net="vgg", spatial=spatial)
    net.eval()
    for param in net.parameters():
        param.requires_grad_(False)
    return net


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    ModelParams(parser)
    PipelineParams(parser)
    parser.add_argument("--start_checkpoint", type=str, default=str(STAGE1_CHECKPOINT))
    parser.add_argument("--lod_checkpoint", type=str, default="")
    parser.add_argument("--max_level", type=int, default=0)
    parser.add_argument("--coverage", type=str, default=str(SCENE_DIR / "EVAL_WINDOWS.json"))
    parser.add_argument("--output_dir", type=str, default=str(EVAL_DIR / "rade_l0"))
    parser.add_argument("--tag", type=str, default="l0")
    parser.add_argument("--gz_root", type=str, default=str(DEFAULT_GZ_ROOT))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    require_rade_gs()
    safe_state(args.quiet)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    windows_payload = json.loads(open(os.path.abspath(args.coverage), encoding="utf-8").read())
    windows = windows_payload.get("windows") or windows_payload.get("eval_windows")
    if not windows:
        raise SystemExit(f"no locked eval windows in {args.coverage}")

    stage1_checkpoint = os.path.abspath(args.start_checkpoint)
    stage1_cfg = load_stage1_cfg(os.path.dirname(stage1_checkpoint))
    apply_stage1_cfg_to_args(args, stage1_cfg)
    extract = argparse.ArgumentParser(add_help=False)
    dataset = ModelParams(extract).extract(args)
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
    test_names = [camera_image_stem(cam.image_name) for cam in test]
    if tuple(test_names) != TEST_VIEWS:
        raise SystemExit(f"test={test_names!r} expected {list(TEST_VIEWS)!r}")

    bundle = import_skyfall_l0(
        gaussians, train, gz_root=args.gz_root, step_scale=2.0, freeze=True,
    )
    if args.lod_checkpoint:
        load_lod_onto_bundle(bundle, os.path.abspath(args.lod_checkpoint), device=str(gaussians.get_xyz.device))
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device=gaussians.get_xyz.device,
    )
    kernel = float(dataset.kernel_size)
    lpips_fn = _lpips().to(device=gaussians.get_xyz.device)
    lpips_spatial = _lpips(spatial=True).to(device=gaussians.get_xyz.device)
    max_level = None if int(args.max_level) < 0 else int(args.max_level)

    full_1x = []
    for camera in test:
        name = camera_image_stem(camera.image_name)
        embedding = bundle.appearance.embedding_for_uid(camera.uid, is_train_view=False)
        rgb = render_lod_appearance(
            bundle, camera, background=background, kernel_size=kernel,
            appearance_embedding=embedding, lod=True, max_level=max_level, compact=False,
        )["render"].clamp(0, 1)
        gt = camera.original_image.to(device=rgb.device).clamp(0, 1)
        mask = _mask(camera)
        folder = os.path.join(output_dir, "full_1x", name)
        os.makedirs(folder, exist_ok=True)
        save_tensor_image(rgb, os.path.join(folder, "render.png"))
        row = {
            "image_name": name,
            "split": "test",
            "scale": 1.0,
            "eval_at": "full_native_image",
            **reconstruction_metrics(rgb, gt, mask, lpips_fn=lpips_fn, lpips_spatial_fn=lpips_spatial),
        }
        full_1x.append(row)

    zoom_rows = []
    for spec in windows:
        if spec.get("split") != "test":
            raise SystemExit(f"eval window is not test: {spec}")
        name = spec["image_name"]
        _, camera = find_named_camera(test, name)
        roi = NormalizedROI(spec["center_x"], spec["center_y"], spec["width"], spec["height"])
        zoom = float(spec["zoom"])
        box = native_crop_box(int(camera.image_width), int(camera.image_height), roi, zoom)
        gt = camera.original_image.to(device=gaussians.get_xyz.device).clamp(0, 1)
        mask = _mask(camera)
        gt_crop = crop_chw(gt, box)
        mask_crop = crop_chw(mask.unsqueeze(0), box)[0] if mask is not None else None
        zoom_cam = make_zoom_camera(
            camera, roi, zoom, uid=int(camera.uid) + 40000,
            image_name=str(spec["window_id"]),
        )
        embedding = bundle.appearance.embedding_for_uid(camera.uid, is_train_view=False)
        zoom_rgb = render_lod_appearance(
            bundle, zoom_cam, background=background, kernel_size=kernel,
            appearance_embedding=embedding, lod=True, max_level=max_level, compact=False,
        )["render"].clamp(0, 1)
        native_h, native_w = box[3] - box[1], box[2] - box[0]
        zoom_native = downsample_render_to_native(zoom_rgb, native_h, native_w)
        metrics = reconstruction_metrics(
            zoom_native, gt_crop, mask_crop,
            lpips_fn=lpips_fn, lpips_spatial_fn=lpips_spatial,
        )
        folder = os.path.join(output_dir, "windows", spec["window_id"])
        os.makedirs(folder, exist_ok=True)
        save_tensor_image(gt_crop, os.path.join(folder, "gt_native_crop.png"))
        save_tensor_image(zoom_native, os.path.join(folder, "render_zoom_at_native.png"))
        zoom_rows.append({
            "window_id": spec["window_id"],
            "image_name": name,
            "roi_id": spec["roi_id"],
            "zoom": zoom,
            "box": list(box),
            **metrics,
        })

    payload = protocol_payload({
        "tag": args.tag,
        "rasterizer": "rade_gs",
        "appearance": "frozen_skyfall_mlp_and_embeddings",
        "max_level": max_level,
        "lod_checkpoint": os.path.abspath(args.lod_checkpoint) if args.lod_checkpoint else None,
        "overlap_windows_not_independent_samples": True,
        "full_1x_test": full_1x,
        "zoom_native_windows": zoom_rows,
        "n_gaussians_l0": int(bundle.layer0().xyz.shape[0]),
        "n_active_levels": len(bundle.lod.layers),
        "iteration": int(first_iter),
        "note": (
            "Official LoD-gain table. Skyfall diff_gauss is not mixed in. "
            "Primary l1/psnr/ssim/lpips are valid-region scores; zero_filled is a companion."
        ),
    })
    write_json(os.path.join(output_dir, "EVAL.json"), payload)
    print(json.dumps({
        "tag": args.tag,
        "rasterizer": "rade_gs",
        "full_1x": [
            {"image_name": row["image_name"], "l1": row["l1"], "psnr": row["psnr"],
             "ssim": row["ssim"], "lpips": row.get("lpips"), "valid_coverage": row["valid_coverage"]}
            for row in full_1x
        ],
        "n_windows": len(zoom_rows),
        "output": os.path.join(output_dir, "EVAL.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
