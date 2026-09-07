#!/usr/bin/env python3
"""
Progressive zoom-refine MVP for Skyfall-GS Stage 2.

Given an existing train/test camera view and ROI, keep camera pose fixed,
progressively increase focal length (zoom in), and at each level run:
  render -> FlowEdit refine -> appearance-only finetune.
"""

from __future__ import annotations

import argparse
import os
import random
from argparse import Namespace
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from submodules.FlowEdit.idu_refine import FlowEditRefineIDU, default_src_prompt, default_tar_prompt
from utils.general_utils import PILtoTorch, safe_state
from utils.loss_utils import l1_loss, ssim
from utils.zoom_camera import (
    NormalizedROI,
    camera_from_pil_image,
    make_zoom_camera,
    nearest_train_cameras,
    save_roi_overlay,
)
from utils.zoom_mvp_utils import (
    GeometrySnapshot,
    assert_post_train_checks,
    compute_post_train_metrics,
    embedding_for_train_camera,
    freeze_geometry_for_zoom,
    save_tensor_image,
    select_appearance_embedding,
    write_json,
)

try:
    from fused_ssim import fused_ssim

    USE_FUSED_SSIM = True
except ImportError:
    USE_FUSED_SSIM = False


def parse_zoom_factors(text: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("zoom_factors must contain at least one value.")
    if any(v <= 1.0 for v in values):
        raise ValueError("zoom_factors must all be > 1.0.")
    return values


def pick_base_camera(scene: Scene, view_index: int, use_test: bool):
    cameras = scene.getTestCameras() if use_test else scene.getTrainCameras()
    if view_index < 0 or view_index >= len(cameras):
        raise IndexError(
            f"view_index={view_index} out of range for {'test' if use_test else 'train'} cameras "
            f"(count={len(cameras)})."
        )
    return cameras[view_index], not use_test


@torch.no_grad()
def render_view(
    camera,
    gaussians: GaussianModel,
    pipe,
    background: torch.Tensor,
    kernel_size: float,
    appearance_embedding: torch.Tensor | None,
) -> torch.Tensor:
    return render(
        camera,
        gaussians,
        pipe,
        background,
        kernel_size=kernel_size,
        testing=False,
        appearance_embedding=appearance_embedding,
    )["render"]


def appearance_only_step(
    gaussians: GaussianModel,
    camera,
    pipe,
    background: torch.Tensor,
    kernel_size: float,
    appearance_embedding: torch.Tensor | None,
    lambda_dssim: float,
) -> torch.Tensor:
    render_pkg = render(
        camera,
        gaussians,
        pipe,
        background,
        kernel_size=kernel_size,
        testing=False,
        appearance_embedding=appearance_embedding,
    )
    image = render_pkg["render"]
    mask = camera.original_mask.cuda()
    gt_image = mask * camera.original_image.cuda()
    image = mask * image

    ll1 = l1_loss(image, gt_image)
    if USE_FUSED_SSIM:
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - lambda_dssim) * ll1 + lambda_dssim * (1.0 - ssim_value)
    else:
        ssim_value = ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - lambda_dssim) * ll1 + lambda_dssim * (1.0 - ssim_value)
    loss.backward()
    return loss.detach()


def train_appearance_only(
    gaussians: GaussianModel,
    zoom_train_cam,
    train_cameras: Sequence,
    pipe,
    background: torch.Tensor,
    kernel_size: float,
    zoom_embedding: torch.Tensor | None,
    num_steps: int,
    mix_ratio: float,
    lambda_dssim: float,
) -> None:
    train_stack = list(train_cameras)
    progress = tqdm(range(num_steps), desc="appearance-only")
    for step in progress:
        if random.random() >= mix_ratio:
            viewpoint = zoom_train_cam
            embedding = zoom_embedding
        else:
            viewpoint = random.choice(train_stack)
            embedding = embedding_for_train_camera(gaussians, viewpoint.uid)

        gaussians.optimizer.zero_grad(set_to_none=True)
        loss = appearance_only_step(
            gaussians,
            viewpoint,
            pipe,
            background,
            kernel_size,
            embedding,
            lambda_dssim,
        )
        gaussians.optimizer.step()
        if step % 20 == 0:
            progress.set_postfix(loss=f"{loss.item():.5f}")


def run_unsharp(render_before: torch.Tensor, save_dir: str) -> Image.Image:
    """FLUX-free stand-in refiner for validating the pipeline end-to-end.

    Produces a plausibly "more detailed" target without any generative model, so the
    camera / freeze / training / check plumbing can be exercised offline.
    """
    os.makedirs(save_dir, exist_ok=True)
    arr = (render_before.detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    pil = pil.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=2))
    pil = ImageEnhance.Contrast(pil).enhance(1.08)
    pil.save(os.path.join(save_dir, "00000.png"))
    return pil


def run_flowedit(
    render_before: torch.Tensor,
    src_prompt: str,
    tar_prompt: str,
    save_dir: str,
    model_type: str,
    flow_edit_n_min: int,
    flow_edit_n_max: int,
    flow_edit_n_avg: int,
    model_path: str | None,
) -> Image.Image:
    os.makedirs(save_dir, exist_ok=True)
    refine_pipe = FlowEditRefineIDU(
        save_path=save_dir, device="cuda:0", model_type=model_type, model_path=model_path
    )
    img_np = render_before.detach().cpu().permute(1, 2, 0).numpy()
    refined_list = refine_pipe.run(
        [img_np],
        src_prompt=src_prompt,
        tar_prompt=tar_prompt,
        n_min=flow_edit_n_min,
        n_max=flow_edit_n_max,
        n_avg=flow_edit_n_avg,
    )
    del refine_pipe
    torch.cuda.empty_cache()
    return refined_list[0]


def load_stage1_cfg(stage1_dir: str) -> Namespace:
    cfg_path = os.path.join(stage1_dir, "cfg_args")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Stage1 cfg_args not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return eval(f.read())


def apply_stage1_cfg_to_args(args: argparse.Namespace, stage1_cfg: Namespace) -> None:
    """Fill dataset-related fields from Stage1 cfg (MVP expects matching Stage1 settings)."""
    for key in (
        "source_path",
        "sh_degree",
        "appearance_enabled",
        "appearance_n_fourier_freqs",
        "appearance_embedding_dim",
        "images",
        "resolution",
        "white_background",
        "data_device",
        "kernel_size",
        "eval",
        "ray_jitter",
        "resample_gt_image",
        "load_allres",
        "sample_more_highres",
    ):
        if hasattr(stage1_cfg, key):
            setattr(args, key, getattr(stage1_cfg, key))


def refined_pil_to_tensor(refined_pil: Image.Image, width: int, height: int, device: torch.device) -> torch.Tensor:
    return PILtoTorch(refined_pil, (width, height)).clamp(0.0, 1.0).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Progressive zoom-refine MVP")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--start_checkpoint", type=str, required=True, help="Stage1 checkpoint (.pth), read-only")
    parser.add_argument("--output_dir", type=str, required=True, help="All MVP outputs go here, never Stage1 dir")
    parser.add_argument("--view_index", type=int, default=0)
    parser.add_argument("--use_test_view", action="store_true", help="Pick view from test cameras instead of train")
    parser.add_argument("--roi_center_x", type=float, required=True)
    parser.add_argument("--roi_center_y", type=float, required=True)
    parser.add_argument("--roi_width", type=float, required=True)
    parser.add_argument("--roi_height", type=float, required=True)
    parser.add_argument("--zoom_factors", type=str, default="2,4,8")
    parser.add_argument("--steps_per_level", type=int, default=500)
    parser.add_argument("--mix_ratio", type=float, default=0.2, help="Fraction of steps using original train cameras")
    parser.add_argument("--src_prompt", type=str, default=default_src_prompt)
    parser.add_argument("--tar_prompt", type=str, default=default_tar_prompt)
    parser.add_argument("--flow_edit_n_min", type=int, default=4)
    parser.add_argument("--flow_edit_n_max", type=int, default=10)
    parser.add_argument("--flow_edit_n_avg", type=int, default=1)
    parser.add_argument(
        "--refine_mode",
        type=str,
        default="flowedit",
        choices=["flowedit", "unsharp"],
        help="'unsharp' is a FLUX-free dry run for validating the pipeline.",
    )
    parser.add_argument(
        "--flux_model_path",
        type=str,
        default=None,
        help="Local diffusers checkpoint dir; falls back to the HuggingFace hub id.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    safe_state(args.quiet)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    zoom_factors = parse_zoom_factors(args.zoom_factors)
    roi = NormalizedROI(args.roi_center_x, args.roi_center_y, args.roi_width, args.roi_height)
    roi.validate_in_bounds()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    stage1_dir = os.path.dirname(os.path.abspath(args.start_checkpoint))
    if os.path.abspath(output_dir) == os.path.abspath(stage1_dir):
        raise ValueError("output_dir must differ from Stage1 checkpoint directory.")

    stage1_cfg = load_stage1_cfg(stage1_dir)
    apply_stage1_cfg_to_args(args, stage1_cfg)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    # Force MVP training settings
    dataset.model_path = output_dir
    opt.lambda_depth = 0.0
    opt.lambda_pseudo_depth = 0.0
    opt.lambda_opacity = 0.0

    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs,
        dataset.appearance_embedding_dim,
    )

    checkpoint_path = os.path.abspath(args.start_checkpoint)
    model_params, first_iter = torch.load(checkpoint_path, weights_only=False)

    scene = Scene(
        dataset,
        gaussians,
        load_iteration=first_iter,
        shuffle=False,
        ply_path=stage1_dir,
    )
    gaussians.load_from_checkpoints(model_params)

    train_cameras = scene.getTrainCameras()
    base_cam, is_train_view = pick_base_camera(scene, args.view_index, args.use_test_view)
    appearance_embedding = select_appearance_embedding(gaussians, base_cam.uid, is_train_view)

    opt.position_lr_max_steps = args.steps_per_level * len(zoom_factors)
    gaussians.training_setup(
        opt,
        num_train_cameras=len(train_cameras),
        from_scratch=False,
    )
    freeze_geometry_for_zoom(gaussians)
    geometry_snapshot = GeometrySnapshot.from_gaussians(gaussians)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    manifest = {
        "start_checkpoint": checkpoint_path,
        "output_dir": output_dir,
        "view_index": args.view_index,
        "use_test_view": args.use_test_view,
        "base_image_name": base_cam.image_name,
        "base_uid": int(base_cam.uid),
        "roi": {
            "center_x": roi.center_x,
            "center_y": roi.center_y,
            "width": roi.width,
            "height": roi.height,
        },
        "zoom_factors": zoom_factors,
        "steps_per_level": args.steps_per_level,
        "mix_ratio": args.mix_ratio,
        "levels": [],
    }

    save_roi_overlay(base_cam, roi, os.path.join(output_dir, "roi_overlay.png"))
    with open(os.path.join(output_dir, "cfg_args"), "w", encoding="utf-8") as f:
        f.write(str(Namespace(**vars(args))))
    write_json(os.path.join(output_dir, "manifest.json"), manifest)

    nearest_cams = nearest_train_cameras(base_cam, train_cameras, k=2)
    nearest_names = [cam.image_name for cam in nearest_cams]

    # Reference point for detecting whether zoom-refine corrupts untouched views.
    gaussians.compute_3D_filter(cameras=list(train_cameras))
    neighbor_baseline = {}
    for neighbor in nearest_cams:
        neighbor_render = render_view(
            neighbor, gaussians, pipe, background, dataset.kernel_size,
            embedding_for_train_camera(gaussians, neighbor.uid),
        )
        save_tensor_image(
            neighbor_render,
            os.path.join(output_dir, f"neighbor_baseline_{neighbor.image_name}.png"),
        )
        neighbor_baseline[neighbor.image_name] = float(
            torch.abs(neighbor_render - neighbor.original_image.cuda()).mean().item()
        )
    manifest["neighbor_baseline_l1"] = neighbor_baseline
    write_json(os.path.join(output_dir, "manifest.json"), manifest)
    print(f"Neighbor baseline L1: {neighbor_baseline}")

    global_step = 0
    for level_idx, zoom_factor in enumerate(zoom_factors):
        level_dir = os.path.join(output_dir, f"zoom_{zoom_factor:g}x")
        os.makedirs(level_dir, exist_ok=True)
        print(f"\n=== Zoom level {level_idx + 1}/{len(zoom_factors)}: {zoom_factor:g}x ===")

        roi.validate_for_zoom(zoom_factor)
        zoom_cam = make_zoom_camera(
            base_cam,
            roi,
            zoom_factor,
            uid=10000 + level_idx,
            image_name=f"{base_cam.image_name}_zoom{zoom_factor:g}",
        )

        gaussians.compute_3D_filter(cameras=list(train_cameras) + [zoom_cam])

        render_before = render_view(
            zoom_cam, gaussians, pipe, background, dataset.kernel_size, appearance_embedding
        )
        save_tensor_image(render_before, os.path.join(level_dir, "render_before.png"))

        if args.refine_mode == "unsharp":
            refined_pil = run_unsharp(render_before, save_dir=os.path.join(level_dir, "unsharp"))
        else:
            refined_pil = run_flowedit(
                render_before,
                args.src_prompt,
                args.tar_prompt,
                save_dir=os.path.join(level_dir, "flowedit"),
                model_type=args.idu_model_type,
                flow_edit_n_min=args.flow_edit_n_min,
                flow_edit_n_max=args.flow_edit_n_max,
                flow_edit_n_avg=args.flow_edit_n_avg,
                model_path=args.flux_model_path,
            )
        refined_pil.save(os.path.join(level_dir, "refined.png"))

        zoom_train_cam = camera_from_pil_image(
            zoom_cam,
            refined_pil,
            image_name=f"{zoom_cam.image_name}_train",
            uid=11000 + level_idx,
        )

        train_appearance_only(
            gaussians,
            zoom_train_cam,
            train_cameras,
            pipe,
            background,
            dataset.kernel_size,
            appearance_embedding,
            num_steps=args.steps_per_level,
            mix_ratio=args.mix_ratio,
            lambda_dssim=opt.lambda_dssim,
        )
        global_step += args.steps_per_level

        render_after = render_view(
            zoom_cam, gaussians, pipe, background, dataset.kernel_size, appearance_embedding
        )
        save_tensor_image(render_after, os.path.join(level_dir, "render_after.png"))

        refined_tensor = refined_pil_to_tensor(
            refined_pil, zoom_cam.image_width, zoom_cam.image_height, render_after.device
        )
        metrics = compute_post_train_metrics(render_before, render_after, refined_tensor)

        ckpt_path = os.path.join(level_dir, f"chkpnt_zoom{zoom_factor:g}.pth")
        torch.save((gaussians.capture(), global_step), ckpt_path)

        neighbor_metrics = {}
        for idx, neighbor in enumerate(nearest_cams):
            neighbor_embedding = embedding_for_train_camera(gaussians, neighbor.uid)
            neighbor_render = render_view(
                neighbor, gaussians, pipe, background, dataset.kernel_size, neighbor_embedding
            )
            save_tensor_image(
                neighbor_render,
                os.path.join(level_dir, f"neighbor_{idx}_{neighbor.image_name}.png"),
            )
            neighbor_l1 = float(
                torch.abs(neighbor_render - neighbor.original_image.cuda()).mean().item()
            )
            neighbor_metrics[neighbor.image_name] = {"l1_to_gt": neighbor_l1}

        level_record = {
            "zoom_factor": zoom_factor,
            "checkpoint": ckpt_path,
            "metrics": metrics,
            "neighbor_cameras": nearest_names,
            "neighbor_metrics": neighbor_metrics,
        }
        manifest["levels"].append(level_record)
        write_json(os.path.join(output_dir, "manifest.json"), manifest)
        print(f"Level {zoom_factor:g}x done. metrics={metrics}")

        # Assert only after every artifact is durable, so a failure is still diagnosable.
        assert_post_train_checks(geometry_snapshot, gaussians, metrics)

    final_ckpt = os.path.join(output_dir, "chkpnt_final.pth")
    torch.save((gaussians.capture(), global_step), final_ckpt)
    manifest["final_checkpoint"] = final_ckpt
    write_json(os.path.join(output_dir, "manifest.json"), manifest)
    print(f"\nMVP complete. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
