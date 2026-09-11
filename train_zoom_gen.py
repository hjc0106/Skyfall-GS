#!/usr/bin/env python3
"""Independent progressive zoom-generation entry point.

The entry point owns the camera curriculum, 3DGS optimization, checkpointing,
and artifact layout.  Image generation is intentionally delegated to a
``refinement.Refiner`` so FlowEdit, recorded images, and future SR/VLM
backends can be compared without changing the MVP baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from argparse import Namespace
from typing import Any, List, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from refinement.base import CachedRefiner
from refinement.camera_adapter import build_supervision_camera, validate_sr_scale
from refinement.dloral_backend import DLoRALBackend
from refinement.geometry_warp import (
    build_aligned_multiview_inputs,
    estimate_depth_scene_scale,
    estimate_spatial_target,
    expected_surface_depth,
    pixel_footprint,
    rank_cameras_by_spatial_overlap,
)
from refinement.multiview_sr_backend import MultiViewSRBackend
from refinement.sr_backend import (
    DEFAULT_SOURCE_PROMPT,
    DEFAULT_TARGET_PROMPT,
    FlowEditBackend,
    ReuseImageBackend,
    TextConditionalSRBackend,
    UnsharpBackend,
)
from refinement.types import CameraSnapshot, RefinementRequest, RenderBundle, to_jsonable
from refinement.vlm_prompt import FixedPromptProvider, JsonPromptProvider, PromptCache, PromptManager
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.zoom_camera import NormalizedROI, make_zoom_camera, nearest_train_cameras, save_roi_overlay
from utils.zoom_mvp_utils import (
    GeometrySnapshot,
    assert_post_train_checks,
    compute_post_train_metrics,
    crop_chw,
    embedding_for_train_camera,
    freeze_geometry_for_zoom,
    image_hf_l1,
    image_l1,
    save_tensor_image,
    select_appearance_embedding,
)

try:
    from fused_ssim import fused_ssim

    USE_FUSED_SSIM = True
except ImportError:
    USE_FUSED_SSIM = False


def parse_zoom_factors(text: str) -> List[float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("zoom_factors must contain at least one value.")
    if any(value <= 1.0 for value in values):
        raise ValueError("zoom_factors must all be > 1.0.")
    if any(current <= previous for previous, current in zip(values, values[1:])):
        raise ValueError("zoom_factors must be strictly increasing for a progressive course.")
    return values


ABSORPTION_CROPS = {
    "building": (0.04, 0.20, 0.44, 0.59),
    "vehicles": (0.24, 0.62, 0.78, 0.98),
    "trees": (0.34, 0.37, 0.63, 0.61),
}


def parse_eval_steps(text: str | None) -> List[int]:
    if not text:
        return []
    steps = sorted({int(part.strip()) for part in str(text).split(",") if part.strip()})
    if any(step < 0 for step in steps):
        raise ValueError("eval_steps must be non-negative.")
    return steps


def _view_means(views: dict[str, dict[str, float]]) -> dict[str, Any]:
    if not views:
        return {"count": 0, "l1_to_gt_mean": None, "hf_l1_to_gt_mean": None}
    l1_values = [float(item["l1_to_gt"]) for item in views.values()]
    hf_values = [float(item["hf_l1_to_gt"]) for item in views.values()]
    return {
        "count": len(l1_values),
        "l1_to_gt_mean": sum(l1_values) / len(l1_values),
        "hf_l1_to_gt_mean": sum(hf_values) / len(hf_values),
    }


def _fractional_crop(width: int, height: int, box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x0 = int(round(box[0] * width))
    y0 = int(round(box[1] * height))
    x1 = int(round(box[2] * width))
    y1 = int(round(box[3] * height))
    return max(0, x0), max(0, y0), min(width, max(x0 + 1, x1)), min(height, max(y0 + 1, y1))


def pick_base_camera(scene: Scene, view_index: int, use_test: bool, image_name: str | None = None):
    from lod.camera import resolve_scene_camera

    cameras = scene.getTestCameras() if use_test else scene.getTrainCameras()
    split = "test" if use_test else "train"
    name = str(image_name or "").strip() or None
    camera, _index = resolve_scene_camera(
        cameras, view_index=view_index, image_name=name, split=split,
    )
    return camera, not use_test


@torch.no_grad()
def render_bundle(
    camera,
    gaussians: GaussianModel,
    pipe,
    background: torch.Tensor,
    kernel_size: float,
    appearance_embedding: torch.Tensor | None,
) -> RenderBundle:
    package = render(
        camera,
        gaussians,
        pipe,
        background,
        kernel_size=kernel_size,
        testing=False,
        appearance_embedding=appearance_embedding,
    )
    return RenderBundle(
        rgb=package["render"],
        depth=package.get("render_depth"),
        alpha=package.get("render_alpha"),
        camera=camera,
        metadata={"image_width": int(camera.image_width), "image_height": int(camera.image_height)},
    )


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    # Match save_tensor_image's uint8 conversion so persisted render_input.png
    # and the in-memory refinement request produce the same cache digest.
    array = (
        image.detach().cpu().float().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0
    ).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def appearance_only_step(
    gaussians: GaussianModel,
    camera,
    pipe,
    background: torch.Tensor,
    kernel_size: float,
    appearance_embedding: torch.Tensor | None,
    lambda_dssim: float,
) -> torch.Tensor:
    package = render(
        camera,
        gaussians,
        pipe,
        background,
        kernel_size=kernel_size,
        testing=False,
        appearance_embedding=appearance_embedding,
    )
    image = package["render"]
    mask = camera.original_mask.to(device=image.device, dtype=image.dtype)
    gt_image = camera.original_image.to(device=image.device, dtype=image.dtype)
    image = mask * image
    gt_image = mask * gt_image

    ll1 = l1_loss(image, gt_image)
    if USE_FUSED_SSIM:
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
    else:
        ssim_value = ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
    loss = (1.0 - lambda_dssim) * ll1 + lambda_dssim * (1.0 - ssim_value)
    loss.backward()
    return loss.detach()


def train_appearance_only(
    gaussians: GaussianModel,
    zoom_train_camera,
    train_cameras: Sequence,
    pipe,
    background: torch.Tensor,
    kernel_size: float,
    zoom_embedding: torch.Tensor | None,
    num_steps: int,
    mix_ratio: float,
    lambda_dssim: float,
    *,
    rng: random.Random | None = None,
    eval_steps: Sequence[int] = (),
    eval_fn=None,
) -> dict[str, Any]:
    if not train_cameras and mix_ratio > 0.0:
        raise ValueError("mix_ratio > 0 requires at least one original train camera.")
    train_stack = list(train_cameras)
    sampler = rng if rng is not None else random
    eval_set = {int(step) for step in eval_steps}
    sequence: list[dict[str, Any]] = []
    if eval_fn is not None and 0 in eval_set:
        eval_fn(0)
    progress = tqdm(range(num_steps), desc="appearance-only")
    for step in progress:
        if sampler.random() >= mix_ratio:
            viewpoint = zoom_train_camera
            embedding = zoom_embedding
            kind = "zoom"
        else:
            viewpoint = sampler.choice(train_stack)
            embedding = embedding_for_train_camera(gaussians, viewpoint.uid)
            kind = "train"
        sequence.append(
            {
                "step": int(step),
                "kind": kind,
                "uid": int(getattr(viewpoint, "uid", -1)),
                "image_name": str(getattr(viewpoint, "image_name", "")),
            }
        )
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
        completed = int(step + 1)
        if step % 20 == 0:
            progress.set_postfix(loss=f"{loss.item():.5f}")
        if eval_fn is not None and completed in eval_set:
            eval_fn(completed)
    return {"view_sequence": sequence}


@torch.no_grad()
def evaluate_absorption_step(
    *,
    step: int,
    level_dir: str,
    gaussians: GaussianModel,
    pipe,
    background: torch.Tensor,
    kernel_size: float,
    supervision_camera,
    zoom_embedding: torch.Tensor | None,
    refined: torch.Tensor,
    neighbor_cameras: Sequence,
    train_cameras: Sequence,
    test_cameras: Sequence,
    render_before: torch.Tensor | None,
) -> dict[str, Any]:
    step_dir = os.path.join(level_dir, "steps", f"{int(step):04d}")
    os.makedirs(step_dir, exist_ok=True)
    target_bundle = render_bundle(
        supervision_camera, gaussians, pipe, background, kernel_size, zoom_embedding
    )
    save_tensor_image(target_bundle.rgb, os.path.join(step_dir, "target.png"))
    refined = refined.to(device=target_bundle.rgb.device, dtype=target_bundle.rgb.dtype)
    height, width = int(target_bundle.rgb.shape[-2]), int(target_bundle.rgb.shape[-1])
    crop_metrics = {}
    for name, frac in ABSORPTION_CROPS.items():
        box = _fractional_crop(width, height, frac)
        save_tensor_image(crop_chw(target_bundle.rgb, box), os.path.join(step_dir, f"crop_{name}.png"))
        save_tensor_image(crop_chw(refined, box), os.path.join(step_dir, f"crop_{name}_refined.png"))
        crop_metrics[name] = {
            "box": list(box),
            "l1_to_refined": image_l1(crop_chw(target_bundle.rgb, box), crop_chw(refined, box)),
            "hf_l1_to_refined": image_hf_l1(crop_chw(target_bundle.rgb, box), crop_chw(refined, box)),
        }
    payload: dict[str, Any] = {
        "step": int(step),
        "target": {
            "l1_to_refined": image_l1(target_bundle.rgb, refined),
            "hf_l1_to_refined": image_hf_l1(target_bundle.rgb, refined),
            "crops": crop_metrics,
        },
        "neighbors": {},
        "train_views": {},
        "heldout_views": {},
    }
    if render_before is not None:
        payload["target"]["l1_from_step0"] = image_l1(target_bundle.rgb, render_before.to(target_bundle.rgb.device))
        payload["target"]["hf_l1_from_step0"] = image_hf_l1(
            target_bundle.rgb, render_before.to(target_bundle.rgb.device)
        )
    for index, camera in enumerate(neighbor_cameras):
        bundle = render_bundle(
            camera,
            gaussians,
            pipe,
            background,
            kernel_size,
            embedding_for_train_camera(gaussians, camera.uid),
        )
        save_tensor_image(bundle.rgb, os.path.join(step_dir, f"neighbor_{index}_{_safe_filename(camera.image_name)}.png"))
        payload["neighbors"][camera.image_name] = {
            "l1_to_gt": image_l1(bundle.rgb, camera.original_image.to(bundle.rgb.device)),
            "hf_l1_to_gt": image_hf_l1(bundle.rgb, camera.original_image.to(bundle.rgb.device)),
        }
    for camera in train_cameras:
        bundle = render_bundle(
            camera,
            gaussians,
            pipe,
            background,
            kernel_size,
            embedding_for_train_camera(gaussians, camera.uid),
        )
        payload["train_views"][camera.image_name] = {
            "l1_to_gt": image_l1(bundle.rgb, camera.original_image.to(bundle.rgb.device)),
            "hf_l1_to_gt": image_hf_l1(bundle.rgb, camera.original_image.to(bundle.rgb.device)),
        }
    for index, camera in enumerate(test_cameras):
        bundle = render_bundle(
            camera,
            gaussians,
            pipe,
            background,
            kernel_size,
            select_appearance_embedding(gaussians, camera.uid, False),
        )
        save_tensor_image(bundle.rgb, os.path.join(step_dir, f"heldout_{index}_{_safe_filename(camera.image_name)}.png"))
        payload["heldout_views"][camera.image_name] = {
            "l1_to_gt": image_l1(bundle.rgb, camera.original_image.to(bundle.rgb.device)),
            "hf_l1_to_gt": image_hf_l1(bundle.rgb, camera.original_image.to(bundle.rgb.device)),
        }
    payload["neighbors_mean"] = _view_means(payload["neighbors"])
    payload["train_views_mean"] = _view_means(payload["train_views"])
    payload["heldout_views_mean"] = _view_means(payload["heldout_views"])
    write_json(os.path.join(step_dir, "metrics.json"), payload)
    return payload


def load_stage1_cfg(stage1_dir: str) -> Namespace:
    cfg_path = os.path.join(stage1_dir, "cfg_args")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Stage1 cfg_args not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        return eval(handle.read())


def apply_stage1_cfg_to_args(args: argparse.Namespace, stage1_cfg: Namespace) -> None:
    """Reuse dataset/model-loading fields from Stage1 without touching gen options."""

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


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)

def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in str(value))


def _save_scalar_preview(value: torch.Tensor, path: str, *, mode: str = "depth") -> None:
    tensor = value.detach().float()
    while tensor.ndim > 2:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"Scalar preview expects [H,W], got {tuple(tensor.shape)}")
    if mode == "alpha":
        normalized = tensor.clamp(0.0, 1.0)
    else:
        valid = torch.isfinite(tensor) & (tensor > 0.0)
        normalized = torch.zeros_like(tensor)
        if bool(valid.any()):
            values = tensor[valid]
            low = torch.quantile(values, 0.01)
            high = torch.quantile(values, 0.99)
            if float(high.item()) <= float(low.item()):
                high = low + 1.0
            normalized = ((tensor - low) / (high - low)).clamp(0.0, 1.0)
            normalized = torch.where(valid, normalized, torch.zeros_like(normalized))
    array = (normalized.cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def _save_pixel_flow(flow: torch.Tensor, path: str) -> None:
    array = flow.detach().cpu().float().numpy()
    np.save(path, array)
    magnitude = np.linalg.norm(np.nan_to_num(array, nan=0.0), axis=-1)
    valid = np.isfinite(array).all(axis=-1)
    preview = np.zeros_like(magnitude)
    if valid.any():
        values = magnitude[valid]
        high = float(np.quantile(values, 0.99)) if values.size else 1.0
        if high <= 0.0:
            high = 1.0
        preview = np.where(valid, np.clip(magnitude / high, 0.0, 1.0), 0.0)
    Image.fromarray((preview * 255.0).round().astype(np.uint8), mode="L").save(path.replace(".npy", "_mag.png"))


def _save_reprojection_overlay(
    target_image: torch.Tensor,
    warped_image: torch.Tensor,
    valid_mask: torch.Tensor,
    path: str,
) -> None:
    target = np.asarray(tensor_to_pil(target_image), dtype=np.float32)
    warped = np.asarray(tensor_to_pil(warped_image), dtype=np.float32)
    mask = valid_mask.detach().float().cpu().numpy() > 0.5
    if mask.shape != target.shape[:2]:
        mask = np.asarray(
            Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(
                (target.shape[1], target.shape[0]), Image.Resampling.NEAREST
            )
        ) > 127
    overlay = target.copy()
    overlay[mask] = 0.5 * target[mask] + 0.5 * warped[mask]
    Image.fromarray(overlay.clip(0, 255).round().astype(np.uint8), mode="RGB").save(path)


def run_multiview_geometry(
    target_bundle: RenderBundle,
    target_camera: Any,
    source_bundles: dict[int, RenderBundle],
    source_cameras: Sequence[Any],
    level_dir: str,
    *,
    neighbor_count: int,
    depth_abs_tolerance: float,
    depth_rel_tolerance: float,
    depth_scene_scale: float | None,
    alpha_threshold: float,
    min_depth: float,
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    """Render-independent geometry stage: rank, warp, mask, and persist diagnostics."""

    target_snapshot = CameraSnapshot.from_camera(target_camera).to_dict()
    if target_bundle.depth is None:
        payload = {
            "status": "missing_target_depth",
            "target_camera": target_snapshot,
            "aligned_neighbor_count": 0,
            "candidates": [],
            "selected": [],
        }
        return [], payload, {"status": payload["status"]}

    target_depth = expected_surface_depth(target_bundle.depth, target_bundle.alpha)
    target_alpha = target_bundle.alpha
    target_alpha_plane = target_alpha
    while target_alpha_plane is not None and target_alpha_plane.ndim > 2:
        target_alpha_plane = target_alpha_plane[0]
    target_valid = torch.isfinite(target_depth) & (target_depth > min_depth)
    if target_alpha_plane is not None:
        target_valid &= torch.isfinite(target_alpha_plane.float()) & (target_alpha_plane.float() > alpha_threshold)
    target_coverage = float(target_valid.float().mean().item()) if target_valid.numel() else 0.0
    _save_scalar_preview(target_depth, os.path.join(level_dir, "geometry_target_depth.png"))
    if target_alpha is not None:
        _save_scalar_preview(target_alpha, os.path.join(level_dir, "geometry_target_alpha.png"), mode="alpha")

    source_observations = []
    camera_by_uid: dict[int, Any] = {}
    for camera in source_cameras:
        uid = int(camera.uid)
        bundle = source_bundles.get(uid)
        camera_by_uid[uid] = camera
        if bundle is None or bundle.depth is None:
            source_observations.append((camera, None, None, None))
            continue
        source_depth = expected_surface_depth(bundle.depth, bundle.alpha)
        source_observations.append((camera, bundle.rgb, source_depth, bundle.alpha))

    aligned, records = build_aligned_multiview_inputs(
        target_camera,
        target_depth,
        target_alpha_plane,
        source_observations,
        k=neighbor_count,
        depth_abs_tolerance=depth_abs_tolerance,
        depth_rel_tolerance=depth_rel_tolerance,
        depth_scene_scale=depth_scene_scale,
        alpha_threshold=alpha_threshold,
        min_depth=min_depth,
    )
    for record in records:
        camera = camera_by_uid.get(int(record.get("uid", -1)))
        if camera is not None:
            record["camera"] = CameraSnapshot.from_camera(camera).to_dict()

    source_slot_by_uid = {int(camera.uid): index for index, camera in enumerate(source_cameras)}
    for index, neighbor in enumerate(aligned):
        neighbor_uid = int(getattr(neighbor.camera, "uid", -1))
        slot = source_slot_by_uid.get(neighbor_uid, index)
        name = _safe_filename(neighbor.name)
        tensor_to_pil(neighbor.image).save(os.path.join(level_dir, f"geometry_neighbor_{slot}_{name}_source.png"))
        tensor_to_pil(neighbor.warped_image).save(
            os.path.join(level_dir, f"geometry_neighbor_{slot}_{name}_reprojected.png")
        )
        mask_path = os.path.join(level_dir, f"geometry_neighbor_{slot}_{name}_valid_mask.png")
        _save_scalar_preview(neighbor.valid_mask, mask_path, mode="alpha")
        if neighbor.pixel_flow is not None:
            _save_pixel_flow(
                neighbor.pixel_flow,
                os.path.join(level_dir, f"geometry_neighbor_{slot}_{name}_pixel_flow.npy"),
            )
        if neighbor.source_to_target_flow is not None:
            _save_pixel_flow(
                neighbor.source_to_target_flow,
                os.path.join(level_dir, f"geometry_neighbor_{slot}_{name}_source_to_target_flow.npy"),
            )
        _save_reprojection_overlay(
            target_bundle.rgb,
            neighbor.warped_image,
            neighbor.valid_mask,
            os.path.join(level_dir, f"geometry_neighbor_{slot}_{name}_overlay.png"),
        )

    selected = [record for record in records if record.get("selected")]
    cache_context = {
        "schema_version": 1,
        "selection": "camera_distance_pool_then_depth_alpha_reprojection",
        "target_uid": int(target_camera.uid),
        "candidate_uids": [int(camera.uid) for camera in source_cameras],
        "selected": [
            {"uid": int(item["uid"]), "coverage": float(item["coverage"])}
            for item in selected
        ],
        "neighbor_count": int(neighbor_count),
        "depth_abs_tolerance": float(depth_abs_tolerance),
        "depth_rel_tolerance": float(depth_rel_tolerance),
        "depth_scene_scale": None if depth_scene_scale is None else float(depth_scene_scale),
        "alpha_threshold": float(alpha_threshold),
        "min_depth": float(min_depth),
    }
    payload = {
        "status": "ok",
        "selection": {
            "strategy": cache_context["selection"],
            "candidate_count": len(source_cameras),
            "requested_neighbor_count": int(neighbor_count),
        },
        "target": {
            "camera": target_snapshot,
            "size": [int(target_camera.image_width), int(target_camera.image_height)],
            "valid_coverage": target_coverage,
        },
        "candidates": records,
        "selected": selected,
        "aligned_neighbor_count": len(aligned),
        "cache_context": cache_context,
    }
    return aligned, payload, cache_context


def prepare_level_geometry(
    *,
    target_bundle: RenderBundle,
    target_camera: Any,
    base_camera: Any,
    train_cameras: Sequence[Any],
    gaussians: GaussianModel,
    pipe,
    background: torch.Tensor,
    kernel_size: float,
    zoom_factor: float,
    level_index: int,
    level_dir: str,
    neighbor_count: int,
    pool_size: int,
    depth_abs_tolerance: float,
    depth_rel_tolerance: float,
    depth_scene_scale: float,
    alpha_threshold: float,
    min_depth: float,
    min_target_confidence: float,
    min_in_frustum: float,
    min_reprojection_coverage: float,
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    """Estimate the 3D ROI, pick spatially corresponding neighbors, and warp RGB."""

    empty_cache = {
        "schema_version": 1,
        "selection": "camera_distance_pool_then_spatial_roi_then_depth_alpha_reprojection",
        "target_uid": int(target_camera.uid),
        "selected": [],
        "neighbor_count": int(neighbor_count),
    }
    if neighbor_count <= 0:
        payload = {"status": "skipped", "aligned_neighbor_count": 0, "selected": [], "candidates": []}
        return [], payload, {**empty_cache, "status": "skipped"}

    if target_bundle.depth is None:
        payload = {
            "status": "missing_target_depth",
            "aligned_neighbor_count": 0,
            "selected": [],
            "candidates": [],
        }
        return [], payload, {**empty_cache, "status": payload["status"]}

    target_depth = expected_surface_depth(target_bundle.depth, target_bundle.alpha)
    spatial_target = estimate_spatial_target(
        target_camera,
        target_depth,
        target_bundle.alpha,
        alpha_threshold=alpha_threshold,
        min_depth=min_depth,
    )
    _save_scalar_preview(target_depth, os.path.join(level_dir, "geometry_target_depth.png"))
    if target_bundle.alpha is not None:
        _save_scalar_preview(
            target_bundle.alpha, os.path.join(level_dir, "geometry_target_alpha.png"), mode="alpha"
        )

    if spatial_target.confidence < min_target_confidence:
        payload = {
            "status": "low_target_confidence",
            "spatial_target": spatial_target.to_dict(),
            "min_target_confidence": float(min_target_confidence),
            "aligned_neighbor_count": 0,
            "selected": [],
            "candidates": [],
            "fallback": "single_view",
        }
        return [], payload, {**empty_cache, "status": payload["status"], "fallback": "single_view"}

    if depth_scene_scale < 0.0:
        resolved_scene_scale = estimate_depth_scene_scale(target_camera, spatial_target.median_depth)
    elif depth_scene_scale == 0.0:
        resolved_scene_scale = None
    else:
        resolved_scene_scale = float(depth_scene_scale)
    empty_cache["depth_scene_scale"] = resolved_scene_scale
    empty_cache["pixel_footprint"] = pixel_footprint(target_camera, spatial_target.median_depth)

    selected, candidate_records = rank_cameras_by_spatial_overlap(
        spatial_target,
        train_cameras,
        zoom_factor=zoom_factor,
        k=neighbor_count,
        pool_size=pool_size,
        target_camera=target_camera,
        exclude_uids=(int(base_camera.uid), int(target_camera.uid)),
        min_depth=min_depth,
        min_in_frustum=min_in_frustum,
    )
    if not selected:
        payload = {
            "status": "no_spatial_neighbors",
            "spatial_target": spatial_target.to_dict(),
            "candidates": candidate_records,
            "aligned_neighbor_count": 0,
            "selected": [],
            "fallback": "single_view",
        }
        return [], payload, {**empty_cache, "status": payload["status"], "fallback": "single_view"}

    neighbor_zooms = []
    source_bundles: dict[int, RenderBundle] = {}
    projected_by_uid: dict[int, Any] = {}
    for index, (base_neighbor, projected) in enumerate(selected):
        roi = NormalizedROI(
            projected.center_x, projected.center_y, projected.width, projected.height
        )
        roi.validate_for_zoom(zoom_factor)
        neighbor_zoom = make_zoom_camera(
            base_neighbor,
            roi,
            zoom_factor,
            uid=12000 + level_index * 100 + index,
            image_name=f"{base_neighbor.image_name}_zoom{zoom_factor:g}_spatial",
        )
        neighbor_zooms.append(neighbor_zoom)
        projected_by_uid[int(neighbor_zoom.uid)] = projected
        save_roi_overlay(
            base_neighbor,
            roi,
            os.path.join(
                level_dir,
                f"geometry_neighbor_{index}_{_safe_filename(base_neighbor.image_name)}_spatial_roi.png",
            ),
        )

    gaussians.compute_3D_filter(cameras=list(train_cameras) + [target_camera] + neighbor_zooms)
    for neighbor_zoom, (base_neighbor, _) in zip(neighbor_zooms, selected):
        source_bundles[int(neighbor_zoom.uid)] = render_bundle(
            neighbor_zoom,
            gaussians,
            pipe,
            background,
            kernel_size,
            embedding_for_train_camera(gaussians, base_neighbor.uid),
        )

    aligned, warp_payload, warp_cache = run_multiview_geometry(
        target_bundle,
        target_camera,
        source_bundles,
        neighbor_zooms,
        level_dir,
        neighbor_count=neighbor_count,
        depth_abs_tolerance=depth_abs_tolerance,
        depth_rel_tolerance=depth_rel_tolerance,
        depth_scene_scale=resolved_scene_scale,
        alpha_threshold=alpha_threshold,
        min_depth=min_depth,
    )
    for record in warp_payload.get("candidates", []):
        projected = projected_by_uid.get(int(record.get("uid", -1)))
        if projected is not None:
            record["spatial_roi"] = projected.to_dict()
            record["base_uid"] = int(projected.uid)

    best_coverage = max(
        (float(item.get("coverage", 0.0)) for item in warp_payload.get("selected", [])),
        default=0.0,
    )
    fallback = None
    if not aligned or best_coverage < float(min_reprojection_coverage):
        fallback = "single_view"
    payload = {
        "status": "ok" if fallback is None else "low_reprojection_coverage",
        "spatial_target": spatial_target.to_dict(),
        "selection": {
            "strategy": empty_cache["selection"],
            "pool_size": int(pool_size),
            "requested_neighbor_count": int(neighbor_count),
            "spatial_candidates": candidate_records,
        },
        "warp": warp_payload,
        "aligned_neighbor_count": len(aligned),
        "best_coverage": best_coverage,
        "depth_scene_scale": resolved_scene_scale,
        "pixel_footprint": empty_cache["pixel_footprint"],
        "selected": warp_payload.get("selected", []),
        "fallback": fallback,
    }
    cache_context = {
        **warp_cache,
        "selection": empty_cache["selection"],
        "status": payload["status"],
        "spatial_target": spatial_target.to_dict(),
        "spatial_rois": [projected.to_dict() for _, projected in selected],
        "fallback": fallback,
    }
    return aligned, payload, cache_context


def write_cfg_args(path: str, args: argparse.Namespace) -> None:
    """Keep the repository's conventional Python ``cfg_args`` format."""

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(str(Namespace(**vars(args))))
        handle.write("\n")


def build_prompt_manager(args: argparse.Namespace, prompt_cache_dir: str) -> PromptManager:
    if args.vlm_model_path:
        from refinement.qwen3_vlm import Qwen3PromptProvider
        provider = Qwen3PromptProvider(args.vlm_model_path, python=args.vlm_python,
            device=args.vlm_device, max_new_tokens=args.vlm_max_new_tokens,
            max_image_size=args.vlm_max_image_size)
    elif args.prompt_json:
        provider = JsonPromptProvider(
            args.prompt_json,
            fallback_source_prompt=args.src_prompt,
            fallback_target_prompt=args.tar_prompt,
        )
    else:
        provider = FixedPromptProvider(
            args.src_prompt,
            args.tar_prompt,
            shared_region_description=args.shared_region_description,
            visible_features=tuple(args.visible_feature),
            preserve_structure=tuple(args.preserve_structure),
            uncertain_information=tuple(args.uncertain_information),
            config={"source": "command_line"},
        )
    return PromptManager(provider, PromptCache(prompt_cache_dir))


def build_backend(args: argparse.Namespace, backend_name: str):
    if backend_name == "reuse":
        if not args.refined_image:
            raise ValueError("--refined_image is required when --refine_backend reuse is selected.")
        return ReuseImageBackend(args.refined_image)
    if backend_name == "unsharp":
        return UnsharpBackend()
    if backend_name == "flowedit":
        return FlowEditBackend(
            model_type=args.idu_model_type,
            model_path=args.flux_model_path or None,
            device=args.refine_device,
            n_min=args.flow_edit_n_min,
            n_max=args.flow_edit_n_max,
            n_avg=args.flow_edit_n_avg,
        )
    if backend_name in ("sr", "sr_fixed", "sr_vlm"):
        return TextConditionalSRBackend(
            backend_name=backend_name,
            model_id=args.sr_model_id,
            model_path=args.sr_model_path or None,
            device=args.sr_device,
            dtype=args.sr_dtype,
            num_inference_steps=args.sr_num_inference_steps,
            guidance_scale=args.sr_guidance_scale,
            local_files_only=args.sr_local_files_only,
            model_scale=args.sr_model_scale,
            input_max_size=args.sr_input_max_size,
        )
    if backend_name == "dloral":
        if not args.dloral_sd_path or not args.dloral_ckpt or not args.dloral_spynet:
            raise ValueError("--dloral_sd_path, --dloral_ckpt and --dloral_spynet are required for the dloral backend.")
        return DLoRALBackend(
            repo_root=args.dloral_root or None,
            sd_path=args.dloral_sd_path,
            ckpt_path=args.dloral_ckpt,
            spynet_path=args.dloral_spynet,
            python=args.dloral_python,
            device=args.dloral_device,
            stages=args.dloral_stages,
            process_size=args.dloral_process_size,
            upscale=args.dloral_upscale,
            align_method=args.dloral_align_method,
            alignment=args.dloral_alignment,
            vae_encoder_tiled_size=args.dloral_vae_encoder_tiled_size,
            latent_tiled_size=args.dloral_latent_tiled_size,
            latent_tiled_overlap=args.dloral_latent_tiled_overlap,
        )
    raise ValueError(f"Unsupported refine backend: {backend_name}")


def backend_config(args: argparse.Namespace, backend_name: str) -> dict[str, Any]:
    config: dict[str, Any] = {"backend": backend_name}
    if backend_name == "flowedit":
        config.update(
            {
                "model_type": args.idu_model_type,
                "model_path": args.flux_model_path or None,
                "device": args.refine_device,
                "n_min": args.flow_edit_n_min,
                "n_max": args.flow_edit_n_max,
                "n_avg": args.flow_edit_n_avg,
            }
        )
    elif backend_name == "reuse":
        config["source_path"] = os.path.abspath(args.refined_image)
    elif backend_name in ("sr", "sr_fixed", "sr_vlm"):
        config.update(
            {
                "model_id": args.sr_model_id,
                "model_path": args.sr_model_path or None,
                "device": args.sr_device,
                "dtype": args.sr_dtype,
                "num_inference_steps": args.sr_num_inference_steps,
                "guidance_scale": args.sr_guidance_scale,
                "local_files_only": args.sr_local_files_only,
                "model_scale": args.sr_model_scale,
                "input_max_size": args.sr_input_max_size,
            }
        )
    elif backend_name == "dloral":
        config.update(
            {
                "repo_root": args.dloral_root or None,
                "sd_path": args.dloral_sd_path,
                "ckpt_path": args.dloral_ckpt,
                "spynet_path": args.dloral_spynet,
                "python": args.dloral_python,
                "device": args.dloral_device,
                "stages": args.dloral_stages,
                "process_size": args.dloral_process_size,
                "upscale": args.dloral_upscale,
                "align_method": args.dloral_align_method,
                "alignment": args.dloral_alignment,
                "latent_tiled_size": args.dloral_latent_tiled_size,
                "vae_encoder_tiled_size": args.dloral_vae_encoder_tiled_size,
                "propagation": (
                    "geometry_external_flows"
                    if args.dloral_alignment == "geometry"
                    else "target_feature_fallback"
                    if args.dloral_alignment == "target_only"
                    else "native_spynet"
                ),
                "frame_order": ["neighbor", "target"],
                "output_frame": "target",
            }
        )
    return config


def _roi_dict(roi: NormalizedROI) -> dict[str, float]:
    return {
        "center_x": roi.center_x,
        "center_y": roi.center_y,
        "width": roi.width,
        "height": roi.height,
    }


def build_run_config(
    args: argparse.Namespace,
    checkpoint_path: str,
    base_camera,
    roi: NormalizedROI,
    backend_name: str,
    prompt_manager: PromptManager,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "start_checkpoint": os.path.abspath(checkpoint_path),
        "base_camera": CameraSnapshot.from_camera(base_camera).to_dict(),
        "roi": _roi_dict(roi),
        "zoom_factors": list(args.zoom_factors_parsed),
        "sr_scale": float(args.sr_scale),
        "supervision_mode": args.supervision_mode,
        "steps_per_level": int(args.steps_per_level),
        "mix_ratio": float(args.mix_ratio),
        "eval_steps": list(getattr(args, "eval_steps_parsed", [])),
        "seed": int(args.seed),
        "skip_geometry": bool(args.skip_geometry),
        "absorption_crops": {name: list(box) for name, box in ABSORPTION_CROPS.items()},
        "refine_backend": backend_name,
        "backend_config": backend_config(args, backend_name),
        "prompt_provider": to_jsonable(dict(prompt_manager.provider.cache_config())),
        "prompt_json": os.path.abspath(args.prompt_json) if args.prompt_json else None,
        "vlm_config": {
            "model_path": os.path.abspath(args.vlm_model_path) if args.vlm_model_path else None,
            "python": os.path.abspath(args.vlm_python) if args.vlm_model_path else None,
            "device": args.vlm_device if args.vlm_model_path else None,
            "max_new_tokens": int(args.vlm_max_new_tokens) if args.vlm_model_path else None,
            "max_image_size": int(args.vlm_max_image_size) if args.vlm_model_path else None,
        },
        "geometry": {
            "neighbor_count": int(args.geometry_neighbor_count),
            "pool_size": int(args.geometry_pool_size),
            "multiview_refine": bool(args.multiview_refine),
            "min_target_confidence": float(args.min_target_confidence),
            "min_reprojection_coverage": float(args.min_reprojection_coverage),
            "min_in_frustum": float(args.min_in_frustum),
            "depth_abs_tolerance": float(args.depth_abs_tolerance),
            "depth_rel_tolerance": float(args.depth_rel_tolerance),
            "depth_scene_scale": float(args.depth_scene_scale),
            "alpha_threshold": float(args.alpha_threshold),
            "min_depth": float(args.min_depth),
        },
    }


def load_manifest(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must contain an object: {path}")
    payload.setdefault("levels", [])
    return payload


def latest_resume_level(
    manifest: dict[str, Any], expected_zoom_factors: Sequence[float] | None = None
) -> dict[str, Any] | None:
    completed_by_factor: dict[float, dict[str, Any]] = {}
    for level in manifest.get("levels", []):
        checkpoint = level.get("checkpoint")
        if checkpoint and os.path.isfile(checkpoint):
            completed_by_factor[float(level["zoom_factor"])] = level

    if expected_zoom_factors is not None:
        latest = None
        missing_seen = False
        for factor in expected_zoom_factors:
            level = completed_by_factor.get(float(factor))
            if level is None:
                missing_seen = True
            elif missing_seen:
                raise ValueError(
                    "Resume checkpoints must form a contiguous zoom prefix; "
                    f"found {factor:g}x after a missing scale."
                )
            else:
                latest = level
        return latest

    if not completed_by_factor:
        return None
    return max(completed_by_factor.values(), key=lambda item: int(item.get("level_index", -1)))

def restore_from_level_checkpoint(gaussians: GaussianModel, opt, level: dict[str, Any]) -> int:
    checkpoint_path = os.path.abspath(level["checkpoint"])
    model_params, global_step = torch.load(checkpoint_path, weights_only=False)
    gaussians.restore(model_params, opt, iterative_datasets_update=False)
    freeze_geometry_for_zoom(gaussians)
    return int(global_step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent progressive zoom-generation pipeline")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--start_checkpoint", type=str, required=True, help="Stage1 checkpoint (.pth), read-only")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for this generation run")
    parser.add_argument("--view_index", type=int, default=0)
    parser.add_argument("--view_name", type=str, default="", help="Lock the base camera by image stem. Preferred over view_index.")
    parser.add_argument("--use_test_view", action="store_true")
    parser.add_argument("--roi_center_x", type=float, required=True)
    parser.add_argument("--roi_center_y", type=float, required=True)
    parser.add_argument("--roi_width", type=float, required=True)
    parser.add_argument("--roi_height", type=float, required=True)
    parser.add_argument("--zoom_factors", type=str, default="2,4,8", help="Focal-length multipliers, e.g. 2,4,8")
    parser.add_argument("--sr_scale", type=float, default=1.0, help="Refiner output pixel multiplier; independent of zoom_factors")
    parser.add_argument(
        "--supervision_mode",
        choices=["original", "highres"],
        default="original",
        help="Resize refined output to zoom size or retain SR pixels for supervision",
    )
    parser.add_argument("--steps_per_level", type=int, default=500)
    parser.add_argument("--mix_ratio", type=float, default=0.2, help="Fraction of SH-only steps using original train cameras")
    parser.add_argument(
        "--eval_steps",
        type=str,
        default="",
        help="Comma-separated SH-only step indices to snapshot, e.g. 0,50,100,250,500",
    )

    parser.add_argument(
        "--refine_backend",
        choices=["flowedit", "reuse", "unsharp", "sr", "sr_fixed", "sr_vlm", "dloral"],
        default="flowedit",
        help="Image generation backend; 'sr' uses the optional text-conditioned diffusers upscaler",
    )
    parser.add_argument(
        "--refine_mode",
        choices=["flowedit", "unsharp"],
        default=None,
        help="Compatibility alias for selecting flowedit or the offline unsharp backend",
    )
    parser.add_argument("--refined_image", type=str, default=None, help="Existing image or run directory for reuse")
    parser.add_argument("--refine_device", type=str, default="cuda:0")
    parser.add_argument("--src_prompt", type=str, default=DEFAULT_SOURCE_PROMPT)
    parser.add_argument("--tar_prompt", type=str, default=DEFAULT_TARGET_PROMPT)
    parser.add_argument("--prompt_json", type=str, default=None, help="Recorded structured VLM output")
    parser.add_argument("--vlm_model_path", default=None, help="Local Qwen3-VL-4B-Instruct weights")
    parser.add_argument("--vlm_python", default=sys.executable, help="Python environment supporting Qwen3-VL")
    parser.add_argument("--vlm_device", default="cuda:0")
    parser.add_argument("--vlm_max_new_tokens", type=int, default=768)
    parser.add_argument("--vlm_max_image_size", type=int, default=1024)
    parser.add_argument("--prompt_cache_dir", type=str, default=None)
    parser.add_argument("--shared_region_description", type=str, default="")
    parser.add_argument("--visible_feature", action="append", default=[])
    parser.add_argument("--preserve_structure", action="append", default=[])
    parser.add_argument("--uncertain_information", action="append", default=[])
    parser.add_argument("--flow_edit_n_min", type=int, default=4)
    parser.add_argument("--flow_edit_n_max", type=int, default=10)
    parser.add_argument("--flow_edit_n_avg", type=int, default=1)

    parser.add_argument("--sr_model_id", type=str, default="stabilityai/stable-diffusion-x4-upscaler")
    parser.add_argument("--sr_model_path", type=str, default=None)
    parser.add_argument("--sr_model_scale", type=float, default=4.0, help="Native pixel scale of the SR model")
    parser.add_argument("--sr_input_max_size", type=int, default=1024, help="Maximum long side sent to the SR model")
    parser.add_argument("--sr_device", type=str, default="cuda")
    parser.add_argument("--sr_dtype", type=str, default="float16")
    parser.add_argument("--sr_num_inference_steps", type=int, default=20)
    parser.add_argument("--sr_guidance_scale", type=float, default=9.0)
    parser.add_argument("--sr_local_files_only", action="store_true")

    parser.add_argument("--dloral_root", type=str, default=None, help="Official DLoRAL checkout (pinned in refinement/dloral_backend.py)")
    parser.add_argument("--dloral_python", type=str, default=sys.executable, help="Interpreter for the isolated DLoRAL environment")
    parser.add_argument("--dloral_sd_path", type=str, default=None, help="Local Stable Diffusion 2.1 base directory")
    parser.add_argument("--dloral_ckpt", type=str, default=None, help="Local DLoRAL model.pkl")
    parser.add_argument("--dloral_spynet", type=str, default=None, help="Local SpyNet checkpoint; URLs are rejected")
    parser.add_argument("--dloral_device", type=str, default="cuda:0")
    parser.add_argument("--dloral_stages", type=int, default=1)
    parser.add_argument("--dloral_process_size", type=int, default=512)
    parser.add_argument(
        "--dloral_upscale",
        type=int,
        default=1,
        help="Official video LQ uses 4; GS renders are already at the zoom size so default is 1",
    )
    parser.add_argument("--dloral_align_method", type=str, default="adain", choices=["adain", "wavelet", "nofix"])
    parser.add_argument(
        "--dloral_alignment",
        type=str,
        default="spynet",
        choices=["spynet", "geometry", "target_only"],
        help="spynet: official optical flow. geometry: depth-driven warping. target_only: disable neighbor contribution.",
    )
    parser.add_argument("--dloral_vae_encoder_tiled_size", type=int, default=4096)
    parser.add_argument(
        "--dloral_latent_tiled_size",
        type=int,
        default=96,
        help="Official default 96. 256 keeps 2048 inputs on the full-image path but OOM on 48GB (CFR attention).",
    )
    parser.add_argument("--dloral_latent_tiled_overlap", type=int, default=32)

    parser.add_argument("--cache_dir", type=str, default=None, help="Generation cache root")
    parser.add_argument("--no_generation_cache", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from completed level checkpoints in output_dir")
    parser.add_argument("--skip_post_train_check", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--geometry_neighbor_count",
        type=int,
        default=2,
        help="Neighbor views to render for geometry alignment; 0 disables the geometry stage",
    )
    parser.add_argument(
        "--geometry_pool_size",
        type=int,
        default=8,
        help="Camera-center pool size before scoring spatial overlap",
    )
    parser.add_argument("--skip_geometry", action="store_true", help="Skip ROI spatial target and reprojection")
    parser.add_argument(
        "--multiview_refine",
        action="store_true",
        help="Fuse geometry-aligned neighbor RGB before the selected refine backend",
    )
    parser.add_argument("--min_target_confidence", type=float, default=0.05)
    parser.add_argument("--min_reprojection_coverage", type=float, default=0.05)
    parser.add_argument("--min_in_frustum", type=float, default=1e-3)
    parser.add_argument("--depth_abs_tolerance", type=float, default=5.0)
    parser.add_argument("--depth_rel_tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--depth_scene_scale",
        type=float,
        default=-1.0,
        help="Occlusion cap in scene units. <0 auto from GSD, 0 disables the cap, >0 is explicit.",
    )
    parser.add_argument("--alpha_threshold", type=float, default=1e-4)
    parser.add_argument("--min_depth", type=float, default=1e-4)

    args = parser.parse_args()
    if args.refine_mode is not None:
        args.refine_backend = args.refine_mode
    if args.refined_image and args.refine_backend in ("flowedit", "unsharp"):
        # Matches the MVP's --refined_image precedence while keeping an
        # explicit --refine_backend for model-based comparisons.
        args.refine_backend = "reuse"
    if args.prompt_json and args.vlm_model_path:
        raise ValueError("Choose --prompt_json or --vlm_model_path, not both.")
    if args.refine_backend == "sr_vlm" and not (args.prompt_json or args.vlm_model_path):
        raise ValueError("--prompt_json or --vlm_model_path is required for sr_vlm.")
    if args.refine_backend == "sr_fixed" and (args.prompt_json or args.vlm_model_path):
        raise ValueError("sr_fixed uses command-line prompts; omit --prompt_json/--vlm_model_path or choose sr_vlm.")
    args.zoom_factors_parsed = parse_zoom_factors(args.zoom_factors)
    args.sr_scale = validate_sr_scale(args.sr_scale)
    if args.sr_model_scale <= 0.0:
        raise ValueError("sr_model_scale must be positive.")
    if args.sr_input_max_size < 64:
        raise ValueError("sr_input_max_size must be at least 64.")
    if args.steps_per_level < 0:
        raise ValueError("steps_per_level must be non-negative.")
    args.eval_steps_parsed = parse_eval_steps(args.eval_steps)
    if args.eval_steps_parsed and max(args.eval_steps_parsed) > args.steps_per_level:
        raise ValueError("eval_steps cannot exceed steps_per_level.")
    if not 0.0 <= args.mix_ratio <= 1.0:
        raise ValueError("mix_ratio must lie in [0, 1].")
    if args.geometry_neighbor_count < 0:
        raise ValueError("geometry_neighbor_count must be non-negative.")
    if args.geometry_pool_size < 1:
        raise ValueError("geometry_pool_size must be at least 1.")
    if args.skip_geometry:
        args.geometry_neighbor_count = 0
    if args.multiview_refine and args.geometry_neighbor_count <= 0:
        raise ValueError("--multiview_refine requires geometry_neighbor_count > 0 (omit --skip_geometry).")
    if args.refine_backend == "dloral" and args.multiview_refine:
        raise ValueError("dloral already consumes a neighbor view; omit --multiview_refine (RGB fusion is a separate baseline).")
    if args.refine_backend == "dloral" and args.dloral_alignment == "geometry" and args.geometry_neighbor_count <= 0:
        raise ValueError("--dloral_alignment geometry requires geometry_neighbor_count > 0 (omit --skip_geometry).")
    for name in (
        "min_target_confidence",
        "min_reprojection_coverage",
        "min_in_frustum",
        "depth_abs_tolerance",
        "depth_rel_tolerance",
        "alpha_threshold",
        "min_depth",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"{name} must be non-negative.")
    return args


def main() -> None:
    args = parse_args()
    safe_state(args.quiet)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    roi = NormalizedROI(args.roi_center_x, args.roi_center_y, args.roi_width, args.roi_height)
    roi.validate_in_bounds()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if args.resume:
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Cannot resume without manifest.json: {manifest_path}")
        manifest = load_manifest(manifest_path)
    else:
        if os.path.isfile(manifest_path):
            raise FileExistsError(f"{manifest_path} already exists; pass --resume or choose a new output_dir.")
        if os.listdir(output_dir):
            raise FileExistsError(f"Output directory is not empty; pass --resume: {output_dir}")
        manifest = {"levels": []}

    stage1_checkpoint = os.path.abspath(args.start_checkpoint)
    stage1_dir = os.path.dirname(stage1_checkpoint)
    if output_dir == os.path.abspath(stage1_dir):
        raise ValueError("output_dir must differ from Stage1 checkpoint directory.")
    stage1_cfg = load_stage1_cfg(stage1_dir)
    apply_stage1_cfg_to_args(args, stage1_cfg)

    extract_parser = argparse.ArgumentParser(add_help=False)
    dataset = ModelParams(extract_parser).extract(args)
    opt = OptimizationParams(extract_parser).extract(args)
    pipe = PipelineParams(extract_parser).extract(args)

    dataset.model_path = output_dir
    opt.lambda_depth = 0.0
    opt.lambda_pseudo_depth = 0.0
    opt.lambda_opacity = 0.0
    opt.position_lr_max_steps = args.steps_per_level * len(args.zoom_factors_parsed)
    opt.idu_position_lr_max_steps = opt.position_lr_max_steps

    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs,
        dataset.appearance_embedding_dim,
    )
    model_params, first_iter = torch.load(stage1_checkpoint, weights_only=False)
    scene = Scene(dataset, gaussians, load_iteration=first_iter, shuffle=False, ply_path=stage1_dir)
    gaussians.load_from_checkpoints(model_params)

    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()
    base_camera, is_train_view = pick_base_camera(scene, args.view_index, args.use_test_view, image_name=args.view_name)
    appearance_embedding = select_appearance_embedding(gaussians, base_camera.uid, is_train_view)

    gaussians.training_setup(opt, num_train_cameras=len(train_cameras), from_scratch=False)
    freeze_geometry_for_zoom(gaussians)
    global_step = 0

    prompt_cache_dir = os.path.abspath(args.prompt_cache_dir or os.path.join(output_dir, "prompt_cache"))
    generation_cache_dir = os.path.abspath(args.cache_dir or os.path.join(output_dir, "generation_cache"))
    prompt_manager = build_prompt_manager(args, prompt_cache_dir)
    backend_name = args.refine_backend
    backend = build_backend(args, backend_name)
    if args.multiview_refine:
        backend = MultiViewSRBackend(
            backend,
            min_coverage=args.min_reprojection_coverage,
        )
    refiner = CachedRefiner(backend, generation_cache_dir, enabled=not args.no_generation_cache)

    run_config = build_run_config(args, stage1_checkpoint, base_camera, roi, backend_name, prompt_manager)
    if args.resume and manifest.get("config"):
        if json.dumps(manifest["config"], sort_keys=True) != json.dumps(to_jsonable(run_config), sort_keys=True):
            raise ValueError("Existing manifest configuration differs from the requested resume configuration.")
    manifest.update(
        {
            "schema_version": 1,
            "start_checkpoint": stage1_checkpoint,
            "output_dir": output_dir,
            "view_index": args.view_index,
            "use_test_view": args.use_test_view,
            "base_image_name": base_camera.image_name,
            "base_uid": int(base_camera.uid),
            "roi": _roi_dict(roi),
            "zoom_factors": list(args.zoom_factors_parsed),
            "sr_scale": float(args.sr_scale),
            "supervision_mode": args.supervision_mode,
            "refine_backend": backend_name,
            "multiview_refine": bool(args.multiview_refine),
            "geometry_neighbor_count": int(args.geometry_neighbor_count),
            "generation_cache_dir": generation_cache_dir,
            "prompt_cache_dir": prompt_cache_dir,
            "config": run_config,
        }
    )
    save_roi_overlay(base_camera, roi, os.path.join(output_dir, "roi_overlay.png"))
    write_cfg_args(os.path.join(output_dir, "cfg_args"), args)
    write_json(manifest_path, manifest)

    nearest_cameras = nearest_train_cameras(base_camera, train_cameras, k=2)
    nearest_names = [camera.image_name for camera in nearest_cameras]
    gaussians.compute_3D_filter(cameras=list(train_cameras) + [base_camera])
    if not manifest.get("neighbor_baseline_l1"):
        neighbor_baseline: dict[str, float] = {}
        for neighbor in nearest_cameras:
            neighbor_bundle = render_bundle(
                neighbor,
                gaussians,
                pipe,
                torch.tensor(
                    [1, 1, 1] if dataset.white_background else [0, 0, 0],
                    dtype=torch.float32,
                    device=gaussians.get_xyz.device,
                ),
                dataset.kernel_size,
                embedding_for_train_camera(gaussians, neighbor.uid),
            )
            save_tensor_image(neighbor_bundle.rgb, os.path.join(output_dir, f"neighbor_baseline_{neighbor.image_name}.png"))
            neighbor_baseline[neighbor.image_name] = float(
                torch.abs(neighbor_bundle.rgb - neighbor.original_image.to(neighbor_bundle.rgb.device)).mean().item()
            )
        manifest["neighbor_baseline_l1"] = neighbor_baseline
        write_json(manifest_path, manifest)

    resume_level = latest_resume_level(manifest, args.zoom_factors_parsed) if args.resume else None
    if resume_level is not None:
        global_step = restore_from_level_checkpoint(gaussians, opt, resume_level)
        print(f"Resumed from {resume_level['checkpoint']} at global_step={global_step}.")
    geometry_snapshot = GeometrySnapshot.from_gaussians(gaussians)

    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device=gaussians.get_xyz.device,
    )
    base_snapshot = CameraSnapshot.from_camera(base_camera)
    completed_factors = {
        float(level["zoom_factor"])
        for level in manifest.get("levels", [])
        if level.get("checkpoint") and os.path.isfile(level["checkpoint"])
    }

    for level_index, zoom_factor in enumerate(args.zoom_factors_parsed):
        if zoom_factor in completed_factors:
            print(f"Skipping completed zoom level {zoom_factor:g}x.")
            continue
        level_dir = os.path.join(output_dir, f"zoom_{zoom_factor:g}x")
        os.makedirs(level_dir, exist_ok=True)
        print(f"\n=== Generated zoom level {level_index + 1}/{len(args.zoom_factors_parsed)}: {zoom_factor:g}x ===")

        roi.validate_for_zoom(zoom_factor)
        zoom_camera = make_zoom_camera(
            base_camera,
            roi,
            zoom_factor,
            uid=10000 + level_index,
            image_name=f"{base_camera.image_name}_zoom{zoom_factor:g}",
        )

        gaussians.compute_3D_filter(cameras=list(train_cameras) + [zoom_camera])
        wide_bundle = render_bundle(
            base_camera,
            gaussians,
            pipe,
            background,
            dataset.kernel_size,
            appearance_embedding,
        )
        zoom_input_bundle = render_bundle(
            zoom_camera,
            gaussians,
            pipe,
            background,
            dataset.kernel_size,
            appearance_embedding,
        )
        save_tensor_image(wide_bundle.rgb, os.path.join(level_dir, "wide.png"))
        save_tensor_image(zoom_input_bundle.rgb, os.path.join(level_dir, "render_input.png"))

        aligned_neighbors: list[Any] = []
        geometry_payload: dict[str, Any] = {"status": "skipped", "aligned_neighbor_count": 0}
        geometry_cache_context: dict[str, Any] = {}
        if args.geometry_neighbor_count > 0:
            aligned_neighbors, geometry_payload, geometry_cache_context = prepare_level_geometry(
                target_bundle=zoom_input_bundle,
                target_camera=zoom_camera,
                base_camera=base_camera,
                train_cameras=train_cameras,
                gaussians=gaussians,
                pipe=pipe,
                background=background,
                kernel_size=dataset.kernel_size,
                zoom_factor=zoom_factor,
                level_index=level_index,
                level_dir=level_dir,
                neighbor_count=args.geometry_neighbor_count,
                pool_size=args.geometry_pool_size,
                depth_abs_tolerance=args.depth_abs_tolerance,
                depth_rel_tolerance=args.depth_rel_tolerance,
                depth_scene_scale=args.depth_scene_scale,
                alpha_threshold=args.alpha_threshold,
                min_depth=args.min_depth,
                min_target_confidence=args.min_target_confidence,
                min_in_frustum=args.min_in_frustum,
                min_reprojection_coverage=args.min_reprojection_coverage,
            )
        write_json(os.path.join(level_dir, "geometry.json"), geometry_payload)
        if geometry_payload.get("fallback") == "single_view":
            print(
                f"Geometry fallback to single-view refinement "
                f"({geometry_payload.get('status')}, aligned={geometry_payload.get('aligned_neighbor_count', 0)})."
            )

        if args.vlm_model_path:
            refiner.release_memory()
        prompt = prompt_manager.get_prompt(
            checkpoint=stage1_checkpoint,
            camera=base_snapshot,
            roi=_roi_dict(roi),
            zoom_factor=zoom_factor,
            level_index=level_index,
            wide_image=tensor_to_pil(wide_bundle.rgb),
            zoom_image=tensor_to_pil(zoom_input_bundle.rgb),
            context={
                "base_camera": base_snapshot.to_dict(),
                "zoom_camera": CameraSnapshot.from_camera(zoom_camera).to_dict(),
                "level_dir": level_dir,
            },
        )
        write_json(os.path.join(level_dir, "prompt.json"), prompt.to_dict())

        request = RefinementRequest(
            image=tensor_to_pil(zoom_input_bundle.rgb),
            checkpoint=stage1_checkpoint,
            camera=CameraSnapshot.from_camera(zoom_camera),
            zoom_factor=zoom_factor,
            sr_scale=args.sr_scale,
            prompt=prompt,
            model_config=backend_config(args, backend_name),
            prompt_config={
                "provider": prompt.provider,
                "prompt_json": os.path.abspath(args.prompt_json) if args.prompt_json else None,
                "cache_dir": prompt_cache_dir,
                "vlm_model_path": os.path.abspath(args.vlm_model_path) if args.vlm_model_path else None,
                "vlm_python": os.path.abspath(args.vlm_python) if args.vlm_model_path else None,
                "vlm_device": args.vlm_device if args.vlm_model_path else None,
                "vlm_max_new_tokens": args.vlm_max_new_tokens if args.vlm_model_path else None,
                "vlm_max_image_size": args.vlm_max_image_size if args.vlm_model_path else None,
            },
            cache_context=geometry_cache_context if (args.multiview_refine or args.refine_backend == "dloral") else {},
            level_index=level_index,
            metadata={
                "level_dir": level_dir,
                "backend_save_dir": os.path.join(level_dir, backend_name),
                "seed": args.seed + level_index,
                "neighbor_views": (
                    aligned_neighbors
                    if (
                        args.refine_backend == "dloral"
                        or (args.multiview_refine and geometry_payload.get("fallback") is None)
                    )
                    else []
                ),
            },
        )
        refinement_result = refiner.refine(request)
        refinement_result.image.save(os.path.join(level_dir, "refined.png"))

        supervision_camera = build_supervision_camera(
            zoom_camera,
            refinement_result.image,
            mode=args.supervision_mode,
            sr_scale=args.sr_scale,
            image_name=f"{zoom_camera.image_name}_train",
            uid=11000 + level_index,
        )
        gaussians.compute_3D_filter(cameras=list(train_cameras) + [supervision_camera])
        render_before_bundle = render_bundle(
            supervision_camera,
            gaussians,
            pipe,
            background,
            dataset.kernel_size,
            appearance_embedding,
        )
        save_tensor_image(render_before_bundle.rgb, os.path.join(level_dir, "render_before.png"))
        refined_tensor = supervision_camera.original_image.to(render_before_bundle.rgb.device)
        absorption_curve: list[dict[str, Any]] = []

        def _eval_absorption(step: int) -> None:
            absorption_curve.append(
                evaluate_absorption_step(
                    step=step,
                    level_dir=level_dir,
                    gaussians=gaussians,
                    pipe=pipe,
                    background=background,
                    kernel_size=dataset.kernel_size,
                    supervision_camera=supervision_camera,
                    zoom_embedding=appearance_embedding,
                    refined=refined_tensor,
                    neighbor_cameras=nearest_cameras,
                    train_cameras=train_cameras,
                    test_cameras=test_cameras,
                    render_before=None if step == 0 else render_before_bundle.rgb,
                )
            )

        train_log = train_appearance_only(
            gaussians,
            supervision_camera,
            train_cameras,
            pipe,
            background,
            dataset.kernel_size,
            appearance_embedding,
            num_steps=args.steps_per_level,
            mix_ratio=args.mix_ratio,
            lambda_dssim=opt.lambda_dssim,
            rng=random.Random(int(args.seed) + 10007 * (level_index + 1)),
            eval_steps=args.eval_steps_parsed,
            eval_fn=_eval_absorption if args.eval_steps_parsed else None,
        )
        write_json(os.path.join(level_dir, "view_sequence.json"), train_log)
        if absorption_curve:
            write_json(os.path.join(level_dir, "absorption_curve.json"), absorption_curve)
        global_step += args.steps_per_level

        render_after_bundle = render_bundle(
            supervision_camera,
            gaussians,
            pipe,
            background,
            dataset.kernel_size,
            appearance_embedding,
        )
        save_tensor_image(render_after_bundle.rgb, os.path.join(level_dir, "render_after.png"))
        refined_tensor = supervision_camera.original_image.to(render_after_bundle.rgb.device)
        post_train_metrics = compute_post_train_metrics(
            render_before_bundle.rgb,
            render_after_bundle.rgb,
            refined_tensor,
        )

        checkpoint_path = os.path.join(level_dir, f"chkpnt_zoom{zoom_factor:g}.pth")
        torch.save((gaussians.capture(), global_step), checkpoint_path)

        neighbor_metrics: dict[str, Any] = {}
        for neighbor_index, neighbor in enumerate(nearest_cameras):
            neighbor_bundle = render_bundle(
                neighbor,
                gaussians,
                pipe,
                background,
                dataset.kernel_size,
                embedding_for_train_camera(gaussians, neighbor.uid),
            )
            save_tensor_image(
                neighbor_bundle.rgb,
                os.path.join(level_dir, f"neighbor_{neighbor_index}_{neighbor.image_name}.png"),
            )
            neighbor_metrics[neighbor.image_name] = {
                "l1_to_gt": float(
                    torch.abs(neighbor_bundle.rgb - neighbor.original_image.to(neighbor_bundle.rgb.device)).mean().item()
                )
            }

        metrics_payload = {
            "zoom_factor": zoom_factor,
            "sr_scale": args.sr_scale,
            "supervision_mode": args.supervision_mode,
            "zoom_camera": CameraSnapshot.from_camera(zoom_camera).to_dict(),
            "supervision_camera": CameraSnapshot.from_camera(supervision_camera).to_dict(),
            "render_input_size": [zoom_input_bundle.rgb.shape[-1], zoom_input_bundle.rgb.shape[-2]],
            "supervision_size": [render_after_bundle.rgb.shape[-1], render_after_bundle.rgb.shape[-2]],
            "refinement": refinement_result.to_dict(),
            "geometry": {
                "status": geometry_payload.get("status"),
                "aligned_neighbor_count": geometry_payload.get("aligned_neighbor_count", 0),
                "fallback": geometry_payload.get("fallback"),
                "spatial_target": geometry_payload.get("spatial_target"),
            },
            "post_train": post_train_metrics,
            "neighbor_metrics": neighbor_metrics,
        }
        write_json(os.path.join(level_dir, "metrics.json"), metrics_payload)

        level_record = {
            "level_index": level_index,
            "zoom_factor": zoom_factor,
            "sr_scale": args.sr_scale,
            "supervision_mode": args.supervision_mode,
            "zoom_camera": CameraSnapshot.from_camera(zoom_camera).to_dict(),
            "supervision_camera": CameraSnapshot.from_camera(supervision_camera).to_dict(),
            "checkpoint": checkpoint_path,
            "metrics_path": os.path.join(level_dir, "metrics.json"),
            "prompt_path": os.path.join(level_dir, "prompt.json"),
            "refined_path": os.path.join(level_dir, "refined.png"),
            "geometry_path": os.path.join(level_dir, "geometry.json"),
            "geometry_status": geometry_payload.get("status"),
            "aligned_neighbor_count": geometry_payload.get("aligned_neighbor_count", 0),
            "refinement": refinement_result.to_dict(),
            "neighbor_cameras": nearest_names,
            "neighbor_metrics": neighbor_metrics,
            "metrics": post_train_metrics,
        }
        manifest["levels"] = [
            level
            for level in manifest.get("levels", [])
            if float(level.get("zoom_factor", -1.0)) != zoom_factor
        ] + [level_record]
        manifest["levels"].sort(key=lambda level: int(level.get("level_index", 0)))
        write_json(manifest_path, manifest)
        print(f"Level {zoom_factor:g}x done. metrics={post_train_metrics}")

        if not args.skip_post_train_check:
            # Artifacts and the checkpoint are durable before asserting, so a
            # failed acceptance check remains inspectable and resumable.
            assert_post_train_checks(geometry_snapshot, gaussians, post_train_metrics)

    final_checkpoint = os.path.join(output_dir, "chkpnt_final.pth")
    torch.save((gaussians.capture(), global_step), final_checkpoint)
    manifest["final_checkpoint"] = final_checkpoint
    write_json(manifest_path, manifest)
    print(f"\nGenerated zoom pipeline complete. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
