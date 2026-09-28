#!/usr/bin/env python3
"""Manifest-driven MULTI-VIEW full-scene RaDe LoD trainer (2x then 4x, all samples).

Trains the GaussianZoom LoD detail levels over the ENTIRE supplied
``skyfall_scene_zoom`` supervision manifest (Stage1 real views and Stage2
FlowEdit-repaired views) with real RaDe-GS rendering on the ``lod=True`` path.
No single-ROI shortcut: every zoom sample is a training target and every
level ROI camera drives visibility/densification.

Example:
    python scripts/train_scene_lod.py \\
        --start_checkpoint OUTPUT/stage1/chkpnt3000.pth \\
        --supervision OUTPUT/scene_zoom/supervision.json \\
        --output_dir OUTPUT/scene_lod \\
        --source_path DATASET_ROOT \\
        --steps_per_level 2000 --max_points_per_level 100000 --seed 0

``--steps_per_level`` is a documented MINIMUM per level: the loop keeps
round-robin stepping (seeded shuffle) until every current-level sample has
been visited at least once, even past the requested count. Smoke runs use
small values, e.g. ``--steps_per_level 100 --max_points_per_level 5000``
plus ``--max_views 4``.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch

from arguments import ModelParams, PipelineParams
from lod.path import DEFAULT_GZ_ROOT
from lod.rasterizer import require_rade_gs
from lod.scene_training import TrainConfig, train_scene_from_manifest
from scene import GaussianModel, Scene
from train_zoom_gen import apply_stage1_cfg_to_args, load_stage1_cfg
from utils.general_utils import safe_state


def _parse_positive_int_list(raw: str, flag: str) -> tuple[int, ...] | None:
    """Comma-separated positive integers; empty string -> None (flag unused)."""

    text = (raw or "").strip()
    if not text:
        return None
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part.isdigit() or int(part) < 1:
            raise ValueError(f"{flag} expects comma-separated positive integers, got {text!r}")
        values.append(int(part))
    return tuple(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    ModelParams(parser)
    PipelineParams(parser)
    parser.add_argument("--start_checkpoint", type=str, required=True)
    parser.add_argument("--parent_lod_checkpoint", type=str, default="",
                        help="Optional completed parent LoD bundle (l{N}_final.lod.pt from a "
                             "previous progressive run). Loads the parent's completed levels "
                             "+ appearance unchanged onto the frozen L0, verifies them against "
                             "this combined manifest, and trains only the remaining levels.")
    parser.add_argument("--supervision", type=str, required=True)
    parser.add_argument("--loss_mode", choices=("l1", "multiscale", "frequency"), default="l1",
                        help="'multiscale' blends HR/LR RGB; 'frequency' supervises signed "
                             "SR detail above the LR-anchor band instead of conflicting "
                             "SR low frequencies. Both require native LR anchors and "
                             "retain RaDe geometry and real-view replay.")
    parser.add_argument("--loss_hr", type=float, default=0.6,
                        help="HR RGB weight, or signed-detail weight in frequency mode "
                             "(validated starting recipe: loss_hr24, loss_lr1).")
    parser.add_argument("--loss_lr", type=float, default=0.4)
    parser.add_argument("--loss_geometry", type=float, default=0.05)
    parser.add_argument("--loss_dssim", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--steps_per_level", type=int, default=1000,
                        help="MINIMUM steps per zoom level; coverage may extend it.")
    parser.add_argument("--max_points_per_level", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_views", type=int, default=0,
                        help="Smoke helper: keep only the first N manifest views (0 = all).")
    parser.add_argument("--mix_ratio", type=float, default=0.2)
    parser.add_argument("--densify_from", type=int, default=1)
    parser.add_argument("--densify_fraction", type=float, default=0.6)
    parser.add_argument("--densify_every", type=int, default=25)
    parser.add_argument("--densify_grad_threshold", type=float, default=2e-4)
    parser.add_argument("--densify_bootstrap_fraction", type=float, default=1.0,
                        help="Initial fraction of the point cap; grow linearly to the full "
                             "cap by densify_until. 1 keeps the existing fixed budget.")
    parser.add_argument("--split_radius_pixels", type=float, default=0.0,
                        help="Split eligible primitives above this observed raster radius "
                             "instead of the scene-unit size threshold; 0 keeps the old rule.")
    parser.add_argument("--position_lr", type=float, default=1.6e-4)
    parser.add_argument("--feature_lr", type=float, default=2.5e-3)
    parser.add_argument("--opacity_lr", type=float, default=5e-2)
    parser.add_argument("--scaling_lr", type=float, default=5e-3)
    parser.add_argument("--rotation_lr", type=float, default=1e-3)
    parser.add_argument("--eval_samples", type=int, default=4)
    parser.add_argument("--png_samples", type=int, default=3)
    parser.add_argument("--target_cache_size", type=int, default=128)
    parser.add_argument("--gz_root", type=str, default=str(DEFAULT_GZ_ROOT))
    parser.add_argument("--step_scale", type=float, default=2.0)
    parser.add_argument("--disable_frozen_color_cache", action="store_true",
                        help="Recompute frozen-parent appearance each step (equivalence/performance control).")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--level_steps", type=str, default="",
                        help="Comma-separated per-level step budgets; count must match the "
                             "manifest level count. Empty keeps --steps_per_level for all levels.")
    parser.add_argument("--checkpoint_steps", type=str, default="",
                        help="Comma-separated absolute step milestones; each reached milestone "
                             "saves l{level}_step{N:06d}.lod.pt (+ appearance sidecar).")
    parser.add_argument("--densify_until", type=int, default=0,
                        help="ABSOLUTE densify-until step applied to every level; "
                             "0 derives it from --densify_fraction per level.")
    args = parser.parse_args()

    try:
        level_steps = _parse_positive_int_list(args.level_steps, "--level_steps")
        checkpoint_steps = _parse_positive_int_list(args.checkpoint_steps, "--checkpoint_steps")
    except ValueError as error:
        parser.error(str(error))
    if checkpoint_steps and any(
        later <= earlier for earlier, later in zip(checkpoint_steps, checkpoint_steps[1:])
    ):
        parser.error("--checkpoint_steps must be strictly increasing")
    if args.densify_until < 0:
        parser.error("--densify_until must be a positive integer or 0")
    densify_until = int(args.densify_until) if args.densify_until > 0 else None
    require_rade_gs()
    safe_state(args.quiet)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = os.path.abspath(args.output_dir)
    checkpoint = os.path.abspath(args.start_checkpoint)
    supervision_path = os.path.abspath(args.supervision)
    cli_source = args.source_path
    cfg = load_stage1_cfg(os.path.dirname(checkpoint))
    apply_stage1_cfg_to_args(args, cfg)
    if cli_source:
        args.source_path = cli_source
    args.data_device = "cpu"
    args.eval = True

    extract_parser = argparse.ArgumentParser(add_help=False)
    dataset = ModelParams(extract_parser).extract(args)
    dataset.model_path = output_dir

    with open(supervision_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest.get("schema_version", 0)) != 1 or manifest.get("kind") != "skyfall_scene_zoom":
        raise ValueError(f"{supervision_path} is not a version-1 skyfall_scene_zoom manifest.")

    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs,
        dataset.appearance_embedding_dim,
    )
    # Scene PLY first, then the authoritative checkpoint state (train_lod_l1 pattern).
    model_params, first_iter = torch.load(checkpoint, weights_only=False)
    scene = Scene(
        dataset, gaussians, load_iteration=first_iter, shuffle=False,
        ply_path=os.path.dirname(checkpoint),
    )
    gaussians.load_from_checkpoints(model_params)
    scene_extent = float(scene.cameras_extent)
    # The checkpoint dataset's own TRAIN cameras: authoritative stable physical
    # base-camera reference for the frozen L0 import (psi_ref birth stats and
    # any reported filter fallback), identical to the generation path.
    native_train_cameras = list(scene.getTrainCameras())
    del scene, model_params
    torch.cuda.empty_cache()

    config = TrainConfig(
        steps_per_level=args.steps_per_level,
        level_steps=level_steps,
        checkpoint_steps=checkpoint_steps or (),
        densify_until=densify_until,
        max_points_per_level=args.max_points_per_level,
        seed=args.seed,
        mix_ratio=args.mix_ratio,
        densify_from=args.densify_from,
        densify_fraction=args.densify_fraction,
        densify_every=args.densify_every,
        densify_grad_threshold=args.densify_grad_threshold,
        densify_bootstrap_fraction=args.densify_bootstrap_fraction,
        split_radius_pixels=args.split_radius_pixels,
        position_lr=args.position_lr,
        feature_lr=args.feature_lr,
        opacity_lr=args.opacity_lr,
        scaling_lr=args.scaling_lr,
        rotation_lr=args.rotation_lr,
        eval_samples=args.eval_samples,
        loss_mode=args.loss_mode,
        loss_hr=args.loss_hr,
        loss_lr=args.loss_lr,
        loss_geometry=args.loss_geometry,
        loss_dssim=args.loss_dssim,
        png_samples=args.png_samples,
        target_cache_size=args.target_cache_size,
        kernel_size=float(dataset.kernel_size),
        cache_frozen_colors=not args.disable_frozen_color_cache,
        white_background=bool(dataset.white_background),
        quiet=bool(args.quiet),
    )

    summary = train_scene_from_manifest(
        gaussians=gaussians,
        manifest=manifest,
        manifest_path=supervision_path,
        start_checkpoint=checkpoint,
        output_dir=output_dir,
        gz_root=str(os.path.abspath(args.gz_root)),
        step_scale=float(args.step_scale),
        config=config,
        scene_extent=scene_extent,
        parent_lod_checkpoint=(
            os.path.abspath(args.parent_lod_checkpoint) if args.parent_lod_checkpoint else None
        ),
        native_train_cameras=native_train_cameras,
    )

    tail = {
        "output": summary["output_dir"],
        "levels": [
            {
                "zoom": level["zoom_factor"],
                "steps": level["steps_executed"],
                "unique_targets_seen": level["unique_targets_seen"],
                "n_samples": level["n_samples"],
                "points": level["points_end"],
                "before_psnr": (level["eval_before"].get("_mean") or {}).get("psnr"),
                "after_psnr": (level["eval_after"].get("_mean") or {}).get("psnr"),
                "checkpoint": level["checkpoint"],
            }
            for level in summary["levels"]
        ],
    }
    print(json.dumps(tail, indent=2))


if __name__ == "__main__":
    main()
