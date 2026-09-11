#!/usr/bin/env python3
"""L2 absorption at 4x on frozen L0+L1. Same mix/densify budget as L1. No geometry regularizer."""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from PIL import Image

from arguments import ModelParams, PipelineParams
from lod.camera import zoom_stage_camera
from lod.correspondence import neighbor_correspondence_report
from lod.freeze import assert_snapshot_unchanged, completed_layers_frozen, snapshot_levels
from lod.importer import add_detail_level, densify_with_appearance, import_skyfall_l0, load_lod_onto_bundle, save_bundle
from lod.inspect import active_opacity_stats, l1_geometry_report, layer_weight_means, neighbor_zoom_report, scale_sweep
from lod.lineage import file_identity, roi_dict, write_json, write_train_lineage
from lod.path import DEFAULT_GZ_ROOT
from lod.rasterizer import require_rade_gs
from lod.render import render_lod_appearance
from lod.train_state import (
    TRAIN_STATE_NAME,
    advance_mix_rng,
    capture_train_state,
    densify_spec,
    keep_upto,
    load_json_list,
    load_train_state,
    mix_rng,
    restore_train_state,
    save_train_state,
    sidecar_dir,
    spec_mismatches,
)
from scene import GaussianModel, Scene
from train_zoom_gen import (
    ABSORPTION_CROPS,
    _fractional_crop,
    _view_means,
    apply_stage1_cfg_to_args,
    load_stage1_cfg,
    parse_eval_steps,
    pick_base_camera,
)
from utils.general_utils import PILtoTorch, safe_state
from utils.loss_utils import l1_loss
from utils.zoom_camera import NormalizedROI, make_zoom_camera
from utils.zoom_mvp_utils import (
    crop_chw,
    embedding_for_train_camera,
    image_hf_l1,
    image_l1,
    save_tensor_image,
    select_appearance_embedding,
)


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_refined(path: str, width: int, height: int, device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = PILtoTorch(image, (width, height)).clamp(0.0, 1.0)
    if tensor.shape[0] == 4:
        tensor = tensor[:3]
    return tensor.to(device=device, dtype=torch.float32)


def _n_active(bundle) -> int:
    layer = bundle.active_layer()
    return int(layer.xyz.shape[0]) if layer is not None else 0


@torch.no_grad()
def _eval_step(
    *,
    step: int,
    out_dir: str,
    bundle,
    zoom_camera,
    parent_zoom_camera,
    refined: torch.Tensor,
    train_cameras,
    test_cameras,
    gaussians,
    background,
    kernel: float,
    zoom_embedding,
    render_before: torch.Tensor | None,
    parent_ref: torch.Tensor | None,
    skip_view_eval: bool = False,
) -> dict:
    step_dir = os.path.join(out_dir, "steps", f"{int(step):04d}")
    os.makedirs(step_dir, exist_ok=True)
    pkg = render_lod_appearance(
        bundle, zoom_camera, background=background, kernel_size=kernel,
        appearance_embedding=zoom_embedding, lod=True, compact=False,
    )
    rgb = pkg["render"].clamp(0.0, 1.0)
    save_tensor_image(rgb, os.path.join(step_dir, "target.png"))
    height, width = int(rgb.shape[-2]), int(rgb.shape[-1])
    crops = {}
    for name, frac in ABSORPTION_CROPS.items():
        box = _fractional_crop(width, height, frac)
        save_tensor_image(crop_chw(rgb, box), os.path.join(step_dir, f"crop_{name}.png"))
        crops[name] = {
            "box": list(box),
            "l1_to_refined": image_l1(crop_chw(rgb, box), crop_chw(refined, box)),
            "hf_l1_to_refined": image_hf_l1(crop_chw(rgb, box), crop_chw(refined, box)),
        }
    parent_pkg = render_lod_appearance(
        bundle, parent_zoom_camera, background=background, kernel_size=kernel,
        appearance_embedding=zoom_embedding, lod=True, compact=False,
    )
    parent_frozen_pkg = render_lod_appearance(
        bundle, parent_zoom_camera, background=background, kernel_size=kernel,
        appearance_embedding=zoom_embedding, lod=True, max_level=1, compact=False,
    )
    parent_full = parent_pkg["render"].clamp(0, 1)
    parent_frozen = parent_frozen_pkg["render"].clamp(0, 1)
    payload = {
        "step": int(step),
        "n_l2": _n_active(bundle),
        "active": active_opacity_stats(bundle),
        "target": {
            "l1_to_refined": image_l1(rgb, refined),
            "hf_l1_to_refined": image_hf_l1(rgb, refined),
            "crops": crops,
        },
        "parent_2x": {
            "l1_vs_frozen_l1": image_l1(parent_full, parent_frozen),
            "hf_vs_frozen_l1": image_hf_l1(parent_full, parent_frozen),
            "l2_weight_mean": layer_weight_means(parent_pkg).get("level_2", 0.0),
            "l1_weight_mean": layer_weight_means(parent_pkg).get("level_1", 0.0),
            "joint_weights": layer_weight_means(parent_pkg),
            "frozen_l1_weights": layer_weight_means(parent_frozen_pkg),
        },
        "train_views": {},
        "heldout_views": {},
    }
    if parent_ref is not None:
        payload["parent_2x"]["l1_vs_step0"] = image_l1(parent_full, parent_ref)
        payload["parent_2x"]["hf_vs_step0"] = image_hf_l1(parent_full, parent_ref)
    if render_before is not None:
        payload["target"]["l1_from_step0"] = image_l1(rgb, render_before)
        payload["target"]["hf_l1_from_step0"] = image_hf_l1(rgb, render_before)
    if not skip_view_eval:
        for camera in train_cameras:
            view = render_lod_appearance(
                bundle, camera, background=background, kernel_size=kernel,
                appearance_embedding=embedding_for_train_camera(gaussians, camera.uid),
                lod=True, compact=False,
            )
            gt = camera.original_image.to(device=view["render"].device)
            payload["train_views"][camera.image_name] = {
                "l1_to_gt": image_l1(view["render"].clamp(0, 1), gt),
                "hf_l1_to_gt": image_hf_l1(view["render"].clamp(0, 1), gt),
            }
        for camera in test_cameras:
            view = render_lod_appearance(
                bundle, camera, background=background, kernel_size=kernel,
                appearance_embedding=bundle.appearance.embedding_for_uid(camera.uid, is_train_view=False),
                lod=True, compact=False,
            )
            gt = camera.original_image.to(device=view["render"].device)
            payload["heldout_views"][camera.image_name] = {
                "l1_to_gt": image_l1(view["render"].clamp(0, 1), gt),
                "hf_l1_to_gt": image_hf_l1(view["render"].clamp(0, 1), gt),
            }
    payload["train_views_mean"] = _view_means(payload["train_views"])
    payload["heldout_views_mean"] = _view_means(payload["heldout_views"])
    _write_json(os.path.join(step_dir, "metrics.json"), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    ModelParams(parser)
    PipelineParams(parser)
    parser.add_argument("--start_checkpoint", type=str, required=True)
    parser.add_argument("--l1_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--refined_image", type=str, required=True)
    parser.add_argument("--gz_root", type=str, default=str(DEFAULT_GZ_ROOT))
    parser.add_argument("--view_index", type=int, default=0)
    parser.add_argument("--view_name", type=str, default="", help="Lock the base camera by image stem. Preferred over view_index.")
    parser.add_argument("--roi_center_x", type=float, default=0.592)
    parser.add_argument("--roi_center_y", type=float, default=0.53)
    parser.add_argument("--roi_width", type=float, default=0.1)
    parser.add_argument("--roi_height", type=float, default=0.1)
    parser.add_argument("--zoom_factor", type=float, default=4.0)
    parser.add_argument("--parent_zoom_factor", type=float, default=2.0)
    parser.add_argument("--step_scale", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--eval_steps", type=str, default="0,50,100,250,500")
    parser.add_argument("--mix_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--densify_from", type=int, default=1)
    parser.add_argument("--densify_until", type=int, default=250)
    parser.add_argument("--densify_interval", type=int, default=10)
    parser.add_argument("--densify_grad_threshold", type=float, default=2e-4)
    parser.add_argument("--max_l2_points", type=int, default=50000)
    parser.add_argument("--skip_cross_view", action="store_true")
    parser.add_argument("--skip_view_eval", action="store_true")
    parser.add_argument("--eval_only_listed", action="store_true")
    parser.add_argument(
        "--alignment",
        type=str,
        default="geometry",
        help="Label only: which 4x DLoRAL alignment produced --refined_image.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    require_rade_gs()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    safe_state(args.quiet)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    stage1_checkpoint = os.path.abspath(args.start_checkpoint)
    stage1_cfg = load_stage1_cfg(os.path.dirname(stage1_checkpoint))
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
    model_params, first_iter = torch.load(stage1_checkpoint, weights_only=False)
    scene = Scene(
        dataset, gaussians, load_iteration=first_iter, shuffle=False,
        ply_path=os.path.dirname(stage1_checkpoint),
    )
    gaussians.load_from_checkpoints(model_params)
    train_cameras = list(scene.getTrainCameras())
    test_cameras = list(scene.getTestCameras())
    gaussians.compute_3D_filter(cameras=train_cameras)

    bundle = import_skyfall_l0(
        gaussians, train_cameras, gz_root=args.gz_root, step_scale=args.step_scale, freeze=True,
    )
    device = str(gaussians.get_xyz.device)
    roi = NormalizedROI(args.roi_center_x, args.roi_center_y, args.roi_width, args.roi_height)
    base_camera, is_train_view = pick_base_camera(scene, args.view_index, False, image_name=args.view_name)
    zoom_camera = make_zoom_camera(
        base_camera, roi, args.zoom_factor, uid=10000,
        image_name=f"{base_camera.image_name}_zoom{args.zoom_factor:g}",
    )
    parent_zoom_camera = make_zoom_camera(
        base_camera, roi, args.parent_zoom_factor, uid=10001,
        image_name=f"{base_camera.image_name}_zoom{args.parent_zoom_factor:g}",
    )
    spec = densify_spec(
        densify_from=args.densify_from, densify_until=args.densify_until,
        densify_interval=args.densify_interval, densify_grad_threshold=args.densify_grad_threshold,
        max_points=args.max_l2_points, mix_ratio=args.mix_ratio, seed=args.seed,
    )
    opt_state = None
    resume_path = os.path.abspath(args.resume) if args.resume else ""
    if resume_path:
        opt_state = load_lod_onto_bundle(bundle, resume_path, device=device)
        bundle.stage_cameras = [
            zoom_stage_camera(base_camera, roi, args.zoom_factor, gz_root=args.gz_root, device=device)
        ]
    else:
        load_lod_onto_bundle(bundle, os.path.abspath(args.l1_checkpoint), device=device)
        add_detail_level(bundle, [base_camera], roi, args.zoom_factor, gz_root=args.gz_root)
    frozen_before = snapshot_levels(bundle, (0, 1))

    optimizer = bundle.lod.make_optimizer(
        position_lr=1.6e-4, feature_lr=2.5e-3, opacity_lr=5e-2, scaling_lr=5e-3, rotation_lr=1e-3,
    )
    if opt_state:
        optimizer.load_state_dict(opt_state)
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device=gaussians.get_xyz.device,
    )
    kernel = float(dataset.kernel_size)
    zoom_embedding = select_appearance_embedding(gaussians, base_camera.uid, is_train_view)
    refined = _load_refined(
        args.refined_image, int(zoom_camera.image_width), int(zoom_camera.image_height),
        gaussians.get_xyz.device,
    )
    eval_set = set(parse_eval_steps(args.eval_steps))
    if not args.eval_only_listed:
        eval_set.add(int(args.steps))
        if int(args.start_step) == 0:
            eval_set.add(0)

    parent_ref = render_lod_appearance(
        bundle, parent_zoom_camera, background=background, kernel_size=kernel,
        appearance_embedding=zoom_embedding, lod=True, max_level=1, compact=False,
    )["render"].detach().clamp(0, 1).clone()
    side = sidecar_dir(output_dir, resume_path or None)
    saved_state = load_train_state(side / TRAIN_STATE_NAME)
    mismatches = spec_mismatches(None if saved_state is None else saved_state.get("spec"), spec)
    if mismatches:
        raise ValueError("Resume densify/mix spec differs on: " + ", ".join(mismatches))
    rng = mix_rng(args.seed)
    if saved_state is not None:
        restore_train_state(saved_state, rng)
    elif int(args.start_step) > 0:
        advance_mix_rng(rng, int(args.start_step), mix_ratio=float(args.mix_ratio), n_train=len(train_cameras))

    curve_path = os.path.join(output_dir, "absorption_curve.json")
    curve = keep_upto(load_json_list(side / "absorption_curve.json", "curve"), int(args.start_step))
    densify_log_path = os.path.join(output_dir, "densify_log.json")
    densify_log = keep_upto(load_json_list(side / "densify_log.json"), int(args.start_step))
    render_before = None
    if int(args.start_step) == 0:
        step0 = _eval_step(
            step=0, out_dir=output_dir, bundle=bundle, zoom_camera=zoom_camera,
            parent_zoom_camera=parent_zoom_camera, refined=refined, train_cameras=train_cameras,
            test_cameras=test_cameras, gaussians=gaussians, background=background, kernel=kernel,
            zoom_embedding=zoom_embedding, render_before=None, parent_ref=parent_ref,
            skip_view_eval=bool(args.skip_view_eval),
        )
        render_before = render_lod_appearance(
            bundle, zoom_camera, background=background, kernel_size=kernel,
            appearance_embedding=zoom_embedding, lod=True, compact=False,
        )["render"].detach().clamp(0, 1).clone()
        save_tensor_image(render_before, os.path.join(output_dir, "render_before.png"))
        save_tensor_image(parent_ref, os.path.join(output_dir, "parent_2x_frozen.png"))
        curve = [step0]
        densify_log = [{"step": 0, "loss": None, "mix_train_view": False, **active_opacity_stats(bundle)}]
        _write_json(densify_log_path, densify_log)
    else:
        before_path = side / "render_before.png"
        if before_path.is_file():
            render_before = _load_refined(
                str(before_path), int(zoom_camera.image_width), int(zoom_camera.image_height),
                gaussians.get_xyz.device,
            )

    densify_cameras = bundle.stage_cameras or bundle.l1_cameras
    for step in range(int(args.start_step) + 1, int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        use_train = rng.random() < float(args.mix_ratio)
        if use_train:
            camera = rng.choice(train_cameras)
            embedding = embedding_for_train_camera(gaussians, camera.uid)
            target = camera.original_image.to(device=gaussians.get_xyz.device)
        else:
            camera = zoom_camera
            embedding = zoom_embedding
            target = refined
        pkg = render_lod_appearance(
            bundle, camera, background=background, kernel_size=kernel,
            appearance_embedding=embedding, lod=True, compact=False,
        )
        loss = l1_loss(pkg["render"], target)
        loss.backward()
        densify_stats = None
        if (
            args.densify_from <= step <= args.densify_until
            and step % args.densify_interval == 0
            and pkg["viewspace_points"].grad is not None
        ):
            screen_grad = pkg["viewspace_points"].grad.detach()[:, :2].norm(dim=-1)
            densify_stats = densify_with_appearance(
                bundle, optimizer, screen_grad, densify_cameras,
                threshold=float(args.densify_grad_threshold),
                max_points=int(args.max_l2_points),
                min_opacity=0.005,
                scene_extent=float(scene.cameras_extent),
                percent_dense=0.01,
            )
        optimizer.step()
        if step % 20 == 0 or densify_stats:
            rec = {
                "step": int(step),
                "loss": float(loss.item()),
                "mix_train_view": bool(use_train),
                "densify": densify_stats or {},
                **active_opacity_stats(bundle),
            }
            densify_log.append(rec)
            _write_json(densify_log_path, densify_log)
            print(
                f"step {step} loss={float(loss.item()):.5f} n_l2={rec['n']} "
                f"opacity_mean={rec['opacity_mean']:.3f} frac_lt_0.05={rec['frac_opacity_lt_0.05']:.3f}"
            )
        if step in eval_set:
            curve.append(
                _eval_step(
                    step=step, out_dir=output_dir, bundle=bundle, zoom_camera=zoom_camera,
                    parent_zoom_camera=parent_zoom_camera, refined=refined, train_cameras=train_cameras,
                    test_cameras=test_cameras, gaussians=gaussians, background=background, kernel=kernel,
                    zoom_embedding=zoom_embedding, render_before=render_before, parent_ref=parent_ref,
                    skip_view_eval=bool(args.skip_view_eval),
                )
            )

    _write_json(densify_log_path, densify_log)
    save_bundle(bundle, os.path.join(output_dir, "l2_final.lod.pt"), optimizer=optimizer)
    save_train_state(
        os.path.join(output_dir, TRAIN_STATE_NAME),
        capture_train_state(step=int(args.steps), mix=rng, spec=spec),
    )
    frozen_after = snapshot_levels(bundle, (0, 1))
    assert_snapshot_unchanged(frozen_before, frozen_after)
    write_json(
        os.path.join(output_dir, "freeze.json"),
        {
            "ok": True,
            "completed": completed_layers_frozen(bundle, up_to_level=2),
            "before": frozen_before,
            "after": frozen_after,
        },
    )
    write_train_lineage(
        output_dir,
        {
            "kind": "lod_train",
            "level": 2,
            "start_checkpoint": file_identity(stage1_checkpoint),
            "parent_lod": file_identity(args.l1_checkpoint),
            "resume": file_identity(resume_path) if resume_path else None,
            "refined_image": file_identity(args.refined_image, hash_file=True),
            "roi": roi_dict(args.roi_center_x, args.roi_center_y, args.roi_width, args.roi_height),
            "view_index": int(args.view_index),
            "view_name": str(getattr(base_camera, "image_name", "") or "") or None,
            "zoom_factor": float(args.zoom_factor),
            "parent_zoom_factor": float(args.parent_zoom_factor),
            "step_scale": float(args.step_scale),
            "steps": int(args.steps),
            "start_step": int(args.start_step),
            "mix_ratio": float(args.mix_ratio),
            "seed": int(args.seed),
            "alignment_label": str(args.alignment),
            "densify": spec,
            "train_state": TRAIN_STATE_NAME,
        },
    )
    summary = {
        "backend": "lod_l2_rade_appearance",
        "mode": str(args.alignment),
        "l1_checkpoint": os.path.abspath(args.l1_checkpoint),
        "refined_image": os.path.abspath(args.refined_image),
        "steps": int(args.steps),
        "start_step": int(args.start_step),
        "mix_ratio": float(args.mix_ratio),
        "n_l0": int(bundle.layer0().xyz.shape[0]),
        "n_l1": int(bundle.layer1().xyz.shape[0]) if bundle.layer1() is not None else 0,
        "n_l2": _n_active(bundle),
        "geometry_regularizer": False,
        "curve": curve,
        "densify_log": densify_log,
    }
    _write_json(curve_path, summary)
    if not args.skip_cross_view:
        cross_dir = os.path.join(output_dir, "cross_view")
        os.makedirs(cross_dir, exist_ok=True)
        corr_dir = os.path.join(output_dir, "correspondence")
        os.makedirs(corr_dir, exist_ok=True)
        cross = {
            "geometry": l1_geometry_report(bundle, child_level=int(bundle.lod.active_level)),
            "scale_sweep": scale_sweep(
                bundle, base_camera, roi, background=background, kernel=kernel,
                embedding=zoom_embedding, factors=(2.0, 2.5, 3.0, 3.5, 4.0), out_dir=cross_dir,
            ),
            "views": neighbor_zoom_report(
                bundle, base_camera, train_cameras, test_cameras, roi,
                background=background, kernel=kernel, gaussians=gaussians,
                zoom_factor=args.zoom_factor, out_dir=cross_dir,
            ),
            "parent_2x_views": neighbor_zoom_report(
                bundle, base_camera, train_cameras, test_cameras, roi,
                background=background, kernel=kernel, gaussians=gaussians,
                zoom_factor=args.parent_zoom_factor, out_dir=None,
            ),
        }
        _write_json(os.path.join(output_dir, "cross_view.json"), cross)
        correspondence = neighbor_correspondence_report(
            bundle, base_camera, train_cameras, roi,
            background=background, kernel=kernel, gaussians=gaussians,
            zoom_factor=args.zoom_factor, floor_max_level=1, child_level=int(bundle.lod.active_level),
            out_dir=corr_dir,
        )
        _write_json(os.path.join(output_dir, "correspondence.json"), correspondence)
        summary["cross_view"] = os.path.join(output_dir, "cross_view.json")
        summary["correspondence"] = os.path.join(output_dir, "correspondence.json")
        _write_json(curve_path, summary)
    print(json.dumps({
        "n_l2": summary["n_l2"],
        "step0_l1": curve[0]["target"]["l1_to_refined"] if curve else None,
        "final_l1": curve[-1]["target"]["l1_to_refined"] if curve else None,
        "final_hf": curve[-1]["target"]["hf_l1_to_refined"] if curve else None,
        "train_gt": curve[-1]["train_views_mean"]["l1_to_gt_mean"] if curve else None,
        "parent_2x": curve[-1].get("parent_2x") if curve else None,
    }, indent=2))


if __name__ == "__main__":
    main()
