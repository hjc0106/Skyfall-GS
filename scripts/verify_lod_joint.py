#!/usr/bin/env python3
"""Appearance-compatible L0+L1 checks: empty layer, freeze, color grad, depth pairing.

Does not train a detail layer. Uses the joint RaDe-GS path (not naked SH).
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render as render_skyfall
from lod.camera import ndc_principal_to_pixel
from lod.depth import pair_depth_metrics, principal_rln, rade_image_center_rln
from lod.importer import (
    add_detail_level,
    densify_with_appearance,
    import_skyfall_l0,
    load_lod_onto_bundle,
    save_bundle,
)
from lod.path import DEFAULT_GZ_ROOT
from lod.rasterizer import render_skyfall_rade, require_rade_gs
from lod.render import render_lod_appearance
from scene import GaussianModel, Scene
from train_zoom_gen import apply_stage1_cfg_to_args, load_stage1_cfg, pick_base_camera
from utils.general_utils import safe_state
from utils.zoom_camera import NormalizedROI, make_zoom_camera
from utils.zoom_mvp_utils import (
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


def _spatial(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().float()
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    return value


def _rgb_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    return {
        "rgb_l1": image_l1(a, b),
        "rgb_hf_l1": image_hf_l1(a, b),
        "rgb_max": float((a - b).abs().max().item()),
    }


def _snapshot_l0(bundle) -> dict[str, torch.Tensor]:
    layer = bundle.layer0()
    return {
        "xyz": layer.xyz.detach().clone(),
        "sh": layer.sh.detach().clone(),
        "log_scales": layer.log_scales.detach().clone(),
        "opacity": layer.opacity_logits.detach().clone(),
        "emb": bundle.appearance.gaussian_embeddings.detach().clone(),
    }


def _l0_changed(bundle, snap: dict[str, torch.Tensor]) -> dict[str, float]:
    layer = bundle.layer0()
    return {
        "xyz": float((layer.xyz - snap["xyz"]).abs().max().item()),
        "sh": float((layer.sh - snap["sh"]).abs().max().item()),
        "log_scales": float((layer.log_scales - snap["log_scales"]).abs().max().item()),
        "opacity": float((layer.opacity_logits - snap["opacity"]).abs().max().item()),
        "emb": float((bundle.appearance.gaussian_embeddings - snap["emb"]).abs().max().item()),
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

    require_rade_gs()
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
        dataset, gaussians, load_iteration=first_iter, shuffle=False,
        ply_path=os.path.dirname(stage1_checkpoint),
    )
    gaussians.load_from_checkpoints(model_params)
    train_cameras = scene.getTrainCameras()
    gaussians.compute_3D_filter(cameras=list(train_cameras))

    bundle = import_skyfall_l0(
        gaussians, train_cameras, gz_root=args.gz_root, step_scale=args.step_scale, freeze=True,
    )
    roi = NormalizedROI(args.roi_center_x, args.roi_center_y, args.roi_width, args.roi_height)
    base_camera, is_train_view = pick_base_camera(scene, args.view_index, False, image_name=args.view_name)
    zoom_camera = make_zoom_camera(
        base_camera, roi, args.zoom_factor, uid=10000,
        image_name=f"{base_camera.image_name}_zoom{args.zoom_factor:g}",
    )
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device=gaussians.get_xyz.device,
    )
    zoom_embedding = select_appearance_embedding(gaussians, base_camera.uid, is_train_view)
    kernel = float(dataset.kernel_size)

    errors: list[str] = []
    with torch.no_grad():
        native_zoom = render_skyfall(
            zoom_camera, gaussians, pipe, background, kernel, appearance_embedding=zoom_embedding
        )
        rade_l0 = render_lod_appearance(
            bundle, zoom_camera, background=background, kernel_size=kernel,
            appearance_embedding=zoom_embedding, lod=False, max_level=0, compact=False,
        )
        rade_only = render_skyfall_rade(
            zoom_camera, gaussians, background=background, kernel_size=kernel,
            appearance_embedding=zoom_embedding,
        )
    same_rade = _rgb_metrics(rade_l0["render"], rade_only["render"])
    vs_native = _rgb_metrics(native_zoom["render"], rade_l0["render"])

    rln_center = rade_image_center_rln(
        int(zoom_camera.image_height), int(zoom_camera.image_width),
        float(zoom_camera.focal_x), float(zoom_camera.focal_y),
        device=rade_l0["render"].device,
    )
    rln_principal = principal_rln(
        int(zoom_camera.image_height), int(zoom_camera.image_width),
        float(zoom_camera.focal_x), float(zoom_camera.focal_y),
        ndc_principal_to_pixel(float(zoom_camera.cx), int(zoom_camera.image_width)),
        ndc_principal_to_pixel(float(zoom_camera.cy), int(zoom_camera.image_height)),
        device=rade_l0["render"].device,
    )
    depth_pair = pair_depth_metrics(
        native_zoom["render_depth"], native_zoom["render_alpha"],
        rade_l0["render_depth"], rade_l0["render_alpha"],
        rln_center, rln_principal,
    )
    if depth_pair.get("euclid_principal_vs_rade_rel", 1.0) > 1e-5:
        errors.append(
            "RaDe-GS out_depth is not principal-ray distance of Skyfall D/A "
            f"(rel={depth_pair.get('euclid_principal_vs_rade_rel')})"
        )
    if depth_pair.get("expected_z_vs_rade_rel", 1.0) >= depth_pair.get("raw_accum_vs_rade_rel", 0.0):
        errors.append("expected Z (D/A) is not a better match to RaDe-GS out_depth than raw accum D")

    l1_cameras = add_detail_level(bundle, [base_camera], roi, args.zoom_factor, gz_root=args.gz_root)
    if l1_cameras[0].name != base_camera.image_name:
        errors.append("L1 stage camera name must match the L0 camera")
    if abs(l1_cameras[0].fx / float(base_camera.focal_x) - args.zoom_factor) > 0.02:
        errors.append("L1 fx ratio is not the focal step_scale")

    with torch.no_grad():
        empty_l1 = render_lod_appearance(
            bundle, zoom_camera, background=background, kernel_size=kernel,
            appearance_embedding=zoom_embedding, lod=True, compact=False,
        )
        disabled = render_lod_appearance(
            bundle, zoom_camera, background=background, kernel_size=kernel,
            appearance_embedding=zoom_embedding, lod=True, max_level=0, compact=False,
        )
    empty_vs_l0 = _rgb_metrics(rade_l0["render"], empty_l1["render"])
    disabled_vs_l0 = _rgb_metrics(rade_l0["render"], disabled["render"])
    if empty_vs_l0["rgb_l1"] > 1e-6:
        errors.append(f"empty L1 changed the 2x image (rgb_l1={empty_vs_l0['rgb_l1']:.3g})")
    if disabled_vs_l0["rgb_l1"] > 1e-6:
        errors.append(f"max_level=0 did not restore L0 (rgb_l1={disabled_vs_l0['rgb_l1']:.3g})")

    ckpt = os.path.join(output_dir, "empty_l1.lod.pt")
    save_bundle(bundle, ckpt)
    with torch.no_grad():
        before = empty_l1["render"].detach().clone()
    load_lod_onto_bundle(bundle, ckpt, device=str(gaussians.get_xyz.device))
    with torch.no_grad():
        restored = render_lod_appearance(
            bundle, zoom_camera, background=background, kernel_size=kernel,
            appearance_embedding=zoom_embedding, lod=True, compact=False,
        )
    restore_metrics = _rgb_metrics(before, restored["render"])
    if restore_metrics["rgb_l1"] > 1e-6:
        errors.append(f"save/load changed empty-L1 RGB (rgb_l1={restore_metrics['rgb_l1']:.3g})")

    snap = _snapshot_l0(bundle)
    optimizer = bundle.lod.make_optimizer(
        position_lr=1e-4, feature_lr=1e-3, opacity_lr=1e-3, scaling_lr=1e-3, rotation_lr=1e-3,
    )
    n0 = int(bundle.layer0().xyz.shape[0])
    grads = torch.zeros(n0, device=bundle.layer0().xyz.device)
    grads[:32] = 1.0
    densify_stats = densify_with_appearance(
        bundle, optimizer, grads, bundle.l1_cameras,
        threshold=0.5, max_points=32, min_opacity=0.005,
        scene_extent=float(scene.cameras_extent), percent_dense=0.01,
    )
    n1 = int(bundle.layer1().xyz.shape[0]) if bundle.layer1() is not None else 0
    if n1 <= 0:
        errors.append("densify did not seed any L1 points for the gradient check")

    l0_delta = _l0_changed(bundle, snap)
    if max(l0_delta.values()) > 1e-8:
        errors.append(f"L0 tensors changed during densify: {l0_delta}")

    mlp_before = [p.detach().clone() for p in bundle.appearance.mlp.parameters()]
    emb_l0_before = bundle.appearance.gaussian_embeddings.detach().clone()
    pkg = render_lod_appearance(
        bundle, zoom_camera, background=background, kernel_size=kernel,
        appearance_embedding=zoom_embedding, lod=True, compact=False,
    )
    pkg["render"].sum().backward()
    l1_sh = bundle.layer1().sh
    l1_xyz = bundle.layer1().xyz
    grad_ok = (
        l1_sh.grad is not None and float(l1_sh.grad.abs().sum()) > 0.0
        and l1_xyz.grad is not None and float(l1_xyz.grad.abs().sum()) > 0.0
    )
    if not grad_ok:
        errors.append("L1 SH/xyz color gradient is zero; colors_precomp is probably detached")
    if bundle.layer0().sh.grad is not None:
        errors.append("L0 SH received a gradient")
    if any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in bundle.appearance.mlp.parameters()):
        errors.append("appearance MLP received a gradient")
    mlp_delta = max(
        float((a - b).abs().max().item()) for a, b in zip(bundle.appearance.mlp.parameters(), mlp_before)
    )
    emb_delta = float((bundle.appearance.gaussian_embeddings - emb_l0_before).abs().max().item())
    if mlp_delta > 0 or emb_delta > 0:
        errors.append("MLP or L0 embeddings changed after the gradient check")
    if bundle.appearance.layer_embeddings[1].requires_grad:
        errors.append("inherited L1 embeddings should be frozen constants")

    save_tensor_image(rade_l0["render"], os.path.join(output_dir, "joint_l0_zoom2x.png"))
    save_tensor_image(empty_l1["render"], os.path.join(output_dir, "empty_l1_zoom2x.png"))
    save_tensor_image(
        (rade_l0["render"] - empty_l1["render"]).abs() * 10.0,
        os.path.join(output_dir, "empty_l1_absdiff_x10.png"),
    )

    payload = {
        "accepted": len(errors) == 0,
        "errors": errors,
        "note": (
            "Joint path uses Skyfall view/proj/camera_center, copied L0 filter, frozen appearance. "
            "Empty L1 must match L0. Color grads must reach L1 SH through the frozen MLP. "
            "Depth: RaDe-GS out_depth matches Skyfall (D/A)/rln_principal (ray distance). "
            "Do not warp with raw accum D. Geometry losses stay off until this pairing is used."
        ),
        "n_l0": n0,
        "n_l1_after_seed": n1,
        "densify": densify_stats,
        "same_rade_l0_vs_skyfall_wrapper": same_rade,
        "cross_rasterizer_vs_diff_gauss": vs_native,
        "empty_l1_vs_l0": empty_vs_l0,
        "disabled_l1_vs_l0": disabled_vs_l0,
        "save_restore": restore_metrics,
        "l0_delta_after_densify": l0_delta,
        "l1_sh_grad_abs": float(l1_sh.grad.abs().sum().item()) if l1_sh.grad is not None else 0.0,
        "l1_xyz_grad_abs": float(l1_xyz.grad.abs().sum().item()) if l1_xyz.grad is not None else 0.0,
        "depth_pairing": depth_pair,
        "l1_camera": {"name": l1_cameras[0].name, "fx_ratio": float(l1_cameras[0].fx / base_camera.focal_x)},
    }
    _write_json(os.path.join(output_dir, "joint_checks.json"), payload)
    print(json.dumps({k: payload[k] for k in (
        "accepted", "errors", "empty_l1_vs_l0", "disabled_l1_vs_l0",
        "same_rade_l0_vs_skyfall_wrapper", "depth_pairing", "n_l1_after_seed",
        "l1_sh_grad_abs", "l1_xyz_grad_abs",
    )}, indent=2))
    if errors:
        raise SystemExit("joint LoD checks failed: " + "; ".join(errors))


if __name__ == "__main__":
    main()
