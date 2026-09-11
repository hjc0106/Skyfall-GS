"""Cross-view, scale-sweep, and L1 geometry checks. No training."""

from __future__ import annotations

from typing import Any

import torch

from lod.render import render_lod_appearance
from utils.zoom_camera import NormalizedROI, make_zoom_camera, nearest_train_cameras
from utils.zoom_mvp_utils import image_hf_l1, image_l1, save_tensor_image


def _percentiles(values: torch.Tensor) -> dict[str, float]:
    if int(values.numel()) == 0:
        return {"count": 0}
    flat = values.detach().float().reshape(-1)
    qs = torch.quantile(flat, torch.tensor([0.5, 0.9, 0.99], device=flat.device))
    return {
        "count": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "p50": float(qs[0].item()),
        "p90": float(qs[1].item()),
        "p99": float(qs[2].item()),
        "max": float(flat.max().item()),
    }


def _spatial(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().float()
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    return value


def _weight_means(pkg: dict[str, Any]) -> dict[str, float]:
    weights = pkg.get("weights")
    slices = pkg.get("layer_slices") or []
    if weights is None:
        return {}
    report = {"all": float(weights.mean().item())}
    for level, sl in enumerate(slices):
        if sl.stop > sl.start:
            report[f"level_{level}"] = float(weights[sl].mean().item())
        else:
            report[f"level_{level}"] = 0.0
    return report


def layer_weight_means(pkg: dict[str, Any]) -> dict[str, float]:
    return _weight_means(pkg)


@torch.no_grad()
def active_opacity_stats(bundle) -> dict[str, float]:
    """Point count and opacity of the trainable detail level. Empty level is zeros."""

    layer = bundle.active_layer() if hasattr(bundle, "active_layer") else None
    if layer is None or int(layer.xyz.shape[0]) == 0:
        return {
            "n": 0,
            "opacity_mean": 0.0,
            "opacity_p50": 0.0,
            "frac_opacity_lt_0.05": 0.0,
        }
    opacity = torch.sigmoid(layer.opacity_logits).reshape(-1).float()
    return {
        "n": int(opacity.numel()),
        "opacity_mean": float(opacity.mean().item()),
        "opacity_p50": float(opacity.median().item()),
        "frac_opacity_lt_0.05": float((opacity < 0.05).float().mean().item()),
    }


@torch.no_grad()
def l1_geometry_tensors(bundle, child_level: int = 1) -> dict[str, torch.Tensor] | None:
    """Per-point child vs ancestor geometry. Offsets are parent-scale normalized."""

    child = bundle.layer(child_level) if hasattr(bundle, "layer") else bundle.layer1()
    if child is None or int(child.xyz.shape[0]) == 0:
        return None
    ancestor_xyz = []
    ancestor_scale = []
    ancestor_ids = []
    for level in range(int(child_level)):
        layer = bundle.layer(level) if hasattr(bundle, "layer") else (bundle.layer0() if level == 0 else None)
        if layer is None or int(layer.xyz.shape[0]) == 0:
            continue
        ancestor_ids.append(layer.node_ids)
        ancestor_xyz.append(layer.xyz)
        ancestor_scale.append(torch.exp(layer.log_scales).amax(dim=1))
    if not ancestor_ids:
        return None
    all_ids = torch.cat(ancestor_ids, dim=0)
    all_xyz = torch.cat(ancestor_xyz, dim=0)
    all_scale = torch.cat(ancestor_scale, dim=0)
    mapper_size = int(all_ids.max().item()) + 1
    mapper = torch.full((mapper_size,), -1, dtype=torch.long, device=all_ids.device)
    mapper[all_ids] = torch.arange(int(all_ids.numel()), device=all_ids.device)
    parent = child.parent_ids
    rows = mapper[parent.clamp(min=0, max=mapper_size - 1)]
    valid = (parent >= 0) & (parent < mapper_size) & (rows >= 0)
    parent_xyz = all_xyz[rows.clamp_min(0)]
    parent_scale = all_scale[rows.clamp_min(0)]
    offset = torch.where(valid, (child.xyz - parent_xyz).norm(dim=1), torch.zeros(child.xyz.shape[0], device=child.xyz.device))
    child_scale = torch.exp(child.log_scales).amax(dim=1)
    scale_ratio = child_scale / parent_scale.clamp_min(1e-8)
    offset_norm = offset / parent_scale.clamp_min(1e-8)
    opacity = torch.sigmoid(child.opacity_logits).reshape(-1)
    opaque = valid & (opacity >= 0.05)
    return {
        "xyz": child.xyz,
        "valid": valid,
        "opaque": opaque,
        "offset": offset,
        "offset_norm": offset_norm,
        "scale_ratio": scale_ratio,
        "opacity": opacity,
        "child_scale": child_scale,
        "scale_gt_4x_parent": opaque & (scale_ratio > 4.0),
        "offset_gt_4x_parent": opaque & (offset_norm > 4.0),
    }


@torch.no_grad()
def l1_geometry_report(bundle, child_level: int = 1) -> dict[str, Any]:
    """Where the new Gaussians sit relative to their ancestors. No densify."""

    tensors = l1_geometry_tensors(bundle, child_level=child_level)
    if tensors is None:
        return {"n_l1": 0, "child_level": int(child_level)}
    valid = tensors["valid"]
    scale_ratio = tensors["scale_ratio"]
    offset_norm = tensors["offset_norm"]
    return {
        "n_l1": int(tensors["xyz"].shape[0]),
        "child_level": int(child_level),
        "valid_parent_frac": float(valid.float().mean().item()),
        "offset_to_parent": _percentiles(tensors["offset"][valid]),
        "offset_norm_to_parent": _percentiles(offset_norm[valid]),
        "child_max_scale": _percentiles(tensors["child_scale"]),
        "scale_ratio_to_parent": _percentiles(scale_ratio[valid]),
        "opacity": _percentiles(tensors["opacity"]),
        "frac_opacity_lt_0.05": float((tensors["opacity"] < 0.05).float().mean().item()),
        "frac_scale_gt_parent": float((scale_ratio[valid] > 1.0).float().mean().item()) if bool(valid.any()) else 0.0,
        "frac_scale_gt_4x_parent": float((scale_ratio[valid] > 4.0).float().mean().item()) if bool(valid.any()) else 0.0,
        "frac_offset_gt_4x_parent": float((offset_norm[valid] > 4.0).float().mean().item()) if bool(valid.any()) else 0.0,
    }


@torch.no_grad()
def scale_sweep(
    bundle,
    base_camera,
    roi: NormalizedROI,
    *,
    background,
    kernel: float,
    embedding,
    factors: tuple[float, ...] = (1.25, 1.5, 1.75, 2.0),
    out_dir: str | None = None,
) -> dict[str, Any]:
    """ROI-centered focal zoom through the valid 2× window.

    This JAX_068 ROI is off-center; ``make_zoom_camera`` rejects factor 1.0
    (the 1× crop would leave the image). Native 1× is the unshifted base
    camera, reported separately.
    """

    rows = []
    prev_rgb = None
    prev_mean = None
    native = render_lod_appearance(
        bundle, base_camera, background=background, kernel_size=kernel,
        appearance_embedding=embedding, lod=True, compact=False,
    )
    native_rgb = native["render"].clamp(0, 1)
    rows.append({
        "factor": 1.0,
        "kind": "unshifted_native",
        "mean_rgb": float(native_rgb.mean().item()),
        "mean_alpha": float(_spatial(native["render_alpha"]).mean().item()),
        "weights": _weight_means(native),
    })
    if out_dir is not None:
        save_tensor_image(native_rgb, f"{out_dir}/scale_1_native.png")
    skipped_factors = []
    for index, factor in enumerate(factors):
        if not roi.zoom_is_valid(factor):
            skipped_factors.append(float(factor))
            continue
        camera = make_zoom_camera(
            base_camera, roi, factor, uid=20000 + index,
            image_name=f"{base_camera.image_name}_zoom{factor:g}_sweep",
        )
        pkg = render_lod_appearance(
            bundle, camera, background=background, kernel_size=kernel,
            appearance_embedding=embedding, lod=True, compact=False,
        )
        rgb = pkg["render"].clamp(0, 1)
        alpha = _spatial(pkg["render_alpha"])
        mean_rgb = float(rgb.mean().item())
        row = {
            "factor": float(factor),
            "kind": "roi_centered_zoom",
            "mean_rgb": mean_rgb,
            "mean_alpha": float(alpha.mean().item()),
            "weights": _weight_means(pkg),
        }
        if prev_mean is not None and prev_rgb is not None:
            row["mean_rgb_jump"] = abs(mean_rgb - prev_mean)
            row["rgb_l1_vs_prev"] = image_l1(rgb, prev_rgb)
        if out_dir is not None:
            save_tensor_image(rgb, f"{out_dir}/scale_{factor:g}.png")
        rows.append(row)
        prev_rgb = rgb
        prev_mean = mean_rgb
    jumps = [row["mean_rgb_jump"] for row in rows if "mean_rgb_jump" in row]
    return {
        "factors": list(factors),
        "skipped_factors": skipped_factors,
        "min_valid_zoom": float(roi.min_zoom_factor()),
        "max_mean_rgb_jump": max(jumps) if jumps else 0.0,
        "rows": rows,
    }


@torch.no_grad()
def neighbor_zoom_report(
    bundle,
    base_camera,
    train_cameras,
    test_cameras,
    roi: NormalizedROI,
    *,
    background,
    kernel: float,
    gaussians,
    zoom_factor: float = 2.0,
    neighbor_k: int = 2,
    out_dir: str | None = None,
) -> dict[str, Any]:
    """2× zoom on views that were not the enhanced target."""

    from utils.zoom_mvp_utils import embedding_for_train_camera, select_appearance_embedding

    test_uids = {int(cam.uid) for cam in test_cameras}
    neighbors = nearest_train_cameras(base_camera, train_cameras, k=neighbor_k)
    payload: dict[str, Any] = {"neighbors": {}, "test_2x": {}}
    for index, camera in enumerate(list(neighbors) + list(test_cameras)):
        is_test = int(camera.uid) in test_uids
        zoom = make_zoom_camera(
            camera, roi, zoom_factor, uid=30000 + index,
            image_name=f"{camera.image_name}_zoom{zoom_factor:g}",
        )
        embedding = (
            select_appearance_embedding(gaussians, camera.uid, False)
            if is_test
            else embedding_for_train_camera(gaussians, camera.uid)
        )
        full = render_lod_appearance(
            bundle, zoom, background=background, kernel_size=kernel,
            appearance_embedding=embedding, lod=True, compact=False,
        )
        l0_only = render_lod_appearance(
            bundle, zoom, background=background, kernel_size=kernel,
            appearance_embedding=embedding, lod=True, max_level=0, compact=False,
        )
        rgb = full["render"].clamp(0, 1)
        rgb0 = l0_only["render"].clamp(0, 1)
        alpha = _spatial(full["render_alpha"])
        alpha0 = _spatial(l0_only["render_alpha"])
        residual = (rgb - rgb0).abs()
        low_l0 = alpha0 < 0.05
        floater = low_l0 & (alpha > 0.2)
        rec = {
            "l1_vs_l0_only": image_l1(rgb, rgb0),
            "hf_vs_l0_only": image_hf_l1(rgb, rgb0),
            "alpha_l1_vs_l0_only": float((alpha - alpha0).abs().mean().item()),
            "floater_frac": float(floater.float().mean().item()),
            "residual_in_low_l0_alpha": float(residual.mean(dim=0)[low_l0].mean().item()) if bool(low_l0.any()) else 0.0,
            "mean_rgb": float(rgb.mean().item()),
            "mean_rgb_l0": float(rgb0.mean().item()),
        }
        key = "test_2x" if is_test else "neighbors"
        payload[key][camera.image_name] = rec
        if out_dir is not None:
            tag = f"{'test' if is_test else 'neighbor'}_{camera.image_name}"
            save_tensor_image(rgb, f"{out_dir}/{tag}_l1.png")
            save_tensor_image(rgb0, f"{out_dir}/{tag}_l0.png")
            save_tensor_image(residual.clamp(0, 1) * 6.0, f"{out_dir}/{tag}_absdiff_x6.png")
    return payload
