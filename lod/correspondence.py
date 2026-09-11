"""Depth-driven co-visible correspondence. Does not train or add regularizers.

Warp convention matches ``refinement.geometry_warp``: destination pixels are
unprojected with **camera Z**, then source RGB is sampled. RaDe ``out_depth``
is ray distance and must be converted first.

L0-depth warps hold the surface fixed so appearance changes can be compared
fairly. Trained-depth warps check whether the new layer moved the surface.
The L0-RGB warp residual is the view-dependent SH / embedding floor; excess
over that floor is the correspondence signal. Change-map agreement asks
whether neighbor-view L1 edits land where the warped target edits land.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from lod.depth import depth_from_rade
from lod.inspect import l1_geometry_tensors
from lod.render import render_lod_appearance
from refinement.geometry_warp import (
    build_reprojection_grid,
    estimate_depth_scene_scale,
    project_world_points,
    warp_image,
)
from utils.zoom_camera import NormalizedROI, make_zoom_camera, nearest_train_cameras
from utils.zoom_mvp_utils import embedding_for_train_camera, save_tensor_image

# Same windows as train_zoom_gen.ABSORPTION_CROPS; kept here so lod does not import the trainer.
ABSORPTION_CROPS = {
    "building": (0.04, 0.20, 0.44, 0.59),
    "vehicles": (0.24, 0.62, 0.78, 0.98),
    "trees": (0.34, 0.37, 0.63, 0.61),
}


LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)


def _fractional_crop(width: int, height: int, box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x0 = int(round(box[0] * width))
    y0 = int(round(box[1] * height))
    x1 = int(round(box[2] * width))
    y1 = int(round(box[3] * height))
    return max(0, x0), max(0, y0), min(width, max(x0 + 1, x1)), min(height, max(y0 + 1, y1))


def rade_pkg_camera_z(pkg: dict[str, Any], camera) -> torch.Tensor:
    """RaDe ``render_depth`` (ray distance) → camera Z for unprojection."""

    return depth_from_rade(pkg["render_depth"], camera, alpha=pkg["render_alpha"]).camera_z()


def _spatial(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().float()
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    return value


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    return LUMA_WEIGHTS[0] * rgb[0] + LUMA_WEIGHTS[1] * rgb[1] + LUMA_WEIGHTS[2] * rgb[2]


def _dilate(mask: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    pad = kernel // 2
    return F.max_pool2d(mask.float()[None, None], kernel, stride=1, padding=pad)[0, 0] > 0.5


def _erode(mask: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    return ~_dilate(~mask, kernel)


def _highpass(rgb: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=rgb.device,
        dtype=rgb.dtype,
    ).view(1, 1, 3, 3).expand(rgb.shape[0], 1, 3, 3)
    return F.conv2d(rgb.unsqueeze(0), kernel, padding=1, groups=rgb.shape[0])[0]


def _masked_mean(plane: torch.Tensor, mask: torch.Tensor) -> float | None:
    if not bool(mask.any()):
        return None
    return float(plane[mask].mean().item())


def masked_image_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float | None]:
    """RGB / luma / high-frequency L1 on a co-visible mask. Invalid pixels are ignored."""

    if mask.ndim == 3:
        mask = mask[0]
    mask = mask.bool()
    residual = (pred - target).abs().mean(dim=0)
    luma_residual = (_luma(pred) - _luma(target)).abs()
    hf = (_highpass(pred) - _highpass(target)).abs().mean(dim=0)
    inner = _erode(mask, 3)
    return {
        "coverage": float(mask.float().mean().item()),
        "valid_pixels": int(mask.sum().item()),
        "rgb_l1": _masked_mean(residual, mask),
        "luma_l1": _masked_mean(luma_residual, mask),
        "hf_l1": _masked_mean(hf, inner),
        "rgb_l1_p90": float(residual[mask].quantile(0.9).item()) if bool(mask.any()) else None,
        "hf_l1_p90": float(hf[inner].quantile(0.9).item()) if bool(inner.any()) else None,
    }


def _crop_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, crops: dict[str, tuple[float, float, float, float]]) -> dict[str, Any]:
    height, width = int(pred.shape[-2]), int(pred.shape[-1])
    out = {}
    for name, frac in crops.items():
        x0, y0, x1, y1 = _fractional_crop(width, height, frac)
        region = torch.zeros_like(mask)
        region[y0:y1, x0:x1] = True
        combined = mask & region
        stats = masked_image_metrics(pred, target, combined)
        stats["box"] = [x0, y0, x1, y1]
        out[name] = stats
    return out


def _render_view(bundle, camera, *, background, kernel: float, embedding, max_level: int | None = None) -> dict[str, torch.Tensor]:
    kwargs = dict(
        appearance_embedding=embedding,
        lod=True,
        compact=False,
    )
    if max_level is not None:
        kwargs["max_level"] = max_level
    pkg = render_lod_appearance(bundle, camera, background=background, kernel_size=kernel, **kwargs)
    alpha = _spatial(pkg["render_alpha"])
    return {
        "rgb": pkg["render"].clamp(0, 1),
        "alpha": alpha,
        "camera_z": rade_pkg_camera_z(pkg, camera),
    }


def _scene_scale(camera, camera_z: torch.Tensor, alpha: torch.Tensor) -> float:
    opaque = alpha > 0.05
    if not bool(opaque.any()):
        return estimate_depth_scene_scale(camera, 1.0)
    median = float(camera_z[opaque].median().item())
    return estimate_depth_scene_scale(camera, median)


def warp_source_into_destination(
    *,
    dest_camera,
    source_camera,
    dest_depth: torch.Tensor,
    source_depth: torch.Tensor,
    dest_alpha: torch.Tensor,
    source_alpha: torch.Tensor,
    source_rgb: torch.Tensor,
    scene_scale: float,
    alpha_threshold: float = 0.05,
):
    """Unproject destination pixels and sample source RGB into that grid."""

    warp = build_reprojection_grid(
        dest_camera,
        source_camera,
        dest_depth,
        source_depth,
        source_alpha=source_alpha,
        target_alpha=dest_alpha,
        depth_scene_scale=scene_scale,
        alpha_threshold=alpha_threshold,
    )
    sampled = warp_image(source_rgb, warp)
    return warp, sampled


def _change_agreement(delta_a: torch.Tensor, delta_b: torch.Tensor, mask: torch.Tensor) -> dict[str, float | None]:
    mag_a = delta_a.abs().mean(dim=0)
    mag_b = delta_b.abs().mean(dim=0)
    residual = (delta_a - delta_b).abs().mean(dim=0)
    inner = _erode(mask, 3)
    hf = (_highpass(delta_a) - _highpass(delta_b)).abs().mean(dim=0)
    neighbor_mean = _masked_mean(mag_a, mask)
    warped_mean = _masked_mean(mag_b, mask)
    agreement = _masked_mean(residual, mask)
    denom = neighbor_mean if neighbor_mean else None
    return {
        "neighbor_change_rgb": neighbor_mean,
        "warped_change_rgb": warped_mean,
        "change_l1": agreement,
        "change_hf_l1": _masked_mean(hf, inner),
        "unexplained_ratio": None if not denom else float(agreement / max(denom, 1e-8)),
    }


def _project_flags(xyz: torch.Tensor, flags: torch.Tensor, camera, residual: torch.Tensor, valid: torch.Tensor, high: torch.Tensor) -> dict[str, Any]:
    uv, z = project_world_points(xyz, camera)
    width = int(camera.image_width)
    height = int(camera.image_height)
    u = uv[:, 0].round().long()
    v = uv[:, 1].round().long()
    in_frame = flags & (z > 1e-4) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not bool(in_frame.any()):
        return {"n_flagged": int(flags.sum().item()), "n_in_frame": 0}
    uu = u[in_frame].clamp(0, width - 1)
    vv = v[in_frame].clamp(0, height - 1)
    on_valid = valid[vv, uu]
    values = residual[vv, uu]
    high_hit = high[vv, uu]
    return {
        "n_flagged": int(flags.sum().item()),
        "n_in_frame": int(in_frame.sum().item()),
        "frac_on_covisible": float(on_valid.float().mean().item()),
        "mean_residual": float(values.mean().item()),
        "mean_residual_covisible": _masked_mean(values, on_valid),
        "frac_in_high_residual": float(high_hit.float().mean().item()),
        "frac_in_high_residual_covisible": float(high_hit[on_valid].float().mean().item()) if bool(on_valid.any()) else None,
    }


def _paint_points(image: torch.Tensor, u: torch.Tensor, v: torch.Tensor, color: tuple[float, float, float], radius: int = 1) -> torch.Tensor:
    painted = image.detach().clamp(0, 1).clone()
    height, width = painted.shape[-2], painted.shape[-1]
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            uu = (u + du).clamp(0, width - 1)
            vv = (v + dv).clamp(0, height - 1)
            painted[0, vv, uu] = color[0]
            painted[1, vv, uu] = color[1]
            painted[2, vv, uu] = color[2]
    return painted


def _save_residual(path: str, residual: torch.Tensor, mask: torch.Tensor, gain: float = 6.0) -> None:
    vis = residual.mean(dim=0).clamp(0, 1) * gain
    rgb = vis[None].expand(3, -1, -1).contiguous()
    rgb = rgb * mask.float()[None]
    save_tensor_image(rgb.clamp(0, 1), path)


def _compare_pair(
    *,
    dest_rgb: torch.Tensor,
    warped: torch.Tensor,
    mask: torch.Tensor,
    crops: dict[str, tuple[float, float, float, float]],
    occlusion_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    stats = masked_image_metrics(warped, dest_rgb, mask)
    stats["crops"] = _crop_metrics(warped, dest_rgb, mask, crops)
    if occlusion_mask is not None and bool(occlusion_mask.any()):
        stats["occlusion_boundary"] = masked_image_metrics(warped, dest_rgb, occlusion_mask)
    else:
        stats["occlusion_boundary"] = {"coverage": 0.0, "valid_pixels": 0, "rgb_l1": None, "luma_l1": None, "hf_l1": None}
    return stats


def _excess(current: dict[str, Any], floor: dict[str, Any]) -> dict[str, float | None]:
    def sub(a, b):
        if a is None or b is None:
            return None
        return float(a - b)

    payload = {
        "rgb_l1": sub(current.get("rgb_l1"), floor.get("rgb_l1")),
        "luma_l1": sub(current.get("luma_l1"), floor.get("luma_l1")),
        "hf_l1": sub(current.get("hf_l1"), floor.get("hf_l1")),
    }
    floor_hf = floor.get("hf_l1")
    payload["hf_ratio_to_floor"] = None if not floor_hf else float((current.get("hf_l1") or 0.0) / max(floor_hf, 1e-8))
    payload["crops"] = {}
    for name, rec in (current.get("crops") or {}).items():
        base = (floor.get("crops") or {}).get(name) or {}
        payload["crops"][name] = {
            "rgb_l1": sub(rec.get("rgb_l1"), base.get("rgb_l1")),
            "hf_l1": sub(rec.get("hf_l1"), base.get("hf_l1")),
        }
    occ = current.get("occlusion_boundary") or {}
    occ_floor = floor.get("occlusion_boundary") or {}
    payload["occlusion_boundary"] = {
        "rgb_l1": sub(occ.get("rgb_l1"), occ_floor.get("rgb_l1")),
        "hf_l1": sub(occ.get("hf_l1"), occ_floor.get("hf_l1")),
    }
    return payload


@torch.no_grad()
def neighbor_correspondence_report(
    bundle,
    base_camera,
    train_cameras,
    roi: NormalizedROI,
    *,
    background,
    kernel: float,
    gaussians,
    zoom_factor: float = 2.0,
    neighbor_k: int = 2,
    floor_max_level: int = 0,
    child_level: int = 1,
    crops: dict[str, tuple[float, float, float, float]] | None = None,
    out_dir: str | None = None,
) -> dict[str, Any]:
    """Project the target 2× render into neighbor 2× views and score co-visible residuals."""

    crops = ABSORPTION_CROPS if crops is None else crops
    target_zoom = make_zoom_camera(
        base_camera, roi, zoom_factor, uid=40000,
        image_name=f"{base_camera.image_name}_zoom{zoom_factor:g}_corr",
    )
    target_emb = embedding_for_train_camera(gaussians, base_camera.uid)
    target_l0 = _render_view(bundle, target_zoom, background=background, kernel=kernel, embedding=target_emb, max_level=floor_max_level)
    target_l1 = _render_view(bundle, target_zoom, background=background, kernel=kernel, embedding=target_emb)
    geometry = l1_geometry_tensors(bundle, child_level=child_level)
    neighbors = nearest_train_cameras(base_camera, train_cameras, k=neighbor_k)
    payload: dict[str, Any] = {
        "depth_kind": "camera_z",
        "depth_source": "rade_out_depth * rln_principal",
        "appearance_note": (
            "Own embeddings keep each view's frozen appearance. "
            "L0-RGB warp residual is the SH/embedding floor; excess over that "
            "floor is the correspondence signal. shared_target_embedding repeats "
            "the L0-depth comparison with the target embedding on both views."
        ),
        "floor_max_level": int(floor_max_level),
        "child_level": int(child_level),
        "target": base_camera.image_name,
        "neighbors": {},
    }
    if out_dir is not None:
        save_tensor_image(target_l1["rgb"], f"{out_dir}/target_l1.png")
        save_tensor_image(target_l0["rgb"], f"{out_dir}/target_l0.png")

    for index, camera in enumerate(neighbors):
        neighbor_zoom = make_zoom_camera(
            camera, roi, zoom_factor, uid=41000 + index,
            image_name=f"{camera.image_name}_zoom{zoom_factor:g}_corr",
        )
        own_emb = embedding_for_train_camera(gaussians, camera.uid)
        neigh_l0 = _render_view(bundle, neighbor_zoom, background=background, kernel=kernel, embedding=own_emb, max_level=floor_max_level)
        neigh_l1 = _render_view(bundle, neighbor_zoom, background=background, kernel=kernel, embedding=own_emb)
        neigh_l0_shared = _render_view(bundle, neighbor_zoom, background=background, kernel=kernel, embedding=target_emb, max_level=floor_max_level)
        neigh_l1_shared = _render_view(bundle, neighbor_zoom, background=background, kernel=kernel, embedding=target_emb)
        scale_l0 = _scene_scale(neighbor_zoom, neigh_l0["camera_z"], neigh_l0["alpha"])
        scale_l1 = _scene_scale(neighbor_zoom, neigh_l1["camera_z"], neigh_l1["alpha"])

        warp_l0, warped_l0 = warp_source_into_destination(
            dest_camera=neighbor_zoom, source_camera=target_zoom,
            dest_depth=neigh_l0["camera_z"], source_depth=target_l0["camera_z"],
            dest_alpha=neigh_l0["alpha"], source_alpha=target_l0["alpha"],
            source_rgb=target_l0["rgb"], scene_scale=scale_l0,
        )
        _, warped_l1_l0depth = warp_source_into_destination(
            dest_camera=neighbor_zoom, source_camera=target_zoom,
            dest_depth=neigh_l0["camera_z"], source_depth=target_l0["camera_z"],
            dest_alpha=neigh_l0["alpha"], source_alpha=target_l0["alpha"],
            source_rgb=target_l1["rgb"], scene_scale=scale_l0,
        )
        warp_l1, warped_l1_l1depth = warp_source_into_destination(
            dest_camera=neighbor_zoom, source_camera=target_zoom,
            dest_depth=neigh_l1["camera_z"], source_depth=target_l1["camera_z"],
            dest_alpha=neigh_l1["alpha"], source_alpha=target_l1["alpha"],
            source_rgb=target_l1["rgb"], scene_scale=scale_l1,
        )
        _, warped_l1_shared = warp_source_into_destination(
            dest_camera=neighbor_zoom, source_camera=target_zoom,
            dest_depth=neigh_l0["camera_z"], source_depth=target_l0["camera_z"],
            dest_alpha=neigh_l0["alpha"], source_alpha=target_l0["alpha"],
            source_rgb=target_l1["rgb"], scene_scale=scale_l0,
        )

        mask_l0 = warp_l0.valid_mask
        mask_l1 = warp_l1.valid_mask
        occlusion_l0 = _dilate(mask_l0, 5) ^ _erode(mask_l0, 5)
        occlusion_l1 = _dilate(mask_l1, 5) ^ _erode(mask_l1, 5)
        floor = _compare_pair(dest_rgb=neigh_l0["rgb"], warped=warped_l0, mask=mask_l0, crops=crops, occlusion_mask=occlusion_l0)
        appearance = _compare_pair(dest_rgb=neigh_l1["rgb"], warped=warped_l1_l0depth, mask=mask_l0, crops=crops, occlusion_mask=occlusion_l0)
        trained = _compare_pair(dest_rgb=neigh_l1["rgb"], warped=warped_l1_l1depth, mask=mask_l1, crops=crops, occlusion_mask=occlusion_l1)
        floor_shared = _compare_pair(dest_rgb=neigh_l0_shared["rgb"], warped=warped_l0, mask=mask_l0, crops=crops, occlusion_mask=occlusion_l0)
        appearance_shared = _compare_pair(dest_rgb=neigh_l1_shared["rgb"], warped=warped_l1_shared, mask=mask_l0, crops=crops, occlusion_mask=occlusion_l0)
        change = _change_agreement(
            neigh_l1["rgb"] - neigh_l0["rgb"],
            warped_l1_l0depth - warped_l0,
            mask_l0,
        )
        residual_app = (warped_l1_l0depth - neigh_l1["rgb"]).abs().mean(dim=0)
        high = torch.zeros_like(mask_l0)
        if bool(mask_l0.any()):
            thresh = residual_app[mask_l0].quantile(0.9)
            high = mask_l0 & (residual_app >= thresh)
        all_l1 = None
        outliers = {}
        if geometry is not None:
            all_l1 = geometry["opaque"]
            residual_for_points = residual_app
            outliers = {
                "all_opaque_l1": _project_flags(geometry["xyz"], all_l1, neighbor_zoom, residual_for_points, mask_l0, high),
                "scale_gt_4x_parent": _project_flags(geometry["xyz"], geometry["scale_gt_4x_parent"], neighbor_zoom, residual_for_points, mask_l0, high),
                "offset_gt_4x_parent": _project_flags(geometry["xyz"], geometry["offset_gt_4x_parent"], neighbor_zoom, residual_for_points, mask_l0, high),
            }
            either = geometry["scale_gt_4x_parent"] | geometry["offset_gt_4x_parent"]
            outliers["scale_or_offset_gt_4x"] = _project_flags(geometry["xyz"], either, neighbor_zoom, residual_for_points, mask_l0, high)
            all_frac = outliers["all_opaque_l1"].get("frac_in_high_residual_covisible")
            for key in ("scale_gt_4x_parent", "offset_gt_4x_parent", "scale_or_offset_gt_4x"):
                hit = outliers[key].get("frac_in_high_residual_covisible")
                outliers[key]["concentration_vs_all_l1"] = (
                    None if all_frac in (None, 0) or hit is None else float(hit / max(all_frac, 1e-8))
                )

        rec = {
            "l0_depth": {
                "coverage": float(mask_l0.float().mean().item()),
                "depth_scene_scale": scale_l0,
                "warp_metadata": warp_l0.metadata,
                "floor_l0_rgb": floor,
                "l1_rgb": appearance,
                "excess_over_floor": _excess(appearance, floor),
                "change_agreement": change,
                "shared_target_embedding": {
                    "floor_l0_rgb": floor_shared,
                    "l1_rgb": appearance_shared,
                    "excess_over_floor": _excess(appearance_shared, floor_shared),
                },
            },
            "trained_depth": {
                "coverage": float(mask_l1.float().mean().item()),
                "depth_scene_scale": scale_l1,
                "warp_metadata": warp_l1.metadata,
                "l1_rgb": trained,
                "coverage_delta_vs_l0_depth": float(mask_l1.float().mean().item() - mask_l0.float().mean().item()),
                "frac_newly_occluded": float((mask_l0 & ~mask_l1).float().mean().item()),
                "frac_newly_visible": float((~mask_l0 & mask_l1).float().mean().item()),
            },
            "outliers_on_l0depth_l1_residual": outliers,
            "neighbor_vs_l0_direct": {
                "rgb_l1": masked_image_metrics(neigh_l1["rgb"], neigh_l0["rgb"], torch.ones_like(mask_l0))["rgb_l1"],
                "hf_l1": masked_image_metrics(neigh_l1["rgb"], neigh_l0["rgb"], torch.ones_like(mask_l0))["hf_l1"],
                "rgb_l1_covisible_l0depth": masked_image_metrics(neigh_l1["rgb"], neigh_l0["rgb"], mask_l0)["rgb_l1"],
                "hf_l1_covisible_l0depth": masked_image_metrics(neigh_l1["rgb"], neigh_l0["rgb"], mask_l0)["hf_l1"],
            },
        }
        payload["neighbors"][camera.image_name] = rec

        if out_dir is not None:
            tag = f"neighbor_{camera.image_name}"
            save_tensor_image(neigh_l1["rgb"], f"{out_dir}/{tag}_direct_l1.png")
            save_tensor_image(neigh_l0["rgb"], f"{out_dir}/{tag}_direct_l0.png")
            save_tensor_image(warped_l1_l0depth, f"{out_dir}/{tag}_warped_target_l1_l0depth.png")
            save_tensor_image(warped_l1_l1depth, f"{out_dir}/{tag}_warped_target_l1_trained_depth.png")
            _save_residual(f"{out_dir}/{tag}_residual_floor_x6.png", (warped_l0 - neigh_l0["rgb"]).abs(), mask_l0)
            _save_residual(f"{out_dir}/{tag}_residual_l0depth_x6.png", (warped_l1_l0depth - neigh_l1["rgb"]).abs(), mask_l0)
            _save_residual(f"{out_dir}/{tag}_residual_trained_depth_x6.png", (warped_l1_l1depth - neigh_l1["rgb"]).abs(), mask_l1)
            save_tensor_image(mask_l0.float()[None].expand(3, -1, -1), f"{out_dir}/{tag}_mask_l0depth.png")
            save_tensor_image(mask_l1.float()[None].expand(3, -1, -1), f"{out_dir}/{tag}_mask_trained_depth.png")
            _save_residual(f"{out_dir}/{tag}_residual_occlusion_boundary_x6.png", (warped_l1_l0depth - neigh_l1["rgb"]).abs(), occlusion_l0)
            for name, frac in crops.items():
                x0, y0, x1, y1 = _fractional_crop(int(neigh_l1["rgb"].shape[-1]), int(neigh_l1["rgb"].shape[-2]), frac)
                crop_mask = torch.zeros_like(mask_l0)
                crop_mask[y0:y1, x0:x1] = True
                _save_residual(
                    f"{out_dir}/{tag}_residual_{name}_x6.png",
                    (warped_l1_l0depth - neigh_l1["rgb"]).abs(),
                    mask_l0 & crop_mask,
                )
            if geometry is not None:
                either = geometry["scale_gt_4x_parent"] | geometry["offset_gt_4x_parent"]
                uv, z = project_world_points(geometry["xyz"], neighbor_zoom)
                u = uv[:, 0].round().long()
                v = uv[:, 1].round().long()
                vis = residual_app.clamp(0, 1).mul(6.0)[None].expand(3, -1, -1).contiguous() * mask_l0.float()[None]
                in_frame = either & (z > 1e-4) & (u >= 0) & (u < vis.shape[-1]) & (v >= 0) & (v < vis.shape[-2])
                overlay = _paint_points(vis, u[in_frame], v[in_frame], (1.0, 0.15, 0.1), radius=1)
                save_tensor_image(overlay.clamp(0, 1), f"{out_dir}/{tag}_outliers_on_residual.png")

    return payload
