"""Classify geometry-warp pixels that stay valid despite a visual mismatch.

Reporting thresholds only. Do not feed these back into the global warp test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from refinement.geometry_warp import (
    camera_to_world,
    compute_depth_tolerance,
    project_world_points,
)


# Same windows as train_zoom_gen.ABSORPTION_CROPS, plus a roof-eave strip.
GHOSTING_CROPS = {
    "trees": (0.34, 0.37, 0.63, 0.61),
    "eaves": (0.10, 0.46, 0.44, 0.74),
    "vehicles": (0.24, 0.62, 0.78, 0.98),
}

# Overlay: invalid, ok, mixed-surface ghost, opaque ghost, bad roundtrip, near slack.
CLASS_COLORS = {
    "invalid": (0, 0, 0),
    "ok": (48, 48, 48),
    "ghosted_mixed": (220, 200, 40),
    "ghosted_opaque": (230, 140, 40),
    "ghosted_roundtrip": (210, 40, 40),
    "ghosted_near_slack": (200, 40, 200),
}


def _unproject_pixel_grid(camera, depth: torch.Tensor) -> torch.Tensor:
    from refinement.geometry_warp import _unproject_pixel_grid as _impl

    return _impl(camera, depth)


def fractional_crop(width: int, height: int, box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x0 = int(round(box[0] * width))
    y0 = int(round(box[1] * height))
    x1 = int(round(box[2] * width))
    y1 = int(round(box[3] * height))
    return max(0, x0), max(0, y0), min(width, max(x0 + 1, x1)), min(height, max(y0 + 1, y1))


def sample_plane(plane: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    return F.grid_sample(
        plane.float()[None, None],
        grid.float()[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0]


def depth_consistency_maps(
    dest_camera,
    dest_depth: torch.Tensor,
    source_camera,
    source_depth: torch.Tensor,
    *,
    scene_scale: float,
    abs_tolerance: float = 0.05,
    rel_tolerance: float = 0.02,
) -> dict[str, torch.Tensor]:
    """Dest-unprojected Z vs sampled source depth. Positive slack is closer to rejection."""

    dest_depth = dest_depth.float()
    source_depth = source_depth.float()
    height, width = dest_depth.shape
    src_h, src_w = source_depth.shape
    world = camera_to_world(_unproject_pixel_grid(dest_camera, dest_depth), dest_camera)
    uv, source_z = project_world_points(world, source_camera)
    grid = torch.stack(
        (
            uv[:, 0] * (2.0 / max(src_w - 1, 1)) - 1.0,
            uv[:, 1] * (2.0 / max(src_h - 1, 1)) - 1.0,
        ),
        dim=-1,
    ).reshape(height, width, 2)
    sampled = sample_plane(source_depth, grid)
    projected_z = source_z.reshape(height, width)
    tolerance = compute_depth_tolerance(
        sampled, abs_tolerance=abs_tolerance, rel_tolerance=rel_tolerance, scene_scale=scene_scale,
    )
    residual = projected_z - sampled
    slack = residual / tolerance.clamp_min(1e-8)
    return {
        "sampled_source_z": sampled,
        "projected_source_z": projected_z,
        "depth_residual": residual,
        "depth_tolerance": tolerance,
        "slack_ratio": slack,
        "grid": grid,
    }


def rgb_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean(dim=0)


def upsample_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    plane = mask.float()
    while plane.ndim > 2:
        plane = plane[0]
    return F.interpolate(plane[None, None], size=(height, width), mode="nearest")[0, 0] > 0.5


def _as_hw_bool(mask: torch.Tensor) -> torch.Tensor:
    plane = mask.bool()
    while plane.ndim > 2:
        plane = plane[0]
    return plane


def feature_readmission_stats(
    pixel_region: torch.Tensor,
    feat_valid: torch.Tensor,
    *,
    pixel_valid: torch.Tensor | None = None,
    downsample: int | None = None,
) -> dict[str, Any]:
    """How much of a pixel region still sits in a valid feature cell.

    ``pixels_in_valid_feature_cell_frac`` answers: after aggregation, how many
    of these pixels are still treated as valid at feature resolution.
    If ``pixel_valid`` is the post-gate pixel mask, ``readmitted_pixel_frac``
    is the share of gated-off region pixels that a mixed 8x8 cell put back.
    """

    region = _as_hw_bool(pixel_region)
    feat = _as_hw_bool(feat_valid)
    height, width = region.shape
    feat_h, feat_w = feat.shape
    if downsample is None:
        if height % feat_h or width % feat_w:
            raise ValueError(
                f"pixel {tuple(region.shape)} is not an integer multiple of feature {tuple(feat.shape)}"
            )
        downsample = height // feat_h
        if width // feat_w != downsample:
            raise ValueError("non-square downsample is not supported")
    up = upsample_mask(feat, height, width)
    cell_has_region = (
        F.max_pool2d(region.float()[None, None], kernel_size=downsample, stride=downsample)[0, 0] > 0
    )
    if tuple(cell_has_region.shape) != tuple(feat.shape):
        cell_has_region = F.interpolate(
            cell_has_region.float()[None, None], size=tuple(feat.shape), mode="nearest"
        )[0, 0] > 0.5
    n_pix = int(region.sum().item())
    n_pix_in_valid = int((region & up).sum().item())
    n_cells = int(cell_has_region.sum().item())
    n_cells_valid = int((cell_has_region & feat).sum().item())
    stats: dict[str, Any] = {
        "downsample": int(downsample),
        "pixel_count": n_pix,
        "pixels_in_valid_feature_cell": n_pix_in_valid,
        "pixels_in_valid_feature_cell_frac": None if n_pix == 0 else n_pix_in_valid / n_pix,
        "overlapping_feature_cells": n_cells,
        "overlapping_feature_cells_valid": n_cells_valid,
        "overlapping_feature_cells_valid_frac": None if n_cells == 0 else n_cells_valid / n_cells,
        "readmitted_pixels": None,
        "readmitted_pixel_frac": None,
    }
    if pixel_valid is not None:
        gated_off = region & ~_as_hw_bool(pixel_valid)
        n_off = int(gated_off.sum().item())
        n_readmit = int((gated_off & up).sum().item())
        stats["gated_off_pixels"] = n_off
        stats["readmitted_pixels"] = n_readmit
        stats["readmitted_pixel_frac"] = None if n_off == 0 else n_readmit / n_off
    return stats


def classify_allowed_errors(
    valid: torch.Tensor,
    rgb_residual: torch.Tensor,
    roundtrip: torch.Tensor,
    slack_ratio: torch.Tensor,
    dest_alpha: torch.Tensor,
    sampled_alpha: torch.Tensor,
    *,
    rgb_thresh: float = 0.05,
    roundtrip_thresh: float = 1.0,
    mixed_lo: float = 0.2,
    mixed_hi: float = 0.8,
    slack_near: float = 0.5,
    alpha_disagree: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Label valid pixels. Thresholds are for the report, not a new warp gate."""

    valid = valid.bool()
    ghosted = valid & (rgb_residual > rgb_thresh)
    mixed = (dest_alpha > mixed_lo) & (dest_alpha < mixed_hi)
    disagree = (dest_alpha - sampled_alpha).abs() > alpha_disagree
    rt_finite = torch.isfinite(roundtrip)
    rt_ok = rt_finite & (roundtrip <= roundtrip_thresh)
    rt_bad = rt_finite & (roundtrip > roundtrip_thresh)
    near = slack_ratio > slack_near
    return {
        "ghosted": ghosted,
        "ghosted_mixed": ghosted & (mixed | disagree) & rt_ok,
        "ghosted_opaque": ghosted & ~(mixed | disagree) & rt_ok,
        "ghosted_roundtrip": ghosted & (rt_bad | ~rt_finite),
        "ghosted_no_roundtrip": ghosted & ~rt_finite,
        "ghosted_near_slack": ghosted & near,
        "valid_roundtrip_hit": valid & rt_finite,
    }


def class_id_map(masks: dict[str, torch.Tensor], valid: torch.Tensor) -> torch.Tensor:
    """Priority: bad roundtrip > near slack > mixed > opaque > ok > invalid."""

    ids = torch.zeros(valid.shape, dtype=torch.uint8, device=valid.device)
    ids[valid] = 1
    ids[masks["ghosted_opaque"]] = 3
    ids[masks["ghosted_mixed"]] = 2
    ids[masks["ghosted_near_slack"]] = 5
    ids[masks["ghosted_roundtrip"]] = 4
    return ids


def masked_stats(plane: torch.Tensor, mask: torch.Tensor) -> dict[str, float | None]:
    if not bool(mask.any()):
        return {"count": 0, "mean": None, "p50": None, "p90": None}
    values = plane[mask].float()
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "p50": float(values.median().item()),
        "p90": float(values.quantile(0.9).item()),
    }


def region_report(
    *,
    valid: torch.Tensor,
    feat_valid: torch.Tensor,
    rgb_residual: torch.Tensor,
    roundtrip: torch.Tensor,
    slack_ratio: torch.Tensor,
    dest_alpha: torch.Tensor,
    classes: dict[str, torch.Tensor],
    region: torch.Tensor | None = None,
) -> dict[str, Any]:
    select = valid if region is None else valid & region
    n_region = int(region.sum().item()) if region is not None else int(valid.numel())
    n_valid = int(select.sum().item())
    n_feat = int((feat_valid & (region if region is not None else torch.ones_like(valid))).sum().item())
    ghosted = classes["ghosted"] & select
    out = {
        "pixels": n_region,
        "valid_frac": None if n_region == 0 else n_valid / n_region,
        "feature_valid_frac": None if n_region == 0 else n_feat / n_region,
        "ghosted_frac_of_valid": None if n_valid == 0 else float(ghosted.float().sum().item() / n_valid),
        "rgb": masked_stats(rgb_residual, select),
        "roundtrip_px": masked_stats(roundtrip, select & torch.isfinite(roundtrip)),
        "slack_ratio": masked_stats(slack_ratio, select),
        "dest_alpha": masked_stats(dest_alpha, select),
        "classes_frac_of_valid": {},
    }
    for name in ("ghosted_mixed", "ghosted_opaque", "ghosted_roundtrip", "ghosted_no_roundtrip", "ghosted_near_slack"):
        out["classes_frac_of_valid"][name] = (
            None if n_valid == 0 else float((classes[name] & select).float().sum().item() / n_valid)
        )
    leaked = (~valid) & feat_valid
    if region is not None:
        leaked = leaked & region
    out["feature_leaked_frac"] = None if n_region == 0 else float(leaked.float().sum().item() / n_region)
    return out


def colorize_scalar(plane: torch.Tensor, *, vmax: float, mask: torch.Tensor | None = None) -> Image.Image:
    values = plane.detach().float().cpu().numpy()
    norm = np.clip(values / max(float(vmax), 1e-8), 0.0, 1.0)
    r = np.clip(1.5 * norm, 0.0, 1.0)
    g = np.clip(1.5 * norm - 0.25, 0.0, 1.0)
    b = np.clip(2.0 * norm - 1.0, 0.0, 1.0)
    rgb = np.stack((r, g, b), axis=-1)
    if mask is not None:
        rgb[~mask.detach().cpu().numpy()] = 0.0
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def tensor_to_image(rgb: torch.Tensor) -> Image.Image:
    plane = rgb.detach().float().clamp(0, 1).cpu()
    if plane.ndim == 2:
        plane = plane[None].expand(3, -1, -1)
    array = (plane.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def class_overlay(ids: torch.Tensor) -> Image.Image:
    table = np.array(
        [
            CLASS_COLORS["invalid"],
            CLASS_COLORS["ok"],
            CLASS_COLORS["ghosted_mixed"],
            CLASS_COLORS["ghosted_opaque"],
            CLASS_COLORS["ghosted_roundtrip"],
            CLASS_COLORS["ghosted_near_slack"],
        ],
        dtype=np.uint8,
    )
    return Image.fromarray(table[ids.detach().cpu().numpy()], mode="RGB")


def _caption(image: Image.Image, text: str) -> Image.Image:
    pad = 22
    canvas = Image.new("RGB", (image.width, image.height + pad), (16, 16, 16))
    canvas.paste(image, (0, pad))
    ImageDraw.Draw(canvas).text((4, 4), text, fill=(230, 230, 230))
    return canvas


def contact_sheet(panels: list[tuple[str, Image.Image]], *, columns: int = 5) -> Image.Image:
    labeled = [_caption(image, title) for title, image in panels]
    width, height = labeled[0].size
    rows = (len(labeled) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), (8, 8, 8))
    for index, image in enumerate(labeled):
        row, col = divmod(index, columns)
        sheet.paste(image, (col * width, row * height))
    return sheet


def crop_image(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = box
    return image.crop((x0, y0, x1, y1))
