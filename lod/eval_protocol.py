"""JAX_068 reconstruction eval: native GT pixels, never upsampled GT.

1x full-frame metrics measure Stage1 reconstruction. They do not show L1/L2
gain, because new layers are weighted off at the capture scale.

LoD gain is measured on a focal-zoom camera (where new layers can fire) after
area-downsampling the zoom raster to the native crop of the real image. That
crop is the highest resolution the photograph actually supports.

Forbidden: interpolate a 1x photograph up to the zoom raster and treat that as
high-res ground truth.

Absorption vs generated supervision stays a training diagnostic, not a
reconstruction score. Test images are excluded from prompts, supervision, and
training.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from utils.loss_utils import create_window, ssim
from utils.zoom_camera import NormalizedROI
from utils.zoom_mvp_utils import image_l1


SSIM_WINDOW = 11
SSIM_ERODE_RADIUS = SSIM_WINDOW // 2

# Official scores normalize by valid pixels (MAE / PSNR) or by a spatial
# support that excludes neighborhoods mixed with invalid pixels (SSIM / LPIPS).
# Zero-filling the invalid region and then averaging the whole tensor dilutes
# MAE/PSNR and lets SSIM/LPIPS see black borders. Those numbers are kept as a
# labeled companion, not the official reconstruction score.
MASK_POLICY = {
    "name": "valid_region_mae_psnr_eroded_ssim_lpips",
    "l1": "sum_M_abs_over_3_sum_M",
    "psnr": "mse_sum_M_sq_over_3_sum_M_then_minus_10_log10",
    "ssim": "mean_ssim_map_on_mask_eroded_by_window_radius",
    "ssim_window": SSIM_WINDOW,
    "ssim_erode_radius": SSIM_ERODE_RADIUS,
    "lpips": "mean_spatial_lpips_on_downsampled_mask",
    "report_valid_coverage": True,
    "zero_filled": {
        "label": "zero_filled_full_image_mean",
        "invalid_pixels": "multiplied_by_zero",
        "note": (
            "Companion only. Mean over the whole tensor after zeroing invalid "
            "pixels. Dilutes MAE/PSNR; SSIM/LPIPS neighborhoods see black."
        ),
    },
}

# LoD gain vs L0 must share RaDe-GS and frozen Skyfall appearance. The original
# Skyfall diff_gauss 1x table is listed separately so renderer drift is not
# counted as an algorithm gain.
LOD_GAIN_BASELINE = {
    "rasterizer": "rade_gs",
    "appearance": "frozen_skyfall_mlp_and_embeddings",
    "compare_against": "same_path_l0_empty_detail",
    "skyfall_diff_gauss": "list_separately_not_in_lod_gain",
}

OVERLAP_WINDOWS = {
    "report_per_view_and_per_window": True,
    "not_independent_samples": True,
    "do_not_compute": ["success_rate_over_overlapping_windows"],
}

PROTOCOL = {
    "scene": "JAX_068",
    "reconstruction_gt": "native_image_pixels_only",
    "upsample_gt": False,
    "full_1x_measures": "stage1_reconstruction_not_lod_gain",
    "lod_eval": "focal_zoom_render_area_downsampled_to_native_crop",
    "eval_scale_note": (
        "Downsampled zoom metrics are reconstruction quality at the real "
        "photograph crop. A gain there does not by itself prove that new "
        "high-frequency detail in the 2048 zoom raster is real."
    ),
    "test_images_excluded_from": ["prompt", "supervision", "training"],
    "absorption_vs_generated": "recorded_not_reconstruction_metric",
    "mask_policy": MASK_POLICY,
    "lod_gain_baseline": LOD_GAIN_BASELINE,
    "overlap_windows": OVERLAP_WINDOWS,
    "full_scene_after_this_gate": {
        "joint_l1_then_joint_l2": True,
        "stitch_independent_roi_models": False,
        "reuse_single_roi_50k_budget": False,
        "start_checkpoint": "skyfall-gs_exp/stage1/JAX_068/chkpnt30000.pth",
        "geometry_mainline_only": True,
        "no_spynet_ablation": True,
        "no_8x": True,
        "no_densify_ablation": True,
    },
}

# Stage1 1x crop vs focal-zoom raster downsampled to that crop. Rasterizer and
# integer-box rounding, not LoD. Tuned after the GPU probe; fail closed.
ALIGN_RGB_L1_MAX = 0.01


def native_crop_hw(width: int, height: int, zoom_factor: float) -> tuple[int, int]:
    """Integer native crop size. Zoom must divide the capture raster."""

    zoom = float(zoom_factor)
    if zoom <= 1.0:
        raise ValueError(f"eval zoom must be > 1, got {zoom}")
    crop_w = width / zoom
    crop_h = height / zoom
    if abs(crop_w - round(crop_w)) > 1e-6 or abs(crop_h - round(crop_h)) > 1e-6:
        raise ValueError(
            f"zoom {zoom:g} does not divide {width}x{height}; "
            "refuse a non-integer native crop rather than resample GT"
        )
    return int(round(crop_h)), int(round(crop_w))


def native_crop_box(
    width: int,
    height: int,
    roi: NormalizedROI,
    zoom_factor: float,
) -> tuple[int, int, int, int]:
    """Inclusive-exclusive pixel box (left, top, right, bottom) at 1x."""

    roi.validate_for_zoom(zoom_factor)
    crop_h, crop_w = native_crop_hw(width, height, zoom_factor)
    cx = float(roi.center_x) * float(width)
    cy = float(roi.center_y) * float(height)
    left = int(round(cx - crop_w / 2.0))
    top = int(round(cy - crop_h / 2.0))
    right = left + crop_w
    bottom = top + crop_h
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError(
            f"native crop [{left}:{right}]x[{top}:{bottom}] escapes {width}x{height}"
        )
    return left, top, right, bottom


def crop_chw(image: torch.Tensor, box: Sequence[int]) -> torch.Tensor:
    left, top, right, bottom = (int(v) for v in box)
    return image[..., top:bottom, left:right]


def forbid_gt_upsample(src_h: int, src_w: int, dst_h: int, dst_w: int) -> None:
    if dst_h > src_h or dst_w > src_w:
        raise ValueError(
            "interpolated upsampled GT is not a real high-resolution reference "
            f"(src {src_w}x{src_h} -> dst {dst_w}x{dst_h})"
        )


def downsample_render_to_native(render: torch.Tensor, native_h: int, native_w: int) -> torch.Tensor:
    """Area-downsample a zoom raster onto the native GT crop. Never grows GT."""

    if render.ndim != 3:
        raise ValueError(f"expected CHW render, got {tuple(render.shape)}")
    src_h, src_w = int(render.shape[-2]), int(render.shape[-1])
    forbid_gt_upsample(src_h, src_w, native_h, native_w)
    if src_h == native_h and src_w == native_w:
        return render
    return F.interpolate(
        render.unsqueeze(0).float(), size=(native_h, native_w), mode="area",
    )[0]


def apply_mask(rgb: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return rgb
    keep = mask.detach().float()
    if keep.ndim == 2:
        keep = keep.unsqueeze(0)
    if keep.shape[0] == 1 and rgb.shape[0] == 3:
        keep = keep.expand(3, -1, -1)
    return rgb * keep.to(device=rgb.device, dtype=rgb.dtype)


def _mask_hw(mask: torch.Tensor | None, height: int, width: int, device, dtype) -> torch.Tensor:
    if mask is None:
        return torch.ones((height, width), device=device, dtype=dtype)
    keep = mask.detach().to(device=device, dtype=dtype)
    if keep.ndim == 3:
        keep = keep[0]
    if tuple(keep.shape) != (height, width):
        keep = F.interpolate(keep[None, None], size=(height, width), mode="nearest")[0, 0]
    return keep


def erode_mask(mask_hw: torch.Tensor, radius: int) -> torch.Tensor:
    """Pixels whose SSIM/LPIPS neighborhood is entirely valid."""

    if int(radius) <= 0:
        return (mask_hw > 0.5).to(dtype=mask_hw.dtype)
    kernel = 2 * int(radius) + 1
    invalid = (mask_hw <= 0.5).float()[None, None]
    grown = F.max_pool2d(invalid, kernel_size=kernel, stride=1, padding=int(radius))[0, 0]
    return (grown < 0.5).to(dtype=mask_hw.dtype)


def ssim_map_hw(pred: torch.Tensor, target: torch.Tensor, *, window_size: int = SSIM_WINDOW) -> torch.Tensor:
    """Channel-mean SSIM map, same window as ``utils.loss_utils.ssim``."""

    img1 = pred.unsqueeze(0).float()
    img2 = target.unsqueeze(0).float()
    channel = int(img1.size(-3))
    window = create_window(window_size, channel)
    window = window.to(device=img1.device, dtype=img1.dtype)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean(dim=1)[0]


def valid_mae_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask_hw: torch.Tensor,
) -> tuple[float, float, int]:
    """MAE and PSNR normalized by valid pixels, not by the full raster.

    MAE = sum_{p,c} M_p |I-G| / (3 sum_p M_p)
    MSE = sum_{p,c} M_p (I-G)^2 / (3 sum_p M_p)
    PSNR = -10 log10(MSE) for images in [0, 1].
    """

    keep = (mask_hw > 0.5).to(dtype=pred.dtype)
    n_valid = int(keep.sum().item())
    if n_valid <= 0:
        return float("nan"), float("nan"), 0
    diff = (pred - target).abs() * keep.unsqueeze(0)
    mae = float(diff.sum().item() / (3.0 * n_valid))
    mse = float(((pred - target).pow(2) * keep.unsqueeze(0)).sum().item() / (3.0 * n_valid))
    psnr = float("inf") if mse <= 0.0 else float(-10.0 * math.log10(mse))
    return mae, psnr, n_valid


def reconstruction_metrics(
    render_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    lpips_fn=None,
    lpips_spatial_fn=None,
) -> dict[str, Any]:
    """Official valid-region scores plus labeled zero-filled companions."""

    rgb = render_rgb.clamp(0, 1).float()
    target = gt_rgb.clamp(0, 1).float()
    height, width = int(rgb.shape[-2]), int(rgb.shape[-1])
    mask_hw = _mask_hw(mask, height, width, rgb.device, rgb.dtype)
    n_pixels = height * width
    mae, psnr_valid, n_valid = valid_mae_psnr(rgb, target, mask_hw)
    zero_rgb = apply_mask(rgb, mask_hw)
    zero_target = apply_mask(target, mask_hw)
    zero_filled: dict[str, Any] = {
        "label": MASK_POLICY["zero_filled"]["label"],
        "l1": image_l1(zero_rgb, zero_target),
        "psnr": float(
            (-10.0 * math.log10(((zero_rgb - zero_target).pow(2).mean().clamp_min(1e-12)).item()))
            if n_pixels
            else float("nan")
        ),
        "ssim": float(ssim(zero_rgb.unsqueeze(0), zero_target.unsqueeze(0)).item()),
    }
    ssim_hw = ssim_map_hw(rgb, target)
    ssim_keep = erode_mask(mask_hw, SSIM_ERODE_RADIUS)
    ssim_n = int((ssim_keep > 0.5).sum().item())
    ssim_valid = (
        float(ssim_hw[ssim_keep > 0.5].mean().item()) if ssim_n else float("nan")
    )
    row: dict[str, Any] = {
        "l1": mae,
        "psnr": psnr_valid,
        "ssim": ssim_valid,
        "valid_coverage": float(n_valid / max(n_pixels, 1)),
        "n_valid_pixels": n_valid,
        "n_pixels": n_pixels,
        "ssim_valid_coverage": float(ssim_n / max(n_pixels, 1)),
        "ssim_erode_radius": SSIM_ERODE_RADIUS,
        "zero_filled": zero_filled,
    }
    if lpips_fn is not None:
        pred = zero_rgb.unsqueeze(0) * 2.0 - 1.0
        ref = zero_target.unsqueeze(0) * 2.0 - 1.0
        zero_filled["lpips"] = float(lpips_fn(pred, ref).mean().item())
    spatial = lpips_spatial_fn if lpips_spatial_fn is not None else (
        lpips_fn if lpips_fn is not None and bool(getattr(lpips_fn, "spatial", False)) else None
    )
    if spatial is not None:
        pred = rgb.unsqueeze(0) * 2.0 - 1.0
        ref = target.unsqueeze(0) * 2.0 - 1.0
        lp_map = spatial(pred, ref)
        if lp_map.ndim == 4:
            lp_map = lp_map[0, 0]
        lp_keep = F.interpolate(
            mask_hw[None, None], size=tuple(lp_map.shape[-2:]), mode="area",
        )[0, 0]
        lp_n = int((lp_keep > 0.5).sum().item())
        row["lpips"] = float(lp_map[lp_keep > 0.5].mean().item()) if lp_n else float("nan")
        row["lpips_valid_coverage"] = float(lp_n / max(int(lp_map.numel()), 1))
    return row


def eval_roi_center() -> NormalizedROI:
    """Centered window: 2x/4x divide 2048 with no clamp."""

    return NormalizedROI(0.5, 0.5, 0.1, 0.1)


def protocol_payload(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(PROTOCOL)
    if extra:
        payload.update(dict(extra))
    return payload
