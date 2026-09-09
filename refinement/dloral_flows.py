"""Geometry correspondence → DLoRAL CFR ``external_flows``.

Coordinate and direction contract
---------------------------------
Pixel origin is the **top-left pixel center** ``(u, v) = (0, 0)``.  Integer
locations match ``geometry_warp._unproject_pixel_grid``.  Image-space flow is

    flow_{i→j}(p_i) = p_j - p_i

where ``p_j`` is the source sample for target pixel ``p_i``, obtained with
the already-verified reprojection (depth type and R/T/K as in
``build_reprojection_grid``):

    p_j = π(K_j T_j T_i^{-1} [D_i(p_i) K_i^{-1} p_i])

CFR frames are ``[neighbor, target]``.  Fusion warps the **previous**
(neighbor) feature onto the **current** (target) grid with ``flows_forward``:

    aligned = flow_warp(neighbor_feat, flows_forward)  # target-pixel grid
    flows_forward(p_target) = p_neighbor - p_target

That is the same vector as stored ``pixel_flow`` / ``target_to_source_flow``.
``flow_warp`` uses ``align_corners=True``: feature coordinate ``0`` maps to
``-1`` and ``W_f - 1`` maps to ``+1``.  Feature-scale displacement is
``flow_image / 8`` so one VAE cell stays one SpyNet/CFR pixel (block indexing,
matching native SpyNet units on the ``scale_factor=0.125`` clip).

Image resize (LANCZOS in the worker) scales displacements by ``dst/src``, not
by bilinear-blending NaN.  Validity is a separate nearest-neighbor field.
Depth edges that mix foreground and background in one VAE cell are marked
invalid.  Invalid network samples use a numerically safe offset of 0; the
mask, not the offset, disables neighbor contribution:

    F_i^{out} = M_i ⊙ F_i^{fused} + (1 - M_i) ⊙ F_i

Invalid keys/values are zeroed **before** cross-frame attention so they cannot
contaminate valid queries.  Fully invalid input therefore falls back to the
target feature.

Bidirectional correspondence uses **each view's own depth**.  Scattering
``-flow`` onto the neighbor grid is a diagnostic tool only
(``invert_feature_flow``); it is not the production reverse field.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .types import GeometryCorrespondence, MultiViewInput

VAE_DOWNSAMPLE = 8
FLOW_UNITS = "source_feature_pixel - target_feature_pixel"
ALIGN_CORNERS = True
FRAME_ORDER = ("neighbor", "target")
FLOWS_FORWARD_MEANING = "p_neighbor - p_target at target feature pixels"


def prepared_image_size(
    width: int,
    height: int,
    *,
    process_size: int = 512,
    upscale: int = 1,
) -> tuple[int, int]:
    """Match ``dloral_worker._prepare_pair`` output size ``(width, height)``."""

    if width < process_size // upscale or height < process_size // upscale:
        scale = (process_size // upscale) / min(width, height)
        width = int(scale * width)
        height = int(scale * height)
    if upscale != 1:
        width *= upscale
        height *= upscale
    width = width - width % 8
    height = height - height % 8
    return int(width), int(height)


def latent_hw(width: int, height: int, downsample: int = VAE_DOWNSAMPLE) -> tuple[int, int]:
    if width % downsample or height % downsample:
        raise ValueError(f"Image size {(width, height)} is not divisible by VAE downsample {downsample}.")
    feat_h, feat_w = height // downsample, width // downsample
    if feat_h < 1 or feat_w < 1:
        raise ValueError(f"Image size {(width, height)} is too small for VAE downsample {downsample}.")
    return feat_h, feat_w


def latent_is_tiled(height: int, width: int, tile_size: int) -> bool:
    """DLoRAL tiles when latent area exceeds ``tile_size ** 2``."""

    return int(height) * int(width) > int(tile_size) * int(tile_size)


def _as_hw2(flow: Any) -> torch.Tensor:
    tensor = flow if isinstance(flow, torch.Tensor) else torch.as_tensor(flow)
    tensor = tensor.detach().cpu().float()
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim == 3 and tensor.shape[0] == 2:
        tensor = tensor.permute(1, 2, 0)
    if tensor.ndim != 3 or tensor.shape[-1] != 2:
        raise ValueError(f"pixel_flow must be [H,W,2] or [2,H,W], got {tuple(tensor.shape)}")
    return tensor


def _as_mask(mask: Any, spatial: tuple[int, int] | None = None) -> torch.Tensor:
    if mask is None:
        if spatial is None:
            raise ValueError("valid_mask is required when flow is not provided")
        return torch.ones(spatial, dtype=torch.bool)
    tensor = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask)
    tensor = tensor.detach().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0, 0] if tensor.shape[1] == 1 else tensor[0]
    if tensor.ndim == 3:
        tensor = tensor[0] if tensor.shape[0] == 1 else tensor[..., 0]
    if tensor.ndim != 2:
        raise ValueError(f"valid_mask must be [H,W], got {tuple(tensor.shape)}")
    if tensor.dtype != torch.bool:
        tensor = tensor > 0.5
    if spatial is not None and tuple(tensor.shape) != spatial:
        raise ValueError(f"valid_mask spatial {tuple(tensor.shape)} != {spatial}")
    return tensor


def valid_mask_from_flow(flow_hw2: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(flow_hw2).all(dim=-1)


def _align_corners_grid(height: int, width: int) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, height)
    xs = torch.linspace(-1.0, 1.0, width)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)


def resize_pixel_flow(
    flow_hw2: torch.Tensor,
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Resize image-space flow without bilinear-blending NaN into valid pixels.

    Coordinates and validity are resized separately.  Displacements scale by
    ``dst/src`` so they stay in the resized image's pixel units.  Sampling uses
    nearest + ``align_corners=True``, matching ``flow_warp``.
    """

    src_w, src_h = (int(source_size[0]), int(source_size[1]))
    dst_w, dst_h = (int(target_size[0]), int(target_size[1]))
    if (src_w, src_h) != (flow_hw2.shape[1], flow_hw2.shape[0]):
        raise ValueError(
            f"pixel_flow spatial size {(flow_hw2.shape[1], flow_hw2.shape[0])} "
            f"does not match source_size {(src_w, src_h)}"
        )
    mask = valid_mask_from_flow(flow_hw2) if valid is None else _as_mask(valid, (src_h, src_w))
    if (src_w, src_h) == (dst_w, dst_h):
        return torch.where(mask[..., None], flow_hw2, torch.full_like(flow_hw2, float("nan")))

    grid = _align_corners_grid(dst_h, dst_w)
    filled = torch.nan_to_num(flow_hw2, nan=0.0).permute(2, 0, 1).unsqueeze(0)
    resized = F.grid_sample(filled, grid, mode="nearest", padding_mode="zeros", align_corners=ALIGN_CORNERS)
    resized_valid = F.grid_sample(
        mask.float()[None, None],
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=ALIGN_CORNERS,
    )[0, 0] > 0.5
    resized = resized[0].permute(1, 2, 0)
    resized[..., 0] = resized[..., 0] * (dst_w / src_w)
    resized[..., 1] = resized[..., 1] * (dst_h / src_h)
    return torch.where(resized_valid[..., None], resized, torch.full_like(resized, float("nan")))


def downsample_image_flow(
    flow_hw2: torch.Tensor,
    *,
    downsample: int = VAE_DOWNSAMPLE,
    min_valid_fraction: float = 0.5,
    max_flow_std: float = 4.0,
    valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map image correspondence onto the VAE/CFR grid.

    Each feature cell uses the median of its valid image-space offsets, then
    ``/ downsample`` to feature pixels.  A cell is invalid when too few image
    pixels are valid or when the block's flow standard deviation is high
    (foreground/background mixed at a depth edge).
    """

    height, width, _ = flow_hw2.shape
    if height % downsample or width % downsample:
        raise ValueError(f"Flow {(height, width)} is not divisible by {downsample}.")
    feat_h, feat_w = height // downsample, width // downsample
    mask = valid_mask_from_flow(flow_hw2) if valid is None else _as_mask(valid, (height, width))
    blocks = flow_hw2.reshape(feat_h, downsample, feat_w, downsample, 2)
    valid_blocks = mask.reshape(feat_h, downsample, feat_w, downsample)
    count = valid_blocks.sum(dim=(1, 3))
    filled = torch.where(valid_blocks[..., None], blocks, torch.full_like(blocks, float("nan")))
    flat = filled.permute(0, 2, 1, 3, 4).reshape(feat_h, feat_w, downsample * downsample, 2)
    feature_flow = torch.nanmedian(flat, dim=2).values
    finite = torch.isfinite(flat)
    centered = torch.where(finite, flat - feature_flow.unsqueeze(2), torch.zeros_like(flat))
    var = (centered.square() * finite.float()).sum(dim=2) / count.clamp(min=1).unsqueeze(-1)
    std = var.clamp(min=0).sqrt().amax(dim=-1)
    feature_flow = feature_flow / float(downsample)
    required = float(downsample * downsample) * float(min_valid_fraction)
    feat_valid = (count >= required) & torch.isfinite(feature_flow).all(dim=-1) & (std <= float(max_flow_std))
    feature_flow = torch.where(feat_valid[..., None], feature_flow, torch.full_like(feature_flow, float("nan")))
    return feature_flow, feat_valid


def invert_feature_flow(flow_hw2: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter ``-flow`` onto the neighbor grid.  Diagnostic only; unhit pixels stay NaN."""

    height, width = valid.shape
    device = flow_hw2.device
    dtype = flow_hw2.dtype
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    src_x = xs + flow_hw2[..., 0]
    src_y = ys + flow_hw2[..., 1]
    ix = src_x.round().long()
    iy = src_y.round().long()
    in_bounds = valid & (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    inverse = torch.full((height, width, 2), float("nan"), dtype=dtype, device=device)
    if not bool(in_bounds.any()):
        return inverse, torch.zeros_like(valid)
    linear = iy[in_bounds] * width + ix[in_bounds]
    acc = torch.zeros((height * width, 2), dtype=dtype, device=device)
    count = torch.zeros((height * width,), dtype=dtype, device=device)
    acc.index_add_(0, linear, -flow_hw2[in_bounds])
    count.index_add_(0, linear, torch.ones_like(linear, dtype=dtype))
    hit = count > 0
    acc[hit] = acc[hit] / count[hit].unsqueeze(-1)
    acc[~hit] = float("nan")
    inverse = acc.reshape(height, width, 2)
    return inverse, hit.reshape(height, width)


def sample_flow_at(flow_hw2: torch.Tensor, uv: torch.Tensor, valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a flow field at pixel coordinates with ``align_corners=True``."""

    height, width = flow_hw2.shape[:2]
    if height < 1 or width < 1:
        nan = torch.full_like(uv, float("nan"))
        return nan, torch.zeros(uv.shape[:-1], dtype=torch.bool)

    grid = torch.stack(
        (
            uv[..., 0] * (2.0 / max(width - 1, 1)) - 1.0,
            uv[..., 1] * (2.0 / max(height - 1, 1)) - 1.0,
        ),
        dim=-1,
    ).unsqueeze(0)
    filled = torch.nan_to_num(flow_hw2, nan=0.0).permute(2, 0, 1).unsqueeze(0)
    sampled = F.grid_sample(filled, grid, mode="nearest", padding_mode="zeros", align_corners=ALIGN_CORNERS)
    sampled = sampled[0].permute(1, 2, 0)
    if valid is None:
        valid = valid_mask_from_flow(flow_hw2)
    sampled_valid = F.grid_sample(
        valid.float()[None, None],
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=ALIGN_CORNERS,
    )[0, 0] > 0.5
    in_bounds = (
        torch.isfinite(uv).all(dim=-1)
        & (uv[..., 0] >= 0)
        & (uv[..., 0] <= width - 1)
        & (uv[..., 1] >= 0)
        & (uv[..., 1] <= height - 1)
    )
    sampled_valid = sampled_valid & in_bounds
    sampled = torch.where(sampled_valid[..., None], sampled, torch.full_like(sampled, float("nan")))
    return sampled, sampled_valid


def roundtrip_diagnostics(
    forward_hw2: torch.Tensor,
    forward_valid: torch.Tensor,
    reverse_hw2: torch.Tensor,
    reverse_valid: torch.Tensor,
) -> dict[str, float]:
    """Compose target→source→target using two independently computed fields."""

    height, width = forward_valid.shape
    device = forward_hw2.device
    dtype = forward_hw2.dtype
    forward_valid = forward_valid.to(device=device)
    reverse_valid = reverse_valid.to(device=device)
    reverse_hw2 = reverse_hw2.to(device=device)
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    origin = torch.stack((xs, ys), dim=-1)
    source = origin + forward_hw2
    reverse_at_source, reverse_hit = sample_flow_at(
        reverse_hw2.to(device=device),
        source,
        reverse_valid.to(device=device),
    )
    back = source + reverse_at_source
    error = (back - origin).norm(dim=-1)
    both = forward_valid & reverse_hit & torch.isfinite(error)
    count = int(both.sum().item())
    if count == 0:
        return {
            "forward_coverage": float(forward_valid.float().mean().item()),
            "reverse_coverage": float(reverse_valid.float().mean().item()),
            "roundtrip_hit_rate": 0.0,
            "median_roundtrip_error_px": float("nan"),
            "mean_roundtrip_error_px": float("nan"),
            "hit_pixels": 0,
        }
    return {
        "forward_coverage": float(forward_valid.float().mean().item()),
        "reverse_coverage": float(reverse_valid.float().mean().item()),
        "roundtrip_hit_rate": float(both.float().mean().item()),
        "median_roundtrip_error_px": float(error[both].median().item()),
        "mean_roundtrip_error_px": float(error[both].mean().item()),
        "hit_pixels": count,
    }


def hw2_to_nchw(flow_hw2: torch.Tensor) -> torch.Tensor:
    """``[H,W,2] -> [1,1,2,H,W]`` as consumed by CFR ``external_flows``."""

    return flow_hw2.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)


def sanitize_flow(flow: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf so the tensor can enter ``grid_sample``.  Mask separately."""

    return torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)


def expand_valid_mask(valid_mask: torch.Tensor, *, batch: int, channels: int) -> torch.Tensor:
    mask = valid_mask
    if mask.dtype != torch.bool:
        mask = mask > 0.5
    mask = mask.to(dtype=torch.bool)
    while mask.ndim > 2:
        mask = mask[0]
    return mask.unsqueeze(0).unsqueeze(0).expand(batch, channels, mask.shape[-2], mask.shape[-1])


def zero_invalid_neighbor_features(aligned: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Zero invalid K/V so cross-frame attention cannot read warped garbage."""

    mask = expand_valid_mask(valid_mask, batch=aligned.shape[0], channels=aligned.shape[1])
    return torch.where(mask, aligned, torch.zeros_like(aligned))


def fuse_with_valid_mask(fused: torch.Tensor, current: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """``F_out = M ⊙ F_fused + (1 - M) ⊙ F_i``."""

    mask = expand_valid_mask(valid_mask, batch=fused.shape[0], channels=fused.shape[1])
    return torch.where(mask, fused, current)


def apply_aligned_feature_fallback(
    fused: torch.Tensor,
    aligned: torch.Tensor,
    current: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    target_index: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop invalid neighbor contributions and copy the target feature instead.

    ``fused`` / ``aligned`` / ``current`` are ``[N,T,C,H,W]``.  ``valid_mask`` is
    true where the neighbor sample is geometrically reliable.
    """

    mask = valid_mask
    if mask.dtype != torch.bool:
        mask = mask > 0.5
    while mask.ndim < 5:
        mask = mask.unsqueeze(0)
    mask = mask.to(device=fused.device)
    if mask.shape[-2:] != fused.shape[-2:]:
        raise ValueError(f"valid_mask spatial {tuple(mask.shape[-2:])} != feature {tuple(fused.shape[-2:])}")
    mask = mask.expand(fused.shape[0], 1, fused.shape[2], fused.shape[3], fused.shape[4])
    fused_out = fused.clone()
    aligned_out = aligned.clone()
    fused_out[:, target_index] = fuse_with_valid_mask(
        fused[:, target_index], current[:, target_index], mask[:, 0, 0]
    )
    aligned_out[:, target_index] = torch.where(mask[:, 0], aligned[:, target_index], current[:, target_index])
    return fused_out, aligned_out


def crop_feature_flow(
    flow_nchw: torch.Tensor,
    *,
    origin_hw: tuple[int, int],
    size_hw: tuple[int, int],
) -> torch.Tensor:
    """Crop a latent-scale flow tile.  Relative offsets are not shifted by the origin."""

    top, left = origin_hw
    height, width = size_hw
    return flow_nchw[..., top : top + height, left : left + width]


def iter_dloral_latent_tiles(
    height: int,
    width: int,
    tile_size: int,
    overlap: int,
) -> list[tuple[int, int, int, int]]:
    """Yield ``(top, left, tile_h, tile_w)`` in the official DLoRAL tiled order.

    The generator names the width index ``row`` / ``ofs_x`` and the height index
    ``col`` / ``ofs_y``.  Geometry tiling must use the same origin sequence so
    a cropped valid mask lines up with each CFR call.
    """

    if height * width <= tile_size * tile_size:
        return [(0, 0, int(height), int(width))]
    tile_size = min(int(tile_size), min(int(height), int(width)))
    overlap = int(overlap)
    grid_rows = 0
    cur_x = 0
    while cur_x < width:
        cur_x = max(grid_rows * tile_size - overlap * grid_rows, 0) + tile_size
        grid_rows += 1
    grid_cols = 0
    cur_y = 0
    while cur_y < height:
        cur_y = max(grid_cols * tile_size - overlap * grid_cols, 0) + tile_size
        grid_cols += 1
    tiles = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            ofs_x = max(row * tile_size - overlap * row, 0)
            ofs_y = max(col * tile_size - overlap * col, 0)
            if row == grid_rows - 1:
                ofs_x = width - tile_size
            if col == grid_cols - 1:
                ofs_y = height - tile_size
            tiles.append((int(ofs_y), int(ofs_x), int(tile_size), int(tile_size)))
    return tiles


def flow_warp_nchw(features: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp ``[N,C,H,W]`` with a feature-pixel offset, ``align_corners=True``."""

    if flow.ndim == 4 and flow.shape[1] == 2:
        flow = flow.permute(0, 2, 3, 1)
    if features.shape[-2:] != flow.shape[1:3]:
        raise ValueError(f"feature {tuple(features.shape[-2:])} != flow {tuple(flow.shape[1:3])}")
    _, _, height, width = features.shape
    device, dtype = flow.device, features.dtype
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack((xs, ys), dim=-1) + flow.to(dtype=dtype)
    grid_x = 2.0 * grid[..., 0] / max(width - 1, 1) - 1.0
    grid_y = 2.0 * grid[..., 1] / max(height - 1, 1) - 1.0
    sample_grid = torch.stack((grid_x, grid_y), dim=-1)
    return F.grid_sample(
        features,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=ALIGN_CORNERS,
    )


def align_neighbor_latent(
    latents: torch.Tensor,
    flow: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    neighbor_index: int = 0,
) -> torch.Tensor:
    """Replace the neighbor latent with a globally warped copy on the target grid."""

    if latents.ndim != 4:
        raise ValueError(f"latents must be [N,C,H,W] stacked frames, got {tuple(latents.shape)}")
    if latents.shape[0] < 2:
        return latents
    flow_nchw = flow
    while flow_nchw.ndim > 4:
        flow_nchw = flow_nchw[0]
    if flow_nchw.ndim == 3 and flow_nchw.shape[0] == 2:
        flow_nchw = flow_nchw.unsqueeze(0)
    elif flow_nchw.ndim == 3 and flow_nchw.shape[-1] == 2:
        flow_nchw = flow_nchw.permute(2, 0, 1).unsqueeze(0)
    neighbor = latents[neighbor_index : neighbor_index + 1]
    aligned = flow_warp_nchw(neighbor, sanitize_flow(flow_nchw).to(device=neighbor.device, dtype=neighbor.dtype))
    mask = expand_valid_mask(valid_mask.to(device=aligned.device), batch=1, channels=aligned.shape[1])
    aligned = torch.where(mask, aligned, torch.zeros_like(aligned))
    output = latents.clone()
    output[neighbor_index : neighbor_index + 1] = aligned
    return output


def mask_cross_frame_attention_keys(
    attn_or_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    invalid_logit: float = -1.0e4,
    as_logits: bool = False,
) -> torch.Tensor:
    """Block invalid keys inside cross-frame attention.

    CFR uses thresholded ReLU attention, not softmax, but invalid keys still
    receive weight when the learned threshold is negative.  Zeroing K/V is not
    a substitute: this multiplies the **key** axis by the valid mask, and when
    ``as_logits`` is true also fills invalid logits with ``invalid_logit``.
    """

    spatial = valid_mask
    if spatial.dtype != torch.bool:
        spatial = spatial > 0.5
    while spatial.ndim > 2:
        spatial = spatial[0]
    key_count = int(attn_or_logits.shape[-1])
    height, width = int(spatial.shape[-2]), int(spatial.shape[-1])
    if height * width != key_count:
        feat_h = int(round(key_count ** 0.5))
        feat_w = key_count // feat_h
        spatial = F.interpolate(
            spatial.float()[None, None],
            size=(feat_h, feat_w),
            mode="nearest",
        )[0, 0] > 0.5
    key_mask = spatial.reshape(-1).to(device=attn_or_logits.device)
    view = key_mask.view(1, 1, 1, -1)
    if as_logits:
        filled = torch.where(view, attn_or_logits, attn_or_logits.new_full((), float(invalid_logit)))
        return filled
    return attn_or_logits * key_mask.to(dtype=attn_or_logits.dtype).view(1, 1, 1, -1)


def correspondence_from_neighbor(neighbor: MultiViewInput) -> GeometryCorrespondence:
    """Lift a ``MultiViewInput`` into the explicit geometry correspondence contract."""

    flow = _as_hw2(neighbor.pixel_flow)
    height, width = flow.shape[:2]
    valid = neighbor.valid_mask
    if valid is None:
        valid = valid_mask_from_flow(flow)
    else:
        valid = _as_mask(valid, (height, width))
    source_size = tuple(neighbor.metadata.get("source_size", (width, height)))
    target_size = tuple(neighbor.metadata.get("target_size", (width, height)))
    reverse = None if neighbor.source_to_target_flow is None else _as_hw2(neighbor.source_to_target_flow)
    reverse_valid = None
    if neighbor.reverse_valid_mask is not None:
        reverse_valid = _as_mask(neighbor.reverse_valid_mask)
    elif reverse is not None:
        reverse_valid = valid_mask_from_flow(reverse)
    return GeometryCorrespondence(
        target_to_source_flow=flow,
        valid_mask=valid,
        source_size=(int(source_size[0]), int(source_size[1])),
        target_size=(int(target_size[0]), int(target_size[1])),
        target_to_source_grid=neighbor.sample_grid,
        source_to_target_flow=reverse,
        reverse_valid_mask=reverse_valid,
        confidence=neighbor.metadata.get("coverage"),
        camera_metadata={
            "name": neighbor.name,
            "weight": float(neighbor.weight),
            **{key: value for key, value in dict(neighbor.metadata).items() if key not in {"coverage"}},
        },
    )


def _map_one_flow(
    flow: Any,
    *,
    image_size: tuple[int, int],
    prepared_size: tuple[int, int],
    downsample: int,
    min_valid_fraction: float,
    max_flow_std: float,
    valid: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    flow_hw2 = _as_hw2(flow)
    src_w, src_h = int(image_size[0]), int(image_size[1])
    dst_w, dst_h = prepared_size
    mask = valid_mask_from_flow(flow_hw2) if valid is None else _as_mask(valid, (flow_hw2.shape[0], flow_hw2.shape[1]))
    resized = resize_pixel_flow(
        flow_hw2,
        source_size=(src_w, src_h),
        target_size=(dst_w, dst_h),
        valid=mask,
    )
    return downsample_image_flow(
        resized,
        downsample=downsample,
        min_valid_fraction=min_valid_fraction,
        max_flow_std=max_flow_std,
    )


def correspondence_to_external_flows(
    correspondence: GeometryCorrespondence,
    *,
    process_size: int = 512,
    upscale: int = 1,
    downsample: int = VAE_DOWNSAMPLE,
    min_valid_fraction: float = 0.5,
    max_flow_std: float = 4.0,
    neighbor_contribution: bool = True,
) -> dict[str, Any]:
    """Build CFR ``(flows_forward, flows_backward)`` plus a feature-scale valid mask."""

    src_w, src_h = int(correspondence.target_size[0]), int(correspondence.target_size[1])
    dst_w, dst_h = prepared_image_size(src_w, src_h, process_size=process_size, upscale=upscale)
    feature_flow, feat_valid = _map_one_flow(
        correspondence.target_to_source_flow,
        image_size=(src_w, src_h),
        prepared_size=(dst_w, dst_h),
        downsample=downsample,
        min_valid_fraction=min_valid_fraction,
        max_flow_std=max_flow_std,
        valid=correspondence.valid_mask,
    )
    reverse_source = "depth"
    if correspondence.source_to_target_flow is None:
        backward, backward_valid = invert_feature_flow(feature_flow, feat_valid)
        reverse_source = "scatter_invert_diagnostic_only"
    else:
        source_w, source_h = int(correspondence.source_size[0]), int(correspondence.source_size[1])
        source_prepared = prepared_image_size(source_w, source_h, process_size=process_size, upscale=upscale)
        if source_prepared != (dst_w, dst_h):
            raise ValueError(
                "DLoRAL dual-view path requires neighbor and target prepared sizes to match, "
                f"got source {source_prepared} vs target {(dst_w, dst_h)}"
            )
        backward, backward_valid = _map_one_flow(
            correspondence.source_to_target_flow,
            image_size=(source_w, source_h),
            prepared_size=source_prepared,
            downsample=downsample,
            min_valid_fraction=min_valid_fraction,
            max_flow_std=max_flow_std,
            valid=correspondence.reverse_valid_mask,
        )
    if not neighbor_contribution:
        feat_valid = torch.zeros_like(feat_valid)
        backward_valid = torch.zeros_like(backward_valid)
    roundtrip = roundtrip_diagnostics(feature_flow, feat_valid, backward, backward_valid) if feature_flow.numel() else {
        "roundtrip_hit_rate": 0.0,
        "hit_pixels": 0,
    }
    flows_forward = hw2_to_nchw(feature_flow)
    flows_backward = hw2_to_nchw(backward)
    coverage = float(feat_valid.float().mean().item()) if feat_valid.numel() else 0.0
    return {
        "flows_forward": flows_forward,
        "flows_backward": flows_backward,
        "valid_mask": feat_valid,
        "valid_mask_backward": backward_valid,
        "flows_forward_safe": sanitize_flow(flows_forward),
        "flows_backward_safe": sanitize_flow(flows_backward),
        "image_size": (src_w, src_h),
        "prepared_size": (dst_w, dst_h),
        "feature_size": (int(feature_flow.shape[1]), int(feature_flow.shape[0])),
        "coverage": coverage,
        "downsample": int(downsample),
        "units": FLOW_UNITS,
        "align_corners": ALIGN_CORNERS,
        "frame_order": list(FRAME_ORDER),
        "flows_forward_meaning": FLOWS_FORWARD_MEANING,
        "invalid": "nan_on_disk_zero_in_network",
        "reverse_source": reverse_source,
        "roundtrip": roundtrip,
        "metadata": {
            "valid_feature_pixels": int(feat_valid.sum().item()),
            "feature_hw": [int(feature_flow.shape[0]), int(feature_flow.shape[1])],
            "min_valid_fraction": float(min_valid_fraction),
            "max_flow_std": float(max_flow_std),
            "neighbor_contribution": bool(neighbor_contribution),
            "reverse_source": reverse_source,
        },
    }


def pixel_flow_to_external_flows(
    pixel_flow: Any,
    *,
    image_size: tuple[int, int],
    process_size: int = 512,
    upscale: int = 1,
    downsample: int = VAE_DOWNSAMPLE,
    min_valid_fraction: float = 0.5,
    max_flow_std: float = 4.0,
    valid_mask: Any = None,
    reverse_pixel_flow: Any = None,
    reverse_valid_mask: Any = None,
    neighbor_contribution: bool = True,
    source_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Build CFR flows from a stored target→source ``pixel_flow`` field."""

    flow = _as_hw2(pixel_flow)
    height, width = flow.shape[:2]
    valid = valid_mask_from_flow(flow) if valid_mask is None else _as_mask(valid_mask, (height, width))
    correspondence = GeometryCorrespondence(
        target_to_source_flow=flow,
        valid_mask=valid,
        source_size=tuple(source_size or image_size),
        target_size=tuple(image_size),
        source_to_target_flow=None if reverse_pixel_flow is None else _as_hw2(reverse_pixel_flow),
        reverse_valid_mask=reverse_valid_mask,
    )
    return correspondence_to_external_flows(
        correspondence,
        process_size=process_size,
        upscale=upscale,
        downsample=downsample,
        min_valid_fraction=min_valid_fraction,
        max_flow_std=max_flow_std,
        neighbor_contribution=neighbor_contribution,
    )


def pack_external_flows_for_worker(payload: Mapping[str, Any], directory, *, prefix: str = "") -> dict[str, str]:
    """Write float32 arrays for the isolated worker.  Invalid stays NaN on disk."""

    from pathlib import Path

    root = Path(directory)
    paths = {
        "flows_forward": root / f"{prefix}flows_forward.npy",
        "flows_backward": root / f"{prefix}flows_backward.npy",
        "valid_mask": root / f"{prefix}flow_valid.npy",
    }
    np.save(paths["flows_forward"], payload["flows_forward"].detach().cpu().float().numpy())
    np.save(paths["flows_backward"], payload["flows_backward"].detach().cpu().float().numpy())
    np.save(paths["valid_mask"], payload["valid_mask"].detach().cpu().bool().numpy())
    return {key: str(path) for key, path in paths.items()}


def feature_diagnostics(
    *,
    target: torch.Tensor,
    aligned: torch.Tensor,
    fused: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, float]:
    """Compare target / aligned / fused features on valid vs invalid pixels."""

    mask = valid_mask
    if mask.dtype != torch.bool:
        mask = mask > 0.5
    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(0)
    mask = mask.expand_as(target)
    invalid = ~mask
    def _l1(a, b, select) -> float:
        if not bool(select.any()):
            return float("nan")
        return float((a - b).abs()[select].mean().item())

    return {
        "target_abs_mean": float(target.abs().mean().item()),
        "aligned_abs_mean": float(aligned.abs().mean().item()),
        "fused_abs_mean": float(fused.abs().mean().item()),
        "aligned_target_l1_valid": _l1(aligned, target, mask),
        "aligned_target_l1_invalid": _l1(aligned, target, invalid),
        "fused_target_l1_valid": _l1(fused, target, mask),
        "fused_target_l1_invalid": _l1(fused, target, invalid),
        "aligned_changes_valid": _l1(aligned, target, mask) > 1e-6,
        "coverage": float(valid_mask.float().mean().item()) if valid_mask.numel() else 0.0,
        "has_nan": bool(torch.isnan(target).any() or torch.isnan(aligned).any() or torch.isnan(fused).any()),
    }


def wrap_cfr_geometry_alignment(
    cfr_module,
    flows_forward,
    flows_backward,
    valid_mask,
    *,
    dump: dict[str, Any] | None = None,
    prealigned: bool = False,
    latent_hw_full: tuple[int, int] | None = None,
    tile_size: int | None = None,
    tile_overlap: int | None = None,
):
    """Patch CFR so geometry flows replace SpyNet and invalid keys cannot leak.

    ``prealigned=True`` means the neighbor latent is already on the target grid
    (global warp, then tiled crops).  CFR then uses a zero flow so SpyNet cannot
    warp it a second time.  Tile calls consume official DLoRAL origins to crop
    ``valid_mask``.
    """

    original_forward = cfr_module.forward
    original_attn = cfr_module.cross_attn_module.forward
    full_h, full_w = (None, None) if latent_hw_full is None else (int(latent_hw_full[0]), int(latent_hw_full[1]))
    tiles = None
    tile_iter = None
    if prealigned and full_h is not None and tile_size is not None:
        tiles = iter_dloral_latent_tiles(full_h, full_w, int(tile_size), int(tile_overlap or 0))
        tile_iter = iter(tiles)

    def _tile_mask(feat_h: int, feat_w: int) -> torch.Tensor:
        mask = valid_mask
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        while mask.ndim > 2:
            mask = mask[0]
        if tuple(mask.shape[-2:]) == (feat_h, feat_w):
            return mask
        if tile_iter is None:
            raise RuntimeError(
                f"valid_mask {tuple(mask.shape[-2:])} does not match CFR spatial {(feat_h, feat_w)}"
            )
        try:
            top, left, tile_h, tile_w = next(tile_iter)
        except StopIteration as exc:
            raise RuntimeError("CFR tile count exceeded the official DLoRAL grid.") from exc
        if (tile_h, tile_w) != (feat_h, feat_w):
            raise RuntimeError(f"tile {(tile_h, tile_w)} != CFR spatial {(feat_h, feat_w)}")
        return mask[top : top + tile_h, left : left + tile_w]

    def gated_attn(cur_img, aligned_img, cur_feat, aligned_feat, tile_mask):
        mask = tile_mask.to(device=aligned_feat.device)
        aligned_feat = zero_invalid_neighbor_features(aligned_feat, mask)
        aligned_img = zero_invalid_neighbor_features(aligned_img, mask)
        orig_relu = F.relu

        def masked_relu(input_tensor, inplace=False):
            del inplace
            key_mask = mask
            while key_mask.ndim > 2:
                key_mask = key_mask[0]
            if key_mask.shape[-2] * key_mask.shape[-1] != input_tensor.shape[-1]:
                key_mask = F.interpolate(
                    key_mask.float()[None, None],
                    size=(cur_img.shape[-2], cur_img.shape[-1]),
                    mode="nearest",
                )[0, 0] > 0.5
            before = orig_relu(input_tensor)
            logits = mask_cross_frame_attention_keys(input_tensor, key_mask, as_logits=True)
            attn = orig_relu(logits)
            attn = mask_cross_frame_attention_keys(attn, key_mask, as_logits=False)
            if dump is not None:
                invalid = (~key_mask.reshape(-1)).to(device=attn.device)
                dump["attn_key_mask"] = True
                dump["attn_invalid_key_mass_before"] = float(before[..., invalid].abs().sum().item())
                dump["attn_invalid_key_mass_after"] = float(attn[..., invalid].abs().sum().item())
            return attn

        F.relu = masked_relu
        try:
            fused, residual = original_attn(cur_img, aligned_img, cur_feat, aligned_feat)
        finally:
            F.relu = orig_relu
        fused = fuse_with_valid_mask(fused, cur_feat, mask)
        return fused, residual

    def geometry_forward(lqs, vae_feat, uncertainty_map, external_flows=None):
        if torch.isnan(vae_feat).any():
            raise RuntimeError("NaN in VAE features before CFR")
        tile_mask = _tile_mask(int(vae_feat.shape[-2]), int(vae_feat.shape[-1]))
        cfr_module.cross_attn_module.forward = lambda cur_img, aligned_img, cur_feat, aligned_feat: gated_attn(
            cur_img, aligned_img, cur_feat, aligned_feat, tile_mask
        )
        if prealigned:
            zeros = torch.zeros(
                vae_feat.shape[0],
                max(vae_feat.shape[1] - 1, 1),
                2,
                vae_feat.shape[-2],
                vae_feat.shape[-1],
                device=vae_feat.device,
                dtype=vae_feat.dtype,
            )
            forward, backward = zeros, zeros
        else:
            forward = sanitize_flow(flows_forward).to(device=vae_feat.device, dtype=vae_feat.dtype)
            backward = sanitize_flow(flows_backward).to(device=vae_feat.device, dtype=vae_feat.dtype)
            if tuple(forward.shape[-2:]) != tuple(vae_feat.shape[-2:]):
                raise RuntimeError(
                    f"external_flows spatial {tuple(forward.shape[-2:])} != latent {tuple(vae_feat.shape[-2:])}"
                )
        if torch.isnan(forward).any() or torch.isnan(backward).any():
            raise RuntimeError("NaN remains in sanitized external_flows")
        fused, weight_map, aligned = original_forward(
            lqs, vae_feat, uncertainty_map, external_flows=(forward, backward)
        )
        fused, aligned = apply_aligned_feature_fallback(
            fused, aligned, vae_feat, tile_mask.to(device=vae_feat.device)
        )
        if torch.isnan(fused).any() or torch.isnan(aligned).any():
            raise RuntimeError("NaN in CFR outputs after geometry gating")
        if dump is not None:
            dump.update(
                feature_diagnostics(
                    target=vae_feat[:, 1],
                    aligned=aligned[:, 1],
                    fused=fused[:, 1],
                    valid_mask=tile_mask.to(device=vae_feat.device),
                )
            )
            dump["valid_coverage"] = float(valid_mask.float().mean().item())
            dump["prealigned"] = bool(prealigned)
        return fused, weight_map, aligned

    cfr_module.forward = geometry_forward
    return cfr_module


def wrap_vae_neighbor_alignment(model, flow, valid_mask):
    """Warp neighbor latents onto the target grid before tiled CFR sees them."""

    original_encode = model.vae.encode

    def encode_and_align(images, *args, **kwargs):
        posterior_out = original_encode(images, *args, **kwargs)
        latent_dist = posterior_out.latent_dist
        original_sample = latent_dist.sample

        def sample_and_align(*sample_args, **sample_kwargs):
            latents = original_sample(*sample_args, **sample_kwargs)
            if latents.ndim == 4 and latents.shape[0] >= 2:
                latents = align_neighbor_latent(latents, flow, valid_mask)
            return latents

        latent_dist.sample = sample_and_align
        return posterior_out

    model.vae.encode = encode_and_align
    return model


__all__ = [
    "ALIGN_CORNERS",
    "FLOW_UNITS",
    "FLOWS_FORWARD_MEANING",
    "FRAME_ORDER",
    "VAE_DOWNSAMPLE",
    "align_neighbor_latent",
    "apply_aligned_feature_fallback",
    "correspondence_from_neighbor",
    "correspondence_to_external_flows",
    "crop_feature_flow",
    "downsample_image_flow",
    "expand_valid_mask",
    "feature_diagnostics",
    "flow_warp_nchw",
    "fuse_with_valid_mask",
    "hw2_to_nchw",
    "invert_feature_flow",
    "iter_dloral_latent_tiles",
    "latent_hw",
    "latent_is_tiled",
    "mask_cross_frame_attention_keys",
    "pack_external_flows_for_worker",
    "pixel_flow_to_external_flows",
    "prepared_image_size",
    "resize_pixel_flow",
    "roundtrip_diagnostics",
    "sanitize_flow",
    "sample_flow_at",
    "valid_mask_from_flow",
    "wrap_cfr_geometry_alignment",
    "wrap_vae_neighbor_alignment",
    "zero_invalid_neighbor_features",
]
