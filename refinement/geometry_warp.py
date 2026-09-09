"""Depth-aware RGB reprojection utilities for the multi-view phase."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .types import GeometryCorrespondence, MultiViewInput, ProjectedROI, SpatialTarget, WarpResult


def _as_plane(value: Any, *, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"{name} with four dimensions must have batch size 1, got {tuple(tensor.shape)}")
        tensor = tensor[0]
    if tensor.ndim == 3:
        if tensor.shape[0] == 1:
            tensor = tensor[0]
        elif tensor.shape[-1] == 1:
            tensor = tensor[..., 0]
        else:
            raise ValueError(f"{name} must contain one channel, got {tuple(tensor.shape)}")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be [H,W], [1,H,W], or [1,1,H,W], got {tuple(tensor.shape)}")
    return tensor


def _camera_matrix(camera: Any, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    rotation = torch.as_tensor(camera.R, device=device, dtype=dtype)
    translation = torch.as_tensor(camera.T, device=device, dtype=dtype).reshape(1, 3)
    if rotation.shape != (3, 3):
        raise ValueError(f"Camera R must be [3,3], got {tuple(rotation.shape)}")
    return rotation, translation


def _camera_intrinsics(camera: Any, device: torch.device, dtype: torch.dtype) -> tuple[float, float, float, float]:
    width = float(camera.image_width)
    height = float(camera.image_height)
    fov_x = float(camera.FoVx)
    fov_y = float(camera.FoVy)
    focal_x = float(getattr(camera, "focal_x", width / (2.0 * math.tan(fov_x / 2.0))))
    focal_y = float(getattr(camera, "focal_y", height / (2.0 * math.tan(fov_y / 2.0))))
    cx = (float(camera.cx) / 2.0 + 0.5) * width
    cy = (float(camera.cy) / 2.0 + 0.5) * height
    return focal_x, focal_y, cx, cy


def world_to_camera(points_world: torch.Tensor, camera: Any) -> torch.Tensor:
    """Transform row-vector world points using the repository's R/T convention."""

    if points_world.ndim != 2 or points_world.shape[-1] != 3:
        raise ValueError(f"points_world must be [N,3], got {tuple(points_world.shape)}")
    rotation, translation = _camera_matrix(camera, points_world.device, points_world.dtype)
    return points_world @ rotation + translation


def camera_to_world(points_camera: torch.Tensor, camera: Any) -> torch.Tensor:
    """Inverse of :func:`world_to_camera` for the same R/T convention."""

    if points_camera.ndim != 2 or points_camera.shape[-1] != 3:
        raise ValueError(f"points_camera must be [N,3], got {tuple(points_camera.shape)}")
    rotation, translation = _camera_matrix(camera, points_camera.device, points_camera.dtype)
    return (points_camera - translation) @ rotation.transpose(0, 1)


def project_world_points(points_world: torch.Tensor, camera: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world points to pixel ``(u,v)`` and return camera-space depth."""

    points_camera = world_to_camera(points_world, camera)
    focal_x, focal_y, cx, cy = _camera_intrinsics(camera, points_world.device, points_world.dtype)
    z = points_camera[:, 2]
    safe_z = z.clamp_min(torch.finfo(points_world.dtype).eps)
    uv = torch.stack(
        (
            points_camera[:, 0] / safe_z * focal_x + cx,
            points_camera[:, 1] / safe_z * focal_y + cy,
        ),
        dim=-1,
    )
    return uv, z


def _unproject_pixel_grid(camera: Any, depth: torch.Tensor) -> torch.Tensor:
    height, width = depth.shape
    device, dtype = depth.device, depth.dtype
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    focal_x, focal_y, cx, cy = _camera_intrinsics(camera, device, dtype)
    points_camera = torch.stack(
        (
            (xs - cx) / focal_x * depth,
            (ys - cy) / focal_y * depth,
            depth,
        ),
        dim=-1,
    )
    return points_camera.reshape(-1, 3)


def _normalize_grid(uv: torch.Tensor, width: int, height: int) -> torch.Tensor:
    denominator_x = max(width - 1, 1)
    denominator_y = max(height - 1, 1)
    return torch.stack(
        (
            uv[..., 0] * (2.0 / denominator_x) - 1.0,
            uv[..., 1] * (2.0 / denominator_y) - 1.0,
        ),
        dim=-1,
    )


def _sample_plane(plane: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    return F.grid_sample(
        plane.float()[None, None],
        grid.float()[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0].to(dtype=plane.dtype)


def expected_surface_depth(
    accumulated_depth: torch.Tensor,
    alpha: torch.Tensor | None = None,
    *,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Convert rasterizer depth to camera-space surface depth.

    The repository rasterizer returns ``sum(T * alpha * z)``. Reprojection
    needs the expected camera-space ``z`` coordinate, so accumulated depth is
    divided by rendered alpha before unprojection. Passing ``alpha=None`` is
    supported for callers that already provide surface depth.
    """

    depth = _as_plane(accumulated_depth, name="accumulated_depth").float()
    if alpha is None:
        return depth
    opacity = _as_plane(alpha, name="alpha").float()
    if opacity.shape != depth.shape:
        raise ValueError(
            "accumulated_depth and alpha must have the same spatial shape, "
            f"got {tuple(depth.shape)} and {tuple(opacity.shape)}"
        )
    valid = torch.isfinite(depth) & torch.isfinite(opacity) & (opacity > float(eps))
    surface = depth / opacity.clamp_min(float(eps))
    return torch.where(valid, surface, torch.zeros_like(surface))


def pixel_footprint(camera: Any, depth: float) -> float:
    """Approximate world-unit size of one pixel at ``depth`` along the view ray."""

    width = max(int(getattr(camera, "image_width", 1)), 1)
    fov_x = float(getattr(camera, "FoVx"))
    return float(2.0 * abs(float(depth)) * math.tan(0.5 * fov_x) / width)


def estimate_depth_scene_scale(
    camera: Any,
    median_depth: float,
    *,
    building_pixels: float = 256.0,
    min_scale: float = 5.0,
) -> float:
    """Cap occlusion tolerance using camera GSD instead of a huge relative term.

    Satellite depths are O(1e6).  A relative tolerance of 0.02 then becomes tens
    of thousands of scene units and cannot reject building-scale occluders.
    """

    footprint = pixel_footprint(camera, median_depth)
    return max(float(min_scale), footprint * float(building_pixels))


def normalize_scene_scale(scene_scale: float | None) -> float | None:
    if scene_scale is None:
        return None
    value = float(scene_scale)
    return None if value <= 0.0 else value


def compute_depth_tolerance(
    sampled_depth: torch.Tensor,
    *,
    abs_tolerance: float,
    rel_tolerance: float,
    scene_scale: float | None = None,
) -> torch.Tensor:
    """Return per-pixel occlusion slack ``max(abs, |D|*rel)``, optionally capped.

    The cap is required for far satellite cameras: without it, ``|D|*rel`` grows
    with range and hides foreground/background gaps of building height.
    Invalid flow later uses this mask; do not recover those pixels by zeroing.
    """

    relative = sampled_depth.abs() * float(rel_tolerance)
    tolerance = torch.maximum(
        torch.full_like(sampled_depth, float(abs_tolerance)),
        relative,
    )
    scale = normalize_scene_scale(scene_scale)
    if scale is not None:
        tolerance = torch.minimum(tolerance, torch.full_like(sampled_depth, scale))
        tolerance = torch.maximum(tolerance, torch.full_like(sampled_depth, float(abs_tolerance)))
    return tolerance


def pixel_displacement_flow(
    projected_uv: torch.Tensor,
    height: int,
    width: int,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return ``p_source - p_target`` in pixel units; invalid entries are NaN."""

    device = projected_uv.device
    dtype = projected_uv.dtype
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    target_uv = torch.stack((xs, ys), dim=-1)
    flow = projected_uv.reshape(height, width, 2) - target_uv
    return torch.where(valid_mask[..., None], flow, torch.full_like(flow, float("nan")))


def build_reprojection_grid(
    target_camera: Any,
    source_camera: Any,
    target_depth: torch.Tensor,
    source_depth: torch.Tensor | None = None,
    source_alpha: torch.Tensor | None = None,
    *,
    target_alpha: torch.Tensor | None = None,
    depth_abs_tolerance: float = 0.05,
    depth_rel_tolerance: float = 0.02,
    depth_scene_scale: float | None = None,
    alpha_threshold: float = 1e-4,
    min_depth: float = 1e-4,
) -> WarpResult:
    """Build a target-pixel to source-pixel grid with depth/alpha validity.

    A target pixel is first unprojected using its own camera and rendered depth,
    transformed into world coordinates, and projected into the source camera.
    This is why a neighbor ROI must be found through spatial correspondence;
    copying normalized image coordinates would be incorrect.
    """

    target_depth = _as_plane(target_depth, name="target_depth").float()
    source_depth_plane = _as_plane(source_depth, name="source_depth").float() if source_depth is not None else None
    source_alpha_plane = _as_plane(source_alpha, name="source_alpha").float() if source_alpha is not None else None
    target_alpha_plane = _as_plane(target_alpha, name="target_alpha").float() if target_alpha is not None else None
    if target_alpha_plane is not None and target_alpha_plane.shape != target_depth.shape:
        raise ValueError(
            "target_alpha and target_depth must have the same spatial shape, "
            f"got {tuple(target_alpha_plane.shape)} and {tuple(target_depth.shape)}"
        )

    target_points_camera = _unproject_pixel_grid(target_camera, target_depth)
    target_points_world = camera_to_world(target_points_camera, target_camera)
    projected_uv, source_z = project_world_points(target_points_world, source_camera)
    height, width = target_depth.shape
    source_height, source_width = (
        source_depth_plane.shape if source_depth_plane is not None else (int(source_camera.image_height), int(source_camera.image_width))
    )
    grid = _normalize_grid(projected_uv, int(source_width), int(source_height)).reshape(height, width, 2)

    target_valid = torch.isfinite(target_depth) & (target_depth > min_depth)
    if target_alpha_plane is not None:
        target_valid &= torch.isfinite(target_alpha_plane) & (target_alpha_plane > alpha_threshold)
    source_valid = torch.isfinite(source_z) & (source_z > min_depth)
    source_uv_valid = (
        torch.isfinite(projected_uv).all(dim=-1)
        & (projected_uv[:, 0] >= 0.0)
        & (projected_uv[:, 0] <= float(source_width - 1))
        & (projected_uv[:, 1] >= 0.0)
        & (projected_uv[:, 1] <= float(source_height - 1))
    )
    valid = target_valid.reshape(-1) & source_valid & source_uv_valid

    if source_depth_plane is not None:
        sampled_depth = _sample_plane(source_depth_plane, grid)
        tolerance = compute_depth_tolerance(
            sampled_depth,
            abs_tolerance=depth_abs_tolerance,
            rel_tolerance=depth_rel_tolerance,
            scene_scale=depth_scene_scale,
        )
        source_depth_valid = torch.isfinite(sampled_depth) & (sampled_depth > min_depth)
        source_depth_valid &= source_z.reshape(height, width) <= sampled_depth + tolerance
        valid &= source_depth_valid.reshape(-1)

    if source_alpha_plane is not None:
        sampled_alpha = _sample_plane(source_alpha_plane, grid)
        valid &= (sampled_alpha.reshape(-1) > alpha_threshold)

    valid_mask = valid.reshape(height, width)
    # Invalid coordinates are set outside the sampling domain so grid_sample
    # returns zero even when a caller forgets to apply valid_mask.
    safe_grid = torch.where(
        torch.isfinite(grid) & valid_mask[..., None],
        grid,
        torch.full_like(grid, 2.0),
    )
    coverage = float(valid_mask.float().mean().item()) if valid_mask.numel() else 0.0
    projected = projected_uv.reshape(height, width, 2)
    pixel_flow = pixel_displacement_flow(projected, height, width, valid_mask)
    scene_scale = normalize_scene_scale(depth_scene_scale)
    metadata = {
        "target_size": [int(width), int(height)],
        "source_size": [int(source_width), int(source_height)],
        "target_alpha_check": target_alpha_plane is not None,
        "depth_check": source_depth_plane is not None,
        "alpha_check": source_alpha_plane is not None,
        "valid_pixels": int(valid_mask.sum().item()),
        "depth_abs_tolerance": float(depth_abs_tolerance),
        "depth_rel_tolerance": float(depth_rel_tolerance),
        "depth_scene_scale": scene_scale,
        "pixel_flow_units": "source_pixel - target_pixel",
        "invalid_flow": "nan",
    }
    if source_depth_plane is not None:
        finite_tol = tolerance[torch.isfinite(tolerance)]
        if finite_tol.numel():
            metadata["median_depth_tolerance"] = float(finite_tol.median().item())
            metadata["max_depth_tolerance"] = float(finite_tol.max().item())
    return WarpResult(
        grid=safe_grid,
        valid_mask=valid_mask,
        projected_uv=projected,
        coverage=coverage,
        pixel_flow=pixel_flow,
        metadata=metadata,
    )


def _image_to_tensor(image: Image.Image | torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)
    tensor = image if isinstance(image, torch.Tensor) else torch.as_tensor(image)
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"Image batch must have size 1, got {tuple(tensor.shape)}")
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Image must be CHW or HWC, got {tuple(tensor.shape)}")
    if tensor.shape[0] in (1, 3, 4):
        output = tensor
    elif tensor.shape[-1] in (1, 3, 4):
        output = tensor.permute(2, 0, 1)
    else:
        raise ValueError(f"Could not infer image channel dimension from {tuple(tensor.shape)}")
    output = output.float()
    if output.numel() and float(output.detach().amax().item()) > 1.0:
        output = output / 255.0
    return output


def warp_image(
    source_image: Image.Image | torch.Tensor | np.ndarray,
    warp: WarpResult,
    *,
    fill: float = 0.0,
) -> torch.Tensor:
    """Sample a source RGB image into target coordinates, returning CHW."""

    image = _image_to_tensor(source_image)
    grid = warp.grid.to(device=image.device)
    sampled = F.grid_sample(
        image[None],
        grid.float()[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0]
    valid = warp.valid_mask.to(device=image.device, dtype=sampled.dtype)
    if fill:
        sampled = sampled + (1.0 - valid)[None] * float(fill)
    else:
        sampled = sampled * valid[None]
    return sampled


def _mapping_value(mapping: Mapping[Any, torch.Tensor] | None, camera: Any) -> torch.Tensor | None:
    if mapping is None:
        return None
    for key in (getattr(camera, "uid", None), getattr(camera, "image_name", None)):
        if key is not None:
            value = mapping.get(key)
            if value is not None:
                return value
    return None


def rank_neighbor_cameras(
    target_camera: Any,
    candidates: Sequence[Any],
    *,
    target_depth: torch.Tensor,
    source_depths: Mapping[Any, torch.Tensor],
    target_alpha: torch.Tensor | None = None,
    source_alphas: Mapping[Any, torch.Tensor] | None = None,
    k: int = 2,
    depth_abs_tolerance: float = 0.05,
    depth_rel_tolerance: float = 0.02,
    depth_scene_scale: float | None = None,
    alpha_threshold: float = 1e-4,
    min_depth: float = 1e-4,
) -> list[tuple[Any, WarpResult]]:
    """Rank candidates by depth/alpha-valid reprojection coverage.

    ``target_depth`` and ``source_depths`` must be camera-space surface depth
    (use :func:`expected_surface_depth` for raw rasterizer outputs). The
    returned warp is retained so callers do not recompute visibility tests.
    """

    if k <= 0:
        return []
    filtered = [
        camera
        for camera in candidates
        if camera is not target_camera
        and getattr(camera, "uid", None) != getattr(target_camera, "uid", None)
    ]
    scored: list[tuple[float, Any, WarpResult]] = []
    for camera in filtered:
        source_depth = _mapping_value(source_depths, camera)
        if source_depth is None:
            continue
        warp = build_reprojection_grid(
            target_camera,
            camera,
            target_depth,
            source_depth,
            _mapping_value(source_alphas, camera),
            target_alpha=target_alpha,
            depth_abs_tolerance=depth_abs_tolerance,
            depth_rel_tolerance=depth_rel_tolerance,
            depth_scene_scale=depth_scene_scale,
            alpha_threshold=alpha_threshold,
            min_depth=min_depth,
        )
        scored.append((warp.coverage, camera, warp))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(camera, warp) for _, camera, warp in scored[:k]]


def build_aligned_multiview_inputs(
    target_camera: Any,
    target_depth: torch.Tensor,
    target_alpha: torch.Tensor | None,
    source_observations: Sequence[tuple[Any, Any, Any, Any]],
    *,
    k: int = 2,
    depth_abs_tolerance: float = 0.05,
    depth_rel_tolerance: float = 0.02,
    depth_scene_scale: float | None = None,
    alpha_threshold: float = 1e-4,
    min_depth: float = 1e-4,
) -> tuple[list[MultiViewInput], list[dict[str, Any]]]:
    """Build aligned neighbor inputs and auditable selection records.

    Each source observation is ``(camera, rgb, surface_depth, alpha)``. The
    function scores candidates with target-to-source projection, retains the
    top ``k`` candidates, and attaches warped RGB plus validity masks. Image
    coordinates are never copied between cameras.
    """

    target_depth = _as_plane(target_depth, name="target_depth").float()
    target_alpha_plane = (
        _as_plane(target_alpha, name="target_alpha").float() if target_alpha is not None else None
    )
    if target_alpha_plane is not None and target_alpha_plane.shape != target_depth.shape:
        raise ValueError(
            "target_alpha and target_depth must have the same spatial shape, "
            f"got {tuple(target_alpha_plane.shape)} and {tuple(target_depth.shape)}"
        )

    scored: list[tuple[float, Any, Any, Any, Any, WarpResult]] = []
    records: list[dict[str, Any]] = []
    for camera, image, source_depth, source_alpha in source_observations:
        name = str(getattr(camera, "image_name", getattr(camera, "uid", "neighbor")))
        if source_depth is None or image is None:
            records.append(
                {
                    "name": name,
                    "uid": int(getattr(camera, "uid", -1)),
                    "skipped": "missing_depth_or_rgb",
                }
            )
            continue
        source_depth = _as_plane(source_depth, name=f"source_depth[{name}]").float()
        source_alpha_plane = (
            _as_plane(source_alpha, name=f"source_alpha[{name}]").float()
            if source_alpha is not None
            else None
        )
        warp = build_reprojection_grid(
            target_camera,
            camera,
            target_depth,
            source_depth,
            source_alpha_plane,
            target_alpha=target_alpha_plane,
            depth_abs_tolerance=depth_abs_tolerance,
            depth_rel_tolerance=depth_rel_tolerance,
            depth_scene_scale=depth_scene_scale,
            alpha_threshold=alpha_threshold,
            min_depth=min_depth,
        )
        record = {
            "name": name,
            "uid": int(getattr(camera, "uid", -1)),
            "coverage": warp.coverage,
            "valid_pixels": int(warp.valid_mask.sum().item()),
            "target_size": warp.metadata.get("target_size"),
            "source_size": warp.metadata.get("source_size"),
            "depth_check": bool(warp.metadata.get("depth_check", False)),
            "alpha_check": bool(warp.metadata.get("alpha_check", False)),
            "median_depth_tolerance": warp.metadata.get("median_depth_tolerance"),
            "max_depth_tolerance": warp.metadata.get("max_depth_tolerance"),
            "depth_scene_scale": warp.metadata.get("depth_scene_scale"),
            "selected": False,
        }
        records.append(record)
        scored.append((warp.coverage, camera, image, source_depth, source_alpha_plane, warp))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected_uids = {int(getattr(item[1], "uid", -1)) for item in scored[:k] if item[0] > 0.0}
    for record in records:
        if record.get("uid") in selected_uids:
            record["selected"] = True

    aligned: list[MultiViewInput] = []
    for coverage, camera, image, source_depth, source_alpha_plane, warp in scored[:k]:
        if coverage <= 0.0:
            continue
        reverse = build_reprojection_grid(
            camera,
            target_camera,
            source_depth,
            target_depth,
            target_alpha_plane,
            target_alpha=source_alpha_plane,
            depth_abs_tolerance=depth_abs_tolerance,
            depth_rel_tolerance=depth_rel_tolerance,
            depth_scene_scale=depth_scene_scale,
            alpha_threshold=alpha_threshold,
            min_depth=min_depth,
        )
        aligned.append(
            MultiViewInput(
                name=str(getattr(camera, "image_name", getattr(camera, "uid", "neighbor"))),
                image=image,
                camera=camera,
                depth=source_depth,
                alpha=source_alpha_plane,
                warped_image=warp_image(image, warp),
                valid_mask=warp.valid_mask,
                pixel_flow=warp.pixel_flow,
                sample_grid=warp.grid,
                source_to_target_flow=reverse.pixel_flow,
                reverse_valid_mask=reverse.valid_mask,
                weight=float(coverage),
                metadata={
                    "coverage": float(coverage),
                    "reverse_coverage": float(reverse.coverage),
                    "reverse_source": "depth",
                    **dict(warp.metadata),
                },
            )
        )
    return aligned, records


def correspondence_from_warps(
    forward: WarpResult,
    reverse: WarpResult | None = None,
    *,
    camera_metadata: dict[str, Any] | None = None,
) -> GeometryCorrespondence:
    """Pack a verified target→source warp and an optional depth-based reverse."""

    source_size = tuple(forward.metadata.get("source_size", ()))
    target_size = tuple(forward.metadata.get("target_size", ()))
    if len(source_size) != 2 or len(target_size) != 2:
        raise ValueError("WarpResult metadata must include source_size and target_size")
    return GeometryCorrespondence(
        target_to_source_flow=forward.pixel_flow,
        valid_mask=forward.valid_mask,
        source_size=(int(source_size[0]), int(source_size[1])),
        target_size=(int(target_size[0]), int(target_size[1])),
        target_to_source_grid=forward.grid,
        source_to_target_flow=None if reverse is None else reverse.pixel_flow,
        reverse_valid_mask=None if reverse is None else reverse.valid_mask,
        confidence=forward.coverage,
        camera_metadata=dict(camera_metadata or {}),
    )


def select_neighbor_cameras(
    target_camera: Any,
    candidates: Sequence[Any],
    *,
    k: int = 2,
    target_depth: torch.Tensor | None = None,
    source_depths: Mapping[Any, torch.Tensor] | None = None,
    target_alpha: torch.Tensor | None = None,
    source_alphas: Mapping[Any, torch.Tensor] | None = None,
) -> list[Any]:
    """Select neighbors by reprojection coverage when depth is available.

    Without depth, the function falls back to camera-center distance, matching
    the MVP diagnostic behavior while clearly recording that geometric overlap
    was not evaluated.
    """

    if k <= 0:
        return []
    filtered = [
        camera
        for camera in candidates
        if camera is not target_camera and getattr(camera, "uid", None) != getattr(target_camera, "uid", None)
    ]
    scored: list[tuple[float, Any]] = []
    if target_depth is not None and source_depths is not None:
        for camera, warp in rank_neighbor_cameras(
            target_camera,
            filtered,
            target_depth=target_depth,
            source_depths=source_depths,
            target_alpha=target_alpha,
            source_alphas=source_alphas,
            k=k,
        ):
            scored.append((warp.coverage, camera))
    else:
        target_center = getattr(target_camera, "camera_center", None)
        if target_center is not None:
            target_center = torch.as_tensor(target_center).detach().cpu().float()
        for camera in filtered:
            center = getattr(camera, "camera_center", None)
            if target_center is None or center is None:
                score = 0.0
            else:
                center = torch.as_tensor(center).detach().cpu().float()
                score = 1.0 / (float(torch.linalg.norm(center - target_center).item()) + 1e-6)
            scored.append((score, camera))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [camera for _, camera in scored[:k]]


def camera_center_world(camera: Any) -> torch.Tensor | None:
    """World-space camera center using the repository R/T convention."""

    center = getattr(camera, "camera_center", None)
    if center is not None:
        tensor = torch.as_tensor(center).detach().float().reshape(-1)
        if tensor.numel() >= 3:
            return tensor[:3]
    try:
        rotation = torch.as_tensor(camera.R).detach().float().reshape(3, 3)
        translation = torch.as_tensor(camera.T).detach().float().reshape(3)
    except (AttributeError, ValueError, TypeError, RuntimeError):
        return None
    return -(translation @ rotation.transpose(0, 1))


def estimate_spatial_target(
    camera: Any,
    depth: torch.Tensor,
    alpha: torch.Tensor | None = None,
    *,
    alpha_threshold: float = 1e-4,
    min_depth: float = 1e-4,
    max_points: int = 4096,
    min_points: int = 32,
) -> SpatialTarget:
    """Unproject valid zoom-surface pixels into a world-space target region."""

    depth = _as_plane(depth, name="target_depth").float()
    alpha_plane = _as_plane(alpha, name="target_alpha").float() if alpha is not None else None
    if alpha_plane is not None and alpha_plane.shape != depth.shape:
        raise ValueError(
            "depth and alpha must have the same spatial shape, "
            f"got {tuple(depth.shape)} and {tuple(alpha_plane.shape)}"
        )

    valid = torch.isfinite(depth) & (depth > float(min_depth))
    if alpha_plane is not None:
        valid &= torch.isfinite(alpha_plane) & (alpha_plane > float(alpha_threshold))
    coverage = float(valid.float().mean().item()) if valid.numel() else 0.0
    point_count = int(valid.sum().item())
    metadata = {
        "image_size": [int(camera.image_width), int(camera.image_height)],
        "uid": int(getattr(camera, "uid", -1)),
        "image_name": str(getattr(camera, "image_name", "")),
        "alpha_check": alpha_plane is not None,
        "min_depth": float(min_depth),
        "alpha_threshold": float(alpha_threshold),
    }
    if point_count == 0:
        return SpatialTarget(
            centroid_world=(0.0, 0.0, 0.0),
            median_depth=0.0,
            confidence=0.0,
            valid_coverage=coverage,
            point_count=0,
            points_world=None,
            metadata=metadata,
        )

    points_camera = _unproject_pixel_grid(camera, depth)
    points_world = camera_to_world(points_camera, camera)[valid.reshape(-1)]
    centroid = points_world.mean(dim=0)
    valid_depth = depth[valid]
    median_depth = float(valid_depth.median().item())
    if points_world.shape[0] > int(max_points):
        index = torch.linspace(
            0, points_world.shape[0] - 1, int(max_points), device=points_world.device
        ).round().long()
        points_world = points_world[index]
    confidence = coverage if point_count >= int(min_points) else 0.0
    metadata["stored_points"] = int(points_world.shape[0])
    return SpatialTarget(
        centroid_world=(float(centroid[0].item()), float(centroid[1].item()), float(centroid[2].item())),
        median_depth=median_depth,
        confidence=float(confidence),
        valid_coverage=coverage,
        point_count=point_count,
        points_world=points_world,
        metadata=metadata,
    )


def project_spatial_roi(
    spatial_target: SpatialTarget,
    camera: Any,
    *,
    zoom_factor: float,
    min_depth: float = 1e-4,
    min_in_frustum: float = 1e-3,
) -> ProjectedROI | None:
    """Project a 3D target into ``camera`` and return a zoom ROI.

    The ROI center comes from the projected world points.  Its size is the
    zoom window ``1/zoom_factor``, not a copy of another camera's normalized
    rectangle.
    """

    points = spatial_target.points_world
    if points is None or spatial_target.point_count <= 0:
        return None
    if zoom_factor <= 1.0:
        raise ValueError(f"zoom_factor must be > 1.0, got {zoom_factor}")

    points = torch.as_tensor(points)
    if points.ndim != 2 or points.shape[-1] != 3 or points.shape[0] == 0:
        return None
    uv, source_z = project_world_points(points, camera)
    width = float(camera.image_width)
    height = float(camera.image_height)
    in_front = torch.isfinite(source_z) & (source_z > float(min_depth))
    in_bounds = (
        in_front
        & torch.isfinite(uv).all(dim=-1)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] <= width - 1.0)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] <= height - 1.0)
    )
    in_frustum = float(in_bounds.float().mean().item()) if in_bounds.numel() else 0.0
    if in_frustum < float(min_in_frustum) or not bool(in_bounds.any()):
        return None

    u = uv[in_bounds, 0] / width
    v = uv[in_bounds, 1] / height
    center_x = float(u.median().item())
    center_y = float(v.median().item())
    span_x = float((u.quantile(0.9) - u.quantile(0.1)).item()) if u.numel() > 1 else 0.0
    span_y = float((v.quantile(0.9) - v.quantile(0.1)).item()) if v.numel() > 1 else 0.0

    visible = 1.0 / float(zoom_factor)
    half = 0.5 * visible
    clamped_x = min(max(center_x, half), 1.0 - half)
    clamped_y = min(max(center_y, half), 1.0 - half)
    clamped = abs(clamped_x - center_x) > 1e-4 or abs(clamped_y - center_y) > 1e-4
    confidence = float(in_frustum * spatial_target.confidence)
    if clamped:
        confidence *= 0.5
    return ProjectedROI(
        center_x=clamped_x,
        center_y=clamped_y,
        width=visible,
        height=visible,
        in_frustum_fraction=in_frustum,
        clamped=clamped,
        confidence=confidence,
        uid=int(getattr(camera, "uid", -1)),
        image_name=str(getattr(camera, "image_name", "")),
        unclamped_center=(center_x, center_y),
        projected_span=(span_x, span_y),
    )


def _exclude_camera(camera: Any, excluded_uids: set[int], target_camera: Any | None) -> bool:
    uid = getattr(camera, "uid", None)
    if uid is not None and int(uid) in excluded_uids:
        return True
    if target_camera is not None and camera is target_camera:
        return True
    return False


def pool_cameras_by_distance(
    target_camera: Any,
    candidates: Sequence[Any],
    *,
    k: int,
    exclude_uids: Iterable[int] | None = None,
) -> list[Any]:
    """Keep the ``k`` nearest cameras by world-space center, excluding poses."""

    excluded = {int(uid) for uid in (exclude_uids or ())}
    filtered = [camera for camera in candidates if not _exclude_camera(camera, excluded, target_camera)]
    if k <= 0 or len(filtered) <= k:
        return filtered
    target_center = camera_center_world(target_camera)
    if target_center is None:
        return list(filtered[:k])
    scored: list[tuple[float, Any]] = []
    for camera in filtered:
        center = camera_center_world(camera)
        if center is None:
            scored.append((float("inf"), camera))
            continue
        delta = center.to(device=target_center.device, dtype=target_center.dtype) - target_center
        scored.append((float(torch.linalg.norm(delta).item()), camera))
    scored.sort(key=lambda item: item[0])
    return [camera for _, camera in scored[:k]]


def rank_cameras_by_spatial_overlap(
    spatial_target: SpatialTarget,
    candidates: Sequence[Any],
    *,
    zoom_factor: float,
    k: int,
    pool_size: int | None = None,
    target_camera: Any | None = None,
    exclude_uids: Iterable[int] | None = None,
    min_depth: float = 1e-4,
    min_in_frustum: float = 1e-3,
) -> tuple[list[tuple[Any, ProjectedROI]], list[dict[str, Any]]]:
    """Rank cameras by projected overlap with a 3D target, not copied image ROIs.

    A camera-center pool is applied first when ``pool_size`` is set, then each
    remaining camera receives a spatially projected zoom ROI.
    """

    excluded = {int(uid) for uid in (exclude_uids or ())}
    pool = list(candidates)
    if not pool or k <= 0:
        return [], []
    if pool_size is not None:
        anchor = target_camera if target_camera is not None else pool[0]
        pool = pool_cameras_by_distance(
            anchor,
            pool,
            k=int(pool_size),
            exclude_uids=excluded,
        )
    else:
        pool = [camera for camera in pool if not _exclude_camera(camera, excluded, target_camera)]

    records: list[dict[str, Any]] = []
    scored: list[tuple[float, Any, ProjectedROI]] = []
    for camera in pool:
        projected = project_spatial_roi(
            spatial_target,
            camera,
            zoom_factor=zoom_factor,
            min_depth=min_depth,
            min_in_frustum=min_in_frustum,
        )
        record = {
            "uid": int(getattr(camera, "uid", -1)),
            "image_name": str(getattr(camera, "image_name", "")),
            "selected": False,
        }
        if projected is None:
            record["skipped"] = "low_in_frustum"
            records.append(record)
            continue
        record.update(projected.to_dict())
        records.append(record)
        scored.append((projected.confidence, camera, projected))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [(camera, projected) for _, camera, projected in scored[: max(0, int(k))]]
    selected_uids = {int(getattr(camera, "uid", -1)) for camera, _ in selected}
    for record in records:
        if record.get("uid") in selected_uids and "skipped" not in record:
            record["selected"] = True
    return selected, records


__all__ = [
    "build_aligned_multiview_inputs",
    "build_reprojection_grid",
    "correspondence_from_warps",
    "camera_center_world",
    "camera_to_world",
    "estimate_spatial_target",
    "compute_depth_tolerance",
    "estimate_depth_scene_scale",
    "expected_surface_depth",
    "normalize_scene_scale",
    "pixel_displacement_flow",
    "pixel_footprint",
    "pool_cameras_by_distance",
    "project_spatial_roi",
    "project_world_points",
    "rank_cameras_by_spatial_overlap",
    "rank_neighbor_cameras",
    "select_neighbor_cameras",
    "warp_image",
    "world_to_camera",
]
