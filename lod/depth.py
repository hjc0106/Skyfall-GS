"""Canonical depth quantities for Skyfall ``diff_gauss`` and RaDe-GS.

Two quantities, never mixed:

    camera_z       -- along the optical axis, ``p_view.z``
    ray_distance   -- Euclidean length along the pinhole ray through the
                      Skyfall principal point ``(cx_pixel, cy_pixel)``

Conversion uses ``rln = 1 / ||((x-cx)/fx, (y-cy)/fy, 1)||``:

    ray_distance = camera_z / rln
    camera_z     = ray_distance * rln

Rasterizer outputs:

    Skyfall ``render_depth``  = Σ αT z          → divide by alpha → camera_z
    RaDe-GS ``out_depth``     = already expected  → ray_distance
                                (verified on JAX_068: rel err ~ 2e-7 vs
                                Skyfall camera_z / rln_principal)

Do **not** use image-center ``(W-1)/2`` for this conversion. Do **not** feed
raw Skyfall accum ``D`` into warping. Geometry losses must pick one quantity
and convert both buffers to it via :class:`DepthMap`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .camera import ndc_principal_to_pixel

DepthKind = Literal["camera_z", "ray_distance"]


def _as_hw(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    return value


def skyfall_expected_z(
    accum_depth: torch.Tensor,
    alpha: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Skyfall rasterizer stores ``D = Σ αT z``; this is camera Z."""

    return accum_depth / alpha.clamp(min=eps)


def rade_image_center_rln(
    height: int,
    width: int,
    fx: float,
    fy: float,
    *,
    device,
    dtype=torch.float32,
) -> torch.Tensor:
    """RaDe CUDA ``pixnf`` uses the image center. Not the warping conversion."""

    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    pixnf_x = (grid_x - 0.5 * float(width - 1)) / float(fx)
    pixnf_y = (grid_y - 0.5 * float(height - 1)) / float(fy)
    return torch.rsqrt(pixnf_x * pixnf_x + pixnf_y * pixnf_y + 1.0)


def principal_rln(
    height: int,
    width: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    *,
    device,
    dtype=torch.float32,
) -> torch.Tensor:
    """``rln`` through a pixel principal point ``(cx, cy)``."""

    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    pixnf_x = (grid_x - float(cx)) / float(fx)
    pixnf_y = (grid_y - float(cy)) / float(fy)
    return torch.rsqrt(pixnf_x * pixnf_x + pixnf_y * pixnf_y + 1.0)


def rln_principal_from_skyfall_camera(camera, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Principal-point ``rln`` for a Skyfall camera (NDC ``cx``/``cy`` → pixels)."""

    width = int(camera.image_width)
    height = int(camera.image_height)
    device = device if device is not None else camera.camera_center.device
    return principal_rln(
        height,
        width,
        float(camera.focal_x),
        float(camera.focal_y),
        ndc_principal_to_pixel(float(camera.cx), width),
        ndc_principal_to_pixel(float(camera.cy), height),
        device=device,
        dtype=dtype,
    )


def z_to_ray_distance(z: torch.Tensor, rln: torch.Tensor) -> torch.Tensor:
    return z / rln.clamp_min(1e-8)


def ray_distance_to_z(distance: torch.Tensor, rln: torch.Tensor) -> torch.Tensor:
    return distance * rln


def skyfall_ray_distance(
    accum_depth: torch.Tensor,
    alpha: torch.Tensor,
    rln_principal: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    return z_to_ray_distance(skyfall_expected_z(accum_depth, alpha, eps=eps), rln_principal)


def rade_camera_z(out_depth: torch.Tensor, rln_principal: torch.Tensor) -> torch.Tensor:
    return ray_distance_to_z(out_depth, rln_principal)


@dataclass
class DepthMap:
    """A depth buffer tagged with its geometric meaning."""

    value: torch.Tensor
    kind: DepthKind
    rln_principal: torch.Tensor
    alpha: torch.Tensor | None = None

    def camera_z(self) -> torch.Tensor:
        value = _as_hw(self.value)
        if self.kind == "camera_z":
            return value
        return ray_distance_to_z(value, _as_hw(self.rln_principal))

    def ray_distance(self) -> torch.Tensor:
        value = _as_hw(self.value)
        if self.kind == "ray_distance":
            return value
        return z_to_ray_distance(value, _as_hw(self.rln_principal))

    def as_kind(self, kind: DepthKind) -> torch.Tensor:
        if kind == "camera_z":
            return self.camera_z()
        if kind == "ray_distance":
            return self.ray_distance()
        raise ValueError(f"unknown depth kind {kind!r}")


def depth_from_skyfall(accum_depth: torch.Tensor, alpha: torch.Tensor, camera, *, eps: float = 1e-4) -> DepthMap:
    """Skyfall ``render_depth`` + ``render_alpha`` → camera Z."""

    rln = rln_principal_from_skyfall_camera(camera, device=accum_depth.device, dtype=torch.float32)
    return DepthMap(
        value=skyfall_expected_z(_as_hw(accum_depth), _as_hw(alpha), eps=eps),
        kind="camera_z",
        rln_principal=rln,
        alpha=_as_hw(alpha),
    )


def depth_from_rade(out_depth: torch.Tensor, camera, *, alpha: torch.Tensor | None = None) -> DepthMap:
    """RaDe-GS ``out_depth`` / ``render_depth`` → ray distance."""

    rln = rln_principal_from_skyfall_camera(camera, device=out_depth.device, dtype=torch.float32)
    return DepthMap(
        value=_as_hw(out_depth),
        kind="ray_distance",
        rln_principal=rln,
        alpha=None if alpha is None else _as_hw(alpha),
    )


def pair_depth_metrics(
    skyfall_accum: torch.Tensor,
    skyfall_alpha: torch.Tensor,
    rade_depth: torch.Tensor,
    rade_alpha: torch.Tensor,
    rln_center: torch.Tensor,
    rln_principal: torch.Tensor,
    opaque: torch.Tensor | None = None,
) -> dict[str, float]:
    """Diagnostic pairings. The accepted match is principal-ray distance."""

    d_sf = _as_hw(skyfall_accum.detach())
    a_sf = _as_hw(skyfall_alpha.detach())
    d_rd = _as_hw(rade_depth.detach())
    a_rd = _as_hw(rade_alpha.detach())
    z_sf = skyfall_expected_z(d_sf, a_sf)
    if opaque is None:
        opaque = a_sf > 0.05
    mask = opaque.bool() & torch.isfinite(z_sf) & torch.isfinite(d_rd)
    if not bool(mask.any().item()):
        return {"opaque_pixels": 0}

    def _rel(a, b) -> float:
        return float(((a - b).abs() / a.abs().clamp_min(1e-3))[mask].mean().item())

    def _l1(a, b) -> float:
        return float((a - b).abs()[mask].mean().item())

    euclid_center = z_to_ray_distance(z_sf, rln_center)
    euclid_principal = z_to_ray_distance(z_sf, rln_principal)
    return {
        "opaque_pixels": int(mask.sum().item()),
        "alpha_l1": _l1(a_sf, a_rd),
        "raw_accum_vs_rade_rel": _rel(d_sf, d_rd),
        "expected_z_vs_rade_rel": _rel(z_sf, d_rd),
        "expected_z_vs_rade_l1": _l1(z_sf, d_rd),
        "euclid_image_center_vs_rade_rel": _rel(euclid_center, d_rd),
        "euclid_principal_vs_rade_rel": _rel(euclid_principal, d_rd),
        "rade_over_expected_z": float((d_rd[mask] / z_sf[mask].clamp_min(1e-6)).median().item()),
        "accepted_quantity": "ray_distance",
        "accepted_pairing": "skyfall_(D/A)/rln_principal == rade_out_depth",
    }
