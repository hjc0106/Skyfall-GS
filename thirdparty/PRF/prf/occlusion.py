from __future__ import annotations

import numpy as np

from prf.config import PRFConfig
from prf.io_gs import covariance_world
from prf.patches import sample_points
from prf.types import GaussianScene, PinholeView, SurfacePatch
from prf.visibility import (
    INVALID_PROJECTION,
    OCCLUDED,
    UNCERTAIN,
    VISIBLE,
    aggregate_visibility,
    project_status,
)

OcclusionMaps = dict[str, tuple[np.ndarray, np.ndarray]]


def sample_bilinear(image: np.ndarray, uv: np.ndarray) -> float:
    """Sample a 2D map at pixel coordinate (u, v); origin is the top-left pixel center."""
    height, width = image.shape[:2]
    u = float(uv[0])
    v = float(uv[1])
    if u < 0.0 or v < 0.0 or u > width - 1.0 or v > height - 1.0:
        return float("nan")
    x0 = int(np.floor(u))
    y0 = int(np.floor(v))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    du = u - x0
    dv = v - y0
    return float(
        (1.0 - du) * (1.0 - dv) * image[y0, x0]
        + du * (1.0 - dv) * image[y0, x1]
        + (1.0 - du) * dv * image[y1, x0]
        + du * dv * image[y1, x1]
    )


def patch_thickness_m(patch: SurfacePatch, scene: GaussianScene | None, cfg: PRFConfig) -> float:
    if patch.thickness_m > 0.0:
        return float(patch.thickness_m)
    if scene is None or patch.ids.size == 0:
        return max(float(patch.radius) * 0.25, cfg.delta_min_m)
    cov = covariance_world(scene.rotations[patch.ids], scene.scales[patch.ids])
    sigma_n = np.sqrt(np.maximum(np.einsum("i,nij,j->n", patch.normal, cov, patch.normal), 0.0))
    return cfg.k_perp * float(np.median(sigma_n))


def depth_tolerance_m(
    patch: SurfacePatch,
    view: PinholeView,
    cfg: PRFConfig,
    scene: GaussianScene | None = None,
    point: np.ndarray | None = None,
) -> float:
    """View-dependent depth slack from Gaussian scale, patch thickness, and incidence."""
    origin = np.asarray(patch.center if point is None else point, dtype=np.float64)
    ray = origin - view.center
    ray_norm = float(np.linalg.norm(ray))
    if ray_norm <= cfg.eps:
        return cfg.delta_min_m
    ray = ray / ray_norm
    cos_inc = max(abs(float(np.dot(patch.normal, ray))), cfg.occlusion_min_cos)

    if scene is not None and patch.ids.size:
        cov = covariance_world(scene.rotations[patch.ids], scene.scales[patch.ids])
        sigma_ray = float(np.median(np.sqrt(np.maximum(np.einsum("i,nij,j->n", ray, cov, ray), 0.0))))
    else:
        sigma_ray = max(float(patch.radius) * 0.1, cfg.delta_min_m)

    sigma_thick = patch_thickness_m(patch, scene, cfg) / cos_inc
    _, z = view.project(origin)
    focal = 0.5 * (float(view.fx) + float(view.fy))
    sigma_pix = abs(float(z)) / max(focal, cfg.eps) / cos_inc
    # float32 expected-depth quantization grows with camera range
    sigma_quant = float(np.finfo(np.float32).eps) * abs(float(z))
    return cfg.occlusion_k * (sigma_ray + sigma_thick + sigma_pix + 8.0 * sigma_quant)


def classify_sample(
    point: np.ndarray,
    view: PinholeView,
    cfg: PRFConfig,
    *,
    depth: np.ndarray,
    alpha: np.ndarray,
    patch: SurfacePatch,
    scene: GaussianScene | None = None,
) -> str:
    uv, z = view.project(point)
    status = project_status(uv, float(z), view, cfg)
    if status != VISIBLE:
        return status
    depth_s = sample_bilinear(depth, uv)
    alpha_s = sample_bilinear(alpha, uv)
    if not np.isfinite(depth_s) or not np.isfinite(alpha_s):
        return INVALID_PROJECTION
    if alpha_s < cfg.occlusion_alpha_empty:
        return UNCERTAIN
    tau = depth_tolerance_m(patch, view, cfg, scene=scene, point=point)
    if float(z) > depth_s + tau:
        return OCCLUDED
    if alpha_s < cfg.occlusion_alpha_visible:
        return UNCERTAIN
    return VISIBLE


def classify_patch_view(
    patch: SurfacePatch,
    view: PinholeView,
    cfg: PRFConfig,
    *,
    depth: np.ndarray | None = None,
    alpha: np.ndarray | None = None,
    scene: GaussianScene | None = None,
) -> str:
    uv, z = view.project(patch.center)
    center_frustum = project_status(uv, float(z), view, cfg)
    if depth is None or alpha is None:
        return center_frustum
    if center_frustum != VISIBLE:
        return center_frustum
    points = sample_points(patch)
    states = [
        classify_sample(point, view, cfg, depth=depth, alpha=alpha, patch=patch, scene=scene)
        for point in points
    ]
    return aggregate_visibility(states[0], states[1:])


def maps_for_view(occlusion_maps: OcclusionMaps | None, view: PinholeView) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if occlusion_maps is None:
        return None, None
    payload = occlusion_maps.get(view.view_id)
    if payload is None:
        return None, None
    return payload
