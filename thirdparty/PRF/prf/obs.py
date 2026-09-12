from __future__ import annotations

import math

import numpy as np

from prf.config import PRFConfig
from prf.occlusion import OcclusionMaps, classify_patch_view, maps_for_view
from prf.types import GaussianScene, PinholeView, SurfacePatch
from prf.visibility import VISIBLE, viewing_angle_deg


def jacobian_sampling_resolution(patch: SurfacePatch, view: PinholeView, cfg: PRFConfig) -> float:
    delta = float(np.clip(cfg.jacobian_delta_ratio * patch.radius, cfg.delta_min_m, cfg.delta_max_m))
    p0, z0 = view.project(patch.center)
    p1, z1 = view.project(patch.center + delta * patch.t1)
    p2, z2 = view.project(patch.center + delta * patch.t2)
    if min(float(z0), float(z1), float(z2)) <= cfg.min_camera_z:
        return math.inf
    if not (np.isfinite(p0).all() and np.isfinite(p1).all() and np.isfinite(p2).all()):
        return math.inf
    jacobian = np.column_stack((p1 - p0, p2 - p0)) / delta
    gamma = np.linalg.svd(jacobian, compute_uv=False)
    gamma_min = float(np.min(gamma))
    if gamma_min <= cfg.eps:
        return math.inf
    return 1.0 / gamma_min


def single_view_sampling_resolution(patch: SurfacePatch, view: PinholeView, cfg: PRFConfig) -> float:
    if classify_patch_view(patch, view, cfg) != VISIBLE:
        return math.inf
    return jacobian_sampling_resolution(patch, view, cfg)


def compute_observation_resolution(
    patch: SurfacePatch,
    views: list[PinholeView],
    cfg: PRFConfig,
    *,
    scene: GaussianScene | None = None,
    occlusion_maps: OcclusionMaps | None = None,
) -> tuple[float, dict]:
    candidates: list[tuple[PinholeView, float]] = []
    states: dict[str, str] = {}
    for view in views:
        depth, alpha = maps_for_view(occlusion_maps, view)
        state = classify_patch_view(patch, view, cfg, depth=depth, alpha=alpha, scene=scene)
        states[view.view_id] = state
        if state != VISIBLE:
            continue
        resolution = jacobian_sampling_resolution(patch, view, cfg)
        if math.isfinite(resolution):
            candidates.append((view, resolution))

    best_value = math.inf
    best_pair: tuple[str, str] | None = None
    best_angle = 0.0
    for i, (view_i, res_i) in enumerate(candidates):
        for view_j, res_j in candidates[i + 1 :]:
            angle = viewing_angle_deg(view_i, view_j, patch.center)
            if angle < cfg.min_pair_angle_deg:
                continue
            pair_r = max(res_i, res_j)
            if pair_r < best_value:
                best_value = pair_r
                best_pair = (view_i.view_id, view_j.view_id)
                best_angle = angle

    flags = list(patch.quality_flags)
    if not candidates:
        flags.append("NO_VISIBLE_TRAINING_VIEW")
    if best_pair is None:
        flags.append("NO_VALID_MULTIVIEW_PAIR")
        return math.inf, {
            "num_visible_real_views": len(candidates),
            "best_camera_pair": None,
            "best_pair_angle_deg": 0.0,
            "quality_flags": flags,
            "visibility_states": states,
        }
    if len(candidates) < 2:
        flags.append("LOW_VISIBILITY")
    return best_value, {
        "num_visible_real_views": len(candidates),
        "best_camera_pair": best_pair,
        "best_pair_angle_deg": best_angle,
        "quality_flags": flags,
        "visibility_states": states,
    }
