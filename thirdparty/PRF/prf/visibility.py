from __future__ import annotations

import numpy as np

from prf.config import PRFConfig
from prf.types import PinholeView, SurfacePatch

VISIBLE = "VISIBLE"
OCCLUDED = "OCCLUDED"
OUTSIDE_FRUSTUM = "OUTSIDE_FRUSTUM"
UNCERTAIN = "UNCERTAIN"
INVALID_PROJECTION = "INVALID_PROJECTION"

VISIBILITY_STATES = (VISIBLE, OCCLUDED, OUTSIDE_FRUSTUM, UNCERTAIN, INVALID_PROJECTION)
STATE_CODE = {name: index for index, name in enumerate(VISIBILITY_STATES)}


def pixel_in_image(uv: np.ndarray, view: PinholeView) -> bool:
    return 0.0 <= float(uv[0]) < float(view.width) and 0.0 <= float(uv[1]) < float(view.height)


def project_status(uv: np.ndarray, z: float, view: PinholeView, cfg: PRFConfig) -> str:
    if not np.isfinite(uv).all() or not np.isfinite(z):
        return INVALID_PROJECTION
    if float(z) <= cfg.min_camera_z:
        return OUTSIDE_FRUSTUM
    if not pixel_in_image(uv, view):
        return OUTSIDE_FRUSTUM
    return VISIBLE


def is_visible(patch: SurfacePatch, view: PinholeView, cfg: PRFConfig) -> bool:
    """Frustum + image-bound visibility. Kept for the frustum-only model."""
    uv, z = view.project(patch.center)
    return project_status(uv, float(z), view, cfg) == VISIBLE


def viewing_angle_deg(view_a: PinholeView, view_b: PinholeView, point: np.ndarray) -> float:
    d1 = view_a.center - point
    d2 = view_b.center - point
    n1 = np.linalg.norm(d1)
    n2 = np.linalg.norm(d2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cosine = float(np.clip(np.dot(d1, d2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def aggregate_visibility(center_state: str, sample_states: list[str]) -> str:
    """Conservative patch-view state. Only all-clear samples stay VISIBLE."""
    if center_state in (INVALID_PROJECTION, OUTSIDE_FRUSTUM):
        return center_state
    states = [center_state, *sample_states]
    if OCCLUDED in states:
        return OCCLUDED
    if OUTSIDE_FRUSTUM in states or UNCERTAIN in states or INVALID_PROJECTION in states:
        return UNCERTAIN
    return VISIBLE
