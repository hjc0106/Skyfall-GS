from __future__ import annotations

import math

import numpy as np
import pytest

from prf.config import PRFConfig
from prf.obs import compute_observation_resolution
from prf.occlusion import classify_patch_view, classify_sample, depth_tolerance_m
from prf.types import SurfacePatch
from prf.visibility import (
    INVALID_PROJECTION,
    OCCLUDED,
    OUTSIDE_FRUSTUM,
    UNCERTAIN,
    VISIBLE,
    aggregate_visibility,
)
from tests.test_prf import _nadir_view, _plane_lattice


def _flat_patch() -> SurfacePatch:
    return SurfacePatch(
        patch_id=0,
        center=np.zeros(3),
        normal=np.array([0.0, 0.0, 1.0]),
        t1=np.array([1.0, 0.0, 0.0]),
        t2=np.array([0.0, 1.0, 0.0]),
        tangent=np.eye(3)[:, :2],
        ids=np.array([0, 1, 2, 3], dtype=np.int64),
        radius=1.0,
        valid=True,
        area_m2=1.0,
        hull_uv=np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]),
        thickness_m=0.2,
    )


def test_frustum_outside_and_invalid():
    patch = _flat_patch()
    view = _nadir_view(height=100.0, focal=500.0, size=64)
    cfg = PRFConfig()
    far = SurfacePatch(**{**patch.__dict__, "center": np.array([50.0, 0.0, 0.0])})
    assert classify_patch_view(far, view, cfg) == OUTSIDE_FRUSTUM
    behind = SurfacePatch(**{**patch.__dict__, "center": np.array([0.0, 0.0, 200.0])})
    assert classify_patch_view(behind, view, cfg) == OUTSIDE_FRUSTUM


def test_occluded_behind_closer_surface():
    patch = _flat_patch()
    view = _nadir_view(height=100.0, focal=500.0, size=32)
    cfg = PRFConfig()
    depth = np.full((32, 32), 40.0, dtype=np.float64)
    alpha = np.ones((32, 32), dtype=np.float64)
    assert classify_patch_view(patch, view, cfg, depth=depth, alpha=alpha) == OCCLUDED


def test_visible_when_depth_matches_surface():
    patch = _flat_patch()
    view = _nadir_view(height=100.0, focal=500.0, size=32)
    cfg = PRFConfig()
    depth = np.full((32, 32), 100.0, dtype=np.float64)
    alpha = np.ones((32, 32), dtype=np.float64)
    assert classify_patch_view(patch, view, cfg, depth=depth, alpha=alpha) == VISIBLE


def test_uncertain_translucent_and_empty_alpha():
    patch = _flat_patch()
    view = _nadir_view(height=100.0, focal=500.0, size=32)
    cfg = PRFConfig()
    depth = np.full((32, 32), 100.0, dtype=np.float64)
    mid = np.full((32, 32), 0.2, dtype=np.float64)
    empty = np.full((32, 32), 0.01, dtype=np.float64)
    assert classify_patch_view(patch, view, cfg, depth=depth, alpha=mid) == UNCERTAIN
    assert classify_patch_view(patch, view, cfg, depth=depth, alpha=empty) == UNCERTAIN


def test_tolerance_grows_with_scale_and_grazing_angle():
    cfg = PRFConfig()
    view = _nadir_view(height=20.0, focal=500.0)
    scene = _plane_lattice(spacing=0.5, sigma=0.10, n=9)
    patch = _flat_patch()
    patch.ids = np.arange(scene.num_gaussians)
    patch.thickness_m = 0.0
    small = depth_tolerance_m(patch, view, cfg, scene=scene)
    grown = _plane_lattice(spacing=0.5, sigma=0.40, n=9)
    large = depth_tolerance_m(patch, view, cfg, scene=grown)
    assert large > small * 1.5

    grazing = _flat_patch()
    grazing.normal = np.array([1.0, 0.0, 0.0])
    tau_grazing = depth_tolerance_m(grazing, view, cfg)
    tau_nadir = depth_tolerance_m(_flat_patch(), view, cfg)
    assert tau_grazing > tau_nadir


def test_aggregate_requires_reliable_visible():
    assert aggregate_visibility(VISIBLE, [VISIBLE, VISIBLE]) == VISIBLE
    assert aggregate_visibility(VISIBLE, [VISIBLE, OCCLUDED]) == OCCLUDED
    assert aggregate_visibility(VISIBLE, [UNCERTAIN]) == UNCERTAIN
    assert aggregate_visibility(OUTSIDE_FRUSTUM, [VISIBLE]) == OUTSIDE_FRUSTUM
    assert aggregate_visibility(VISIBLE, [OUTSIDE_FRUSTUM]) == UNCERTAIN


def test_invalid_projection_on_nan_depth():
    patch = _flat_patch()
    view = _nadir_view(height=100.0, focal=500.0, size=8)
    cfg = PRFConfig()
    depth = np.full((8, 8), np.nan)
    alpha = np.ones((8, 8))
    assert classify_sample(patch.center, view, cfg, depth=depth, alpha=alpha, patch=patch) == INVALID_PROJECTION


def test_expected_depth_divides_accumulated_by_alpha():
    from prf.gs_depth import expected_depth_from_accumulated

    alpha = np.array([[0.5, 1.0], [0.0, 0.25]], dtype=np.float32)
    accumulated = np.array([[50.0, 100.0], [0.0, 20.0]], dtype=np.float32)
    expected = expected_depth_from_accumulated(accumulated, alpha)
    assert expected[0, 0] == pytest.approx(100.0)
    assert expected[0, 1] == pytest.approx(100.0)
    assert expected[1, 1] == pytest.approx(80.0)
    assert np.isnan(expected[1, 0])
    patch = _flat_patch()
    cfg = PRFConfig(min_pair_angle_deg=1.0)
    view_a = _nadir_view(100.0, 500.0, size=512)
    view_b = _nadir_view(100.0, 500.0, size=512)
    view_b.view_id = "side"
    view_b.center = np.array([40.0, 0.0, 100.0])
    view_b.c2w = view_a.c2w.copy()
    view_b.c2w[:3, 3] = view_b.center
    view_b.w2c = np.linalg.inv(view_b.c2w)
    depth_ok = np.full((512, 512), 100.0)
    depth_occ = np.full((512, 512), 20.0)
    alpha = np.ones((512, 512))
    maps = {"nadir": (depth_ok, alpha), "side": (depth_occ, alpha)}
    r_obs, info = compute_observation_resolution(patch, [view_a, view_b], cfg, occlusion_maps=maps)
    assert not math.isfinite(r_obs)
    assert info["best_camera_pair"] is None
    assert info["visibility_states"]["nadir"] == VISIBLE
    assert info["visibility_states"]["side"] == OCCLUDED
