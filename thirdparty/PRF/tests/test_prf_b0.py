from __future__ import annotations

import math

import numpy as np
import pytest

from prf.config import PRFConfig
from prf.kernel import compute_kernel_resolution
from prf.pipeline import build_valid_patches, evaluate_patches, scale_view_intrinsics
from prf.spacing import compute_spacing_resolution
from prf.types import GaussianScene, SurfacePatch
from tests.test_prf import _nadir_view, _plane_lattice


def _two_views():
    view_a = _nadir_view(100.0, 500.0, size=256)
    view_b = _nadir_view(100.0, 500.0, size=256)
    view_b.view_id = "side"
    view_b.center = np.array([40.0, 0.0, 100.0])
    view_b.c2w = view_a.c2w.copy()
    view_b.c2w[:3, 3] = view_b.center
    view_b.w2c = np.linalg.inv(view_b.c2w)
    return [view_a, view_b]


def test_b0_image_scale_obs_kernel_spacing():
    scene = _plane_lattice(spacing=0.5, sigma=0.10, n=16)
    cfg = PRFConfig(
        max_anchors=8,
        patch_knn=24,
        use_gaussian_normals=False,
        min_pair_angle_deg=1.0,
        metric_extent_min_m=1.0,
        metric_scale_min_m=1e-3,
    )
    patches = build_valid_patches(scene, cfg)
    views = _two_views()
    base, _ = evaluate_patches(patches, scene, views, cfg)
    assert base
    for scale in (0.5, 0.25):
        scaled_views = [scale_view_intrinsics(view, scale) for view in views]
        scaled, _ = evaluate_patches(patches, scene, scaled_views, cfg)
        assert len(scaled) == len(base)
        for rec_s, rec_0 in zip(scaled, base, strict=True):
            assert rec_s.R_kernel_m_per_equiv_pixel == pytest.approx(rec_0.R_kernel_m_per_equiv_pixel, rel=0, abs=0)
            assert rec_s.R_spacing_m_per_equiv_pixel == pytest.approx(rec_0.R_spacing_m_per_equiv_pixel, rel=0, abs=0)
            if math.isfinite(rec_0.R_obs_m_per_equiv_pixel):
                assert rec_s.R_obs_m_per_equiv_pixel == pytest.approx(
                    rec_0.R_obs_m_per_equiv_pixel / scale, rel=1e-6
                )


def test_b0_frozen_membership_gaussian_scale():
    scene = _plane_lattice(spacing=0.5, sigma=0.10, n=16)
    cfg = PRFConfig(patch_knn=24, min_patch_gaussians=8, use_gaussian_normals=False)
    patches = build_valid_patches(scene, cfg)
    assert patches
    grown = GaussianScene(
        mu=scene.mu,
        scales=scene.scales * 2.0,
        rotations=scene.rotations,
        opacity=scene.opacity,
        normals=scene.normals,
    )
    for patch in patches:
        r_k0 = compute_kernel_resolution(patch, scene, cfg)
        r_k1 = compute_kernel_resolution(patch, grown, cfg)
        r_s0 = compute_spacing_resolution(patch, scene, cfg)
        r_s1 = compute_spacing_resolution(patch, grown, cfg)
        assert r_k1 / r_k0 == pytest.approx(2.0, rel=1e-6)
        assert r_s1 == pytest.approx(r_s0, rel=0, abs=0)


def test_b0_frozen_membership_spacing_thins_monotone():
    scene = _plane_lattice(spacing=0.5, sigma=0.10, n=21)
    cfg = PRFConfig(patch_knn=48, min_patch_gaussians=8, use_gaussian_normals=False)
    patches = build_valid_patches(scene, cfg)
    assert patches
    thicker = 0
    for patch in patches:
        if patch.ids.size < 16:
            continue
        r0 = compute_spacing_resolution(patch, scene, cfg)
        thinned = SurfacePatch(**{**patch.__dict__, "ids": patch.ids[::2]})
        r1 = compute_spacing_resolution(thinned, scene, cfg)
        assert r1 > r0
        thicker += 1
    assert thicker > 0
