from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from prf.cameras import load_training_views
from prf.config import PRFConfig
from prf.fusion import fuse_resolutions
from prf.io_gs import apply_similarity, assert_metric, covariance_world
from prf.kernel import compute_kernel_resolution
from prf.obs import single_view_sampling_resolution
from prf.patches import build_surface_patch
from prf.pipeline import build_physical_resolution_field
from prf.spacing import compute_spacing_resolution
from prf.types import GaussianScene, PinholeView, SurfacePatch


def _identity_quat(n: int) -> np.ndarray:
    q = np.zeros((n, 4), dtype=np.float64)
    q[:, 0] = 1.0
    return q


def _plane_lattice(spacing: float = 0.5, sigma: float = 0.10, n: int = 12) -> GaussianScene:
    xs, ys = np.meshgrid(np.arange(n) * spacing, np.arange(n) * spacing)
    mu = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], axis=1).astype(np.float64)
    mu -= mu.mean(axis=0)
    scales = np.full((mu.shape[0], 3), sigma, dtype=np.float64)
    scales[:, 2] = sigma * 0.15
    return GaussianScene(
        mu=mu,
        scales=scales,
        rotations=_identity_quat(mu.shape[0]),
        opacity=np.ones(mu.shape[0], dtype=np.float64),
        normals=np.repeat(np.array([[0.0, 0.0, 1.0]]), mu.shape[0], axis=0),
    )


def _nadir_view(height: float = 100.0, focal: float = 500.0, size: int = 512) -> PinholeView:
    c2w = np.eye(4)
    c2w[:3, 3] = [0.0, 0.0, height]
    # OpenCV: +Z forward. Looking down -world-Z means camera +Z = -world Z.
    c2w[:3, 0] = [1.0, 0.0, 0.0]
    c2w[:3, 1] = [0.0, -1.0, 0.0]
    c2w[:3, 2] = [0.0, 0.0, -1.0]
    return PinholeView(
        view_id="nadir",
        width=size,
        height=size,
        fx=focal,
        fy=focal,
        cx=size / 2.0,
        cy=size / 2.0,
        c2w=c2w,
        w2c=np.linalg.inv(c2w),
        center=c2w[:3, 3].copy(),
    )


def test_metric_assert_rejects_tiny_normalized_scene():
    scene = GaussianScene(
        mu=np.random.default_rng(0).normal(size=(32, 3)) * 0.1,
        scales=np.full((32, 3), 1e-4),
        rotations=_identity_quat(32),
        opacity=np.ones(32),
    )
    with pytest.raises(ValueError, match="metric-scale"):
        assert_metric(scene, PRFConfig())


def test_kernel_and_spacing_on_known_lattice():
    cfg = PRFConfig(patch_knn=48, min_patch_gaussians=8, use_gaussian_normals=False)
    scene = _plane_lattice(spacing=0.5, sigma=0.10, n=21)
    tree = cKDTree(scene.mu)
    center_id = int(np.argmin(np.linalg.norm(scene.mu, axis=1)))
    patch = build_surface_patch(scene, tree, center_id, 0, cfg)
    assert patch.valid
    assert patch.area_m2 > 0.0
    r_kernel = compute_kernel_resolution(patch, scene, cfg)
    r_spacing = compute_spacing_resolution(patch, scene, cfg)
    assert r_kernel == pytest.approx(cfg.bandwidth_coeff * 0.10, rel=0.15)
    assert 0.45 <= r_spacing <= 1.2

    coarse = _plane_lattice(spacing=1.0, sigma=0.10, n=21)
    coarse_tree = cKDTree(coarse.mu)
    coarse_id = int(np.argmin(np.linalg.norm(coarse.mu, axis=1)))
    coarse_patch = build_surface_patch(coarse, coarse_tree, coarse_id, 0, cfg)
    r_spacing_coarse = compute_spacing_resolution(coarse_patch, coarse, cfg)
    assert r_spacing_coarse / r_spacing == pytest.approx(2.0, rel=0.2)


def test_pinhole_jacobian_matches_d_over_f():
    patch = SurfacePatch(
        patch_id=0,
        center=np.zeros(3),
        normal=np.array([0.0, 0.0, 1.0]),
        t1=np.array([1.0, 0.0, 0.0]),
        t2=np.array([0.0, 1.0, 0.0]),
        tangent=np.eye(3)[:, :2],
        ids=np.array([0]),
        radius=1.0,
        valid=True,
    )
    view = _nadir_view(height=100.0, focal=500.0)
    cfg = PRFConfig()
    r = single_view_sampling_resolution(patch, view, cfg)
    assert r == pytest.approx(100.0 / 500.0, rel=1e-3)


def test_fusion_takes_worst_bottleneck():
    patch = SurfacePatch(
        patch_id=0,
        center=np.zeros(3),
        normal=np.array([0.0, 0.0, 1.0]),
        t1=np.array([1.0, 0.0, 0.0]),
        t2=np.array([0.0, 1.0, 0.0]),
        tangent=np.eye(3)[:, :2],
        ids=np.array([0, 1]),
        radius=1.0,
        valid=True,
    )
    rec = fuse_resolutions(
        patch,
        0.10,
        0.18,
        0.14,
        {"num_visible_real_views": 4, "best_camera_pair": ("a", "b"), "best_pair_angle_deg": 20.0, "quality_flags": []},
    )
    assert rec.bottleneck_type == "KERNEL"
    assert rec.R_phys_m_per_equiv_pixel == pytest.approx(0.18)
    assert rec.R_phys_cm_per_equiv_pixel == pytest.approx(18.0)
    assert rec.patch_radius_m == pytest.approx(1.0)
    assert rec.patch_area_m2 == pytest.approx(0.0)


def test_covariance_uses_squared_scales():
    cov = covariance_world(_identity_quat(1), np.array([[0.2, 0.1, 0.05]]))[0]
    assert np.allclose(np.diag(cov), [0.04, 0.01, 0.0025])


def test_kernel_spacing_ignore_extra_cameras():
    scene = _plane_lattice()
    cfg = PRFConfig(
        max_anchors=16,
        patch_knn=24,
        use_gaussian_normals=False,
        min_pair_angle_deg=1.0,
        metric_extent_min_m=1.0,
        metric_scale_min_m=1e-3,
    )
    view_a = _nadir_view(100.0, 500.0)
    view_b = _nadir_view(100.0, 500.0)
    view_b.view_id = "side"
    view_b.center = np.array([40.0, 0.0, 100.0])
    view_b.c2w = view_a.c2w.copy()
    view_b.c2w[:3, 3] = view_b.center
    view_b.w2c = np.linalg.inv(view_b.c2w)
    field_a = build_physical_resolution_field(scene, [view_a], cfg)
    field_b = build_physical_resolution_field(scene, [view_a, view_b], cfg)
    assert field_a.records and field_b.records
    assert [rec.R_kernel_m_per_equiv_pixel for rec in field_a.records] == [
        rec.R_kernel_m_per_equiv_pixel for rec in field_b.records
    ]
    assert [rec.R_spacing_m_per_equiv_pixel for rec in field_a.records] == [
        rec.R_spacing_m_per_equiv_pixel for rec in field_b.records
    ]


def test_load_skyfall_transforms_roundtrip(tmp_path: Path):
    payload = {
        "camera_model": "PINHOLE",
        "w": 64,
        "h": 64,
        "frames": [
            {
                "file_path": "./images/cam0.png",
                "fl_x": 100.0,
                "fl_y": 100.0,
                "cx": 32.0,
                "cy": 32.0,
                "transform_matrix": np.eye(4).tolist(),
            }
        ],
    }
    path = tmp_path / "transforms.json"
    import json

    path.write_text(json.dumps(payload))
    views = load_training_views(path)
    assert views[0].view_id == "cam0"
    assert views[0].fx == 100.0


def test_load_views_uses_global_intrinsics(tmp_path: Path):
    payload = {
        "w": 32,
        "h": 32,
        "fl_x": 50.0,
        "fl_y": 50.0,
        "cx": 16.0,
        "cy": 16.0,
        "frames": [{"file_path": "renders/nadir/000.png", "transform_matrix": np.eye(4).tolist()}],
    }
    path = tmp_path / "transforms.json"
    import json

    path.write_text(json.dumps(payload))
    views = load_training_views(path)
    assert views[0].fx == 50.0
    assert views[0].file_path == "renders/nadir/000.png"


def test_topdown_map_covers_lattice_points():
    from prf.visualize import topdown_map

    xy = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    values = np.array([1.0, 2.0, 3.0])
    raster, _ = topdown_map(xy, values, size=64, radius_m=2.0)
    assert np.isfinite(raster).sum() > 10
    assert np.nanmin(raster) >= 1.0 - 1e-6
    assert np.nanmax(raster) <= 3.0 + 1e-6


def test_similarity_scales_positions_and_scales():
    scene = _plane_lattice(spacing=1.0, sigma=0.2, n=4)
    scaled = apply_similarity(scene, 2.0, np.eye(3), np.zeros(3))
    assert np.allclose(scaled.mu, scene.mu * 2.0)
    assert np.allclose(scaled.scales, scene.scales * 2.0)
