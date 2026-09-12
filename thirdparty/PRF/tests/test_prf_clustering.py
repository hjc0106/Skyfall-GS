from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from prf.config import PRFConfig
from prf.patches import (
    build_candidate_patches_at_anchor,
    build_surface_patch,
    hull_interior_margin,
    unsigned_normal_clusters,
)
from prf.pipeline import build_valid_patches
from prf.types import GaussianScene
from tests.test_prf import _identity_quat, _plane_lattice


def test_unsigned_clusters_ignore_normal_sign():
    up = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    ids, sizes = unsigned_normal_clusters(up, min_cos=np.cos(np.deg2rad(30.0)), seed=0)
    assert sizes == [3]
    assert set(ids.tolist()) == {0}


def test_unsigned_clusters_split_orthogonal_planes():
    normals = np.vstack(
        [np.tile([0.0, 0.0, 1.0], (10, 1)), np.tile([0.0, 1.0, 0.0], (8, 1))]
    )
    ids, sizes = unsigned_normal_clusters(normals, min_cos=np.cos(np.deg2rad(30.0)), seed=0)
    assert sizes[0] == 10
    assert sizes[1] == 8
    assert set(ids[:10].tolist()) == {0}
    assert set(ids[10:].tolist()) == {1}


def test_crease_anchor_is_mixed_not_knn_border():
    xs, ys = np.meshgrid(np.linspace(-4.0, 4.0, 17), np.linspace(0.4, 4.0, 12))
    floor = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    xs, zs = np.meshgrid(np.linspace(-4.0, 4.0, 17), np.linspace(0.4, 4.0, 12))
    wall = np.stack([xs.ravel(), np.zeros(xs.size), zs.ravel()], axis=1)
    mu = np.vstack([floor, wall])
    n = mu.shape[0]
    scales = np.full((n, 3), 0.08)
    scales[: floor.shape[0], 2] = 0.012
    scales[floor.shape[0] :, 1] = 0.012
    normals = np.vstack(
        [np.tile([0.0, 0.0, 1.0], (floor.shape[0], 1)), np.tile([0.0, 1.0, 0.0], (wall.shape[0], 1))]
    )
    scene = GaussianScene(
        mu=mu,
        scales=scales,
        rotations=_identity_quat(n),
        opacity=np.ones(n),
        normals=normals,
    )
    tree = cKDTree(scene.mu)
    cfg = PRFConfig(patch_knn=48, min_patch_gaussians=8, use_gaussian_normals=True, cluster_normals=True)
    crease = int(np.argmin(np.linalg.norm(scene.mu - np.array([0.0, 0.4, 0.0]), axis=1)))
    patch = build_surface_patch(scene, tree, crease, 0, cfg)
    assert patch.valid
    assert "MIXED_NEIGHBORHOOD" in patch.quality_flags
    assert patch.cluster_fraction < 0.75
    assert patch.inlier_fraction > 0.5
    assert abs(patch.normal[2]) > abs(patch.normal[1])
    if patch.hull_margin >= cfg.border_hull_margin:
        assert "BORDER" not in patch.quality_flags


def test_spatial_fallback_keeps_orientation_cluster():
    scene = _plane_lattice(spacing=0.5, sigma=0.10, n=21)
    tree = cKDTree(scene.mu)
    center_id = int(np.argmin(np.linalg.norm(scene.mu, axis=1)))
    _, nbr = tree.query(scene.mu[center_id], k=48)
    scene.mu = scene.mu.copy()
    scene.mu[nbr[1::2]] += np.array([0.0, 0.0, 30.0])
    cfg = PRFConfig(patch_knn=48, min_patch_gaussians=8, use_gaussian_normals=True, cluster_normals=True)
    patch = build_surface_patch(scene, cKDTree(scene.mu), center_id, 0, cfg)
    assert patch.valid
    assert patch.ids.size >= cfg.min_patch_gaussians


def test_interior_plane_is_not_mixed():
    scene = _plane_lattice(spacing=0.5, sigma=0.10, n=21)
    tree = cKDTree(scene.mu)
    cfg = PRFConfig(patch_knn=48, min_patch_gaussians=8, use_gaussian_normals=True, cluster_normals=True)
    center_id = int(np.argmin(np.linalg.norm(scene.mu, axis=1)))
    patch = build_surface_patch(scene, tree, center_id, 0, cfg)
    assert patch.valid
    assert "MIXED_NEIGHBORHOOD" not in patch.quality_flags
    assert patch.cluster_fraction == 1.0
    assert patch.inlier_fraction > 0.8
    assert patch.hull_margin >= cfg.border_hull_margin
    assert "BORDER" not in patch.quality_flags


def test_hull_margin_unit_square():
    hull = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    assert hull_interior_margin(hull) == pytest.approx(1.0 / np.sqrt(2.0), rel=1e-6)


def _crease_scene() -> tuple[GaussianScene, int]:
    xs, ys = np.meshgrid(np.linspace(-4.0, 4.0, 17), np.linspace(0.4, 4.0, 12))
    floor = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    xs, zs = np.meshgrid(np.linspace(-4.0, 4.0, 17), np.linspace(0.4, 4.0, 12))
    wall = np.stack([xs.ravel(), np.zeros(xs.size), zs.ravel()], axis=1)
    mu = np.vstack([floor, wall])
    n = mu.shape[0]
    scales = np.full((n, 3), 0.08)
    scales[: floor.shape[0], 2] = 0.012
    scales[floor.shape[0] :, 1] = 0.012
    normals = np.vstack(
        [np.tile([0.0, 0.0, 1.0], (floor.shape[0], 1)), np.tile([0.0, 1.0, 0.0], (wall.shape[0], 1))]
    )
    scene = GaussianScene(
        mu=mu,
        scales=scales,
        rotations=_identity_quat(n),
        opacity=np.ones(n),
        normals=normals,
    )
    crease = int(np.argmin(np.linalg.norm(scene.mu - np.array([0.0, 0.4, 0.0]), axis=1)))
    return scene, crease


def test_crease_emits_floor_and_wall_candidates():
    scene, crease = _crease_scene()
    tree = cKDTree(scene.mu)
    cfg = PRFConfig(
        patch_knn=48,
        min_patch_gaussians=8,
        use_gaussian_normals=True,
        cluster_normals=True,
        split_all_orientation_clusters=True,
        anchor_voxel_m=0.5,
        max_anchors=200,
    )
    candidates = build_candidate_patches_at_anchor(scene, tree, crease, cfg)
    assert len(candidates) >= 2
    has_floor = any(abs(patch.normal[2]) > 0.8 for patch in candidates)
    has_wall = any(abs(patch.normal[1]) > 0.8 for patch in candidates)
    assert has_floor and has_wall
    for patch in candidates:
        if abs(patch.normal[2]) > 0.8:
            assert float(np.quantile(np.abs(scene.mu[patch.ids, 2]), 0.9)) < 0.35
        if abs(patch.normal[1]) > 0.8:
            assert float(np.quantile(np.abs(scene.mu[patch.ids, 1]), 0.9)) < 0.35
        assert patch.geometry_valid


def test_two_pass_pca_rejects_normal_outliers():
    scene = _plane_lattice(spacing=0.5, sigma=0.10, n=21)
    scene.mu = scene.mu.copy()
    scene.mu[:6] += np.array([0.0, 0.0, 2.0])
    tree = cKDTree(scene.mu)
    cfg = PRFConfig(patch_knn=48, min_patch_gaussians=8, use_gaussian_normals=True, cluster_normals=True)
    center_id = int(np.argmin(np.linalg.norm(scene.mu, axis=1)))
    patch = build_surface_patch(scene, tree, center_id, 0, cfg)
    assert patch.valid
    assert patch.inlier_fraction >= 0.8
    assert patch.plane_residual_m < 0.05
    assert patch.geometry_valid
    assert "NONPLANAR" not in patch.quality_flags
    assert float(np.max(np.abs(scene.mu[patch.ids, 2]))) < 0.5


def test_geometry_valid_false_for_nonplanar_cluster():
    xs, ys, zs = np.meshgrid(np.linspace(-2.0, 2.0, 6), np.linspace(-2.0, 2.0, 6), np.linspace(-2.0, 2.0, 6))
    mu = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    n = mu.shape[0]
    scene = GaussianScene(
        mu=mu,
        scales=np.full((n, 3), 2.0),
        rotations=_identity_quat(n),
        opacity=np.ones(n),
        normals=np.repeat(np.array([[0.0, 0.0, 1.0]]), n, axis=0),
    )
    tree = cKDTree(scene.mu)
    cfg = PRFConfig(patch_knn=48, min_patch_gaussians=8, use_gaussian_normals=False, cluster_normals=False)
    center_id = int(np.argmin(np.linalg.norm(scene.mu, axis=1)))
    patch = build_surface_patch(scene, tree, center_id, 0, cfg)
    assert patch.valid
    assert "NONPLANAR" in patch.quality_flags
    assert not patch.geometry_valid


def test_valid_patches_are_deduplicated():
    scene = _plane_lattice(spacing=0.5, sigma=0.10, n=21)
    cfg = PRFConfig(
        patch_knn=48,
        min_patch_gaussians=8,
        use_gaussian_normals=True,
        cluster_normals=True,
        max_anchors=40,
        anchor_voxel_m=0.5,
        metric_extent_min_m=1.0,
        metric_scale_min_m=1e-3,
    )
    patches = build_valid_patches(scene, cfg)
    assert patches
    centers = np.stack([patch.center for patch in patches])
    for i in range(len(patches)):
        for j in range(i + 1, len(patches)):
            dist = float(np.linalg.norm(centers[i] - centers[j]))
            iou_den = len(set(patches[i].ids.tolist()) | set(patches[j].ids.tolist()))
            iou = len(set(patches[i].ids.tolist()) & set(patches[j].ids.tolist())) / max(iou_den, 1)
            same_normal = abs(float(np.dot(patches[i].normal, patches[j].normal))) >= np.cos(np.deg2rad(25.0))
            if same_normal and dist <= 0.75 * min(patches[i].radius, patches[j].radius):
                assert iou < 0.40
