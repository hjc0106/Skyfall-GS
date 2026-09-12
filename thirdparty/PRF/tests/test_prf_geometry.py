from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from prf.color import obs_enhanced_codes, resolution_bin, resolution_codes
from prf.config import PRFConfig
from prf.export import NPZ_REQUIRED, export_field, validate_npz_schema_v2
from prf.fusion import fuse_resolutions
from prf.gaussian_field import map_patch_fields_to_points
from prf.patches import build_surface_patch, convex_hull_uv, pack_csr
from prf.pipeline import build_physical_resolution_field
from prf.types import SurfacePatch
from tests.test_prf import _nadir_view, _plane_lattice


def test_convex_hull_area_unit_square():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    area, hull = convex_hull_uv(coords)
    assert area == pytest.approx(1.0)
    assert hull.shape[0] == 4


def test_patch_radius_area_and_csr():
    scene = _plane_lattice(spacing=0.5, sigma=0.10, n=21)
    cfg = PRFConfig(patch_knn=48, min_patch_gaussians=8, use_gaussian_normals=False)
    tree = cKDTree(scene.mu)
    center_id = int(np.argmin(np.linalg.norm(scene.mu, axis=1)))
    patch = build_surface_patch(scene, tree, center_id, 7, cfg)
    assert patch.valid
    assert patch.radius > 0.0
    area, _ = convex_hull_uv((scene.mu[patch.ids] - patch.center) @ patch.tangent)
    assert patch.area_m2 == pytest.approx(area)
    offsets, ids = pack_csr([patch.ids, patch.ids[:3]])
    assert offsets.tolist() == [0, int(patch.ids.size), int(patch.ids.size) + 3]
    assert ids.shape[0] == int(offsets[-1])


def test_npz_schema_v2(tmp_path: Path):
    scene = _plane_lattice(spacing=0.5, sigma=0.10, n=16)
    cfg = PRFConfig(
        max_anchors=8,
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
    field = build_physical_resolution_field(scene, [view_a, view_b], cfg)
    export_field(field, tmp_path)
    data = np.load(tmp_path / "physical_resolution_field.npz", allow_pickle=True)
    missing = sorted(set(NPZ_REQUIRED).difference(data.files))
    assert missing == []
    assert int(np.asarray(data["schema_version"]).reshape(-1)[0]) == 2
    assert data["member_offsets"].shape[0] == len(field.records) + 1
    assert data["hull_offsets"].shape[0] == len(field.records) + 1
    assert data["hull_uv"].ndim == 2 and data["hull_uv"].shape[1] == 2
    assert data["best_camera_a"].dtype.kind in ("U", "S")
    assert validate_npz_schema_v2({key: data[key] for key in data.files}, scene_size=scene.num_gaussians) == len(field.records)
    assert "geometry_valid" in data.files
    assert "plane_residual_m" in data.files
    assert "support_count" in data.files
    assert int(np.asarray(data["geometry_valid"]).sum()) >= 1


def test_npz_schema_v2_rejects_broken_csr(tmp_path: Path):
    scene = _plane_lattice(spacing=0.5, sigma=0.10, n=16)
    cfg = PRFConfig(
        max_anchors=4,
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
    export_field(build_physical_resolution_field(scene, [view_a, view_b], cfg), tmp_path)
    with np.load(tmp_path / "physical_resolution_field.npz", allow_pickle=False) as data:
        archive = {key: data[key] for key in data.files}
    archive["hull_offsets"] = archive["hull_offsets"].copy()
    archive["hull_offsets"][-1] -= 1
    with pytest.raises(ValueError, match="hull_offsets"):
        validate_npz_schema_v2(archive)


def test_viewer_assets_are_self_contained():
    root = Path(__file__).parents[1] / "prf" / "assets"
    html = (root / "physical_resolution_3d.html").read_text()
    javascript = (root / "prf-viewer.js").read_text()
    assert '"three": "./vendor/three.module.js"' in html
    assert "cdn.jsdelivr.net" not in html
    assert (root / "vendor" / "three.module.js").is_file()
    assert (root / "vendor" / "OrbitControls.js").is_file()
    assert "buildPatchSurface" in javascript
    assert "GSF1" in javascript
    assert "GSF2" in javascript


def test_training_image_projection_error():
    view = _nadir_view(height=100.0, focal=500.0, size=512)
    point = np.array([2.0, -1.0, 0.0])
    uv, z = view.project(point)
    assert float(z) == pytest.approx(100.0)
    assert abs(float(uv[0]) - (256 + 10)) < 1.0
    assert abs(float(uv[1]) - (256 + 5)) < 1.0


def test_absolute_color_bins():
    assert resolution_bin(0.3) == 0
    assert resolution_bin(0.7) == 1
    assert resolution_bin(1.5) == 2
    assert resolution_bin(3.0) == 3
    assert resolution_bin(7.0) == 4
    assert resolution_bin(12.0) == 5
    assert resolution_bin(math.inf) == -1
    assert resolution_codes(np.array([0.5, 1.1, math.inf])).tolist() == [0, 2, 255]
    assert obs_enhanced_codes(np.array([0.35, 0.6, 1.2, 2.1, math.inf])).tolist() == [0, 2, 4, 6, 255]


def test_chunked_multi_field_mapping():
    points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (2, 1))
    centers = points.copy()
    t1 = np.tile(np.array([[1.0, 0.0, 0.0]]), (2, 1))
    t2 = np.tile(np.array([[0.0, 1.0, 0.0]]), (2, 1))
    values = np.array([[0.5, 2.0], [4.0, math.inf]])
    mapped, nearest = map_patch_fields_to_points(
        points,
        normals,
        centers,
        normals,
        t1,
        t2,
        np.array([2.0, 2.0]),
        values,
        k=1,
        chunk_size=1,
    )
    assert nearest.tolist() == [0, 1]
    assert mapped[0].tolist() == pytest.approx([0.5, 2.0])
    assert mapped[1, 0] == pytest.approx(4.0)
    assert np.isnan(mapped[1, 1])


def test_inf_observation_is_invalid():
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
    rec = fuse_resolutions(patch, math.inf, 0.18, 0.14, {"quality_flags": ["NO_VALID_MULTIVIEW_PAIR"]})
    assert rec.bottleneck_type == "INVALID"
    assert not math.isfinite(rec.R_phys_m_per_equiv_pixel)


def test_visualize_load_field_schema_v2(tmp_path: Path):
    from prf.visualize import load_field

    scene = _plane_lattice(spacing=0.5, sigma=0.10, n=12)
    cfg = PRFConfig(
        max_anchors=4,
        patch_knn=16,
        use_gaussian_normals=False,
        min_pair_angle_deg=1.0,
        metric_extent_min_m=1.0,
        metric_scale_min_m=1e-3,
    )
    view = _nadir_view(100.0, 500.0)
    view_b = _nadir_view(100.0, 500.0)
    view_b.view_id = "b"
    view_b.center = np.array([30.0, 0.0, 100.0])
    view_b.c2w = view.c2w.copy()
    view_b.c2w[:3, 3] = view_b.center
    view_b.w2c = np.linalg.inv(view_b.c2w)
    export_field(build_physical_resolution_field(scene, [view, view_b], cfg), tmp_path)
    field = load_field(tmp_path)
    for key in ("center", "R_obs", "R_kernel", "R_spacing", "R_phys", "bottleneck"):
        assert key in field
    assert field["center"].ndim == 2
