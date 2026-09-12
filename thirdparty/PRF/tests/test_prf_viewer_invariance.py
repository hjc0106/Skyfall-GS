from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prf.config import PRFConfig
from prf.export import export_field
from prf.manifest import sha256_file
from prf.pipeline import build_physical_resolution_field
from prf.visualize_3d import build_vis3d
from tests.test_prf import _nadir_view, _plane_lattice

METRIC_KEYS = ("R_obs", "R_kernel", "R_spacing", "R_phys")


def _write_gs_ply(path: Path, scene) -> None:
    n = scene.num_gaussians
    props = [
        "x", "y", "z", "nx", "ny", "nz", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {name}\n" for name in props)
        + "end_header\n"
    )
    opacity = np.clip(scene.opacity, 1e-6, 1.0 - 1e-6)
    logit = np.log(opacity / (1.0 - opacity))
    scale_log = np.log(np.maximum(scene.scales, 1e-8))
    payload = np.column_stack(
        [
            scene.mu,
            scene.normals,
            logit[:, None],
            scale_log,
            scene.rotations,
        ]
    ).astype("<f4")
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        payload.tofile(handle)


def _write_transforms(path: Path, views) -> None:
    frames = []
    for view in views:
        frames.append(
            {
                "file_path": f"./images/{view.view_id}.png",
                "fl_x": view.fx,
                "fl_y": view.fy,
                "cx": view.cx,
                "cy": view.cy,
                "w": view.width,
                "h": view.height,
                "transform_matrix": view.c2w.tolist(),
            }
        )
    path.write_text(
        json.dumps(
            {
                "camera_model": "PINHOLE",
                "w": views[0].width,
                "h": views[0].height,
                "frames": frames,
            }
        )
        + "\n"
    )


def _two_views():
    view = _nadir_view(100.0, 500.0, size=64)
    view_b = _nadir_view(100.0, 500.0, size=64)
    view_b.view_id = "b"
    view_b.center = np.array([30.0, 0.0, 100.0])
    view_b.c2w = view.c2w.copy()
    view_b.c2w[:3, 3] = view_b.center
    view_b.w2c = np.linalg.inv(view_b.c2w)
    return [view, view_b]


def _metric_digest(path: Path) -> dict[str, bytes]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]).tobytes() for key in METRIC_KEYS}


def test_viewer_display_params_do_not_change_npz_r(tmp_path: Path):
    scene = _plane_lattice(spacing=0.5, sigma=0.10, n=12)
    ply_path = tmp_path / "scene.ply"
    transforms = tmp_path / "transforms.json"
    views = _two_views()
    _write_gs_ply(ply_path, scene)
    _write_transforms(transforms, views)
    cfg = PRFConfig(
        max_anchors=6,
        patch_knn=16,
        use_gaussian_normals=False,
        min_pair_angle_deg=1.0,
        metric_extent_min_m=1.0,
        metric_scale_min_m=1e-3,
    )
    field_dir = tmp_path / "field"
    export_field(
        build_physical_resolution_field(
            scene, views, cfg, source_ply=str(ply_path), source_transforms=str(transforms)
        ),
        field_dir,
    )
    npz = field_dir / "physical_resolution_field.npz"
    before_hash = sha256_file(npz)
    before_metrics = _metric_digest(npz)

    for image_size, context_points in ((32, 8), (128, 0), (96, 16)):
        build_vis3d(
            field_dir,
            ply_path,
            transforms,
            tmp_path / f"vis_{image_size}_{context_points}",
            opengl_c2w=False,
            context_points=context_points,
            image_size=image_size,
            image_quality=40,
            max_images=0,
        )
        assert sha256_file(npz) == before_hash
        assert _metric_digest(npz) == before_metrics

    html = Path(__file__).parents[1] / "prf" / "assets" / "physical_resolution_3d.html"
    js = Path(__file__).parents[1] / "prf" / "assets" / "prf-viewer.js"
    client = html.read_text() + "\n" + js.read_text()
    assert "physical_resolution_field.npz" not in client
    assert "R_obs" in js.read_text()
