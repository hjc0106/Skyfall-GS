"""JAX_068 nine-center × six-azimuth orbit: Episode 1 pose + focal zoom.

Layout borrows the official JAX IDU 3×3 lookat grid and 6 azimuths. Resolution,
single-sample DLoRAL, and LoD training follow this experiment, not IDU.

First delivery is coverage, neighbor selection, and the origin 1-vs-6 control.
Do not write locked scene / Episode 1 / 80K archives. Do not change Qwen,
DLoRAL, or parent-training policy. 8× is a later independent stage.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from lod.episode1 import (
    BASE_FOV_DEG,
    CENTER_ROI,
    ELEVATION_DEG,
    IMAGE_SIZE,
    MODEL_STEP_SCALE,
    RADIUS,
    dump_skyfall_camera,
    expected_fovs_deg,
    horizontal_fov_deg,
    orbit_c2w,
    orbit_eye,
    pose_rt,
    project_lookat,
    stage_scale_report,
    structure_metrics,
    zoom_intrinsics,
)
from lod.jax068 import DATASET_DIR, SCENE, SCENE_80K_DIR, SCENE_DIR, STAGE1_CHECKPOINT, TEST_VIEWS


SCENE_NAME = SCENE
STAGE1_PATH = STAGE1_CHECKPOINT
DATASET_PATH = DATASET_DIR
OUTPUT_ROOT = Path("skyfall-gs_exp/lod_jax068_orbit_9c6a")
REPO_ROOT = Path(__file__).resolve().parents[1]

LOCKED_ARCHIVE_ROOTS = (
    SCENE_DIR,
    SCENE_80K_DIR,
    Path("skyfall-gs_exp/lod_jax068_episode1"),
    Path("skyfall-gs_exp/lod_jax068_episode1_parent_sh"),
    Path("skyfall-gs_exp/lod_jax068_episode1_parent_full"),
)

CENTER_XY = (-128.0, 0.0, 128.0)
LOOKAT_Z = 0.0
AZIMUTHS_DEG = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
DIAGNOSTIC_AZIMUTHS_DEG = (30.0, 90.0, 150.0, 210.0, 270.0, 330.0)
PREFERRED_NEIGHBOR_DELTAS_DEG = (-10.0, 10.0)
# Geometry-validity rule already used by prepare_lod_l2_supervision / Episode 1.
MIN_REPROJECTION_COVERAGE = 0.05
ALPHA_THRESHOLD = 0.05
LOOKAT_CENTER_PX = 4.0
MIN_ALPHA_FRAC = 0.50
MIN_LAPLACIAN = 1e-4
NEAR_SURFACE_Z = 50.0
ZOOM_FACTORS_ROUND1 = (1.0, 4.0)
SAMPLES_PER_POSE = 1
N_LOOKATS = 9
N_AZIMUTHS = 6
N_BASE_POSES = N_LOOKATS * N_AZIMUTHS  # 54
ORIGIN_LOOKAT = (0.0, 0.0, 0.0)

UID_BASE_1X = 20_000
UID_BASE_4X = 30_000
UID_BASE_NEIGHBOR = 40_000
UID_DIAG = 50_000

SEED = 0
CENTER_STEPS = 500
CENTER_MAX_POINTS = 50_000
MIX_RATIO = 0.2
PROBE_STEPS = 30
PROBE_RESUME_AT = 20
EVAL_STEPS = "0,50,100,250,500"
DENSIFY_FROM = 1
DENSIFY_UNTIL = 250
DENSIFY_INTERVAL = 10
DENSIFY_GRAD_THRESHOLD = 2e-4

# Full nine-center × six-azimuth 4× L1 round. These values are deliberately
# separate from the center 1-vs-6 control budget; the full run must not inherit
# the local 500-step / 50k-point cap.
FULL_L1_STEPS = 4_000
FULL_L1_MAX_POINTS = 450_000
FULL_L1_VISITS_EXPECTED = 59
FULL_L1_CKPT_STEPS = (0, 50, 500, 1_000, 2_000, 4_000)
FULL_L1_DENSIFY_UNTIL = DENSIFY_UNTIL * FULL_L1_STEPS // CENTER_STEPS
FULL_PROBE_STEPS = 60
FULL_PROBE_RESUME_AT = 30
FULL_CORRESPONDENCE_STEPS = (0, 500, 1_000, 2_000, 4_000)

L1_POSITION_LR = 1.6e-4
L1_FEATURE_LR = 2.5e-3
L1_OPACITY_LR = 5e-2
L1_SCALING_LR = 5e-3
L1_ROTATION_LR = 1e-3

NOTE = (
    "Nine lookats do not imply 4× covers the scene. Coverage is measured from "
    "frozen L0 renders. DLoRAL neighbors are same-center ±10°, never the next "
    "60° training camera. Low-altitude poses have no co-located GT; pseudo-label "
    "scores are supervision absorption only. Test views 003/012 are not used to "
    "pick this design."
)


def lookats() -> tuple[tuple[float, float, float], ...]:
    """Official JAX IDU 3×3 order: meshgrid(x, y) with border stripped."""

    return tuple((float(x), float(y), LOOKAT_Z) for y in CENTER_XY for x in CENTER_XY)


def lookat_index(lookat: Sequence[float]) -> int:
    target = (float(lookat[0]), float(lookat[1]), float(lookat[2]) if len(lookat) > 2 else LOOKAT_Z)
    for index, item in enumerate(lookats()):
        if all(abs(a - b) < 1e-6 for a, b in zip(item, target)):
            return index
    raise ValueError(f"lookat {lookat} is not one of the locked 9 centers")


def azimuth_index(azimuth_deg: float) -> int:
    for index, value in enumerate(AZIMUTHS_DEG):
        if abs(float(azimuth_deg) - value) < 1e-6:
            return index
    raise ValueError(f"azimuth {azimuth_deg} is not a training azimuth")


def _num_tag(value: float) -> str:
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return text


def pose_id(lookat: Sequence[float], azimuth_deg: float) -> str:
    cx, cy = float(lookat[0]), float(lookat[1])
    return f"c{_num_tag(cx)}_{_num_tag(cy)}_az{_num_tag(azimuth_deg)}"


def pose_name(lookat: Sequence[float], azimuth_deg: float) -> str:
    return (
        f"orbit9c6a_{pose_id(lookat, azimuth_deg)}"
        f"_e{ELEVATION_DEG:g}_r{RADIUS:g}"
    )


def pose_index(lookat: Sequence[float], azimuth_deg: float) -> int:
    return lookat_index(lookat) * N_AZIMUTHS + azimuth_index(azimuth_deg)


def all_base_poses() -> tuple[dict[str, Any], ...]:
    rows = []
    for lookat in lookats():
        for azimuth in AZIMUTHS_DEG:
            index = pose_index(lookat, azimuth)
            rows.append(
                {
                    "index": index,
                    "pose_id": pose_id(lookat, azimuth),
                    "name": pose_name(lookat, azimuth),
                    "lookat": list(lookat),
                    "lookat_index": lookat_index(lookat),
                    "azimuth_deg": float(azimuth),
                    "azimuth_index": azimuth_index(azimuth),
                    "uid_1x": UID_BASE_1X + index,
                    "uid_4x": UID_BASE_4X + index,
                    "is_origin": all(abs(float(v)) < 1e-6 for v in lookat),
                }
            )
    return tuple(rows)


def origin_poses(*, azimuths: Sequence[float] = AZIMUTHS_DEG) -> tuple[dict[str, Any], ...]:
    wanted = {float(v) for v in azimuths}
    return tuple(row for row in all_base_poses() if row["is_origin"] and float(row["azimuth_deg"]) in wanted)


def neighbor_azimuth(azimuth_deg: float, delta_deg: float) -> float:
    return (float(azimuth_deg) + float(delta_deg)) % 360.0


def neighbor_candidates(azimuth_deg: float) -> tuple[dict[str, Any], ...]:
    rows = []
    for delta in PREFERRED_NEIGHBOR_DELTAS_DEG:
        az = neighbor_azimuth(azimuth_deg, delta)
        if az in DIAGNOSTIC_AZIMUTHS_DEG:
            raise ValueError(f"neighbor azimuth {az} collides with a held-out diagnostic view")
        if az in AZIMUTHS_DEG:
            raise ValueError(f"neighbor azimuth {az} collides with a training camera")
        rows.append(
            {
                "delta_deg": float(delta),
                "azimuth_deg": float(az),
                "kind": "same_center_pm10",
            }
        )
    return tuple(rows)


def is_training_azimuth(azimuth_deg: float) -> bool:
    return any(abs(float(azimuth_deg) - value) < 1e-6 for value in AZIMUTHS_DEG)


def is_diagnostic_azimuth(azimuth_deg: float) -> bool:
    return any(abs(float(azimuth_deg) - value) < 1e-6 for value in DIAGNOSTIC_AZIMUTHS_DEG)


def expected_visits(n_targets: int, *, steps: int = CENTER_STEPS, mix_ratio: float = MIX_RATIO) -> dict[str, Any]:
    n = max(int(n_targets), 1)
    enhance = int(steps) * (1.0 - float(mix_ratio))
    per = enhance / n
    return {
        "sampling": "random_with_replacement",
        "steps": int(steps),
        "mix_ratio": float(mix_ratio),
        "n_targets": n,
        "expected_enhance_steps": enhance,
        "expected_visits_per_target": per,
        "expected_is_not_guaranteed": True,
        "note": (
            "Six-view has the same 500-step budget as single-view, so each "
            "supervision image is visited fewer times. Counts are expectations "
            "under uniform draws, not a round-robin guarantee."
        ),
    }


def lock_neighbor(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pick a same-center ±10° neighbor by bidirectional coverage. Never 60°."""

    if not records:
        return {
            "selected": None,
            "fallback": "target_only",
            "reason": "no_neighbor_candidates",
            "min_reprojection_coverage": MIN_REPROJECTION_COVERAGE,
        }
    forbidden = []
    scored = []
    for rec in records:
        azimuth = float(rec["azimuth_deg"])
        delta = float(rec.get("delta_deg", math.nan))
        if is_training_azimuth(azimuth) or is_diagnostic_azimuth(azimuth):
            forbidden.append(
                {
                    "azimuth_deg": azimuth,
                    "reason": "training_or_diagnostic_azimuth",
                }
            )
            continue
        if not any(abs(delta - pref) < 1e-6 for pref in PREFERRED_NEIGHBOR_DELTAS_DEG):
            forbidden.append({"azimuth_deg": azimuth, "reason": "not_preferred_pm10"})
            continue
        coverage = float(rec.get("coverage") or 0.0)
        reverse = float(rec.get("reverse_coverage") or 0.0)
        item = dict(rec)
        item["score"] = min(coverage, reverse)
        scored.append(item)
    scored.sort(key=lambda item: item["score"], reverse=True)
    if not scored:
        return {
            "selected": None,
            "fallback": "target_only",
            "reason": "all_candidates_rejected",
            "forbidden": forbidden,
            "min_reprojection_coverage": MIN_REPROJECTION_COVERAGE,
        }
    best = scored[0]
    if float(best["score"]) < MIN_REPROJECTION_COVERAGE:
        return {
            "selected": best,
            "fallback": "target_only",
            "reason": "low_reprojection_coverage",
            "candidates": scored,
            "forbidden": forbidden,
            "min_reprojection_coverage": MIN_REPROJECTION_COVERAGE,
        }
    return {
        "selected": best,
        "fallback": None,
        "reason": "same_center_pm10",
        "candidates": scored,
        "forbidden": forbidden,
        "min_reprojection_coverage": MIN_REPROJECTION_COVERAGE,
    }


def camera_payload(
    lookat: Sequence[float],
    azimuth_deg: float,
    zoom_factor: float,
    *,
    uid: int,
) -> dict[str, Any]:
    target = tuple(float(v) for v in lookat)
    c2w = orbit_c2w(azimuth_deg, lookat=target)
    w2c = np.linalg.inv(c2w)
    R, T = pose_rt(azimuth_deg, lookat=target)
    intra = zoom_intrinsics(zoom_factor)
    cx_pixel = 0.5 * IMAGE_SIZE * (intra["cx_ndc"] + 1.0)
    cy_pixel = 0.5 * IMAGE_SIZE * (intra["cy_ndc"] + 1.0)
    lookat_proj = project_lookat(
        w2c, target, fx=intra["fx"], fy=intra["fy"], cx_pixel=cx_pixel, cy_pixel=cy_pixel,
    )
    return {
        "pose_id": pose_id(target, azimuth_deg),
        "name": pose_name(target, azimuth_deg),
        "uid": int(uid),
        "index": pose_index(target, azimuth_deg) if is_training_azimuth(azimuth_deg) else None,
        "azimuth_deg": float(azimuth_deg),
        "elevation_deg": float(ELEVATION_DEG),
        "radius": float(RADIUS),
        "lookat": list(target),
        "zoom_factor": float(zoom_factor),
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "fov_x_deg": intra["fov_deg"],
        "fov_y_deg": intra["fov_deg"],
        "fx": intra["fx"],
        "fy": intra["fy"],
        "cx_ndc": intra["cx_ndc"],
        "cy_ndc": intra["cy_ndc"],
        "cx_pixel": cx_pixel,
        "cy_pixel": cy_pixel,
        "R": R.tolist(),
        "T": T.tolist(),
        "w2c": w2c.tolist(),
        "c2w": c2w.tolist(),
        "camera_center": c2w[:3, 3].tolist(),
        "lookat_projection": lookat_proj,
        "roi": {
            "center_x": CENTER_ROI.center_x,
            "center_y": CENTER_ROI.center_y,
            "width": CENTER_ROI.width,
            "height": CENTER_ROI.height,
        },
        "extrinsics_unchanged_under_zoom": True,
    }


def dump_orbit_camera(camera, *, zoom_factor: float, azimuth_deg: float, lookat: Sequence[float]) -> dict[str, Any]:
    payload = dump_skyfall_camera(camera, zoom_factor=zoom_factor, azimuth_deg=azimuth_deg)
    w2c = np.asarray(payload["w2c"], dtype=np.float64)
    cx_pixel = float(payload["cx_pixel"])
    cy_pixel = float(payload["cy_pixel"])
    target = tuple(float(v) for v in lookat)
    payload["lookat"] = list(target)
    payload["pose_id"] = pose_id(target, azimuth_deg)
    payload["lookat_projection"] = project_lookat(
        w2c, target, fx=float(payload["fx"]), fy=float(payload["fy"]),
        cx_pixel=cx_pixel, cy_pixel=cy_pixel,
    )
    return payload


def make_orbit_camera(
    lookat: Sequence[float],
    azimuth_deg: float,
    *,
    uid: int,
    data_device: str = "cuda",
    elevation_deg: float = ELEVATION_DEG,
    radius: float = RADIUS,
    image_name: str | None = None,
):
    """1× Skyfall camera with optional orbit elevation.

    The locked training cameras keep the default 85° elevation. Video and
    visualization callers may lower it for an oblique view without changing
    the training-camera convention.
    """

    import torch
    from PIL import Image

    from scene.cameras import Camera
    from utils.general_utils import PILtoTorch

    target = tuple(float(v) for v in lookat)
    R, T = pose_rt(
        azimuth_deg,
        elevation_deg=float(elevation_deg),
        radius=float(radius),
        lookat=target,
    )
    fov = math.radians(BASE_FOV_DEG)
    image = PILtoTorch(Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0)), (IMAGE_SIZE, IMAGE_SIZE))
    if image.shape[0] == 4:
        image = image[:3]
    return Camera(
        colmap_id=int(uid),
        R=R,
        T=T,
        FoVx=fov,
        FoVy=fov,
        cx=0.0,
        cy=0.0,
        image=image,
        gt_alpha_mask=None,
        image_name=(
            str(image_name)
            if image_name is not None
            else pose_name(target, azimuth_deg)
        ),
        uid=int(uid),
        depth=None,
        mask=torch.ones((1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32),
        data_device=data_device,
        optimizing=False,
    )


def zoomed_orbit_camera(base_camera, zoom_factor: float, *, uid: int | None = None):
    from utils.zoom_camera import make_zoom_camera

    factor = float(zoom_factor)
    return make_zoom_camera(
        base_camera,
        CENTER_ROI,
        factor,
        uid=int(uid if uid is not None else base_camera.uid + int(round(1000 * factor))),
        image_name=f"{base_camera.image_name}_zoom{factor:g}",
    )


def unproject_pixel_to_z0(
    c2w: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx_pixel: float,
    cy_pixel: float,
    u: float,
    v: float,
    z_plane: float = 0.0,
) -> np.ndarray | None:
    x = (float(u) - float(cx_pixel)) / float(fx)
    y = (float(v) - float(cy_pixel)) / float(fy)
    direction_cam = np.array([x, y, 1.0], dtype=np.float64)
    rotation = np.asarray(c2w, dtype=np.float64)[:3, :3]
    origin = np.asarray(c2w, dtype=np.float64)[:3, 3]
    direction = rotation @ direction_cam
    if abs(float(direction[2])) < 1e-8:
        return None
    scale = (float(z_plane) - float(origin[2])) / float(direction[2])
    if scale <= 0.0:
        return None
    return origin + scale * direction


def ground_quad(
    lookat: Sequence[float],
    azimuth_deg: float,
    zoom_factor: float,
) -> np.ndarray:
    """Image-corner rays intersect z=0. Near-nadir 4× footprints are almost squares."""

    payload = camera_payload(lookat, azimuth_deg, zoom_factor, uid=0)
    c2w = np.asarray(payload["c2w"], dtype=np.float64)
    corners = ((0.0, 0.0), (IMAGE_SIZE - 1.0, 0.0), (IMAGE_SIZE - 1.0, IMAGE_SIZE - 1.0), (0.0, IMAGE_SIZE - 1.0))
    points = []
    for u, v in corners:
        point = unproject_pixel_to_z0(
            c2w,
            fx=payload["fx"],
            fy=payload["fy"],
            cx_pixel=payload["cx_pixel"],
            cy_pixel=payload["cy_pixel"],
            u=u,
            v=v,
        )
        if point is None:
            raise ValueError(f"corner ({u},{v}) does not hit z=0 for {pose_id(lookat, azimuth_deg)}")
        points.append(point[:2])
    return np.asarray(points, dtype=np.float64)


def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def aabb_iou(left: np.ndarray, right: np.ndarray) -> float:
    def box(poly):
        return float(poly[:, 0].min()), float(poly[:, 1].min()), float(poly[:, 0].max()), float(poly[:, 1].max())

    ax0, ay0, ax1, ay1 = box(left)
    bx0, by0, bx1, by1 = box(right)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def occupancy_union(
    polygons: Sequence[np.ndarray],
    *,
    bounds: tuple[float, float, float, float] = (-256.0, 256.0, -256.0, 256.0),
    resolution: float = 4.0,
) -> dict[str, Any]:
    x0, x1, y0, y1 = bounds
    xs = np.arange(x0, x1, resolution)
    ys = np.arange(y0, y1, resolution)
    grid = np.zeros((len(ys), len(xs)), dtype=np.uint8)
    cell = float(resolution)
    for poly in polygons:
        minx, miny = poly[:, 0].min(), poly[:, 1].min()
        maxx, maxy = poly[:, 0].max(), poly[:, 1].max()
        i0 = max(int((miny - y0) / cell), 0)
        i1 = min(int(math.ceil((maxy - y0) / cell)), len(ys))
        j0 = max(int((minx - x0) / cell), 0)
        j1 = min(int(math.ceil((maxx - x0) / cell)), len(xs))
        path = _Poly(poly)
        for i in range(i0, i1):
            cy = y0 + (i + 0.5) * cell
            for j in range(j0, j1):
                cx = x0 + (j + 0.5) * cell
                if path.contains(cx, cy):
                    grid[i, j] = 1
    covered = int(grid.sum())
    total = int(grid.size)
    return {
        "bounds": list(bounds),
        "resolution": float(resolution),
        "covered_cells": covered,
        "total_cells": total,
        "coverage_frac": 0.0 if total == 0 else covered / total,
        "grid_shape": [int(grid.shape[0]), int(grid.shape[1])],
        "grid": grid,
    }


class _Poly:
    def __init__(self, vertices: np.ndarray):
        self.x = vertices[:, 0]
        self.y = vertices[:, 1]

    def contains(self, x: float, y: float) -> bool:
        inside = False
        n = int(self.x.shape[0])
        j = n - 1
        for i in range(n):
            xi, yi = float(self.x[i]), float(self.y[i])
            xj, yj = float(self.x[j]), float(self.y[j])
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi):
                inside = not inside
            j = i
        return inside


def classify_pose(structure: Mapping[str, Any], lookat_projection: Mapping[str, Any]) -> dict[str, Any]:
    proj = lookat_projection
    in_frame = bool(proj.get("in_frame"))
    u = float(proj.get("u") or math.nan)
    v = float(proj.get("v") or math.nan)
    off = (not in_frame) or abs(u - IMAGE_SIZE / 2.0) > LOOKAT_CENTER_PX or abs(v - IMAGE_SIZE / 2.0) > LOOKAT_CENTER_PX
    alpha_frac = float(structure.get("alpha_frac_gt_0.05") or 0.0)
    lap = float(structure.get("laplacian_energy_valid") or 0.0)
    center_z = float(structure.get("center_camera_z") or math.nan)
    flags = {
        "lookat_off_center": bool(off),
        "lookat_in_frame": bool(in_frame),
        "low_alpha": bool(alpha_frac < MIN_ALPHA_FRAC),
        "low_structure": bool(lap < MIN_LAPLACIAN),
        "near_surface": bool(center_z == center_z and center_z < NEAR_SURFACE_Z),
    }
    fail = flags["lookat_off_center"] or flags["near_surface"] or not flags["lookat_in_frame"]
    warn = flags["low_alpha"] or flags["low_structure"]
    return {
        "flags": flags,
        "severity": "fail" if fail else ("warn" if warn else "ok"),
        "invalid_for_supervision": bool(fail or (flags["low_alpha"] and flags["low_structure"])),
        "alpha_frac": alpha_frac,
        "laplacian_energy_valid": lap,
        "center_camera_z": center_z,
        "lookat_uv": [u, v],
    }


def locked_archive_hit(path: str | Path) -> Path | None:
    dest = Path(path).resolve()
    for root in LOCKED_ARCHIVE_ROOTS:
        locked = (REPO_ROOT / root).resolve()
        if dest == locked or locked in dest.parents:
            return root
    return None


def assert_writable_output(path: str | Path, *, protect: bool = True) -> None:
    if not protect:
        return
    hit = locked_archive_hit(path)
    if hit is not None:
        raise ValueError(f"refusing to write {path} under locked archive {hit}. Use {OUTPUT_ROOT}.")


def output_paths(root: str | Path | None = None) -> dict[str, Path]:
    base = Path(root or OUTPUT_ROOT)
    return {
        "root": base,
        "coverage": base / "coverage",
        "neighbors": base / "neighbors",
        "supervise_center": base / "supervise_center",
        "supervise_full": base / "supervise_full",
        "center_probe": base / "center_probe",
        "center_single": base / "center_single",
        "center_six": base / "center_six",
        "center_compare": base / "center_compare",
        "full_probe": base / "full_probe",
        "full_l1": base / "full_l1",
        "full_report": base / "full_report",
        "report": base / "report",
    }


def experiment_lock() -> dict[str, Any]:
    return {
        "scene": SCENE_NAME,
        "archive": str(OUTPUT_ROOT),
        "layout": "official_jax_idu_9_lookats_x_6_azimuths",
        "not_full_idu_replica": True,
        "pose": {
            "elevation_deg": ELEVATION_DEG,
            "radius": RADIUS,
            "base_fov_deg": BASE_FOV_DEG,
            "image_size": [IMAGE_SIZE, IMAGE_SIZE],
            "lookats": [list(item) for item in lookats()],
            "azimuths_deg": list(AZIMUTHS_DEG),
            "diagnostic_azimuths_deg": list(DIAGNOSTIC_AZIMUTHS_DEG),
            "n_base_poses": N_BASE_POSES,
            "samples_per_pose": SAMPLES_PER_POSE,
        },
        "zoom": {
            "round1": list(ZOOM_FACTORS_ROUND1),
            "extrinsics_unchanged": True,
            "center_roi": {
                "center_x": CENTER_ROI.center_x,
                "center_y": CENTER_ROI.center_y,
                "width": CENTER_ROI.width,
                "height": CENTER_ROI.height,
            },
            "expected_fov_deg": {k: v for k, v in expected_fovs_deg().items() if k in ("1x", "4x")},
        },
        "neighbors": {
            "preferred_deltas_deg": list(PREFERRED_NEIGHBOR_DELTAS_DEG),
            "same_center": True,
            "never_use_next_training_camera": True,
            "min_reprojection_coverage": MIN_REPROJECTION_COVERAGE,
            "fallback": "target_only",
            "not_lod_supervision": True,
        },
        "center_control": {
            "lookat": list(ORIGIN_LOOKAT),
            "single_azimuths_deg": [0.0],
            "six_azimuths_deg": list(AZIMUTHS_DEG),
            "shared_supervision": "origin_az0_4x",
            "steps": CENTER_STEPS,
            "max_points": CENTER_MAX_POINTS,
            "seed": SEED,
            "mix_ratio": MIX_RATIO,
            "densify": {
                "from": DENSIFY_FROM,
                "until": DENSIFY_UNTIL,
                "interval": DENSIFY_INTERVAL,
                "grad_threshold": DENSIFY_GRAD_THRESHOLD,
            },
            "learning_rate": {
                "xyz": L1_POSITION_LR,
                "sh": L1_FEATURE_LR,
                "opacity_logits": L1_OPACITY_LR,
                "log_scales": L1_SCALING_LR,
                "rotations": L1_ROTATION_LR,
            },
            "visits": {
                "single": expected_visits(1),
                "six": expected_visits(6),
            },
            "parent_update": "frozen",
            "appearance": "frozen",
        },
        "full_54_l1": {
            "locked": True,
            "status": "budget_locked_before_training",
            "steps": FULL_L1_STEPS,
            "max_points": FULL_L1_MAX_POINTS,
            "mix_ratio": MIX_RATIO,
            "expected_visits_per_target": FULL_L1_VISITS_EXPECTED,
            "checkpoint_steps": list(FULL_L1_CKPT_STEPS),
            "densify": {
                "from": DENSIFY_FROM,
                "until": FULL_L1_DENSIFY_UNTIL,
                "interval": DENSIFY_INTERVAL,
                "grad_threshold": DENSIFY_GRAD_THRESHOLD,
            },
            "do_not_copy_center_50k_500": True,
            "do_not_copy_80k": True,
            "single_shared_l1": True,
            "reuse_origin_six_only_if_source_matches": True,
            "test_views_not_for_selection": list(TEST_VIEWS),
            "no_8x": True,
        },
        "x8": "independent_later_stage_from_this_L0_plus_L1",
        "test_views_003_012": "report_after_lock_only_not_for_design",
        "step_scale": MODEL_STEP_SCALE,
        "note": NOTE,
    }


__all__ = [
    "AZIMUTHS_DEG",
    "CENTER_ROI",
    "CENTER_MAX_POINTS",
    "DIAGNOSTIC_AZIMUTHS_DEG",
    "FULL_CORRESPONDENCE_STEPS",
    "FULL_L1_CKPT_STEPS",
    "FULL_L1_DENSIFY_UNTIL",
    "FULL_L1_MAX_POINTS",
    "FULL_L1_STEPS",
    "FULL_L1_VISITS_EXPECTED",
    "IMAGE_SIZE",
    "MIN_REPROJECTION_COVERAGE",
    "N_BASE_POSES",
    "OUTPUT_ROOT",
    "aabb_iou",
    "all_base_poses",
    "assert_writable_output",
    "camera_payload",
    "classify_pose",
    "dump_orbit_camera",
    "expected_fovs_deg",
    "expected_visits",
    "experiment_lock",
    "ground_quad",
    "horizontal_fov_deg",
    "lock_neighbor",
    "lookats",
    "make_orbit_camera",
    "neighbor_candidates",
    "occupancy_union",
    "origin_poses",
    "output_paths",
    "pose_id",
    "pose_index",
    "pose_name",
    "stage_scale_report",
    "structure_metrics",
    "zoomed_orbit_camera",
]
