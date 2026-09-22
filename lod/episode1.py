"""JAX_068 Episode 1: near-nadir pose with 1× → 4× → 8× focal zoom.

1× is context and the frozen parent baseline. L1 absorbs 4×, L2 absorbs 8×.
There is no 1× detail layer. Outputs stay under ``skyfall-gs_exp/lod_jax068_episode1``;
the full-scene archive is not touched.

This pose has no co-located real photograph. Pseudo-label PSNR is not real
reconstruction quality.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from lod.jax068 import DATASET_DIR, SCENE, STAGE1_CHECKPOINT
from utils.camera_utils import look_at_to_c2w
from utils.zoom_camera import NormalizedROI, zoom_fov, zoom_principal_point


SCENE_NAME = SCENE
STAGE1_PATH = STAGE1_CHECKPOINT
DATASET_PATH = DATASET_DIR
OUTPUT_ROOT = Path("skyfall-gs_exp/lod_jax068_episode1")
PARENT_SH_OUTPUT_ROOT = Path("skyfall-gs_exp/lod_jax068_episode1_parent_sh")
PARENT_FULL_OUTPUT_ROOT = Path("skyfall-gs_exp/lod_jax068_episode1_parent_full")
SCENE_ARCHIVE = Path("skyfall-gs_exp/lod_jax068_scene")

ELEVATION_DEG = 85.0
RADIUS = 300.0
AZIMUTH_DEG = 0.0
LOOKAT = (0.0, 0.0, 0.0)
UP = (0.0, 0.0, 1.0)
BASE_FOV_DEG = 60.0
IMAGE_SIZE = 2048
ZOOM_FACTORS = (1.0, 4.0, 8.0)
LEVEL_ZOOM = {1: 4.0, 2: 8.0}
# Adjacent intervals are 4× then 2×. Do not set the model-wide step_scale to 4.
STAGE_SCALES = (1.0, 4.0, 8.0)
MODEL_STEP_SCALE = 2.0
CENTER_ROI = NormalizedROI(0.5, 0.5, 0.1, 0.1)
NEIGHBOR_AZIMUTHS_DEG = (-10.0, 10.0, -20.0, 20.0, -30.0, 30.0)
TARGET_UID = 9000
NEIGHBOR_UID0 = 9100
SEED = 0
STEPS_PER_LEVEL = 500
SHORT_STEPS = 50
MAX_NEW_POINTS = 50_000
MIX_RATIO = 0.2
EVAL_STEPS = "0,50,100,250,500"

NOTE = (
    "No co-located real image at this pose. DLoRAL pseudo-label metrics are not "
    "a real reconstruction score. This run asks whether generated detail can be "
    "absorbed while coarser scales and neighbor correspondence stay stable."
)


def horizontal_fov_deg(base_fov_deg: float = BASE_FOV_DEG, zoom_factor: float = 1.0) -> float:
    """Horizontal FoV after the existing focal-zoom mapping."""

    base = math.radians(float(base_fov_deg))
    return math.degrees(zoom_fov(base, float(zoom_factor)))


def expected_fovs_deg() -> dict[str, float]:
    return {f"{int(factor)}x": horizontal_fov_deg(zoom_factor=factor) for factor in ZOOM_FACTORS}


def orbit_eye(
    azimuth_deg: float = AZIMUTH_DEG,
    *,
    elevation_deg: float = ELEVATION_DEG,
    radius: float = RADIUS,
    lookat: Sequence[float] = LOOKAT,
) -> np.ndarray:
    target = np.asarray(lookat, dtype=np.float64).reshape(3)
    theta = math.radians(float(azimuth_deg))
    phi = math.radians(float(elevation_deg))
    return target + np.array(
        [
            float(radius) * math.cos(theta) * math.cos(phi),
            float(radius) * math.sin(theta) * math.cos(phi),
            float(radius) * math.sin(phi),
        ],
        dtype=np.float64,
    )


def orbit_c2w(
    azimuth_deg: float = AZIMUTH_DEG,
    *,
    elevation_deg: float = ELEVATION_DEG,
    radius: float = RADIUS,
    lookat: Sequence[float] = LOOKAT,
    up: Sequence[float] = UP,
) -> np.ndarray:
    eye = orbit_eye(azimuth_deg, elevation_deg=elevation_deg, radius=radius, lookat=lookat)
    return np.asarray(look_at_to_c2w(eye, np.asarray(lookat, dtype=np.float64), np.asarray(up, dtype=np.float64)))


def orbit_w2c(
    azimuth_deg: float = AZIMUTH_DEG,
    **kwargs,
) -> np.ndarray:
    return np.linalg.inv(orbit_c2w(azimuth_deg, **kwargs))


def camera_name(azimuth_deg: float = AZIMUTH_DEG) -> str:
    value = float(azimuth_deg)
    tag = f"{value:g}".replace("-", "m").replace(".", "p")
    return f"episode1_e{ELEVATION_DEG:g}_r{RADIUS:g}_az{tag}"


def pose_rt(azimuth_deg: float = AZIMUTH_DEG, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Skyfall ``R`` (transposed w2c rotation) and ``T``."""

    w2c = orbit_w2c(azimuth_deg, **kwargs)
    return np.transpose(w2c[:3, :3]), w2c[:3, 3].copy()


def zoom_intrinsics(
    zoom_factor: float,
    *,
    width: int = IMAGE_SIZE,
    height: int = IMAGE_SIZE,
    base_fov_deg: float = BASE_FOV_DEG,
    roi: NormalizedROI = CENTER_ROI,
) -> dict[str, float]:
    fov = zoom_fov(math.radians(float(base_fov_deg)), float(zoom_factor))
    cx, cy = zoom_principal_point(0.0, 0.0, roi, float(zoom_factor))
    fx = float(width) / (2.0 * math.tan(fov / 2.0))
    fy = float(height) / (2.0 * math.tan(fov / 2.0))
    return {
        "zoom_factor": float(zoom_factor),
        "fov_rad": float(fov),
        "fov_deg": math.degrees(fov),
        "cx_ndc": float(cx),
        "cy_ndc": float(cy),
        "fx": fx,
        "fy": fy,
        "width": float(width),
        "height": float(height),
    }


def project_lookat(
    w2c: np.ndarray,
    lookat: Sequence[float] = LOOKAT,
    *,
    fx: float,
    fy: float,
    cx_pixel: float,
    cy_pixel: float,
) -> dict[str, float]:
    """Project the look-at point with pixel pinhole convention used by GaussianZoom."""

    point = np.append(np.asarray(lookat, dtype=np.float64).reshape(3), 1.0)
    cam = np.asarray(w2c, dtype=np.float64) @ point
    z = float(cam[2])
    if z <= 1e-8:
        return {"camera_z": z, "u": float("nan"), "v": float("nan"), "in_frame": False}
    u = fx * float(cam[0]) / z + float(cx_pixel)
    v = fy * float(cam[1]) / z + float(cy_pixel)
    return {
        "camera_z": z,
        "u": u,
        "v": v,
        "in_frame": 0.0 <= u < IMAGE_SIZE and 0.0 <= v < IMAGE_SIZE,
    }


def camera_payload(
    azimuth_deg: float,
    zoom_factor: float,
    *,
    uid: int,
) -> dict[str, Any]:
    c2w = orbit_c2w(azimuth_deg)
    w2c = np.linalg.inv(c2w)
    R, T = pose_rt(azimuth_deg)
    intra = zoom_intrinsics(zoom_factor)
    cx_pixel = 0.5 * IMAGE_SIZE * (intra["cx_ndc"] + 1.0)
    cy_pixel = 0.5 * IMAGE_SIZE * (intra["cy_ndc"] + 1.0)
    lookat = project_lookat(
        w2c, LOOKAT, fx=intra["fx"], fy=intra["fy"], cx_pixel=cx_pixel, cy_pixel=cy_pixel,
    )
    return {
        "name": camera_name(azimuth_deg),
        "uid": int(uid),
        "azimuth_deg": float(azimuth_deg),
        "elevation_deg": float(ELEVATION_DEG),
        "radius": float(RADIUS),
        "lookat": list(LOOKAT),
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
        "lookat_projection": lookat,
        "roi": {
            "center_x": CENTER_ROI.center_x,
            "center_y": CENTER_ROI.center_y,
            "width": CENTER_ROI.width,
            "height": CENTER_ROI.height,
        },
    }


def make_episode_camera(
    azimuth_deg: float = AZIMUTH_DEG,
    *,
    uid: int = TARGET_UID,
    data_device: str = "cuda",
):
    """1× Skyfall camera on the Episode 1 orbit. Zoom with ``make_zoom_camera``."""

    import torch
    from PIL import Image

    from scene.cameras import Camera
    from utils.general_utils import PILtoTorch

    R, T = pose_rt(azimuth_deg)
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
        image_name=camera_name(azimuth_deg),
        uid=int(uid),
        depth=None,
        mask=torch.ones((1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32),
        data_device=data_device,
        optimizing=False,
    )


def zoomed_episode_camera(base_camera, zoom_factor: float, *, uid: int | None = None):
    from utils.zoom_camera import make_zoom_camera

    factor = float(zoom_factor)
    return make_zoom_camera(
        base_camera,
        CENTER_ROI,
        factor,
        uid=int(uid if uid is not None else base_camera.uid + int(round(1000 * factor))),
        image_name=f"{base_camera.image_name}_zoom{factor:g}",
    )


def dump_skyfall_camera(camera, *, zoom_factor: float, azimuth_deg: float) -> dict[str, Any]:
    w2c = camera.world_view_transform.detach().transpose(0, 1).contiguous().cpu().float().numpy()
    center = camera.camera_center.detach().cpu().float().numpy()
    payload = camera_payload(azimuth_deg, zoom_factor, uid=int(camera.uid))
    payload.update(
        {
            "name": str(camera.image_name),
            "fx": float(camera.focal_x),
            "fy": float(camera.focal_y),
            "cx_ndc": float(camera.cx),
            "cy_ndc": float(camera.cy),
            "fov_x_deg": math.degrees(float(camera.FoVx)),
            "fov_y_deg": math.degrees(float(camera.FoVy)),
            "w2c": w2c.tolist(),
            "c2w": np.linalg.inv(w2c).tolist(),
            "camera_center": center.tolist(),
            "R": np.asarray(camera.R).tolist(),
            "T": np.asarray(camera.T).tolist(),
            "image_size": [int(camera.image_width), int(camera.image_height)],
        }
    )
    cx_pixel = 0.5 * float(camera.image_width) * (float(camera.cx) + 1.0)
    cy_pixel = 0.5 * float(camera.image_height) * (float(camera.cy) + 1.0)
    payload["cx_pixel"] = cx_pixel
    payload["cy_pixel"] = cy_pixel
    payload["lookat_projection"] = project_lookat(
        w2c, LOOKAT, fx=float(camera.focal_x), fy=float(camera.focal_y),
        cx_pixel=cx_pixel, cy_pixel=cy_pixel,
    )
    payload["torch_device"] = str(camera.world_view_transform.device)
    return payload


def structure_metrics(rgb, alpha, camera_z) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    color = rgb.detach().float()
    if color.ndim == 3 and color.shape[0] in (1, 3, 4):
        pass
    else:
        raise ValueError(f"rgb must be CHW, got {tuple(color.shape)}")
    if color.shape[0] == 4:
        color = color[:3]
    cover = alpha.detach().float()
    if cover.ndim == 3:
        cover = cover[0]
    depth = camera_z.detach().float()
    if depth.ndim == 3:
        depth = depth[0]
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=color.device,
        dtype=color.dtype,
    ).view(1, 1, 3, 3).expand(color.shape[0], 1, 3, 3)
    lap = F.conv2d(color.unsqueeze(0), kernel, padding=1, groups=color.shape[0])[0].abs()
    valid = cover > 0.05
    h, w = int(cover.shape[-2]), int(cover.shape[-1])
    cy, cx = h // 2, w // 2
    return {
        "mean_rgb": float(color.mean().item()),
        "mean_alpha": float(cover.mean().item()),
        "alpha_frac_gt_0.05": float(valid.float().mean().item()),
        "center_alpha": float(cover[cy, cx].item()),
        "center_rgb": [float(c) for c in color[:, cy, cx].reshape(-1).tolist()],
        "laplacian_energy": float(lap.mean().item()),
        "laplacian_energy_valid": float(lap.mean(dim=0)[valid].mean().item()) if bool(valid.any()) else 0.0,
        "center_camera_z": float(depth[cy, cx].item()),
        "median_camera_z_valid": float(depth[valid].median().item()) if bool(valid.any()) else float("nan"),
    }


def stage_scale_report(bundle) -> dict[str, Any]:
    records = list(getattr(bundle.lod, "stage_records", []) or [])
    scales = [float(stage["scale"]) for stage in records]
    adjacent = [scales[i] / scales[i - 1] for i in range(1, len(scales))]
    return {
        "n_layers": int(len(bundle.lod.layers)),
        "active_level": int(bundle.lod.active_level),
        "model_step_scale": float(bundle.lod.step_scale),
        "stage_scales": scales,
        "adjacent_ratios": adjacent,
        "expected_stage_scales": list(STAGE_SCALES[: len(scales)]),
        "weight_policy": str(getattr(bundle.lod, "weight_policy", "")),
    }


def reuse_episode1_assets(dest_root: str | Path, src_root: str | Path) -> dict[str, str]:
    """Symlink archived 4× DLoRAL and locked neighbor files into a new run dir."""

    src = output_paths(src_root)
    dest = output_paths(dest_root)
    dest["supervise_l1"].mkdir(parents=True, exist_ok=True)
    dest["neighbors"].mkdir(parents=True, exist_ok=True)
    linked: dict[str, str] = {}
    pairs = (
        ("refined", src["supervise_l1"] / "refined.png", dest["supervise_l1"] / "refined.png"),
        ("supervision", src["supervise_l1"] / "SUPERVISION.json", dest["supervise_l1"] / "SUPERVISION.json"),
        ("neighbors", src["neighbors"] / "NEIGHBORS.json", dest["neighbors"] / "NEIGHBORS.json"),
    )
    for key, source, target in pairs:
        if not source.is_file():
            raise FileNotFoundError(f"missing reusable Episode 1 asset {source}")
        if not (target.exists() or target.is_symlink()):
            target.symlink_to(source.resolve())
        linked[key] = str(target.resolve() if target.exists() else source.resolve())
    return linked


def output_paths(root: str | Path | None = None) -> dict[str, Path]:
    base = Path(root or OUTPUT_ROOT)
    return {
        "root": base,
        "baseline": base / "baseline",
        "neighbors": base / "neighbors",
        "accept": base / "accept",
        "supervise_l1": base / "supervise_l1",
        "train_l1": base / "train_l1",
        "supervise_l2": base / "supervise_l2",
        "train_l2": base / "train_l2",
        "report": base / "report",
        "video": base / "video",
        "visual_check": base / "visual_check",
    }


def parent_max_level_for_zoom(zoom_factor: float) -> int:
    """Parent of the layer that this zoom is meant to absorb: L0 at 4×, L0+L1 at 8×."""

    if float(zoom_factor) <= 4.0 + 1e-6:
        return 0
    return 1


def assert_not_scene_archive(path: str | Path) -> None:
    resolved = Path(path).resolve()
    archive = SCENE_ARCHIVE.resolve()
    if resolved == archive or archive in resolved.parents:
        raise ValueError(f"Episode 1 must not write into the scene archive {archive}")


def assert_not_original_episode1_archive(path: str | Path) -> None:
    """Parent-SH controls write outside the original Episode 1 and scene archives."""

    resolved = Path(path).resolve()
    assert_not_scene_archive(resolved)
    episode = OUTPUT_ROOT.resolve()
    if resolved == episode or episode in resolved.parents:
        raise ValueError(
            f"parent-SH experiment must not write into the original Episode 1 archive {episode}"
        )
