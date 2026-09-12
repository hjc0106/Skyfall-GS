"""Build a 3D patch-surface viewer with Gaussian context and training-image linking."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import struct
from collections import Counter
from dataclasses import fields
from pathlib import Path

import numpy as np
from PIL import Image

from prf.cameras import load_training_views
from prf.color import (
    INVALID_RGB,
    OBS_ENHANCED_BINS_M,
    OBS_ENHANCED_RGB,
    RESOLUTION_BINS_M,
    RESOLUTION_RGB,
    obs_enhanced_codes,
    resolution_codes,
)
from prf.config import PRFConfig
from prf.export import validate_npz_schema_v2
from prf.gaussian_field import map_patch_fields_to_points, sample_context_indices
from prf.io_gs import load_gaussians_ply
from prf.obs import single_view_sampling_resolution
from prf.types import PinholeView, SurfacePatch

BOTTLENECK_CODE = {"KERNEL": 0, "SPACING": 1, "OBS": 2, "INVALID": 3}
METRIC_KEYS = ("R_obs", "R_kernel", "R_spacing", "R_phys")
ASSETS = Path(__file__).with_name("assets")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an interactive 3D PRF viewer.")
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--transforms", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opengl-c2w", action="store_true")
    parser.add_argument(
        "--context-points",
        type=int,
        default=0,
        help="Original-Ply points to map; 0 keeps every Gaussian (default).",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--image-quality", type=int, default=70)
    parser.add_argument("--max-images", type=int, default=17)
    return parser


def _finite_or_neg(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else -1.0


def _resolve_summary_path(field_dir: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (field_dir / path).resolve()


def _load_summary(field_dir: Path) -> dict:
    path = field_dir / "summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _load_sources(field_dir: Path, scene: Path | None, transforms: Path | None) -> tuple[Path, Path]:
    summary = _load_summary(field_dir)
    scene_path = scene.resolve() if scene else None
    transforms_path = transforms.resolve() if transforms else None
    if scene_path is None and summary.get("source_ply"):
        scene_path = _resolve_summary_path(field_dir, str(summary["source_ply"]))
    if transforms_path is None and summary.get("source_transforms"):
        transforms_path = _resolve_summary_path(field_dir, str(summary["source_transforms"]))
    if scene_path is None or transforms_path is None:
        raise ValueError("--scene and --transforms are required when summary.json has no source paths")
    return scene_path, transforms_path


def _config_from_summary(field_dir: Path) -> PRFConfig:
    raw = _load_summary(field_dir).get("config", {})
    allowed = {item.name for item in fields(PRFConfig)}
    return PRFConfig(**{key: value for key, value in raw.items() if key in allowed})


def _resolve_image(transforms_path: Path, file_path: str) -> Path:
    path = Path(file_path)
    return path if path.is_absolute() else (transforms_path.parent / path).resolve()


def _adjusted_w2c(w2c: np.ndarray, origin: np.ndarray) -> list[float]:
    adjusted = np.asarray(w2c[:3, :4], dtype=np.float64).copy()
    adjusted[:, 3] = adjusted[:, :3] @ origin + adjusted[:, 3]
    return adjusted.reshape(-1).tolist()


def _select_views(
    views: list[PinholeView],
    pair_a: list[str],
    pair_b: list[str],
    max_images: int,
) -> tuple[list[PinholeView], int]:
    """Prefer cameras referenced by best pairs, then fill in source order."""
    limit = max(int(max_images), 0)
    if limit == 0:
        return [], len({name for name in pair_a + pair_b if name})
    by_id = {view.view_id: view for view in views}
    counts = Counter(name for name in pair_a + pair_b if name in by_id)
    original_order = {view.view_id: index for index, view in enumerate(views)}
    ranked = sorted(counts, key=lambda name: (-counts[name], original_order[name]))
    chosen_ids = ranked[:limit]
    if len(chosen_ids) < limit:
        chosen = set(chosen_ids)
        chosen_ids.extend(view.view_id for view in views if view.view_id not in chosen)
        chosen_ids = chosen_ids[:limit]
    selected = [by_id[name] for name in chosen_ids]
    omitted = len(set(counts).difference(chosen_ids))
    return selected, omitted


def _per_view_metrics(
    centers: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    radii: np.ndarray,
    views: list[PinholeView],
    cfg: PRFConfig,
) -> tuple[np.ndarray, np.ndarray]:
    n_patch = centers.shape[0]
    n_view = len(views)
    resolutions = np.full((n_patch, n_view), -1.0, dtype=np.float32)
    inside = np.zeros((n_patch, n_view), dtype=np.uint8)
    for j, view in enumerate(views):
        for i in range(n_patch):
            patch = SurfacePatch(
                patch_id=i,
                center=centers[i],
                normal=np.cross(t1[i], t2[i]),
                t1=t1[i],
                t2=t2[i],
                tangent=np.column_stack((t1[i], t2[i])),
                ids=np.zeros(0, dtype=np.int64),
                radius=float(radii[i]),
                valid=True,
            )
            uv, z = view.project(centers[i])
            if np.isfinite(uv).all() and np.isfinite(z) and float(z) > cfg.min_camera_z:
                if 0.0 <= float(uv[0]) < view.width and 0.0 <= float(uv[1]) < view.height:
                    inside[i, j] = 1
            value = single_view_sampling_resolution(patch, view, cfg)
            if math.isfinite(value):
                resolutions[i, j] = np.float32(value)
    return resolutions, inside


def _write_binary(path: Path, magic: bytes, header: tuple[int, ...], chunks: list[np.ndarray]) -> None:
    with path.open("wb") as handle:
        handle.write(magic)
        handle.write(struct.pack("<" + "I" * len(header), *header))
        for chunk in chunks:
            handle.write(np.ascontiguousarray(chunk).tobytes())


def _write_mapped_ply(
    path: Path,
    xyz: np.ndarray,
    normals: np.ndarray,
    source_rgb: np.ndarray,
    mapped: np.ndarray,
    codes: np.ndarray,
    bottleneck: np.ndarray,
) -> None:
    """Write original Gaussian centres with PRF scalar fields and R_phys RGB."""
    dtype = np.dtype(
        [
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("source_red", "u1"), ("source_green", "u1"), ("source_blue", "u1"),
            ("prf_red", "u1"), ("prf_green", "u1"), ("prf_blue", "u1"),
            ("R_obs_m", "<f4"), ("R_kernel_m", "<f4"),
            ("R_spacing_m", "<f4"), ("R_phys_m", "<f4"),
            ("bottleneck_code", "u1"), ("prf_valid", "u1"),
            ("prf_covered", "u1"),
        ]
    )
    payload = np.empty(xyz.shape[0], dtype=dtype)
    for axis, name in enumerate(("x", "y", "z")):
        payload[name] = xyz[:, axis]
    for axis, name in enumerate(("nx", "ny", "nz")):
        payload[name] = normals[:, axis]
    display_rgb = np.full((xyz.shape[0], 3), INVALID_RGB, dtype=np.uint8)
    valid_color = codes[:, 3] < 254
    unmapped = codes[:, 3] == 254
    palette = np.asarray(RESOLUTION_RGB, dtype=np.uint8)
    display_rgb[valid_color] = palette[codes[valid_color, 3]]
    display_rgb[unmapped] = source_rgb[unmapped]
    blended_rgb = np.rint(0.68 * display_rgb + 0.32 * source_rgb).astype(np.uint8)
    for axis, name in enumerate(("red", "green", "blue")):
        payload[name] = blended_rgb[:, axis]
    for axis, name in enumerate(("source_red", "source_green", "source_blue")):
        payload[name] = source_rgb[:, axis]
    for axis, name in enumerate(("prf_red", "prf_green", "prf_blue")):
        payload[name] = display_rgb[:, axis]
    for axis, name in enumerate(("R_obs_m", "R_kernel_m", "R_spacing_m", "R_phys_m")):
        payload[name] = np.where(np.isfinite(mapped[:, axis]), mapped[:, axis], -1.0)
    payload["bottleneck_code"] = bottleneck
    payload["prf_valid"] = valid_color.astype(np.uint8)
    payload["prf_covered"] = (~unmapped).astype(np.uint8)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment PRF fields are surface-aware interpolations from schema-v2 patches\n"
        "comment bottleneck_code 0=KERNEL 1=SPACING 2=OBS 3=INVALID 4=UNMAPPED\n"
        f"element vertex {xyz.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar source_red\nproperty uchar source_green\nproperty uchar source_blue\n"
        "property uchar prf_red\nproperty uchar prf_green\nproperty uchar prf_blue\n"
        "property float R_obs_m\nproperty float R_kernel_m\n"
        "property float R_spacing_m\nproperty float R_phys_m\n"
        "property uchar bottleneck_code\nproperty uchar prf_valid\n"
        "property uchar prf_covered\nend_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        payload.tofile(handle)


def _preview_filename(index: int, view_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", view_id).strip("._") or f"view_{index:03d}"
    return f"{index:03d}_{safe}.jpg"


def build_vis3d(
    field_dir: Path,
    scene_path: Path,
    transforms_path: Path,
    output: Path,
    *,
    opengl_c2w: bool,
    context_points: int,
    image_size: int,
    image_quality: int,
    max_images: int,
) -> dict:
    field_dir = Path(field_dir)
    with np.load(field_dir / "physical_resolution_field.npz", allow_pickle=False) as data:
        archive = {key: data[key] for key in data.files}
    n_patch = validate_npz_schema_v2(archive)

    centers = np.asarray(archive["center"], dtype=np.float64)
    origin = 0.5 * (centers.min(axis=0) + centers.max(axis=0)) if len(centers) else np.zeros(3)
    relative = (centers - origin).astype(np.float32)
    normals = np.asarray(archive["normal"], dtype=np.float32)
    t1 = np.asarray(archive["tangent_1"], dtype=np.float32)
    t2 = np.asarray(archive["tangent_2"], dtype=np.float32)
    radius = np.asarray(archive["patch_radius"], dtype=np.float32)
    area = np.asarray(archive["patch_area"], dtype=np.float32)
    r_obs = np.array([_finite_or_neg(v) for v in archive["R_obs"]], dtype=np.float32)
    r_kernel = np.array([_finite_or_neg(v) for v in archive["R_kernel"]], dtype=np.float32)
    r_spacing = np.array([_finite_or_neg(v) for v in archive["R_spacing"]], dtype=np.float32)
    r_phys = np.array([_finite_or_neg(v) for v in archive["R_phys"]], dtype=np.float32)
    labels = archive["bottleneck"].astype(str)
    bottleneck = np.array([BOTTLENECK_CODE[name] for name in labels], dtype=np.uint8)
    visible = np.asarray(archive["num_visible_real_views"], dtype=np.uint16)
    pair_angle = np.asarray(archive["best_pair_angle_deg"], dtype=np.float32)
    pair_a = [str(v) for v in archive["best_camera_a"]]
    pair_b = [str(v) for v in archive["best_camera_b"]]
    hull_uv = np.asarray(archive["hull_uv"], dtype=np.float32).reshape(-1, 2)
    hull_offsets = np.asarray(archive["hull_offsets"], dtype=np.int32)
    member_ids = np.asarray(archive["member_gaussian_ids"], dtype=np.int64).reshape(-1)
    member_offsets = np.asarray(archive["member_offsets"], dtype=np.int32)

    scene = load_gaussians_ply(scene_path)
    validate_npz_schema_v2(archive, scene_size=scene.num_gaussians)
    member_xyz = (scene.mu[member_ids] - origin).astype(np.float32) if member_ids.size else np.zeros((0, 3), np.float32)

    all_views = load_training_views(transforms_path, opengl_c2w=opengl_c2w)
    views, omitted_best_pair_views = _select_views(all_views, pair_a, pair_b, max_images)
    r_view, inside = _per_view_metrics(
        centers,
        t1.astype(np.float64),
        t2.astype(np.float64),
        radius.astype(np.float64),
        views,
        _config_from_summary(field_dir),
    )

    if context_points <= 0 or context_points >= scene.num_gaussians:
        context_idx = np.arange(scene.num_gaussians, dtype=np.int64)
    else:
        context_idx = sample_context_indices(scene, context_points)
    field_xyz_world = scene.mu[context_idx]
    field_normals = scene.normals[context_idx]
    patch_values = np.column_stack(
        [np.asarray(archive[key], dtype=np.float64) for key in METRIC_KEYS]
    )
    mapped, nearest_patch = map_patch_fields_to_points(
        field_xyz_world,
        field_normals,
        centers,
        np.asarray(archive["normal"], dtype=np.float64),
        np.asarray(archive["tangent_1"], dtype=np.float64),
        np.asarray(archive["tangent_2"], dtype=np.float64),
        np.asarray(archive["patch_radius"], dtype=np.float64),
        patch_values,
        normal_angle_deg=80.0,
    )
    unsupported = ~np.isfinite(mapped)
    nearest_invalid_by_metric = np.zeros_like(unsupported)
    for metric_index in range(len(METRIC_KEYS)):
        nearest_invalid = ~np.isfinite(patch_values[nearest_patch, metric_index])
        nearest_invalid_by_metric[:, metric_index] = nearest_invalid
        mapped[nearest_invalid, metric_index] = np.nan
    field_codes = np.column_stack(
        [resolution_codes(mapped[:, metric_index]) for metric_index in range(len(METRIC_KEYS))]
    )
    field_codes[unsupported & ~nearest_invalid_by_metric] = np.uint8(254)
    obs_detail_codes = obs_enhanced_codes(mapped[:, 0])
    obs_detail_codes[unsupported[:, 0] & ~nearest_invalid_by_metric[:, 0]] = np.uint8(254)
    field_bottleneck = bottleneck[nearest_patch]
    field_bottleneck[unsupported[:, 1]] = np.uint8(4)
    field_xyz = (field_xyz_world - origin).astype(np.float32)
    if scene.rgb is None:
        field_rgb = np.full((context_idx.size, 3), 166, dtype=np.uint8)
    else:
        field_rgb = np.rint(np.clip(scene.rgb[context_idx], 0.0, 1.0) * 255.0).astype(np.uint8)

    output.mkdir(parents=True, exist_ok=True)
    mapped_ply_name = f"{scene_path.stem}_prf_mapped.ply"
    _write_mapped_ply(
        output / mapped_ply_name,
        field_xyz_world,
        field_normals,
        field_rgb,
        mapped,
        field_codes,
        field_bottleneck,
    )
    preview_dir = output / "training_previews"
    preview_dir.mkdir(exist_ok=True)
    for stale in preview_dir.glob("*.jpg"):
        stale.unlink()
    view_meta = []
    for index, view in enumerate(views):
        image_path = _resolve_image(transforms_path, view.file_path)
        preview_name = _preview_filename(index, view.view_id)
        preview_width = preview_height = 0
        if image_path.exists():
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
                preview_width, preview_height = image.size
                image.save(preview_dir / preview_name, format="JPEG", quality=image_quality, optimize=True)
        view_meta.append(
            {
                "id": view.view_id,
                "w": view.width,
                "h": view.height,
                "fx": view.fx,
                "fy": view.fy,
                "cx": view.cx,
                "cy": view.cy,
                "w2c": _adjusted_w2c(view.w2c, origin),
                "preview": f"training_previews/{preview_name}" if image_path.exists() else "",
                "preview_w": preview_width,
                "preview_h": preview_height,
            }
        )

    _write_binary(
        output / "patches.bin",
        b"PRF2",
        (n_patch, int(hull_uv.shape[0]), int(member_xyz.shape[0]), len(views)),
        [
            np.asarray(archive["patch_id"], dtype="<i4"),
            relative.reshape(-1).astype("<f4"),
            normals.reshape(-1).astype("<f4"),
            t1.reshape(-1).astype("<f4"),
            t2.reshape(-1).astype("<f4"),
            radius.astype("<f4"),
            area.astype("<f4"),
            r_obs.astype("<f4"),
            r_kernel.astype("<f4"),
            r_spacing.astype("<f4"),
            r_phys.astype("<f4"),
            bottleneck,
            visible.astype("<u2"),
            pair_angle.astype("<f4"),
            hull_offsets.astype("<i4"),
            hull_uv.reshape(-1).astype("<f4"),
            member_offsets.astype("<i4"),
            member_xyz.reshape(-1).astype("<f4"),
            r_view.astype("<f4").reshape(-1),
            inside.reshape(-1),
        ],
    )
    _write_binary(
        output / "gaussian_context.bin",
        b"GSF2",
        (int(field_xyz.shape[0]),),
        [
            field_xyz.reshape(-1).astype("<f4"),
            field_rgb.reshape(-1),
            field_codes[:, 0],
            field_codes[:, 1],
            field_codes[:, 2],
            field_codes[:, 3],
            obs_detail_codes,
            field_bottleneck,
        ],
    )

    metadata = {
        "schema_version": int(np.asarray(archive["schema_version"]).reshape(-1)[0]),
        "viewer_protocol": "prf-patch-surface-v2",
        "patch_geometry": "tangent-plane-convex-hull",
        "origin": origin.tolist(),
        "visibility_model": str(_load_summary(field_dir).get("visibility_model", "unknown")),
        "gaussian_context_format": "GSF2: xyz-float32 + rgb-uint8 + 4 absolute-bin uint8 + R_obs enhanced-bin uint8 + bottleneck uint8",
        "gaussian_mapping": "surface-aware 8-nearest-patch interpolation; nearest-patch validity",
        "mapped_ply": mapped_ply_name,
        "mapped_valid_fraction": {
            key: float(np.mean(field_codes[:, index] < 254))
            for index, key in enumerate(METRIC_KEYS)
        },
        "mapped_coverage_fraction": {
            key: float(np.mean(field_codes[:, index] != 254))
            for index, key in enumerate(METRIC_KEYS)
        },
        "color_bins_m": list(RESOLUTION_BINS_M),
        "R_obs_enhanced_bins_m": list(OBS_ENHANCED_BINS_M),
        "R_obs_enhanced_rgb": [list(rgb) for rgb in OBS_ENHANCED_RGB],
        "scene": scene_path.name,
        "pair_a": pair_a,
        "pair_b": pair_b,
        "views": view_meta,
        "omitted_best_pair_views": omitted_best_pair_views,
        "counts": {
            "patches": n_patch,
            "gaussians": int(field_xyz.shape[0]),
            "hull_vertices": int(hull_uv.shape[0]),
            "members": int(member_xyz.shape[0]),
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copyfile(ASSETS / "physical_resolution_3d.html", output / "index.html")
    shutil.copyfile(ASSETS / "prf-viewer.js", output / "prf-viewer.js")
    vendor_src = ASSETS / "vendor"
    vendor_dst = output / "vendor"
    vendor_dst.mkdir(exist_ok=True)
    for asset in ("three.module.js", "OrbitControls.js", "THREE-LICENSE.txt"):
        shutil.copyfile(vendor_src / asset, vendor_dst / asset)
    return metadata["counts"]


def main() -> None:
    args = build_parser().parse_args()
    scene_path, transforms_path = _load_sources(args.field, args.scene, args.transforms)
    counts = build_vis3d(
        args.field,
        scene_path,
        transforms_path,
        args.output,
        opengl_c2w=args.opengl_c2w,
        context_points=args.context_points,
        image_size=args.image_size,
        image_quality=args.image_quality,
        max_images=args.max_images,
    )
    print(json.dumps({"output": str(args.output.resolve()), **counts}))


if __name__ == "__main__":
    main()
