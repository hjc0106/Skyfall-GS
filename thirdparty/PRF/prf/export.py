from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from prf.manifest import describe_path
from prf.patches import pack_csr
from prf.types import SCHEMA_VERSION, PhysicalResolutionField
from prf.visibility import STATE_CODE, VISIBILITY_STATES

NPZ_REQUIRED = (
    "schema_version",
    "patch_id",
    "center",
    "normal",
    "tangent_1",
    "tangent_2",
    "patch_radius",
    "patch_area",
    "num_gaussians",
    "num_visible_real_views",
    "best_pair_angle_deg",
    "best_camera_a",
    "best_camera_b",
    "quality_flags",
    "R_obs",
    "R_kernel",
    "R_spacing",
    "R_phys",
    "bottleneck",
    "member_offsets",
    "member_gaussian_ids",
    "hull_offsets",
    "hull_uv",
)

NPZ_VECTOR_KEYS = ("center", "normal", "tangent_1", "tangent_2")
NPZ_SCALAR_KEYS = (
    "patch_radius",
    "patch_area",
    "R_obs",
    "R_kernel",
    "R_spacing",
    "R_phys",
    "bottleneck",
)
BOTTLENECK_LABELS = {"KERNEL", "SPACING", "OBS", "INVALID"}


def validate_npz_schema_v2(
    archive: Mapping[str, np.ndarray],
    *,
    scene_size: int | None = None,
) -> int:
    """Validate the public schema-v2 patch geometry contract.

    Returns the number of patches. ``scene_size`` additionally checks that CSR
    member IDs refer to the supplied Gaussian scene.
    """
    missing = sorted(set(NPZ_REQUIRED).difference(archive.keys()))
    if missing:
        raise ValueError("NPZ missing schema v2 keys: " + ", ".join(missing))

    version = int(np.asarray(archive["schema_version"]).reshape(-1)[0])
    if version != SCHEMA_VERSION:
        raise ValueError(f"NPZ schema_version={version}, expected {SCHEMA_VERSION}")

    patch_id = np.asarray(archive["patch_id"])
    if patch_id.ndim != 1 or patch_id.dtype.kind not in ("i", "u"):
        raise ValueError("patch_id must be a 1D integer array")
    n_patch = int(patch_id.shape[0])
    if np.unique(patch_id).size != n_patch:
        raise ValueError("patch_id values must be unique")

    for key in NPZ_VECTOR_KEYS:
        values = np.asarray(archive[key])
        if values.shape != (n_patch, 3):
            raise ValueError(f"{key} must have shape ({n_patch}, 3), got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{key} contains non-finite values")
    for key in NPZ_SCALAR_KEYS:
        values = np.asarray(archive[key])
        if values.shape != (n_patch,):
            raise ValueError(f"{key} must have shape ({n_patch},), got {values.shape}")
    for key in (
        "num_gaussians",
        "num_visible_real_views",
        "best_pair_angle_deg",
        "best_camera_a",
        "best_camera_b",
        "quality_flags",
    ):
        values = np.asarray(archive[key])
        if values.shape != (n_patch,):
            raise ValueError(f"{key} must have shape ({n_patch},), got {values.shape}")

    radius = np.asarray(archive["patch_radius"], dtype=np.float64)
    area = np.asarray(archive["patch_area"], dtype=np.float64)
    if not np.isfinite(radius).all() or np.any(radius <= 0.0):
        raise ValueError("patch_radius must be finite and positive")
    if not np.isfinite(area).all() or np.any(area <= 0.0):
        raise ValueError("patch_area must be finite and positive")

    normal = np.asarray(archive["normal"], dtype=np.float64)
    tangent_1 = np.asarray(archive["tangent_1"], dtype=np.float64)
    tangent_2 = np.asarray(archive["tangent_2"], dtype=np.float64)
    basis = np.stack((normal, tangent_1, tangent_2), axis=1)
    gram = basis @ np.swapaxes(basis, 1, 2)
    if not np.allclose(gram, np.eye(3)[None, :, :], atol=1e-5, rtol=1e-5):
        raise ValueError("normal/tangent basis must be orthonormal")

    labels = np.asarray(archive["bottleneck"]).astype(str)
    unknown = sorted(set(labels.tolist()).difference(BOTTLENECK_LABELS))
    if unknown:
        raise ValueError("Unknown bottleneck labels: " + ", ".join(unknown))
    for key in ("R_obs", "R_kernel", "R_spacing", "R_phys"):
        resolution = np.asarray(archive[key], dtype=np.float64)
        if np.isnan(resolution).any() or np.any(resolution < 0.0):
            raise ValueError(f"{key} may be non-negative or +inf, but not negative or NaN")
    r_phys = np.asarray(archive["R_phys"], dtype=np.float64)
    expected_invalid = ~np.isfinite(r_phys)
    if not np.array_equal(labels == "INVALID", expected_invalid):
        raise ValueError("bottleneck must be INVALID exactly where R_phys is non-finite")

    member_ids = np.asarray(archive["member_gaussian_ids"])
    member_offsets = _validate_csr_offsets(
        "member_offsets", archive["member_offsets"], n_patch, member_ids.size
    )
    if member_ids.ndim != 1 or member_ids.dtype.kind not in ("i", "u"):
        raise ValueError("member_gaussian_ids must be a 1D integer array")
    if np.any(member_ids < 0):
        raise ValueError("member_gaussian_ids contains negative IDs")
    if scene_size is not None and member_ids.size and int(member_ids.max()) >= int(scene_size):
        raise ValueError("member_gaussian_ids references a Gaussian outside the supplied scene")
    member_counts = np.diff(member_offsets)
    if np.any(member_counts < 3):
        raise ValueError("each patch must reference at least three member Gaussians")
    if not np.array_equal(np.asarray(archive["num_gaussians"], dtype=np.int64), member_counts):
        raise ValueError("num_gaussians does not match member_offsets")

    hull_uv = np.asarray(archive["hull_uv"], dtype=np.float64)
    if hull_uv.ndim != 2 or hull_uv.shape[1] != 2 or not np.isfinite(hull_uv).all():
        raise ValueError("hull_uv must be a finite [M, 2] array")
    hull_offsets = _validate_csr_offsets("hull_offsets", archive["hull_offsets"], n_patch, len(hull_uv))
    if np.any(np.diff(hull_offsets) < 3):
        raise ValueError("each patch hull must contain at least three vertices")

    hull_area = np.empty(n_patch, dtype=np.float64)
    for index in range(n_patch):
        polygon = hull_uv[hull_offsets[index] : hull_offsets[index + 1]]
        hull_area[index] = 0.5 * abs(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
    if not np.allclose(hull_area, area, atol=1e-8, rtol=1e-7):
        raise ValueError("patch_area does not match the tangent-plane convex-hull area")
    return n_patch


def _validate_csr_offsets(name: str, values: np.ndarray, n_patch: int, payload_size: int) -> np.ndarray:
    offsets = np.asarray(values)
    if offsets.shape != (n_patch + 1,) or offsets.dtype.kind not in ("i", "u"):
        raise ValueError(f"{name} must be a 1D integer array with {n_patch + 1} values")
    offsets64 = offsets.astype(np.int64, copy=False)
    if int(offsets64[0]) != 0 or np.any(np.diff(offsets64) < 0) or int(offsets64[-1]) != payload_size:
        raise ValueError(f"{name} is not a valid CSR offset array")
    return offsets64


def export_field(field: PhysicalResolutionField, output_dir: Path | str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = field.records
    csv_path = output / "patch_resolution.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "patch_id",
                "x_m",
                "y_m",
                "z_m",
                "nx",
                "ny",
                "nz",
                "t1x",
                "t1y",
                "t1z",
                "t2x",
                "t2y",
                "t2z",
                "patch_radius_m",
                "patch_area_m2",
                "num_gaussians",
                "num_visible_real_views",
                "best_camera_pair",
                "best_pair_angle_deg",
                "R_obs_m",
                "R_kernel_m",
                "R_spacing_m",
                "R_phys_m",
                "R_phys_cm",
                "bottleneck_type",
                "quality_flags",
                "geometry_valid",
                "geometry_confidence",
                "plane_residual_m",
                "support_count",
                "inlier_fraction",
                "eigen_ratio",
            ]
        )
        for rec in records:
            pair = "" if rec.best_camera_pair is None else ",".join(rec.best_camera_pair)
            writer.writerow(
                [
                    rec.patch_id,
                    *rec.center_xyz_m.tolist(),
                    *rec.normal_xyz.tolist(),
                    *rec.tangent_1_xyz.tolist(),
                    *rec.tangent_2_xyz.tolist(),
                    rec.patch_radius_m,
                    rec.patch_area_m2,
                    rec.num_gaussians,
                    rec.num_visible_real_views,
                    pair,
                    rec.best_pair_angle_deg,
                    rec.R_obs_m_per_equiv_pixel,
                    rec.R_kernel_m_per_equiv_pixel,
                    rec.R_spacing_m_per_equiv_pixel,
                    rec.R_phys_m_per_equiv_pixel,
                    rec.R_phys_cm_per_equiv_pixel,
                    rec.bottleneck_type,
                    "|".join(rec.quality_flags),
                    int(rec.geometry_valid),
                    rec.geometry_confidence,
                    rec.plane_residual_m,
                    rec.support_count,
                    rec.inlier_fraction,
                    rec.eigen_ratio,
                ]
            )

    member_offsets, member_ids = pack_csr(
        [np.asarray(rec.member_ids, dtype=np.int64).reshape(-1) for rec in records]
    )
    hull_offsets, hull_uv = pack_csr(
        [np.asarray(rec.hull_uv, dtype=np.float64).reshape(-1, 2) for rec in records],
        last_axis=2,
    )
    pair_a = np.array(
        ["" if rec.best_camera_pair is None else rec.best_camera_pair[0] for rec in records],
        dtype="<U64",
    )
    pair_b = np.array(
        ["" if rec.best_camera_pair is None else rec.best_camera_pair[1] for rec in records],
        dtype="<U64",
    )
    centers = np.stack([rec.center_xyz_m for rec in records]) if records else np.zeros((0, 3))
    np.savez_compressed(
        output / "physical_resolution_field.npz",
        schema_version=np.int32(SCHEMA_VERSION),
        patch_id=np.array([rec.patch_id for rec in records], dtype=np.int64),
        center=centers,
        normal=np.stack([rec.normal_xyz for rec in records]) if records else np.zeros((0, 3)),
        tangent_1=np.stack([rec.tangent_1_xyz for rec in records]) if records else np.zeros((0, 3)),
        tangent_2=np.stack([rec.tangent_2_xyz for rec in records]) if records else np.zeros((0, 3)),
        patch_radius=np.array([rec.patch_radius_m for rec in records], dtype=np.float64),
        patch_area=np.array([rec.patch_area_m2 for rec in records], dtype=np.float64),
        num_gaussians=np.array([rec.num_gaussians for rec in records], dtype=np.int32),
        num_visible_real_views=np.array([rec.num_visible_real_views for rec in records], dtype=np.int16),
        best_pair_angle_deg=np.array([rec.best_pair_angle_deg for rec in records], dtype=np.float64),
        best_camera_a=pair_a,
        best_camera_b=pair_b,
        quality_flags=np.array(["|".join(rec.quality_flags) for rec in records], dtype="<U256"),
        R_obs=np.array([rec.R_obs_m_per_equiv_pixel for rec in records], dtype=np.float64),
        R_kernel=np.array([rec.R_kernel_m_per_equiv_pixel for rec in records], dtype=np.float64),
        R_spacing=np.array([rec.R_spacing_m_per_equiv_pixel for rec in records], dtype=np.float64),
        R_phys=np.array([rec.R_phys_m_per_equiv_pixel for rec in records], dtype=np.float64),
        bottleneck=np.array([rec.bottleneck_type for rec in records], dtype="<U16"),
        member_offsets=member_offsets,
        member_gaussian_ids=np.asarray(member_ids, dtype=np.int64).reshape(-1),
        hull_offsets=hull_offsets,
        hull_uv=np.asarray(hull_uv, dtype=np.float64).reshape(-1, 2),
        geometry_valid=np.array([int(rec.geometry_valid) for rec in records], dtype=np.uint8),
        geometry_confidence=np.array([rec.geometry_confidence for rec in records], dtype=np.float64),
        plane_residual_m=np.array([rec.plane_residual_m for rec in records], dtype=np.float64),
        support_count=np.array([rec.support_count for rec in records], dtype=np.int32),
        inlier_fraction=np.array([rec.inlier_fraction for rec in records], dtype=np.float64),
        eigen_ratio=np.array([rec.eigen_ratio for rec in records], dtype=np.float64),
    )
    with np.load(output / "physical_resolution_field.npz", allow_pickle=False) as archive:
        validate_npz_schema_v2({key: archive[key] for key in archive.files})
    _write_patch_ply(output / "physical_resolution_field.ply", field)

    finite = [rec for rec in records if np.isfinite(rec.R_phys_m_per_equiv_pixel)]
    ply_info = describe_path(field.source_ply) if field.source_ply else {}
    transforms_info = describe_path(field.source_transforms) if field.source_transforms else {}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_ply": ply_info.get("path", field.source_ply),
        "source_ply_repo_relative": ply_info.get("path_repo_relative"),
        "source_ply_sha256": ply_info.get("sha256"),
        "source_transforms": transforms_info.get("path", field.source_transforms),
        "source_transforms_repo_relative": transforms_info.get("path_repo_relative"),
        "source_transforms_sha256": transforms_info.get("sha256"),
        "config": field.config,
        "num_patches": len(records),
        "num_finite": len(finite),
        "visibility_model": field.visibility_model,
        "R_phys_cm_quantiles": {},
        "bottleneck_counts": {},
    }
    if finite:
        values = np.array([rec.R_phys_cm_per_equiv_pixel for rec in finite])
        summary["R_phys_cm_quantiles"] = {
            str(q): float(np.quantile(values, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        }
        obs = np.array([rec.R_obs_m_per_equiv_pixel for rec in finite if np.isfinite(rec.R_obs_m_per_equiv_pixel)])
        if obs.size:
            summary["R_obs_m_quantiles"] = {
                str(q): float(np.quantile(obs, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
            }
        for rec in finite:
            summary["bottleneck_counts"][rec.bottleneck_type] = (
                summary["bottleneck_counts"].get(rec.bottleneck_type, 0) + 1
            )
    invalid = sum(1 for rec in records if rec.bottleneck_type == "INVALID")
    summary["num_invalid"] = invalid
    geometry_valid = [rec for rec in records if rec.geometry_valid]
    summary["num_geometry_valid"] = len(geometry_valid)
    summary["num_geometry_valid_finite"] = sum(
        1 for rec in geometry_valid if np.isfinite(rec.R_phys_m_per_equiv_pixel)
    )
    if records:
        inliers = np.array([rec.inlier_fraction for rec in records], dtype=np.float64)
        trusted_inliers = np.array([rec.inlier_fraction for rec in geometry_valid], dtype=np.float64)
        summary["inlier_fraction_median"] = float(np.median(inliers))
        if trusted_inliers.size:
            summary["geometry_valid_inlier_fraction_median"] = float(np.median(trusted_inliers))
        trusted_nonplanar = sum(1 for rec in geometry_valid if "NONPLANAR" in rec.quality_flags)
        summary["geometry_valid_nonplanar_ratio"] = float(trusted_nonplanar / max(len(geometry_valid), 1))
        summary["border_ratio"] = float(sum("BORDER" in rec.quality_flags for rec in records) / max(len(records), 1))
    if field.visibility_by_patch:
        counts: dict[str, int] = {}
        for row in field.visibility_by_patch:
            for state in row.values():
                counts[state] = counts.get(state, 0) + 1
        summary["visibility_state_counts"] = counts
        _write_visibility_npz(output / "visibility_states.npz", field)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def _write_visibility_npz(path: Path, field: PhysicalResolutionField) -> None:
    view_ids = sorted({view_id for row in field.visibility_by_patch for view_id in row})
    index = {name: i for i, name in enumerate(view_ids)}
    codes = np.full((len(field.records), len(view_ids)), 255, dtype=np.uint8)
    for row_i, row in enumerate(field.visibility_by_patch):
        for view_id, state in row.items():
            codes[row_i, index[view_id]] = STATE_CODE.get(state, 255)
    np.savez_compressed(
        path,
        view_id=np.array(view_ids, dtype="<U64"),
        patch_id=np.array([rec.patch_id for rec in field.records], dtype=np.int64),
        state_code=codes,
        state_name=np.array(VISIBILITY_STATES, dtype="<U32"),
    )


def _write_patch_ply(path: Path, field: PhysicalResolutionField) -> None:
    records = field.records
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(records)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property int patch_id\n"
        "property float t1x\nproperty float t1y\nproperty float t1z\n"
        "property float t2x\nproperty float t2y\nproperty float t2z\n"
        "property float patch_radius_m\nproperty float patch_area_m2\n"
        "property float R_obs_m\nproperty float R_kernel_m\nproperty float R_spacing_m\n"
        "property float R_phys_m\nproperty float R_phys_cm\nproperty uchar bottleneck_code\n"
        "end_header\n"
    )
    lines = [header]
    for rec in records:
        r_obs = rec.R_obs_m_per_equiv_pixel if np.isfinite(rec.R_obs_m_per_equiv_pixel) else -1.0
        r_ker = rec.R_kernel_m_per_equiv_pixel if np.isfinite(rec.R_kernel_m_per_equiv_pixel) else -1.0
        r_sp = rec.R_spacing_m_per_equiv_pixel if np.isfinite(rec.R_spacing_m_per_equiv_pixel) else -1.0
        r_ph = rec.R_phys_m_per_equiv_pixel if np.isfinite(rec.R_phys_m_per_equiv_pixel) else -1.0
        r_cm = rec.R_phys_cm_per_equiv_pixel if np.isfinite(rec.R_phys_cm_per_equiv_pixel) else -1.0
        bottleneck_code = {"KERNEL": 0, "SPACING": 1, "OBS": 2, "INVALID": 3}.get(rec.bottleneck_type, 3)
        xyzn = " ".join(f"{v:.6f}" for v in (*rec.center_xyz_m, *rec.normal_xyz))
        rest = " ".join(
            f"{v:.6f}"
            for v in (
                *rec.tangent_1_xyz,
                *rec.tangent_2_xyz,
                rec.patch_radius_m,
                rec.patch_area_m2,
                r_obs,
                r_ker,
                r_sp,
                r_ph,
                r_cm,
            )
        )
        lines.append(f"{xyzn} {rec.patch_id} {rest} {bottleneck_code}\n")
    path.write_text("".join(lines))
