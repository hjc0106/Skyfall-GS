from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from prf.config import PRFConfig
from prf.manifest import canonical_path
from prf.patches import select_patch_anchors
from prf.pipeline import build_valid_patches
from prf.types import GaussianScene

REGION_SPECS = ("roof", "road", "facade")


def _json_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _flag_rates(flag_rows: list[list[str]], n_patch: int) -> dict[str, dict[str, float]]:
    counts: Counter[str] = Counter()
    for flags in flag_rows:
        counts.update(flags)
    return {
        name: {"count": int(count), "ratio": float(count) / float(max(n_patch, 1))}
        for name, count in sorted(counts.items())
    }


def _parse_flags(values: np.ndarray) -> list[list[str]]:
    rows = []
    for raw in np.asarray(values).astype(str):
        rows.append([name for name in raw.split("|") if name])
    return rows


def typical_region_mask(
    centers: np.ndarray,
    normals: np.ndarray,
    region: str,
    *,
    z_low: float,
    z_high: float,
    up_cos: float = 0.85,
    facade_cos: float = 0.40,
) -> np.ndarray:
    nz = np.abs(np.asarray(normals, dtype=np.float64)[:, 2])
    z = np.asarray(centers, dtype=np.float64)[:, 2]
    if region == "roof":
        return (nz >= up_cos) & (z >= z_high)
    if region == "road":
        return (nz >= up_cos) & (z <= z_low)
    if region == "facade":
        return nz <= facade_cos
    raise ValueError(f"unknown region {region}")


def _sample_region(
    mask: np.ndarray,
    flags: list[list[str]],
    rng: np.random.Generator,
    n_planar: int = 8,
    n_nonplanar: int = 4,
) -> list[int]:
    indices = np.nonzero(mask)[0]
    if indices.size == 0:
        return []
    planar = [int(i) for i in indices if "NONPLANAR" not in flags[int(i)]]
    nonplanar = [int(i) for i in indices if "NONPLANAR" in flags[int(i)]]
    picked: list[int] = []
    if planar:
        take = min(n_planar, len(planar))
        picked.extend(rng.choice(planar, size=take, replace=False).tolist())
    if nonplanar:
        take = min(n_nonplanar, len(nonplanar))
        picked.extend(rng.choice(nonplanar, size=take, replace=False).tolist())
    if len(picked) < n_planar + n_nonplanar:
        remain = [int(i) for i in indices if int(i) not in picked]
        extra = min((n_planar + n_nonplanar) - len(picked), len(remain))
        if extra:
            picked.extend(rng.choice(remain, size=extra, replace=False).tolist())
    return picked


def diagnose_field_archive(archive: dict[str, np.ndarray], *, seed: int = 0) -> dict:
    centers = np.asarray(archive["center"], dtype=np.float64)
    normals = np.asarray(archive["normal"], dtype=np.float64)
    flags = _parse_flags(archive["quality_flags"])
    n_patch = centers.shape[0]
    z = centers[:, 2]
    z_low, z_high = np.quantile(z, [0.30, 0.70])
    rng = np.random.default_rng(seed)
    samples = []
    region_counts = {}
    for region in REGION_SPECS:
        mask = typical_region_mask(centers, normals, region, z_low=float(z_low), z_high=float(z_high))
        region_counts[region] = int(mask.sum())
        for index in _sample_region(mask, flags, rng):
            samples.append(
                {
                    "patch_id": int(archive["patch_id"][index]),
                    "index": int(index),
                    "region": region,
                    "center": centers[index].tolist(),
                    "normal": normals[index].tolist(),
                    "quality_flags": flags[index],
                    "patch_radius_m": float(archive["patch_radius"][index]),
                    "patch_area_m2": float(archive["patch_area"][index]),
                    "num_gaussians": int(archive["num_gaussians"][index]),
                    "R_obs": _json_float(archive["R_obs"][index]),
                    "R_kernel": _json_float(archive["R_kernel"][index]),
                    "R_spacing": _json_float(archive["R_spacing"][index]),
                    "R_phys": _json_float(archive["R_phys"][index]),
                    "bottleneck": str(archive["bottleneck"][index]),
                }
            )
    finite = np.isfinite(np.asarray(archive["R_phys"], dtype=np.float64))
    geometry_valid = (
        np.asarray(archive["geometry_valid"]).astype(bool)
        if "geometry_valid" in archive
        else np.zeros(n_patch, dtype=bool)
    )
    trusted_flags = [flags[i] for i in range(n_patch) if geometry_valid[i]]
    result = {
        "num_patches": n_patch,
        "num_finite": int(finite.sum()),
        "num_geometry_valid": int(geometry_valid.sum()),
        "region_label": "geometric_heuristic",
        "region_label_note": (
            "roof/road/facade are |nz| and z-quantile heuristics, not semantic classes. "
            "Confirm on original PLY members and training-image projections before treating a sample as that surface."
        ),
        "flag_rates": _flag_rates(flags, n_patch),
        "geometry_valid_flag_rates": _flag_rates(trusted_flags, int(geometry_valid.sum())),
        "z_quantiles": {"0.3": float(z_low), "0.7": float(z_high)},
        "region_counts": region_counts,
        "samples": samples,
    }
    if "inlier_fraction" in archive:
        inliers = np.asarray(archive["inlier_fraction"], dtype=np.float64)
        result["inlier_fraction"] = _describe(inliers)
        if geometry_valid.any():
            result["geometry_valid_inlier_fraction"] = _describe(inliers[geometry_valid])
    if "plane_residual_m" in archive:
        result["plane_residual_m"] = _describe(np.asarray(archive["plane_residual_m"], dtype=np.float64))
    if "eigen_ratio" in archive:
        result["eigen_ratio"] = _describe(np.asarray(archive["eigen_ratio"], dtype=np.float64))
    return result


def diagnose_rebuilt_geometry(scene: GaussianScene, cfg: PRFConfig) -> dict:
    patches = build_valid_patches(scene, cfg)
    eigen = []
    keep = []
    cluster = []
    inlier = []
    second = []
    margin = []
    thickness = []
    area = []
    residual = []
    confidence = []
    flags: list[list[str]] = []
    trusted = 0
    for patch in patches:
        eigen.append(patch.eigen_ratio)
        keep.append(patch.keep_fraction)
        cluster.append(patch.cluster_fraction)
        inlier.append(patch.inlier_fraction)
        second.append(patch.second_cluster_fraction)
        margin.append(patch.hull_margin)
        thickness.append(patch.thickness_m)
        area.append(patch.area_m2)
        residual.append(patch.plane_residual_m)
        confidence.append(patch.geometry_confidence)
        flags.append(patch.quality_flags)
        trusted += int(patch.geometry_valid)
    valid = len(patches)
    trusted_flags = [patch.quality_flags for patch in patches if patch.geometry_valid]
    trusted_inlier = [patch.inlier_fraction for patch in patches if patch.geometry_valid]
    return {
        "num_anchors": int(select_patch_anchors(scene, cfg).size),
        "num_valid_patches": valid,
        "num_geometry_valid": trusted,
        "flag_rates": _flag_rates(flags, valid),
        "geometry_valid_flag_rates": _flag_rates(trusted_flags, trusted),
        "eigen_ratio": _describe(np.asarray(eigen, dtype=np.float64)),
        "keep_fraction": _describe(np.asarray(keep, dtype=np.float64)),
        "cluster_fraction": _describe(np.asarray(cluster, dtype=np.float64)),
        "inlier_fraction": _describe(np.asarray(inlier, dtype=np.float64)),
        "geometry_valid_inlier_fraction": _describe(np.asarray(trusted_inlier, dtype=np.float64)),
        "second_cluster_fraction": _describe(np.asarray(second, dtype=np.float64)),
        "hull_margin": _describe(np.asarray(margin, dtype=np.float64)),
        "thickness_m": _describe(np.asarray(thickness, dtype=np.float64)),
        "plane_residual_m": _describe(np.asarray(residual, dtype=np.float64)),
        "geometry_confidence": _describe(np.asarray(confidence, dtype=np.float64)),
        "patch_area_m2": _describe(np.asarray(area, dtype=np.float64)),
        "mixed_and_nonplanar_ratio": float(
            sum("MIXED_NEIGHBORHOOD" in row and "NONPLANAR" in row for row in flags) / max(valid, 1)
        ),
        "border_and_nonplanar_ratio": float(
            sum("BORDER" in row and "NONPLANAR" in row for row in flags) / max(valid, 1)
        ),
        "geometry_valid_nonplanar_ratio": float(
            sum("NONPLANAR" in row for row in trusted_flags) / max(trusted, 1)
        ),
    }


def _describe(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
    }


def compare_observation_archives(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> dict:
    """Compare frozen-geometry observation fields. Kernel/spacing must match by patch_id."""
    before_ids = np.asarray(before["patch_id"])
    after_ids = np.asarray(after["patch_id"])
    if before_ids.shape != after_ids.shape or not np.array_equal(before_ids, after_ids):
        raise ValueError("patch_id arrays differ; freeze geometry before comparing R_obs")
    kernel_delta = np.asarray(after["R_kernel"], dtype=np.float64) - np.asarray(before["R_kernel"], dtype=np.float64)
    spacing_delta = np.asarray(after["R_spacing"], dtype=np.float64) - np.asarray(before["R_spacing"], dtype=np.float64)
    obs_before = np.asarray(before["R_obs"], dtype=np.float64)
    obs_after = np.asarray(after["R_obs"], dtype=np.float64)
    both = np.isfinite(obs_before) & np.isfinite(obs_after)
    lost = np.isfinite(obs_before) & ~np.isfinite(obs_after)
    gained = ~np.isfinite(obs_before) & np.isfinite(obs_after)
    ratio = np.full(obs_before.shape, np.nan)
    ratio[both] = obs_after[both] / np.maximum(obs_before[both], 1e-12)
    bottleneck_before = np.asarray(before["bottleneck"]).astype(str)
    bottleneck_after = np.asarray(after["bottleneck"]).astype(str)
    changed = bottleneck_before != bottleneck_after
    return {
        "num_patches": int(before_ids.size),
        "kernel_max_abs_delta": float(np.nanmax(np.abs(kernel_delta))) if kernel_delta.size else 0.0,
        "spacing_max_abs_delta": float(np.nanmax(np.abs(spacing_delta))) if spacing_delta.size else 0.0,
        "num_finite_obs_before": int(np.isfinite(obs_before).sum()),
        "num_finite_obs_after": int(np.isfinite(obs_after).sum()),
        "num_lost_obs": int(lost.sum()),
        "num_gained_obs": int(gained.sum()),
        "R_obs_ratio_both_finite": _describe(ratio[both]) if both.any() else {},
        "R_obs_abs_delta_both_finite": _describe(np.abs(obs_after[both] - obs_before[both])) if both.any() else {},
        "bottleneck_changed": int(changed.sum()),
        "bottleneck_before": {str(k): int(v) for k, v in zip(*np.unique(bottleneck_before, return_counts=True))},
        "bottleneck_after": {str(k): int(v) for k, v in zip(*np.unique(bottleneck_after, return_counts=True))},
    }


def compare_field_statistics(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> dict:
    """Compare two PRF fields without requiring identical patch IDs."""

    def stats(archive: dict[str, np.ndarray]) -> dict:
        r_obs = np.asarray(archive["R_obs"], dtype=np.float64)
        r_ker = np.asarray(archive["R_kernel"], dtype=np.float64)
        r_sp = np.asarray(archive["R_spacing"], dtype=np.float64)
        r_ph = np.asarray(archive["R_phys"], dtype=np.float64)
        finite = np.isfinite(r_ph)
        flags = _parse_flags(archive["quality_flags"])
        return {
            "num_patches": int(np.asarray(archive["patch_id"]).size),
            "num_finite": int(finite.sum()),
            "finite_rate": float(finite.mean()) if finite.size else 0.0,
            "flag_rates": _flag_rates(flags, int(np.asarray(archive["patch_id"]).size)),
            "patch_area_m2": _describe(np.asarray(archive["patch_area"], dtype=np.float64)),
            "R_obs": _describe(r_obs[np.isfinite(r_obs)]),
            "R_kernel": _describe(r_ker[np.isfinite(r_ker)]),
            "R_spacing": _describe(r_sp[np.isfinite(r_sp)]),
            "R_phys": _describe(r_ph[finite]),
            "bottleneck": {
                str(k): int(v) for k, v in zip(*np.unique(np.asarray(archive["bottleneck"]).astype(str), return_counts=True))
            },
        }

    return {"before": stats(before), "after": stats(after), "spatial_match": match_patches_spatial(before, after)}


def match_patches_spatial(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    *,
    center_tol_m: float = 3.0,
    min_normal_abs_dot: float = 0.9397,
) -> dict:
    """Match patches by centre proximity and unsigned normal agreement."""
    before_c = np.asarray(before["center"], dtype=np.float64)
    after_c = np.asarray(after["center"], dtype=np.float64)
    before_n = np.asarray(before["normal"], dtype=np.float64)
    after_n = np.asarray(after["normal"], dtype=np.float64)
    if before_c.size == 0 or after_c.size == 0:
        return {"num_matched": 0}
    tree = cKDTree(before_c)
    dist, index = tree.query(after_c, k=1)
    cosine = np.abs(np.einsum("ij,ij->i", after_n, before_n[index]))
    matched = (dist <= center_tol_m) & (cosine >= min_normal_abs_dot)
    before_flags = _parse_flags(before["quality_flags"])
    after_flags = _parse_flags(after["quality_flags"])
    planar = np.array(
        [
            "NONPLANAR" not in before_flags[int(i)] and "NONPLANAR" not in after_flags[int(j)]
            for i, j in zip(index.tolist(), range(after_c.shape[0]))
        ],
        dtype=bool,
    )
    stable = matched & planar
    payload = {
        "center_tol_m": float(center_tol_m),
        "min_normal_abs_dot": float(min_normal_abs_dot),
        "num_matched": int(matched.sum()),
        "num_stable_planar": int(stable.sum()),
        "match_rate_after": float(matched.mean()),
    }
    if "geometry_valid" in after:
        payload["num_matched_geometry_valid"] = int(
            (matched & np.asarray(after["geometry_valid"]).astype(bool)).sum()
        )
    for key in ("R_kernel", "R_spacing"):
        b_val = np.asarray(before[key], dtype=np.float64)[index]
        a_val = np.asarray(after[key], dtype=np.float64)
        both = matched & np.isfinite(b_val) & np.isfinite(a_val)
        ratio = a_val[both] / np.maximum(b_val[both], 1e-12)
        payload[f"{key}_ratio_matched"] = _describe(ratio)
        both_stable = stable & np.isfinite(b_val) & np.isfinite(a_val)
        payload[f"{key}_ratio_stable_planar"] = _describe(
            a_val[both_stable] / np.maximum(b_val[both_stable], 1e-12)
        )
    return payload


def write_diagnostics(payload: dict, output_dir: Path | str) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "patch_geometry_diagnostics.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    samples = payload.get("samples", [])
    if not samples and isinstance(payload.get("field"), dict):
        samples = payload["field"].get("samples", [])
    lines = [
        "patch_id,region,x_m,y_m,z_m,nx,ny,nz,flags,radius_m,area_m2,num_gaussians,R_obs,R_kernel,R_spacing,R_phys,bottleneck"
    ]
    for sample in samples:
        xyz = sample["center"]
        nxyz = sample["normal"]
        lines.append(
            ",".join(
                [
                    str(sample["patch_id"]),
                    sample["region"],
                    *(f"{v:.6f}" for v in (*xyz, *nxyz)),
                    "|".join(sample["quality_flags"]),
                    f"{sample['patch_radius_m']:.6f}",
                    f"{sample['patch_area_m2']:.6f}",
                    str(sample["num_gaussians"]),
                    f"{sample['R_obs'] if sample['R_obs'] is not None else 'inf'}",
                    f"{sample['R_kernel'] if sample['R_kernel'] is not None else 'inf'}",
                    f"{sample['R_spacing'] if sample['R_spacing'] is not None else 'inf'}",
                    f"{sample['R_phys'] if sample['R_phys'] is not None else 'inf'}",
                    sample["bottleneck"],
                ]
            )
        )
    (output / "typical_region_samples.csv").write_text("\n".join(lines) + "\n")
    return path


def write_region_spot_checks(
    archive: dict[str, np.ndarray],
    samples: list[dict],
    scene: GaussianScene,
    views: list,
    transforms_path: Path | str,
    output_dir: Path | str,
    *,
    crop: int = 256,
) -> Path:
    """Project geometric-heuristic samples onto training images and dump member PLYs."""
    from prf.types import SurfacePatch

    output = Path(output_dir) / "region_spotchecks"
    output.mkdir(parents=True, exist_ok=True)
    image_roots = _image_roots(transforms_path)
    member_offsets = np.asarray(archive["member_offsets"], dtype=np.int64)
    member_ids = np.asarray(archive["member_gaussian_ids"], dtype=np.int64)
    hull_offsets = np.asarray(archive["hull_offsets"], dtype=np.int64)
    hull_uv = np.asarray(archive["hull_uv"], dtype=np.float64).reshape(-1, 2)
    index_by_id = {int(pid): i for i, pid in enumerate(np.asarray(archive["patch_id"]).tolist())}
    records = []
    for sample in samples:
        patch_id = int(sample["patch_id"])
        index = index_by_id[patch_id]
        ids = member_ids[member_offsets[index] : member_offsets[index + 1]]
        ply_path = output / f"{sample['region']}_{patch_id}_members.ply"
        _write_member_ply(ply_path, scene, ids)
        patch = SurfacePatch(
            patch_id=patch_id,
            center=np.asarray(archive["center"][index], dtype=np.float64),
            normal=np.asarray(archive["normal"][index], dtype=np.float64),
            t1=np.asarray(archive["tangent_1"][index], dtype=np.float64),
            t2=np.asarray(archive["tangent_2"][index], dtype=np.float64),
            tangent=np.column_stack(
                (np.asarray(archive["tangent_1"][index]), np.asarray(archive["tangent_2"][index]))
            ),
            ids=ids,
            radius=float(archive["patch_radius"][index]),
            valid=True,
            hull_uv=hull_uv[hull_offsets[index] : hull_offsets[index + 1]],
        )
        overlays = []
        for view in views:
            image_path = _find_view_image(view.file_path, image_roots)
            if image_path is None:
                continue
            uv, z = view.project(patch.center)
            if float(z) <= 1.0 or not (0.0 <= float(uv[0]) < view.width and 0.0 <= float(uv[1]) < view.height):
                continue
            overlay = output / f"{sample['region']}_{patch_id}_{view.view_id}.jpg"
            _draw_patch_overlay(image_path, overlay, view, patch, ids, scene, crop=crop)
            overlays.append({"view_id": view.view_id, "image": overlay.name, "uv": [float(uv[0]), float(uv[1])]})
            if len(overlays) >= 2:
                break
        records.append(
            {
                "patch_id": patch_id,
                "region": sample["region"],
                "region_label": "geometric_heuristic",
                "member_ply": ply_path.name,
                "num_gaussians": int(ids.size),
                "center": sample["center"],
                "overlays": overlays,
            }
        )
    index_path = output / "index.json"
    index_path.write_text(json.dumps({"note": "roof/road/facade are geometric heuristics.", "samples": records}, indent=2) + "\n")
    return index_path


def _image_roots(transforms_path: Path | str) -> list[Path]:
    import os

    path = canonical_path(transforms_path)
    roots = [path.parent, Path(os.path.realpath(path)).parent]
    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def _find_view_image(file_path: str, roots: list[Path]) -> Path | None:
    relative = Path(file_path)
    candidates = []
    for root in roots:
        candidates.append(root / relative)
        candidates.append(root / relative.name)
        candidates.append(root / "images" / relative.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _write_member_ply(path: Path, scene: GaussianScene, ids: np.ndarray) -> None:
    pts = scene.mu[ids]
    rgb = np.full((ids.size, 3), 220, dtype=np.uint8)
    if scene.rgb is not None:
        rgb = (np.clip(scene.rgb[ids], 0.0, 1.0) * 255.0).astype(np.uint8)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {ids.size}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for (x, y, z), (r, g, b) in zip(pts, rgb, strict=True):
        lines.append(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}")
    path.write_text("\n".join(lines) + "\n")


def _draw_patch_overlay(image_path: Path, out_path: Path, view, patch, ids, scene: GaussianScene, *, crop: int) -> None:
    from PIL import Image, ImageDraw

    from prf.patches import hull_world_points

    image = Image.open(image_path).convert("RGB")
    uv_c, _ = view.project(patch.center)
    cx, cy = float(uv_c[0]), float(uv_c[1])
    half = crop // 2
    box = (
        int(round(cx)) - half,
        int(round(cy)) - half,
        int(round(cx)) + half,
        int(round(cy)) + half,
    )
    cropped = image.crop(box)
    draw = ImageDraw.Draw(cropped)
    hull = hull_world_points(patch)
    if hull.shape[0] >= 3:
        uv_h, _ = view.project(hull)
        poly = [(float(u) - box[0], float(v) - box[1]) for u, v in uv_h]
        draw.polygon(poly, outline=(255, 48, 48))
    if ids.size:
        uv_m, z_m = view.project(scene.mu[ids])
        for (u, v), z in zip(uv_m, z_m):
            if z <= 1.0:
                continue
            draw.ellipse((u - box[0] - 1, v - box[1] - 1, u - box[0] + 1, v - box[1] + 1), fill=(255, 220, 40))
    draw.ellipse((cx - box[0] - 3, cy - box[1] - 3, cx - box[0] + 3, cy - box[1] + 3), outline=(80, 220, 255), width=2)
    cropped.save(out_path, quality=92)
