from __future__ import annotations

import math

import numpy as np

from prf.types import PatchResolutionRecord, SurfacePatch


def fuse_resolutions(
    patch: SurfacePatch,
    r_obs: float,
    r_kernel: float,
    r_spacing: float,
    obs_info: dict,
) -> PatchResolutionRecord:
    values = {"OBS": r_obs, "KERNEL": r_kernel, "SPACING": r_spacing}
    r_phys = max(values.values())
    finite = {name: value for name, value in values.items() if math.isfinite(value)}
    if not math.isfinite(r_phys) or not finite:
        bottleneck = "INVALID"
    else:
        bottleneck = max(finite, key=finite.get)
    flags = list(obs_info.get("quality_flags", patch.quality_flags))
    pair = obs_info.get("best_camera_pair")
    return PatchResolutionRecord(
        patch_id=patch.patch_id,
        center_xyz_m=patch.center.copy(),
        normal_xyz=patch.normal.copy(),
        tangent_1_xyz=patch.t1.copy(),
        tangent_2_xyz=patch.t2.copy(),
        patch_radius_m=float(patch.radius),
        patch_area_m2=float(patch.area_m2),
        member_ids=np.asarray(patch.ids, dtype=np.int64).copy(),
        hull_uv=np.asarray(patch.hull_uv, dtype=np.float64).reshape(-1, 2).copy(),
        num_gaussians=int(patch.ids.size),
        num_visible_real_views=int(obs_info.get("num_visible_real_views", 0)),
        best_camera_pair=tuple(pair) if pair is not None else None,
        best_pair_angle_deg=float(obs_info.get("best_pair_angle_deg", 0.0)),
        R_obs_m_per_equiv_pixel=float(r_obs),
        R_kernel_m_per_equiv_pixel=float(r_kernel),
        R_spacing_m_per_equiv_pixel=float(r_spacing),
        R_phys_m_per_equiv_pixel=float(r_phys),
        R_phys_cm_per_equiv_pixel=100.0 * float(r_phys) if math.isfinite(r_phys) else float("inf"),
        bottleneck_type=bottleneck,
        quality_flags=flags,
        geometry_valid=bool(patch.geometry_valid),
        geometry_confidence=float(patch.geometry_confidence),
        plane_residual_m=float(patch.plane_residual_m),
        support_count=int(patch.support_count if patch.support_count else patch.ids.size),
        inlier_fraction=float(patch.inlier_fraction),
        eigen_ratio=float(patch.eigen_ratio),
    )
