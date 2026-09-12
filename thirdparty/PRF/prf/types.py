from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SCHEMA_VERSION = 2


@dataclass
class GaussianScene:
    """Metric-frame 3D Gaussians. scales are linear meters, opacity in [0, 1]."""

    mu: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    opacity: np.ndarray
    normals: np.ndarray | None = None
    rgb: np.ndarray | None = None

    @property
    def num_gaussians(self) -> int:
        return int(self.mu.shape[0])

    def is_metric(self, extent_min: float, extent_max: float, scale_min: float, scale_max: float) -> bool:
        if self.num_gaussians == 0:
            return False
        extent = float(np.median(np.ptp(self.mu, axis=0)))
        scale = float(np.median(self.scales.max(axis=1)))
        return extent_min <= extent <= extent_max and scale_min <= scale <= scale_max


@dataclass
class PinholeView:
    view_id: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    c2w: np.ndarray
    w2c: np.ndarray
    center: np.ndarray
    file_path: str = ""

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project world points to pixels. Returns (uv [..., 2], z_cam [...])."""
        pts = np.asarray(points, dtype=np.float64)
        single = pts.ndim == 1
        if single:
            pts = pts[None, :]
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        homo = np.concatenate([pts, ones], axis=1)
        cam = (self.w2c @ homo.T).T
        z = cam[:, 2]
        uv = np.empty((pts.shape[0], 2), dtype=np.float64)
        valid = np.abs(z) > 1e-12
        uv[:, 0] = self.fx * cam[:, 0] / np.where(valid, z, 1.0) + self.cx
        uv[:, 1] = self.fy * cam[:, 1] / np.where(valid, z, 1.0) + self.cy
        if single:
            return uv[0], z[0]
        return uv, z


@dataclass
class SurfacePatch:
    patch_id: int
    center: np.ndarray
    normal: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    tangent: np.ndarray
    ids: np.ndarray
    radius: float
    valid: bool
    area_m2: float = 0.0
    hull_uv: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    quality_flags: list[str] = field(default_factory=list)
    thickness_m: float = 0.0
    eigen_ratio: float = 0.0
    keep_fraction: float = 0.0
    cluster_fraction: float = 0.0
    inlier_fraction: float = 0.0
    second_cluster_fraction: float = 0.0
    hull_margin: float = 0.0
    geometry_valid: bool = False
    geometry_confidence: float = 0.0
    plane_residual_m: float = 0.0
    support_count: int = 0


@dataclass
class PatchResolutionRecord:
    patch_id: int
    center_xyz_m: np.ndarray
    normal_xyz: np.ndarray
    tangent_1_xyz: np.ndarray
    tangent_2_xyz: np.ndarray
    patch_radius_m: float
    patch_area_m2: float
    member_ids: np.ndarray
    hull_uv: np.ndarray
    num_gaussians: int
    num_visible_real_views: int
    best_camera_pair: tuple[str, str] | None
    best_pair_angle_deg: float
    R_obs_m_per_equiv_pixel: float
    R_kernel_m_per_equiv_pixel: float
    R_spacing_m_per_equiv_pixel: float
    R_phys_m_per_equiv_pixel: float
    R_phys_cm_per_equiv_pixel: float
    bottleneck_type: str
    quality_flags: list[str]
    geometry_valid: bool = False
    geometry_confidence: float = 0.0
    plane_residual_m: float = 0.0
    support_count: int = 0
    inlier_fraction: float = 0.0
    eigen_ratio: float = 0.0


@dataclass
class PhysicalResolutionField:
    records: list[PatchResolutionRecord]
    config: dict
    source_ply: str = ""
    source_transforms: str = ""
    schema_version: int = SCHEMA_VERSION
    visibility_model: str = "frustum"
    visibility_by_patch: list[dict[str, str]] = field(default_factory=list)
