from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PRFConfig:
    """V1 defaults from 局部物理分辨率计算_工程实现设计.md §14."""

    patch_knn: int = 48
    k_perp: float = 2.5
    theta_surface_deg: float = 30.0
    use_gaussian_normals: bool = True
    min_opacity: float = 0.01
    min_patch_gaussians: int = 8
    nonplanar_eigen_ratio: float = 0.35
    cluster_normals: bool = True
    split_all_orientation_clusters: bool = True
    spatial_connect_knn: int = 6
    mixed_cluster_fraction: float = 0.50
    mixed_second_fraction: float = 0.20
    border_hull_margin: float = 0.15
    geometry_min_inlier_fraction: float = 0.80
    geometry_severe_border_margin: float = 0.0
    dedup_center_radius_scale: float = 0.75
    dedup_normal_deg: float = 25.0
    dedup_member_iou: float = 0.40

    jacobian_delta_ratio: float = 0.05
    delta_min_m: float = 0.01
    delta_max_m: float = 0.10
    min_pair_angle_deg: float = 12.0
    min_camera_z: float = 1.0
    eps: float = 1e-12
    occlusion_k: float = 2.5
    occlusion_alpha_visible: float = 0.50
    occlusion_alpha_empty: float = 0.05
    occlusion_min_cos: float = 0.15

    bandwidth_coeff: float = 3.77
    kernel_quantile: float = 0.90
    spacing_knn: int = 6
    spacing_quantile: float = 0.90

    anchor_voxel_m: float = 4.0
    max_anchors: int = 8000
    metric_extent_min_m: float = 10.0
    metric_extent_max_m: float = 1.0e5
    metric_scale_min_m: float = 0.01
    metric_scale_max_m: float = 50.0
