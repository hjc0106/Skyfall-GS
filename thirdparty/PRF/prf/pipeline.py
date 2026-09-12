from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from scipy.spatial import cKDTree

from prf.cameras import load_training_views
from prf.config import PRFConfig
from prf.fusion import fuse_resolutions
from prf.io_gs import assert_metric, load_gaussians_ply
from prf.kernel import compute_kernel_resolution
from prf.manifest import canonical_path
from prf.obs import compute_observation_resolution
from prf.occlusion import OcclusionMaps
from prf.patches import (
    build_candidate_patches_at_anchor,
    deduplicate_patches,
    patches_from_archive,
    select_patch_anchors,
)
from prf.spacing import compute_spacing_resolution
from prf.types import GaussianScene, PhysicalResolutionField, PinholeView, SurfacePatch, PatchResolutionRecord


def build_valid_patches(scene: GaussianScene, cfg: PRFConfig) -> list[SurfacePatch]:
    tree = cKDTree(scene.mu)
    patches: list[SurfacePatch] = []
    for anchor_id in select_patch_anchors(scene, cfg).tolist():
        patches.extend(build_candidate_patches_at_anchor(scene, tree, int(anchor_id), cfg))
    return [patch for patch in deduplicate_patches(patches, cfg) if patch.valid]


def evaluate_patches(
    patches: list[SurfacePatch],
    scene: GaussianScene,
    training_views: list[PinholeView],
    cfg: PRFConfig,
    *,
    occlusion_maps: OcclusionMaps | None = None,
    kernel_values: list[float] | None = None,
    spacing_values: list[float] | None = None,
) -> tuple[list[PatchResolutionRecord], list[dict[str, str]]]:
    records = []
    visibility = []
    for index, patch in enumerate(patches):
        r_obs, obs_info = compute_observation_resolution(
            patch,
            training_views,
            cfg,
            scene=scene,
            occlusion_maps=occlusion_maps,
        )
        r_kernel = (
            float(kernel_values[index])
            if kernel_values is not None
            else compute_kernel_resolution(patch, scene, cfg)
        )
        r_spacing = (
            float(spacing_values[index])
            if spacing_values is not None
            else compute_spacing_resolution(patch, scene, cfg)
        )
        records.append(fuse_resolutions(patch, r_obs, r_kernel, r_spacing, obs_info))
        visibility.append(dict(obs_info.get("visibility_states", {})))
    return records, visibility


def build_physical_resolution_field(
    scene: GaussianScene,
    training_views: list[PinholeView],
    cfg: PRFConfig | None = None,
    *,
    source_ply: str = "",
    source_transforms: str = "",
    occlusion_maps: OcclusionMaps | None = None,
    visibility_model: str = "frustum",
) -> PhysicalResolutionField:
    cfg = cfg or PRFConfig()
    assert_metric(scene, cfg)
    patches = build_valid_patches(scene, cfg)
    records, visibility = evaluate_patches(
        patches, scene, training_views, cfg, occlusion_maps=occlusion_maps
    )
    return PhysicalResolutionField(
        records=records,
        config=asdict(cfg),
        source_ply=source_ply,
        source_transforms=source_transforms,
        visibility_model="gs_expected_depth" if occlusion_maps is not None else visibility_model,
        visibility_by_patch=visibility,
    )


def recompute_observation_from_archive(
    archive: dict,
    scene: GaussianScene,
    training_views: list[PinholeView],
    cfg: PRFConfig | None = None,
    *,
    occlusion_maps: OcclusionMaps | None = None,
    source_ply: str = "",
    source_transforms: str = "",
) -> PhysicalResolutionField:
    """Reuse frozen patch geometry and kernel/spacing; recompute R_obs only."""
    cfg = cfg or PRFConfig()
    assert_metric(scene, cfg)
    patches = patches_from_archive(archive)
    records, visibility = evaluate_patches(
        patches,
        scene,
        training_views,
        cfg,
        occlusion_maps=occlusion_maps,
        kernel_values=list(archive["R_kernel"]),
        spacing_values=list(archive["R_spacing"]),
    )
    return PhysicalResolutionField(
        records=records,
        config=asdict(cfg),
        source_ply=source_ply,
        source_transforms=source_transforms,
        visibility_model="gs_expected_depth" if occlusion_maps is not None else "frustum",
        visibility_by_patch=visibility,
    )


def compute_from_paths(
    ply_path: Path | str,
    transforms_path: Path | str,
    cfg: PRFConfig | None = None,
    *,
    opengl_c2w: bool = False,
    occlusion_maps: OcclusionMaps | None = None,
    visibility_model: str = "frustum",
) -> PhysicalResolutionField:
    cfg = cfg or PRFConfig()
    scene = load_gaussians_ply(ply_path)
    views = load_training_views(transforms_path, opengl_c2w=opengl_c2w)
    return build_physical_resolution_field(
        scene,
        views,
        cfg,
        source_ply=str(canonical_path(ply_path)),
        source_transforms=str(canonical_path(transforms_path)),
        occlusion_maps=occlusion_maps,
        visibility_model=visibility_model,
    )


def scale_view_intrinsics(view: PinholeView, scale: float) -> PinholeView:
    """Scale image size and focal length together. Camera rays stay the same."""
    return replace(
        view,
        width=max(int(round(view.width * scale)), 1),
        height=max(int(round(view.height * scale)), 1),
        fx=float(view.fx) * scale,
        fy=float(view.fy) * scale,
        cx=float(view.cx) * scale,
        cy=float(view.cy) * scale,
    )
