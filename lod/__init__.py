"""Skyfall adapter for GaussianZoom LoD. Does not replace train_zoom_gen or the MVP.

The LoD implementation lives in GaussianZoom_distill. This package only converts
Skyfall checkpoints, cameras, appearance, and RaDe-GS rendering into that container.
"""

from .camera import (
    CameraWithStoredCenter,
    ndc_principal_to_pixel,
    pixel_principal_to_ndc,
    projection_offset_delta_ndc,
    skyfall_camera_to_lod,
    skyfall_w2c,
    zoom_stage_camera,
)
from .depth import (
    DepthMap,
    depth_from_rade,
    depth_from_skyfall,
    pair_depth_metrics,
    rade_camera_z,
    rln_principal_from_skyfall_camera,
    skyfall_expected_z,
    skyfall_ray_distance,
)
from .importer import (
    FrozenAppearance,
    SkyfallL0,
    add_detail_level,
    densify_with_appearance,
    import_skyfall_l0,
    load_lod_onto_bundle,
    save_bundle,
    sync_layer_embeddings,
)
from .path import DEFAULT_GZ_ROOT, add_gz_src
from .rasterizer import (
    appearance_colors_precomp,
    render_lod,
    render_skyfall_rade,
    require_rade_gs,
)
from .render import appearance_colors_lod, render_lod_appearance

__all__ = [
    "CameraWithStoredCenter",
    "DepthMap",
    "DEFAULT_GZ_ROOT",
    "FrozenAppearance",
    "SkyfallL0",
    "add_detail_level",
    "add_gz_src",
    "appearance_colors_lod",
    "appearance_colors_precomp",
    "depth_from_rade",
    "depth_from_skyfall",
    "densify_with_appearance",
    "import_skyfall_l0",
    "load_lod_onto_bundle",
    "ndc_principal_to_pixel",
    "pair_depth_metrics",
    "pixel_principal_to_ndc",
    "projection_offset_delta_ndc",
    "rade_camera_z",
    "rln_principal_from_skyfall_camera",
    "render_lod",
    "render_lod_appearance",
    "render_skyfall_rade",
    "require_rade_gs",
    "save_bundle",
    "skyfall_camera_to_lod",
    "skyfall_expected_z",
    "skyfall_ray_distance",
    "skyfall_w2c",
    "sync_layer_embeddings",
    "zoom_stage_camera",
]
