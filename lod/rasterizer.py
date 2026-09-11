"""Bind GaussianZoom's RaDe-GS rasterizer without touching Skyfall's ``diff_gauss``."""

from __future__ import annotations

import math
from typing import Any

import torch

from .path import add_gz_src
from utils.sh_utils import eval_sh

REQUIRED_SETTINGS = (
    "image_height",
    "image_width",
    "tanfovx",
    "tanfovy",
    "kernel_size",
    "bg",
    "scale_modifier",
    "viewmatrix",
    "projmatrix",
    "sh_degree",
    "campos",
    "prefiltered",
    "require_depth",
    "debug",
)


def require_rade_gs():
    """Import official ``diff_gaussian_rasterization`` with the RaDe-GS settings."""

    try:
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    except ImportError as exc:
        raise ImportError(
            "RaDe-GS diff_gaussian_rasterization is not installed. "
            "Run scripts/setup_rade_gs.sh in the skyfall-gs environment."
        ) from exc
    fields = getattr(GaussianRasterizationSettings, "_fields", ())
    missing = [name for name in ("require_depth", "kernel_size", "campos") if name not in fields]
    if missing:
        raise ImportError(
            "Installed diff_gaussian_rasterization is not the RaDe-GS API "
            f"(missing {missing}; fields={fields})."
        )
    return GaussianRasterizationSettings, GaussianRasterizer


def appearance_colors_precomp(camera, gaussians, appearance_embedding=None) -> torch.Tensor:
    """Same frozen Skyfall appearance path as ``gaussian_renderer.render``."""

    embedding = appearance_embedding
    if embedding is None and gaussians.appearance_enabled:
        try:
            embedding = gaussians.appearance_embeddings[camera.uid]
        except Exception:
            embedding = torch.mean(gaussians.appearance_embeddings, dim=0)
    if gaussians.appearance_enabled and embedding is not None:
        embedding_expanded = embedding[None].repeat(int(gaussians.get_xyz.shape[0]), 1)
        colors_toned = gaussians.appearance_mlp(
            gaussians._embeddings, embedding_expanded, gaussians.get_features
        ).clamp_max(1.0)
        shdim = (gaussians.max_sh_degree + 1) ** 2
        colors_toned = colors_toned.view(-1, shdim, 3).transpose(1, 2).contiguous().clamp_max(1.0)
        dir_pp = gaussians.get_xyz - camera.camera_center.repeat(gaussians.get_features.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        colors_toned = eval_sh(gaussians.active_sh_degree, colors_toned, dir_pp_normalized)
        return torch.clamp_min(colors_toned + 0.5, 0.0)
    shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
    dir_pp = gaussians.get_xyz - camera.camera_center.repeat(gaussians.get_features.shape[0], 1)
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    return torch.clamp_min(eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized) + 0.5, 0.0)


def render_skyfall_rade(
    camera,
    gaussians,
    *,
    background: torch.Tensor,
    kernel_size: float,
    appearance_embedding=None,
    scaling_modifier: float = 1.0,
    require_depth: bool = True,
    debug: bool = False,
) -> dict[str, torch.Tensor]:
    """Rasterize a Skyfall model with RaDe-GS, keeping appearance as ``colors_precomp``.

    Uses Skyfall's view/projection matrices and stored ``camera_center``. Does not
    go through GaussianZoom's rebuilt pinhole projection (half-pixel / zfar=1e4).
    """

    settings_cls, rasterizer_cls = require_rade_gs()
    screenspace = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=True) + 0
    try:
        screenspace.retain_grad()
    except Exception:
        pass
    colors = appearance_colors_precomp(camera, gaussians, appearance_embedding)
    settings = settings_cls(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx * 0.5),
        tanfovy=math.tan(camera.FoVy * 0.5),
        kernel_size=float(kernel_size),
        bg=background,
        scale_modifier=float(scaling_modifier),
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        require_depth=require_depth,
        debug=debug,
    )
    image, radii, depth, median, alpha, normal = rasterizer_cls(settings)(
        means3D=gaussians.get_xyz,
        means2D=screenspace,
        opacities=gaussians.get_opacity_with_3D_filter.float(),
        colors_precomp=colors,
        scales=gaussians.get_scaling_with_3D_filter.float(),
        rotations=gaussians.get_rotation,
    )
    return {
        "render": image,
        "render_depth": depth,
        "render_median_depth": median,
        "render_norm": normal,
        "render_alpha": alpha,
        "radii": radii,
        "viewspace_points": screenspace,
        "visibility_filter": radii > 0,
    }


def render_lod(model, camera, *, gz_root: str | None = None, **kwargs) -> dict[str, Any]:
    """Render through GaussianZoom's RaDe-GS path (``lod=False`` still applies filters)."""

    add_gz_src(gz_root)
    require_rade_gs()
    from gaussianzoom_lod.renderer import render

    return render(model, camera, **kwargs)
