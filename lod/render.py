"""Appearance-compatible L0+L1 render path used for LoD training.

Merges layers, applies LoD opacity weights, rasterizes once with RaDe-GS, and
evaluates Skyfall's frozen appearance MLP without detaching SH or geometry.
Camera matrices and ``camera_center`` come from the Skyfall camera, not from
GaussianZoom's rebuilt pinhole (half-pixel / zfar=1e4).
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .path import add_gz_src
from .rasterizer import require_rade_gs
from utils.sh_utils import eval_sh


def appearance_colors_lod(
    xyz: torch.Tensor,
    sh: torch.Tensor,
    gaussian_embeddings: torch.Tensor,
    appearance,
    camera_center: torch.Tensor,
    image_embedding: torch.Tensor | None,
    active_sh_degree: int,
    frozen_colors: torch.Tensor | None = None,
) -> torch.Tensor:
    """Frozen MLP; cache only completed prefixes, never detach active SH/xyz."""

    if frozen_colors is not None:
        count = int(frozen_colors.shape[0])
        if frozen_colors.shape != (count, 3) or count > xyz.shape[0] or frozen_colors.requires_grad:
            raise ValueError("Frozen colors must be a detached RGB prefix")
        if count == xyz.shape[0]:
            return frozen_colors
        active = appearance_colors_lod(
            xyz[count:], sh[count:], gaussian_embeddings[count:], appearance,
            camera_center, image_embedding, active_sh_degree,
        )
        return torch.cat((frozen_colors, active), dim=0)

    if appearance.enabled and appearance.mlp is not None and image_embedding is not None:
        aemb = image_embedding.reshape(1, -1).expand(int(xyz.shape[0]), -1)
        colors_toned = appearance.mlp(gaussian_embeddings, aemb, sh).clamp_max(1.0)
        shdim = int(sh.shape[1])
        colors_toned = colors_toned.view(-1, shdim, 3).transpose(1, 2).contiguous().clamp_max(1.0)
        dir_pp = xyz - camera_center.reshape(1, 3).to(device=xyz.device, dtype=xyz.dtype)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True).clamp_min(1e-8)
        colors_toned = eval_sh(int(active_sh_degree), colors_toned, dir_pp_normalized)
        return torch.clamp_min(colors_toned + 0.5, 0.0)
    shs_view = sh.transpose(1, 2).reshape(-1, 3, int(sh.shape[1]))
    dir_pp = xyz - camera_center.reshape(1, 3).to(device=xyz.device, dtype=xyz.dtype)
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return torch.clamp_min(eval_sh(int(active_sh_degree), shs_view, dir_pp_normalized) + 0.5, 0.0)


def _lod_opacity_weights(model, data, camera, lod: bool) -> torch.Tensor:
    add_gz_src()
    from gaussianzoom_lod.renderer import interval_weights, lod_weight, topology_weight

    if not lod:
        return torch.ones_like(data["opacity"])
    center = camera.camera_center.to(device=data["xyz"].device, dtype=data["xyz"].dtype)
    distance = (data["xyz"] - center).norm(dim=1, keepdim=True)
    psi_current = distance / float(camera.focal_x)
    if model.topology_mode == "legacy_flat":
        return lod_weight(psi_current, data["psi_ref"], model.step_scale)
    if model.weight_policy == "legacy_tent":
        return topology_weight(
            psi_current, data["psi_ref"], model.step_scale, data["has_parent"], data["has_child"]
        )
    model.require_stage_records()
    return interval_weights(
        psi_current,
        data["psi_ref"],
        [stage["scale"] for stage in model.stage_records],
        data,
        data["layer_slices"],
    )


def render_lod_appearance(
    bundle,
    camera,
    *,
    background: torch.Tensor,
    kernel_size: float,
    appearance_embedding=None,
    lod: bool = True,
    max_level: int | None = None,
    compact: bool = True,
    require_depth: bool = True,
    debug: bool = False,
    scaling_modifier: float = 1.0,
    gz_root: str | None = None,
    frozen_colors: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Training rasterizer: merged L0+L1, LoD opacity, frozen appearance, Skyfall pose."""

    add_gz_src(gz_root)
    settings_cls, rasterizer_cls = require_rade_gs()
    data = bundle.lod.tensors(max_level=max_level)
    xyz = data["xyz"]
    n_levels = len(data["layer_slices"])
    weights = _lod_opacity_weights(bundle.lod, data, camera, lod)
    effective_opacity = data["opacity"] * weights
    image_embedding = appearance_embedding
    if image_embedding is None and bundle.appearance.enabled:
        image_embedding = bundle.appearance.embedding_for_uid(int(camera.uid), is_train_view=True)
    if frozen_colors is not None:
        count = 0
        for layer in bundle.lod.layers[:n_levels]:
            if count == len(frozen_colors):
                break
            if not layer.frozen or any(parameter.requires_grad for parameter in layer.parameters()):
                raise ValueError("Color caching requires a completely frozen layer prefix")
            count += len(layer.xyz)
        if count != len(frozen_colors):
            raise ValueError("Frozen-color prefix must end on a layer boundary")
        if bundle.appearance.enabled and (
            any(parameter.requires_grad for parameter in bundle.appearance.mlp.parameters())
            or (image_embedding is not None and image_embedding.requires_grad)
        ):
            raise ValueError("Color caching requires frozen appearance")
    if bundle.appearance.enabled:
        gaussian_embeddings = bundle.appearance.embeddings_for_levels(n_levels)
        if int(gaussian_embeddings.shape[0]) != int(xyz.shape[0]):
            raise ValueError(
                f"appearance embeddings {tuple(gaussian_embeddings.shape)} do not match "
                f"merged Gaussians {tuple(xyz.shape)}"
            )
        colors = appearance_colors_lod(
            xyz,
            data["sh"],
            gaussian_embeddings,
            bundle.appearance,
            camera.camera_center,
            image_embedding,
            bundle.sh_degree,
            frozen_colors=frozen_colors,
        )
    else:
        colors = appearance_colors_lod(
            xyz, data["sh"], xyz.new_zeros((xyz.shape[0], 1)), bundle.appearance,
            camera.camera_center, None, bundle.sh_degree, frozen_colors=frozen_colors,
        )

    screen = torch.zeros_like(xyz, requires_grad=True)
    if torch.is_grad_enabled():
        screen.retain_grad()
    indices = torch.nonzero(weights[:, 0] > 0, as_tuple=False)[:, 0] if lod and compact else None
    if indices is not None and indices.numel() == 0:
        zero = (screen.sum() + xyz.sum() + effective_opacity.sum() + colors.sum()) * 0
        scalar = torch.zeros((1, int(camera.image_height), int(camera.image_width)), device=xyz.device) + zero
        image = scalar.expand(3, -1, -1).contiguous()
        return {
            "render": image, "render_depth": scalar, "render_median_depth": scalar,
            "render_norm": image, "render_alpha": scalar,
            "radii": torch.zeros(len(xyz), device=xyz.device, dtype=torch.int32),
            "viewspace_points": screen, "visibility_filter": torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device),
            "weights": weights, "layer_slices": data["layer_slices"],
        }

    def selected(tensor):
        return tensor if indices is None else tensor.index_select(0, indices)

    settings = settings_cls(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=math.tan(float(camera.FoVx) * 0.5),
        tanfovy=math.tan(float(camera.FoVy) * 0.5),
        kernel_size=float(kernel_size),
        bg=background,
        scale_modifier=float(scaling_modifier),
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=int(bundle.sh_degree),
        campos=camera.camera_center,
        prefiltered=False,
        require_depth=require_depth,
        debug=debug,
    )
    image, radii, depth, median, alpha, normal = rasterizer_cls(settings)(
        means3D=selected(xyz),
        means2D=selected(screen),
        opacities=selected(effective_opacity),
        colors_precomp=selected(colors),
        scales=selected(data["scales"]),
        rotations=selected(data["rotations"]),
    )
    if indices is not None:
        full_radii = torch.zeros(len(xyz), device=xyz.device, dtype=radii.dtype)
        radii = full_radii.scatter_(0, indices, radii)
    return {
        "render": image,
        "render_depth": depth,
        "render_median_depth": median,
        "render_norm": normal,
        "render_alpha": alpha,
        "radii": radii,
        "viewspace_points": screen,
        "visibility_filter": radii > 0,
        "weights": weights,
        "layer_slices": data["layer_slices"],
        "colors_precomp": colors,
    }
