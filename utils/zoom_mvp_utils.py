"""Utilities for progressive zoom-refine MVP."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from scene.gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid
from utils.loss_utils import l1_loss
from utils.sh_utils import RGB2SH


PARKING_CROP_BOX = (727, 808, 1239, 1320)


def freeze_appearance_for_zoom(gaussians: GaussianModel) -> None:
    if not gaussians.appearance_enabled:
        return
    gaussians.appearance_embeddings.requires_grad_(False)
    if gaussians._embeddings is not None:
        gaussians._embeddings.requires_grad_(False)
    if gaussians.appearance_mlp is not None:
        for param in gaussians.appearance_mlp.parameters():
            param.requires_grad_(False)


def freeze_geometry_for_zoom(gaussians: GaussianModel) -> None:
    """Freeze geometry/opacity/appearance; keep SH features trainable."""
    for name in ("xyz", "scaling", "rotation", "opacity"):
        for group in gaussians.optimizer.param_groups:
            if group["name"] == name:
                group["params"][0].requires_grad_(False)
    freeze_appearance_for_zoom(gaussians)


def configure_optimizer_for_detail(gaussians: GaussianModel, freeze_scale: bool = False) -> None:
    """Train new-point SH / opacity, and scale unless frozen. xyz and rotation stay in the graph but lr=0."""
    frozen = {"xyz", "rotation", "appearance_embeddings", "embeddings", "appearance_mlp"}
    if freeze_scale:
        frozen.add("scaling")
    for group in gaussians.optimizer.param_groups:
        if group["name"] in frozen:
            group["lr"] = 0.0
    freeze_appearance_for_zoom(gaussians)


def apply_detail_grad_mask(
    gaussians: GaussianModel,
    n_coarse: int,
    freeze_scale: bool = False,
) -> None:
    """Zero grads on frozen coarse Gaussians and on detail xyz/rotation/embeddings."""

    def _mask(param: Optional[torch.Tensor], freeze_tail: bool) -> None:
        if param is None or param.grad is None:
            return
        param.grad[:n_coarse].zero_()
        if freeze_tail:
            param.grad[n_coarse:].zero_()

    _mask(gaussians._xyz, freeze_tail=True)
    _mask(gaussians._rotation, freeze_tail=True)
    _mask(gaussians._scaling, freeze_tail=freeze_scale)
    _mask(gaussians._opacity, freeze_tail=False)
    _mask(gaussians._features_dc, freeze_tail=False)
    _mask(gaussians._features_rest, freeze_tail=False)
    if gaussians.appearance_enabled:
        _mask(gaussians._embeddings, freeze_tail=True)


@dataclass
class GeometrySnapshot:
    xyz: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    opacity: torch.Tensor

    @classmethod
    def from_gaussians(cls, gaussians: GaussianModel) -> "GeometrySnapshot":
        return cls(
            xyz=gaussians._xyz.detach().clone(),
            scaling=gaussians._scaling.detach().clone(),
            rotation=gaussians._rotation.detach().clone(),
            opacity=gaussians._opacity.detach().clone(),
        )

    def assert_unchanged(self, gaussians: GaussianModel, rtol: float = 0.0, atol: float = 0.0) -> None:
        checks = {
            "_xyz": self.xyz,
            "_scaling": self.scaling,
            "_rotation": self.rotation,
            "_opacity": self.opacity,
        }
        for name, expected in checks.items():
            current = getattr(gaussians, name).detach()
            if not torch.equal(current, expected):
                max_diff = (current - expected).abs().max().item()
                raise AssertionError(
                    f"Geometry parameter {name} changed during appearance-only training (max diff={max_diff})."
                )


@dataclass
class DetailLayerSnapshot:
    n_coarse: int
    n_total: int
    coarse_xyz: torch.Tensor
    coarse_scaling: torch.Tensor
    coarse_rotation: torch.Tensor
    coarse_opacity: torch.Tensor
    coarse_f_dc: torch.Tensor
    coarse_f_rest: torch.Tensor
    detail_xyz: torch.Tensor
    detail_rotation: torch.Tensor
    detail_scaling: torch.Tensor
    detail_init_scale: torch.Tensor
    detail_init_opacity: torch.Tensor
    freeze_detail_scale: bool = False

    @classmethod
    def from_gaussians(
        cls,
        gaussians: GaussianModel,
        n_coarse: int,
        freeze_detail_scale: bool = False,
    ) -> "DetailLayerSnapshot":
        return cls(
            n_coarse=n_coarse,
            n_total=int(gaussians._xyz.shape[0]),
            coarse_xyz=gaussians._xyz[:n_coarse].detach().clone(),
            coarse_scaling=gaussians._scaling[:n_coarse].detach().clone(),
            coarse_rotation=gaussians._rotation[:n_coarse].detach().clone(),
            coarse_opacity=gaussians._opacity[:n_coarse].detach().clone(),
            coarse_f_dc=gaussians._features_dc[:n_coarse].detach().clone(),
            coarse_f_rest=gaussians._features_rest[:n_coarse].detach().clone(),
            detail_xyz=gaussians._xyz[n_coarse:].detach().clone(),
            detail_rotation=gaussians._rotation[n_coarse:].detach().clone(),
            detail_scaling=gaussians._scaling[n_coarse:].detach().clone(),
            detail_init_scale=gaussians.get_scaling[n_coarse:].detach().clone(),
            detail_init_opacity=gaussians.get_opacity[n_coarse:].detach().clone(),
            freeze_detail_scale=freeze_detail_scale,
        )

    def assert_invariants(self, gaussians: GaussianModel) -> None:
        if int(gaussians._xyz.shape[0]) != self.n_total:
            raise AssertionError(
                f"Point count changed: {self.n_total} -> {int(gaussians._xyz.shape[0])}. "
                "densify/prune should be off for this experiment."
            )
        checks = {
            "coarse _xyz": (gaussians._xyz[: self.n_coarse], self.coarse_xyz),
            "coarse _scaling": (gaussians._scaling[: self.n_coarse], self.coarse_scaling),
            "coarse _rotation": (gaussians._rotation[: self.n_coarse], self.coarse_rotation),
            "coarse _opacity": (gaussians._opacity[: self.n_coarse], self.coarse_opacity),
            "coarse _features_dc": (gaussians._features_dc[: self.n_coarse], self.coarse_f_dc),
            "coarse _features_rest": (gaussians._features_rest[: self.n_coarse], self.coarse_f_rest),
            "detail _xyz": (gaussians._xyz[self.n_coarse :], self.detail_xyz),
            "detail _rotation": (gaussians._rotation[self.n_coarse :], self.detail_rotation),
        }
        if self.freeze_detail_scale:
            checks["detail _scaling"] = (
                gaussians._scaling[self.n_coarse :],
                self.detail_scaling,
            )
        for name, (current, expected) in checks.items():
            if not torch.equal(current.detach(), expected):
                max_diff = (current.detach() - expected).abs().max().item()
                raise AssertionError(f"{name} changed during detail-layer training (max diff={max_diff}).")


def select_appearance_embedding(
    gaussians: GaussianModel,
    base_cam_uid: int,
    is_train_view: bool,
) -> torch.Tensor:
    if not gaussians.appearance_enabled:
        return None

    if is_train_view:
        if base_cam_uid >= gaussians.appearance_embeddings.shape[0]:
            raise IndexError(
                f"Train view uid {base_cam_uid} exceeds appearance embedding count "
                f"{gaussians.appearance_embeddings.shape[0]}."
            )
        embedding = gaussians.appearance_embeddings[base_cam_uid]
    else:
        embedding = torch.mean(gaussians.appearance_embeddings, dim=0)
    return embedding.detach()


def embedding_for_train_camera(gaussians: GaussianModel, uid: int) -> Optional[torch.Tensor]:
    """A train camera's own frozen embedding.

    Supervising an original train view with a *different* view's embedding would ask the
    SH features to absorb an illumination mismatch, so each original view keeps its own.
    """
    if not gaussians.appearance_enabled:
        return None
    return gaussians.appearance_embeddings[uid].detach()


def save_tensor_image(image: torch.Tensor, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    arr = (image.detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(out_path)


def save_residual_image(a: torch.Tensor, b: torch.Tensor, out_path: str, gain: float = 6.0) -> None:
    save_tensor_image((a - b).abs().clamp(0.0, 1.0) * gain, out_path)


def crop_chw(image: torch.Tensor, crop_box: Sequence[int]) -> torch.Tensor:
    x0, y0, x1, y1 = crop_box
    return image[:, y0:y1, x0:x1]


def image_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(l1_loss(a, b).item())


def image_hf_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    """L1 on a 3x3 Laplacian high-pass; tracks edge/texture mismatch."""
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=a.device,
        dtype=a.dtype,
    ).view(1, 1, 3, 3).expand(a.shape[0], 1, 3, 3)
    hp_a = torch.nn.functional.conv2d(a.unsqueeze(0), kernel, padding=1, groups=a.shape[0])
    hp_b = torch.nn.functional.conv2d(b.unsqueeze(0), kernel, padding=1, groups=b.shape[0])
    return float(torch.mean(torch.abs(hp_a - hp_b)).item())


def parse_crop_box(text: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if text is None:
        return None
    stripped = text.strip()
    if stripped.lower() in ("", "none", "off"):
        return None
    parts = [int(x.strip()) for x in stripped.split(",")]
    if len(parts) != 4:
        raise ValueError(f"crop_box must be x0,y0,x1,y1, got {text!r}.")
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"crop_box must have x1>x0 and y1>y0, got {parts}.")
    return x0, y0, x1, y1


def compute_post_train_metrics(
    render_before: torch.Tensor,
    render_after: torch.Tensor,
    refined: torch.Tensor,
    crop_box: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    dist_before = image_l1(render_before, refined)
    dist_after = image_l1(render_after, refined)
    hf_before = image_hf_l1(render_before, refined)
    hf_after = image_hf_l1(render_after, refined)
    metrics: Dict[str, float] = {
        "l1_before_to_refined": dist_before,
        "l1_after_to_refined": dist_after,
        "l1_moved_toward_refined": bool(dist_after < dist_before),
        "hf_l1_before_to_refined": hf_before,
        "hf_l1_after_to_refined": hf_after,
        "hf_l1_moved_toward_refined": bool(hf_after < hf_before),
    }
    if crop_box is not None:
        before_c = crop_chw(render_before, crop_box)
        after_c = crop_chw(render_after, crop_box)
        refined_c = crop_chw(refined, crop_box)
        crop_l1_before = image_l1(before_c, refined_c)
        crop_l1_after = image_l1(after_c, refined_c)
        crop_hf_before = image_hf_l1(before_c, refined_c)
        crop_hf_after = image_hf_l1(after_c, refined_c)
        metrics.update(
            {
                "crop_box": list(crop_box),
                "crop_l1_before_to_refined": crop_l1_before,
                "crop_l1_after_to_refined": crop_l1_after,
                "crop_l1_moved_toward_refined": bool(crop_l1_after < crop_l1_before),
                "crop_hf_l1_before_to_refined": crop_hf_before,
                "crop_hf_l1_after_to_refined": crop_hf_after,
                "crop_hf_l1_moved_toward_refined": bool(crop_hf_after < crop_hf_before),
            }
        )
    return metrics


def assert_post_train_checks(
    snapshot: GeometrySnapshot,
    gaussians: GaussianModel,
    metrics: Dict[str, float],
    detail_snapshot: Optional[DetailLayerSnapshot] = None,
) -> None:
    """Run after artifacts are on disk, so a failure still leaves images to inspect."""
    if detail_snapshot is not None:
        detail_snapshot.assert_invariants(gaussians)
    else:
        snapshot.assert_unchanged(gaussians)
    if not metrics["l1_moved_toward_refined"]:
        raise AssertionError(
            f"render_after is not closer to refined than render_before "
            f"(before={metrics['l1_before_to_refined']:.6f}, "
            f"after={metrics['l1_after_to_refined']:.6f})."
        )


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return _json_ready(obj.item())
        return [_json_ready(v) for v in obj.detach().cpu().flatten().tolist()]
    if isinstance(obj, np.ndarray):
        return _json_ready(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, indent=2, ensure_ascii=False)


def squeeze_hw(tensor: torch.Tensor) -> torch.Tensor:
    while tensor.dim() > 2:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 2:
        raise ValueError(f"Expected a HxW map after squeeze, got shape {tuple(tensor.shape)}.")
    return tensor


def _percentiles(values: torch.Tensor, qs: Sequence[float] = (0, 10, 25, 50, 75, 90, 99, 100)) -> Dict[str, float]:
    flat = values.detach().float().reshape(-1)
    out: Dict[str, float] = {"count": int(flat.numel())}
    if flat.numel() == 0:
        out["mean"] = float("nan")
        for q in qs:
            out[f"p{q:g}"] = float("nan")
        return out
    q_tensor = torch.tensor([q / 100.0 for q in qs], device=flat.device, dtype=flat.dtype)
    pct = torch.quantile(flat, q_tensor)
    out["mean"] = float(flat.mean().item())
    for q, v in zip(qs, pct):
        out[f"p{q:g}"] = float(v.item())
    return out


def camera_pixel_origin(camera) -> Tuple[float, float]:
    cx_ori = camera.cx / 2.0 * camera.image_width + camera.image_width / 2.0
    cy_ori = camera.cy / 2.0 * camera.image_height + camera.image_height / 2.0
    return float(cx_ori), float(cy_ori)


def world_to_camera_xyz(xyz: torch.Tensor, camera) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match ``GaussianModel.compute_3D_filter``: xyz_cam = xyz @ R + T."""
    R = torch.as_tensor(camera.R, device=xyz.device, dtype=xyz.dtype)
    T = torch.as_tensor(camera.T, device=xyz.device, dtype=xyz.dtype).reshape(3)
    return xyz @ R + T[None, :], R, T


def camera_to_world_xyz(xyz_cam: torch.Tensor, R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    return (xyz_cam - T[None, :]) @ torch.linalg.inv(R)


def project_points(xyz: torch.Tensor, camera) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xyz_cam, _, _ = world_to_camera_xyz(xyz, camera)
    z = xyz_cam[:, 2]
    z_safe = z.clamp(min=1e-6)
    cx_ori, cy_ori = camera_pixel_origin(camera)
    u = xyz_cam[:, 0] / z_safe * camera.focal_x + cx_ori
    v = xyz_cam[:, 1] / z_safe * camera.focal_y + cy_ori
    in_screen = (
        (z > 0.2)
        & (u >= 0)
        & (u < camera.image_width)
        & (v >= 0)
        & (v < camera.image_height)
    )
    return u, v, z, in_screen


def expected_render_depth(accum_depth: torch.Tensor, alpha: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Convert rasterizer accumulated depth D=Σ Tαz into expected surface depth D/A."""
    return accum_depth / alpha.clamp(min=eps)


def float32_ulp(values: torch.Tensor) -> torch.Tensor:
    """Unit in the last place of the float32 representation of ``values``."""
    v64 = values.to(torch.float64).abs().clamp(min=1e-12)
    _, exp = torch.frexp(v64)
    return torch.ldexp(torch.ones_like(v64), exp - 24)


def depth_continuity_mask(
    depth: torch.Tensor,
    camera,
    max_jump_px: float,
) -> torch.Tensor:
    """Reject pixels whose 3x3 depth range exceeds ``max_jump_px`` world-pixel sizes."""
    pixel_world = depth * 0.5 * (1.0 / camera.focal_x + 1.0 / camera.focal_y)
    padded = torch.nn.functional.pad(depth[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    h, w = depth.shape
    diffs = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neigh = padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
            diffs.append((neigh - depth).abs())
    max_diff = torch.stack(diffs, dim=0).max(dim=0).values
    return max_diff <= (max_jump_px * pixel_world.clamp(min=1e-8))


@torch.no_grad()
def verify_seed_geometry(
    xyz: torch.Tensor,
    camera,
    pixel_x: torch.Tensor,
    pixel_y: torch.Tensor,
    expected_z: torch.Tensor,
    intended_offset: torch.Tensor,
) -> Dict[str, float]:
    """Reproject seeded points and measure pixel / depth error after float32 storage."""
    u, v, z, _ = project_points(xyz, camera)
    du = u - (pixel_x.to(u.dtype) + 0.5)
    dv = v - (pixel_y.to(v.dtype) + 0.5)
    pixel_err = torch.sqrt(du.square() + dv.square())
    actual_offset = expected_z.to(z.dtype) - z
    intended = intended_offset.to(z.dtype)
    return {
        "pixel_err_mean": float(pixel_err.mean().item()),
        "pixel_err_p50": float(pixel_err.median().item()),
        "pixel_err_p90": float(torch.quantile(pixel_err, 0.9).item()),
        "pixel_err_p99": float(torch.quantile(pixel_err, 0.99).item()),
        "frac_pixel_err_gt_0.5": float((pixel_err > 0.5).float().mean().item()),
        "frac_pixel_err_gt_1.0": float((pixel_err > 1.0).float().mean().item()),
        "intended_front_offset_mean": float(intended.mean().item()),
        "actual_front_offset_mean": float(actual_offset.mean().item()),
        "actual_front_offset_p50": float(actual_offset.median().item()),
        "frac_in_front": float((actual_offset > 0).float().mean().item()),
        "frac_offset_lt_half_intended": float(
            (actual_offset < 0.5 * intended.clamp(min=1e-12)).float().mean().item()
        ),
        "mean_expected_z": float(expected_z.mean().item()),
        "mean_reprojected_z": float(z.mean().item()),
    }


@torch.no_grad()
def clamp_detail_scale(
    gaussians: GaussianModel,
    n_coarse: int,
    camera,
    min_px: float,
    max_px: float,
) -> None:
    """Keep detail Gaussian world scale inside ``[min_px, max_px]`` screen pixels."""
    xyz = gaussians.get_xyz[n_coarse:]
    _, _, z, _ = project_points(xyz, camera)
    pixel_world = (z * 0.5 * (1.0 / camera.focal_x + 1.0 / camera.focal_y)).clamp(min=1e-8)
    lo = (min_px * pixel_world).unsqueeze(1)
    hi = (max_px * pixel_world).unsqueeze(1)
    scales = gaussians.get_scaling[n_coarse:].clamp(min=lo, max=hi)
    gaussians._scaling.data[n_coarse:] = gaussians.scaling_inverse_activation(scales)


@torch.no_grad()
def diagnose_projection_and_filter(
    gaussians: GaussianModel,
    camera,
    radii: Optional[torch.Tensor] = None,
    subset: Optional[torch.Tensor] = None,
    label: str = "all",
) -> Dict:
    """Screen-space scale vs filter_3D floor for Gaussians visible in ``camera``."""
    xyz = gaussians.get_xyz
    if subset is None:
        subset = torch.ones((xyz.shape[0],), device=xyz.device, dtype=torch.bool)
    u, v, z, in_screen = project_points(xyz, camera)
    visible = subset & in_screen
    if radii is not None:
        visible = visible & (radii.reshape(-1) > 0)

    n_vis = int(visible.sum().item())
    scales = gaussians.get_scaling
    max_scale = scales.max(dim=1).values
    filter_3d = gaussians.filter_3D.reshape(-1)
    scales_eff = gaussians.get_scaling_with_3D_filter.max(dim=1).values
    screen_px = max_scale * camera.focal_x / z.clamp(min=1e-6)
    screen_px_eff = scales_eff * camera.focal_x / z.clamp(min=1e-6)
    filter_screen_px = filter_3d * camera.focal_x / z.clamp(min=1e-6)
    filter_over_scale = filter_3d / max_scale.clamp(min=1e-12)

    vis_z = z[visible]
    pixel_world = vis_z / camera.focal_x
    theory_filter = vis_z / camera.focal_x * math.sqrt(0.2)
    # If this camera's focal is the one used by compute_3D_filter, 1px Gaussians
    # become sqrt(1 + 0.2) screen pixels after the 3D filter.
    one_px_eff_screen = torch.sqrt(pixel_world.square() + filter_3d[visible].square()) * camera.focal_x / vis_z.clamp(
        min=1e-6
    )

    diag = {
        "label": label,
        "n_gaussians": int(xyz.shape[0]),
        "n_visible": n_vis,
        "camera_focal_x": float(camera.focal_x),
        "camera_focal_y": float(camera.focal_y),
        "kernel_size_note": "rasterizer kernel_size is extra 2D AA on top of filter_3D",
        "max_scale": _percentiles(max_scale[visible]),
        "filter_3d": _percentiles(filter_3d[visible]),
        "filter_over_scale": _percentiles(filter_over_scale[visible]),
        "screen_px": _percentiles(screen_px[visible]),
        "screen_px_with_filter": _percentiles(screen_px_eff[visible]),
        "filter_screen_px": _percentiles(filter_screen_px[visible]),
        "one_px_world": _percentiles(pixel_world),
        "theory_filter_if_this_focal": _percentiles(theory_filter),
        "one_px_effective_screen_px": _percentiles(one_px_eff_screen),
        "frac_filter_dominates_scale": float((filter_over_scale[visible] > 1.0).float().mean().item()) if n_vis else float("nan"),
        "frac_screen_gt_4px": float((screen_px_eff[visible] > 4.0).float().mean().item()) if n_vis else float("nan"),
        "frac_screen_gt_8px": float((screen_px_eff[visible] > 8.0).float().mean().item()) if n_vis else float("nan"),
        "new_1px_survives_filter": bool(
            n_vis > 0 and float(one_px_eff_screen.median().item()) < 2.5
        ),
    }
    return diag


def _grid_sample_high_residual(
    residual: torch.Tensor,
    valid: torch.Tensor,
    n_target: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One highest-residual valid pixel per grid cell, then top-n_target."""
    h, w = residual.shape
    masked = residual.clone()
    masked[~valid] = -1.0
    stride = max(1, int(math.sqrt((h * w) / max(n_target, 1))))
    ys_all: List[torch.Tensor] = []
    xs_all: List[torch.Tensor] = []
    scores_all: List[torch.Tensor] = []

    for _ in range(6):
        pad_h = (stride - h % stride) % stride
        pad_w = (stride - w % stride) % stride
        padded = torch.nn.functional.pad(masked, (0, pad_w, 0, pad_h), value=-1.0)
        ph, pw = padded.shape
        n_gy, n_gx = ph // stride, pw // stride
        cells = padded.view(n_gy, stride, n_gx, stride).permute(0, 2, 1, 3).contiguous().view(n_gy * n_gx, stride * stride)
        scores, local = cells.max(dim=1)
        keep = scores > 0
        if keep.any():
            gy = torch.div(torch.arange(n_gy * n_gx, device=residual.device)[keep], n_gx, rounding_mode="floor")
            gx = torch.arange(n_gy * n_gx, device=residual.device)[keep] % n_gx
            ly = torch.div(local[keep], stride, rounding_mode="floor")
            lx = local[keep] % stride
            ys = gy * stride + ly
            xs = gx * stride + lx
            in_img = (ys < h) & (xs < w)
            ys_all.append(ys[in_img])
            xs_all.append(xs[in_img])
            scores_all.append(scores[keep][in_img])
        if scores_all and int(torch.cat(ys_all).shape[0]) >= n_target:
            break
        if stride == 1:
            break
        stride = max(1, stride // 2)
        ys_all, xs_all, scores_all = [], [], []

    if not ys_all:
        return torch.empty(0, dtype=torch.long, device=residual.device), torch.empty(
            0, dtype=torch.long, device=residual.device
        )

    ys = torch.cat(ys_all)
    xs = torch.cat(xs_all)
    scores = torch.cat(scores_all)
    if ys.numel() > n_target:
        top = torch.topk(scores, n_target, largest=True).indices
        ys, xs = ys[top], xs[top]
    return ys, xs


def _voxel_dedup(
    xyz: torch.Tensor,
    scores: torch.Tensor,
    voxel_size: float,
    n_target: int,
) -> torch.Tensor:
    if xyz.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=xyz.device)
    keys = torch.round(xyz / max(voxel_size, 1e-8)).to(torch.int64)
    _, inverse = torch.unique(keys, dim=0, return_inverse=True)
    n_vox = int(inverse.max().item()) + 1
    order = torch.argsort(scores, descending=True)
    inv = inverse[order]
    positions = torch.arange(inv.shape[0], device=xyz.device)
    best_pos = torch.empty(n_vox, dtype=torch.long, device=xyz.device)
    best_pos.scatter_(0, inv.flip(0), positions.flip(0))
    kept = order[best_pos]
    if kept.numel() > n_target:
        kept_scores = scores[kept]
        kept = kept[torch.topk(kept_scores, n_target, largest=True).indices]
    return kept


@torch.no_grad()
def _copy_nearest_embeddings(
    new_xyz: torch.Tensor,
    src_xyz: torch.Tensor,
    src_emb: torch.Tensor,
    src_mask: Optional[torch.Tensor] = None,
    max_src: int = 8192,
    chunk: int = 4096,
) -> torch.Tensor:
    if src_mask is not None:
        src_xyz = src_xyz[src_mask]
        src_emb = src_emb[src_mask]
    if src_xyz.shape[0] == 0:
        return torch.zeros((new_xyz.shape[0], src_emb.shape[1]), device=new_xyz.device, dtype=src_emb.dtype)
    if src_xyz.shape[0] > max_src:
        idx = torch.randperm(src_xyz.shape[0], device=src_xyz.device)[:max_src]
        src_xyz = src_xyz[idx]
        src_emb = src_emb[idx]
    parts = []
    for start in range(0, new_xyz.shape[0], chunk):
        dist = torch.cdist(new_xyz[start : start + chunk], src_xyz)
        parts.append(src_emb[dist.argmin(dim=1)])
    return torch.cat(parts, dim=0)


@torch.no_grad()
def seed_detail_gaussians(
    gaussians: GaussianModel,
    camera,
    render_rgb: torch.Tensor,
    refined_rgb: torch.Tensor,
    depth: torch.Tensor,
    alpha: torch.Tensor,
    n_target: int,
    min_alpha: float = 0.9,
    min_depth: float = 0.2,
    init_opacity: float = 0.25,
    front_offset_px: float = 0.25,
    max_depth_jump_px: float = 2.0,
    depth_eps: float = 1e-4,
    min_front_ulps: float = 4.0,
) -> Dict[str, torch.Tensor]:
    """Backproject high-residual pixels into a small surface detail layer.

    Rasterizer ``render_depth`` is accumulated ``D = Σ Tαz``, not camera Z. Seeding
    uses expected depth ``D / A``. Front offset is applied in float64 and enlarged
    to a few float32 ULPs when the requested 0.25 px shift would round away.
    """
    accum_hw = squeeze_hw(depth)
    alpha_hw = squeeze_hw(alpha)
    expected_hw = expected_render_depth(accum_hw, alpha_hw, eps=depth_eps)
    residual = (render_rgb - refined_rgb).abs().mean(dim=0)
    valid = (
        (alpha_hw > min_alpha)
        & (expected_hw > min_depth)
        & torch.isfinite(expected_hw)
        & torch.isfinite(alpha_hw)
    )
    if max_depth_jump_px > 0:
        valid = valid & depth_continuity_mask(expected_hw, camera, max_depth_jump_px)
    if int(valid.sum().item()) == 0:
        raise RuntimeError("No valid depth/alpha pixels to seed detail Gaussians.")

    ys, xs = _grid_sample_high_residual(residual, valid, n_target=max(n_target * 2, n_target))
    if ys.numel() == 0:
        raise RuntimeError("Grid sampling found no high-residual pixels.")

    z_raw = accum_hw[ys, xs]
    z_expected = expected_hw[ys, xs]
    pixel_world = z_expected * 0.5 * (1.0 / camera.focal_x + 1.0 / camera.focal_y)
    n_before_dedup = int(ys.numel())

    z64 = z_expected.to(torch.float64)
    pixel_world64 = pixel_world.to(torch.float64)
    requested_offset = front_offset_px * pixel_world64
    ulp = float32_ulp(z64)
    min_offset = min_front_ulps * ulp
    offset64 = torch.maximum(requested_offset, min_offset)
    z_front64 = (z64 - offset64).clamp(min=float(min_depth))

    R = torch.as_tensor(camera.R, device=z64.device, dtype=torch.float64)
    T = torch.as_tensor(camera.T, device=z64.device, dtype=torch.float64).reshape(3)
    cx_ori, cy_ori = camera_pixel_origin(camera)
    u = xs.to(torch.float64) + 0.5
    v = ys.to(torch.float64) + 0.5
    x = (u - cx_ori) / camera.focal_x * z_front64
    y = (v - cy_ori) / camera.focal_y * z_front64
    xyz_cam = torch.stack([x, y, z_front64], dim=-1)
    xyz64 = camera_to_world_xyz(xyz_cam, R, T)
    xyz = xyz64.to(torch.float32)

    scores = residual[ys, xs]
    voxel = float(pixel_world.median().item()) * 1.5
    keep = _voxel_dedup(xyz, scores, voxel_size=voxel, n_target=n_target)
    xyz = xyz[keep]
    ys, xs = ys[keep], xs[keep]
    pixel_world = pixel_world[keep]
    z_raw = z_raw[keep]
    z_expected = z_expected[keep]
    offset64 = offset64[keep]
    n = xyz.shape[0]
    if n == 0:
        raise RuntimeError("Voxel dedup removed every candidate detail Gaussian.")

    colors = refined_rgb[:, ys, xs].T.clamp(0.0, 1.0)
    sh_dc = RGB2SH(colors).unsqueeze(1)  # (N, 1, 3)
    rest_dim = gaussians._features_rest.shape[1]
    sh_rest = torch.zeros((n, rest_dim, 3), device=xyz.device, dtype=gaussians._features_rest.dtype)
    scales = torch.log(pixel_world.clamp(min=1e-8)).unsqueeze(1).repeat(1, 3)
    rots = torch.zeros((n, 4), device=xyz.device, dtype=gaussians._rotation.dtype)
    rots[:, 0] = 1.0
    opacities = inverse_sigmoid(
        torch.full((n, 1), init_opacity, device=xyz.device, dtype=gaussians._opacity.dtype)
    )

    new_embeddings = None
    if gaussians.appearance_enabled and gaussians._embeddings is not None:
        _, _, _, in_screen = project_points(gaussians.get_xyz, camera)
        new_embeddings = _copy_nearest_embeddings(
            xyz, gaussians.get_xyz, gaussians._embeddings, src_mask=in_screen
        )

    seed_map = torch.zeros((3, residual.shape[0], residual.shape[1]), device=residual.device)
    seed_map[:, ys, xs] = 1.0
    geometry = verify_seed_geometry(
        xyz, camera, xs, ys, z_expected, offset64.to(torch.float32)
    )
    raw_over_expected = z_raw / z_expected.clamp(min=1e-8)

    return {
        "xyz": xyz,
        "features_dc": sh_dc,
        "features_rest": sh_rest,
        "opacity": opacities,
        "scaling": scales,
        "rotation": rots,
        "embeddings": new_embeddings,
        "pixel_y": ys,
        "pixel_x": xs,
        "pixel_world": pixel_world,
        "seed_map": seed_map,
        "n_candidates_before_dedup": n_before_dedup,
        "n_valid_seed_pixels": int(valid.sum().item()),
        "voxel_size": voxel,
        "mean_residual": float(scores[keep].mean().item()),
        "mean_depth": float(z_expected.mean().item()),
        "mean_accum_depth": float(z_raw.mean().item()),
        "mean_pixel_world": float(pixel_world.mean().item()),
        "mean_alpha_proxy": float(raw_over_expected.mean().item()),
        "mean_float32_ulp": float(float32_ulp(z_expected).mean().item()),
        "mean_requested_offset": float((front_offset_px * pixel_world.double()).mean().item()),
        "mean_applied_offset": float(offset64.mean().item()),
        "frac_offset_bumped_for_ulp": float((offset64 > (front_offset_px * pixel_world.double() * 1.01)).float().mean().item()),
        "seed_geometry": geometry,
    }


@torch.no_grad()
def append_detail_gaussians(gaussians: GaussianModel, seeded: Dict[str, torch.Tensor]) -> int:
    n_coarse = int(gaussians._xyz.shape[0])
    gaussians.densification_postfix(
        seeded["xyz"],
        seeded["features_dc"],
        seeded["features_rest"],
        seeded["opacity"],
        seeded["scaling"],
        seeded["rotation"],
        seeded["embeddings"],
    )
    return n_coarse


@torch.no_grad()
def compensate_detail_opacity_for_filter(
    gaussians: GaussianModel,
    n_coarse: int,
    target_opacity: float,
) -> Dict[str, float]:
    """Keep effective (filtered) opacity near ``target_opacity`` so new points are not silent."""
    scales = gaussians.get_scaling[n_coarse:]
    filter_3d = gaussians.filter_3D[n_coarse:]
    scales_square = torch.square(scales)
    det1 = scales_square.prod(dim=1)
    det2 = (scales_square + torch.square(filter_3d)).prod(dim=1)
    coef = torch.sqrt((det1 / det2.clamp(min=1e-12)).clamp(min=1e-8))
    raw = (target_opacity / coef).clamp(max=0.99)
    gaussians._opacity.data[n_coarse:] = inverse_sigmoid(raw)[..., None]
    return {
        "filter_opacity_coef_mean": float(coef.mean().item()),
        "filter_opacity_coef_median": float(coef.median().item()),
        "filter_opacity_coef_min": float(coef.min().item()),
        "raw_opacity_mean": float(raw.mean().item()),
        "effective_opacity_target": target_opacity,
    }


@torch.no_grad()
def detail_layer_stats(
    gaussians: GaussianModel,
    camera,
    snapshot: DetailLayerSnapshot,
    radii: Optional[torch.Tensor] = None,
) -> Dict:
    n0 = snapshot.n_coarse
    scales = gaussians.get_scaling[n0:]
    scales_eff = gaussians.get_scaling_with_3D_filter[n0:]
    opacity = gaussians.get_opacity[n0:]
    opacity_f = gaussians.get_opacity_with_3D_filter[n0:]
    init_scale = snapshot.detail_init_scale.max(dim=1).values
    cur_scale = scales.max(dim=1).values
    inflation = cur_scale / init_scale.clamp(min=1e-12)
    u, v, z, in_screen = project_points(gaussians.get_xyz[n0:], camera)
    visible = in_screen
    if radii is not None:
        visible = visible & (radii.reshape(-1)[n0:] > 0)
    screen_px = scales_eff.max(dim=1).values * camera.focal_x / z.clamp(min=1e-6)
    return {
        "n_detail": int(gaussians._xyz.shape[0] - n0),
        "n_visible": int(visible.sum().item()),
        "opacity": _percentiles(opacity.squeeze(-1)),
        "opacity_with_filter": _percentiles(opacity_f.squeeze(-1)),
        "frac_opacity_lt_0.01": float((opacity.squeeze(-1) < 0.01).float().mean().item()),
        "frac_opacity_lt_0.05": float((opacity.squeeze(-1) < 0.05).float().mean().item()),
        "frac_filtered_opacity_lt_0.01": float((opacity_f.squeeze(-1) < 0.01).float().mean().item()),
        "scale": _percentiles(cur_scale),
        "scale_inflation": _percentiles(inflation),
        "frac_scale_gt_4x_init": float((inflation > 4.0).float().mean().item()),
        "frac_scale_gt_8x_init": float((inflation > 8.0).float().mean().item()),
        "screen_px_with_filter": _percentiles(screen_px[visible] if visible.any() else screen_px[:0]),
        "frac_visible": float(visible.float().mean().item()),
    }


def save_gray_image(map_hw: torch.Tensor, out_path: str, gain: float = 1.0) -> None:
    vis = map_hw.detach().clamp(0.0, 1.0) * gain
    save_tensor_image(vis.unsqueeze(0).repeat(3, 1, 1).clamp(0.0, 1.0), out_path)


def _coverage_stats(map_hw: torch.Tensor, prefix: str) -> Dict[str, float]:
    flat = map_hw.detach().float()
    return {
        f"{prefix}_mean": float(flat.mean().item()),
        f"{prefix}_p50": float(flat.median().item()),
        f"{prefix}_frac_gt_0.05": float((flat > 0.05).float().mean().item()),
        f"{prefix}_frac_gt_0.2": float((flat > 0.2).float().mean().item()),
        f"{prefix}_frac_gt_0.5": float((flat > 0.5).float().mean().item()),
    }


@torch.no_grad()
def diagnose_detail_contribution(
    gaussians: GaussianModel,
    camera,
    pipe,
    kernel_size: float,
    n_coarse: int,
):
    """Solo detail alpha vs detail ΣTα inside the full composite.

    Solo alpha has no base occlusion. Composite weight uses override colors
    (detail=1, coarse=0, bg=0) so the rendered value equals Σ Tα of detail
    Gaussians in the real depth order.
    """
    from gaussian_renderer import render

    device = gaussians.get_xyz.device
    n_total = int(gaussians._xyz.shape[0])
    if n_coarse < 0 or n_coarse >= n_total:
        raise ValueError(f"n_coarse={n_coarse} is invalid for {n_total} Gaussians.")

    prev_appearance = gaussians.appearance_enabled
    gaussians.appearance_enabled = False
    black = torch.zeros(3, device=device)

    saved_opacity = gaussians._opacity.data[:n_coarse].clone()
    near_zero = gaussians.inverse_opacity_activation(
        torch.tensor(1e-8, device=device, dtype=gaussians._opacity.dtype)
    )
    gaussians._opacity.data[:n_coarse] = near_zero
    solo_pkg = render(
        camera,
        gaussians,
        pipe,
        black,
        kernel_size=kernel_size,
        testing=True,
        appearance_embedding=None,
    )
    gaussians._opacity.data[:n_coarse] = saved_opacity
    solo_alpha = squeeze_hw(solo_pkg["render_alpha"])

    override = torch.zeros((n_total, 3), device=device, dtype=torch.float32)
    override[n_coarse:] = 1.0
    composite_pkg = render(
        camera,
        gaussians,
        pipe,
        black,
        kernel_size=kernel_size,
        testing=True,
        appearance_embedding=None,
        override_color=override,
    )
    composite_weight = composite_pkg["render"].mean(dim=0)
    gaussians.appearance_enabled = prev_appearance

    covered = solo_alpha > 0.05
    blocked = (solo_alpha - composite_weight).clamp(min=0.0)
    survival = torch.zeros_like(solo_alpha)
    survival[covered] = (composite_weight[covered] / solo_alpha[covered].clamp(min=1e-6)).clamp(0.0, 1.0)
    return {
        "solo_alpha": solo_alpha,
        "composite_weight": composite_weight,
        "blocked": blocked,
        "survival": survival,
        "stats": {
            **_coverage_stats(solo_alpha, "solo_alpha"),
            **_coverage_stats(composite_weight, "composite_weight"),
            "frac_solo_covered_surviving": float(survival[covered].mean().item()) if covered.any() else float("nan"),
            "mean_blocked_where_solo": float(blocked[covered].mean().item()) if covered.any() else float("nan"),
            "n_coarse": n_coarse,
            "n_detail": n_total - n_coarse,
        },
    }


def write_level_review_html(
    path: str,
    title: str,
    metrics: Dict,
    neighbor_metrics: Dict,
    neighbor_baseline: Dict,
    filter_diag: Optional[Dict] = None,
    detail_before: Optional[Dict] = None,
    detail_after: Optional[Dict] = None,
    image_entries: Optional[List[Tuple[str, str]]] = None,
) -> None:
    rows = []
    for key, value in metrics.items():
        if key == "crop_box":
            rows.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
        elif isinstance(value, bool):
            rows.append(f"<tr><td>{key}</td><td>{'yes' if value else 'no'}</td></tr>")
        elif isinstance(value, (int, float)):
            rows.append(f"<tr><td>{key}</td><td>{value:.6f}</td></tr>" if isinstance(value, float) else f"<tr><td>{key}</td><td>{value}</td></tr>")
    neighbor_rows = []
    for name, rec in neighbor_metrics.items():
        base = neighbor_baseline.get(name)
        after = rec.get("l1_to_gt") if isinstance(rec, dict) else rec
        delta = None if base is None or after is None else after - base
        neighbor_rows.append(
            f"<tr><td>{name}</td><td>{base:.5f}</td><td>{after:.5f}</td><td>{delta:+.5f}</td></tr>"
            if base is not None and after is not None and delta is not None
            else f"<tr><td>{name}</td><td>{base}</td><td>{after}</td><td></td></tr>"
        )
    figures = []
    for caption, rel in image_entries or []:
        figures.append(f"<figure><figcaption>{caption}</figcaption><img src='{rel}' /></figure>")

    def _json_block(label: str, payload: Optional[Dict]) -> str:
        if payload is None:
            return ""
        return f"<h2>{label}</h2><pre>{json.dumps(_json_ready(payload), indent=2, ensure_ascii=False)}</pre>"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, sans-serif; background: #111; color: #eee; }}
    main {{ padding: 20px; }}
    table {{ border-collapse: collapse; font-size: 13px; margin: 0 0 18px; }}
    th, td {{ border: 1px solid #333; padding: 6px 10px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }}
    img {{ width: 100%; background: #000; border-radius: 6px; }}
    pre {{ background: #1a1a1a; padding: 12px; overflow: auto; font-size: 12px; }}
    .note {{ color: #bbb; line-height: 1.5; }}
  </style>
</head>
<body>
<main>
  <h1>{title}</h1>
  <p class="note">细节 Gaussian 走普通 alpha 合成，不是可正负相加的图像残差。看停车线/车沿是否变清晰，而不是更多彩色噪声；同时看新点是否膨胀或 opacity 掉到接近 0。</p>
  <h2>指标</h2>
  <table>{''.join(rows)}</table>
  <h2>邻视角 L1</h2>
  <table><tr><th>view</th><th>baseline</th><th>after</th><th>delta</th></tr>{''.join(neighbor_rows)}</table>
  <div class="grid">{''.join(figures)}</div>
  {_json_block("过滤器 / 投影诊断", filter_diag)}
  {_json_block("细节层训前", detail_before)}
  {_json_block("细节层训后", detail_after)}
</main>
</body>
</html>
"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
