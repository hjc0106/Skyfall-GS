"""Gaussian LoD primitives for the GaussianZoom continuous-LoD component.

This module is fully standalone: it imports nothing from Skyfall-GS and never
touches the official rasterizer.  It owns the per-level learnable tensors
(:class:`GaussianLayer`) and the level-of-detail container
(:class:`GaussianLoD`) that freezes old levels, grows the hierarchy from
screen-space gradients across the whole tree, and round-trips checkpoints.

Each Gaussian primitive belongs to exactly one LoD level and stores the
creation-scale reference coefficient ``psi_ref = d/f`` (Eq. 8 of the
GaussianZoom paper).  The reference-anchor policy used here (an engineering
choice, documented in the scope doc because the paper leaves it open):

    ``psi`` of a new primitive = minimum ``distance(camera_center, xyz) / fx``
    over the level's training cameras that see the primitive center projected
    inside the raster and in front of the camera; if no camera of the level
    sees it, ``psi`` = the same minimum taken over *all* level cameras.
    ``psi`` is fixed when the primitive is created (layer creation or a
    densification-split child) and never mutated afterwards.

The levels form a forest: every primitive row carries a globally unique
``node_id`` and a ``parent_id`` referencing a node of a strictly coarser
level (``-1`` marks a root; same-level edges never exist, so level-0
children are roots).  Frozen levels are immutable, but their screen-space
gradients still drive densification: fixed splits/clones append children to
the active level under the source's node id, while active splits replace a
row in place by two children that inherit its parent link and active clones
append a sibling under it.  ``GaussianLoD.topology_mode`` is ``'tree'`` for
real ancestry or ``'legacy_flat'`` for ancestry-free legacy checkpoints
(diagnostics only).

Rendering never happens here: :meth:`GaussianLoD.tensors` returns the merged,
depth-sortable view (one flat set of points, per-level row slices) that the
renderer consumes in a single global rasterization.

Every layer also owns the RaDe-GS 3D sampling-filter radius ``filter_3d`` (a
``[N, 1]`` nonnegative buffer; the needle-artifact repair).  It is anchored
from the layer's own training cameras when the layer is created
(:func:`sampling_filter`) and refreshed only on the *active* layer via
:meth:`GaussianLoD.update_filter`; frozen layers keep theirs bit-for-bit.
:meth:`GaussianLoD.tensors` therefore returns the *filtered* rendering view:
``scales = sqrt(exp(2 log_scales) + filter_3d**2)`` and ``opacity`` scaled by
the stable determinant coefficient ``prod_axes exp(log_scales) / scales``
(the official RaDe-GS ``sqrt(det1 / det2)`` in per-axis form), plus the
unfiltered values under ``raw_scales`` / ``raw_opacity`` for diagnostics.
Version-1 checkpoints (written before the repair) load with zero filters and
reproduce the old unfiltered renders exactly; version 2 stores and validates
``filter_3d`` per layer, and version 3 adds the full tree topology (node
ids, parent links, the id allocator state and ``topology_mode``).
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from .stages import capture_stage, validate_next_stage, validate_stage_records, validate_stage_use


__all__ = ["GaussianLayer", "GaussianLoD"]

# SH DC basis constant C0 = 1 / (2 * sqrt(pi)); SH DC init uses (rgb-0.5)/C0
# so that degree-0 rendering reproduces rgb exactly (color = 0.5 + C0 * dc).
C0 = 0.28209479177387814

# Floor for psi_ref so the stored reference coefficient is always strictly
# positive and finite even for a primitive exactly at a camera center.
_PSI_FLOOR = 1e-12

# Densification split geometry (standard 3DGS numbers): each split parent is
# replaced by _SPLIT_CHILDREN children jittered along the parent's rotated
# frame; child physical scales are divided by _SPLIT_SCALE_SHRINK = 0.8 * 2.
_SPLIT_CHILDREN = 2
_SPLIT_SCALE_SHRINK = 0.8 * _SPLIT_CHILDREN
_LOG_SPLIT_SHRINK = math.log(_SPLIT_SCALE_SHRINK)

# Parameter attribute names (exact optimizer group "name" keys, in a fixed
# order).  psi_ref is a registered buffer and lives outside the optimizer.
_PARAM_NAMES: Tuple[str, ...] = ("xyz", "sh", "log_scales", "rotations", "opacity_logits")

# Registered buffer names that follow layer rows (densify prune/append).
_BUFFER_NAMES: Tuple[str, ...] = ("psi_ref", "filter_3d", "node_ids", "parent_ids")

# Checkpoint identification/version (tensor-only torch payload).  Version 2
# adds the per-layer ``filter_3d`` sampling radius (RaDe-GS needle-artifact
# repair); version 3 adds the tree topology (per-layer ``node_ids`` /
# ``parent_ids``, the ``next_node_id`` allocator state, ``topology_mode``).
# Version 1 payloads are unfiltered and load with zero filters under
# ``model_lod["legacy_unfiltered"] = True`` so old renders reproduce exactly.
_FORMAT_TAG = "gaussianzoom_lod.gaussianlod"
_LEGACY_PAYLOAD_VERSION = 1
_PAYLOAD_VERSION = 4

# RaDe-GS 3D sampling filter (official scene/gaussian_model.py
# ``compute_3D_filter`` / ``get_scaling_n_opacity_with_3D_filter``): a point
# keeps its camera-space depth only when ``z > _FILTER_DEPTH_MIN`` and its
# projected center stays inside the raster support padded by _FILTER_MARGIN
# (0.575 -> [-0.075 W, 1.075 W] for a centered principal point, same vertical
# margin); ``filter = min_visible_depth / max_fx * _FILTER_FACTOR``.
_FILTER_DEPTH_MIN = 0.2
_FILTER_MARGIN = 0.575
_FILTER_FACTOR = math.sqrt(0.2)


_PSI_REFERENCE_POLICY = {
    "name": "min-visible-d-over-fx",
    "rule": (
        "psi = min over the level training cameras whose frustum contains the "
        "primitive center of ||xyz - camera_center|| / fx; fallback min over all "
        "level cameras when no camera of the level sees it; floored at 1e-12."
    ),
    "scope": (
        "fixed when a primitive is created (layer creation or a densification "
        "split/clone child); survivors keep their psi; engineering choice documented "
        "in the component scope doc (the paper leaves the anchor rule open)."
    ),
}

try:
    from simple_knn._C import distCUDA2 as _dist_cuda2
except ImportError:  # pragma: no cover - environment without simple-knn
    _dist_cuda2 = None


# --------------------------------------------------------------------------- #
# Small pure-tensor helpers
# --------------------------------------------------------------------------- #


def _inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """logit of ``x``, i.e. the inverse of the logistic sigmoid."""
    return torch.log(x / (1.0 - x))


def _quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Batched quaternion ``(w, x, y, z)`` -> ``[..., 3, 3]`` rotation matrix.

    Input rows are normalized first; the layout matches the rasterizer's
    convention (quaternion [w, x, y, z], active rotation as used by the
    official Gaussian rasterizers).
    """
    q = F.normalize(q.float(), dim=-1)
    w, x, y, z = torch.unbind(q, dim=-1)
    r = torch.zeros(q.shape[:-1] + (3, 3), dtype=q.dtype, device=q.device)
    r[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    r[..., 0, 1] = 2.0 * (x * y - w * z)
    r[..., 0, 2] = 2.0 * (x * z + w * y)
    r[..., 1, 0] = 2.0 * (x * y + w * z)
    r[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    r[..., 1, 2] = 2.0 * (y * z - w * x)
    r[..., 2, 0] = 2.0 * (x * z - w * y)
    r[..., 2, 1] = 2.0 * (y * z + w * x)
    r[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return r


def _camera_center(cam: Any) -> torch.Tensor:
    """World-space camera center of a duck-typed camera object.

    Uses the documented ``center`` property when present (per the camera
    contract it equals ``inverse(w2c)[:3, 3]``); otherwise derives it from the
    ``w2c`` field with ``-R^T t``.
    """
    center = getattr(cam, "center", None)
    if center is not None:
        return center
    w2c = cam.w2c
    return -(w2c[:3, :3].transpose(0, 1) @ w2c[:3, 3])


def _visible_mask(xyz: torch.Tensor, cam: Any) -> torch.Tensor:
    """Boolean mask: primitives whose center projects inside the raster.

    ``cam`` must expose ``w2c`` ([4,4] row-major, COLMAP convention with the
    camera looking down +z), ``fx``/``fy``/``cx``/``cy`` (pixels) and
    ``width``/``height``.  A primitive counts as visible when it is in front
    of the camera and its projected center falls in ``[0, width) x
    [0, height)``.
    """
    w2c = cam.w2c
    xyz_h = F.pad(xyz, (0, 1), value=1.0)
    cam_space = xyz_h @ w2c.transpose(0, 1)  # [N, 4] rows in camera space
    x_c, y_c, z_c = cam_space[:, 0], cam_space[:, 1], cam_space[:, 2]
    fx, fy = float(cam.fx), float(cam.fy)
    cx, cy = float(cam.cx), float(cam.cy)
    width, height = int(cam.width), int(cam.height)
    in_front = z_c > 0.0
    safe_z = torch.where(in_front, z_c, torch.ones_like(z_c))
    u = fx * x_c / safe_z + cx
    v = fy * y_c / safe_z + cy
    inside = (u >= 0.0) & (u < float(width)) & (v >= 0.0) & (v < float(height))
    return in_front & inside


@torch.no_grad()
def _creation_psi(xyz: torch.Tensor, cameras: Sequence[Any]) -> torch.Tensor:
    """Reference ``psi = min visible d/fx`` (fallback: min d/fx over cameras).

    See the module docstring for the anchor policy.  ``xyz`` is ``[N, 3]``
    float; returns a contiguous ``[N, 1]`` float32 tensor of strictly
    positive, finite reference coefficients.  Raises ``ValueError`` when no
    camera is provided or when a camera has invalid intrinsics.
    """
    if len(cameras) == 0:
        raise ValueError(
            "psi reference requires at least one camera (the level's training "
            "cameras); got an empty camera list"
        )
    xyz = xyz.detach().float()
    n = int(xyz.shape[0])
    device = xyz.device
    best_visible = torch.full((n,), float("inf"), dtype=torch.float32, device=device)
    best_all = torch.full((n,), float("inf"), dtype=torch.float32, device=device)
    for cam in cameras:
        fx = float(cam.fx)
        if not math.isfinite(fx) or fx <= 0.0:
            raise ValueError("camera fx must be finite and positive for psi anchoring")
        center = _camera_center(cam).to(dtype=torch.float32, device=device).reshape(1, 3)
        dist = torch.sqrt(((xyz - center) ** 2).sum(dim=-1))
        ratio = dist / fx
        torch.minimum(best_all, ratio, out=best_all)
        visible = _visible_mask(xyz, cam)
        if bool(visible.any().item()):
            best_visible = torch.where(
                visible, torch.minimum(best_visible, ratio), best_visible
            )
    # Never-visible primitives fall back to the min over all cameras; the
    # floor keeps the stored coefficient strictly positive.
    psi = torch.where(torch.isfinite(best_visible), best_visible, best_all)
    psi = torch.clamp_min(psi, _PSI_FLOOR).reshape(n, 1).contiguous()
    if not bool(torch.isfinite(psi).all().item()) or not bool((psi > 0.0).all().item()):
        raise RuntimeError("psi validation failed: non-finite input coordinates")
    return psi


@torch.no_grad()
def sampling_filter(xyz: torch.Tensor, cameras: Sequence[Any]) -> torch.Tensor:
    """Per-point RaDe-GS 3D sampling-filter radius ``filter_3d``.

    Follows the official ``compute_3D_filter`` (vendor RaDe-GS
    ``scene/gaussian_model.py``) made principal-point aware: for every point
    of ``xyz`` (``[N, 3]`` float),

        filter = min_visible_depth / max_fx * sqrt(0.2)

    where ``max_fx`` is the largest focal length over *all* cameras (the
    highest-resolution camera) and a point is *visible* in a camera when its
    center sits at camera-space depth ``z > 0.2`` and projects inside the
    padded raster support ``[-0.075 W, 1.075 W]`` by
    ``[-0.075 H, 1.075 H]`` in pixels; projection includes cx/cy.
    This equals the official 0.575 symmetric bound for centered intrinsics.
    Points no camera sees fall back to the *largest visible depth*,
    matching the official implementation; when *no* point of the set is
    visible anywhere, a ``ValueError`` is raised instead of silently
    inventing a scene extent.

    Raises ``ValueError`` when ``cameras`` is empty or exposes invalid
    intrinsics/raster, or when no point is visible in any camera.  Returns a
    contiguous ``[N, 1]`` float32 tensor of strictly positive finite values
    (``filter`` cannot be zero for a non-empty set with a visible point).
    """
    if len(cameras) == 0:
        raise ValueError(
            "sampling_filter requires at least one camera (the level's "
            "training cameras); got an empty camera list"
        )
    xyz = xyz.detach().float()
    n = int(xyz.shape[0])
    device = xyz.device
    best = torch.full((n,), float("inf"), dtype=torch.float32, device=device)
    seen = torch.zeros(n, dtype=torch.bool, device=device)
    max_fx = 0.0
    for cam in cameras:
        fx = float(cam.fx)
        fy = float(cam.fy)
        if not math.isfinite(fx) or fx <= 0.0 or not math.isfinite(fy) or fy <= 0.0:
            raise ValueError(
                "camera fx/fy must be finite and positive for the sampling filter"
            )
        cx, cy = float(cam.cx), float(cam.cy)
        if not math.isfinite(cx) or not math.isfinite(cy):
            raise ValueError("camera cx/cy must be finite for the sampling filter")
        width, height = int(cam.width), int(cam.height)
        if width < 1 or height < 1:
            raise ValueError(
                f"camera raster must be at least 1x1, got {width}x{height}"
            )
        max_fx = max(max_fx, fx)
        w2c = torch.as_tensor(cam.w2c, dtype=torch.float32, device=device)
        cam_space = F.pad(xyz, (0, 1), value=1.0) @ w2c.transpose(0, 1)
        x_c, y_c, z_c = cam_space[:, 0], cam_space[:, 1], cam_space[:, 2]
        valid_depth = z_c > _FILTER_DEPTH_MIN
        safe_z = torch.where(z_c > 0.0, z_c, torch.ones_like(z_c))
        u = fx * x_c / safe_z + cx
        v = fy * y_c / safe_z + cy
        margin_x = _FILTER_MARGIN * float(width)
        margin_y = _FILTER_MARGIN * float(height)
        in_support = (
            (u >= 0.5 * width - margin_x)
            & (u <= 0.5 * width + margin_x)
            & (v >= 0.5 * height - margin_y)
            & (v <= 0.5 * height + margin_y)
        )
        visible = valid_depth & in_support
        best = torch.where(visible, torch.minimum(best, z_c), best)
        seen = seen | visible
    if not math.isfinite(max_fx) or max_fx <= 0.0:
        raise ValueError("cameras expose no positive finite focal length")
    n_visible = int(seen.sum().item())
    if n_visible == 0:
        raise ValueError(
            "no Gaussian center is visible in any camera (depth > 0.2 inside "
            "the padded raster support); refusing to invent a scene-extent "
            "sampling filter"
        )
    largest_visible = float(best[seen].max().item())  # scene-extent fallback
    filter_3d = torch.where(seen, best, torch.full_like(best, largest_visible))
    filter_3d = filter_3d / max_fx * _FILTER_FACTOR
    return filter_3d.reshape(n, 1).contiguous()


# --------------------------------------------------------------------------- #
# GaussianLayer
# --------------------------------------------------------------------------- #


class GaussianLayer(nn.Module):
    """One LoD level: a single set of Gaussian primitives plus metadata.

    Parameters (all float32, CUDA in the training pipeline):

    ================ ============ =========================================
    name             shape        meaning
    ================ ============ =========================================
    ``xyz``          [N, 3]       primitive centers (world)
    ``sh``           [N, K, 3]    spherical-harmonic coefficients,
                                  K = (degree + 1)**2, channels last
    ``log_scales``   [N, 3]       log of physical per-axis scales
    ``rotations``    [N, 4]       raw quaternions (w, x, y, z); the
                                  renderer consumes normalized versions
    ``opacity_logits`` [N, 1]     logits; renderer consumes sigmoid(opacity)
    ``psi_ref``      [N, 1]       registered buffer (not a parameter): the
                                  creation-scale reference ``d/fx`` used for
                                  the continuous-LoD weight, fixed per row
    ``filter_3d``    [N, 1]       registered buffer (not a parameter): the
                                  RaDe-GS sampling-filter radius per row
                                  (nonnegative, finite), anchored at layer
                                  creation and refreshed only on the active
                                  layer via GaussianLoD.update_filter
    ``node_ids``     [N]          registered buffer: globally unique int64
                                  node identifiers (allocator-issued; the
                                  deterministic default for hand-built layers
                                  is ``(level << 48) + arange(N)``)
    ``parent_ids``   [N]          registered buffer: int64 parent node ids,
                                  ``-1`` for roots; a parent always lives on
                                  a strictly coarser level
    ================ ============ =========================================

    ``level`` is the integer LoD index (0 for the base layer).  A layer whose
    parameters have ``requires_grad=False`` is frozen and must stay immutable;
    the LoD container only optimizes/densifies the currently active layer.

    Parameters are stored per level (this module is per level), so freezing
    old levels and optimizing the new one never needs masked optimizers.
    """

    def __init__(
        self,
        level: int,
        xyz: torch.Tensor,
        sh: torch.Tensor,
        log_scales: torch.Tensor,
        rotations: torch.Tensor,
        opacity_logits: torch.Tensor,
        psi_ref: torch.Tensor,
        frozen: bool = False,
        filter_3d: Optional[torch.Tensor] = None,
        node_ids: Optional[torch.Tensor] = None,
        parent_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Build a level owning copies-free references to the given tensors.

        Tensors are detached, cast to float32 and made contiguous; they must
        share one row count ``N >= 0`` (``N == 0`` is the empty fine level
        created by :meth:`GaussianLoD.add_level`; level 0 requires
        ``N >= 1``).  ``sh`` must be ``[N, K, 3]`` with ``K >= 1``.
        ``psi_ref`` must be ``[N, 1]`` with strictly positive finite values.
        ``filter_3d`` must be ``[N, 1]``, nonnegative and finite; when
        omitted the layer is *unfiltered* (an exactly zero buffer, for raw
        tests and version-1 checkpoints).  The physical scale
        ``exp(log_scales)`` must be finite.

        ``node_ids``/``parent_ids`` are int64 ``[N]`` topology metadata kept
        as persistent buffers.  Defaults are deterministic for hand-built
        layers -- ``(level << 48) + arange(N)`` ids and all-``-1`` (root)
        parents; the LoD container passes allocator-issued ids instead.
        """
        super().__init__()
        level = int(level)
        if level < 0:
            raise ValueError(f"level must be a non-negative integer, got {level}")
        if not isinstance(frozen, bool):
            raise TypeError("frozen must be a bool")
        self.level = level

        def prep(t: torch.Tensor, shape_tail: Tuple[int, ...]) -> torch.Tensor:
            t = t.detach()
            if t.dtype != torch.float32:
                t = t.float()
            t = t.contiguous()
            if not torch.isfinite(t).all().item():
                raise ValueError(f"layer {level}: non-finite tensor in {shape_tail}")
            return t

        n = int(torch.as_tensor(xyz).shape[0])
        if n < 1 and level == 0:
            raise ValueError("a level-0 GaussianLayer must contain at least one primitive")
        row_shapes = {
            "xyz": (3,),
            "sh": (None, 3),  # [N, (degree+1)^2, 3]: per-row tail has 2 dims
            "log_scales": (3,),
            "rotations": (4,),
            "opacity_logits": (1,),
        }
        check = {
            "xyz": tuple(xyz.shape), "sh": tuple(sh.shape),
            "log_scales": tuple(log_scales.shape), "rotations": tuple(rotations.shape),
            "opacity_logits": tuple(opacity_logits.shape),
        }
        for name, shape in check.items():
            tail = row_shapes[name]
            if (
                shape[0] != n
                or len(shape) != 1 + len(tail)
                or any(
                    want is not None and got != want
                    for got, want in zip(shape[1:], tail)
                )
            ):
                raise ValueError(
                    f"layer {level}: {name} has shape {shape}, expected leading "
                    f"dimension {n} and per-row tail {tail}"
                )
        if int(sh.shape[1]) < 1:
            raise ValueError(f"layer {level}: sh needs at least the DC coefficient")

        psi = prep(psi_ref, (1,))
        if psi.shape != (n, 1):
            raise ValueError(f"layer {level}: psi_ref must be [N, 1], got {tuple(psi.shape)}")
        if not bool((psi > 0.0).all().item()) or not bool(torch.isfinite(psi).all().item()):
            raise ValueError(f"layer {level}: psi_ref must be strictly positive and finite")

        if filter_3d is None:  # unfiltered layer: raw tests, legacy checkpoints
            filt = torch.zeros((n, 1), dtype=torch.float32, device=psi.device)
        else:
            filt = prep(filter_3d, (1,))
            if filt.shape != (n, 1):
                raise ValueError(
                    f"layer {level}: filter_3d must be [N, 1], got {tuple(filt.shape)}"
                )
            if bool((filt < 0.0).any().item()):
                raise ValueError(f"layer {level}: filter_3d must be nonnegative")

        if node_ids is None:  # deterministic ids for hand-built layers
            ids = (level << 48) + torch.arange(n, dtype=torch.int64, device=psi.device)
        else:
            ids = torch.as_tensor(node_ids, dtype=torch.int64, device=psi.device).reshape(-1)
        if tuple(ids.shape) != (n,):
            raise ValueError(f"layer {level}: node_ids must be [{n}], got {tuple(ids.shape)}")
        if n and bool((ids < 0).any().item()):
            raise ValueError(f"layer {level}: node_ids must be nonnegative")
        if n and int(torch.unique(ids).numel()) != n:
            raise ValueError(f"layer {level}: node_ids must be unique within the layer")
        if parent_ids is None:  # default: roots
            parents = torch.full((n,), -1, dtype=torch.int64, device=psi.device)
        else:
            parents = torch.as_tensor(parent_ids, dtype=torch.int64, device=psi.device).reshape(-1)
        if tuple(parents.shape) != (n,):
            raise ValueError(f"layer {level}: parent_ids must be [{n}], got {tuple(parents.shape)}")

        log_s = prep(log_scales, (3,))
        if not bool(torch.isfinite(torch.exp(log_s)).all().item()):
            raise ValueError(
                f"layer {level}: physical scale exp(log_scales) must be finite "
                "(log_scales too large for float32)"
            )

        grad_flag = not frozen
        self.xyz = nn.Parameter(prep(xyz, (3,)), requires_grad=grad_flag)
        self.sh = nn.Parameter(prep(sh, (None, 3)), requires_grad=grad_flag)
        self.log_scales = nn.Parameter(log_s, requires_grad=grad_flag)
        self.rotations = nn.Parameter(prep(rotations, (4,)), requires_grad=grad_flag)
        self.opacity_logits = nn.Parameter(
            prep(opacity_logits, (1,)), requires_grad=grad_flag
        )
        self.register_buffer("psi_ref", psi, persistent=True)
        self.register_buffer("filter_3d", filt, persistent=True)
        self.register_buffer("node_ids", ids.contiguous(), persistent=True)
        self.register_buffer("parent_ids", parents.contiguous(), persistent=True)

    # -- filtered rendering view (RaDe-GS sampling filter) ------------------- #

    def filter_terms(self):
        """Raw scales, filtered scales and opacity coefficient, computed once.

        The zero-filter identity branch avoids evaluating hypot(0,0) in a
        differentiable path: its undefined derivative can poison masked grads.
        """
        raw = torch.exp(self.log_scales)
        enabled = self.filter_3d > 0
        radius = torch.where(enabled, self.filter_3d, torch.ones_like(self.filter_3d))
        smoothed = torch.hypot(raw, radius)
        filtered = torch.where(enabled, smoothed, raw)
        ratio = torch.where(enabled, raw / smoothed, torch.ones_like(raw))
        return raw, filtered, ratio.prod(dim=1, keepdim=True)

    def filtered_scales(self) -> torch.Tensor:
        """``[N, 3]`` per-axis effective scales of the filtered footprint.

        ``sqrt(exp(2 log_scales) + filter_3d**2)`` computed with
        ``torch.hypot`` (no intermediate square overflow).  With an exactly
        zero filter this equals ``exp(log_scales)``, so legacy unfiltered
        layers keep their exact old scales.
        """
        return self.filter_terms()[1]

    def filter_coef(self) -> torch.Tensor:
        """``[N, 1]`` determinant-compensation opacity coefficient.

        Stable per-axis equivalent of the official RaDe-GS
        ``sqrt(det1 / det2)``: the product over axes of
        ``exp(log_scales) / filtered_scales``, each factor
        ``s / sqrt(s**2 + f**2)`` in ``[0, 1]`` -- no six-factor squared
        determinant product that could underflow before the factors are
        combined.  An exactly zero filter gives the exact identity ``1.0``
        (the zero/zero axes that would otherwise divide 0/0 into NaN are
        pinned to 1), so unfiltered layers multiply opacity by exactly one.
        """
        return self.filter_terms()[2]

    # -- freeze helpers ----------------------------------------------------- #

    @property
    def frozen(self) -> bool:
        """True when none of this layer's parameters requires gradients."""
        return all(not p.requires_grad for p in self.parameters())

    def freeze(self) -> "GaussianLayer":
        """Make every parameter non-trainable (requires_grad=False)."""
        for p in self.parameters():
            p.requires_grad_(False)
            p.grad = None
        return self

    def unfreeze(self) -> "GaussianLayer":
        """Make every parameter trainable (requires_grad=True)."""
        for p in self.parameters():
            p.requires_grad_(True)
        return self

    def __repr__(self) -> str:  # compact, informative
        return (
            f"GaussianLayer(level={self.level}, points={int(self.xyz.shape[0])}, "
            f"sh_degree={(int(self.sh.shape[1]) ** 0.5 - 1):.0f}, frozen={self.frozen})"
        )


# --------------------------------------------------------------------------- #
# GaussianLoD
# --------------------------------------------------------------------------- #


class GaussianLoD(nn.Module):
    """Multi-level Gaussian hierarchy with frozen older levels.

    State:
        ``layers`` -- ``nn.ModuleList`` of :class:`GaussianLayer` ordered by
        increasing level (level ``i`` at index ``i``); exactly
        ``active_level + 1`` layers exist, levels ``0 .. active_level - 1``
        frozen, level ``active_level`` trainable.
        ``next_node_id`` -- the monotonic node-id allocator state; every new
        primitive row takes fresh ids from here (persisted in v3 payloads).
        ``topology_mode`` -- ``'tree'`` (real parent/child ancestry) or
        ``'legacy_flat'`` (ancestry-free legacy payload; diagnostics only).
        ``degree`` -- SH degree shared by every layer (sh width
        ``(degree + 1) ** 2``).
        ``step_scale`` -- the adjacent-level scale ratio ``s > 1`` of the
        log-space cross-fade (Eq. 9; 4.0 for the paper's experiment) and of
        the per-level scale division when seeding a new level.

    Only the active layer is ever optimized; frozen layers are immutable and
    excluded from the optimizer (parameters are stored per level, so no
    masked-Adam bookkeeping is needed) -- yet their screen-space gradients
    still drive densification through :meth:`densify`.
    """

    def __init__(self, degree: int = 1, step_scale: float = 4.0) -> None:
        """Empty LoD container; populate it with :meth:`from_points`."""
        super().__init__()
        degree = int(degree)
        if degree < 0 or degree > 3:
            raise ValueError(f"degree must be in [0, 3], got {degree}")
        step_scale = float(step_scale)
        if not math.isfinite(step_scale) or step_scale <= 1.0:
            raise ValueError(f"step_scale must be finite and > 1, got {step_scale}")
        self.degree = degree
        self.step_scale = step_scale
        self.layers: nn.ModuleList = nn.ModuleList()
        self.active_level: int = -1
        self.next_node_id: int = 0
        self.topology_mode: str = "tree"
        self.stage_records: List[Dict[str, Any]] = []
        self.weight_policy = "interval"

    # -- construction ------------------------------------------------------- #

    def _active(self) -> GaussianLayer:
        if len(self.layers) == 0 or self.active_level < 0:
            raise RuntimeError(
                "no level exists yet: create the model with GaussianLoD.from_points() "
                "or add levels with add_level()"
            )
        return self.layers[self.active_level]

    @property
    def active(self) -> GaussianLayer:
        """The currently trainable layer (level ``active_level``)."""
        return self._active()

    def require_stage_records(self):
        """Reject mutation/rendering under invented or missing scale history."""
        if self.topology_mode != "tree" or self.weight_policy != "interval" or len(self.stage_records) != len(self.layers) or not self.stage_records:
            raise ValueError("Verified stage camera records are required; bind a filtered L0 or start from_points")

    def bind_base_stage(self, cameras):
        """Explicitly attach known calibration to a legacy single-layer L0.

        This never infers a scale from primitive psi statistics, and never
        fabricates the unavailable camera history of a multi-level checkpoint.
        """
        if len(self.layers) != 1 or self.active_level != 0 or self.topology_mode != "tree":
            raise ValueError("Only a single filtered root layer can bind base calibration")
        if self.stage_records:
            raise ValueError("Base stage calibration is already bound")
        record = capture_stage(cameras, scale=1.)
        self.stage_records = [record]
        self.weight_policy = "interval"

    def _check_stage_use(self, cameras):
        self.require_stage_records()
        signature = (self.active_level, id(self.stage_records[-1]), tuple(
            (cam.name, cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy,
             id(cam.w2c), cam.w2c._version) for cam in cameras))
        if getattr(self, "_camera_stage_check", None) != signature:
            validate_stage_use(self.stage_records[-1], cameras)
            # Hold camera tensors so Python cannot reuse identities in a cache hit.
            self._camera_stage_refs = tuple(cam.w2c for cam in cameras)
            self._camera_stage_check = signature

    @classmethod
    def from_points(
        cls,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        cameras: Sequence[Any],
        degree: int = 1,
        step_scale: float = 4.0,
    ) -> "GaussianLoD":
        """Build the L0 base level from an SfM point cloud.

        Args:
            xyz: ``[N, 3]`` world positions (CUDA float32 in the pipeline;
                moved/cast as needed).
            rgb: ``[N, 3]`` colors in ``[0, 1]`` (clamped), one per point.
            cameras: the base level's training cameras (duck-typed, per the
                camera contract); they anchor ``psi_ref`` and must be
                non-empty.
            degree: SH degree for every layer of the model (0..3).
            step_scale: adjacent-level scale ratio ``s > 1``.

        Initialization follows standard 3DGS: per-point scale from
        ``distCUDA2`` nearest-neighbor distances (simple-knn extension), SH DC
        from ``(rgb - 0.5) / C0`` with higher bands zero, identity rotations,
        opacity 0.1 (logits).  ``psi_ref`` is anchored with the module's
        min-visible-``d/fx`` policy and ``filter_3d`` with
        :func:`sampling_filter` -- both over ``cameras``, so a freshly
        initialized model is always filtered.  Raises ``ValueError`` (from
        ``sampling_filter``) when no point of the cloud is visible in any
        camera.
        """
        if _dist_cuda2 is None:
            raise ImportError(
                "GaussianLoD.from_points needs the simple-knn CUDA extension "
                "(distCUDA2) for scale initialization: pip install simple-knn"
            )
        model = cls(degree=degree, step_scale=step_scale)
        model.stage_records = [capture_stage(cameras, scale=1.)]

        p = torch.as_tensor(xyz)
        c = torch.as_tensor(rgb)
        if p.dim() != 2 or p.shape[1] != 3:
            raise ValueError(f"xyz must be [N, 3], got {tuple(p.shape)}")
        if tuple(c.shape) != tuple(p.shape):
            raise ValueError(
                f"rgb must match xyz: got {tuple(c.shape)} vs {tuple(p.shape)}"
            )
        if p.shape[0] < 1:
            raise ValueError("from_points requires at least one point")
        if not bool(torch.isfinite(p).all().item()) or not bool(
            torch.isfinite(c).all().item()
        ):
            raise ValueError("from_points inputs must be finite")
        if not p.is_cuda:
            p = p.cuda()
        if not c.is_cuda:
            c = c.cuda()
        if p.dtype != torch.float32:
            p = p.float()
        if c.dtype != torch.float32:
            c = c.float()
        p = p.contiguous().clone()  # model owns its parameter storage
        c = c.contiguous().clamp(0.0, 1.0)
        n = int(p.shape[0])

        dist2 = torch.clamp_min(_dist_cuda2(p.detach().clone()), 1e-7)
        log_scales = (0.5 * torch.log(dist2))[:, None].repeat(1, 3)

        rotations = torch.zeros((n, 4), dtype=torch.float32, device=p.device)
        rotations[:, 0] = 1.0

        opacity_logits = _inverse_sigmoid(
            torch.full((n, 1), 0.1, dtype=torch.float32, device=p.device)
        )

        sh_channels = (int(degree) + 1) ** 2
        sh = torch.zeros((n, sh_channels, 3), dtype=torch.float32, device=p.device)
        sh[:, 0, :] = (c - 0.5) / C0

        psi_ref = _creation_psi(p, cameras)
        filter_3d = sampling_filter(p, cameras)

        layer = GaussianLayer(
            level=0,
            xyz=p,
            sh=sh,
            log_scales=log_scales,
            rotations=rotations,
            opacity_logits=opacity_logits,
            psi_ref=psi_ref,
            frozen=False,
            filter_3d=filter_3d,
            node_ids=model._alloc_ids(n, device=p.device),
        )  # parent_ids default to roots: L0 primitives are tree roots
        model.layers.append(layer)
        model.active_level = 0
        return model

    def add_level(self, cameras: Sequence[Any]) -> GaussianLayer:
        """Freeze every old level and append a new, EMPTY trainable level.

        The fine level starts with zero primitives -- no blanket uniform
        copies of the coarser level (supplement B.1: the fine level is born
        from fixed-level screen-space gradients, not from copies).  The
        empty layer is valid everywhere: :meth:`tensors` still returns the
        frozen rows so their screen gradients accumulate, the optimizer may
        hold empty params, :meth:`update_filter` is a silent no-op, and
        :meth:`densify` seeds the first rows from fixed-level candidates,
        anchoring each child's ``psi_ref``/``filter_3d`` over ``cameras`` at
        that moment.  Old layers are frozen only after the cheap argument
        checks pass, so a rejected call leaves the model untouched.

        Args:
            cameras: training cameras of the level being created (non-empty).

        Returns:
            The newly created, active, empty :class:`GaussianLayer`.
        """
        if self.topology_mode != "tree":
            raise ValueError(
                "legacy_flat models carry no ancestry and are diagnostic-only; "
                "rebuild a tree with GaussianLoD.from_points()"
            )
        if len(cameras) == 0:
            raise ValueError(
                "add_level requires the new level's training cameras (they "
                "anchor child psi_ref/filter_3d during densification)"
            )
        prev = self._active()
        self.require_stage_records()
        validate_stage_records(self.stage_records, self.step_scale)
        new_stage = validate_next_stage(self.stage_records[0], self.stage_records[-1],
                                        cameras, self.step_scale)
        device = prev.xyz.device
        zeros = lambda *shape: torch.zeros(shape, dtype=torch.float32, device=device)  # noqa: E731
        layer = GaussianLayer(
            level=prev.level + 1,
            xyz=zeros(0, 3),
            sh=zeros(0, (self.degree + 1) ** 2, 3),
            log_scales=zeros(0, 3),
            rotations=zeros(0, 4),
            opacity_logits=zeros(0, 1),
            psi_ref=zeros(0, 1),
            frozen=False,
            filter_3d=None,
        )
        for old in self.layers:
            old.freeze()
        self.layers.append(layer)
        self.active_level = layer.level
        self.stage_records.append(new_stage)
        return layer

    # -- node-id allocation and topology validation -------------------------- #

    def _alloc_ids(self, count: int, device: torch.device) -> torch.Tensor:
        """Issue ``count`` fresh, globally unique, monotonic node ids."""
        count = int(count)
        if count < 0:
            raise ValueError(f"node-id count must be nonnegative, got {count}")
        ids = torch.arange(
            self.next_node_id, self.next_node_id + count,
            dtype=torch.int64, device=device,
        )
        self.next_node_id += count
        return ids

    def _sync_allocator(self) -> int:
        """Raise the allocator above every issued id; return the new state.

        Keeps hand-built layers (deterministic default ids, untouched
        ``next_node_id``) collision-safe: :meth:`densify` and :meth:`save`
        call this before allocating or persisting ids.
        """
        issued = [
            int(lid.node_ids.max().item())
            for lid in self.layers
            if int(lid.node_ids.numel()) > 0
        ]
        self.next_node_id = max(int(self.next_node_id), (max(issued) + 1) if issued else 0)
        return self.next_node_id

    def validate_topology(self) -> None:
        """Validate node-id/parent-link invariants; raise ``ValueError``.

        Checks the ``topology_mode`` value, per-layer ``node_ids``/
        ``parent_ids`` dtype and shape, global node-id uniqueness, that every
        parent link points at an existing node of a strictly coarser level
        (which rules out cycles and same-level edges), that ``legacy_flat``
        models carry no links at all, and that the allocator sits strictly
        above every issued id (:meth:`densify`/:meth:`save` re-synchronize it,
        so hand-built layers with deterministic default ids validate right
        after a densify or a checkpoint round-trip).
        """
        if self.topology_mode not in ("tree", "legacy_flat"):
            raise ValueError(f"unknown topology_mode {self.topology_mode!r}")
        id_blocks: List[torch.Tensor] = []
        lower: List[torch.Tensor] = []
        for i, lid in enumerate(self.layers):
            n = int(lid.xyz.shape[0])
            for name in ("node_ids", "parent_ids"):
                t = getattr(lid, name)
                if t.dtype != torch.int64 or tuple(t.shape) != (n,):
                    raise ValueError(
                        f"layer {i}: {name} must be int64 [{n}], got "
                        f"{tuple(t.shape)} {t.dtype}"
                    )
            if n:
                if self.topology_mode == "tree":
                    if not lower:
                        if bool((lid.parent_ids != -1).any().item()):
                            raise ValueError(f"layer {i}: roots must have parent_id -1")
                    else:
                        known = (lid.parent_ids == -1) | torch.isin(
                            lid.parent_ids, torch.cat(lower)
                        )
                        if not bool(known.all().item()):
                            raise ValueError(
                                f"layer {i}: {int((~known).sum().item())} parent "
                                "link(s) do not point at an existing lower-level node"
                            )
                elif bool((lid.parent_ids != -1).any().item()):
                    raise ValueError(
                        f"layer {i}: legacy_flat topology must not carry parent links"
                    )
            id_blocks.append(lid.node_ids)
            lower.append(lid.node_ids)
        flat = torch.cat(id_blocks) if id_blocks else None
        if flat is not None and int(flat.numel()) > 0:
            if int(torch.unique(flat).numel()) != int(flat.numel()):
                raise ValueError("node ids must be globally unique across layers")
            top = int(flat.max().item())
            if self.next_node_id <= top:
                raise ValueError(
                    f"next_node_id {self.next_node_id} must exceed the largest "
                    f"issued node id {top}"
                )

    # -- rendering view ----------------------------------------------------- #

    def tensors(self, max_level: Optional[int] = None) -> Dict[str, Any]:
        """Merged single-point-set view of levels ``0 .. max_level``.

        Concatenates every included level into one flat point set (the
        renderer performs a single, globally depth-sorted rasterization):
        ``xyz``, ``scales`` (RaDe-GS *filtered* footprint
        ``sqrt(exp(2 log_scales) + filter_3d**2)``), ``rotations``
        (L2-normalized), ``opacity`` (*filtered*: ``sigmoid(opacity_logits)``
        times the stable determinant-compensation coefficient
        ``prod_axes exp(log_scales) / scales``), ``sh`` and ``psi_ref``, plus
        ``layer_slices`` -- one ``slice`` per included level over the row
        dimension of every other tensor, ordered by level.  Gradient flow is
        preserved through the active layer's parameters only (frozen layers
        contribute constants), which is what lets fixed-level screen-space
        gradients accumulate and bootstrap an initially empty fine level
        (an empty level contributes an empty slice).

        Args:
            max_level: include levels ``0 .. min(max_level, active_level)``;
                ``None`` (default) includes every existing level.

        Returns:
            Dict with keys ``xyz``, ``scales``, ``rotations``, ``opacity``,
            ``sh``, ``psi_ref`` (all ``[total_N, ...]`` float32 contiguous,
            ``sh`` is ``[total_N, (degree+1)^2, 3]``), ``raw_scales``
            (``exp(log_scales)``) and ``raw_opacity`` (``sigmoid(opacity_logits)``)
            -- the unfiltered diagnostics, identical to the filtered values
            for zero ``filter_3d`` -- plus the topology view: ``node_ids`` /
            ``parent_ids`` (int64 ``[total_N]``) and ``has_parent`` /
            ``has_child`` (bool ``[total_N]``), derived from the actual
            parent links of the in-scope levels only (never from layer
            counts, and never written back into any layer's state), and
            ``layer_slices`` (``list[slice]``, index ``i`` = level ``i``).
        """
        n_layers = len(self.layers)
        if n_layers == 0:
            raise RuntimeError("model has no levels; nothing to render")
        top = n_layers - 1
        if max_level is not None:
            max_level = int(max_level)
            if max_level < 0:
                raise ValueError(f"max_level must be >= 0, got {max_level}")
            top = min(top, max_level)
        sh_channels: Optional[int] = None
        parts: Dict[str, List[torch.Tensor]] = {
            "xyz": [], "sh": [], "rotations": [], "psi_ref": [],
            "scales": [], "raw_scales": [], "opacity": [], "raw_opacity": [],
            "node_ids": [], "parent_ids": [],
        }
        slices: List[slice] = []
        start = 0
        for idx in range(top + 1):
            layer = self.layers[idx]
            n = int(layer.xyz.shape[0])
            channels = int(layer.sh.shape[1])
            if sh_channels is None:
                sh_channels = channels
            elif channels != sh_channels:
                raise RuntimeError(
                    f"layer {idx} has {channels} SH channels but level 0 has "
                    f"{sh_channels}; all layers must share one degree"
                )
            raw_scales, filtered_scales, coefficient = layer.filter_terms()
            raw_opacity = torch.sigmoid(layer.opacity_logits)
            parts["xyz"].append(layer.xyz)
            parts["sh"].append(layer.sh)
            parts["rotations"].append(F.normalize(layer.rotations, dim=-1))
            parts["scales"].append(filtered_scales)
            parts["raw_scales"].append(raw_scales)
            parts["opacity"].append(raw_opacity * coefficient)
            parts["raw_opacity"].append(raw_opacity)
            parts["psi_ref"].append(layer.psi_ref)
            parts["node_ids"].append(layer.node_ids)
            parts["parent_ids"].append(layer.parent_ids)
            slices.append(slice(start, start + n))
            start += n
        # Topology is invariant between density events. Cache on buffer identity
        # and mutation version, retaining references to prevent identity reuse.
        refs = tuple(t for lid in self.layers[:top+1] for t in (lid.node_ids, lid.parent_ids))
        key = (top, tuple((id(t), t._version, t.numel()) for t in refs))
        cache = getattr(self, "_topology_view_cache", None)
        if cache is None or cache[0] != key:
            ids = torch.cat(parts["node_ids"])
            parents = torch.cat(parts["parent_ids"])
            sorted_ids, order = torch.sort(ids)
            where = torch.searchsorted(sorted_ids, parents)
            valid = (parents >= 0) & (where < len(ids))
            valid &= sorted_ids[where.clamp_max(len(ids)-1)] == parents
            parent_index = torch.full_like(ids, -1)
            parent_index[valid] = order[where[valid]]
            levels = torch.cat([torch.full((len(lid.xyz),), i, device=ids.device, dtype=torch.int64)
                                for i, lid in enumerate(self.layers[:top+1])])
            sentinel = top+1
            first = torch.full_like(ids, sentinel)
            first.scatter_reduce_(0, parent_index[valid], levels[valid], reduce="amin", include_self=True)
            first[first == sentinel] = -1
            previous = torch.full_like(ids, -1)
            following = torch.full_like(ids, -1)
            # Cohorts are direct children with the same parent AND birth stage.
            # A later cohort must not overlap fully with an earlier cohort.
            for level in range(1, top+1):
                at_level = valid & (levels == level)
                prior_by_parent = torch.full_like(ids, -1)
                later_by_parent = torch.full_like(ids, sentinel)
                prior_by_parent.scatter_reduce_(
                    0, parent_index[valid], torch.where(levels[valid] < level, levels[valid], -1),
                    reduce="amax", include_self=True)
                later_by_parent.scatter_reduce_(
                    0, parent_index[valid], torch.where(levels[valid] > level, levels[valid], sentinel),
                    reduce="amin", include_self=True)
                p = parent_index[at_level]
                prior = prior_by_parent[p]
                previous[at_level] = torch.where(prior >= 0, prior, levels[p])
                later = later_by_parent[p]
                following[at_level] = torch.where(later < sentinel, later, -1)
            topology = dict(node_levels=levels, parent_index=parent_index,
                            first_child_level=first, previous_sibling_level=previous,
                            next_sibling_level=following)
            cache = (key, ids, parents, valid, first >= 0, refs, topology)
            self._topology_view_cache = cache
        return {
            "xyz": torch.cat(parts["xyz"], dim=0),
            "scales": torch.cat(parts["scales"], dim=0),
            "raw_scales": torch.cat(parts["raw_scales"], dim=0),
            "rotations": torch.cat(parts["rotations"], dim=0),
            "opacity": torch.cat(parts["opacity"], dim=0),
            "raw_opacity": torch.cat(parts["raw_opacity"], dim=0),
            "sh": torch.cat(parts["sh"], dim=0),
            "psi_ref": torch.cat(parts["psi_ref"], dim=0),
            "node_ids": cache[1],
            "parent_ids": cache[2],
            "has_parent": cache[3],
            "has_child": cache[4],
            "layer_slices": slices,
            **cache[6],
        }

    # -- optimization ------------------------------------------------------- #

    def make_optimizer(
        self,
        position_lr: float,
        feature_lr: float,
        opacity_lr: float,
        scaling_lr: float,
        rotation_lr: float,
    ) -> torch.optim.Adam:
        """Build an Adam optimizer over the active layer's parameters only.

        One parameter group per tensor, named exactly ``xyz``, ``sh``,
        ``log_scales``, ``rotations``, ``opacity_logits`` (group ``"name"``
        keys) with the respective learning rates; frozen levels are never
        part of the optimizer.  Group learning rates follow the arguments
        literally -- scale ``position_lr`` by the scene size at the call site
        if desired.  Returns the optimizer (not stored on the model); rebuild
        it with this method after every :meth:`add_level`.
        """
        layer = self._active()
        groups = [
            {"params": [layer.xyz], "lr": float(position_lr), "name": "xyz"},
            {"params": [layer.sh], "lr": float(feature_lr), "name": "sh"},
            {"params": [layer.log_scales], "lr": float(scaling_lr), "name": "log_scales"},
            {"params": [layer.rotations], "lr": float(rotation_lr), "name": "rotations"},
            {"params": [layer.opacity_logits], "lr": float(opacity_lr), "name": "opacity_logits"},
        ]
        return torch.optim.Adam(groups, lr=0.0, eps=1e-15)

    # -- filter maintenance (RaDe-GS needle-artifact repair) ---------------- #

    def update_filter(self, cameras: Sequence[Any]) -> None:
        """Re-anchor the active layer's ``filter_3d`` buffer only.

        Recomputes the per-row sampling-filter radius with
        :func:`sampling_filter` over ``cameras`` (the active level's own
        training cameras) and stores it in the active layer's persistent
        ``filter_3d`` buffer.  Frozen layers keep their creation-time filter
        untouched -- this method never mutates them.  A truly empty active
        set is a silent no-op (nothing to anchor until densify seeds it).
        Otherwise raises ``ValueError`` (from ``sampling_filter``) when no
        active point is visible in any camera; the active buffer is then
        left unchanged.

        Args:
            cameras: the active level's training cameras (non-empty).
        """
        layer = self._active()
        self._check_stage_use(cameras)
        if int(layer.xyz.shape[0]) == 0:
            return  # truly empty active set: nothing to anchor yet
        layer.filter_3d = sampling_filter(layer.xyz, cameras)

    def reset_opacity(
        self, optimizer: torch.optim.Optimizer, maximum: float = 0.01
    ) -> None:
        """Cap the active layer's *filtered* opacity; reset only its moments.

        RaDe-GS-style reset (official vendor ``reset_opacity``): the filtered
        opacity ``sigmoid(logits) * filter_coef`` is capped at ``maximum``
        and the raw logits are set to the exact inverse-compensated target
        ``min(sigmoid(logits), maximum / filter_coef)`` (equal to
        ``min(filtered, maximum) / filter_coef`` for a positive coefficient),
        so the filtered opacity after the reset never exceeds ``maximum``.
        Rows already below the cap retain their original finite logits,
        including zero-contribution and sigmoid-saturated rows. Only rows
        actually reduced need inverse compensation, avoiding logit(1).

        The active layer's ``opacity_logits`` parameter is replaced inside
        ``optimizer`` (stale-optimizer groups are rejected) and only its Adam
        ``exp_avg`` / ``exp_avg_sq`` moments are zeroed -- the shared step
        counter is kept, as in the official implementation.  Every other
        optimizer group and every frozen layer is untouched.

        Args:
            optimizer: the Adam from :meth:`make_optimizer` for the active
                layer.
            maximum: positive normal float32 cap smaller than one.
        """
        layer = self._active()
        maximum = float(maximum)
        if not math.isfinite(maximum) or not (torch.finfo(torch.float32).tiny <= maximum < 1.0):
            raise ValueError(f"maximum must be representable as positive float32 and < 1, got {maximum}")
        self._require_optimizer(optimizer)
        with torch.no_grad():
            coef = layer.filter_coef()
            raw = torch.sigmoid(layer.opacity_logits.detach())
            capped = (maximum / coef).clamp_max(1.0 - torch.finfo(raw.dtype).eps)
            new_logits = torch.where(
                raw * coef > maximum, _inverse_sigmoid(capped),
                layer.opacity_logits.detach(),
            ).contiguous()
        new_param = nn.Parameter(
            new_logits, requires_grad=layer.opacity_logits.requires_grad
        )

        def reset_moments(state: Dict[str, Any]) -> None:
            if "exp_avg" in state:
                state["exp_avg"].zero_()
                state["exp_avg_sq"].zero_()

        self._swap_active_param(
            optimizer, layer, "opacity_logits", new_param, reset_moments
        )

    # -- optimizer-state plumbing ------------------------------------------- #

    @staticmethod
    def _find_optimizer_group(optimizer: torch.optim.Optimizer, name: str) -> Dict[str, Any]:
        for group in optimizer.param_groups:
            if group.get("name") == name:
                return group
        raise ValueError(
            f"optimizer has no group named {name!r}; rebuild it with make_optimizer()"
        )

    @staticmethod
    def _require_optimizer(optimizer: Optional[torch.optim.Optimizer]) -> torch.optim.Optimizer:
        if optimizer is None:
            raise ValueError(
                "densify changed the active layer topology but received "
                "optimizer=None; pass the Adam returned by make_optimizer()"
            )
        return optimizer

    def _swap_active_param(
        self,
        optimizer: torch.optim.Optimizer,
        layer: GaussianLayer,
        name: str,
        new_param: nn.Parameter,
        prep_state,
    ) -> None:
        """Replace one active-layer parameter inside ``optimizer``.

        Optimizer moments follow the rows: ``prep_state(state)`` mutates the
        existing Adam ``exp_avg``/``exp_avg_sq`` (row slice or zero-pad) and
        the state dict is re-keyed to the new parameter object.  Raises when
        the optimizer group references a stale parameter (built before the
        current ``add_level``).
        """
        group = self._find_optimizer_group(optimizer, name)
        old = group["params"][0]
        if old is not getattr(layer, name):
            raise ValueError(
                f"optimizer group {name!r} does not reference the active layer's "
                "parameter (stale optimizer): call make_optimizer() after add_level()"
            )
        state = optimizer.state.pop(old, None)
        group["params"][0] = new_param
        if state is not None:
            if prep_state is not None:
                prep_state(state)
            optimizer.state[new_param] = state
        setattr(layer, name, new_param)

    def _remove_active_rows(
        self,
        optimizer: Optional[torch.optim.Optimizer],
        layer: GaussianLayer,
        keep: torch.Tensor,
    ) -> None:
        """Drop rows where ``keep`` is False from the active layer + optimizer.

        Masks every row tensor and registered row buffer (``psi_ref``,
        ``filter_3d``, ``node_ids``, ``parent_ids``) of the layer;
        optimizer moments follow the rows.
        """
        keep = keep.reshape(-1).to(layer.xyz.device)
        if int(keep.shape[0]) != int(layer.xyz.shape[0]):
            raise ValueError(
                f"row mask length {int(keep.shape[0])} != active layer "
                f"points {int(layer.xyz.shape[0])}"
            )
        self._require_optimizer(optimizer)

        def prep_slice(state: Dict[str, Any]) -> None:
            if "exp_avg" in state:
                state["exp_avg"] = state["exp_avg"][keep]
                state["exp_avg_sq"] = state["exp_avg_sq"][keep]

        for name in _PARAM_NAMES:
            param = getattr(layer, name)
            new_param = nn.Parameter(
                param.detach()[keep].contiguous(), requires_grad=param.requires_grad
            )
            self._swap_active_param(optimizer, layer, name, new_param, prep_slice)
        for name in _BUFFER_NAMES:
            buf = getattr(layer, name)
            setattr(layer, name, buf[keep].contiguous())

    def _append_active_rows(
        self,
        optimizer: Optional[torch.optim.Optimizer],
        layer: GaussianLayer,
        extension: Dict[str, torch.Tensor],
    ) -> None:
        """Append rows (children) to the active layer with zeroed moments.

        ``extension`` must carry every param and registered row buffer;
        optimizer moments are zero-padded for the new rows.
        """
        self._require_optimizer(optimizer)
        rows = int(extension["xyz"].shape[0])
        for name in _PARAM_NAMES + _BUFFER_NAMES:
            ext = extension.get(name)
            if ext is None:
                raise ValueError(f"append extension is missing {name!r}")
            if int(ext.shape[0]) != rows:
                raise ValueError(
                    f"extension {name} row count {int(ext.shape[0])} does not "
                    f"match xyz extension {rows}"
                )

        def prep_cat(state: Dict[str, Any]) -> None:
            if "exp_avg" in state:
                state["exp_avg"] = torch.cat(
                    [state["exp_avg"], torch.zeros_like(extension[name])], dim=0
                )
                state["exp_avg_sq"] = torch.cat(
                    [state["exp_avg_sq"], torch.zeros_like(extension[name])], dim=0
                )

        for name in _PARAM_NAMES:
            param = getattr(layer, name)
            new_param = nn.Parameter(
                torch.cat([param.detach(), extension[name]], dim=0).contiguous(),
                requires_grad=param.requires_grad,
            )
            self._swap_active_param(optimizer, layer, name, new_param, prep_cat)
        for name in _BUFFER_NAMES:
            buf = getattr(layer, name)
            setattr(
                layer, name,
                torch.cat([buf, extension[name]], dim=0).contiguous(),
            )

    # -- densification ------------------------------------------------------ #

    @staticmethod
    def _split_rows(src: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Standard 3DGS split children of ``src`` rows (pure tensor math).

        Each source row becomes ``_SPLIT_CHILDREN`` children: centers offset
        by a zero-mean gaussian sample scaled by the parent's physical scale
        and rotated into world space by the parent's (normalized) rotation,
        physical scales divided by ``_SPLIT_SCALE_SHRINK``; SH/rotation/
        opacity cloned.  Nothing is mutated.
        """
        rep = lambda t: t.repeat_interleave(_SPLIT_CHILDREN, dim=0)  # noqa: E731
        stds = torch.exp(src["log_scales"]).repeat_interleave(_SPLIT_CHILDREN, dim=0)
        frame = _quaternion_to_matrix(src["rotations"]).repeat_interleave(
            _SPLIT_CHILDREN, dim=0
        )
        eps = torch.randn_like(stds)
        children = {
            "xyz": rep(src["xyz"])
            + torch.bmm(frame, (eps * stds).unsqueeze(-1)).squeeze(-1),
            "sh": rep(src["sh"]),
            "log_scales": rep(src["log_scales"]) - _LOG_SPLIT_SHRINK,
            "rotations": rep(src["rotations"]),
            "opacity_logits": rep(src["opacity_logits"]),
        }
        return {name: t.contiguous() for name, t in children.items()}

    @torch.no_grad()
    def densify(
        self,
        optimizer: Optional[torch.optim.Optimizer],
        grads: torch.Tensor,
        threshold: float,
        max_points: int,
        cameras: Sequence[Any],
        min_opacity: float = 0.005,
        scene_extent: float = 1.0,
        percent_dense: float = 0.01,
        split_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, int]:
        """Graph-aware densification driven by screen-space gradients.

        ``grads`` holds one nonnegative finite screen-space gradient norm per
        MODEL row (fixed AND active levels, in :meth:`tensors` row order) --
        frozen geometry is never optimized, but its gradients still trigger
        the birth of children.  Rows with ``grads >= threshold`` are
        candidates; the standard size decision splits candidates whose raw
        max scale exceeds ``percent_dense * scene_extent`` and clones the
        rest.

        Fixed rows are immutable: a fixed split appends two children to the
        active level under the source's node id (+2 against the budget), a
        fixed clone appends one child under the source's node id (+1).  An
        active split removes the source row and appends two children that
        inherit its parent link (+1 net); an active clone appends one sibling
        under the source's parent link (+1; reading the paper's "child of the
        parent primitive" as the tree parent -- same-level edges are
        impossible -- is the documented unresolved supplement detail).
        Same-level edges never exist,
        so level-0 children are roots.  ``max_points`` caps the active-level
        total; candidates are prioritized by descending screen gradient. Net
        growth costs are charged before allocation (a fixed parent may grow
        children even if it already has some).

        New rows take fresh node ids from the allocator -- which is
        re-synchronized above every issued id first, so hand-built models
        never collide -- plus re-anchored ``psi_ref`` and their own
        ``filter_3d`` over ``cameras``; a child set that no camera sees
        falls back to the full surviving scene's depth envelope (via
        :func:`sampling_filter`).  Active survivors keep
        their Adam moments bit-for-bit; children start from zeroed moments.
        Afterwards active rows with ``sigmoid(opacity) < min_opacity`` are
        pruned (derived ``has_child`` flags follow automatically); fixed rows
        are never pruned.  When no gradient qualifies, nothing is mutated:
        an empty active level stays empty instead of receiving bogus
        fallback seeds.

        Args:
            optimizer: the Adam from :meth:`make_optimizer` for the active
                level; required whenever the topology changes.
            grads: gradient norm per model row (length = fixed + active).
            threshold: candidate when ``grads >= threshold``.
            max_points: growth cap on the active level total (>= 1).
            cameras: the active level's training cameras (re-anchor child
                ``psi_ref``/``filter_3d``).
            min_opacity: prune threshold on sigmoid opacity, in [0, 1).
            scene_extent: scene size for the split/clone size decision (> 0).
            percent_dense: split size threshold as a fraction of it (>= 0).
            split_mask: optional per-model-row boolean split decision from observed
                projected footprints; otherwise use the scene-unit size threshold.

        Returns:
            ``{}`` when nothing changed; otherwise a JSON-serializable dict
            of python ints: ``fixed_split``, ``fixed_clone``,
            ``active_split``, ``active_clone``, ``pruned`` and ``points``
            (final active count).
        """
        if self.topology_mode != "tree":
            raise ValueError(
                "legacy_flat models carry no ancestry and are diagnostic-only; "
                "densify needs a tree built by from_points()/add_level()"
            )
        layer = self._active()
        self._check_stage_use(cameras)
        sizes = [int(lid.xyz.shape[0]) for lid in self.layers]
        total = sum(sizes)
        g = torch.as_tensor(grads).detach().reshape(-1)
        if int(g.numel()) != total:
            raise ValueError(
                f"grads length {int(g.numel())} does not match the model total "
                f"{total} rows (fixed + active); reset accumulators after every "
                "densify call"
            )
        if not bool(torch.isfinite(g).all().item()):
            raise ValueError("grads must be finite screen-space gradient norms")
        if bool((g < 0).any().item()):
            raise ValueError("grads must be nonnegative screen-space gradient norms")
        threshold = float(threshold)
        if not math.isfinite(threshold):
            raise ValueError(f"threshold must be finite, got {threshold}")
        max_points = int(max_points)
        if max_points < 1:
            raise ValueError(f"max_points must be >= 1, got {max_points}")
        min_opacity = float(min_opacity)
        if not (0.0 <= min_opacity < 1.0):
            raise ValueError(f"min_opacity must be in [0, 1), got {min_opacity}")
        scene_extent = float(scene_extent)
        percent_dense = float(percent_dense)
        if not math.isfinite(scene_extent) or scene_extent <= 0.0:
            raise ValueError(f"scene_extent must be finite and > 0, got {scene_extent}")
        if not math.isfinite(percent_dense) or percent_dense < 0.0:
            raise ValueError(f"percent_dense must be finite and >= 0, got {percent_dense}")
        self._sync_allocator()
        active_idx = self.active_level
        size_gate = percent_dense * scene_extent
        remaining = max_points - sizes[active_idx]
        fixed_split: List[Optional[torch.Tensor]] = [None] * active_idx
        fixed_clone: List[Optional[torch.Tensor]] = [None] * active_idx
        act_split: Optional[torch.Tensor] = None
        act_clone: Optional[torch.Tensor] = None
        if split_mask is None:
            all_big = torch.cat([
                torch.exp(lid.log_scales.detach()).amax(dim=1) > size_gate
                for lid in self.layers
            ])
        else:
            all_big = torch.as_tensor(split_mask, device=layer.xyz.device)
            if all_big.dtype != torch.bool or all_big.shape != (total,):
                raise ValueError("split_mask must contain one boolean per model row")
        g = g.to(layer.xyz.device)
        candidates = torch.where((g >= threshold) & (g > 0))[0]
        candidates = candidates[torch.argsort(g[candidates], descending=True, stable=True)]
        fixed_count = sum(sizes[:active_idx])
        costs = torch.where((candidates < fixed_count) & all_big[candidates], 2, 1)
        budget = max(remaining, 0)
        cumulative = costs.cumsum(0)
        accepted = cumulative <= budget
        used = int(costs[accepted].sum())
        # If the first rejected split costs two but only one slot remains,
        # the next one-slot operation can still use that slot.
        if budget-used == 1:
            extra = torch.where((~accepted) & (costs == 1))[0]
            if extra.numel():
                accepted[extra[0]] = True
        taken = torch.zeros(total, dtype=torch.bool, device=layer.xyz.device)
        taken[candidates[accepted]] = True
        offset = 0
        for idx, n in enumerate(sizes):
            local = taken[offset:offset+n]
            big = all_big[offset:offset+n]
            split_rows = torch.where(local & big)[0]
            clone_rows = torch.where(local & ~big)[0]
            if idx < active_idx:
                fixed_split[idx], fixed_clone[idx] = split_rows, clone_rows
            else:
                act_split, act_clone = split_rows, clone_rows
            offset += n

        def gather(pairs: List[Tuple[GaussianLayer, torch.Tensor]]) -> Optional[Dict[str, torch.Tensor]]:
            if not any(int(rows.numel()) for _, rows in pairs):
                return None
            return {
                name: torch.cat(
                    [getattr(lid, name).detach()[rows]
                     for lid, rows in pairs if int(rows.numel())],
                    dim=0,
                ).contiguous()
                for name in _PARAM_NAMES
            }

        split_pairs = [(self.layers[i], rows) for i, rows in enumerate(fixed_split)
                       if rows is not None]
        clone_pairs = [(self.layers[i], rows) for i, rows in enumerate(fixed_clone)
                       if rows is not None]
        if act_split is not None:
            split_pairs.append((layer, act_split))
        if act_clone is not None:
            clone_pairs.append((layer, act_clone))
        split_src = gather(split_pairs)
        clone_src = gather(clone_pairs)

        def parents_of(pairs: List[Tuple[GaussianLayer, torch.Tensor]]) -> torch.Tensor:
            # Fixed clone/split children hang under the source node; active
            # children inherit the source's own parent link (same-level
            # edges never exist).
            return torch.cat(
                [lid.node_ids[rows] if lid.level < active_idx else layer.parent_ids[rows]
                 for lid, rows in pairs if int(rows.numel())],
                dim=0,
            )

        blocks: List[Dict[str, torch.Tensor]] = []
        parent_blocks: List[torch.Tensor] = []
        if split_src is not None:
            blocks.append(self._split_rows(split_src))
            parent_blocks.append(
                parents_of(split_pairs).repeat_interleave(_SPLIT_CHILDREN, dim=0)
            )
        if clone_src is not None:
            blocks.append({name: src.contiguous() for name, src in clone_src.items()})
            parent_blocks.append(parents_of(clone_pairs))

        stats: Dict[str, int] = {}
        if blocks:
            ext = {name: torch.cat([b[name] for b in blocks], dim=0) for name in _PARAM_NAMES}
            n_children = int(ext["xyz"].shape[0])
            ext["parent_ids"] = torch.cat(parent_blocks, dim=0).contiguous()
            ext["node_ids"] = self._alloc_ids(n_children, device=layer.xyz.device)
            ext["psi_ref"] = _creation_psi(ext["xyz"], cameras)
            keep = torch.ones(sizes[active_idx], dtype=torch.bool, device=layer.xyz.device)
            if act_split is not None:
                keep[act_split] = False
            try:  # child subset first; an all-invisible set needs the scene
                ext["filter_3d"] = sampling_filter(ext["xyz"], cameras)
            except ValueError:
                scene = torch.cat(
                    [lid.xyz.detach() for lid in self.layers[:active_idx]]
                    + [layer.xyz.detach()[keep], ext["xyz"]],
                    dim=0,
                )
                ext["filter_3d"] = sampling_filter(scene, cameras)[
                    int(scene.shape[0]) - n_children:
                ]
            if not bool(keep.all().item()):
                self._remove_active_rows(optimizer, layer, keep)
            self._append_active_rows(optimizer, layer, ext)
            stats = {
                "fixed_split": sum(int(r.numel()) for r in fixed_split if r is not None),
                "fixed_clone": sum(int(r.numel()) for r in fixed_clone if r is not None),
                "active_split": 0 if act_split is None else int(act_split.numel()),
                "active_clone": 0 if act_clone is None else int(act_clone.numel()),
            }

        opacity = torch.sigmoid(layer.opacity_logits.detach()).reshape(-1)
        low = opacity < min_opacity
        pruned = int(low.sum().item())
        if pruned:
            self._remove_active_rows(optimizer, layer, ~low)
        if not stats and not pruned:
            return {}
        stats["pruned"] = pruned
        stats["points"] = int(layer.xyz.shape[0])
        return stats

    # -- checkpointing ------------------------------------------------------ #

    def save(
        self,
        path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metadata: Optional[Any] = None,
    ) -> None:
        """Atomically write a versioned, tensor-only torch checkpoint.

        Payload (version 3) contains, per layer, every parameter plus the
        ``psi_ref``, ``filter_3d``, ``node_ids`` and ``parent_ids`` buffers
        (never optimizer/module objects), the explicit scalar model state
        (``degree``, ``step_scale``, ``active_level``), the tree state
        (``next_node_id`` allocator, ``topology_mode``), per-layer
        ``level``/``frozen`` flags, auto model metadata (including the psi
        reference policy and the topology mode) and the caller's
        ``metadata`` (plain data, e.g. python RNG states / tensors /
        JSON-ish config; objects are neither stored nor expected back).
        With ``optimizer`` given, its ``state_dict()`` is stored under
        ``"optimizer"``.  The allocator is re-synchronized here, so even
        hand-assembled layers with deterministic default ids leave a valid
        ``next_node_id`` in the file.

        The file is written to a temporary sibling and ``os.replace``-d into
        place, so a crash never leaves a truncated checkpoint at ``path``.
        """
        self._sync_allocator()
        self.validate_topology()
        if self.weight_policy == "interval":
            self.require_stage_records()
            validate_stage_records(self.stage_records, self.step_scale)
        payload: Dict[str, Any] = {
            "format": _FORMAT_TAG,
            "version": _PAYLOAD_VERSION,
            "degree": int(self.degree),
            "step_scale": float(self.step_scale),
            "active_level": int(self.active_level),
            "next_node_id": int(self.next_node_id),
            "topology_mode": self.topology_mode,
            "stage_records": self.stage_records,
            "weight_policy": self.weight_policy,
            "layers": [
                {
                    "level": layer.level,
                    "frozen": bool(layer.frozen),
                    "xyz": layer.xyz.detach().cpu(),
                    "sh": layer.sh.detach().cpu(),
                    "log_scales": layer.log_scales.detach().cpu(),
                    "rotations": layer.rotations.detach().cpu(),
                    "opacity_logits": layer.opacity_logits.detach().cpu(),
                    "psi_ref": layer.psi_ref.detach().cpu(),
                    "filter_3d": layer.filter_3d.detach().cpu(),
                    "node_ids": layer.node_ids.detach().cpu(),
                    "parent_ids": layer.parent_ids.detach().cpu(),
                }
                for layer in self.layers
            ],
            "model_metadata": {
                "psi_reference_policy": _PSI_REFERENCE_POLICY,
                "layer_levels": [int(layer.level) for layer in self.layers],
                "points_per_layer": [int(layer.xyz.shape[0]) for layer in self.layers],
                "frozen_per_layer": [bool(layer.frozen) for layer in self.layers],
                "topology_mode": self.topology_mode,
                "weight_policy": self.weight_policy,
                "stage_scales": [stage["scale"] for stage in self.stage_records],
            },
            "metadata": metadata,
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()

        path = os.fspath(path)
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory
        )
        os.close(fd)
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @classmethod
    def load(cls, path, device: str = "cuda") -> Tuple["GaussianLoD", Dict[str, Any], Any]:
        """Load a checkpoint written by :meth:`save`.

        Args:
            path: checkpoint file.
            device: where to place the model tensors (``torch.load``
                ``map_location``).

        Returns:
            ``(model, metadata, optimizer_state)``.  ``model`` reproduces the
            saved tensors exactly (same values across the checkpoint), with
            frozen flags restored per layer, per-layer ``filter_3d`` restored
            and validated, and the active layer trainable.  Version-3
            payloads restore the tree topology (``node_ids``/``parent_ids``,
            the ``next_node_id`` allocator and ``topology_mode``) and are
            re-validated with :meth:`validate_topology`.  Legacy payloads
            are adapted without inventing ancestry: version-1 checkpoints
            load with exactly zero ``filter_3d`` buffers and
            ``metadata["model_lod"]["legacy_unfiltered"] = True`` under
            ``topology_mode='legacy_flat'`` (diagnostic renders only); a
            version-2 SINGLE-L0 payload is promoted to a valid tree of
            allocator-issued roots, while a multi-level version-2 payload
            stays ``'legacy_flat'``.  ``metadata`` is the caller metadata
            dict merged with the auto model metadata under the reserved
            ``"model_lod"`` key (so e.g. ``metadata["dataset"]`` works when
            a dict was saved); non-dict metadata is preserved under
            ``"user_metadata"``.  ``metadata`` is ``{}`` when the checkpoint
            holds none.  ``optimizer_state`` is the saved Adam state dict,
            or ``None``.

        Raises:
            ValueError: unsupported format/version or an inconsistent/corrupt
                payload (bad level layout, empty level-0 layer, non-finite or
                non-positive ``psi_ref``, missing/invalid ``filter_3d``,
                mismatched SH widths, invalid node ids or parent links).
        """
        payload = torch.load(path, map_location=device, weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("checkpoint is not a gaussianzoom_lod model payload")
        if payload.get("format") != _FORMAT_TAG:
            raise ValueError(
                f"checkpoint format {payload.get('format')!r} is not "
                f"{_FORMAT_TAG!r}; refusing to load"
            )
        version = payload.get("version")
        if version not in (_LEGACY_PAYLOAD_VERSION, 2, 3, _PAYLOAD_VERSION):
            raise ValueError(
                f"checkpoint version {version} is unsupported (this build reads versions 1 to 4)"
            )
        legacy_unfiltered = version == _LEGACY_PAYLOAD_VERSION
        degree = int(payload["degree"])
        step_scale = float(payload["step_scale"])
        active_level = int(payload["active_level"])
        raw_layers = payload["layers"]
        if not isinstance(raw_layers, list) or len(raw_layers) < 1:
            raise ValueError("checkpoint has no layers")
        if active_level != len(raw_layers) - 1:
            raise ValueError(
                f"checkpoint active_level {active_level} inconsistent with "
                f"{len(raw_layers)} layers"
            )
        topology_mode: Optional[str] = None
        if version >= 3:
            topology_mode = payload.get("topology_mode")
            if topology_mode not in ("tree", "legacy_flat"):
                raise ValueError(
                    f"checkpoint topology_mode {topology_mode!r} is invalid"
                )
            if "next_node_id" not in payload:
                raise ValueError("version-3 payload is missing next_node_id")
        model = cls(degree=degree, step_scale=step_scale)
        sh_channels: Optional[int] = None
        for idx, raw in enumerate(raw_layers):
            level = int(raw["level"])
            if level != idx:
                raise ValueError(
                    f"checkpoint layer {idx} has level {level}; expected "
                    f"contiguous levels 0..{len(raw_layers) - 1}"
                )
            n = int(raw["xyz"].shape[0])
            if n < 1 and (version < 3 or level == 0):
                raise ValueError(
                    f"checkpoint layer {level} is empty (only a non-L0 "
                    "version-3 layer may await gradient bootstrap)"
                )
            channels = int(raw["sh"].shape[1])
            if channels != (degree + 1) ** 2:
                raise ValueError("checkpoint SH width does not match declared degree")
            if bool(raw["frozen"]) != (idx < active_level):
                raise ValueError("checkpoint must freeze exactly the older layers")
            if sh_channels is None:
                sh_channels = channels
            elif channels != sh_channels:
                raise ValueError("checkpoint layers disagree on SH width")
            if legacy_unfiltered:
                # Pre-repair payloads carry no filter: exactly zero filters
                # reproduce the old unfiltered renders (identity coefficient).
                filter_3d = torch.zeros(
                    (n, 1), dtype=torch.float32, device=raw["xyz"].device
                )
            elif "filter_3d" not in raw:
                raise ValueError(
                    f"checkpoint layer {level}: payload is missing filter_3d "
                    "(corrupt checkpoint)"
                )
            else:
                filter_3d = raw["filter_3d"]
            if version >= 3:
                if "node_ids" not in raw or "parent_ids" not in raw:
                    raise ValueError(
                        f"checkpoint layer {level}: version-3 payload is "
                        "missing node_ids/parent_ids (corrupt checkpoint)"
                    )
                node_ids: Optional[torch.Tensor] = raw["node_ids"]
                parent_ids: Optional[torch.Tensor] = raw["parent_ids"]
            else:
                # Legacy payloads carry no ids: issue fresh monotonic roots.
                node_ids = model._alloc_ids(n, device=raw["xyz"].device)
                parent_ids = None
            layer = GaussianLayer(
                level=level,
                xyz=raw["xyz"],
                sh=raw["sh"],
                log_scales=raw["log_scales"],
                rotations=raw["rotations"],
                opacity_logits=raw["opacity_logits"],
                psi_ref=raw["psi_ref"],
                frozen=bool(raw["frozen"]),
                filter_3d=filter_3d,
                node_ids=node_ids,
                parent_ids=parent_ids,
            )
            model.layers.append(layer)
        model.active_level = active_level
        if version >= 3:
            model.next_node_id = int(payload["next_node_id"])
            model.topology_mode = str(topology_mode)
        else:
            # Version-2 single-L0 payloads promote to a valid tree of roots
            # so the freshly repaired L0 artifact trains; every other legacy
            # payload stays ancestry-free (diagnostics only, never trained).
            model.topology_mode = (
                "tree" if version == 2 and len(raw_layers) == 1 else "legacy_flat"
            )
        model.validate_topology()
        if version >= 4:
            model.weight_policy = payload.get("weight_policy", "")
            if model.weight_policy not in ("interval", "legacy_tent"):
                raise ValueError("Invalid checkpoint weight policy")
            model.stage_records = payload.get("stage_records", [])
            if model.weight_policy == "interval":
                model.require_stage_records()
                validate_stage_records(model.stage_records, model.step_scale)
            elif model.stage_records:
                raise ValueError("Legacy diagnostic checkpoints cannot claim validated stage records")
        else:
            model.weight_policy = "legacy_tent"
            model.stage_records = []
        metadata: Dict[str, Any] = {}
        model_metadata = payload.get("model_metadata") or {}
        if not isinstance(model_metadata, dict):
            model_metadata = {}
        if legacy_unfiltered:
            model_metadata["legacy_unfiltered"] = True
        model_metadata["topology_mode"] = model.topology_mode
        model_metadata["weight_policy"] = model.weight_policy
        model_metadata["stage_scales"] = [stage["scale"] for stage in model.stage_records]
        user_metadata = payload.get("metadata")
        if isinstance(user_metadata, dict):
            metadata.update(user_metadata)
        elif user_metadata is not None:
            metadata["user_metadata"] = user_metadata
        metadata["model_lod"] = model_metadata
        optimizer_state = payload.get("optimizer")
        return model, metadata, optimizer_state
