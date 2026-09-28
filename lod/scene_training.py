"""Manifest-driven MULTI-VIEW full-scene RaDe LoD trainer (Skyfall scene enhancement).

Shared training kernel for ``scripts/train_scene_lod.py``. One process trains
every zoom level of a ``skyfall_scene_zoom`` supervision manifest sequentially
on the existing frozen-import / ``add_level`` / ``densify_with_appearance`` /
RaDe-GS ``lod=True`` path, with NO single-ROI shortcut:

- Base cameras are rebuilt from the manifest's serialized ``CameraSnapshot``
  dicts at their serialized raster (R/T, cx/cy, znear/zfar preserved). They
  never own image tensors; targets stay CPU/disk-backed and only the active
  target moves to GPU per step.
- The frozen L0 is imported over the STABLE physical base-camera reference --
  preferably the checkpoint dataset's own TRAIN cameras (``Scene.getTrainCameras()``
  passed by the caller, identical to the generation path), else the manifest's
  native TRAIN views (``source_stage`` ``stage1``/``real_replay``), else the
  deduped manifest physical set -- so ``psi_ref`` birth stats never drift with
  the manifest's changing synthetic FlowEdit camera sets. Every fallback step
  is reported.
- The RaDe-GS sampling filter is the authoritative checkpoint filter_3D
  exactly as persisted next to the source checkpoint; it is recomputed ONLY
  when the checkpoint carries none (native fallback over the same stable
  reference cameras, reported). Generation and training therefore import the
  identical frozen G_R filter and a trained parent loads unchanged.
- A base stage record covering the union of all level ROI aliases is bound once
  via the supported ``GaussianLoD.bind_base_stage`` (aliases share the physical
  calibration, only the name differs), BEFORE any ``add_level``.
- Each zoom level creates its full varied-ROI stage camera set (subset of the
  bound base names, exact focal step) through ``add_level``; the same set drives
  visibility/densification. Vendor stage checks are used as-is.
- Densification ordering is the vendor-sound order: collect screen gradients ->
  ``optimizer.step()`` -> ``densify_with_appearance`` -> ``zero_grad`` next
  iteration (``_swap_active_param``/``_append_active_rows`` never carry ``.grad``).
- ``steps_per_level`` is a documented MINIMUM: the level loop keeps stepping
  round-robin (shuffled) until every current-level sample has been visited at
  least once, even past the requested step count.
- ``level_steps`` overrides the scalar per level (count must match manifest levels);
  ``checkpoint_steps`` persists ``l{level}_step{step:06d}.lod.pt`` bundles after the
  optimizer update + densification + zero_grad (never rebuilding model/optimizer);
  ``densify_until`` pins an ABSOLUTE per-level densify horizon that wins over the
  ``densify_fraction`` derivation.

Optional frequency supervision separates SR detail from the native LR anchor.
"""

from __future__ import annotations

import math
import os
import random
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

import torch.nn.functional as F

from lod.camera import CameraWithStoredCenter, ndc_principal_to_pixel, skyfall_camera_to_lod
from lod.depth import ray_distance_to_z, rln_principal_from_skyfall_camera
from lod.freeze import (
    assert_snapshot_unchanged,
    completed_layers_frozen,
    snapshot_levels,
    tensor_digest,
)
from lod.importer import (
    assert_l0_tensors_match,
    densify_with_appearance,
    import_skyfall_l0,
    load_lod_onto_bundle,
    save_bundle,
    sync_layer_embeddings,
)
from lod.lineage import file_identity, write_json
from lod.path import add_gz_src
from lod.render import appearance_colors_lod, render_lod_appearance
from lod.train_state import densify_spec
from utils.general_utils import PILtoTorch
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.loss_utils import create_window
from utils.zoom_camera import NormalizedROI, zoom_fov, zoom_principal_point
from utils.zoom_mvp_utils import save_tensor_image
MANIFEST_KIND = "skyfall_scene_zoom"

#: Manifest ``source_stage`` values that identify native checkpoint TRAIN
#: views: the stable physical base-camera reference for L0 birth stats.
NATIVE_VIEW_STAGES = ("stage1", "real_replay")



# --------------------------------------------------------------------------- #
# Camera reconstruction (no image tensors on the camera)
# --------------------------------------------------------------------------- #


class RenderCamera:
    """Skyfall-style camera rebuilt from a serialized ``CameraSnapshot``.

    Mirrors ``scene.cameras.Camera`` projection math (getWorld2View2 /
    getProjectionMatrix / focal from FoV, NDC principal point) while carrying NO
    raster tensor: supervision targets are loaded from disk on demand.
    """

    __slots__ = (
        "image_name", "uid", "colmap_id", "R", "T", "FoVx", "FoVy", "cx", "cy",
        "image_width", "image_height", "znear", "zfar",
        "world_view_transform", "projection_matrix", "full_proj_transform",
        "camera_center", "focal_x", "focal_y",
    )

    def __init__(
        self,
        *,
        image_name: str,
        uid: int,
        colmap_id: Any,
        R: Any,
        T: Any,
        fov_x: float,
        fov_y: float,
        cx: float,
        cy: float,
        image_width: int,
        image_height: int,
        znear: float = 0.01,
        zfar: float = 100.0,
        device: Any = "cuda",
    ) -> None:
        self.image_name = str(image_name)
        self.uid = int(uid)
        self.colmap_id = colmap_id
        self.R = np.asarray(R, dtype=np.float64)
        self.T = np.asarray(T, dtype=np.float64).reshape(3)
        self.FoVx = float(fov_x)
        self.FoVy = float(fov_y)
        self.cx = float(cx)
        self.cy = float(cy)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.znear = float(znear)
        self.zfar = float(zfar)
        device = torch.device(device)
        self.world_view_transform = (
            torch.tensor(getWorld2View2(self.R, self.T)).transpose(0, 1).to(device)
        )
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy,
            cx=self.cx, cy=self.cy,
        ).transpose(0, 1).to(device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        tan_fovx = math.tan(self.FoVx / 2.0)
        tan_fovy = math.tan(self.FoVy / 2.0)
        self.focal_x = self.image_width / (2.0 * tan_fovx)
        self.focal_y = self.image_height / (2.0 * tan_fovy)

    def zoomed(
        self, factor: float, center_u: float, center_v: float, image_name: str | None = None
    ) -> "RenderCamera":
        """Pure intrinsic zoom (fixed R/T, same raster), matching ``make_zoom_camera``."""

        roi = NormalizedROI(float(center_u), float(center_v), 1.0 / factor, 1.0 / factor)
        roi.validate_for_zoom(factor)
        fov_x = zoom_fov(self.FoVx, factor)
        fov_y = zoom_fov(self.FoVy, factor)
        cx, cy = zoom_principal_point(self.cx, self.cy, roi, factor)
        return RenderCamera(
            image_name=image_name or f"{self.image_name}_zoom{factor:g}",
            uid=self.uid,
            colmap_id=self.colmap_id,
            R=self.R,
            T=self.T,
            fov_x=fov_x,
            fov_y=fov_y,
            cx=cx,
            cy=cy,
            image_width=self.image_width,
            image_height=self.image_height,
            znear=self.znear,
            zfar=self.zfar,
            device=self.world_view_transform.device,
        )

    def physical_key(self) -> tuple:
        """Identity of the physical viewpoint (pose + intrinsics + raster)."""

        w2c = (
            self.world_view_transform.detach().transpose(0, 1).contiguous().cpu()
            .numpy().astype(np.float64).reshape(-1)
        )
        return (
            int(self.image_width), int(self.image_height),
            round(float(self.focal_x), 6), round(float(self.focal_y), 6),
            round(float(0.5 * self.image_width * (1.0 + self.cx)), 6),
            round(float(0.5 * self.image_height * (1.0 + self.cy)), 6),
            tuple(round(float(v), 7) for v in w2c),
        )


def render_camera_from_snapshot(snapshot: dict, *, device: Any = "cuda") -> RenderCamera:
    return RenderCamera(
        image_name=str(snapshot.get("image_name", "view")),
        uid=int(snapshot.get("uid", -1)),
        colmap_id=snapshot.get("colmap_id"),
        R=snapshot["R"],
        T=snapshot["T"],
        fov_x=float(snapshot["fov_x"]),
        fov_y=float(snapshot["fov_y"]),
        cx=float(snapshot["cx"]),
        cy=float(snapshot["cy"]),
        image_width=int(snapshot["image_width"]),
        image_height=int(snapshot["image_height"]),
        znear=float(snapshot.get("znear", 0.01)),
        zfar=float(snapshot.get("zfar", 100.0)),
        device=device,
    )


# --------------------------------------------------------------------------- #
# Manifest views: physical dedupe
# --------------------------------------------------------------------------- #


@dataclass
class SceneViews:
    """Unique physical base cameras plus the view-id -> physical index map."""

    render_cameras: list[RenderCamera]      # one per unique physical camera
    view_to_physical: dict[str, int]        # manifest view id -> physical index
    lod_cameras: list[Any]                  # vendor pinhole cams of the above


def build_scene_views(manifest: dict, *, max_views: int = 0, device: Any = "cuda") -> SceneViews:
    """Dedupe manifest views by physical camera identity (pose+intrinsics+raster)."""

    key_to_physical: dict[tuple, int] = {}
    render_cameras: list[RenderCamera] = []
    view_to_physical: dict[str, int] = {}
    for view in manifest["views"]:
        view_id = str(view["id"])
        if view_id in view_to_physical:
            continue
        if max_views and len(view_to_physical) >= int(max_views):
            break
        camera = render_camera_from_snapshot(view["camera"], device=device)
        key = camera.physical_key()
        index = key_to_physical.get(key)
        if index is None:
            index = len(render_cameras)
            key_to_physical[key] = index
            camera.image_name = f"phys{index:04d}"
            render_cameras.append(camera)
        view_to_physical[view_id] = index
    if not render_cameras:
        raise ValueError("Supervision manifest produced no usable views.")
    lod_cameras = [
        skyfall_camera_to_lod(camera, gz_root=None, device=None, use_skyfall_center=False)
        for camera in render_cameras
    ]
    return SceneViews(
        render_cameras=render_cameras,
        view_to_physical=view_to_physical,
        lod_cameras=lod_cameras,
    )


# --------------------------------------------------------------------------- #
# Vendor camera helpers (aliases + stage sets)
# --------------------------------------------------------------------------- #


def alias_base_camera(bundle, physical_index: int, name: str):
    """Scale-1 pinhole camera for the base stage record under a unique alias name."""

    base = bundle.lod_cameras[physical_index]
    return replace(base, name=name)


def make_stage_camera(
    bundle, *, alias: str, physical_index: int, zoom: float,
    center_u: float, center_v: float, stored_center,
):
    """Level stage camera: subset-of-base alias name, focal x zoom, same pose.

    Identical math to ``lod.camera.zoom_stage_camera`` (``use_skyfall_center=True``
    so RaDe-GS ``campos`` keeps Skyfall's stored center), with a unique name.
    """

    add_gz_src()
    from gaussianzoom_lod.camera import Camera as LodCamera

    base = bundle.lod_cameras[physical_index]
    width = int(base.width)
    height = int(base.height)
    lod = LodCamera(
        name=alias,
        width=width,
        height=height,
        fx=float(base.fx) * float(zoom),
        fy=float(base.fy) * float(zoom),
        cx=float(zoom) * (float(base.cx) - float(center_u) * width) + width / 2.0,
        cy=float(zoom) * (float(base.cy) - float(center_v) * height) + height / 2.0,
        w2c=base.w2c,
    )

    return CameraWithStoredCenter(lod, stored_center)



def native_reference_cameras(
    manifest: dict,
    scene_views: SceneViews,
    gaussians,
    *,
    native_train_cameras: Sequence[Any] | None = None,
) -> tuple[list, str, str]:
    """Stable physical base-camera reference for the frozen L0 import.

    LoD birth stats (``psi_ref``) and any filter fallback must not depend on
    the synthetic FlowEdit camera sets, which change per manifest and level.
    Preference order: the checkpoint dataset's own TRAIN cameras
    (``native_train_cameras``, exactly what the generation path imports over),
    then the manifest's native TRAIN views (``source_stage`` ``stage1`` /
    ``real_replay``), then the full deduped manifest physical set. Also
    enforces the filter contract: keep the checkpoint's own ``filter_3D``;
    recompute it over the reference cameras ONLY when the checkpoint carries
    none (never silently).

    Returns ``(cameras, filter_source, reference_source)``.
    """

    reference: list
    if native_train_cameras:
        reference = list(native_train_cameras)
        reference_source = "scene_train_cameras"
    else:
        stage_by_view = {
            str(view["id"]): str(view.get("source_stage", "stage1"))
            for view in manifest["views"]
        }
        reference = []
        seen: set[int] = set()
        for view_id, physical in scene_views.view_to_physical.items():
            if stage_by_view.get(view_id) not in NATIVE_VIEW_STAGES or physical in seen:
                continue
            seen.add(physical)
            reference.append(scene_views.render_cameras[physical])
        if reference:
            reference_source = "manifest_native_views"
        else:
            reference = list(scene_views.render_cameras)
            reference_source = "manifest_physicals"
            print(
                "[scene-training] WARNING: no native TRAIN cameras available; "
                "L0 birth stats use the deduped manifest physical cameras as "
                "the base-camera reference."
            )
    filter_source = "checkpoint_ply"
    if getattr(gaussians, "filter_3D", None) is None:
        filter_source = "recomputed_native_reference_cameras"
        print(
            "[scene-training] WARNING: base checkpoint carries no filter_3D; "
            "recomputing it over the stable physical base-camera reference "
            f"({reference_source})."
        )
        gaussians.compute_3D_filter(cameras=reference)
    return reference, filter_source, reference_source



def bind_union_base_stage(bundle, union_cameras) -> None:
    """Register every base name (physical + ROI aliases) once via ``bind_base_stage``."""

    model = bundle.lod
    model.stage_records = []
    model.bind_base_stage(union_cameras)



def manifest_alias_specs(
    manifest: dict, allowed_view_ids: set[str] | None, view_to_physical: dict[str, int]
) -> dict[tuple, str]:
    """Unique ``(physical, zoom, u, v) -> base alias name`` table over all levels."""

    alias_specs: dict[tuple, str] = {}
    for level in sorted(manifest["levels"], key=lambda item: float(item["zoom_factor"])):
        zoom = float(level["zoom_factor"])
        for sample in level["samples"]:
            view_id = str(sample["view_id"])
            if allowed_view_ids is not None and view_id not in allowed_view_ids:
                continue
            roi = sample["roi"]
            u = float(roi["center_x"])
            v = float(roi["center_y"])
            physical = view_to_physical[view_id]
            key = (physical, zoom, round(u, 6), round(v, 6))
            if key not in alias_specs:
                alias_specs[key] = f"p{physical:04d}_z{zoom:g}_u{u:.4f}_v{v:.4f}"
    return alias_specs


def manifest_union_base_cameras(bundle, scene_views: SceneViews, alias_specs: dict[tuple, str]) -> list:
    """Physical base cameras plus one scale-1 alias per unique level ROI."""

    union_cameras = list(scene_views.lod_cameras)

    for (physical, _zoom, _u, _v), name in sorted(alias_specs.items()):
        union_cameras.append(alias_base_camera(bundle, physical, name))
    return union_cameras


def resolve_lr_anchor(
    raw: dict, camera, zoom: float, key: str, *, loss_mode: str,
) -> tuple[str | None, str | None, int | None, int | None]:
    """Read a real crop at camera-raster/zoom, allowing only pixel-grid rounding."""
    if loss_mode == "l1":
        return None, None, None, None
    if not raw.get("lr_image_path"):
        raise ValueError(f"sample {key}: multiscale training requires lr_image_path")
    image_path = os.path.abspath(raw["lr_image_path"])
    mask_path = os.path.abspath(raw["lr_mask_path"]) if raw.get("lr_mask_path") else None
    with Image.open(image_path) as image:
        width, height = image.size
    expected = (int(camera.image_width) / zoom, int(camera.image_height) / zoom)
    if any(abs(actual - want) > 1 for actual, want in zip((width, height), expected)):
        raise ValueError(f"sample {key}: LR raster {width}x{height} differs from real crop {expected}")
    if mask_path:
        with Image.open(mask_path) as mask:
            if mask.size != (width, height):
                raise ValueError(f"sample {key}: LR mask and image rasters differ")
    return image_path, mask_path, width, height


def add_scene_level(bundle, stage_cameras):
    """Vendor ``add_level`` on the full (deduped) stage camera set, plus appearance bookkeeping."""

    model = bundle.lod
    layer = model.add_level(stage_cameras)
    if bundle.appearance.gaussian_embeddings is not None:
        dim = int(bundle.appearance.gaussian_embeddings.shape[1])
        device = bundle.appearance.gaussian_embeddings.device
        while len(bundle.appearance.layer_embeddings) < len(model.layers):
            bundle.appearance.layer_embeddings.append(
                bundle.appearance.gaussian_embeddings.new_zeros((0, dim)).to(device)
            )
    bundle.stage_cameras = list(stage_cameras)
    index = len(model.layers) - 1
    if index == 1:
        bundle.l1_cameras = list(stage_cameras)
    elif index == 2:
        bundle.l2_cameras = list(stage_cameras)
    sync_layer_embeddings(bundle)
    return layer


# --------------------------------------------------------------------------- #
# Appearance embeddings
# --------------------------------------------------------------------------- #


class AppearanceResolver:
    """Train uid -> own frozen embedding; generated views -> native novel-view policy (mean)."""

    def __init__(self, gaussians) -> None:
        self.gaussians = gaussians
        self._cache: dict[Any, Any] = {}

    def for_uid(self, uid: int):
        if not self.gaussians.appearance_enabled:
            return None
        uid = int(uid)
        if uid not in self._cache:
            embeddings = self.gaussians.appearance_embeddings
            if uid >= int(embeddings.shape[0]):
                raise IndexError(
                    f"Train view uid {uid} exceeds appearance embedding count "
                    f"{int(embeddings.shape[0])}."
                )
            self._cache[uid] = embeddings[uid].detach()
        return self._cache[uid]

    def novel(self):
        """Native novel-view policy (gaussian_renderer): row min(6, n-1), never the mean."""

        if not self.gaussians.appearance_enabled:
            return None
        if "novel" not in self._cache:
            embeddings = self.gaussians.appearance_embeddings
            uid = min(6, int(embeddings.shape[0]) - 1)
            self._cache["novel"] = embeddings[uid].detach()
        return self._cache["novel"]

    def for_view(self, appearance_uid) -> Any:
        if appearance_uid is None:
            return self.novel()
        return self.for_uid(appearance_uid)


# --------------------------------------------------------------------------- #
# CPU/disk-backed targets
# --------------------------------------------------------------------------- #


class TargetCache:
    """CPU-backed PNG cache; only the active target is moved to GPU per step."""

    def __init__(self, maxsize: int = 128) -> None:
        self.maxsize = int(maxsize)
        self._store: OrderedDict[tuple, torch.Tensor] = OrderedDict()

    def _get(self, key, loader):
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        value = loader()
        self._store[key] = value
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)
        return value

    def image(self, path: str, width: int, height: int) -> torch.Tensor:
        def loader():
            with Image.open(path) as handle:
                tensor = PILtoTorch(handle.convert("RGB"), (int(width), int(height))).clamp(0.0, 1.0)
            if tensor.shape[0] == 4:
                tensor = tensor[:3]
            return tensor.contiguous()

        return self._get((str(os.path.abspath(path)), int(width), int(height), "rgb"), loader)

    def frequency(self, path: str, width: int, height: int, low_width: int, low_height: int) -> torch.Tensor:
        """Cache signed teacher detail instead of retaining a second RGB copy."""
        def loader():
            with Image.open(path) as handle:
                high = PILtoTorch(handle.convert("RGB"), (width, height)).clamp(0.0, 1.0)
            low = F.interpolate(
                high[None], size=(low_height, low_width), mode="bicubic",
                align_corners=False, antialias=True,
            )[0]
            return frequency_residual(high, low).contiguous()

        return self._get(
            (os.path.abspath(path), width, height, low_width, low_height, "frequency"), loader
        )

    def frequency_mask(self, path: str, width: int, height: int, low_width: int, low_height: int) -> torch.Tensor:
        def loader():
            with Image.open(path) as handle:
                mask = PILtoTorch(handle.convert("L"), (width, height)) > 0.5
            return frequency_support(mask.float(), (low_height, low_width))

        return self._get(
            (os.path.abspath(path), width, height, low_width, low_height, "frequency_mask"), loader
        )

    def mask(self, path: str, width: int, height: int) -> torch.Tensor:
        def loader():
            with Image.open(path) as handle:
                tensor = PILtoTorch(handle.convert("L"), (int(width), int(height)))
            if tensor.shape[0] == 3:
                tensor = tensor[:1]
            return (tensor > 0.5).float().contiguous()

        return self._get((str(os.path.abspath(path)), int(width), int(height), "mask"), loader)


# --------------------------------------------------------------------------- #
# Metrics and PNG evidence
# --------------------------------------------------------------------------- #


def masked_l1(render: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return (render - target).abs().mean()
    keep = mask.to(device=render.device, dtype=render.dtype)
    if keep.shape[0] == 1 and render.shape[0] != 1:
        keep = keep.expand(render.shape[0], -1, -1)
    denominator = keep.sum().clamp_min(1.0)
    return ((render - target).abs() * keep).sum() / denominator


def masked_mse(render: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> float:
    if mask is None:
        mse = (render - target).square().mean()
    else:
        keep = mask.to(device=render.device, dtype=render.dtype)
        if keep.shape[0] == 1 and render.shape[0] != 1:
            keep = keep.expand(render.shape[0], -1, -1)
        mse = ((render - target).square() * keep).sum() / keep.sum().clamp_min(1.0)
    return float(mse.detach().item())


def psnr_value(mse: float) -> float:
    if not math.isfinite(mse) or mse <= 0.0:
        return 60.0 if mse == 0.0 else float("nan")
    return float(-10.0 * math.log10(mse))

# --------------------------------------------------------------------------- #
# Multiscale loss helpers (opt-in via ``loss_mode="multiscale"``)
# --------------------------------------------------------------------------- #


def frequency_residual(high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
    """Signed detail above the LR anchor band; constants are not detail."""
    return high - F.interpolate(
        low[None], size=high.shape[-2:], mode="bicubic",
        align_corners=False, antialias=True,
    )[0]


def frequency_support(mask: torch.Tensor, low_size: tuple[int, int]) -> torch.Tensor:
    """Exclude invalid pixels touched by the bicubic down/up footprint."""
    radius = math.ceil(4 * max(mask.shape[-2] / low_size[0], mask.shape[-1] / low_size[1]))
    invalid = 1.0 - (mask > 0.5).to(dtype=torch.float32)
    touched = F.max_pool2d(invalid[None], 2 * radius + 1, stride=1, padding=radius)[0]
    return (touched == 0).to(dtype=torch.float32)


SSIM_WINDOW_SIZE = 11


@lru_cache(maxsize=8)
def _ssim_window(window_size: int, channel: int, device: torch.device, dtype: torch.dtype):
    return create_window(window_size, channel).to(device=device, dtype=dtype)


def masked_dssim(
    render: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None,
    *, window_size: int = SSIM_WINDOW_SIZE,
) -> torch.Tensor:
    """1-SSIM on fully observed interior windows; empty support contributes zero."""
    if render.shape != target.shape or render.ndim != 3:
        raise ValueError("SSIM requires matching CHW images")
    if min(render.shape[-2:]) < window_size:
        return render.sum() * 0.0
    channel = int(render.shape[0])
    window = _ssim_window(window_size, channel, render.device, render.dtype)

    def conv(value):
        return F.conv2d(value.unsqueeze(0), window, padding=0, groups=channel)

    mu1, mu2 = conv(render), conv(target)
    sigma1 = conv(render.square()) - mu1.square()
    sigma2 = conv(target.square()) - mu2.square()
    sigma12 = conv(render * target) - mu1 * mu2
    score = ((2 * mu1 * mu2 + 0.01**2) * (2 * sigma12 + 0.03**2)) / (
        (mu1.square() + mu2.square() + 0.01**2)
        * (sigma1 + sigma2 + 0.03**2)
    ).clamp_min(1e-12)
    error = (1.0 - score).squeeze(0).mean(dim=0)
    if mask is None:
        return error.mean()
    keep = mask.detach().to(device=render.device)
    if keep.ndim == 2:
        keep = keep.unsqueeze(0)
    if keep.shape[-2:] != render.shape[-2:] or keep.shape[0] not in (1, channel):
        raise ValueError("SSIM mask must cover the image raster")
    invalid = (~(torch.isfinite(keep) & (keep > 0.999))).any(dim=0)
    valid = F.max_pool2d(
        invalid.to(render.dtype)[None, None], window_size, stride=1
    )[0, 0] == 0
    return torch.where(valid, error, 0.0).sum() / valid.sum().clamp_min(1)


def masked_rgb_blend(
    render: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None,
    *, dssim_weight: float = 0.2,
) -> torch.Tensor:
    """Masked RGB loss: ``(1 - dssim) * masked L1 + dssim * masked DSSIM``."""
    l1 = masked_l1(render, target, mask)
    if dssim_weight == 0:
        return l1
    dssim = masked_dssim(render, target, mask)
    return (1.0 - float(dssim_weight)) * l1 + float(dssim_weight) * dssim


def rade_depth_normal_consistency(pkg: dict, camera) -> torch.Tensor:
    """RaDe-GS depth-normal consistency (GaussianZoom Eq. 10 stencil).

    Repository-verified RaDe depth convention applies: ``pkg["render_depth"]``
    is the expected principal-ray distance, so per-pixel camera
    Z is ``render_depth * rln`` (``lod.depth.ray_distance_to_z`` through the
    principal point). Camera-space points are rebuilt as ``z * ray``; the
    depth-derived normal uses the official ``cross(dy, dx)`` stencil; the
    rasterizer's camera-space ``pkg["render_norm"]`` is compared with
    ``1 - dot`` over fully observed interior pixels (alpha > 0.05, positive
    finite depth, whole stencil observed). Gradients flow through the
    rendered depth/normal buffers; validity masks are detached.
    """
    missing = [
        key for key in ("render_depth", "render_norm", "render_alpha")
        if pkg.get(key) is None
    ]
    if missing:
        raise ValueError(
            f"RaDe geometry loss requires rasterizer outputs {missing}; "
            f"available keys: {sorted(pkg)}"
        )
    depth = pkg["render_depth"]
    normal = pkg["render_norm"]
    alpha = pkg["render_alpha"]
    if depth.dim() == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if alpha.dim() == 3 and alpha.shape[0] == 1:
        alpha = alpha[0]
    if normal.dim() == 4 and normal.shape[0] == 1:
        normal = normal[0]
    height, width = int(depth.shape[0]), int(depth.shape[1])
    if depth.shape != (camera.image_height, camera.image_width) or normal.shape != (3, height, width):
        raise ValueError("RaDe depth/normal raster must match the camera")
    if alpha.shape != depth.shape:
        raise ValueError("RaDe alpha must match the depth raster")
    observed = torch.isfinite(depth.detach()) & (depth.detach() > 0)
    observed &= torch.isfinite(alpha.detach()) & (alpha.detach() > 0.05)
    safe_depth = torch.where(observed, depth, 0.0)
    rln = rln_principal_from_skyfall_camera(camera, device=depth.device, dtype=depth.dtype)
    z = ray_distance_to_z(safe_depth, rln)
    ys = torch.arange(height, device=depth.device, dtype=depth.dtype)
    xs = torch.arange(width, device=depth.device, dtype=depth.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    cx = ndc_principal_to_pixel(float(camera.cx), width)
    cy = ndc_principal_to_pixel(float(camera.cy), height)
    rays = torch.stack(
        (
            (grid_x - cx) / float(camera.focal_x),
            (grid_y - cy) / float(camera.focal_y),
            torch.ones_like(grid_x),
        ),
        dim=0,
    )
    points = rays * z.unsqueeze(0)
    dy = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]
    dx = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
    normals = F.normalize(torch.cross(dy, dx, dim=0), dim=0, eps=1e-8)
    finite_normal = torch.isfinite(normal.detach()).all(dim=0)
    rendered = torch.where(torch.isfinite(normal), normal, 0.0)[:, 1:-1, 1:-1]
    error = 1.0 - (rendered * normals).sum(dim=0).clamp(-1.0, 1.0)
    valid = (
        observed[1:-1, 1:-1] & observed[2:, 1:-1] & observed[:-2, 1:-1]
        & observed[1:-1, 2:] & observed[1:-1, :-2] & finite_normal[1:-1, 1:-1]
    )
    return (error * valid).sum() / valid.sum().clamp_min(1.0)

def _compact(image: torch.Tensor, max_side: int = 512) -> torch.Tensor:
    _, height, width = image.shape
    scale = min(1.0, float(max_side) / float(max(height, width)))
    if scale >= 1.0:
        return image
    return torch.nn.functional.interpolate(
        image.unsqueeze(0), scale_factor=scale, mode="bilinear", align_corners=False,
        antialias=True,
    ).squeeze(0).clamp(0.0, 1.0)


def save_compact_triplet(
    *, before: torch.Tensor | None, target: torch.Tensor, after: torch.Tensor,
    out_dir: str, name: str, max_side: int = 512,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    paths = {"target": os.path.join(out_dir, f"{name}_target.png")}
    save_tensor_image(_compact(target, max_side), paths["target"])
    if before is not None:
        paths["before"] = os.path.join(out_dir, f"{name}_before.png")
        save_tensor_image(_compact(before, max_side), paths["before"])
    paths["after"] = os.path.join(out_dir, f"{name}_after.png")
    save_tensor_image(_compact(after, max_side), paths["after"])
    return paths


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    steps_per_level: int
    max_points_per_level: int
    seed: int
    mix_ratio: float = 0.2
    densify_from: int = 1
    densify_fraction: float = 0.6
    densify_every: int = 25
    densify_grad_threshold: float = 2e-4
    position_lr: float = 1.6e-4
    feature_lr: float = 2.5e-3
    opacity_lr: float = 5e-2
    scaling_lr: float = 5e-3
    rotation_lr: float = 1e-3
    eval_samples: int = 4
    png_samples: int = 3
    target_cache_size: int = 128
    kernel_size: float = 0.1
    white_background: bool = False
    quiet: bool = False
    # Step-ablation controls. All default to the legacy single-scalar behavior.
    level_steps: tuple[int, ...] | None = None
    checkpoint_steps: tuple[int, ...] = ()
    # "multiscale" fits HR/LR RGB; "frequency" replaces HR with signed
    # teacher detail above the native LR band. Both retain RaDe geometry.
    # "l1" keeps the original masked RGB objective.
    loss_mode: str = "l1"
    loss_hr: float = 0.6
    loss_lr: float = 0.4
    loss_geometry: float = 0.05
    loss_dssim: float = 0.2
    densify_until: int | None = None
    densify_bootstrap_fraction: float = 1.0
    split_radius_pixels: float = 0.0
    cache_frozen_colors: bool = True

    def __post_init__(self) -> None:
        if self.loss_mode not in ("l1", "multiscale", "frequency"):
            raise ValueError(f"Unsupported loss_mode {self.loss_mode!r}")
        for name in ("loss_hr", "loss_lr", "loss_geometry", "loss_dssim"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative float (got {value})")
            setattr(self, name, value)
        if self.loss_dssim > 1:
            raise ValueError("loss_dssim must be in [0, 1]")
        if self.loss_mode != "l1" and self.loss_hr + self.loss_lr <= 0.0:
            raise ValueError("Dual-scale loss requires loss_hr + loss_lr > 0")
        if not math.isfinite(self.densify_bootstrap_fraction) or not 0 < self.densify_bootstrap_fraction <= 1:
            raise ValueError("densify_bootstrap_fraction must be in (0,1]")
        if not math.isfinite(self.split_radius_pixels) or self.split_radius_pixels < 0:
            raise ValueError("split_radius_pixels must be finite and nonnegative")
        if self.level_steps is not None:
            budgets = tuple(int(value) for value in self.level_steps)
            if not budgets or any(value < 1 for value in budgets):
                raise ValueError(
                    "level_steps must be a non-empty sequence of positive integers "
                    f"(got {self.level_steps!r})"
                )
            self.level_steps = budgets
        milestones = tuple(int(value) for value in self.checkpoint_steps)
        if any(value < 1 for value in milestones):
            raise ValueError(f"checkpoint_steps must be positive integers (got {self.checkpoint_steps!r})")
        if any(later <= earlier for earlier, later in zip(milestones, milestones[1:])):
            raise ValueError(
                f"checkpoint_steps must be strictly increasing (got {self.checkpoint_steps!r})"
            )
        self.checkpoint_steps = milestones
        if self.densify_until is not None:
            self.densify_until = int(self.densify_until)
            if self.densify_until < 1:
                raise ValueError(f"densify_until must be a positive integer (got {self.densify_until})")


@dataclass
class LevelSample:
    key: str
    view_id: str
    zoom: float
    roi: dict
    image_path: str
    mask_path: str | None
    camera: Any          # zoomed RenderCamera (Skyfall-style)
    stage_camera: Any    # zoomed vendor pinhole camera (for stage checks/densify)
    embedding: Any
    appearance_uid: Any
    # Real LR anchor supervision (loss_mode="multiscale"): the original-view
    # crop at the base raster for this level's cumulative zoom, provided by
    # the manifest. ``lr_mask_path`` optionally excludes the edited region.
    lr_image_path: str | None = None
    lr_mask_path: str | None = None
    lr_width: int | None = None
    lr_height: int | None = None


@dataclass
class ReplaySample:
    label: str
    camera: Any
    embedding: Any
    image_path: str
    mask_path: str | None
    width: int
    height: int


@dataclass
class LevelResult:
    zoom_factor: float
    level_index: int
    stage_scale: float
    n_stage_cameras: int
    n_samples: int
    bootstrap_views: int
    steps_requested: int
    steps_executed: int
    coverage_steps: int
    replay_steps: int
    unique_targets_seen: int
    densify_events: int
    densify_totals: dict
    points_start: int
    points_end: int
    l0_points: int
    eval_before: dict
    eval_after: dict
    pngs: dict
    seconds: float
    densify_until: int = 0
    densify_spec: dict = field(default_factory=dict)
    checkpoints: list = field(default_factory=list)


class SceneLodTrainer:
    """Owns the bundle, targets, RNG and per-level training across a manifest."""

    def __init__(
        self,
        *,
        gaussians,
        bundle,
        scene_views: SceneViews,
        resolver: AppearanceResolver,
        background: torch.Tensor,
        kernel: float,
        config: TrainConfig,
        output_dir: str,
        scene_extent: float,
    ) -> None:
        self.gaussians = gaussians
        self.bundle = bundle
        self.views = scene_views
        self.resolver = resolver
        self.background = background
        self.kernel = float(kernel)
        self.config = config
        self.output_dir = os.path.abspath(output_dir)
        self.scene_extent = float(scene_extent)
        self.device = str(gaussians.get_xyz.device)
        self.targets = TargetCache(maxsize=config.target_cache_size)
        self.densify_log: list[dict] = []
        self.level_checkpoints: list[dict] = []
        self.manifest: dict | None = None
        self.allowed_view_ids: set[str] | None = None
        self.views_by_id: dict[str, dict] = {}
        self._frozen_color_inputs = None
        self._frozen_colors: dict[tuple, torch.Tensor] = {}
        self._color_pose_keys: dict[int, tuple] = {}

    # -- sample assembly ----------------------------------------------------- #

    def build_samples(
        self, manifest: dict, allowed_view_ids: set[str] | None,
        *, bind_union_base: bool = True,
    ) -> tuple[list[LevelSample], list[list[LevelSample]]]:
        """Register ROI aliases for every sample and return per-level sample lists.

        ``bind_union_base=False`` skips the base-stage bind: parent-continuation
        runs extend the loaded parent's stage records instead (the vendor model
        refuses a second ``bind_base_stage`` on a multi-level checkpoint).
        """

        self.manifest = manifest
        self.allowed_view_ids = allowed_view_ids
        self.views_by_id = {str(view["id"]): view for view in manifest["views"]}
        stored_centers: dict[int, torch.Tensor] = {}
        level_infos = [
            {"zoom": float(level["zoom_factor"]), "level": level, "order": order}
            for order, level in enumerate(
                sorted(manifest["levels"], key=lambda item: float(item["zoom_factor"]))
            )
        ]
        alias_specs = manifest_alias_specs(
            manifest, allowed_view_ids, self.views.view_to_physical
        )
        if bind_union_base:
            union_cameras = manifest_union_base_cameras(self.bundle, self.views, alias_specs)
            bind_union_base_stage(self.bundle, union_cameras)

        samples: list[LevelSample] = []
        per_level: list[list[LevelSample]] = [[] for _ in level_infos]
        for info in level_infos:
            zoom = info["zoom"]
            for position, raw in enumerate(info["level"]["samples"]):
                view_id = str(raw["view_id"])
                if allowed_view_ids is not None and view_id not in allowed_view_ids:
                    continue
                roi = raw["roi"]
                u = float(roi["center_x"])
                v = float(roi["center_y"])
                physical = self.views.view_to_physical[view_id]
                alias = alias_specs[(physical, zoom, round(u, 6), round(v, 6))]
                render_base = self.views.render_cameras[physical]
                camera = render_base.zoomed(
                    zoom, u, v, image_name=f"{view_id}_z{zoom:g}_u{u:.4f}_v{v:.4f}"
                )
                if physical not in stored_centers:
                    stored_centers[physical] = render_base.camera_center.detach().to(
                        device=camera.world_view_transform.device, dtype=torch.float32
                    ).reshape(3).contiguous()
                stage_camera = make_stage_camera(
                    self.bundle, alias=alias, physical_index=physical, zoom=zoom,
                    center_u=u, center_v=v, stored_center=stored_centers[physical],
                )
                view_record = self.views_by_id[view_id]
                key = f"z{zoom:g}/{view_id}/u{u:.4f}_v{v:.4f}#{position}"
                lr_image_path, lr_mask_path, lr_width, lr_height = resolve_lr_anchor(
                    raw, camera, zoom, key,
                    loss_mode=(
                        self.config.loss_mode
                        if info["order"] >= len(self.bundle.lod.layers) - 1 else "l1"
                    ),
                )
                sample = LevelSample(
                    key=key,
                    view_id=view_id,
                    zoom=zoom,
                    roi=dict(roi),
                    image_path=os.path.abspath(raw["image_path"]),
                    mask_path=os.path.abspath(raw["mask_path"]) if raw.get("mask_path") else None,
                    camera=camera,
                    stage_camera=stage_camera,
                    embedding=self.resolver.for_view(view_record.get("appearance_uid")),
                    appearance_uid=view_record.get("appearance_uid"),
                    lr_image_path=lr_image_path,
                    lr_mask_path=lr_mask_path,
                    lr_width=lr_width,
                    lr_height=lr_height,
                )
                per_level[info["order"]].append(sample)
        for group in per_level:
            samples.extend(group)
        return samples, per_level

    def replay_pool_for_level(
        self, level_order: int, per_level: list[list[LevelSample]]
    ) -> list[ReplaySample]:
        """Original views plus all earlier-level supervision, vendor-style multiscale replay."""

        pool: list[ReplaySample] = []
        allowed = self.allowed_view_ids
        for view in self.manifest["views"]:
            view_id = str(view["id"])
            if allowed is not None and view_id not in allowed:
                continue
            physical = self.views.view_to_physical[view_id]
            camera = self.views.render_cameras[physical]
            pool.append(ReplaySample(
                label=f"base/{view_id}",
                camera=camera,
                embedding=self.resolver.for_view(view.get("appearance_uid")),
                image_path=os.path.abspath(view["image_path"]),
                mask_path=None,
                width=int(camera.image_width),
                height=int(camera.image_height),
            ))
        for earlier in per_level[: max(0, int(level_order))]:
            for sample in earlier:
                pool.append(ReplaySample(
                    label=f"earlier/{sample.key}",
                    camera=sample.camera,
                    embedding=sample.embedding,
                    image_path=sample.image_path,
                    mask_path=sample.mask_path,
                    width=int(sample.camera.image_width),
                    height=int(sample.camera.image_height),
                ))
        return pool

    def steps_budget_for(self, level_order: int) -> int:
        """Per-level step budget: explicit ``level_steps`` when set, else the scalar."""

        if self.config.level_steps is not None:
            return max(1, int(self.config.level_steps[int(level_order)]))
        return max(1, int(self.config.steps_per_level))

    def densify_until_for(self, steps_budget: int) -> int:
        """Absolute densify-until step: explicit ``densify_until`` wins, else derived."""

        if self.config.densify_until is not None:
            return max(1, int(self.config.densify_until))
        return max(1, int(math.ceil(self.config.densify_fraction * steps_budget)))

    def point_budget_for(self, step: int, horizon: int) -> int:
        """Reserve capacity for later residuals instead of filling it at bootstrap."""
        fraction = self.config.densify_bootstrap_fraction
        progress = min(1.0, max(0.0, step / max(1, horizon)))
        return max(1, math.ceil(self.config.max_points_per_level * (fraction + (1 - fraction) * progress)))

    def save_level_checkpoint(
        self,
        *,
        level_index: int,
        step: int,
        zoom: float,
        steps_budget: int,
        densify_until: int,
        eval_set: dict[str, LevelSample],
    ) -> None:
        """Persist a midlevel bundle AFTER the optimizer/densify update, then record it."""

        path = os.path.join(self.output_dir, f"l{level_index}_step{step:06d}.lod.pt")
        save_bundle(self.bundle, path, optimizer=None)
        eval_payload = self.evaluate_samples(list(eval_set.values())) if eval_set else {}
        self.level_checkpoints.append({
            "step": int(step),
            "checkpoint": os.path.abspath(path),
            "appearance_sidecar": os.path.abspath(path) + ".appearance.pt",
            "eval": eval_payload,
            "points": int(self.bundle.active_layer().xyz.shape[0]),
        })
        # Progressive index so remote snapshots stay discoverable while training runs.
        write_json(
            os.path.join(self.output_dir, f"l{level_index}_checkpoints.json"),
            {
                "kind": "skyfall_scene_lod_step_checkpoints",
                "level_index": int(level_index),
                "zoom_factor": float(zoom),
                "steps_budget": int(steps_budget),
                "densify_until": int(densify_until),
                "checkpoints": [dict(entry) for entry in self.level_checkpoints],
            },
        )


    def _reset_frozen_color_cache(self) -> None:
        self._frozen_colors.clear()
        self._color_pose_keys.clear()
        self._frozen_color_inputs = None
        if not self.config.cache_frozen_colors:
            return
        layers = list(self.bundle.lod.layers[:-1])
        if not layers:
            return
        if any(not layer.frozen for layer in layers):
            raise ValueError("Parent color caching requires completed frozen levels")

        def joined(name):
            values = [getattr(layer, name).detach() for layer in layers]
            return values[0] if len(values) == 1 else torch.cat(values)

        xyz, sh = joined("xyz"), joined("sh")
        appearance = self.bundle.appearance
        if appearance.enabled:
            embeddings = appearance.embeddings_for_levels(len(layers))
        else:
            embeddings = xyz.new_zeros((len(xyz), 1))
        self._frozen_color_inputs = (xyz, sh, embeddings)

    def _parent_colors(self, camera, embedding) -> torch.Tensor | None:
        if self._frozen_color_inputs is None:
            return None
        pose = self._color_pose_keys.get(id(camera))
        if pose is None:
            pose = tuple(camera.camera_center.detach().cpu().tolist())
            self._color_pose_keys[id(camera)] = pose
        key = (pose, None if embedding is None else embedding.data_ptr())
        if key not in self._frozen_colors:
            with torch.no_grad():
                self._frozen_colors[key] = appearance_colors_lod(
                    *self._frozen_color_inputs, self.bundle.appearance,
                    camera.camera_center, embedding, self.bundle.sh_degree,
                ).detach()
        return self._frozen_colors[key]

    # -- core loop ----------------------------------------------------------- #

    def _accumulate_screen_grad(self, pkg, grad_sum: torch.Tensor, grad_count: torch.Tensor, radius_max=None) -> None:
        if radius_max is not None:
            torch.maximum(radius_max, pkg["radii"].detach(), out=radius_max)
        grad = pkg["viewspace_points"].grad
        if grad is None:
            return
        visible = pkg["radii"] > 0
        norms = grad.detach()[:, :2].norm(dim=-1).float()
        if bool(visible.any().item()):
            index = torch.nonzero(visible, as_tuple=False)[:, 0]
            grad_sum.index_add_(0, index, norms[index])
            grad_count.index_add_(0, index, torch.ones_like(norms[index]))

    def _render_loss(self, camera, embedding, image_path, mask_path, width, height):
        target = self.targets.image(image_path, width, height).to(
            device=self.device, dtype=torch.float32
        )
        mask = None
        if mask_path:
            mask = self.targets.mask(mask_path, width, height).to(device=self.device)
        pkg = render_lod_appearance(
            self.bundle, camera, background=self.background, kernel_size=self.kernel,
            appearance_embedding=embedding, lod=True, compact=False,
            frozen_colors=self._parent_colors(camera, embedding),
        )
        loss = masked_l1(pkg["render"], target, mask)
        return loss, pkg, target, mask


    def _multiscale_loss(self, sample: LevelSample) -> tuple[torch.Tensor, dict, dict]:
        """Dual-scale RGB or frequency-residual loss plus RaDe geometry.

        Frequency mode replaces the HR term below with signed residual L1;
        the teacher band is cached and the same render supplies the LR term.

        HR: masked RGB blend (L1/DSSIM) between the render and the level
        target at the level raster. LR: the SAME HR render bicubic-downsampled
        to the real LR anchor's raster (never a second render with different
        LoD weights) against the manifest's ``lr_image_path``; ``lr_mask_path``
        keeps edited pixels out of the LR support. Geometry: RaDe depth-normal
        consistency on the rendered buffers. Replay steps keep ordinary
        masked-L1 RGB loss regardless of ``loss_mode``.
        """
        cfg = self.config
        if not sample.lr_image_path:
            raise ValueError(
                f"sample {sample.key}: dual-scale training requires lr_image_path"
            )
        width = int(sample.camera.image_width)
        height = int(sample.camera.image_height)
        mask = None
        if sample.mask_path:
            if cfg.loss_mode == "frequency":
                mask = self.targets.frequency_mask(
                    sample.mask_path, width, height, int(sample.lr_width), int(sample.lr_height)
                ).to(device=self.device, dtype=torch.float32)
            else:
                mask = self.targets.mask(sample.mask_path, width, height).to(
                    device=self.device, dtype=torch.float32
                )
        pkg = render_lod_appearance(
            self.bundle, sample.camera, background=self.background, kernel_size=self.kernel,
            appearance_embedding=sample.embedding, lod=True, compact=False,
            frozen_colors=self._parent_colors(sample.camera, sample.embedding),
        )
        render = pkg["render"]
        lr_target = self.targets.image(
            sample.lr_image_path, int(sample.lr_width), int(sample.lr_height)
        ).to(device=self.device, dtype=torch.float32)
        lr_mask = None
        if sample.lr_mask_path:
            lr_mask = self.targets.mask(
                sample.lr_mask_path, int(sample.lr_width), int(sample.lr_height)
            ).to(device=self.device, dtype=torch.float32)
        down = F.interpolate(
            render.unsqueeze(0), size=lr_target.shape[-2:], mode="bicubic",
            align_corners=False, antialias=True,
        ).squeeze(0)
        if cfg.loss_mode == "frequency":
            detail_target = self.targets.frequency(
                sample.image_path, width, height, int(sample.lr_width), int(sample.lr_height)
            ).to(device=self.device, dtype=torch.float32)
            hr = masked_l1(frequency_residual(render, down), detail_target, mask)
        else:
            hr_target = self.targets.image(sample.image_path, width, height).to(
                device=self.device, dtype=torch.float32
            )
            hr = masked_rgb_blend(render, hr_target, mask, dssim_weight=cfg.loss_dssim)
        lr = masked_rgb_blend(down, lr_target, lr_mask, dssim_weight=cfg.loss_dssim)
        if cfg.loss_geometry > 0.0:
            geometry = rade_depth_normal_consistency(pkg, sample.camera)
        else:
            geometry = render.sum() * 0.0
        total = cfg.loss_hr * hr + cfg.loss_lr * lr + cfg.loss_geometry * geometry
        components = {
            "hf" if cfg.loss_mode == "frequency" else "hr": hr.detach(),
            "lr": lr.detach(),
            "geometry": geometry.detach(),
            "total": total.detach(),
        }
        if not bool(torch.isfinite(total)):
            raise RuntimeError(
                f"sample {sample.key}: non-finite multiscale loss components {components}"
            )
        return total, pkg, components

    @torch.no_grad()
    def evaluate_samples(self, samples: Sequence[LevelSample]) -> dict:
        payload: dict[str, Any] = {}
        values: list[dict] = []
        for sample in samples:
            mask = None
            if sample.mask_path:
                mask = self.targets.mask(
                    sample.mask_path, int(sample.camera.image_width), int(sample.camera.image_height)
                )
            pkg = render_lod_appearance(
                self.bundle, sample.camera, background=self.background, kernel_size=self.kernel,
                appearance_embedding=sample.embedding, lod=True, compact=False,
                frozen_colors=self._parent_colors(sample.camera, sample.embedding),
            )
            render = pkg["render"].detach()
            target = self.targets.image(
                sample.image_path, int(sample.camera.image_width), int(sample.camera.image_height)
            ).to(device=render.device)
            mse = masked_mse(render.clamp(0.0, 1.0), target, mask)
            entry = {"mse": mse, "psnr": psnr_value(mse)}
            entry["l1"] = float(masked_l1(render.clamp(0.0, 1.0), target, mask).item())
            payload[sample.key] = entry
            values.append(entry)
        if values:
            payload["_mean"] = {
                "l1": float(np.mean([v["l1"] for v in values])),
                "psnr": float(np.mean([v["psnr"] for v in values])),
            }
        return payload

    def train_level(
        self,
        *,
        level_order: int,
        level_index: int,
        samples: list[LevelSample],
        densify_cameras: Sequence[Any],
        replay_pool: Sequence[ReplaySample],
        level_dir: str,
    ) -> LevelResult:
        if not samples:
            raise ValueError(f"level {level_index} has no samples")
        cfg = self.config
        zoom = samples[0].zoom
        started = time.perf_counter()
        self.densify_log = []
        self.level_checkpoints = []
        points_start = int(self.bundle.active_layer().xyz.shape[0])
        stage_scale = float(self.bundle.lod.stage_records[-1]["scale"])
        self._reset_frozen_color_cache()

        optimizer = self.bundle.lod.make_optimizer(
            position_lr=cfg.position_lr, feature_lr=cfg.feature_lr, opacity_lr=cfg.opacity_lr,
            scaling_lr=cfg.scaling_lr, rotation_lr=cfg.rotation_lr,
        )
        total_rows = sum(int(layer.xyz.shape[0]) for layer in self.bundle.lod.layers)
        grad_sum = torch.zeros(total_rows, device=self.device)
        grad_count = torch.zeros(total_rows, device=self.device)
        radius_max = (
            torch.zeros(total_rows, device=self.device, dtype=torch.int32)
            if cfg.split_radius_pixels > 0 else None
        )
        rng = random.Random(cfg.seed * 1009 + level_order * 97 + 17)

        steps_min = self.steps_budget_for(level_order)
        densify_until = self.densify_until_for(steps_min)
        per_level_densify_spec = densify_spec(
            densify_from=max(1, int(cfg.densify_from)), densify_until=densify_until,
            densify_interval=cfg.densify_every, densify_grad_threshold=cfg.densify_grad_threshold,
            max_points=cfg.max_points_per_level, mix_ratio=cfg.mix_ratio, seed=cfg.seed,
        )
        per_level_densify_spec.update(
            bootstrap_fraction=cfg.densify_bootstrap_fraction,
            split_radius_pixels=cfg.split_radius_pixels,
        )
        densify_from = max(1, int(cfg.densify_from))
        eval_indices: list[int] = []
        if cfg.eval_samples > 0:
            stride = max(1, len(samples) // max(1, cfg.eval_samples))
            eval_indices = list(range(0, len(samples), stride))[: cfg.eval_samples]
            if len(samples) - 1 not in eval_indices:
                eval_indices.append(len(samples) - 1)
        eval_set = {samples[index].key: samples[index] for index in eval_indices}

        # "Before" evidence: active level is still empty -> render of completed levels only.
        eval_before: dict = {}
        eval_before_render: dict[str, torch.Tensor] = {}
        if eval_set:
            eval_before = self.evaluate_samples(list(eval_set.values()))
            for sample in list(eval_set.values())[: max(0, cfg.png_samples)]:
                pkg = render_lod_appearance(
                    self.bundle, sample.camera, background=self.background,
                    kernel_size=self.kernel, appearance_embedding=sample.embedding,
                    lod=True, compact=False,
                    frozen_colors=self._parent_colors(sample.camera, sample.embedding),
                )
                eval_before_render[sample.key] = pkg["render"].detach().clamp(0.0, 1.0).cpu()

        # Allocate the finite detail budget from the entire scene, not the first
        # random crop. A single-crop bootstrap otherwise fills max_points at step 1.
        for sample in samples:
            if cfg.loss_mode != "l1":
                loss, pkg, _components = self._multiscale_loss(sample)
            else:
                loss, pkg, _target, _mask = self._render_loss(
                    sample.camera, sample.embedding, sample.image_path, sample.mask_path,
                    int(sample.camera.image_width), int(sample.camera.image_height),
                )
            loss.backward()
            self._accumulate_screen_grad(pkg, grad_sum, grad_count, radius_max)
        bootstrap_stats = densify_with_appearance(
            self.bundle, optimizer, grad_sum / grad_count.clamp_min(1.0), densify_cameras,
            threshold=cfg.densify_grad_threshold, max_points=self.point_budget_for(0, densify_until),
            min_opacity=0.005, scene_extent=self.scene_extent, percent_dense=0.01,
            split_mask=None if radius_max is None else radius_max > cfg.split_radius_pixels,
        )
        if int(self.bundle.active_layer().xyz.shape[0]) == 0:
            raise RuntimeError(f"LoD level {level_index} could not initialize any detail points")
        optimizer.zero_grad(set_to_none=True)
        rows = sum(int(layer.xyz.shape[0]) for layer in self.bundle.lod.layers)
        grad_sum = torch.zeros(rows, device=self.device)
        grad_count = torch.zeros(rows, device=self.device)
        if radius_max is not None:
            radius_max = torch.zeros(rows, device=self.device, dtype=torch.int32)
        self.densify_log.append({
            "step": 0, "kind": "scene_bootstrap", "views": len(samples),
            "densify": bootstrap_stats, "n_active": int(self.bundle.active_layer().xyz.shape[0]),
            "point_budget": self.point_budget_for(0, densify_until),
        })

        queue: deque = deque()
        visited: set[int] = set()
        loss_components: dict | None = None
        densify_events = 1
        densify_totals = {key: int(value) for key, value in bootstrap_stats.items() if key != "points"}
        coverage_steps = 0
        replay_steps = 0
        step = 0
        while True:
            remaining = len(samples) - len(visited)
            if step >= steps_min and remaining == 0:
                break
            step += 1
            use_replay = (
                bool(replay_pool)
                and rng.random() < cfg.mix_ratio
                and (step + remaining) <= steps_min
            )
            loss_components = None
            if use_replay:
                replay = replay_pool[rng.randrange(len(replay_pool))]
                loss, pkg, _target, _mask = self._render_loss(
                    replay.camera, replay.embedding, replay.image_path, replay.mask_path,
                    replay.width, replay.height,
                )
                replay_steps += 1
            else:
                if not queue:
                    queue = deque(rng.sample(range(len(samples)), len(samples)))
                index = queue.popleft()
                visited.add(index)
                sample = samples[index]
                if cfg.loss_mode != "l1":
                    loss, pkg, loss_components = self._multiscale_loss(sample)
                else:
                    loss, pkg, _target, _mask = self._render_loss(
                        sample.camera, sample.embedding, sample.image_path, sample.mask_path,
                        int(sample.camera.image_width), int(sample.camera.image_height),
                    )
                coverage_steps += 1
            loss.backward()
            self._accumulate_screen_grad(pkg, grad_sum, grad_count, radius_max)
            active = self.bundle.active_layer()
            # Vendor-sound ordering: apply the collected gradients to the current
            # parameters FIRST, then let densify replace/append rows.
            if int(active.xyz.shape[0]) > 0:
                optimizer.step()
            densify_stats = None
            scheduled = (
                cfg.densify_every > 0
                and densify_from <= step <= densify_until
                and step % cfg.densify_every == 0
                and int(active.xyz.shape[0]) > 0
            )
            if scheduled:
                grads = (grad_sum / grad_count.clamp_min(1.0)).float()
                densify_stats = densify_with_appearance(
                    self.bundle, optimizer, grads, densify_cameras,
                    threshold=cfg.densify_grad_threshold,
                    max_points=self.point_budget_for(step, densify_until),
                    min_opacity=0.005,
                    scene_extent=self.scene_extent,
                    percent_dense=0.01,
                    split_mask=None if radius_max is None else radius_max > cfg.split_radius_pixels,
                )
                rows = sum(int(layer.xyz.shape[0]) for layer in self.bundle.lod.layers)
                grad_sum = torch.zeros(rows, device=self.device)
                grad_count = torch.zeros(rows, device=self.device)
                if radius_max is not None:
                    radius_max = torch.zeros(rows, device=self.device, dtype=torch.int32)
                if densify_stats:
                    densify_events += 1
                    for key, value in densify_stats.items():
                        if key == "points":
                            continue
                        densify_totals[key] = densify_totals.get(key, 0) + int(value)
            optimizer.zero_grad(set_to_none=True)
            if cfg.checkpoint_steps and step in cfg.checkpoint_steps:
                self.save_level_checkpoint(
                    level_index=level_index, step=step, zoom=zoom,
                    steps_budget=steps_min, densify_until=densify_until,
                    eval_set=eval_set,
                )
            if densify_stats or step % 20 == 0 or step == steps_min:
                record = {
                    "step": step,
                    "loss": float(loss.detach().item()),
                    "kind": "replay" if use_replay else "coverage",
                    "densify": densify_stats or {},
                    "n_active": int(self.bundle.active_layer().xyz.shape[0]),
                    "point_budget": self.point_budget_for(step, densify_until),
                }
                if loss_components is not None:
                    record["loss_components"] = {name: float(value) for name, value in loss_components.items()}
                self.densify_log.append(record)
                if not cfg.quiet:
                    print(
                        f"[level {level_index} z{zoom:g}] step {step} loss={record['loss']:.5f} "
                        f"kind={record['kind']} n_active={record['n_active']}"
                    )

        eval_after: dict = {}
        if eval_set:
            eval_after = self.evaluate_samples(list(eval_set.values()))
        pngs: dict = {}
        for sample in list(eval_set.values())[: max(0, cfg.png_samples)]:
            pkg = render_lod_appearance(
                self.bundle, sample.camera, background=self.background,
                kernel_size=self.kernel, appearance_embedding=sample.embedding,
                lod=True, compact=False,
                frozen_colors=self._parent_colors(sample.camera, sample.embedding),
            )
            target = self.targets.image(
                sample.image_path, int(sample.camera.image_width), int(sample.camera.image_height)
            ).to(device=pkg["render"].device)
            pngs[sample.key] = save_compact_triplet(
                before=eval_before_render.get(sample.key),
                target=target,
                after=pkg["render"].detach().clamp(0.0, 1.0),
                out_dir=os.path.join(level_dir, "eval"),
                name=sample.key.replace("/", "__"),
            )

        return LevelResult(
            zoom_factor=zoom,
            level_index=level_index,
            stage_scale=stage_scale,
            n_stage_cameras=len(densify_cameras),
            n_samples=len(samples),
            bootstrap_views=len(samples),
            steps_requested=steps_min,
            steps_executed=step,
            coverage_steps=coverage_steps,
            replay_steps=replay_steps,
            unique_targets_seen=len(visited),
            densify_events=densify_events,
            densify_totals=densify_totals,
            densify_until=densify_until,
            densify_spec=per_level_densify_spec,
            checkpoints=list(self.level_checkpoints),
            points_start=points_start,
            points_end=int(self.bundle.active_layer().xyz.shape[0]),
            l0_points=int(self.bundle.layer0().xyz.shape[0]),
            eval_before=eval_before,
            eval_after=eval_after,
            pngs=pngs,
            seconds=time.perf_counter() - started,
        )


def train_scene_from_manifest(
    *,
    gaussians,
    manifest: dict,
    manifest_path: str,
    start_checkpoint: str,
    output_dir: str,
    gz_root: str,
    step_scale: float,
    config: TrainConfig,
    scene_extent: float,
    max_views: int = 0,
    parent_lod_checkpoint: str | None = None,
    native_train_cameras: Sequence[Any] | None = None,
) -> dict:
    """Full orchestration: build cameras, import frozen L0, train every level, save artifacts.

    With ``parent_lod_checkpoint`` set, the parent's completed LoD levels and
    per-level appearance are loaded unchanged onto the freshly imported frozen
    L0 (exact tensor identity is verified against ``gaussians``), the parent's
    stage records are validated against the combined manifest, the base stage
    record is extended with the combined manifest's scale-1 alias cameras, and
    only the remaining manifest levels are trained.

    ``native_train_cameras`` are the checkpoint dataset's own TRAIN cameras
    (``Scene.getTrainCameras()``): the authoritative stable physical
    base-camera reference for the L0 import, identical to the generation path.

    Returns the ``training_summary.json`` payload.
    """

    from lod.lineage import write_train_lineage, write_json

    started = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)
    device = str(gaussians.get_xyz.device)
    declared_base = manifest.get("base_checkpoint")
    if declared_base and os.path.realpath(declared_base) != os.path.realpath(start_checkpoint):
        raise ValueError("Supervision and trainer must use the same base checkpoint")

    scene_views = build_scene_views(manifest, max_views=max_views, device=device)
    # Authoritative checkpoint state (including the PLY-persisted native
    # filter_3D) is already loaded into ``gaussians`` by the caller. The filter
    # is kept EXACTLY as stored -- never recomputed against the manifest's
    # synthetic camera sets -- so generation and training import the identical
    # frozen G_R filter and a trained parent loads unchanged at the next level.
    import_cameras, filter_source, reference_source = native_reference_cameras(
        manifest, scene_views, gaussians, native_train_cameras=native_train_cameras,
    )
    bundle = import_skyfall_l0(
        gaussians, import_cameras, gz_root=gz_root,
        step_scale=step_scale, freeze=True,
    )
    # The import only consumes cameras for psi_ref/initial bind; keep the
    # bundle's physical camera table index-aligned with scene_views so
    # alias/stage cameras resolve by physical index as before.
    bundle.lod_cameras = list(scene_views.lod_cameras)
    assert_l0_tensors_match(gaussians, bundle)

    parent_report: dict | None = None
    if parent_lod_checkpoint:
        parent_report = load_parent_lod_for_continuation(
            bundle, gaussians, path=parent_lod_checkpoint,
            device=device, step_scale=float(step_scale),
        )

    background = torch.tensor(
        [1, 1, 1] if config.white_background else [0, 0, 0],
        dtype=torch.float32, device=device,
    )
    resolver = AppearanceResolver(gaussians)
    trainer = SceneLodTrainer(
        gaussians=gaussians, bundle=bundle, scene_views=scene_views, resolver=resolver,
        background=background, kernel=float(config.kernel_size),
        config=config, output_dir=output_dir,
        scene_extent=float(scene_extent),
    )
    allowed_ids = set(scene_views.view_to_physical.keys())
    if parent_report is None:
        samples, per_level = trainer.build_samples(manifest, allowed_ids)
    else:
        # Extend the parent's frozen base stage record with the combined
        # manifest's scale-1 alias cameras BEFORE binding/sampling; the loaded
        # multi-level vendor model refuses a second bind_base_stage.
        alias_specs = manifest_alias_specs(manifest, allowed_ids, scene_views.view_to_physical)
        union = manifest_union_base_cameras(bundle, scene_views, alias_specs)
        parent_report["stage_extension"] = extend_parent_base_stage(
            bundle, union, step_scale=float(step_scale)
        )
        samples, per_level = trainer.build_samples(manifest, allowed_ids, bind_union_base=False)

    zooms = sorted({sample.zoom for sample in samples})
    step_scale_value = float(step_scale)
    for level_order, zoom in enumerate(zooms):
        expected = step_scale_value ** (level_order + 1)
        if not math.isclose(zoom, expected, rel_tol=1e-3):
            raise ValueError(
                f"Level {level_order + 1} zoom {zoom:g} does not match step_scale "
                f"{step_scale_value:g}^(level) = {expected:g}; pass a matching --step_scale."
            )
    if config.level_steps is not None and len(config.level_steps) != len(per_level):
        raise ValueError(
            f"level_steps has {len(config.level_steps)} budgets but the manifest defines "
            f"{len(per_level)} levels; counts must match."
        )

    completed_levels = int(parent_report["completed_levels"]) if parent_report else 0
    if parent_report is not None:
        if completed_levels >= len(per_level):
            raise ValueError(
                f"parent checkpoint already covers all {len(per_level)} manifest level(s); "
                "nothing to train. Pass the combined manifest's remaining levels or drop "
                "--parent_lod_checkpoint."
            )
        for order in range(completed_levels):
            zoom = zooms[order]
            expected = step_scale_value ** (order + 1)
            if not math.isclose(zoom, expected, rel_tol=1e-3):
                raise ValueError(
                    f"Combined manifest level {order + 1} zoom {zoom:g} does not match the "
                    f"parent checkpoint stage scale (step_scale {step_scale_value:g}^{order + 1})."
                )
            record_scale = float(bundle.lod.stage_records[order + 1]["scale"])
            if not math.isclose(record_scale, zoom, rel_tol=1e-3):
                raise ValueError(
                    f"Parent stage record {order + 1} scale {record_scale:g} does not match the "
                    f"combined manifest level zoom {zoom:g}."
                )
            from gaussianzoom_lod.stages import validate_stage_use
            validate_stage_use(
                bundle.lod.stage_records[order + 1], group_dedupe_stage(per_level[order])
            )

    level_summaries = []
    for level_order, group in enumerate(per_level):
        level_index = level_order + 1
        if completed_levels and level_order < completed_levels:
            # Completed under the parent checkpoint; verified above, not retrained.
            continue
        if not group:
            raise ValueError(f"level {level_index} has no usable samples")
        level_dir = os.path.join(output_dir, f"level{level_index}")
        os.makedirs(level_dir, exist_ok=True)
        stage_cameras = group_dedupe_stage(group)
        # Freeze invariant covers L0 AND every completed parent level
        # (0 .. level_index-1), not just the levels below the current order.
        frozen_before = snapshot_levels(bundle, tuple(range(level_index)))
        add_scene_level(bundle, stage_cameras)
        replay_pool = trainer.replay_pool_for_level(level_order, per_level)
        result = trainer.train_level(
            level_order=level_order, level_index=level_index, samples=group,
            densify_cameras=stage_cameras,
            replay_pool=replay_pool, level_dir=level_dir,
        )
        frozen_after = snapshot_levels(bundle, tuple(range(level_index)))
        assert_snapshot_unchanged(frozen_before, frozen_after)
        frozen_report = {
            "ok": True,
            "completed": completed_layers_frozen(bundle, up_to_level=level_index),
            "before": frozen_before,
            "after": frozen_after,
        }
        if not frozen_report["completed"]["ok"]:
            raise AssertionError("A completed parent or appearance parameter became trainable")
        write_json(os.path.join(level_dir, "freeze.json"), frozen_report)
        checkpoint_path = os.path.join(output_dir, f"l{level_index}_final.lod.pt")
        save_bundle(bundle, checkpoint_path, optimizer=None)
        densify_log_path = os.path.join(level_dir, "densify_log.json")
        write_json(densify_log_path, trainer.densify_log)
        level_summaries.append(level_result_to_dict(result, densify_log_path, checkpoint_path))

    summary = {
        "schema_version": 1,
        "kind": "skyfall_scene_lod_training",
        "start_checkpoint": os.path.abspath(start_checkpoint),
        "checkpoint_identity": file_identity(start_checkpoint),
        "supervision": os.path.abspath(manifest_path),
        "supervision_identity": file_identity(manifest_path, hash_file=True),
        "output_dir": os.path.abspath(output_dir),
        "gz_root": str(gz_root),
        "step_scale": float(step_scale),
        "seed": int(config.seed),
        "mix_ratio": float(config.mix_ratio),
        "level_steps": [int(v) for v in config.level_steps] if config.level_steps is not None else None,
        "checkpoint_steps": list(config.checkpoint_steps),
        "densify_until": config.densify_until,
        "densify_spec": (
            level_summaries[0]["densify_spec"]
            if level_summaries and all(
                level["densify_spec"] == level_summaries[0]["densify_spec"]
                for level in level_summaries
            )
            else None
        ),
        "frozen_color_cache": bool(config.cache_frozen_colors),
        "loss": {
            "mode": config.loss_mode,
            "hr_weight": float(config.loss_hr),
            "lr_weight": float(config.loss_lr),
            "geometry_weight": float(config.loss_geometry),
            "dssim_weight": float(config.loss_dssim),
            "hr_term": "signed_frequency_residual_l1" if config.loss_mode == "frequency" else "rgb",
            "geometry_kind": (
                "rade_camera_z_depth_normal_consistency"
                if config.loss_mode != "l1" and config.loss_geometry > 0 else None
            ),
        },
        "parent_lod_checkpoint": (
            os.path.abspath(parent_lod_checkpoint) if parent_lod_checkpoint else None
        ),
        "parent_checkpoint_identity": (
            file_identity(parent_lod_checkpoint, hash_file=True) if parent_lod_checkpoint else None
        ),
        "parent": parent_report,
        "completed_levels_skipped": completed_levels,
        "base_points": int(bundle.layer0().xyz.shape[0]),
        "unique_physical_cameras": len(scene_views.render_cameras),
        "l0_import_cameras": len(import_cameras),
        "l0_import_camera_reference": reference_source,
        "filter_3d_source": filter_source,
        "views_total": len(scene_views.view_to_physical),
        "real_replay_views": sum(
            1
            for view in manifest["views"]
            if str(view.get("source_stage", "stage1")) == "real_replay"
        ),
        "levels": level_summaries,
        "seconds": time.perf_counter() - started,
        "densify_log": trainer.densify_log,
    }
    summary_path = os.path.join(output_dir, "training_summary.json")
    write_json(summary_path, summary)
    write_train_lineage(output_dir, {
        "kind": "scene_lod_train",
        "start_checkpoint": file_identity(start_checkpoint),
        "supervision": file_identity(manifest_path, hash_file=True),
        "steps_per_level": int(config.steps_per_level),
        "level_steps": [int(v) for v in config.level_steps] if config.level_steps is not None else None,
        "checkpoint_steps": [int(v) for v in config.checkpoint_steps],
        "densify_until": int(config.densify_until) if config.densify_until is not None else None,
        "max_points_per_level": int(config.max_points_per_level),
        "loss_mode": config.loss_mode,
        "parent_lod_checkpoint": bool(parent_lod_checkpoint),
        "seed": int(config.seed),
    })
    return summary



def load_parent_lod_for_continuation(
    bundle, gaussians, *, path: str, device: str, step_scale: float,
) -> dict:
    """Load a completed parent LoD (+ appearance sidecar) for continuation.

    The parent tensors are restored unchanged (exact bytes, including L0 and
    per-level appearance embeddings); the frozen Skyfall appearance MLP stays
    in place from the base checkpoint. Validates:
    - L0 identity against the base ``gaussians`` (exact, includes filter_3d),
    - step_scale consistency,
    - completed levels frozen with non-empty layers and a consistent
      active level,
    - per-level appearance embeddings present, sized to their layers and
      frozen (never regenerated).
    """

    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise ValueError(f"parent_lod_checkpoint does not exist: {path}")
    side_path = path + ".appearance.pt"
    if bundle.appearance.enabled and not os.path.isfile(side_path):
        raise ValueError(
            f"parent_lod_checkpoint appearance sidecar missing: {side_path}. "
            "The parent run must save bundles through save_bundle()."
        )
    parent_appearance_before = appearance_digests(bundle)
    try:
        load_lod_onto_bundle(bundle, path, device=device)
    except FileNotFoundError as exc:
        raise ValueError(f"parent checkpoint load failed: {exc}") from exc

    # Exact base identity: the parent must sit on the SAME frozen L0 the base
    # checkpoint imports (bytes, including filter_3d); assert_l0_tensors_match
    # also re-checks the frozen flag and the frozen appearance MLP.
    assert_l0_tensors_match(gaussians, bundle)
    if bundle.appearance.enabled and appearance_digests(bundle)[0] != parent_appearance_before[0]:
        raise ValueError("Parent L0 appearance differs from the supplied base checkpoint")
    if not math.isclose(float(bundle.lod.step_scale), float(step_scale), rel_tol=1e-6):
        raise ValueError(
            f"parent checkpoint step_scale {float(bundle.lod.step_scale):g} does not match "
            f"the requested step_scale {float(step_scale):g}"
        )
    layers = list(bundle.lod.layers)
    completed = len(layers) - 1
    if completed < 1:
        raise ValueError(
            f"parent checkpoint has {len(layers)} layer(s); at least one completed detail "
            "level is required for continuation"
        )
    for index, layer in enumerate(layers):
        if index < completed and not bool(getattr(layer, "frozen", False)):
            raise ValueError(f"parent level {index} is not frozen in the parent checkpoint")
        if int(layer.xyz.shape[0]) == 0:
            raise ValueError(f"parent level {index} is empty in the parent checkpoint")
    if int(bundle.lod.active_level) != completed:
        raise ValueError(
            f"parent checkpoint active_level {int(bundle.lod.active_level)} does not match "
            f"its completed level count {completed}"
        )
    if bundle.appearance.enabled:
        embeddings = list(bundle.appearance.layer_embeddings or [])
        if len(embeddings) != len(layers):
            raise ValueError(
                f"parent appearance sidecar has {len(embeddings)} layer embedding(s) "
                f"for {len(layers)} layer(s)"
            )
        for index, (emb, layer) in enumerate(zip(embeddings, layers)):
            if not torch.is_tensor(emb):
                raise ValueError(f"parent appearance layer {index} embedding is missing")
            if int(emb.shape[0]) != int(layer.xyz.shape[0]):
                raise ValueError(
                    f"parent appearance layer {index} has {int(emb.shape[0])} row(s) "
                    f"for {int(layer.xyz.shape[0])} Gaussian(s)"
                )
            if bool(emb.requires_grad):
                raise ValueError(f"parent appearance layer {index} embedding is trainable")
    return {
        "checkpoint": path,
        "completed_levels": completed,
        "stage_scales": [float(record["scale"]) for record in bundle.lod.stage_records],
        "points_per_level": [int(layer.xyz.shape[0]) for layer in layers],
        "appearance_digests": appearance_digests(bundle),
        "appearance_digests_before_load": parent_appearance_before,
    }


def appearance_digests(bundle) -> list[dict | None]:
    """Byte-level digests of every per-level appearance embedding (evidence)."""

    embeddings = list(getattr(bundle.appearance, "layer_embeddings", None) or [])
    return [tensor_digest(emb) if torch.is_tensor(emb) else None for emb in embeddings]


def extend_parent_base_stage(bundle, union_cameras, *, step_scale: float) -> dict:
    """Extend the loaded parent's base stage record to the combined union.

    The vendor ``validate_next_stage`` requires every new stage camera name to
    exist in the base stage record. A parent trained on the z2 manifest only
    declared z2 aliases; the combined z2/z4 manifest introduces new scale-1
    aliases built from the SAME physical cameras. This replaces ONLY
    ``stage_records[0]`` with a re-captured scale-1 record over the combined
    union after verifying every parent base camera is unchanged (raster,
    intrinsics, pose) in the combined union, then re-validates the whole
    chain. Completed stage records are preserved untouched and no parent
    tensor (layers, psi_ref, filter_3d, appearance) is modified.
    """

    add_gz_src()
    from gaussianzoom_lod.stages import capture_stage, validate_stage_records

    records = list(bundle.lod.stage_records)
    if not records:
        raise ValueError("parent checkpoint has no stage records to extend")
    old_base = records[0]
    if float(old_base["scale"]) != 1.0:
        raise ValueError(
            f"parent base stage scale must be 1.0 (got {float(old_base['scale']):g})"
        )
    new_base = capture_stage(union_cameras, scale=1.0)
    new_index = {cam["name"]: cam for cam in new_base["cameras"]}
    old_names = {cam["name"] for cam in old_base["cameras"]}
    added = [cam["name"] for cam in new_base["cameras"] if cam["name"] not in old_names]
    for old_cam in old_base["cameras"]:
        name = old_cam["name"]
        match = new_index.get(name)
        if match is None:
            raise ValueError(
                f"parent base stage camera {name!r} is missing from the combined manifest union"
            )
        for key in ("width", "height"):
            if int(old_cam[key]) != int(match[key]):
                raise ValueError(
                    f"parent base stage camera {name!r}: {key} {old_cam[key]} changed in the "
                    f"combined manifest ({match[key]})"
                )
        for key in ("fx", "fy", "cx", "cy"):
            if not math.isclose(float(old_cam[key]), float(match[key]), rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError(
                    f"parent base stage camera {name!r}: {key} {old_cam[key]!r} changed in the "
                    f"combined manifest ({match[key]!r})"
                )
        for row_old, row_new in zip(old_cam["w2c"], match["w2c"]):
            for value_old, value_new in zip(row_old, row_new):
                if not math.isclose(float(value_old), float(value_new), rel_tol=1e-5, abs_tol=1e-6):
                    raise ValueError(
                        f"parent base stage camera {name!r}: pose changed in the combined manifest"
                    )
    chain = [new_base] + [dict(record) for record in records[1:]]
    validate_stage_records(chain, float(step_scale))
    # Only now mutate the plain-data stage record; layers/appearance untouched.
    bundle.lod.stage_records[0] = new_base
    return {
        "preserved_base_cameras": len(old_base["cameras"]),
        "combined_base_cameras": len(new_base["cameras"]),
        "added_alias_names": added,
        "stage_scales": [float(record["scale"]) for record in chain],
    }


# --------------------------------------------------------------------------- #
# Orchestration helpers
# --------------------------------------------------------------------------- #


def group_dedupe_stage(group: Sequence[LevelSample]) -> list:
    """Dedupe a level's stage cameras by alias name (identical physical ROI sets)."""

    seen: dict[str, Any] = {}
    for sample in group:
        name = sample.stage_camera.name
        if name not in seen:
            seen[name] = sample.stage_camera
    return list(seen.values())


def level_result_to_dict(result: LevelResult, densify_log_path: str, checkpoint_path: str) -> dict:
    return {
        "zoom_factor": float(result.zoom_factor),
        "level_index": int(result.level_index),
        "stage_scale": float(result.stage_scale),
        "n_stage_cameras": int(result.n_stage_cameras),
        "n_samples": int(result.n_samples),
        "bootstrap_views": int(result.bootstrap_views),
        "steps_requested": int(result.steps_requested),
        "steps_executed": int(result.steps_executed),
        "coverage_steps": int(result.coverage_steps),
        "replay_steps": int(result.replay_steps),
        "unique_targets_seen": int(result.unique_targets_seen),
        "coverage_complete": bool(result.unique_targets_seen >= result.n_samples),
        "densify_events": int(result.densify_events),
        "densify_totals": dict(result.densify_totals),
        "points_start": int(result.points_start),
        "points_end": int(result.points_end),
        "l0_points": int(result.l0_points),
        "eval_before": dict(result.eval_before),
        "eval_after": dict(result.eval_after),
        "pngs": {key: dict(value) for key, value in result.pngs.items()},
        "checkpoint": os.path.abspath(checkpoint_path),
        "appearance_sidecar": os.path.abspath(checkpoint_path) + ".appearance.pt",
        "densify_until": int(result.densify_until),
        "densify_spec": dict(result.densify_spec),
        "checkpoints": [dict(entry) for entry in result.checkpoints],
        "densify_log": os.path.abspath(densify_log_path),
        "seconds": float(result.seconds),
    }
