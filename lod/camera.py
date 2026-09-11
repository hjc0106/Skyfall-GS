"""Convert Skyfall cameras into GaussianZoom pixel-pinhole cameras.

Skyfall stores FoV plus an NDC principal-point offset ``cx``/``cy`` consumed by
``getProjectionMatrix`` as ``x_ndc = (x/z)/tan(FoVx/2) + cx``. GaussianZoom uses
COLMAP pixel intrinsics. The conversions below match Skyfall's own
``compute_3D_filter`` pixel projection:

    cx_pixel = width * (cx_ndc + 1) / 2
    fx = width / (2 tan(FoVx/2))

Pose: Skyfall's ``world_view_transform`` is ``getWorld2View2(R, T).T`` for GLM.
GaussianZoom ``w2c`` is the untransposed 4x4, so ``w2c.T`` is the rasterizer
view matrix. Raster size is unchanged under focal zoom; do not infer LoD scale
from image width.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from .path import add_gz_src


def camera_image_stem(name: str | None) -> str:
    return Path(str(name or "")).stem


def find_named_camera(cameras: Sequence[Any], image_name: str) -> tuple[int, Any]:
    """Select a camera by image stem. Names, not list order, bind an ROI."""

    want = camera_image_stem(image_name)
    hits = [
        (index, camera)
        for index, camera in enumerate(cameras)
        if camera_image_stem(getattr(camera, "image_name", "")) == want
    ]
    if not hits:
        available = [camera_image_stem(getattr(camera, "image_name", "")) for camera in cameras]
        raise LookupError(f"no camera named {want!r}; available={available}")
    if len(hits) > 1:
        raise LookupError(f"multiple cameras named {want!r}")
    return hits[0]


def resolve_scene_camera(
    cameras: Sequence[Any],
    *,
    view_index: int,
    image_name: str | None = None,
    split: str = "train",
) -> tuple[Any, int]:
    """Prefer ``image_name``. Index is only a check, not the identity.

    If both are given and disagree, raise rather than put a locked ROI on
    a different image after a load-order change.
    """

    if view_index < 0 or view_index >= len(cameras):
        raise IndexError(f"view_index={view_index} out of range for {split} cameras (count={len(cameras)}).")
    if image_name:
        named_index, named = find_named_camera(cameras, image_name)
        indexed = cameras[view_index]
        if camera_image_stem(getattr(indexed, "image_name", "")) != camera_image_stem(image_name):
            raise RuntimeError(
                f"{split} view_index={view_index} is {camera_image_stem(indexed.image_name)!r}, "
                f"but the locked camera is {camera_image_stem(image_name)!r} at index {named_index}. "
                "ROI is bound to the image name, not the list index."
            )
        return named, named_index
    return cameras[view_index], view_index


class CameraWithStoredCenter:
    """GaussianZoom camera whose ``center`` is Skyfall's stored ``camera_center``.

    Satellite ``w2c`` is well-conditioned for the view matrix, but ``inv(w2c)``
    can drift from the value Skyfall stored as ``camera_center``. RaDe-GS uses
    ``campos`` for SH view directions, so keep the Skyfall center.
    """

    def __init__(self, camera, center: torch.Tensor):
        self._camera = camera
        self._center = center.detach().to(device=camera.w2c.device, dtype=camera.w2c.dtype).reshape(3).contiguous()

    @property
    def center(self) -> torch.Tensor:
        return self._center

    def __getattr__(self, name: str):
        return getattr(self._camera, name)

    def to(self, device) -> "CameraWithStoredCenter":
        return CameraWithStoredCenter(self._camera.to(device), self._center.to(device))

    def resized(self, *args, **kwargs) -> "CameraWithStoredCenter":
        return CameraWithStoredCenter(self._camera.resized(*args, **kwargs), self._center)

    def zoomed(self, *args, **kwargs) -> "CameraWithStoredCenter":
        return CameraWithStoredCenter(self._camera.zoomed(*args, **kwargs), self._center)


def ndc_principal_to_pixel(offset: float, size: int) -> float:
    return 0.5 * float(size) * (float(offset) + 1.0)


def pixel_principal_to_ndc(pixel: float, size: int) -> float:
    return 2.0 * float(pixel) / float(size) - 1.0


def projection_offset_delta_ndc(size: int) -> float:
    """NDC gap between Skyfall's ``P[0,2]=cx_ndc`` and GaussianZoom's COLMAP half-pixel.

    GaussianZoom uses ``(2 * cx_pixel + 1) / width - 1`` = ``cx_ndc + 1/width``.
    """

    return 1.0 / float(size)


def skyfall_w2c(camera) -> torch.Tensor:
    """Row-major world-to-camera matching GaussianZoom's ``Camera.w2c``."""

    return camera.world_view_transform.detach().transpose(0, 1).contiguous()


def skyfall_camera_to_lod(
    camera,
    *,
    gz_root: str | None = None,
    device=None,
    use_skyfall_center: bool = False,
) -> Any:
    """Build a GaussianZoom ``Camera`` from a Skyfall ``scene.cameras.Camera``.

    Default ``center`` follows GaussianZoom (``inv(w2c)``) so L0 ``psi_ref``
    stays a pose quantity. Pass ``use_skyfall_center=True`` when calling
    RaDe-GS: SH ``campos`` should be Skyfall's stored ``camera_center``.
    """

    add_gz_src(gz_root)
    from gaussianzoom_lod.camera import Camera as LodCamera

    width = int(camera.image_width)
    height = int(camera.image_height)
    w2c = skyfall_w2c(camera)
    stored = camera.camera_center.detach()
    if device is not None:
        w2c = w2c.to(device)
        stored = stored.to(device)
    lod = LodCamera(
        name=str(camera.image_name),
        width=width,
        height=height,
        fx=float(camera.focal_x),
        fy=float(camera.focal_y),
        cx=ndc_principal_to_pixel(float(camera.cx), width),
        cy=ndc_principal_to_pixel(float(camera.cy), height),
        w2c=w2c,
    )
    if not use_skyfall_center:
        return lod
    return CameraWithStoredCenter(lod, stored)


def zoom_stage_camera(skyfall_camera, roi, factor: float, *, gz_root: str | None = None, device=None):
    """L1 stage camera: same name, same pose, focal * factor, raster unchanged.

    ``validate_next_stage`` keys cameras by name and requires fx/fy to step by
    ``step_scale``. Do not use ``make_zoom_camera``'s ``_zoom2`` suffix here.
    ``campos`` uses Skyfall's stored center.
    """

    base = skyfall_camera_to_lod(
        skyfall_camera, gz_root=gz_root, device=device, use_skyfall_center=True
    )
    return base.zoomed(float(factor), center_uv=(float(roi.center_x), float(roi.center_y)))


def lod_camera_diagnostics(skyfall_camera, lod_camera) -> dict[str, float]:
    """Numbers that must be inspected for satellite FoV and large principal offsets."""

    width = int(skyfall_camera.image_width)
    height = int(skyfall_camera.image_height)
    w2c = skyfall_w2c(skyfall_camera)
    view_delta = (lod_camera.w2c.T - skyfall_camera.world_view_transform).abs().max().item()
    view_center = skyfall_camera.world_view_transform.inverse()[3, :3]
    inv_center = torch.linalg.inv(lod_camera.w2c)[:3, 3]
    used_center = lod_camera.center.to(device=view_center.device, dtype=view_center.dtype)
    stored_center = skyfall_camera.camera_center.to(device=view_center.device, dtype=view_center.dtype)
    return {
        "width": float(width),
        "height": float(height),
        "fx": float(lod_camera.fx),
        "fy": float(lod_camera.fy),
        "cx_pixel": float(lod_camera.cx),
        "cy_pixel": float(lod_camera.cy),
        "cx_ndc": float(skyfall_camera.cx),
        "cy_ndc": float(skyfall_camera.cy),
        "gz_proj_cx_minus_skyfall_ndc": projection_offset_delta_ndc(width),
        "gz_proj_cy_minus_skyfall_ndc": projection_offset_delta_ndc(height),
        "viewmatrix_max_abs": float(view_delta),
        "center_inv_vs_view": float((view_center - inv_center.to(device=view_center.device, dtype=view_center.dtype)).abs().max().item()),
        "center_used_vs_stored": float((used_center - stored_center).abs().max().item()),
        "center_stored_vs_view": float((stored_center - view_center).abs().max().item()),
        "w2c_det": float(torch.linalg.det(w2c).item()),
        "znear": float(skyfall_camera.znear),
        "zfar": float(skyfall_camera.zfar),
    }
