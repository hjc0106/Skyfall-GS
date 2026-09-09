"""Camera/image adaptation for original-size and high-resolution supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from PIL import Image

from scene.cameras import Camera
from utils.general_utils import PILtoTorch

SupervisionMode = Literal["original", "highres"]


@dataclass(frozen=True)
class SupervisionSpec:
    """Explicit distinction between zoom/camera scale and SR pixel scale."""

    mode: SupervisionMode = "original"
    sr_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in ("original", "highres"):
            raise ValueError(f"Unsupported supervision mode: {self.mode}")
        if self.sr_scale <= 0.0:
            raise ValueError(f"sr_scale must be positive, got {self.sr_scale}")


def validate_sr_scale(sr_scale: float) -> float:
    value = float(sr_scale)
    if value <= 0.0:
        raise ValueError(f"sr_scale must be positive, got {sr_scale}")
    return value


def supervision_size(camera, mode: SupervisionMode, sr_scale: float) -> tuple[int, int]:
    """Return ``(width, height)`` expected by the supervision camera."""

    spec = SupervisionSpec(mode=mode, sr_scale=validate_sr_scale(sr_scale))
    if spec.mode == "original":
        return int(camera.image_width), int(camera.image_height)
    return (
        max(1, int(round(camera.image_width * spec.sr_scale))),
        max(1, int(round(camera.image_height * spec.sr_scale))),
    )


def adapt_image_for_supervision(
    image: Image.Image,
    camera,
    *,
    mode: SupervisionMode,
    sr_scale: float,
) -> Image.Image:
    """Normalize a backend output to the image size used by its camera."""

    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL image, got {type(image)!r}")
    width, height = supervision_size(camera, mode, sr_scale)
    image = image.convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image


def _resized_mask(camera, width: int, height: int) -> torch.Tensor:
    mask = getattr(camera, "original_mask", None)
    if mask is None or not isinstance(mask, torch.Tensor) or mask.numel() == 0:
        return torch.ones((1, height, width), dtype=torch.float32)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"Camera mask must be [C,H,W] or [H,W], got {tuple(mask.shape)}")
    mask = mask[:1].float().unsqueeze(0)
    return F.interpolate(mask, size=(height, width), mode="nearest")[0]


def _clone_camera_with_image(
    base_camera,
    image: torch.Tensor,
    *,
    width: int,
    height: int,
    image_name: str,
    uid: int,
) -> Camera:
    return Camera(
        colmap_id=base_camera.colmap_id,
        R=base_camera.R,
        T=base_camera.T,
        FoVx=base_camera.FoVx,
        FoVy=base_camera.FoVy,
        # cx/cy are normalized projection offsets, not pixel coordinates.  They
        # therefore stay unchanged when only the image resolution changes.
        cx=base_camera.cx,
        cy=base_camera.cy,
        image=image,
        gt_alpha_mask=None,
        image_name=image_name,
        uid=uid,
        depth=None,
        mask=_resized_mask(base_camera, width, height),
        data_device=str(base_camera.data_device),
        trans=getattr(base_camera, "trans", (0.0, 0.0, 0.0)),
        scale=getattr(base_camera, "scale", 1.0),
        optimizing=False,
    )


def build_supervision_camera(
    base_camera,
    refined_image: Image.Image,
    *,
    mode: SupervisionMode = "original",
    sr_scale: float = 1.0,
    image_name: str | None = None,
    uid: int | None = None,
) -> Camera:
    """Attach a refined image while preserving the zoom camera projection.

    In ``original`` mode the image is resized back to the zoom render size.  In
    ``highres`` mode the SR pixels are retained and the Camera constructor
    recomputes pixel focal lengths from the larger image dimensions.  FoV and
    normalized principal-point offsets remain exactly those of ``base_camera``.
    """

    image = adapt_image_for_supervision(
        refined_image,
        base_camera,
        mode=mode,
        sr_scale=sr_scale,
    )
    width, height = image.size
    image_tensor = PILtoTorch(image, image.size).clamp(0.0, 1.0)
    return _clone_camera_with_image(
        base_camera,
        image_tensor,
        width=width,
        height=height,
        image_name=image_name or f"{base_camera.image_name}_supervision",
        uid=uid if uid is not None else int(base_camera.uid) + 20000,
    )


def build_render_camera(
    base_camera,
    *,
    mode: SupervisionMode = "original",
    sr_scale: float = 1.0,
    image_name: str | None = None,
    uid: int | None = None,
) -> Camera:
    """Create a blank camera with the supervision resolution for rendering."""

    width, height = supervision_size(base_camera, mode, sr_scale)
    blank = torch.zeros((3, height, width), dtype=torch.float32)
    return _clone_camera_with_image(
        base_camera,
        blank,
        width=width,
        height=height,
        image_name=image_name or f"{base_camera.image_name}_render",
        uid=uid if uid is not None else int(base_camera.uid) + 21000,
    )


__all__ = [
    "SupervisionMode",
    "SupervisionSpec",
    "adapt_image_for_supervision",
    "build_render_camera",
    "build_supervision_camera",
    "supervision_size",
    "validate_sr_scale",
]
