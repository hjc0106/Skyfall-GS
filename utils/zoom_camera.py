"""Zoom camera helpers for progressive zoom-refine MVP."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from scene.cameras import Camera
from utils.general_utils import PILtoTorch


@dataclass(frozen=True)
class NormalizedROI:
    """ROI in normalized image coordinates [0, 1]."""

    center_x: float
    center_y: float
    width: float
    height: float

    def validate_in_bounds(self) -> None:
        if not (0.0 <= self.center_x <= 1.0 and 0.0 <= self.center_y <= 1.0):
            raise ValueError(
                f"ROI center ({self.center_x}, {self.center_y}) must lie in [0, 1]."
            )
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError(f"ROI width/height must be positive, got {self.width}x{self.height}.")

        left = self.center_x - self.width / 2.0
        right = self.center_x + self.width / 2.0
        top = self.center_y - self.height / 2.0
        bottom = self.center_y + self.height / 2.0
        if left < 0.0 or right > 1.0 or top < 0.0 or bottom > 1.0:
            raise ValueError(
                "ROI rectangle exceeds image bounds: "
                f"center=({self.center_x}, {self.center_y}), size=({self.width}, {self.height})."
            )

    def validate_for_zoom(self, zoom_factor: float) -> None:
        self.validate_in_bounds()
        visible = 1.0 / zoom_factor
        if self.width > visible + 1e-6:
            raise ValueError(
                f"ROI width {self.width:.4f} exceeds zoom-{zoom_factor:g} visible window {visible:.4f}."
            )
        if self.height > visible + 1e-6:
            raise ValueError(
                f"ROI height {self.height:.4f} exceeds zoom-{zoom_factor:g} visible window {visible:.4f}."
            )

        half = 0.5 / zoom_factor
        if self.center_x - half < -1e-6 or self.center_x + half > 1.0 + 1e-6:
            raise ValueError(
                f"Zoom-{zoom_factor:g} crop centered at x={self.center_x:.4f} exceeds image horizontally."
            )
        if self.center_y - half < -1e-6 or self.center_y + half > 1.0 + 1e-6:
            raise ValueError(
                f"Zoom-{zoom_factor:g} crop centered at y={self.center_y:.4f} exceeds image vertically."
            )


def zoom_fov(base_fov: float, zoom_factor: float) -> float:
    return 2.0 * math.atan(math.tan(base_fov / 2.0) / zoom_factor)


def zoom_principal_point(
    base_cx: float,
    base_cy: float,
    roi: NormalizedROI,
    zoom_factor: float,
) -> Tuple[float, float]:
    """Principal point that recenters the ROI after zooming.

    ``getProjectionMatrix`` builds ``x_ndc = (x/z) / tan(FoVx/2) + cx``, and the satellite
    cameras carry a large ``cx`` that shifts the frustum onto the scene, so the base offset
    must be transformed rather than replaced. Requiring the point at normalized ``u`` to
    land at ``u = 0.5`` after scaling the focal length by ``zoom_factor`` gives:

        cx_zoom = (cx_base - (2u - 1)) * zoom_factor
    """
    cx = (base_cx - (2.0 * roi.center_x - 1.0)) * zoom_factor
    cy = (base_cy - (2.0 * roi.center_y - 1.0)) * zoom_factor
    return cx, cy


def make_zoom_camera(
    base_cam: Camera,
    roi: NormalizedROI,
    zoom_factor: float,
    image: torch.Tensor | None = None,
    image_name: str | None = None,
    uid: int | None = None,
) -> Camera:
    """Clone a camera with fixed R/T and zoomed intrinsics."""
    roi.validate_for_zoom(zoom_factor)

    fov_x = zoom_fov(float(base_cam.FoVx), zoom_factor)
    fov_y = zoom_fov(float(base_cam.FoVy), zoom_factor)
    cx, cy = zoom_principal_point(float(base_cam.cx), float(base_cam.cy), roi, zoom_factor)

    if image is None:
        image = torch.zeros((3, base_cam.image_height, base_cam.image_width), dtype=torch.float32)

    return Camera(
        colmap_id=base_cam.colmap_id,
        R=base_cam.R,
        T=base_cam.T,
        FoVx=fov_x,
        FoVy=fov_y,
        cx=cx,
        cy=cy,
        image=image,
        gt_alpha_mask=None,
        image_name=image_name or f"{base_cam.image_name}_zoom{zoom_factor:g}",
        uid=uid if uid is not None else base_cam.uid + 10000,
        depth=None,
        mask=torch.ones((1, base_cam.image_height, base_cam.image_width), dtype=torch.float32),
        data_device=str(base_cam.data_device),
        optimizing=False,
    )


def camera_from_pil_image(
    base_cam: Camera,
    pil_image: Image.Image,
    image_name: str,
    uid: int,
) -> Camera:
    image = PILtoTorch(pil_image, (base_cam.image_width, base_cam.image_height)).clamp(0.0, 1.0)
    return Camera(
        colmap_id=base_cam.colmap_id,
        R=base_cam.R,
        T=base_cam.T,
        FoVx=base_cam.FoVx,
        FoVy=base_cam.FoVy,
        cx=base_cam.cx,
        cy=base_cam.cy,
        image=image,
        gt_alpha_mask=None,
        image_name=image_name,
        uid=uid,
        depth=None,
        mask=torch.ones((1, base_cam.image_height, base_cam.image_width), dtype=torch.float32),
        data_device=str(base_cam.data_device),
        optimizing=False,
    )


def save_roi_overlay(
    base_cam: Camera,
    roi: NormalizedROI,
    out_path: str,
) -> None:
    img = (base_cam.original_image.detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    w, h = pil.size
    left = (roi.center_x - roi.width / 2.0) * w
    right = (roi.center_x + roi.width / 2.0) * w
    top = (roi.center_y - roi.height / 2.0) * h
    bottom = (roi.center_y + roi.height / 2.0) * h
    draw.rectangle([left, top, right, bottom], outline=(255, 64, 64), width=3)
    draw.ellipse(
        [roi.center_x * w - 4, roi.center_y * h - 4, roi.center_x * w + 4, roi.center_y * h + 4],
        fill=(255, 255, 0),
    )
    pil.save(out_path)


def nearest_train_cameras(
    target_cam: Camera,
    train_cameras: Sequence[Camera],
    k: int = 2,
) -> List[Camera]:
    candidates = [cam for cam in train_cameras if cam is not target_cam and cam.uid != target_cam.uid]
    if len(candidates) <= k:
        return candidates

    target_center = target_cam.camera_center.detach().cpu().numpy()
    dists = []
    for cam in candidates:
        center = cam.camera_center.detach().cpu().numpy()
        dist = float(np.linalg.norm(center - target_center))
        dists.append((dist, cam))
    dists.sort(key=lambda item: item[0])
    return [cam for _, cam in dists[:k]]
