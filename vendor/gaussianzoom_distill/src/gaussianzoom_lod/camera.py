"""Shared pinhole camera for the GaussianZoom LoD component.

Contract (fixed by the LoD design):
    Camera(name, width, height, fx, fy, cx, cy, w2c)
        name     -- image basename ("" when synthetic)
        width, height -- raster pixel size the intrinsics refer to
        fx, fy, cx, cy -- pinhole intrinsics in pixels of that raster
        w2c      -- [4, 4] float32 row-major world-to-camera transform,
                    COLMAP convention: R*world + t (last row [0, 0, 0, 1]),
                    so ``center = inverse(w2c)[:3, 3]`` is the camera center
                    in world coordinates.

All pose math is standard (column-vector convention, ``w2c @ [x, y, z, 1]``).
Instances are effectively immutable: ``resized``/``zoomed``/``to`` return new
cameras sharing the (never mutated) ``w2c`` tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Tuple, Union

import torch

DeviceLike = Union[torch.device, str]


@dataclass(eq=False)
class Camera:
    """Pinhole camera with fixed intrinsics and world-to-camera pose."""

    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    w2c: torch.Tensor = field(repr=False)  # [4, 4] float32 row-major

    @property
    def center(self) -> torch.Tensor:
        """Camera center in world coordinates: ``inverse(w2c)[:3, 3]``."""
        return torch.linalg.inv(self.w2c)[:3, 3]

    @property
    def device(self) -> torch.device:
        return self.w2c.device

    def resized(self, width: int, height: int) -> "Camera":
        """New Camera for the same pose/content on a ``width`` x ``height`` raster.

        Intrinsics are re-expressed in the new pixel grid with the exact
        per-axis ratios ``width / self.width`` and ``height / self.height``
        (no nearest/downscale assumptions).
        """
        sx = float(width) / self.width
        sy = float(height) / self.height
        return replace(
            self,
            width=width,
            height=height,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
        )

    def zoomed(self, factor: float, center_uv: Tuple[float, float] = (0.5, 0.5)) -> "Camera":
        """New Camera with virtual focal zoom, raster size kept.

        ``fx``/``fy`` are multiplied by ``factor`` (same image content, now
        covering a ``factor``-times smaller world angle) while the raster
        stays ``width`` x ``height``.  The principal point is shifted so the
        world point projecting to ``center_uv`` (fractions of the raster,
        default image center) stays put:

            cx' = factor * (cx - center_u * width)  + width / 2
            cy' = factor * (cy - center_v * height) + height / 2

        Pose (``w2c``) is unchanged: zoom is a pure intrinsic change.
        """
        cu, cv = center_uv
        cx = factor * (self.cx - cu * self.width) + self.width / 2.0
        cy = factor * (self.cy - cv * self.height) + self.height / 2.0
        return replace(
            self,
            fx=self.fx * factor,
            fy=self.fy * factor,
            cx=cx,
            cy=cy,
        )

    def to(self, device: DeviceLike) -> "Camera":
        """New Camera with ``w2c`` moved to ``device`` (scalars unchanged)."""
        if self.w2c.device == torch.device(device):
            return self
        return replace(self, w2c=self.w2c.to(device))
