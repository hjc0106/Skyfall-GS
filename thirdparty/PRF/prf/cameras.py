from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prf.types import PinholeView

CAMERA_CONVENTION_FLIP = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


def load_training_views(path: Path | str, *, opengl_c2w: bool = False) -> list[PinholeView]:
    """Load Skyfall/NeRF-style transforms JSON as pinhole training cameras.

    Skyfall JAX satellite transforms store COLMAP/OpenCV C2W (no Y/Z flip).
    Pass opengl_c2w=True only for Blender/NeRFStudio OpenGL poses.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    width = int(data.get("w", 0))
    height = int(data.get("h", 0))
    views: list[PinholeView] = []
    for index, frame in enumerate(data["frames"]):
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if opengl_c2w:
            c2w = c2w @ CAMERA_CONVENTION_FLIP
        w2c = np.linalg.inv(c2w)
        image_w = int(frame.get("w", width))
        image_h = int(frame.get("h", height))
        file_path = str(frame.get("file_path", f"frame_{index:05d}"))
        views.append(
            PinholeView(
                view_id=Path(file_path).stem,
                width=image_w,
                height=image_h,
                fx=float(frame["fl_x"] if "fl_x" in frame else data["fl_x"]),
                fy=float(frame["fl_y"] if "fl_y" in frame else data["fl_y"]),
                cx=float(frame["cx"] if "cx" in frame else data["cx"]),
                cy=float(frame["cy"] if "cy" in frame else data["cy"]),
                c2w=c2w,
                w2c=w2c,
                center=c2w[:3, 3].copy(),
                file_path=file_path,
            )
        )
    if not views:
        raise ValueError(f"{path} contains no frames")
    return views
