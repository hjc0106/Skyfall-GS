"""Pluggable image-refinement components for the independent zoom pipeline.

The package intentionally keeps the entry-point responsibilities separate from
the 3DGS training loop.  A refiner only receives a :class:`RefinementRequest`
and returns a :class:`RefinementResult`; camera courses and checkpointing stay
in ``train_zoom_gen.py``.
"""

from .types import (
    CameraSnapshot,
    GeometryCorrespondence,
    ImageData,
    MultiViewInput,
    ProjectedROI,
    PromptDescription,
    RefinementRequest,
    RefinementResult,
    RenderBundle,
    SpatialTarget,
    WarpResult,
)

__all__ = [
    "CameraSnapshot",
    "GeometryCorrespondence",
    "ImageData",
    "MultiViewInput",
    "ProjectedROI",
    "PromptDescription",
    "RefinementRequest",
    "RefinementResult",
    "RenderBundle",
    "SpatialTarget",
    "WarpResult",
]
