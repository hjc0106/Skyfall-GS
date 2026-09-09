"""Data contracts shared by refinement backends and the zoom entry point.

The contracts are deliberately lightweight.  They carry enough camera and
generation information to make an output reproducible and cache-safe without
coupling the refinement package to a particular VLM, SR model, or rasterizer
implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from PIL import Image


def _jsonable(value: Any) -> Any:
    """Convert common NumPy/PyTorch/dataclass values to JSON-safe objects."""

    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    # Torch tensors and NumPy arrays/scalars expose one or both of these APIs.
    detached = getattr(value, "detach", None)
    if callable(detached):
        return _jsonable(value.detach().cpu().tolist())
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _jsonable(tolist())

    return str(value)


def to_jsonable(value: Any) -> Any:
    """Public JSON conversion helper used by manifests and cache metadata."""

    return _jsonable(value)


def canonical_json(value: Any) -> str:
    """Return a stable JSON representation for cache keys."""

    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _nested_tuple(value: Any) -> Tuple[Tuple[float, ...], ...]:
    value = _jsonable(value)
    if value is None:
        return tuple()
    if value and not isinstance(value[0], (list, tuple)):
        value = [value]
    return tuple(tuple(float(item) for item in row) for row in value)


def _vector_tuple(value: Any) -> Tuple[float, ...]:
    value = _jsonable(value)
    if value is None:
        return tuple()
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return tuple(float(item) for item in value)


@dataclass(frozen=True)
class CameraSnapshot:
    """Serializable camera identity and intrinsics used for generation/cache keys."""

    image_name: str
    uid: int
    colmap_id: Any
    image_width: int
    image_height: int
    fov_x: float
    fov_y: float
    cx: float
    cy: float
    R: Tuple[Tuple[float, ...], ...]
    T: Tuple[float, ...]
    znear: float = 0.01
    zfar: float = 100.0

    @classmethod
    def from_camera(cls, camera: Any) -> "CameraSnapshot":
        """Capture the fields that affect the 3DGS projection and image size."""

        return cls(
            image_name=str(getattr(camera, "image_name", "")),
            uid=int(getattr(camera, "uid", -1)),
            colmap_id=_jsonable(getattr(camera, "colmap_id", None)),
            image_width=int(getattr(camera, "image_width")),
            image_height=int(getattr(camera, "image_height")),
            fov_x=float(getattr(camera, "FoVx")),
            fov_y=float(getattr(camera, "FoVy")),
            cx=float(getattr(camera, "cx")),
            cy=float(getattr(camera, "cy")),
            R=_nested_tuple(getattr(camera, "R")),
            T=_vector_tuple(getattr(camera, "T")),
            znear=float(getattr(camera, "znear", 0.01)),
            zfar=float(getattr(camera, "zfar", 100.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_name": self.image_name,
            "uid": self.uid,
            "colmap_id": _jsonable(self.colmap_id),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "fov_x": self.fov_x,
            "fov_y": self.fov_y,
            "cx": self.cx,
            "cy": self.cy,
            "R": [list(row) for row in self.R],
            "T": list(self.T),
            "znear": self.znear,
            "zfar": self.zfar,
        }


@dataclass
class ImageData:
    """An image plus provenance metadata shared by backends."""

    image: Image.Image
    name: str = ""
    source: str = ""
    scale: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.image, Image.Image):
            raise TypeError(f"image must be a PIL.Image.Image, got {type(self.image)!r}")
        if self.image.mode not in ("RGB", "RGBA"):
            self.image = self.image.convert("RGB")

    @property
    def width(self) -> int:
        return int(self.image.width)

    @property
    def height(self) -> int:
        return int(self.image.height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "scale": self.scale,
            "width": self.width,
            "height": self.height,
            "metadata": _jsonable(self.metadata),
        }


@dataclass(frozen=True)
class PromptDescription:
    """Structured VLM/fixed-prompt output.

    ``shared_region_description`` is intended to remain stable across zoom
    levels.  ``current_scale_description`` is the level-specific part and is
    regenerated or loaded independently at each scale.
    """

    shared_region_description: str = ""
    current_scale_description: str = ""
    visible_features: Tuple[str, ...] = tuple()
    preserve_structure: Tuple[str, ...] = tuple()
    uncertain_information: Tuple[str, ...] = tuple()
    source_prompt: str = ""
    target_prompt: str = ""
    provider: str = "fixed"
    config: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PromptDescription":
        if not isinstance(data, Mapping):
            raise TypeError(f"PromptDescription must be a mapping, got {type(data)!r}")

        def _strings(key: str) -> Tuple[str, ...]:
            value = data.get(key, ())
            if value is None:
                return tuple()
            if isinstance(value, str):
                return (value,) if value else tuple()
            return tuple(str(item) for item in value)

        return cls(
            shared_region_description=str(data.get("shared_region_description", "")),
            current_scale_description=str(data.get("current_scale_description", "")),
            visible_features=_strings("visible_features"),
            preserve_structure=_strings("preserve_structure"),
            uncertain_information=_strings("uncertain_information"),
            source_prompt=str(data.get("source_prompt", "")),
            target_prompt=str(data.get("target_prompt", "")),
            provider=str(data.get("provider", "fixed")),
            config=dict(data.get("config") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shared_region_description": self.shared_region_description,
            "current_scale_description": self.current_scale_description,
            "visible_features": list(self.visible_features),
            "preserve_structure": list(self.preserve_structure),
            "uncertain_information": list(self.uncertain_information),
            "source_prompt": self.source_prompt,
            "target_prompt": self.target_prompt,
            "provider": self.provider,
            "config": _jsonable(dict(self.config)),
        }


@dataclass
class RenderBundle:
    """RGB/depth/alpha outputs of one rasterizer call."""

    rgb: Any
    depth: Any = None
    alpha: Any = None
    camera: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RefinementRequest:
    """Input contract for every refinement backend."""

    image: Image.Image
    checkpoint: str
    camera: CameraSnapshot
    zoom_factor: float
    sr_scale: float
    prompt: PromptDescription
    model_config: Mapping[str, Any] = field(default_factory=dict)
    prompt_config: Mapping[str, Any] = field(default_factory=dict)
    cache_context: Mapping[str, Any] = field(default_factory=dict)
    level_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.image, Image.Image):
            raise TypeError(f"image must be a PIL.Image.Image, got {type(self.image)!r}")
        self.image = self.image.convert("RGB")
        self.checkpoint = os.path.abspath(os.fspath(self.checkpoint))
        if self.zoom_factor <= 0.0:
            raise ValueError(f"zoom_factor must be positive, got {self.zoom_factor}")
        if self.sr_scale <= 0.0:
            raise ValueError(f"sr_scale must be positive, got {self.sr_scale}")

    @property
    def expected_output_size(self) -> tuple[int, int]:
        return (
            max(1, int(round(self.camera.image_width * self.sr_scale))),
            max(1, int(round(self.camera.image_height * self.sr_scale))),
        )

    def image_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.image.mode.encode("utf-8"))
        digest.update(str(self.image.size).encode("ascii"))
        digest.update(self.image.tobytes())
        return digest.hexdigest()

    def cache_payload(self, backend_name: str) -> dict[str, Any]:
        checkpoint_stat: dict[str, Any] = {}
        try:
            stat = os.stat(self.checkpoint)
            checkpoint_stat = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        except OSError:
            # A synthetic checkpoint identifier is valid for tests and remote
            # runners; the path itself still participates in the key.
            checkpoint_stat = {"missing": True}

        return {
            "schema_version": 1,
            "backend": backend_name,
            "checkpoint": self.checkpoint,
            "checkpoint_stat": checkpoint_stat,
            "camera": self.camera.to_dict(),
            "zoom_factor": float(self.zoom_factor),
            "sr_scale": float(self.sr_scale),
            "prompt": self.prompt.to_dict(),
            "prompt_config": _jsonable(dict(self.prompt_config)),
            "cache_context": _jsonable(dict(self.cache_context)),
            "model_config": _jsonable(dict(self.model_config)),
            "input_image_sha256": self.image_sha256(),
        }


@dataclass
class RefinementResult:
    """Output contract returned by a refinement backend."""

    image: Image.Image
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)
    cache_key: str | None = None
    cache_hit: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.image, Image.Image):
            raise TypeError(f"image must be a PIL.Image.Image, got {type(self.image)!r}")
        self.image = self.image.convert("RGB")

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "width": self.image.width,
            "height": self.image.height,
            "metadata": _jsonable(self.metadata),
            "cache_key": self.cache_key,
            "cache_hit": self.cache_hit,
        }


@dataclass
class GeometryCorrespondence:
    """One view-pair correspondence used by DLoRAL geometry alignment.

    Pixel origin is the top-left pixel center ``(u, v) = (0, 0)``.  Stored
    flows are ``p_source - p_query`` in the query image's pixel units.
    Invalid locations are NaN on disk and never sent into ``grid_sample``.

    CFR frames are ``[neighbor, target]``.  ``target_to_source_flow`` is
    ``p_neighbor - p_target`` at target pixels and is the field that warps
    the previous (neighbor) feature onto the current (target) grid, matching
    ``flows_forward`` in ``CFR_model``.  ``source_to_target_flow`` is the
    reverse field computed from the neighbor's own depth, not by scattering
    the forward flow.
    """

    target_to_source_flow: Any
    valid_mask: Any
    source_size: Tuple[int, int]
    target_size: Tuple[int, int]
    target_to_source_grid: Any = None
    source_to_target_flow: Any = None
    reverse_valid_mask: Any = None
    confidence: Any = None
    camera_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiViewInput:
    """A neighboring view and its optional geometry-aligned observations."""

    name: str
    image: Image.Image | Any
    camera: Any = None
    depth: Any = None
    alpha: Any = None
    warped_image: Image.Image | Any = None
    valid_mask: Any = None
    pixel_flow: Any = None
    sample_grid: Any = None
    source_to_target_flow: Any = None
    reverse_valid_mask: Any = None
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WarpResult:
    """Target-to-source sampling grid and validity diagnostics.

    ``grid`` is the ``[-1, 1]`` sampler used by ``grid_sample``.  Feature
    propagation should consume ``pixel_flow`` instead: ``p_source - p_target``
    in pixel units.  Invalid locations are NaN, not zero, so a later warper
    cannot silently sample the neighbor at the same coordinates.
    """

    grid: Any
    valid_mask: Any
    projected_uv: Any
    coverage: float
    pixel_flow: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpatialTarget:
    """World-space region corresponding to a zoom view's valid surface.

    Neighbor ROIs must be derived from this 3D target rather than by copying
    normalized image coordinates from the source camera.
    """

    centroid_world: Tuple[float, float, float]
    median_depth: float
    confidence: float
    valid_coverage: float
    point_count: int
    points_world: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "centroid_world": list(self.centroid_world),
            "median_depth": float(self.median_depth),
            "confidence": float(self.confidence),
            "valid_coverage": float(self.valid_coverage),
            "point_count": int(self.point_count),
            "metadata": _jsonable(self.metadata),
        }


@dataclass(frozen=True)
class ProjectedROI:
    """Normalized ROI obtained by projecting a :class:`SpatialTarget`."""

    center_x: float
    center_y: float
    width: float
    height: float
    in_frustum_fraction: float
    clamped: bool
    confidence: float
    uid: int
    image_name: str
    unclamped_center: Tuple[float, float] = (0.0, 0.0)
    projected_span: Tuple[float, float] = (0.0, 0.0)

    def as_roi_dict(self) -> dict[str, float]:
        return {
            "center_x": float(self.center_x),
            "center_y": float(self.center_y),
            "width": float(self.width),
            "height": float(self.height),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.as_roi_dict(),
            "in_frustum_fraction": float(self.in_frustum_fraction),
            "clamped": bool(self.clamped),
            "confidence": float(self.confidence),
            "uid": int(self.uid),
            "image_name": self.image_name,
            "unclamped_center": [float(self.unclamped_center[0]), float(self.unclamped_center[1])],
            "projected_span": [float(self.projected_span[0]), float(self.projected_span[1])],
        }
