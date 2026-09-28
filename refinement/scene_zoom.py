"""Whole-scene multiscale zoom supervision (``prepare_scene_zoom``).

Turns one Skyfall checkpoint plus its dataset configuration into the
``skyfall_scene_zoom`` supervision manifest consumed by
``scripts/train_scene_lod.py``:

* **Stage 1** (no ``--flowedit_manifests``): every real TRAIN view of the
  checkpoint provides the base RGB.  Zoom targets are true super-resolution
  crops of the real photograph -- never GS renders.
* **Stage 2** (``--flowedit_manifests``): the base RGB of each view comes from
  the ``flowedit_views.json`` manifest written by the FlowEdit stage (only
  ``kind == 'skyfall_flowedit_views'`` is accepted).  A prior Stage 1
  ``supervision.json`` (``--real_supervision``) is carried over verbatim, so
  real views are never regenerated.

``--include_real_replay`` (Stage 2 only) additionally collects the checkpoint's
native TRAIN views as replay-only manifest entries (``source_stage
real_replay``): their base images/cameras feed base replay supervision in the
LoD trainer, but they never receive tile plans, prompts or SR targets and never
count against ``--max_views``. The L0 import in both generation and training
uses these native TRAIN cameras as the stable physical base-camera reference
for LoD birth stats, and the checkpoint's own PLY-persisted ``filter_3D`` is
kept verbatim (native fallback over the same reference cameras, reported, when
the checkpoint carries none).

``--target_view_ids`` (sharded generation) restricts the new SR targets to the
named collected views.  View collection, replay entries, manifest view entries
and the geometry neighbor pool stay complete, so independent output
directories can be generated in parallel and merged without changing neighbor
selection or parent-render semantics.

For each view and zoom factor ``z`` the whole frame is covered by an exact
``z x z`` tile grid.  Every tile uses the normalized ROI
``center=((c+0.5)/z, (r+0.5)/z), width=height=1/z`` so the zoom camera derives
through the repository zoom math (``zoom_fov``/``zoom_principal_point``, the
same math as ``make_zoom_camera``).  The SR input is the tile crop at
``base/z`` pixels and ``request.sr_scale=z``; each zoom level therefore runs
its own DLoRAL backend instance with ``upscale=z`` so the worker prepares and
generates at the native SR raster (``raw_output_size == requested_size``).

Geometry (depth/alpha and cross-view flows) is estimated from checkpoint
renders at the tile's crop raster, while every RGB image -- target crop and
neighbor crop alike -- comes from the real/repaired photograph.  GPU geometry,
Qwen3-VL prompting and DLoRAL refinement run strictly sequentially; the scene
is released before any generative phase.  Per-tile geometry scratch is deleted
as soon as the matching SR target exists; the DLoRAL/CachedRefiner ephemeral

Depth provenance: every raw ``render_depth`` is converted to expected
camera-space Z through ``lod.depth`` using the backend that actually rendered
it, and each tile/sample carries a versioned ``geometry_identity`` (backend,
raw depth meaning, camera-Z output).  Resume and Stage 2
``--real_supervision`` refuse targets whose geometry contract is missing or
mismatched instead of reinterpreting legacy buffers.

Alignment modes (``--dloral_alignment``): ``geometry`` reuses the existing
external-flow path built from depth-reprojected checkpoint renders; ``spynet``
selects neighbors the same way but lets DLoRAL's native SpyNet warp the real
neighbor frame; ``target_only`` is explicit single-view DLoRAL.  A tile whose
geometry stage found no usable neighbor runs ``target_only`` and is reported
in the manifest provenance -- never silently.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import time
import zlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from lod.depth import depth_from_rade, skyfall_expected_z
from refinement.base import CachedRefiner
from refinement.dloral_backend import DLoRALBackend
from refinement.geometry_warp import (
    build_aligned_multiview_inputs,
    estimate_depth_scene_scale,
    estimate_spatial_target,
    pixel_footprint,
    rank_cameras_by_spatial_overlap,
)
from refinement.qwen3_vlm import Qwen3PromptProvider
from refinement.types import (
    CameraSnapshot,
    MultiViewInput,
    PromptDescription,
    RefinementRequest,
    to_jsonable,
)
from refinement.vlm_prompt import PromptCache, PromptManager
from train_zoom_gen import apply_stage1_cfg_to_args, load_stage1_cfg
from utils.zoom_camera import NormalizedROI, zoom_fov, zoom_principal_point


FLOWEDIT_KIND = "skyfall_flowedit_views"
SCENE_ZOOM_KIND = "skyfall_scene_zoom"
REAL_REPLAY_STAGE = "real_replay"
SCRATCH_DIRNAME = "_scratch"
PROMPTS_DIRNAME = "prompts"
BASE_DIRNAME = "base"
TARGETS_DIRNAME = "targets"
GENERATION_CACHE_DIRNAME = "generation_cache"
PROMPT_CACHE_DIRNAME = "prompt_cache"

LOCAL_EDIT_DIRNAME = "local_edit"
PROGRESSIVE_RGB_CONTRACT = "normalized_float_clamped_v1"

#: Schema of the recorded ``geometry_identity`` provenance payload.
GEOMETRY_IDENTITY_VERSION = 1

#: Raw ``render_depth`` meaning per rasterizer backend; must stay in sync with
#: ``lod/depth.py``.  Backends without a listed conversion are rejected instead
#: of silently guessed.
_SOURCE_DEPTH_BY_BACKEND = {
    "diff_gauss": "accumulated_camera_z",
    "rade": "ray_distance",
}

#: Alpha under which a rendered depth sample is treated as empty (no surface).
_DEPTH_VALID_EPS = 1e-4


@dataclass
class SceneZoomConfig:
    """All knobs of one ``prepare_scene_zoom`` run."""

    start_checkpoint: str
    output_dir: str
    resume: bool = False
    zoom_factors: Sequence[float] = (2.0, 4.0)
    resolution: int = 0
    max_views: int = 0
    #: Sharded generation: restrict NEW SR targets to these collected view ids.
    #: The full view/context/neighbor pool is still collected.  Empty = all.
    target_view_ids: Sequence[str] = ()
    seed: int = 0
    source_path: str | None = None
    flowedit_manifests: Sequence[str] = ()
    real_supervision: str | None = None
    vlm_model_path: str | None = None
    vlm_python: str | None = None
    vlm_device: str = "cuda:0"
    vlm_max_new_tokens: int = 768
    vlm_max_image_size: int = 1024
    dloral_root: str | None = None
    dloral_sd_path: str | None = None
    dloral_ckpt: str | None = None
    dloral_spynet: str | None = None
    dloral_python: str | None = None
    dloral_device: str = "cuda:0"
    dloral_stages: int = 1
    dloral_process_size: int = 512
    dloral_align_method: str = "adain"
    dloral_alignment: str = "geometry"
    dloral_max_roundtrip_error_px: float | None = None
    # Reported per tile; a tile with an available geometry neighbor keeps it.
    min_reprojection_coverage: float = 0.0
    geometry_neighbor_count: int = 1
    geometry_pool_size: int = 8
    min_target_confidence: float = 0.05
    min_in_frustum: float = 1e-3
    depth_abs_tolerance: float = 5.0
    depth_rel_tolerance: float = 1e-5
    depth_scene_scale: float = -1.0
    alpha_threshold: float = 1e-4
    min_depth: float = 1e-4
    geometry_batch_views: int = 4
    prompt_cache_dir: str | None = None
    generation_cache_dir: str | None = None
    # Progressive supervision: render G(t-1) instead of the G0 source model.
    progressive: bool = False
    parent_lod_checkpoint: str | None = None
    previous_supervision: str | None = None
    local_flowedit_config: str | None = None
    # Stage 2 only: also collect the checkpoint's native TRAIN views as
    # replay-only manifest entries (no SR targets are ever generated for them).
    include_real_replay: bool = False
    step_scale: float = 2.0
    no_generation_cache: bool = False


@dataclass
class ViewTask:
    """One base view to supervise."""

    view_id: str
    snapshot: CameraSnapshot
    base_image_path: str
    source_stage: str
    appearance_uid: int | None
    episode_idx: int | None = None
    source_checkpoint: str = ""
    mask_path: str | None = None
    context_image_path: str | None = None


@dataclass
class TileSpec:
    """One zoom tile of one view."""

    zoom_factor: float
    level_index: int
    row: int
    col: int
    sample_id: str
    alias: str
    uid: int
    roi: NormalizedROI
    roi_dict: dict[str, float]
    crop_box: tuple[int, int, int, int]
    crop_size: tuple[int, int]


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _write_json(path: str | os.PathLike[str], payload: Any) -> None:
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(os.path.abspath(os.fspath(path)), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _snapshot_from_dict(data: Mapping[str, Any]) -> CameraSnapshot:
    return CameraSnapshot(
        image_name=str(data["image_name"]),
        uid=int(data.get("uid", -1)),
        colmap_id=data.get("colmap_id"),
        image_width=int(data["image_width"]),
        image_height=int(data["image_height"]),
        fov_x=float(data["fov_x"]),
        fov_y=float(data["fov_y"]),
        cx=float(data["cx"]),
        cy=float(data["cy"]),
        R=tuple(tuple(float(item) for item in row) for row in data["R"]),
        T=tuple(float(item) for item in data["T"]),
        znear=float(data.get("znear", 0.01)),
        zfar=float(data.get("zfar", 100.0)),
    )


def _tensor_to_pil(image: Any) -> Image.Image:
    """Encode normalized float RGB (including HDR highlights), or uint8 RGB."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    tensor = image if isinstance(image, torch.Tensor) else torch.as_tensor(image)
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
        byte_units = tensor.dtype == torch.uint8
        if not byte_units and not tensor.is_floating_point():
            raise TypeError("RGB tensors must be normalized floating point or uint8")
        tensor = tensor[:3].detach().cpu().float()
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("Cannot encode non-finite RGB pixels")
        if byte_units:
            tensor = tensor / 255.0
        array = (tensor.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(array, mode="RGB")
    raise TypeError(f"Unsupported image payload: {type(image)!r}")


def _plane(tensor: Any) -> torch.Tensor | None:
    if tensor is None:
        return None
    value = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
    while value.ndim > 2:
        value = value[0]
    return value


def _save_png(image: Image.Image, path: str | os.PathLike[str]) -> str:
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path, compress_level=1)
    return path


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def _save_view_mask(
    camera: Any,
    width: int,
    height: int,
    base_dir: str,
    filename: str,
) -> str | None:
    """Persist a nontrivial training mask at the supervision raster; else None."""
    mask = getattr(camera, "original_mask", None)
    if mask is None:
        return None
    while mask.ndim > 2:
        mask = mask[0]
    if mask.numel() == 0 or float((mask < 0.999).float().mean().item()) <= 1e-6:
        return None
    array = (mask.detach().float().cpu().numpy() > 0.5).astype(np.uint8) * 255
    image = Image.fromarray(array, mode="L")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.NEAREST)
    return _save_png(image, os.path.join(base_dir, filename))



def _file_identity(path: str | os.PathLike[str] | None) -> dict[str, Any] | None:
    """Content identity with stat-keyed hashing for immutable run artifacts."""
    if path is None:
        return None
    path = os.path.abspath(os.fspath(path))
    stat = os.stat(path)
    return {
        "path": path, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "sha256": _file_digest(path, stat.st_size, stat.st_mtime_ns),
    }


def _identities_match(left: Any, right: Any) -> bool:
    if left is None and right is None:
        return True
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    digest = left.get("sha256")
    return isinstance(digest, str) and len(digest) == 64 and digest == right.get("sha256")


@lru_cache(maxsize=128)
def _file_digest(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_sha256(path: str | os.PathLike[str]) -> str:
    path = os.path.abspath(os.fspath(path))
    stat = os.stat(path)
    return _file_digest(path, stat.st_size, stat.st_mtime_ns)


def load_local_flowedit_config(path: str | None) -> dict[str, Any] | None:
    """Load the optional LocalizedFlowEdit batch config (JSON)."""

    if path is None:
        return None
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Local FlowEdit config must be a JSON object: {path}")
    return payload


def _local_generation_identity(cfg: SceneZoomConfig) -> dict | None:
    payload = load_local_flowedit_config(cfg.local_flowedit_config)
    if payload is None:
        return None
    return {
        "config": _file_identity(cfg.local_flowedit_config),
        "view_masks": {
            str(name): _file_identity(path)
            for name, path in sorted(payload.get("view_masks", {}).items())
        },
        "module": _file_identity(os.path.join(os.path.dirname(__file__), "local_flowedit.py")),
    }


def _local_edit_dir(cfg: SceneZoomConfig) -> str:
    """Persistent per-tile local-FlowEdit evidence root (never under _scratch)."""

    return os.path.abspath(os.path.join(cfg.output_dir, LOCAL_EDIT_DIRNAME))


def _view_edit_mask_path(local_cfg: Mapping[str, Any], view_id: str) -> str | None:
    """Absolute base-raster edit-mask path configured for one view; else None."""

    masks = local_cfg.get("view_masks") or {}
    if not isinstance(masks, Mapping):
        raise ValueError("local_flowedit_config.view_masks must be an object.")
    path = masks.get(view_id)
    return None if path is None else os.path.abspath(str(path))


def _load_edit_mask_image(
    path: str, expected_size: tuple[int, int], context: str
) -> Image.Image:
    """Load one base-raster edit mask, refusing any raster drift."""

    with Image.open(path) as handle:
        mask = handle.convert("L")
    if mask.size != tuple(expected_size):
        raise ValueError(
            f"Edit mask {path} raster {mask.size} does not match base raster "
            f"{tuple(expected_size)} ({context})."
        )
    return mask


def _crop_edit_mask(
    view_mask: Image.Image,
    tile: TileSpec,
    raster: tuple[int, int],
) -> Image.Image | None:
    """Tile window of the view edit mask at ``raster``; None when empty."""

    crop = view_mask.crop(tile.crop_box)
    if crop.size != tuple(raster):
        crop = crop.resize(tuple(raster), Image.Resampling.NEAREST)
    if crop.getextrema()[1] == 0:
        return None
    return crop


def _invalidate_target_support(
    tile_dir: str, edit_mask: Image.Image, *, dilation_radius: int = 4,
) -> tuple[int, int]:
    """Invalidate edited target feature cells through the shared mask contract."""
    from refinement.local_flowedit import invalidate_target_side_geometry

    support = np.asarray(edit_mask, dtype=np.uint8) > 0
    invalidated = 0
    for name in sorted(os.listdir(tile_dir)):
        if not name.endswith(".npz"):
            continue
        path = os.path.join(tile_dir, name)
        with np.load(path) as loaded:
            payload = {key: loaded[key] for key in loaded.files}
        result = invalidate_target_side_geometry(
            payload["flow"], payload["valid_mask"], support, dilation_radius=dilation_radius
        )
        payload["flow"], payload["valid_mask"] = result["pixel_flow"], result["valid_mask"]
        invalidated += result["invalidated_pixels"]
        np.savez(path, **payload)
    return int(support.sum()), invalidated





def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)



def _progressive_work_raster(
    cfg: SceneZoomConfig, snapshot: CameraSnapshot
) -> tuple[int, int]:
    """SR-input raster of a progressive tile: ``base_raster / step_scale``.

    Per-level SRscale (step_scale, default 2) stays independent of the
    cumulative zoom: the tile zoom camera keeps the SAME cumulative-zoom FoV,
    just a lower raster, so DLoRAL's ``upscale=step_scale`` reproduces the full
    base raster. The real LR anchor is the original view crop at
    ``base_raster / cumulative_zoom`` (TileSpec.crop_size).
    """

    if not cfg.progressive:
        raise ValueError("_progressive_work_raster requires a progressive run.")
    step = float(cfg.step_scale)
    if abs(step - round(step)) > 1e-9:
        raise ValueError(f"step_scale must be an integer SR factor, got {step:g}")
    step_int = int(round(step))
    width = int(snapshot.image_width)
    height = int(snapshot.image_height)
    if width % step_int or height % step_int:
        raise ValueError(
            f"Base raster {width}x{height} must be divisible by step_scale {step_int}"
        )
    return width // step_int, height // step_int


def apply_local_flowedit(
    cfg: SceneZoomConfig, local_cfg: Mapping[str, Any],
    batch: Sequence[tuple[ViewTask, Mapping[float, Sequence[TileSpec]]]],
) -> dict[str, int]:
    """Edit a batch after releasing 3D tensors; keep every editing artifact."""
    from refinement.local_flowedit import refine_local_flowedit_jobs

    jobs, entries = [], []
    for view, levels in batch:
        mask_path = _view_edit_mask_path(local_cfg, view.view_id)
        if not mask_path:
            continue
        view_mask = _load_edit_mask_image(
            mask_path, (view.snapshot.image_width, view.snapshot.image_height), view.view_id
        )
        for tiles in levels.values():
            for tile in tiles:
                tile_dir = os.path.join(cfg.output_dir, SCRATCH_DIRNAME, view.view_id, tile.alias)
                input_path = os.path.join(tile_dir, "input.png")
                mask = _crop_edit_mask(view_mask, tile, _pil_size(input_path))
                if mask is None:
                    continue
                evidence = os.path.join(_local_edit_dir(cfg), tile.sample_id)
                os.makedirs(evidence, exist_ok=True)
                original = os.path.join(evidence, "pre_edit.png")
                shutil.copyfile(input_path, original)
                mask_png = _save_png(mask, os.path.join(evidence, "edit_mask.png"))
                output = os.path.join(evidence, "post_edit.png")
                metadata = os.path.join(evidence, "report.json")
                jobs.append({"image_path": original, "mask_path": mask_png,
                             "output_path": output, "metadata_path": metadata})
                entries.append((tile_dir, input_path, tile))
    if not jobs:
        return {"jobs": 0, "edited": 0}
    results = refine_local_flowedit_jobs(jobs, local_cfg)
    if len(results) != len(jobs):
        raise RuntimeError("Local FlowEdit batch result count differs from its jobs")
    for (tile_dir, input_path, tile), result in zip(entries, results):
        if result["status"] != "edited":
            raise RuntimeError(f"Nonempty edit mask was not processed: {tile.sample_id}")
        if _pil_size(result["output_path"]) != _pil_size(input_path):
            raise ValueError(f"Local FlowEdit changed the SR raster: {tile.sample_id}")
        shutil.copyfile(result["output_path"], input_path)
        record_path = os.path.join(tile_dir, "tile.json")
        record = _load_json(record_path)
        record["local_edit"] = {
            **(record.get("local_edit") or {}),
            "status": "edited", "metadata_path": result["metadata_path"],
            "pre_edit_path": result["image_path"], "post_edit_path": result["output_path"],
            "mask_path": result["mask_path"],
        }
        _write_json(record_path, record)
    return {"jobs": len(jobs), "edited": len(jobs)}


def _pil_size(path: str) -> tuple[int, int]:
    with Image.open(path) as handle:
        return handle.size


# ---------------------------------------------------------------------------
# Tile planning
# ---------------------------------------------------------------------------


def _sample_id(view_id: str, zoom_factor: float, row: int, col: int) -> str:
    return f"{view_id}__z{zoom_factor:g}__r{row}c{col}"


def _stable_uid(text: str) -> int:
    return 500_000_000 + int(zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF)


def build_view_tiles(
    view: ViewTask,
    zoom_factors: Sequence[float],
    *,
    level_offset: int = 0,
) -> dict[float, list[TileSpec]]:
    """Exact ``z x z`` tile grid covering the whole base frame."""
    width = int(view.snapshot.image_width)
    height = int(view.snapshot.image_height)
    per_zoom: dict[float, list[TileSpec]] = {}
    for offset, zoom_factor in enumerate(zoom_factors):
        level_index = level_offset + offset
        if zoom_factor <= 1.0:
            raise ValueError(f"zoom_factor must be > 1.0, got {zoom_factor}")
        if abs(zoom_factor - round(zoom_factor)) > 1e-9:
            raise ValueError(f"zoom_factor must be an integer grid size, got {zoom_factor}")
        zoom = int(round(zoom_factor))
        if width % zoom or height % zoom:
            raise ValueError(f"Base raster {width}x{height} must be divisible by zoom {zoom}")
        tiles: list[TileSpec] = []
        for row in range(zoom):
            for col in range(zoom):
                cx = (col + 0.5) / zoom
                cy = (row + 0.5) / zoom
                roi = NormalizedROI(cx, cy, 1.0 / zoom, 1.0 / zoom)
                roi.validate_for_zoom(float(zoom_factor))
                left = int(round(col * width / zoom))
                right = int(round((col + 1) * width / zoom))
                top = int(round(row * height / zoom))
                bottom = int(round((row + 1) * height / zoom))
                alias = f"{view.snapshot.image_name}__zoom{zoom_factor:g}__r{row}c{col}"
                tiles.append(
                    TileSpec(
                        zoom_factor=float(zoom_factor),
                        level_index=level_index,
                        row=row,
                        col=col,
                        sample_id=_sample_id(view.view_id, zoom_factor, row, col),
                        alias=alias,
                        uid=_stable_uid(alias),
                        roi=roi,
                        roi_dict={
                            "center_x": cx,
                            "center_y": cy,
                            "width": 1.0 / zoom,
                            "height": 1.0 / zoom,
                        },
                        crop_box=(left, top, right, bottom),
                        crop_size=(right - left, bottom - top),
                    )
                )
        per_zoom[float(zoom_factor)] = tiles
    return per_zoom


def _selected_target_views(
    tasks: Sequence[ViewTask], cfg: SceneZoomConfig
) -> list[ViewTask]:
    """Supervised views whose new SR targets this run plans to generate.

    ``target_view_ids`` only narrows the pending generation work: every view of
    the run still gets collected (base evidence, replay entries, manifest view
    entries) and the geometry neighbor pool stays the checkpoint's full camera
    set.  Unknown ids and replay-only ids are refused instead of being skipped.
    """

    eligible = [view for view in tasks if view.source_stage != REAL_REPLAY_STAGE]
    requested = [str(view_id) for view_id in cfg.target_view_ids]
    if not requested:
        return eligible
    by_id = {view.view_id: view for view in tasks}
    duplicated = sorted({view_id for view_id in requested if requested.count(view_id) > 1})
    if duplicated:
        raise ValueError(f"--target_view_ids repeats view id(s): {duplicated}")
    unknown = sorted(view_id for view_id in requested if view_id not in by_id)
    if unknown:
        raise ValueError(f"--target_view_ids names unknown view id(s): {unknown}")
    replay = sorted(
        view_id for view_id in requested if by_id[view_id].source_stage == REAL_REPLAY_STAGE
    )
    if replay:
        raise ValueError(
            f"--target_view_ids names replay-only view id(s) {replay}; real replay "
            "views never receive SR targets."
        )
    wanted = set(requested)
    return [view for view in eligible if view.view_id in wanted]


def build_tile_plans(
    tasks: Sequence[ViewTask],
    cfg: SceneZoomConfig,
    *,
    level_offset: int = 0,
) -> list[tuple[ViewTask, dict[float, list[TileSpec]]]]:
    """Tile plans for every supervised view; replay-only views never get tiles.

    Replay-only views (``source_stage == real_replay``) carry base evidence
    into the manifest and the trainer's replay pool, but they never receive
    geometry, prompts or SR targets and never count against ``--max_views``.

    ``target_view_ids`` (sharded generation) restricts the plans to the
    requested collected views; every other view keeps its collected evidence,
    manifest entry and membership in the global geometry neighbor pool.
    """

    return [
        (view, build_view_tiles(view, cfg.zoom_factors, level_offset=level_offset))
        for view in _selected_target_views(tasks, cfg)
    ]






# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------


def _camera_from_fields(
    *,
    colmap_id: Any,
    R: Any,
    T: Any,
    fov_x: float,
    fov_y: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    uid: int,
    name: str,
    data_device: str = "cuda",
):
    from scene.cameras import Camera

    image = torch.zeros((3, int(height), int(width)), dtype=torch.float32)
    mask = torch.ones((1, int(height), int(width)), dtype=torch.float32)
    return Camera(
        colmap_id=colmap_id,
        R=np.asarray(R, dtype=np.float64),
        T=np.asarray(T, dtype=np.float64).reshape(3),
        FoVx=float(fov_x),
        FoVy=float(fov_y),
        cx=float(cx),
        cy=float(cy),
        image=image,
        gt_alpha_mask=None,
        image_name=str(name),
        uid=int(uid),
        depth=None,
        mask=mask,
        data_device=str(data_device),
        optimizing=False,
    )


def _zoom_camera(
    base: Any,
    roi: NormalizedROI,
    zoom_factor: float,
    width: int,
    height: int,
    *,
    name: str,
    uid: int,
):
    """Zoom camera through the repository zoom math at an explicit raster."""
    snapshot = base if isinstance(base, CameraSnapshot) else CameraSnapshot.from_camera(base)
    fov_x = zoom_fov(snapshot.fov_x, zoom_factor)
    fov_y = zoom_fov(snapshot.fov_y, zoom_factor)
    cx, cy = zoom_principal_point(float(base.cx), float(base.cy), roi, zoom_factor)
    return _camera_from_fields(
        colmap_id=getattr(base, "colmap_id", None),
        R=base.R,
        T=base.T,
        fov_x=fov_x,
        fov_y=fov_y,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        uid=uid,
        name=name,
        data_device="cpu",
    )


def _zoom_snapshot(
    base_snapshot: CameraSnapshot,
    roi: NormalizedROI,
    zoom_factor: float,
    width: int,
    height: int,
    *,
    name: str,
    uid: int,
) -> CameraSnapshot:
    fov_x = zoom_fov(base_snapshot.fov_x, zoom_factor)
    fov_y = zoom_fov(base_snapshot.fov_y, zoom_factor)
    cx, cy = zoom_principal_point(base_snapshot.cx, base_snapshot.cy, roi, zoom_factor)
    return CameraSnapshot(
        image_name=name,
        uid=uid,
        colmap_id=base_snapshot.colmap_id,
        image_width=int(width),
        image_height=int(height),
        fov_x=fov_x,
        fov_y=fov_y,
        cx=cx,
        cy=cy,
        R=base_snapshot.R,
        T=base_snapshot.T,
        znear=base_snapshot.znear,
        zfar=base_snapshot.zfar,
    )


# ---------------------------------------------------------------------------
# Scene loading (checkpoint renders for geometry only)
# ---------------------------------------------------------------------------




def build_arg_namespace(cfg: SceneZoomConfig) -> argparse.Namespace:
    """Reuse the checkpoint dataset contract with complete pipeline defaults."""
    from arguments import ModelParams, PipelineParams

    parser = argparse.ArgumentParser(add_help=False)
    ModelParams(parser)
    PipelineParams(parser)
    namespace = parser.parse_args([])
    apply_stage1_cfg_to_args(namespace, load_stage1_cfg(os.path.dirname(cfg.start_checkpoint)))
    if cfg.source_path:
        namespace.source_path = cfg.source_path
    namespace.model_path = os.path.abspath(cfg.output_dir)
    namespace.resolution = 1
    namespace.eval = True
    namespace.data_device = "cpu"
    return namespace




def load_scene_context(cfg: SceneZoomConfig, namespace: argparse.Namespace) -> dict[str, Any]:
    """Load the checkpoint scene for render-only geometry (no training setup)."""
    from arguments import ModelParams, PipelineParams
    from scene import GaussianModel, Scene

    checkpoint = os.path.abspath(cfg.start_checkpoint)
    checkpoint_dir = os.path.dirname(checkpoint)
    model_params, first_iter = torch.load(checkpoint, weights_only=False)
    ply_path = os.path.join(
        checkpoint_dir, "point_cloud", f"iteration_{first_iter}", "point_cloud.ply"
    )
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(
            "Matching filter_3D PLY missing next to the checkpoint "
            f"({ply_path}); the Skyfall checkpoint directory must keep both."
        )

    extract_parser = argparse.ArgumentParser(add_help=False)
    dataset = ModelParams(extract_parser).extract(namespace)
    pipe = PipelineParams(extract_parser).extract(namespace)
    dataset.model_path = os.path.abspath(cfg.output_dir)

    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs,
        dataset.appearance_embedding_dim,
    )
    scene = Scene(
        dataset, gaussians, load_iteration=first_iter, shuffle=False, ply_path=checkpoint_dir
    )
    gaussians.load_from_checkpoints(model_params)
    del model_params
    if getattr(gaussians, "filter_3D", None) is not None:
        filter_source = "checkpoint_ply"
    else:
        # Native fallback: the checkpoint PLY carried no filter_3D, so rebuild
        # it over the STABLE physical base cameras (the native TRAIN set) and
        # report. Never invented silently; progressive/training paths must make
        # the same call over the same reference cameras.
        filter_source = "recomputed_native_train_cameras"
        print(
            "[scene-zoom] WARNING: checkpoint PLY carries no filter_3D; "
            "recomputing it over the native TRAIN cameras (native fallback)."
        )
        gaussians.compute_3D_filter(cameras=list(scene.getTrainCameras()))
    gaussians.freeze_params()
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device=gaussians.get_xyz.device,
    )
    return {
        "scene": scene,
        "gaussians": gaussians,
        "train_cameras": list(scene.getTrainCameras()),
        "pipe": pipe,
        "background": background,
        "kernel_size": float(dataset.kernel_size),
        "first_iter": first_iter,
        "filter_3d_source": filter_source,
    }


# ---------------------------------------------------------------------------
# Progressive parent: G(t-1) renders at the next-level low-res raster
# ---------------------------------------------------------------------------


def _parent_bundle_identity(cfg: SceneZoomConfig) -> dict[str, Any]:
    """Checkpoint AND appearance provenance of the parent model."""

    identity: dict[str, Any] = {
        "start_checkpoint": _file_identity(cfg.start_checkpoint),
    }
    if cfg.progressive and cfg.parent_lod_checkpoint:
        parent = os.path.abspath(cfg.parent_lod_checkpoint)
        identity["parent_lod_checkpoint"] = _file_identity(parent)
        identity["parent_appearance"] = _file_identity(parent + ".appearance.pt")
    return identity


def _require_parent_checkpoint(cfg: SceneZoomConfig) -> str | None:
    parent = os.path.abspath(cfg.parent_lod_checkpoint) if cfg.parent_lod_checkpoint else None
    if not parent and not math.isclose(float(cfg.zoom_factors[0]), float(cfg.step_scale)):
        raise ValueError("Only the first progressive level may use G0 without a parent checkpoint")
    if parent and not os.path.isfile(parent):
        raise FileNotFoundError(f"Parent LoD checkpoint missing: {parent}")
    return parent


def load_parent_render_context(
    cfg: SceneZoomConfig, namespace: argparse.Namespace
) -> dict[str, Any]:
    """Progressive context: frozen Skyfall G0 bundle + parent G(t-1) LoD tensors.

    The G0 source checkpoint keeps its dataset contract and appearance MLP
    (frozen); ``--parent_lod_checkpoint`` swaps the L0 point set for the
    trained G(t-1) layers and layer appearance embeddings. The bundle then
    renders the SAME model the previous training level converged to.

    ``filter_3D`` stays exactly the native source checkpoint's PLY filter
    (or the identical native fallback recorded by ``load_scene_context``):
    it is NEVER recomputed against this run's synthetic camera sets, so the
    training-side import and the parent equality check agree level over level.
    """

    from lod.importer import assert_l0_tensors_match, import_skyfall_l0, load_lod_onto_bundle
    from lod.rasterizer import require_rade_gs

    require_rade_gs()
    parent = _require_parent_checkpoint(cfg)
    ctx = load_scene_context(cfg, namespace)
    bundle = import_skyfall_l0(
        ctx["gaussians"],
        ctx["train_cameras"],
        gz_root=None,
        step_scale=float(cfg.step_scale),
        freeze=True,
    )
    if parent:
        device = str(ctx["gaussians"].get_xyz.device)
        load_lod_onto_bundle(bundle, parent, device=device)
        assert_l0_tensors_match(ctx["gaussians"], bundle)
        if not torch.equal(bundle.appearance.layer_embeddings[0], ctx["gaussians"]._embeddings):
            raise ValueError("Parent L0 appearance differs from the source checkpoint")
    expected_zoom = float(cfg.step_scale) ** len(bundle.lod.layers)
    if not math.isclose(float(cfg.zoom_factors[0]), expected_zoom):
        raise ValueError(f"Parent requires next cumulative zoom {expected_zoom:g}")
    if not math.isclose(float(bundle.lod.step_scale), float(cfg.step_scale)):
        raise ValueError("Parent and requested step_scale differ")
    ctx["parent_bundle"] = bundle
    ctx["parent_identity"] = _parent_bundle_identity(cfg)
    return ctx


def _parent_lod_render(
    ctx: Mapping[str, Any],
    camera: Any,
    *,
    background: Any,
    kernel_size: float,
    embedding: Any,
) -> dict[str, Any]:
    """Render the parent LoD (merged layers) with RaDe camera-Z semantics."""

    from lod.render import render_lod_appearance

    return render_lod_appearance(
        ctx["parent_bundle"],
        camera,
        background=background,
        kernel_size=kernel_size,
        appearance_embedding=embedding,
        lod=True,
        compact=True,
        require_depth=True,
    )




# ---------------------------------------------------------------------------
# View collection
# ---------------------------------------------------------------------------



def _base_raster(native_width: int, native_height: int, resolution: int) -> tuple[int, int]:
    """Aspect-preserving supervision raster; 0 keeps the input's own raster."""
    if resolution <= 0:
        return int(native_width), int(native_height)
    scale = float(resolution) / float(max(native_width, native_height))
    width = max(1, int(round(native_width * scale)))
    height = max(1, int(round(native_height * scale)))
    return width, height


def collect_stage1_views(
    ctx: Mapping[str, Any],
    cfg: SceneZoomConfig,
    *,
    source_stage: str = "stage1",
) -> list[ViewTask]:
    """Real TRAIN views; base RGB and optional mask saved at the supervision raster.

    ``source_stage="real_replay"`` marks replay-only entries for FlowEdit runs:
    identical cameras/images/ids as stage1 collection, but the trainer and the
    manifest treat them as base replay evidence, never as SR-target sources.
    """
    base_dir = os.path.abspath(os.path.join(cfg.output_dir, BASE_DIRNAME))
    os.makedirs(base_dir, exist_ok=True)
    tasks: list[ViewTask] = []
    for camera in ctx["train_cameras"]:
        native = CameraSnapshot.from_camera(camera)
        width, height = _base_raster(native.image_width, native.image_height, cfg.resolution)
        view_id = f"real_{native.image_name}"
        base_path = os.path.abspath(os.path.join(base_dir, f"{view_id}.png"))
        wide = _tensor_to_pil(camera.original_image)
        if (wide.width, wide.height) != (width, height):
            wide = wide.resize((width, height), Image.Resampling.LANCZOS)
        _save_png(wide, base_path)
        mask_path = _save_view_mask(camera, width, height, base_dir, f"{view_id}.mask.png")
        snapshot = CameraSnapshot(
            image_name=native.image_name,
            uid=native.uid,
            colmap_id=native.colmap_id,
            image_width=width,
            image_height=height,
            fov_x=native.fov_x,
            fov_y=native.fov_y,
            cx=native.cx,
            cy=native.cy,
            R=native.R,
            T=native.T,
            znear=native.znear,
            zfar=native.zfar,
        )
        tasks.append(
            ViewTask(
                view_id=view_id,
                snapshot=snapshot,
                base_image_path=base_path,
                source_stage=source_stage,
                appearance_uid=native.uid,
                mask_path=mask_path,
            )
        )
    return tasks


def collect_flowedit_views(
    manifest_paths: Sequence[str], cfg: SceneZoomConfig
) -> list[ViewTask]:
    """Accepts only ``kind == 'skyfall_flowedit_views'`` manifests."""
    base_dir = os.path.abspath(os.path.join(cfg.output_dir, BASE_DIRNAME))
    os.makedirs(base_dir, exist_ok=True)
    tasks: list[ViewTask] = []
    seen_ids: set[str] = set()
    for manifest_path in manifest_paths:
        payload = _load_json(manifest_path)
        if payload.get("kind") != FLOWEDIT_KIND:
            raise ValueError(
                f"Refusing non-FlowEdit manifest {manifest_path}: "
                f"kind={payload.get('kind')!r} (expected {FLOWEDIT_KIND!r})."
            )
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError(f"Unsupported flowedit schema_version in {manifest_path}.")
        episode_idx = int(payload.get("episode_idx", -1))
        source_checkpoint = str(payload.get("checkpoint", ""))
        for entry in payload.get("views", ()):
            view_id = str(entry["id"])
            if view_id in seen_ids:
                raise ValueError(f"Duplicate flowedit view id across manifests: {view_id}")
            seen_ids.add(view_id)
            native = _snapshot_from_dict(entry["camera"])
            image_path = os.path.abspath(str(entry["image_path"]))
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Repaired view image missing: {image_path}")
            with Image.open(image_path) as handle:
                native_size = handle.size
            if native_size != (native.image_width, native.image_height):
                raise ValueError(
                    f"Repaired image raster {native_size} does not match camera raster "
                    f"{(native.image_width, native.image_height)} for view {view_id}."
                )
            width, height = _base_raster(native.image_width, native.image_height, cfg.resolution)
            if (width, height) != (native.image_width, native.image_height):
                resized_path = os.path.abspath(os.path.join(base_dir, f"{view_id}.png"))
                with Image.open(image_path) as handle:
                    resized = handle.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
                _save_png(resized, resized_path)
                base_path = resized_path
            else:
                base_path = image_path
            snapshot = CameraSnapshot(
                image_name=native.image_name,
                uid=native.uid,
                colmap_id=native.colmap_id,
                image_width=width,
                image_height=height,
                fov_x=native.fov_x,
                fov_y=native.fov_y,
                cx=native.cx,
                cy=native.cy,
                R=native.R,
                T=native.T,
                znear=native.znear,
                zfar=native.zfar,
            )
            appearance_uid = entry.get("appearance_uid")
            tasks.append(
                ViewTask(
                    view_id=view_id,
                    snapshot=snapshot,
                    base_image_path=base_path,
                    source_stage="stage2",
                    appearance_uid=None if appearance_uid is None else int(appearance_uid),
                    episode_idx=episode_idx,
                    source_checkpoint=source_checkpoint,
                )
            )
    return tasks


# ---------------------------------------------------------------------------
# Geometry phase: checkpoint renders -> flows; RGB stays real
# ---------------------------------------------------------------------------


@torch.no_grad()
def _render_package(ctx: Mapping[str, Any], camera: Any, embedding: Any) -> dict[str, Any]:
    from gaussian_renderer import render

    return render(
        camera,
        ctx["gaussians"],
        ctx["pipe"],
        ctx["background"],
        kernel_size=ctx["kernel_size"],
        testing=False,
        appearance_embedding=embedding,
    )


def _target_embedding(ctx: Mapping[str, Any], view: ViewTask) -> Any:
    gaussians = ctx["gaussians"]
    if not getattr(gaussians, "appearance_enabled", False):
        return None
    if view.appearance_uid is not None:
        return gaussians.appearance_embeddings[int(view.appearance_uid)].detach()
    return gaussians.appearance_embeddings[min(6, len(gaussians.appearance_embeddings) - 1)].detach()


def _embedding_for_train_camera(ctx: Mapping[str, Any], uid: int) -> Any:
    gaussians = ctx["gaussians"]
    if not getattr(gaussians, "appearance_enabled", False):
        return None
    return gaussians.appearance_embeddings[int(uid)].detach()


def _neighbor_crop(
    neighbor_camera: Any,
    projected_center: tuple[float, float],
    zoom_factor: float,
    crop_size: tuple[int, int],
) -> Image.Image:
    """Real RGB of the neighbor's zoom window, resampled to the target crop raster."""
    width = int(neighbor_camera.image_width)
    height = int(neighbor_camera.image_height)
    half_w = width / (2.0 * zoom_factor)
    half_h = height / (2.0 * zoom_factor)
    left = int(round(projected_center[0] * width - half_w))
    right = int(round(projected_center[0] * width + half_w))
    top = int(round(projected_center[1] * height - half_h))
    bottom = int(round(projected_center[1] * height + half_h))
    left, right = max(0, min(left, width - 1)), max(left + 1, min(right, width))
    top, bottom = max(0, min(top, height - 1)), max(top + 1, min(bottom, height))
    neighbor = _tensor_to_pil(neighbor_camera.original_image)
    return neighbor.crop((left, top, right, bottom)).resize(crop_size, Image.Resampling.LANCZOS)


def _resolve_depth_scene_scale(cfg: SceneZoomConfig, camera: Any, spatial: Any) -> float | None:
    if cfg.depth_scene_scale < 0.0:
        return estimate_depth_scene_scale(camera, spatial.median_depth)
    if cfg.depth_scene_scale == 0.0:
        return None
    return float(cfg.depth_scene_scale)


def geometry_identity(
    backend: str, parent_identity: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Versioned depth contract of scene-zoom geometry for one renderer backend.

    Recorded in the tile geometry scratch, every supervision sample and the
    generation-cache request identity so stale or legacy buffers are refused
    instead of silently reinterpreted.  ``parent_identity`` (progressive runs)
    fingerprints the exact parent checkpoint AND appearance sidecar; identity
    comparisons are exact, so a parent change fails closed.
    """

    source = _SOURCE_DEPTH_BY_BACKEND.get(backend)
    if source is None:
        raise ValueError(
            f"Unsupported rasterizer backend {backend!r}; expected one of "
            f"{sorted(_SOURCE_DEPTH_BY_BACKEND)}"
        )
    identity = {
        "schema_version": GEOMETRY_IDENTITY_VERSION,
        "renderer_backend": backend,
        "source_depth": source,
        "target_depth": "camera_z",
        "neighbor_depth": "camera_z",
    }
    if parent_identity is not None:
        identity["parent"] = dict(parent_identity)
    return identity


def _geometry_identity_is_valid(identity: Any) -> bool:
    """True when ``identity`` is the current, self-consistent depth contract."""

    if not isinstance(identity, Mapping):
        return False
    backend = identity.get("renderer_backend")
    source = _SOURCE_DEPTH_BY_BACKEND.get(backend) if isinstance(backend, str) else None
    parent = identity.get("parent")
    if parent is not None:
        if not isinstance(parent, Mapping) or not parent:
            return False
        start = parent.get("start_checkpoint")
        if not isinstance(start, Mapping) or not _identities_match(start, start):
            return False
        has_lod = parent.get("parent_lod_checkpoint") is not None
        has_appearance = parent.get("parent_appearance") is not None
        if has_lod != has_appearance:
            return False
        for key in ("parent_lod_checkpoint", "parent_appearance"):
            value = parent.get(key)
            if value is not None and not _identities_match(value, value):
                return False
    return (
        identity.get("schema_version") == GEOMETRY_IDENTITY_VERSION
        and source is not None
        and source == identity.get("source_depth")
        and identity.get("target_depth") == "camera_z"
        and identity.get("neighbor_depth") == "camera_z"
    )


def _require_geometry_identity(sample: Mapping[str, Any], context: str) -> dict[str, Any]:
    """Refuse a supervision sample whose geometry contract is missing/invalid."""

    geometry = sample.get("geometry") or {}
    identity = geometry.get("geometry_identity") if isinstance(geometry, Mapping) else None
    if not _geometry_identity_is_valid(identity):
        raise ValueError(
            f"{context} carries no valid geometry identity (schema "
            f"{GEOMETRY_IDENTITY_VERSION}, camera-Z outputs); refusing legacy target."
        )
    return dict(identity)


def _effective_rasterizer_backend(ctx: Mapping[str, Any]) -> str:
    """Backend the geometry depth renders of this context dispatch to.

    Progressive parent-bundle contexts always render through RaDe-GS
    (``render_lod_appearance``); otherwise the shared ``gaussian_renderer``
    pipeline decides.
    """

    if ctx.get("parent_bundle") is not None:
        return "rade"
    from gaussian_renderer import _resolve_rasterizer_backend

    return _resolve_rasterizer_backend(ctx["pipe"])


def _checkpoint_rasterizer_backend(checkpoint: str) -> str:
    """Backend recorded by the Stage 1 run that owns ``checkpoint``.

    ``build_arg_namespace`` replays the same field into the render pipeline, so
    this is the backend every geometry render of the run will use.
    """

    stage1_cfg = load_stage1_cfg(os.path.dirname(os.path.abspath(checkpoint)))
    backend = getattr(stage1_cfg, "rasterizer_backend", None)
    return "diff_gauss" if backend is None else str(backend)


def _expected_camera_z(
    backend: str,
    render_depth: Any,
    render_alpha: Any,
    camera: Any,
) -> torch.Tensor:
    """Raw ``render_depth`` of ``backend`` → expected camera-space Z.

    ``diff_gauss`` accumulates ``sum(alpha * T * z)`` and keeps the historical
    accumulate-then-divide semantics; ``rade`` already stores the expected
    distance along the camera's principal ray, which is multiplied by the
    principal-point ``rln`` and must never be divided by alpha again. Empty or
    non-finite alpha stays zero depth for both backends.
    """

    depth = _plane(render_depth)
    alpha = _plane(render_alpha)
    if backend == "diff_gauss":
        camera_z = skyfall_expected_z(depth, alpha, eps=_DEPTH_VALID_EPS)
    elif backend == "rade":
        camera_z = depth_from_rade(depth, camera, alpha=alpha).camera_z()
    else:
        raise ValueError(
            f"Unsupported rasterizer backend {backend!r}; expected one of "
            f"{sorted(_SOURCE_DEPTH_BY_BACKEND)}"
        )
    if alpha is None:
        return camera_z
    valid = torch.isfinite(camera_z) & torch.isfinite(alpha) & (alpha > _DEPTH_VALID_EPS)
    return torch.where(valid, camera_z, torch.zeros_like(camera_z))


@torch.no_grad()
def _render_view_package(
    ctx: Mapping[str, Any], camera: Any, embedding: Any
) -> dict[str, Any]:
    """Render one camera with the model source this context carries.

    Progressive parent-bundle contexts render merged LoD layers through
    RaDe-GS; every other context uses the shared Skyfall render pipeline.
    """

    if ctx.get("parent_bundle") is not None:
        return _parent_lod_render(
            ctx,
            camera,
            background=ctx["background"],
            kernel_size=float(ctx["kernel_size"]),
            embedding=embedding,
        )
    return _render_package(ctx, camera, embedding)


@torch.no_grad()
def _render_neighbor_observations(
    ctx: Mapping[str, Any],
    view: ViewTask,
    tile: TileSpec,
    zoom_factor: float,
    selected: Sequence[tuple[Any, Any]],
    backend: str,
    raster: tuple[int, int],
) -> tuple[list[tuple[Any, Image.Image, Any, Any]], dict[int, dict[str, Any]]]:
    """Render neighbor depth/alpha at the tile raster; crop REAL neighbor RGB."""
    observations: list[tuple[Any, Image.Image, Any, Any]] = []
    projected_by_uid: dict[int, dict[str, Any]] = {}
    for neighbor_camera, projected in selected:
        neighbor_roi = NormalizedROI(
            float(projected.center_x),
            float(projected.center_y),
            float(projected.width),
            float(projected.height),
        )
        neighbor_name = (
            f"{neighbor_camera.image_name}__zoom{zoom_factor:g}"
            f"__to_{view.view_id}__r{tile.row}c{tile.col}"
        )
        neighbor_cam = _zoom_camera(
            neighbor_camera,
            neighbor_roi,
            zoom_factor,
            raster[0],
            raster[1],
            name=neighbor_name,
            uid=_stable_uid(neighbor_name),
        )
        projected_by_uid[int(neighbor_cam.uid)] = projected.to_dict()
        package = _render_view_package(
            ctx,
            neighbor_cam,
            _embedding_for_train_camera(ctx, int(neighbor_camera.uid)),
        )
        neighbor_depth = _expected_camera_z(
            backend, package["render_depth"], package["render_alpha"], neighbor_cam
        )
        neighbor_alpha = _plane(package["render_alpha"])
        neighbor_crop = _neighbor_crop(
            neighbor_camera,
            (float(projected.center_x), float(projected.center_y)),
            zoom_factor,
            raster,
        )
        observations.append((neighbor_cam, neighbor_crop, neighbor_depth, neighbor_alpha))
    return observations, projected_by_uid
def _save_tile_neighbors(
    tile_dir: str,
    tile: TileSpec,
    aligned: Sequence[MultiViewInput],
    projected_by_uid: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Persist per-neighbor scratch: real RGB PNG plus optional flow arrays."""
    neighbors: list[dict[str, Any]] = []
    for index, item in enumerate(aligned):
        png_path = _save_png(
            _tensor_to_pil(item.image), os.path.join(tile_dir, f"neighbor_{index}.png")
        )
        npz_path = None
        if item.pixel_flow is not None:
            crop_h, crop_w = item.pixel_flow.shape[:2]
            npz_path = os.path.abspath(os.path.join(tile_dir, f"neighbor_{index}.npz"))
            np.savez(
                npz_path,
                flow=item.pixel_flow.detach().cpu().numpy().astype(np.float32),
                valid_mask=item.valid_mask.detach().cpu().numpy() > 0.5,
                reverse_flow=(
                    item.source_to_target_flow.detach().cpu().numpy().astype(np.float32)
                    if item.source_to_target_flow is not None
                    else np.full((crop_h, crop_w, 2), np.nan, np.float32)
                ),
                reverse_valid_mask=(
                    item.reverse_valid_mask.detach().cpu().numpy() > 0.5
                    if item.reverse_valid_mask is not None
                    else np.zeros((crop_h, crop_w), dtype=bool)
                ),
            )
        zoom_uid = int(getattr(item.camera, "uid", -1))
        projected = projected_by_uid.get(zoom_uid)
        neighbors.append(
            {
                "index": index,
                "name": str(item.name),
                "uid": zoom_uid,
                "base_uid": None if projected is None else int(projected.get("uid", -1)),
                "weight": float(item.weight),
                "coverage": float(item.metadata.get("coverage", 0.0)),
                "reverse_coverage": float(item.metadata.get("reverse_coverage", 0.0)),
                "npz": npz_path,
                "png": png_path,
                "camera": CameraSnapshot.from_camera(item.camera).to_dict(),
                "projected": projected,
            }
        )
    return neighbors


@torch.no_grad()
def prepare_view_geometry(
    view: ViewTask,
    ctx: Mapping[str, Any],
    cfg: SceneZoomConfig,
    tiles_by_zoom: Mapping[float, Sequence[TileSpec]],
    *,
    view_edit_mask_path: str | None = None,
) -> None:
    """Per tile: render depth/alpha, rank spatial neighbors, save geometry scratch.

    Progressive runs render the tile through the parent bundle at the SR-input
    raster (``crop_raster * step_scale == base_raster``) and persist that RGB as
    the tile SR input plus the real base crop as the permanent LR anchor.
    """
    wide = Image.open(view.base_image_path).convert("RGB")
    scratch_root = os.path.abspath(os.path.join(cfg.output_dir, SCRATCH_DIRNAME, view.view_id))
    os.makedirs(scratch_root, exist_ok=True)
    target_embedding = _target_embedding(ctx, view)
    base_uid = int(view.snapshot.uid)
    train_cameras = list(ctx["train_cameras"])
    backend = _effective_rasterizer_backend(ctx)
    parent_identity = ctx.get("parent_identity") if ctx.get("parent_bundle") is not None else None
    identity = geometry_identity(backend, parent_identity=parent_identity)
    if cfg.progressive and view.source_stage == "stage2":
        # FlowEdit stage2: the repaired base photograph IS the wide context.
        # The parent render stays the tile SR input and geometry source, but it
        # never substitutes the repaired image for prompting; the LR anchor
        # crop below keeps every valid repaired pixel constrained (mask 255),
        # minus any explicit local-edit support when local FlowEdit is enabled.
        view.context_image_path = view.base_image_path
    elif cfg.progressive:
        wide_camera = _zoom_camera(
            view.snapshot, NormalizedROI(0.5, 0.5, 1.0, 1.0), 1.0,
            view.snapshot.image_width, view.snapshot.image_height,
            name=view.view_id, uid=view.snapshot.uid,
        )
        wide_package = _render_view_package(ctx, wide_camera, target_embedding)
        view.context_image_path = _save_png(
            _tensor_to_pil(wide_package["render"]),
            os.path.join(cfg.output_dir, "context", f"{view.view_id}.png"),
        )
        del wide_package
    edit_mask = (
        _load_edit_mask_image(
            view_edit_mask_path,
            (int(view.snapshot.image_width), int(view.snapshot.image_height)),
            view.view_id,
        )
        if view_edit_mask_path
        else None
    )

    for zoom_factor, tiles in tiles_by_zoom.items():
        for tile in tiles:
            tile_dir = os.path.abspath(os.path.join(scratch_root, tile.alias))
            os.makedirs(tile_dir, exist_ok=True)
            raster = _progressive_work_raster(cfg, view.snapshot) if cfg.progressive else tile.crop_size
            zoom_cam = _zoom_camera(
                view.snapshot,
                tile.roi,
                zoom_factor,
                raster[0],
                raster[1],
                name=tile.alias,
                uid=tile.uid,
            )
            needs_render = cfg.progressive or (
                cfg.geometry_neighbor_count > 0 and cfg.dloral_alignment != "target_only"
            )
            package = _render_view_package(ctx, zoom_cam, target_embedding) if needs_render else None
            if cfg.progressive:
                rendered = _tensor_to_pil(package["render"])
                if rendered.size != raster:
                    raise ValueError(
                        f"Progressive parent render {rendered.size} does not match "
                        f"SR-input raster {raster} for {tile.sample_id}."
                    )
                _save_png(rendered, os.path.join(tile_dir, "input.png"))
                lr_anchor = wide.crop(tile.crop_box)
                _save_png(
                    lr_anchor,
                    os.path.abspath(
                        os.path.join(
                            cfg.output_dir, TARGETS_DIRNAME, f"{tile.sample_id}.lr.png"
                        )
                    ),
                )
                lr_keep = np.full((tile.crop_size[1], tile.crop_size[0]), 255, dtype=np.uint8)
                if view.mask_path:
                    with Image.open(view.mask_path) as mask_image:
                        lr_keep = np.array(mask_image.convert("L").crop(tile.crop_box))
                if edit_mask is not None:
                    lr_keep[np.asarray(edit_mask.crop(tile.crop_box)) > 0] = 0
                _save_png(
                    Image.fromarray(lr_keep),
                    os.path.join(cfg.output_dir, TARGETS_DIRNAME, f"{tile.sample_id}.lr_mask.png"),
                )
            else:
                crop = wide.crop(tile.crop_box)
                _save_png(crop, os.path.join(tile_dir, "input.png"))
            if cfg.geometry_neighbor_count <= 0 or cfg.dloral_alignment == "target_only":
                _write_json(os.path.join(tile_dir, "tile.json"), {
                    "mode": "target_only", "status": "target_only_requested",
                    "fallback": "single_view_requested", "neighbors": [],
                    "cache_context": {
                        "geometry_identity": dict(identity),
                        "status": "target_only_requested",
                        **({"local_flowedit": ctx["local_edit_identity"]} if ctx.get("local_edit_identity") else {}),
                        **({"render_rgb_contract": PROGRESSIVE_RGB_CONTRACT} if cfg.progressive else {}),
                    },
                })
                continue

            depth = _expected_camera_z(
                backend, package["render_depth"], package["render_alpha"], zoom_cam
            )
            alpha_plane = _plane(package["render_alpha"])
            spatial = estimate_spatial_target(
                zoom_cam,
                depth,
                alpha_plane,
                alpha_threshold=cfg.alpha_threshold,
                min_depth=cfg.min_depth,
            )

            record: dict[str, Any] = {
                "sample_id": tile.sample_id,
                "alias": tile.alias,
                "zoom_factor": float(zoom_factor),
                "row": int(tile.row),
                "col": int(tile.col),
                "mode": cfg.dloral_alignment,
                "status": "ok",
                "fallback": None,
                "zoom_camera": CameraSnapshot.from_camera(zoom_cam).to_dict(),
                "spatial_target": spatial.to_dict(),
                "neighbors": [],
                "best_coverage": None,
                "candidates": [],
                "cache_context": {},
            }
            cache_context: dict[str, Any] = {
                "schema_version": 1,
                "selection": "spatial_target_projection_depth_reprojection",
                "geometry_identity": dict(identity),
                "target_uid": int(tile.uid),
                "base_uid": base_uid,
                "neighbor_count": int(cfg.geometry_neighbor_count),
                "depth_abs_tolerance": float(cfg.depth_abs_tolerance),
                "depth_rel_tolerance": float(cfg.depth_rel_tolerance),
                "alpha_threshold": float(cfg.alpha_threshold),
                "min_depth": float(cfg.min_depth),
                "crop_box": list(tile.crop_box),
                "crop_size": list(raster),
                "spatial_target": spatial.to_dict(),
                "status": "ok",
                "selected": [],
            }
            if cfg.progressive:
                cache_context["render_rgb_contract"] = PROGRESSIVE_RGB_CONTRACT

            failed_status: str | None = None
            if spatial.confidence < float(cfg.min_target_confidence):
                failed_status = "low_target_confidence"
            else:
                cache_context["depth_scene_scale"] = _float_or_none(
                    _resolve_depth_scene_scale(cfg, zoom_cam, spatial)
                )
                cache_context["pixel_footprint"] = pixel_footprint(zoom_cam, spatial.median_depth)
                selected, candidates = rank_cameras_by_spatial_overlap(
                    spatial,
                    train_cameras,
                    zoom_factor=zoom_factor,
                    k=cfg.geometry_neighbor_count,
                    pool_size=cfg.geometry_pool_size,
                    target_camera=zoom_cam,
                    exclude_uids=(base_uid,),
                    min_depth=cfg.min_depth,
                    min_in_frustum=cfg.min_in_frustum,
                )
                record["candidates"] = to_jsonable(candidates)
                if not selected:
                    failed_status = "no_spatial_neighbors"
                else:
                    observations, projected_by_uid = _render_neighbor_observations(
                        ctx, view, tile, zoom_factor, selected, backend, raster
                    )
                    scene_scale = _resolve_depth_scene_scale(cfg, zoom_cam, spatial)
                    aligned, candidates = build_aligned_multiview_inputs(
                        zoom_cam,
                        depth,
                        alpha_plane,
                        observations,
                        k=cfg.geometry_neighbor_count,
                        depth_abs_tolerance=cfg.depth_abs_tolerance,
                        depth_rel_tolerance=cfg.depth_rel_tolerance,
                        depth_scene_scale=scene_scale,
                        alpha_threshold=cfg.alpha_threshold,
                        min_depth=cfg.min_depth,
                    )
                    cache_context["selected"] = [
                        {
                            "uid": int(getattr(item.camera, "uid", -1)),
                            "name": str(item.name),
                            "coverage": float(item.metadata.get("coverage", 0.0)),
                        }
                        for item in aligned
                    ]
                    if not aligned:
                        failed_status = "no_aligned_neighbor"
                    else:
                        best_coverage = max(float(item.weight) for item in aligned)
                        record["best_coverage"] = float(best_coverage)
                        cache_context["best_coverage"] = float(best_coverage)
                        record["status"] = (
                            "ok"
                            if best_coverage >= float(cfg.min_reprojection_coverage)
                            else "low_reprojection_coverage"
                        )
                        record["neighbors"] = _save_tile_neighbors(
                            tile_dir, tile, aligned, projected_by_uid
                        )

            if failed_status is not None:
                record["status"] = failed_status
                record["fallback"] = "single_view"
            tile_edit_mask = (
                _crop_edit_mask(edit_mask, tile, raster) if edit_mask is not None else None
            )
            if tile_edit_mask is not None:
                mask_png = _save_png(
                    tile_edit_mask, os.path.join(tile_dir, "edit_mask.png")
                )
                support_pixels, invalidated = _invalidate_target_support(
                    tile_dir, tile_edit_mask,
                    dilation_radius=math.ceil(8 / (cfg.step_scale if cfg.progressive else zoom_factor)),
                )
                record["local_edit"] = {
                    "status": "geometry_invalidated",
                    "view_mask": os.path.abspath(view_edit_mask_path) if view_edit_mask_path else None,
                    "tile_mask": os.path.abspath(mask_png),
                    "edit_mask_pixels": int(support_pixels),
                    "dilated_invalidated_target_pixels": int(invalidated),
                    "dilation_radius": math.ceil(8 / (cfg.step_scale if cfg.progressive else zoom_factor)),
                }
                cache_context["local_edit"] = {
                    "invalidated_target_pixels": int(invalidated),
                    "edit_mask_pixels": int(support_pixels),
                }
            if ctx.get("local_edit_identity"):
                cache_context["local_flowedit"] = ctx["local_edit_identity"]
            cache_context["status"] = record["status"]
            record["cache_context"] = cache_context
            _write_json(os.path.join(tile_dir, "tile.json"), record)


# ---------------------------------------------------------------------------
# Prompt phase
# ---------------------------------------------------------------------------


def build_prompt_backend(cfg: SceneZoomConfig) -> tuple[Any, PromptManager]:
    provider = Qwen3PromptProvider(
        cfg.vlm_model_path,
        python=cfg.vlm_python,
        device=cfg.vlm_device,
        max_new_tokens=cfg.vlm_max_new_tokens,
        max_image_size=cfg.vlm_max_image_size,
    )
    cache_dir = os.path.abspath(
        cfg.prompt_cache_dir or os.path.join(cfg.output_dir, PROMPT_CACHE_DIRNAME)
    )
    return provider, PromptManager(provider, PromptCache(cache_dir))


def ensure_prompts(
    cfg: SceneZoomConfig,
    provider: Any,
    prompt_manager: PromptManager,
    view: ViewTask,
    pending_tiles: Sequence[TileSpec],
) -> None:
    """Load or generate the per-tile prompt JSON (permanent provenance)."""
    if not pending_tiles:
        return
    prompts_dir = os.path.abspath(os.path.join(cfg.output_dir, PROMPTS_DIRNAME))
    os.makedirs(prompts_dir, exist_ok=True)
    wide = Image.open(view.context_image_path or view.base_image_path).convert("RGB")
    scratch_root = os.path.abspath(os.path.join(cfg.output_dir, SCRATCH_DIRNAME, view.view_id))
    for tile in pending_tiles:
        prompt_path = os.path.join(prompts_dir, f"{tile.sample_id}.json")
        crop = Image.open(os.path.join(scratch_root, tile.alias, "input.png")).convert("RGB")
        zoom_snapshot = _zoom_snapshot(
            view.snapshot,
            tile.roi,
            tile.zoom_factor,
            crop.width,
            crop.height,
            name=tile.alias,
            uid=tile.uid,
        )
        prompt = prompt_manager.get_prompt(
            checkpoint=os.path.abspath(cfg.start_checkpoint),
            camera=view.snapshot,
            roi=tile.roi_dict,
            zoom_factor=tile.zoom_factor,
            level_index=tile.level_index,
            wide_image=wide,
            zoom_image=crop,
            context={
                "base_camera": view.snapshot.to_dict(),
                "view_id": view.view_id,
                "source_stage": view.source_stage,
                "zoom_camera": zoom_snapshot.to_dict(),
            },
        )
        _write_json(prompt_path, prompt.to_dict())


# ---------------------------------------------------------------------------
# SR phase: DLoRAL with per-level ``upscale=z``
# ---------------------------------------------------------------------------


def _dloral_model_config(cfg: SceneZoomConfig, zoom_factor: float, alignment: str) -> dict[str, Any]:
    """Per-tile DLoRAL config; progressive runs keep the adjacent SR step 2."""

    per_level_scale = float(cfg.step_scale) if cfg.progressive else float(zoom_factor)
    return {
        "backend": "dloral",
        "repo_root": os.path.abspath(cfg.dloral_root) if cfg.dloral_root else None,
        "sd_path": os.path.abspath(cfg.dloral_sd_path) if cfg.dloral_sd_path else None,
        "ckpt_path": os.path.abspath(cfg.dloral_ckpt) if cfg.dloral_ckpt else None,
        "spynet_path": os.path.abspath(cfg.dloral_spynet) if cfg.dloral_spynet else None,
        "python": os.path.abspath(cfg.dloral_python) if cfg.dloral_python else None,
        "device": cfg.dloral_device,
        "stages": int(cfg.dloral_stages),
        "process_size": int(cfg.dloral_process_size),
        "upscale": int(round(per_level_scale)),
        "align_method": cfg.dloral_align_method,
        "alignment": alignment,
        "max_roundtrip_error_px": (
            None
            if cfg.dloral_max_roundtrip_error_px is None
            else float(cfg.dloral_max_roundtrip_error_px)
        ),
        "latent_tiled_size": 96,
        "vae_encoder_tiled_size": 4096,
        "latent_tiled_overlap": 32,
        "sr_scale": per_level_scale,
        "frame_order": ["neighbor", "target"],
        "output_frame": "target",
    }


def build_dloral_backend(cfg: SceneZoomConfig, zoom_factor: float, alignment: str) -> DLoRALBackend:
    """One isolated backend per (zoom level, alignment); per-level upscale only."""
    if not (cfg.dloral_sd_path and cfg.dloral_ckpt and cfg.dloral_spynet):
        raise ValueError("--dloral_sd_path, --dloral_ckpt and --dloral_spynet are required.")
    per_level_scale = float(cfg.step_scale) if cfg.progressive else float(zoom_factor)
    return DLoRALBackend(
        repo_root=cfg.dloral_root or None,
        sd_path=cfg.dloral_sd_path,
        ckpt_path=cfg.dloral_ckpt,
        spynet_path=cfg.dloral_spynet,
        python=cfg.dloral_python,
        device=cfg.dloral_device,
        stages=cfg.dloral_stages,
        process_size=cfg.dloral_process_size,
        upscale=int(round(per_level_scale)),
        align_method=cfg.dloral_align_method,
        alignment=alignment,
        max_roundtrip_error_px=cfg.dloral_max_roundtrip_error_px,
    )


def _tile_mode(cfg: SceneZoomConfig, record: Mapping[str, Any]) -> str:
    """Geometry/spynet when a neighbor exists; explicit target_only otherwise."""
    if record.get("fallback") is not None:
        return "target_only"
    mode = str(record.get("mode", cfg.dloral_alignment))
    return mode if mode in ("geometry", "spynet") else "target_only"


def _load_geometry_neighbors(tile: TileSpec, record: Mapping[str, Any]) -> list[MultiViewInput]:
    neighbors: list[MultiViewInput] = []
    for meta in record.get("neighbors", ()):
        npz_payload = None
        if meta.get("npz"):
            with np.load(meta["npz"]) as archive:
                npz_payload = {name: archive[name] for name in archive.files}
        raster = [int(meta["camera"]["image_width"]), int(meta["camera"]["image_height"])]
        neighbors.append(
            MultiViewInput(
                name=str(meta["name"]),
                image=Image.open(meta["png"]).convert("RGB"),
                camera=_snapshot_from_dict(meta["camera"]),
                valid_mask=None if npz_payload is None else npz_payload["valid_mask"],
                pixel_flow=None if npz_payload is None else npz_payload["flow"],
                source_to_target_flow=(
                    None if npz_payload is None else npz_payload["reverse_flow"]
                ),
                reverse_valid_mask=(
                    None if npz_payload is None else npz_payload["reverse_valid_mask"]
                ),
                weight=float(meta["weight"]),
                metadata={
                    "coverage": float(meta.get("coverage", 0.0)),
                    "reverse_coverage": float(meta.get("reverse_coverage", 0.0)),
                    "source_size": raster,
                    "target_size": raster,
                    "reverse_source": "depth" if npz_payload is not None else "none",
                },
            )
        )
    return neighbors


def refine_tile(
    cfg: SceneZoomConfig,
    refiner: CachedRefiner,
    view: ViewTask,
    tile: TileSpec,
    record: Mapping[str, Any],
    prompt_payload: Mapping[str, Any],
    alignment: str,
) -> dict[str, Any]:
    """Run DLoRAL on one tile crop and return the supervision sample."""
    scratch_root = os.path.abspath(os.path.join(cfg.output_dir, SCRATCH_DIRNAME, view.view_id))
    tile_dir = os.path.join(scratch_root, tile.alias)
    input_path = os.path.join(tile_dir, "input.png")
    crop = Image.open(input_path).convert("RGB")
    per_level_scale = float(cfg.step_scale) if cfg.progressive else float(tile.zoom_factor)
    request_raster = crop.size
    zoom_snapshot = _zoom_snapshot(
        view.snapshot,
        tile.roi,
        tile.zoom_factor,
        request_raster[0],
        request_raster[1],
        name=tile.alias,
        uid=tile.uid,
    )
    cache_context = record.get("cache_context") or {}
    identity = cache_context.get("geometry_identity")
    if not _geometry_identity_is_valid(identity):
        raise ValueError(
            f"Geometry scratch for {tile.sample_id} carries no valid geometry identity "
            f"(schema {GEOMETRY_IDENTITY_VERSION}, camera-Z outputs)."
        )
    neighbors = _load_geometry_neighbors(tile, record) if alignment != "target_only" else []
    seed = int(cfg.seed) + int(zlib.crc32(tile.sample_id.encode("utf-8")) % (2**31 - 1))
    request = RefinementRequest(
        image=crop,
        checkpoint=os.path.abspath(cfg.start_checkpoint),
        camera=zoom_snapshot,
        zoom_factor=tile.zoom_factor,
        sr_scale=per_level_scale,
        prompt=PromptDescription.from_dict(prompt_payload),
        model_config=_dloral_model_config(cfg, tile.zoom_factor, alignment),
        prompt_config={
            "provider": "qwen3_vl",
            "vlm_model_path": os.path.abspath(cfg.vlm_model_path) if cfg.vlm_model_path else None,
            "vlm_python": os.path.abspath(cfg.vlm_python) if cfg.vlm_python else None,
            "vlm_device": cfg.vlm_device if cfg.vlm_model_path else None,
            "vlm_max_new_tokens": int(cfg.vlm_max_new_tokens),
            "vlm_max_image_size": int(cfg.vlm_max_image_size),
            "prompt_cache_dir": os.path.abspath(
                cfg.prompt_cache_dir or os.path.join(cfg.output_dir, PROMPT_CACHE_DIRNAME)
            ),
        },
        cache_context=to_jsonable(dict(record.get("cache_context", {}))),
        level_index=int(tile.level_index),
        metadata={"seed": seed, "neighbor_views": neighbors},
    )
    result = refiner.refine(request)
    image = result.image
    base_raster = (int(view.snapshot.image_width), int(view.snapshot.image_height))
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if tuple(metadata.get("raw_output_size", ())) != base_raster or image.size != base_raster:
        raise ValueError(
            f"DLoRAL must synthesize at {base_raster}, got native {metadata.get('raw_output_size')} "
            f"and returned {image.size}; use a raster supported by its VAE grid"
        )
    cache_path = metadata.get("cache_path")
    fallback_target = os.path.abspath(
        os.path.join(cfg.output_dir, TARGETS_DIRNAME, f"{tile.sample_id}.png")
    )
    if cache_path and os.path.isfile(cache_path):
        image_path = os.path.abspath(cache_path)
    else:
        image_path = _save_png(image, fallback_target)
    sample_mask_path = None
    if view.mask_path:
        with Image.open(view.mask_path) as handle:
            tile_mask = handle.convert("L").crop(tile.crop_box)
        if tile_mask.size != base_raster:
            tile_mask = tile_mask.resize(base_raster, Image.Resampling.NEAREST)
        sample_mask_path = _save_png(
            tile_mask, os.path.join(cfg.output_dir, TARGETS_DIRNAME, f"{tile.sample_id}.mask.png")
        )
    neighbor_meta = record.get("neighbors") or [None]
    first_neighbor = neighbor_meta[0] if neighbor_meta else None
    sample: dict[str, Any] = {
        "view_id": view.view_id,
        "roi": dict(tile.roi_dict),
        "image_path": image_path,
        "sample_id": tile.sample_id,
        "mask_path": os.path.abspath(sample_mask_path) if sample_mask_path else None,
        "zoom_factor": float(tile.zoom_factor),
        "tile_row": int(tile.row),
        "tile_col": int(tile.col),
        "crop_box": list(tile.crop_box),
        "crop_size": list(tile.crop_size),
        "base_raster": [base_raster[0], base_raster[1]],
        "sr_input_size": [request_raster[0], request_raster[1]],
        "sr_scale": float(per_level_scale),
        "source_stage": view.source_stage,
        "appearance_uid": view.appearance_uid,
        "zoom_camera": _zoom_snapshot(
            view.snapshot, tile.roi, tile.zoom_factor, *base_raster,
            name=tile.alias, uid=tile.uid,
        ).to_dict(),
        "prompt_target": str(prompt_payload.get("target_prompt", "")),
        "geometry": {
            "status": record.get("status"),
            "fallback": record.get("fallback"),
            "mode": alignment,
            "best_coverage": record.get("best_coverage"),
            "neighbor": (
                {
                    "name": str(first_neighbor["name"]),
                    "uid": int(first_neighbor["uid"]),
                    "coverage": float(first_neighbor["coverage"]),
                }
                if first_neighbor is not None
                else None
            ),
            "geometry_identity": dict(identity),
        },
        "backend": {
            "backend": metadata.get("backend", "dloral"),
            "alignment": metadata.get("alignment", alignment),
            "propagation": metadata.get("propagation"),
            "prompt_used": metadata.get("prompt_used"),
            "prepared_size": metadata.get("prepared_size"),
            "latent_size": metadata.get("latent_size"),
            "tiled": metadata.get("tiled"),
            "raw_output_size": metadata.get("raw_output_size"),
            "requested_size": metadata.get("requested_size"),
            "elapsed_sec": metadata.get("elapsed_sec"),
            "seed": metadata.get("seed", seed),
            "seed_source": metadata.get("seed_source"),
            "neighbor_name": metadata.get("neighbor_name"),
            "geometry_coverage": metadata.get("geometry_coverage"),
            "cache_key": result.cache_key,
            "cache_hit": bool(getattr(result, "cache_hit", False)),
        },
    }
    sample["local_edit"] = record.get("local_edit")
    if cfg.progressive:
        lr_image_path = os.path.abspath(
            os.path.join(cfg.output_dir, TARGETS_DIRNAME, f"{tile.sample_id}.lr.png")
        )
        if not os.path.isfile(lr_image_path):
            raise FileNotFoundError(
                f"Progressive LR anchor missing for {tile.sample_id}: {lr_image_path}"
            )
        sample["lr_image_path"] = lr_image_path
        sample["lr_mask_path"] = os.path.abspath(
            os.path.join(cfg.output_dir, TARGETS_DIRNAME, f"{tile.sample_id}.lr_mask.png")
        )
        if record.get("local_edit", {}).get("status") == "edited":
            with Image.open(record["local_edit"]["mask_path"]) as mask_image:
                edit_hr = mask_image.resize(base_raster, Image.Resampling.NEAREST)
            sample["edit_mask_path"] = _save_png(
                edit_hr, os.path.join(cfg.output_dir, TARGETS_DIRNAME, f"{tile.sample_id}.edit_mask.png")
            )
        sample["progressive"] = {
            "render_rgb_contract": PROGRESSIVE_RGB_CONTRACT,
            "parent_lod_checkpoint": (
                os.path.abspath(cfg.parent_lod_checkpoint)
                if cfg.parent_lod_checkpoint
                else None
            ),
            "parent_identity": _parent_bundle_identity(cfg),
            "step_scale": float(cfg.step_scale),
            "cumulative_zoom": float(tile.zoom_factor),
            "sr_input_size": [request_raster[0], request_raster[1]],
            "sr_input_sha256": _file_sha256(input_path),
            "lr_size": [int(tile.crop_size[0]), int(tile.crop_size[1])],
            "wide_image_path": view.context_image_path,
            "real_base_identity": _file_identity(view.base_image_path),
            "lr_image_identity": _file_identity(lr_image_path),
            "lr_mask_identity": _file_identity(sample["lr_mask_path"]),
        }
    shutil.rmtree(tile_dir)
    if not os.listdir(scratch_root):
        os.rmdir(scratch_root)
    return sample


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------


class _ManifestState:
    """Incrementally assembled ``skyfall_scene_zoom`` supervision manifest."""

    def __init__(self, cfg: SceneZoomConfig):
        self.path = os.path.abspath(os.path.join(cfg.output_dir, "supervision.json"))
        self.base_checkpoint = os.path.abspath(cfg.start_checkpoint)
        self.views: dict[str, dict[str, Any]] = {}
        self.view_order: list[str] = []
        self.samples: dict[float, dict[str, dict[str, Any]]] = {}
        self.cfg = cfg
        self.carried: set[tuple[float, str]] = set()
        self.carried_from: dict[str, Any] | None = None

    def add_view(self, entry: Mapping[str, Any]) -> None:
        view_id = str(entry["id"])
        if view_id not in self.views:
            self.view_order.append(view_id)
        self.views[view_id] = dict(entry)

    def add_sample(
        self,
        zoom_factor: float,
        sample: Mapping[str, Any],
        *,
        write: bool = True,
    ) -> None:
        zoom = float(zoom_factor)
        bucket = self.samples.setdefault(zoom, {})
        bucket[str(sample["sample_id"])] = dict(sample)
        if write:
            self.write()

    def sample_exists(self, zoom_factor: float, sample_id: str) -> bool:
        return sample_id in self.samples.get(float(zoom_factor), {})

    def write(self) -> None:
        levels = []
        for zoom in sorted(self.samples):
            bucket = self.samples[zoom]
            order = {view_id: index for index, view_id in enumerate(self.view_order)}
            samples = sorted(
                bucket.values(),
                key=lambda item: (
                    order.get(str(item["view_id"]), -1),
                    int(item["tile_row"]),
                    int(item["tile_col"]),
                ),
            )
            levels.append({"zoom_factor": zoom, "samples": samples})
        payload = {
            "schema_version": 1,
            "kind": SCENE_ZOOM_KIND,
            "base_checkpoint": self.base_checkpoint,
            "views": [self.views[view_id] for view_id in self.view_order],
            "levels": levels,
        }
        if self.cfg.progressive:
            payload["progressive"] = _progressive_manifest_payload(
                self.cfg, self.carried_from
            )
        _write_json(self.path, payload)



def _progressive_manifest_payload(
    cfg: SceneZoomConfig, carried_from: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Manifest-level progressive provenance (parent checkpoint AND appearance)."""

    parent = os.path.abspath(cfg.parent_lod_checkpoint) if cfg.parent_lod_checkpoint else None
    payload: dict[str, Any] = {
        "enabled": True,
        "render_rgb_contract": PROGRESSIVE_RGB_CONTRACT,
        "step_scale": float(cfg.step_scale),
        "start_checkpoint": _file_identity(cfg.start_checkpoint),
        "parent_lod_checkpoint": _file_identity(parent) if parent else None,
        "parent_appearance": _file_identity(parent + ".appearance.pt") if parent else None,
        "local_flowedit_config": (
            os.path.abspath(cfg.local_flowedit_config) if cfg.local_flowedit_config else None
        ),
        "local_flowedit_identity": _local_generation_identity(cfg),
        "carried_from": dict(carried_from) if carried_from else None,
    }
    return payload


def ingest_previous_supervision(
    state: "_ManifestState",
    payload: Mapping[str, Any],
    *,
    manifest_path: str,
    collected_entries: Mapping[str, Mapping[str, Any]],
) -> list[float]:
    """Carry prior progressive levels verbatim, without regenerating.

    Unlike ``--real_supervision`` this works for Stage 1 progressive chains:
    each sample keeps its own geometry identity and parent provenance (it was
    generated under the PRIOR parent), and a carried zoom that this run plans
    to regenerate is refused so stale parent targets cannot survive.
    """

    if payload.get("kind") != SCENE_ZOOM_KIND or int(payload.get("schema_version", 0)) != 1:
        raise ValueError(
            f"--previous_supervision is not a {SCENE_ZOOM_KIND} manifest: "
            f"{payload.get('kind')!r}."
        )
    if os.path.realpath(payload["base_checkpoint"]) != os.path.realpath(state.base_checkpoint):
        raise ValueError("Previous supervision belongs to a different base checkpoint")
    carried_zooms: list[float] = []
    for entry in payload.get("views", ()):
        view_id = str(entry["id"])
        current = collected_entries.get(view_id)
        if current is None or any(
            current.get(key) != entry.get(key) for key in ("camera", "appearance_uid", "source_stage")
        ):
            raise ValueError(f"Previous supervision changed camera identity: {view_id}")
        for key in ("image_path", "mask_path"):
            if not _identities_match(_file_identity(current.get(key)), _file_identity(entry.get(key))):
                raise ValueError(f"Previous supervision changed real observation: {view_id}/{key}")
        state.add_view(current)
    for level in payload.get("levels", ()):
        zoom = float(level["zoom_factor"])
        carried_zooms.append(zoom)
        for sample in level.get("samples", ()):
            _require_geometry_identity(
                sample, f"--previous_supervision sample {sample.get('sample_id')}"
            )
            if sample.get("progressive") and sample["progressive"].get("render_rgb_contract") != PROGRESSIVE_RGB_CONTRACT:
                raise ValueError("Previous supervision uses an obsolete render RGB contract")
            state.add_sample(zoom, dict(sample), write=False)
            state.carried.add((zoom, str(sample["sample_id"])))
    carried_zooms = sorted(set(carried_zooms))
    planned = {float(zoom) for zoom in state.cfg.zoom_factors}
    collisions = sorted(zoom for zoom in carried_zooms if zoom in planned)
    if collisions:
        raise ValueError(
            "--previous_supervision already contains zoom level(s) "
            f"{[f'{zoom:g}' for zoom in collisions]} planned for this run; "
            "refusing stale parent targets."
        )
    state.carried_from = {
        "manifest": os.path.abspath(manifest_path),
        "base_checkpoint": payload.get("base_checkpoint"),
        "levels": carried_zooms,
        "samples": sum(len(level.get("samples", ())) for level in payload.get("levels", ())),
    }
    return carried_zooms


def ingest_real_supervision(state: "_ManifestState", payload: Mapping[str, Any]) -> None:
    """Carry a Stage 1 ``skyfall_scene_zoom`` manifest over verbatim."""
    if payload.get("kind") != SCENE_ZOOM_KIND or int(payload.get("schema_version", 0)) != 1:
        raise ValueError(
            f"--real_supervision is not a {SCENE_ZOOM_KIND} manifest: {payload.get('kind')!r}."
        )
    expected_identity = None
    base_checkpoint = payload.get("base_checkpoint")
    if base_checkpoint:
        expected_identity = geometry_identity(_checkpoint_rasterizer_backend(str(base_checkpoint)))
    for entry in payload.get("views", ()):
        state.add_view(dict(entry))
    for level in payload.get("levels", ()):
        zoom = float(level["zoom_factor"])
        for sample in level.get("samples", ()):
            identity = _require_geometry_identity(
                sample, f"--real_supervision sample {sample.get('sample_id')}"
            )
            if expected_identity is not None and identity != expected_identity:
                raise ValueError(
                    f"--real_supervision sample {sample.get('sample_id')} was generated under "
                    f"geometry identity {identity}, not {expected_identity}; refusing legacy target."
                )
            state.add_sample(zoom, dict(sample), write=False)


def _view_entry(view: ViewTask) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": view.view_id,
        "camera": view.snapshot.to_dict(),
        "image_path": os.path.abspath(view.base_image_path),
        "source_stage": view.source_stage,
        "appearance_uid": view.appearance_uid,
    }
    if view.mask_path:
        entry["mask_path"] = os.path.abspath(view.mask_path)
    return entry


def restore_completed_supervision(
    state: _ManifestState, previous: Mapping[str, Any], cfg: SceneZoomConfig,
    plans: Sequence[tuple[ViewTask, Mapping[float, Sequence[TileSpec]]]],
) -> int:
    """Reuse completed targets only for the same checkpoint, cameras and generation settings."""
    if (previous.get("kind") != SCENE_ZOOM_KIND or previous.get("schema_version") != 1
            or previous.get("base_checkpoint") != os.path.abspath(cfg.start_checkpoint)):
        raise ValueError("Cannot resume a different supervision run")
    stat = os.stat(cfg.start_checkpoint)
    checkpoint_stat = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    old_views = {view["id"]: view for view in previous["views"]}
    planned_zooms = {float(zoom) for zoom in cfg.zoom_factors}
    expected_identity = geometry_identity(
        "rade" if cfg.progressive else _checkpoint_rasterizer_backend(cfg.start_checkpoint),
        parent_identity=_parent_bundle_identity(cfg) if cfg.progressive else None,
    )
    local_identity = _local_generation_identity(cfg)
    planned = {view.view_id: (view, {tile.sample_id: tile for tiles in levels.values() for tile in tiles})
               for view, levels in plans}
    prompt_settings = {
        "vlm_model_path": os.path.abspath(cfg.vlm_model_path) if cfg.vlm_model_path else None,
        "vlm_python": os.path.abspath(cfg.vlm_python) if cfg.vlm_python else None,
        "vlm_device": cfg.vlm_device,
        "vlm_max_new_tokens": cfg.vlm_max_new_tokens,
        "vlm_max_image_size": cfg.vlm_max_image_size,
    }
    restored = 0
    for level in previous["levels"]:
        for sample in level["samples"]:
            sample_id = str(sample["sample_id"])
            sample_zoom = float(level["zoom_factor"])
            identity = _require_geometry_identity(sample, f"Resume sample {sample_id}")
            if sample_zoom not in planned_zooms or (sample_zoom, sample_id) in state.carried:
                if (sample_zoom, sample_id) not in state.carried:
                    raise ValueError("Resume contains a level not explicitly carried by previous_supervision")
                continue
            view_id = sample["view_id"]
            if view_id not in planned:
                # Carried real targets are validated by ingest_real_supervision
                # against their own manifest checkpoint, not this run's backend.
                continue
            if identity != expected_identity:
                raise ValueError(f"Resume geometry identity changed: {sample['sample_id']}")
            if cfg.progressive:
                progressive = sample.get("progressive")
                if not isinstance(progressive, Mapping):
                    raise ValueError(
                        f"Resume sample {sample['sample_id']} lacks progressive provenance."
                    )
                if progressive.get("render_rgb_contract") != PROGRESSIVE_RGB_CONTRACT:
                    raise ValueError(f"Resume render RGB contract changed: {sample_id}")
                recorded = progressive.get("parent_identity") or {}
                current = _parent_bundle_identity(cfg)
                for key, value in current.items():
                    if not _identities_match(recorded.get(key), value):
                        raise ValueError(
                            f"Resume parent identity changed for {sample['sample_id']}: {key}"
                        )
            view, tiles = planned[view_id]
            tile = tiles.get(sample["sample_id"])
            if old_views.get(view_id) != _view_entry(view) or tile is None:
                raise ValueError(f"Resume camera or zoom grid changed: {sample['sample_id']}")
            if sample["crop_box"] != list(tile.crop_box) or sample["roi"] != tile.roi_dict:
                raise ValueError(f"Resume crop changed: {sample['sample_id']}")
            image_path = sample["image_path"]
            cache_path = os.path.splitext(image_path)[0] + ".json"
            auxiliary = [sample.get(key) for key in ("lr_image_path", "lr_mask_path", "edit_mask_path")]
            if (not os.path.isfile(image_path) or not os.path.isfile(cache_path)
                    or (sample.get("mask_path") and not os.path.isfile(sample["mask_path"]))
                    or any(path and not os.path.isfile(path) for path in auxiliary)):
                continue
            cached = _load_json(cache_path)
            request = cached["request"]
            if (request.get("cache_context") or {}).get("geometry_identity") != expected_identity:
                raise ValueError(f"Resume request geometry identity changed: {sample['sample_id']}")
            if (request.get("cache_context") or {}).get("local_flowedit") != local_identity:
                raise ValueError(f"Resume local edit identity changed: {sample_id}")
            if cfg.progressive:
                if (request.get("cache_context") or {}).get("render_rgb_contract") != PROGRESSIVE_RGB_CONTRACT:
                    raise ValueError(f"Resume cached RGB contract changed: {sample_id}")
                saved = sample["progressive"]
                for key, path in (
                    ("real_base_identity", view.base_image_path),
                    ("lr_image_identity", sample["lr_image_path"]),
                    ("lr_mask_identity", sample["lr_mask_path"]),
                ):
                    if not _identities_match(saved.get(key), _file_identity(path)):
                        raise ValueError(f"Resume real anchor identity changed: {sample_id}/{key}")
            alignment = sample["geometry"]["mode"]
            request_raster = _progressive_work_raster(cfg, view.snapshot) if cfg.progressive else tile.crop_size
            expected_camera = _zoom_snapshot(
                view.snapshot, tile.roi, tile.zoom_factor, *request_raster,
                name=tile.alias, uid=tile.uid,
            ).to_dict()
            if (request["checkpoint_stat"] != checkpoint_stat
                    or request["checkpoint"] != os.path.abspath(cfg.start_checkpoint)
                    or request["camera"] != expected_camera
                    or request["model_config"] != _dloral_model_config(cfg, tile.zoom_factor, alignment)
                    or any(request["prompt_config"].get(key) != value for key, value in prompt_settings.items())
                    or sample["backend"].get("seed") != int(cfg.seed) + int(zlib.crc32(tile.sample_id.encode()) % (2**31 - 1))):
                raise ValueError(f"Resume generation identity changed: {sample['sample_id']}")
            raster = (view.snapshot.image_width, view.snapshot.image_height)
            with Image.open(image_path) as image:
                if image.size != raster:
                    raise ValueError(f"Resume image raster changed: {image_path}")
            state.add_sample(tile.zoom_factor, dict(sample), write=False)
            restored += 1
    return restored


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_scene_zoom(cfg: SceneZoomConfig) -> dict[str, Any]:
    """Full pipeline: view collection, geometry, prompts, SR, manifest."""
    started = time.perf_counter()
    output_dir = os.path.abspath(cfg.output_dir)
    if cfg.previous_supervision and cfg.real_supervision is not None:
        raise ValueError(
            "--previous_supervision and --real_supervision are mutually exclusive."
        )
    if cfg.real_supervision is not None and not cfg.flowedit_manifests:
        raise ValueError("--real_supervision is only valid together with --flowedit_manifests.")
    if cfg.include_real_replay:
        if not cfg.flowedit_manifests:
            raise ValueError(
                "--include_real_replay is only valid together with --flowedit_manifests."
            )
        if cfg.real_supervision is not None:
            raise ValueError(
                "--include_real_replay supersedes --real_supervision: real base views are "
                "collected fresh as replay-only entries and must not also be carried from a "
                "Stage 1 manifest (identical real_* view ids)."
            )
    if not cfg.flowedit_manifests and cfg.real_supervision is None and cfg.resolution < 0:
        raise ValueError("resolution must be >= 0.")
    if cfg.parent_lod_checkpoint and not cfg.progressive:
        raise ValueError("--parent_lod_checkpoint requires --progressive.")
    if cfg.progressive:
        if len(cfg.zoom_factors) != 1:
            raise ValueError(
                "--progressive prepares exactly ONE new cumulative zoom level per "
                f"invocation, got {len(cfg.zoom_factors)} factor(s)."
            )
        if not math.isfinite(float(cfg.step_scale)) or float(cfg.step_scale) <= 1.0:
            raise ValueError("Progressive step_scale must exceed 1.0.")
        _require_parent_checkpoint(cfg)
    local_cfg = load_local_flowedit_config(cfg.local_flowedit_config)
    local_identity = _local_generation_identity(cfg)
    os.makedirs(output_dir, exist_ok=True)
    namespace = build_arg_namespace(cfg)
    state = _ManifestState(cfg)
    previous = _load_json(state.path) if cfg.resume and os.path.isfile(state.path) else None

    if cfg.flowedit_manifests:
        stage = "stage2"
        if cfg.real_supervision:
            ingest_real_supervision(state, _load_json(cfg.real_supervision))
            print(
                f"[scene-zoom] stage2: carrying {len(state.views)} carried real view(s) "
                f"from {os.path.abspath(cfg.real_supervision)}."
            )
        elif not cfg.include_real_replay:
            print("[scene-zoom] WARNING: Stage2 without --real_supervision; manifest will contain only FlowEdit views.")
    else:
        if cfg.real_supervision is not None:
            raise ValueError("--real_supervision is only valid together with --flowedit_manifests.")
        stage = "stage1"

    if stage == "stage1":
        ctx = load_scene_context(cfg, namespace)
        try:
            tasks = collect_stage1_views(ctx, cfg)
        finally:
            del ctx
            _free_gpu()
        print(f"[scene-zoom] stage1: {len(tasks)} real train view(s).")
    else:
        tasks = collect_flowedit_views(cfg.flowedit_manifests, cfg)
        print(f"[scene-zoom] stage2: {len(tasks)} repaired view(s).")
        if cfg.include_real_replay:
            ctx = load_scene_context(cfg, namespace)
            try:
                replay_tasks = collect_stage1_views(ctx, cfg, source_stage=REAL_REPLAY_STAGE)
            finally:
                del ctx
                _free_gpu()
            print(
                f"[scene-zoom] stage2: {len(replay_tasks)} real base replay view(s) "
                "collected (replay-only; never SR targets, never pre-repair supervision)."
            )
            tasks.extend(replay_tasks)

    for view in tasks:
        state.add_view(_view_entry(view))
    collected_entries = {str(view["id"]): view for view in (_view_entry(v) for v in tasks)}
    carried_zooms: list[float] = []
    if cfg.previous_supervision:
        carried_zooms = ingest_previous_supervision(
            state,
            _load_json(cfg.previous_supervision),
            manifest_path=os.path.abspath(cfg.previous_supervision),
            collected_entries=collected_entries,
        )
        print(
            f"[scene-zoom] previous_supervision: carried "
            f"{state.carried_from['samples']} sample(s) at zoom(s) "
            f"{[f'{zoom:g}' for zoom in carried_zooms]}."
        )
    if cfg.progressive:
        expected_prior = [float(cfg.step_scale) ** i for i in range(1, len(carried_zooms) + 1)]
        if carried_zooms != expected_prior or not math.isclose(
            float(cfg.zoom_factors[0]), float(cfg.step_scale) ** (len(carried_zooms) + 1)
        ):
            raise ValueError("Progressive levels must form an unbroken parent-to-child chain")
    plans = build_tile_plans(tasks, cfg, level_offset=len(carried_zooms))
    supervised_views = sum(1 for view in tasks if view.source_stage != REAL_REPLAY_STAGE)
    views_skipped_by_target_selection = max(0, supervised_views - len(plans))
    if cfg.progressive:
        for view, _tiles in plans:
            _progressive_work_raster(cfg, view.snapshot)
    edit_mask_paths = {
        view.view_id: _view_edit_mask_path(local_cfg, view.view_id)
        for view in tasks
    } if local_cfg else {}
    restored = restore_completed_supervision(state, previous, cfg, plans) if previous else 0
    if cfg.resume:
        print(f"[scene-zoom] resume: reused {restored} completed SR targets", flush=True)
    state.write()

    pending: list[tuple[ViewTask, dict[float, list[TileSpec]]]] = []
    for view, tiles_by_zoom in plans:
        remainder = {
            zoom: [tile for tile in tiles if tile.sample_id not in state.samples.get(zoom, {})]
            for zoom, tiles in tiles_by_zoom.items()
        }
        if any(remainder.values()):
            pending.append((view, remainder))
    skipped_by_max_views = 0
    if cfg.max_views > 0:
        skipped_by_max_views = max(0, len(pending) - int(cfg.max_views))
        pending = pending[: int(cfg.max_views)]
    pending_samples = sum(len(tiles) for _, remainder in pending for tiles in remainder.values())
    print(f"[scene-zoom] pending: {pending_samples} targets from {len(pending)} views", flush=True)

    provider, prompt_manager = build_prompt_backend(cfg)
    generation_cache_dir = os.path.abspath(
        cfg.generation_cache_dir or os.path.join(output_dir, GENERATION_CACHE_DIRNAME)
    )
    samples_generated = 0
    batch_count = 0
    local_edit_jobs = 0
    local_edit_edited = 0
    batch_size = max(1, int(cfg.geometry_batch_views))
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start : batch_start + batch_size]
        batch_count += 1
        if cfg.progressive:
            ctx = load_parent_render_context(cfg, namespace)
        else:
            ctx = load_scene_context(cfg, namespace)
        ctx["local_edit_identity"] = local_identity
        try:
            for view, tiles_by_zoom in batch:
                active = {zoom: tiles for zoom, tiles in tiles_by_zoom.items() if tiles}
                if active:
                    prepare_view_geometry(
                        view,
                        ctx,
                        cfg,
                        active,
                        view_edit_mask_path=edit_mask_paths.get(view.view_id),
                    )
        finally:
            del ctx
            _free_gpu()

        if local_cfg is not None:
            stats = apply_local_flowedit(cfg, local_cfg, batch)
            local_edit_jobs += int(stats["jobs"])
            local_edit_edited += int(stats["edited"])
            if stats["edited"]:
                print(
                    f"[scene-zoom] local flowedit: edited {stats['edited']} tile input(s) "
                    f"in batch {batch_count}."
                )

        with provider.session():
            for view, tiles_by_zoom in batch:
                pending_tiles = [tile for tiles in tiles_by_zoom.values() for tile in tiles]
                ensure_prompts(cfg, provider, prompt_manager, view, pending_tiles)

        for zoom_factor in cfg.zoom_factors:
            zoom = float(zoom_factor)
            for alignment in ("geometry", "spynet", "target_only"):
                jobs = []
                for view, tiles_by_zoom in batch:
                    for tile in tiles_by_zoom.get(zoom, ()):
                        if state.sample_exists(zoom, tile.sample_id):
                            continue
                        tile_path = os.path.join(
                            output_dir, SCRATCH_DIRNAME, view.view_id, tile.alias, "tile.json"
                        )
                        if not os.path.isfile(tile_path):
                            raise FileNotFoundError(f"Geometry scratch missing: {tile_path}")
                        record = _load_json(tile_path)
                        if _tile_mode(cfg, record) != alignment:
                            continue
                        prompt_path = os.path.join(
                            output_dir, PROMPTS_DIRNAME, f"{tile.sample_id}.json"
                        )
                        jobs.append((view, tile, record, _load_json(prompt_path)))
                if not jobs:
                    continue
                backend = build_dloral_backend(cfg, zoom, alignment)
                refiner = CachedRefiner(
                    backend, generation_cache_dir, enabled=not cfg.no_generation_cache
                )
                with backend.session():
                    for view, tile, record, prompt_payload in jobs:
                        sample = refine_tile(
                            cfg, refiner, view, tile, record, prompt_payload, alignment
                        )
                        state.add_sample(zoom, sample, write=False)
                        samples_generated += 1
                    state.write()
                del backend, refiner
                _free_gpu()

    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "skyfall_scene_zoom_preparation",
        "stage": stage,
        "start_checkpoint": os.path.abspath(cfg.start_checkpoint),
        "output_dir": output_dir,
        "zoom_factors": [float(zoom) for zoom in cfg.zoom_factors],
        "resolution": int(cfg.resolution),
        "max_views": int(cfg.max_views),
        "target_view_ids": [str(view_id) for view_id in cfg.target_view_ids],
        "views_skipped_by_target_selection": int(views_skipped_by_target_selection),
        "dloral_alignment": cfg.dloral_alignment,
        "samples_reused": restored,
        "geometry_neighbor_count": int(cfg.geometry_neighbor_count),
        "views_collected": len(tasks),
        "flowedit_views": sum(
            1 for view in tasks if view.source_stage != REAL_REPLAY_STAGE
        ),
        "real_replay_views": sum(
            1 for view in tasks if view.source_stage == REAL_REPLAY_STAGE
        ),
        "include_real_replay": bool(cfg.include_real_replay),
        "views_pending": len(pending),
        "views_skipped_by_max_views": int(skipped_by_max_views),
        "pending_tile_targets": int(pending_samples),
        "samples_generated": int(samples_generated),
        "batches": int(batch_count),
        "progressive": bool(cfg.progressive),
        "parent_lod_checkpoint": (
            os.path.abspath(cfg.parent_lod_checkpoint) if cfg.parent_lod_checkpoint else None
        ),
        "previous_supervision": (
            os.path.abspath(cfg.previous_supervision) if cfg.previous_supervision else None
        ),
        "carried_levels": carried_zooms,
        "carried_samples": len(state.carried),
        "local_flowedit_config": (
            os.path.abspath(cfg.local_flowedit_config) if cfg.local_flowedit_config else None
        ),
        "local_edit_jobs": int(local_edit_jobs),
        "local_edit_edited": int(local_edit_edited),
        "supervision_manifest": state.path,
        "elapsed_sec": time.perf_counter() - started,
    }
    _write_json(os.path.join(output_dir, "prepare_scene_zoom_summary.json"), summary)
    return summary
