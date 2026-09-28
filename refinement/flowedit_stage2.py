"""Pure FlowEdit refinement for the extrinsic IDU curriculum.

This restores the original Skyfall-GS baseline refinement: renders are repaired
directly with ``submodules/FlowEdit/idu_refine.FlowEditRefineIDU`` using the
default source/target prompts and the original sampler settings. One FLUX load
serves a whole episode. No geometry flow, Qwen prompts or DLoRAL models are
involved; the caller must release its Scene/Gaussians between the render and
refinement phases so FLUX fits on the same device.
"""
from __future__ import annotations

import gc
import hashlib
import inspect
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from PIL import Image

from .types import CameraSnapshot

SCHEMA_VERSION = 1
FLOWEDIT_VIEWS_KIND = "skyfall_flowedit_views"
_FLOWEDIT_MODULE = (
    Path(__file__).resolve().parents[1] / "submodules" / "FlowEdit" / "idu_refine.py"
)


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _open_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def validate_flowedit_options(options) -> None:
    """Check only the assets the pure FlowEdit backend uses (no Qwen/DLoRAL)."""
    model_type = getattr(options, "idu_model_type", "FLUX")
    if model_type not in ("FLUX", "SD3"):
        raise ValueError(f"FlowEdit supports model types 'FLUX' and 'SD3', got {model_type!r}")
    flux_model_path = getattr(options, "flux_model_path", "") or ""
    if flux_model_path:
        try:
            has_config = (Path(flux_model_path).expanduser() / "model_index.json").is_file()
        except OSError as error:
            raise FileNotFoundError(f"Cannot read local FLUX pipeline {flux_model_path}: {error}") from error
        if not has_config:
            raise FileNotFoundError(f"Missing local FLUX pipeline config: {flux_model_path}")
    if torch.device(getattr(options, "idu_refine_device", "cuda:0")).type != "cuda":
        raise ValueError("FlowEdit requires a CUDA device")
    n_min = int(getattr(options, "idu_flow_edit_n_min", 4))
    n_max = int(getattr(options, "idu_flow_edit_n_max", 10))
    n_max_end = int(getattr(options, "idu_flow_edit_n_max_end", -1))
    n_avg = int(getattr(options, "idu_flow_edit_n_avg", 1))
    if n_avg < 1 or n_min < 0 or n_min > n_max:
        raise ValueError("FlowEdit sampler needs n_avg >= 1 and 0 <= n_min <= n_max")
    if n_max_end != -1 and n_min > n_max_end:
        raise ValueError("FlowEdit sampling needs n_min <= n_max_end")
    if not _FLOWEDIT_MODULE.is_file():
        raise FileNotFoundError(
            "submodules/FlowEdit/idu_refine.py is missing; restore the FlowEdit "
            "submodule before selecting --idu_refine_backend flowedit"
        )


@torch.no_grad()
def render_flowedit_views(
    views: Sequence[Any], render_view: Callable,
    *, checkpoint_path: str, episode_dir: str, episode_idx: int, options,
) -> dict:
    """Render each unique curriculum pose once and store the RGB renders.

    Only raster data and serializable camera metadata are produced here; no
    geometry flows, prompts or generative models are involved. The caller must
    release its Scene/Gaussians afterwards and call :func:`refine_flowedit_views`.
    """
    validate_flowedit_options(options)
    if not views:
        raise ValueError("FlowEdit needs curriculum target views")
    if len({view.uid for view in views}) != len(views):
        raise ValueError("FlowEdit target views must have unique IDs")
    sizes = {(view.image_width, view.image_height) for view in views}
    if len(sizes) != 1:
        raise ValueError("FlowEdit target views must share a raster size")
    width, height = sizes.pop()
    if width % 16 or height % 16:
        raise ValueError(
            f"FlowEdit crops rasters to multiples of 16, so {width}x{height} "
            "cannot be repaired without changing the camera raster"
        )
    checkpoint = str(Path(checkpoint_path).resolve())
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    samples = int(options.idu_num_samples_per_view)
    root = Path(episode_dir).resolve()
    render_dir = root / "render"
    render_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, view in enumerate(views):
        rendered = render_view(view)
        rgb = rendered.rgb.detach().float()
        if tuple(rgb.shape) != (3, height, width) or not torch.isfinite(rgb).all():
            raise ValueError(f"Invalid RGB render for IDU view {index}")
        image = Image.fromarray((rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
        pose_path = render_dir / f"view_{index:05d}.png"
        image.save(pose_path)
        records.append(dict(
            index=index, uid=int(view.uid),
            camera=CameraSnapshot.from_camera(view).to_dict(),
            render_path=str(pose_path),
            rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        ))
        # Keep the per-sample render naming of the GaussianZoom Stage 2 layout so
        # compare.html and the episode archives stay consistent.
        for sample in range(samples):
            shutil.copyfile(pose_path, render_dir / f"{index * samples + sample:05d}.png")
        del rendered, rgb, image
    manifest = dict(
        schema_version=SCHEMA_VERSION, kind="flowedit_stage2_prepared",
        checkpoint=checkpoint,
        checkpoint_stat={"size": Path(checkpoint).stat().st_size, "mtime_ns": Path(checkpoint).stat().st_mtime_ns},
        episode_idx=int(episode_idx), episode_dir=str(root),
        num_views=len(views), samples_per_view=samples, width=width, height=height,
        camera_motion="extrinsic_low_elevation_orbit", zoom_factor=1.0, sr_scale=1.0,
        focal_zoom=False, views=records,
    )
    _write_json(root / "flowedit_prepared.json", manifest)
    return manifest


@torch.no_grad()
def refine_flowedit_views(prepared: dict, *, options) -> list[Image.Image]:
    """Repair every rendered pose with the original FlowEdit sampler.

    One FLUX load serves the whole episode. Each diffusion sample receives the
    same source render for its pose, as in the original IDU pipeline.
    """
    validate_flowedit_options(options)
    if prepared.get("schema_version") != SCHEMA_VERSION or prepared.get("kind") != "flowedit_stage2_prepared":
        raise ValueError("Unsupported FlowEdit preparation manifest")
    samples = int(options.idu_num_samples_per_view)
    if samples != prepared["samples_per_view"] or len(prepared["views"]) != prepared["num_views"]:
        raise ValueError("FlowEdit view/sample count changed after preparation")
    checkpoint = Path(prepared["checkpoint"])
    if prepared["checkpoint_stat"] != {"size": checkpoint.stat().st_size, "mtime_ns": checkpoint.stat().st_mtime_ns}:
        raise ValueError("Stage2 checkpoint changed after preparation")
    width, height = prepared["width"], prepared["height"]
    root = Path(prepared["episode_dir"])
    render_refine_dir = root / "render_refine"
    render_refine_dir.mkdir(parents=True, exist_ok=True)

    from submodules.FlowEdit.idu_refine import FlowEditRefineIDU

    model_path = getattr(options, "flux_model_path", "") or None
    if model_path and "model_path" not in inspect.signature(FlowEditRefineIDU.__init__).parameters:
        raise RuntimeError(
            "The FlowEdit submodule lacks model_path support; apply "
            "patches/flowedit-local.patch to submodules/FlowEdit/idu_refine.py"
        )
    sampler = dict(
        n_min=int(getattr(options, "idu_flow_edit_n_min", 4)),
        n_max=int(getattr(options, "idu_flow_edit_n_max", 10)),
        n_max_end=int(getattr(options, "idu_flow_edit_n_max_end", -1)),
        n_avg=int(getattr(options, "idu_flow_edit_n_avg", 1)),
    )
    if sampler["n_max_end"] == -1:
        sampler["n_max_end"] = None
    expected = len(prepared["views"]) * samples
    refine_pipe = FlowEditRefineIDU(
        save_path=str(render_refine_dir),
        device=getattr(options, "idu_refine_device", "cuda:0"),
        model_type=getattr(options, "idu_model_type", "FLUX"),
        model_path=model_path,
    )
    started = time.perf_counter()
    try:
        expanded = []
        for view in prepared["views"]:
            image = _open_image(view["render_path"])
            if image.size != (width, height) or hashlib.sha256(image.tobytes()).hexdigest() != view["rgb_sha256"]:
                raise ValueError("FlowEdit target image changed after preparation")
            array = np.asarray(image, dtype=np.float32) / 255.0
            expanded.extend([array] * samples)
            del image
        refined = refine_pipe.run(
            expanded,
            n_min=sampler["n_min"], n_max=sampler["n_max"],
            n_max_end=sampler["n_max_end"], n_avg=sampler["n_avg"],
        )
        del expanded
    finally:
        # Release CUDA weights directly; the upstream destructor otherwise copies
        # the entire FLUX pipeline to CPU before deleting it.
        refine_pipe.pipe = None
        del refine_pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if len(refined) != expected:
        raise RuntimeError(f"Expected {expected} FlowEdit-repaired IDU images, got {len(refined)}")
    if any(image.size != (width, height) for image in refined):
        raise ValueError("FlowEdit changed the IDU camera raster")
    report = dict(
        kind="flowedit_stage2_synthesis", prepared_episode_dir=str(root),
        episode_idx=prepared["episode_idx"], checkpoint=prepared["checkpoint"],
        model_type=getattr(options, "idu_model_type", "FLUX"), model_path=model_path,
        sampler=sampler, n_images=expected,
        timings={"refinement_seconds": time.perf_counter() - started},
        status="complete",
    )
    _write_json(root / "flowedit_synthesis.json", report)
    return refined


def build_flowedit_views_manifest(
    prepared: dict, *, checkpoint: str, episode_idx: int, samples_per_view: int,
) -> dict:
    """Build the reusable ``flowedit_views.json`` contract for the zoom stage.

    Each entry is one per-view sample: a stable id, the original (unzoomed) IDU
    camera snapshot, and the absolute path of the repaired PNG, which is the
    exact camera raster before any zoom SR.
    """
    if prepared.get("kind") != "flowedit_stage2_prepared":
        raise ValueError("Zoom supervision requires FlowEdit-repaired views")
    root = Path(prepared["episode_dir"]).resolve()
    views = []
    for view in prepared["views"]:
        camera = dict(view["camera"])
        for sample in range(samples_per_view):
            flat = view["index"] * samples_per_view + sample
            image_path = (root / "render_refine" / f"{flat:05d}.png").resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing FlowEdit-repaired image: {image_path}")
            views.append(dict(
                id=f"idu_e{episode_idx:02d}_v{view['index']:05d}_s{sample:02d}",
                camera=camera,
                image_path=str(image_path),
                source_stage="stage2",
                appearance_uid=None,
            ))
    return dict(
        schema_version=1,
        kind=FLOWEDIT_VIEWS_KIND,
        checkpoint=str(Path(checkpoint).resolve()),
        episode_idx=int(episode_idx),
        views=views,
    )


#: Appearance embedding index the native IDU render path uses for synthetic
#: views (``render(..., testing=True)`` picks ``appearance_embeddings[min(6, N-1)]``).
IDU_APPEARANCE_UID = 6


_REQUIRED_CAMERA_KEYS = (
    "image_width", "image_height", "fov_x", "fov_y", "cx", "cy", "R", "T",
)


def _validate_camera_snapshot(camera: Any, where: str) -> None:
    if not isinstance(camera, dict):
        raise ValueError(f"{where}: camera must be an object")
    missing = [key for key in _REQUIRED_CAMERA_KEYS if key not in camera]
    if missing:
        raise ValueError(f"{where}: camera snapshot missing fields {missing}")
    for key in ("image_width", "image_height"):
        value = camera[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{where}: camera.{key} must be a positive int, got {value!r}")
    for key in ("fov_x", "fov_y", "cx", "cy"):
        value = camera[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{where}: camera.{key} must be a finite number, got {value!r}")
        if key in ("fov_x", "fov_y") and not 0.0 < float(value) < math.pi:
            raise ValueError(f"{where}: camera.{key} must be radians in (0, pi), got {value!r}")
    rotation = camera["R"]
    if (
        not isinstance(rotation, list) or len(rotation) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in rotation)
        or any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for row in rotation for value in row)
    ):
        raise ValueError(f"{where}: camera.R must be a 3x3 numeric matrix")
    translation = camera["T"]
    if (
        not isinstance(translation, list) or len(translation) != 3
        or any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in translation)
    ):
        raise ValueError(f"{where}: camera.T must be a length-3 numeric vector")


def load_flowedit_views_manifest(
    path: str | os.PathLike,
    *,
    expected_checkpoint: str | None = None,
) -> dict:
    """Load and validate a ``skyfall_flowedit_views`` supervision manifest.

    Enforces kind/schema, that ``checkpoint`` is an existing file (equal to
    ``expected_checkpoint`` when given, so foreign-source teachers cannot slip
    in), per-view provenance (``source_stage``/``appearance_uid``), and that
    every image file exists with exactly its snapshot's raster. A uniform
    raster, FoV and principal point are required, matching the uniform IDU
    camera course the manifest was generated from.

    Returns the parsed payload with every ``image_path`` resolved to an
    absolute path (relative paths are interpreted against the manifest file).
    """
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"FlowEdit supervision manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read FlowEdit supervision manifest {manifest_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{manifest_path}: manifest must be a JSON object")
    if payload.get("kind") != FLOWEDIT_VIEWS_KIND:
        raise ValueError(
            f"{manifest_path}: expected kind {FLOWEDIT_VIEWS_KIND!r}, got {payload.get('kind')!r}"
        )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{manifest_path}: unsupported schema_version {payload.get('schema_version')!r}"
        )
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError(f"{manifest_path}: manifest checkpoint must be a non-empty string")
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{manifest_path}: manifest checkpoint not found: {checkpoint_path}")
    if expected_checkpoint:
        expected = Path(expected_checkpoint).expanduser().resolve()
        if checkpoint_path.resolve() != expected:
            raise ValueError(
                f"{manifest_path}: supervision teacher checkpoint {checkpoint_path} does not match "
                f"the training start checkpoint {expected}; refusing incompatible teachers"
            )
    episode_idx = payload.get("episode_idx")
    if not isinstance(episode_idx, int) or isinstance(episode_idx, bool) or episode_idx < 0:
        raise ValueError(f"{manifest_path}: episode_idx must be a non-negative int, got {episode_idx!r}")
    views = payload.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError(f"{manifest_path}: views must be a non-empty list")

    seen_ids: set[str] = set()
    rasters: set[tuple[int, int]] = set()
    intrinsics: set[tuple[float, float, float, float]] = set()
    for index, view in enumerate(views):
        where = f"{manifest_path}: views[{index}]"
        if not isinstance(view, dict):
            raise ValueError(f"{where}: view must be an object")
        view_id = view.get("id")
        if not isinstance(view_id, str) or not view_id:
            raise ValueError(f"{where}: id must be a non-empty string")
        if view_id in seen_ids:
            raise ValueError(f"{where}: duplicate view id {view_id!r}")
        seen_ids.add(view_id)
        source_stage = view.get("source_stage")
        if source_stage != "stage2":
            raise ValueError(
                f"{where} ({view_id}): source_stage {source_stage!r} is not FlowEdit stage2 "
                "supervision; real photographs must use the native dataset replay path"
            )
        appearance_uid = view.get("appearance_uid")
        if appearance_uid is not None and appearance_uid != IDU_APPEARANCE_UID:
            raise ValueError(
                f"{where} ({view_id}): appearance_uid {appearance_uid!r} != {IDU_APPEARANCE_UID}; "
                "the native IDU render path uses that embedding index for synthetic views"
            )
        _validate_camera_snapshot(view.get("camera"), where)
        camera = view["camera"]
        image_path = view.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            raise ValueError(f"{where} ({view_id}): image_path must be a non-empty string")
        resolved_image = Path(image_path).expanduser()
        if not resolved_image.is_absolute():
            resolved_image = (manifest_path.parent / resolved_image).resolve()
        if not resolved_image.is_file():
            raise FileNotFoundError(f"{where} ({view_id}): supervision image not found: {resolved_image}")
        with Image.open(resolved_image) as probe:
            if probe.size != (int(camera["image_width"]), int(camera["image_height"])):
                raise ValueError(
                    f"{where} ({view_id}): image {resolved_image.name} is {probe.size}, "
                    f"expected {(int(camera['image_width']), int(camera['image_height']))} "
                    "per its camera snapshot"
                )
        view["image_path"] = str(resolved_image)
        rasters.add((int(camera["image_width"]), int(camera["image_height"])))
        intrinsics.add((float(camera["fov_x"]), float(camera["fov_y"]), float(camera["cx"]), float(camera["cy"])))

    if len(rasters) != 1:
        raise ValueError(f"{manifest_path}: supervision views must share one raster, got {sorted(rasters)}")
    if len(intrinsics) != 1:
        raise ValueError(
            f"{manifest_path}: supervision views must share one FoV/principal point, got {sorted(intrinsics)}"
        )
    return payload


__all__ = [
    "validate_flowedit_options",
    "render_flowedit_views",
    "refine_flowedit_views",
    "build_flowedit_views_manifest",
    "load_flowedit_views_manifest",
]


