"""Localized FlowEdit: masked ROI editing with the original FlowEdit sampler.

Every job carries an explicit RGB image and a same-size ``0..255`` edit mask.
The editor crops a context window around the mask support, runs the ORIGINAL
``submodules.FlowEdit.idu_refine.FlowEditRefineIDU`` sampler on that crop with
full FP16 weights (no quantization, no offload, no hidden resize), and
composites the refined crop back into the full-raster image strictly inside
the mask support with a feathered alpha.  Pixels whose feathered alpha is
exactly zero are byte-identical to the input image.

A crop is aligned to multiples of 16, which is the actual FLUX requirement:
the VAE downsamples by 8 and the transformer patchifies latents by 2.  When
the aligned context box is clipped by an image border that is not itself a
multiple of 16, the crop is reflect-padded outwards (bounded, recorded) and
the pad columns are discarded on compositing.  If the crop would still exceed
``max_crop_pixels`` it is resampled down (recorded) and the refined crop is
resampled back to the exact source crop size, so the saved output always
matches the input raster.

A job whose mask has no support pixels never invokes the editor: its output is
a byte-identical copy of the input and its metadata records ``skipped``.

The module also ships :func:`invalidate_target_side_geometry`, the reusable
invalidation helper for DLoRAL target-side geometry correspondence.  Per
``refinement.types.GeometryCorrespondence``, ``target_to_source_flow`` is
``p_neighbor - p_target`` at *target* pixels and ``valid_mask`` lives on the
target grid; it is the field that warps the neighbor onto the target grid.
When an edit region replaces target content, the old neighbor correspondence
in that region is stale and must not be consumed: the helper sets the edited
support invalid on the target grid (``valid_mask := valid & ~support`` and
``flow := NaN`` there) and leaves the reverse/source side untouched.

The batch API is :func:`refine_local_flowedit_jobs(jobs, config)` where each
job is ``{"image_path", "mask_path", "output_path", "metadata_path"}`` and the
result repeats those keys plus ``status`` and the persisted ``metadata``.
Config keys: ``weights_path, n_min, n_max, T_steps, n_avg, src_guidance,
tar_guidance, seed, context_padding, feather_radius, source_prompt,
target_prompt`` plus the optional operational keys ``device, model_type,
max_crop_pixels, min_free_vram_mb``.  Unknown keys (e.g. ``view_masks``) are
ignored and echoed back in ``metadata["config_extra"]``.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageFilter

CROP_ALIGN = 16  # FLUX: VAE downsample 8 x transformer patch 2



class LocalFlowEditError(RuntimeError):
    """Actionable failure; the message states what is missing and what to do."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    """JSON-safe projection with deterministic ordering."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: str | os.PathLike, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(_plain(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, target)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

_INT_FIELDS = ("n_min", "n_max", "T_steps", "n_avg", "context_padding",
               "feather_radius", "max_crop_pixels", "min_free_vram_mb", "seed")
_FLOAT_FIELDS = ("src_guidance", "tar_guidance")
_STR_FIELDS = ("weights_path", "source_prompt", "target_prompt", "device",
               "model_type")


@dataclass
class LocalFlowEditConfig:
    """Validated local editor configuration (see module docstring for keys)."""

    weights_path: str = ""
    n_min: int = 4
    n_max: int = 10
    T_steps: int = 28
    n_avg: int = 1
    src_guidance: float = 1.5
    tar_guidance: float = 5.5
    seed: int = 0
    context_padding: int = 64
    feather_radius: int = 12
    source_prompt: str = ""
    target_prompt: str = ""
    device: str = "cuda:0"
    model_type: str = "FLUX"
    max_crop_pixels: int = 0
    min_free_vram_mb: int = 0
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in _INT_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise LocalFlowEditError(
                    f"local_flowedit config field {name!r} must be a non-negative "
                    f"int, got {value!r}."
                )
        for name in _FLOAT_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) \
                    or not math.isfinite(float(value)) or not float(value) > 0.0:
                raise LocalFlowEditError(
                    f"local_flowedit config field {name!r} must be a positive "
                    f"number, got {value!r}."
                )
        for name in _STR_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str):
                raise LocalFlowEditError(
                    f"local_flowedit config field {name!r} must be a string, got "
                    f"{type(value).__name__}."
                )
        if self.model_type not in ("FLUX", "SD3"):
            raise LocalFlowEditError(
                f"local_flowedit config model_type must be 'FLUX' or 'SD3', got "
                f"{self.model_type!r}."
            )
        if self.n_max < self.n_min:
            raise LocalFlowEditError(
                f"local_flowedit config n_max ({self.n_max}) must be >= n_min "
                f"({self.n_min})."
            )
        if self.T_steps < 1 or self.n_avg < 1 or self.n_max > self.T_steps:
            raise LocalFlowEditError("Require positive T_steps/n_avg and n_max <= T_steps.")
        if not isinstance(self.extra, dict):
            raise LocalFlowEditError("local_flowedit config extra must be a dict.")

    def to_dict(self) -> dict:
        payload = {name: getattr(self, name) for name in (
            "weights_path", "n_min", "n_max", "T_steps", "n_avg",
            "src_guidance", "tar_guidance", "seed", "context_padding",
            "feather_radius", "source_prompt", "target_prompt", "device",
            "model_type", "max_crop_pixels", "min_free_vram_mb",
        )}
        payload["extra"] = dict(self.extra)
        return payload

    @classmethod
    def from_dict(cls, config: Mapping[str, Any] | None) -> "LocalFlowEditConfig":
        data = dict(config or {})
        extra = {}
        known = set(_INT_FIELDS) | set(_FLOAT_FIELDS) | set(_STR_FIELDS)
        for key in data:
            if key not in known:
                extra[str(key)] = data[key]
        fields = {name: data[name] for name in known if name in data}
        return cls(extra=extra, **fields)


# ---------------------------------------------------------------------------
# crop planning, feathering and compositing (pure, no editor needed)
# ---------------------------------------------------------------------------

def plan_local_crop(
    support: np.ndarray,
    context_padding: int,
    max_crop_pixels: int = 0,
) -> dict:
    """Plan the 16-aligned context crop around the edit support.

    ``support`` is a boolean ``(H, W)`` mask of the pixels the edit may touch.
    The returned plan contains:

    * ``box``: ``(left, top, right, bottom)`` exclusive window in image coords;
      ``left``/``top`` are floored to a multiple of 16 and ``right``/``bottom``
      are ceiled, never beyond the image.
    * ``pad``: ``(pad_left, pad_top, pad_right, pad_bottom)`` reflect padding
      added beyond the image border when the border itself is not 16-aligned.
    * ``padded_size``: exact crop array size fed to the sampler (multiple of 16).
    * ``resample``: ``None`` or a record with ``scale``, ``source_crop`` and
      ``sampler_crop`` when the crop was resampled down to ``max_crop_pixels``.
    """
    if support.ndim != 2 or not np.issubdtype(support.dtype, np.bool_):
        raise LocalFlowEditError(
            f"plan_local_crop expects a boolean (H, W) support mask, got shape "
            f"{tuple(np.shape(support))} dtype {np.asarray(support).dtype}."
        )
    height, width = support.shape
    ys, xs = np.nonzero(support)
    if ys.size == 0:
        raise LocalFlowEditError(
            "plan_local_crop called with an empty support mask; zero-mask jobs "
            "must be skipped without planning."
        )
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    padding = int(context_padding)
    left = max(0, x0 - padding)
    top = max(0, y0 - padding)
    right = min(width, x1 + padding)
    bottom = min(height, y1 + padding)
    left -= left % CROP_ALIGN
    top -= top % CROP_ALIGN
    right_aligned = -(-right // CROP_ALIGN) * CROP_ALIGN
    bottom_aligned = -(-bottom // CROP_ALIGN) * CROP_ALIGN
    pad_right = pad_bottom = 0
    if right_aligned <= width:
        right = right_aligned
    else:
        right = width
        pad_right = (CROP_ALIGN - (right - left) % CROP_ALIGN) % CROP_ALIGN
    if bottom_aligned <= height:
        bottom = bottom_aligned
    else:
        bottom = height
        pad_bottom = (CROP_ALIGN - (bottom - top) % CROP_ALIGN) % CROP_ALIGN
    crop_w, crop_h = (right - left) + pad_right, (bottom - top) + pad_bottom
    resample = None
    if max_crop_pixels and crop_w * crop_h > int(max_crop_pixels):
        scale = (float(max_crop_pixels) / float(crop_w * crop_h)) ** 0.5
        new_w = max(CROP_ALIGN, (int(crop_w * scale) // CROP_ALIGN) * CROP_ALIGN)
        new_h = max(CROP_ALIGN, (int(crop_h * scale) // CROP_ALIGN) * CROP_ALIGN)
        resample = {
            "scale": scale,
            "source_crop": [int(crop_w), int(crop_h)],
            "sampler_crop": [int(new_w), int(new_h)],
        }
        crop_w, crop_h = new_w, new_h
    return {
        "box": [int(left), int(top), int(right), int(bottom)],
        "pad": [0, 0, int(pad_right), int(pad_bottom)],
        "padded_size": [int(crop_w), int(crop_h)],
        "resample": resample,
    }


def extract_crop(image: np.ndarray, plan: Mapping[str, Any]) -> np.ndarray:
    """Cut the planned crop out of an RGB image and reflect-pad its borders."""
    left, top, right, bottom = plan["box"]
    pad_left, pad_top, pad_right, pad_bottom = plan["pad"]
    if pad_left or pad_top:
        raise LocalFlowEditError(
            "plan_local_crop never pads left/top; unexpected pad values "
            f"{plan['pad']}."
        )
    crop = np.asarray(image)[top:bottom, left:right]
    if pad_right or pad_bottom:
        crop = np.pad(
            crop,
            ((0, pad_bottom), (0, pad_right), (0, 0)),
            mode="reflect" if (crop.shape[0] > pad_bottom and crop.shape[1] > pad_right) else "edge",
        )
    expected = tuple(plan["padded_size"])
    if crop.shape[:2] != (expected[1], expected[0]):
        raise LocalFlowEditError(
            f"Crop extraction produced {crop.shape[:2]}, expected "
            f"{(expected[1], expected[0])}."
        )
    return np.ascontiguousarray(crop)


def feather_alpha(edit_mask: np.ndarray, feather_radius: int) -> np.ndarray:
    """Feathered compositing alpha, strictly inside the edit support.

    ``feather_radius`` is the Gaussian sigma in pixels; 0 keeps the binary
    support fully hard.  The blurred support is normalized to its own peak
    so interior pixels keep full edit strength, then multiplied by the
    support itself so alpha is exactly zero outside the edit mask -- every
    pixel outside the mask is byte-identical to the original image.
    """
    mask = np.asarray(edit_mask).astype(bool)
    if mask.ndim != 2:
        raise LocalFlowEditError(
            f"feather_alpha expects a 2-D mask, got shape {mask.shape}."
        )
    radius = int(feather_radius)
    if radius < 0:
        raise LocalFlowEditError(f"feather_radius must be >= 0, got {radius}.")
    mask_image = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), "L")
    if radius:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=radius))
    alpha = np.asarray(mask_image, dtype=np.float64) / 255.0
    peak = float(alpha.max())
    if peak > 0.0:
        alpha = np.minimum(alpha / peak, 1.0)
    alpha *= mask
    return alpha


def compose_local_edit(
    source: np.ndarray,
    edited_crop: np.ndarray,
    plan: Mapping[str, Any],
    alpha: np.ndarray,
) -> np.ndarray:
    """Composite the refined crop back into the full image.

    Pixels with alpha exactly zero are copied byte-identically from
    ``source``; the feathered transition lives only inside the crop box.
    Reflect-pad columns of the crop are never pasted.
    """
    height, width = np.asarray(source).shape[:2]
    if np.asarray(alpha).shape != (height, width):
        raise LocalFlowEditError(
            f"compositing alpha shape {np.asarray(alpha).shape} does not match "
            f"the image {(height, width)}."
        )
    left, top, right, bottom = plan["box"]
    pad_left, pad_top, pad_right, pad_bottom = plan["pad"]
    crop_w, crop_h = (right - left) + pad_right, (bottom - top) + pad_bottom
    crop = np.asarray(edited_crop)
    if crop.shape[:2] != (crop_h, crop_w):
        if plan["resample"]:
            crop = np.asarray(
                Image.fromarray(np.asarray(crop, dtype=np.uint8)).resize(
                    (crop_w, crop_h), Image.Resampling.LANCZOS
                )
            )
        else:
            raise LocalFlowEditError(
                f"refined crop shape {crop.shape[:2]} does not match the planned "
                f"crop {(crop_h, crop_w)}."
            )
    source = np.asarray(source, dtype=np.float64)
    crop = crop.astype(np.float64)
    alpha = np.asarray(alpha, dtype=np.float64).copy()
    alpha[:, :left] = 0.0
    alpha[:, right:] = 0.0
    alpha[:top, :] = 0.0
    alpha[bottom:, :] = 0.0
    placed = np.zeros((height, width, 3), dtype=np.float64)
    placed[top:bottom, left:right] = crop[: bottom - top, : right - left]
    composite = source * (1.0 - alpha[..., None]) + placed * alpha[..., None]
    return np.rint(composite).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# reusable DLoRAL target-side geometry invalidation
# ---------------------------------------------------------------------------

def _as_numpy(array: Any) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array)


def dilate_support(support: np.ndarray, radius: int) -> np.ndarray:
    """Boolean dilation of a support mask by a square ``2*radius + 1`` kernel."""
    mask = np.asarray(support).astype(bool)
    if radius <= 0:
        return mask.copy()
    size = 2 * int(radius) + 1
    if mask.shape[0] < size or mask.shape[1] < size:
        # MaxFilter needs at least the kernel footprint; fall back to an
        # equivalent numpy dilation via maximum filtering in two passes.
        grown = mask.copy()
        for axis, amount in ((0, int(radius)), (1, int(radius))):
            if amount == 0:
                continue
            padded = np.pad(grown, [(amount, amount) if a == axis else (0, 0)
                                    for a in range(2)],
                            mode="edge")
            grown = np.zeros_like(mask)
            for shift in range(2 * amount + 1):
                if axis == 0:
                    grown |= padded[shift:shift + mask.shape[0], :]
                else:
                    grown |= padded[:, shift:shift + mask.shape[1]]
        return grown
    image = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), "L")
    return np.asarray(image.filter(ImageFilter.MaxFilter(size=size))) > 0


def invalidate_target_side_geometry(
    pixel_flow: Any,
    valid_mask: Any,
    edit_support: Any,
    dilation_radius: int = 0,
) -> dict:
    """Invalidate DLoRAL target-side geometry inside an edit support.

    Consumes the stored ``target_to_source_flow`` (``p_neighbor - p_target``
    at target pixels, NaN when invalid) and its boolean ``valid_mask`` on the
    target grid, plus a boolean ``edit_support`` on the same grid.  Returns
    new arrays with the dilated support excluded from the target side; the
    reverse (``source_to_target_flow`` / ``reverse_valid_mask``) is NOT
    touched and must keep the real neighbor correspondence.

    Returns ``{"pixel_flow", "valid_mask", "support_pixels",
    "invalidated_pixels", "dilation_radius", "flow_sha256", "valid_sha256"}``.
    """
    flow = np.array(_as_numpy(pixel_flow), dtype=np.float32, copy=True)
    valid = np.array(_as_numpy(valid_mask), dtype=bool, copy=True)
    support = np.asarray(_as_numpy(edit_support), dtype=bool)
    if support.shape != valid.shape:
        raise LocalFlowEditError(
            f"Edit support shape {support.shape} does not match the target-grid "
            f"geometry validity {valid.shape}; masks must be raster-contained."
        )
    if valid.ndim != 2 or flow.shape != (*valid.shape, 2):
        raise LocalFlowEditError(
            f"pixel_flow spatial shape {flow.shape[:2]} does not match the "
            f"validity mask {valid.shape}."
        )
    invalid = dilate_support(support, int(dilation_radius))
    invalidated = valid & invalid
    valid = valid & ~invalid
    flow[invalid] = np.float32("nan")
    return {
        "pixel_flow": np.ascontiguousarray(flow),
        "valid_mask": np.ascontiguousarray(valid),
        "support_pixels": int(support.sum()),
        "invalidated_pixels": int(invalidated.sum()),
        "dilation_radius": int(dilation_radius),
        "flow_sha256": sha256_array(flow),
        "valid_sha256": sha256_array(valid),
    }


def sha256_array(array: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(array).tobytes())


# ---------------------------------------------------------------------------
# batch execution
# ---------------------------------------------------------------------------

def reset_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _require_editor_weights(config: LocalFlowEditConfig) -> Path:
    weights_path = Path(config.weights_path).expanduser().resolve()
    if not str(config.weights_path).strip():
        raise LocalFlowEditError(
            "local_flowedit config needs weights_path pointing at a local "
            "FLUX.1-dev diffusers directory (model_index.json); hub download "
            "is forbidden."
        )
    if not (weights_path / "model_index.json").is_file():
        raise LocalFlowEditError(
            f"local_flowedit weights_path {weights_path} lacks model_index.json; "
            "it must be a local FLUX.1-dev (or SD3) diffusers directory."
        )
    return weights_path


def _vram_free_mb(device: str) -> int | None:
    import torch

    if not torch.cuda.is_available():
        return None
    try:
        free, _ = torch.cuda.mem_get_info(torch.device(device))
    except Exception:  # pragma: no cover - exotic device strings
        return None
    return int(free) >> 20


def _ensure_flowedit_import_path() -> None:
    import sys

    flowedit_dir = str(Path(__file__).resolve().parents[1] / "submodules" / "FlowEdit")
    if flowedit_dir not in sys.path:
        sys.path.insert(0, flowedit_dir)


def _load_editor(config: LocalFlowEditConfig, work_dir: Path):
    """Load the original full-FP16 FlowEdit editor once for a whole batch."""
    weights_path = _require_editor_weights(config)
    if config.min_free_vram_mb:
        free = _vram_free_mb(config.device)
        if free is not None and free < int(config.min_free_vram_mb):
            raise LocalFlowEditError(
                f"FlowEdit load aborted: only {free} MiB free on {config.device} "
                f"but the gate requires {config.min_free_vram_mb} MiB. Free GPU "
                "memory or lower the gate explicitly; no quantization/offload "
                "will be attempted."
            )
    _ensure_flowedit_import_path()
    from submodules.FlowEdit.idu_refine import (
        FlowEditRefineIDU, default_src_prompt, default_tar_prompt,
    )

    reset_seed(config.seed)
    work_dir.mkdir(parents=True, exist_ok=True)
    pipe = FlowEditRefineIDU(
        save_path=str(work_dir), device=config.device, model_type=config.model_type,
        model_path=str(weights_path),
    )
    return pipe, default_src_prompt, default_tar_prompt


def _release_editor(pipe) -> None:
    """Release CUDA weights directly; upstream's destructor copies to CPU."""
    import torch

    pipe.pipe = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _validate_job(job: Mapping[str, Any]) -> dict:
    if not isinstance(job, Mapping):
        raise LocalFlowEditError(
            f"Each local-flowedit job must be a dict with image_path, mask_path, "
            f"output_path, metadata_path; got {type(job).__name__}."
        )
    missing = [key for key in ("image_path", "mask_path", "output_path",
                               "metadata_path") if not job.get(key)]
    if missing:
        raise LocalFlowEditError(
            f"local-flowedit job is missing required paths {missing}: {dict(job)!r}."
        )
    spec = {key: Path(job[key]).expanduser().resolve() for key in (
        "image_path", "mask_path", "output_path", "metadata_path")}
    for key in ("image_path", "mask_path"):
        if not spec[key].is_file():
            raise LocalFlowEditError(
                f"local-flowedit job {key} not found: {spec[key]}."
            )
    # Optional per-job sampler overrides (used by batch runners whose cases
    # carry different n_min/n_max windows).  Absent keys fall back to config.
    spec["overrides"] = {}
    for key in ("n_min", "n_max", "seed"):
        if key in job and job[key] is not None:
            value = int(job[key])
            if value < 0:
                raise LocalFlowEditError(
                    f"local-flowedit job override {key!r} must be >= 0, got {value}."
                )
            spec["overrides"][key] = value
    return spec


def _load_image_and_mask(spec: Mapping[str, Path]) -> tuple[np.ndarray, np.ndarray, dict]:
    with Image.open(spec["image_path"]) as image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_arr = np.asarray(image, dtype=np.uint8).copy()
        image_size = [int(image.width), int(image.height)]
    with Image.open(spec["mask_path"]) as mask:
        if mask.mode != "L":
            mask = mask.convert("L")
        mask_arr = np.asarray(mask, dtype=np.uint8).copy()
        mask_size = [int(mask.width), int(mask.height)]
    if mask_size != image_size:
        raise LocalFlowEditError(
            f"Edit mask {spec['mask_path']} is {mask_size} but its image "
            f"{spec['image_path']} is {image_size}; mask must be raster-contained."
        )
    return image_arr, mask_arr, {"image_size": image_size, "mask_size": mask_size}


def refine_local_flowedit_jobs(
    jobs: list,
    config: Mapping[str, Any] | LocalFlowEditConfig,
) -> list:
    """Run masked localized FlowEdit over a batch of jobs, one FLUX load.

    Every job is ``{"image_path", "mask_path", "output_path", "metadata_path"}``.
    A zero-support mask short-circuits: the output is a byte-identical copy of
    the input and no editor is invoked.  A job may carry optional ``n_min``,
    ``n_max`` and ``seed`` overrides for batch runners whose cases use
    different sampler windows; absent keys fall back to the config.  Returns
    one result dict per job in input order with ``status`` and the persisted
    ``metadata``.
    """
    parsed = config if isinstance(config, LocalFlowEditConfig) \
        else LocalFlowEditConfig.from_dict(config)
    specs = [_validate_job(job) for job in jobs]
    results: list[dict | None] = [None] * len(specs)
    edited: list[tuple[int, dict, np.ndarray, np.ndarray]] = []

    for index, spec in enumerate(specs):
        image_arr, mask_arr, sizes = _load_image_and_mask(spec)
        support = mask_arr > 0
        base = {
            "input": {
                "path": str(spec["image_path"]),
                "sha256": sha256_file(spec["image_path"]),
                "size": sizes["image_size"],
            },
            "mask": {
                "path": str(spec["mask_path"]),
                "sha256": sha256_file(spec["mask_path"]),
                "support_pixels": int(support.sum()),
            },
        }
        spec["output_path"].parent.mkdir(parents=True, exist_ok=True)
        if not support.any():
            shutil.copyfile(spec["image_path"], spec["output_path"])
            metadata = {
                **base,
                "output": {
                    "path": str(spec["output_path"]),
                    "sha256": sha256_file(spec["output_path"]),
                    "size": sizes["image_size"],
                },
                "status": "skipped",
                "editor_invoked": False,
                "skip_reason": "edit mask has no support pixels; no editor invocation",
                "config_extra": _plain(parsed.extra),
                "edit_support": {"mask_pixels": 0, "alpha_pixels": 0},
            }
            write_json(spec["metadata_path"], metadata)
            results[index] = {
                "image_path": str(spec["image_path"]),
                "mask_path": str(spec["mask_path"]),
                "output_path": str(spec["output_path"]),
                "metadata_path": str(spec["metadata_path"]),
                "status": "skipped",
                "metadata": metadata,
            }
            continue
        edited.append((index, spec, image_arr, mask_arr))

    if not edited:
        return results  # type: ignore[return-value]

    work_dir = Path(specs[0]["output_path"]).parent / "local_flowedit_work"
    pipe, default_src, default_tar = _load_editor(parsed, work_dir)
    try:
        for index, spec, image_arr, mask_arr in edited:
            support = mask_arr > 0
            plan = plan_local_crop(support, parsed.context_padding,
                                   parsed.max_crop_pixels)
            crop = extract_crop(image_arr, plan)
            overrides = spec.get("overrides") or {}
            job_seed = overrides.get("seed", parsed.seed)
            job_n_min = overrides.get("n_min", parsed.n_min)
            job_n_max = overrides.get("n_max", parsed.n_max)
            if job_n_max < job_n_min:
                raise LocalFlowEditError(
                    f"Sampler window n_max ({job_n_max}) must be >= n_min "
                    f"({job_n_min}) for job {spec['image_path']}."
                )
            reset_seed(job_seed)
            refined = pipe.run(
                [crop],
                src_prompt=parsed.source_prompt or default_src,
                tar_prompt=parsed.target_prompt or default_tar,
                T_steps=parsed.T_steps, n_avg=parsed.n_avg,
                src_guidance_scale=parsed.src_guidance,
                tar_guidance_scale=parsed.tar_guidance,
                n_min=job_n_min, n_max=job_n_max, n_max_end=None,
            )
            sampler_window = {
                "T_steps": int(parsed.T_steps), "n_avg": int(parsed.n_avg),
                "n_min": int(job_n_min), "n_max": int(job_n_max),
                "n_max_end": None, "seed": int(job_seed),
                "src_guidance_scale": float(parsed.src_guidance),
                "tar_guidance_scale": float(parsed.tar_guidance),
            }
            prompts_origin = ("config" if parsed.source_prompt or parsed.target_prompt
                              else "original_upstream_defaults")
            if len(refined) != 1:
                raise LocalFlowEditError(
                    f"FlowEdit returned {len(refined)} outputs for crop at "
                    f"{spec['image_path']}; expected exactly one."
                )
            refined_crop = np.asarray(refined[0].convert("RGB"), dtype=np.uint8)
            expected = tuple(plan["padded_size"])
            if refined_crop.shape[:2] != (expected[1], expected[0]):
                raise LocalFlowEditError(
                    f"FlowEdit crop output {refined_crop.shape[:2]} does not "
                    f"match the planned crop {(expected[1], expected[0])}."
                )
            alpha = feather_alpha(mask_arr, parsed.feather_radius)
            composite = compose_local_edit(image_arr, refined_crop, plan, alpha)
            output_image = Image.fromarray(composite, "RGB")
            temporary = Path(spec["output_path"]).with_name(
                f".{Path(spec['output_path']).name}.tmp")
            temporary.parent.mkdir(parents=True, exist_ok=True)
            output_image.save(temporary, format="PNG")
            os.replace(temporary, spec["output_path"])
            evidence_dir = spec["metadata_path"].parent / (spec["metadata_path"].stem + "_crops")
            evidence_dir.mkdir(parents=True, exist_ok=True)
            crop_input_path = evidence_dir / "input.png"
            crop_output_path = evidence_dir / "edited.png"
            Image.fromarray(crop).save(crop_input_path)
            Image.fromarray(refined_crop).save(crop_output_path)
            outside_unchanged = np.array_equal(composite[~support], image_arr[~support])
            if not outside_unchanged:
                raise LocalFlowEditError("Localized FlowEdit modified pixels outside the mask")
            metadata = {
                "input": {
                    "path": str(spec["image_path"]),
                    "sha256": sha256_file(spec["image_path"]),
                    "size": [int(image_arr.shape[1]), int(image_arr.shape[0])],
                },
                "mask": {
                    "path": str(spec["mask_path"]),
                    "sha256": sha256_file(spec["mask_path"]),
                    "support_pixels": int(support.sum()),
                },
                "output": {
                    "path": str(spec["output_path"]),
                    "sha256": sha256_file(spec["output_path"]),
                    "size": [int(composite.shape[1]), int(composite.shape[0])],
                },
                "crop": {
                    "box": plan["box"],
                    "padded_size": plan["padded_size"],
                    "pad": plan["pad"],
                    "resample": plan["resample"],
                    "input_path": str(crop_input_path),
                    "refined_path": str(crop_output_path),
                    "input_sha256": sha256_array(crop),
                    "refined_sha256": sha256_array(refined_crop),
                },
                "sampler": dict(sampler_window),
                "compositing": {
                    "feather_radius": int(parsed.feather_radius),
                    "context_padding": int(parsed.context_padding),
                    "alpha_support_pixels": int((alpha > 0).sum()),
                    "byte_identical_outside_support": bool(outside_unchanged),
                },
                "edit_support": {
                    "mask_pixels": int(support.sum()),
                    "alpha_pixels": int((alpha > 0).sum()),
                },
                "status": "edited",
                "editor_invoked": True,
                "config_extra": _plain(parsed.extra),
                "prompts_origin": prompts_origin,
            }
            write_json(spec["metadata_path"], metadata)
            results[index] = {
                "image_path": str(spec["image_path"]),
                "mask_path": str(spec["mask_path"]),
                "output_path": str(spec["output_path"]),
                "metadata_path": str(spec["metadata_path"]),
                "status": "edited",
                "metadata": metadata,
            }
    finally:
        _release_editor(pipe)
    return results

__all__ = [
    "CROP_ALIGN",
    "LocalFlowEditConfig",
    "LocalFlowEditError",
    "compose_local_edit",
    "dilate_support",
    "extract_crop",
    "feather_alpha",
    "invalidate_target_side_geometry",
    "plan_local_crop",
    "refine_local_flowedit_jobs",
    "sha256_array",
    "sha256_file",
]
