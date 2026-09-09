"""Single-view refinement backends.

FlowEdit and the optional diffusers SR pipeline are imported lazily so that
the offline/reuse modes remain usable without loading large generative models.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from .base import Refiner
from .types import RefinementRequest, RefinementResult

DEFAULT_SOURCE_PROMPT = (
    "Satellite image of an urban area with modern and older buildings, roads, green spaces, "
    "and a unique white angular structure. Some areas appear distorted, with blurring and "
    "warping artifacts near edges and trees."
)
DEFAULT_TARGET_PROMPT = (
    "Clear satellite image of an urban area with sharp buildings, smooth edges, and no "
    "distortions. Roads, green spaces, and the white angular structure are crisp, with "
    "natural lighting and well-defined textures."
)


def _expected_size(request: RefinementRequest) -> tuple[int, int]:
    return request.expected_output_size


class ReuseImageBackend(Refiner):
    """Use a previously generated image without invoking a model."""

    name = "reuse"

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path).expanduser().resolve()

    def _resolve_path(self, request: RefinementRequest) -> Path:
        if self.path.is_file():
            return self.path
        if self.path.is_dir():
            factor = f"{request.zoom_factor:g}"
            candidates = (
                self.path / f"zoom_{factor}x" / "refined.png",
                self.path / f"zoom_{factor}x.png",
                self.path / f"{factor}x.png",
                self.path / "refined.png",
            )
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
        raise FileNotFoundError(
            f"No reusable refined image found at {self.path} for zoom {request.zoom_factor:g}x."
        )

    def refine(self, request: RefinementRequest) -> RefinementResult:
        path = self._resolve_path(request)
        with Image.open(path) as image:
            output = image.convert("RGB").copy()
        return RefinementResult(
            image=output,
            backend=self.name,
            metadata={"source_path": str(path), "requested_size": _expected_size(request)},
        )


class UnsharpBackend(Refiner):
    """Deterministic, model-free backend for plumbing and smoke tests."""

    name = "unsharp"

    def __init__(self, radius: float = 2.0, percent: int = 140, threshold: int = 2, contrast: float = 1.08):
        self.radius = radius
        self.percent = percent
        self.threshold = threshold
        self.contrast = contrast

    def refine(self, request: RefinementRequest) -> RefinementResult:
        image = request.image.convert("RGB")
        requested_size = _expected_size(request)
        if image.size != requested_size:
            image = image.resize(requested_size, Image.Resampling.LANCZOS)
        image = image.filter(
            ImageFilter.UnsharpMask(
                radius=self.radius,
                percent=self.percent,
                threshold=self.threshold,
            )
        )
        image = ImageEnhance.Contrast(image).enhance(self.contrast)
        return RefinementResult(
            image=image,
            backend=self.name,
            metadata={
                "deterministic": True,
                "radius": self.radius,
                "percent": self.percent,
                "threshold": self.threshold,
                "contrast": self.contrast,
                "requested_size": requested_size,
            },
        )


class FlowEditBackend(Refiner):
    """Adapter for the repository's existing FlowEdit implementation."""

    name = "flowedit"

    def __init__(
        self,
        *,
        model_type: str = "FLUX",
        model_path: str | None = None,
        device: str = "cuda:0",
        n_min: int = 4,
        n_max: int = 10,
        n_avg: int = 1,
    ):
        self.model_type = model_type
        self.model_path = model_path or None
        self.device = device
        self.n_min = n_min
        self.n_max = n_max
        self.n_avg = n_avg

    def refine(self, request: RefinementRequest) -> RefinementResult:
        # Keep this import lazy: ``reuse`` and ``unsharp`` are intentionally
        # runnable on machines without diffusers or the FlowEdit submodule.
        from submodules.FlowEdit.idu_refine import FlowEditRefineIDU

        save_dir = request.metadata.get("backend_save_dir")
        if not save_dir:
            save_dir = os.path.join(os.getcwd(), "flowedit")
        os.makedirs(save_dir, exist_ok=True)

        refine_pipe = FlowEditRefineIDU(
            save_path=save_dir,
            device=self.device,
            model_type=self.model_type,
            model_path=self.model_path,
        )
        image_np = np.asarray(request.image.convert("RGB"), dtype=np.float32) / 255.0
        refined = refine_pipe.run(
            [image_np],
            src_prompt=request.prompt.source_prompt or DEFAULT_SOURCE_PROMPT,
            tar_prompt=request.prompt.target_prompt or DEFAULT_TARGET_PROMPT,
            n_min=self.n_min,
            n_max=self.n_max,
            n_avg=self.n_avg,
        )[0]
        del refine_pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return RefinementResult(
            image=refined.convert("RGB"),
            backend=self.name,
            metadata={
                "model_type": self.model_type,
                "model_path": self.model_path,
                "device": self.device,
                "n_min": self.n_min,
                "n_max": self.n_max,
                "n_avg": self.n_avg,
                "save_dir": str(save_dir),
            },
        )


class TextConditionalSRBackend(Refiner):
    """Optional text-conditioned SR backend based on diffusers.

    The default model is the public Stable Diffusion x4 upscaler.  A different
    compatible pipeline can be supplied by replacing this adapter; DLoRAL and
    other multi-view models should be integrated only after their feature and
    text interfaces have been verified.
    """

    name = "sr"

    def __init__(
        self,
        *,
        backend_name: str = "sr",
        model_id: str = "stabilityai/stable-diffusion-x4-upscaler",
        model_path: str | None = None,
        device: str = "cuda",
        dtype: str = "float16",
        num_inference_steps: int = 20,
        guidance_scale: float = 9.0,
        local_files_only: bool = False,
        model_scale: float = 4.0,
        input_max_size: int = 1024,
    ):
        self.model_id = model_id
        self.model_path = model_path or None
        self.device = device
        self.dtype = dtype
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.local_files_only = local_files_only
        if model_scale <= 0.0:
            raise ValueError(f"SR model scale must be positive, got {model_scale}")
        if input_max_size < 64:
            raise ValueError(f"SR input_max_size must be at least 64, got {input_max_size}")
        self.name = str(backend_name)
        self.model_scale = float(model_scale)
        self.input_max_size = int(input_max_size)
        self._pipe: Any = None

    def release_memory(self) -> None:
        pipe = self._pipe
        self._pipe = None
        if pipe is not None:
            del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _round_model_dimension(value: float) -> int:
        # VAE-backed diffusion pipelines are most reliable on dimensions that
        # are multiples of 64. The final resize restores the exact requested
        # supervision resolution and aspect ratio.
        return max(64, int(round(value / 64.0)) * 64)

    def _model_input_size(self, request: RefinementRequest) -> tuple[int, int]:
        target_width, target_height = _expected_size(request)
        width = self._round_model_dimension(target_width / self.model_scale)
        height = self._round_model_dimension(target_height / self.model_scale)
        longest = max(width, height)
        if longest > self.input_max_size:
            ratio = self.input_max_size / float(longest)
            width = self._round_model_dimension(width * ratio)
            height = self._round_model_dimension(height * ratio)
        return width, height

    def _torch_dtype(self) -> torch.dtype:
        try:
            value = getattr(torch, self.dtype)
        except AttributeError as exc:
            raise ValueError(f"Unsupported torch dtype for SR backend: {self.dtype}") from exc
        if not isinstance(value, torch.dtype):
            raise ValueError(f"Unsupported torch dtype for SR backend: {self.dtype}")
        return value

    def _load_pipeline(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("SR backend requested a CUDA device, but CUDA is unavailable.")
        try:
            from diffusers import StableDiffusionUpscalePipeline
        except ImportError as exc:
            raise RuntimeError(
                "The SR backend requires diffusers and a compatible StableDiffusionUpscalePipeline."
            ) from exc

        model_source = self.model_path or self.model_id
        self._pipe = StableDiffusionUpscalePipeline.from_pretrained(
            model_source,
            torch_dtype=self._torch_dtype(),
            local_files_only=self.local_files_only,
        ).to(self.device)
        return self._pipe

    def refine(self, request: RefinementRequest) -> RefinementResult:
        pipe = self._load_pipeline()
        prompt = request.prompt.target_prompt or DEFAULT_TARGET_PROMPT
        model_input_size = self._model_input_size(request)
        model_input = request.image.convert("RGB")
        if model_input.size != model_input_size:
            model_input = model_input.resize(model_input_size, Image.Resampling.LANCZOS)
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "image": model_input,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }
        seed = request.metadata.get("seed")
        if seed is not None:
            kwargs["generator"] = torch.Generator(device=self.device).manual_seed(int(seed))
        output = pipe(**kwargs).images[0].convert("RGB")
        requested_size = _expected_size(request)
        raw_output_size = output.size
        if output.size != requested_size:
            output = output.resize(requested_size, Image.Resampling.LANCZOS)
        return RefinementResult(
            image=output,
            backend=self.name,
            metadata={
                "model_id": self.model_id,
                "model_path": self.model_path,
                "device": self.device,
                "dtype": self.dtype,
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "prompt": prompt,
                "model_scale": self.model_scale,
                "model_input_size": model_input_size,
                "raw_output_size": raw_output_size,
                "requested_size": requested_size,
            },
        )


__all__ = [
    "DEFAULT_SOURCE_PROMPT",
    "DEFAULT_TARGET_PROMPT",
    "FlowEditBackend",
    "ReuseImageBackend",
    "TextConditionalSRBackend",
    "UnsharpBackend",
]
