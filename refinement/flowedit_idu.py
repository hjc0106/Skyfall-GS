"""Repository-owned FlowEdit adapter used by the LoD experiments.

The upstream FlowEdit checkout provides the sampler implementation.  This
adapter keeps experiment-specific concerns in the main repository:

* local model paths instead of a hard-coded Hub model;
* deterministic, per-image seeds without leaking RNG state into training;
* component offload before 2048px VAE decode; and
* optional deferred decode plus lightweight timing metadata.

Keeping this shim outside the submodule makes a normal clone reproducible
without requiring uncommitted edits inside ``submodules/FlowEdit``.
"""

from __future__ import annotations

import gc
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from diffusers import FluxPipeline, StableDiffusion3Pipeline
from PIL import Image
from PIL.Image import Image as PILImage
from tqdm import tqdm


_FLOWEDIT_ROOT = Path(__file__).resolve().parents[1] / "submodules" / "FlowEdit"
if str(_FLOWEDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLOWEDIT_ROOT))

from FlowEdit_utils import FlowEditFLUX, FlowEditSD3  # noqa: E402


DEFAULT_SOURCE_PROMPT = (
    "Satellite image of an urban area with modern and older buildings, roads, "
    "green spaces, and some distorted edges."
)
DEFAULT_TARGET_PROMPT = (
    "Clear satellite image of an urban area with sharp buildings, smooth "
    "edges, natural lighting, and well-defined textures."
)


@contextmanager
def seeded_inference(seed: int | None):
    """Isolate all RNG changes made by the reference FlowEdit sampler."""

    if seed is None:
        yield
        return

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _as_pil(image: np.ndarray | PILImage) -> PILImage:
    if isinstance(image, PILImage):
        return image
    value = np.asarray(image)
    if value.dtype != np.uint8:
        value = (value * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    if value.ndim == 2:
        return Image.fromarray(value, mode="L")
    if value.ndim == 3 and value.shape[2] in (3, 4):
        return Image.fromarray(value, mode="RGB" if value.shape[2] == 3 else "RGBA")
    raise ValueError(f"unsupported image shape {value.shape}")


class FlowEditRefineIDU:
    """Small compatibility wrapper around the upstream FlowEdit samplers."""

    def __init__(
        self,
        save_path: str,
        device: str = "cuda:0",
        model_type: str = "FLUX",
        model_path: str | None = None,
    ):
        self.device = device
        self.save_path = save_path
        self.model_type = model_type
        self.pipe = None
        self.last_run_timing: dict | None = None
        self._sampling_components = ("transformer", "text_encoder", "text_encoder_2")
        self._sampling_components_offloaded = False

        started = time.perf_counter()
        if model_type == "FLUX":
            pipe = FluxPipeline.from_pretrained(
                model_path or "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.float16,
            )
        elif model_type == "SD3":
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_path or "stabilityai/stable-diffusion-3-medium-diffusers",
                torch_dtype=torch.float16,
            )
        else:
            raise NotImplementedError(f"model type {model_type!r} is not supported")
        self.scheduler = pipe.scheduler
        self.pipe = pipe.to(self.device)
        self.initialization_sec = time.perf_counter() - started
        os.makedirs(save_path, exist_ok=True)

    def __del__(self):
        pipe = getattr(self, "pipe", None)
        if pipe is not None:
            try:
                pipe.to("cpu")
                del self.pipe
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _move_sampling_components(self, device: str) -> None:
        if self.pipe is None:
            return
        for name in self._sampling_components:
            component = getattr(self.pipe, name, None)
            if component is not None:
                component.to(device)
        self._sampling_components_offloaded = device == "cpu"
        if self._sampling_components_offloaded:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _restore_sampling_components(self) -> None:
        if self._sampling_components_offloaded:
            self._move_sampling_components(self.device)

    def _decode(self, latent: torch.Tensor) -> PILImage:
        with torch.autocast("cuda"), torch.inference_mode():
            decoded = self.pipe.vae.decode(latent, return_dict=False)[0]
        return self.pipe.image_processor.postprocess(decoded)[0]

    def _sample_latent(
        self,
        image: np.ndarray | PILImage,
        *,
        src_prompt: str,
        tar_prompt: str,
        T_steps: int,
        n_avg: int,
        src_guidance_scale: float,
        tar_guidance_scale: float,
        n_min: int,
        n_max: int,
    ) -> torch.Tensor:
        self._restore_sampling_components()
        pil = _as_pil(image)
        pil = pil.crop((0, 0, pil.width - pil.width % 16, pil.height - pil.height % 16))
        source = self.pipe.image_processor.preprocess(pil).to(self.device).half()
        with torch.autocast("cuda"), torch.inference_mode():
            encoded = self.pipe.vae.encode(source).latent_dist.mode()
        x0_src = (
            encoded - self.pipe.vae.config.shift_factor
        ) * self.pipe.vae.config.scaling_factor
        sampler = FlowEditSD3 if self.model_type == "SD3" else FlowEditFLUX
        target = sampler(
            self.pipe,
            self.scheduler,
            x0_src,
            src_prompt,
            tar_prompt,
            "",
            T_steps,
            n_avg,
            src_guidance_scale,
            tar_guidance_scale,
            n_min,
            n_max,
        )
        return (
            target / self.pipe.vae.config.scaling_factor
        ) + self.pipe.vae.config.shift_factor

    @torch.no_grad()
    def run(
        self,
        imgs: Sequence[np.ndarray | PILImage],
        src_prompt: str = DEFAULT_SOURCE_PROMPT,
        tar_prompt: str = DEFAULT_TARGET_PROMPT,
        T_steps: int = 28,
        n_avg: int = 1,
        src_guidance_scale: float = 1.5,
        tar_guidance_scale: float = 5.5,
        n_min: int = 0,
        n_max: int = 15,
        n_max_end: int | None = None,
        seed: int | None = None,
        seeds: Sequence[int] | None = None,
        defer_decode: bool = False,
    ) -> list[PILImage]:
        if seed is not None and seeds is not None:
            raise ValueError("pass either seed or seeds, not both")
        if seeds is not None and len(seeds) != len(imgs):
            raise ValueError(f"got {len(seeds)} seeds for {len(imgs)} images")
        image_seeds = list(seeds) if seeds is not None else [
            None if seed is None else int(seed) + index for index in range(len(imgs))
        ]

        started = time.perf_counter()
        latents: list[tuple[int, torch.Tensor]] = []
        outputs: list[PILImage] = []
        records = []
        for index, image in enumerate(tqdm(imgs, desc="Refining images using FlowEdit")):
            with seeded_inference(image_seeds[index]):
                current_n_max = (
                    random.randint(n_min, n_max_end)
                    if n_max_end is not None and n_max_end != -1
                    else n_max
                )
                image_started = time.perf_counter()
                latent = self._sample_latent(
                    image,
                    src_prompt=src_prompt,
                    tar_prompt=tar_prompt,
                    T_steps=T_steps,
                    n_avg=n_avg,
                    src_guidance_scale=src_guidance_scale,
                    tar_guidance_scale=tar_guidance_scale,
                    n_min=n_min,
                    n_max=current_n_max,
                )
            if defer_decode:
                latents.append((index, latent))
                records.append({"n_max": int(current_n_max), "sample_sec": time.perf_counter() - image_started})
                continue
            self._move_sampling_components("cpu")
            output = self._decode(latent)
            output.save(os.path.join(self.save_path, f"{index:05d}.png"))
            outputs.append(output)
            records.append({"n_max": int(current_n_max), "total_sec": time.perf_counter() - image_started})

        if defer_decode:
            self._move_sampling_components("cpu")
            for index, latent in latents:
                decode_started = time.perf_counter()
                output = self._decode(latent)
                output.save(os.path.join(self.save_path, f"{index:05d}.png"))
                outputs.append(output)
                records[index]["decode_sec"] = time.perf_counter() - decode_started

        self.last_run_timing = {
            "mode": "defer_decode" if defer_decode else "baseline",
            "count": len(outputs),
            "wall_sec": time.perf_counter() - started,
            "initialization_sec": float(self.initialization_sec),
            "images": records,
        }
        return outputs
