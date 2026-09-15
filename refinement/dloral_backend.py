"""Isolated DLoRAL dual-view backend.

The official generator accepts a string ``prompt`` and a two-frame clip
``[neighbor, target]``.  The decoded image corresponds to the **target**
frame (the second slot), matching ``src_idx = start + 1`` in
``src/test_DLoRAL.py``.  RAM/DAPE captioning is skipped: Qwen3-VL or a
fixed prompt is passed in directly.

Alignment modes:

- ``spynet``: official dual-view path (neighbor → target, native SpyNet).
- ``geometry``: same generator, but CFR consumes geometry correspondence.
  Full-image path warps inside CFR.  Tiled path warps neighbor VAE features
  globally, then feeds pre-aligned tiles with identity flow so SpyNet cannot
  warp a second time.  Invalid attention keys are masked on the logits.
- ``target_only``: same geometry path with neighbor contribution disabled.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from PIL import Image

from .base import Refiner
from .dloral_flows import (
    VAE_DOWNSAMPLE,
    correspondence_from_neighbor,
    correspondence_to_external_flows,
    latent_hw,
    latent_is_tiled,
    pack_external_flows_for_worker,
    pixel_flow_to_external_flows,
    prepared_image_size,
)
from .types import MultiViewInput, RefinementRequest, RefinementResult, to_jsonable

DLORAL_PINNED_COMMIT = "e8a5574124dd18d7d6ea71d6974bab6705f6e1f4"
DLORAL_WEIGHT_NOTE = (
    "Official improved checkpoint (2025-10-16): "
    "https://drive.google.com/file/d/1C2TLERta3a-PkoMpqQhHoM_S54pNEASO/view"
)
CLIP_TOKEN_LIMIT = 77
DEFAULT_REPO = Path(__file__).resolve().parents[1] / "submodules" / "DLoRAL"
_SD_WEIGHT_NAMES = (
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.fp16.safetensors",
    "diffusion_pytorch_model.bin",
    "diffusion_pytorch_model.fp16.bin",
)


def default_dloral_root() -> Path:
    return Path(os.environ.get("DLORAL_ROOT", DEFAULT_REPO)).expanduser().resolve()


def _dir_has_weight_file(directory: Path) -> bool:
    return any((directory / name).is_file() for name in _SD_WEIGHT_NAMES)


def assert_local_dloral_assets(
    *,
    repo_root: str | os.PathLike[str],
    sd_path: str | os.PathLike[str],
    ckpt_path: str | os.PathLike[str],
    spynet_path: str | os.PathLike[str],
) -> dict[str, str]:
    """Fail before launch if any required local file is missing."""

    repo = Path(repo_root).expanduser().resolve()
    sd = Path(sd_path).expanduser().resolve()
    ckpt = Path(ckpt_path).expanduser().resolve()
    spynet = Path(spynet_path).expanduser().resolve()
    missing: list[str] = []
    if not (repo / "src" / "DLoRAL_model.py").is_file():
        missing.append(f"DLoRAL repo (expected src/DLoRAL_model.py): {repo}")
    if not (sd / "model_index.json").is_file() and not (sd / "config.json").is_file():
        missing.append(f"SD 2.1 base (local, needs model_index.json): {sd}")
    elif not _dir_has_weight_file(sd / "unet") or not _dir_has_weight_file(sd / "vae"):
        missing.append(f"SD 2.1 unet/vae weight files (download incomplete): {sd}")
    if not ckpt.is_file():
        missing.append(f"DLoRAL checkpoint (.pkl): {ckpt}")
    if not spynet.is_file():
        missing.append(f"SpyNet weights (local, no URL): {spynet}")
    if missing:
        raise FileNotFoundError(
            "DLoRAL local assets are incomplete (no implicit download). Missing:\n- "
            + "\n- ".join(missing)
            + f"\nPinned commit {DLORAL_PINNED_COMMIT}. {DLORAL_WEIGHT_NOTE}"
        )
    return {
        "repo_root": str(repo),
        "sd_path": str(sd),
        "ckpt_path": str(ckpt),
        "spynet_path": str(spynet),
    }


def _neighbor_from_request(request: RefinementRequest) -> MultiViewInput | None:
    values = request.metadata.get("neighbor_views", ())
    best: MultiViewInput | None = None
    best_weight = -1.0
    for value in values:
        neighbor = value if isinstance(value, MultiViewInput) else MultiViewInput(**value)
        weight = float(neighbor.weight)
        if weight > best_weight:
            best = neighbor
            best_weight = weight
    return best


def _to_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    import numpy as np
    import torch

    tensor = image if isinstance(image, torch.Tensor) else torch.as_tensor(image)
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
        tensor = tensor[:3].detach().cpu().float()
        if float(tensor.amax().item()) > 1.0:
            tensor = tensor / 255.0
        array = (tensor.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        return Image.fromarray(array, mode="RGB")
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255.0
        array = array.clip(0, 255).round().astype("uint8")
    return Image.fromarray(array[..., :3], mode="RGB")


class DLoRALBackend(Refiner):
    """Launch official DLoRAL inference in an isolated Python process."""

    name = "dloral"

    def __init__(
        self,
        *,
        repo_root: str | os.PathLike[str] | None = None,
        sd_path: str | os.PathLike[str],
        ckpt_path: str | os.PathLike[str],
        spynet_path: str | os.PathLike[str],
        python: str | None = None,
        device: str = "cuda:0",
        stages: int = 1,
        process_size: int = 512,
        upscale: int = 1,
        align_method: str = "adain",
        vae_encoder_tiled_size: int = 4096,
        latent_tiled_size: int = 96,
        latent_tiled_overlap: int = 32,
        mixed_precision: str = "fp16",
        prompt_max_chars: int = 300,
        alignment: str = "spynet",
        max_roundtrip_error_px: float | None = None,
        dump_spatial_features: bool = False,
    ):
        if alignment not in ("spynet", "geometry", "target_only"):
            raise ValueError("dloral alignment must be 'spynet', 'geometry', or 'target_only'")
        self.assets = assert_local_dloral_assets(
            repo_root=repo_root or default_dloral_root(),
            sd_path=sd_path,
            ckpt_path=ckpt_path,
            spynet_path=spynet_path,
        )
        self.python = python or sys.executable
        self.device = device
        self.stages = int(stages)
        self.process_size = int(process_size)
        self.upscale = int(upscale)
        self.align_method = align_method
        self.alignment = alignment
        self.vae_encoder_tiled_size = int(vae_encoder_tiled_size)
        self.latent_tiled_size = int(latent_tiled_size)
        self.latent_tiled_overlap = int(latent_tiled_overlap)
        self.mixed_precision = mixed_precision
        self.prompt_max_chars = int(prompt_max_chars)
        self.max_roundtrip_error_px = (
            None if max_roundtrip_error_px is None else float(max_roundtrip_error_px)
        )
        self.dump_spatial_features = bool(dump_spatial_features)
        self._worker_session = None
        if self.upscale < 1:
            raise ValueError("dloral upscale must be >= 1")
        interpreter = shutil.which(self.python) or self.python
        if not Path(interpreter).exists():
            raise FileNotFoundError(f"DLoRAL python interpreter not found: {self.python}")
        self.python = interpreter

    def _propagation_name(self) -> str:
        if self.alignment == "geometry":
            return "geometry_external_flows"
        if self.alignment == "target_only":
            return "target_feature_fallback"
        return "native_spynet"

    def cache_config(self) -> Mapping[str, Any]:
        ckpt = Path(self.assets["ckpt_path"])
        spynet = Path(self.assets["spynet_path"])
        return {
            "backend": self.name,
            "pinned_commit": DLORAL_PINNED_COMMIT,
            "repo_root": self.assets["repo_root"],
            "sd_path": self.assets["sd_path"],
            "ckpt_path": str(ckpt),
            "ckpt_stat": {"size": ckpt.stat().st_size, "mtime_ns": ckpt.stat().st_mtime_ns},
            "spynet_path": str(spynet),
            "spynet_sha256": hashlib.sha256(spynet.read_bytes()).hexdigest() if spynet.stat().st_size < 64_000_000 else None,
            "python": str(Path(self.python).resolve()),
            "device": self.device,
            "stages": self.stages,
            "process_size": self.process_size,
            "upscale": self.upscale,
            "align_method": self.align_method,
            "alignment": self.alignment,
            "max_roundtrip_error_px": self.max_roundtrip_error_px,
            "dump_spatial_features": self.dump_spatial_features,
            "latent_tiled_size": self.latent_tiled_size,
            "vae_encoder_tiled_size": self.vae_encoder_tiled_size,
            "frame_order": ["neighbor", "target"],
            "output_frame": "target",
            "propagation": self._propagation_name(),
        }

    def _worker_env(self) -> dict[str, str]:
        """Environment for the isolated worker, shared by one-shot and session runs."""

        env = os.environ.copy()
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        project_root = str(Path(__file__).resolve().parents[1])
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            project_root if not existing_pythonpath else project_root + os.pathsep + existing_pythonpath
        )
        return env

    @contextmanager
    def session(self) -> Iterator["DLoRALBackend"]:
        """Reuse one worker process, and one loaded DLoRAL model, per ``refine`` block.

        The worker is launched lazily on the first ``refine`` and torn down when
        the block exits, so the model is freed before the next refinement stage.
        Outside the block every ``refine`` keeps spawning the standalone
        one-shot worker, so existing single-call behavior is unchanged.
        """

        from .worker_session import WorkerSession

        worker = Path(__file__).with_name("dloral_worker.py")
        project_root = Path(__file__).resolve().parents[1]
        session = WorkerSession(
            [self.python, str(worker), "--serve"],
            env=self._worker_env(),
            cwd=str(project_root),
        )
        with session:
            self._worker_session = session
            try:
                yield self
            finally:
                self._worker_session = None

    def refine(self, request: RefinementRequest) -> RefinementResult:
        neighbor = _neighbor_from_request(request)
        target = request.image.convert("RGB")
        neighbor_image = _to_pil(neighbor.image) if neighbor is not None else target.copy()
        fallback = None if neighbor is not None else "duplicated_target_no_neighbor"
        prompt = (request.prompt.target_prompt or "").strip()
        if not prompt:
            raise ValueError("DLoRAL requires a non-empty target_prompt (Qwen3-VL or --tar_prompt).")
        if len(prompt) > self.prompt_max_chars:
            prompt = prompt[: self.prompt_max_chars].rstrip()

        prepared_w, prepared_h = prepared_image_size(
            target.width, target.height, process_size=self.process_size, upscale=self.upscale
        )
        feat_h, feat_w = latent_hw(prepared_w, prepared_h)
        tiled = latent_is_tiled(feat_h, feat_w, self.latent_tiled_size)

        geometry_info: dict[str, Any] | None = None
        uses_geometry = self.alignment in ("geometry", "target_only")
        if uses_geometry:
            import torch

            neighbor_contribution = self.alignment != "target_only"
            if neighbor is not None and neighbor.pixel_flow is not None:
                geometry_info = correspondence_to_external_flows(
                    correspondence_from_neighbor(neighbor),
                    process_size=self.process_size,
                    upscale=self.upscale,
                    neighbor_contribution=neighbor_contribution,
                    max_roundtrip_error_px=self.max_roundtrip_error_px,
                )
            elif self.alignment == "target_only":
                geometry_info = pixel_flow_to_external_flows(
                    torch.zeros(target.height, target.width, 2),
                    image_size=(target.width, target.height),
                    process_size=self.process_size,
                    upscale=self.upscale,
                    neighbor_contribution=False,
                )
            else:
                raise ValueError(
                    "dloral alignment=geometry requires a neighbor with pixel_flow; "
                    "omit --skip_geometry and keep geometry_neighbor_count > 0."
                )

        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="dloral-") as directory:
            root = Path(directory)
            target.save(root / "target.png", compress_level=1)
            neighbor_image.save(root / "neighbor.png", compress_level=1)
            payload = {
                "repo_root": self.assets["repo_root"],
                "sd_path": self.assets["sd_path"],
                "ckpt_path": self.assets["ckpt_path"],
                "spynet_path": self.assets["spynet_path"],
                "device": self.device,
                "stages": self.stages,
                "process_size": self.process_size,
                "upscale": self.upscale,
                "align_method": self.align_method,
                "alignment": self.alignment,
                "vae_encoder_tiled_size": self.vae_encoder_tiled_size,
                "latent_tiled_size": self.latent_tiled_size,
                "latent_tiled_overlap": self.latent_tiled_overlap,
                "mixed_precision": self.mixed_precision,
                "prompt": prompt,
                "seed": int(request.metadata.get("seed", 0)),
                "clip_token_limit": CLIP_TOKEN_LIMIT,
                "frame_order": ["neighbor", "target"],
                "output_frame": "target",
                "fallback": fallback,
                "prepared_size": [prepared_w, prepared_h],
                "latent_size": [feat_w, feat_h],
                "tiled": tiled,
                "vae_downsample": VAE_DOWNSAMPLE,
                "dump_diagnostics": True,
                "dump_spatial_features": self.dump_spatial_features,
            }
            if geometry_info is not None:
                flow_paths = pack_external_flows_for_worker(geometry_info, root)
                payload["external_flows"] = flow_paths
                payload["geometry_coverage"] = geometry_info["coverage"]
                payload["reverse_source"] = geometry_info.get("reverse_source")
                payload["roundtrip"] = geometry_info.get("roundtrip")
                payload["image_roundtrip"] = geometry_info.get("image_roundtrip")
                payload["roundtrip_gate"] = geometry_info.get("roundtrip_gate")
            (root / "request.json").write_text(json.dumps(payload), encoding="utf-8")
            worker = Path(__file__).with_name("dloral_worker.py")
            project_root = Path(__file__).resolve().parents[1]
            if self._worker_session is None:
                env = self._worker_env()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                completed = subprocess.run(
                    [self.python, str(worker), str(root)],
                    check=False,
                    env=env,
                    cwd=str(project_root),
                    capture_output=True,
                    text=True,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        "DLoRAL worker failed.\n"
                        f"stdout:\n{completed.stdout[-4000:]}\n"
                        f"stderr:\n{completed.stderr[-4000:]}"
                    )
            else:
                # The long-lived worker reports failures through WorkerSession,
                # including the diagnostic tail of its stderr log.
                self._worker_session.run(root)
            result_path = root / "result.json"
            image_path = root / "result.png"
            if not result_path.is_file() or not image_path.is_file():
                raise RuntimeError("DLoRAL worker exited without result.png/result.json.")
            info = json.loads(result_path.read_text(encoding="utf-8"))
            with Image.open(image_path) as image:
                output = image.convert("RGB").copy()
            save_dir = request.metadata.get("backend_save_dir")
            if save_dir:
                persist = Path(save_dir)
                persist.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / "target.png", persist / "target.png")
                shutil.copy2(root / "neighbor.png", persist / "neighbor.png")
                shutil.copy2(image_path, persist / "result_native.png")
                shutil.copy2(result_path, persist / "result.json")
                persist_payload = dict(payload)
                if geometry_info is not None:
                    persist_payload["external_flows"] = pack_external_flows_for_worker(geometry_info, persist)
                    persist_payload["roundtrip"] = geometry_info.get("roundtrip")
                    persist_payload["image_roundtrip"] = geometry_info.get("image_roundtrip")
                    persist_payload["roundtrip_gate"] = geometry_info.get("roundtrip_gate")
                    persist_payload["reverse_source"] = geometry_info.get("reverse_source")
                for extra in ("coverage.png", "feature_diag.json"):
                    extra_path = root / extra
                    if extra_path.is_file():
                        shutil.copy2(extra_path, persist / extra)
                for extra_path in sorted(root.glob("spatial_*.npy")):
                    shutil.copy2(extra_path, persist / extra_path.name)
                (persist / "request.json").write_text(
                    json.dumps(to_jsonable(persist_payload), indent=2), encoding="utf-8"
                )

        requested = request.expected_output_size
        raw_size = output.size
        if output.size != requested:
            output = output.resize(requested, Image.Resampling.LANCZOS)
        elapsed = time.perf_counter() - started
        metadata = {
            **self.cache_config(),
            "fallback": fallback,
            "feature_propagation": True,
            "propagation": self._propagation_name(),
            "alignment": self.alignment,
            "prompt_submitted": prompt,
            "prompt_used": info.get("prompt_used", prompt),
            "prompt_truncated": bool(info.get("prompt_truncated", False)),
            "frame_order": ["neighbor", "target"],
            "output_frame": "target",
            "neighbor_name": None if neighbor is None else neighbor.name,
            "prepared_size": [prepared_w, prepared_h],
            "latent_size": [feat_w, feat_h],
            "tiled": bool(info.get("tiled", tiled)),
            "raw_output_size": list(raw_size),
            "requested_size": list(requested),
            "elapsed_sec": elapsed,
            "worker_elapsed_sec": info.get("elapsed_sec"),
            "worker_gpu_elapsed_sec": info.get("gpu_elapsed_sec"),
            "worker_preprocess_sec": info.get("preprocess_sec"),
            "worker_total_sec": info.get("total_sec"),
            "worker_init_sec": info.get("init_elapsed_sec"),
            "worker_request_index": info.get("request_index"),
            "worker_session": self._worker_session is not None,
            "seed": info.get("seed"),
            "seed_source": info.get("seed_source"),
            "init_cuda_rng_consumed": info.get("init_cuda_rng_consumed"),
            "init_cpu_rng_consumed": info.get("init_cpu_rng_consumed"),
            "peak_cuda_memory_bytes": info.get("peak_cuda_memory_bytes"),
            "init_peak_cuda_memory_bytes": info.get("init_peak_cuda_memory_bytes"),
            "geometry_coverage": None if geometry_info is None else geometry_info["coverage"],
            "reverse_source": None if geometry_info is None else geometry_info.get("reverse_source"),
            "roundtrip": None if geometry_info is None else geometry_info.get("roundtrip"),
            "image_roundtrip": None if geometry_info is None else geometry_info.get("image_roundtrip"),
            "roundtrip_gate": None if geometry_info is None else geometry_info.get("roundtrip_gate"),
            "worker": info,
        }
        return RefinementResult(image=output, backend=self.name, metadata=to_jsonable(dict(metadata)))


__all__ = [
    "CLIP_TOKEN_LIMIT",
    "DLORAL_PINNED_COMMIT",
    "DLoRALBackend",
    "assert_local_dloral_assets",
    "default_dloral_root",
]
