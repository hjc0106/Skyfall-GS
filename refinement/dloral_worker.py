#!/usr/bin/env python3
"""Isolated DLoRAL inference worker. Run only in the DLoRAL environment.

The worker loads one ``Generator_eval`` and can serve either a single request
directory (the standalone path used outside a session) or a stdin stream of
request directories (``--serve``).  Only the weights, tokenizer and immutable
load configuration survive between requests; geometry hooks, feature dumps,
tile iterators and RNG state are rebuilt per request.
"""

from __future__ import annotations

import os
import sys

# ``python refinement/dloral_worker.py`` puts this directory on sys.path[0] and
# shadows the stdlib ``types`` module with ``refinement/types.py``.  Fix that
# before importing anything else.
_WORKER_DIR = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_WORKER_DIR)


def _unshadow_stdlib(path_entries):
    cleaned = []
    for entry in path_entries:
        candidate = os.getcwd() if entry == "" else entry
        try:
            if os.path.realpath(candidate) == _WORKER_DIR:
                continue
        except OSError:
            pass
        cleaned.append(entry)
    if _PROJECT_ROOT not in cleaned:
        cleaned.insert(0, _PROJECT_ROOT)
    return cleaned


sys.path[:] = _unshadow_stdlib(sys.path)

import json
import time
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Request fields that determine the loaded model.  A session may only change
# prompt/seed/alignment/geometry between requests, never these.
_LOAD_KEYS = (
    "repo_root",
    "sd_path",
    "ckpt_path",
    "spynet_path",
    "device",
    "stages",
    "vae_encoder_tiled_size",
    "vae_decoder_tiled_size",
    "latent_tiled_size",
    "latent_tiled_overlap",
    "mixed_precision",
)


def _load_signature(request: dict) -> tuple:
    return tuple((key, request.get(key)) for key in _LOAD_KEYS)


def _force_offline_from_pretrained() -> None:
    import diffusers
    import transformers

    def wrap(fn):
        def wrapped(*args, **kwargs):
            kwargs["local_files_only"] = True
            return fn(*args, **kwargs)

        return wrapped

    transformers.AutoTokenizer.from_pretrained = wrap(transformers.AutoTokenizer.from_pretrained)
    transformers.CLIPTextModel.from_pretrained = wrap(transformers.CLIPTextModel.from_pretrained)
    diffusers.AutoencoderKL.from_pretrained = wrap(diffusers.AutoencoderKL.from_pretrained)
    diffusers.UNet2DConditionModel.from_pretrained = wrap(diffusers.UNet2DConditionModel.from_pretrained)
    diffusers.DDPMScheduler.from_pretrained = wrap(diffusers.DDPMScheduler.from_pretrained)


def _prepare_pair(target, neighbor, *, process_size: int, upscale: int):
    from PIL import Image

    def _one(image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        if width < process_size // upscale or height < process_size // upscale:
            scale = (process_size // upscale) / min(width, height)
            image = image.resize((int(scale * width), int(scale * height)), Image.Resampling.LANCZOS)
        if upscale != 1:
            image = image.resize((image.size[0] * upscale, image.size[1] * upscale), Image.Resampling.LANCZOS)
        new_width = image.width - image.width % 8
        new_height = image.height - image.height % 8
        if (new_width, new_height) != image.size:
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image

    neighbor_p = _one(neighbor)
    target_p = _one(target)
    if neighbor_p.size != target_p.size:
        neighbor_p = neighbor_p.resize(target_p.size, Image.Resampling.LANCZOS)
    return target_p, neighbor_p


def _uncertainty_map(frames_gray, device):
    import torch

    stacked = torch.stack(frames_gray, dim=0)
    ambi = stacked.var(dim=0)
    threshold = ambi.mean().item()
    mask = torch.where(ambi >= threshold, torch.ones_like(ambi), torch.zeros_like(ambi))
    return mask.unsqueeze(0).to(device)


def _bind_cuda_device(requested: str) -> str:
    """Pin the worker to one GPU before importing torch.

    ``cuda:N`` is a physical index unless the parent already set
    ``CUDA_VISIBLE_DEVICES``, in which case N is a local index into that list.
    """

    if not requested.startswith("cuda:"):
        return requested
    index = requested.split(":", 1)[1]
    already = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if already:
        visible = [item.strip() for item in already.split(",") if item.strip()]
        local = int(index)
        if len(visible) == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible[0]
        elif 0 <= local < len(visible):
            os.environ["CUDA_VISIBLE_DEVICES"] = visible[local]
        else:
            raise ValueError(f"device {requested} is outside CUDA_VISIBLE_DEVICES={already}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = index
    return "cuda:0"


class _DLoRALWorker:
    """Reusable DLoRAL context: one loaded model, one request directory at a time.

    Seeded-sample semantics.  ``Generator_eval`` construction (module creation,
    ``nn.init`` resets, checkpoint copy, ``_init_tiled_vae``, ``.to``) draws only
    from the CPU generator; it never touches the CUDA generator (the training-only
    ``cal_csd`` is the only ``torch.randint`` site and is unreachable from
    ``Generator_eval.forward``).  The eval forward consumes randomness exactly
    once, at ``latent_dist.sample()``, which the installed diffusers routes to the
    VAE's parameter device — CUDA here (``diffusers/models/autoencoders/vae.py``,
    ``DiagonalGaussianDistribution.sample``).  Therefore reseeding CUDA per
    request reproduces the standalone worker's sampled latents exactly; the CPU
    delta flag is recorded only to make the pixel-irrelevant build-time CPU
    consumption visible.

    The model is built under the first request's seed; the post-init RNG states
    are snapshotted, so a request that reuses the load seed restores them
    bit-for-bit instead of reseeding.
    """

    def __init__(self):
        self._model = None
        self._torch = None
        self._transforms = None
        self._adain_color_fix = None
        self._wavelet_color_fix = None
        self._device = None
        self._signature = None
        self._init_seed = None
        self._post_init_cpu_state = None
        self._post_init_cuda_states = None
        self._pristine_cfr_forward = None
        self._pristine_attn_forward = None
        self._pristine_vae_encode = None
        self._request_index = 0
        self._init_cuda_rng_consumed = None
        self.init_elapsed_sec = None
        self.init_peak_cuda_memory_bytes = None
        self.init_cpu_rng_consumed = None

    def _build(self, request: dict) -> None:
        self._device = _bind_cuda_device(str(request["device"]))
        repo_root = Path(request["repo_root"]).resolve()
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        _force_offline_from_pretrained()
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

        import torch
        from torchvision import transforms

        from src.cross_frame_retrieval import cfr_main
        from src.DLoRAL_model import Generator_eval
        from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

        try:
            import xformers  # noqa: F401
        except ImportError:
            from diffusers import UNet2DConditionModel

            UNet2DConditionModel.enable_xformers_memory_efficient_attention = lambda self: None

        spynet = str(Path(request["spynet_path"]).resolve())
        original_cfr_init = cfr_main.CFR_model.__init__

        def patched_cfr_init(self, *args, **kwargs):
            kwargs["spynet_pretrained"] = spynet
            return original_cfr_init(self, *args, **kwargs)

        cfr_main.CFR_model.__init__ = patched_cfr_init

        args = Namespace(
            pretrained_path=request["ckpt_path"],
            pretrained_model_path=request["sd_path"],
            pretrained_model_name_or_path=request["sd_path"],
            vae_encoder_tiled_size=int(request["vae_encoder_tiled_size"]),
            vae_decoder_tiled_size=int(request.get("vae_decoder_tiled_size", 224)),
            latent_tiled_size=int(request["latent_tiled_size"]),
            latent_tiled_overlap=int(request["latent_tiled_overlap"]),
            merge_and_unload_lora=False,
            load_cfr=True,
            stages=int(request["stages"]),
        )

        seed = int(request.get("seed", 0))
        cuda = torch.cuda.is_available()
        torch.manual_seed(seed)
        cpu_before = torch.get_rng_state()
        cuda_before = None
        if cuda:
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
            cuda_before = torch.cuda.get_rng_state_all()

        started = time.perf_counter()
        try:
            model = Generator_eval(args)
        finally:
            # The pin only has to exist while the generator builds its CFR net.
            cfr_main.CFR_model.__init__ = original_cfr_init
        model.set_eval()
        model.unet.set_adapter(
            [
                "default_encoder_quality",
                "default_decoder_quality",
                "default_others_quality",
                "default_encoder_consistency",
                "default_decoder_consistency",
                "default_others_consistency",
            ]
        )

        weight_dtype = torch.float16 if request.get("mixed_precision", "fp16") == "fp16" else torch.float32
        model.vae = model.vae.to(dtype=weight_dtype)
        model.unet = model.unet.to(dtype=weight_dtype)
        model.cfr_main_net = model.cfr_main_net.to(dtype=weight_dtype)
        if cuda:
            torch.cuda.synchronize()
        self.init_elapsed_sec = time.perf_counter() - started
        self.init_peak_cuda_memory_bytes = int(torch.cuda.max_memory_allocated()) if cuda else None
        self._init_cuda_rng_consumed = bool(
            cuda_before is not None
            and any(
                not torch.equal(before, after)
                for before, after in zip(cuda_before, torch.cuda.get_rng_state_all())
            )
        )
        self._post_init_cpu_state = torch.get_rng_state()
        self._post_init_cuda_states = torch.cuda.get_rng_state_all() if cuda else None
        self.init_cpu_rng_consumed = not torch.equal(cpu_before, self._post_init_cpu_state)
        self._init_seed = seed
        self._signature = _load_signature(request)
        self._torch = torch
        self._transforms = transforms
        self._adain_color_fix = adain_color_fix
        self._wavelet_color_fix = wavelet_color_fix
        self._model = model
        self._snapshot_hooks(model)

    def _snapshot_hooks(self, model) -> None:
        """Record the unpatched geometry entry points of ``model``."""

        cfr = model.cfr_main_net
        self._pristine_cfr_forward = cfr.forward
        self._pristine_attn_forward = cfr.cross_attn_module.forward
        self._pristine_vae_encode = model.vae.encode

    def _reset_geometry_hooks(self) -> None:
        # ``wrap_cfr_geometry_alignment`` chains off ``cfr.forward`` and installs
        # a per-request attention lambda, while ``wrap_vae_neighbor_alignment``
        # chains off ``model.vae.encode``.  Rebase them every request so a stale
        # closure from a previous sample (tiled or not) can never be reused.
        cfr = self._model.cfr_main_net
        cfr.forward = self._pristine_cfr_forward
        cfr.cross_attn_module.forward = self._pristine_attn_forward
        self._model.vae.encode = self._pristine_vae_encode

    def run(self, root: Path, request: dict) -> None:
        torch = self._torch
        model = self._model
        transforms = self._transforms
        request_started = time.perf_counter()

        alignment = str(request.get("alignment", "spynet"))
        if alignment not in ("spynet", "geometry", "target_only"):
            raise ValueError(f"Unknown DLoRAL alignment: {alignment}")

        seed = int(request.get("seed", 0))
        if seed == self._init_seed and self._post_init_cpu_state is not None:
            # Identical to a fresh process: the standalone worker reached its
            # forward pass from exactly this post-init state.
            torch.set_rng_state(self._post_init_cpu_state)
            if self._post_init_cuda_states is not None:
                torch.cuda.set_rng_state_all(self._post_init_cuda_states)
            seed_source = "post_init"
        else:
            # The standalone worker seeded both streams and then built the model;
            # the build advanced only the CPU stream, so the CUDA generator it
            # sampled from was exactly a fresh seed.  Reseed both here.
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            seed_source = "reseeded"
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._reset_geometry_hooks()

        weight_dtype = torch.float16 if request.get("mixed_precision", "fp16") == "fp16" else torch.float32
        tokenizer = model.tokenizer
        raw_prompt = str(request["prompt"])
        encoded = tokenizer(
            raw_prompt,
            max_length=int(request.get("clip_token_limit", 77)),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        prompt_used = tokenizer.decode(encoded.input_ids[0], skip_special_tokens=True)
        truncated = prompt_used.strip() != raw_prompt.strip()

        from PIL import Image

        target, neighbor = _prepare_pair(
            Image.open(root / "target.png"),
            Image.open(root / "neighbor.png"),
            process_size=int(request["process_size"]),
            upscale=int(request["upscale"]),
        )
        to_tensor = transforms.ToTensor()
        frames = [to_tensor(neighbor), to_tensor(target)]
        grays = [
            torch.nn.functional.interpolate(
                transforms.functional.to_tensor(image.convert("L")).unsqueeze(0),
                scale_factor=0.125,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            for image in (neighbor, target)
        ]
        c_t = torch.stack(frames, dim=0).unsqueeze(0).to(device=self._device, dtype=weight_dtype) * 2 - 1
        uncertainty = _uncertainty_map(grays, self._device)
        feat_h, feat_w = int(c_t.shape[-2] // 8), int(c_t.shape[-1] // 8)
        tile_size = int(request["latent_tiled_size"])
        tiled = feat_h * feat_w > tile_size * tile_size

        dump: dict = {}
        if alignment in ("geometry", "target_only"):
            import numpy as np

            skyfall_root = Path(__file__).resolve().parents[1]
            if str(skyfall_root) not in sys.path:
                sys.path.insert(0, str(skyfall_root))
            from refinement.dloral_flows import wrap_cfr_geometry_alignment, wrap_vae_neighbor_alignment

            flows_forward = torch.from_numpy(np.load(root / "flows_forward.npy"))
            flows_backward = torch.from_numpy(np.load(root / "flows_backward.npy"))
            valid_mask = torch.from_numpy(np.load(root / "flow_valid.npy"))
            if tuple(valid_mask.shape[-2:]) != (feat_h, feat_w):
                raise RuntimeError(
                    f"flow valid_mask spatial {tuple(valid_mask.shape[-2:])} != latent {(feat_h, feat_w)}"
                )
            if request.get("dump_spatial_features"):
                dump["dump_spatial"] = True
                dump["spatial_hw"] = (feat_h, feat_w)
            if tiled:
                wrap_vae_neighbor_alignment(model, flows_forward, valid_mask)
                wrap_cfr_geometry_alignment(
                    model.cfr_main_net,
                    flows_forward,
                    flows_backward,
                    valid_mask,
                    dump=dump,
                    prealigned=True,
                    latent_hw_full=(feat_h, feat_w),
                    tile_size=tile_size,
                    tile_overlap=int(request.get("latent_tiled_overlap", 32)),
                )
            else:
                if tuple(flows_forward.shape[-2:]) != (feat_h, feat_w):
                    raise RuntimeError(
                        f"external_flows spatial {tuple(flows_forward.shape[-2:])} "
                        f"!= latent {(feat_h, feat_w)}"
                    )
                wrap_cfr_geometry_alignment(
                    model.cfr_main_net,
                    flows_forward,
                    flows_backward,
                    valid_mask,
                    dump=dump,
                    prealigned=False,
                )

        if torch.cuda.is_available():
            # Flush preparation kernels so the timed region covers forward only.
            torch.cuda.synchronize()
        preprocess_sec = time.perf_counter() - request_started
        started = time.perf_counter()
        with torch.no_grad():
            output_image, _, _, _, _ = model(
                stages=int(request["stages"]),
                c_t=c_t,
                uncertainty_map=uncertainty.unsqueeze(0),
                prompt=prompt_used,
                weight_dtype=weight_dtype,
            )
        elapsed = time.perf_counter() - started
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gpu_elapsed_sec = time.perf_counter() - started
        frame = (output_image[0].detach().float().cpu() * 0.5 + 0.5).clamp(0.0, 1.0)
        output = transforms.ToPILImage()(frame)
        align = request.get("align_method", "adain")
        if align == "adain":
            output = self._adain_color_fix(target=output, source=target)
        elif align == "wavelet":
            output = self._wavelet_color_fix(target=output, source=target)
        output.save(root / "result.png", compress_level=1)

        peak = None
        if torch.cuda.is_available():
            peak = int(torch.cuda.max_memory_allocated())
        scalar_dump: dict = {}
        if dump:
            from refinement.dloral_flows import finalize_spatial_maps, jsonable_feature_dump

            scalar_dump = jsonable_feature_dump(dump)
            (root / "feature_diag.json").write_text(
                json.dumps(scalar_dump, default=float),
                encoding="utf-8",
            )
            if dump.get("dump_spatial"):
                import numpy as np

                for name, tensor in finalize_spatial_maps(dump).items():
                    np.save(root / f"spatial_{name}.npy", tensor.numpy())
            if alignment in ("geometry", "target_only"):
                import numpy as np
                from PIL import Image as PILImage

                coverage = np.load(root / "flow_valid.npy")
                coverage_img = (np.asarray(coverage).astype("float32") * 255.0).clip(0, 255).astype("uint8")
                PILImage.fromarray(coverage_img, mode="L").resize(target.size, PILImage.Resampling.NEAREST).save(
                    root / "coverage.png"
                )
        total_sec = time.perf_counter() - request_started
        request_index = self._request_index
        self._request_index += 1
        (root / "result.json").write_text(
            json.dumps(
                {
                    "prompt_used": prompt_used,
                    "prompt_truncated": truncated,
                    "elapsed_sec": elapsed,
                    "gpu_elapsed_sec": gpu_elapsed_sec,
                    "preprocess_sec": preprocess_sec,
                    "total_sec": total_sec,
                    "init_elapsed_sec": self.init_elapsed_sec,
                    "peak_cuda_memory_bytes": peak,
                    "init_peak_cuda_memory_bytes": self.init_peak_cuda_memory_bytes,
                    "request_index": request_index,
                    "seed": seed,
                    "seed_source": seed_source,
                    "init_cuda_rng_consumed": self._init_cuda_rng_consumed,
                    "init_cpu_rng_consumed": self.init_cpu_rng_consumed,
                    "output_size": list(output.size),
                    "input_size": list(target.size),
                    "neighbor_size": list(neighbor.size),
                    "latent_size": [feat_w, feat_h],
                    "tiled": tiled,
                    "alignment": alignment,
                    "frame_order": request.get("frame_order"),
                    "output_frame": request.get("output_frame"),
                    "propagation": (
                        "geometry_external_flows"
                        if alignment == "geometry"
                        else "target_feature_fallback"
                        if alignment == "target_only"
                        else "native_spynet"
                    ),
                    "feature_diag": scalar_dump if dump else {},
                    "offline": True,
                },
                default=float,
            ),
            encoding="utf-8",
        )

    def process(self, directory) -> None:
        root = Path(directory)
        request = json.loads((root / "request.json").read_text(encoding="utf-8"))
        if self._model is None:
            self._build(request)
        else:
            signature = _load_signature(request)
            if signature != self._signature:
                raise RuntimeError(
                    "DLoRAL worker model config changed during one session: "
                    f"{self._signature} -> {signature}"
                )
        self.run(root, request)

    def close(self) -> None:
        if self._model is None:
            return
        self._model = None
        self._post_init_cpu_state = None
        self._post_init_cuda_states = None
        self._pristine_cfr_forward = None
        self._pristine_attn_forward = None
        self._pristine_vae_encode = None
        self._adain_color_fix = None
        self._wavelet_color_fix = None
        self._transforms = None
        self._torch = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def worker(directory: str) -> None:
    """Standalone worker: run one request directory and exit."""
    session = _DLoRALWorker()
    try:
        session.process(directory)
    finally:
        session.close()


def _serve() -> None:
    """Run the shared stdin request loop, reusing one loaded model."""
    try:
        from .worker_session import serve
    except ImportError:
        # ``python refinement/dloral_worker.py --serve`` has no package context.
        project_root = str(Path(__file__).resolve().parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from refinement.worker_session import serve

    session = _DLoRALWorker()
    try:
        serve(session.process)
    finally:
        session.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        _serve()
    else:
        worker(sys.argv[1])
