#!/usr/bin/env python3
"""Isolated DLoRAL inference worker. Run only in the DLoRAL environment."""

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


def worker(directory: str) -> None:
    root = Path(directory)
    request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    device = _bind_cuda_device(str(request["device"]))

    repo_root = Path(request["repo_root"]).resolve()
    sys.path.insert(0, str(repo_root))
    _force_offline_from_pretrained()

    import torch
    from PIL import Image
    from torchvision import transforms

    from src.cross_frame_retrieval import cfr_main
    from src.DLoRAL_model import Generator_eval
    from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

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

    torch.manual_seed(int(request.get("seed", 0)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(request.get("seed", 0)))
        torch.cuda.reset_peak_memory_stats()

    model = Generator_eval(args)
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

    alignment = str(request.get("alignment", "spynet"))
    if alignment not in ("spynet", "geometry", "target_only"):
        raise ValueError(f"Unknown DLoRAL alignment: {alignment}")

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
    c_t = torch.stack(frames, dim=0).unsqueeze(0).to(device=device, dtype=weight_dtype) * 2 - 1
    uncertainty = _uncertainty_map(grays, device)
    feat_h, feat_w = int(c_t.shape[-2] // 8), int(c_t.shape[-1] // 8)
    tile_size = int(request["latent_tiled_size"])
    tiled = feat_h * feat_w > tile_size * tile_size

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
        dump: dict = {}
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
    else:
        dump = {}

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
    frame = (output_image[0].detach().float().cpu() * 0.5 + 0.5).clamp(0.0, 1.0)
    output = transforms.ToPILImage()(frame)
    align = request.get("align_method", "adain")
    if align == "adain":
        output = adain_color_fix(target=output, source=target)
    elif align == "wavelet":
        output = wavelet_color_fix(target=output, source=target)
    output.save(root / "result.png")

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
    (root / "result.json").write_text(
        json.dumps(
            {
                "prompt_used": prompt_used,
                "prompt_truncated": truncated,
                "elapsed_sec": elapsed,
                "peak_cuda_memory_bytes": peak,
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


if __name__ == "__main__":
    worker(sys.argv[1])
