"""Shared runtime helpers for the self-contained JAX_068 LoD runners."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from arguments import ModelParams
from lod.correspondence import rade_pkg_camera_z, warp_source_into_destination
from lod.freeze import appearance_freeze_report
from lod.importer import extend_base_stage, import_skyfall_l0, load_lod_onto_bundle
from lod.inspect import layer_weight_means
from lod.orbit_9c6a import ALPHA_THRESHOLD
from lod.rasterizer import require_rade_gs
from lod.render import render_lod_appearance
from refinement.dloral_backend import DLoRALBackend
from refinement.geometry_warp import estimate_depth_scene_scale
from refinement.types import MultiViewInput
from scene import GaussianModel, Scene
from train_zoom_gen import apply_stage1_cfg_to_args, load_stage1_cfg
from utils.general_utils import PILtoTorch
from utils.zoom_mvp_utils import select_appearance_embedding


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        return {"path": str(target), "missing": True}
    stat = target.stat()
    return {
        "path": str(target),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(target),
    }


def gpu_snapshot() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        payload.update(
            {
                "device": int(torch.cuda.current_device()),
                "name": torch.cuda.get_device_name(),
                "free_gb": float(free) / 1024**3,
                "total_gb": float(total) / 1024**3,
                "allocated_gb": float(torch.cuda.memory_allocated()) / 1024**3,
                "max_allocated_gb": float(torch.cuda.max_memory_allocated()) / 1024**3,
            }
        )
    return payload


def camera_payload(camera) -> dict[str, Any]:
    return {
        "image_name": str(camera.image_name),
        "uid": int(camera.uid),
        "width": int(camera.image_width),
        "height": int(camera.image_height),
        "FoVx": float(camera.FoVx),
        "FoVy": float(camera.FoVy),
        "cx": float(camera.cx),
        "cy": float(camera.cy),
        "R": np.asarray(camera.R).tolist(),
        "T": np.asarray(camera.T).tolist(),
        "camera_center": camera.camera_center.detach().cpu().tolist(),
    }


def save_gray(value: torch.Tensor, path: str | Path) -> None:
    image = value.detach().float()
    while image.ndim > 2:
        image = image.squeeze(0)
    Image.fromarray(
        (image.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8),
        mode="L",
    ).save(path)


def save_depth(value: torch.Tensor, path: str | Path) -> None:
    image = value.detach().float()
    while image.ndim > 2:
        image = image.squeeze(0)
    np.save(path, image.cpu().numpy())


def load_rgb(
    path: str | Path,
    device: str | torch.device,
    *,
    image_size: int,
) -> torch.Tensor:
    with Image.open(path) as image:
        value = PILtoTorch(image.convert("RGB"), (int(image_size), int(image_size)))
    return value[:3].to(device=device, dtype=torch.float32).clamp(0, 1)


def _fit_image(path: Path, cell: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((cell, cell), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (cell, cell), "black")
        canvas.paste(image, ((cell - image.width) // 2, (cell - image.height) // 2))
    return canvas


def save_mosaic(
    rows: Sequence[Sequence[tuple[str, Path]]],
    output: Path,
    *,
    cell: int = 256,
) -> None:
    cols = max((len(row) for row in rows), default=0)
    label_h = 24
    canvas = Image.new("RGB", (cols * cell, len(rows) * (cell + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(rows):
        y = row_index * (cell + label_h)
        for col_index, (label, path) in enumerate(row):
            x = col_index * cell
            draw.rectangle((x, y, x + cell, y + label_h), fill=(245, 245, 245))
            draw.text((x + 4, y + 4), label, fill="black")
            canvas.paste(_fit_image(path, cell), (x, y + label_h))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)


def make_dloral_backend(args) -> DLoRALBackend:
    weight_root = os.environ.get("DLORAL_WEIGHT_ROOT", args.dloral_weight_root)
    return DLoRALBackend(
        repo_root=args.dloral_root,
        sd_path=args.sd_path or os.path.join(weight_root, "stable-diffusion-2-1-base"),
        ckpt_path=args.ckpt or os.path.join(weight_root, "model_enhanced.pkl"),
        spynet_path=args.spynet
        or os.path.join(weight_root, "spynet_20210409-c6c1bd09.pth"),
        python=args.dloral_python,
        device=args.dloral_device,
        stages=1,
        process_size=512,
        upscale=1,
        align_method="adain",
        alignment="geometry",
        latent_tiled_size=96,
        max_roundtrip_error_px=None,
    )


def neighbor_input(source: Path, selected: Mapping[str, Any]) -> MultiViewInput:
    return MultiViewInput(
        name=str(selected["name"]),
        image=Image.open(source / "neighbor.png").convert("RGB"),
        pixel_flow=torch.from_numpy(np.load(source / "target_to_source_flow.npy")),
        valid_mask=torch.from_numpy(
            np.asarray(Image.open(source / "flow_valid.png").convert("L"))
        )
        > 127,
        source_to_target_flow=torch.from_numpy(
            np.load(source / "source_to_target_flow.npy")
        ),
        reverse_valid_mask=torch.from_numpy(
            np.asarray(Image.open(source / "reverse_valid.png").convert("L"))
        )
        > 127,
        weight=float(selected["score"]),
        metadata={
            "kind": "same_center_same_elevation_pm10",
            "geometry_guided": True,
        },
    )


def freeze_for_l1(bundle) -> dict[str, Any]:
    for index, layer in enumerate(bundle.lod.layers):
        for parameter in layer.parameters():
            parameter.requires_grad_(index == 1)
    return {
        "active_level": int(bundle.lod.active_level),
        "l0_trainable": any(
            parameter.requires_grad for parameter in bundle.layer(0).parameters()
        ),
        "l1_trainable": any(
            parameter.requires_grad for parameter in bundle.layer(1).parameters()
        ),
        "l2_present": len(bundle.lod.layers) > 2,
        "appearance": appearance_freeze_report(bundle.appearance),
    }


def render_bundle(
    bundle,
    camera,
    *,
    background,
    kernel: float,
    embedding,
    max_level: int | None = None,
) -> dict[str, Any]:
    kwargs = {
        "appearance_embedding": embedding,
        "lod": True,
        "compact": False,
    }
    if max_level is not None:
        kwargs["max_level"] = int(max_level)
    package = render_lod_appearance(
        bundle,
        camera,
        background=background,
        kernel_size=kernel,
        **kwargs,
    )
    alpha = package["render_alpha"]
    alpha_hw = alpha[0] if alpha.ndim == 3 and alpha.shape[0] == 1 else alpha.reshape(alpha.shape[-2:])
    return {
        "pkg": package,
        "rgb": package["render"].clamp(0, 1),
        "alpha": alpha_hw,
        "camera_z": rade_pkg_camera_z(package, camera),
        "weights": layer_weight_means(package),
    }


def drop_context(context: dict[str, Any]) -> None:
    for key in (
        "bundle",
        "gaussians",
        "scene",
        "train_cameras",
        "background",
        "embedding",
    ):
        context[key] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_context(args, *, extra_1x=None) -> dict[str, Any]:
    require_rade_gs()
    stage1_checkpoint = os.path.abspath(args.start_checkpoint)
    stage1_cfg = load_stage1_cfg(os.path.dirname(stage1_checkpoint))
    apply_stage1_cfg_to_args(args, stage1_cfg)
    extract_parser = argparse.ArgumentParser(add_help=False)
    dataset = ModelParams(extract_parser).extract(args)
    dataset.model_path = os.path.abspath(args.output_dir)
    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs,
        dataset.appearance_embedding_dim,
    )
    model_params, first_iter = torch.load(stage1_checkpoint, weights_only=False)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=first_iter,
        shuffle=False,
        ply_path=os.path.dirname(stage1_checkpoint),
    )
    gaussians.load_from_checkpoints(model_params)
    train_cameras = list(scene.getTrainCameras())
    gaussians.compute_3D_filter(cameras=train_cameras)
    bundle = import_skyfall_l0(
        gaussians,
        train_cameras,
        gz_root=args.gz_root,
        step_scale=float(args.step_scale),
        freeze=True,
    )
    extras = list(extra_1x or [])
    if extras and len(bundle.lod.layers) == 1:
        names = {camera["name"] for camera in bundle.lod.stage_records[0]["cameras"]}
        missing = [camera for camera in extras if camera.image_name not in names]
        if missing:
            extend_base_stage(bundle, missing, gz_root=args.gz_root)
    optimizer_state = None
    if getattr(args, "lod_checkpoint", ""):
        optimizer_state = load_lod_onto_bundle(
            bundle,
            os.path.abspath(args.lod_checkpoint),
            device=str(gaussians.get_xyz.device),
        )
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device=gaussians.get_xyz.device,
    )
    return {
        "dataset": dataset,
        "gaussians": gaussians,
        "scene": scene,
        "bundle": bundle,
        "train_cameras": train_cameras,
        "background": background,
        "kernel": float(dataset.kernel_size),
        "device": str(gaussians.get_xyz.device),
        "lod_opt_state": optimizer_state,
        "orbit_1x": extras,
    }


def appearance_embedding(context: Mapping[str, Any], uid: int):
    return select_appearance_embedding(context["gaussians"], int(uid), False)


def pair_warp(
    destination_camera,
    destination: Mapping[str, Any],
    source_camera,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    valid = destination["alpha"] > ALPHA_THRESHOLD
    median_z = float(destination["camera_z"][valid].median().item()) if bool(valid.any()) else 1.0
    scene_scale = estimate_depth_scene_scale(
        destination_camera,
        median_z if median_z > 0 else 1.0,
    )
    forward, warped = warp_source_into_destination(
        dest_camera=destination_camera,
        source_camera=source_camera,
        dest_depth=destination["camera_z"],
        source_depth=source["camera_z"],
        dest_alpha=destination["alpha"],
        source_alpha=source["alpha"],
        source_rgb=source["rgb"],
        scene_scale=scene_scale,
        alpha_threshold=ALPHA_THRESHOLD,
    )
    reverse, _ = warp_source_into_destination(
        dest_camera=source_camera,
        source_camera=destination_camera,
        dest_depth=source["camera_z"],
        source_depth=destination["camera_z"],
        dest_alpha=source["alpha"],
        source_alpha=destination["alpha"],
        source_rgb=destination["rgb"],
        scene_scale=scene_scale,
        alpha_threshold=ALPHA_THRESHOLD,
    )
    return {
        "coverage": float(forward.coverage),
        "reverse_coverage": float(reverse.coverage),
        "scene_scale": float(scene_scale),
        "warp": forward,
        "reverse": reverse,
        "warped": warped,
    }
