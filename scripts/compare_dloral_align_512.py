#!/usr/bin/env python3
"""512 full-image DLoRAL alignment comparison. Does not run 3DGS training.

Reuses the JAX_068 2x SpyNet baseline pair, prompt, weights, and seed.
Modes: target_only, native SpyNet, depth-driven geometry warping.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from refinement.dloral_backend import DLoRALBackend
from refinement.dloral_flows import resize_pixel_flow, valid_mask_from_flow
from refinement.types import CameraSnapshot, MultiViewInput, PromptDescription, RefinementRequest

DEFAULT_BASELINE = Path("skyfall-gs_exp/zoom_gen_dloral_spynet_2x/zoom_2x")
DEFAULT_OUT = Path("skyfall-gs_exp/zoom_gen_dloral_align_512")
WEIGHT_ROOT = Path(os.environ.get("DLORAL_WEIGHT_ROOT", "/datacc05/hongjiacheng/sr_models/dloral"))
NEIGHBOR_FLOW = "geometry_neighbor_1_JAX_068_019_RGB_zoom2_spatial_pixel_flow.npy"


def _camera(snapshot: dict, width: int, height: int) -> CameraSnapshot:
    return CameraSnapshot(
        image_name=str(snapshot.get("image_name", "zoom512")),
        uid=int(snapshot.get("uid", 0)),
        colmap_id=snapshot.get("colmap_id"),
        image_width=int(width),
        image_height=int(height),
        fov_x=float(snapshot.get("fov_x", snapshot.get("FoVx", 1.0))),
        fov_y=float(snapshot.get("fov_y", snapshot.get("FoVy", 1.0))),
        cx=float(snapshot.get("cx", 0.0)),
        cy=float(snapshot.get("cy", 0.0)),
        R=tuple(tuple(float(item) for item in row) for row in snapshot["R"]),
        T=tuple(float(item) for item in snapshot["T"]),
        znear=float(snapshot.get("znear", 0.01)),
        zfar=float(snapshot.get("zfar", 100.0)),
    )


def _l1(a: Image.Image, b: Image.Image) -> float:
    left = np.asarray(a.convert("RGB"), dtype=np.float32)
    right = np.asarray(b.convert("RGB"), dtype=np.float32)
    return float(np.abs(left - right).mean() / 255.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", type=str, default=os.environ.get("DLORAL_DEVICE", "cuda:2"))
    parser.add_argument("--dloral_python", type=str, default=os.environ.get("DLORAL_PYTHON", str(Path.home() / "miniconda3/envs/dloral/bin/python")))
    parser.add_argument("--sd_path", type=str, default=str(WEIGHT_ROOT / "stable-diffusion-2-1-base"))
    parser.add_argument("--ckpt", type=str, default=str(WEIGHT_ROOT / "model.pkl"))
    parser.add_argument("--spynet", type=str, default=str(WEIGHT_ROOT / "spynet_20210409-c6c1bd09.pth"))
    parser.add_argument("--dloral_root", type=str, default="submodules/DLoRAL")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--geometry_dir", type=Path, default=None, help="Directory with dual-depth npy from export_dloral_geometry_pair.py")
    parser.add_argument("--forward_flow", type=Path, default=None)
    parser.add_argument("--reverse_flow", type=Path, default=None)
    args = parser.parse_args()

    baseline = args.baseline_dir.resolve()
    size = int(args.size)
    target = Image.open(baseline / "dloral" / "target.png").convert("RGB")
    neighbor = Image.open(baseline / "dloral" / "neighbor.png").convert("RGB")
    if target.size != (size, size):
        target = target.resize((size, size), Image.Resampling.LANCZOS)
        neighbor = neighbor.resize((size, size), Image.Resampling.LANCZOS)

    geometry_dir = args.geometry_dir
    forward_path = args.forward_flow
    reverse_path = args.reverse_flow
    if geometry_dir is not None:
        geometry_dir = geometry_dir.resolve()
        forward_path = forward_path or (geometry_dir / "target_to_source_flow.npy")
        reverse_path = reverse_path or (geometry_dir / "source_to_target_flow.npy")
    if forward_path is None:
        forward_path = baseline / NEIGHBOR_FLOW
    flow2048 = torch.from_numpy(np.load(forward_path)).float()
    if flow2048.shape[-1] != 2:
        raise ValueError(f"pixel_flow must be HWC2, got {tuple(flow2048.shape)}")
    flow = resize_pixel_flow(
        flow2048,
        source_size=(flow2048.shape[1], flow2048.shape[0]),
        target_size=(size, size),
    )
    reverse = None
    reverse_valid = None
    reverse_source_hint = "scatter_invert_diagnostic_only"
    if reverse_path is not None and Path(reverse_path).is_file():
        reverse2048 = torch.from_numpy(np.load(reverse_path)).float()
        reverse = resize_pixel_flow(
            reverse2048,
            source_size=(reverse2048.shape[1], reverse2048.shape[0]),
            target_size=(size, size),
        )
        reverse_valid = valid_mask_from_flow(reverse)
        reverse_source_hint = "depth"
    prompt_json = json.loads((baseline / "prompt.json").read_text(encoding="utf-8"))
    prompt_text = str(prompt_json["target_prompt"])
    geometry = json.loads((baseline / "geometry.json").read_text(encoding="utf-8"))
    selected = geometry.get("selected") or []
    neighbor_cam = next(
        (item["camera"] for item in selected if "019" in str(item.get("name", ""))),
        selected[0]["camera"] if selected else {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "T": [0, 0, 0]},
    )
    valid = valid_mask_from_flow(flow)
    neighbor_input = MultiViewInput(
        name="JAX_068_019",
        image=neighbor,
        pixel_flow=flow,
        valid_mask=valid,
        source_to_target_flow=reverse,
        reverse_valid_mask=reverse_valid,
        weight=1.0,
        metadata={
            "source_size": [size, size],
            "target_size": [size, size],
            "coverage": float(valid.float().mean()),
            "reverse_source": reverse_source_hint,
        },
    )
    camera = _camera(neighbor_cam, size, size)
    prompt = PromptDescription(target_prompt=prompt_text, provider="archived_qwen3vl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target.save(args.output_dir / f"target_{size}.png")
    neighbor.save(args.output_dir / f"neighbor_{size}.png")
    Image.fromarray((valid.numpy().astype("uint8") * 255), mode="L").save(args.output_dir / f"flow_valid_{size}.png")

    results = {}
    images = {}
    for alignment in ("target_only", "spynet", "geometry"):
        out_dir = args.output_dir / alignment
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        backend = DLoRALBackend(
            repo_root=args.dloral_root,
            sd_path=args.sd_path,
            ckpt_path=args.ckpt,
            spynet_path=args.spynet,
            python=args.dloral_python,
            device=args.device,
            stages=1,
            process_size=512,
            upscale=1,
            align_method="adain",
            alignment=alignment,
            latent_tiled_size=96,
        )
        request = RefinementRequest(
            image=target,
            checkpoint=str(baseline.parents[1] / "stage1" / "JAX_068" / "chkpnt30000.pth"),
            camera=camera,
            zoom_factor=2.0,
            sr_scale=1.0,
            prompt=prompt,
            metadata={
                "seed": int(args.seed),
                "neighbor_views": [neighbor_input],
                "backend_save_dir": str(out_dir),
            },
        )
        result = backend.refine(request)
        result.image.save(out_dir / "refined.png")
        images[alignment] = result.image
        results[alignment] = {
            "alignment": alignment,
            "prepared_size": result.metadata.get("prepared_size"),
            "latent_size": result.metadata.get("latent_size"),
            "tiled": result.metadata.get("tiled"),
            "elapsed_sec": result.metadata.get("elapsed_sec"),
            "worker_elapsed_sec": result.metadata.get("worker_elapsed_sec"),
            "peak_cuda_memory_bytes": result.metadata.get("peak_cuda_memory_bytes"),
            "geometry_coverage": result.metadata.get("geometry_coverage"),
            "reverse_source": result.metadata.get("reverse_source"),
            "roundtrip": result.metadata.get("roundtrip"),
            "feature_diag": (result.metadata.get("worker") or {}).get("feature_diag"),
            "prompt_used": result.metadata.get("prompt_used"),
            "propagation": result.metadata.get("propagation"),
        }

    results["comparisons"] = {
        "l1_geometry_vs_spynet": _l1(images["geometry"], images["spynet"]),
        "l1_geometry_vs_target_only": _l1(images["geometry"], images["target_only"]),
        "l1_spynet_vs_target_only": _l1(images["spynet"], images["target_only"]),
        "geometry_differs_from_spynet": _l1(images["geometry"], images["spynet"]) > 1e-4,
        "geometry_differs_from_target_only": _l1(images["geometry"], images["target_only"]) > 1e-4,
        "input_size": [size, size],
        "seed": int(args.seed),
        "prompt": prompt_text,
        "neighbor": "JAX_068_019",
        "flow_valid_fraction": float(valid.float().mean()),
        "reverse_source": reverse_source_hint,
        "l1_is_not_quality_ranking": True,
        "note": (
            "Image L1 is a difference between modes, not a quality ranking. "
            "Inspect building boundaries, occlusions, and tile seams visually."
        ),
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["comparisons"], indent=2))


if __name__ == "__main__":
    main()
