#!/usr/bin/env python3
"""Re-render the JAX_068 zoom pair and export dual-depth GeometryCorrespondence.

This does not run DLoRAL or 3DGS training.  It only loads the frozen Stage1
checkpoint, renders target + neighbor depth with the stored cameras, and
builds both flow directions from each view's own depth.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Camera.world_view_transform is created with ``.cuda()`` on the first visible GPU.
_device_flag = "cuda:2"
if "--device" in sys.argv:
    _device_flag = sys.argv[sys.argv.index("--device") + 1]
elif os.environ.get("DLORAL_DEVICE"):
    _device_flag = os.environ["DLORAL_DEVICE"]
if _device_flag.startswith("cuda:"):
    os.environ["CUDA_VISIBLE_DEVICES"] = _device_flag.split(":", 1)[1]

import numpy as np
import torch
from PIL import Image

from arguments import ModelParams, PipelineParams
from refinement.dloral_flows import correspondence_from_neighbor, correspondence_to_external_flows, roundtrip_diagnostics
from refinement.geometry_warp import build_reprojection_grid, correspondence_from_warps, expected_surface_depth
from refinement.types import MultiViewInput, to_jsonable
from scene import GaussianModel, Scene
from scene.cameras import Camera
from train_zoom_gen import apply_stage1_cfg_to_args, load_stage1_cfg, render_bundle
from utils.zoom_mvp_utils import embedding_for_train_camera, select_appearance_embedding


def _camera_from_snapshot(data: dict, device: str) -> Camera:
    height = int(data["image_height"])
    width = int(data["image_width"])
    image = torch.zeros(3, height, width)
    camera = Camera(
        colmap_id=data.get("colmap_id", 0),
        R=np.asarray(data["R"], dtype=np.float64),
        T=np.asarray(data["T"], dtype=np.float64),
        FoVx=float(data["fov_x"]),
        FoVy=float(data["fov_y"]),
        cx=float(data["cx"]),
        cy=float(data["cy"]),
        image=image,
        gt_alpha_mask=None,
        image_name=str(data["image_name"]),
        uid=int(data["uid"]),
        data_device=device,
    )
    camera.znear = float(data.get("znear", 0.01))
    camera.zfar = float(data.get("zfar", 100.0))
    return camera


def _save_flow(path: Path, flow: torch.Tensor) -> None:
    np.save(path, flow.detach().cpu().float().numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_dir", type=Path, default=Path("skyfall-gs_exp/zoom_gen_dloral_spynet_2x/zoom_2x"))
    parser.add_argument("--output_dir", type=Path, default=Path("skyfall-gs_exp/dloral_geometry_pair"))
    parser.add_argument("--start_checkpoint", type=str, default="skyfall-gs_exp/stage1/JAX_068/chkpnt30000.pth")
    parser.add_argument("--neighbor_substr", type=str, default="019")
    parser.add_argument("--device", type=str, default=os.environ.get("DLORAL_DEVICE", "cuda:2"))
    args = parser.parse_args()
    device = "cuda:0" if args.device.startswith("cuda:") else args.device

    geometry = json.loads((args.baseline_dir / "geometry.json").read_text(encoding="utf-8"))
    target_snap = (geometry.get("target") or geometry.get("warp", {}).get("target") or {})["camera"]
    selected = geometry.get("selected") or geometry.get("warp", {}).get("selected") or []
    neighbor_rec = next(
        (item for item in selected if args.neighbor_substr in str(item.get("name", ""))),
        None,
    )
    if neighbor_rec is None:
        raise FileNotFoundError(f"No selected neighbor matching {args.neighbor_substr!r}")
    neighbor_snap = neighbor_rec["camera"]
    scene_scale = neighbor_rec.get("depth_scene_scale", geometry.get("depth_scene_scale"))

    checkpoint = os.path.abspath(args.start_checkpoint)
    stage1_dir = os.path.dirname(checkpoint)
    dummy = argparse.Namespace(
        source_path="",
        model_path=str(args.output_dir),
        sh_degree=1,
        appearance_enabled=True,
        appearance_n_fourier_freqs=4,
        appearance_embedding_dim=32,
        images="images",
        resolution=1,
        white_background=False,
        data_device=device,
        kernel_size=0.1,
        eval=True,
        ray_jitter=False,
        resample_gt_image=False,
        load_allres=False,
        sample_more_highres=False,
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
    )
    apply_stage1_cfg_to_args(dummy, load_stage1_cfg(stage1_dir))
    dummy.data_device = device
    extract_parser = argparse.ArgumentParser(add_help=False)
    dataset = ModelParams(extract_parser).extract(dummy)
    pipe = PipelineParams(extract_parser).extract(dummy)
    dataset.model_path = str(args.output_dir.resolve())
    dataset.data_device = device

    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs,
        dataset.appearance_embedding_dim,
    )
    model_params, first_iter = torch.load(checkpoint, weights_only=False)
    scene = Scene(dataset, gaussians, load_iteration=first_iter, shuffle=False, ply_path=stage1_dir)
    gaussians.load_from_checkpoints(model_params)
    for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity", "_embeddings", "appearance_embeddings", "max_radii2D"):
        tensor = getattr(gaussians, name, None)
        if torch.is_tensor(tensor):
            setattr(gaussians, name, tensor.to(device))
    if getattr(gaussians, "appearance_mlp", None) is not None:
        gaussians.appearance_mlp.to(device)

    target_cam = _camera_from_snapshot(target_snap, device)
    neighbor_cam = _camera_from_snapshot(neighbor_snap, device)
    train_cameras = scene.getTrainCameras()
    gaussians.compute_3D_filter(cameras=list(train_cameras) + [target_cam, neighbor_cam])
    background = torch.tensor([0.0, 0.0, 0.0], device=device)
    kernel_size = float(getattr(dataset, "kernel_size", 0.1))
    target_embed = select_appearance_embedding(gaussians, int(target_snap.get("colmap_id", 0)), True)
    neighbor_embed = embedding_for_train_camera(gaussians, int(neighbor_rec.get("base_uid", neighbor_snap["colmap_id"])))

    with torch.no_grad():
        target_bundle = render_bundle(target_cam, gaussians, pipe, background, kernel_size, target_embed)
        neighbor_bundle = render_bundle(neighbor_cam, gaussians, pipe, background, kernel_size, neighbor_embed)

    target_depth = expected_surface_depth(target_bundle.depth, target_bundle.alpha)
    neighbor_depth = expected_surface_depth(neighbor_bundle.depth, neighbor_bundle.alpha)
    forward = build_reprojection_grid(
        target_cam,
        neighbor_cam,
        target_depth,
        neighbor_depth,
        neighbor_bundle.alpha,
        target_alpha=target_bundle.alpha,
        depth_abs_tolerance=5.0,
        depth_rel_tolerance=1e-5,
        depth_scene_scale=None if scene_scale is None else float(scene_scale),
    )
    reverse = build_reprojection_grid(
        neighbor_cam,
        target_cam,
        neighbor_depth,
        target_depth,
        target_bundle.alpha,
        target_alpha=neighbor_bundle.alpha,
        depth_abs_tolerance=5.0,
        depth_rel_tolerance=1e-5,
        depth_scene_scale=None if scene_scale is None else float(scene_scale),
    )
    correspondence = correspondence_from_warps(
        forward,
        reverse,
        camera_metadata={
            "target": target_snap,
            "neighbor": neighbor_snap,
            "reverse_source": "depth",
        },
    )
    roundtrip = roundtrip_diagnostics(
        forward.pixel_flow, forward.valid_mask, reverse.pixel_flow, reverse.valid_mask
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_flow(args.output_dir / "target_to_source_flow.npy", forward.pixel_flow)
    _save_flow(args.output_dir / "source_to_target_flow.npy", reverse.pixel_flow)
    np.save(args.output_dir / "forward_valid.npy", forward.valid_mask.detach().cpu().bool().numpy())
    np.save(args.output_dir / "reverse_valid.npy", reverse.valid_mask.detach().cpu().bool().numpy())
    Image.fromarray((forward.valid_mask.cpu().numpy().astype("uint8") * 255), mode="L").save(
        args.output_dir / "forward_valid.png"
    )
    Image.fromarray((reverse.valid_mask.cpu().numpy().astype("uint8") * 255), mode="L").save(
        args.output_dir / "reverse_valid.png"
    )
    neighbor = MultiViewInput(
        name=str(neighbor_rec.get("name", "neighbor")),
        image=neighbor_bundle.rgb,
        pixel_flow=forward.pixel_flow,
        valid_mask=forward.valid_mask,
        source_to_target_flow=reverse.pixel_flow,
        reverse_valid_mask=reverse.valid_mask,
        metadata={
            "source_size": [int(neighbor_cam.image_width), int(neighbor_cam.image_height)],
            "target_size": [int(target_cam.image_width), int(target_cam.image_height)],
            "coverage": forward.coverage,
            "reverse_coverage": reverse.coverage,
            "reverse_source": "depth",
        },
    )
    feature_payload = correspondence_to_external_flows(correspondence_from_neighbor(neighbor), process_size=512, upscale=1)
    summary = {
        "reverse_source": "depth",
        "forward_coverage": forward.coverage,
        "reverse_coverage": reverse.coverage,
        "roundtrip": roundtrip,
        "feature_reverse_source": feature_payload["reverse_source"],
        "feature_coverage": feature_payload["coverage"],
        "feature_roundtrip": feature_payload["roundtrip"],
        "neighbor": neighbor.name,
        "target": target_snap["image_name"],
        "scene_scale": scene_scale,
        "note": "L1 between SpyNet and geometry is a difference, not a quality ranking.",
    }
    (args.output_dir / "correspondence.json").write_text(
        json.dumps(to_jsonable(summary), indent=2), encoding="utf-8"
    )
    print(json.dumps(to_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
