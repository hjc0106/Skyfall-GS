#!/usr/bin/env python3
"""Render frozen L0+L1 at 4x and rebuild DLoRAL geometry supervision. Does not train."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from arguments import ModelParams, PipelineParams
from lod.correspondence import rade_pkg_camera_z, warp_source_into_destination
from lod.importer import import_skyfall_l0, load_lod_onto_bundle
from lod.camera import camera_image_stem
from lod.lineage import roi_dict, supervision_lineage, write_json
from lod.path import DEFAULT_GZ_ROOT
from lod.rasterizer import require_rade_gs
from lod.render import render_lod_appearance
from refinement.dloral_backend import DLoRALBackend
from refinement.geometry_warp import estimate_depth_scene_scale, estimate_spatial_target, rank_cameras_by_spatial_overlap
from refinement.types import CameraSnapshot, MultiViewInput, PromptDescription, RefinementRequest
from scene import GaussianModel, Scene
from train_zoom_gen import apply_stage1_cfg_to_args, load_stage1_cfg, pick_base_camera, tensor_to_pil
from utils.general_utils import safe_state
from utils.zoom_camera import NormalizedROI, make_zoom_camera, nearest_train_cameras, save_roi_overlay
from utils.zoom_mvp_utils import embedding_for_train_camera, save_tensor_image, select_appearance_embedding


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def _spatial(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().float()
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    return value


def _render(bundle, camera, *, background, kernel: float, embedding):
    pkg = render_lod_appearance(
        bundle, camera, background=background, kernel_size=kernel,
        appearance_embedding=embedding, lod=True, compact=False,
    )
    return {
        "rgb": pkg["render"].clamp(0, 1),
        "alpha": _spatial(pkg["render_alpha"]),
        "camera_z": rade_pkg_camera_z(pkg, camera),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    ModelParams(parser)
    PipelineParams(parser)
    parser.add_argument("--start_checkpoint", type=str, required=True)
    parser.add_argument("--l1_checkpoint", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompt_json", type=str, default="")
    parser.add_argument("--vlm_model_path", type=str, default="")
    parser.add_argument("--vlm_python", type=str, default="")
    parser.add_argument("--vlm_device", type=str, default="cuda:0")
    parser.add_argument("--vlm_max_new_tokens", type=int, default=768)
    parser.add_argument("--vlm_max_image_size", type=int, default=1024)
    parser.add_argument("--gz_root", type=str, default=str(DEFAULT_GZ_ROOT))
    parser.add_argument("--view_index", type=int, default=0)
    parser.add_argument("--view_name", type=str, default="", help="Lock the base camera by image stem. Preferred over view_index.")
    parser.add_argument("--roi_center_x", type=float, default=0.592)
    parser.add_argument("--roi_center_y", type=float, default=0.53)
    parser.add_argument("--roi_width", type=float, default=0.1)
    parser.add_argument("--roi_height", type=float, default=0.1)
    parser.add_argument("--zoom_factor", type=float, default=4.0)
    parser.add_argument("--step_scale", type=float, default=2.0)
    parser.add_argument("--min_reprojection_coverage", type=float, default=0.05)
    parser.add_argument("--alpha_threshold", type=float, default=0.05)
    parser.add_argument("--geometry_neighbor_count", type=int, default=2)
    parser.add_argument("--geometry_pool_size", type=int, default=8)
    parser.add_argument("--dloral_alignment", type=str, default="auto", choices=["auto", "geometry", "spynet", "target_only"])
    parser.add_argument("--run_dloral", action="store_true")
    parser.add_argument("--dloral_python", type=str, default="")
    parser.add_argument("--dloral_device", type=str, default="cuda:0")
    parser.add_argument("--dloral_root", type=str, default="submodules/DLoRAL")
    parser.add_argument("--sd_path", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--spynet", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max_roundtrip_error_px",
        type=float,
        default=None,
        help="Optional full-res roundtrip gate in image pixels, applied before feature downsample.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    require_rade_gs()
    safe_state(args.quiet)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    stage1_cfg = load_stage1_cfg(os.path.dirname(os.path.abspath(args.start_checkpoint)))
    apply_stage1_cfg_to_args(args, stage1_cfg)
    extract_parser = argparse.ArgumentParser(add_help=False)
    dataset = ModelParams(extract_parser).extract(args)
    dataset.model_path = output_dir

    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs,
        dataset.appearance_embedding_dim,
    )
    model_params, first_iter = torch.load(args.start_checkpoint, weights_only=False)
    scene = Scene(
        dataset, gaussians, load_iteration=first_iter, shuffle=False,
        ply_path=os.path.dirname(os.path.abspath(args.start_checkpoint)),
    )
    gaussians.load_from_checkpoints(model_params)
    train_cameras = list(scene.getTrainCameras())
    gaussians.compute_3D_filter(cameras=train_cameras)
    bundle = import_skyfall_l0(
        gaussians, train_cameras, gz_root=args.gz_root, step_scale=args.step_scale, freeze=True,
    )
    if args.l1_checkpoint:
        load_lod_onto_bundle(bundle, os.path.abspath(args.l1_checkpoint), device=str(gaussians.get_xyz.device))
    roi = NormalizedROI(args.roi_center_x, args.roi_center_y, args.roi_width, args.roi_height)
    roi.validate_for_zoom(args.zoom_factor)
    base_camera, is_train_view = pick_base_camera(scene, args.view_index, False, image_name=args.view_name)
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device=gaussians.get_xyz.device,
    )
    kernel = float(dataset.kernel_size)
    embedding = select_appearance_embedding(gaussians, base_camera.uid, is_train_view)
    target_zoom = make_zoom_camera(
        base_camera, roi, args.zoom_factor, uid=50000,
        image_name=f"{base_camera.image_name}_zoom{args.zoom_factor:g}",
    )
    target = _render(bundle, target_zoom, background=background, kernel=kernel, embedding=embedding)
    save_tensor_image(target["rgb"], os.path.join(output_dir, "render_input.png"))
    save_tensor_image(target["rgb"], os.path.join(output_dir, "target.png"))
    save_roi_overlay(base_camera, roi, os.path.join(output_dir, "roi_overlay.png"))

    spatial = estimate_spatial_target(
        target_zoom, target["camera_z"], target["alpha"],
        alpha_threshold=args.alpha_threshold,
    )
    scene_scale = estimate_depth_scene_scale(target_zoom, spatial.median_depth or 1.0)
    selected, spatial_records = rank_cameras_by_spatial_overlap(
        spatial, train_cameras, zoom_factor=args.zoom_factor,
        k=int(args.geometry_neighbor_count), pool_size=int(args.geometry_pool_size),
        target_camera=target_zoom, exclude_uids=(int(base_camera.uid), int(target_zoom.uid)),
    )
    aligned = []
    skipped = []
    for index, (base_neighbor, projected) in enumerate(selected):
        neighbor_roi = NormalizedROI(projected.center_x, projected.center_y, projected.width, projected.height)
        try:
            neighbor_zoom = make_zoom_camera(
                base_neighbor, neighbor_roi, args.zoom_factor, uid=51000 + index,
                image_name=f"{base_neighbor.image_name}_zoom{args.zoom_factor:g}_spatial",
            )
        except ValueError as exc:
            skipped.append({"name": base_neighbor.image_name, "reason": str(exc), "kind": "spatial_roi"})
            continue
        neigh = _render(
            bundle, neighbor_zoom, background=background, kernel=kernel,
            embedding=embedding_for_train_camera(gaussians, base_neighbor.uid),
        )
        warp, warped = warp_source_into_destination(
            dest_camera=target_zoom, source_camera=neighbor_zoom,
            dest_depth=target["camera_z"], source_depth=neigh["camera_z"],
            dest_alpha=target["alpha"], source_alpha=neigh["alpha"],
            source_rgb=neigh["rgb"], scene_scale=scene_scale,
            alpha_threshold=args.alpha_threshold,
        )
        reverse, _ = warp_source_into_destination(
            dest_camera=neighbor_zoom, source_camera=target_zoom,
            dest_depth=neigh["camera_z"], source_depth=target["camera_z"],
            dest_alpha=neigh["alpha"], source_alpha=target["alpha"],
            source_rgb=target["rgb"], scene_scale=scene_scale,
            alpha_threshold=args.alpha_threshold,
        )
        rec = {
            "name": base_neighbor.image_name,
            "coverage": float(warp.coverage),
            "reverse_coverage": float(reverse.coverage),
            "kind": "spatial_roi",
            "spatial_roi": projected.to_dict(),
        }
        aligned.append((rec, base_neighbor, neighbor_zoom, neigh, warp, reverse, warped))

    if not aligned:
        for index, camera in enumerate(nearest_train_cameras(base_camera, train_cameras, k=2)):
            try:
                neighbor_zoom = make_zoom_camera(
                    camera, roi, args.zoom_factor, uid=52000 + index,
                    image_name=f"{camera.image_name}_zoom{args.zoom_factor:g}",
                )
            except ValueError as exc:
                skipped.append({"name": camera.image_name, "reason": str(exc), "kind": "same_roi"})
                continue
            neigh = _render(
                bundle, neighbor_zoom, background=background, kernel=kernel,
                embedding=embedding_for_train_camera(gaussians, camera.uid),
            )
            warp, warped = warp_source_into_destination(
                dest_camera=target_zoom, source_camera=neighbor_zoom,
                dest_depth=target["camera_z"], source_depth=neigh["camera_z"],
                dest_alpha=target["alpha"], source_alpha=neigh["alpha"],
                source_rgb=neigh["rgb"], scene_scale=scene_scale,
                alpha_threshold=args.alpha_threshold,
            )
            reverse, _ = warp_source_into_destination(
                dest_camera=neighbor_zoom, source_camera=target_zoom,
                dest_depth=neigh["camera_z"], source_depth=target["camera_z"],
                dest_alpha=neigh["alpha"], source_alpha=target["alpha"],
                source_rgb=target["rgb"], scene_scale=scene_scale,
                alpha_threshold=args.alpha_threshold,
            )
            rec = {
                "name": camera.image_name,
                "coverage": float(warp.coverage),
                "reverse_coverage": float(reverse.coverage),
                "kind": "same_roi_fallback",
            }
            aligned.append((rec, camera, neighbor_zoom, neigh, warp, reverse, warped))

    aligned.sort(key=lambda item: item[0]["coverage"], reverse=True)
    best_coverage = aligned[0][0]["coverage"] if aligned else 0.0
    fallback = None if aligned and best_coverage >= float(args.min_reprojection_coverage) else "target_only"
    if args.dloral_alignment == "auto":
        alignment = "geometry" if fallback is None else "target_only"
    else:
        alignment = args.dloral_alignment
        if alignment == "geometry" and fallback is not None:
            alignment = "target_only"

    neighbor_input = None
    if aligned:
        rec, base_neighbor, neighbor_zoom, neigh, warp, reverse, warped = aligned[0]
        save_tensor_image(neigh["rgb"], os.path.join(output_dir, "neighbor.png"))
        save_tensor_image(warped, os.path.join(output_dir, "neighbor_warped.png"))
        np.save(os.path.join(output_dir, "target_to_source_flow.npy"), warp.pixel_flow.detach().cpu().float().numpy())
        np.save(os.path.join(output_dir, "source_to_target_flow.npy"), reverse.pixel_flow.detach().cpu().float().numpy())
        Image.fromarray((warp.valid_mask.detach().cpu().numpy().astype("uint8") * 255), mode="L").save(
            os.path.join(output_dir, "flow_valid.png")
        )
        neighbor_input = MultiViewInput(
            name=base_neighbor.image_name,
            image=tensor_to_pil(neigh["rgb"]),
            pixel_flow=warp.pixel_flow,
            valid_mask=warp.valid_mask,
            source_to_target_flow=reverse.pixel_flow,
            reverse_valid_mask=reverse.valid_mask,
            weight=float(warp.coverage),
            metadata={"coverage": float(warp.coverage), "reverse_source": "depth", "kind": rec["kind"]},
        )

    target_view_name = camera_image_stem(getattr(base_camera, "image_name", ""))
    geometry_payload = {
        "status": "ok" if fallback is None else "low_reprojection_coverage",
        "fallback": fallback,
        "alignment": alignment,
        "requested_alignment": args.dloral_alignment,
        "target_view": {
            "image_name": target_view_name,
            "view_index": int(args.view_index),
            "requested_view_name": str(getattr(args, "view_name", "") or "") or None,
        },
        "best_coverage": best_coverage,
        "spatial_target": spatial.to_dict(),
        "depth_scene_scale": scene_scale,
        "spatial_candidates": spatial_records,
        "skipped": skipped,
        "selected": [item[0] for item in aligned],
        "min_reprojection_coverage": float(args.min_reprojection_coverage),
        "depth_kind": "camera_z",
        "note": "Invalid warp pixels fall back to target features inside DLoRAL geometry alignment.",
    }
    _write_json(os.path.join(output_dir, "geometry.json"), geometry_payload)

    refined_path = os.path.join(output_dir, "refined.png")
    if args.run_dloral:
        prompt_text = ""
        prompt_path = os.path.join(output_dir, "prompt.json")
        if args.vlm_model_path:
            from refinement.qwen3_vlm import Qwen3PromptProvider

            provider = Qwen3PromptProvider(
                args.vlm_model_path,
                python=args.vlm_python or os.path.expanduser("~/miniconda3/envs/fixanything/bin/python3.10"),
                device=args.vlm_device,
                max_new_tokens=int(args.vlm_max_new_tokens),
                max_image_size=int(args.vlm_max_image_size),
            )
            prompt = provider.describe(
                tensor_to_pil(base_camera.original_image),
                tensor_to_pil(target["rgb"]),
                zoom_factor=float(args.zoom_factor),
                level_index=1 if args.l1_checkpoint else 0,
                context={},
            )
            _write_json(prompt_path, prompt.to_dict())
            prompt_text = prompt.target_prompt
        else:
            prompt_json_path = os.path.abspath(args.prompt_json)
            if not prompt_json_path or not os.path.isfile(prompt_json_path):
                raise ValueError("DLoRAL needs --prompt_json or --vlm_model_path")
            prompt_json = json.loads(open(prompt_json_path, encoding="utf-8").read())
            prompt_text = str(prompt_json.get("target_prompt") or prompt_json.get("prompt") or "")
            if not prompt_text and isinstance(prompt_json.get("config"), dict):
                prompt_text = str(prompt_json["config"].get("target_prompt") or "")
            if prompt_json_path != os.path.abspath(prompt_path):
                _write_json(prompt_path, prompt_json)
        if not prompt_text.strip():
            raise ValueError("resolved DLoRAL target_prompt is empty")
        weight_root = os.environ.get("DLORAL_WEIGHT_ROOT", "weights/dloral")
        backend = DLoRALBackend(
            repo_root=args.dloral_root,
            sd_path=args.sd_path or os.path.join(weight_root, "stable-diffusion-2-1-base"),
            ckpt_path=args.ckpt or os.path.join(weight_root, "model_enhanced.pkl"),
            spynet_path=args.spynet or os.path.join(weight_root, "spynet_20210409-c6c1bd09.pth"),
            python=args.dloral_python or os.path.expanduser("~/miniconda3/envs/dloral/bin/python"),
            device=args.dloral_device,
            stages=1,
            process_size=512,
            upscale=1,
            align_method="adain",
            alignment=alignment,
            latent_tiled_size=96,
            max_roundtrip_error_px=args.max_roundtrip_error_px,
        )
        request = RefinementRequest(
            image=tensor_to_pil(target["rgb"]),
            checkpoint=os.path.abspath(args.start_checkpoint),
            camera=CameraSnapshot.from_camera(target_zoom),
            zoom_factor=float(args.zoom_factor),
            sr_scale=1.0,
            prompt=PromptDescription(target_prompt=prompt_text, provider="archived_qwen3vl"),
            metadata={
                "seed": int(args.seed),
                "neighbor_views": (
                    [] if neighbor_input is None or alignment in ("target_only",)
                    else [neighbor_input]
                ),
                "backend_save_dir": os.path.join(output_dir, "dloral"),
            },
        )
        os.makedirs(os.path.join(output_dir, "dloral"), exist_ok=True)
        # Drop imported LoD before the DLoRAL subprocess. JAX_214 Stage1 is 1.35M
        # points; keeping both on the same card filled 46,909 MiB on this run.
        bundle = None
        gaussians = None
        scene = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result = backend.refine(request)
        result.image.save(refined_path)
        _write_json(os.path.join(output_dir, "dloral_result.json"), result.to_dict())
    prompt_path = os.path.join(output_dir, "prompt.json")
    weight_root = os.environ.get("DLORAL_WEIGHT_ROOT", "weights/dloral")
    write_json(
        os.path.join(output_dir, "lineage.json"),
        supervision_lineage(
            start_checkpoint=os.path.abspath(args.start_checkpoint),
            parent_lod=os.path.abspath(args.l1_checkpoint) if args.l1_checkpoint else None,
            roi=roi_dict(args.roi_center_x, args.roi_center_y, args.roi_width, args.roi_height),
            view_index=int(args.view_index),
            zoom_factor=float(args.zoom_factor),
            step_scale=float(args.step_scale),
            alignment=str(alignment),
            seed=int(args.seed),
            max_roundtrip_error_px=args.max_roundtrip_error_px,
            dloral_ckpt=args.ckpt or os.path.join(weight_root, "model_enhanced.pkl"),
            prompt_path=prompt_path if os.path.isfile(prompt_path) else None,
            refined_image=refined_path if os.path.isfile(refined_path) else None,
            image_name=target_view_name,
        ),
    )
    print(json.dumps({
        "alignment": alignment,
        "view_name": target_view_name,
        "best_coverage": best_coverage,
        "fallback": fallback,
        "refined": refined_path if os.path.isfile(refined_path) else None,
        "selected": [item[0] for item in aligned[:2]],
    }, indent=2))


if __name__ == "__main__":
    main()
