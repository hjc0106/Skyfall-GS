"""Read-only recovery of a 2x->4x LoD chain: freeze, parent identity, 4x input linkage."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from arguments import ModelParams, PipelineParams
from lod.camera import camera_image_stem
from lod.freeze import (
    appearance_freeze_report,
    completed_layers_frozen,
    snapshot_levels,
    snapshot_mismatches,
)
from lod.importer import assert_l0_tensors_match, import_skyfall_l0, load_lod_onto_bundle
from lod.lineage import file_identity, roi_dict, write_json
from lod.path import DEFAULT_GZ_ROOT
from lod.rasterizer import require_rade_gs
from lod.render import render_lod_appearance
from scene import GaussianModel, Scene
from train_zoom_gen import apply_stage1_cfg_to_args, load_stage1_cfg, pick_base_camera
from utils.general_utils import safe_state
from utils.zoom_camera import NormalizedROI, make_zoom_camera
from utils.zoom_mvp_utils import image_l1, save_tensor_image, select_appearance_embedding


STAGE_LINK_L1_MAX = 1e-4


def _load_png(path: str, device) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).to(device=device, dtype=torch.float32)


def recover_scale_chain(
    *,
    start_checkpoint: str,
    l1_checkpoint: str,
    l2_checkpoint: str | None,
    l2_render_input: str | None,
    output_dir: str,
    gz_root: str,
    view_index: int,
    image_name: str | None = None,
    roi: NormalizedROI,
    zoom_l1: float,
    zoom_l2: float,
    step_scale: float,
    quiet: bool = False,
) -> dict:
    """Load existing checkpoints. Does not train and does not write into those files."""

    require_rade_gs()
    safe_state(quiet)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    start_checkpoint = os.path.abspath(start_checkpoint)
    l1_checkpoint = os.path.abspath(l1_checkpoint)
    parser = argparse.ArgumentParser()
    ModelParams(parser)
    PipelineParams(parser)
    parser.add_argument("--start_checkpoint", type=str, required=True)
    parser.add_argument("--quiet", action="store_true")
    argv = ["--start_checkpoint", start_checkpoint]
    if quiet:
        argv.append("--quiet")
    args = parser.parse_args(argv)
    stage1_cfg = load_stage1_cfg(os.path.dirname(start_checkpoint))
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
    model_params, first_iter = torch.load(start_checkpoint, weights_only=False)
    scene = Scene(
        dataset, gaussians, load_iteration=first_iter, shuffle=False,
        ply_path=os.path.dirname(start_checkpoint),
    )
    gaussians.load_from_checkpoints(model_params)
    train_cameras = list(scene.getTrainCameras())
    gaussians.compute_3D_filter(cameras=train_cameras)
    bundle = import_skyfall_l0(
        gaussians, train_cameras, gz_root=gz_root, step_scale=step_scale, freeze=True,
    )
    import_delta = assert_l0_tensors_match(gaussians, bundle)
    snap_l0 = snapshot_levels(bundle, (0,))
    device = str(gaussians.get_xyz.device)
    load_lod_onto_bundle(bundle, l1_checkpoint, device=device)
    snap_l1 = snapshot_levels(bundle, (0, 1))
    l0_vs_l1 = snapshot_mismatches(snap_l0, {"levels": {"0": snap_l1["levels"]["0"]}, "appearance": snap_l1["appearance"]})
    l1_completed = completed_layers_frozen(bundle, up_to_level=1)

    base_camera, is_train_view = pick_base_camera(scene, view_index, False, image_name=image_name)
    zoom4 = make_zoom_camera(
        base_camera, roi, zoom_l2, uid=50000,
        image_name=f"{base_camera.image_name}_zoom{zoom_l2:g}",
    )
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device=gaussians.get_xyz.device,
    )
    kernel = float(dataset.kernel_size)
    embedding = select_appearance_embedding(gaussians, base_camera.uid, is_train_view)
    rendered_4x = render_lod_appearance(
        bundle, zoom4, background=background, kernel_size=kernel,
        appearance_embedding=embedding, lod=True, compact=False,
    )["render"].clamp(0, 1)
    stage_link_rgb = None
    stage_link_rgb_float = None
    stage_link_ok = None
    if l2_render_input and os.path.isfile(l2_render_input):
        saved = _load_png(l2_render_input, gaussians.get_xyz.device)
        stage_link_rgb_float = image_l1(rendered_4x, saved)
        rendered_path = os.path.join(output_dir, "l1_render_4x.png")
        save_tensor_image(rendered_4x, rendered_path)
        rerendered = _load_png(rendered_path, gaussians.get_xyz.device)
        stage_link_rgb = image_l1(rerendered, saved)
        stage_link_ok = stage_link_rgb <= STAGE_LINK_L1_MAX

    l1_in_l2 = None
    l2_completed = None
    l2_vs_l1 = None
    if l2_checkpoint:
        load_lod_onto_bundle(bundle, os.path.abspath(l2_checkpoint), device=device)
        snap_l2 = snapshot_levels(bundle, (0, 1, 2))
        l2_vs_l1 = snapshot_mismatches(
            {"levels": {"0": snap_l1["levels"]["0"], "1": snap_l1["levels"]["1"]}, "appearance": {
                "ok": True,
                "layer_embeddings": (snap_l1.get("appearance") or {}).get("layer_embeddings") or [],
            }},
            {"levels": {"0": snap_l2["levels"]["0"], "1": snap_l2["levels"]["1"]}, "appearance": snap_l2["appearance"]},
        )
        l2_completed = completed_layers_frozen(bundle, up_to_level=2)
        l1_in_l2 = snap_l2["levels"]["1"]

    freeze_ok = (
        not l0_vs_l1
        and bool(l1_completed["ok"])
        and (l2_vs_l1 is None or not l2_vs_l1)
        and (l2_completed is None or bool(l2_completed["ok"]))
        and bool(appearance_freeze_report(bundle.appearance)["ok"])
    )
    resume_ok = bool(os.path.isfile(l1_checkpoint)) and (not l2_checkpoint or os.path.isfile(l2_checkpoint))
    payload = {
        "ok": bool(freeze_ok and resume_ok and (stage_link_ok is not False)),
        "freeze_ok": freeze_ok,
        "resume_ok": resume_ok,
        "stage_link_ok": stage_link_ok,
        "stage_link_rgb_l1": stage_link_rgb,
        "stage_link_rgb_l1_float": stage_link_rgb_float,
        "stage_link_threshold": STAGE_LINK_L1_MAX,
        "import_l0_max_abs": import_delta,
        "l0_changed_after_l1_load": l0_vs_l1,
        "l1_in_l2_mismatches": l2_vs_l1,
        "l1_completed": l1_completed,
        "l2_completed": l2_completed,
        "start_checkpoint": file_identity(start_checkpoint),
        "l1_checkpoint": file_identity(l1_checkpoint),
        "l2_checkpoint": file_identity(l2_checkpoint) if l2_checkpoint else None,
        "roi": roi_dict(roi.center_x, roi.center_y, roi.width, roi.height),
        "view_index": int(view_index),
        "view_name": camera_image_stem(getattr(base_camera, "image_name", "")),
        "zoom_l1": float(zoom_l1),
        "zoom_l2": float(zoom_l2),
        "n_l1": None if snap_l1["levels"]["1"] is None else snap_l1["levels"]["1"]["n"],
        "n_l1_in_l2": None if l1_in_l2 is None else l1_in_l2["n"],
    }
    write_json(os.path.join(output_dir, "recover.json"), payload)
    return payload
