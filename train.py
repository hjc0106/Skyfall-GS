#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import copy
import shutil
import gc
import time
from io import BytesIO
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from random import randint
from utils.general_utils import get_expon_lr_func
from utils.loss_utils import l1_loss, ssim
from torchmetrics.functional.regression import pearson_corrcoef
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, IDUParams, RASTERIZER_BACKENDS, has_explicit_flag
from utils.camera_utils import gen_idu_orbit_camera, cameraList_from_camInfos
from scene.dataset_readers import CameraInfo

import json

from PIL import Image
from submodules.MoGe.idu_depth import MoGeIDU

from refinement.stage2_gaussianzoom import (
    prepare_stage2_inputs,
    refine_stage2_inputs,
    validate_stage2_options,
)
from refinement.flowedit_stage2 import (
    build_flowedit_views_manifest,
    load_flowedit_views_manifest,
    refine_flowedit_views,
    render_flowedit_views,
    validate_flowedit_options,
)
from refinement.types import RenderBundle

from utils.compact_retention import (
    build_stage_comparison,
    finalize_episode,
    prune_stage_scratch,
    retire_prior_iteration,
)

# fused SSIM, for faster training

from fused_ssim import fused_ssim

# from utils.gpu_utils import GPUManager

import lpips
import math

from torchvision.transforms.functional import to_pil_image

try:
    from tensorboardX import SummaryWriter
    from tensorboardX.proto.summary_pb2 import Summary
    from tensorboardX.summary import _clean_tag
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


@torch.no_grad()
def create_offset_gt(image, offset):
    height, width = image.shape[1:]
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords).cuda()
    
    id_coords = id_coords.permute(1, 2, 0) + offset
    id_coords[..., 0] /= (width - 1)
    id_coords[..., 1] /= (height - 1)
    id_coords = id_coords * 2 - 1
    
    image = torch.nn.functional.grid_sample(image[None], id_coords[None], align_corners=True, padding_mode="border")[0]
    return image

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if opt.use_lpips_loss:
        lpips_loss_fn = lpips.LPIPS(net=opt.lpips_net)
        for param in lpips_loss_fn.parameters():
            param.requires_grad = False
        lpips_loss_fn.cuda()
        print("Initialized LPIPS loss")
    first_iter = 0
    # cfg_args is written from the ModelParams group only; carry the
    # rasterizer backend so later renders/resume can honor it.
    dataset.rasterizer_backend = pipe.rasterizer_backend
    tb_writer = prepare_output_and_logger(dataset)
    moge_standalone = (
        MoGeIDU(os.path.join(dataset.model_path, "depth_tmp"), "cuda:0", 60.0)
        if opt.lambda_pseudo_depth > 0 else None
    )
    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs,
        dataset.appearance_embedding_dim
    )
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt, num_train_cameras=len(scene.getTrainCameras()))
    if checkpoint:
        print("Restoring model from checkpoint")
        # original implementation
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)
        # set correct xyz lr scheduler
        opt.position_lr_max_steps = opt.iterations
        opt.densify_until_iter = opt.iterations
        opt.densify_from_iter = 0
        gaussians.xyz_scheduler_args = get_expon_lr_func(lr_init=opt.position_lr_init * gaussians.spatial_lr_scale,
                                                        lr_final=opt.position_lr_final * gaussians.spatial_lr_scale,
                                                        lr_delay_mult=opt.position_lr_delay_mult,
                                                        max_steps=opt.position_lr_max_steps)
        print("Restored model from checkpoint at iteration {}".format(first_iter))


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    testCameras = scene.getTestCameras().copy()
    allCameras = trainCameras + testCameras

    num_train_cams = len(trainCameras)
    
    # highresolution index
    highresolution_index = []
    for index, camera in enumerate(trainCameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    gaussians.compute_3D_filter(cameras=trainCameras) # + pseudoCameras)

    viewpoint_stack = None
    pseudo_stack = None
    ema_loss_for_log = 0.0
    ema_depth_loss_for_log = 0.0
    ema_opacity_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    opacity_cooldown_iter = None
    origin_lambda_opacity = opt.lambda_opacity
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        if opacity_cooldown_iter is not None:
            if opacity_cooldown_iter > 0:
                opacity_cooldown_iter -= 1
            else:
                opacity_cooldown_iter = None
                opt.lambda_opacity = origin_lambda_opacity
                print(f"Restore lambda opacity to {opt.lambda_opacity}")


        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Pick a random high resolution camera
        if random.random() < 0.3 and dataset.sample_more_highres:
            viewpoint_cam = trainCameras[highresolution_index[randint(0, len(highresolution_index)-1)]]
            
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        #TODO ignore border pixels
        if dataset.ray_jitter:
            subpixel_offset = torch.rand((int(viewpoint_cam.image_height), int(viewpoint_cam.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
            # subpixel_offset *= 0.0
        else:
            subpixel_offset = None

        render_pkg = render(
            viewpoint_cam, 
            gaussians, 
            pipe, 
            background, 
            kernel_size=dataset.kernel_size, 
            subpixel_offset=subpixel_offset
        )
        image, depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["render_depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        mask = viewpoint_cam.original_mask.cuda()
        gt_image = mask * viewpoint_cam.original_image.cuda()
        gt_depth = mask * viewpoint_cam.original_depth.cuda()

        image = mask * image
        depth = mask * depth
        
        # sample gt_image with subpixel offset
        if dataset.resample_gt_image:
            gt_image = create_offset_gt(gt_image, subpixel_offset)

        Ll1 = l1_loss(image, gt_image)
        if opt.use_lpips_loss:
            lpips_value = lpips_loss_fn(image.unsqueeze(0)*2.0-1.0,  gt_image.unsqueeze(0)*2.0-1.0).mean()
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * lpips_value
        else:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        depth_loss = 0.0
        if opt.lambda_depth > 0:
            gt_depth = gt_depth.reshape(-1, 1)
            depth = depth.reshape(-1, 1)
            nan_inf_mask = torch.isnan(depth) | torch.isinf(depth) | torch.isnan(gt_depth) | torch.isinf(gt_depth)
            depth[nan_inf_mask] = 0.0
            gt_depth[nan_inf_mask] = 0.0
            depth_loss += depth_loss_func(gt_depth, depth)

            loss += opt.lambda_depth * depth_loss
        
        opacity_loss = 0.0
        if opt.lambda_opacity > 0:
            # Get each gaussians' opacity and use cross entropy loss
            opacity = gaussians.get_opacity.clamp(1.0e-3, 1.0 - 1.0e-3)
            opacity_loss = torch.nn.functional.binary_cross_entropy(opacity, opacity)
            # opacity_loss = torch.mean(-opacity * torch.log(opacity + 1e-6))
            loss += opt.lambda_opacity * opacity_loss


        if opt.lambda_pseudo_depth > 0 and iteration % opt.sample_pseudo_interval == 0 and iteration > opt.start_sample_pseudo and iteration < opt.end_sample_pseudo:
            if not pseudo_stack:
                # sample elevation from 80 to 45
                elevation = (opt.end_sample_pseudo - iteration) / (opt.end_sample_pseudo - opt.start_sample_pseudo) * (80 - 45) + 45
                # For Satellite
                radius = (opt.end_sample_pseudo - iteration) / (opt.end_sample_pseudo - opt.start_sample_pseudo) * (300 - 250) + 250
                # For GES
                # radius = (opt.end_sample_pseudo - iteration) / (opt.end_sample_pseudo - opt.start_sample_pseudo) * (100 - 50) + 50
                pseudo_stack = generate_pseudo_cams(dataset, opt.num_pseudo_cams, num_train_cams, elevation, radius, target_std=opt.target_std)
            
            pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))
            render_pkg = render(
                pseudo_cam, 
                gaussians, 
                pipe, 
                background, 
                kernel_size=dataset.kernel_size, 
                subpixel_offset=subpixel_offset
            )
            render_image, render_depth = render_pkg["render"], render_pkg["render_depth"]
            
            render_image_pil = to_pil_image(render_image)
            moge_depth = moge_standalone.run([render_image_pil], pbar=False)[0]
            gt_depth = torch.tensor(moge_depth).to(render_depth.device)

            gt_depth = gt_depth.reshape(-1, 1)
            render_depth = render_depth.reshape(-1, 1)
            depth_loss_pseudo = depth_loss_func(gt_depth, render_depth)

            if torch.isnan(depth_loss_pseudo).sum() == 0:
                loss_scale = min((iteration - args.start_sample_pseudo) / 500., 1)
                loss += loss_scale * opt.lambda_pseudo_depth * depth_loss_pseudo
                depth_loss += depth_loss_pseudo

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if (opt.lambda_depth > 0 or opt.lambda_pseudo_depth > 0) and not isinstance(depth_loss, float):
                if math.isnan(ema_depth_loss_for_log):
                    ema_depth_loss_for_log = depth_loss.item()
                else:
                    ema_depth_loss_for_log = 0.4 * depth_loss.item() + 0.6 * ema_depth_loss_for_log
            else:
                ema_depth_loss_for_log = 0
            if opt.lambda_opacity > 0:
                ema_opacity_loss_for_log = 0.4 * opacity_loss.item() + 0.6 * ema_opacity_loss_for_log
            else:
                ema_opacity_loss_for_log = 0.6 * ema_opacity_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}", 
                    "Depth Loss": f"{ema_depth_loss_for_log:.{7}f}",
                    "Opacity Loss": f"{ema_opacity_loss_for_log:.{7}f}",
                    "# of GS": f"{gaussians.get_xyz.shape[0]}"
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, dataset.kernel_size))

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    size_threshold = opt.size_threshold
                    # size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    gaussians.compute_3D_filter(cameras=trainCameras) # + pseudoCameras)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    opt.lambda_opacity = 0.01
                    opacity_cooldown_iter = 500
                    print(f"Turn off opacity regularization for {opacity_cooldown_iter} iterations")



            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras) # + pseudoCameras)
        
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

def idu_episode_dirname(episode_idx: int, elevation, radius) -> str:
    if elevation is None or radius is None:
        # External supervision episodes carry no orbit elevation/radius.
        return f"episode_{episode_idx:02d}"
    if isinstance(elevation, list) or isinstance(radius, list):
        return f"episode_{episode_idx:02d}"
    return f"episode_{episode_idx:02d}_e{elevation:g}_r{radius:g}"


def write_idu_compare_html(episode_dir: str, n_images: int, episode_idx: int, elevation, radius, middle_label: str = "GaussianZoom / DLoRAL") -> None:
    rows = []
    for idx in range(n_images):
        name = f"{idx:05d}.png"
        rows.append(
            "<tr>"
            f"<td>{idx:05d}</td>"
            f'<td><img src="render/{name}" /></td>'
            f'<td><img src="render_refine/{name}" /></td>'
            f'<td><img src="render_after_train/{name}" /></td>'
            "</tr>"
        )
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8" />
<title>IDU episode {episode_idx:02d}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 16px; font-family: ui-sans-serif, system-ui, sans-serif; background: #111; color: #eee; }}
  img {{ width: 280px; height: auto; background: #000; }}
  table {{ border-collapse: collapse; }}
  td, th {{ padding: 6px; vertical-align: top; }}
</style></head><body>
<h2>Episode {episode_idx:02d} · e={elevation} · r={radius}</h2>
<p>left: 3DGS render · middle: {middle_label} · right: after this episode's 3DGS training (filled later)</p>
<table>
<tr><th>id</th><th>render</th><th>render_refine</th><th>render_after_train</th></tr>
{''.join(rows)}
</table></body></html>
"""
    with open(os.path.join(episode_dir, "compare.html"), "w", encoding="utf-8") as f:
        f.write(html)


def update_idu_root_index(model_path: str, record: dict) -> None:
    index_path = os.path.join(model_path, "idu", "manifest.json")
    payload = {"episodes": []}
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    episodes = [e for e in payload.get("episodes", []) if e.get("episode_idx") != record["episode_idx"]]
    episodes.append(record)
    episodes.sort(key=lambda e: e["episode_idx"])
    payload["episodes"] = episodes
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    links = [
        f'<li><a href="{e["dirname"]}/compare.html">episode {e["episode_idx"]:02d}</a> '
        f'(e={e["elevation"]}, r={e["radius"]}, n={e["n_images"]})</li>'
        for e in episodes
    ]
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8' /><title>IDU episodes</title></head>"
        "<body style='font-family:sans-serif;background:#111;color:#eee;padding:20px'>"
        "<h2>Skyfall-GS Stage 2 IDU</h2><ul>"
        + "".join(links)
        + "</ul></body></html>"
    )
    with open(os.path.join(model_path, "idu", "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


@torch.no_grad()
def save_idu_after_train_renders(views, gaussians, pipeline, background, kernel_size, save_dir: str, *, description: str = "IDU after-train render") -> None:
    os.makedirs(save_dir, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc=description)):
        rendering = render(view, gaussians, pipeline, background, kernel_size=kernel_size, testing=True)["render"]
        img = rendering.detach().cpu().permute(1, 2, 0).numpy()
        Image.fromarray((img * 255 + 0.5).clip(0, 255).astype(np.uint8)).save(
            os.path.join(save_dir, f"{idx:05d}.png")
        )


@torch.no_grad()
def generate_idu_training_set(
    dataset: ModelParams,
    checkpoint_path: str,
    pipeline: PipelineParams,
    targets,
    elevation,
    radius,
    idu_num_cams: int,
    idu_num_samples_per_view: int,
    *,
    options,
    height: int = 1024,
    width: int = 1024,
    fov_x: float = 60.0,
    episode_idx: int = 0,
):
    """Refine every curriculum view with the selected Stage 2 backend, then infer depth.

    Rendering (and geometry, for the GaussianZoom backend), generative
    refinement, and MoGe depth are separate phases. Only CPU CameraInfo
    metadata and disk-backed inputs cross phases.
    """
    refine_backend = getattr(options, "idu_refine_backend", "flowedit")
    if refine_backend == "flowedit":
        validate_flowedit_options(options)
    elif refine_backend == "gaussianzoom":
        validate_stage2_options(options)
    else:
        raise ValueError(f"Unknown idu_refine_backend: {refine_backend!r}")
    if idu_num_cams < 1 or idu_num_samples_per_view < 1:
        raise ValueError("IDU needs positive camera and per-view sample counts")

    episode_name = idu_episode_dirname(episode_idx, elevation, radius)
    episode_root = os.path.join(dataset.model_path, "idu", episode_name)
    os.makedirs(episode_root, exist_ok=True)
    if isinstance(elevation, list) or isinstance(radius, list):
        if not options.idu_no_curriculum or not isinstance(elevation, list) or not isinstance(radius, list):
            raise ValueError("Multiple IDU elevations/radii require idu_no_curriculum")
        if not elevation or len(elevation) != len(radius):
            raise ValueError("IDU elevation and radius lists must have equal nonzero length")
        courses = list(zip(elevation, radius))
    else:
        courses = [(elevation, radius)]

    # Keep the original extrinsic camera course, but render each pose once.
    # Independent diffusion samples are expanded only after view-pair selection.
    unique_infos = []
    for ele, rad in courses:
        for target in targets:
            unique_infos.extend(gen_idu_orbit_camera(
                target, ele, rad, idu_num_cams, 1, height, width, fov_x,
            ))
    if len(courses) > 1:
        unique_infos = random.sample(unique_infos, len(unique_infos) // len(courses))
    unique_infos = [
        info._replace(uid=1000 + index, image_name=f"idu_view_{index:05d}.png")
        for index, info in enumerate(unique_infos)
    ]
    if not unique_infos:
        raise ValueError("The IDU camera course produced no views")

    gaussians = GaussianModel(
        dataset.sh_degree, dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs, dataset.appearance_embedding_dim,
    )
    model_params, first_iter = torch.load(checkpoint_path, weights_only=False)
    scene = Scene(
        dataset, gaussians, load_iteration=first_iter, shuffle=False,
        ply_path=os.path.dirname(checkpoint_path),
    )
    # Scene loads a PLY; restore the authoritative checkpoint afterwards.
    gaussians.load_from_checkpoints(model_params)
    generated_dataset = copy.copy(dataset)
    generated_dataset.resolution = 1
    views = cameraList_from_camInfos(unique_infos, 1, generated_dataset, is_idu=True)
    gaussians.compute_3D_filter(cameras=list(scene.getTrainCameras()) + views)
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda",
    )
    if refine_backend == "gaussianzoom":
        context_images = [
            (camera, to_pil_image(camera.original_image.detach().cpu()))
            for camera in scene.getTrainCameras()
        ]
    else:
        context_images = []

    def render_view(camera):
        package = render(
            camera, gaussians, pipeline, background,
            kernel_size=dataset.kernel_size, testing=True,
        )
        return RenderBundle(
            rgb=package["render"], depth=package["render_depth"],
            alpha=package["render_alpha"], camera=camera,
        )

    if refine_backend == "flowedit":
        # Pure FlowEdit: render each unique pose once, then repair the renders
        # with the original FlowEdit sampler. No geometry flow, Qwen prompts
        # or DLoRAL models are involved.
        prepared = render_flowedit_views(
            views, render_view,
            checkpoint_path=checkpoint_path, episode_dir=episode_root,
            episode_idx=episode_idx, options=options,
        )
        # Release the scene, Gaussians and camera tensors before FLUX loads.
        del render_view, context_images, views, scene, gaussians, model_params, background
        gc.collect()
        torch.cuda.empty_cache()
        final_imgs = refine_flowedit_views(prepared, options=options)
    else:
        prepared = prepare_stage2_inputs(
            views, context_images, render_view,
            checkpoint_path=checkpoint_path, episode_dir=episode_root,
            episode_idx=episode_idx, options=options,
        )
        del render_view, context_images, views, scene, gaussians, model_params, background
        gc.collect()
        torch.cuda.empty_cache()
        final_imgs = refine_stage2_inputs(prepared, options=options)
    expected_count = len(unique_infos) * idu_num_samples_per_view
    if len(final_imgs) != expected_count:
        raise RuntimeError(f"Expected {expected_count} refined IDU samples, got {len(final_imgs)}")
    if any(image.size != (width, height) for image in final_imgs):
        raise ValueError("Stage2 refinement must preserve the IDU camera raster")

    depth_path = os.path.join(episode_root, "render_depth")
    os.makedirs(depth_path, exist_ok=True)
    moge = MoGeIDU(depth_path, device="cuda:0", fov_x=fov_x)
    try:
        depths = moge.run(final_imgs)
    finally:
        del moge
        gc.collect()
        torch.cuda.empty_cache()
    if len(depths) != expected_count:
        raise RuntimeError("MoGe returned a different number of depths than refined IDU images")

    final_idu_infos = []
    for view_index, info in enumerate(unique_infos):
        for sample_index in range(idu_num_samples_per_view):
            index = view_index * idu_num_samples_per_view + sample_index
            np.save(os.path.join(depth_path, f"{index:05d}.npy"), depths[index])
            final_idu_infos.append(info._replace(
                uid=1000 + index, image=final_imgs[index], depth=depths[index],
                image_name=f"idu_view_{view_index:05d}_sample_{sample_index:02d}.png",
                mask=None,
            ))
    final_cameras = cameraList_from_camInfos(
        final_idu_infos, 1, generated_dataset, is_idu=True,
    )
    refinement_method = "flowedit" if refine_backend == "flowedit" else "gaussianzoom_dloral"
    refine_label = "FlowEdit" if refine_backend == "flowedit" else "GaussianZoom / DLoRAL"
    meta = {
        "episode_idx": episode_idx,
        "dirname": episode_name,
        "elevation": elevation,
        "radius": radius,
        "n_views": len(unique_infos),
        "samples_per_view": idu_num_samples_per_view,
        "n_images": expected_count,
        "render_dir": os.path.join(episode_root, "render"),
        "refine_dir": os.path.join(episode_root, "render_refine"),
        "depth_dir": depth_path,
        "checkpoint_used": checkpoint_path,
        "refinement_method": refinement_method,
        "refinement_inputs": prepared,
        "cameras": [
            {"index": index, "uid": info.uid, "image_name": info.image_name}
            for index, info in enumerate(final_idu_infos)
        ],
    }
    with open(os.path.join(episode_root, "episode_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    if refine_backend == "flowedit":
        with open(os.path.join(episode_root, "flowedit_views.json"), "w", encoding="utf-8") as handle:
            json.dump(
                build_flowedit_views_manifest(
                    prepared, checkpoint=checkpoint_path,
                    episode_idx=episode_idx, samples_per_view=idu_num_samples_per_view,
                ),
                handle, indent=2, ensure_ascii=False,
            )
    write_idu_compare_html(episode_root, expected_count, episode_idx, elevation, radius,
                           middle_label=refine_label)
    update_idu_root_index(dataset.model_path, {
        "episode_idx": episode_idx, "dirname": episode_name,
        "elevation": elevation, "radius": radius, "n_images": expected_count,
        "refinement_method": refinement_method,
    })
    print(f"Saved {refine_label} IDU episode artifacts to {episode_root}")
    return final_cameras

@torch.no_grad()
def load_idu_supervision_training_set(
    dataset: ModelParams,
    manifest_path: str,
    *,
    episode_idx: int,
    start_checkpoint: str,
):
    """Build native IDU cameras from an external ``skyfall_flowedit_views`` manifest.

    The manifest lists fixed FlowEdit-repaired targets rendered from the ORIGINAL
    source checkpoint. They are consumed as-is (no re-render, no extra FlowEdit
    pass, no resize); only the MoGe depths are newly inferred, reusing the
    native ``render_depth`` persistence. Cameras keep the manifest snapshot's
    R/T/FoV/principal point and join the standard full-parameter IDU episode as
    ``scene.train_idu_cameras[1.0]``, where the native render path applies the
    appearance-6 embedding to synthetic views exactly as the built-in curriculum.
    """
    manifest = load_flowedit_views_manifest(
        manifest_path, expected_checkpoint=start_checkpoint)
    views = manifest["views"]
    episode_name = idu_episode_dirname(episode_idx, None, None)
    episode_root = os.path.join(dataset.model_path, "idu", episode_name)
    os.makedirs(episode_root, exist_ok=True)
    render_dir = os.path.join(episode_root, "render")
    refine_dir = os.path.join(episode_root, "render_refine")
    depth_path = os.path.join(episode_root, "render_depth")
    for directory in (render_dir, refine_dir, depth_path):
        os.makedirs(directory, exist_ok=True)

    width = int(views[0]["camera"]["image_width"])
    height = int(views[0]["camera"]["image_height"])
    fov_x_radians = float(views[0]["camera"]["fov_x"])

    images = []
    for index, view in enumerate(views):
        image = Image.open(view["image_path"]).convert("RGB")
        if image.size != (width, height):
            raise ValueError(
                f"Supervision image {view['image_path']} is {image.size}, expected {(width, height)}"
            )
        target_path = os.path.join(refine_dir, f"{index:05d}.png")
        shutil.copyfile(view["image_path"], target_path)
        images.append(image)

    moge = MoGeIDU(depth_path, device="cuda:0", fov_x=math.degrees(fov_x_radians))
    try:
        depths = moge.run(images)
    finally:
        del moge
        gc.collect()
        torch.cuda.empty_cache()
    if len(depths) != len(views):
        raise RuntimeError("MoGe returned a different number of depths than supervision images")
    # Same render_depth/*.npy persistence as the native generation path.
    for index, depth in enumerate(depths):
        np.save(os.path.join(depth_path, f"{index:05d}.npy"), depth)

    final_idu_infos = []
    for index, (view, image, depth) in enumerate(zip(views, images, depths)):
        camera = view["camera"]
        final_idu_infos.append(CameraInfo(
            uid=1000 + index,
            R=np.array(camera["R"], dtype=np.float64),
            T=np.array(camera["T"], dtype=np.float64),
            FovY=float(camera["fov_y"]),
            FovX=float(camera["fov_x"]),
            cx=float(camera["cx"]),
            cy=float(camera["cy"]),
            image=image,
            image_path=view["image_path"],
            image_name=view["id"],
            depth=depth,
            mask=None,
            width=width,
            height=height,
        ))
    generated_dataset = copy.copy(dataset)
    generated_dataset.resolution = 1
    final_cameras = cameraList_from_camInfos(
        final_idu_infos, 1, generated_dataset, is_idu=True,
    )

    meta = {
        "episode_idx": episode_idx,
        "dirname": episode_name,
        "elevation": None,
        "radius": None,
        "supervision_manifest": os.path.abspath(manifest_path),
        "manifest_episode_idx": manifest["episode_idx"],
        "checkpoint_used": manifest["checkpoint"],
        "n_views": len(views),
        "samples_per_view": 1,
        "n_images": len(views),
        "width": width,
        "height": height,
        "render_dir": render_dir,
        "refine_dir": refine_dir,
        "depth_dir": depth_path,
        "refinement_method": "flowedit_supervision",
        "cameras": [
            {"index": index, "uid": 1000 + index, "image_name": view["id"]}
            for index, view in enumerate(views)
        ],
    }
    with open(os.path.join(episode_root, "episode_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(episode_root, "flowedit_views.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    write_idu_compare_html(
        episode_root, len(views), episode_idx, None, None,
        middle_label="FlowEdit supervision",
    )
    update_idu_root_index(dataset.model_path, {
        "episode_idx": episode_idx, "dirname": episode_name,
        "elevation": None, "radius": None, "n_images": len(views),
        "refinement_method": "flowedit_supervision",
    })
    print(f"Loaded {len(views)} FlowEdit supervision views for IDU episode {episode_idx} "
          f"from {manifest_path}")
    return final_cameras


@torch.no_grad()
def generate_pseudo_cams(
    dataset : ModelParams,
    num_cams: int,
    num_train_cams: int,
    elevation: float=80.0,
    radius: float=300.0,
    target_std: float=64.0
):
    idu_cam_infos = []
    for _ in range(num_cams):
        mean = torch.tensor([0., 0.])
        std = torch.tensor([target_std, target_std])
        xy = torch.normal(mean, std)
        z = torch.tensor([0])
        target = torch.cat((xy, z)).tolist()
        gen_cams = gen_idu_orbit_camera(
            target,
            elevation=elevation,
            radius=radius,
            num_cams=12,
            num_samples=1,
            height=1024,
            width=1024,
            fov=60.0,
            use_new_id=False,
            num_train_cams=num_train_cams
        )
        gen_cam = random.choice(gen_cams)
        idu_cam_infos.append(gen_cam)

    print(f"Generated {len(idu_cam_infos)} pseudo cameras with e={elevation:.2f} r={radius:.2f}")

    final_idu_cam_infos = []
    # Save to cam_infos
    for idx, cam_info in enumerate(idu_cam_infos):
        final_cam_info = CameraInfo(
            uid=cam_info.uid, R=cam_info.R, T=cam_info.T, 
            FovY=cam_info.FovY, FovX=cam_info.FovX, 
            cx=0, cy=0,
            image=Image.new("1", (cam_info.width, cam_info.height), (0)), image_path=cam_info.image_path,
            image_name=cam_info.image_name, 
            depth=None, mask=None,
            width=cam_info.width, height=cam_info.height
        )
        final_idu_cam_infos.append(final_cam_info)

    final_cam_lists = cameraList_from_camInfos(final_idu_cam_infos, 1, dataset, is_pseudo_cam=True)
    

    return final_cam_lists

def training_idu_episode(
        dataset, opt, pipe, 
        checkpoint_path,
        targets, elevation, radius, fov,
        idu_num_cams, idu_num_samples_per_view,
        episode_idx: int = 0,
        supervision_cameras=None,
    ):
    # Refine all extrapolated curriculum views, then optimize the full 3DGS.
    if supervision_cameras is not None:
        # External ``skyfall_flowedit_views`` supervision: the cameras were built
        # from the manifest and are reused as-is for this native IDU episode.
        idu_cam_list = supervision_cameras
    else:
        # Generate IDU training set
        if not opt.idu_no_curriculum:
            assert isinstance(elevation, float) and isinstance(radius, float)
        else:
            assert isinstance(elevation, list) and isinstance(radius, list), "Elevation and radius should be list when no_curriculum is True"
        idu_cam_list = generate_idu_training_set(
            dataset, checkpoint_path, pipe, targets, elevation, radius,
            idu_num_cams, idu_num_samples_per_view,
            options=opt, height=opt.idu_render_size, width=opt.idu_render_size,
            fov_x=fov, episode_idx=episode_idx,
        )

    # load Gaussians and scene
    # cfg_args is written from the ModelParams group only; carry the
    # rasterizer backend so later renders/resume can honor it.
    dataset.rasterizer_backend = pipe.rasterizer_backend
    tb_writer = prepare_output_and_logger(dataset)
    if opt.use_lpips_loss:
        lpips_loss_fn = lpips.LPIPS(net=opt.lpips_net)
        for param in lpips_loss_fn.parameters():
            param.requires_grad = False
        lpips_loss_fn.cuda()
        print("Initialized LPIPS loss")
    moge_standalone = (
        MoGeIDU(os.path.join(dataset.model_path, "depth_tmp"), "cuda:0", 60.0)
        if opt.lambda_pseudo_depth > 0 else None
    )
    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs,
        dataset.appearance_embedding_dim
    )
    scene = Scene(dataset, gaussians)
    # set IDU cameras
    scene.train_idu_cameras[1.0] = idu_cam_list
    gaussians.training_setup(
        opt,
        num_train_cameras=len(scene.getTrainCameras()),
        from_scratch=False  
        # NOTE: set appearacne lr to zero and set the xyz lr scheduler
    )
    if checkpoint_path:
        print(f"Restoring model from checkpoint {checkpoint_path}")
        # original implementation
        (model_params, first_iter) = torch.load(checkpoint_path, weights_only=False)
        gaussians.restore(model_params, opt, iterative_datasets_update=True)
        print("Restored model from checkpoint at iteration {}".format(first_iter))
        opt.iterations = first_iter + opt.idu_episode_iterations  # TODO: make this a parameter
        idu_densify_until_iter = first_iter + opt.idu_densify_until_iter
        assert idu_densify_until_iter < opt.iterations
        print(f"Set iterations to {opt.iterations}, densify until {idu_densify_until_iter}")
    else:
        raise ValueError("Checkpoint is required for iterative datasets update")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    testCameras = scene.getTestCameras().copy()
    trainIDUCameras = scene.getTrainIDUCameras().copy()
    allCameras = trainCameras + trainIDUCameras + testCameras

    num_train_cams = len(trainCameras)
    
    # highresolution index
    highresolution_index = []
    for index, camera in enumerate(trainCameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    gaussians.compute_3D_filter(cameras=trainCameras + trainIDUCameras)
    if supervision_cameras is not None:
        before_dir = os.path.join(
            dataset.model_path, "idu", idu_episode_dirname(episode_idx, None, None), "render"
        )
        save_idu_after_train_renders(
            trainIDUCameras, gaussians, pipe, background, dataset.kernel_size,
            before_dir, description="IDU before-train render",
        )

    viewpoint_train_stack = None
    viewpoint_train_idu_stack = None
    pseudo_stack = None
    ema_loss_for_log = 0.0
    ema_depth_loss_for_log = 0.0
    ema_opacity_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    episode_first_iteration = first_iter
    episode_started = time.time()
    episode_metrics = {}
    first_iter += 1
    testing_iterations = [iter for iter in range(first_iter, opt.iterations + 2, opt.idu_testing_interval)][1:] # skip first iter
    if opt.iterations not in testing_iterations:
        testing_iterations.append(opt.iterations)
    checkpoint_iterations = [opt.iterations]


    checkpoint_path = None

    opacity_cooldown_iter = None
    origin_lambda_opacity = opt.lambda_opacity

    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        
        if opacity_cooldown_iter is not None:
            if opacity_cooldown_iter > 0:
                opacity_cooldown_iter -= 1
            else:
                opacity_cooldown_iter = None
                opt.lambda_opacity = origin_lambda_opacity
                print(f"Restore lambda opacity to {opt.lambda_opacity}")

        iter_start.record()

        gaussians.update_learning_rate(iteration - first_iter)  # NOTE: modified for IDU

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        # Pick a random Camera
        idu_viewpoint = None

        if iteration + opt.idu_iter_full_train <= opt.iterations and random.random() < opt.idu_train_ratio:
            idu_viewpoint = True
            if not viewpoint_train_idu_stack:
                viewpoint_train_idu_stack = scene.getTrainIDUCameras().copy()
            viewpoint_cam = viewpoint_train_idu_stack.pop(randint(0, len(viewpoint_train_idu_stack)-1))
            lambda_depth = opt.lambda_depth
        else:
            idu_viewpoint = False
            if not viewpoint_train_stack:
                viewpoint_train_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_train_stack.pop(randint(0, len(viewpoint_train_stack)-1))
            lambda_depth = 0
        

        #TODO ignore border pixels
        if dataset.ray_jitter:
            subpixel_offset = torch.rand((int(viewpoint_cam.image_height), int(viewpoint_cam.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
            # subpixel_offset *= 0.0
        else:
            subpixel_offset = None

        render_pkg = render(
            viewpoint_cam, 
            gaussians, 
            pipe, 
            background, 
            kernel_size=dataset.kernel_size, 
            subpixel_offset=subpixel_offset,
            testing=bool(idu_viewpoint)
            # Use the same fixed IDU appearance policy as the generation pass.
        )
        image, depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["render_depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        mask = viewpoint_cam.original_mask.cuda()
        gt_image = mask * viewpoint_cam.original_image.cuda()
        gt_depth = mask * viewpoint_cam.original_depth.cuda()

        image = mask * image
        depth = mask * depth
        
        # sample gt_image with subpixel offset
        loss = None
        if dataset.resample_gt_image:
            gt_image = create_offset_gt(gt_image, subpixel_offset)
        Ll1 = l1_loss(image, gt_image)
        if opt.use_lpips_loss:
            lpips_value = lpips_loss_fn(image.unsqueeze(0)*2.0-1.0, gt_image.unsqueeze(0)*2.0-1.0).mean()
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * lpips_value
        else:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)


        depth_loss = 0.0
        if lambda_depth > 0:
            gt_depth = gt_depth.reshape(-1, 1)
            depth = depth.reshape(-1, 1)
            nan_inf_mask = torch.isnan(depth) | torch.isinf(depth) | torch.isnan(gt_depth) | torch.isinf(gt_depth)
            depth = depth[~nan_inf_mask]
            gt_depth = gt_depth[~nan_inf_mask]
            depth_loss += depth_loss_func(gt_depth, depth)
            if torch.isnan(depth_loss).sum() == 0:
                if loss:
                    loss += lambda_depth * depth_loss
                else:
                    loss = lambda_depth * depth_loss
            else:
                depth_loss = 0.0

            # loss += lambda_depth * depth_loss
        if opt.lambda_pseudo_depth > 0 and iteration % opt.sample_pseudo_interval == 0:
            if not pseudo_stack:
                pseudo_elevation = (first_iter + opt.idu_episode_iterations - iteration) / opt.idu_episode_iterations * (85 - 45) + 45
                pseudo_radius = (first_iter + opt.idu_episode_iterations - iteration) / opt.idu_episode_iterations * (150 - 75) + 75
                pseudo_stack = generate_pseudo_cams(dataset, opt.num_pseudo_cams, num_train_cams, pseudo_elevation, pseudo_radius)
            
            pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))
            render_pkg = render(
                pseudo_cam, 
                gaussians, 
                pipe, 
                background, 
                kernel_size=dataset.kernel_size, 
                subpixel_offset=subpixel_offset
            )
            render_image, render_depth = render_pkg["render"], render_pkg["render_depth"]
            
            render_image_pil = to_pil_image(render_image)
            moge_depth = moge_standalone.run([render_image_pil], pbar=False)[0]
            gt_depth = torch.tensor(moge_depth).to(render_depth.device)

            gt_depth = gt_depth.reshape(-1, 1)
            render_depth = render_depth.reshape(-1, 1)
            depth_loss_pseudo = depth_loss_func(gt_depth, render_depth)

            if torch.isnan(depth_loss_pseudo).sum() == 0:
                loss_scale = 1.0
                loss += loss_scale * opt.lambda_pseudo_depth * depth_loss_pseudo
                depth_loss += depth_loss_pseudo
        
        opacity_loss = 0.0
        if opt.lambda_opacity > 0:
            # Get each gaussians' opacity and use cross entropy loss
            opacity = gaussians.get_opacity.clamp(1.0e-3, 1.0 - 1.0e-3)
            opacity_loss = torch.nn.functional.binary_cross_entropy(opacity, opacity)
            # opacity_loss = torch.mean(-opacity * torch.log(opacity + 1e-6))
            if loss:
                loss += opt.lambda_opacity * opacity_loss
            else:
                loss = opt.lambda_opacity * opacity_loss

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if (lambda_depth > 0 or opt.lambda_pseudo_depth > 0) and not isinstance(depth_loss, float):
                if math.isnan(ema_depth_loss_for_log):
                    ema_depth_loss_for_log = depth_loss.item()
                else:
                    ema_depth_loss_for_log = 0.4 * depth_loss.item() + 0.6 * ema_depth_loss_for_log
            else:
                ema_depth_loss_for_log = 0
            if opt.lambda_opacity > 0:
                ema_opacity_loss_for_log = 0.4 * opacity_loss.item() + 0.6 * ema_opacity_loss_for_log
            else:
                ema_opacity_loss_for_log = 0.6 * ema_opacity_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}", 
                    "Depth Loss": f"{ema_depth_loss_for_log:.{7}f}",
                    "Opacity Loss": f"{ema_opacity_loss_for_log:.{7}f}",
                    "# of GS": f"{gaussians.get_xyz.shape[0]}"
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            report = training_report(
                tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, dataset.kernel_size),
                iterative_datasets_update=True
            )
            if report:
                episode_metrics[str(iteration)] = report

            # Densification
            if iteration < idu_densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    print("densification!")
                    size_threshold = opt.size_threshold
                    # size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    gaussians.compute_3D_filter(cameras=trainCameras + trainIDUCameras)

                if (iteration % opt.opacity_reset_interval == 0 and iteration < opt.iterations - 100) or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    opt.lambda_opacity = 0.0
                    opacity_cooldown_iter = opt.idu_opacity_cooling_iterations
                    print(f"Turn off opacity regularization for {opacity_cooldown_iter} iterations")

            if iteration % 100 == 0 and iteration > idu_densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras + trainIDUCameras)
        
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                checkpoint_path = scene.model_path + "/chkpnt" + str(iteration) + ".pth"
                torch.save((gaussians.capture(), iteration), checkpoint_path)
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

    episode_name = idu_episode_dirname(episode_idx, elevation, radius)
    after_dir = os.path.join(dataset.model_path, "idu", episode_name, "render_after_train")
    save_idu_after_train_renders(
        scene.getTrainIDUCameras(),
        gaussians,
        pipe,
        background,
        dataset.kernel_size,
        after_dir,
    )

    if getattr(opt, "compact_retention", False):
        # Panels and provenance first, then the consumed dense scratch goes away.
        summary = finalize_episode(
            dataset.model_path,
            episode_dir_name=episode_name,
            episode_idx=episode_idx,
            elevation=elevation,
            radius=radius,
            iteration_start=episode_first_iteration,
            iteration_end=opt.iterations,
            checkpoint_path=checkpoint_path,
            point_cloud_path=os.path.join(
                dataset.model_path, "point_cloud", f"iteration_{opt.iterations}",
                "point_cloud.ply"),
            metrics=episode_metrics,
            timings={
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(episode_started)),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "seconds": round(time.time() - episode_started, 1),
            },
        )
        print(
            f"[compact] episode {episode_idx:02d}: retained {summary['retained_bytes'] / 1e6:.1f} MB, "
            f"removed {summary['removed_bytes'] / 1e9:.2f} GB of consumed scratch"
        )
        del summary

    return checkpoint_path

def training_idu(dataset, opt, pipe, init_checkpoint_path):
    refine_backend = getattr(opt, "idu_refine_backend", "flowedit")
    if refine_backend == "flowedit":
        validate_flowedit_options(opt)
        print("===== IDU Stage 2 refinement backend: FlowEdit (original baseline) =====")
    elif refine_backend == "gaussianzoom":
        validate_stage2_options(opt)
        print("===== IDU Stage 2 refinement backend: GaussianZoom + DLoRAL =====")
    else:
        raise ValueError(f"Unknown idu_refine_backend: {refine_backend!r}")
    if not init_checkpoint_path or not os.path.isfile(init_checkpoint_path):
        raise ValueError("Stage2 requires an existing --start_checkpoint")
    start_checkpoint_path = init_checkpoint_path
    opt.opacity_reset_interval = opt.idu_opacity_reset_interval
    opt.idu_testing_interval = opt.idu_episode_iterations // 4
    if opt.idu_testing_interval < 1:
        raise ValueError("idu_episode_iterations must be at least 4")
    if not 0 <= opt.idu_densify_until_iter < opt.idu_episode_iterations:
        raise ValueError("idu_densify_until_iter must be within the episode")
    if not 0.0 <= opt.idu_train_ratio <= 1.0:
        raise ValueError("idu_train_ratio must be between 0 and 1")
    random.seed(opt.idu_seed)
    np.random.seed(opt.idu_seed)
    torch.manual_seed(opt.idu_seed)
    opt.idu_position_lr_max_steps = opt.idu_episode_iterations
    # extract idu params
    idu_params: IDUParams = opt.idu_params[opt.datasets_type]
    opt.idu_radius_list = idu_params.radius_list
    opt.idu_elevation_list = idu_params.elevation_list
    opt.idu_fov = idu_params.fov
    print("===== IDU Params =====")
    print(f"Datasets Type: {opt.datasets_type}")
    print(f"Radius List: {opt.idu_radius_list}")
    print(f"Elevation List: {opt.idu_elevation_list}")
    print(f"FOV: {opt.idu_fov}")
    print("======================")
    # generate targets
    x = np.linspace(-opt.idu_grid_width/2, opt.idu_grid_width/2, opt.idu_grid_size+2)
    y = np.linspace(-opt.idu_grid_height/2, opt.idu_grid_height/2, opt.idu_grid_size+2)
    # remove border
    x = x[1:-1]
    y = y[1:-1]
    xx, yy = np.meshgrid(x, y)
    targets = np.stack([xx, yy, np.zeros_like(xx)], axis=-1).reshape(-1, 3).tolist()
    assert len(targets) == opt.idu_grid_size * opt.idu_grid_size

    first_stage2_iteration = None

    def retire_superseded_pair(episode_idx: int) -> None:
        """Drop the previous episode's checkpoint/PLY once the newer pair exists.

        ``first_stage2_iteration`` protects the Stage 1 pair (chkpnt30000 lives in
        the Stage 1 directory and is never a Stage 2 artifact).
        """
        nonlocal first_stage2_iteration
        if first_stage2_iteration is None:
            first_stage2_iteration = opt.iterations
        if not getattr(opt, "compact_retention", False) or episode_idx == 0:
            return
        retired = retire_prior_iteration(
            dataset.model_path,
            prior_iteration=opt.iterations - opt.idu_episode_iterations,
            current_iteration=opt.iterations,
            protected_below=first_stage2_iteration,
        )
        if retired["removed"]:
            print(f"[compact] retired superseded pair {retired['removed']} "
                  f"({retired['bytes'] / 1e9:.2f} GB)")
        if retired["skipped"]:
            print(f"[compact] superseded-pair retirement skipped: {retired['skipped']}")

    supervision_manifest = getattr(opt, "idu_supervision_manifest", "") or ""
    episode_cap = int(getattr(opt, "idu_episode_count", 0) or 0)
    if episode_cap < 0:
        raise ValueError("idu_episode_count must be >= 0 (0 keeps the full curriculum)")
    if supervision_manifest:
        if refine_backend != "flowedit":
            raise ValueError(
                "--idu_supervision_manifest consumes plain FlowEdit-repaired views and "
                "requires --idu_refine_backend flowedit"
            )
        if episode_cap < 1:
            raise ValueError(
                "--idu_episode_count must be >= 1 when --idu_supervision_manifest is set"
            )
        print(f"===== IDU supervision manifest: {supervision_manifest} =====")

    if supervision_manifest:
        for episode_idx in range(episode_cap):
            supervision_cameras = load_idu_supervision_training_set(
                dataset, supervision_manifest,
                episode_idx=episode_idx, start_checkpoint=init_checkpoint_path,
            )
            start_checkpoint_path = training_idu_episode(
                dataset, opt, pipe,
                checkpoint_path=start_checkpoint_path,
                targets=targets, elevation=None, radius=None, fov=opt.idu_fov,
                idu_num_cams=opt.idu_num_cams,
                idu_num_samples_per_view=opt.idu_num_samples_per_view,
                episode_idx=episode_idx,
                supervision_cameras=supervision_cameras,
            )
            retire_superseded_pair(episode_idx)
    elif not opt.idu_no_curriculum:
        course = list(zip(opt.idu_radius_list, opt.idu_elevation_list))
        if episode_cap > 0:
            course = course[:episode_cap]
        for episode_idx, (radius, elevation) in enumerate(course):
            print(f"Training IDU episode {episode_idx} with elevation {elevation} and radius {radius}")
            print(f"# of IDU targets: {len(targets)}")
            start_checkpoint_path = training_idu_episode(
                dataset, opt, pipe, 
                checkpoint_path=start_checkpoint_path,
                targets=targets, elevation=elevation, radius=radius, fov=opt.idu_fov,
                idu_num_cams=opt.idu_num_cams,
                idu_num_samples_per_view=opt.idu_num_samples_per_view,
                episode_idx=episode_idx,
            )
            retire_superseded_pair(episode_idx)
    else:
        print("===== Disable IDU curriculum learning =====")
        assert opt.idu_episode_iterations == 10000, "IDU episode iterations should be 10000"
        assert opt.idu_densify_until_iter == 9000, "IDU episode iterations should be 9000"
        episodes = list(range(5))
        if episode_cap > 0:
            episodes = episodes[:episode_cap]
        for episode_idx in episodes:
            start_checkpoint_path = training_idu_episode(
                dataset, opt, pipe, 
                checkpoint_path=start_checkpoint_path,
                targets=targets, elevation=opt.idu_elevation_list, radius=opt.idu_radius_list, fov=opt.idu_fov,
                idu_num_cams=opt.idu_num_cams,
                idu_num_samples_per_view=opt.idu_num_samples_per_view,
                episode_idx=episode_idx,
            )
            retire_superseded_pair(episode_idx)

    if getattr(opt, "compact_retention", False):
        comparison = build_stage_comparison(dataset.model_path)
        if comparison["path"]:
            print(f"[compact] stage comparison panel: {comparison['path']} "
                  f"({comparison['panels']} episode panels)")
        prune_stage_scratch(dataset.model_path)
        print("[compact] Stage 2 retains the final checkpoint + matching PLY, cfg_args, "
              "cameras.json, compact logs and per-episode summaries")
        


def depth_loss_func(gt_depth, depth):
    # gt_depth = torch.nan_to_num(gt_depth, nan=0.0, posinf=0.0, neginf=0.0)
    # depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return (1 - pearson_corrcoef(gt_depth, depth)).mean()
    # return min(
    #     1 - pearson_corrcoef(gt_depth, depth),
    #     1 - pearson_corrcoef(1 / (gt_depth + 200.), depth)
    # )

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def colorize_depth_torch(depth_tensor, mask=None, normalize=True, cmap='Spectral'):
    """
    Colorize depth map using matplotlib colormap, implemented for PyTorch tensors.
    Args:
        depth_tensor: Input depth tensor [B, H, W] or [H, W]
        mask: Optional mask tensor [B, H, W] or [H, W]
        normalize: Whether to normalize the depth values
        cmap: Matplotlib colormap name
    Returns:
        CPU float RGB tensor [3, H, W] for image logging.
    """

    # Process each item in batch
    # Convert to numpy for matplotlib colormap
    depth = depth_tensor[0].detach().cpu().numpy()
    
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        mask_b = mask[0].detach().cpu().numpy()
        depth = np.where((depth > 0) & mask_b, depth, np.nan)
    
    # Convert to disparity (inverse depth)
    disp = 1 / depth
    if np.isnan(disp).all():
        return torch.zeros((3, *disp.shape), dtype=torch.float32)
    
    # Normalize disparity
    if normalize:
        min_disp = np.nanquantile(disp, 0.01)
        max_disp = np.nanquantile(disp, 0.99)
        disp = (disp - min_disp) / (max_disp - min_disp)
    
    # Apply colormap
    colored = plt.get_cmap(cmap)(1.0 - disp)
    colored = np.nan_to_num(colored, 0)
    colored = (colored.clip(0, 1) * 255).astype(np.uint8)[:, :, :3]
    
    # Convert back to torch tensor and rearrange dimensions
    colored = torch.from_numpy(colored).float() / 255.0
    colored = colored.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
    
    return colored

#: Set from ``--compact_retention``.  Keeps scalar/ histogram logging but skips
#: the raw full-resolution TensorBoard image events, which are the bulk of the
#: event files and are explicitly not retained (do_not_archive).
_COMPACT_RETENTION = False


def _log_tensorboard_image(writer, tag, image, iteration):
    """Encode the same full-resolution pixels with fast, lossless PNG compression."""
    image = image.detach()
    if image.dtype != torch.uint8:
        image = image.mul(255.0).to(torch.uint8)
    array = image.permute(1, 2, 0).contiguous().cpu().numpy()
    with BytesIO() as buffer:
        Image.fromarray(array).save(buffer, format="PNG", compress_level=1)
        encoded = buffer.getvalue()
    summary = Summary(value=[Summary.Value(
        tag=_clean_tag(tag),
        image=Summary.Image(height=array.shape[0], width=array.shape[1],
                            colorspace=array.shape[2], encoded_image_string=encoded),
    )])
    writer._get_file_writer().add_summary(summary, iteration)
    writer._get_comet_logger().log_image_encoded(encoded, tag, step=iteration)


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, iterative_datasets_update=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    metrics = {}
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = [{'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()[::4]}]

        if iterative_datasets_update:
            validation_configs.append({'name': 'train_idu', 'cameras' : scene.getTrainIDUCameras()[::3]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(
                        viewpoint, scene.gaussians, *renderArgs,
                        testing=config['name'] in ('test', 'train_idu'),
                    )
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and not _COMPACT_RETENTION and (idx < 5):
                        tag = config['name'] + f"_view_{viewpoint.image_name}"
                        _log_tensorboard_image(tb_writer, tag + "/render", image, iteration)
                        mask = viewpoint.original_mask.cuda()
                        depth_vis = torch.nan_to_num(mask * render_pkg["render_depth"], nan=0, posinf=0, neginf=0)
                        colored_depth = colorize_depth_torch(depth_vis)
                        _log_tensorboard_image(tb_writer, tag + "/depth_colored", colored_depth, iteration)
                        if iteration == testing_iterations[0]:
                            gt_depth = mask * viewpoint.original_depth.to("cuda")
                            colored_gt_depth = colorize_depth_torch(mask * gt_depth)
                            _log_tensorboard_image(tb_writer, tag + "/ground_truth", gt_image, iteration)
                            _log_tensorboard_image(tb_writer, tag + "/depth", colored_gt_depth, iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])       
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                metrics[config['name']] = {"psnr": psnr_test.item(), "l1": l1_test.item(),
                                           "cameras": len(config['cameras'])}
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        metrics["total_points"] = int(scene.gaussians.get_xyz.shape[0])
        torch.cuda.empty_cache()
    return metrics

def honor_saved_rasterizer_backend(args: Namespace) -> None:
    """Keep a saved run's rasterizer backend across resume/re-runs.

    ``cfg_args`` records the backend the run was trained with. Without an
    explicit ``--rasterizer_backend`` on the command line, re-running into the
    same ``--model_path`` (resume, continued stages) or resuming from a
    checkpoint directory must restore that backend instead of silently
    reverting to the ``diff_gauss`` default. Unreadable or invalid saved
    configs raise instead of falling back silently.
    """

    if has_explicit_flag("--rasterizer_backend"):
        return
    candidates = []
    if args.model_path:
        candidates.append(os.path.join(args.model_path, "cfg_args"))
    start_checkpoint = getattr(args, "start_checkpoint", None)
    if start_checkpoint:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(start_checkpoint)), "cfg_args"))
    for cfg_path in candidates:
        if not os.path.exists(cfg_path):
            continue
        try:
            with open(cfg_path, "r") as cfg_file:
                saved = eval(cfg_file.read())
            saved_backend = getattr(saved, "rasterizer_backend", None)
        except Exception as exc:
            raise ValueError(f"Could not read saved config {cfg_path}: {exc}") from exc
        if saved_backend is None:
            # Pre-backend cfg_args: the run was trained with diff_gauss.
            continue
        if saved_backend not in RASTERIZER_BACKENDS:
            raise ValueError(
                f"{cfg_path} has invalid rasterizer_backend={saved_backend!r}; "
                f"expected one of {list(RASTERIZER_BACKENDS)}"
            )
        if saved_backend != args.rasterizer_backend:
            print(f"[rasterizer_backend] Restoring saved backend '{saved_backend}' from {cfg_path} "
                  f"(command line default was '{args.rasterizer_backend}')")
            args.rasterizer_backend = saved_backend
        return


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2000, 3050, 7_000, 10000, 15000, 20000, 21000, 22000, 23000, 30_000, 60100, 61000, 62000, 65000, 67500, 70000, 70100, 71000, 72000, 75000, 77500, 80000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2000, 3050, 7_000, 10000, 15000, 20000, 21000, 22000, 23000, 30_000, 60100, 61000, 62000, 65000, 67500, 70000, 70100, 71000, 72000, 75000, 77500, 80000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[2000, 3050, 7_000, 10000, 15000, 20000, 21000, 22000, 23000, 30_000, 60100, 61000, 62000, 65000, 67500, 70000, 70000, 70100, 71000, 72000, 75000, 77500, 80000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--iterative_datasets_update", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    honor_saved_rasterizer_backend(args)
    print("Rasterizer backend: {}".format(args.rasterizer_backend))
    args.save_iterations.append(args.iterations)
    _COMPACT_RETENTION = bool(getattr(args, "compact_retention", False))
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if not args.iterative_datasets_update:
        training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
        if args.compact_retention:
            scratch = prune_stage_scratch(args.model_path)
            print(f"[compact] Stage 1 keeps the final checkpoint + matching PLY; removed "
                  f"{scratch['bytes'] / 1e9:.2f} GB of consumed scratch")
    else:
    # Start running iterative datasets update
        training_idu(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint)
    # All done
    print("\nTraining complete.")
