#!/usr/bin/env python3
"""Train the full-scene JAX WideFE + GSZoom method without machine-bound paths.

Stages (strictly ordered):
  G0: train 30k native steps, or reuse --base-checkpoint with matching base_iterations.
  G_R: for each 85/75/65/55/45-degree course, render -> two FlowEdit samples/view
       -> 10k native GS updates. Do not generate all courses from the same G0.
  L1: generate every 2x SR tile, then train the first detail level with G_R frozen.
  L2: generate every 4x SR tile from the TRAINED G_R+L1 parent, then train L2
      with G_R and L1 frozen; carry the earlier supervision into its manifest.

Native refinement uses 0.8 RGB-L1 + 0.2 (1-SSIM), plus pseudo-depth weight 0.5
at the native sampler interval. Direct depth and opacity losses are disabled.
LoD target steps use 24 signed high-frequency L1 + 1 LR anchor + .05 RaDe
geometry consistency. Replay uses ordinary RGB-L1; mix_ratio=.2 describes
sampling from the complete replay pool, NOT 20% guaranteed real-photo steps.

Usage (activate the base Skyfall environment first):
  python scripts/train_widefe_gszoom.py --config configs/widefe_gszoom.example.json --plan-only
  python scripts/train_widefe_gszoom.py --config /path/to/your-training.json
  python scripts/train_widefe_gszoom.py --config /path/to/your-training.json --base-checkpoint /path/to/chkpnt30000.pth

The config supplies dataset/model paths and the separate Qwen3-VL and DLoRAL
Python environments. Relative paths are relative to the CONFIG FILE, not the
shell working directory. Model weights are never copied into this repository.
MoGe's pretrained weights must be available through the normal HF cache.
Initialize the pinned submodules and CUDA rasterizers, then run
``python scripts/apply_training_patches.py --apply`` before training.
--plan-only uses no CUDA, requires no installed model weights, and writes nothing.

All GPU stages use wait_for_idle_gpu.py. lock_dir must be shared with your other
GPU jobs on the same machine. Resume requires unchanged configuration and
completed-job evidence; failed/interrupted monitor states are not auto-retried.
Outputs include protocol.json, run_status.json, per-stage monitors/manifests,
curriculum/e04_45/gs/chkpnt80000.pth (default course), and the two LoD checkpoints
with their .appearance.pt sidecars. No shutdown or remote-host action is taken.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULTS = {
    "source_path": None,
    "output_dir": None,
    "base_checkpoint": None,
    "base_iterations": 30000,
    "python": None,
    "vlm_python": None,
    "dloral_python": None,
    "flux_model_path": None,
    "vlm_model_path": None,
    "dloral_root": None,
    "dloral_sd_path": None,
    "dloral_ckpt": None,
    "dloral_spynet": None,
    "gpu_indices": [0],
    "lock_dir": None,
    "idle_seconds": 60,
    "poll_seconds": 10,
    "min_free_mib": 40000,
    "max_utilization": 5,
    "port_base": 16540,
    "elevations": [85, 75, 65, 55, 45],
    "radii": [300, 275, 275, 250, 250],
    "grid_size": 3,
    "grid_width": 512,
    "grid_height": 512,
    "cameras_per_target": 6,
    "samples_per_pose": 2,
    "raster": 1024,
    "fov": 60,
    "idu_steps_per_episode": 10000,
    "flowedit_images_per_shard": 18,
    "sr_views_per_shard": 6,
    "sampler": {
        "n_min": 4, "n_max": 10, "n_max_end": None, "steps": 28, "n_avg": 1,
        "src_guidance": 1.5, "tar_guidance": 5.5,
        "seed_policy": "sample index; reset per image, independent of shard scheduling",
        "original_upstream_prompts": True,
    },
    "lod": {
        "1": {"steps": 10800, "max_points": 800000, "densify_until": 6480,
              "densify_every": 250, "checkpoints": [2700, 5400, 10800]},
        "2": {"steps": 43200, "max_points": 2000000, "densify_until": 25920,
              "densify_every": 500, "checkpoints": [10800, 21600, 43200]},
    },
    "lod_loss": {"mode": "frequency", "detail_weight": 24.0,
                 "lr_anchor_weight": 1.0, "geometry_weight": 0.05,
                 "dssim_weight": 0.2, "mix_ratio": 0.2},
}
PATH_KEYS = ("source_path", "output_dir", "base_checkpoint", "flux_model_path",
             "vlm_model_path", "dloral_root", "dloral_sd_path", "dloral_ckpt",
             "dloral_spynet", "lock_dir")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def merge_config(defaults: dict, overrides: dict, prefix: str = "") -> dict:
    if not isinstance(overrides, dict):
        raise ValueError(f"{prefix or 'config'} must be a JSON object")
    result = copy.deepcopy(defaults)
    for key, value in overrides.items():
        if key not in defaults:
            raise ValueError(f"Unknown configuration key: {prefix}{key}")
        result[key] = (merge_config(defaults[key], value, prefix + key + ".")
                       if isinstance(defaults[key], dict) else value)
    return result


def resolve_config(args) -> dict:
    config_path = args.config.expanduser().resolve()
    protocol = merge_config(DEFAULTS, load(config_path))
    for key in ("source_path", "output_dir", "base_checkpoint"):
        override = getattr(args, key)
        if override is not None:
            protocol[key] = str(Path(override).expanduser().resolve())
    if args.gpu_indices is not None:
        protocol["gpu_indices"] = [int(value) for value in args.gpu_indices.split(",")]
    protocol["python"] = protocol["python"] or sys.executable
    protocol["dloral_root"] = protocol["dloral_root"] or str(REPO_ROOT / "submodules/DLoRAL")
    for key in PATH_KEYS:
        value = protocol[key]
        if value is None and key == "base_checkpoint":
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Set a nonempty {key} in {config_path}")
        path = Path(os.path.expandvars(value)).expanduser()
        protocol[key] = str((path if path.is_absolute() else config_path.parent / path).resolve())
    for key in ("python", "vlm_python", "dloral_python"):
        value = protocol[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Set {key}; the VLM and DLoRAL environments are separate prerequisites")
        expanded = os.path.expandvars(os.path.expanduser(value))
        found = shutil.which(expanded) if Path(expanded).name == expanded else None
        if found:
            protocol[key] = os.path.abspath(found)
        else:
            path = Path(expanded)
            # Preserve venv/bin/python symlinks: resolving them discards the venv.
            protocol[key] = os.path.abspath(path if path.is_absolute() else config_path.parent / path)
    for key in ("base_iterations", "grid_size", "cameras_per_target", "samples_per_pose",
                "raster", "idu_steps_per_episode", "flowedit_images_per_shard", "sr_views_per_shard"):
        if type(protocol[key]) is not int or protocol[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if protocol["raster"] % 4:
        raise ValueError("raster must be divisible by four for the 2x/4x tile pyramid")
    indices = protocol["gpu_indices"]
    if not isinstance(indices, list) or not indices or any(type(i) is not int or i < 0 for i in indices):
        raise ValueError("gpu_indices must explicitly select nonnegative physical GPU indices")
    if len(indices) != len(set(indices)):
        raise ValueError("gpu_indices must not contain duplicates")
    elevations, radii = protocol["elevations"], protocol["radii"]
    if not isinstance(elevations, list) or not isinstance(radii, list) or not elevations or len(elevations) != len(radii):
        raise ValueError("elevations and radii must be nonempty equal-length lists")
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in elevations + radii):
        raise ValueError("Camera courses must be finite numbers")
    if any(not 0 < x < 90 for x in elevations) or any(x <= 0 for x in radii):
        raise ValueError("Course elevations must be in (0,90) and radii must be positive")
    if not 0 < protocol["fov"] < 180:
        raise ValueError("fov must be in (0,180)")
    if type(protocol["port_base"]) is not int or not 1 < protocol["port_base"] < 65535 - len(elevations):
        raise ValueError("port_base must leave room for the base and each curriculum episode")
    if protocol["lod_loss"]["mode"] != "frequency":
        raise ValueError("This entrypoint implements the current frequency-supervised method")
    for level in ("1", "2"):
        settings = protocol["lod"][level]
        for key in ("steps", "max_points", "densify_until", "densify_every"):
            if type(settings[key]) is not int or settings[key] <= 0:
                raise ValueError(f"lod.{level}.{key} must be a positive integer")
        if settings["densify_until"] > settings["steps"]:
            raise ValueError(f"lod.{level}.densify_until cannot exceed the level budget")
        if any(type(step) is not int or not 0 < step <= settings["steps"] for step in settings["checkpoints"]):
            raise ValueError(f"lod.{level}.checkpoints must fall inside the level budget")
    protocol["repo_root"] = str(REPO_ROOT)
    protocol["reuse_verified_stage1"] = protocol["base_checkpoint"] is not None
    if protocol["base_checkpoint"] is None:
        protocol["base_checkpoint"] = str(Path(protocol["output_dir"]) / "stage1_base" / f"chkpnt{protocol['base_iterations']}.pth")
    views = protocol["grid_size"] ** 2 * protocol["cameras_per_target"]
    images = len(elevations) * views * protocol["samples_per_pose"]
    protocol["flowedit_image_count"] = images
    protocol["sr_teacher_counts"] = {str(zoom): images * zoom * zoom for zoom in (2, 4)}
    protocol["run"] = Path(protocol["output_dir"]).name
    return protocol


def gpu_command(protocol: dict, script: str, *argv) -> list[str]:
    return [protocol["python"], "-u", str(REPO_ROOT / script), *map(str, argv)]


def base_command(protocol: dict) -> list[str]:
    # Reuse the existing frozen JAX recipe; do not invent another base objective.
    from scripts.train_full_dataset import STAGE1_JAX_ARGS
    iterations = protocol["base_iterations"]
    return gpu_command(protocol, "train.py", "-s", protocol["source_path"],
        "-m", Path(protocol["base_checkpoint"]).parent, "--eval",
        "--port", protocol["port_base"] - 1, "--rasterizer_backend", "rade",
        *STAGE1_JAX_ARGS, "--iterations", iterations, "--position_lr_max_steps", iterations,
        "--densify_until_iter", min(21000, int(iterations * 0.7)),
        "--end_sample_pseudo", min(21000, int(iterations * 0.7)),
        "--test_iterations", iterations, "--save_iterations", iterations,
        "--checkpoint_iterations", iterations, "--data_device", "cpu")


def episode_directory(protocol: dict, index: int) -> Path:
    return Path(protocol["output_dir"]) / "curriculum" / f"e{index:02d}_{protocol['elevations'][index]:02g}"


def native_command(protocol: dict, index: int, checkpoint: Path, manifest: Path) -> list[str]:
    sampler = protocol["sampler"]
    return gpu_command(protocol, "train.py", "-s", protocol["source_path"],
        "-m", episode_directory(protocol, index) / "gs", "--eval",
        "--port", protocol["port_base"] + index, "--start_checkpoint", checkpoint,
        "--rasterizer_backend", "rade", "--iterative_datasets_update",
        "--kernel_size", "0.1", "--resolution", "1", "--sh_degree", "1",
        "--appearance_enabled", "--lambda_dssim", "0.2", "--lambda_depth", "0",
        "--lambda_opacity", "0", "--opacity_reset_interval", "10000000",
        "--idu_opacity_reset_interval", "5000", "--densify_grad_threshold", "0.0002",
        "--datasets_type", "jax_v1", "--idu_num_cams", protocol["cameras_per_target"],
        "--idu_grid_size", protocol["grid_size"], "--idu_grid_width", protocol["grid_width"],
        "--idu_grid_height", protocol["grid_height"], "--idu_episode_iterations", protocol["idu_steps_per_episode"],
        "--idu_opacity_cooling_iterations", "500", "--lambda_pseudo_depth", "0.5",
        "--sample_pseudo_interval", "10", "--idu_densify_until_iter", min(9000, protocol["idu_steps_per_episode"]),
        "--idu_train_ratio", "0.75", "--idu_render_size", protocol["raster"], "--idu_seed", "0",
        "--idu_refine_backend", "flowedit", "--idu_supervision_manifest", manifest,
        "--idu_episode_count", "1", "--idu_num_samples_per_view", protocol["samples_per_pose"],
        "--idu_flow_edit_n_min", sampler["n_min"], "--idu_flow_edit_n_max", sampler["n_max"],
        "--idu_flow_edit_n_max_end", -1 if sampler["n_max_end"] is None else sampler["n_max_end"],
        "--flux_model_path", protocol["flux_model_path"], "--data_device", "cpu")


def lod_command(protocol: dict, level: int, checkpoint: Path, supervision: Path, parent: Path | None) -> list[str]:
    settings, loss = protocol["lod"][str(level)], protocol["lod_loss"]
    command = gpu_command(protocol, "scripts/train_scene_lod.py",
        "--start_checkpoint", checkpoint, "--source_path", protocol["source_path"],
        "--supervision", supervision, "--output_dir", Path(protocol["output_dir"]) / f"l{level}_train",
        "--steps_per_level", settings["steps"], "--max_points_per_level", settings["max_points"],
        "--seed", "0", "--mix_ratio", loss["mix_ratio"], "--step_scale", "2",
        "--loss_mode", loss["mode"], "--loss_hr", loss["detail_weight"],
        "--loss_lr", loss["lr_anchor_weight"], "--loss_geometry", loss["geometry_weight"],
        "--loss_dssim", loss["dssim_weight"], "--densify_from", "1",
        "--densify_until", settings["densify_until"], "--densify_every", settings["densify_every"],
        "--densify_grad_threshold", "0.0002", "--densify_bootstrap_fraction", "0.25",
        "--split_radius_pixels", "8", "--eval_samples", "18", "--png_samples", "9",
        "--target_cache_size", "32", "--checkpoint_steps", ",".join(map(str, settings["checkpoints"])))
    if parent is not None:
        command += ["--parent_lod_checkpoint", str(parent)]
    return command


def training_plan(protocol: dict) -> list[dict]:
    root = Path(protocol["output_dir"])
    checkpoint = Path(protocol["base_checkpoint"])
    plan = [{"stage": "g0", "action": "reuse" if protocol["reuse_verified_stage1"] else "train",
             "checkpoint": str(checkpoint),
             "command": None if protocol["reuse_verified_stage1"] else base_command(protocol)}]
    manifests = []
    for index, (elevation, radius) in enumerate(zip(protocol["elevations"], protocol["radii"])):
        folder = episode_directory(protocol, index)
        manifest = folder / "flowedit_views.json"
        count = protocol["grid_size"] ** 2 * protocol["cameras_per_target"] * protocol["samples_per_pose"]
        iteration = protocol["base_iterations"] + (index + 1) * protocol["idu_steps_per_episode"]
        plan.append({"stage": f"curriculum_{index}", "elevation": elevation, "radius": radius,
                     "input_checkpoint": str(checkpoint), "teacher_images": count,
                     "flowedit_shards": math.ceil(count / protocol["flowedit_images_per_shard"]),
                     "prepare_command": gpu_command(protocol, "scripts/prepare_widefe.py",
                         "--protocol", root / "protocol.json", "--checkpoint", checkpoint,
                         "--output-dir", folder, "--episode-index", index, "--elevation", elevation, "--radius", radius),
                     "train_command": native_command(protocol, index, checkpoint, manifest)})
        checkpoint = folder / "gs" / f"chkpnt{iteration}.pth"
        manifests.append(str(manifest))
    parent = previous = None
    for level, zoom in ((1, 2), (2, 4)):
        sr_dir = root / f"sr{zoom}"
        command = gpu_command(protocol, "scripts/prepare_scene_zoom_shards.py", "plan",
            "--protocol", root / "protocol.json", "--checkpoint", checkpoint,
            "--zoom", zoom, "--output-dir", sr_dir, "--flowedit-manifests", *manifests,
            "--views-per-shard", protocol["sr_views_per_shard"])
        if parent is not None:
            command += ["--parent-lod-checkpoint", str(parent), "--previous-supervision", str(previous)]
        plan.append({"stage": f"l{level}", "cumulative_zoom": zoom,
                     "new_teacher_images": protocol["sr_teacher_counts"][str(zoom)],
                     "parent_lod_checkpoint": None if parent is None else str(parent),
                     "prepare_command": command,
                     "merge_command": gpu_command(protocol, "scripts/prepare_scene_zoom_shards.py", "merge", "--plan", sr_dir / "plan.json"),
                     "train_command": lod_command(protocol, level, checkpoint, sr_dir / "supervision.json", parent)})
        parent, previous = root / f"l{level}_train/l{level}_final.lod.pt", sr_dir / "supervision.json"
    return plan


def check_assets(protocol: dict) -> None:
    for key in ("source_path", "flux_model_path", "vlm_model_path", "dloral_root", "dloral_sd_path"):
        if not Path(protocol[key]).is_dir():
            raise FileNotFoundError(f"{key}: {protocol[key]}")
    for key in ("python", "vlm_python", "dloral_python", "dloral_ckpt", "dloral_spynet"):
        if not Path(protocol[key]).is_file():
            raise FileNotFoundError(f"{key}: {protocol[key]}")
    if protocol["reuse_verified_stage1"] and not Path(protocol["base_checkpoint"]).is_file():
        raise FileNotFoundError(protocol["base_checkpoint"])
    for relative in ("submodules/FlowEdit/idu_refine.py", "submodules/MoGe/idu_depth.py",
                     "scripts/prepare_widefe.py", "scripts/run_widefe_shard.py",
                     "scripts/prepare_scene_zoom_shards.py", "scripts/wait_for_idle_gpu.py"):
        if not (REPO_ROOT / relative).is_file():
            raise FileNotFoundError(f"Required source dependency missing: {relative}")


def _execute(protocol: dict) -> None:
    from scripts.idle_dispatch import run_jobs
    check_assets(protocol)
    root = Path(protocol["output_dir"])
    if root == REPO_ROOT or root == Path(protocol["source_path"]):
        raise ValueError("output_dir cannot be the source checkout or input dataset")
    status_path = root / "run_status.json"
    if root.exists() and any(path.name != ".training.lock" for path in root.iterdir()) and not status_path.exists():
        raise ValueError("output_dir is nonempty without this trainer's run_status.json; use a fresh run directory")
    request_sha = hashlib.sha256(json.dumps(protocol, sort_keys=True, allow_nan=False).encode()).hexdigest()
    state = load(status_path) if status_path.exists() else {
        "status": "preparing", "request_sha256": request_sha, "completed": [],
        "episodes": [], "stage_seconds": {}, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if state.get("request_sha256") != request_sha:
        raise ValueError("Training configuration changed; refusing to mix an existing run with new settings")
    root.mkdir(parents=True, exist_ok=True)
    save(root / "request.json", protocol)
    save(status_path, state)

    def phase(name: str, jobs: list[dict]) -> None:
        state.update(status="running", active_stage=name)
        save(status_path, state)
        print("PHASE_START " + name, flush=True)
        started = time.monotonic()
        records = run_jobs(jobs, protocol=protocol, stage_dir=str(root / "monitors" / name))
        if name not in state["completed"]:
            state["completed"].append(name)
            state["stage_seconds"][name] = time.monotonic() - started
        state["last_dispatch"] = records
        save(status_path, state)
        print("PHASE_DONE " + name, flush=True)

    def cpu(command: list[str]) -> None:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
        subprocess.run(command, check=True, cwd=REPO_ROOT, env=environment)

    try:
        checkpoint = Path(protocol["base_checkpoint"])
        if not protocol["reuse_verified_stage1"]:
            phase("stage1_base", [{"id": "train", "command": base_command(protocol),
                "expected_outputs": [str(checkpoint), str(checkpoint.parent / "cfg_args")]}])
        base_sha = digest(checkpoint)
        protocol_path = root / "protocol.json"
        if protocol_path.exists():
            existing = load(protocol_path)
            if existing.get("base_sha256") != base_sha:
                raise ValueError("The run's verified G0 checkpoint changed")
        protocol["base_sha256"] = base_sha
        save(protocol_path, protocol)
        from PIL import Image
        from refinement.flowedit_stage2 import build_flowedit_views_manifest
        plan = training_plan(protocol)
        expected_poses = protocol["grid_size"] ** 2 * protocol["cameras_per_target"]
        manifests = []
        for index, (elevation, radius) in enumerate(zip(protocol["elevations"], protocol["radii"])):
            key = f"e{index:02d}_{elevation:02g}"
            folder = episode_directory(protocol, index)
            folder.mkdir(parents=True, exist_ok=True)
            prepared_path = folder / "flowedit_prepared.json"
            phase("prepare_" + key, [{"id": "render", "command": plan[index + 1]["prepare_command"],
                                      "expected_outputs": [str(prepared_path)]}])
            prepared = load(prepared_path)
            count = prepared["num_views"] * prepared["samples_per_view"]
            if prepared["num_views"] != expected_poses or prepared["samples_per_view"] != protocol["samples_per_pose"]:
                raise ValueError(f"Incomplete or incompatible camera course: {key}")
            jobs = []
            for start in range(0, count, protocol["flowedit_images_per_shard"]):
                indices = list(range(start, min(count, start + protocol["flowedit_images_per_shard"])))
                result_path = folder / "shards" / f"{start:05d}.json"
                jobs.append({"id": f"images_{start:05d}",
                    "command": gpu_command(protocol, "scripts/run_widefe_shard.py", "--protocol", protocol_path,
                        "--prepared", prepared_path, "--indices", ",".join(map(str, indices)), "--result-json", result_path),
                    "expected_outputs": [str(result_path)] + [str(folder / "render_refine" / f"{i:05d}.png") for i in indices]})
            phase("flowedit_" + key, jobs)
            manifest = build_flowedit_views_manifest(prepared, checkpoint=str(checkpoint),
                episode_idx=index, samples_per_view=protocol["samples_per_pose"])
            manifest.update(checkpoint_sha256=digest(checkpoint), elevation=elevation,
                            radius=radius, sampler=protocol["sampler"])
            for flat, entry in enumerate(manifest["views"]):
                output = Path(entry["image_path"])
                with Image.open(output) as image:
                    if image.size != (protocol["raster"], protocol["raster"]):
                        raise ValueError(f"Unexpected native FlowEdit raster: {output}")
                    image.verify()
                source = Path(prepared["views"][flat // protocol["samples_per_pose"]]["render_path"])
                entry.update(appearance_uid=6, image_sha256=digest(output), input_image_path=str(source),
                             input_sha256=digest(source), sample_seed=flat % protocol["samples_per_pose"],
                             synthesis_stage="pure_wide_flowedit")
            manifest_path = folder / "flowedit_views.json"
            save(manifest_path, manifest)
            manifests.append(str(manifest_path))
            iteration = protocol["base_iterations"] + (index + 1) * protocol["idu_steps_per_episode"]
            next_checkpoint = folder / "gs" / f"chkpnt{iteration}.pth"
            phase("native_" + key, [{"id": "train", "command": plan[index + 1]["train_command"],
                "expected_outputs": [str(next_checkpoint), str(folder / "gs/cfg_args"),
                    str(folder / "gs/point_cloud" / f"iteration_{iteration}" / "point_cloud.ply")]}])
            record = {"episode": index, "elevation": elevation, "radius": radius,
                      "input_checkpoint": str(checkpoint), "output_checkpoint": str(next_checkpoint),
                      "output_sha256": digest(next_checkpoint), "flowedit_manifest": str(manifest_path),
                      "unique_camera_poses": expected_poses, "flowedit_images": count,
                      "native_steps": protocol["idu_steps_per_episode"]}
            previous_record = next((item for item in state["episodes"] if item["episode"] == index), None)
            if previous_record and previous_record["output_sha256"] != record["output_sha256"]:
                raise ValueError(f"Previously completed checkpoint changed: {next_checkpoint}")
            state["episodes"] = [item for item in state["episodes"] if item["episode"] != index] + [record]
            checkpoint = next_checkpoint
            save(status_path, state)
        for level, zoom in ((1, 2), (2, 4)):
            stage = plan[len(protocol["elevations"]) + level]
            directory = root / f"sr{zoom}"
            cpu(stage["prepare_command"])
            phase(f"sr{zoom}_generation", load(directory / "jobs.json"))
            cpu(stage["merge_command"])
            supervision = directory / "supervision.json"
            payload = load(supervision)
            samples = next(item["samples"] for item in payload["levels"] if float(item["zoom_factor"]) == zoom)
            if len(samples) != protocol["sr_teacher_counts"][str(zoom)]:
                raise ValueError(f"{zoom}x teacher count mismatch: {len(samples)}")
            training = root / f"l{level}_train"
            final = training / f"l{level}_final.lod.pt"
            phase(f"l{level}_training", [{"id": "train", "command": stage["train_command"],
                "expected_outputs": [str(final), str(final) + ".appearance.pt", str(training / "training_summary.json")]}])
            summary = load(training / "training_summary.json")
            matching = [item for item in summary["levels"] if float(item["zoom_factor"]) == zoom]
            if len(matching) != 1 or not matching[0]["coverage_complete"]:
                raise ValueError(f"{zoom}x did not cover every supervision target")
            record = {"checkpoint": str(final), "sha256": digest(final),
                "appearance_sha256": digest(Path(str(final) + ".appearance.pt")),
                "supervision": str(supervision), "new_teacher_count": len(samples),
                "coverage_complete": True, "summary": matching[0]}
            previous_record = state.get(f"l{level}")
            if previous_record and any(previous_record[key] != record[key] for key in ("sha256", "appearance_sha256")):
                raise ValueError(f"Previously completed LoD or appearance state changed: {final}")
            state[f"l{level}"] = record
            save(status_path, state)
        state.update(status="training_complete", active_stage=None,
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            final_native_checkpoint=str(checkpoint), final_lod_checkpoint=state["l2"]["checkpoint"])
        save(status_path, state)
        print("FULL_TRAINING_COMPLETE " + str(status_path), flush=True)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        save(status_path, state)
        raise


def execute(protocol: dict) -> None:
    root = Path(protocol["output_dir"])
    if root == REPO_ROOT or root == Path(protocol["source_path"]):
        raise ValueError("output_dir cannot be the source checkout or input dataset")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".training.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another training controller already owns {root}") from error
        _execute(protocol)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="JSON config; paths are relative to this file")
    parser.add_argument("--source-path", dest="source_path", help="Override the dataset path")
    parser.add_argument("--output-dir", dest="output_dir", help="Override the artifact directory")
    parser.add_argument("--base-checkpoint", dest="base_checkpoint", help="Reuse G0 instead of training it")
    parser.add_argument("--gpu-indices", help="Explicit comma-separated physical GPU indices")
    parser.add_argument("--plan-only", action="store_true", help="Print the full stage plan without file writes, model loads, or GPU work")
    args = parser.parse_args(argv)
    try:
        protocol = resolve_config(args)
        if args.plan_only:
            print(json.dumps({"method": "widefe_gszoom", "protocol": protocol, "stages": training_plan(protocol)}, indent=2, allow_nan=False))
            return 0
        execute(protocol)
        return 0
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
