#!/usr/bin/env python3
"""Full-dataset GaussianZoom Stage 1 + Stage 2 training queue (remote GPU 0).

This is the training-owner entry point for the 12-scene run defined by
``/dataset/Skyfall-GS/experiments/stage2_gaussianzoom_20260913/pipeline_manifest.json``.
It is intentionally self-contained: the scene table and the per-recipe argument
lists are data in this file (copied from the qualified, GPU-verified source
recipes), so importing it never starts a job and running it never needs the
manifest.  ``--manifest`` cross-checks the embedded table against the shared
contract when the manifest is reachable.

Recipe provenance (remote project, branch dev-stage2-gaussianzoom):
  * JAX Stage 1: ``scripts/run_jax.py`` recipe, as frozen in the verified
    ``skyfall-gs_exp/jax068_gszoom_20260912_01/stage1-command.json``
    (dataset root ``data/datasets_JAX/<scene>``; the stale ``outputs_skew``
    subdirectory does not exist, so the real root holding
    ``transforms_train.json`` is resolved at preflight).
  * NYC Stage 1: ``scripts/run_nyc.py`` (``--target_std 32``, grad threshold
    0.0002, opacity reset 4000).
  * JAX Stage 2 spatial/optimization recipe: original ``scripts/run_jax_idu.py``
    (``idu_grid_size 3``, ``lambda_pseudo_depth 0.5``, ``idu_num_cams 6``,
    ``idu_num_samples_per_view 2``, ``idu_episode_iterations 10000``,
    ``idu_densify_until_iter 9000``, ``idu_train_ratio 0.75``).
  * NYC Stage 2 spatial/optimization recipe: original ``scripts/run_nyc_idu.py``
    (``idu_grid_size 4``, ``lambda_pseudo_depth 0.0``, ``lambda_opacity 10``, ``--target_std 32``).

This frozen queue explicitly uses GaussianZoom synthesis. The generic IDU
default and the current JAX/NYC launchers instead select pure FlowEdit.

The five-elevation course and 10000-step episodes come from
``arguments/__init__.py`` (``jax_v1`` / ``nyc_v1``) and are untouched.  The
``nyc_v1`` elevation list has a sixth value with no matching radius; the
curriculum zips the two lists, so the actual course is five episodes for both
datasets.  That existing behaviour is preserved exactly.

No ``render.py`` / ``metrics.py`` / fused-PLY step is invoked here: evaluation
is the local-GPU evaluator's job and the raw ``.ply`` does not carry the learned
appearance model.

Subcommands
-----------
``plan``          exact per-job argv/env, reserves and the queue launch command
``run``           durable queue: train, verify, publish markers, gate on disk
``status``        queue state table + which scenes/stages are missing
``publish-reuse`` hard-link an already-verified run (JAX_068) + publish markers
``compact``       offline compact retention for a finished/reused stage dir
``reap``          delete remote finals only after local archive + load clearance

Cleanup gate: ``<archive_root>/<scene>/<stage>/archive_status.json`` == verified
AND ``<archive_root>/<scene>/evaluation/<stage>/evaluation_status.json`` has
``model_load_verified: true`` (the evaluator publishes that as soon as the model
actually loads and the first renders succeed, so remote training keeps going
while the full metric pass finishes locally).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

SCHEMA_VERSION = 1
METHOD = "gaussianzoom_stage2"
BRANCH = "dev-stage2-gaussianzoom"
MARKER_NAME = "stage_complete.json"
RUN_STATUS_NAME = "run_status.json"
CLEARANCE_NAME = "remote_cleanup.json"

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RUN_ROOT = "/root/autodl-tmp/Skyfall-GS/skyfall-gs_exp/full_dataset_gz_20260913"
DEFAULT_CONTROL_ROOT = "/root/autodl-tmp/skyfall-setup/full-dataset"
DEFAULT_MANIFEST = "/dataset/Skyfall-GS/experiments/stage2_gaussianzoom_20260913/pipeline_manifest.json"

# --------------------------------------------------------------------------- #
# Scene table (mirrors pipeline_manifest.json "scenes")
# --------------------------------------------------------------------------- #

SCENES = [
    {"scene": "JAX_004", "dataset": "datasets_JAX", "recipe": "jax_v1"},
    {"scene": "JAX_068", "dataset": "datasets_JAX", "recipe": "jax_v1",
     "reuse_stage1": "/root/autodl-tmp/Skyfall-GS/skyfall-gs_exp/jax068_gszoom_20260912_01/stage1",
     "reuse_stage2": "/root/autodl-tmp/Skyfall-GS/skyfall-gs_exp/jax068_stage2_gz_20260913_082647/full"},
    {"scene": "JAX_164", "dataset": "datasets_JAX", "recipe": "jax_v1"},
    {"scene": "JAX_168", "dataset": "datasets_JAX", "recipe": "jax_v1"},
    {"scene": "JAX_175", "dataset": "datasets_JAX", "recipe": "jax_v1"},
    {"scene": "JAX_214", "dataset": "datasets_JAX", "recipe": "jax_v1"},
    {"scene": "JAX_260", "dataset": "datasets_JAX", "recipe": "jax_v1"},
    {"scene": "JAX_264", "dataset": "datasets_JAX", "recipe": "jax_v1"},
    {"scene": "NYC_004", "dataset": "datasets_NYC", "recipe": "nyc_v1"},
    {"scene": "NYC_010", "dataset": "datasets_NYC", "recipe": "nyc_v1"},
    {"scene": "NYC_219", "dataset": "datasets_NYC", "recipe": "nyc_v1"},
    {"scene": "NYC_336", "dataset": "datasets_NYC", "recipe": "nyc_v1"},
]

STAGE1_ITERATIONS = 30000
STAGE2_EPISODE_ITERATIONS = 10000
STAGE2_EPISODES = 5
STAGE2_FINAL_ITERATION = STAGE1_ITERATIONS + STAGE2_EPISODES * STAGE2_EPISODE_ITERATIONS

# --------------------------------------------------------------------------- #
# Recipe argument lists (frozen from the scripts named in the module docstring)
# --------------------------------------------------------------------------- #

# JAX Stage 1 == jax068_gszoom_20260912_01/stage1-command.json, except that only
# the final checkpoint/PLY pair is written (compact-retention request).
STAGE1_JAX_ARGS = [
    "--iterations", str(STAGE1_ITERATIONS),
    "--kernel_size", "0.1",
    "--resolution", "1",
    "--sh_degree", "1",
    "--appearance_enabled",
    "--lambda_depth", "0",
    "--lambda_opacity", "10",
    "--densify_until_iter", "21000",
    "--densify_grad_threshold", "0.0001",
    "--lambda_pseudo_depth", "0.5",
    "--start_sample_pseudo", "1000",
    "--end_sample_pseudo", "21000",
    "--size_threshold", "20",
    "--scaling_lr", "0.001",
    "--rotation_lr", "0.001",
    "--opacity_reset_interval", "3000",
    "--sample_pseudo_interval", "10",
    "--test_iterations", "2000", "7000", "15000", "21000", "30000",
    "--save_iterations", str(STAGE1_ITERATIONS),
    "--checkpoint_iterations", str(STAGE1_ITERATIONS),
]

# NYC Stage 1 == scripts/run_nyc.py (its test_iterations default is kept).
STAGE1_NYC_ARGS = [
    "--iterations", str(STAGE1_ITERATIONS),
    "--kernel_size", "0.1",
    "--resolution", "1",
    "--sh_degree", "1",
    "--appearance_enabled",
    "--lambda_depth", "0",
    "--lambda_opacity", "10",
    "--densify_until_iter", "21000",
    "--densify_grad_threshold", "0.0002",
    "--lambda_pseudo_depth", "0.5",
    "--start_sample_pseudo", "1000",
    "--end_sample_pseudo", "21000",
    "--size_threshold", "20",
    "--scaling_lr", "0.001",
    "--rotation_lr", "0.001",
    "--opacity_reset_interval", "4000",
    "--sample_pseudo_interval", "10",
    "--target_std", "32",
    "--datasets_type", "nyc_v1",
    "--test_iterations", "2000", "3050", "7000", "10000", "15000", "20000", "21000",
    "22000", "23000", "30000", "60100", "61000", "62000", "65000", "67500", "70000",
    "70100", "71000", "72000", "75000", "77500", "80000",
    "--save_iterations", str(STAGE1_ITERATIONS),
    "--checkpoint_iterations", str(STAGE1_ITERATIONS),
]

# Shared JAX Stage 2 spatial/optimization recipe; backend selected by caller.
STAGE2_JAX_ARGS = [
    "--iterative_datasets_update",
    "--kernel_size", "0.1",
    "--resolution", "1",
    "--sh_degree", "1",
    "--appearance_enabled",
    "--lambda_depth", "0.0",
    "--lambda_opacity", "0.0",
    "--opacity_reset_interval", "10000000",
    "--idu_opacity_reset_interval", "5000",
    "--idu_num_samples_per_view", "2",
    "--densify_grad_threshold", "0.0002",
    "--datasets_type", "jax_v1",
    "--idu_num_cams", "6",
    "--idu_grid_size", "3",
    "--idu_grid_width", "512",
    "--idu_grid_height", "512",
    "--idu_episode_iterations", str(STAGE2_EPISODE_ITERATIONS),
    "--idu_opacity_cooling_iterations", "500",
    "--lambda_pseudo_depth", "0.5",
    "--idu_densify_until_iter", "9000",
    "--idu_train_ratio", "0.75",
    "--idu_render_size", "1024",
    "--idu_seed", "0",
]

# Shared NYC Stage 2 spatial/optimization recipe; backend selected by caller.
STAGE2_NYC_ARGS = [
    "--iterative_datasets_update",
    "--kernel_size", "0.1",
    "--resolution", "1",
    "--sh_degree", "1",
    "--appearance_enabled",
    "--lambda_depth", "0.0",
    "--lambda_opacity", "10",
    "--opacity_reset_interval", "10000000",
    "--idu_opacity_reset_interval", "5000",
    "--idu_num_samples_per_view", "2",
    "--densify_grad_threshold", "0.0002",
    "--datasets_type", "nyc_v1",
    "--idu_num_cams", "6",
    "--idu_grid_size", "4",
    "--idu_grid_width", "512",
    "--idu_grid_height", "512",
    "--idu_episode_iterations", str(STAGE2_EPISODE_ITERATIONS),
    "--idu_opacity_cooling_iterations", "500",
    "--lambda_pseudo_depth", "0.0",
    "--idu_densify_until_iter", "9000",
    "--idu_train_ratio", "0.75",
    "--target_std", "32",
    "--idu_render_size", "1024",
    "--idu_seed", "0",
]

# Compact retention: bounded per-episode provenance + panels.
COMPACT_ARGS = ["--compact_retention"]

RECIPES = {
    ("stage1", "jax_v1"): STAGE1_JAX_ARGS,
    ("stage1", "nyc_v1"): STAGE1_NYC_ARGS,
    ("stage2", "jax_v1"): STAGE2_JAX_ARGS,
    ("stage2", "nyc_v1"): STAGE2_NYC_ARGS,
}

# --------------------------------------------------------------------------- #
# Runtime environment (defaults mirror /root/autodl-tmp/skyfall-setup/activate.sh)
# --------------------------------------------------------------------------- #

ENV_DEFAULTS = {
    "SKYFALL_ROOT": "/root/autodl-tmp/Skyfall-GS",
    "SKYFALL_PYTHON": "/root/autodl-tmp/skyfall-envs/skyfall-gs/bin/python",
    "VLM_PYTHON": "/root/autodl-tmp/skyfall-envs/qwen3vl/bin/python",
    "VLM_MODEL_PATH": "/root/autodl-tmp/weights/Qwen3-VL-4B-Instruct",
    "DLORAL_PYTHON": "/root/autodl-tmp/skyfall-envs/dloral/bin/python",
    "DLORAL_WEIGHT_ROOT": "/root/autodl-tmp/weights/dloral",
    "HF_HOME": "/root/autodl-tmp/skyfall-cache/huggingface",
    "TORCH_HOME": "/root/autodl-tmp/skyfall-cache/torch",
    "XDG_CACHE_HOME": "/root/autodl-tmp/skyfall-cache/xdg",
    "TMPDIR": "/root/autodl-tmp/skyfall-tmp",
}

STAGE_RESERVE_GB = {"stage1": 6.0, "stage2": 16.0}

GPU_INDEX = "0"


def project_root() -> Path:
    return Path(os.environ.get("SKYFALL_ROOT", ENV_DEFAULTS["SKYFALL_ROOT"]))


def runtime_env(project: Path) -> dict:
    env = dict(os.environ)
    for name, default in ENV_DEFAULTS.items():
        env.setdefault(name, default)
    env.update({
        "CUDA_VISIBLE_DEVICES": GPU_INDEX,
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONPATH": f"{project}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep),
    })
    return env


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(str(path)).free / 1e9
    except OSError:
        return shutil.disk_usage(str(path.parent)).free / 1e9


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Job construction
# --------------------------------------------------------------------------- #

def source_candidates(project: Path, dataset: str, scene: str) -> list:
    base = project / "data" / dataset / scene
    return [base, base / "outputs_skew"]


def _has_transforms(candidate: Path) -> bool:
    for name in ("transforms_train.json", "transforms.json"):
        try:
            if (candidate / name).is_file():
                return True
        except OSError:
            continue
    return False


def resolve_source(project: Path, dataset: str, scene: str, *, strict: bool = True) -> Path:
    """Resolve the dataset root that actually holds the transforms files.

    ``scripts/run_jax.py`` still points at ``<scene>/outputs_skew``, which does
    not exist in the shipped layout; the real root (with ``transforms_train.json``
    / ``transforms.json``) is preferred and the stale path is only used if it
    genuinely contains the transforms.  ``strict=False`` is used by ``plan`` so
    the command manifest can be generated off-host.
    """
    candidates = source_candidates(project, dataset, scene)
    valid = [candidate for candidate in candidates if _has_transforms(candidate)]
    if valid:
        return valid[0]
    if strict:
        raise FileNotFoundError(
            f"No dataset root with transforms_train.json/transforms.json under "
            f"{candidates[0]} (checked {[str(c) for c in candidates]})")
    return candidates[0]


def build_jobs(project: Path, run_root: Path, only: list, stages: list) -> list:
    jobs = []
    for index, entry in enumerate(SCENES):
        scene, dataset, recipe = entry["scene"], entry["dataset"], entry["recipe"]
        if only and scene not in only:
            continue
        for stage_offset, stage in enumerate(("stage1", "stage2")):
            if stages and stage not in stages:
                continue
            reuse_dir = entry.get("reuse_stage1" if stage == "stage1" else "reuse_stage2")
            iteration = STAGE1_ITERATIONS if stage == "stage1" else STAGE2_FINAL_ITERATION
            job = {
                "key": f"{scene}/{stage}",
                "scene": scene,
                "stage": stage,
                "dataset": dataset,
                "recipe": recipe,
                "output_dir": str(run_root / scene / stage),
                "marker": str(run_root / scene / stage / MARKER_NAME),
                "expected_iteration": iteration,
                "reserve_gb": STAGE_RESERVE_GB[stage],
                "kind": "reuse" if reuse_dir else "train",
                "reuse_dir": reuse_dir,
                "port": str(6300 + 2 * index + stage_offset),
            }
            jobs.append(job)
    return jobs


def job_command(job: dict, project: Path, source_path: Path, start_checkpoint: str | None,
                compact: bool = True) -> list:
    args = list(RECIPES[(job["stage"], job["recipe"])])
    if job["stage"] == "stage2":
        # This queue publishes gaussianzoom_stage2 markers; never inherit a
        # different backend when the generic train.py default changes.
        args += ["--idu_refine_backend", "gaussianzoom"]
    if compact:
        args += COMPACT_ARGS
    command = [os.environ.get("SKYFALL_PYTHON", ENV_DEFAULTS["SKYFALL_PYTHON"]), "-u",
               str(project / "train.py"), "-s", str(source_path), "-m", job["output_dir"],
               "--eval", "--port", str(job["port"])]
    if start_checkpoint:
        command += ["--start_checkpoint", str(start_checkpoint)]
    return command + args


def dependency_key(job: dict) -> str | None:
    return f"{job['scene']}/stage1" if job["stage"] == "stage2" else None


def plan_payload(project: Path, run_root: Path, control_root: Path, jobs: list,
                 compact: bool = True) -> dict:
    entries = []
    for job in jobs:
        entry = dict(job)
        entry["stage1_checkpoint"] = str(Path(job["output_dir"]).parent / "stage1"
                                        / f"chkpnt{STAGE1_ITERATIONS}.pth") \
            if job["stage"] == "stage2" else None
        if job["kind"] == "reuse":
            entry["source_path"] = str(resolve_source(project, job["dataset"], job["scene"],
                                                      strict=False))
            entry["command"] = None
            entry["note"] = "reused qualified run: hard-link finals + publish marker"
        else:
            source = resolve_source(project, job["dataset"], job["scene"], strict=False)
            start = entry["stage1_checkpoint"]
            entry["source_path"] = str(source)
            entry["source_exists"] = _has_transforms(source)
            entry["command"] = job_command(job, project, source, start, compact)
        entries.append(entry)
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "branch": BRANCH,
        "generated_at": now_iso(),
        "project_root": str(project),
        "run_root": str(run_root),
        "control_root": str(control_root),
        "archive_root": DEFAULT_ARCHIVE_ROOT,
        "manifest": DEFAULT_MANIFEST,
        "scenes": [entry["scene"] for entry in SCENES],
        "stage1_iterations": STAGE1_ITERATIONS,
        "stage2_episode_iterations": STAGE2_EPISODE_ITERATIONS,
        "stage2_episodes": STAGE2_EPISODES,
        "stage2_final_iteration": STAGE2_FINAL_ITERATION,
        "compact_retention": compact,
        "gpu": GPU_INDEX,
        "cpu_threads": 8,
        "matrix_note": ("Scenes run strictly sequentially on remote GPU 0; a train job starts only "
                        "when its Stage 1 checkpoint exists and the filesystem reserve is free."),
        "jobs": entries,
    }


# --------------------------------------------------------------------------- #
# Preflight / verification / markers
# --------------------------------------------------------------------------- #

def preflight_train(job: dict, state: dict, project: Path, run_root: Path) -> tuple:
    """Return (source_path, start_checkpoint) or raise with a precise reason."""
    source = resolve_source(project, job["dataset"], job["scene"])
    if job["stage"] == "stage2":
        dep = state["jobs"].get(dependency_key(job), {})
        start = dep.get("checkpoint") or str(run_root / job["scene"] / "stage1"
                                            / f"chkpnt{STAGE1_ITERATIONS}.pth")
        if not Path(start).is_file():
            raise FileNotFoundError(f"missing Stage 1 checkpoint for {job['scene']}: {start}")
        return source, start
    return source, None


def detect_iteration(stage_dir: Path) -> int | None:
    checkpoints = [int(path.stem.removeprefix("chkpnt")) for path in stage_dir.glob("chkpnt*.pth")
                   if path.stem.removeprefix("chkpnt").isdigit()]
    return max(checkpoints) if checkpoints else None


def verify_stage_artifacts(stage_dir: Path, iteration: int | None = None,
                           *, expect_episodes: int | None = None) -> dict:
    """Verify the final checkpoint + matching PLY pair exists (both are required)."""
    iteration = iteration if iteration is not None else detect_iteration(stage_dir)
    result = {"iteration": iteration, "errors": [], "warnings": [],
              "checkpoint": None, "point_cloud": None}
    if iteration is None:
        result["errors"].append("no chkpnt*.pth in stage directory")
        return result
    checkpoint = stage_dir / f"chkpnt{iteration}.pth"
    point_cloud = stage_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        result["errors"].append(f"checkpoint missing/empty: {checkpoint}")
    else:
        result["checkpoint"] = str(checkpoint)
    if not point_cloud.is_file() or point_cloud.stat().st_size == 0:
        result["errors"].append(
            f"matching PLY missing/empty: {point_cloud} "
            "(filter_3D lives in the PLY, capture() alone is not a faithful model)")
    else:
        result["point_cloud"] = str(point_cloud)
    for name in ("cfg_args", "cameras.json"):
        if not (stage_dir / name).is_file():
            result["errors"].append(f"missing {name}")
    if expect_episodes:
        manifest = read_json(stage_dir / "idu" / "manifest.json") or {}
        episodes = manifest.get("episodes", [])
        if len(episodes) != expect_episodes:
            result["errors"].append(
                f"Stage 2 recorded {len(episodes)} IDU episodes, expected {expect_episodes}")
        missing_meta = [entry.get("dirname") for entry in episodes
                        if entry.get("dirname")
                        and not (stage_dir / "idu" / entry["dirname"] / "episode_meta.json").is_file()]
        if missing_meta:
            result["errors"].append(f"episodes without episode_meta.json: {missing_meta}")
        missing_summary = [entry.get("dirname") for entry in episodes
                           if entry.get("dirname")
                           and not (stage_dir / "idu" / entry["dirname"]
                                    / "episode_summary.json").is_file()]
        if missing_summary:
            result["warnings"].append(
                f"episodes without compact summary (panels/retention may be incomplete): "
                f"{missing_summary}")
    return result


def publish_marker(job: dict, source_path: str, verification: dict, *, output_dir: Path,
                   extra: dict) -> dict:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "scene": job["scene"],
        "stage": job["stage"],
        "iteration": verification["iteration"],
        "output_dir": str(output_dir),
        "checkpoint": verification["checkpoint"],
        "point_cloud": verification["point_cloud"],
        "source_path": source_path,
        "method": METHOD,
        "branch": BRANCH,
        "recipe": job["recipe"],
        "created_at": now_iso(),
        "compact_retention": True,
        "artifacts": [],
        "warnings": verification.get("warnings", []),
        "notes": [
            "Final checkpoint and matching filter_3D PLY are both required for a faithful load; "
            "capture() omits filter_3D while the raw PLY omits the learned appearance "
            "embeddings/MLP.",
            "Raw datasets, pretrained weights and native CUDA builds stay on the training host.",
        ],
    }
    for role, path in (("checkpoint", verification["checkpoint"]),
                       ("point_cloud", verification["point_cloud"])):
        if not path:
            continue
        entry = {"role": role, "path": path, "bytes": os.path.getsize(path)}
        if extra.get("hash_artifacts"):
            entry["sha256"] = sha256_file(Path(path))
        payload["artifacts"].append(entry)
    payload.update({key: value for key, value in extra.items() if key != "hash_artifacts"})
    atomic_write_json(Path(job["marker"]), payload)
    return payload


def write_run_status(output_dir: Path, job: dict, status: str, *, exit_code=None,
                     error: str | None = None, command=None) -> None:
    path = output_dir / RUN_STATUS_NAME
    payload = read_json(path, {}) or {}
    payload.update({
        "schema_version": SCHEMA_VERSION,
        "scene": job["scene"],
        "stage": job["stage"],
        "status": status,
        "exit_code": exit_code,
        "updated_at": now_iso(),
        "output_dir": str(output_dir),
        "recipe": job["recipe"],
    })
    if command:
        payload["command"] = command
    if error:
        payload["error"] = error
    payload.setdefault("started_at", now_iso())
    if status in ("completed", "failed"):
        payload["finished_at"] = now_iso()
    atomic_write_json(path, payload)


def retained_inventory(stage_dir: Path) -> list:
    keep = ["cfg_args", "cameras.json", "console.log", "train.log", "input.ply",
            "run_status.json", "experiment_summary.json", "episode_comparison.jpg",
            "compact_retention.json"]
    items = []
    for name in keep:
        path = stage_dir / name
        if path.is_file():
            items.append({"path": str(path), "bytes": path.stat().st_size})
    for pattern in ("idu/*/episode_summary.json", "idu/*/panels/*.jpg", "idu/*/retained_views/*.png",
                    "idu/manifest.json", "idu/index.html"):
        for path in sorted(stage_dir.glob(pattern)):
            items.append({"path": str(path), "bytes": path.stat().st_size})
    for tfevent in sorted(stage_dir.glob("events.out.tfevents.*")):
        items.append({"path": str(tfevent), "bytes": tfevent.stat().st_size, "role": "compact_log"})
    return items


def episode_summaries(stage_dir: Path) -> list:
    summaries = []
    for path in sorted(stage_dir.glob("idu/episode_*/episode_summary.json")):
        data = read_json(path)
        if data:
            summaries.append({"episode": data.get("episode_idx"), "path": str(path),
                              "iteration_end": data.get("iteration_end"),
                              "removed_bytes": data.get("removed_bytes"),
                              "retained_bytes": data.get("retained_bytes"),
                              "metrics": data.get("metrics")})
    return summaries


# --------------------------------------------------------------------------- #
# Reuse (JAX_068) publication
# --------------------------------------------------------------------------- #

REUSE_FILES = {
    "stage1": [f"chkpnt{STAGE1_ITERATIONS}.pth", "cfg_args", "cameras.json", "input.ply",
               f"point_cloud/iteration_{STAGE1_ITERATIONS}/point_cloud.ply"],
    "stage2": [f"chkpnt{STAGE2_FINAL_ITERATION}.pth", "cfg_args", "cameras.json", "input.ply",
               "console.log", "run_status.json", "experiment_summary.json",
               "episode_comparison.jpg", "idu/manifest.json", "idu/index.html",
               f"point_cloud/iteration_{STAGE2_FINAL_ITERATION}/point_cloud.ply"],
}

# Compact provenance that may only exist after the origin run is compacted; the
# publisher is idempotent, so re-running it picks these up.
REUSE_GLOBS = {
    "stage1": ["compact_retention.json"],
    "stage2": ["compact_retention.json", "idu/*/episode_meta.json", "idu/*/episode_summary.json",
               "idu/*/compare.html", "idu/*/panels/*.jpg", "idu/*/retained_views/*.png",
               "idu/*/stage2_prepared.json", "idu/*/stage2_synthesis.json"],
}


def expected_episodes(job: dict) -> int | None:
    return STAGE2_EPISODES if job["stage"] == "stage2" else None


def publish_reuse(job: dict, state: dict, *, apply: bool = True, hash_artifacts: bool = True,
                  notify=None) -> dict:
    source_dir = Path(job["reuse_dir"])
    target_dir = Path(job["output_dir"])
    iteration = STAGE1_ITERATIONS if job["stage"] == "stage1" else STAGE2_FINAL_ITERATION
    verification = verify_stage_artifacts(source_dir, iteration,
                                          expect_episodes=expected_episodes(job))
    if verification["errors"]:
        raise RuntimeError(f"reused run for {job['key']} is incomplete: {verification['errors']}")
    if not apply:
        return {"action": "would_publish_reuse", "key": job["key"], "source_dir": str(source_dir),
                "target_dir": str(target_dir), "iteration": iteration}

    target_dir.mkdir(parents=True, exist_ok=True)
    link_paths = []

    def _link(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if (target.stat().st_ino == source.stat().st_ino
                    and target.stat().st_dev == source.stat().st_dev):
                link_paths.append(str(target))
                return
            target.unlink()
        os.link(source, target)
        link_paths.append(str(target))

    for name in REUSE_FILES[job["stage"]]:
        source = source_dir / name
        if source.is_file():
            _link(source, target_dir / name)
    for pattern in REUSE_GLOBS[job["stage"]]:
        for source in sorted(source_dir.glob(pattern)):
            if source.is_file():
                _link(source, target_dir / source.relative_to(source_dir))

    verification = verify_stage_artifacts(target_dir, iteration,
                                          expect_episodes=expected_episodes(job))
    if verification["errors"]:
        raise RuntimeError(f"hard-linked reuse target failed verification: {verification['errors']}")
    payload = publish_marker(
        job, str(resolve_source(project_root(), job["dataset"], job["scene"], strict=False)),
        verification,
        output_dir=target_dir,
        extra={
            "reused_from": str(source_dir),
            "hard_link_paths": link_paths,
            "hash_artifacts": hash_artifacts,
            "retained": retained_inventory(target_dir),
            "episode_summaries": episode_summaries(target_dir),
            "provenance": {
                "origin_run": str(source_dir),
                "note": ("Qualified completed run reused as-is; no retraining. Files in the stage "
                         "directory are hard links to the origin run, so no extra space is used "
                         "and the bytes are identical."),
            },
        },
    )
    if notify:
        notify("stage_published", {"scene": job["scene"], "stage": job["stage"],
                                   "marker": job["marker"], "reused_from": str(source_dir),
                                   "iteration": iteration})
    return {"action": "published_reuse", "key": job["key"], "marker": job["marker"],
            "iteration": iteration, "hard_links": len(link_paths), "artifacts": payload["artifacts"]}


# --------------------------------------------------------------------------- #
# Queue execution
# --------------------------------------------------------------------------- #

class Notifier:
    def __init__(self, control_root: Path):
        self.path = control_root / "notifications.jsonl"

    def __call__(self, kind: str, payload: dict) -> None:
        record = {"ts": now_iso(), "kind": kind, **payload}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[notify] {kind} {json.dumps(payload, ensure_ascii=False)}", flush=True)


def state_path(control_root: Path) -> Path:
    return control_root / "queue_state.json"


def load_state(control_root: Path) -> dict:
    return read_json(state_path(control_root), {}) or {}


def save_state(control_root: Path, state: dict) -> None:
    state["updated_at"] = now_iso()
    atomic_write_json(state_path(control_root), state)


def init_state(jobs: list, run_root: Path) -> dict:
    state = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "branch": BRANCH,
        "gpu": GPU_INDEX,
        "run_root": str(run_root),
        "queue_status": "running",
        "started_at": now_iso(),
        "jobs": {},
    }
    for job in jobs:
        status = "pending"
        marker = read_json(Path(job["marker"]), {})
        if marker.get("status") == "completed":
            status = "completed"
        state["jobs"][job["key"]] = {
            "scene": job["scene"], "stage": job["stage"], "kind": job["kind"],
            "status": status, "attempts": 0, "marker": job["marker"],
            "output_dir": job["output_dir"], "reserve_gb": job["reserve_gb"],
        }
    return state


DEFAULT_ARCHIVE_ROOT = "/dataset/Skyfall-GS/experiments/stage2_gaussianzoom_20260913"


def cleanup_clearance(job: dict, control_root: Path, archive_root: Path,
                      run_root: Path | None = None,
                      published_artifacts: Sequence[dict] | None = None) -> dict:
    """Decide whether a published stage's remote finals may be deleted.

    Gate, in order (all must hold):
      1. ``archive_status.json`` status == ``verified`` (ArchiveCurator, after SHA256
         verification of the transferred set);
      2. ``evaluation_status.json`` ``model_load_verified == true`` (DatasetEvaluator,
         published as soon as the model loads and the first renders succeed, so the
         full video/CLIP-FID/CMMD pass can finish later);
      3. the evaluation's recorded ``checkpoint_sha256`` / ``point_cloud_sha256`` equal
         the hashes of the assets THIS queue published.  Without this, a same-named
         scene's stale marker (or a blanket bypass file) could release a newer model
         that was never verified at all;
      4. Stage 1 only: the same scene's Stage 2 has published ``stage_complete.json``
         with status completed -- Stage 2 re-reads and re-validates the Stage 1
         checkpoint during its episodes.

    Anything that cannot be proven denies deletion, which is the safe direction.
    """
    archive_marker = read_json(archive_root / job["scene"] / job["stage"]
                               / "archive_status.json", {}) or {}
    eval_dir = archive_root / job["scene"] / "evaluation" / job["stage"]
    evaluation_marker = read_json(eval_dir / "evaluation_status.json", {}) or {}
    override = read_json(control_root / "clearance" / job["scene"]
                         / f"{job['stage']}.json", {}) or {}

    sha_verified = (archive_marker.get("status") == "verified"
                    or override.get("local_sha256_verified") is True)
    load_verified = (evaluation_marker.get("model_load_verified") is True
                     or override.get("model_load_verified") is True)

    published = {entry["role"]: (entry.get("sha256") or "").lower()
                 for entry in (published_artifacts or []) if entry.get("role") and entry.get("sha256")}
    hash_reasons = []
    if not published:
        hash_reasons.append("this queue's stage marker records no artifact sha256, so asset "
                            "identity cannot be proven")
    else:
        for role, key in (("checkpoint", "checkpoint_sha256"),
                          ("point_cloud", "point_cloud_sha256")):
            expected = published.get(role)
            recorded = (evaluation_marker.get(key) or "").lower() or None
            if expected is None:
                hash_reasons.append(f"stage marker has no {role} sha256")
            elif recorded is None:
                hash_reasons.append(f"evaluation_status has no {key}")
            elif recorded != expected:
                hash_reasons.append(
                    f"{key} mismatch: evaluation verified {recorded[:16]}... but this queue "
                    f"published {expected[:16]}...")
    identity_ok = not hash_reasons

    stage2_done = None
    stage2_marker = None
    if job["stage"] == "stage1":
        base = Path(run_root) if run_root else Path(job["output_dir"]).parent.parent
        stage2_marker = base / job["scene"] / "stage2" / MARKER_NAME
        stage2_done = (read_json(stage2_marker, {}) or {}).get("status") == "completed"

    reasons = []
    if not sha_verified:
        reasons.append(f"archive_status at {archive_marker.get('status')!r} is not 'verified'")
    if not load_verified:
        reasons.append("no evaluation_status with model_load_verified=true")
    reasons.extend(hash_reasons)
    if stage2_done is False:
        reasons.append(
            f"same-scene Stage 2 has not completed ({stage2_marker}); Stage 1 finals must stay "
            "remote while Stage 2 still reads and validates the Stage 1 checkpoint")
    return {
        "allowed": sha_verified and load_verified and identity_ok and stage2_done is not False,
        "sha256_verified": sha_verified,
        "model_load_verified": load_verified,
        "asset_identity_verified": identity_ok,
        "published_sha256": published,
        "same_scene_stage2_completed": stage2_done,
        "stage2_marker": str(stage2_marker) if stage2_marker else None,
        "reasons": reasons,
        "archive_status": str(archive_root / job["scene"] / job["stage"] / "archive_status.json"),
        "evaluation_status": str(eval_dir / "evaluation_status.json"),
        "override": str(control_root / "clearance" / job["scene"] / f"{job['stage']}.json"),
    }


def reap(jobs: list, state: dict, control_root: Path, notify: Notifier, *, apply: bool = True,
         archive_root: Path | None = None) -> list:
    """Delete remote finals of a published stage once local verification is signalled."""
    archive_root = Path(archive_root or DEFAULT_ARCHIVE_ROOT)
    actions = []
    # The cleanup gate must identify the CURRENT frozen assets, not just trust a
    # same-named scene's booleans.  Require the evaluation to have hashed exactly the
    # checkpoint+PLY this queue published, so a stale/other-run marker cannot release
    # a newer model.
    for job in jobs:
        record = state["jobs"].get(job["key"], {})
        if record.get("status") == "cleared":
            continue
        if record.get("status") not in ("completed", "completed_pinned_until_stage2"):
            continue
        run_root = Path(record.get("output_dir", "")).parent.parent if record.get("output_dir") else None
        marker = read_json(Path(job["marker"]), {}) or {}
        clearance = cleanup_clearance(
            job, control_root, archive_root, run_root=run_root,
            published_artifacts=[{"role": a.get("role"), "sha256": (a.get("sha256") or "").lower()}
                                 for a in marker.get("artifacts", []) if a.get("sha256")])
        if not clearance["allowed"]:
            if clearance["same_scene_stage2_completed"] is False:
                record["status"] = "completed_pinned_until_stage2"
                record["pinned_reason"] = clearance["reasons"][-1]
            actions.append({"key": job["key"], "action": "clearance_incomplete",
                            "reasons": clearance["reasons"]})
            continue
        if record.get("status") == "completed_pinned_until_stage2":
            record["status"] = "completed"          # pin released by the completed Stage 2
            record.pop("pinned_reason", None)
        marker = read_json(Path(job["marker"]), {})
        targets = []
        for artifact in marker.get("artifacts", []):
            if artifact.get("path"):
                targets.append(artifact["path"])
        targets.extend(marker.get("hard_link_paths", []))
        removed = []
        freed = 0
        seen = set()
        for path in targets:
            if path in seen:
                continue
            seen.add(path)
            candidate = Path(path)
            if not candidate.is_file():
                continue
            size = candidate.stat().st_size
            if apply:
                candidate.unlink()
            removed.append(str(candidate))
            freed += size
        if apply:
            report = {
                "schema_version": SCHEMA_VERSION,
                "scene": job["scene"], "stage": job["stage"],
                "status": "deleted_remote_final",
                "deleted_at": now_iso(),
                "clearance": clearance,
                "deleted": removed,
                "bytes_freed": freed,
                "note": ("Remote finals deleted only after a locally verified archive and a "
                         "successful local model load; the local archive is now authoritative."),
            }
            atomic_write_json(Path(job["output_dir"]) / CLEARANCE_NAME, report)
            record["status"] = "cleared"
            record["cleared_at"] = report["deleted_at"]
            record["bytes_freed"] = freed
            notify("stage_final_cleared", {"scene": job["scene"], "stage": job["stage"],
                                           "bytes_freed": freed, "deleted": len(removed)})
        actions.append({"key": job["key"], "action": "deleted_remote_final", "freed_bytes": freed,
                        "files": removed})
    return actions


def run_train_job(job: dict, state: dict, project: Path, run_root: Path, control_root: Path,
                  notify: Notifier, args) -> dict:
    record = state["jobs"][job["key"]]
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    record.update({"status": "running", "started_at": now_iso(), "attempts": record["attempts"] + 1})
    save_state(control_root, state)
    notify("stage_started", {"scene": job["scene"], "stage": job["stage"], "attempt": record["attempts"]})
    try:
        source, start = preflight_train(job, state, project, run_root)
    except Exception as exc:  # a bad scene must not take down the whole queue
        record.update({"status": "failed", "finished_at": now_iso(),
                       "error": f"preflight: {exc}"})
        write_run_status(output_dir, job, "failed", error=f"preflight: {exc}")
        save_state(control_root, state)
        notify("stage_failed", {"scene": job["scene"], "stage": job["stage"],
                                "reason": "preflight", "error": str(exc)})
        print(f"[queue] {job['key']} preflight failed: {exc}", flush=True)
        return {"status": "failed", "error": str(exc)}
    command = job_command(job, project, source, start, compact=not args.no_compact)
    record.update({"source_path": str(source), "start_checkpoint": start, "command": command})
    write_run_status(output_dir, job, "running", command=command)
    log_path = output_dir / "train.log"
    started = time.time()
    env = runtime_env(project)
    print(f"[queue] {job['key']} start ({time.time() - started:.1f}s preflight) -> {log_path}",
          flush=True)
    with open(log_path, "ab") as job_log:
        job_log.write((f"\n=== {job['key']} {now_iso()} ===\n"
                       + " ".join(command) + "\n").encode("utf-8"))
        job_log.flush()
        process = subprocess.Popen(command, cwd=str(project), env=env,
                                   stdout=job_log, stderr=subprocess.STDOUT)
    record["pid"] = process.pid
    save_state(control_root, state)
    return_code = process.wait()
    elapsed = time.time() - started
    record["exit_code"] = return_code
    record["seconds"] = round(elapsed, 1)

    if return_code != 0:
        tail = _tail(log_path, 40)
        record.update({"status": "failed", "finished_at": now_iso(), "error": f"exit {return_code}",
                       "log_tail": tail})
        write_run_status(output_dir, job, "failed", exit_code=return_code,
                         error=f"exit {return_code}", command=command)
        save_state(control_root, state)
        notify("stage_failed", {"scene": job["scene"], "stage": job["stage"],
                                "exit_code": return_code, "log": str(log_path)})
        return {"status": "failed", "exit_code": return_code}

    verification = verify_stage_artifacts(output_dir, job["expected_iteration"],
                                          expect_episodes=expected_episodes(job))
    if verification["errors"]:
        record.update({"status": "failed", "finished_at": now_iso(),
                       "error": "artifact verification failed: " + "; ".join(verification["errors"])})
        write_run_status(output_dir, job, "failed", exit_code=return_code,
                         error="; ".join(verification["errors"]), command=command)
        save_state(control_root, state)
        notify("stage_unverified", {"scene": job["scene"], "stage": job["stage"],
                                    "errors": verification["errors"]})
        return {"status": "failed", "errors": verification["errors"]}

    payload = publish_marker(
        job, str(source), verification, output_dir=output_dir,
        extra={
            "command": command,
            "environment": {key: env[key] for key in
                            ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                             "OPENBLAS_NUM_THREADS", "HF_HUB_OFFLINE", "VLM_PYTHON",
                             "DLORAL_PYTHON", "VLM_MODEL_PATH", "DLORAL_WEIGHT_ROOT")},
            "hash_artifacts": not args.no_hash,
            "retained": retained_inventory(output_dir),
            "episode_summaries": episode_summaries(output_dir),
            "compact": {
                "enabled": not args.no_compact,
                "intermediate_checkpoints": "removed",
                "episode_scratch": "removed after each completed episode",
            },
        },
    )
    write_run_status(output_dir, job, "completed", exit_code=return_code, command=command)
    record.update({"status": "completed", "finished_at": now_iso(), "iteration": verification["iteration"],
                   "checkpoint": verification["checkpoint"], "point_cloud": verification["point_cloud"],
                   "marker": job["marker"], "seconds": round(elapsed, 1),
                   "sha256": {item["role"]: item.get("sha256") for item in payload["artifacts"]}})
    save_state(control_root, state)
    notify("stage_published", {"scene": job["scene"], "stage": job["stage"], "marker": job["marker"],
                               "iteration": verification["iteration"],
                               "checkpoint": verification["checkpoint"],
                               "point_cloud": verification["point_cloud"],
                               "seconds": round(elapsed, 1)})
    return {"status": "completed", "iteration": verification["iteration"]}


def _tail(path: Path, lines: int) -> str:
    try:
        with open(path, "r", errors="replace") as handle:
            return "".join(handle.readlines()[-lines:])
    except OSError:
        return ""


def deps_satisfied(job: dict, state: dict) -> tuple:
    """Return (satisfied, terminal_reason).

    ``cleared`` counts as satisfied: the dependency's work is done, and whether its
    artifacts are still present is decided by the stage's own preflight (which
    reads the real files) rather than by queue bookkeeping.
    """
    dep = dependency_key(job)
    if dep is None:
        return True, None
    record = state["jobs"].get(dep, {})
    status = record.get("status")
    if status in ("completed", "cleared", "completed_pinned_until_stage2"):
        return True, None
    if status in ("failed", "blocked_disk", "blocked_dependency"):
        return False, f"dependency {dep} is {status}"
    return False, None


def reconcile_running_jobs(state: dict, jobs: list, notify: Notifier, save) -> None:
    """Adopt an existing trainer and reconcile it only after its process exits."""
    for job in jobs:
        record = state["jobs"].get(job["key"], {})
        output_dir = Path(job["output_dir"])
        status = read_json(output_dir / RUN_STATUS_NAME, {}) or {}
        if record.get("status") != "running":
            if (record.get("status") not in ("pending", "blocked_disk", "blocked_dependency")
                    or status.get("status") != "running"):
                continue
            record.update(status="running", pid=status.get("pid"),
                          command=status.get("command", record.get("command")),
                          attempts=max(1, record.get("attempts", 0)))
        pid = record.get("pid")
        alive = False
        if isinstance(pid, int) and pid > 0:
            try:
                argv = (Path(f"/proc/{pid}/cmdline").read_bytes()
                        .decode("utf-8", "replace").split("\0"))
                alive = (str(output_dir) in argv
                         and any(Path(arg).name == "train.py" for arg in argv if arg))
            except OSError:
                pass
        if alive:
            if record.get("recovered") != "waiting_for_existing_trainer":
                record["recovered"] = "waiting_for_existing_trainer"
                notify("stage_recovered_alive", {"scene": job["scene"], "stage": job["stage"],
                                                 "pid": pid})
                save()
            continue
        verification = verify_stage_artifacts(output_dir, job["expected_iteration"],
                                              expect_episodes=expected_episodes(job))
        if not verification["errors"] and status.get("status") != "failed":
            record["status"] = "pending"
            record["recovered"] = "trainer exited with complete artifacts; publish marker"
            notify("stage_recovered_complete", {"scene": job["scene"], "stage": job["stage"],
                                                "iteration": verification["iteration"]})
        else:
            record["status"] = "failed"
            record["error"] = ("previous trainer exited without verified success: "
                               + "; ".join(verification["errors"]))
            record["run_status_at_recovery"] = status.get("status")
            notify("stage_recovered_failed", {"scene": job["scene"], "stage": job["stage"],
                                              "errors": verification["errors"],
                                              "run_status": status.get("status")})
        save()


def run_queue(args) -> int:
    project = project_root()
    run_root = Path(args.run_root)
    control_root = Path(args.control_root)
    run_root.mkdir(parents=True, exist_ok=True)
    control_root.mkdir(parents=True, exist_ok=True)
    # One controller per queue; keep this handle live until run_queue returns.
    queue_lock = (control_root / ".queue.lock").open("a+")
    try:
        if args.wait_lock:
            print("[queue] waiting for exclusive queue ownership; active training is untouched", flush=True)
        fcntl.flock(queue_lock.fileno(), fcntl.LOCK_EX | (0 if args.wait_lock else fcntl.LOCK_NB))
    except BlockingIOError:
        queue_lock.close()
        print("[queue] another controller owns this queue", flush=True)
        return 2
    notify = Notifier(control_root)
    only = [item for item in (args.only or "").split(",") if item]
    stages = [item for item in (args.stages or "").split(",") if item]
    all_jobs = build_jobs(project, run_root, [], [])
    jobs = build_jobs(project, run_root, only, stages)
    if not jobs:
        print("[queue] no jobs selected")
        return 2
    state = load_state(control_root)
    if not state.get("jobs") or args.reset:
        state = init_state(all_jobs, run_root)
        save_state(control_root, state)
    else:
        for job in all_jobs:
            state["jobs"].setdefault(job["key"], {
                "scene": job["scene"], "stage": job["stage"], "kind": job["kind"],
                "status": "pending", "attempts": 0, "marker": job["marker"],
                "output_dir": job["output_dir"], "reserve_gb": job["reserve_gb"]})
    if args.retry_failed:
        for job in jobs:
            record = state["jobs"][job["key"]]
            if record["status"] == "failed" and record["attempts"] < args.max_attempts:
                record["status"] = "failed_retry"
                notify("stage_requeued", {"scene": job["scene"], "stage": job["stage"],
                                          "attempts": record["attempts"]})
    state["queue_status"] = "running"
    save_state(control_root, state)

    idle = 0
    # blocked_* states are re-evaluated every round: disk can be freed and a failed
    # dependency can be retried, so they must come back into the runnable set.
    RE_EVALUATED = ("pending", "failed_retry", "blocked_disk", "blocked_dependency")
    while True:
        reconcile_running_jobs(state, all_jobs, notify, save=lambda: save_state(control_root, state))
        if any(record.get("status") == "running" for record in state["jobs"].values()):
            # A surviving/bootstrap trainer owns GPU0. Never start a second job.
            time.sleep(min(args.poll_seconds, 30))
            continue
        pending = [job for job in jobs
                   if state["jobs"][job["key"]]["status"] in RE_EVALUATED
                   and state["jobs"][job["key"]]["attempts"] < args.max_attempts]
        if not pending:
            break
        action = None
        blocked_disk = []
        blocked_dep = []
        for job in pending:
            record = state["jobs"][job["key"]]
            ok, reason = deps_satisfied(job, state)
            if not ok:
                if reason:
                    record["status"] = "blocked_dependency"
                    record["error"] = reason
                blocked_dep.append(job)
                continue
            record.pop("error", None)
            if job["kind"] == "reuse":
                action = ("reuse", job)
                break
            already = verify_stage_artifacts(Path(job["output_dir"]), job["expected_iteration"],
                                             expect_episodes=expected_episodes(job))
            if not already["errors"]:
                action = ("mark", job)
                break
            if record["status"] == "blocked_disk" and not already["errors"]:
                record["status"] = "pending"
            free = free_gb(run_root)
            if free >= job["reserve_gb"]:
                action = ("train", job)
                break
            record["blocked_reason"] = f"free {free:.1f} GB < reserve {job['reserve_gb']:.1f} GB"
            record["status"] = "blocked_disk"
            blocked_disk.append(job)

        if action:
            kind, job = action
            record = state["jobs"][job["key"]]
            record["status"] = "running"
            save_state(control_root, state)
            try:
                if kind == "reuse":
                    result = publish_reuse(job, state, apply=True,
                                           hash_artifacts=not args.no_hash, notify=notify)
                    record.update({"status": "completed", "finished_at": now_iso(),
                                   "iteration": result["iteration"],
                                   "reused_from": job["reuse_dir"]})
                    save_state(control_root, state)
                elif kind == "mark":
                    # Training already produced the pair (queue restarted mid-flight).
                    verification = verify_stage_artifacts(Path(job["output_dir"]),
                                                          job["expected_iteration"],
                                                          expect_episodes=expected_episodes(job))
                    if verification["errors"]:
                        raise RuntimeError("; ".join(verification["errors"]))
                    source = resolve_source(project, job["dataset"], job["scene"])
                    publish_marker(job, str(source), verification,
                                   output_dir=Path(job["output_dir"]),
                                   extra={"hash_artifacts": not args.no_hash,
                                          "retained": retained_inventory(Path(job["output_dir"])),
                                          "episode_summaries": episode_summaries(Path(job["output_dir"])),
                                          "recovered": "marker published from existing artifacts"})
                    write_run_status(Path(job["output_dir"]), job, "completed")
                    record.update({"status": "completed", "finished_at": now_iso(),
                                   "iteration": verification["iteration"],
                                   "checkpoint": verification["checkpoint"],
                                   "point_cloud": verification["point_cloud"]})
                    save_state(control_root, state)
                    notify("stage_published", {"scene": job["scene"], "stage": job["stage"],
                                               "marker": job["marker"],
                                               "recovered_from_existing_artifacts": True})
                else:
                    result = run_train_job(job, state, project, run_root, control_root,
                                           notify, args)
                    if result["status"] == "failed" and args.retry_failed:
                        state["jobs"][job["key"]]["status"] = "failed_retry"
                        save_state(control_root, state)
            except Exception as exc:  # one scene's bookkeeping error must not kill the queue
                record.update({"status": "failed", "finished_at": now_iso(),
                               "error": f"{kind}: {exc}"})
                save_state(control_root, state)
                notify("stage_failed", {"scene": job["scene"], "stage": job["stage"],
                                        "reason": kind, "error": str(exc)})
                print(f"[queue] {job['key']} {kind} failed: {exc}", flush=True)
            idle = 0
            continue

        # Nothing runnable: try to free space through verified archival clearance.
        reap(all_jobs, state, control_root, notify, apply=not args.no_reap,
             archive_root=Path(args.archive_root))
        free = free_gb(run_root)
        save_state(control_root, state)
        if not blocked_disk:
            # No job is waiting on space, so waiting cannot help: pending jobs are
            # either dependency-blocked (with or without a terminal reason) or
            # otherwise unrunnable. Never spin.
            unresolved = [job for job in blocked_dep
                          if state["jobs"][job["key"]]["status"] == "blocked_dependency"]
            detail = {job["key"]: state["jobs"][job["key"]].get("error", "dependency not completed")
                      for job in blocked_dep if job not in unresolved}
            state["queue_status"] = "blocked_dependency"
            state["blocked_detail"] = detail
            save_state(control_root, state)
            notify("queue_blocked", {"reason": "no_runnable_job",
                                     "blocked": [job["key"] for job in blocked_dep],
                                     "detail": detail})
            print(f"[queue] no runnable job; blocked={detail or [j['key'] for j in unresolved]}",
                  flush=True)
            return 1
        state["queue_status"] = "blocked_disk"
        state["disk"] = {"free_gb": round(free, 2), "path": str(run_root),
                         "blocked": [job["key"] for job in blocked_disk]}
        save_state(control_root, state)
        idle += 1
        if idle == 1:
            notify("queue_blocked", {"reason": "disk_reserve",
                                     "free_gb": round(free, 2),
                                     "blocked": [job["key"] for job in blocked_disk]})
        print(f"[queue] blocked on disk: free {free:.1f} GB, waiting "
              f"{args.poll_seconds}s for archival clearance", flush=True)
        if args.max_idle_polls and idle >= args.max_idle_polls:
            state["queue_status"] = "blocked_disk"
            save_state(control_root, state)
            return 1
        time.sleep(args.poll_seconds)

    missing = [job["key"] for job in all_jobs
               if state["jobs"].get(job["key"], {}).get("status") not in
               ("completed", "cleared", "completed_pinned_until_stage2")]
    not_selected = [job["key"] for job in all_jobs
                    if job not in jobs and state["jobs"].get(job["key"], {}).get("status") == "pending"]
    failed = [job["key"] for job in all_jobs
              if state["jobs"].get(job["key"], {}).get("status") in
              ("failed", "blocked_disk", "blocked_dependency")]
    outstanding = [key for key in missing if key not in not_selected]
    # A scoped run (--only/--stages) legitimately leaves unselected jobs pending; that is
    # not an error.  Only failed/blocked/unaccounted-for jobs make the queue unsuccessful.
    state["queue_status"] = "completed" if not failed and not outstanding else "completed_with_errors"
    state["missing"] = missing
    state["outstanding"] = outstanding
    state["not_selected"] = not_selected
    state["failed"] = failed
    state["finished_at"] = now_iso()
    save_state(control_root, state)
    notify("queue_finished", {"queue_status": state["queue_status"], "failed": failed,
                              "outstanding": outstanding, "not_selected": not_selected})
    print(f"[queue] finished: {state['queue_status']}; failed={failed}; "
          f"outstanding={outstanding}; not_selected={len(not_selected)}", flush=True)
    return 0 if not failed and not outstanding else 1


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #

def cmd_plan(args) -> int:
    project = project_root()
    jobs = build_jobs(project, Path(args.run_root),
                      [item for item in (args.only or "").split(",") if item],
                      [item for item in (args.stages or "").split(",") if item])
    payload = plan_payload(project, Path(args.run_root), Path(args.control_root), jobs,
                           compact=not args.no_compact)
    if args.manifest and Path(args.manifest).is_file():
        payload["manifest_check"] = check_manifest(Path(args.manifest))
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def check_manifest(manifest_path: Path) -> dict:
    manifest = read_json(manifest_path) or {}
    problems = []
    manifest_scenes = [entry.get("scene") for entry in manifest.get("scenes", [])]
    embedded = [entry["scene"] for entry in SCENES]
    if manifest_scenes and manifest_scenes != embedded:
        problems.append(f"scene list/order differs: manifest={manifest_scenes} embedded={embedded}")
    training = manifest.get("training", {})
    if training.get("stage1_iterations") not in (None, STAGE1_ITERATIONS):
        problems.append("stage1_iterations mismatch")
    if training.get("stage2_episode_iterations") not in (None, STAGE2_EPISODE_ITERATIONS):
        problems.append("stage2_episode_iterations mismatch")
    if training.get("stage2_episodes") not in (None, STAGE2_EPISODES):
        problems.append("stage2_episodes mismatch")
    if training.get("final_iteration") not in (None, STAGE2_FINAL_ITERATION):
        problems.append("final_iteration mismatch")
    contract = manifest.get("stage_publish_contract", {})
    if contract.get("status_value") not in (None, "completed"):
        problems.append("stage marker status_value mismatch")
    return {"manifest": str(manifest_path), "problems": problems,
            "ok": not problems, "manifest_scenes": len(manifest_scenes)}


def cmd_status(args) -> int:
    control_root = Path(args.control_root)
    state = load_state(control_root)
    if not state:
        print(f"[status] no queue state at {state_path(control_root)}")
        return 1
    rows = []
    for key, record in state.get("jobs", {}).items():
        rows.append((key, record.get("status"), record.get("iteration"),
                     record.get("seconds"), record.get("error") or record.get("blocked_reason") or ""))
    width = max(len(key) for key, *_ in rows) if rows else 6
    for key, status, iteration, seconds, note in rows:
        print(f"{key:<{width}}  {status:<20} iter={iteration} "
              f"seconds={seconds} {note}")
    missing = [key for key, record in state.get("jobs", {}).items()
               if record.get("status") not in ("completed", "cleared", "completed_pinned_until_stage2")]
    print(f"queue_status={state.get('queue_status')} missing={missing}")
    if args.json:
        print(json.dumps(state, indent=2, ensure_ascii=False))
    return 0 if not missing else 1


def cmd_publish_reuse(args) -> int:
    project = project_root()
    run_root = Path(args.run_root)
    control_root = Path(args.control_root)
    notify = Notifier(control_root)
    jobs = [job for job in build_jobs(project, run_root, [], [])
            if job["kind"] == "reuse"]
    if args.only:
        wanted = set(args.only.split(","))
        jobs = [job for job in jobs if job["scene"] in wanted]
    if not jobs:
        print("[reuse] no reuse jobs configured")
        return 0
    state = load_state(control_root) or init_state(build_jobs(project, run_root, [], []), run_root)
    results = []
    for job in jobs:
        if args.apply:
            result = publish_reuse(job, state, apply=True, hash_artifacts=not args.no_hash,
                                  notify=notify)
            state["jobs"].setdefault(job["key"], {})["status"] = "completed"
            state["jobs"][job["key"]]["reused_from"] = job["reuse_dir"]
            state["jobs"][job["key"]]["iteration"] = result["iteration"]
        else:
            result = publish_reuse(job, state, apply=False)
        results.append(result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    save_state(control_root, state)
    return 0


def reuse_origin_dirs() -> list:
    """Directories that belong to a previous run and are NOT owned by this queue.

    Compact retention must never touch them: their cleanup is exclusively the
    archive worker's job, gated on a verified local archive plus a successful
    local model load.  Only directories this queue created (or an explicit
    scratch/test path) may be compacted.
    """
    origins = []
    for entry in SCENES:
        for key in ("reuse_stage1", "reuse_stage2"):
            if entry.get(key):
                origins.append(Path(entry[key]))
    return origins


def is_reuse_origin(path: Path) -> Path | None:
    resolved = Path(path).resolve()
    for origin in reuse_origin_dirs():
        try:
            if resolved == origin.resolve() or origin.resolve() in resolved.parents:
                return origin
        except OSError:
            continue
    return None


def allowed_compact_roots(run_root: Path, extra: Sequence[str]) -> list:
    """Roots where ``compact`` is allowed to delete anything.

    There is deliberately no override: a previous run's directory (any
    ``reuse_stage1``/``reuse_stage2`` origin in the manifest) can never be
    compacted by this queue.  Its cleanup is the archive worker's job, gated on
    ``archive_status=verified`` plus the evaluator's ``model_load_verified``.
    """
    roots = [Path(run_root)]
    for item in extra:
        if item:
            roots.append(Path(item))
    for entry in os.environ.get("GZ_COMPACT_ALLOWED_ROOTS", "").split(os.pathsep):
        if entry:
            roots.append(Path(entry))
    return roots


def compact_guard(directory: Path, run_root: Path, extra: Sequence[str]) -> dict:
    """Decide whether ``compact`` may operate on ``directory`` (no bypass).

    Three independent refusals, none of them overridable:
      1. the directory belongs to a previous run (manifest reuse origin),
      2. it is outside the compactor's allowed roots,
      3. a ``stage_complete.json`` marker is already published there, so it is the
         frozen archive/evaluation source and must not change underneath them.
    """
    resolved = Path(directory).resolve()
    origin = is_reuse_origin(resolved)
    roots = allowed_compact_roots(run_root, extra)
    inside = None
    for root in roots:
        try:
            root_resolved = root.resolve()
        except OSError:
            continue
        if resolved == root_resolved or root_resolved in resolved.parents:
            inside = root_resolved
            break
    published = (resolved / MARKER_NAME).is_file()
    allowed = origin is None and inside is not None and not published
    reasons = []
    if origin is not None:
        reasons.append(
            f"belongs to a previous run ({origin}); previous-run cleanup is owned by the "
            "archive worker and gated on archive_status=verified + model_load_verified")
    if inside is None:
        reasons.append(
            f"outside the compactor's allowed roots {[str(r) for r in roots]}; this queue may "
            "only compact directories it created (the current run root) or an explicitly "
            "declared scratch root")
    if published:
        reasons.append(
            f"{MARKER_NAME} is published here, so this stage is the frozen archive/evaluation "
            "source; compacting it would invalidate the archive and evaluation gates")
    return {"allowed": allowed, "resolved": str(resolved), "origin": str(origin) if origin else None,
            "inside_root": str(inside) if inside else None, "published": published,
            "reasons": reasons, "allowed_roots": [str(r) for r in roots]}


def cmd_compact(args) -> int:
    sys.path.insert(0, str(REPO_ROOT))
    from utils.compact_retention import (compact_stage1_output, compact_stage_output)

    run_root = Path(args.run_root)
    extra_roots = [item for item in (args.allow_root or "").split(",") if item]
    targets = []
    if args.dir:
        targets.append((args.stage or "stage2", Path(args.dir)))
    else:
        for entry in SCENES:
            if args.only and entry["scene"] not in args.only.split(","):
                continue
            for stage in ("stage1", "stage2"):
                if args.stage and args.stage != stage:
                    continue
                targets.append((stage, run_root / entry["scene"] / stage))
    if not targets:
        print("[compact] no target directories selected")
        return 2

    refusals = []
    for stage, directory in targets:
        guard = compact_guard(directory, run_root, extra_roots)
        if not guard["allowed"]:
            refusals.append((stage, str(directory), guard))
    if refusals:
        for stage, directory, guard in refusals:
            print(f"[compact] REFUSED {stage} {directory}: {'; '.join(guard['reasons'])}")
        print("[compact] refusing the whole request; nothing was deleted")
        return 3

    report = {"apply": args.apply, "run_root": str(run_root),
              "allowed_roots": [str(r) for r in allowed_compact_roots(run_root, extra_roots)],
              "targets": []}
    for stage, directory in targets:
        if stage == "stage1":
            result = compact_stage1_output(directory, apply=args.apply,
                                           drop_tfevents=not args.keep_tfevents,
                                           keep_tfevents=args.keep_tfevents)
        else:
            result = compact_stage_output(directory, apply=args.apply,
                                          drop_tfevents=not args.keep_tfevents,
                                          keep_tfevents=args.keep_tfevents,
                                          drop_intermediate_iterations=not args.keep_intermediate)
        result["stage"] = stage
        report["targets"].append(result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        atomic_write_json(Path(args.out), report)
    return 0


def cmd_reap(args) -> int:
    project = project_root()
    run_root = Path(args.run_root)
    control_root = Path(args.control_root)
    notify = Notifier(control_root)
    jobs = build_jobs(project, run_root, [], [])
    state = load_state(control_root)
    if not state:
        print("[reap] no queue state; nothing to do")
        return 1
    actions = reap(jobs, state, control_root, notify, apply=args.apply,
                   archive_root=Path(args.archive_root))
    save_state(control_root, state)
    print(json.dumps(actions, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", default=os.environ.get("GZ_RUN_ROOT", DEFAULT_RUN_ROOT))
    parser.add_argument("--control-root",
                        default=os.environ.get("GZ_CONTROL_ROOT", DEFAULT_CONTROL_ROOT))
    parser.add_argument("--archive-root",
                        default=os.environ.get("GZ_ARCHIVE_ROOT", DEFAULT_ARCHIVE_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="emit the exact per-job commands and reserves")
    plan.add_argument("--only", default="")
    plan.add_argument("--stages", default="")
    plan.add_argument("--manifest", default=DEFAULT_MANIFEST)
    plan.add_argument("--out", default="")
    plan.add_argument("--no-compact", action="store_true")
    plan.set_defaults(func=cmd_plan)

    run = sub.add_parser("run", help="run the durable queue")
    run.add_argument("--only", default="")
    run.add_argument("--stages", default="")
    run.add_argument("--reset", action="store_true")
    run.add_argument("--poll-seconds", type=int, default=300)
    run.add_argument("--max-idle-polls", type=int, default=0)
    run.add_argument("--max-attempts", type=int, default=2)
    run.add_argument("--retry-failed", action="store_true",
                     help="explicitly requeue selected failures below max-attempts")
    run.add_argument("--wait-lock", action="store_true",
                     help="wait for the active controller to exit without interrupting training")
    run.add_argument("--no-reap", action="store_true")
    run.add_argument("--no-hash", action="store_true")
    run.add_argument("--no-compact", action="store_true")
    run.set_defaults(func=run_queue)

    status = sub.add_parser("status", help="print queue state")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    reuse = sub.add_parser("publish-reuse", help="publish reused qualified runs")
    reuse.add_argument("--only", default="")
    reuse.add_argument("--apply", action="store_true")
    reuse.add_argument("--no-hash", action="store_true")
    reuse.set_defaults(func=cmd_publish_reuse)

    compact = sub.add_parser("compact", help="offline compact retention for a finished stage")
    compact.add_argument("--stage", choices=["stage1", "stage2"], default="")
    compact.add_argument("--dir", default="")
    compact.add_argument("--only", default="")
    compact.add_argument("--apply", action="store_true")
    compact.add_argument("--allow-root", default="",
                         help="comma-separated extra scratch roots the compactor may operate in; "
                              "previous-run directories are refused regardless of this value")
    compact.add_argument("--keep-tfevents", action="store_true")
    compact.add_argument("--keep-intermediate", action="store_true")
    compact.add_argument("--out", default="")
    compact.set_defaults(func=cmd_compact)

    reap_cmd = sub.add_parser("reap", help="delete remote finals after verified archival")
    reap_cmd.add_argument("--apply", action="store_true")
    reap_cmd.set_defaults(func=cmd_reap)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
