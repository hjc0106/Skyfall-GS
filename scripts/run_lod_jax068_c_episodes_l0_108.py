#!/usr/bin/env python3
"""L0 30K start, freeze L0, five-episode shared L1, 108 supervision views.

Independent archive: skyfall-gs_exp/lod_jax068_c_episodes_l0_start_108/

Training densify / LR / Adam follow ``scripts/run_lod_jax068_c_two_scale.py``
on a cumulative step clock.  This is not the old L1-40K 0.1x continuation.
Probe writes ``probe/`` and is never a formal starting checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image

from arguments import ModelParams, PipelineParams
from lod.camera import zoom_stage_camera
from lod.coverage import visit_stats
from lod.dloral_flowedit_compare import (
    FLOWEDIT_MODEL_TYPE,
    FLOWEDIT_N_AVG,
    FLOWEDIT_N_MAX,
    FLOWEDIT_N_MIN,
    FLOWEDIT_SOURCE_PROMPT,
    FLOWEDIT_SRC_GUIDANCE,
    FLOWEDIT_TAR_GUIDANCE,
    FLOWEDIT_T_STEPS,
    FLOWEDIT_TARGET_PROMPT,
)
from lod.episodes_l0_108 import (
    ARCHIVE_NAME,
    CHECKPOINT_STEPS,
    DENSIFY_FROM,
    DENSIFY_GRAD_THRESHOLD,
    DENSIFY_INTERVAL,
    DENSIFY_UNTIL,
    EMPTY_L1_RGB_L1,
    EPISODE_SPECS,
    EXPECTED_VISITS_PER_IMAGE,
    GENERATION_SEED,
    IMAGE_SIZE,
    MAX_POINTS_L1,
    MIX_RATIO,
    N_EPISODES,
    N_POSES,
    N_SUPERVISION,
    PREVIOUS_L1_CONTINUATION_ARCHIVE,
    SAMPLES_PER_POSE,
    SCHEMA,
    STAGE1_PATH,
    STEPS_PER_EPISODE,
    TRAINING_SEED,
    TWO_SCALE_ENTRY,
    ZOOM_FACTOR,
    base_camera,
    camera_records,
    cumulative_step,
    densify_at_cumulative,
    episode_spec,
    heldout_records,
    neighbor_azimuths,
    neighbor_camera,
    parent_hash_mismatch,
    protocol_payload,
    spawn_camera_records,
    training_items,
    two_scale_l1_training_rules,
    zoom_camera,
)
from lod.freeze import (
    appearance_freeze_report,
    assert_snapshot_unchanged,
    completed_layers_frozen,
    snapshot_levels,
)
from lod.importer import add_detail_level, densify_with_appearance, save_bundle
from lod.inspect import active_opacity_stats
from lod.jax068 import TEST_VIEWS
from lod.jax068_runtime import (
    appearance_embedding as _embedding,
    build_context as _build_context,
    camera_payload as _camera_payload,
    drop_context as _drop_scene,
    file_identity as _identity,
    freeze_for_l1 as _freeze_for_l1,
    gpu_snapshot as _gpu,
    load_json as _load_json,
    load_rgb as _load_rgb,
    make_dloral_backend as _dloral_backend,
    neighbor_input as _neighbor_input,
    pair_warp as _pair_warp,
    render_bundle as _render,
    save_depth as _save_depth,
    save_gray as _save_gray,
    save_mosaic as _mosaic,
    sha256_file as _sha256,
    write_json as _write,
)
from lod.orbit_9c6a import (
    CENTER_ROI,
    L1_FEATURE_LR,
    L1_OPACITY_LR,
    L1_POSITION_LR,
    L1_ROTATION_LR,
    L1_SCALING_LR,
    MODEL_STEP_SCALE,
    assert_writable_output,
    make_orbit_camera,
    zoomed_orbit_camera,
)
from lod.path import DEFAULT_GZ_ROOT
from lod.render import render_lod_appearance
from lod.train_state import (
    TRAIN_STATE_NAME,
    capture_train_state,
    densify_spec,
    load_train_state,
    mix_rng,
    restore_train_state,
    save_train_state,
    spec_mismatches,
)
from refinement.qwen3_vlm import Qwen3PromptProvider
from refinement.flowedit_idu import FlowEditRefineIDU
from refinement.types import CameraSnapshot, PromptDescription, RefinementRequest
from train_zoom_gen import apply_stage1_cfg_to_args, load_stage1_cfg
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss
from utils.zoom_mvp_utils import image_hf_l1, image_l1, save_tensor_image


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "skyfall-gs_exp" / ARCHIVE_NAME
PREVIOUS_ARCHIVE = REPO_ROOT / "skyfall-gs_exp" / PREVIOUS_L1_CONTINUATION_ARCHIVE
ORIGINAL_TARGET_KEY = "__train_1x__"
PHASES = (
    "preflight",
    "probe",
    "supervise",
    "flowedit",
    "train",
    "evaluate",
    "episode",
    "all",
    "report",
)
PROBE_N_POSES = 2
PROBE_STEPS = 25
PROBE_INTERRUPT_AT = 15
PROBE_SWITCH_STEPS = 8
ORIGIN_LOOKAT = (0.0, 0.0, 0.0)


def _root(args) -> Path:
    return Path(args.output_dir).expanduser().resolve()


def _assert_output_isolated(path: str | Path) -> None:
    destination = Path(path).expanduser().resolve()
    previous = PREVIOUS_ARCHIVE.resolve()
    if destination == previous or previous in destination.parents:
        raise ValueError(
            f"refusing to write the L0-start experiment under completed archive {previous}"
        )


def _episode_root(args, index: int) -> Path:
    return _root(args) / f"episode{int(index):02d}"


def _supervision_root(args, index: int) -> Path:
    return _episode_root(args, index) / "supervision"


def _source_dir(args, item: Mapping[str, Any]) -> Path:
    return _supervision_root(args, int(item["episode"])) / "sources" / str(item["pose_id"])


def _dloral_dir(args, item: Mapping[str, Any]) -> Path:
    return _source_dir(args, item) / f"dloral_s{int(item['sample_id'])}"


def _target_dir(args, item: Mapping[str, Any]) -> Path:
    return _supervision_root(args, int(item["episode"])) / "targets" / str(item["supervision_id"])


def _train_root(args, index: int) -> Path:
    return _episode_root(args, index) / "train"


def _eval_root(args, index: int) -> Path:
    return _episode_root(args, index) / "eval"


def _items(args, episode: int) -> list[dict[str, Any]]:
    items = [dict(item) for item in training_items(episode)]
    n_poses = int(getattr(args, "n_poses", 0) or 0)
    if n_poses > 0:
        allowed = {rec["pose_id"] for rec in camera_records(episode)[:n_poses]}
        items = [item for item in items if item["pose_id"] in allowed]
    ids = [item["supervision_id"] for item in items]
    if len(set(ids)) != len(ids):
        raise RuntimeError("supervision IDs are not unique")
    return items


def _unique_cameras(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen = {}
    for item in items:
        seen.setdefault(item["pose_id"], {k: v for k, v in item.items() if k != "sample_id" and k != "supervision_id" and "seed" not in k})
    return list(seen.values())


def _code_version() -> dict[str, Any]:
    def git(*parts: str) -> str:
        return subprocess.check_output(["git", *parts], cwd=REPO_ROOT, text=True).strip()

    try:
        return {
            "git_commit": git("rev-parse", "HEAD"),
            "git_commit_short": git("rev-parse", "--short", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": str(exc)}


def _stage1_identities(checkpoint: str | Path) -> dict[str, Any]:
    path = Path(checkpoint).expanduser().resolve()
    root = path.parent
    files = [path, root / "cfg_args", root / "cameras.json"]
    files.extend(sorted(root.glob("*.appearance*")))
    files.extend(sorted(root.glob("*mlp*")))
    payload = {}
    for item in files:
        if item.is_file():
            payload[item.name] = _identity(item)
    return payload


def _is_stage1(path: Path) -> bool:
    return path.suffix == ".pth"


def _flowedit_src_prompt() -> str:
    return FLOWEDIT_SOURCE_PROMPT.replace("4x focal crop", f"{float(ZOOM_FACTOR):g}x focal crop")


def _release(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _spawn_extras(device: str):
    return [base_camera(rec, data_device=device) for rec in spawn_camera_records()]


def _stage1_args(args, output_dir: Path, lod_checkpoint: Path | str | None):
    local = copy.copy(args)
    local.output_dir = str(output_dir)
    if lod_checkpoint and not _is_stage1(Path(lod_checkpoint)):
        local.lod_checkpoint = str(lod_checkpoint)
    else:
        local.lod_checkpoint = ""
    stage1_cfg = load_stage1_cfg(str(Path(args.start_checkpoint).expanduser().resolve().parent))
    apply_stage1_cfg_to_args(local, stage1_cfg)
    return local


def _bind_spawn_stage_cameras(args, ctx: Mapping[str, Any]) -> None:
    extras = list(ctx.get("orbit_1x") or _spawn_extras(ctx["device"]))
    ctx["bundle"].stage_cameras = [
        zoom_stage_camera(
            camera,
            CENTER_ROI,
            ZOOM_FACTOR,
            gz_root=args.gz_root,
            device=ctx["device"],
        )
        for camera in extras
    ]


def _load_parent_context(args, checkpoint: Path, output_dir: Path, *, spawn_l1: bool = False):
    extras = _spawn_extras("cuda" if torch.cuda.is_available() else "cpu")
    local = _stage1_args(args, output_dir, checkpoint)
    ctx = _build_context(local, extra_1x=extras)
    if spawn_l1:
        if len(ctx["bundle"].lod.layers) != 1:
            raise RuntimeError(f"stage1 import must be L0-only, got {len(ctx['bundle'].lod.layers)} levels")
        add_detail_level(ctx["bundle"], extras, CENTER_ROI, ZOOM_FACTOR, gz_root=args.gz_root)
        if int(ctx["bundle"].layer(1).xyz.shape[0]) != 0:
            raise RuntimeError("spawned L1 must start empty")
    _bind_spawn_stage_cameras(args, ctx)
    return ctx


def _parent_for_episode(args, index: int) -> Path:
    if int(index) == 1:
        path = Path(args.start_checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    previous = _train_root(args, int(index) - 1) / "l1_final.lod.pt"
    if not previous.is_file():
        raise FileNotFoundError(f"episode {index} needs {previous}")
    appearance = Path(str(previous) + ".appearance.pt")
    if not appearance.is_file():
        raise FileNotFoundError(appearance)
    return previous


def _required_source_files(source: Path) -> tuple[Path, ...]:
    return (
        source / "render_input.png",
        source / "wide_1x.png",
        source / "neighbor.png",
        source / "geometry.json",
        source / "SOURCE_LOCK.json",
        source / "target_to_source_flow.npy",
        source / "source_to_target_flow.npy",
        source / "flow_valid.png",
        source / "reverse_valid.png",
    )


def _source_ready(source: Path, parent_sha: str, *, force: bool) -> bool:
    if not all(path.is_file() for path in _required_source_files(source)):
        return False
    saved = _load_json(source / "SOURCE_LOCK.json")
    saved_sha = str(((saved.get("config") or {}).get("parent_checkpoint") or {}).get("sha256") or "")
    if parent_hash_mismatch(saved_sha, parent_sha):
        if force:
            return False
        raise RuntimeError(
            f"parent checkpoint hash changed for {source}; refusing to reuse supervision "
            f"(saved={saved_sha} current={parent_sha})"
        )
    return True


def _dloral_ready(args, item: Mapping[str, Any], *, force: bool) -> bool:
    dloral = _dloral_dir(args, item)
    result_path = dloral / "dloral_result.json"
    image_path = dloral / "refined.png"
    if force or not image_path.is_file() or not result_path.is_file():
        return False
    result = _load_json(result_path)
    metadata = result.get("metadata") or {}
    seed = result.get("seed", metadata.get("seed", metadata.get("dloral_seed", -1)))
    return result.get("backend") == "dloral" and int(seed) == int(item["dloral_seed"])


def _flowedit_ready(args, item: Mapping[str, Any], parent_sha: str, *, force: bool) -> bool:
    target = _target_dir(args, item)
    if force or not (target / "refined.png").is_file() or not (target / "SUPERVISION.json").is_file():
        return False
    payload = _load_json(target / "SUPERVISION.json")
    saved_sha = str((payload.get("parent_checkpoint") or {}).get("sha256") or "")
    if parent_hash_mismatch(saved_sha, parent_sha):
        raise RuntimeError(
            f"parent checkpoint hash changed for {item['supervision_id']}; refusing to reuse FlowEdit"
        )
    return (
        int(payload.get("sample_id", -1)) == int(item["sample_id"])
        and int(payload.get("dloral_seed", -1)) == int(item["dloral_seed"])
        and int(payload.get("flowedit_seed", -1)) == int(item["flowedit_seed"])
        and _sha256(target / "refined.png") == payload.get("output_sha256")
    )


def phase_preflight(args) -> dict[str, Any]:
    root = _root(args)
    _assert_output_isolated(root)
    assert_writable_output(root)
    root.mkdir(parents=True, exist_ok=True)
    payload = protocol_payload()
    payload["training_rules"] = two_scale_l1_training_rules()
    payload["stage1"] = _stage1_identities(args.start_checkpoint)
    payload["code_version"] = _code_version()
    payload["output_dir"] = str(root)
    payload["generation_seed"] = int(args.generation_seed)
    payload["training_seed"] = int(args.training_seed)
    cameras = []
    for index in range(1, N_EPISODES + 1):
        for rec in camera_records(index):
            cameras.append(rec)
        payload.setdefault("heldout", []).append(
            {"episode": index, "n": len(heldout_records(index))}
        )
    protocol_path = root / "PROTOCOL.json"
    if protocol_path.is_file() and not args.force:
        saved = _load_json(protocol_path)
        locked = (
            "schema",
            "episodes",
            "camera",
            "generation",
            "training",
            "generation_seed",
            "training_seed",
        )
        mismatch = [key for key in locked if saved.get(key) != payload.get(key)]
        saved_stage1 = {
            name: value.get("sha256")
            for name, value in (saved.get("stage1") or {}).items()
        }
        current_stage1 = {
            name: value.get("sha256")
            for name, value in (payload.get("stage1") or {}).items()
        }
        if saved_stage1 != current_stage1:
            mismatch.append("stage1")
        if mismatch:
            raise SystemExit(
                f"{protocol_path} is locked with different {mismatch}; use a new output directory"
            )
    _write(protocol_path, payload)
    _write(root / "CAMERAS.json", {"n": len(cameras), "cameras": cameras})
    print(json.dumps({"preflight": str(protocol_path), "n_cameras": len(cameras)}, indent=2))
    return payload


def _save_pose_source(args, ctx, rec: Mapping[str, Any], parent: Path) -> dict[str, Any]:
    source = _source_dir(args, rec)
    source.mkdir(parents=True, exist_ok=True)
    base = base_camera(rec, data_device=ctx["device"])
    zoom = zoom_camera(rec, data_device=ctx["device"])
    embedding = _embedding(ctx, int(rec["uid_1x"]))
    target_render = _render(
        ctx["bundle"], zoom, background=ctx["background"], kernel=ctx["kernel"], embedding=embedding
    )
    wide = _render(
        ctx["bundle"], base, background=ctx["background"], kernel=ctx["kernel"], embedding=embedding
    )
    save_tensor_image(target_render["rgb"], str(source / "render_input.png"))
    save_tensor_image(wide["rgb"], str(source / "wide_1x.png"))
    _save_gray(target_render["alpha"], source / "alpha.png")
    _save_depth(target_render["camera_z"], source / "camera_z.npy")
    _write(source / "camera.json", {"base": _camera_payload(base), "zoom": _camera_payload(zoom), "pose": dict(rec)})
    candidates = []
    previews = {}
    for candidate_index, candidate_azimuth in enumerate(neighbor_azimuths(float(rec["azimuth_deg"]))):
        nbase = neighbor_camera(
            rec,
            candidate_azimuth,
            uid=int(rec["uid_neighbor"]) + candidate_index,
            data_device=ctx["device"],
        )
        nzoom = zoomed_orbit_camera(nbase, ZOOM_FACTOR, uid=int(rec["uid_neighbor"]) + candidate_index + 100)
        nrender = _render(
            ctx["bundle"], nzoom, background=ctx["background"], kernel=ctx["kernel"], embedding=embedding
        )
        pair = _pair_warp(zoom, target_render, nzoom, nrender)
        row = {
            "delta_deg": float(candidate_azimuth - float(rec["azimuth_deg"])),
            "azimuth_deg": float(candidate_azimuth),
            "name": str(nbase.image_name),
            "coverage": float(pair["coverage"]),
            "reverse_coverage": float(pair["reverse_coverage"]),
            "score": float(min(pair["coverage"], pair["reverse_coverage"])),
        }
        candidates.append(row)
        previews[float(candidate_azimuth)] = (nzoom, {**nrender, "pair": pair})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = candidates[0]
    selected_zoom, selected_render = previews[float(selected["azimuth_deg"])]
    pair = selected_render["pair"]
    save_tensor_image(selected_render["rgb"], str(source / "neighbor.png"))
    save_tensor_image(pair["warped"], str(source / "neighbor_warped.png"))
    Image.fromarray(pair["warp"].valid_mask.detach().cpu().numpy().astype(np.uint8) * 255, mode="L").save(
        source / "flow_valid.png"
    )
    Image.fromarray(pair["reverse"].valid_mask.detach().cpu().numpy().astype(np.uint8) * 255, mode="L").save(
        source / "reverse_valid.png"
    )
    np.save(source / "target_to_source_flow.npy", pair["warp"].pixel_flow.detach().cpu().float().numpy())
    np.save(source / "source_to_target_flow.npy", pair["reverse"].pixel_flow.detach().cpu().float().numpy())
    geometry = {
        "schema": "jax068_l0_start_108_geometry_v1",
        "pose_id": rec["pose_id"],
        "episode": int(rec["episode"]),
        "parent_checkpoint_sha256": _sha256(parent),
        "neighbor_lock": {"selected": selected, "candidates": candidates},
        "coverage": float(pair["coverage"]),
        "reverse_coverage": float(pair["reverse_coverage"]),
        "scene_scale": float(pair["scene_scale"]),
        "target_camera": _camera_payload(zoom),
        "neighbor_camera": _camera_payload(selected_zoom),
    }
    _write(source / "geometry.json", geometry)
    config = {
        "schema": "jax068_l0_start_108_source_lock_v1",
        "pose_id": rec["pose_id"],
        "episode": int(rec["episode"]),
        "zoom_factor": float(ZOOM_FACTOR),
        "parent_checkpoint": _identity(parent),
        "generation_seed": int(args.generation_seed),
        "neighbor": dict(selected),
    }
    _write(source / "SOURCE_LOCK.json", {"config": config})
    return {"pose_id": rec["pose_id"], "neighbor": selected, "source": str(source)}


def phase_render_inputs(args, index: int, parent: Path) -> dict[str, Any]:
    items = _items(args, index)
    cameras = _unique_cameras(items)
    parent_sha = _sha256(parent)
    todo = [
        rec for rec in cameras if not _source_ready(_source_dir(args, rec), parent_sha, force=args.force)
    ]
    if not todo:
        return {"episode": index, "rendered": 0, "reused": len(cameras)}
    ctx = _load_parent_context(args, parent, _episode_root(args, index) / "generation_context", spawn_l1=False)
    try:
        rows = [_save_pose_source(args, ctx, rec, parent) for rec in todo]
    finally:
        _drop_scene(ctx)
        _release(ctx)
    return {"episode": index, "rendered": len(rows), "reused": len(cameras) - len(rows)}


def phase_qwen(args, index: int) -> dict[str, Any]:
    cameras = _unique_cameras(_items(args, index))
    missing = [rec for rec in cameras if not (_source_dir(args, rec) / "prompt.json").is_file() or args.force]
    if not missing:
        return {"episode": index, "prompts": 0, "reused": len(cameras)}
    if not args.vlm_model_path:
        raise SystemExit("supervision generation requires --vlm_model_path")
    provider = Qwen3PromptProvider(
        args.vlm_model_path,
        python=args.vlm_python or sys.executable,
        device=args.vlm_device,
        max_new_tokens=int(args.vlm_max_new_tokens),
        max_image_size=int(args.vlm_max_image_size),
    )
    try:
        for rec in missing:
            source = _source_dir(args, rec)
            prompt = provider.describe(
                Image.open(source / "wide_1x.png").convert("RGB"),
                Image.open(source / "render_input.png").convert("RGB"),
                zoom_factor=ZOOM_FACTOR,
                level_index=0,
                context={"experiment": SCHEMA, "episode": int(index), "pose_id": rec["pose_id"]},
            )
            _write(source / "prompt.json", prompt.to_dict())
            print(json.dumps({"qwen": rec["pose_id"], "episode": index}, ensure_ascii=False), flush=True)
    finally:
        _release(provider)
    return {"episode": index, "prompts": len(missing), "reused": len(cameras) - len(missing)}


def _run_one_dloral(args, parent: Path, item: Mapping[str, Any], backend) -> None:
    source = _source_dir(args, item)
    dloral = _dloral_dir(args, item)
    dloral.mkdir(parents=True, exist_ok=True)
    prompt = PromptDescription.from_dict(_load_json(source / "prompt.json"))
    selected = _load_json(source / "geometry.json")["neighbor_lock"]["selected"]
    result = backend.refine(
        RefinementRequest(
            image=Image.open(source / "render_input.png").convert("RGB"),
            checkpoint=str(parent),
            camera=CameraSnapshot.from_camera(zoom_camera(item, data_device="cpu")),
            zoom_factor=ZOOM_FACTOR,
            sr_scale=1.0,
            prompt=prompt,
            level_index=0,
            metadata={
                "generation_seed": int(args.generation_seed),
                "seed": int(item["dloral_seed"]),
                "dloral_seed": int(item["dloral_seed"]),
                "sample_id": int(item["sample_id"]),
                "episode": int(item["episode"]),
                "pose_id": item["pose_id"],
                "geometry_guided": True,
                "alignment": "geometry",
                "neighbor_views": [_neighbor_input(source, selected)],
                "backend_save_dir": str(dloral / "worker"),
            },
        )
    )
    result.image.convert("RGB").save(dloral / "refined.png")
    payload = result.to_dict()
    payload["seed"] = int(item["dloral_seed"])
    payload["sample_id"] = int(item["sample_id"])
    payload["supervision_id"] = item["supervision_id"]
    _write(dloral / "dloral_result.json", payload)


def phase_dloral(args, index: int, parent: Path) -> dict[str, Any]:
    items = _items(args, index)
    pending = [item for item in items if not _dloral_ready(args, item, force=args.force)]
    if not pending:
        return {"episode": index, "dloral": 0, "reused": len(items)}
    backend = _dloral_backend(args)
    try:
        for item in pending:
            _run_one_dloral(args, parent, item, backend)
            print(json.dumps({"dloral": item["supervision_id"]}, ensure_ascii=False), flush=True)
    finally:
        _release(backend)
    return {"episode": index, "dloral": len(pending), "reused": len(items) - len(pending)}


def _finish_flowedit(args, item: Mapping[str, Any], parent: Path, pipe: FlowEditRefineIDU) -> dict[str, Any]:
    source = _source_dir(args, item)
    dloral = _dloral_dir(args, item)
    target = _target_dir(args, item)
    target.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with Image.open(dloral / "refined.png") as image:
        image_np = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    result = pipe.run(
        [image_np],
        src_prompt=_flowedit_src_prompt(),
        tar_prompt=FLOWEDIT_TARGET_PROMPT,
        T_steps=FLOWEDIT_T_STEPS,
        n_avg=FLOWEDIT_N_AVG,
        src_guidance_scale=FLOWEDIT_SRC_GUIDANCE,
        tar_guidance_scale=FLOWEDIT_TAR_GUIDANCE,
        n_min=FLOWEDIT_N_MIN,
        n_max=FLOWEDIT_N_MAX,
        seeds=[int(item["flowedit_seed"])],
        defer_decode=bool(getattr(args, "flowedit_defer_decode", False)),
    )[0].convert("RGB")
    result.save(target / "refined.png")
    flowedit_elapsed = time.perf_counter() - started
    flowedit_timing = copy.deepcopy(pipe.last_run_timing or {})
    payload = {
        "schema": "jax068_l0_start_108_supervision_item_v1",
        "episode": int(item["episode"]),
        "pose_id": item["pose_id"],
        "sample_id": int(item["sample_id"]),
        "supervision_id": item["supervision_id"],
        "parent_checkpoint": _identity(parent),
        "dloral_seed": int(item["dloral_seed"]),
        "flowedit_seed": int(item["flowedit_seed"]),
        "zoom_factor": float(ZOOM_FACTOR),
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "source": str((source / "render_input.png").resolve()),
        "dloral": str((dloral / "refined.png").resolve()),
        "refined": str((target / "refined.png").resolve()),
        "output_sha256": _sha256(target / "refined.png"),
        "flowedit_elapsed_sec": float(flowedit_elapsed),
        "flowedit_timing": flowedit_timing,
        "geometry_guided": True,
        "qwen_prompt": str((source / "prompt.json").resolve()),
    }
    _write(
        target / "flowedit_result.json",
        {
            "seed": int(item["flowedit_seed"]),
            "supervision_id": item["supervision_id"],
            "input": str((dloral / "refined.png").resolve()),
            "input_sha256": _sha256(dloral / "refined.png"),
            "output_sha256": _sha256(target / "refined.png"),
            "elapsed_sec": float(flowedit_elapsed),
            "timing": flowedit_timing,
            "defer_decode": bool(getattr(args, "flowedit_defer_decode", False)),
        },
    )
    _write(target / "SUPERVISION.json", payload)
    return payload


def _finish_flowedit_batch(args, items: Sequence[Mapping[str, Any]], parent: Path, pipe: FlowEditRefineIDU) -> list[dict[str, Any]]:
    images = []
    seeds = []
    for item in items:
        dloral = _dloral_dir(args, item)
        with Image.open(dloral / "refined.png") as image:
            images.append(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0)
        seeds.append(int(item["flowedit_seed"]))
    started = time.perf_counter()
    results = pipe.run(
        images,
        src_prompt=_flowedit_src_prompt(),
        tar_prompt=FLOWEDIT_TARGET_PROMPT,
        T_steps=FLOWEDIT_T_STEPS,
        n_avg=FLOWEDIT_N_AVG,
        src_guidance_scale=FLOWEDIT_SRC_GUIDANCE,
        tar_guidance_scale=FLOWEDIT_TAR_GUIDANCE,
        n_min=FLOWEDIT_N_MIN,
        n_max=FLOWEDIT_N_MAX,
        seeds=seeds,
        defer_decode=True,
    )
    batch_elapsed = time.perf_counter() - started
    summary = copy.deepcopy(pipe.last_run_timing or {})
    image_timings = list(summary.get("images") or [])
    batch_timing = {key: value for key, value in summary.items() if key != "images"}
    payloads = []
    for offset, (item, result) in enumerate(zip(items, results)):
        source = _source_dir(args, item)
        dloral = _dloral_dir(args, item)
        target = _target_dir(args, item)
        target.mkdir(parents=True, exist_ok=True)
        result = result.convert("RGB")
        result.save(target / "refined.png")
        image_timing = copy.deepcopy(image_timings[offset]) if offset < len(image_timings) else {}
        item_elapsed = float(image_timing.get("image_total_sec", batch_elapsed / max(len(items), 1)))
        flowedit_timing = {"batch": batch_timing, "image": image_timing}
        payload = {
            "schema": "jax068_l0_start_108_supervision_item_v1",
            "episode": int(item["episode"]),
            "pose_id": item["pose_id"],
            "sample_id": int(item["sample_id"]),
            "supervision_id": item["supervision_id"],
            "parent_checkpoint": _identity(parent),
            "dloral_seed": int(item["dloral_seed"]),
            "flowedit_seed": int(item["flowedit_seed"]),
            "zoom_factor": float(ZOOM_FACTOR),
            "image_size": [IMAGE_SIZE, IMAGE_SIZE],
            "source": str((source / "render_input.png").resolve()),
            "dloral": str((dloral / "refined.png").resolve()),
            "refined": str((target / "refined.png").resolve()),
            "output_sha256": _sha256(target / "refined.png"),
            "flowedit_elapsed_sec": item_elapsed,
            "flowedit_batch_elapsed_sec": float(batch_elapsed),
            "flowedit_timing": flowedit_timing,
            "geometry_guided": True,
            "qwen_prompt": str((source / "prompt.json").resolve()),
        }
        _write(
            target / "flowedit_result.json",
            {
                "seed": int(item["flowedit_seed"]),
                "supervision_id": item["supervision_id"],
                "input": str((dloral / "refined.png").resolve()),
                "input_sha256": _sha256(dloral / "refined.png"),
                "output_sha256": _sha256(target / "refined.png"),
                "elapsed_sec": item_elapsed,
                "batch_elapsed_sec": float(batch_elapsed),
                "timing": flowedit_timing,
                "defer_decode": True,
            },
        )
        _write(target / "SUPERVISION.json", payload)
        payloads.append(payload)
    return payloads


def phase_flowedit(args, index: int, parent: Path) -> dict[str, Any]:
    items = _items(args, index)
    parent_sha = _sha256(parent)
    pending = [item for item in items if not _flowedit_ready(args, item, parent_sha, force=args.force)]
    if not pending:
        return _write_manifest(args, index, parent, items)
    for item in pending:
        if not _dloral_ready(args, item, force=False):
            raise RuntimeError(f"FlowEdit requested before DLoRAL for {item['supervision_id']}")
    pipe = FlowEditRefineIDU(
        save_path=str(_supervision_root(args, index) / "flowedit_work"),
        device=args.flowedit_device,
        model_type=FLOWEDIT_MODEL_TYPE,
        model_path=args.flowedit_model_path,
    )
    try:
        if bool(getattr(args, "flowedit_defer_decode", False)):
            _finish_flowedit_batch(args, pending, parent, pipe)
            for item in pending:
                print(json.dumps({"flowedit": item["supervision_id"], "mode": "defer_decode"}, ensure_ascii=False), flush=True)
        else:
            for item in pending:
                _finish_flowedit(args, item, parent, pipe)
                print(json.dumps({"flowedit": item["supervision_id"]}, ensure_ascii=False), flush=True)
    finally:
        _release(pipe)
    return _write_manifest(args, index, parent, items)


def _write_manifest(args, index: int, parent: Path, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    parent_sha = _sha256(parent)
    rows = []
    for item in items:
        if not _flowedit_ready(args, item, parent_sha, force=False):
            raise RuntimeError(f"incomplete supervision {item['supervision_id']}")
        target = _target_dir(args, item) / "refined.png"
        with Image.open(target) as image:
            if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise RuntimeError(f"wrong size {target}: {image.size}")
        rows.append(
            {
                "supervision_id": item["supervision_id"],
                "pose_id": item["pose_id"],
                "sample_id": int(item["sample_id"]),
                "episode": int(item["episode"]),
                "dloral_seed": int(item["dloral_seed"]),
                "flowedit_seed": int(item["flowedit_seed"]),
                "uid_1x": int(item["uid_1x"]),
                "refined": str(target.resolve()),
                "sha256": _sha256(target),
            }
        )
    ids = [row["supervision_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate supervision IDs in manifest")
    by_pose = {}
    for row in rows:
        by_pose.setdefault(row["pose_id"], []).append(row)
    for pose_id, pair in by_pose.items():
        if len(pair) != SAMPLES_PER_POSE:
            raise RuntimeError(f"{pose_id} does not have {SAMPLES_PER_POSE} samples")
        if pair[0]["uid_1x"] != pair[1]["uid_1x"]:
            raise RuntimeError(f"{pose_id} dual samples do not share a camera uid")
        if pair[0]["dloral_seed"] == pair[1]["dloral_seed"]:
            raise RuntimeError(f"{pose_id} dual samples share a DLoRAL seed")
    manifest = {
        "schema": "jax068_l0_start_108_supervision_manifest_v1",
        "episode": int(index),
        "n_targets": len(rows),
        "n_poses": len(by_pose),
        "samples_per_pose": SAMPLES_PER_POSE,
        "parent_checkpoint": _identity(parent),
        "sampling": "current_episode_108_restored_images_only",
        "items": rows,
        "tiles": {row["supervision_id"]: row["refined"] for row in rows},
    }
    _write(_supervision_root(args, index) / "SUPERVISION_MANIFEST.json", manifest)
    _write(_episode_root(args, index) / "SUPERVISION.json", manifest)
    return manifest


def phase_supervise(args, index: int, parent: Path) -> dict[str, Any]:
    render = phase_render_inputs(args, index, parent)
    qwen = phase_qwen(args, index)
    dloral = phase_dloral(args, index, parent)
    return {"render": render, "qwen": qwen, "dloral": dloral}


def _validate_pool(args, index: int, parent: Path) -> list[dict[str, Any]]:
    manifest = _load_json(_supervision_root(args, index) / "SUPERVISION_MANIFEST.json")
    parent_sha = _sha256(parent)
    saved_sha = str((manifest.get("parent_checkpoint") or {}).get("sha256") or "")
    probe_switch = bool(getattr(args, "probe", False)) and int(index) > 1
    if not probe_switch and parent_hash_mismatch(saved_sha, parent_sha):
        raise RuntimeError("manifest parent hash does not match the training parent")
    items = _items(args, index)
    if int(manifest["n_targets"]) != len(items):
        raise RuntimeError(f"manifest n_targets {manifest['n_targets']} != {len(items)}")
    expected = {item["supervision_id"] for item in items}
    if set(manifest["tiles"]) != expected:
        raise RuntimeError("manifest IDs are not the current-episode pool")
    return items


def _plot_curve(curve: Sequence[Mapping[str, Any]], output: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    steps = [row["step"] for row in curve]
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    if "mean_rgb_l1_to_target" in curve[0]:
        ax.plot(steps, [row["mean_rgb_l1_to_target"] for row in curve], label="RGB L1")
    if "mean_hf_l1_to_target" in curve[0]:
        ax.plot(steps, [row["mean_hf_l1_to_target"] for row in curve], label="HF L1")
    ax.set_xlabel("local step")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _l1_param_snapshot(layer) -> dict[str, torch.Tensor]:
    return {
        name: getattr(layer, name).detach().float().cpu().clone()
        for name in ("xyz", "sh", "log_scales", "rotations", "opacity_logits")
        if getattr(layer, name, None) is not None
    }


def _required_optimizer_state(context: Mapping[str, Any], source: Path):
    state = context.get("lod_opt_state")
    if state is None:
        raise RuntimeError(
            f"{source} has no Adam state; keep_adam_across_episodes forbids a reset"
        )
    return state


def _required_train_state(path: Path, *, reason: str) -> dict[str, Any]:
    state = load_train_state(path)
    if state is None:
        raise RuntimeError(f"{reason} requires {path}")
    return state


def phase_train(args, index: int, parent: Path, *, steps: int | None = None, interrupt_at: int | None = None) -> dict[str, Any]:
    items = _validate_pool(args, index, parent)
    root = _train_root(args, index)
    root.mkdir(parents=True, exist_ok=True)
    n_steps = int(steps if steps is not None else args.steps)
    if n_steps != STEPS_PER_EPISODE and not bool(getattr(args, "probe", False)):
        raise SystemExit(f"formal episode steps are locked to {STEPS_PER_EPISODE}")
    state_path = root / TRAIN_STATE_NAME
    saved_state = load_train_state(state_path) if state_path.is_file() else None
    if (root / "TRAIN.json").is_file() and not args.force and interrupt_at is None:
        return _load_json(root / "TRAIN.json")
    orphan_checkpoints = sorted(root.glob("l1_step*.lod.pt"))
    if saved_state is None and orphan_checkpoints and not args.force:
        raise RuntimeError(
            f"{root} has local checkpoints but no {TRAIN_STATE_NAME}; "
            "refusing to restart with a different RNG/Adam trajectory"
        )
    resume = bool(saved_state is not None and not (root / "TRAIN.json").is_file())
    spawn_l1 = int(index) == 1 and not resume
    if resume:
        start_step = int(saved_state.get("local_step", saved_state.get("step") or 0))
        ckpt = root / f"l1_step{start_step:05d}.lod.pt"
        if not ckpt.is_file():
            raise RuntimeError(
                f"{state_path} points to local step {start_step}, but {ckpt} is missing"
            )
        ctx = _load_parent_context(args, ckpt, root / "context", spawn_l1=False)
    else:
        start_step = 0
        ctx = _load_parent_context(args, parent, root / "context", spawn_l1=spawn_l1)
    try:
        bundle = ctx["bundle"]
        freeze_report = _freeze_for_l1(bundle)
        if freeze_report["l0_trainable"] or not freeze_report["appearance"]["ok"]:
            raise RuntimeError(f"L0/appearance not frozen: {freeze_report}")
        if int(bundle.lod.active_level) != 1:
            raise RuntimeError(f"active level {bundle.lod.active_level} is not L1")
        optimizer = bundle.lod.make_optimizer(
            position_lr=L1_POSITION_LR,
            feature_lr=L1_FEATURE_LR,
            opacity_lr=L1_OPACITY_LR,
            scaling_lr=L1_SCALING_LR,
            rotation_lr=L1_ROTATION_LR,
        )
        if resume or int(index) > 1:
            source = ckpt if resume else parent
            optimizer_state = _required_optimizer_state(ctx, source)
            optimizer.load_state_dict(optimizer_state)
        expected_spec = densify_spec(
            densify_from=int(DENSIFY_FROM),
            densify_until=int(DENSIFY_UNTIL),
            densify_interval=int(DENSIFY_INTERVAL),
            densify_grad_threshold=float(DENSIFY_GRAD_THRESHOLD),
            max_points=int(MAX_POINTS_L1),
            mix_ratio=float(MIX_RATIO),
            seed=int(args.training_seed),
        )
        rng = mix_rng(int(args.training_seed))
        if resume and saved_state is not None:
            mismatch = spec_mismatches(saved_state.get("spec"), expected_spec)
            if mismatch:
                raise ValueError(f"resume spec differs on: {', '.join(mismatch)}")
            restore_train_state(saved_state, rng)
        elif int(index) > 1:
            previous_path = _train_root(args, int(index) - 1) / TRAIN_STATE_NAME
            previous_state = _required_train_state(
                previous_path,
                reason=(
                    f"episode {index} sampling-RNG continuation from episode {index - 1}"
                ),
            )
            restore_train_state(previous_state, rng)
        views = []
        for item in items:
            views.append(
                {
                    **item,
                    "base": base_camera(item, data_device=ctx["device"]),
                    "zoom": zoom_camera(item, data_device=ctx["device"]),
                    "target_path": _target_dir(args, item) / "refined.png",
                }
            )
        targets = [
            _load_rgb(view["target_path"], ctx["device"], image_size=IMAGE_SIZE)
            for view in views
        ]
        parents = {}
        with torch.no_grad():
            for view in views:
                rgb = _render(
                    bundle,
                    view["zoom"],
                    background=ctx["background"],
                    kernel=ctx["kernel"],
                    embedding=_embedding(ctx, int(view["uid_1x"])),
                    max_level=0 if spawn_l1 and start_step == 0 else None,
                )["rgb"]
                parents[view["supervision_id"]] = rgb.detach().clone()
                save_tensor_image(parents[view["supervision_id"]], str(root / "parents" / f"{view['supervision_id']}.png"))
        if spawn_l1 and start_step == 0:
            empty_rows = []
            with torch.no_grad():
                for view in _unique_cameras(items):
                    camera = zoom_camera(view, data_device=ctx["device"])
                    embedding = _embedding(ctx, int(view["uid_1x"]))
                    l0 = _render(
                        bundle, camera, background=ctx["background"], kernel=ctx["kernel"], embedding=embedding, max_level=0
                    )["rgb"]
                    both = _render(
                        bundle, camera, background=ctx["background"], kernel=ctx["kernel"], embedding=embedding
                    )["rgb"]
                    err = image_l1(both, l0)
                    empty_rows.append({"pose_id": view["pose_id"], "rgb_l1": err, "ok": bool(err < EMPTY_L1_RGB_L1)})
            _write(root / "EMPTY_L1.json", {"poses": empty_rows})
            if not all(row["ok"] for row in empty_rows):
                raise RuntimeError("empty L1 does not reproduce L0")
        frozen_before = snapshot_levels(bundle, (0,))
        appearance_before = appearance_freeze_report(bundle.appearance)
        sample_counts: Counter[str] = Counter()
        sequence: list[dict[str, Any]] = []
        visits_path = root / "VISITS.json"
        if visits_path.is_file() and start_step > 0:
            saved_visits = _load_json(visits_path)
            if int(saved_visits.get("step", -1)) != int(start_step):
                raise RuntimeError(
                    f"{visits_path} is at step {saved_visits.get('step')}, "
                    f"but resume checkpoint is step {start_step}"
                )
            sample_counts.update(saved_visits.get("counts") or {})
            sequence.extend(saved_visits.get("sequence") or [])
        for view in views:
            sample_counts.setdefault(view["supervision_id"], 0)
        sample_counts.setdefault(ORIGINAL_TARGET_KEY, 0)
        previous_ids = []
        if int(index) > 1:
            previous_ids = [item["supervision_id"] for item in training_items(int(index) - 1)]
            for item_id in previous_ids:
                sample_counts.setdefault(item_id, 0)
                if start_step == 0:
                    sample_counts[item_id] = 0
        densify_log = []
        if (root / "densify_log.json").is_file() and start_step > 0:
            densify_log = [
                row for row in _load_json(root / "densify_log.json") if int(row.get("step", 0)) <= start_step
            ]
        curve = []
        if (root / "absorption_curve.json").is_file() and start_step > 0:
            curve = [
                row
                for row in _load_json(root / "absorption_curve.json").get("curve") or []
                if int(row.get("step", 0)) <= start_step
            ]
        ckpt_steps = {value for value in CHECKPOINT_STEPS if 0 <= value <= n_steps}
        ckpt_steps.add(n_steps)
        if interrupt_at is not None:
            ckpt_steps.add(int(interrupt_at))

        def evaluate(local_step: int) -> dict[str, Any]:
            rows = []
            with torch.no_grad():
                for view, target in zip(views, targets):
                    rgb = _render(
                        bundle,
                        view["zoom"],
                        background=ctx["background"],
                        kernel=ctx["kernel"],
                        embedding=_embedding(ctx, int(view["uid_1x"])),
                    )["rgb"]
                    rows.append(
                        {
                            "supervision_id": view["supervision_id"],
                            "rgb_l1_to_target": image_l1(rgb, target),
                            "hf_l1_to_target": image_hf_l1(rgb, target),
                            "rgb_l1_vs_parent": image_l1(rgb, parents[view["supervision_id"]]),
                            "hf_l1_vs_parent": image_hf_l1(rgb, parents[view["supervision_id"]]),
                        }
                    )
                    if local_step in ckpt_steps:
                        save_tensor_image(rgb, str(root / "steps" / f"{local_step:05d}" / f"{view['supervision_id']}.png"))
            payload = {
                "episode": int(index),
                "step": int(local_step),
                "cumulative_step": cumulative_step(index, local_step),
                "mean_rgb_l1_to_target": float(np.mean([row["rgb_l1_to_target"] for row in rows])),
                "mean_hf_l1_to_target": float(np.mean([row["hf_l1_to_target"] for row in rows])),
                "mean_rgb_l1_vs_parent": float(np.mean([row["rgb_l1_vs_parent"] for row in rows])),
                "mean_hf_l1_vs_parent": float(np.mean([row["hf_l1_vs_parent"] for row in rows])),
                "n_points": {
                    f"L{level}": int(bundle.layer(level).xyz.shape[0])
                    for level in range(len(bundle.lod.layers))
                },
                "active": active_opacity_stats(bundle),
                "rows": rows,
            }
            return payload

        def write_visits(local_step: int, *, durable: bool = False) -> dict[str, Any]:
            current_ids = [view["supervision_id"] for view in views]
            previous_steps = int(sum(int(sample_counts[item_id]) for item_id in previous_ids))
            payload = {
                "episode": int(index),
                "step": int(local_step),
                "cumulative_step": cumulative_step(index, local_step),
                "mix_ratio": 0.0,
                "counts": dict(sample_counts),
                "sequence": list(sequence) if durable else [],
                "current_episode_steps": int(sum(int(sample_counts[item_id]) for item_id in current_ids)),
                "previous_episode_steps": previous_steps,
                "original_photo_steps": int(sample_counts[ORIGINAL_TARGET_KEY]),
                "n_targets": len(current_ids),
                "expected_visits_per_image": float(n_steps / max(len(current_ids), 1)),
                "stats": visit_stats(
                    {key: value for key, value in sample_counts.items() if key in current_ids},
                    expected=None,
                ),
            }
            _write(root / "VISITS_PROGRESS.json", payload)
            if durable:
                _write(visits_path, payload)
            return payload

        def write_state(local_step: int) -> None:
            payload = capture_train_state(step=int(local_step), mix=rng, spec=expected_spec)
            payload["local_step"] = int(local_step)
            payload["cumulative_step"] = cumulative_step(index, local_step)
            payload["episode"] = int(index)
            save_bundle(
                bundle,
                str(root / f"l1_step{local_step:05d}.lod.pt"),
                optimizer=optimizer,
            )
            save_train_state(str(state_path), payload)

        if start_step == 0:
            curve = [evaluate(0)]
            _write(root / "absorption_curve.json", {"curve": curve})
            write_state(0)
            write_visits(0, durable=True)

        first_grad = None
        param_before = _l1_param_snapshot(bundle.layer(1))
        densify_after_resume = 0
        started = time.time()
        stop_at = int(interrupt_at) if interrupt_at is not None else n_steps
        for local_step in range(int(start_step) + 1, stop_at + 1):
            optimizer.zero_grad(set_to_none=True)
            choice = int(rng.choice(range(len(views))))
            view = views[choice]
            sample_counts[view["supervision_id"]] += 1
            sequence.append({"step": int(local_step), "id": view["supervision_id"], "index": choice})
            package = render_lod_appearance(
                bundle,
                view["zoom"],
                background=ctx["background"],
                kernel_size=ctx["kernel"],
                appearance_embedding=_embedding(ctx, int(view["uid_1x"])),
                lod=True,
                compact=False,
            )
            loss = l1_loss(package["render"], targets[choice])
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at episode {index} step {local_step}")
            loss.backward()
            active = bundle.layer(1)
            if not any(getattr(active, name).requires_grad for name in ("xyz", "sh", "opacity_logits")):
                raise RuntimeError("L1 has no trainable tensors")
            if first_grad is None and int(active.xyz.shape[0]) > 0 and active.xyz.grad is not None:
                first_grad = float(active.xyz.grad.detach().float().abs().mean().item())
            densify_stats = None
            global_step = cumulative_step(index, local_step)
            if densify_at_cumulative(global_step) and package["viewspace_points"].grad is not None:
                screen_grad = package["viewspace_points"].grad.detach()[:, :2].norm(dim=-1)
                densify_stats = densify_with_appearance(
                    bundle,
                    optimizer,
                    screen_grad,
                    bundle.stage_cameras,
                    threshold=float(DENSIFY_GRAD_THRESHOLD),
                    max_points=int(MAX_POINTS_L1),
                    min_opacity=0.005,
                    scene_extent=float(ctx["scene"].cameras_extent),
                    percent_dense=0.01,
                )
                if resume and local_step > start_step:
                    densify_after_resume += 1
                if int(bundle.layer(0).xyz.shape[0]) != int(frozen_before["levels"]["0"]["n"]):
                    raise RuntimeError("frozen L0 point count changed")
            optimizer.step()
            if local_step % 50 == 0 or densify_stats or local_step in ckpt_steps:
                row = {
                    "step": int(local_step),
                    "cumulative_step": global_step,
                    "loss": float(loss.item()),
                    "densify": densify_stats or {},
                    "n_l1": int(bundle.layer(1).xyz.shape[0]),
                    "cuda": _gpu(),
                }
                densify_log.append(row)
                _write(root / "densify_log.json", densify_log)
                write_visits(local_step)
                print(
                    f"E{index} local {local_step} cum {global_step} loss={row['loss']:.6f} n_l1={row['n_l1']}",
                    flush=True,
                )
            if local_step in ckpt_steps:
                curve.append(evaluate(local_step))
                _write(root / "absorption_curve.json", {"curve": curve})
                _plot_curve(curve, root / "absorption_curve.png")
                write_state(local_step)
                write_visits(local_step, durable=True)
        visits = write_visits(stop_at, durable=stop_at in ckpt_steps)
        if visits["original_photo_steps"] != 0:
            raise AssertionError("original photos were sampled")
        if visits["previous_episode_steps"] != 0:
            raise AssertionError("previous-episode supervision was sampled")
        assert_snapshot_unchanged(frozen_before, snapshot_levels(bundle, (0,)))
        freeze = {
            "ok": bool(
                completed_layers_frozen(bundle, up_to_level=1)["ok"]
                and appearance_freeze_report(bundle.appearance)["ok"]
            ),
            "before": appearance_before,
            "after": appearance_freeze_report(bundle.appearance),
            "l0": freeze_report,
        }
        if not freeze["ok"]:
            raise AssertionError("freeze validation failed")
        final_path = root / "l1_final.lod.pt"
        if stop_at == n_steps:
            save_bundle(bundle, str(final_path), optimizer=optimizer)
            payload_state = capture_train_state(step=int(n_steps), mix=rng, spec=expected_spec)
            payload_state["local_step"] = int(n_steps)
            payload_state["cumulative_step"] = cumulative_step(index, n_steps)
            payload_state["episode"] = int(index)
            save_train_state(str(state_path), payload_state)
        param_after = _l1_param_snapshot(bundle.layer(1))
        updated = int(param_after["xyz"].shape[0]) != int(param_before["xyz"].shape[0]) or (
            int(param_after["xyz"].shape[0]) > 0
            and float((param_after["xyz"][: min(param_after["xyz"].shape[0], param_before["xyz"].shape[0])]
                       - param_before["xyz"][: min(param_after["xyz"].shape[0], param_before["xyz"].shape[0])]).abs().mean())
            > 0
        )
        payload = {
            "schema": "jax068_l0_start_108_train_v1",
            "episode": int(index),
            "probe": bool(getattr(args, "probe", False)),
            "steps": int(stop_at),
            "requested_steps": int(n_steps),
            "start_step": int(start_step),
            "cumulative_step": cumulative_step(index, stop_at),
            "n_targets": len(items),
            "checkpoint": str((final_path if stop_at == n_steps else root / f"l1_step{stop_at:05d}.lod.pt").resolve()),
            "n_l1": int(bundle.layer(1).xyz.shape[0]),
            "visits": visits,
            "freeze": freeze,
            "first_l1_xyz_grad": first_grad,
            "l1_updated": bool(updated),
            "densify_events_this_segment": densify_after_resume if resume else sum(
                1 for row in densify_log if row.get("densify")
            ),
            "seconds": time.time() - started,
            "gpu": _gpu(),
            "learning_rate": two_scale_l1_training_rules()["learning_rate"],
            "densify": two_scale_l1_training_rules()["densify"],
            "optimizer_policy": "keep_adam_across_episodes",
            "interrupted": interrupt_at is not None and stop_at < n_steps,
        }
        if stop_at == n_steps:
            _write(root / "TRAIN.json", payload)
            _write(root / "freeze.json", freeze)
        else:
            _write(root / "PROGRESS.json", payload)
        return payload
    finally:
        _drop_scene(ctx)
        _release(ctx)


def _ffmpeg_video(frame_dir: Path, output: Path, fps: int = 12) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required for evaluation videos")
    subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
            "-i", str(frame_dir / "%05d.png"),
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(output),
        ],
        check=True,
    )


def _render_trajectory(ctx, output: Path, *, mode: str, n_frames: int = 48) -> dict[str, Any]:
    frame_dir = output.parent / f"{output.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    cameras = []
    spec1 = episode_spec(1)
    spec5 = episode_spec(5)
    with torch.inference_mode():
        for index in range(n_frames):
            phase = float(index) / max(n_frames - 1, 1)
            if mode == "fixed_elevation_orbit":
                azimuth = 360.0 * phase
                elevation = float(spec1["elevation_deg"])
                radius = float(spec1["radius"])
            elif mode == "elevation_85_to_45":
                azimuth = 0.0
                elevation = float(spec1["elevation_deg"]) + (
                    float(spec5["elevation_deg"]) - float(spec1["elevation_deg"])
                ) * phase
                radius = float(spec1["radius"]) + (
                    float(spec5["radius"]) - float(spec1["radius"])
                ) * phase
            else:
                raise ValueError(mode)
            rec = {
                "episode": 1,
                "lookat": list(ORIGIN_LOOKAT),
                "azimuth_deg": float(azimuth),
                "elevation_deg": float(elevation),
                "radius": float(radius),
                "uid_1x": 900_000 + index,
                "uid_2x": 910_000 + index,
                "pose_id": f"traj_{mode}_{index}",
                "zoom_factor": ZOOM_FACTOR,
            }
            camera = zoom_camera(rec, data_device=ctx["device"])
            rgb = _render(
                ctx["bundle"],
                camera,
                background=ctx["background"],
                kernel=ctx["kernel"],
                embedding=_embedding(ctx, int(spawn_camera_records()[0]["uid_1x"])),
            )["rgb"]
            save_tensor_image(rgb.clamp(0, 1), str(frame_dir / f"{index:05d}.png"))
            cameras.append({"index": index, "mode": mode, **_camera_payload(camera), "elevation_deg": elevation, "radius": radius, "azimuth_deg": azimuth})
    _write(output.parent / f"{output.stem}_cameras.json", {"frames": cameras})
    _ffmpeg_video(frame_dir, output)
    return {"video": str(output), "n_frames": n_frames, "cameras": str(output.parent / f"{output.stem}_cameras.json")}


def _origin_items(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in items
        if abs(float(item["lookat"][0])) < 1e-6 and abs(float(item["lookat"][1])) < 1e-6
    ]


def phase_evaluate(args, index: int, parent: Path) -> dict[str, Any]:
    train = _load_json(_train_root(args, index) / "TRAIN.json")
    items = _items(args, index)
    output = _eval_root(args, index)
    output.mkdir(parents=True, exist_ok=True)
    ckpt = Path(train["checkpoint"])
    ctx = _load_parent_context(args, ckpt, output / "context", spawn_l1=False)
    try:
        origin = _origin_items(items)
        mosaic_rows = []
        for item in origin:
            source = _source_dir(args, item)
            dloral = _dloral_dir(args, item) / "refined.png"
            flowedit = _target_dir(args, item) / "refined.png"
            trained = _train_root(args, index) / "steps" / f"{int(args.steps):05d}" / f"{item['supervision_id']}.png"
            if not trained.is_file():
                camera = zoom_camera(item, data_device=ctx["device"])
                rgb = _render(
                    ctx["bundle"], camera, background=ctx["background"], kernel=ctx["kernel"],
                    embedding=_embedding(ctx, int(item["uid_1x"])),
                )["rgb"]
                trained = output / "trained" / f"{item['supervision_id']}.png"
                save_tensor_image(rgb, str(trained))
            mosaic_rows.append(
                [
                    (f"parent {item['supervision_id']}", source / "render_input.png"),
                    ("dloral", dloral),
                    ("flowedit", flowedit),
                    ("trained", trained),
                ]
            )
        if mosaic_rows:
            _mosaic(mosaic_rows, output / "absorption_mosaic.png", cell=256)
        cross_rows = []
        origin_s0 = [
            item
            for item in items
            if abs(float(item["lookat"][0])) < 1e-6
            and abs(float(item["lookat"][1])) < 1e-6
            and int(item["sample_id"]) == 0
        ]
        origin_s0.sort(key=lambda item: float(item["azimuth_deg"]))
        with torch.no_grad():
            rendered = {}
            for item in origin_s0:
                camera = zoom_camera(item, data_device=ctx["device"])
                pack = _render(
                    ctx["bundle"],
                    camera,
                    background=ctx["background"],
                    kernel=ctx["kernel"],
                    embedding=_embedding(ctx, int(item["uid_1x"])),
                )
                rendered[item["supervision_id"]] = (camera, pack)
            for left, right in zip(origin_s0, origin_s0[1:] + origin_s0[:1]):
                lcam, lpack = rendered[left["supervision_id"]]
                rcam, rpack = rendered[right["supervision_id"]]
                pair = _pair_warp(lcam, lpack, rcam, rpack)
                mask = pair["warp"].valid_mask.detach().float()
                warped = pair["warped"]
                rgb_err = float(((lpack["rgb"] - warped).abs() * mask).sum() / mask.sum().clamp_min(1.0)) if mask.numel() else None
                cross_rows.append(
                    {
                        "left": left["supervision_id"],
                        "right": right["supervision_id"],
                        "coverage": float(pair["coverage"]),
                        "rgb_l1_on_mask": rgb_err,
                    }
                )
        _write(output / "CROSS_VIEW.json", {"pairs": cross_rows})
        heldout_rows = []
        for rec in heldout_records(index):
            camera = zoom_camera(rec, data_device=ctx["device"])
            rgb = _render(
                ctx["bundle"], camera, background=ctx["background"], kernel=ctx["kernel"],
                embedding=_embedding(ctx, int(rec["uid_1x"])),
            )["rgb"]
            path = output / "heldout" / f"{rec['pose_id']}.png"
            save_tensor_image(rgb, str(path))
            heldout_rows.append({"pose_id": rec["pose_id"], "azimuth_deg": rec["azimuth_deg"], "path": str(path)})
        history = []
        for past in range(1, int(index)):
            for rec in camera_records(past):
                if abs(float(rec["lookat"][0])) > 1e-6 or abs(float(rec["lookat"][1])) > 1e-6:
                    continue
                camera = zoom_camera(rec, data_device=ctx["device"])
                rgb = _render(
                    ctx["bundle"], camera, background=ctx["background"], kernel=ctx["kernel"],
                    embedding=_embedding(ctx, int(rec["uid_1x"])),
                )["rgb"]
                path = output / "history" / f"{rec['pose_id']}_e{past}.png"
                save_tensor_image(rgb, str(path))
                history.append({"episode": past, "pose_id": rec["pose_id"], "path": str(path)})
        scale = {}
        rec0 = next(
            rec
            for rec in spawn_camera_records()
            if abs(float(rec["lookat"][0])) < 1e-6
            and abs(float(rec["lookat"][1])) < 1e-6
            and abs(float(rec["azimuth_deg"])) < 1e-6
        )
        rec0 = dict(rec0)
        for factor in (1.0, 2.0, 4.0):
            base = base_camera({**rec0, "azimuth_deg": 0.0, "uid_1x": rec0["uid_1x"]}, data_device=ctx["device"])
            camera = base if factor == 1.0 else zoomed_orbit_camera(base, factor, uid=int(rec0["uid_2x"]) + int(factor * 10))
            rgb = _render(
                ctx["bundle"], camera, background=ctx["background"], kernel=ctx["kernel"],
                embedding=_embedding(ctx, int(rec0["uid_1x"])),
            )["rgb"]
            path = output / "scale" / f"origin_az0_{factor:g}x.png"
            save_tensor_image(rgb, str(path))
            scale[f"{factor:g}x"] = str(path)
        videos = {}
        if not bool(getattr(args, "probe", False)):
            videos["orbit"] = _render_trajectory(ctx, output / "fixed_elevation_orbit.mp4", mode="fixed_elevation_orbit")
            videos["elevation"] = _render_trajectory(ctx, output / "elevation_85_to_45.mp4", mode="elevation_85_to_45")
        curve = _load_json(_train_root(args, index) / "absorption_curve.json")["curve"]
        _plot_curve(curve, output / "rgb_hf_curve.png")
        payload = {
            "episode": int(index),
            "heldout": heldout_rows,
            "cross_view": cross_rows,
            "history": history,
            "scale": scale,
            "videos": videos,
            "visits": _load_json(_train_root(args, index) / "VISITS.json"),
            "n_points": train.get("n_l1"),
            "seconds": train.get("seconds"),
            "gpu": train.get("gpu"),
        }
        _write(output / "EVAL.json", payload)
        return payload
    finally:
        _drop_scene(ctx)
        _release(ctx)


def _copy_pool_for_switch(args, src_episode: int, dst_episode: int) -> None:
    src_items = _items(args, src_episode)
    dst_items = _items(args, dst_episode)
    if len(src_items) != len(dst_items):
        raise RuntimeError("probe pool copy requires equal pool sizes")
    for src, dst in zip(src_items, dst_items):
        src_target = _target_dir(args, src)
        dst_target = _target_dir(args, dst)
        dst_target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_target / "refined.png", dst_target / "refined.png")
        payload = _load_json(src_target / "SUPERVISION.json")
        payload["episode"] = int(dst_episode)
        payload["pose_id"] = dst["pose_id"]
        payload["sample_id"] = int(dst["sample_id"])
        payload["supervision_id"] = dst["supervision_id"]
        payload["dloral_seed"] = int(dst["dloral_seed"])
        payload["flowedit_seed"] = int(dst["flowedit_seed"])
        payload["output_sha256"] = _sha256(dst_target / "refined.png")
        payload["copied_from"] = src["supervision_id"]
        payload["note"] = "probe pool-switch copy; not formal supervision"
        _write(dst_target / "SUPERVISION.json", payload)
        flow = _load_json(src_target / "flowedit_result.json")
        flow["seed"] = int(dst["flowedit_seed"])
        flow["supervision_id"] = dst["supervision_id"]
        _write(dst_target / "flowedit_result.json", flow)
        src_source = _source_dir(args, src)
        dst_source = _source_dir(args, dst)
        if src_source.is_dir():
            shutil.copytree(src_source, dst_source, dirs_exist_ok=True)
        dloral_src = _dloral_dir(args, src)
        dloral_dst = _dloral_dir(args, dst)
        if dloral_src.is_dir():
            shutil.copytree(dloral_src, dloral_dst, dirs_exist_ok=True)
    _write_manifest(args, dst_episode, _parent_for_episode(args, src_episode), dst_items)


def phase_probe(args) -> dict[str, Any]:
    probe_args = copy.copy(args)
    probe_args.output_dir = str(_root(args) / "probe")
    probe_args.n_poses = int(args.probe_n_poses)
    probe_args.steps = int(args.probe_steps)
    probe_args.probe = True
    probe_root = _root(probe_args)
    if (probe_root / "PROBE.json").is_file() and not args.force:
        return _load_json(probe_root / "PROBE.json")
    phase_preflight(probe_args)
    parent = _parent_for_episode(probe_args, 1)
    phase_supervise(probe_args, 1, parent)
    phase_flowedit(probe_args, 1, parent)
    first = _items(probe_args, 1)
    dual = [item for item in first if item["pose_id"] == first[0]["pose_id"]]
    dual_ok = (
        len(dual) == 2
        and dual[0]["uid_1x"] == dual[1]["uid_1x"]
        and dual[0]["dloral_seed"] != dual[1]["dloral_seed"]
        and dual[0]["flowedit_seed"] != dual[1]["flowedit_seed"]
    )
    e1_train = _train_root(probe_args, 1) / "TRAIN.json"
    if e1_train.is_file() and not args.force:
        resumed = _load_json(e1_train)
        progress = _train_root(probe_args, 1) / "PROGRESS.json"
        interrupted = _load_json(progress) if progress.is_file() else resumed
    else:
        interrupted = phase_train(
            probe_args, 1, parent, steps=int(args.probe_steps), interrupt_at=int(args.probe_interrupt_at)
        )
        resumed = phase_train(probe_args, 1, parent, steps=int(args.probe_steps))
    _copy_pool_for_switch(probe_args, 1, 2)
    switched = phase_train(
        probe_args, 2, Path(resumed["checkpoint"]), steps=int(args.probe_switch_steps)
    )
    visits2 = switched["visits"]
    payload = {
        "schema": "jax068_l0_start_108_probe_v1",
        "ok": bool(
            dual_ok
            and resumed.get("freeze", {}).get("ok")
            and resumed.get("n_l1", 0) > 0
            and resumed.get("l1_updated")
            and int(visits2.get("previous_episode_steps", -1)) == 0
            and int(visits2.get("original_photo_steps", -1)) == 0
            and (probe_root / "episode01" / "train" / "EMPTY_L1.json").is_file()
        ),
        "empty_l1": _load_json(probe_root / "episode01" / "train" / "EMPTY_L1.json")
        if (probe_root / "episode01" / "train" / "EMPTY_L1.json").is_file()
        else {},
        "dual_sample_ok": dual_ok,
        "n_unique_ids": len({item["supervision_id"] for item in first}),
        "interrupt": interrupted,
        "resume": resumed,
        "episode_switch": switched,
        "resource": _gpu(),
        "note": "Probe is resource/integration validation only and is not a formal starting checkpoint.",
        "formal_output": str(_root(args)),
    }
    _write(probe_root / "PROBE.json", payload)
    if not payload["ok"]:
        raise RuntimeError("probe failed")
    print(json.dumps({"probe": "ok", "path": str(probe_root / "PROBE.json")}, indent=2))
    return payload


def run_episode(args, index: int) -> dict[str, Any]:
    parent = _parent_for_episode(args, index)
    train_path = _train_root(args, index) / "TRAIN.json"
    if not train_path.is_file() or args.force:
        print(
            f"=== episode {index}: e={episode_spec(index)['elevation_deg']:g} "
            f"r={episode_spec(index)['radius']:g} parent={parent} ===",
            flush=True,
        )
        phase_supervise(args, index, parent)
        phase_flowedit(args, index, parent)
    trained = phase_train(args, index, parent)
    eval_path = _eval_root(args, index) / "EVAL.json"
    eval_payload = (
        _load_json(eval_path)
        if eval_path.is_file() and not args.force
        else phase_evaluate(args, index, parent)
    )
    chain_path = _root(args) / "CHAIN.json"
    chain = _load_json(chain_path) if chain_path.is_file() else {"schema": "jax068_l0_start_108_chain_v1", "episodes": []}
    rows = [row for row in chain.get("episodes") or [] if int(row.get("index", -1)) != int(index)]
    rows.append(
        {
            "index": int(index),
            "spec": episode_spec(index),
            "parent": _identity(parent),
            "checkpoint": _identity(trained["checkpoint"]),
            "n_targets": trained["n_targets"],
            "previous_episode_steps": 0,
            "original_photo_steps": 0,
            "eval": str(_eval_root(args, index) / "EVAL.json"),
            "status": "complete",
        }
    )
    rows.sort(key=lambda item: int(item["index"]))
    chain["episodes"] = rows
    chain["sampling"] = "current_episode_108_restored_images_only"
    _write(chain_path, chain)
    trained["eval"] = eval_payload
    return trained


def _require_probe(args) -> None:
    if args.skip_probe_gate:
        return
    path = _root(args) / "probe" / "PROBE.json"
    if not path.is_file() or not bool(_load_json(path).get("ok")):
        raise SystemExit(f"formal run requires a passing probe at {path}")


def phase_all(args) -> dict[str, Any]:
    _require_probe(args)
    phase_preflight(args)
    results = {}
    for index in range(1, N_EPISODES + 1):
        results[str(index)] = run_episode(args, index)
    payload = {"phase": "all", "n_episodes": N_EPISODES, "episodes": {k: {"checkpoint": v["checkpoint"]} for k, v in results.items()}}
    _write(_root(args) / "ALL.json", payload)
    return payload


def _test_view_eval(args, checkpoint: Path, tag: str) -> dict[str, Any]:
    output = _root(args) / "test_eval" / tag
    output.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_lod_jax068.py"),
        "--start_checkpoint",
        str(Path(args.start_checkpoint).resolve()),
        "--lod_checkpoint",
        str(checkpoint.resolve()),
        "--max_level",
        "1" if checkpoint.suffix == ".pt" else "0",
        "--output_dir",
        str(output),
        "--tag",
        tag,
        "--gz_root",
        str(args.gz_root),
        "--quiet",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    metrics = output / "metrics.json"
    return _load_json(metrics) if metrics.is_file() else {"output": str(output)}


def phase_report(args) -> dict[str, Any]:
    chain = _load_json(_root(args) / "CHAIN.json") if (_root(args) / "CHAIN.json").is_file() else {"episodes": []}
    rows = []
    for item in chain.get("episodes") or []:
        train = _load_json(_train_root(args, int(item["index"])) / "TRAIN.json")
        visits = _load_json(_train_root(args, int(item["index"])) / "VISITS.json")
        curve = _load_json(_train_root(args, int(item["index"])) / "absorption_curve.json")["curve"]
        rows.append(
            {
                "index": int(item["index"]),
                "spec": episode_spec(int(item["index"])),
                "n_l1": train.get("n_l1"),
                "current_episode_steps": visits.get("current_episode_steps"),
                "previous_episode_steps": visits.get("previous_episode_steps"),
                "original_photo_steps": visits.get("original_photo_steps"),
                "mean_visits": visits.get("stats", {}).get("median"),
                "expected_visits": EXPECTED_VISITS_PER_IMAGE,
                "start_rgb": curve[0]["mean_rgb_l1_to_target"] if curve else None,
                "final_rgb": curve[-1]["mean_rgb_l1_to_target"] if curve else None,
                "checkpoint": train["checkpoint"],
            }
        )
    test_eval = {}
    if rows:
        test_eval["l0"] = _test_view_eval(args, Path(args.start_checkpoint), "l0")
        test_eval["this_e5"] = _test_view_eval(args, Path(rows[-1]["checkpoint"]), "this_e5")
        previous = PREVIOUS_ARCHIVE / "episode05" / "train" / "l1_final.lod.pt"
        if previous.is_file():
            test_eval["previous_l1_continuation_e5"] = _test_view_eval(args, previous, "previous_e5")
    report = {
        "schema": "jax068_l0_start_108_report_v1",
        "protocol": str(_root(args) / "PROTOCOL.json"),
        "comparison": {
            "original_l0": str(Path(args.start_checkpoint).resolve()),
            "this_run": str(_root(args)),
            "previous_five_episode_from_old_l1_40k": str(PREVIOUS_ARCHIVE),
            "note": (
                "Start point, supervision count, and cumulative budget differ. "
                "Do not attribute all gaps to removing old L1 noise. "
                "If absorption is weak, inspect curves and visit counts before LoD capacity."
            ),
        },
        "episodes": rows,
        "test_views": list(TEST_VIEWS),
        "test_eval": test_eval,
        "test_views_not_used_to_pick_steps": True,
        "source_entry": TWO_SCALE_ENTRY,
    }
    _write(_root(args) / "REPORT.json", report)
    print(json.dumps({"report": str(_root(args) / "REPORT.json")}, indent=2))
    return report


def _parse_phases(text: str) -> list[str]:
    if text == "all":
        return ["all"]
    phases = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [item for item in phases if item not in PHASES]
    if unknown:
        raise SystemExit(f"unknown phases {unknown}; choose from {PHASES}")
    return phases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    ModelParams(parser)
    PipelineParams(parser)
    parser.add_argument("--start_checkpoint", type=str, default=str(STAGE1_PATH))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--gz_root", type=str, default=str(DEFAULT_GZ_ROOT))
    parser.add_argument("--phase", type=str, default="preflight")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--steps", type=int, default=STEPS_PER_EPISODE)
    parser.add_argument("--n_poses", type=int, default=0, help="0 means the locked 54 poses")
    parser.add_argument("--probe_n_poses", type=int, default=PROBE_N_POSES)
    parser.add_argument("--probe_steps", type=int, default=PROBE_STEPS)
    parser.add_argument("--probe_interrupt_at", type=int, default=PROBE_INTERRUPT_AT)
    parser.add_argument("--probe_switch_steps", type=int, default=PROBE_SWITCH_STEPS)
    parser.add_argument("--generation_seed", type=int, default=GENERATION_SEED)
    parser.add_argument("--training_seed", type=int, default=TRAINING_SEED)
    parser.add_argument("--step_scale", type=float, default=MODEL_STEP_SCALE)
    parser.add_argument("--vlm_model_path", type=str, default=os.environ.get("VLM_MODEL_PATH", ""))
    parser.add_argument("--vlm_python", type=str, default=sys.executable)
    parser.add_argument("--vlm_device", type=str, default="cuda:0")
    parser.add_argument("--vlm_max_new_tokens", type=int, default=768)
    parser.add_argument("--vlm_max_image_size", type=int, default=1024)
    parser.add_argument("--flowedit_model_path", type=str, default=os.environ.get("FLOWEDIT_MODEL_PATH", ""))
    parser.add_argument("--flowedit_device", type=str, default="cuda:0")
    parser.add_argument("--flowedit_defer_decode", action="store_true", help="sample all FlowEdit images before one shared component offload and VAE decode")
    parser.add_argument("--dloral_python", type=str, default=sys.executable)
    parser.add_argument("--dloral_device", type=str, default="cuda:0")
    parser.add_argument("--dloral_root", type=str, default="submodules/DLoRAL")
    parser.add_argument(
        "--dloral_weight_root",
        type=str,
        default=os.environ.get("DLORAL_WEIGHT_ROOT", ""),
    )
    parser.add_argument("--sd_path", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--spynet", type=str, default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_probe_gate", action="store_true")
    args = parser.parse_args()
    args.output_dir = os.path.abspath(args.output_dir)
    args.start_checkpoint = os.path.abspath(args.start_checkpoint)
    args.gz_root = os.path.abspath(args.gz_root)
    if args.flowedit_model_path:
        args.flowedit_model_path = os.path.abspath(os.path.expanduser(args.flowedit_model_path))
    args.probe = False
    if abs(float(args.step_scale) - float(MODEL_STEP_SCALE)) > 1e-9:
        raise SystemExit(f"locked step_scale is {MODEL_STEP_SCALE}")
    if int(args.generation_seed) != int(GENERATION_SEED):
        raise SystemExit(f"locked generation_seed is {GENERATION_SEED}")
    if int(args.training_seed) != int(TRAINING_SEED):
        raise SystemExit(f"locked training_seed is {TRAINING_SEED}")
    if int(args.steps) != int(STEPS_PER_EPISODE):
        raise SystemExit(f"locked formal steps are {STEPS_PER_EPISODE}; use --probe_steps for the probe")
    if int(args.n_poses) != 0:
        raise SystemExit("formal camera set is locked to all 54 poses; use --probe_n_poses for the probe")
    _assert_output_isolated(args.output_dir)
    assert_writable_output(args.output_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    phases = _parse_phases(args.phase)
    if any(phase not in {"preflight"} for phase in phases):
        safe_state(args.quiet)
    random.seed(int(args.training_seed))
    for phase in phases:
        if phase == "preflight":
            phase_preflight(args)
        elif phase == "probe":
            phase_probe(args)
        elif phase == "all":
            phase_all(args)
            phase_report(args)
        elif phase == "report":
            phase_report(args)
        elif phase == "episode":
            if int(args.episode) < 1:
                raise SystemExit("--episode N is required")
            _require_probe(args)
            phase_preflight(args)
            run_episode(args, int(args.episode))
        elif phase in {"supervise", "flowedit", "train", "evaluate"}:
            if int(args.episode) < 1:
                raise SystemExit("--episode N is required")
            if phase in {"supervise", "flowedit", "train"}:
                _require_probe(args)
            parent = _parent_for_episode(args, int(args.episode))
            if phase == "supervise":
                phase_supervise(args, int(args.episode), parent)
            elif phase == "flowedit":
                phase_flowedit(args, int(args.episode), parent)
            elif phase == "train":
                phase_train(args, int(args.episode), parent)
            else:
                phase_evaluate(args, int(args.episode), parent)
        else:
            raise SystemExit(phase)


if __name__ == "__main__":
    main()
