#!/usr/bin/env python3
"""Recoverable 2x->4x LoD entry. Reuses prepare/train scripts; stays off the MVP path.

Stages: supervise_l1, train_l1, supervise_l2, train_l2, eval, video, recover, archive.
Geometry is the main alignment. SpyNet is opt-in on the same frozen L1. Roundtrip
gating stays off unless --max_roundtrip_error_px is set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from lod.archive import build_acceptance, experiment_index_entry, merge_experiment_index, write_acceptance
from lod.lineage import (
    POLICY,
    file_identity,
    reuse_supervision_dir,
    roi_dict,
    supervision_lineage,
    write_json,
)
from lod.panel import (
    GpuSampler,
    PANEL_NOTES,
    PANEL_ROIS,
    PANEL_STATUS,
    StageLog,
    run_logged,
    source_side_record,
)
from lod.path import DEFAULT_GZ_ROOT
from utils.zoom_camera import NormalizedROI


def _zoom_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


ROOT = Path(__file__).resolve().parents[1]
ALL_TRAIN_STAGES = (
    "supervise_l1", "train_l1", "supervise_l2", "train_l2", "eval", "video", "recover", "archive",
)
_PANEL_ZOOM = {
    "status": "pending_visual",
    "passed": None,
    "unresolved": None,
    "check_manually": ["building edges", "vehicle silhouettes", "tree mass as unresolved only"],
}


def _panel_preset(item: dict) -> dict:
    return {
        "name": item["name"],
        "scene": "JAX_068",
        "view_index": 0,
        "roi": {
            "center_x": item["center_x"],
            "center_y": item["center_y"],
            "width": item["width"],
            "height": item["height"],
        },
        "archive": f"skyfall-gs_exp/lod_panel_3roi/{item['id']}",
        "status": PANEL_STATUS,
        "notes": list(PANEL_NOTES),
        "continuous_zoom": dict(_PANEL_ZOOM),
    }


PRESETS = {
    "ybuilding": {
        "name": "jax068_ybuilding",
        "scene": "JAX_068",
        "view_index": 0,
        "roi": {"center_x": 0.592, "center_y": 0.53, "width": 0.1, "height": 0.1},
        "existing_l1": "skyfall-gs_exp/lod_l1_absorb_2x/geometry",
        "existing_l2": "skyfall-gs_exp/lod_l2_geometry_4x",
        "l1_supervision": "skyfall-gs_exp/zoom_gen_dloral_align_2048/geometry",
        "archive": "skyfall-gs_exp/lod_passed_roi_jax068_ybuilding",
        "status": "passed_small_scope",
        "video_start": 1.25,
        "notes": [
            "Reused existing 2x geometry supervision and 2x/4x checkpoints; did not retrain the 500-step runs.",
            "Resume probe wrote skyfall-gs_exp/lod_chain_ybuilding/resume_probe and did not overwrite archived checkpoints. That L2 probe is not L1/L2 full-stage resume.",
            "Old checkpoints have no lineage.json; --allow_unlineaged_reuse is an explicit compatibility channel, not a complete source record.",
            "PNG stage-link proves input content match only; it does not reconstruct historical execution order.",
            "Step-50 RGB drop coincides with filling the 50k budget; 500 steps recover. Do not change densify.",
            "Labeled local-reprojection regions are not confirmed artifacts.",
            "Continuous zoom remains a visual check of building edges and vehicle silhouettes.",
        ],
        "continuous_zoom": {
            "status": "visual_checked",
            "passed": [
                "Y-building silhouette stays connected from 1.25x through 2.9x while the building is in frame",
                "Parking-lot vehicle rows keep their layout; no whole row appears or vanishes",
                "Road edges stay continuous; no building-split jump",
            ],
            "unresolved": [
                "4x roof outlines show a gold fringe; stills cannot rule out flicker",
                "At 4x the crop is the ROI center, so the Y-building is mostly out of frame",
                "Tree mass softens; not used as a pass/fail item",
            ],
            "frames": "skyfall-gs_exp/lod_chain_entry_probe/zoom_frames/ybuilding",
        },
    },
    "harder_0p28": {
        "name": "jax068_0p28_0p28",
        "scene": "JAX_068",
        "view_index": 0,
        "roi": {"center_x": 0.28, "center_y": 0.28, "width": 0.1, "height": 0.1},
        "existing_l1": "skyfall-gs_exp/lod_harder_roi_0p28_0p28/l1",
        "existing_l2": "skyfall-gs_exp/lod_harder_roi_0p28_0p28/l2_geometry",
        "spynet_l2": "skyfall-gs_exp/lod_harder_roi_0p28_0p28/l2_spynet",
        "archive": "skyfall-gs_exp/lod_harder_roi_0p28_0p28",
        "status": "regression_sample",
        "video_start": 1.786,
        "notes": [
            "Same 2x->4x settings as Y-building, different ROI. Not a Y-building-specific config.",
            "Old L1/L2 directories remain source-record incomplete; --allow_unlineaged_reuse is compatibility only.",
            "SpyNet is a same-frozen-L1 control only; absorption is not used to rank modes.",
            "Roundtrip gate stayed off. Labeled local-reprojection regions are not confirmed artifacts.",
            "Continuous zoom remains a visual check; do not write video_fully_passed.",
        ],
        "continuous_zoom": {
            "status": "visual_checked",
            "passed": [
                "Main roof and helipad H remain in place from 1.79x to 4x",
                "Parking vehicles stay as rows of car-shaped blobs; no sudden extra aisle",
                "No building contour that splits or merges between sampled frames",
            ],
            "unresolved": [
                "4x roof edges have a similar gold fringe to the Y-building clip",
                "Stills do not certify temporal flicker; max_mean_rgb_jump is recorded only",
                "Foliage blobs; not used as a pass/fail item",
            ],
            "frames": "skyfall-gs_exp/lod_chain_entry_probe/zoom_frames/harder",
        },
    },
}
PRESETS.update({f"panel_{item['id']}": _panel_preset(item) for item in PANEL_ROIS})
PRESETS.update({item["id"]: _panel_preset(item) for item in PANEL_ROIS})
PRESETS["jax214_building_parking"] = {
    "name": "jax214_building_parking",
    "scene": "JAX_214",
    "view_index": 0,
    "view_name": "JAX_214_018_RGB",
    "roi": {"center_x": 0.66, "center_y": 0.66, "width": 0.1, "height": 0.1},
    "start_checkpoint": "skyfall-gs_exp/stage1/JAX_214/chkpnt30000.pth",
    "archive": "skyfall-gs_exp/lod_jax214",
    "status": "second_scene_pipeline",
    "notes": [
        "Second-scene 2x->4x source chain on a new JAX_214 Stage1. Same fixed settings as the JAX_068 panel.",
        "One ROI: building/parking junction on train view 0. Not a mode ranking and not a generalization claim.",
        "Do not copy JAX_068 ROI coordinates. Roundtrip gate stays off. No --allow_unlineaged_reuse.",
        "Absolute MAE is not required to match JAX_068. Video remains a visual check.",
    ],
    "continuous_zoom": {
        "status": "pending_visual",
        "passed": None,
        "unresolved": None,
        "check_manually": ["building edges", "vehicle rows", "do not use water as a pass item"],
        "do_not_write": "video_fully_passed",
    },
}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _run(cmd: list[str], *, env: dict[str, str] | None = None, stage: str, log: StageLog) -> None:
    print("+", " ".join(cmd), flush=True)
    run_logged(cmd, cwd=str(ROOT), env=env or os.environ.copy(), stage=stage, log=log)


def _roi_args(args) -> list[str]:
    cmd = [
        "--view_index", str(args.view_index),
        "--roi_center_x", str(args.roi_center_x),
        "--roi_center_y", str(args.roi_center_y),
        "--roi_width", str(args.roi_width),
        "--roi_height", str(args.roi_height),
        "--step_scale", str(args.step_scale),
    ]
    if str(getattr(args, "view_name", "") or "").strip():
        cmd += ["--view_name", str(args.view_name).strip()]
    return cmd


def _dloral_args(args) -> list[str]:
    weight_root = Path(os.environ.get("DLORAL_WEIGHT_ROOT", "weights/dloral"))
    cmd = [
        "--run_dloral",
        "--dloral_alignment", args.alignment,
        "--dloral_python", args.dloral_python,
        "--dloral_device", args.dloral_device,
        "--dloral_root", args.dloral_root,
        "--sd_path", args.sd_path or str(weight_root / "stable-diffusion-2-1-base"),
        "--ckpt", args.ckpt or str(weight_root / "model_enhanced.pkl"),
        "--spynet", args.spynet or str(weight_root / "spynet_20210409-c6c1bd09.pth"),
        "--seed", str(args.seed),
    ]
    if args.max_roundtrip_error_px is not None:
        cmd += ["--max_roundtrip_error_px", str(args.max_roundtrip_error_px)]
    if args.prompt_json:
        cmd += ["--prompt_json", args.prompt_json]
    else:
        cmd += [
            "--vlm_model_path", args.vlm_model_path,
            "--vlm_python", args.vlm_python,
            "--vlm_device", args.vlm_device,
        ]
    return cmd


def _current_supervision(args, *, zoom: float, parent_lod: str | None, prompt_path: str | None) -> dict:
    weight_root = Path(os.environ.get("DLORAL_WEIGHT_ROOT", "weights/dloral"))
    return supervision_lineage(
        start_checkpoint=args.start_checkpoint,
        parent_lod=parent_lod,
        roi=roi_dict(args.roi_center_x, args.roi_center_y, args.roi_width, args.roi_height),
        view_index=int(args.view_index),
        image_name=str(getattr(args, "view_name", "") or "") or None,
        zoom_factor=float(zoom),
        step_scale=float(args.step_scale),
        alignment=str(args.alignment),
        seed=int(args.seed),
        max_roundtrip_error_px=args.max_roundtrip_error_px,
        dloral_ckpt=args.ckpt or str(weight_root / "model_enhanced.pkl"),
        prompt_path=prompt_path,
    )


def _reuse_refined(directory: Path, current: dict, *, force: bool, allow_unlineaged: bool) -> dict:
    return reuse_supervision_dir(directory, current, force=force, allow_unlineaged=allow_unlineaged)


def _write_run_json(args, l1_dir: Path, l2_dir: Path) -> None:
    write_json(
        Path(args.output_dir) / "run.json",
        {
            "policy": POLICY,
            "preset": args.preset,
            "start_checkpoint": file_identity(args.start_checkpoint),
            "roi": roi_dict(args.roi_center_x, args.roi_center_y, args.roi_width, args.roi_height),
            "view_index": int(args.view_index),
            "view_name": str(getattr(args, "view_name", "") or "") or None,
            "alignment": args.alignment,
            "seed": int(args.seed),
            "steps": int(args.steps),
            "mix_ratio": float(args.mix_ratio),
            "max_roundtrip_error_px": args.max_roundtrip_error_px,
            "l1_dir": str(l1_dir),
            "l2_dir": str(l2_dir),
            "gz_root": args.gz_root,
        },
    )


def _archive_index(entries: list[dict], path: Path) -> None:
    write_json(
        path,
        {
            "schema": "lod_scale_chain_v1",
            "policy": POLICY,
            "entry": "scripts/run_lod_scale_chain.py",
            "experiments": entries,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", type=str, default="", choices=["", *PRESETS])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--start_checkpoint", type=str, default="skyfall-gs_exp/stage1/JAX_068/chkpnt30000.pth")
    parser.add_argument("--stages", type=str, default="all")
    parser.add_argument("--alignment", type=str, default="geometry", choices=["geometry", "spynet", "target_only"])
    parser.add_argument("--l1_dir", type=str, default="")
    parser.add_argument("--l2_dir", type=str, default="")
    parser.add_argument("--existing_l1", type=str, default="")
    parser.add_argument("--existing_l2", type=str, default="")
    parser.add_argument("--archive_dir", type=str, default="")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--scene", type=str, default="JAX_068")
    parser.add_argument("--status", type=str, default="recorded")
    parser.add_argument("--view_index", type=int, default=0)
    parser.add_argument("--view_name", type=str, default="", help="Lock the base camera by image stem. Preferred over view_index.")
    parser.add_argument("--roi_center_x", type=float, default=None)
    parser.add_argument("--roi_center_y", type=float, default=None)
    parser.add_argument("--roi_width", type=float, default=0.1)
    parser.add_argument("--roi_height", type=float, default=0.1)
    parser.add_argument("--zoom_l1", type=float, default=2.0)
    parser.add_argument("--zoom_l2", type=float, default=4.0)
    parser.add_argument("--step_scale", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=str, default="0,50,100,250,500")
    parser.add_argument("--mix_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gz_root", type=str, default=str(DEFAULT_GZ_ROOT))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow_unlineaged_reuse", action="store_true")
    parser.add_argument("--max_roundtrip_error_px", type=float, default=None)
    parser.add_argument("--prompt_json", type=str, default="")
    parser.add_argument("--vlm_model_path", type=str, default=os.environ.get("VLM_MODEL_PATH", "weights/Qwen3-VL-4B-Instruct"))
    parser.add_argument("--vlm_python", type=str, default=str(Path.home() / "miniconda3/envs/fixanything/bin/python3.10"))
    parser.add_argument("--vlm_device", type=str, default="cuda:0")
    parser.add_argument("--dloral_python", type=str, default=str(Path.home() / "miniconda3/envs/dloral/bin/python"))
    parser.add_argument("--dloral_device", type=str, default="cuda:0")
    parser.add_argument("--dloral_root", type=str, default="submodules/DLoRAL")
    parser.add_argument("--sd_path", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--spynet", type=str, default="")
    parser.add_argument("--video_start", type=float, default=None)
    parser.add_argument("--probe_resume_steps", type=int, default=0)
    parser.add_argument("--probe_resume_level", type=str, default="l2", choices=["l1", "l2", "both"])
    parser.add_argument("--spynet_control", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    preset = PRESETS.get(args.preset or "", {})
    if preset:
        args.view_index = int(preset.get("view_index", args.view_index))
        if preset.get("view_name") and not str(getattr(args, "view_name", "") or "").strip():
            args.view_name = preset["view_name"]
        roi = preset["roi"]
        if args.roi_center_x is None:
            args.roi_center_x = roi["center_x"]
        if args.roi_center_y is None:
            args.roi_center_y = roi["center_y"]
        args.roi_width = roi.get("width", args.roi_width)
        args.roi_height = roi.get("height", args.roi_height)
        args.name = args.name or preset["name"]
        args.scene = preset.get("scene", args.scene)
        args.status = preset.get("status", args.status)
        args.existing_l1 = args.existing_l1 or preset.get("existing_l1", "")
        args.existing_l2 = args.existing_l2 or preset.get("existing_l2", "")
        args.archive_dir = args.archive_dir or preset.get("archive", "")
        default_ckpt = parser.get_default("start_checkpoint")
        if preset.get("start_checkpoint") and args.start_checkpoint == default_ckpt:
            args.start_checkpoint = preset["start_checkpoint"]
        if args.video_start is None:
            args.video_start = preset.get("video_start")
        if not args.prompt_json:
            for candidate in (
                ROOT / preset.get("existing_l1", "") / "prompt.json" if preset.get("existing_l1") else None,
                ROOT / preset.get("l1_supervision", "") / "prompt.json" if preset.get("l1_supervision") else None,
            ):
                if candidate is not None and candidate.is_file():
                    args.prompt_json = str(candidate)
                    break
    if args.roi_center_x is None or args.roi_center_y is None:
        raise SystemExit("ROI center is required (set --preset or --roi_center_x/y).")

    args.start_checkpoint = str((ROOT / args.start_checkpoint).resolve()) if not os.path.isabs(args.start_checkpoint) else args.start_checkpoint
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    args.output_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    l1_dir = Path(args.l1_dir or args.existing_l1 or (output_dir / "l1"))
    l2_dir = Path(args.l2_dir or args.existing_l2 or (output_dir / "l2"))
    if not l1_dir.is_absolute():
        l1_dir = (ROOT / l1_dir).resolve()
    if not l2_dir.is_absolute():
        l2_dir = (ROOT / l2_dir).resolve()
    archive_dir = Path(args.archive_dir or output_dir)
    if not archive_dir.is_absolute():
        archive_dir = (ROOT / archive_dir).resolve()
    if not args.prompt_json:
        for candidate in (l1_dir / "prompt.json", l2_dir / "prompt.json"):
            if candidate.is_file():
                args.prompt_json = str(candidate)
                break

    if args.stages == "all":
        stages = list(ALL_TRAIN_STAGES)
    elif args.stages == "verify":
        stages = ["recover", "eval", "archive"]
    else:
        stages = [item.strip() for item in args.stages.split(",") if item.strip()]
    py = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    roi = NormalizedROI(args.roi_center_x, args.roi_center_y, args.roi_width, args.roi_height)
    _write_run_json(args, l1_dir, l2_dir)
    write_json(output_dir / "roi.json", roi_dict(roi.center_x, roi.center_y, roi.width, roi.height) | {
        "view_index": args.view_index,
        "view_name": str(getattr(args, "view_name", "") or "") or None,
    })

    recover_payload = None
    source_records = []
    stage_log = StageLog(output_dir / "stage_log.json")
    stage_log.load()

    if "supervise_l1" in stages:
        l1_dir.mkdir(parents=True, exist_ok=True)
        current = _current_supervision(
            args, zoom=args.zoom_l1, parent_lod=None,
            prompt_path=str(l1_dir / "prompt.json") if (l1_dir / "prompt.json").is_file() else args.prompt_json,
        )
        decision = _reuse_refined(l1_dir, current, force=args.force, allow_unlineaged=args.allow_unlineaged_reuse)
        source_records.append({"stage": "supervise_l1", **decision})
        if args.force or not decision["reuse"]:
            cmd = [
                py, "scripts/prepare_lod_l2_supervision.py",
                "--start_checkpoint", args.start_checkpoint,
                "--output_dir", str(l1_dir),
                "--gz_root", args.gz_root,
                "--zoom_factor", str(args.zoom_l1),
                *_roi_args(args),
                *_dloral_args(args),
            ]
            _run(cmd, env=env, stage="supervise_l1", log=stage_log)
        else:
            stage_log.record({
                "name": "supervise_l1", "ok": True, "skipped": True,
                "reason": decision.get("source_record"), "elapsed_s": 0, "peak_mem_mib": None,
            })

    if "train_l1" in stages:
        ckpt = l1_dir / "l1_final.lod.pt"
        if args.force or not ckpt.is_file():
            _run(
                [
                    py, "scripts/train_lod_l1.py",
                    "--start_checkpoint", args.start_checkpoint,
                    "--output_dir", str(l1_dir),
                    "--refined_image", str(l1_dir / "refined.png"),
                    "--gz_root", args.gz_root,
                    "--zoom_factor", str(args.zoom_l1),
                    "--steps", str(args.steps),
                    "--eval_steps", args.eval_steps,
                    "--mix_ratio", str(args.mix_ratio),
                    "--seed", str(args.seed),
                    *_roi_args(args),
                ],
                env=env, stage="train_l1", log=stage_log,
            )
        else:
            stage_log.record({
                "name": "train_l1", "ok": True, "skipped": True,
                "reason": "existing_ckpt", "elapsed_s": 0, "peak_mem_mib": None,
            })

    if "supervise_l2" in stages:
        l2_dir.mkdir(parents=True, exist_ok=True)
        l1_ckpt = str(l1_dir / "l1_final.lod.pt")
        prompt = args.prompt_json or (str(l1_dir / "prompt.json") if (l1_dir / "prompt.json").is_file() else "")
        current = _current_supervision(
            args, zoom=args.zoom_l2, parent_lod=l1_ckpt,
            prompt_path=str(l2_dir / "prompt.json") if (l2_dir / "prompt.json").is_file() else prompt,
        )
        decision = _reuse_refined(l2_dir, current, force=args.force, allow_unlineaged=args.allow_unlineaged_reuse)
        source_records.append({"stage": "supervise_l2", **decision})
        if args.force or not decision["reuse"]:
            cmd = [
                py, "scripts/prepare_lod_l2_supervision.py",
                "--start_checkpoint", args.start_checkpoint,
                "--l1_checkpoint", l1_ckpt,
                "--output_dir", str(l2_dir),
                "--gz_root", args.gz_root,
                "--zoom_factor", str(args.zoom_l2),
                *_roi_args(args),
            ]
            saved_prompt = args.prompt_json
            if prompt:
                args.prompt_json = prompt
            try:
                cmd += _dloral_args(args)
            finally:
                args.prompt_json = saved_prompt
            _run(cmd, env=env, stage="supervise_l2", log=stage_log)
        else:
            stage_log.record({
                "name": "supervise_l2", "ok": True, "skipped": True,
                "reason": decision.get("source_record"), "elapsed_s": 0, "peak_mem_mib": None,
            })

    if "train_l2" in stages:
        ckpt = l2_dir / "l2_final.lod.pt"
        if args.force or not ckpt.is_file():
            _run(
                [
                    py, "scripts/train_lod_l2.py",
                    "--start_checkpoint", args.start_checkpoint,
                    "--l1_checkpoint", str(l1_dir / "l1_final.lod.pt"),
                    "--output_dir", str(l2_dir),
                    "--refined_image", str(l2_dir / "refined.png"),
                    "--gz_root", args.gz_root,
                    "--zoom_factor", str(args.zoom_l2),
                    "--parent_zoom_factor", str(args.zoom_l1),
                    "--steps", str(args.steps),
                    "--eval_steps", args.eval_steps,
                    "--mix_ratio", str(args.mix_ratio),
                    "--seed", str(args.seed),
                    "--alignment", args.alignment,
                    *_roi_args(args),
                ],
                env=env, stage="train_l2", log=stage_log,
            )
        else:
            stage_log.record({
                "name": "train_l2", "ok": True, "skipped": True,
                "reason": "existing_ckpt", "elapsed_s": 0, "peak_mem_mib": None,
            })

    if source_records:
        write_json(output_dir / "source_record.json", {"records": source_records})

    if "eval" in stages and (l2_dir / "absorption_curve.json").is_file():
        _run([py, "scripts/summarize_lod_l2.py", "--root", str(l2_dir)], env=env, stage="eval", log=stage_log)

    if "video" in stages:
        start_factor = args.video_start if args.video_start is not None else roi.min_zoom_factor()
        video_dir = l2_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        _run(
            [
                py, "scripts/render_lod_zoom_video.py",
                "--start_checkpoint", args.start_checkpoint,
                "--lod_checkpoint", str(l2_dir / "l2_final.lod.pt"),
                "--output", str(video_dir / f"zoom_{_zoom_tag(start_factor)}_to_{_zoom_tag(args.zoom_l2)}.mp4"),
                "--gz_root", args.gz_root,
                "--start_factor", str(start_factor),
                "--end_factor", str(args.zoom_l2),
                *_roi_args(args),
            ],
            env=env, stage="video", log=stage_log,
        )

    if "recover" in stages:
        from lod.recover import recover_scale_chain
        import time

        l1_ckpt = l1_dir / "l1_final.lod.pt"
        l2_ckpt = l2_dir / "l2_final.lod.pt"
        sampler = GpuSampler()
        sampler.start()
        started = time.time()
        error = None
        try:
            recover_payload = recover_scale_chain(
                start_checkpoint=args.start_checkpoint,
                l1_checkpoint=str(l1_ckpt),
                l2_checkpoint=str(l2_ckpt) if l2_ckpt.is_file() else None,
                l2_render_input=str(l2_dir / "render_input.png") if (l2_dir / "render_input.png").is_file() else None,
                output_dir=str(output_dir / "recover"),
                gz_root=args.gz_root,
                view_index=int(args.view_index),
                image_name=str(getattr(args, "view_name", "") or "") or None,
                roi=roi,
                zoom_l1=float(args.zoom_l1),
                zoom_l2=float(args.zoom_l2),
                step_scale=float(args.step_scale),
                quiet=bool(args.quiet),
            )
            if args.probe_resume_steps:
                start_step = int(args.steps)
                levels = ["l1", "l2"] if args.probe_resume_level == "both" else [args.probe_resume_level]
                if "l1" in levels and l1_ckpt.is_file() and (l1_dir / "refined.png").is_file():
                    probe_l1 = output_dir / "resume_probe_l1"
                    probe_l1.mkdir(parents=True, exist_ok=True)
                    _run(
                        [
                            py, "scripts/train_lod_l1.py",
                            "--start_checkpoint", args.start_checkpoint,
                            "--resume", str(l1_ckpt),
                            "--start_step", str(start_step),
                            "--output_dir", str(probe_l1),
                            "--refined_image", str(l1_dir / "refined.png"),
                            "--gz_root", args.gz_root,
                            "--zoom_factor", str(args.zoom_l1),
                            "--steps", str(start_step + int(args.probe_resume_steps)),
                            "--eval_steps", str(start_step + int(args.probe_resume_steps)),
                            "--mix_ratio", str(args.mix_ratio),
                            "--seed", str(args.seed),
                            "--skip_cross_view",
                            "--skip_view_eval",
                            *_roi_args(args),
                        ],
                        env=env, stage="resume_probe_l1", log=stage_log,
                    )
                    recover_payload["resume_probe_l1"] = str(probe_l1 / "l1_final.lod.pt")
                if "l2" in levels and l2_ckpt.is_file() and (l2_dir / "refined.png").is_file():
                    probe_dir = output_dir / "resume_probe"
                    probe_dir.mkdir(parents=True, exist_ok=True)
                    _run(
                        [
                            py, "scripts/train_lod_l2.py",
                            "--start_checkpoint", args.start_checkpoint,
                            "--l1_checkpoint", str(l1_ckpt),
                            "--resume", str(l2_ckpt),
                            "--start_step", str(start_step),
                            "--output_dir", str(probe_dir),
                            "--refined_image", str(l2_dir / "refined.png"),
                            "--gz_root", args.gz_root,
                            "--zoom_factor", str(args.zoom_l2),
                            "--parent_zoom_factor", str(args.zoom_l1),
                            "--steps", str(start_step + int(args.probe_resume_steps)),
                            "--eval_steps", str(start_step + int(args.probe_resume_steps)),
                            "--mix_ratio", str(args.mix_ratio),
                            "--seed", str(args.seed),
                            "--skip_cross_view",
                            "--skip_view_eval",
                            "--alignment", args.alignment,
                            *_roi_args(args),
                        ],
                        env=env, stage="resume_probe_l2", log=stage_log,
                    )
                    recover_payload["resume_probe"] = str(probe_dir / "l2_final.lod.pt")
                write_json(output_dir / "recover" / "recover.json", recover_payload)
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            stage_log.record({
                "name": "recover",
                "ok": error is None,
                "elapsed_s": round(time.time() - started, 3),
                "peak_mem_mib": sampler.stop(),
                "error": error,
            })

    if "archive" in stages:
        recover_path = output_dir / "recover" / "recover.json"
        if recover_payload is None and recover_path.is_file():
            recover_payload = _load(recover_path)
        extra = {
            "archive_dir": str(archive_dir),
            "recover": recover_payload,
            "recover_path": str(recover_path) if recover_path.is_file() else None,
            "notes": list(preset.get("notes") or []),
            "continuous_zoom": preset.get("continuous_zoom") or {},
            "source_record": {
                "l1": source_side_record(l1_dir, require_prompt=True),
                "l2": source_side_record(l2_dir, require_prompt=True),
                "allow_unlineaged_reuse": bool(args.allow_unlineaged_reuse),
                "this_run": source_records,
            },
            "resume_boundary": {
                "segment_restart": True,
                "l1_resume": True,
                "l2_resume": True,
                "not": "all_training_stages_historically_checkpointed",
            },
            "sources": {
                "chain_dir": str(output_dir),
                "l1_dir": str(l1_dir),
                "l2_dir": str(l2_dir),
                "l1_supervision": None if not preset.get("l1_supervision") else str(ROOT / preset["l1_supervision"]),
                "spynet_l2": None if not preset.get("spynet_l2") else str(ROOT / preset["spynet_l2"]),
                "alignment": args.alignment,
                "seed": int(args.seed),
                "prompt_json": args.prompt_json or None,
                "dloral_request": next(
                    (
                        str(path)
                        for path in (
                            l2_dir / "dloral" / "request.json",
                            l1_dir / "dloral" / "request.json",
                            None if not preset.get("l1_supervision") else ROOT / preset["l1_supervision"] / "request.json",
                        )
                        if path is not None and path.is_file()
                    ),
                    None,
                ),
            },
        }
        acceptance = build_acceptance(
            name=args.name or output_dir.name,
            scene=args.scene,
            view_index=int(args.view_index),
            roi=roi_dict(roi.center_x, roi.center_y, roi.width, roi.height),
            l1_dir=l1_dir,
            l2_dir=l2_dir,
            start_checkpoint=args.start_checkpoint,
            status=args.status,
            extra=extra,
        )
        write_acceptance(archive_dir / "ACCEPTANCE.json", acceptance)
        index_path = ROOT / "skyfall-gs_exp" / "lod_index.json"
        entries = []
        if index_path.is_file():
            entries = list((_load(index_path).get("experiments") or []))
        entry = experiment_index_entry(acceptance)
        entry["archive"] = str(archive_dir)
        entry["chain_dir"] = str(output_dir)
        entries = merge_experiment_index(entries, entry)
        _archive_index(entries, index_path)
        print(json.dumps({"acceptance": str(archive_dir / "ACCEPTANCE.json"), "index": str(index_path)}, indent=2))

    print(json.dumps({"output_dir": str(output_dir), "stages": stages, "l1_dir": str(l1_dir), "l2_dir": str(l2_dir)}, indent=2))


if __name__ == "__main__":
    main()
