#!/usr/bin/env python3
"""Independent short-run checks for the LoD scale-chain entry.

Verifies supervision cache hit/reject, unlineaged compatibility, and L1 interrupt
vs contiguous training. Does not overwrite the two archived ROI experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from lod.lineage import reuse_supervision_dir, supervision_lineage, write_json
from lod.train_state import layer_param_delta


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "skyfall-gs_exp" / "lod_chain_entry_probe"
REFINED = ROOT / "skyfall-gs_exp" / "zoom_gen_dloral_align_2048" / "geometry" / "refined.png"
STAGE1 = ROOT / "skyfall-gs_exp" / "stage1" / "JAX_068" / "chkpnt30000.pth"
PROMPT = "Zoom in on the central Y-shaped building and its immediate surroundings."
ROI = {"center_x": 0.592, "center_y": 0.53, "width": 0.1, "height": 0.1}


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)


def _current(*, seed: int, directory: Path) -> dict:
    return supervision_lineage(
        start_checkpoint=str(STAGE1),
        parent_lod=None,
        roi=ROI,
        view_index=0,
        zoom_factor=2.0,
        step_scale=2.0,
        alignment="geometry",
        seed=int(seed),
        max_roundtrip_error_px=None,
        dloral_ckpt=os.environ.get("DLORAL_WEIGHT_ROOT", "weights/dloral")
        + "/model_enhanced.pkl",
        prompt_text=PROMPT,
        prompt_path=str(directory / "prompt.json") if (directory / "prompt.json").is_file() else None,
        refined_image=str(directory / "refined.png") if (directory / "refined.png").is_file() else None,
    )


def _lineage_checks(out: Path) -> dict:
    hit_dir = out / "lineage_hit"
    miss_dir = out / "lineage_seed_change"
    unlineaged = out / "lineage_unlineaged"
    for directory in (hit_dir, miss_dir, unlineaged):
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REFINED, directory / "refined.png")
        write_json(directory / "prompt.json", {"target_prompt": PROMPT, "prompt": PROMPT})
    write_json(hit_dir / "lineage.json", _current(seed=0, directory=hit_dir))
    write_json(miss_dir / "lineage.json", _current(seed=0, directory=miss_dir))
    hit = reuse_supervision_dir(hit_dir, _current(seed=0, directory=hit_dir))
    rejected = None
    try:
        reuse_supervision_dir(miss_dir, _current(seed=1, directory=miss_dir))
    except ValueError as exc:
        rejected = str(exc)
    blocked = None
    try:
        reuse_supervision_dir(unlineaged, _current(seed=0, directory=unlineaged))
    except ValueError as exc:
        blocked = str(exc)
    allowed = reuse_supervision_dir(
        unlineaged, _current(seed=0, directory=unlineaged), allow_unlineaged=True,
    )
    payload = {
        "cache_hit": hit,
        "seed_change_rejected": rejected is not None,
        "seed_change_error": rejected,
        "unlineaged_blocked_without_flag": blocked is not None,
        "unlineaged_allowed_incomplete": allowed,
        "ok": bool(hit["reuse"] and hit["complete"] and rejected and blocked and allowed["reuse"] and not allowed["complete"]),
    }
    write_json(out / "lineage_probe.json", payload)
    return payload


def _load_payload(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _compare_l1(left: Path, right: Path) -> dict:
    a = _load_payload(left)
    b = _load_payload(right)
    class _Wrap:
        def __init__(self, row):
            for key, value in row.items():
                setattr(self, key, value)

    delta = layer_param_delta(_Wrap(a["layers"][1]), _Wrap(b["layers"][1]))
    opt_left = a.get("optimizer")
    opt_right = b.get("optimizer")
    opt_ok = (opt_left is None) == (opt_right is None)
    xyz = delta["xyz"]["max_abs"]
    payload = {
        "delta": delta,
        "optimizer_present": {"left": opt_left is not None, "right": opt_right is not None},
        "optimizer_both": opt_ok and opt_left is not None,
        "n_match": delta["n_left"] == delta["n_right"],
        "xyz_max_abs": xyz,
        "ok": bool(delta["n_left"] == delta["n_right"] and xyz is not None and xyz < 1e-3),
    }
    return payload


def _train_l1(out: Path, *, steps: int, start_step: int, resume: Path | None, eval_steps: str) -> None:
    py = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [
        py, "scripts/train_lod_l1.py",
        "--start_checkpoint", str(STAGE1),
        "--output_dir", str(out),
        "--refined_image", str(REFINED),
        "--view_index", "0",
        "--roi_center_x", str(ROI["center_x"]),
        "--roi_center_y", str(ROI["center_y"]),
        "--roi_width", str(ROI["width"]),
        "--roi_height", str(ROI["height"]),
        "--zoom_factor", "2",
        "--step_scale", "2",
        "--steps", str(steps),
        "--start_step", str(start_step),
        "--eval_steps", eval_steps,
        "--eval_only_listed",
        "--mix_ratio", "0.2",
        "--seed", "0",
        "--skip_cross_view",
        "--skip_view_eval",
        "--quiet",
    ]
    if resume is not None:
        cmd += ["--resume", str(resume)]
    _run(cmd, env=env)


def _l1_resume_probe(out: Path, *, interrupt: int, total: int) -> dict:
    contiguous = out / "l1_contiguous"
    first = out / "l1_interrupt"
    resume_dir = out / "l1_resume"
    contiguous.mkdir(parents=True, exist_ok=True)
    first.mkdir(parents=True, exist_ok=True)
    resume_dir.mkdir(parents=True, exist_ok=True)
    _train_l1(contiguous, steps=total, start_step=0, resume=None, eval_steps=f"0,{total}")
    _train_l1(first, steps=interrupt, start_step=0, resume=None, eval_steps="0")
    _train_l1(
        resume_dir, steps=total, start_step=interrupt,
        resume=first / "l1_final.lod.pt", eval_steps=str(total),
    )
    compare = _compare_l1(contiguous / "l1_final.lod.pt", resume_dir / "l1_final.lod.pt")
    densify_c = json.loads((contiguous / "densify_log.json").read_text(encoding="utf-8")) if (contiguous / "densify_log.json").is_file() else []
    densify_r = json.loads((resume_dir / "densify_log.json").read_text(encoding="utf-8")) if (resume_dir / "densify_log.json").is_file() else []
    mix_c = [(row.get("step"), row.get("mix_train_view"), row.get("n")) for row in densify_c]
    mix_r = [(row.get("step"), row.get("mix_train_view"), row.get("n")) for row in densify_r]
    payload = {
        "interrupt": interrupt,
        "total": total,
        "contiguous": str(contiguous / "l1_final.lod.pt"),
        "resume": str(resume_dir / "l1_final.lod.pt"),
        "compare": compare,
        "densify_steps_contiguous": mix_c,
        "densify_steps_resume": mix_r,
        "densify_n_match": mix_c == mix_r or (
            {row[0]: row[2] for row in mix_c} == {row[0]: row[2] for row in mix_r}
        ),
        "train_state_saved": (resume_dir / "train_state.pt").is_file(),
        "ok": bool(compare["ok"] and (resume_dir / "train_state.pt").is_file()),
    }
    write_json(out / "l1_resume_probe.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--interrupt", type=int, default=12)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not REFINED.is_file():
        raise SystemExit(f"missing refined image: {REFINED}")
    lineage = _lineage_checks(out)
    train = None
    if args.skip_train:
        previous = out / "l1_resume_probe.json"
        if previous.is_file():
            train = json.loads(previous.read_text(encoding="utf-8"))
    else:
        train = _l1_resume_probe(out, interrupt=int(args.interrupt), total=int(args.steps))
    report = {
        "output_dir": str(out),
        "lineage": lineage,
        "l1_resume": train,
        "ok": bool(lineage.get("ok") and (train is None or train.get("ok"))),
        "boundaries": {
            "segment_restart_is_not_historical_full_resume": True,
            "allow_unlineaged_reuse_is_not_complete_source_record": True,
        },
    }
    write_json(out / "report.json", report)
    print(json.dumps({
        "ok": report["ok"],
        "lineage_ok": lineage.get("ok"),
        "l1_resume_ok": None if train is None else train.get("ok"),
        "xyz_max_abs": None if train is None else train["compare"].get("xyz_max_abs"),
        "report": str(out / "report.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
