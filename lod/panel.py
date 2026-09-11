"""Fixed-setting 3-ROI small-sample panel. Not a general success-rate claim."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from lod.lineage import POLICY, write_json
from utils.zoom_camera import NormalizedROI


PANEL_DIR = Path("skyfall-gs_exp/lod_panel_3roi")
VIEW_IMAGE = "JAX_068_011_RGB"
VIEW_INDEX = 0
PANEL_STATUS = "small_sample_stability"

FIXED_SETTINGS = {
    "alignment": "geometry",
    "zoom_l1": 2.0,
    "zoom_l2": 4.0,
    "step_scale": 2.0,
    "seed": 0,
    "steps": 500,
    "mix_ratio": 0.2,
    "max_points_per_level": 50000,
    "densify_from": 1,
    "densify_until": 250,
    "densify_interval": 10,
    "roundtrip_gate": False,
    "spynet_control": False,
    "allow_unlineaged_reuse": False,
}

EXCLUDED_HISTORICAL = [
    {
        "name": "jax068_ybuilding",
        "center_x": 0.592,
        "center_y": 0.53,
        "width": 0.1,
        "height": 0.1,
        "on_view0": "parking lot east of the Y-building, not the Y silhouette",
        "keep": {"status": "passed_small_scope", "source_complete": False},
    },
    {
        "name": "jax068_0p28_0p28",
        "center_x": 0.28,
        "center_y": 0.28,
        "width": 0.1,
        "height": 0.1,
        "on_view0": "NW warehouse roofs plus a small parking strip",
        "keep": {"status": "regression_sample", "source_complete": False},
    },
]

PANEL_ROIS: list[dict[str, Any]] = [
    {
        "id": "building",
        "name": "jax068_panel_building",
        "run_order": 1,
        "dominant": "building_boundary",
        "center_x": 0.38,
        "center_y": 0.50,
        "width": 0.1,
        "height": 0.1,
        "mixed": ["adjacent dark roofs", "courtyard"],
        "rationale": (
            "Y-building roof silhouette and wing edges on train view 0. "
            "Chosen as building-boundary primary before any training metric. "
            "Not the historical (0.592, 0.53) box, which on this view is a parking lot."
        ),
    },
    {
        "id": "trees",
        "name": "jax068_panel_trees",
        "run_order": 2,
        "dominant": "trees",
        "center_x": 0.25,
        "center_y": 0.48,
        "width": 0.1,
        "height": 0.1,
        "mixed": ["roads", "lawn", "courtyard"],
        "rationale": (
            "Tree canopy west of the Y-building. First lock used (0.20, 0.48); "
            "that box fails the 2x crop bound (center_x must be in [0.25, 0.75]). "
            "Shifted to x=0.25 before any trees training, same grove and same "
            "selection rule, not a metric-based resample. Pond-shore boxes fail 2x "
            "for the same reason. Not the historical (0.28, 0.28) warehouse roof."
        ),
        "amendment": {
            "from": {"center_x": 0.20, "center_y": 0.48},
            "reason": "zoom_2_crop_exceeds_image_horizontally",
            "before_trees_training": True,
        },
    },
    {
        "id": "parking",
        "name": "jax068_panel_parking",
        "run_order": 3,
        "dominant": "parking",
        "center_x": 0.72,
        "center_y": 0.68,
        "width": 0.1,
        "height": 0.1,
        "mixed": ["building edge on the right", "adjacent road"],
        "rationale": (
            "SE surface lot with readable vehicle rows. Parking is the selection "
            "basis; the neighboring roof is mixed content, not the reason for picking. "
            "Away from both historical centers."
        ),
    },
]

PANEL_CLAIM = "jax068_in_scene_small_sample_stability_and_supervision_absorption"
PANEL_DO_NOT_CLAIM = [
    "generation_authenticity",
    "cross_scene_generalization",
    "geometry_advantage",
    "video_fully_passed",
    "general_success_rate",
]
PANEL_NOTES = [
    "Closed as JAX_068 in-scene small-sample stability and supervision absorption.",
    "Supports: three content types absorb supervision; tree high-frequency gain is limited; old-scale change is small.",
    "Does not support: generation authenticity, cross-scene generalization, or geometry advantage. Video flicker stays unverified.",
    "Fixed-setting 2x->4x geometry panel, n=3. Do not claim a general success rate.",
    "Execution success and visual quality are recorded separately.",
    "New source chain: no --allow_unlineaged_reuse; roundtrip gate stays off.",
    "No SpyNet full control in this panel. Local reprojection stays low priority and is not handled.",
    "Historical jax068_ybuilding / jax068_0p28_0p28 stay passed_small_scope / regression_sample with source_complete false.",
    "Short-run resume is numerically close (xyz max abs 3.8e-6), not elementwise identical. 12->16 restored a post-densify state and did not cross the next densify event.",
    "Next scientific step, if any: one full 2x->4x source chain on another scene. Do not add similar JAX_068 ROIs. Do not change the algorithm.",
]


def panel_roi(item: Mapping[str, Any]) -> NormalizedROI:
    return NormalizedROI(
        float(item["center_x"]),
        float(item["center_y"]),
        float(item.get("width", 0.1)),
        float(item.get("height", 0.1)),
    )


def selection_payload(*, locked_at: str, view_image: str = VIEW_IMAGE) -> dict[str, Any]:
    rois = []
    for item in PANEL_ROIS:
        roi = panel_roi(item)
        rois.append({
            **item,
            "min_zoom": roi.min_zoom_factor(),
            "zoom_2x_valid": roi.zoom_is_valid(2.0),
            "zoom_4x_valid": roi.zoom_is_valid(4.0),
        })
        if not roi.zoom_is_valid(2.0) or not roi.zoom_is_valid(4.0):
            raise ValueError(f"panel ROI {item['id']} is outside the 2x/4x crop bounds")
    return {
        "schema": "lod_panel_3roi_v1",
        "locked_at": locked_at,
        "locked_before_training": True,
        "claim": "small_sample_stability_n=3",
        "do_not_claim": ["general_success_rate", "geometry_better_than_spynet"],
        "view": {
            "scene": "JAX_068",
            "view_index": VIEW_INDEX,
            "image_name": view_image,
            "source": "first transforms_train.json frame; cfg eval=True; Scene shuffle=False",
        },
        "fixed_settings": FIXED_SETTINGS,
        "policy": POLICY,
        "excluded_historical": EXCLUDED_HISTORICAL,
        "rois": rois,
        "run_order": [item["id"] for item in sorted(PANEL_ROIS, key=lambda row: int(row["run_order"]))],
        "notes": PANEL_NOTES,
        "aiming_crops": str(PANEL_DIR / "_candidates"),
        "aiming_note": "Candidate crops were inspected to aim the three boxes. They are not a post-hoc sample from training metrics.",
    }


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def source_side_record(directory: str | Path, *, require_prompt: bool) -> dict[str, Any]:
    root = Path(directory)
    lineage = _load(root / "lineage.json")
    train_lineage = _load(root / "train_lineage.json")
    prompt = (root / "prompt.json").is_file()
    train_state = (root / "train_state.pt").is_file()
    supervision = lineage.get("kind") == "lod_supervision"
    train_ok = bool(train_lineage) or lineage.get("kind") == "lod_train"
    complete = bool(supervision and train_state and train_ok and (prompt or not require_prompt))
    return {
        "path": str(root),
        "lineage": bool(lineage),
        "lineage_kind": lineage.get("kind"),
        "supervision_lineage": supervision,
        "train_lineage": bool(train_lineage) or lineage.get("kind") == "lod_train",
        "train_state": train_state,
        "prompt": prompt,
        "freeze": (root / "freeze.json").is_file(),
        "refined": (root / "refined.png").is_file(),
        "complete": complete,
        "channel": "lineage" if supervision else ("unlineaged_compat" if not lineage else lineage.get("kind")),
    }


def gpu_memory_used_mib() -> int | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    if visible.isdigit():
        cmd += ["-i", visible]
    try:
        raw = subprocess.check_output(cmd, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    values = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(float(line)))
        except ValueError:
            continue
    if not values:
        return None
    return max(values)


class GpuSampler:
    def __init__(self, interval_s: float = 2.0):
        self.interval_s = interval_s
        self.peak_mib: int | None = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            used = gpu_memory_used_mib()
            if used is not None:
                self.samples += 1
                self.peak_mib = used if self.peak_mib is None else max(self.peak_mib, used)
            self._stop.wait(self.interval_s)

    def stop(self) -> int | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        return self.peak_mib


class StageLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.payload: dict[str, Any] = {
            "stages": [],
            "auto_retries": 0,
            "recoveries": 0,
            "failed_stage": None,
        }

    def load(self) -> None:
        if self.path.is_file():
            self.payload = _load(self.path) or self.payload

    def write(self) -> None:
        write_json(self.path, self.payload)

    def record(self, row: Mapping[str, Any]) -> None:
        self.payload.setdefault("stages", []).append(dict(row))
        if not row.get("ok"):
            self.payload["failed_stage"] = row.get("name")
        self.write()


def run_logged(cmd: Sequence[str], *, cwd: str, env: dict[str, str], stage: str, log: StageLog) -> None:
    sampler = GpuSampler()
    sampler.start()
    started = time.time()
    error = None
    try:
        subprocess.run(list(cmd), check=True, cwd=cwd, env=env)
        ok = True
    except subprocess.CalledProcessError as exc:
        ok = False
        error = f"exit {exc.returncode}"
        raise
    finally:
        peak = sampler.stop()
        log.record({
            "name": stage,
            "ok": error is None,
            "elapsed_s": round(time.time() - started, 3),
            "peak_mem_mib": peak,
            "gpu_samples": sampler.samples,
            "error": error,
            "cmd": list(cmd),
        })


def _curve_endpoints(curve: list[Mapping[str, Any]], *, rgb_key="l1_to_refined", hf_key="hf_l1_to_refined") -> dict[str, Any]:
    if not curve:
        return {}
    first = curve[0]
    last = curve[-1]
    by_step = {int(item.get("step", -1)): item for item in curve}

    def target(item, key):
        return ((item or {}).get("target") or {}).get(key)

    s0 = by_step.get(0, first)
    s500 = by_step.get(500, last)
    return {
        "rgb_mae": {"0": target(s0, rgb_key), "500": target(s500, rgb_key)},
        "laplacian_mae": {"0": target(s0, hf_key), "500": target(s500, hf_key)},
        "n_final": s500.get("n_l1") or s500.get("n_l2") or (s500.get("active") or {}).get("n"),
        "train_gt_1x": ((s500.get("train_views_mean") or {}).get("l1_to_gt_mean")),
        "parent_2x_vs_frozen_l1": ((s500.get("parent_2x") or {}).get("l1_vs_frozen_l1")),
    }


def summarize_roi(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    l1 = root / "l1"
    l2 = root / "l2"
    acceptance = _load(root / "ACCEPTANCE.json")
    stage_log = _load(root / "stage_log.json")
    source_record = _load(root / "source_record.json")
    l1_curve = _load(l1 / "absorption_curve.json")
    l2_curve = _load(l2 / "absorption_curve.json")
    l1_summary = _load(l1 / "lod_l1_summary.json") if (l1 / "lod_l1_summary.json").is_file() else l1_curve
    l2_summary = _load(l2 / "lod_l2_summary.json")
    correspondence = _load(l2 / "correspondence.json")
    run = _load(root / "run.json")
    stages = list(stage_log.get("stages") or [])
    failed = [row for row in stages if not row.get("ok")]
    gen = [row for row in stages if str(row.get("name", "")).startswith("supervise")]
    train = [row for row in stages if str(row.get("name", "")).startswith("train")]
    execution_ok = bool(stages) and not failed and (root / "ACCEPTANCE.json").is_file()
    visual = ((acceptance.get("checks") or {}).get("continuous_zoom") or {})
    return {
        "root": str(root),
        "name": run.get("preset") or root.name,
        "execution": {
            "success": execution_ok,
            "stages_completed": [row.get("name") for row in stages if row.get("ok")],
            "failed": failed,
            "auto_retries": stage_log.get("auto_retries", 0),
            "recoveries": stage_log.get("recoveries", 0),
            "not": "visual_quality",
        },
        "source_completeness": {
            "l1": source_side_record(l1, require_prompt=True),
            "l2": source_side_record(l2, require_prompt=True),
            "this_run": source_record.get("records") or source_record,
            "start_checkpoint": (run.get("start_checkpoint") or (acceptance.get("start_checkpoint"))),
            "allow_unlineaged_reuse": False,
        },
        "resources": {
            "generate_s": round(sum(float(row.get("elapsed_s") or 0) for row in gen), 3),
            "train_s": round(sum(float(row.get("elapsed_s") or 0) for row in train), 3),
            "peak_mem_mib": max((row.get("peak_mem_mib") or 0) for row in stages) if stages else None,
            "n_l0": l1_summary.get("n_l0") or l2_summary.get("n_l0"),
            "n_l1": l1_summary.get("n_l1") or l2_summary.get("n_l1"),
            "n_l2": l2_summary.get("n_l2"),
            "stage_log": str(root / "stage_log.json") if (root / "stage_log.json").is_file() else None,
        },
        "supervision_absorption": {
            "l1": _curve_endpoints(list(l1_curve.get("curve") or [])),
            "l2": _curve_endpoints(list(l2_curve.get("curve") or [])),
            "not_for": "mode_ranking",
        },
        "old_scale": {
            "train_gt_1x": ((acceptance.get("checks") or {}).get("old_scale") or {}).get("train_gt_1x"),
            "parent_2x_vs_frozen_l1": ((acceptance.get("checks") or {}).get("old_scale") or {}).get("parent_2x_rgb_vs_frozen_l1"),
            "require_zero_overlap": False,
        },
        "correspondence_and_transition": {
            "coverage": ((l2_summary.get("correspondence") or {}).get("coverage")
                         or ((correspondence.get("neighbors") or {}) and None)),
            "best_coverage": l2_summary.get("best_coverage"),
            "correspondence": l2_summary.get("correspondence"),
            "continuous_zoom": {
                "status": visual.get("status") or "pending_visual",
                "passed": visual.get("passed"),
                "unresolved": visual.get("unresolved"),
                "video": visual.get("video"),
                "max_mean_rgb_jump": visual.get("max_mean_rgb_jump"),
                "do_not_write": "video_fully_passed",
            },
        },
        "visual_quality": {
            "status": visual.get("status") or "pending_visual",
            "passed": visual.get("passed"),
            "unresolved": visual.get("unresolved"),
            "not_used_for_execution_success": True,
            "do_not_write": "video_fully_passed",
        },
    }


def summarize_panel(panel_dir: str | Path, *, selection: Mapping[str, Any] | None = None) -> dict[str, Any]:
    panel_dir = Path(panel_dir)
    selection = dict(selection or _load(panel_dir / "SELECTION.json"))
    rois = []
    for item in selection.get("rois") or PANEL_ROIS:
        roi_dir = panel_dir / item["id"]
        row = {"id": item["id"], "name": item.get("name"), "dominant": item.get("dominant")}
        if roi_dir.is_dir():
            row.update(summarize_roi(roi_dir))
        else:
            row["execution"] = {"success": False, "stages_completed": [], "note": "not_started"}
        rois.append(row)
    execution_n = sum(1 for row in rois if (row.get("execution") or {}).get("success"))
    return {
        "schema": "lod_panel_3roi_v1",
        "claim": PANEL_CLAIM,
        "sample": "small_sample_stability_n=3",
        "closed": True,
        "do_not_claim": list(PANEL_DO_NOT_CLAIM),
        "execution_success_count": execution_n,
        "visual_quality_separate": True,
        "spynet_control": False,
        "local_reprojection": "not_handled",
        "selection": str(panel_dir / "SELECTION.json"),
        "fixed_settings": selection.get("fixed_settings") or FIXED_SETTINGS,
        "rois": rois,
        "historical_unchanged": EXCLUDED_HISTORICAL,
        "notes": PANEL_NOTES,
    }


def _display_mib(mib: int | None) -> str | None:
    if mib is None:
        return None
    return f"{mib:,} MiB ≈ {mib / 1024.0:.2f} GiB"


def _display_allocated_bytes(raw: int | None) -> str | None:
    if raw is None:
        return None
    return f"{raw:,} B ≈ {raw / (1024 ** 3):.2f} GiB ({raw / 1e9:.2f} GB)"


def memory_accounting(panel_dir: str | Path) -> dict[str, Any]:
    """Keep nvidia-smi board use and DLoRAL worker allocated peaks separate.

    Report both with explicit units. Do not subtract them and read the
    remainder as parent-process occupancy, and do not treat the gap as a
    performance regression.
    """

    panel_dir = Path(panel_dir)
    stages: list[dict[str, Any]] = []
    worker: list[dict[str, Any]] = []
    for item in PANEL_ROIS:
        log = _load(panel_dir / item["id"] / "stage_log.json")
        for row in log.get("stages") or []:
            stages.append({
                "roi": item["id"],
                "stage": row.get("name"),
                "nvidia_smi_used_mib": row.get("peak_mem_mib"),
                "elapsed_s": row.get("elapsed_s"),
            })
        for level in ("l1", "l2"):
            result = _load(panel_dir / item["id"] / level / "dloral" / "result.json")
            raw = result.get("peak_cuda_memory_bytes")
            if raw is None:
                continue
            worker.append({
                "roi": item["id"],
                "level": level,
                "allocated_bytes": int(raw),
                "allocated_gib": int(raw) / (1024 ** 3),
                "allocated_gb": int(raw) / 1e9,
                "elapsed_sec": result.get("elapsed_sec"),
                "tiled": result.get("tiled"),
            })
    smi_values = [int(row["nvidia_smi_used_mib"]) for row in stages if row.get("nvidia_smi_used_mib")]
    worker_values = [int(row["allocated_bytes"]) for row in worker]
    peak_smi = max(smi_values) if smi_values else None
    peak_worker = max(worker_values) if worker_values else None
    return {
        "nvidia_smi_memory_used": {
            "peak_mib": peak_smi,
            "peak_gib": None if peak_smi is None else peak_smi / 1024.0,
            "display": _display_mib(peak_smi),
            "note": "Whole visible GPU via nvidia-smi memory.used. Parent 3DGS plus any child worker; not parent occupancy alone.",
            "query": "nvidia-smi --query-gpu=memory.used",
            "stages": stages,
        },
        "dloral_worker_allocated": {
            "peak_bytes": peak_worker,
            "peak_gib": None if peak_worker is None else peak_worker / (1024 ** 3),
            "peak_gb": None if peak_worker is None else peak_worker / 1e9,
            "display": _display_allocated_bytes(peak_worker),
            "note": "torch.cuda.max_memory_allocated inside the isolated DLoRAL worker. Tensor allocation, not whole-GPU use.",
            "same_as_prior_worker_report": True,
            "runs": worker,
        },
        "do_not": [
            "treat_board_peak_vs_worker_allocated_as_regression",
            "subtract_as_parent_occupancy",
        ],
    }


def archive_gate(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    l1 = source_side_record(root / "l1", require_prompt=True)
    l2 = source_side_record(root / "l2", require_prompt=True)
    acceptance = (root / "ACCEPTANCE.json").is_file()
    missing = []
    for label, ok in (
        ("l1_complete", l1["complete"]),
        ("l2_complete", l2["complete"]),
        ("acceptance", acceptance),
        ("l1_ckpt", (root / "l1" / "l1_final.lod.pt").is_file()),
        ("l2_ckpt", (root / "l2" / "l2_final.lod.pt").is_file()),
    ):
        if not ok:
            missing.append(label)
    return {
        "ok": not missing,
        "missing": missing,
        "l1": l1,
        "l2": l2,
        "acceptance": str(root / "ACCEPTANCE.json") if acceptance else None,
    }
