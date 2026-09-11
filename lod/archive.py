"""Build a uniform LoD acceptance table from existing stage outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from lod.lineage import POLICY, file_identity, write_json


def _load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _curve_rows(curve: list[Mapping[str, Any]], *, n_key: str) -> list[dict[str, Any]]:
    rows = []
    for item in curve:
        active = item.get("active") or {}
        parent = item.get("parent_2x") or {}
        rows.append({
            "step": int(item["step"]),
            "n": item.get(n_key, active.get("n")),
            "rgb": (item.get("target") or {}).get("l1_to_refined"),
            "hf": (item.get("target") or {}).get("hf_l1_to_refined"),
            "train_gt_1x": (item.get("train_views_mean") or {}).get("l1_to_gt_mean"),
            "parent_2x_rgb": parent.get("l1_vs_frozen_l1"),
            "l2_weight_at_2x": parent.get("l2_weight_mean"),
            "crops": {
                name: (crop or {}).get("l1_to_refined")
                for name, crop in ((item.get("target") or {}).get("crops") or {}).items()
            },
        })
    return rows


def _row_at(rows: list[Mapping[str, Any]], step: int) -> Mapping[str, Any] | None:
    for row in rows:
        if int(row["step"]) == step:
            return row
    return None


def _rel(before, after):
    if before in (None, 0):
        return None
    return (after - before) / before


def _status(pass_now: bool | None, pending: bool = False) -> str:
    if pending:
        return "pending"
    if pass_now is True:
        return "pass"
    if pass_now is False:
        return "fail"
    return "pending"


def _find_video_meta(*directories: Path | str | None) -> dict[str, Any]:
    for directory in directories:
        if not directory:
            continue
        root = Path(directory)
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.json")):
            data = _load(path)
            if data.get("output") or data.get("max_mean_rgb_jump") is not None:
                return data
    return {}


def _archive_links(archive: Path | None) -> dict[str, str]:
    if archive is None or not archive.is_dir():
        return {}
    links: dict[str, str] = {}
    for path in archive.rglob("*"):
        if path.is_symlink():
            links[str(path.relative_to(archive))] = str(path.resolve())
    return links


def _compact_recover(recover: Mapping[str, Any] | None, *, path: str | None = None) -> dict[str, Any] | None:
    if recover is None:
        return None
    return {
        "ok": recover.get("ok"),
        "freeze_ok": recover.get("freeze_ok"),
        "resume_ok": recover.get("resume_ok"),
        "resume_probe": recover.get("resume_probe"),
        "stage_link_ok": recover.get("stage_link_ok"),
        "stage_link_rgb_l1": recover.get("stage_link_rgb_l1"),
        "stage_link_rgb_l1_float": recover.get("stage_link_rgb_l1_float"),
        "n_l1": recover.get("n_l1"),
        "n_l1_in_l2": recover.get("n_l1_in_l2"),
        "l0_changed_after_l1_load": recover.get("l0_changed_after_l1_load"),
        "l1_in_l2_mismatches": recover.get("l1_in_l2_mismatches"),
        "path": path,
    }


def _l2_weight_at_2x(
    l2: Path | None,
    row: Mapping[str, Any] | None,
    summary: Mapping[str, Any],
) -> float | None:
    if row and row.get("l2_weight_at_2x") is not None:
        return row["l2_weight_at_2x"]
    frozen = (summary.get("acceptance") or {}).get("frozen_params_vs_parent_scale") or {}
    if frozen.get("l2_weight_at_2x") is not None:
        return frozen["l2_weight_at_2x"]
    if l2 is None:
        return None
    cross = _load(l2 / "cross_view.json") if (l2 / "cross_view.json").is_file() else {}
    for item in ((cross.get("scale_sweep") or {}).get("rows") or []):
        if abs(float(item.get("factor", 0)) - 2.0) < 1e-6:
            weight = (item.get("weights") or {}).get("level_2")
            if weight is not None:
                return weight
    return None


def build_acceptance(
    *,
    name: str,
    scene: str,
    view_index: int,
    roi: Mapping[str, float],
    l1_dir: str | Path | None,
    l2_dir: str | Path | None,
    start_checkpoint: str | None = None,
    status: str = "recorded",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    l1 = Path(l1_dir).resolve() if l1_dir else None
    l2 = Path(l2_dir).resolve() if l2_dir else None
    l1_curve = _load(l1 / "absorption_curve.json") if l1 and (l1 / "absorption_curve.json").is_file() else {}
    l2_curve = _load(l2 / "absorption_curve.json") if l2 and (l2 / "absorption_curve.json").is_file() else {}
    l1_rows = _curve_rows(list(l1_curve.get("curve") or []), n_key="n_l1")
    l2_rows = _curve_rows(list(l2_curve.get("curve") or []), n_key="n_l2")
    extra = dict(extra or {})
    archive_dir = Path(extra["archive_dir"]).resolve() if extra.get("archive_dir") else None
    l1_0, l1_500 = _row_at(l1_rows, 0), _row_at(l1_rows, 500)
    l2_0, l2_50, l2_500 = _row_at(l2_rows, 0), _row_at(l2_rows, 50), _row_at(l2_rows, 500)
    video_meta = extra.get("video") or _find_video_meta(
        extra.get("video_dir"),
        None if archive_dir is None else archive_dir / "video",
        None if l2 is None else l2 / "video",
    )
    freeze_l1 = _load(l1 / "freeze.json") if l1 and (l1 / "freeze.json").is_file() else {}
    freeze_l2 = _load(l2 / "freeze.json") if l2 and (l2 / "freeze.json").is_file() else {}
    lineage_l1 = _load(l1 / "lineage.json") if l1 and (l1 / "lineage.json").is_file() else {}
    lineage_l2 = _load(l2 / "lineage.json") if l2 and (l2 / "lineage.json").is_file() else {}
    summary = _load(l2 / "lod_l2_summary.json") if l2 and (l2 / "lod_l2_summary.json").is_file() else {}
    recover = extra.get("recover")

    absorption_ok = None
    if l2_0 and l2_500 and l2_0.get("rgb") and l2_500.get("rgb"):
        absorption_ok = l2_500["rgb"] < l2_0["rgb"] and (
            l2_500.get("hf") is None or l2_0.get("hf") is None or l2_500["hf"] <= l2_0["hf"] * 1.05
        )
    train_gt = None if l2_500 is None else l2_500.get("train_gt_1x")
    old_scale = None if l2_500 is None else l2_500.get("parent_2x_rgb")

    freeze_flag = None
    if recover is not None:
        freeze_flag = recover.get("freeze_ok")
    elif freeze_l1 or freeze_l2:
        freeze_flag = bool(freeze_l1.get("ok", True) and freeze_l2.get("ok", True) if (freeze_l1 and freeze_l2) else (freeze_l1.get("ok") or freeze_l2.get("ok")))
    freeze_l1_ok = freeze_l1.get("ok")
    freeze_l2_ok = freeze_l2.get("ok")
    if freeze_l1_ok is None and recover is not None:
        freeze_l1_ok = (recover.get("l1_completed") or {}).get("ok")
    if freeze_l2_ok is None and recover is not None:
        freeze_l2_ok = (recover.get("l2_completed") or {}).get("ok")
    l2_weight = _l2_weight_at_2x(l2, l2_500, summary)
    links = extra.get("links") or _archive_links(archive_dir)
    checks = {
        "stage_link": {
            "status": _status(None if recover is None else recover.get("stage_link_ok"), pending=recover is None),
            "l1_checkpoint": None if l1 is None else str(l1 / "l1_final.lod.pt"),
            "l2_render_input": None if l2 is None else str(l2 / "render_input.png"),
            "rgb_l1_vs_saved_4x_input": None if recover is None else recover.get("stage_link_rgb_l1"),
            "rgb_l1_vs_saved_4x_input_float": None if recover is None else recover.get("stage_link_rgb_l1_float"),
            "note": "PNG L1 gates input content match with the archived 4x render. It does not reconstruct historical execution order. Float L1 records PNG quantization (~0.002) and is not the gate.",
        },
        "resume_cache": {
            "status": _status(None if recover is None else recover.get("resume_ok"), pending=recover is None),
            "l1_lineage": bool(lineage_l1),
            "l2_lineage": bool(lineage_l2),
            "resume_probe": None if recover is None else recover.get("resume_probe"),
            "note": "Resume must keep Stage1, ROI, prompt, seed, DLoRAL weights, and parent LoD. Unlineaged reused runs are gated by recover freeze+stage_link.",
        },
        "freeze": {
            "status": _status(freeze_flag, pending=freeze_flag is None),
            "l1_ok": freeze_l1_ok,
            "l2_ok": freeze_l2_ok,
            "note": "L0, completed layers, and appearance stay frozen. Old-scale RGB change is recorded separately.",
        },
        "absorption": {
            "status": _status(absorption_ok, pending=l2_500 is None),
            "l1_rgb": None if l1_0 is None or l1_500 is None else {"0": l1_0["rgb"], "500": l1_500["rgb"], "rel": _rel(l1_0["rgb"], l1_500["rgb"])},
            "l2_rgb": None if l2_0 is None or l2_500 is None else {
                "0": l2_0["rgb"],
                "50": None if l2_50 is None else l2_50["rgb"],
                "500": l2_500["rgb"],
                "rel": _rel(l2_0["rgb"], l2_500["rgb"]),
            },
            "l2_hf": None if l2_0 is None or l2_500 is None else {
                "0": l2_0["hf"],
                "50": None if l2_50 is None else l2_50["hf"],
                "500": l2_500["hf"],
                "rel": _rel(l2_0["hf"], l2_500["hf"]),
            },
            "crops_500": None if l2_500 is None else l2_500.get("crops"),
            "not_for": "mode_ranking",
        },
        "old_scale": {
            "status": _status(train_gt is not None, pending=train_gt is None),
            "train_gt_1x": train_gt,
            "parent_2x_rgb_vs_frozen_l1": old_scale,
            "l2_weight_at_2x": l2_weight,
            "require_zero_overlap": False,
            "label": "l2_contribution_at_parent_scale",
            "not": "leak",
        },
        "continuous_zoom": {
            "status": (extra.get("continuous_zoom") or {}).get("status") or "pending_visual",
            "video": video_meta.get("output"),
            "max_mean_rgb_jump": video_meta.get("max_mean_rgb_jump"),
            "check_manually": ["building edges", "vehicle silhouettes"],
            "do_not_compare_sparse_and_dense_jumps": True,
            "do_not_write": "video_fully_passed",
            "passed": (extra.get("continuous_zoom") or {}).get("passed"),
            "unresolved": (extra.get("continuous_zoom") or {}).get("unresolved"),
            "frames": (extra.get("continuous_zoom") or {}).get("frames"),
        },
    }
    payload: dict[str, Any] = {
        "schema": "lod_scale_chain_v1",
        "name": name,
        "scene": scene,
        "view_index": int(view_index),
        "roi": dict(roi),
        "status": status,
        "policy": POLICY,
        "start_checkpoint": file_identity(start_checkpoint) if start_checkpoint else None,
        "l1_dir": None if l1 is None else str(l1),
        "l2_dir": None if l2 is None else str(l2),
        "alignment_main": "geometry",
        "spynet": "control_only",
        "roundtrip_gate_default": False,
        "l1_eval": l1_rows,
        "l2_eval": l2_rows,
        "checks": checks,
        "local_reprojection": {
            "priority": "low",
            "labeled_regions_are_not_confirmed_artifacts": True,
        },
        "recover": _compact_recover(recover, path=extra.get("recover_path")),
        "links": links or None,
        "sources": extra.get("sources"),
        "source_record": extra.get("source_record"),
        "resume_boundary": extra.get("resume_boundary") or {
            "segment_restart": True,
            "l1_resume": True,
            "l2_resume": True,
            "not": "all_training_stages_historically_checkpointed",
        },
        "notes": extra.get("notes"),
        "early_densify": extra.get("early_densify") or (summary.get("acceptance") or {}).get("early_densify"),
    }
    return payload


HISTORICAL_INDEX_LOCK = {
    "jax068_ybuilding": {"status": "passed_small_scope", "source_complete": False},
    "jax068_0p28_0p28": {"status": "regression_sample", "source_complete": False},
}


def merge_experiment_index(
    existing: list[Mapping[str, Any]] | None,
    new_entry: Mapping[str, Any],
    *,
    lock: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Replace or append by name. Locked historical fields are not upgraded."""

    lock = HISTORICAL_INDEX_LOCK if lock is None else lock
    entries = [dict(item) for item in (existing or [])]
    name = new_entry.get("name")
    replaced = False
    merged: list[dict[str, Any]] = []
    for item in entries:
        if name and item.get("name") == name:
            merged.append(dict(new_entry))
            replaced = True
        else:
            merged.append(item)
    if not replaced:
        merged.append(dict(new_entry))
    for locked_name, fields in lock.items():
        for item in merged:
            if item.get("name") == locked_name:
                item.update(dict(fields))
    return merged


def write_acceptance(path: str | Path, payload: Mapping[str, Any]) -> None:
    write_json(path, payload)


def experiment_index_entry(acceptance: Mapping[str, Any]) -> dict[str, Any]:
    checks = acceptance.get("checks") or {}
    return {
        "name": acceptance.get("name"),
        "scene": acceptance.get("scene"),
        "roi": acceptance.get("roi"),
        "status": acceptance.get("status"),
        "l1_dir": acceptance.get("l1_dir"),
        "l2_dir": acceptance.get("l2_dir"),
        "alignment_main": acceptance.get("alignment_main"),
        "start_checkpoint": ((acceptance.get("start_checkpoint") or {}).get("path")),
        "source_complete": bool(
            ((acceptance.get("source_record") or {}).get("l1") or {}).get("complete")
            and ((acceptance.get("source_record") or {}).get("l2") or {}).get("complete")
        ),
        "pending": [key for key, value in checks.items() if str((value or {}).get("status") or "").startswith("pending")],
        "checks": {key: (value or {}).get("status") for key, value in checks.items()},
    }
