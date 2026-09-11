"""Run fingerprints so LoD stages cannot silently reuse the wrong cache."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


POLICY = {
    "main_alignment": "geometry",
    "spynet": "control_only_same_frozen_upstream",
    "roundtrip_gate_default": False,
    "absorption_not_for_mode_ranking": True,
    "labeled_regions_are_not_confirmed_artifacts": True,
    "entry": "scripts/run_lod_scale_chain.py",
    "not_entry": ["train_zoom_mvp.py", "train_zoom_gen.py"],
    "do_not": [
        "go_to_8x",
        "change_densify",
        "add_scale_offset_constraint",
        "use_train_zoom_mvp_for_lod",
        "use_train_zoom_gen_for_lod_absorb",
        "make_roundtrip_gate_default",
        "rank_geometry_vs_spynet_by_absorption",
    ],
    "scope": ["supervision_absorption", "old_scale_influence", "co_visible_stability"],
    "out_of_scope": [
        "generated_detail_photorealism",
        "geometry_better_than_spynet",
        "confirmed_structural_artifacts_from_labeled_regions",
        "local_reprojection_as_blocker",
    ],
}

SUPERVISION_KEYS = (
    "start_checkpoint",
    "parent_lod",
    "roi",
    "view_index",
    "zoom_factor",
    "step_scale",
    "alignment",
    "seed",
    "max_roundtrip_error_px",
    "dloral_ckpt",
    "prompt_sha256",
)


def write_json(path: str | os.PathLike[str], payload: Any) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_train_lineage(output_dir: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    """Keep supervision fingerprints in ``lineage.json``; training writes a sidecar.

    Overwriting ``lineage.json`` with ``kind=lod_train`` would make a later
    supervise-stage reuse check look at the wrong record.
    """

    root = Path(output_dir)
    write_json(root / "train_lineage.json", payload)
    lineage_path = root / "lineage.json"
    if lineage_path.is_file():
        try:
            saved = json.loads(lineage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            saved = {}
        if saved.get("kind") == "lod_supervision":
            return
    write_json(lineage_path, payload)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str], *, max_bytes: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = max_bytes
    with open(path, "rb") as handle:
        while True:
            chunk_n = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            if chunk_n <= 0:
                break
            chunk = handle.read(chunk_n)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return digest.hexdigest()


def file_identity(path: str | os.PathLike[str] | None, *, hash_file: bool = False) -> dict[str, Any] | None:
    if path is None or str(path).strip() == "":
        return None
    dest = Path(path)
    if not dest.is_file():
        return {"path": str(dest.resolve()) if dest.exists() else str(dest), "missing": True}
    stat = dest.stat()
    payload: dict[str, Any] = {
        "path": str(dest.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "missing": False,
    }
    if hash_file:
        payload["sha256"] = sha256_file(dest)
    return payload


def prompt_sha256(path: str | os.PathLike[str] | None, text: str | None = None) -> str | None:
    if text is None and path and Path(path).is_file():
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        text = str(payload.get("target_prompt") or payload.get("prompt") or "")
        if not text and isinstance(payload.get("config"), dict):
            text = str(payload["config"].get("target_prompt") or "")
    if not text:
        return None
    return sha256_bytes(text.strip().encode("utf-8"))


def roi_dict(center_x: float, center_y: float, width: float, height: float) -> dict[str, float]:
    return {
        "center_x": float(center_x),
        "center_y": float(center_y),
        "width": float(width),
        "height": float(height),
    }


def supervision_lineage(
    *,
    start_checkpoint: str,
    parent_lod: str | None,
    roi: Mapping[str, float],
    view_index: int,
    zoom_factor: float,
    step_scale: float,
    alignment: str,
    seed: int,
    max_roundtrip_error_px: float | None,
    dloral_ckpt: str | None,
    prompt_path: str | None = None,
    prompt_text: str | None = None,
    refined_image: str | None = None,
    image_name: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": "lod_supervision",
        "start_checkpoint": file_identity(start_checkpoint),
        "parent_lod": file_identity(parent_lod),
        "roi": dict(roi),
        "view_index": int(view_index),
        "view_name": str(image_name or "").strip() or None,
        "zoom_factor": float(zoom_factor),
        "step_scale": float(step_scale),
        "alignment": str(alignment),
        "seed": int(seed),
        "max_roundtrip_error_px": None if max_roundtrip_error_px is None else float(max_roundtrip_error_px),
        "dloral_ckpt": file_identity(dloral_ckpt),
        "prompt_sha256": prompt_sha256(prompt_path, prompt_text),
        "refined_image": file_identity(refined_image, hash_file=True) if refined_image else None,
        "roundtrip_gate_default": False,
    }


def identities_match(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None) -> bool:
    if left is None and right is None:
        return True
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    if left.get("missing") or right.get("missing"):
        return False
    if left.get("sha256") and right.get("sha256"):
        return left["sha256"] == right["sha256"]
    return (
        left.get("path") == right.get("path")
        and left.get("size") == right.get("size")
        and left.get("mtime_ns") == right.get("mtime_ns")
    )


def lineage_mismatches(saved: Mapping[str, Any], current: Mapping[str, Any], *, keys: Sequence[str] = SUPERVISION_KEYS) -> list[str]:
    """Return the keys that would make cache reuse unsafe."""

    mismatches: list[str] = []
    for key in keys:
        old = saved.get(key)
        new = current.get(key)
        if key in ("start_checkpoint", "parent_lod", "dloral_ckpt", "refined_image"):
            if not identities_match(old if isinstance(old, Mapping) else None, new if isinstance(new, Mapping) else None):
                mismatches.append(key)
            continue
        if old != new:
            mismatches.append(key)
    return mismatches


def assert_lineage_reusable(saved: Mapping[str, Any], current: Mapping[str, Any], *, keys: Sequence[str] = SUPERVISION_KEYS) -> None:
    bad = lineage_mismatches(saved, current, keys=keys)
    if bad:
        raise ValueError(
            "Refusing to reuse supervision/cache; lineage differs on: "
            + ", ".join(bad)
            + ". Pass --force to ignore, or regenerate that stage."
        )


def reuse_supervision_dir(
    directory: str | os.PathLike[str],
    current: Mapping[str, Any],
    *,
    force: bool = False,
    allow_unlineaged: bool = False,
) -> dict[str, Any]:
    """Decide whether a stage directory can be reused.

    ``allow_unlineaged`` is an explicit compatibility channel. It does not make
    the historical source record complete.
    """

    root = Path(directory)
    refined = root / "refined.png"
    if not refined.is_file():
        return {"reuse": False, "source_record": "missing_refined", "complete": False, "path": str(root)}
    lineage_path = root / "lineage.json"
    if not lineage_path.is_file():
        if allow_unlineaged or force:
            return {
                "reuse": True,
                "source_record": "unlineaged_compat",
                "complete": False,
                "path": str(root),
                "note": "--allow_unlineaged_reuse permits cache reuse; it does not verify historical provenance.",
            }
        raise ValueError(
            f"{root} has refined.png but no lineage.json; pass --allow_unlineaged_reuse or regenerate."
        )
    saved = json.loads(lineage_path.read_text(encoding="utf-8"))
    if force:
        return {"reuse": True, "source_record": "forced", "complete": bool(saved), "path": str(root)}
    assert_lineage_reusable(saved, current)
    return {"reuse": True, "source_record": "lineage_match", "complete": True, "path": str(root)}
