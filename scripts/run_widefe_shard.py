#!/usr/bin/env python3
"""Repair assigned prepared-view samples with the pinned FlowEdit sampler.

Requires an activated Skyfall-GS Python environment, the FlowEdit submodule,
a prepared manifest with intact native renders, and the local FLUX model named
in the protocol. CUDA assignment is inherited from the dispatcher; this worker
uses cuda:0 within CUDA_VISIBLE_DEVICES and does not select or change a device.
Each requested global flat image index is generated or verified as reused, and
--result-json records completion for the shard.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
import random
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

PREPARED_KIND = "flowedit_stage2_prepared"
PREPARED_SCHEMA_VERSION = 1
RESULT_KIND = "skyfall_flowedit_worker_result"
SIDECAR_KIND = "skyfall_flowedit_image"
SIDECAR_SCHEMA_VERSION = 1
RGB_CONTRACT = "normalized_float_clamped_v1"
PIPELINE = "FlowEditRefineIDU"
PROMPT_ORIGIN = "original_upstream_defaults"

#: The published Wide FlowEdit IDU recipe; the protocol must not change it.
FIXED_SAMPLER = {
    "T_steps": 28,
    "n_avg": 1,
    "src_guidance_scale": 1.5,
    "tar_guidance_scale": 5.5,
    "n_min": 4,
    "n_max": 10,
    "n_max_end": None,
}
_SAMPLER_ALIASES = {
    "T_steps": ("T_steps", "steps"),
    "n_avg": ("n_avg",),
    "src_guidance_scale": ("src_guidance_scale", "src_guidance"),
    "tar_guidance_scale": ("tar_guidance_scale", "tar_guidance"),
    "n_min": ("n_min",),
    "n_max": ("n_max",),
    "n_max_end": ("n_max_end",),
}
_INT_SAMPLER_KEYS = ("T_steps", "n_avg", "n_min", "n_max")


def _repo_root() -> Path:
    """Return the Skyfall-GS checkout containing this script."""
    root = Path(__file__).resolve().parents[1]
    if not (root / "train.py").is_file():
        raise SystemExit(f"error: Skyfall-GS checkout not found at {root}")
    return root


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


# ---------------------------------------------------------------------------
# small IO / identity helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _write_json_atomic(path: str | os.PathLike[str], payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _load_json(path: str | os.PathLike[str], what: str) -> dict:
    target = Path(path).expanduser()
    if not target.is_file():
        raise SystemExit(f"error: {what} not found: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: could not read {what} {target}: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"error: {what} {target} must be a JSON object")
    return payload


def _ensure_flowedit_import_path() -> None:
    """Make ``submodules/FlowEdit`` absolute before the upstream module runs.

    ``idu_refine.py`` appends the CWD-relative ``submodules/FlowEdit`` to
    ``sys.path`` and then imports ``FlowEdit_utils`` from it, so this must be
    installed first for any working directory.
    """

    flowedit_dir = REPO_ROOT / "submodules" / "FlowEdit"
    if not (flowedit_dir / "idu_refine.py").is_file():
        raise SystemExit(f"error: FlowEdit sampler missing: {flowedit_dir / 'idu_refine.py'}")
    path = str(flowedit_dir)
    if path not in sys.path:
        sys.path.insert(0, path)


def _reset_seed(seed: int) -> None:
    """Reset every RNG FlowEdit draws from, so scheduling cannot change pixels."""

    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gpu_block() -> dict:
    import torch

    block = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device": "cuda:0",
        "torch": torch.__version__,
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        block.update({
            "name": torch.cuda.get_device_name(0),
            "total_mib": int(total) >> 20,
            "free_mib_at_start": int(free) >> 20,
        })
    return block


_FLOWEDIT_CODE: dict | None = None


def _flowedit_code_identity() -> dict:
    """sha256 of the pinned FlowEdit sampler sources (computed once)."""

    global _FLOWEDIT_CODE
    if _FLOWEDIT_CODE is None:
        root = REPO_ROOT / "submodules" / "FlowEdit"
        identity = {}
        for name in ("idu_refine.py", "FlowEdit_utils.py"):
            path = root / name
            if not path.is_file():
                raise SystemExit(f"error: FlowEdit source missing: {path}")
            identity[name] = _sha256_file(path)
        _FLOWEDIT_CODE = identity
    return _FLOWEDIT_CODE


def _flux_model_identity(model_path: str) -> dict:
    index = Path(model_path) / "model_index.json"
    if not index.is_file():
        raise SystemExit(f"error: FLUX pipeline config missing: {index}")
    stat = index.stat()
    return {
        "path": str(Path(model_path).resolve()),
        "model_index_sha256": _sha256_file(index),
        "model_index_size": int(stat.st_size),
        "model_index_mtime_ns": int(stat.st_mtime_ns),
    }


class _RefineOptions:
    """Minimal IDU options object for ``validate_flowedit_options`` (pure FlowEdit)."""

    def __init__(self, sampler: dict, model_path: str, device: str) -> None:
        self.idu_model_type = "FLUX"
        self.flux_model_path = model_path
        self.idu_refine_device = device
        self.idu_flow_edit_n_min = int(sampler["n_min"])
        self.idu_flow_edit_n_max = int(sampler["n_max"])
        self.idu_flow_edit_n_max_end = -1 if sampler["n_max_end"] is None else int(sampler["n_max_end"])
        self.idu_flow_edit_n_avg = int(sampler["n_avg"])


# ---------------------------------------------------------------------------
# protocol + prepared manifests
# ---------------------------------------------------------------------------


def _protocol_settings(protocol: dict) -> dict:
    """Validate the protocol's sampler/raster contract for this worker."""

    sampler = protocol.get("sampler")
    if sampler is None:
        sampler = {}
    if not isinstance(sampler, dict):
        raise SystemExit("error: protocol 'sampler' must be an object")
    resolved = {}
    for key, expected in FIXED_SAMPLER.items():
        value = expected
        for alias in _SAMPLER_ALIASES[key]:
            if alias in sampler:
                value = sampler[alias]
                break
        if isinstance(value, bool) or not (value is None or isinstance(value, (int, float))):
            raise SystemExit(f"error: protocol sampler {key!r} must be a number or null, got {value!r}")
        if key == "n_max_end" and value == -1:
            # The repository's own convention: -1 means "no random n_max".
            value = None
        if value is not None and key in _INT_SAMPLER_KEYS:
            value = int(value)
        if value != expected:
            raise SystemExit(
                f"error: protocol sampler {key!r} is {value!r} but the Wide FlowEdit IDU recipe "
                f"requires {expected!r}; refusing to repair with a different sampler"
            )
        resolved[key] = expected

    raster = protocol.get("raster")
    if raster is not None and (isinstance(raster, bool) or not isinstance(raster, int) or raster < 1):
        raise SystemExit(f"error: protocol 'raster' must be a positive int, got {raster!r}")
    samples = protocol.get("samples_per_pose")
    if samples is not None and (isinstance(samples, bool) or not isinstance(samples, int) or samples < 1):
        raise SystemExit(f"error: protocol 'samples_per_pose' must be a positive int, got {samples!r}")
    flux_model_path = protocol.get("flux_model_path")
    if not isinstance(flux_model_path, str) or not flux_model_path.strip():
        raise SystemExit("error: protocol 'flux_model_path' must be a non-empty local directory")
    return {
        "sampler": resolved,
        "raster": raster,
        "samples_per_pose": samples,
        "model_path": str(Path(flux_model_path).expanduser().resolve()),
    }


def _load_manifest_entries(container_path: Path) -> list[dict]:
    """Ordered prepared manifests from a manifest path, list or bundle object."""

    payload = _load_json(container_path, "prepared manifest")
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload.get("episodes"), list):
        entries = payload["episodes"]
    elif isinstance(payload.get("manifests"), list):
        entries = payload["manifests"]
    else:
        return [{"payload": payload, "path": container_path}]
    if not entries:
        raise SystemExit(f"error: {container_path} lists no prepared manifests")
    loaded = []
    for position, entry in enumerate(entries):
        if isinstance(entry, str):
            path = Path(entry).expanduser()
            if not path.is_absolute():
                path = (container_path.parent / path).resolve()
            loaded.append({"payload": _load_json(path, f"prepared manifest #{position}"), "path": path})
        elif isinstance(entry, dict):
            loaded.append({"payload": entry, "path": None})
        else:
            raise SystemExit(
                f"error: {container_path} entry #{position} must be a manifest path or object, "
                f"got {type(entry).__name__}"
            )
    return loaded


def _validate_manifest(entry: dict, protocol: dict) -> dict:
    """Structural + checkpoint validation of one prepared episode manifest."""

    payload = entry["payload"]
    label = str(entry["path"]) if entry["path"] else "inline prepared manifest"
    if payload.get("kind") != PREPARED_KIND:
        raise SystemExit(f"error: {label}: expected kind {PREPARED_KIND!r}, got {payload.get('kind')!r}")
    if payload.get("schema_version") != PREPARED_SCHEMA_VERSION:
        raise SystemExit(f"error: {label}: unsupported schema_version {payload.get('schema_version')!r}")
    checkpoint = Path(str(payload.get("checkpoint", ""))).expanduser()
    if not checkpoint.is_file():
        raise SystemExit(f"error: {label}: checkpoint not found: {checkpoint}")
    checkpoint_stat = {"size": checkpoint.stat().st_size, "mtime_ns": checkpoint.stat().st_mtime_ns}
    if payload.get("checkpoint_stat") != checkpoint_stat:
        raise SystemExit(
            f"error: {label}: checkpoint {checkpoint} changed after preparation "
            f"({payload.get('checkpoint_stat')} -> {checkpoint_stat}); re-render before repairing"
        )
    samples = payload.get("samples_per_view")
    views = payload.get("views")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
        raise SystemExit(f"error: {label}: samples_per_view must be a positive int, got {samples!r}")
    if not isinstance(views, list) or not views:
        raise SystemExit(f"error: {label}: views must be a non-empty list")
    if payload.get("num_views") != len(views):
        raise SystemExit(f"error: {label}: num_views {payload.get('num_views')!r} != {len(views)} view records")
    width, height = payload.get("width"), payload.get("height")
    for name, value in (("width", width), ("height", height)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise SystemExit(f"error: {label}: {name} must be a positive int, got {value!r}")
        if value % 16:
            raise SystemExit(
                f"error: {label}: raster {name}={value} is not a multiple of 16; FlowEdit would crop it"
            )
    if protocol.get("raster") is not None and int(protocol["raster"]) != width:
        raise SystemExit(f"error: {label}: raster {width}x{height} != protocol raster {protocol['raster']}")
    if protocol.get("samples_per_pose") is not None and int(protocol["samples_per_pose"]) != samples:
        raise SystemExit(
            f"error: {label}: samples_per_view {samples} != protocol samples_per_pose "
            f"{protocol['samples_per_pose']}"
        )
    for position, view in enumerate(views):
        if not isinstance(view, dict):
            raise SystemExit(f"error: {label}: views[{position}] must be an object")
        if view.get("index") != position:
            raise SystemExit(f"error: {label}: views[{position}] index {view.get('index')!r} is not positional")
        for key in ("uid", "camera", "render_path", "rgb_sha256"):
            if key not in view:
                raise SystemExit(f"error: {label}: views[{position}] is missing {key!r}")
    episode_dir = payload.get("episode_dir")
    if not isinstance(episode_dir, str) or not episode_dir:
        raise SystemExit(f"error: {label}: episode_dir must be a non-empty string")
    if entry["path"] is not None:
        manifest_sha = _sha256_file(entry["path"])
    else:
        manifest_sha = _sha256_bytes(_canonical_json(payload).encode("utf-8"))
    return {
        "path": entry["path"],
        "label": label,
        "manifest_sha256": manifest_sha,
        "episode_dir": str(Path(episode_dir).resolve()),
        "episode_idx": payload.get("episode_idx"),
        "episode_name": Path(episode_dir).name,
        "samples_per_view": samples,
        "width": width,
        "height": height,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_stat": checkpoint_stat,
        "views": views,
    }


def _planned_global_indices(episodes: list[dict], indices: list[int]) -> list[dict]:
    """Map global flat indices onto (episode, view, sample) targets."""

    total = 0
    targets = []
    wanted = set(indices)
    for ordinal, episode in enumerate(episodes):
        images = len(episode["views"]) * episode["samples_per_view"]
        for global_index in range(total, total + images):
            if global_index in wanted:
                local = global_index - total
                view_index, sample_index = divmod(local, episode["samples_per_view"])
                targets.append({
                    "global_index": global_index,
                    "episode_ordinal": ordinal,
                    "flat_index": local,
                    "view_index": view_index,
                    "sample_index": sample_index,
                })
        total += images
    covered = {target["global_index"] for target in targets}
    missing = sorted(wanted - covered)
    if missing:
        raise SystemExit(
            f"error: global index/indices {missing} are outside the prepared image space "
            f"[0, {total}) of {len(episodes)} manifest(s)"
        )
    return sorted(targets, key=lambda target: target["global_index"])


def _reuse_reason(output_path: Path, sidecar_path: Path, identity: dict) -> str | None:
    """None when the existing output is verified reusable; else the refusal reason."""

    if not output_path.is_file():
        return "missing_output"
    if not sidecar_path.is_file():
        return "missing_sidecar"
    try:
        stored = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unreadable_sidecar"
    if not isinstance(stored, dict) or stored.get("kind") != SIDECAR_KIND:
        return "foreign_sidecar"
    if stored.get("status") != "complete":
        return "incomplete_sidecar"
    if stored.get("identity") != identity:
        return "identity_mismatch"
    output = stored.get("output")
    if not isinstance(output, dict):
        return "malformed_sidecar"
    if output.get("size") != identity["output_size"]:
        return "raster_mismatch"
    if _sha256_file(output_path) != output.get("png_sha256"):
        return "output_sha_mismatch"
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repair requested prepared-view images with the original FlowEdit sampler "
            "(one FLUX load per shard, deterministic sample-index seeds). Requires the "
            "FlowEdit submodule, prepared native renders, and protocol FLUX model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--protocol", required=True, help="Canonical WideFE protocol JSON with FLUX model and lock directory.")
    parser.add_argument("--prepared", required=True,
                        help="flowedit_prepared.json, a JSON list of them, or an object with "
                             "an 'episodes'/'manifests' list.")
    parser.add_argument("--indices", required=True,
                        help="Comma-separated GLOBAL flat image indices to produce.")
    parser.add_argument("--result-json", required=True, help="Atomic per-worker result JSON path.")
    parser.add_argument("--work-dir", default=None,
                        help="Private directory for the upstream auto-saved PNGs "
                             "(default: <result-json dir>/flowedit_work/<unique>).")
    return parser


def _parse_indices(text: str) -> tuple[list[int], int]:
    values: list[int] = []
    if not text or not text.strip():
        raise SystemExit("error: --indices must list at least one global flat image index")
    for chunk in text.split(","):
        token = chunk.strip()
        if not token:
            raise SystemExit(f"error: --indices contains an empty entry in {text!r}")
        if not token.isdigit():
            raise SystemExit(f"error: --indices entry {token!r} is not a non-negative integer")
        values.append(int(token))
    unique = sorted(set(values))
    return unique, len(values) - len(unique)


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    indices, duplicates = _parse_indices(args.indices)
    # The dispatcher owns GPU assignment. Preserve CUDA_VISIBLE_DEVICES exactly;
    # cuda:0 is the worker-local device in that inherited visibility set.
    device = "cuda:0"
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if not visible:
        raise SystemExit("error: CUDA_VISIBLE_DEVICES must be assigned by the dispatcher")

    result_path = Path(args.result_json).expanduser().resolve()
    protocol_path = Path(args.protocol).expanduser().resolve()
    protocol = _load_json(protocol_path, "protocol")
    settings = _protocol_settings(protocol)
    protocol_repo = protocol.get("repo_root")
    if not isinstance(protocol_repo, str) or not protocol_repo.strip():
        raise SystemExit("error: protocol 'repo_root' must be a non-empty checkout path")
    if Path(protocol_repo).expanduser().resolve() != REPO_ROOT:
        raise SystemExit(f"error: protocol repo_root {protocol_repo!r} does not match script checkout {REPO_ROOT}")
    prepared_path = Path(args.prepared).expanduser().resolve()

    episodes = [_validate_manifest(entry, protocol) for entry in _load_manifest_entries(prepared_path)]
    seen_dirs: dict[str, str] = {}
    for episode in episodes:
        previous = seen_dirs.get(episode["episode_dir"])
        if previous is not None:
            raise SystemExit(
                f"error: prepared manifests {previous} and {episode['label']} share the episode directory "
                f"{episode['episode_dir']}; global indices would alias one output file"
            )
        seen_dirs[episode["episode_dir"]] = episode["label"]

    targets = _planned_global_indices(episodes, indices)

    # The original upstream prompt strings are the identity's prompt block, so
    # they come from the pinned FlowEdit module itself (never re-typed here).
    _ensure_flowedit_import_path()
    from submodules.FlowEdit.idu_refine import default_src_prompt, default_tar_prompt

    static_identity = {
        "pipeline": PIPELINE,
        "model_type": "FLUX",
        "model": _flux_model_identity(settings["model_path"]),
        "flowedit_code": _flowedit_code_identity(),
        "prompts": {
            "source": default_src_prompt,
            "target": default_tar_prompt,
            "origin": PROMPT_ORIGIN,
        },
        "sampler": settings["sampler"],
        "rgb_contract": RGB_CONTRACT,
    }
    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else (
        result_path.parent / "flowedit_work" / f"w{os.getpid()}_{uuid.uuid4().hex[:8]}"
    )
    result = {
        "kind": RESULT_KIND,
        "schema_version": 1,
        "status": "running",
        "started_at": _utc_now(),
        "worker": {
            "pid": os.getpid(),
            "host": os.uname().nodename,
            "work_dir": str(work_dir),
            "device_requested": device,
            "device_used": device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "visible_devices": visible,
        },
        "protocol": {"path": str(protocol_path), "sha256": _sha256_file(protocol_path)},
        "prepared": [
            {
                "path": str(episode["path"]) if episode["path"] else None,
                "sha256": episode["manifest_sha256"],
                "episode_idx": episode["episode_idx"],
                "episode_dir": episode["episode_dir"],
                "checkpoint": episode["checkpoint"],
                "num_views": len(episode["views"]),
                "samples_per_view": episode["samples_per_view"],
                "width": episode["width"],
                "height": episode["height"],
            }
            for episode in episodes
        ],
        "requested_indices": indices,
        "duplicates_removed": duplicates,
        "identity": static_identity,
        "images": [],
        "counts": {"requested": len(targets), "generated": 0, "reused": 0, "failed": 0},
        "gpu": None,
        "timings": {},
    }
    records: list[dict] = result["images"]
    pending: list[dict] = []

    def _publish(status: str) -> None:
        result["status"] = status
        result["counts"]["generated"] = sum(1 for record in records if record["status"] == "generated")
        result["counts"]["reused"] = sum(1 for record in records if record["status"] == "reused")
        result["counts"]["failed"] = sum(1 for record in records if record["status"] == "failed")
        result["timings"]["total_seconds"] = round(time.perf_counter() - started, 3)
        result["finished_at"] = _utc_now()
        _write_json_atomic(result_path, result)

    # ---- resolve every requested index: verify the input, then try to reuse ----
    for target in targets:
        episode = episodes[target["episode_ordinal"]]
        view = episode["views"][target["view_index"]]
        input_path = Path(str(view["render_path"])).expanduser().resolve()
        output_dir = Path(episode["episode_dir"]) / "render_refine"
        output_path = output_dir / f"{target['flat_index']:05d}.png"
        sidecar_path = output_dir / f"{target['flat_index']:05d}.flowedit.json"
        if not input_path.is_file():
            raise SystemExit(
                f"error: episode render missing for global index {target['global_index']}: {input_path}"
            )
        with Image.open(input_path) as probe:
            size = [int(probe.width), int(probe.height)]
            pixel_sha = _sha256_bytes(probe.convert("RGB").tobytes())
        if size != [episode["width"], episode["height"]]:
            raise SystemExit(
                f"error: {input_path} is {size} but the manifest declares "
                f"{[episode['width'], episode['height']]}"
            )
        if pixel_sha != str(view["rgb_sha256"]):
            raise SystemExit(
                f"error: {input_path} raw RGB sha256 {pixel_sha} does not match the prepared manifest "
                f"{view['rgb_sha256']}; re-render the episode before repairing it"
            )
        identity = {
            **static_identity,
            "prepared_kind": PREPARED_KIND,
            "prepared_schema_version": PREPARED_SCHEMA_VERSION,
            "prepared_manifest": str(episode["path"]) if episode["path"] else None,
            "prepared_manifest_sha256": episode["manifest_sha256"],
            "checkpoint": episode["checkpoint"],
            "checkpoint_stat": episode["checkpoint_stat"],
            "episode_idx": episode["episode_idx"],
            "episode_dir": episode["episode_dir"],
            "flat_index": target["flat_index"],
            "view_index": target["view_index"],
            "sample_index": target["sample_index"],
            "uid": int(view["uid"]),
            "input_path": str(input_path),
            "input_png_sha256": _sha256_file(input_path),
            "input_rgb_sha256": pixel_sha,
            "input_size": size,
            "seed": int(target["sample_index"]),
            "output_path": str(output_path),
            "output_size": size,
        }
        record = {
            "global_index": target["global_index"],
            "episode_ordinal": target["episode_ordinal"],
            "episode_idx": episode["episode_idx"],
            "episode_dir": episode["episode_dir"],
            "flat_index": target["flat_index"],
            "view_index": target["view_index"],
            "sample_index": target["sample_index"],
            "uid": int(view["uid"]),
            "input_path": str(input_path),
            "input_png_sha256": identity["input_png_sha256"],
            "input_rgb_sha256": pixel_sha,
            "output_path": str(output_path),
            "identity_path": str(sidecar_path),
            "seed": int(target["sample_index"]),
            "prompts": dict(static_identity["prompts"]),
            "sampler": dict(settings["sampler"]),
            "status": None,
            "seconds": None,
            "gpu": None,
        }
        reuse_reason = _reuse_reason(output_path, sidecar_path, identity)
        if reuse_reason is None:
            stored = _load_json(sidecar_path, "sidecar")
            record.update({
                "status": "reused",
                "seconds": 0.0,
                "output_png_sha256": stored["output"]["png_sha256"],
                "output_rgb_sha256": stored["output"]["rgb_sha256"],
                "output_size": stored["output"]["size"],
                "identity_sha256": stored.get("identity_sha256"),
                "gpu": stored.get("gpu"),
                "reuse": {"reused": True, "reason": "verified_sidecar_identity"},
            })
        else:
            record["reuse"] = {"reused": False, "rejected_cache": reuse_reason}
            pending.append({
                "record": record, "identity": identity, "input_path": input_path,
                "output_path": output_path, "sidecar_path": sidecar_path,
                "reuse_rejected": reuse_reason,
            })
        records.append(record)

    if not pending:
        print(
            f"[flowedit-worker] all {len(records)} requested image(s) already verified; "
            "no FLUX load needed",
            flush=True,
        )
        _publish("complete")
        print(f"FLOWEDIT_WORKER_COMPLETE {result_path}", flush=True)
        return 0

    # ---- validate the FlowEdit assets, then load FLUX exactly once ----
    from refinement.flowedit_stage2 import validate_flowedit_options

    validate_flowedit_options(_RefineOptions(settings["sampler"], settings["model_path"], device))

    work_dir.mkdir(parents=True, exist_ok=True)
    from submodules.FlowEdit.idu_refine import FlowEditRefineIDU

    gpu = _gpu_block()
    result["gpu"] = gpu
    print(
        f"[flowedit-worker] loading FLUX {settings['model_path']} on {device} "
        f"({gpu.get('name')}); {len(pending)} pending of {len(records)} requested",
        flush=True,
    )
    load_started = time.perf_counter()
    # FLUX first materializes CPU weights before moving them to its leased GPU.
    # Serialize checkpoint staging, not inference, through the protocol lock directory.
    load_lock_path = Path(protocol["lock_dir"]) / "flowedit-model-load.lock"
    with load_lock_path.open("a+") as load_lock:
        fcntl.flock(load_lock, fcntl.LOCK_EX)
        model_load_started = time.perf_counter()
        result["timings"]["pipeline_load_wait_seconds"] = model_load_started - load_started
        pipe = FlowEditRefineIDU(
            save_path=str(work_dir), device=device, model_type="FLUX", model_path=settings["model_path"],
        )
    result["timings"]["pipeline_load_seconds"] = round(time.perf_counter() - model_load_started, 3)

    failure: str | None = None
    try:
        for position, item in enumerate(pending):
            record = item["record"]
            identity = item["identity"]
            sample_index = identity["sample_index"]
            _reset_seed(sample_index)
            image_started = time.perf_counter()
            with Image.open(item["input_path"]) as handle:
                source = handle.convert("RGB")
            if (source.width, source.height) != (identity["input_size"][0], identity["input_size"][1]):
                raise RuntimeError(f"input raster changed while repairing: {item['input_path']}")
            array = np.asarray(source, dtype=np.float32) / 255.0
            del source
            repaired = pipe.run(
                [array],
                src_prompt=static_identity["prompts"]["source"],
                tar_prompt=static_identity["prompts"]["target"],
                T_steps=settings["sampler"]["T_steps"],
                n_avg=settings["sampler"]["n_avg"],
                src_guidance_scale=settings["sampler"]["src_guidance_scale"],
                tar_guidance_scale=settings["sampler"]["tar_guidance_scale"],
                n_min=settings["sampler"]["n_min"],
                n_max=settings["sampler"]["n_max"],
                n_max_end=settings["sampler"]["n_max_end"],
            )
            del array
            if len(repaired) != 1 or repaired[0].mode != "RGB":
                raise RuntimeError(
                    f"FlowEdit returned {len(repaired)} image(s) for global index {record['global_index']}"
                )
            output = repaired[0]
            del repaired
            expected_size = (identity["output_size"][0], identity["output_size"][1])
            if output.size != expected_size:
                raise RuntimeError(
                    f"FlowEdit changed the raster of global index {record['global_index']}: "
                    f"{output.size} != {expected_size}"
                )
            pixel_sha = _sha256_bytes(output.tobytes())
            item["output_path"].parent.mkdir(parents=True, exist_ok=True)
            temporary = item["output_path"].with_name(f".{item['output_path'].name}.{os.getpid()}.tmp.png")
            output.save(temporary, format="PNG")
            del output
            os.replace(temporary, item["output_path"])
            # Upstream auto-save target of this private work directory.
            upstream_auto = work_dir / "00000.png"
            if upstream_auto.is_file():
                upstream_auto.unlink()
            png_sha = _sha256_file(item["output_path"])
            with Image.open(item["output_path"]) as probe:
                written_size = [int(probe.width), int(probe.height)]
            if written_size != identity["output_size"]:
                raise RuntimeError(
                    f"written raster {written_size} != {identity['output_size']} for {item['output_path']}"
                )

            seconds = round(time.perf_counter() - image_started, 3)
            sidecar = {
                "schema_version": SIDECAR_SCHEMA_VERSION,
                "kind": SIDECAR_KIND,
                "created_at": _utc_now(),
                "status": "complete",
                "global_index": record["global_index"],
                "identity": identity,
                "identity_sha256": _sha256_bytes(_canonical_json(identity).encode("utf-8")),
                "output": {
                    "path": str(item["output_path"]),
                    "png_sha256": png_sha,
                    "rgb_sha256": pixel_sha,
                    "size": written_size,
                },
                "prompts": static_identity["prompts"],
                "sampler": settings["sampler"],
                "seed": sample_index,
                "seconds": seconds,
                "gpu": {**gpu, "device_used": device},
                "reuse_rejected": item["reuse_rejected"],
            }
            _write_json_atomic(item["sidecar_path"], sidecar)
            record.update({
                "status": "generated",
                "seconds": seconds,
                "output_png_sha256": png_sha,
                "output_rgb_sha256": pixel_sha,
                "output_size": written_size,
                "identity_sha256": sidecar["identity_sha256"],
                "gpu": sidecar["gpu"],
            })
            print(
                f"[flowedit-worker] {position + 1}/{len(pending)} global {record['global_index']} "
                f"flat {record['flat_index']:05d} seed {sample_index} in {seconds}s -> {item['output_path']}",
                flush=True,
            )
            if (position + 1) % 8 == 0:
                gc.collect()
        _publish("complete")
    except BaseException:  # noqa: BLE001 - record the exact failure, then stop
        failure = traceback.format_exc()
        for record in records:
            if record["status"] is None:
                record["status"] = "failed"
                record["error"] = failure.splitlines()[-1]
        result["error"] = failure
        _publish("failed")
    finally:
        # The upstream destructor copies the whole pipeline to CPU first, which
        # does not fit beside other stages; release CUDA weights directly.
        try:
            pipe.pipe = None
        except Exception:  # pragma: no cover - defensive
            pass
        del pipe
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            work_dir.rmdir()
        except OSError:
            pass

    if failure is not None:
        print(f"[flowedit-worker] FAILED into {result_path}\n{failure}", file=sys.stderr, flush=True)
        return 1
    print(f"FLOWEDIT_WORKER_COMPLETE {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
