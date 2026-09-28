#!/usr/bin/env python3
"""Sharded plan/merge CLI for the wide-FlowEdit GaussianZoom supervision.

``plan`` splits one monolithic ``prepare_scene_zoom`` run into independent
per-shard jobs.  Every job still collects ALL FlowEdit views and the
checkpoint's native real replay views (complete view/context/neighbor pool),
but generates new SR targets only for its own disjoint ``--target_view_ids``
shard, into its own output directory::

    python scripts/prepare_scene_zoom_shards.py plan --protocol <protocol.json> \\
        --checkpoint <checkpoint.pth> --zoom 2 --output-dir <plan-root> \\
        --flowedit-manifests <manifest.json> [...] --views-per-shard 6

It writes ``<root>/plan.json`` (self-contained merge contract) and
``<root>/jobs.json`` (the idle-dispatch job contract: ``id`` / ``command`` /
``expected_outputs``).  The planner is CPU-only and never stats the remote
runtime paths it records.

``merge`` verifies every shard summary and supervision manifest, then folds
the shards into one trainable ``skyfall_scene_zoom`` manifest plus a report::

    python scripts/prepare_scene_zoom_shards.py merge --plan <plan-root>/plan.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
PLAN_KIND = "skyfall_sr_shard_plan"
REPORT_KIND = "skyfall_sr_shard_merge_report"
SUMMARY_FILENAME = "prepare_scene_zoom_summary.json"
MANIFEST_FILENAME = "supervision.json"
REPORT_FILENAME = "merge-report.json"
PLAN_FILENAME = "plan.json"
JOBS_FILENAME = "jobs.json"

#: Portable runtime fields shared by every generated shard job.
PROTOCOL_PATHS = (
    "python",
    "repo_root",
    "output_dir",
    "source_path",
    "vlm_python",
    "vlm_model_path",
    "dloral_python",
    "dloral_root",
    "dloral_sd_path",
    "dloral_ckpt",
    "dloral_spynet",
)

#: Protocol paths that are model/script environments rather than the shared
#: dataset root; recorded with the plan so merge stays protocol-independent.
ENV_PATHS = (
    "vlm_python",
    "vlm_model_path",
    "dloral_python",
    "dloral_root",
    "dloral_sd_path",
    "dloral_ckpt",
    "dloral_spynet",
)

#: Path fields of a manifest view entry that legitimately differ between
#: shards (each shard collects its own base evidence) as long as the bytes are
#: identical: replay-only views are collected fresh by every shard.
VIEW_CONTENT_PATHS = ("image_path", "mask_path")


_REPO = str(Path(__file__).resolve().parents[1])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from refinement.scene_zoom import (  # noqa: E402
    FLOWEDIT_KIND,
    PROGRESSIVE_RGB_CONTRACT,
    REAL_REPLAY_STAGE,
    SCENE_ZOOM_KIND,
)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _load_json(path: str | os.PathLike[str]) -> Any:
    with open(os.path.abspath(os.fspath(path)), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str | os.PathLike[str], payload: Any) -> str:
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    os.replace(temporary, path)
    return path


@lru_cache(maxsize=8192)
def _file_digest(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_sha256(path: str) -> str:
    """Content digest of one artifact; never called on multi-GB checkpoints."""

    absolute = os.path.abspath(path)
    stat = os.stat(absolute)
    return _file_digest(absolute, stat.st_size, stat.st_mtime_ns)


def _real(path: Any) -> str | None:
    return None if path is None else os.path.realpath(os.path.abspath(str(path)))


def _same_path(left: Any, right: Any) -> bool:
    if not left or not right:
        return not left and not right
    return _real(left) == _real(right)


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object.")
    return value


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def _protocol_string(protocol: Mapping[str, Any], key: str) -> str | None:
    """First string value for ``key`` in the protocol, at any nesting depth."""

    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                found.add(value)
            for item in node.values():
                walk(item)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(protocol)
    if len(found) > 1:
        raise ValueError(f"protocol declares conflicting {key!r} paths: {sorted(found)}")
    return next(iter(found)) if found else None


def _protocol_path(protocol: Mapping[str, Any], key: str) -> str:
    value = _protocol_string(protocol, key)
    if value is None:
        raise ValueError(f"protocol has no {key!r} path.")
    return os.path.abspath(value)

def _manifest_view_ids(manifest_paths: Sequence[str]) -> list[str]:
    """FlowEdit view ids in collection order; exactly ``collect_flowedit_views``."""

    view_ids: list[str] = []
    seen: set[str] = set()
    for manifest_path in manifest_paths:
        payload = _load_json(manifest_path)
        if payload.get("kind") != FLOWEDIT_KIND:
            raise ValueError(
                f"Refusing non-FlowEdit manifest {manifest_path}: "
                f"kind={payload.get('kind')!r} (expected {FLOWEDIT_KIND!r})."
            )
        if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError(f"Unsupported flowedit schema_version in {manifest_path}.")
        entries = payload.get("views")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"FlowEdit manifest {manifest_path} has no views.")
        for entry in entries:
            view_id = str(entry["id"])
            if view_id in seen:
                raise ValueError(f"Duplicate flowedit view id across manifests: {view_id}")
            seen.add(view_id)
            view_ids.append(view_id)
    return view_ids


def _shard_command(
    *,
    python: str,
    repo_root: str,
    checkpoint: str,
    source_path: str,
    flowedit_manifests: Sequence[str],
    zoom: float,
    output_dir: str,
    view_ids: Sequence[str],
    parent_lod_checkpoint: str | None,
    previous_supervision: str | None,
    env: Mapping[str, str],
) -> list[str]:
    command = [
        python, "-u",
        os.path.join(repo_root, "scripts", "prepare_scene_zoom.py"),
        "--start_checkpoint", checkpoint,
        "--source_path", source_path,
        "--include_real_replay",
        "--progressive",
        "--step_scale", "2",
        "--resolution", "0",
        "--seed", "0",
        "--vlm_python", env["vlm_python"],
        "--vlm_model_path", env["vlm_model_path"],
        "--dloral_python", env["dloral_python"],
        "--dloral_root", env["dloral_root"],
        "--dloral_sd_path", env["dloral_sd_path"],
        "--dloral_ckpt", env["dloral_ckpt"],
        "--dloral_spynet", env["dloral_spynet"],
        "--dloral_alignment", "geometry",
        "--zoom_factors", f"{zoom:g}",
        "--output_dir", output_dir,
    ]
    command.extend(["--flowedit_manifests", *flowedit_manifests])
    if parent_lod_checkpoint:
        command.extend(["--parent_lod_checkpoint", parent_lod_checkpoint])
    if previous_supervision:
        command.extend(["--previous_supervision", previous_supervision])
    command.append("--target_view_ids")
    command.extend(str(view_id) for view_id in view_ids)
    return command


def plan_shards(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = os.path.abspath(args.protocol)
    protocol = _require_mapping(_load_json(protocol_path), protocol_path)
    resolved = {key: _protocol_path(protocol, key) for key in PROTOCOL_PATHS}
    repo_root = resolved["repo_root"]
    python = resolved["python"]
    source_path = resolved["source_path"]
    env = {key: resolved[key] for key in ENV_PATHS}
    checkpoint = os.path.abspath(args.checkpoint)
    zoom = float(args.zoom)
    output_dir = os.path.abspath(args.output_dir)
    views_per_shard = int(args.views_per_shard)
    if views_per_shard < 1:
        raise ValueError("--views-per-shard must be >= 1.")
    flowedit_manifests = [os.path.abspath(path) for path in args.flowedit_manifests]
    if not flowedit_manifests:
        raise ValueError("--flowedit-manifests needs at least one manifest.")
    parent_lod_checkpoint = (
        os.path.abspath(args.parent_lod_checkpoint) if args.parent_lod_checkpoint else None
    )
    previous_supervision = (
        os.path.abspath(args.previous_supervision) if args.previous_supervision else None
    )
    if zoom >= 4.0:
        if not parent_lod_checkpoint:
            raise ValueError("--parent-lod-checkpoint is mandatory for 4x supervision.")
        if not previous_supervision:
            raise ValueError("--previous-supervision is mandatory for 4x supervision.")
    view_ids = _manifest_view_ids(flowedit_manifests)
    tiles = int(round(zoom)) ** 2

    shards: list[dict[str, Any]] = []
    for start in range(0, len(view_ids), views_per_shard):
        chunk = view_ids[start : start + views_per_shard]
        shard_id = f"z{zoom:g}_s{len(shards):04d}"
        shard_dir = os.path.join(output_dir, shard_id)
        command = _shard_command(
            python=python,
            repo_root=repo_root,
            checkpoint=checkpoint,
            source_path=source_path,
            flowedit_manifests=flowedit_manifests,
            zoom=zoom,
            output_dir=shard_dir,
            view_ids=chunk,
            parent_lod_checkpoint=parent_lod_checkpoint,
            previous_supervision=previous_supervision,
            env=env,
        )
        shards.append({
            "id": shard_id,
            "output_dir": shard_dir,
            "views": list(chunk),
            "expected_samples": len(chunk) * tiles,
            "command": command,
            "expected_outputs": [
                os.path.join(shard_dir, MANIFEST_FILENAME),
                os.path.join(shard_dir, SUMMARY_FILENAME),
            ],
        })

    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": PLAN_KIND,
        "protocol": protocol_path,
        "protocol_output_dir": resolved["output_dir"],
        "repo_root": repo_root,
        "python": python,
        "source_path": source_path,
        "checkpoint": checkpoint,
        "zoom": zoom,
        "step_scale": 2.0,
        "resolution": 0,
        "seed": 0,
        "dloral_alignment": "geometry",
        "expected_tiles_per_view": tiles,
        "expected_samples_total": len(view_ids) * tiles,
        "views_per_shard": views_per_shard,
        "flowedit_manifests": flowedit_manifests,
        "parent_lod_checkpoint": parent_lod_checkpoint,
        "previous_supervision": previous_supervision,
        "env": env,
        "view_ids": view_ids,
        "output_dir": output_dir,
        "shards": shards,
    }
    plan_path = _write_json(os.path.join(output_dir, PLAN_FILENAME), plan)
    jobs_path = _write_json(
        os.path.join(output_dir, JOBS_FILENAME),
        [
            {
                "id": shard["id"],
                "command": shard["command"],
                "expected_outputs": shard["expected_outputs"],
            }
            for shard in shards
        ],
    )
    print(
        f"[sr-shards] planned {len(shards)} shard(s) x {views_per_shard} view(s) "
        f"({len(view_ids)} view(s), {tiles} teacher(s) per view, zoom {zoom:g}); "
        f"plan: {plan_path}; jobs: {jobs_path}"
    )
    return plan


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def _expected_sample_id(view_id: str, zoom: float, row: int, col: int) -> str:
    return f"{view_id}__z{zoom:g}__r{row}c{col}"


def _check_summary(
    summary: Mapping[str, Any],
    *,
    shard: Mapping[str, Any],
    manifest_path: str,
    checkpoint: str,
    zoom: float,
    tiles: int,
    parent_lod_checkpoint: str | None,
    previous_supervision: str | None,
    planned_view_count: int,
) -> None:
    shard_id = shard["id"]
    shard_dir = os.path.abspath(str(shard["output_dir"]))
    shard_views = [str(view_id) for view_id in shard["views"]]

    def fail(message: str) -> None:
        raise ValueError(f"shard {shard_id}: {message}")

    if summary.get("stage") != "stage2" or summary.get("progressive") is not True:
        fail("summary is not a progressive Stage 2 run.")
    if not _same_path(summary.get("start_checkpoint"), checkpoint):
        fail("summary base checkpoint differs from the plan checkpoint.")
    if [float(value) for value in summary.get("zoom_factors", ())] != [zoom]:
        fail(f"summary zoom factors {summary.get('zoom_factors')!r} are not [{zoom:g}].")
    if summary.get("dloral_alignment") != "geometry":
        fail("summary alignment is not 'geometry'.")
    if summary.get("include_real_replay") is not True:
        fail("summary did not collect real replay views.")
    if not _same_path(summary.get("output_dir"), shard_dir):
        fail("summary output_dir differs from the plan shard directory.")
    if not _same_path(summary.get("supervision_manifest"), manifest_path):
        fail("summary supervision_manifest differs from the shard manifest path.")
    if sorted(str(view_id) for view_id in summary.get("target_view_ids", ())) != sorted(shard_views):
        fail("summary target_view_ids differ from the planned shard views.")
    collected = int(summary.get("flowedit_views", -1))
    replay = int(summary.get("real_replay_views", -1))
    if collected != int(planned_view_count) or replay < 0:
        fail("summary did not collect every planned FlowEdit view.")
    if int(summary.get("views_collected", -1)) != collected + replay:
        fail("summary collected view accounting is inconsistent.")
    if int(summary.get("views_pending", -1)) != len(shard_views):
        fail("summary pending view count differs from the planned shard views.")
    if int(summary.get("views_skipped_by_max_views", -1)) != 0:
        fail("summary skipped views via --max_views.")
    if int(summary.get("views_skipped_by_target_selection", -1)) != collected - len(shard_views):
        fail("summary target selection accounting is inconsistent with the plan.")
    generated = int(summary.get("samples_generated", -1))
    reused = int(summary.get("samples_reused", -1))
    if generated < 0 or reused < 0 or generated + reused != len(shard_views) * tiles:
        fail(
            "summary generated/reused targets do not cover the shard "
            f"({generated} + {reused} != {len(shard_views) * tiles})."
        )
    if int(summary.get("geometry_neighbor_count", 0)) < 1:
        fail("summary ran without geometry neighbors.")
    if not _same_path(summary.get("parent_lod_checkpoint"), parent_lod_checkpoint):
        fail("summary parent checkpoint differs from the plan.")
    if not _same_path(summary.get("previous_supervision"), previous_supervision):
        fail("summary previous supervision differs from the plan.")


def _merge_view_entries(view_id: str, entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One manifest view entry for every shard's copy of ``view_id``."""

    base = dict(entries[0])
    for other in entries[1:]:
        for key in sorted(set(base) | set(other)):
            left, right = base.get(key), other.get(key)
            if key == "image_path" and (not left or not right):
                raise ValueError(f"View {view_id!r} has no base image path.")
            if key not in VIEW_CONTENT_PATHS:
                if left != right:
                    raise ValueError(f"Conflicting view entry {view_id!r}: field {key!r} differs.")
                continue
            if left == right:
                if left and not os.path.isfile(str(left)):
                    raise ValueError(f"View {view_id!r} base evidence is missing: {left!r}")
                continue
            for path in (left, right):
                if not path or not os.path.isfile(str(path)):
                    raise ValueError(f"View {view_id!r} base evidence is missing: {path!r}")
            if _content_sha256(str(left)) != _content_sha256(str(right)):
                raise ValueError(
                    f"Conflicting view entry {view_id!r}: {key!r} content differs "
                    f"({left} vs {right})."
                )
    return base


def merge_shards(plan_path: str) -> dict[str, Any]:
    plan_path = os.path.abspath(plan_path)
    plan = _require_mapping(_load_json(plan_path), plan_path)
    if plan.get("kind") != PLAN_KIND or int(plan.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"{plan_path} is not a version-{SCHEMA_VERSION} {PLAN_KIND} plan.")

    output_dir = os.path.abspath(str(plan["output_dir"]))
    checkpoint = os.path.abspath(str(plan["checkpoint"]))
    zoom = float(plan["zoom"])
    tiles = int(plan["expected_tiles_per_view"])
    grid = int(round(zoom))
    planned_ids = [str(view_id) for view_id in plan["view_ids"]]
    planned_set = set(planned_ids)
    shards = list(plan.get("shards", ()))
    if not shards:
        raise ValueError("plan has no shards.")
    parent_lod_checkpoint = plan.get("parent_lod_checkpoint")
    previous_path = plan.get("previous_supervision")

    previous_samples: dict[str, Mapping[str, Any]] = {}
    previous_zooms: list[float] = []
    if previous_path:
        previous = _require_mapping(_load_json(str(previous_path)), str(previous_path))
        if previous.get("kind") != SCENE_ZOOM_KIND or int(previous.get("schema_version", 0)) != 1:
            raise ValueError(f"{previous_path} is not a version-1 {SCENE_ZOOM_KIND} manifest.")
        if not _same_path(previous.get("base_checkpoint"), checkpoint):
            raise ValueError("Previous supervision belongs to a different base checkpoint.")
        for level in previous.get("levels", ()):
            level_zoom = float(level["zoom_factor"])
            if level_zoom == zoom:
                raise ValueError(
                    f"Previous supervision already contains zoom {zoom:g}; refusing stale parents."
                )
            previous_zooms.append(level_zoom)
            for sample in level.get("samples", ()):
                sample_id = str(sample["sample_id"])
                if sample_id in previous_samples:
                    raise ValueError(f"Previous supervision repeats sample id {sample_id!r}.")
                previous_samples[sample_id] = sample
        if not previous_samples:
            raise ValueError("Previous supervision has no samples to carry.")
    carried_needed = set(previous_samples)

    progressive_payload: Mapping[str, Any] | None = None
    view_entries: dict[str, list[Mapping[str, Any]]] = {}
    view_order: list[str] | None = None
    replay_ids: set[str] = set()
    base_checkpoint_values: set[str] = set()
    merged_planned: dict[str, Mapping[str, Any]] = {}
    merged_carried: dict[str, Mapping[str, Any]] = {}
    shard_records: list[dict[str, Any]] = []

    for shard in shards:
        shard_id = str(shard["id"])
        shard_dir = os.path.abspath(str(shard["output_dir"]))
        shard_views = [str(view_id) for view_id in shard["views"]]
        shard_manifest_path = os.path.join(shard_dir, MANIFEST_FILENAME)
        shard_summary_path = os.path.join(shard_dir, SUMMARY_FILENAME)

        def fail(message: str) -> None:
            raise ValueError(f"shard {shard_id}: {message}")

        for path in (shard_manifest_path, shard_summary_path):
            if not os.path.isfile(path):
                fail(f"missing {path}")
        manifest = _require_mapping(_load_json(shard_manifest_path), shard_manifest_path)
        summary = _require_mapping(_load_json(shard_summary_path), shard_summary_path)
        _check_summary(
            summary,
            shard=shard,
            manifest_path=shard_manifest_path,
            checkpoint=checkpoint,
            zoom=zoom,
            tiles=tiles,
            parent_lod_checkpoint=parent_lod_checkpoint,
            previous_supervision=previous_path,
            planned_view_count=len(planned_ids),
        )

        if manifest.get("kind") != SCENE_ZOOM_KIND or int(manifest.get("schema_version", 0)) != 1:
            fail(f"manifest is not a version-1 {SCENE_ZOOM_KIND} manifest.")
        recorded_base = manifest.get("base_checkpoint")
        if not _same_path(recorded_base, checkpoint):
            fail("manifest base checkpoint differs from the plan checkpoint.")
        base_checkpoint_values.add(str(recorded_base))
        progressive = _require_mapping(
            manifest.get("progressive"), f"manifest {shard_manifest_path} progressive"
        )
        if progressive.get("enabled") is not True:
            fail("manifest is not progressive.")
        if progressive.get("render_rgb_contract") != PROGRESSIVE_RGB_CONTRACT:
            fail(f"manifest render RGB contract is not {PROGRESSIVE_RGB_CONTRACT!r}.")
        recorded_parent = progressive.get("parent_lod_checkpoint")
        recorded_parent_path = (
            recorded_parent.get("path") if isinstance(recorded_parent, Mapping) else None
        )
        if not _same_path(recorded_parent_path, parent_lod_checkpoint):
            fail("manifest parent checkpoint differs from the plan.")
        if progressive_payload is None:
            progressive_payload = progressive
        elif progressive != progressive_payload:
            fail("parent checkpoint/appearance or progressive contract differs between shards.")

        entries = manifest.get("views")
        if not isinstance(entries, list) or not entries:
            fail("manifest has no views.")
        shard_view_ids = [str(entry["id"]) for entry in entries]
        if len(set(shard_view_ids)) != len(shard_view_ids):
            fail("manifest repeats a view id.")
        missing_views = sorted(set(shard_views) - set(shard_view_ids))
        if missing_views:
            fail(f"manifest does not list planned view(s) {missing_views}.")
        replay_here = {
            str(entry["id"])
            for entry in entries
            if str(entry.get("source_stage")) == REAL_REPLAY_STAGE
        }
        if not replay_here:
            fail("manifest has no native real replay views.")
        if int(summary.get("real_replay_views", -1)) != len(replay_here):
            fail("summary real replay view count differs from the manifest.")
        if view_order is None:
            view_order = list(shard_view_ids)
            replay_ids = set(replay_here)
        else:
            if set(shard_view_ids) != set(view_order):
                fail("manifest view ids differ from the other shards.")
            if replay_here != replay_ids:
                fail("real replay collection differs from the other shards.")
        for entry in entries:
            view_id = str(entry["id"])
            view_entries.setdefault(view_id, []).append(dict(entry))

        shard_planned = 0
        shard_carried = 0
        seen_carried: set[str] = set()
        for level in manifest.get("levels", ()):
            level_zoom = float(level["zoom_factor"])
            samples = level.get("samples")
            if not isinstance(samples, list):
                fail(f"level {level_zoom:g} has no sample list.")
            if level_zoom == zoom:
                coverage: dict[str, set[tuple[int, int]]] = {
                    view_id: set() for view_id in shard_views
                }
                for sample in samples:
                    view_id = str(sample["view_id"])
                    if view_id not in coverage:
                        fail(
                            f"unplanned sample {sample.get('sample_id')!r} for view {view_id!r}."
                        )
                    row, col = int(sample["tile_row"]), int(sample["tile_col"])
                    if not (0 <= row < grid and 0 <= col < grid):
                        fail(f"sample {sample['sample_id']!r} lies outside the {zoom:g}x grid.")
                    if str(sample["sample_id"]) != _expected_sample_id(view_id, zoom, row, col):
                        fail(f"sample id {sample['sample_id']!r} does not match its tile.")
                    if (row, col) in coverage[view_id]:
                        fail(f"sample {sample['sample_id']!r} is duplicated inside the shard.")
                    if float(sample["zoom_factor"]) != zoom:
                        fail(f"sample {sample['sample_id']!r} has a mismatched zoom factor.")
                    coverage[view_id].add((row, col))
                    sample_id = str(sample["sample_id"])
                    if sample_id in merged_planned:
                        fail(
                            f"teacher {sample_id!r} is already planned by another shard "
                            "(overlapping shard plans)."
                        )
                    merged_planned[sample_id] = sample
                    shard_planned += 1
                for view_id, cells in coverage.items():
                    if len(cells) != tiles:
                        fail(
                            f"view {view_id!r} has {len(cells)} of {tiles} expected "
                            f"{zoom:g}x teachers."
                        )
            elif previous_samples and level_zoom in previous_zooms:
                for sample in samples:
                    sample_id = str(sample["sample_id"])
                    reference = previous_samples.get(sample_id)
                    if reference is None:
                        fail(f"unplanned carried sample {sample_id!r}.")
                    if dict(sample) != dict(reference):
                        fail(f"carried sample {sample_id!r} differs from previous supervision.")
                    seen_carried.add(sample_id)
                    merged_carried[sample_id] = sample
                    shard_carried += 1
            else:
                fail(f"unplanned zoom level {level_zoom:g}.")

        if seen_carried != carried_needed:
            fail(
                f"carries {len(seen_carried)} of {len(carried_needed)} previous-level "
                "sample(s); the carried level must be complete."
            )
        shard_records.append({
            "id": shard_id,
            "output_dir": shard_dir,
            "manifest": shard_manifest_path,
            "summary": shard_summary_path,
            "views": len(shard_views),
            "planned_samples": shard_planned,
            "carried_samples": shard_carried,
        })

    if len(base_checkpoint_values) != 1:
        raise ValueError("Shard manifests disagree on the base checkpoint.")
    assert view_order is not None

    missing_views = sorted(planned_set - set(view_entries))
    if missing_views:
        raise ValueError(f"Planned view(s) never appeared in a manifest: {missing_views}")
    unplanned_views = sorted(set(view_entries) - planned_set - replay_ids)
    if unplanned_views:
        raise ValueError(f"Manifests contain unplanned view(s): {unplanned_views}")

    expected_planned = len(planned_ids) * tiles
    if len(merged_planned) != expected_planned:
        raise ValueError(
            f"Merged teachers cover {len(merged_planned)} of {expected_planned} planned targets."
        )
    if carried_needed and set(merged_carried) != carried_needed:
        raise ValueError("Merged carried samples are incomplete.")
    if not carried_needed and merged_carried:
        raise ValueError("Merged manifest carries samples without a previous supervision.")

    missing_paths: list[str] = []
    checked_paths: set[str] = set()
    for sample in list(merged_planned.values()) + list(merged_carried.values()):
        for key in ("image_path", "lr_image_path"):
            path = sample.get(key)
            if not path or str(path) in checked_paths:
                continue
            checked_paths.add(str(path))
            if not os.path.isfile(str(path)):
                missing_paths.append(str(path))
    if missing_paths:
        raise ValueError(
            f"{len(missing_paths)} teacher/anchor path(s) are missing, e.g. {missing_paths[:3]}"
        )

    merged_views = [
        _merge_view_entries(view_id, view_entries[view_id])
        for view_id in list(planned_ids) + [v for v in view_order if v not in planned_set]
    ]
    order = {str(view["id"]): index for index, view in enumerate(merged_views)}

    def sort_samples(samples: Mapping[str, Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        return sorted(
            samples.values(),
            key=lambda item: (
                order.get(str(item["view_id"]), -1),
                int(item["tile_row"]),
                int(item["tile_col"]),
            ),
        )

    levels = [
        {"zoom_factor": level_zoom, "samples": sort_samples(merged_carried)}
        for level_zoom in sorted(set(previous_zooms))
    ] + [{"zoom_factor": zoom, "samples": sort_samples(merged_planned)}]

    manifest_payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": SCENE_ZOOM_KIND,
        "base_checkpoint": sorted(base_checkpoint_values)[0],
        "views": merged_views,
        "levels": levels,
    }
    if progressive_payload is not None:
        manifest_payload["progressive"] = dict(progressive_payload)
    manifest_path = _write_json(os.path.join(output_dir, MANIFEST_FILENAME), manifest_payload)
    teacher_identity = {
        sample["sample_id"]: {
            "image_path": sample["image_path"],
            "image_sha256": _content_sha256(sample["image_path"]),
            "lr_image_path": sample.get("lr_image_path"),
            "lr_image_sha256": _content_sha256(sample["lr_image_path"])
            if sample.get("lr_image_path") else None,
        }
        for level in levels for sample in level["samples"]
    }
    identity_path = _write_json(
        os.path.join(output_dir, "teacher-identity.json"),
        {"schema_version": 1, "samples": teacher_identity},
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "plan": plan_path,
        "output_dir": output_dir,
        "supervision": manifest_path,
        "teacher_identity": identity_path,
        "checkpoint": checkpoint,
        "zoom": zoom,
        "expected_tiles_per_view": tiles,
        "parent_lod_checkpoint": parent_lod_checkpoint,
        "previous_supervision": previous_path,
        "views": {
            "planned": len(planned_ids),
            "real_replay": len(replay_ids),
            "total": len(merged_views),
        },
        "teachers": {
            "zooms": [float(level["zoom_factor"]) for level in levels],
            "planned_level": len(merged_planned),
            "planned_expected": expected_planned,
            "carried_level": len(merged_carried),
            "checked_paths": len(checked_paths),
        },
        "shards": shard_records,
    }
    report_path = _write_json(os.path.join(output_dir, REPORT_FILENAME), report)
    print(
        f"[sr-shards] merged {len(shard_records)} shard(s): {len(merged_planned)} new "
        f"teacher(s) at zoom {zoom:g}, {len(merged_carried)} carried, "
        f"{len(merged_views)} view(s); manifest: {manifest_path}; report: {report_path}"
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sharded plan/merge CLI for prepare_scene_zoom supervision.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser(
        "plan",
        help="Write plan.json plus jobs.json for one sharded supervision level.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    plan_parser.add_argument("--protocol", required=True, help="Run protocol JSON.")
    plan_parser.add_argument("--checkpoint", required=True,
                             help="Stage 2 (FlowEdit) checkpoint shared by every shard.")
    plan_parser.add_argument("--zoom", type=int, choices=(2, 4), required=True,
                             help="Tile-grid zoom factor of this level.")
    plan_parser.add_argument("--output-dir", required=True,
                             help="Plan root; one separate output directory per shard.")
    plan_parser.add_argument("--flowedit-manifests", nargs="+", required=True,
                             help="One or more skyfall_flowedit_views manifests; every shard "
                                  "receives all of them (the neighbor pool stays global).")
    plan_parser.add_argument("--views-per-shard", type=int, required=True,
                             help="Planned FlowEdit views per shard job.")
    plan_parser.add_argument("--parent-lod-checkpoint", default=None,
                             help="Trained G(t-1) LoD checkpoint; mandatory for 4x.")
    plan_parser.add_argument("--previous-supervision", default=None,
                             help="Merged lower-level supervision.json carried verbatim; "
                                  "mandatory for 4x.")

    merge_parser = subparsers.add_parser(
        "merge",
        help="Verify every shard and fold them into one supervision manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    merge_parser.add_argument("--plan", required=True, help="plan.json written by 'plan'.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        plan_shards(args)
    else:
        merge_shards(args.plan)
    return 0


if __name__ == "__main__":
    sys.exit(main())
