#!/usr/bin/env python3
"""Selective Skyfall-GS result archival worker (GaussianZoom Stage1/Stage2).

This worker owns, for every scene/stage of the 12-scene pipeline described by
``<archive_root>/pipeline_manifest.json``:

* the *selected* file manifest (final model artifacts + metrics + provenance),
* the rsync transfer (resumable, compressed; no dense per-episode scratch),
* SHA256 verification of every selected file on *both* ends,
* the atomic ``<archive_root>/<scene>/<stage>/archive_status.json`` publish
  (``status: verified``) -- this file is written by this worker only,
* gated remote cleanup with JSON receipts,
* a recurring watcher over remote ``stage_complete.json`` markers.

Retention contract (see pipeline_manifest.json -> retention):

* keep per stage: final checkpoint + *matching* point_cloud PLY (the checkpoint
  stores learned appearance embeddings/MLP, the PLY stores ``filter_3D``; both
  are required, no lossy conversion is invented here), cfg_args, cameras.json,
  run arguments/provenance, metrics, compact console log, representative
  visual comparison, plus small prompt/camera/episode metadata,
* never transfer: intermediate checkpoints/PLY, geometry flow/depth/alpha
  scratch, DLoRAL diagnostic tensors, generation cache, temporary MoGe data,
  redundant generated samples, raw TensorBoard image events,
* remote final model deletion requires BOTH a local SHA256-verified archive and
  a local ``evaluation_status.json`` with ``model_load_verified: true``.

Cleanup never touches legacy experiment *parents* (old LoD assets) without an
explicit ``cleanup_approval.json`` signed by the main agent.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

SCHEMA_VERSION = 1
ARCHIVE_STATUS_NAME = "archive_status.json"
ARCHIVE_MANIFEST_NAME = "archive_manifest.json"
RECEIPTS_DIR = "archive_receipts"
WORKER_STATE_NAME = "archive_worker_state.json"
CLEANUP_APPROVAL_NAME = "cleanup_approval.json"
CLEANUP_POLICY_NAME = "cleanup_policy.json"
WORKER_LOCK_NAME = "archive_worker.lock"
FAILURES_NAME = "archive_failures.json"
NOTIFICATIONS_DIR = "notifications"
STAGES = ("stage1", "stage2")
POINT_CLOUD_MARKER = "point_cloud"
SSH_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=6",
    "-o", "StrictHostKeyChecking=yes",
]
# Hard safety net: never let a selection accidentally transfer these trees.
FORBIDDEN_PATH_PARTS = (
    "/geometry/", "/generation_cache/", "/dloral/", "/render/", "/render_after_train/",
    "/render_depth/", "/render_refine/", "/steps/", "/cross_view/",
    "/correspondence/", "/depth_tmp/", "/smoke/", "/smoke_optimized/",
)
# Scratch directories that legitimately hold a few small metadata files: only
# the listed categories may be selected from inside them.
SCRATCH_METADATA_RULES = {
    "/prompt_cache/": {"prompt_metadata"},
}
FORBIDDEN_SUFFIXES = (".pth.appearance.pt", ".lod.pt")


# --------------------------------------------------------------------------- #
# generic helpers
# --------------------------------------------------------------------------- #
def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` as JSON atomically (tmp file + fsync + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=False, sort_keys=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _jsonable(value):
    """Drop NaN/Inf so manifests stay strictly valid JSON."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_files_parallel(root: Path, rels: Sequence[str], workers: int = 4) -> dict[str, str]:
    def one(rel: str) -> tuple[str, str]:
        return rel, sha256_file(root / rel)

    if not rels:
        return {}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(rels)))) as pool:
        return dict(pool.map(one, rels))


def human_bytes(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f}{unit}"
        value /= 1024
    return f"{value:.2f}TiB"


def safe_rel(path: str) -> str:
    if not path or path.startswith("/") or "\n" in path or ".." in Path(path).parts:
        raise ValueError(f"unsafe relative path: {path!r}")
    return path


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
class Manifest:
    def __init__(self, root: Path):
        self.path = root / "pipeline_manifest.json"
        if not self.path.is_file():
            raise SystemExit(f"missing pipeline manifest: {self.path}")
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        self.archive_root = Path(self.data.get("archive_root") or root)
        self.host: str = self.data["remote_host"]
        self.run_root: str = self.data["remote_run_root"].rstrip("/")
        self.remote_project: str = self.data["remote_project"].rstrip("/")
        self.local_dataset_root = Path(self.data["local_dataset_root"])
        self.training = self.data["training"]

    def scene(self, name: str) -> dict:
        for entry in self.data["scenes"]:
            if entry["scene"] == name:
                return entry
        raise SystemExit(f"scene {name} not in pipeline manifest")

    def scene_names(self) -> list[str]:
        return [entry["scene"] for entry in self.data["scenes"]]

    def stage_iteration(self, stage: str) -> int:
        if stage == "stage1":
            return int(self.training["stage1_iterations"])
        return int(self.training["final_iteration"])

    def remote_dataset_dir(self, scene: str) -> str:
        entry = self.scene(scene)
        return f"{self.remote_project}/data/{entry['dataset']}/{scene}"

    def local_dataset_dir(self, scene: str) -> Path:
        entry = self.scene(scene)
        return self.local_dataset_root / entry["dataset"] / scene

    def remote_scene_root(self, scene: str) -> str:
        return f"{self.run_root}/{scene}"

    def remote_marker(self, scene: str, stage: str) -> str:
        return f"{self.remote_scene_root(scene)}/{stage}/stage_complete.json"

    def archive_dir(self, scene: str, stage: str) -> Path:
        return self.archive_root / scene / stage

    def evaluation_marker(self, scene: str, stage: str) -> Path:
        return self.archive_root / scene / "evaluation" / stage / "evaluation_status.json"


# --------------------------------------------------------------------------- #
# selection spec
# --------------------------------------------------------------------------- #
def stage_dir_patterns(stage: str, iteration: int) -> list[tuple[str, str, bool]]:
    """Files inside the stage output directory itself. -> (pattern, category, required)"""
    if stage == "stage1":
        return [
            (f"chkpnt{iteration}.pth", "final_checkpoint", True),
            (f"{POINT_CLOUD_MARKER}/iteration_{iteration}/point_cloud.ply", "final_point_cloud", True),
            ("cfg_args", "config", True),
            ("cameras.json", "config", True),
            ("run_status.json", "status", False),
            ("command.json", "run_arguments", False),
            ("train.log", "console_log", False),
            ("compact_retention.json", "retention_record", False),
            ("console.log", "console_log", False),
            ("metrics.json", "metrics", False),
            ("source.sha256", "source_checksum", False),
            ("source_*.sha256", "source_checksum", False),
            ("input.ply", "input_provenance", False),
            ("experiment_summary.json", "metrics", False),
            ("episode_comparison.jpg", "visual_comparison", False),
        ]
    return [
        (f"chkpnt{iteration}.pth", "final_checkpoint", True),
        (f"{POINT_CLOUD_MARKER}/iteration_{iteration}/point_cloud.ply", "final_point_cloud", True),
        ("cfg_args", "config", True),
        ("cameras.json", "config", True),
        ("run_status.json", "status", False),
        ("command.json", "run_arguments", False),
        ("train.log", "console_log", False),
        ("compact_retention.json", "retention_record", False),
        ("console.log", "console_log", False),
        ("metrics.json", "metrics", False),
        ("source.sha256", "source_checksum", False),
        ("source_*.sha256", "source_checksum", False),
        ("input.ply", "input_provenance", False),
        ("experiment_summary.json", "metrics", False),
        ("episode_comparison.jpg", "visual_comparison", False),
        ("idu/manifest.json", "episode_metadata", False),
        ("idu/index.html", "episode_metadata", False),
        ("idu/*/episode_meta.json", "episode_metadata", False),
        ("idu/*/episode_summary.json", "episode_metrics", False),
        ("idu/*/compare.html", "episode_metadata", False),
        ("idu/*/stage2_prepared.json", "episode_metadata", False),
        ("idu/*/stage2_synthesis.json", "episode_metadata", False),
        ("idu/*/panels/*", "visual_comparison", False),
        ("idu/*/retained_views/*", "visual_comparison", False),
        ("idu/*/context/prompt_*.json", "prompt_metadata", False),
        ("idu/*/prompt_cache/*.json", "prompt_metadata", False),
    ]


def scene_provenance_patterns() -> list[tuple[str, str, bool]]:
    """Small prior-run provenance kept beside the stage dir (explains reuse)."""
    base = [
        ("stage1-command.json", "stage1_provenance", False),
        ("stage1-stage-log.json", "stage1_provenance", False),
        ("RESULTS.txt", "prior_results", False),
        ("results.json", "prior_results", False),
        ("protocol.json", "protocol", False),
        ("logs/*.log", "compact_log", False),
        ("visual-previews/*", "visual_comparison", False),
        ("evaluation/heldout_metrics.csv", "prior_metrics", False),
        ("evaluation/heldout_metrics.json", "prior_metrics", False),
        ("evaluation/no_reference_metrics.csv", "prior_metrics", False),
        ("evaluation/no_reference_metrics.json", "prior_metrics", False),
        ("evaluation/protocol.json", "protocol", False),
        ("evaluation/summary.json", "prior_metrics", False),
        ("chain/*.json", "chain_provenance", False),
        ("chain/recover/recover.json", "chain_provenance", False),
        ("run_stage1.py", "stage1_provenance", False),
        ("prepare_stage1.py", "stage1_provenance", False),
        ("prepare_metrics.sh", "stage1_provenance", False),
        ("evaluate_run.py", "stage1_provenance", False),
        ("record_results.py", "stage1_provenance", False),
        ("run_chain.py", "stage1_provenance", False),
    ]
    for level in ("l1", "l2"):
        for name in (
            "lineage.json", "prompt.json", "freeze.json", "geometry.json", "cross_view.json",
            "correspondence.json", "densify_log.json", "absorption_curve.json",
            "dloral_result.json", "train_lineage.json", "lod_l2_summary.json",
        ):
            base.append((f"chain/{level}/{name}", "chain_provenance", False))
    return base


def resolve_stage_source(remote: Remote, manifest: Manifest, scene: str, stage: str) -> tuple[str, str]:
    """Pick the remote source directory for a stage.

    Precedence:
    1. The run's own published ``stage_complete.json`` (trainer-owned). It names
       ``output_dir``, which after compaction holds exactly the retained final
       set; archiving from it guarantees the archive matches the trainer's
       frozen manifest.
    2. ``reuse_stage1``/``reuse_stage2`` from the pipeline manifest (legacy runs
       qualified before the canonical run root existed).
    3. ``<remote_run_root>/<scene>/<stage>``.
    """
    marker = remote.read_json(manifest.remote_marker(scene, stage))
    if marker and marker.get("status") == "completed" and marker.get("output_dir"):
        return marker["output_dir"].rstrip("/"), "published_marker"
    entry = manifest.scene(scene)
    reuse_key = "reuse_stage1" if stage == "stage1" else "reuse_stage2"
    if entry.get(reuse_key):
        return entry[reuse_key].rstrip("/"), "manifest_reuse_path"
    return f"{manifest.remote_scene_root(scene)}/{stage}", "run_root"


def build_selection(manifest: Manifest, scene: str, stage: str,
                    remote: Remote | None = None) -> list[dict]:
    """Ordered rsync groups: (name, source dir, archive subdir, patterns)."""
    iteration = manifest.stage_iteration(stage)
    stage_patterns = stage_dir_patterns(stage, iteration)
    provenance = scene_provenance_patterns()
    if remote is not None:
        stage_source, origin = resolve_stage_source(remote, manifest, scene, stage)
    else:
        entry = manifest.scene(scene)
        reuse_key = "reuse_stage1" if stage == "stage1" else "reuse_stage2"
        if entry.get(reuse_key):
            stage_source, origin = entry[reuse_key].rstrip("/"), "manifest_reuse_path"
        else:
            stage_source, origin = f"{manifest.remote_scene_root(scene)}/{stage}", "run_root"
    legacy = origin == "manifest_reuse_path"
    groups: list[dict] = [{
        "name": stage,
        "source": stage_source,
        "source_origin": origin,
        "dest_rel": "",
        "legacy_source": legacy,
        "patterns": stage_patterns,
    }]
    # Small prior-run provenance that explains the reused/qualified run.  It is
    # resolved independently of the stage dir: a run whose stage output was
    # re-published under the canonical run root still keeps its provenance in
    # the original experiment directory.
    entry = manifest.scene(scene)
    reuse_key = "reuse_stage1" if stage == "stage1" else "reuse_stage2"
    provenance_source = None
    if entry.get(reuse_key):
        provenance_source = os.path.dirname(entry[reuse_key].rstrip("/"))
    elif origin == "published_marker" and os.path.dirname(stage_source) != manifest.remote_scene_root(scene):
        provenance_source = os.path.dirname(stage_source)
    if provenance_source and provenance_source != stage_source:
        groups.append({
            "name": "run_provenance",
            "source": provenance_source,
            "source_origin": "sibling_run_dir",
            "dest_rel": "run_provenance",
            "legacy_source": True,
            "patterns": provenance,
        })
    # Prompt/metadata that only the original run directory still holds (the
    # trainer's compaction of a republished stage drops episode scratch into the
    # legacy origin while the canonical run root keeps the retained set).
    if stage == "stage2" and entry.get(reuse_key):
        overlay_source = entry[reuse_key].rstrip("/")
        if overlay_source != stage_source:
            groups.append({
                "name": "legacy_prompt_metadata",
                "source": overlay_source,
                "source_origin": "sibling_run_dir",
                "dest_rel": "",
                "legacy_source": True,
                "patterns": [
                    ("idu/*/prompt_cache/*.json", "prompt_metadata", False),
                    ("idu/*/context/prompt_*.json", "prompt_metadata", False),
                ],
            })
    return groups


def category_of(rel: str, patterns: Sequence[tuple[str, str, bool]]) -> str:
    best = None
    for pattern, category, _required in patterns:
        if "*" in pattern or "?" in pattern:
            if fnmatch.fnmatch(rel, pattern):
                best = best or category
        elif rel == pattern:
            return category
    return best or "other"


# --------------------------------------------------------------------------- #
# remote / rsync
# --------------------------------------------------------------------------- #
class Remote:
    def __init__(self, host: str):
        self.host = host

    def run(self, script: str, *, input_bytes: bytes | None = None, check: bool = True,
            timeout: float | None = None) -> subprocess.CompletedProcess:
        cmd = ["ssh", *SSH_OPTS, self.host, script]
        result = subprocess.run(cmd, input=input_bytes, capture_output=True, timeout=timeout)
        if check and result.returncode != 0:
            raise SystemExit(
                f"remote command failed ({result.returncode}): {script}\n"
                f"stderr: {result.stderr.decode('utf-8', 'replace')[-4000:]}"
            )
        return result

    def stat_exists(self, path: str) -> bool:
        return self.run(f"test -e {shlex.quote(path)}", check=False).returncode == 0

    def read_json(self, path: str) -> dict | None:
        result = self.run(f"cat {shlex.quote(path)}", check=False)
        if result.returncode != 0:
            return None
        try:
            return json.loads(result.stdout.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            return None

    def sha256(self, source: str, rels: Sequence[str], chunk: int = 400) -> dict[str, str]:
        out: dict[str, str] = {}
        for start in range(0, len(rels), chunk):
            batch = [safe_rel(rel) for rel in rels[start:start + chunk]]
            if not batch:
                continue
            script = f"cd {shlex.quote(source)} && tr '\\n' '\\0' | xargs -0 -r sha256sum"
            result = self.run(script, input_bytes=("\n".join(batch) + "\n").encode("utf-8"))
            for line in result.stdout.decode("utf-8", "replace").splitlines():
                if not line.strip():
                    continue
                digest, _, name = line.partition("  ")
                out[name.strip()] = digest.strip().lower()
        return out


def rsync_remote_arg(host: str, source: str, directory: bool = True) -> str:
    path = source.rstrip("/") + "/" if directory else source
    return f"{host}:{shlex.quote(path)}"


def rsync_filter_args(patterns: Sequence[str]) -> list[str]:
    args = ["--include=*/"]
    args += [f"--include={pattern}" for pattern in patterns]
    args.append("--exclude=*")
    return args


_LIST_LINE = re.compile(
    r"^(?P<kind>[dlcbps-])(?P<perm>\S*)\s+(?P<size>[\d,]+)\s+"
    r"(?P<date>\d{4}/\d{2}/\d{2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+(?P<name>.+?)$"
)


def rsync_list(remote: Remote, source: str, patterns: Sequence[str]) -> dict[str, int]:
    """List the regular files that the transfer filters would select.

    ``rsync --list-only`` ignores ``--out-format`` and prints the long listing
    format, so parse that (and tolerate tabular variants if a future rsync does
    honour ``--out-format``).  An absent source directory yields an empty
    selection: for the 11 not-yet-trained scenes nothing exists yet, which is a
    normal state for planning, not an error.
    """
    if not remote.stat_exists(source):
        return {}
    cmd = [
        "rsync", "-r", "--list-only",
        *rsync_filter_args(patterns),
        "-e", "ssh " + " ".join(SSH_OPTS),
        rsync_remote_arg(remote.host, source),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise SystemExit(
            f"rsync --list-only failed for {source}:\n{result.stderr.decode('utf-8', 'replace')[-4000:]}"
        )
    listing: dict[str, int] = {}
    for line in result.stdout.decode("utf-8", "replace").splitlines():
        line = line.rstrip("\n")
        if not line or line.startswith("receiving ") or line.startswith("sending "):
            continue
        if "\t" in line:
            size_text, name = line.split("\t", 1)
            name = name.strip()
            if name and not name.endswith("/"):
                listing[safe_rel(name)] = int(size_text.replace(",", "").strip() or 0)
            continue
        match = _LIST_LINE.match(line)
        if not match or match.group("kind") == "d":
            continue
        name = match.group("name").strip()
        if not name or name.endswith("/"):
            continue
        listing[safe_rel(name)] = int(match.group("size").replace(",", ""))
    return listing


def rsync_transfer(remote: Remote, source: str, destination: Path, patterns: Sequence[str]) -> str:
    destination.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync", "-a", "--no-owner", "--no-group", "--compress", "--compress-level=6",
        "--partial", "--inplace", "--prune-empty-dirs", "--stats", "--timeout=600",
        *rsync_filter_args(patterns),
        "-e", "ssh " + " ".join(SSH_OPTS),
        rsync_remote_arg(remote.host, source),
        str(destination) + "/",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise SystemExit(
            f"rsync failed for {source} -> {destination}:\n"
            f"{result.stderr.decode('utf-8', 'replace')[-4000:]}"
        )
    return result.stdout.decode("utf-8", "replace")


def local_file_sizes(root: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            sizes[str(path.relative_to(root))] = path.stat().st_size
    return sizes


# --------------------------------------------------------------------------- #
# readiness
# --------------------------------------------------------------------------- #
def marker_state(remote: Remote, manifest: Manifest, scene: str, stage: str,
                 groups: Sequence[dict]) -> dict:
    state: dict = {"marker_path": manifest.remote_marker(scene, stage), "marker_present": False}
    legacy = bool(groups[0].get("legacy_source"))
    marker = remote.read_json(state["marker_path"])
    if marker is not None:
        state.update({
            "marker_present": True,
            "marker_status": marker.get("status"),
            "marker_iteration": marker.get("iteration"),
        })
    stage_dir = groups[0]["source"]
    run_status = remote.read_json(f"{stage_dir}/run_status.json")
    state["legacy_source"] = legacy
    state["source_dir"] = stage_dir
    if run_status is not None:
        state["run_status"] = run_status.get("status")
        state["run_status_exit_code"] = run_status.get("exit_code")
    missing = []
    for pattern, _category, required in groups[0]["patterns"]:
        if not required:
            continue
        if not remote.stat_exists(f"{stage_dir}/{pattern}"):
            missing.append(pattern)
    state["missing_required"] = missing
    marker_ok = (not state["marker_present"]) or state.get("marker_status") == "completed"
    run_ok = state.get("run_status") in (None, "completed")
    state["ready"] = marker_ok and run_ok and not missing
    return state


# --------------------------------------------------------------------------- #
# dataset compatibility
# --------------------------------------------------------------------------- #
def dataset_inventory(remote: Remote, remote_dir: str) -> dict[str, int]:
    script = (
        f"cd {shlex.quote(remote_dir)} && "
        "find . -type f -printf '%P\\t%s\\n' | LC_ALL=C sort"
    )
    result = remote.run(script)
    inventory: dict[str, int] = {}
    for line in result.stdout.decode("utf-8", "replace").splitlines():
        if "\t" not in line:
            continue
        name, size = line.rsplit("\t", 1)
        inventory[name] = int(size)
    return inventory


def dataset_digests(remote: Remote, remote_dir: str, rels: Sequence[str]) -> dict[str, str]:
    if not rels:
        return {}
    return remote.sha256(remote_dir, list(rels))


def local_dataset_cache_path(manifest: Manifest, scene: str) -> Path:
    return manifest.archive_root / scene / "dataset_checksums.json"


def local_dataset_digests(manifest: Manifest, scene: str, local_dir: Path,
                          rels: Sequence[str]) -> tuple[dict[str, str], bool]:
    """SHA256 of local dataset files, memoized per scene (the datasets are the
    shared training inputs, so re-hashing ~200MB per stage is wasted work).

    Returns ``(digests, cache_hit)``.
    """
    cache_path = local_dataset_cache_path(manifest, scene)
    cached = _read_local_json(cache_path) or {}
    stored = {entry["path"]: entry["sha256"] for entry in cached.get("files", [])}
    inventory = {str(path.relative_to(local_dir)): path.stat().st_size
                 for path in sorted(local_dir.rglob("*")) if path.is_file()}
    reusable = all(
        name in stored and cached.get("sizes", {}).get(name) == size
        for name, size in inventory.items()
    )
    if reusable and set(rels) <= set(stored):
        return {name: stored[name] for name in rels}, True
    digests = sha256_files_parallel(local_dir, sorted(inventory), workers=4)
    atomic_write_json(cache_path, {
        "schema_version": SCHEMA_VERSION,
        "scene": scene,
        "local_dir": str(local_dir),
        "algorithm": "sha256",
        "hashed_at": utcnow(),
        "sizes": inventory,
        "files": [{"path": name, "bytes": inventory[name], "sha256": digests[name]}
                  for name in sorted(digests)],
    })
    return {name: digests[name] for name in rels}, False


def dataset_compatibility(remote: Remote, manifest: Manifest, scene: str, *, deep: bool = False) -> dict:
    remote_dir = manifest.remote_dataset_dir(scene)
    local_dir = manifest.local_dataset_dir(scene)
    report: dict = {
        "scene": scene,
        "remote_dir": remote_dir,
        "local_dir": str(local_dir),
        "local_reused": True,
        "checked_at": utcnow(),
    }
    if not local_dir.is_dir():
        report.update({"status": "missing_local_dataset"})
        return report
    remote_files = dataset_inventory(remote, remote_dir)
    local_files = local_file_sizes(local_dir)
    only_remote = sorted(set(remote_files) - set(local_files))
    only_local = sorted(set(local_files) - set(remote_files))
    size_mismatch = sorted(
        name for name in set(remote_files) & set(local_files)
        if remote_files[name] != local_files[name]
    )
    report.update({
        "files_remote": len(remote_files),
        "files_local": len(local_files),
        "only_remote": only_remote,
        "only_local": only_local,
        "size_mismatch": size_mismatch,
        "bytes": sum(local_files.values()),
    })
    if deep:
        shared = sorted(set(remote_files) & set(local_files))
        remote_digests = dataset_digests(remote, remote_dir, shared)
        local_digests, cache_hit = local_dataset_digests(manifest, scene, local_dir, shared)
        digest_mismatch = sorted(
            name for name in shared
            if remote_digests.get(name) != local_digests.get(name)
        )
        report.update({
            "deep_verified": True,
            "files_digest_compared": len(shared),
            "digest_mismatch": digest_mismatch,
            "local_digest_cache_hit": cache_hit,
            "local_checksum_cache": str(local_dataset_cache_path(manifest, scene)),
        })
    else:
        report["deep_verified"] = False
    report["status"] = "identical" if not (only_remote or only_local or size_mismatch) and (
        report.get("deep_verified") is False or not report.get("digest_mismatch")
    ) else "divergent"
    return report


# --------------------------------------------------------------------------- #
# derived receipts
# --------------------------------------------------------------------------- #
def _read_local_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def receipts_path(archive_dir: Path, name: str) -> Path:
    """Derived receipts live in a dedicated subtree so they can never collide
    with a file transferred from the run directory."""
    directory = archive_dir / RECEIPTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def is_receipt(rel: str) -> bool:
    return rel == RECEIPTS_DIR or rel.startswith(RECEIPTS_DIR + "/") or rel.startswith(
        "." + RECEIPTS_DIR + ".tmp")


DERIVED_FILES = (ARCHIVE_STATUS_NAME, ARCHIVE_MANIFEST_NAME, "stage_complete.json",
                 "source.sha256.json")


def is_derived(rel: str) -> bool:
    """Files this worker generates next to (not from) the transferred selection."""
    return (is_receipt(rel) or rel in DERIVED_FILES or rel.startswith(".")
            or rel.startswith(".rsync-partial"))


def write_command_record(archive_dir: Path, stage: str, source_root: str, remote: Remote) -> dict | None:
    if stage == "stage2":
        run_status = remote.read_json(f"{source_root}/run_status.json")
        if not run_status:
            return None
        payload = {
            "schema_version": SCHEMA_VERSION,
            "scene_stage": "stage2",
            "origin": f"{source_root}/run_status.json",
            "started_at": run_status.get("started_at"),
            "parameters": run_status.get("parameters", {}),
        }
        log_head = remote.run(
            f"head -1 {shlex.quote(source_root + '/console.log')}", check=False).stdout.decode(
                "utf-8", "replace").strip()
        if log_head:
            payload["command_line"] = log_head
        atomic_write_json(receipts_path(archive_dir, "command.json"), payload)
        return payload
    command = remote.read_json(f"{source_root}/stage1-command.json")
    stage_log = remote.read_json(f"{source_root}/stage1-stage-log.json")
    if not (command or stage_log):
        run_command = remote.read_json(f"{source_root}/command.json")
        run_status = remote.read_json(f"{source_root}/run_status.json")
        if not (run_command or run_status):
            return None
        payload = {
            "schema_version": SCHEMA_VERSION,
            "scene_stage": "stage1",
            "origin": f"{source_root}/command.json",
            "cmd": (run_command or {}).get("cmd") or (run_command or {}).get("command", []),
            "note": (run_command or {}).get("note"),
            "environment_overrides": (run_command or {}).get("environment_overrides", {}),
            "run_status_parameters": (run_status or {}).get("parameters", {}),
        }
        atomic_write_json(receipts_path(archive_dir, "command.json"), payload)
        return payload
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scene_stage": "stage1",
        "origin": f"{source_root}/stage1-command.json",
        "cmd": (command or {}).get("cmd", []),
        "environment_overrides": (command or {}).get("environment_overrides", {}),
        "stage_log": stage_log,
    }
    atomic_write_json(receipts_path(archive_dir, "command.json"), payload)
    return payload


def write_source_checksum(archive_dir: Path, stage: str, source_root: str, remote: Remote,
                          transferred: set[str]) -> dict:
    """Freeze the source-code checksums that produced the run, when captured.

    The run-provided ``source_*.sha256`` files are transferred verbatim; this
    receipt records which of them belongs to the final run and whether the run
    captured any checksums at all (the legacy stage1 run predates capture).
    """
    provided = sorted(rel for rel in transferred if os.path.basename(rel).startswith("source_")
                      and rel.endswith(".sha256"))
    if not provided:
        payload = {
            "status": "unavailable",
            "reason": (
                "the run directory captured no source_*.sha256; the exact command and environment are "
                "frozen in archive_receipts/command.json and run_provenance/stage1-command.json"
            ) if stage == "stage1" else "the run directory captured no source_*.sha256",
            "origin": f"{source_root}/source_*.sha256",
        }
        atomic_write_json(receipts_path(archive_dir, "source_checksum.json"), payload)
        return payload
    latest_rel = provided[-1]
    matches = sorted(archive_dir.rglob(os.path.basename(latest_rel)))
    latest_path = matches[0] if matches else None
    payload = {
        "status": "captured",
        "run_dir": source_root,
        "captures": provided,
        "final_capture": latest_rel,
        "final_capture_content": latest_path.read_text(encoding="utf-8", errors="replace")
        if latest_path is not None else None,
        "note": "checksums of the repository sources used by the final run, as captured by the run itself",
    }
    atomic_write_json(receipts_path(archive_dir, "source_checksum.json"), payload)
    return payload


def write_stage_complete(archive_dir: Path, manifest: Manifest, scene: str, stage: str,
                         remote: Remote, marker: dict, archive_status: dict,
                         transferred: set[str]) -> dict:
    iteration = manifest.stage_iteration(stage)
    source_dir = marker["source_dir"]
    checkpoint = f"chkpnt{iteration}.pth"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "scene": scene,
        "stage": stage,
        "iteration": iteration,
        "output_dir": source_dir,
        "checkpoint": f"{source_dir}/{checkpoint}",
        "point_cloud": f"{source_dir}/{POINT_CLOUD_MARKER}/iteration_{iteration}/point_cloud.ply",
        "source_path": manifest.remote_dataset_dir(scene),
        "origin": (
            "run stage_complete.json" if "stage_complete.json" in transferred
            else "derived_from_run_status.json"
        ),
        "remote_marker_present": bool(marker.get("marker_present")),
        "run_status": marker.get("run_status"),
        "run_status_exit_code": marker.get("run_status_exit_code"),
        "archive_status": ARCHIVE_STATUS_NAME,
        "archive_verified": archive_status.get("status") == "verified",
        "created_at": utcnow(),
        "derivation": (
            "the run's own stage_complete.json was transferred and digest-verified against the source"
            if "stage_complete.json" in transferred
            else "synthesized by the archive worker from the remote run_status.json (status=completed); "
                 "this legacy run published no stage_complete.json marker"
        ),
    }
    if "stage_complete.json" in transferred:
        # keep the run's own marker at the stage root untouched
        atomic_write_json(receipts_path(archive_dir, "marker_check.json"), payload)
        return payload
    atomic_write_json(archive_dir / "stage_complete.json", payload)
    return payload


def reconcile_published_marker(manifest: Manifest, remote: Remote, scene: str, stage: str,
                               status: dict) -> dict:
    """Cross-check the trainer's published marker digests against our archive.

    The trainer owns ``stage_complete.json``; this receipt is read-only evidence
    that the bytes it declares are exactly the bytes we archived, so a later
    remote deletion cannot desynchronize the two records.
    """
    marker = remote.read_json(manifest.remote_marker(scene, stage))
    archive_dir = manifest.archive_dir(scene, stage)
    local = {entry["path"]: entry for entry in status.get("files", [])}
    declared = []
    if marker:
        for artifact in marker.get("artifacts", []):
            rel = os.path.relpath(artifact.get("path", ""), marker.get("output_dir", "/")) \
                if artifact.get("path") else None
            entry = local.get(rel) if rel else None
            declared.append({
                "role": artifact.get("role"),
                "remote_path": artifact.get("path"),
                "remote_sha256": artifact.get("sha256"),
                "remote_bytes": artifact.get("bytes"),
                "archive_path": entry["path"] if entry else None,
                "archive_sha256": entry["sha256"] if entry else None,
                "matches": bool(entry and entry["sha256"] == artifact.get("sha256")),
            })
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scene": scene,
        "stage": stage,
        "marker_path": manifest.remote_marker(scene, stage),
        "marker_present": marker is not None,
        "marker_status": (marker or {}).get("status"),
        "marker_iteration": (marker or {}).get("iteration"),
        "marker_created_at": (marker or {}).get("created_at"),
        "marker_reused_from": (marker or {}).get("reused_from"),
        "marker_content": marker,
        "declared_artifacts": declared,
        "all_declared_match": bool(declared) and all(item["matches"] for item in declared),
        "note": ("the run's own stage_complete.json is read and preserved here verbatim; the stage root "
                 "carries this worker's derived stage_complete.json so the archive never mixes the "
                 "publisher's marker with archive-owned fields"),
        "checked_at": utcnow(),
    }
    atomic_write_json(receipts_path(archive_dir, "marker_reconciliation.json"), payload)
    return payload


def summarize_metrics(archive_dir: Path, stage: str) -> dict:
    """Small, honest metric summary; authoritative numbers live in evaluation/."""
    summary: dict = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "authoritative_evaluation": (
            f"../evaluation/{stage}/evaluation_status.json (DatasetEvaluator uniform protocol)"
        ),
    }
    experiment = _read_local_json(archive_dir / "experiment_summary.json")
    if experiment:
        summary["run_summary"] = experiment
    run_status = _read_local_json(archive_dir / "run_status.json")
    if run_status:
        summary["run_status"] = run_status.get("status")
        summary["exit_code"] = run_status.get("exit_code")
    if stage == "stage1":
        prior = _read_local_json(archive_dir / "run_provenance" / "evaluation" / "heldout_metrics.json")
        if prior:
            summary["self_eval_heldout_legacy_protocol"] = prior
        no_ref = _read_local_json(archive_dir / "run_provenance" / "evaluation" / "no_reference_metrics.json")
        if no_ref:
            summary["self_eval_no_reference_legacy_protocol"] = no_ref
    else:
        manifest_json = _read_local_json(archive_dir / "idu" / "manifest.json")
        if manifest_json:
            summary["episodes"] = manifest_json.get("episodes", [])
    atomic_write_json(receipts_path(archive_dir, "metrics.json"), summary)
    return summary


# --------------------------------------------------------------------------- #
# transfer + verify
# --------------------------------------------------------------------------- #
def assert_selectable(source: str, rel: str, patterns: Sequence[tuple[str, str, bool]]) -> None:
    """Refuse anything that is not part of the frozen retention contract.

    rsync's own filters already restrict the transfer, so this is defence in
    depth against a pattern typo silently pulling dense scratch trees.
    """
    probe = "/" + rel
    for part in FORBIDDEN_PATH_PARTS:
        if part in probe:
            raise SystemExit(f"selection contains forbidden scratch path: {source}/{rel}")
    for part, allowed in SCRATCH_METADATA_RULES.items():
        if part in probe and category_of(rel, patterns) not in allowed:
            raise SystemExit(
                f"selection pulls {category_of(rel, patterns)!r} from scratch dir {part}: {source}/{rel}")
    if rel.endswith(FORBIDDEN_SUFFIXES):
        raise SystemExit(f"selection contains non-archived model blob: {source}/{rel}")


def evaluation_gate(manifest: Manifest, scene: str, stage: str) -> dict:
    """Snapshot of the DatasetEvaluator gate used before any remote deletion.

    Per the scheduling agreement the evaluator may publish ``status: running``
    with ``model_load_verified: true`` as soon as the final checkpoint + matching
    PLY load and the first renders succeed; the cleanup gate therefore keys on
    ``model_load_verified``, not on ``status == completed``.
    """
    path = manifest.evaluation_marker(scene, stage)
    payload = _read_local_json(path) or {}
    verified = payload.get("model_load_verified") is True
    return {
        "reference_only": True,
        "evaluation_status_path": str(path),
        "evaluation_status_present": path.is_file(),
        "evaluation_status": payload.get("status"),
        "model_load_verified": payload.get("model_load_verified"),
        "satisfied": verified,
        "satisfied_by": "model_load_verified=true (running or completed)" if verified else None,
        "read_at": utcnow(),
    }


def run_transfer(manifest: Manifest, remote: Remote, scene: str, stage: str, *,
                 verify_only: bool = False, deep_compat: bool = True) -> dict:
    groups = build_selection(manifest, scene, stage, remote)
    archive_dir = manifest.archive_dir(scene, stage)
    archive_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_state(remote, manifest, scene, stage, groups)
    if not marker["ready"]:
        raise SystemExit(
            f"{scene}/{stage} not ready for archival: missing={marker['missing_required']} "
            f"marker={marker.get('marker_status')} run_status={marker.get('run_status')}"
        )
    previous = _read_local_json(archive_dir / ARCHIVE_STATUS_NAME) or {}
    reuse_frozen = verify_only
    prior_by_group: dict[str, dict[str, dict]] = {}
    frozen_by_group: dict[str, set[str]] = {}
    if reuse_frozen:
        for entry in previous.get("files", []):
            source = entry.get("source", "")
            for group in groups:
                if source.startswith(group["source"] + "/"):
                    prior_by_group.setdefault(group["name"], {})[
                        source[len(group["source"]) + 1:]] = entry
        for entry in previous.get("sources", []):
            if entry.get("frozen_files"):
                frozen_by_group[entry["group"]] = set(entry["frozen_files"])

    transfers = []
    listings: dict[str, dict[str, int]] = {}
    for group in groups:
        dest = archive_dir / group["dest_rel"] if group["dest_rel"] else archive_dir
        patterns = [pattern for pattern, _c, _r in group["patterns"]]
        listed = rsync_list(remote, group["source"], patterns)
        if not listed and not frozen_by_group.get(group["name"]):
            transfers.append({"group": group["name"], "source": group["source"],
                              "dest_rel": group["dest_rel"], "skipped": "no matching files",
                              "ok": True, "transferred": 0, "bytes": 0, "files": []})
            continue
        if not verify_only:
            rsync_transfer(remote, group["source"], dest, patterns)
        listings[group["name"]] = listed

    # Groups sharing a destination directory are verified together: the archive
    # holds their union, so comparing each group against the whole local subtree
    # would flag every sibling's files as unexpected.
    by_dest: dict[str, list[dict]] = {}
    for group in groups:
        if group["name"] in listings or frozen_by_group.get(group["name"]):
            by_dest.setdefault(group["dest_rel"], []).append(group)
    for dest_rel, dest_groups in by_dest.items():
        dest = archive_dir / dest_rel if dest_rel else archive_dir
        expected: dict[str, str] = {}
        for group in dest_groups:
            for pattern, _c, _r in group["patterns"]:
                for rel in listings.get(group["name"], {}):
                    if fnmatch.fnmatch(rel, pattern) or rel == pattern:
                        expected.setdefault(rel, group["name"])
            for rel in frozen_by_group.get(group["name"], set()):
                expected.setdefault(rel, group["name"])
        local_files = local_file_sizes(dest) if dest.is_dir() else {}
        frozen = {
            rel for rel in local_files
            if not is_derived(rel) and rel in expected
        }
        for rel in sorted(frozen):
            assert_selectable(dest.as_posix(), rel,
                              dest_groups[0]["patterns"] if len(dest_groups) == 1 else
                              [item for group in dest_groups for item in group["patterns"]])
        absent = sorted(rel for rel in expected if rel not in local_files)
        unexpected = sorted(rel for rel in frozen if rel not in expected)
        for group in dest_groups:
            own = {rel for rel, owner in expected.items() if owner == group["name"]}
            own_local = own & set(local_files)
            remote_digests = remote.sha256(group["source"], sorted(own & set(listings[group["name"]])))
            local_digests = sha256_files_parallel(dest, sorted(own_local), workers=4)
            mismatched = sorted(
                rel for rel in own_local
                if rel in remote_digests and local_digests.get(rel) != remote_digests[rel]
            )
            prior = prior_by_group.get(group["name"], {})
            remote_compacted = sorted(
                rel for rel in own if rel not in listings.get(group["name"], {}) and rel in prior)
            files = []
            for rel in sorted(own):
                files.append({
                    "path": f"{dest_rel}/{rel}" if dest_rel else rel,
                    "source": f"{group['source']}/{rel}",
                    "bytes": local_files.get(rel, 0),
                    "sha256": local_digests.get(rel),
                    "remote_sha256": remote_digests.get(rel, prior.get(rel, {}).get("remote_sha256")),
                    "remote_present": rel in listings.get(group["name"], {}),
                    "remote_compacted": rel in remote_compacted,
                    "category": category_of(rel, group["patterns"]),
                })
            required_missing = [
                pattern for pattern, _category, is_required in group["patterns"]
                if is_required and not any(entry["path"].endswith(pattern) for entry in files)
            ]
            result = {
                "group": group["name"],
                "source": group["source"],
                "dest_rel": group["dest_rel"],
                "remote_listed": len(listings.get(group["name"], {})),
                "transferred": len(own),
                "bytes": sum(local_files.get(rel, 0) for rel in own),
                "missing_local": [rel for rel in absent if rel in own],
                "absent_local": [rel for rel in absent if rel in own],
                "unexpected_local": unexpected,
                "remote_compacted": remote_compacted,
                "digest_mismatch": mismatched,
                "hash_missing": sorted(rel for rel in own_local if rel not in local_digests),
                "required_missing": sorted(set(required_missing)),
                "ok": not (absent or unexpected or mismatched or required_missing
                           or any(rel for rel in own_local if rel not in local_digests)),
                "frozen": sorted(own),
                "files": files,
            }
            transfers.append(result)
    all_files = [entry for result in transfers for entry in result.get("files", [])]
    ok = all(result.get("ok") for result in transfers)
    compatibility = dataset_compatibility(remote, manifest, scene, deep=deep_compat)
    iteration = manifest.stage_iteration(stage)
    transferred_rel = {entry["source"].split("/", 1)[-1] for entry in all_files}
    status = {
        "schema_version": SCHEMA_VERSION,
        "status": "verified" if ok else "failed",
        "scene": scene,
        "stage": stage,
        "iteration": iteration,
        "archive_dir": str(archive_dir),
        "remote_host": remote.host,
        "source_path": manifest.remote_dataset_dir(scene),
        "local_dataset_dir": str(manifest.local_dataset_dir(scene)),
        "sources": [{
            "group": result["group"], "source": result["source"], "dest_rel": result["dest_rel"],
            "transferred": result.get("transferred"), "bytes": result.get("bytes"),
            "ok": result.get("ok"), "digest_mismatch": result.get("digest_mismatch", []),
            "required_missing": result.get("required_missing", []),
            "missing_local": result.get("missing_local", []),
            "absent_local": result.get("absent_local", []),
            "unexpected_local": result.get("unexpected_local", []),
            "remote_compacted": result.get("remote_compacted", []),
            "hash_missing": result.get("hash_missing", []),
            "frozen_files": result.get("frozen", []),
        } for result in transfers if result.get("files")],
        "model_artifacts": {},
        "files": all_files,
        "totals": {"files": len(all_files), "bytes": sum(entry["bytes"] for entry in all_files)},
        "dataset_compatibility": compatibility,
        "marker": marker,
        "verification": {
            "method": "sha256 on sender and receiver; digests compared per file",
            "verified_at": utcnow(),
            "files_compared": len(all_files),
            "digest_mismatch": sorted(
                entry["path"] for entry in all_files
                if entry.get("sha256") != entry.get("remote_sha256")
            ),
        },
        "evaluation_gate_snapshot": evaluation_gate(manifest, scene, stage),
        "tool": {
            "script": "scripts/archive_dataset_results.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "created_at": previous.get("created_at") or utcnow(),
        "updated_at": utcnow(),
    }
    for entry in all_files:
        if entry["category"] == "final_checkpoint":
            status["model_artifacts"]["checkpoint"] = {
                "path": entry["path"], "bytes": entry["bytes"], "sha256": entry["sha256"],
                "iteration": iteration,
            }
        if entry["category"] == "final_point_cloud":
            status["model_artifacts"]["point_cloud"] = {
                "path": entry["path"], "bytes": entry["bytes"], "sha256": entry["sha256"],
                "iteration": iteration,
            }
    missing_artifacts = [key for key in ("checkpoint", "point_cloud")
                         if key not in status["model_artifacts"]]
    if missing_artifacts:
        status["status"] = "failed"
        status["verification"]["missing_model_artifacts"] = missing_artifacts
        ok = False
    if ok:
        stage_group = next(group for group in groups if group["name"] == stage)
        provenance_group = next((group for group in groups if group["name"] == "run_provenance"), None)
        provenance_root = provenance_group["source"] if provenance_group else stage_group["source"]
        write_command_record(archive_dir, stage, stage_group["source"] if stage == "stage2"
                             else provenance_root, remote)
        write_source_checksum(archive_dir, stage, stage_group["source"], remote, transferred_rel)
        summarize_metrics(archive_dir, stage)
        write_stage_complete(archive_dir, manifest, scene, stage, remote, marker, status, transferred_rel)
        status["marker_reconciliation"] = reconcile_published_marker(
            manifest, remote, scene, stage, status)
        frozen_manifest = {
            "schema_version": SCHEMA_VERSION,
            "scene": scene,
            "stage": stage,
            "iteration": iteration,
            "frozen_at": utcnow(),
            "sources": [{"group": result["group"], "source": result["source"],
                         "dest_rel": result["dest_rel"]} for result in transfers],
            "files": [
                {key: entry[key] for key in ("path", "source", "bytes", "sha256", "category")}
                for entry in all_files
            ],
        }
        atomic_write_json(archive_dir / ARCHIVE_MANIFEST_NAME, frozen_manifest)
    atomic_write_json(archive_dir / ARCHIVE_STATUS_NAME, status)
    write_scene_summary(manifest, scene)
    write_global_summary(manifest, manifest.scene_names())
    return status


# --------------------------------------------------------------------------- #
# cleanup
# --------------------------------------------------------------------------- #
def list_remote_tree(remote: Remote, root: str) -> dict[str, int]:
    listing = remote.run(
        f"cd {shlex.quote(root)} && find . -type f -printf '%P\\t%s\\n' | LC_ALL=C sort", check=False)
    entries: dict[str, int] = {}
    if listing.returncode != 0:
        return entries
    for line in listing.stdout.decode("utf-8", "replace").splitlines():
        if "\t" in line:
            name, size = line.rsplit("\t", 1)
            entries[name] = int(size)
    return entries


def dir_bytes(remote: Remote, path: str) -> int:
    result = remote.run(f"du -sb {shlex.quote(path)} 2>/dev/null | cut -f1", check=False)
    try:
        return int(result.stdout.decode().strip() or 0)
    except ValueError:
        return 0


def stage_cleanup_candidates(remote: Remote, manifest: Manifest, scene: str, stage: str) -> list[dict]:
    """Expendable remote artifacts inside one completed stage output directory.

    Never returns the final model artifacts; those are handled separately and
    gated on the local model-load verification.
    """
    groups = build_selection(manifest, scene, stage, remote)
    source_dir = groups[0]["source"]
    iteration = manifest.stage_iteration(stage)
    entries = list_remote_tree(remote, source_dir)
    candidates: list[dict] = []

    def add(name: str, size: int, reason: str) -> None:
        candidates.append({"path": f"{source_dir}/{name}", "rel": name, "bytes": size, "reason": reason})

    for name, size in entries.items():
        final_checkpoint = f"chkpnt{iteration}.pth"
        final_ply = f"{POINT_CLOUD_MARKER}/iteration_{iteration}/point_cloud.ply"
        if name == final_checkpoint or name == final_ply:
            continue
        if name.startswith("chkpnt") and name.endswith(".pth"):
            add(name, size, "intermediate checkpoint (final checkpoint archived)")
        elif name.startswith("events.out.tfevents"):
            add(name, size, "raw TensorBoard events (redundant image events)")
        elif name.startswith(f"{POINT_CLOUD_MARKER}/iteration_"):
            add(name, size, "intermediate point_cloud PLY (final PLY archived)")
        elif stage == "stage2" and name.startswith("idu/"):
            parts = name.split("/")
            if len(parts) >= 3 and parts[1] in {"geometry", "generation_cache", "dloral", "render",
                                                "render_after_train", "render_depth", "render_refine",
                                                "prompt_cache"}:
                add(name, size, "consumed per-episode dense flow/cache scratch")
            elif len(parts) == 3 and parts[1] == "context" and parts[2].startswith("view_"):
                add(name, size, "redundant per-episode context renders (prompt metadata archived)")
            elif len(parts) == 3 and parts[1] == "context" and parts[2].startswith("mask_"):
                add(name, size, "redundant per-episode context masks")
        elif stage == "stage2" and name.startswith("depth_tmp/"):
            add(name, size, "temporary MoGe depth scratch")
    return candidates


def identity_gate(manifest: Manifest, remote: Remote, scene: str, stage: str,
                  status: dict) -> dict:
    """Compare the identity of the bytes, not just the booleans.

    The cleanup gate must prove that the checkpoint/PLY the evaluator loaded are
    the same bytes the training side published: evaluation_status.json's
    ``checkpoint_sha256``/``point_cloud_sha256`` must equal the digests in the
    run-root ``stage_complete.json`` and in our own archive manifest.
    """
    evaluation = _read_local_json(manifest.evaluation_marker(scene, stage)) or {}
    marker = remote.read_json(manifest.remote_marker(scene, stage)) or {}
    declared = {}
    for artifact in marker.get("artifacts", []):
        role = artifact.get("role")
        if role in ("checkpoint", "point_cloud"):
            declared[role] = artifact.get("sha256")
    archived = {
        "checkpoint": (status.get("model_artifacts", {}).get("checkpoint") or {}).get("sha256"),
        "point_cloud": (status.get("model_artifacts", {}).get("point_cloud") or {}).get("sha256"),
    }
    checks = {}
    for role, key in (("checkpoint", "checkpoint_sha256"), ("point_cloud", "point_cloud_sha256")):
        evaluated = evaluation.get(key)
        checks[role] = {
            "evaluation_sha256": evaluated,
            "marker_sha256": declared.get(role),
            "archive_sha256": archived.get(role),
            "evaluation_matches_marker": bool(evaluated) and evaluated == declared.get(role),
            "evaluation_matches_archive": bool(evaluated) and evaluated == archived.get(role),
        }
    # The evaluator's own fingerprint repeats the digests it loaded, so a stale
    # marker refreshed against different bytes is caught even if a top-level
    # digest field were left behind.
    fingerprint = evaluation.get("protocol_fingerprint") or {}
    fingerprint_agrees = all(
        (fingerprint.get(role) or {}).get("sha256") in (None, archived.get(role))
        for role in ("checkpoint", "point_cloud")
    )
    required = all(item["evaluation_matches_marker"] and item["evaluation_matches_archive"]
                   for item in checks.values()) and fingerprint_agrees
    return {
        "required": "evaluation_status sha256 == run-root stage_complete.json sha256 == archive sha256",
        "protocol_fingerprint_digest": fingerprint.get("digest"),
        "protocol_fingerprint_agrees": fingerprint_agrees,
        "checks": checks,
        "declared_in_marker": declared,
        "archived": archived,
        "satisfied": required,
        "marker_present": bool(marker),
        "evaluation_present": bool(evaluation),
        "read_at": utcnow(),
    }


def scene_evaluation_gates(manifest: Manifest, scene: str) -> dict:
    return {
        "stage1": evaluation_gate(manifest, scene, "stage1"),
        "stage2": evaluation_gate(manifest, scene, "stage2"),
    }


def cleanup_gate_state(candidate: dict, verified: bool, model_ok: bool,
                       dependency: dict | None, scene_gates: dict) -> tuple[bool, list[str]]:
    """Evaluate a candidate's gate; return ``(allowed, missing_reasons)``.

    Gates are cumulative by design: an expendable-scratch candidate approved by
    Main additionally requires *both* stages of the scene to have loaded and
    rendered locally, so a scene's scratch is never reclaimed while one of its
    two models is still unverified.
    """
    gate = candidate.get("gate")
    missing: list[str] = []
    if gate in ("archive_verified", "model_load_verified", "model_load_verified_and_stage2_done"):
        if not verified:
            missing.append("archive_status=verified")
    if gate in ("model_load_verified", "model_load_verified_and_stage2_done"):
        if not model_ok:
            missing.append("evaluation model_load_verified=true")
    if gate == "model_load_verified_and_stage2_done" and not (
            dependency and dependency["satisfied"]):
        missing.append("stage2 stage_complete.json status=completed (stage1 assets are read "
                       "during stage2 episodes)")
    if gate == "scene_archives_verified_and_evaluated":
        if not verified:
            missing.append("archive_status=verified")
        for stage, state in sorted(scene_gates.items()):
            if not state["satisfied"]:
                missing.append(f"{stage} evaluation model_load_verified=true")
    if gate is None:
        return False, ["candidate declares no gate"]
    return not missing, missing


def approved_scratch_candidates(remote: Remote, approvals: dict, scene: str) -> list[dict]:
    """Expendable do_not_archive scratch inside explicitly approved roots.

    Each entry of ``approved_scratch`` in cleanup_approval.json declares a root,
    the retention categories that may be removed there, and the gates that must
    hold.  Everything else under the root is left alone, so an approval can never
    widen into "delete the directory".
    """
    out: list[dict] = []
    for entry in approvals.get("approved_scratch", []):
        if scene not in (entry.get("scenes") or [scene]):
            continue
        root = entry["path"].rstrip("/")
        entries = list_remote_tree(remote, root)
        for name, size in entries.items():
            category = scratch_category(name)
            if category is None or category not in set(entry.get("categories", [])):
                continue
            out.append({
                "path": f"{root}/{name}",
                "rel": name,
                "bytes": size,
                "reason": f"{entry.get('reason', 'approved expendable scratch')} [{category}]",
                "gate": "scene_archives_verified_and_evaluated",
                "requires": entry.get("requires", []),
                "approved": True,
                "scratch_root": root,
                "category": category,
            })
    return out


def scratch_category(rel: str) -> str | None:
    """Classify a file using the manifest's do_not_archive vocabulary."""
    if rel.startswith("chkpnt") and rel.endswith(".pth"):
        return "intermediate_checkpoint_ply"
    if rel.startswith("point_cloud/iteration_"):
        return "intermediate_checkpoint_ply"
    if rel.startswith("events.out.tfevents"):
        return "raw_tensorboard_image_events"
    if rel.startswith("depth_tmp/"):
        return "temporary_moge_data"
    if "/dloral/" in f"/{rel}" or rel.startswith("dloral/"):
        return "dloral_diagnostic_tensors"
    if "/generation_cache/" in f"/{rel}" or rel.startswith("generation_cache/"):
        return "generation_cache"
    if "/geometry/" in f"/{rel}" or rel.startswith("geometry/"):
        return "geometry_flow_scratch"
    if "/prompt_cache/" in f"/{rel}" or rel.startswith("prompt_cache/"):
        return "generation_cache"
    for prefix in ("render/", "render_after_train/", "render_depth/", "render_refine/"):
        if rel.startswith(prefix) or f"/{prefix}" in f"/{rel}":
            return "redundant_generated_samples"
    if "/context/" in f"/{rel}" and (rel.endswith(".png")):
        return "redundant_generated_samples"
    return None


def approved_extra_candidates(remote: Remote, approvals: dict, scene: str) -> list[dict]:
    """Explicitly approved deletions outside the stage output dir (signed off)."""
    out: list[dict] = []
    for entry in approvals.get("approved_paths", []):
        path = entry["path"] if isinstance(entry, dict) else entry
        reason = entry.get("reason", "approved by cleanup_approval.json") if isinstance(entry, dict) else "approved"
        required = entry.get("requires", "archive_verified") if isinstance(entry, dict) else "archive_verified"
        size = dir_bytes(remote, path)
        out.append({"path": path, "rel": os.path.basename(path.rstrip("/")), "bytes": size,
                    "reason": reason, "gate": required, "directory": True, "approved": True})
    return out


def load_cleanup_policy(manifest: Manifest) -> dict:
    """Worker-owned cleanup policy: which paths are deliberately protected.

    Kept separate from ``cleanup_approval.json`` (the main agent's sign-off file)
    so the two cannot be confused: policy states intent to *keep*, approval states
    permission to *remove*.
    """
    policy = _read_local_json(manifest.archive_root / CLEANUP_POLICY_NAME) or {}
    return {
        "protect_paths": [str(path).rstrip("/") for path in policy.get("protect_paths", [])],
        "protect_reason": policy.get("protect_reason"),
        "read_at": utcnow(),
        "source": str(manifest.archive_root / CLEANUP_POLICY_NAME),
    }


def canonical_final_candidates(manifest: Manifest, scene: str, stage: str, status: dict,
                               dependency: dict, source_dir: str) -> list[dict]:
    """The run-root final assets themselves; this worker owns their cleanup.

    DatasetTrainer runs with ``--no-reap``, so no second remote deletion path
    exists: every scene's run-root finals are gated here and removed here.
    """
    artifacts = status.get("model_artifacts", {})
    gate = "model_load_verified_and_stage2_done" if stage == "stage1" else "model_load_verified"
    out = []
    for key in ("checkpoint", "point_cloud"):
        artifact = artifacts.get(key)
        if not artifact:
            continue
        out.append({
            "path": f"{source_dir}/{artifact['path']}",
            "rel": artifact["path"],
            "bytes": artifact["bytes"],
            "reason": (f"run-root final {key}; this worker owns the cleanup because the trainer runs "
                       "with --no-reap"),
            "gate": gate,
            "dependency": dependency if stage == "stage1" else None,
            "owner": "ArchiveCurator",
            "canonical": True,
        })
    return out


def stage2_dependency_gate(manifest: Manifest, remote: Remote, scene: str) -> dict:
    """Stage1 final assets must stay remote until the same scene's stage2 ran.

    Stage2 reads and re-validates the stage1 path on every episode; deleting the
    stage1 checkpoint mid-stage2 breaks the run.  The conservative rule is
    therefore: stage1 final-asset deletion additionally requires
    ``<scene>/stage2/stage_complete.json`` with ``status: completed``.
    """
    marker_path = manifest.remote_marker(scene, "stage2")
    marker = remote.read_json(marker_path) or {}
    completed = marker.get("status") == "completed"
    same_dir = marker.get("output_dir", "").rstrip("/") == manifest.remote_scene_root(scene) + "/stage2"
    return {
        "requires": "stage2 stage_complete.json status=completed",
        "marker_path": marker_path,
        "marker_present": bool(marker),
        "marker_status": marker.get("status"),
        "marker_iteration": marker.get("iteration"),
        "marker_in_new_run_root": bool(marker) and same_dir,
        "satisfied": completed,
        "read_at": utcnow(),
    }


def filesystem_usage(remote: Remote, path: str) -> dict:
    result = remote.run(f"df -B1 --output=used,avail {shlex.quote(path)} | tail -1", check=False)
    fields = result.stdout.decode().split()
    try:
        return {"used_bytes": int(fields[0]), "avail_bytes": int(fields[1]), "probe": path}
    except (IndexError, ValueError):
        return {"used_bytes": None, "avail_bytes": None, "probe": path}


def link_info(remote: Remote, paths: Sequence[str]) -> dict[str, dict]:
    """Link count, inode and peer hard links for each path (empty dict if absent)."""
    if not paths:
        return {}
    script = (
        "for p in " + " ".join(shlex.quote(path) for path in paths) + "; do "
        "if [ -e \"$p\" ]; then "
        "printf '%s\\t%s\\t%s\\n' \"$p\" \"$(stat -c %h \"$p\")\" \"$(stat -c %i \"$p\")\"; "
        "else printf '%s\\tMISSING\\tMISSING\\n' \"$p\"; fi; done"
    )
    result = remote.run(script, check=False)
    out: dict[str, dict] = {}
    for line in result.stdout.decode().splitlines():
        if "\t" not in line:
            continue
        path, links, inode = line.split("\t", 2)
        out[path] = {"links": None if links == "MISSING" else int(links),
                     "inode": None if inode == "MISSING" else int(inode)}
    return out


def mirror_candidates(manifest: Manifest, scene: str, stage: str, status: dict,
                      source_dir: str, remote: Remote | None = None) -> list[dict]:
    """Duplicates of the archived files that live in the manifest's reuse path.

    The training side republished this run by hard-linking the qualified run's
    files into the canonical run root, so each archived byte exists twice on the
    same filesystem.  Deleting only one link reclaims nothing; both mirrors must
    go together, which is why they are enumerated here as an explicit pair.
    """
    entry = manifest.scene(scene)
    reuse_key = "reuse_stage1" if stage == "stage1" else "reuse_stage2"
    legacy_root = (entry.get(reuse_key) or "").rstrip("/")
    if not legacy_root or legacy_root == source_dir:
        return []
    out: list[dict] = []
    pending: list[dict] = []
    for record in status.get("files", []):
        source = record.get("source", "")
        if not source.startswith(source_dir + "/"):
            continue
        if record.get("category") not in {"final_checkpoint", "final_point_cloud"}:
            continue
        rel = source[len(source_dir) + 1:]
        pending.append({
            "path": f"{legacy_root}/{rel}",
            "rel": rel,
            "bytes": record.get("bytes", 0),
            "reason": (f"hard-linked duplicate of the archived {record['category']} in the legacy "
                       "origin dir; space is reclaimed only when both links are removed"),
            "gate": "model_load_verified_and_stage2_done" if stage == "stage1" else "model_load_verified",
            "mirror_of": source,
        })
    if remote is not None and pending:
        info = link_info(remote, [item["path"] for item in pending])
        for item in pending:
            record = info.get(item["path"], {})
            item["links_at_plan"] = record.get("links")
            item["inode"] = record.get("inode")
            item["peer_survives_here"] = item["mirror_of"]
            item["reclaimable_bytes_if_peer_kept"] = (0 if (record.get("links") or 0) > 1
                                                     else item["bytes"])
        # a mirror that is already gone is not a deletion candidate
        pending = [item for item in pending if item["links_at_plan"] is not None]
    out.extend(pending)
    return out


def legacy_final_candidates(manifest: Manifest, scene: str, stage: str, status: dict,
                           dependency: dict, source_dir: str,
                           remote: Remote | None = None) -> list[dict]:
    """Hard-link mirrors of the archived finals that live in the reuse origin.

    The trainer republishes a qualified run by hard-linking it into the run root,
    so each archived byte can exist twice on the same filesystem; only removing
    the legacy link while the run-root link survives frees nothing, but the legacy
    link is still this worker's to remove. Run-root finals themselves are produced
    by :func:`canonical_final_candidates` (this worker owns both, since the trainer
    runs with ``--no-reap``).
    """
    gate = "model_load_verified_and_stage2_done" if stage == "stage1" else "model_load_verified"
    out = []
    for candidate in mirror_candidates(manifest, scene, stage, status, source_dir, remote):
        candidate.update({"gate": gate,
                          "dependency": dependency if stage == "stage1" else None,
                          "owner": "ArchiveCurator"})
        out.append(candidate)
    return out


def legacy_parent_guard(manifest: Manifest, scene: str, stage: str, candidate: dict,
                        approvals: dict, allowed_roots: Sequence[str]) -> str | None:
    """Decide whether a deletion candidate is inside this worker's scope.

    Allowed without further sign-off:

    * the stage output directory itself (canonical run root), and
    * the manifest reuse root for *this* scene/stage (the legacy origin), i.e.
      the hard-linked mirror of the archived finals plus its leftover scratch.

    Additionally blocked: removing a whole legacy experiment parent, or any path
    outside the roots above, which needs an explicit ``cleanup_approval.json``
    entry from the main agent.
    """
    path = candidate["path"].rstrip("/")
    approved = {entry if isinstance(entry, str) else entry.get("path")
                for entry in approvals.get("approved_paths", [])}
    parents_approved = set(approvals.get("legacy_parents_approved", []))
    entry = manifest.scene(scene)
    reuse_key = "reuse_stage1" if stage == "stage1" else "reuse_stage2"
    reuse_root = (entry.get(reuse_key) or "").rstrip("/")
    if candidate.get("directory") or candidate.get("whole_parent"):
        if path not in parents_approved and path not in approved:
            return (f"whole-directory removal of {path} requires cleanup_approval.json "
                    "(legacy parents and their sub-trees are out of scope for automatic cleanup)")
    for root in list(allowed_roots) + ([reuse_root] if reuse_root else []) + sorted(approved):
        root_norm = root.rstrip("/")
        if root_norm and (path == root_norm or path.startswith(root_norm + "/")):
            return None
    for key in ("reuse_stage1", "reuse_stage2"):
        root = entry.get(key)
        if not root:
            continue
        parent_base = os.path.dirname(root.rstrip("/"))
        if path == parent_base or path.startswith(parent_base + "/"):
            if parent_base in parents_approved:
                return None
            return (f"inside legacy experiment parent {parent_base} but outside the archived reuse root; "
                    "requires cleanup_approval.json from Main")
    return f"path is outside every recognized cleanup root ({path}); requires cleanup_approval.json"


def run_cleanup(manifest: Manifest, remote: Remote, scene: str, stage: str, *,
                execute: bool = False, include_scratch: bool = True,
                include_final: bool = False) -> dict:
    archive_dir = manifest.archive_dir(scene, stage)
    status = _read_local_json(archive_dir / ARCHIVE_STATUS_NAME) or {}
    gate = evaluation_gate(manifest, scene, stage)
    verified = status.get("status") == "verified"
    model_ok = gate["satisfied"]
    approvals = _read_local_json(manifest.archive_root / CLEANUP_APPROVAL_NAME) or {}
    groups = build_selection(manifest, scene, stage, remote)
    source_dir = groups[0]["source"]
    candidates: list[dict] = []
    if include_scratch:
        for candidate in stage_cleanup_candidates(remote, manifest, scene, stage):
            candidate["gate"] = "archive_verified"
            candidates.append(candidate)
    candidates.extend(approved_scratch_candidates(remote, approvals, scene))
    candidates.extend(approved_extra_candidates(remote, approvals, scene))
    dependency = stage2_dependency_gate(manifest, remote, scene) if stage == "stage1" else None
    policy = load_cleanup_policy(manifest)
    protected = policy["protect_paths"]
    canonical: list[dict] = []
    if include_final:
        mine = legacy_final_candidates(manifest, scene, stage, status, dependency,
                                       source_dir, remote)
        # Run-root finals: this worker owns them, unless the policy protects the path.
        for candidate in canonical_final_candidates(manifest, scene, stage, status, dependency,
                                                   source_dir):
            if any(candidate["path"] == guard_path
                   or candidate["path"].startswith(guard_path + "/") for guard_path in protected):
                candidate["decision_hint"] = "retained"
                candidate["blocked_by"] = (f"protected by {policy['source']}"
                                           + (f": {policy['protect_reason']}" if policy["protect_reason"]
                                              else ""))
                canonical.append(candidate)
            else:
                candidates.append(candidate)
        # Legacy-origin mirrors stay this worker's scope as well.
        for candidate in mine:
            if any(candidate["path"] == guard_path
                   or candidate["path"].startswith(guard_path + "/") for guard_path in protected):
                canonical.append({**candidate, "blocked_by": f"protected by {policy['source']}"})
            else:
                candidates.append(candidate)
    scene_gates = scene_evaluation_gates(manifest, scene)
    identity = identity_gate(manifest, remote, scene, stage, status)
    decisions = []
    for candidate in candidates:
        if candidate.get("decision_hint") == "retained":
            decisions.append(candidate)
            continue
        guard = legacy_parent_guard(manifest, scene, stage, candidate, approvals, [source_dir])
        if guard:
            decisions.append({**candidate, "decision": "retained", "blocked_by": guard})
            continue
        gate_ok, missing = cleanup_gate_state(candidate, verified, model_ok, dependency, scene_gates)
        if gate_ok and candidate.get("gate", "").startswith("model_load_verified") \
                and not identity["satisfied"]:
            gate_ok = False
            missing.append("evaluation/model/archive sha256 identity check failed")
        elif gate_ok and candidate.get("gate", "").startswith("model_load_verified"):
            candidate.setdefault("identity_verified", True)
        decisions.append({**candidate, "decision": "delete" if gate_ok else "retained",
                          **({} if gate_ok else {"blocked_by": ", ".join(missing)})})
    removed, failed = [], []
    fs_before = filesystem_usage(remote, source_dir)
    if execute:
        for decision in decisions:
            if decision["decision"] != "delete":
                continue
            flag = "-r" if decision.get("directory") else ""
            path = shlex.quote(decision["path"])
            result = remote.run(
                f"if [ -e {path} ] || [ -L {path} ]; then "
                f"rm {flag} -- {path} && printf removed; fi", check=False)
            if result.returncode != 0:
                failed.append({"path": decision["path"], "stderr": result.stderr.decode("utf-8", "replace")})
            elif result.stdout.strip() == b"removed":
                removed.append(decision["path"])
            else:
                decision["decision"] = "absent"
        remote.run(f"cd {shlex.quote(source_dir)} && find . -type d -empty -delete", check=False)
    fs_after = filesystem_usage(remote, source_dir) if execute else fs_before
    removed_records = [d for d in decisions if d["decision"] == "delete" and d["path"] in removed]
    mirror_free = sum(d.get("reclaimable_bytes_if_peer_kept") or 0 for d in removed_records)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": "cleanup_receipt",
        "scene": scene,
        "stage": stage,
        "source_dir": source_dir,
        "executed": execute,
        "created_at": utcnow(),
        "gates": {
            "archive_status": status.get("status"),
            "archive_verified": verified,
            "archive_status_path": str(archive_dir / ARCHIVE_STATUS_NAME),
            "evaluation_model_load_verified": model_ok,
            "evaluation_status": gate["evaluation_status"],
            "evaluation_status_path": gate["evaluation_status_path"],
            "legacy_parents_approved": approvals.get("legacy_parents_approved", []),
            "approved_paths": [entry.get("path") for entry in approvals.get("approved_paths", [])],
            "stage2_dependency": dependency,
            "scene_evaluation_gates": scene_gates,
            "identity": identity,
            "cleanup_policy": policy,
        },
        "candidates": decisions,
        "canonical_finals_not_deleted_here": canonical,
        "planned_bytes": sum(d["bytes"] for d in decisions if d["decision"] == "delete"),
        "deleted": removed,
        "deleted_bytes": sum(d["bytes"] for d in decisions
                             if d["decision"] == "delete" and d["path"] in removed),
        "filesystem": {
            "before": fs_before,
            "after": fs_after,
            "delta_used_bytes": ((fs_after["used_bytes"] - fs_before["used_bytes"])
                                 if fs_before["used_bytes"] is not None
                                 and fs_after["used_bytes"] is not None else None),
            "note": ("df-based measurement; a removed hard-link mirror frees nothing until the peer "
                     "link is removed, which is why deleted_bytes can exceed the delta_used_bytes"),
            "mirror_bytes_not_yet_free": mirror_free,
        },
        "retained": [d["path"] for d in decisions if d["decision"] == "retained"],
        "failed": failed,
    }
    # An execution is appended to a cumulative ledger, so a later no-op execution
    # (or a dry-run) can never erase the record of what this worker removed. The
    # per-run copy is immutable; the plan is a separate artefact.
    if execute:
        atomic_write_json(
            receipts_path(archive_dir, f"cleanup_{utcnow().replace(':', '').replace('-', '')}.json"),
            receipt)
        ledger = _read_local_json(receipts_path(archive_dir, "cleanup_ledger.json")) or {
            "schema_version": SCHEMA_VERSION, "kind": "cleanup_ledger", "scene": scene,
            "stage": stage, "runs": [], "deleted": [], "deleted_bytes": 0,
        }
        prior = {entry["path"]: entry for entry in ledger["deleted"]}
        for record in removed_records:
            prior[record["path"]] = {
                "path": record["path"], "bytes": record["bytes"],
                "reason": record.get("reason"), "removed_at": receipt["created_at"],
                "mirror_of": record.get("mirror_of"),
            }
        ledger["runs"].append({
            "at": receipt["created_at"], "gates": receipt["gates"], "removed": removed,
            "planned": len([d for d in decisions if d["decision"] == "delete"]),
        })
        ledger["deleted"] = [prior[path] for path in sorted(prior)]
        ledger["deleted_bytes"] = sum(entry["bytes"] for entry in ledger["deleted"])
        ledger["updated_at"] = utcnow()
        atomic_write_json(receipts_path(archive_dir, "cleanup_ledger.json"), ledger)
        receipt["ledger"] = {
            "runs": len(ledger["runs"]),
            "deleted_total": ledger["deleted_bytes"],
            "path": str(receipts_path(archive_dir, "cleanup_ledger.json")),
        }
        atomic_write_json(receipts_path(archive_dir, "cleanup_receipt.json"), receipt)
    else:
        atomic_write_json(receipts_path(archive_dir, "cleanup_plan.json"), receipt)
    return receipt


# --------------------------------------------------------------------------- #
# worker state / notifications
# --------------------------------------------------------------------------- #
def load_state(manifest: Manifest) -> dict:
    return _read_local_json(manifest.archive_root / WORKER_STATE_NAME) or {
        "schema_version": SCHEMA_VERSION, "scenes": {}
    }


def save_state(manifest: Manifest, state: dict) -> None:
    state["updated_at"] = utcnow()
    atomic_write_json(manifest.archive_root / WORKER_STATE_NAME, state)


def publish_notification(manifest: Manifest, kind: str, payload: dict) -> Path:
    directory = manifest.archive_root / NOTIFICATIONS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = utcnow().replace(":", "").replace("-", "")
    path = directory / f"{stamp}_{kind}_{payload.get('scene')}_{payload.get('stage')}.json"
    payload = {"schema_version": SCHEMA_VERSION, "kind": kind, "created_at": utcnow(), **payload}
    atomic_write_json(path, payload)
    return path


def notify(manifest: Manifest, target: str, scene: str, stage: str, message: str, **extra) -> None:
    publish_notification(manifest, "notify", {
        "target": target, "scene": scene, "stage": stage, "message": message, **extra,
    })
    print(f"NOTIFY {target} {scene}/{stage}: {message}", flush=True)


def iter_targets(args, manifest: Manifest):
    scenes = manifest.scene_names() if args.scene in (None, "all") else [args.scene]
    stages = STAGES if args.stage in (None, "all") else (args.stage,)
    for scene in scenes:
        for stage in stages:
            yield scene, stage


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_plan(args, manifest: Manifest, remote: Remote) -> dict:
    report = {"schema_version": SCHEMA_VERSION, "generated_at": utcnow(), "scenes": {}}
    for scene, stage in iter_targets(args, manifest):
        groups = build_selection(manifest, scene, stage, remote)
        entry = {"stage": stage, "iteration": manifest.stage_iteration(stage), "groups": []}
        total = 0
        for group in groups:
            patterns = [pattern for pattern, _c, _r in group["patterns"]]
            listed = rsync_list(remote, group["source"], patterns)
            if not listed:
                entry["groups"].append({
                    "name": group["name"], "source": group["source"], "dest_rel": group["dest_rel"],
                    "files": 0, "bytes": 0, "absent": True,
                })
                continue
            total += sum(listed.values())
            entry["groups"].append({
                "name": group["name"], "source": group["source"], "dest_rel": group["dest_rel"],
                "files": len(listed), "bytes": sum(listed.values()),
                "categories": sorted({category_of(rel, group["patterns"]) for rel in listed}),
            })
        entry["total_bytes"] = total
        entry["total_human"] = human_bytes(total)
        entry["readiness"] = marker_state(remote, manifest, scene, stage, groups)
        entry["archive_status"] = (_read_local_json(manifest.archive_dir(scene, stage) / ARCHIVE_STATUS_NAME)
                                   or {}).get("status")
        report["scenes"].setdefault(scene, {})[stage] = entry
    return report


def cmd_transfer(args, manifest: Manifest, remote: Remote) -> dict:
    results = {}
    for scene, stage in iter_targets(args, manifest):
        print(f"[{utcnow()}] transfer {scene}/{stage} ...", flush=True)
        status = run_transfer(manifest, remote, scene, stage, verify_only=args.verify_only,
                              deep_compat=not args.skip_deep_compat)
        results[f"{scene}/{stage}"] = {
            "status": status["status"], "files": status["totals"]["files"],
            "bytes": status["totals"]["bytes"], "digest_mismatch": status["verification"]["digest_mismatch"],
            "dataset_compatibility": status["dataset_compatibility"]["status"],
        }
        print(f"[{utcnow()}] transfer {scene}/{stage}: {status['status']} "
              f"({status['totals']['files']} files, {human_bytes(status['totals']['bytes'])})", flush=True)
        if status["status"] == "verified":
            notify(manifest, "DatasetEvaluator", scene, stage,
                   f"archive verified at {manifest.archive_dir(scene, stage)} "
                   f"(checkpoint {status['model_artifacts'].get('checkpoint', {}).get('path')}), "
                   "archive_status.json published; model-load verification still required before "
                   "any remote final-model deletion",
                   archive_dir=str(manifest.archive_dir(scene, stage)))
    return results


def cmd_verify(args, manifest: Manifest, remote: Remote) -> dict:
    report = {}
    for scene, stage in iter_targets(args, manifest):
        status = run_transfer(manifest, remote, scene, stage, verify_only=True,
                              deep_compat=not args.skip_deep_compat)
        report[f"{scene}/{stage}"] = {
            "status": status["status"],
            "digest_mismatch": status["verification"]["digest_mismatch"],
            "files": status["totals"]["files"],
            "bytes": status["totals"]["bytes"],
        }
    return report


def cmd_compat(args, manifest: Manifest, remote: Remote) -> dict:
    report = {"schema_version": SCHEMA_VERSION, "generated_at": utcnow(), "scenes": {}}
    for scene, _stage in iter_targets(args, manifest):
        entry = dataset_compatibility(remote, manifest, scene, deep=not args.skip_deep_compat)
        report["scenes"][scene] = entry
        scene_dir = manifest.archive_root / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(scene_dir / "dataset_compatibility.json", entry)
        write_global_summary(manifest, manifest.scene_names())
        print(f"[{utcnow()}] compat {scene}: {entry['status']}", flush=True)
    return report


def cmd_cleanup(args, manifest: Manifest, remote: Remote) -> dict:
    report = {"schema_version": SCHEMA_VERSION, "generated_at": utcnow(), "receipts": {}}
    for scene, stage in iter_targets(args, manifest):
        receipt = run_cleanup(manifest, remote, scene, stage, execute=args.execute,
                              include_scratch=not args.final_only,
                              include_final=not args.scratch_only)
        report["receipts"][f"{scene}/{stage}"] = receipt
        logger = print
        logger(f"[{utcnow()}] cleanup {scene}/{stage} executed={args.execute} "
               f"planned={len([d for d in receipt['candidates'] if d['decision'] == 'delete'])} "
               f"deleted={len(receipt['deleted'])} freed={human_bytes(receipt['deleted_bytes'])}", flush=True)
        if args.execute and receipt["deleted"]:
            notify(manifest, "DatasetTrainer", scene, stage,
                   f"reclaimed {human_bytes(receipt['deleted_bytes'])} from {receipt['source_dir']} "
                   f"({len(receipt['deleted'])} expendable files); receipt "
                   f"{manifest.archive_dir(scene, stage)}/cleanup_receipt.json",
                   reclaimed_bytes=receipt["deleted_bytes"])
        for decision in receipt["candidates"]:
            if decision["decision"] == "retained" and decision.get("blocked_by"):
                print(f"    retained {decision['rel']}: {decision['blocked_by']}", flush=True)
    return report


def acquire_lock(manifest: Manifest) -> Path:
    """Single-instance lock for the watcher (stale locks from dead pids are taken over)."""
    path = manifest.archive_root / WORKER_LOCK_NAME
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            pid = int(payload.get("pid", -1))
        except (json.JSONDecodeError, TypeError, ValueError):
            pid = -1
        alive = False
        if pid > 0:
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                alive = False
        if alive:
            raise SystemExit(
                f"another archive watcher is already running (pid {pid}, lock {path}); "
                "stop it or delete the lock if that process is gone")
    atomic_write_json(path, {"schema_version": SCHEMA_VERSION, "pid": os.getpid(),
                             "host": manifest.host, "acquired_at": utcnow()})
    return path


def release_lock(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def record_failure(manifest: Manifest, kind: str, scene: str, stage: str, error: str) -> dict:
    path = manifest.archive_root / FAILURES_NAME
    payload = _read_local_json(path) or {"schema_version": SCHEMA_VERSION, "kind": "archive_failures",
                                         "failures": []}
    payload["failures"].append({
        "kind": kind, "scene": scene, "stage": stage, "error": error[:4000], "at": utcnow(),
    })
    payload["failures"] = payload["failures"][-200:]
    payload["updated_at"] = utcnow()
    atomic_write_json(path, payload)
    return payload["failures"][-1]


def cleanup_state_path(manifest: Manifest, scene: str, stage: str) -> Path:
    """Where the durable evidence of an executed cleanup lives for a stage."""
    return receipts_path(manifest.archive_dir(scene, stage), "cleanup_ledger.json")


def cmd_watch(args, manifest: Manifest, remote: Remote) -> dict:
    """Recurring worker: archive each published stage, then clean it once gated.

    Per pass, for every scene/stage:
      1. archive when the remote marker/run is complete and no verified archive exists;
      2. when a verified archive exists, *re-check* the cleanup gates (a verified
         archive alone releases nothing) and clean the stage's authorized remote
         assets as soon as archive + evaluation load + sha256 identity + the
         stage1-depends-on-stage2 rule all hold;
      3. isolate failures: one bad scene/stage is recorded and skipped, never
         stalling the others and never reported as verified.
    """
    lock = acquire_lock(manifest)
    state = load_state(manifest)
    deadline = None if args.duration <= 0 else time.time() + args.duration
    processed = cleaned = failures = 0
    print(f"[{utcnow()}] archive watcher start interval={args.interval}s duration={args.duration}s "
          f"scenes={manifest.scene_names()} lock={lock}", flush=True)
    try:
        while True:
            state.setdefault("runs", []).append({"at": utcnow(), "pid": os.getpid()})
            state["runs"] = state["runs"][-50:]
            for scene, stage in iter_targets(args, manifest):
                scene_state = state.setdefault("scenes", {}).setdefault(scene, {}).setdefault(stage, {})
                try:
                    groups = build_selection(manifest, scene, stage, remote)
                    archive_dir = manifest.archive_dir(scene, stage)
                    status = _read_local_json(archive_dir / ARCHIVE_STATUS_NAME) or {}
                    verified = status.get("status") == "verified"
                    if not verified:
                        marker = marker_state(remote, manifest, scene, stage, groups)
                        scene_state.update({
                            "archive_status": status.get("status"),
                            "ready": marker["ready"], "marker_present": marker["marker_present"],
                            "missing_required": marker["missing_required"], "checked_at": utcnow(),
                        })
                        if not marker["ready"]:
                            continue
                        print(f"[{utcnow()}] marker detected {scene}/{stage}; archiving", flush=True)
                        status = run_transfer(manifest, remote, scene, stage,
                                              deep_compat=not args.skip_deep_compat)
                        processed += 1
                        scene_state["archive_status"] = status["status"]
                        scene_state["last_error"] = None
                        if status["status"] != "verified":
                            failures += 1
                            record_failure(manifest, "archive_not_verified", scene, stage,
                                           f"run_transfer returned {status['status']}")
                            continue
                        notify(manifest, "DatasetEvaluator", scene, stage,
                               f"archive verified at {archive_dir}; model-load verification requested "
                               "before this stage's remote finals are released")
                    scene_state["archive_status"] = "verified"
                    scene_state["last_error"] = None
                    if args.no_cleanup:
                        continue
                    gate = evaluation_gate(manifest, scene, stage)
                    dependency = (stage2_dependency_gate(manifest, remote, scene)
                                  if stage == "stage1" else None)
                    identity = identity_gate(manifest, remote, scene, stage, status)
                    scene_state["cleanup_gates"] = {
                        "archive_verified": True,
                        "model_load_verified": gate["satisfied"],
                        "identity": identity["satisfied"],
                        "stage2_dependency": (dependency or {}).get("satisfied"),
                        "checked_at": utcnow(),
                    }
                    # Cleanup runs on every pass for a verified archive: the
                    # per-candidate gates decide what may go, so expendable
                    # scratch is reclaimed as soon as the archive is verified,
                    # while finals wait for the evaluation load (and, for
                    # stage1, for the same scene's stage2 to finish). A later
                    # evaluation marker therefore releases the pinned finals on
                    # a subsequent pass instead of never being re-checked.
                    receipt = run_cleanup(manifest, remote, scene, stage, execute=True,
                                          include_scratch=True, include_final=True)
                    planned = [d for d in receipt.get("candidates", []) if d["decision"] == "delete"]
                    scene_state["cleanup_planned"] = len(planned)
                    scene_state["cleanup_retained"] = [
                        {"path": d["path"], "blocked_by": d.get("blocked_by")}
                        for d in receipt.get("candidates", []) if d["decision"] == "retained"
                    ][:40]
                    scene_state["cleanup_checked_at"] = utcnow()
                    # The ledger deduplicates paths, including receipts from
                    # before missing files stopped being reported as deletions.
                    scene_state["cleanup_deleted_bytes_total"] = receipt["ledger"]["deleted_total"]
                    if receipt.get("deleted"):
                        cleaned += 1
                        notify(manifest, "DatasetTrainer", scene, stage,
                               f"archive-gated cleanup removed {receipt['deleted_bytes']} bytes "
                               f"({len(receipt['deleted'])} files); receipt "
                               f"{cleanup_state_path(manifest, scene, stage)}",
                               reclaimed_bytes=receipt["deleted_bytes"])
                    if receipt.get("failed"):
                        failures += 1
                        record_failure(manifest, "cleanup_partial", scene, stage,
                                       json.dumps(receipt["failed"])[:2000])
                    if not planned and not receipt.get("retained"):
                        scene_state["cleanup_completed_at"] = utcnow()
                except BaseException as error:  # noqa: BLE001 - isolate one target's failure
                    failures += 1
                    scene_state["last_error"] = f"{type(error).__name__}: {error}"[:2000]
                    scene_state["last_error_at"] = utcnow()
                    record_failure(manifest, "exception", scene, stage,
                                   f"{type(error).__name__}: {error}")
                    print(f"[{utcnow()}] FAILED {scene}/{stage}: {type(error).__name__}: "
                          f"{str(error)[:200]}", file=sys.stderr, flush=True)
            save_state(manifest, state)
            for scene in manifest.scene_names():
                write_scene_summary(manifest, scene)
            write_global_summary(manifest, manifest.scene_names())
            if deadline is not None and time.time() >= deadline:
                break
            time.sleep(args.interval)
    finally:
        release_lock(lock)
    print(f"[{utcnow()}] archive watcher stop processed={processed} cleaned={cleaned} "
          f"failures={failures}", flush=True)
    return {"processed": processed, "cleaned": cleaned, "failures": failures,
            "state": str(manifest.archive_root / WORKER_STATE_NAME),
            "failures_log": str(manifest.archive_root / FAILURES_NAME)}


def cmd_handoff(args, manifest: Manifest, remote: Remote) -> dict:
    """Freeze the 12-scene plan and the launch spec for the recurring worker."""
    script = Path(__file__).resolve()
    plan_args = argparse.Namespace(scene="all", stage="all", skip_deep_compat=True)
    plan = cmd_plan(plan_args, manifest, remote)
    scene_plan: dict[str, dict] = {}
    ready_now, waiting, not_started = [], [], []
    for scene in manifest.scene_names():
        entry = {"stages": {}}
        for stage in STAGES:
            data = plan["scenes"].get(scene, {}).get(stage, {})
            groups = build_selection(manifest, scene, stage, remote)
            archive_status = _read_local_json(manifest.archive_dir(scene, stage) / ARCHIVE_STATUS_NAME) or {}
            known = scene == "JAX_068" or bool(archive_status)
            entry["stages"][stage] = {
                "iteration": manifest.stage_iteration(stage),
                "remote_marker": manifest.remote_marker(scene, stage),
                "source_origin": groups[0]["source_origin"],
                "source": groups[0]["source"],
                "archive_dir": str(manifest.archive_dir(scene, stage)),
                "selection_bytes": data.get("total_bytes"),
                "selection_human": data.get("total_human"),
                "readiness": data.get("readiness", {}).get("ready"),
                "archive_status": archive_status.get("status"),
                "evaluation_gate": evaluation_gate(manifest, scene, stage),
            }
            if data.get("readiness", {}).get("ready") and not known:
                waiting.append(f"{scene}/{stage}")
            elif not known and not data.get("readiness", {}).get("ready"):
                not_started.append(f"{scene}/{stage}")
        ready_now.append(scene) if all(
            entry["stages"][stage]["archive_status"] == "verified" for stage in STAGES) else None
        scene_plan[scene] = entry
    spec = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utcnow(),
        "worker": "ArchiveCurator",
        "script": str(script),
        "script_sha256": sha256_file(script),
        "archive_root": str(manifest.archive_root),
        "remote_host": manifest.host,
        "watch": {
            "command": [sys.executable, str(script), "--archive-root", str(manifest.archive_root),
                        "watch", "--interval", "180"],
            "interval_seconds": 180,
            "purpose": ("poll every scene/stage for a remote stage_complete.json plus complete "
                        "run_status.json, then rsync+sha256-verify and publish archive_status.json"),
            "hub_start_spec": {
                "op": "start", "name": "archive-curator",
                "application": sys.executable,
                "args": [str(script), "--archive-root", str(manifest.archive_root),
                         "watch", "--interval", "180"],
                "cwd": str(script.parents[1]),
                "ready": {"log": "archive watcher start", "timeout": 30},
            },
        },
        "one_shot": {
            "transfer_all": [sys.executable, str(script), "--archive-root", str(manifest.archive_root), "transfer", "--scene", "all", "--stage", "all"],
            "transfer_one": [sys.executable, str(script), "--archive-root", str(manifest.archive_root), "transfer", "--scene", "<scene>", "--stage", "<stage1|stage2>"],
            "verify": [sys.executable, str(script), "--archive-root", str(manifest.archive_root), "verify", "--scene", "<scene>", "--stage", "<stage>"],
            "cleanup_account": [sys.executable, str(script), "--archive-root", str(manifest.archive_root), "cleanup-account", "--scene", "<scene>", "--stage", "<stage>"],
            "handoff": [sys.executable, str(script), "--archive-root", str(manifest.archive_root), "handoff"],
            "cleanup_dry_run": [sys.executable, str(script), "--archive-root", str(manifest.archive_root), "cleanup", "--scene", "<scene>", "--stage", "<stage>"],
            "cleanup_execute": [sys.executable, str(script), "--archive-root", str(manifest.archive_root), "cleanup", "--scene", "<scene>", "--stage", "<stage>", "--execute"],
        },
        "gates": {
            "archive_ready": "archive_status.json status=verified (written only by this worker)",
            "remote_scratch_delete": "archive_status=verified",
            "stage2_final_delete": "archive_status=verified AND evaluation_status.json model_load_verified=true",
            "stage1_final_delete": ("archive_status=verified AND evaluation model_load_verified=true AND "
                                    "<scene>/stage2/stage_complete.json status=completed"),
            "identity": ("sha256 identity, not just booleans: evaluation_status.json's checkpoint/PLY "
                         "sha256 (and its protocol_fingerprint) must equal the run-root "
                         "stage_complete.json sha256 and the archive manifest sha256"),
            "legacy_parent_delete": "cleanup_approval.json from Main naming the parent",
            "approved_scratch": ("cleanup_approval.json entry approved_scratch with path, categories, "
                                 "requires and scenes"),
        },
        "ownership": {
            "archive_status.json": "ArchiveCurator only",
            "archive_receipts/*": "ArchiveCurator only",
            "stage_complete.json (remote)": "DatasetTrainer only",
            "evaluation_status.json": "DatasetEvaluator only",
            "canonical run-root final assets": ("ArchiveCurator (the trainer runs --no-reap, so this worker "
                                                "is the only remote deletion path for finals)"),
        },
        "local_archive_root_disk": str(manifest.archive_root),
        "scenes": scene_plan,
        "summary": {
            "scenes": len(manifest.scene_names()),
            "archived_verified": sorted(s for s in ready_now),
            "waiting_for_remote_marker_or_readiness": sorted(waiting),
            "not_started_remote": sorted(not_started),
            "note": ("JAX_068 is the only pre-existing qualified run; the other 11 scenes archive as "
                     "DatasetTrainer publishes stage_complete.json per stage"),
        },
    }
    atomic_write_json(manifest.archive_root / "archive_worker_launch.json", spec)
    print(f"[{utcnow()}] handoff written: {manifest.archive_root / 'archive_worker_launch.json'}",
          flush=True)
    return spec


def cmd_cleanup_account(args, manifest: Manifest, remote: Remote) -> dict:
    """Re-derive the true filesystem effect of an already-executed cleanup.

    The first execution of this worker (JAX_068) measured the filesystem delta
    only after the change, so this subcommand reconstructs the accounting from
    the live state: for every path the receipt reports as deleted it records
    whether the path is gone, which peer hard link survives, and the resulting
    df delta measured now versus the receipt's recorded usage.
    """
    report = {"schema_version": SCHEMA_VERSION, "generated_at": utcnow(), "accounts": {}}
    for scene, stage in iter_targets(args, manifest):
        archive_dir = manifest.archive_dir(scene, stage)
        ledger = _read_local_json(receipts_path(archive_dir, "cleanup_ledger.json"))
        receipt = _read_local_json(receipts_path(archive_dir, "cleanup_receipt.json"))
        if not ledger:
            report["accounts"][f"{scene}/{stage}"] = {
                "status": "no_ledger",
                "detail": "no executed cleanup was recorded for this stage",
            }
            continue
        # The ledger is the durable record of what this worker removed; the
        # receipt only describes the most recent run.
        deleted = [entry["path"] for entry in ledger["deleted"]]
        peers = [entry["mirror_of"] for entry in ledger["deleted"] if entry.get("mirror_of")]
        info = link_info(remote, list(deleted) + peers)
        usage_now = filesystem_usage(remote, ledger.get("deleted", [{}])[0].get("path", "/")
                                    if deleted else receipt.get("source_dir", "/"))
        accounted = []
        for entry in ledger["deleted"]:
            path = entry["path"]
            peer = entry.get("mirror_of")
            record = info.get(path, {})
            peer_info = info.get(peer or "", {})
            accounted.append({
                "path": path,
                "bytes": entry.get("bytes"),
                "removed_at": entry.get("removed_at"),
                "present_now": record.get("links") is not None,
                "peer_path": peer,
                "peer_present_now": peer_info.get("links") is not None,
                "peer_links_now": peer_info.get("links"),
                "peer_owner": "DatasetTrainer (canonical run root)",
                "frees_space_only_with_peer": bool(peer),
            })
        peers_gone = [p for p in accounted if not p["peer_present_now"]]
        peers_kept = [p for p in accounted if p["peer_present_now"]]
        if accounted and not peers_kept:
            conclusion = ("every removed path was a hard link whose canonical peer has since also been "
                          "removed, so the space is reclaimed")
        elif peers_kept:
            conclusion = ("some canonical peers survive, so those removed mirrors free nothing until "
                          "the peer owner removes the surviving link")
        else:
            conclusion = "no paths were removed by this cleanup"
        recorded = (receipt or {}).get("filesystem")
        before_used = (recorded or {}).get("before", {}).get("used_bytes")
        account = {
            "status": "reconstructed",
            "scene": scene,
            "stage": stage,
            "ledger": str(receipts_path(archive_dir, "cleanup_ledger.json")),
            "ledger_runs": len(ledger.get("runs", [])),
            "latest_receipt": str(receipts_path(archive_dir, "cleanup_receipt.json")),
            "removed_paths": len(deleted),
            "removed_file_bytes": ledger.get("deleted_bytes"),
            "recorded_filesystem": recorded,
            "filesystem_now": usage_now,
            "measured_filesystem_delta_bytes": (
                (usage_now["used_bytes"] - before_used)
                if before_used is not None and usage_now["used_bytes"] is not None else None),
            "delta_unavailable_because": None if before_used is not None else (
                "no execution recorded a pre-cleanup filesystem sample; only the post-state is "
                "measurable now"),
            "paths": accounted,
            "peers_surviving": [p["peer_path"] for p in peers_kept],
            "peers_removed": [p["peer_path"] for p in peers_gone],
            "supersedes": ["cleanup_receipt.deleted_bytes as a freed-space claim"],
            "conclusion": conclusion,
            "accounted_at": utcnow(),
        }
        report["accounts"][f"{scene}/{stage}"] = account
        atomic_write_json(receipts_path(archive_dir, "cleanup_accounting.json"), account)
        print(f"[{utcnow()}] account {scene}/{stage}: removed_file_bytes="
              f"{account['removed_file_bytes']} delta={account['measured_filesystem_delta_bytes']}",
              file=sys.stderr, flush=True)
    return report


def write_scene_summary(manifest: Manifest, scene: str) -> dict:
    """Small per-scene index of what was archived, for the 12-scene handoff."""
    entry = manifest.scene(scene)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "scene": scene,
        "dataset": entry.get("dataset"),
        "recipe": entry.get("recipe"),
        "archive_root": str(manifest.archive_root / scene),
        "updated_at": utcnow(),
        "stages": {},
    }
    for stage in STAGES:
        archive_dir = manifest.archive_dir(scene, stage)
        status = _read_local_json(archive_dir / ARCHIVE_STATUS_NAME) or {}
        summary["stages"][stage] = {
            "archive_status": status.get("status"),
            "iteration": status.get("iteration"),
            "source_path": status.get("source_path"),
            "archive_dir": str(archive_dir),
            "model_artifacts": status.get("model_artifacts", {}),
            "totals": status.get("totals"),
            "dataset_compatibility": (status.get("dataset_compatibility") or {}).get("status"),
            "evaluation_gate": evaluation_gate(manifest, scene, stage),
            "files": len(status.get("files") or []),
        }
    compat = _read_local_json(manifest.archive_root / scene / "dataset_compatibility.json")
    if compat:
        summary["dataset_compatibility"] = {key: compat.get(key) for key in
                                            ("status", "deep_verified", "files_local",
                                             "files_remote", "digest_mismatch", "local_dir",
                                             "remote_dir", "bytes")}
    atomic_write_json(manifest.archive_root / scene / "archive_summary.json", summary)
    return summary


def write_global_summary(manifest: Manifest, scenes: Sequence[str]) -> dict:
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "archive_summary",
        "archive_root": str(manifest.archive_root),
        "method": manifest.data.get("method"),
        "branch": manifest.data.get("branch"),
        "updated_at": utcnow(),
        "scenes": {},
    }
    for scene in scenes:
        entry: dict = {"dataset": manifest.scene(scene).get("dataset"), "stages": {}}
        for stage in STAGES:
            status = _read_local_json(manifest.archive_dir(scene, stage) / ARCHIVE_STATUS_NAME) or {}
            evaluation = _read_local_json(manifest.evaluation_marker(scene, stage)) or {}
            entry["stages"][stage] = {
                "archive_status": status.get("status"),
                "files": (status.get("totals") or {}).get("files"),
                "bytes": (status.get("totals") or {}).get("bytes"),
                "checkpoint_sha256": (status.get("model_artifacts", {}).get("checkpoint") or {}).get("sha256"),
                "point_cloud_sha256": (status.get("model_artifacts", {}).get("point_cloud") or {}).get("sha256"),
                "evaluation_status": evaluation.get("status"),
                "model_load_verified": evaluation.get("model_load_verified"),
            }
        summary["scenes"][scene] = entry
    complete = [s for s in scenes if all(
        summary["scenes"][s]["stages"][st]["archive_status"] == "verified" for st in STAGES)]
    summary["summary"] = {
        "scenes_total": len(scenes),
        "scenes_archived_verified": complete,
        "scenes_pending": [s for s in scenes if s not in complete],
    }
    atomic_write_json(manifest.archive_root / "archive_summary.json", summary)
    return summary


def cmd_status(args, manifest: Manifest, remote: Remote) -> dict:
    report = {"schema_version": SCHEMA_VERSION, "generated_at": utcnow(), "scenes": {}}
    for scene in manifest.scene_names():
        entry = {}
        for stage in STAGES:
            status = _read_local_json(manifest.archive_dir(scene, stage) / ARCHIVE_STATUS_NAME) or {}
            evaluation = _read_local_json(manifest.evaluation_marker(scene, stage)) or {}
            entry[stage] = {
                "archive_status": status.get("status"),
                "files": (status.get("totals") or {}).get("files"),
                "bytes": (status.get("totals") or {}).get("bytes"),
                "evaluation_model_load_verified": evaluation.get("model_load_verified"),
            }
        report["scenes"][scene] = entry
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive-root",
                        default=os.environ.get("SKYFALL_ARCHIVE_ROOT",
                                               "/dataset/Skyfall-GS/experiments/stage2_gaussianzoom_20260913"))
    parser.add_argument("--host", default=None, help="ssh alias (defaults to pipeline_manifest.json remote_host)")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, with_targets=True):
        if with_targets:
            p.add_argument("--scene", default="all")
            p.add_argument("--stage", default="all", choices=("all", *STAGES))
        p.add_argument("--skip-deep-compat", action="store_true",
                       help="compare dataset file sizes only (no per-file sha256)")

    p = sub.add_parser("plan", help="list the frozen selection and readiness per scene/stage")
    common(p)
    p = sub.add_parser("transfer", help="rsync the selection, verify sha256 on both ends, publish archive_status.json")
    common(p)
    p.add_argument("--verify-only", action="store_true", help="verify existing local archive against remote")
    p = sub.add_parser("verify", help="re-verify an existing archive without transferring")
    common(p)
    p = sub.add_parser("compat", help="verify local/remote dataset input compatibility")
    common(p)
    p = sub.add_parser("cleanup", help="plan (or execute) gated remote cleanup with a JSON receipt")
    common(p)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--final-only", action="store_true", help="only consider the final model artifacts")
    p.add_argument("--scratch-only", action="store_true", help="only consider expendable scratch")
    p = sub.add_parser("watch", help="recurring worker over remote stage_complete.json markers")
    common(p)
    p.add_argument("--interval", type=int, default=180)
    p.add_argument("--duration", type=int, default=0, help="seconds; 0 = run forever")
    p.add_argument("--cleanup-scratch", action="store_true",
                   help="after a verified archive, also reclaim that stage's expendable remote scratch")
    p.add_argument("--no-cleanup", action="store_true",
                   help="archive only; never delete remote assets even when gates are satisfied")
    p = sub.add_parser("status", help="local archive/evaluation state summary")
    common(p, with_targets=False)
    p = sub.add_parser("cleanup-account",
                       help="reconstruct the measured filesystem effect of an executed cleanup")
    common(p)
    p = sub.add_parser("handoff", help="write the recurring-worker launch spec and 12-scene readiness plan")
    common(p, with_targets=False)

    args = parser.parse_args(argv)
    manifest = Manifest(Path(args.archive_root))
    if args.host:
        manifest.host = args.host
    remote = Remote(manifest.host)
    handler = {
        "plan": cmd_plan, "transfer": cmd_transfer, "verify": cmd_verify, "compat": cmd_compat,
        "cleanup": cmd_cleanup, "watch": cmd_watch, "status": cmd_status,
        "cleanup-account": cmd_cleanup_account, "handoff": cmd_handoff,
    }[args.command]
    report = handler(args, manifest, remote)
    if args.command in {"plan", "verify", "compat", "status", "cleanup-account"}:
        print(json.dumps(_jsonable(report), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
