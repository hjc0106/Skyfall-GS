from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PRF_DIR = Path(__file__).resolve().parent

DEPTH_CACHE_PROTOCOL = {
    "renderer": "prf.gs_depth -> gsplat.rasterization",
    "render_mode": "RGB+D",
    "depth_kind": "expected",
    "conversion": "expected_z = accumulated_z / alpha",
    "min_alpha": 1e-6,
    "sampling": "bilinear at projected pixel (u, v)",
    "states": ["VISIBLE", "OCCLUDED", "OUTSIDE_FRUSTUM", "UNCERTAIN", "INVALID_PROJECTION"],
    "pair_rule": "only VISIBLE views enter the best camera pair",
}


def canonical_path(path: Path | str) -> Path:
    """Absolute path without following symlinks."""
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = Path.cwd() / value
    return Path(os.path.abspath(str(value)))


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def describe_path(path: Path | str, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    given = str(path)
    abs_path = canonical_path(path)
    real_path = Path(os.path.realpath(abs_path)) if abs_path.exists() else None
    readable = real_path if real_path is not None and real_path.is_file() else abs_path
    repo_relative = _repo_relative(abs_path, repo_root)
    if repo_relative is None and real_path is not None:
        repo_relative = _repo_relative(real_path, repo_root)
    info: dict[str, Any] = {
        "path_given": given,
        "path": str(abs_path),
        "path_repo_relative": repo_relative,
        "path_realpath": str(real_path) if real_path is not None else None,
        "exists": bool(readable.exists()),
        "sha256": None,
        "size_bytes": None,
    }
    if readable.is_file():
        info["sha256"] = sha256_file(readable)
        info["size_bytes"] = int(readable.stat().st_size)
    return info


def _repo_relative(path: Path, repo_root: Path) -> str | None:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return None


def code_file_hashes(*, prf_dir: Path = PRF_DIR) -> dict[str, str]:
    hashes = {}
    for path in sorted(prf_dir.glob("*.py")):
        hashes[f"prf/{path.name}"] = sha256_file(path)
    schema = prf_dir.parent / "SCHEMA_V2.md"
    if schema.is_file():
        hashes["SCHEMA_V2.md"] = sha256_file(schema)
    return hashes


def git_identity(*, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    identity: dict[str, Any] = {"root": str(repo_root), "rev": None, "dirty": None}
    try:
        identity["rev"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "thirdparty/PRF"], cwd=repo_root, text=True
        )
        identity["dirty"] = bool(status.strip())
        identity["prf_status"] = [line for line in status.splitlines() if line.strip()]
    except (OSError, subprocess.CalledProcessError):
        pass
    return identity


def describe_depth_cache(cache_dir: Path | str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"protocol": DEPTH_CACHE_PROTOCOL, "dir": None, "files": {}}
    if cache_dir is None:
        return payload
    directory = canonical_path(cache_dir)
    payload["dir"] = describe_path(directory)
    if not directory.is_dir():
        return payload
    for path in sorted(directory.glob("*.npz")):
        payload["files"][path.name] = {"sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)}
    return payload


def build_experiment_manifest(
    *,
    name: str,
    ply: Path | str,
    transforms: Path | str,
    output: Path | str,
    config: dict | None,
    visibility_model: str,
    reuse_patches: Path | str | None = None,
    depth_cache: Path | str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "name": name,
        "visibility_model": visibility_model,
        "config": config or {},
        "git": git_identity(),
        "code_sha256": code_file_hashes(),
        "ply": describe_path(ply),
        "transforms": describe_path(transforms),
        "output": describe_path(output),
        "reuse_patches": None if reuse_patches is None else describe_path(reuse_patches),
        "depth_cache": describe_depth_cache(depth_cache),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(manifest: dict[str, Any], output_dir: Path | str) -> Path:
    path = canonical_path(output_dir) / "freeze_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def config_dict(cfg) -> dict[str, Any]:
    return asdict(cfg)
