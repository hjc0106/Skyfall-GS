#!/usr/bin/env python3
"""All-scene Skyfall-GS Stage1/Stage2 evaluator (local GPU, owner: DatasetEvaluator).

Consumes the frozen stage archives produced by the training/archiving pipeline:

    <archive_root>/<scene>/<stage>/              stage1|stage2
        chkpnt<iter>.pth                          final checkpoint (authoritative)
        point_cloud/iteration_<iter>/point_cloud.ply   matching PLY (carries filter_3D)
        cfg_args, cameras.json, metrics.json, source.sha256, ...

and writes, per stage:

    <archive_root>/<scene>/evaluation/<stage>/
        protocol.json                 what is measured, on what, with which policy
        heldout_metrics.json/.csv     per-view + aggregate PSNR/SSIM/LPIPS/L1
        renders/heldout/<view>_{render,gt,absdiff_x4}.png
        novel/<path>.mp4              fixed checked-in camera trajectories
        novel/<path>/frame_*.png      deterministic subset used for panels
        official/external_metrics.json/.csv    official external-view protocol
        evaluation.json               full evidence record
        evaluation_status.json        published LAST, atomically (model_load_verified)

Model loading is faithful and 24GB-safe:
  * the checkpoint (which contains learned per-Gaussian appearance embeddings + MLP,
    but NOT filter_3D) is loaded with ``map_location='cpu'``;
  * the matching PLY (which carries filter_3D) supplies filter_3D;
  * optimizer / densification-only tensors are dropped and never moved to the GPU;
  * only rendering tensors (geometry, SH features, appearance state, filter_3D) are
    moved to ``cuda``, one model and one frame at a time.

Held-out metrics: every camera of ``transforms_test.json`` for the scene, exactly as
``train.py::training_report`` scores them (full-frame, unmasked, clamped [0,1] RGB,
``render(..., testing=True)``, i.e. the checkpoint's learned test appearance row
``min(6, n_train-1)``), so the archived checkpoints must reproduce the values that
the remote training console printed for the same iteration.

Official external-view protocol uses the released Skyfall-GS eval data
(``data_eval_JAX`` / ``data_eval_NYC``).  Paired PSNR/SSIM/LPIPS against the official
GT is only computed after a *verified* frame-index correspondence test; otherwise
only distributional CLIP-FID/CMMD is reported and the paired fields stay null with an
explicit reason.  Bundled other-method videos are never used as our output.

Usage::

    # single scene, both stages (default)
    python scripts/evaluate_dataset.py --scene JAX_068

    # the long-running consumer for newly verified archives
    python scripts/evaluate_dataset.py --queue --exit-when-complete

    # skip already completed stages, keep the queue cheap
    python scripts/evaluate_dataset.py --queue --skip-completed
"""

from __future__ import annotations

import argparse
import ast
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state
from utils.camera_utils import cameraList_from_camInfos
from utils.image_utils import psnr as psnr_metric
from utils.loss_utils import l1_loss, ssim as ssim_metric

DEFAULT_MANIFEST = "/dataset/Skyfall-GS/experiments/stage2_gaussianzoom_20260913/pipeline_manifest.json"
EVAL_SCHEMA = "skyfall_gs_dataset_evaluation_v1"
STAGE_STATUS_SCHEMA = "skyfall_gs_evaluation_status_v1"
STATUS_COMPLETED = "completed"
ARCHIVE_READY = "verified"

# Remote training console values for the two reused JAX_068 stages.  These are the
# exact numbers train.py::training_report printed for the final iteration, so they
# are the ground truth an independent reload must reproduce.
KNOWN_REPRODUCTION: dict[tuple[str, str], dict[str, Any]] = {
    ("JAX_068", "stage1"): {
        "psnr": 18.431320190429688,
        "l1": 0.07893401011824608,
        "iteration": 30000,
        "source": "remote stage1 console.log [ITER 30000] Evaluating test",
    },
    ("JAX_068", "stage2"): {
        "psnr": 17.668739318847656,
        "l1": 0.09248843416571617,
        "iteration": 80000,
        "source": "remote stage2 console.log [ITER 80000] Evaluating test",
    },
}
DEFAULT_PSNR_TOL = 0.1
DEFAULT_L1_TOL = 1e-3

# Official external-view sampling (mirrors the released repo eval.py convention:
# n uniformly spread frames per sequence, paired by index / relative position).
OFFICIAL_FRAMES = 24
PATCH_SIZE = 512
PATCH_MIN = (9, 16)


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path) -> Any:
    with open(path, "r") as fh:
        return json.load(fh)


def write_json(path: str | Path, payload: Any) -> None:
    """Atomic JSON write: temp file in the same directory, then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{}]
    fields = list(fieldnames) if fieldnames else list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: str | Path, do_hash: bool = True) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {"path": str(p), "exists": False}
    info: dict[str, Any] = {
        "path": str(p),
        "exists": True,
        "size": p.stat().st_size,
        "mtime": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
    }
    if do_hash:
        info["sha256"] = sha256_file(p)
    return info


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def parse_cfg_args(path: str | Path) -> dict[str, Any]:
    """Parse the ``Namespace(...)`` repr that train.py writes to ``cfg_args``.

    Values are kept verbatim when they are literals (numbers, strings, bools, lists,
    dicts).  Anything else (e.g. a nested dataclass repr) is preserved as its source
    text so no information is silently dropped.
    """
    text = Path(path).read_text().strip()
    start = text.find("Namespace(")
    if start < 0:
        raise ValueError(f"not a Namespace repr: {path}")
    tree = ast.parse(text[start:], mode="eval")
    if not (isinstance(tree.body, ast.Call) and getattr(tree.body.func, "id", None) == "Namespace"):
        raise ValueError(f"unexpected cfg_args expression in {path}")
    parsed: dict[str, Any] = {}
    for kw in tree.body.keywords:
        if kw.arg is None:
            raise ValueError(f"positional argument in cfg_args {path}")
        try:
            parsed[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            parsed[kw.arg] = ast.unparse(kw.value)
    return parsed


def gpu_mem_mib() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tensor_to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().float().clamp(0.0, 1.0).cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr[:3].transpose(1, 2, 0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


# --------------------------------------------------------------------------- #
# metric stack
# --------------------------------------------------------------------------- #
class MetricStack:
    """Reference metrics with one fixed, documented definition per metric."""

    def __init__(self, device: torch.device, lpips_net: str = "alex") -> None:
        self.device = device
        self.available: dict[str, Any] = {}
        self.prerequisites: list[dict[str, str]] = []
        try:
            import lpips  # noqa: WPS433 (optional heavy dependency)

            net = lpips.LPIPS(net=lpips_net, verbose=False).to(device).eval()
            for param in net.parameters():
                param.requires_grad_(False)
            self.available["lpips"] = net
            self.lpips_net = lpips_net
        except Exception as exc:  # pragma: no cover - reported as prerequisite
            self.prerequisites.append(
                {"metric": "lpips", "status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
            )
            self.lpips_net = None

    @torch.no_grad()
    def reference(self, image: torch.Tensor, target: torch.Tensor) -> dict[str, float | None]:
        """image/target: [3,H,W] float in [0,1]; same raster required."""
        if image.shape != target.shape:
            raise ValueError(f"raster mismatch: render {tuple(image.shape)} vs gt {tuple(target.shape)}")
        out: dict[str, float | None] = {}
        out["psnr"] = float(psnr_metric(image, target).mean().item())
        out["ssim"] = float(ssim_metric(image.unsqueeze(0), target.unsqueeze(0)).item())
        out["l1"] = float(l1_loss(image, target).mean().item())
        net = self.available.get("lpips")
        if net is None:
            out["lpips"] = None
        else:
            a = image.unsqueeze(0) * 2.0 - 1.0
            b = target.unsqueeze(0) * 2.0 - 1.0
            out["lpips"] = float(net(a, b).mean().item())
        return out

    def metric_definitions(self) -> dict[str, Any]:
        return {
            "psnr_db": {
                "implementation": "utils.image_utils.psnr",
                "formula": "20*log10(1/sqrt(MSE)) per channel over all pixels, mean over RGB",
                "data_range": 1.0,
                "inputs": "clamped [0,1] float32 RGB, full frame",
            },
            "ssim": {
                "implementation": "utils.loss_utils.ssim",
                "params": {"window_size": 11, "sigma": 1.5, "size_average": True},
                "note": "pure torch SSIM on RGB (not the fused_ssim CUDA kernel)",
            },
            "lpips": {
                "implementation": "lpips.LPIPS",
                "net": self.lpips_net,
                "input_scaling": "image*2-1 (canonical AlexNet v0.1, normalize=False on [0,1] inputs)",
                "available": self.lpips_net is not None,
            },
            "l1": {"implementation": "utils.loss_utils.l1_loss", "formula": "mean |render-gt|"},
        }


# --------------------------------------------------------------------------- #
# archive / dataset plumbing
# --------------------------------------------------------------------------- #
def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = read_json(path)
    manifest["_manifest_path"] = str(Path(path).resolve())
    return manifest


def scene_dataset_dir(manifest: dict[str, Any], scene: str) -> Path:
    root = Path(manifest["local_dataset_root"])
    entry = next((s for s in manifest["scenes"] if s["scene"] == scene), None)
    if entry is None:
        raise KeyError(f"scene {scene!r} not in manifest")
    return root / entry["dataset"] / scene


def scene_camera_path_dir(manifest: dict[str, Any], scene: str) -> Path:
    prefix, num = scene.split("_", 1)
    return Path(manifest["local_project"]) / "camera_paths" / prefix / num


def split_of(scene: str) -> str:
    return "data_eval_JAX" if scene.startswith("JAX") else "data_eval_NYC"


def ensure_eval_data(manifest: dict[str, Any], scene: str) -> Path | None:
    """Return <eval_data>/data_eval_<split>/<scene>, extracting the zip if needed."""
    root = Path(manifest["local_eval_archive_root"])
    target = root / split_of(scene) / scene
    if target.is_dir():
        return target
    zip_path = root / f"{split_of(scene)}.zip"
    if zip_path.is_file():
        import zipfile

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root)
        if target.is_dir():
            return target
    return None


def find_final_assets(stage_dir: Path) -> tuple[Path, Path, int]:
    """Resolve the single final checkpoint and its matching PLY."""
    ckpts: dict[int, Path] = {}
    for cand in sorted(stage_dir.glob("chkpnt*.pth")) + sorted(stage_dir.glob("*.pth")):
        stem = cand.stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        if digits:
            ckpts[int(digits)] = cand
    ckpts = {it: p for it, p in ckpts.items() if p.stat().st_size > 0}
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint under {stage_dir}")
    if len(ckpts) > 1:  # the contract keeps only the final checkpoint archived
        raise RuntimeError(f"expected one final checkpoint under {stage_dir}, found {sorted(ckpts)}")
    iteration = max(ckpts)
    checkpoint = ckpts[iteration]
    ply = stage_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply.is_file():
        ply_dirs = sorted(p for p in (stage_dir / "point_cloud").glob("iteration_*") if p.is_dir())
        raise FileNotFoundError(
            f"checkpoint iteration {iteration} has no matching PLY at {ply} (found {[p.name for p in ply_dirs]})"
        )
    return checkpoint, ply, iteration


def archive_state(stage_dir: Path) -> dict[str, Any]:
    marker = stage_dir / "archive_status.json"
    if not marker.is_file():
        return {"present": False, "ready": False, "path": str(marker)}
    payload = read_json(marker)
    return {"present": True, "ready": payload.get("status") == ARCHIVE_READY, "path": str(marker), "payload": payload}


def evaluation_state(eval_dir: Path) -> dict[str, Any]:
    marker = eval_dir / "evaluation_status.json"
    if not marker.is_file():
        return {"present": False, "completed": False, "path": str(marker)}
    payload = read_json(marker)
    return {
        "present": True,
        "completed": payload.get("status") == STATUS_COMPLETED,
        "path": str(marker),
        "payload": payload,
    }


# --------------------------------------------------------------------------- #
# input data compatibility
# --------------------------------------------------------------------------- #
FINGERPRINT_NAME = "source_fingerprint.json"


def dataset_files(dataset_dir: Path) -> list[Path]:
    files: list[Path] = []
    for name in ("transforms_train.json", "transforms_test.json", "points3D.ply", "points3D.txt"):
        cand = dataset_dir / name
        if cand.is_file():
            files.append(cand)
    for sub in ("images", "masks"):
        d = dataset_dir / sub
        if d.is_dir():
            files.extend(sorted(p for p in d.rglob("*") if p.is_file()))
    return files


def dataset_fingerprint(dataset_dir: Path, cache_path: Path | None = None) -> dict[str, Any]:
    """sha256 over the dataset payload; cached on (path,size,mtime)."""
    stat_sig = []
    for p in dataset_files(dataset_dir):
        st = p.stat()
        stat_sig.append((str(p.relative_to(dataset_dir)), st.st_size, int(st.st_mtime)))
    sig_hash = hashlib.sha256(json.dumps(stat_sig, sort_keys=True).encode()).hexdigest()
    if cache_path is not None and cache_path.is_file():
        cached = read_json(cache_path)
        if cached.get("signature") == sig_hash:
            return cached
    digests = {}
    for p in dataset_files(dataset_dir):
        digests[str(p.relative_to(dataset_dir))] = sha256_file(p)
    payload = {
        "dataset_dir": str(dataset_dir),
        "signature": sig_hash,
        "file_count": len(digests),
        "total_bytes": sum(p.stat().st_size for p in dataset_files(dataset_dir)),
        "files": digests,
    }
    if cache_path is not None:
        write_json(cache_path, payload)
    return payload


def check_input_data(
    manifest: dict[str, Any],
    scene: str,
    stage_dir: Path,
    cfg: dict[str, Any],
    eval_dir: Path,
) -> dict[str, Any]:
    """Confirm the stage was trained on exactly this local dataset copy."""
    checks: dict[str, Any] = {}
    dataset_dir = scene_dataset_dir(manifest, scene)
    checks["dataset_dir_present"] = dataset_dir.is_dir()
    checks["cfg_source_path"] = cfg.get("source_path")
    checks["cfg_source_path_basename_matches"] = (
        Path(str(cfg.get("source_path", ""))).name == scene and dataset_dir.is_dir()
    )
    checks["cfg_eval_split_enabled"] = bool(cfg.get("eval"))
    checks["cfg_resolution"] = cfg.get("resolution")
    checks["cfg_kernel_size"] = cfg.get("kernel_size")
    checks["cfg_appearance_enabled"] = bool(cfg.get("appearance_enabled"))

    train = read_json(dataset_dir / "transforms_train.json")
    test = read_json(dataset_dir / "transforms_test.json")
    checks["train_frames"] = len(train["frames"])
    checks["test_frames"] = len(test["frames"])
    checks["test_frame_files"] = [f["file_path"] for f in test["frames"]]
    checks["all_frame_files_present"] = all(
        (dataset_dir / str(f["file_path"]).lstrip("./")).is_file() for f in list(train["frames"]) + list(test["frames"])
    )

    fingerprint = dataset_fingerprint(dataset_dir, eval_dir / FINGERPRINT_NAME)
    checks["fingerprint_signature"] = fingerprint["signature"]
    checks["fingerprint_file_count"] = fingerprint["file_count"]
    checks["fingerprint_total_bytes"] = fingerprint["total_bytes"]

    # Cross-check whatever provenance the archive recorded.
    recorded = {}
    for name in ("metrics.json", "dataset_compatibility.json"):
        cand = stage_dir / name
        if not cand.is_file():
            continue
        try:
            payload = read_json(cand)
        except Exception:
            continue
        if isinstance(payload, dict) and "dataset_compatibility" in payload:
            recorded = payload["dataset_compatibility"]
            break
    checks["archive_dataset_compatibility"] = recorded
    mismatch: list[str] = []
    if isinstance(recorded, dict):
        raw = recorded.get("files") or recorded.get("file_md5") or recorded.get("md5") or {}
        if isinstance(raw, dict):
            for rel, digest in raw.items():
                local = dataset_dir / str(rel).lstrip("./")
                if not local.is_file():
                    mismatch.append(f"{rel}: missing locally")
                    continue
                if isinstance(digest, str) and len(digest) == 64:
                    if sha256_file(local) != digest:
                        mismatch.append(f"{rel}: sha256 mismatch")
                elif isinstance(digest, str) and len(digest) == 32:
                    if hashlib.md5(local.read_bytes()).hexdigest() != digest:
                        mismatch.append(f"{rel}: md5 mismatch")
    checks["recorded_digest_mismatches"] = mismatch

    ok = bool(
        checks["dataset_dir_present"]
        and checks["cfg_source_path_basename_matches"]
        and checks["cfg_eval_split_enabled"]
        and checks["all_frame_files_present"]
        and not mismatch
    )
    checks["compatible"] = ok
    checks["resolution_definition"] = (
        "stage cfg_args resolution maps to the native held-out raster: resolution=1 -> 2048x2048 for JAX scenes"
    )
    return checks


# --------------------------------------------------------------------------- #
# faithful checkpoint + PLY loading (CPU checkpoint, GPU rendering state)
# --------------------------------------------------------------------------- #
class LoadedModel:
    def __init__(self, gaussians: GaussianModel, dataset: Any, pipe: Any, scene: Scene, iteration: int):
        self.gaussians = gaussians
        self.dataset = dataset
        self.pipe = pipe
        self.scene = scene
        self.iteration = iteration


def build_dataset_args(cfg: dict[str, Any], model_path: str, local_source_path: Path | None = None) -> tuple[Any, Any, dict[str, Any]]:
    parser = argparse.ArgumentParser(add_help=False)
    mp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    args = parser.parse_args([])
    for key, value in cfg.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.model_path = model_path
    if local_source_path is not None:
        args.source_path = str(local_source_path)
    dataset = mp.extract(args)
    pipe = pp.extract(args)
    return dataset, pipe, vars(args)


def load_archive_model(
    manifest: dict[str, Any],
    scene: str,
    stage_dir: Path,
    device: torch.device,
    do_hash: bool = True,
) -> tuple[LoadedModel, dict[str, Any]]:
    """Load the archived checkpoint + matching PLY faithfully, GPU-lean."""
    checkpoint, ply_path, iteration = find_final_assets(stage_dir)
    cfg = parse_cfg_args(stage_dir / "cfg_args")
    local_source = scene_dataset_dir(manifest, scene)
    if not local_source.is_dir():
        raise FileNotFoundError(f"local dataset for {scene} not found at {local_source}")
    dataset, pipe, extracted = build_dataset_args(cfg, str(stage_dir), local_source)

    report: dict[str, Any] = {
        "checkpoint": file_identity(checkpoint, do_hash),
        "ply": file_identity(ply_path, do_hash),
        "iteration": iteration,
        "cfg_args": {k: v for k, v in cfg.items() if k != "model_path"},
        "dataset_args": {
            "source_path": dataset.source_path,
            "archived_source_path": cfg.get("source_path"),
            "source_path_remapped_to_local": cfg.get("source_path") != dataset.source_path,
            "resolution": dataset.resolution,
            "kernel_size": dataset.kernel_size,
            "sh_degree": dataset.sh_degree,
            "appearance_enabled": dataset.appearance_enabled,
            "appearance_n_fourier_freqs": dataset.appearance_n_fourier_freqs,
            "appearance_embedding_dim": dataset.appearance_embedding_dim,
            "white_background": dataset.white_background,
            "data_device": dataset.data_device,
            "eval": dataset.eval,
        },
        "checkpoint_load_map_location": "cpu",
        "optimizer_state_loaded": False,
    }

    started = time.perf_counter()
    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs,
        dataset.appearance_embedding_dim,
    )
    # filter_3D lives only in the PLY; load it first, before the checkpoint tensors.
    gaussians.load_ply(str(ply_path))
    report["ply_has_filter_3D"] = True
    filter_cpu = gaussians.filter_3D
    report["filter_3D"] = {
        "shape": list(filter_cpu.shape),
        "finite": bool(torch.isfinite(filter_cpu).all().item()),
        "min": float(filter_cpu.min().item()),
        "max": float(filter_cpu.max().item()),
    }

    model_params, ckpt_iter = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if int(ckpt_iter) != iteration:
        raise RuntimeError(f"checkpoint iteration {ckpt_iter} != resolved iteration {iteration}")
    if not isinstance(model_params, (tuple, list)) or len(model_params) != 15:
        raise RuntimeError(f"unexpected capture() arity in {checkpoint}: {len(model_params)}")

    (
        active_sh_degree,
        p_xyz,
        p_features_dc,
        p_features_rest,
        p_scaling,
        p_rotation,
        p_opacity,
        p_embeddings,
        p_appearance_embeddings,
        p_appearance_mlp,
        p_max_radii2D,
        p_xyz_gradient_accum,
        p_denom,
        p_opt_state,
        spatial_lr_scale,
    ) = model_params

    report["checkpoint_contents"] = {
        "active_sh_degree": int(active_sh_degree),
        "num_gaussians": int(p_xyz.shape[0]),
        "has_appearance_embeddings": p_appearance_embeddings is not None,
        "appearance_embeddings_shape": list(p_appearance_embeddings.shape) if p_appearance_embeddings is not None else None,
        "has_appearance_mlp": p_appearance_mlp is not None,
        "has_embeddings": p_embeddings is not None,
        "embeddings_shape": list(p_embeddings.shape) if p_embeddings is not None else None,
        "spatial_lr_scale": float(spatial_lr_scale),
        "optimizer_state_param_groups": len(p_opt_state.get("param_groups", [])) if isinstance(p_opt_state, dict) else None,
        "training_only_tensors_dropped": [
            "max_radii2D",
            "xyz_gradient_accum",
            "denom",
            "optimizer.state_dict()",
        ],
    }

    ply_count = int(filter_cpu.shape[0])
    if ply_count != int(p_xyz.shape[0]):
        raise RuntimeError(f"PLY/checkpoint Gaussians disagree: PLY {ply_count} vs checkpoint {int(p_xyz.shape[0])}")

    # Assign without autograd, keep everything on CPU, then move only what renders.
    def _hold(t: torch.Tensor | None) -> torch.Tensor | None:
        if t is None:
            return None
        return t.detach().to(torch.float32).requires_grad_(False)

    gaussians._xyz = _hold(p_xyz)
    gaussians._features_dc = _hold(p_features_dc)
    gaussians._features_rest = _hold(p_features_rest)
    gaussians._scaling = _hold(p_scaling)
    gaussians._rotation = _hold(p_rotation)
    gaussians._opacity = _hold(p_opacity)
    gaussians._embeddings = _hold(p_embeddings)
    gaussians.appearance_embeddings = _hold(p_appearance_embeddings)
    if p_appearance_mlp is not None:
        gaussians.appearance_mlp = p_appearance_mlp
    gaussians.active_sh_degree = int(active_sh_degree)
    gaussians.max_sh_degree = dataset.sh_degree
    gaussians.spatial_lr_scale = float(spatial_lr_scale)

    # The optimizer state and densification buffers are never moved to the GPU.
    n_opt_bytes = 0
    if isinstance(p_opt_state, dict):
        for state in p_opt_state.get("state", {}).values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    n_opt_bytes += value.numel() * value.element_size()
    del model_params, p_opt_state, p_max_radii2D, p_xyz_gradient_accum, p_denom
    gc.collect()
    report["optimizer_state_bytes_in_checkpoint"] = n_opt_bytes
    report["optimizer_state_bytes_skipped_mib"] = n_opt_bytes / 2**20

    # Now move the rendering-only state to the GPU.
    def _to_gpu(t: torch.Tensor | None) -> torch.Tensor | None:
        return None if t is None else t.to(device)

    for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity", "_embeddings"):
        setattr(gaussians, name, _to_gpu(getattr(gaussians, name)))
    gaussians.appearance_embeddings = _to_gpu(gaussians.appearance_embeddings)
    gaussians.filter_3D = _to_gpu(gaussians.filter_3D)
    if gaussians.appearance_mlp is not None:
        gaussians.appearance_mlp = gaussians.appearance_mlp.to(device).eval()
        for param in gaussians.appearance_mlp.parameters():
            param.requires_grad_(False)

    gpu_bytes = 0
    for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity", "_embeddings"):
        t = getattr(gaussians, name)
        if isinstance(t, torch.Tensor):
            gpu_bytes += t.numel() * t.element_size()
    if isinstance(gaussians.appearance_embeddings, torch.Tensor):
        gpu_bytes += gaussians.appearance_embeddings.numel() * gaussians.appearance_embeddings.element_size()
    gpu_bytes += gaussians.filter_3D.numel() * gaussians.filter_3D.element_size()
    if gaussians.appearance_mlp is not None:
        gpu_bytes += sum(p.numel() * p.element_size() for p in gaussians.appearance_mlp.parameters())
    report["gpu_rendering_state_bytes"] = gpu_bytes
    report["gpu_rendering_state_mib"] = gpu_bytes / 2**20

    if not bool(dataset.eval):
        raise RuntimeError("archived cfg_args has eval=False: no held-out split to score")

    scene_obj = Scene(
        dataset,
        gaussians,
        load_iteration=iteration,
        shuffle=False,
        ply_path=str(stage_dir),
    )
    train_cameras = list(scene_obj.getTrainCameras())
    test_cameras = list(scene_obj.getTestCameras())
    if not test_cameras:
        raise RuntimeError("held-out split is empty")
    if p_appearance_embeddings is not None and gaussians.appearance_embeddings.shape[0] != len(train_cameras):
        report["appearance_train_row_warning"] = (
            f"appearance_embeddings rows {int(gaussians.appearance_embeddings.shape[0])} != "
            f"train cameras {len(train_cameras)}"
        )
    report["train_cameras"] = len(train_cameras)
    report["heldout_cameras"] = len(test_cameras)
    report["load_seconds"] = time.perf_counter() - started
    report["gpu_mem_after_load"] = gpu_mem_mib()
    return LoadedModel(gaussians, dataset, pipe, scene_obj, iteration), report


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
@torch.no_grad()
def render_camera(model: LoadedModel, camera: Any, device: torch.device, testing: bool = True) -> dict[str, torch.Tensor]:
    background = torch.tensor(
        [1.0, 1.0, 1.0] if model.dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    )
    return render(
        camera,
        model.gaussians,
        model.pipe,
        background,
        model.dataset.kernel_size,
        testing=testing,
    )


def heldout_evaluate(
    model: LoadedModel,
    out_dir: Path,
    metrics: MetricStack,
    device: torch.device,
    mask_mode: str,
    save_renders: bool,
    on_first_view: Any = None,
) -> dict[str, Any]:
    per_view: list[dict[str, Any]] = []
    render_dir = out_dir / "renders" / "heldout"
    if save_renders:
        render_dir.mkdir(parents=True, exist_ok=True)
    for camera in model.scene.getTestCameras():
        pkg = render_camera(model, camera, device, testing=True)
        image = torch.clamp(pkg["render"], 0.0, 1.0)
        gt = torch.clamp(camera.original_image.to(device), 0.0, 1.0)
        if image.shape != gt.shape:
            raise RuntimeError(
                f"raster mismatch for {camera.image_name}: render {tuple(image.shape)} vs gt {tuple(gt.shape)}"
            )
        mask = None
        if hasattr(camera, "original_mask") and camera.original_mask.numel() > 1:
            raw_mask = camera.original_mask.to(device)
            if raw_mask.shape[-2:] == image.shape[-2:]:
                mask = raw_mask.to(torch.float32)
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
        scored_image, scored_gt = image, gt
        if mask_mode == "zero-fill" and mask is not None:
            scored_image, scored_gt = image * mask, gt * mask
        values = metrics.reference(scored_image, scored_gt)
        row = {
            "image_name": str(camera.image_name),
            "uid": int(camera.uid),
            "width": int(camera.image_width),
            "height": int(camera.image_height),
            "focal_x": float(camera.focal_x),
            **values,
            "mask_mode": mask_mode,
            "mask_coverage": float(mask.mean().item()) if mask is not None else None,
            "render_min": float(image.min().item()),
            "render_max": float(image.max().item()),
            "gt_min": float(gt.min().item()),
            "gt_max": float(gt.max().item()),
        }
        per_view.append(row)
        if save_renders:
            from PIL import Image

            Image.fromarray(tensor_to_uint8_hwc(image)).save(render_dir / f"{camera.image_name}_render.png")
            Image.fromarray(tensor_to_uint8_hwc(gt)).save(render_dir / f"{camera.image_name}_gt.png")
            diff = torch.clamp((image - gt).abs() * 4.0, 0.0, 1.0)
            Image.fromarray(tensor_to_uint8_hwc(diff)).save(render_dir / f"{camera.image_name}_absdiff_x4.png")
        del pkg, image, gt
        free_cuda()
        if on_first_view is not None and len(per_view) == 1:
            on_first_view(per_view[0])
    aggregate: dict[str, Any] = {"num_views": len(per_view)}
    for key in ("psnr", "ssim", "lpips", "l1"):
        vals = [row[key] for row in per_view if row.get(key) is not None]
        aggregate[key] = float(np.mean(vals)) if vals else None
        aggregate[f"{key}_std"] = float(np.std(vals)) if vals else None
    return {"per_view": per_view, "aggregate": aggregate, "mask_mode": mask_mode}


def reproduction_check(scene: str, stage: str, aggregate: dict[str, Any], psnr_tol: float, l1_tol: float) -> dict[str, Any]:
    expected = KNOWN_REPRODUCTION.get((scene, stage))
    if expected is None:
        return {"applicable": False, "reason": "no remote console reference recorded for this scene/stage"}
    obs_psnr, obs_l1 = aggregate.get("psnr"), aggregate.get("l1")
    out = {
        "applicable": True,
        "expected_psnr": expected["psnr"],
        "expected_l1": expected["l1"],
        "expected_iteration": expected["iteration"],
        "reference": expected["source"],
        "observed_psnr": obs_psnr,
        "observed_l1": obs_l1,
        "tolerance_psnr": psnr_tol,
        "tolerance_l1": l1_tol,
        "psnr_abs_diff": abs(obs_psnr - expected["psnr"]) if obs_psnr is not None else None,
        "l1_abs_diff": abs(obs_l1 - expected["l1"]) if obs_l1 is not None else None,
    }
    out["passed"] = bool(
        obs_psnr is not None
        and obs_l1 is not None
        and abs(obs_psnr - expected["psnr"]) <= psnr_tol
        and abs(obs_l1 - expected["l1"]) <= l1_tol
    )
    return out


# --------------------------------------------------------------------------- #
# fixed camera-path novel views (shared checked-in trajectories)
# --------------------------------------------------------------------------- #
def camera_path_files(manifest: dict[str, Any], scene: str) -> list[Path]:
    d = scene_camera_path_dir(manifest, scene)
    if not d.is_dir():
        return []
    return sorted(d.glob("*.json"))


@torch.no_grad()
def render_camera_path(
    model: LoadedModel,
    path_json: Path,
    out_dir: Path,
    device: torch.device,
    frame_subset: int,
    save_video: bool = True,
    keep_frames: bool = False,
) -> tuple[dict[str, Any], list[np.ndarray] | None]:
    from PIL import Image
    from render_video import get_path_from_json

    payload = read_json(path_json)
    cam_infos, radius = get_path_from_json(payload)
    cams = cameraList_from_camInfos(cam_infos, 1, model.dataset, is_testing=True)
    path_name = path_json.stem
    frames_dir = out_dir / "novel" / path_name
    frames_dir.mkdir(parents=True, exist_ok=True)
    kept: list[int] = []
    step = max(1, len(cams) // max(1, frame_subset))
    frames: list[np.ndarray] = []
    for idx, cam in enumerate(cams):
        pkg = render_camera(model, cam, device, testing=True)
        image = torch.clamp(pkg["render"], 0.0, 1.0)
        arr = tensor_to_uint8_hwc(image)
        frames.append(arr)
        if frame_subset > 0 and idx % step == 0 and len(kept) < frame_subset:
            Image.fromarray(arr).save(frames_dir / f"frame_{idx:05d}.png")
            kept.append(idx)
        del pkg, image
    video_rel = None
    if save_video and frames:
        try:
            import mediapy as media

            video_rel = f"novel/{path_name}.mp4"
            with media.VideoWriter(
                path=str(out_dir / video_rel),
                shape=frames[0].shape[:2],
                fps=float(payload.get("fps", 24)),
            ) as writer:
                for arr in frames:
                    writer.add_image(arr)
        except Exception as exc:  # reported, not silently dropped
            video_rel = None
            frames_dir.joinpath("_video_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
    meta = {
        "path": path_name,
        "path_json": str(path_json),
        "num_frames": len(cams),
        "render_width": int(payload["render_width"]),
        "render_height": int(payload["render_height"]),
        "fps": float(payload.get("fps", 24)),
        "radius": float(radius) if radius is not None else None,
        "frame_subset_indices": kept,
        "video": video_rel,
    }
    return meta, (frames if keep_frames else None)


def build_panels(archive_root: Path, scene: str, stages: Sequence[str]) -> list[str]:
    """Compose Stage1-vs-Stage2 (+GT) comparison grids from saved novel frames."""
    from PIL import Image

    scene_eval = archive_root / scene / "evaluation"
    frames_by_stage: dict[str, dict[str, list[Path]]] = {}
    for stage in stages:
        root = scene_eval / stage / "novel"
        if not root.is_dir():
            continue
        frames_by_stage[stage] = {
            p.name: sorted(p.glob("frame_*.png")) for p in sorted(root.iterdir()) if p.is_dir()
        }
    if len(frames_by_stage) < 2:
        return []
    panels: list[str] = []
    panel_dir = scene_eval / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    common = sorted(set(frames_by_stage[stages[0]]).intersection(*[set(frames_by_stage[s]) for s in stages[1:]]))
    for path_name in common:
        lists = [frames_by_stage[s][path_name] for s in stages]
        n = min(len(x) for x in lists)
        if n == 0:
            continue
        cols = list(range(n))
        tiles = []
        for ci in cols:
            row_imgs = [Image.open(lists[si][ci]).convert("RGB") for si in range(len(stages))]
            row = np.concatenate([np.asarray(im) for im in row_imgs], axis=1)
            tiles.append(row)
        grid = np.concatenate(tiles, axis=0)
        out = panel_dir / f"{scene}_{path_name}_stage1_vs_stage2.png"
        Image.fromarray(grid).save(out)
        panels.append(str(out.relative_to(archive_root / scene)))
    return panels


# --------------------------------------------------------------------------- #
# official external-view protocol
# --------------------------------------------------------------------------- #
def grayscale_sequence_from_video(path: Path, size: int = 64) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    return np.asarray(frames)


def grayscale_sequence_from_images(paths: Sequence[Path], size: int = 64) -> np.ndarray:
    import cv2

    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"cannot read {p}")
        frames.append(cv2.cvtColor(cv2.resize(img, (size, size)), cv2.COLOR_BGR2GRAY).astype(np.float32))
    return np.asarray(frames)


def grayscale_sequence_from_arrays(arrays: Sequence[np.ndarray], size: int = 64) -> np.ndarray:
    import cv2

    return np.asarray(
        [cv2.cvtColor(cv2.resize(a, (size, size)), cv2.COLOR_RGB2GRAY).astype(np.float32) for a in arrays]
    )


def _normalize_rows(seq: np.ndarray) -> np.ndarray:
    flat = seq.reshape(len(seq), -1)
    flat = flat - flat.mean(1, keepdims=True)
    return flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)


def verify_correspondence(gt_seq: np.ndarray, ref_seq: np.ndarray, window: int = 15) -> dict[str, Any]:
    """Check that a reference sequence follows the GT sequence index-for-index.

    Both sequences describe the same closed orbit; the expected map is linear with
    wraparound.  A frame is accepted when the best normalized-correlation match
    inside the expected window agrees with the windowed optimum, and when no
    frame outside the window is a better match.
    """
    n, m = len(gt_seq), len(ref_seq)
    if n == 0 or m == 0:
        return {"verified": False, "reason": "empty sequence"}
    g = _normalize_rows(gt_seq)
    r = _normalize_rows(ref_seq)
    corr = g @ r.T
    expected = np.arange(n) * m / n
    offsets, windowed_scores, global_agreement = [], [], 0
    for i in range(n):
        j0 = int(round(expected[i]))
        cand = np.arange(j0 - window, j0 + window + 1) % m
        windowed = corr[i, cand]
        best_local = int(cand[int(windowed.argmax())])
        best_global = int(corr[i].argmax())
        offsets.append(int(((best_local - expected[i] + m / 2) % m) - m / 2))
        windowed_scores.append(float(windowed.max()))
        if best_global in set(cand.tolist()):
            global_agreement += 1
    offsets_arr = np.asarray(offsets)
    agreement = global_agreement / n
    abs_off = np.abs(offsets_arr)
    mean_abs = float(abs_off.mean())
    p95_abs = float(np.percentile(abs_off, 95))
    # Calibrated against the released official pairs and against roll/shuffle controls:
    # short, near-static NYC orbits carry little photometric signal between adjacent
    # frames, so the frame-offset criterion (relative to the orbit length) does the work
    # there while the correlation criteria dominate on the 300-frame JAX orbits.
    mean_limit = max(1.0, 0.02 * m)
    p95_limit = max(3.0, 0.05 * m)
    monotone = bool(mean_abs <= mean_limit and p95_abs <= p95_limit)
    mean_corr = float(np.mean(windowed_scores))
    agreement_min = 0.85
    verified = bool(monotone and agreement >= agreement_min and mean_corr >= 0.10)
    return {
        "verified": verified,
        "gt_frames": n,
        "reference_frames": m,
        "window_radius": window,
        "mean_frame_offset": float(offsets_arr.mean()),
        "mean_abs_frame_offset": mean_abs,
        "p95_abs_frame_offset": p95_abs,
        "max_abs_frame_offset": float(abs_off.max()),
        "global_argmax_inside_window_fraction": float(agreement),
        "mean_windowed_correlation": mean_corr,
        "min_windowed_correlation": float(np.min(windowed_scores)),
        "criteria": {
            "identity_linear_map": monotone,
            "mean_abs_offset_limit": mean_limit,
            "p95_abs_offset_limit": p95_limit,
            f"agreement_fraction>={agreement_min:.2f}": bool(agreement >= agreement_min),
            "mean_correlation>=0.10": bool(mean_corr >= 0.10),
        },
        "offsets_preview": [int(v) for v in offsets_arr[:: max(1, n // 16)]],
    }


def relative_sample_indices(n_total: int, k: int) -> list[int]:
    if n_total <= 0:
        return []
    if n_total <= k:
        return list(range(n_total))
    return sorted({int(round(v)) for v in np.linspace(0, n_total - 1, k)})


def read_video_frames(path: Path) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def patchify(image: np.ndarray, patch_size: int = PATCH_SIZE, min_patches: tuple[int, int] = PATCH_MIN) -> list[np.ndarray]:
    """Identical to eval.py::patchify (the released evaluation convention).

    A raster smaller than one patch cannot be tiled; such an image is emitted whole
    and the caller reports it, rather than silently yielding zero samples.
    """
    height, width = image.shape[:2]
    patch_height = patch_width = patch_size
    if height < patch_height or width < patch_width:
        return [image]
    min_h, min_w = min_patches
    h_stride = max(1, (height - patch_height) // max(min_h - 1, 1))
    w_stride = max(1, (width - patch_width) // max(min_w - 1, 1))
    stride = min(h_stride, w_stride)
    num_h = max(1, (height - patch_height) // stride + 1)
    num_w = max(1, (width - patch_width) // stride + 1)
    if num_h < min_h or num_w < min_w:
        h_stride = (height - patch_height) / max(min_h - 1, 1)
        w_stride = (width - patch_width) / max(min_w - 1, 1)
        patches = []
        for y in [int(i * h_stride) for i in range(min_h)]:
            for x in [int(i * w_stride) for i in range(min_w)]:
                y_end, x_end = min(y + patch_height, height), min(x + patch_width, width)
                y, x = max(0, y_end - patch_height), max(0, x_end - patch_width)
                patch = image[y:y_end, x:x_end]
                if patch.shape[0] == patch_height and patch.shape[1] == patch_width:
                    patches.append(patch)
        return patches
    patches = []
    for i in range(num_h):
        for j in range(num_w):
            y, x = i * stride, j * stride
            if y + patch_height <= height and x + patch_width <= width:
                patches.append(image[y : y + patch_height, x : x + patch_width])
    return patches


def write_patch_dir(frames: Sequence[np.ndarray], out_dir: Path) -> int:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for fi, frame in enumerate(frames):
        for pi, patch in enumerate(patchify(frame)):
            cv2.imwrite(str(out_dir / f"f{fi:03d}_p{pi:03d}.png"), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            count += 1
    return count


def distributional_metrics(gt_frames: Sequence[np.ndarray], our_frames: Sequence[np.ndarray], work_dir: Path) -> dict[str, Any]:
    """CLIP-FID + CMMD on patchified frames (same convention as the repo eval.py)."""
    import cv2

    out: dict[str, Any] = {
        "scope": "distributional (no frame pairing required); patchified 512 with min_patches=(9,16)",
        "frame_count_per_side": [len(gt_frames), len(our_frames)],
    }
    patch_root = work_dir / "patchify"
    if patch_root.exists():
        shutil.rmtree(patch_root)
    try:
        gt_dir = patch_root / "gt"
        our_dir = patch_root / "ours"
        out["gt_patch_count"] = write_patch_dir(gt_frames, gt_dir)
        out["ours_patch_count"] = write_patch_dir(our_frames, our_dir)

        try:
            from cleanfid import fid

            out["clip_fid"] = float(fid.compute_fid(str(gt_dir), str(our_dir), mode="clean", model_name="clip_vit_b_32"))
            out["clip_fid_status"] = "computed"
        except Exception as exc:
            out["clip_fid"] = None
            out["clip_fid_status"] = "unavailable"
            out["clip_fid_error"] = f"{type(exc).__name__}: {exc}"

        try:
            from cmmd_pytorch.main import compute_cmmd

            out["cmmd"] = float(compute_cmmd(str(gt_dir), str(our_dir), batch_size=32))
            out["cmmd_status"] = "computed"
        except Exception as exc:
            out["cmmd"] = None
            out["cmmd_status"] = "unavailable"
            out["cmmd_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # Consumed intermediate: the patch rasters are never retained.
        for d in (patch_root / "gt", patch_root / "ours"):
            shutil.rmtree(d, ignore_errors=True)
    return out


def official_external_view(
    manifest: dict[str, Any],
    scene: str,
    stage: str,
    our_frames: list[np.ndarray],
    our_path_name: str,
    metrics: MetricStack,
    device: torch.device,
    out_dir: Path,
    work_dir: Path,
    frames_per_side: int = OFFICIAL_FRAMES,
    allow_inferred_paired: bool = False,
) -> dict[str, Any]:
    """Official external-view protocol on the released Skyfall-GS eval data.

    Three distinct scopes, never conflated:

    * ``distributional`` — CLIP-FID/CMMD between our rendered frames and the released GT
      frames.  Requires no frame pairing, so it is always reportable when GT exists.
    * ``paired_vs_released_render`` — paired PSNR/SSIM/LPIPS against the *released render*
      of the same method and stage.  This pairing is provable by construction: both our
      render and the released render are produced by the same pipeline from the same
      checked-in camera path, so frame i of each is camera i.
    * ``paired_vs_ground_truth`` — paired metrics against the released GT video.  The
      released data publishes no GT-to-camera index contract (the GT is a merged
      reference video), so this pairing is **inference from appearance, not camera
      truth**.  It is therefore reported only as explicitly-labelled inferred evidence
      and is never presented as verified physical correspondence.
    """
    scene_dir = ensure_eval_data(manifest, scene)
    result: dict[str, Any] = {
        "scene": scene,
        "stage": stage,
        "our_trajectory": our_path_name,
        "official_eval_dir": str(scene_dir) if scene_dir else None,
        "scope_definitions": {
            "distributional": "CLIP-FID + CMMD; no frame pairing required",
            "paired_vs_released_render": (
                "paired PSNR/SSIM/LPIPS against the released render of the same method/stage; "
                "pairing provable by construction (same checked-in camera path, same pipeline)"
            ),
            "paired_vs_ground_truth": (
                "paired PSNR/SSIM/LPIPS against the released GT video; no published "
                "GT-to-camera index contract exists, so this is appearance-inferred only"
            ),
        },
    }
    if scene_dir is None or not (scene_dir / "GT").is_dir():
        result["status"] = "no_official_gt"
        result["reason"] = (
            f"official Skyfall-GS eval data has no {scene} ground truth (released coverage is "
            "JAX_004/068/214/260 and NYC_004/010/219/336 only; checked "
            "https://huggingface.co/datasets/jayinnn/Skyfall-GS-eval and the local zips)"
        )
        result["paired_vs_ground_truth"] = None
        result["paired_vs_released_render"] = None
        result["distributional"] = None
        return result

    gt_dir = scene_dir / "GT"
    gt_videos = sorted(gt_dir.glob("*.mp4"))
    gt_images = sorted(p for p in gt_dir.glob("*.jpg"))
    method_dirs = [d for d in scene_dir.iterdir() if d.is_dir() and d.name != "GT"]
    released_videos = sorted(v for d in method_dirs for v in d.glob(f"{our_path_name}.mp4"))
    released_same_stage = [v for v in released_videos if v.parent.name == f"ours_{stage}"]

    if gt_videos:
        gt_path = gt_videos[0]
        gt_frames_all = read_video_frames(gt_path)
        gt_seq = grayscale_sequence_from_video(gt_path)
        gt_kind = "video"
        gt_source = str(gt_path)
    elif gt_images:
        gt_frames_all = [cv2_imread_rgb(p) for p in gt_images]
        gt_seq = grayscale_sequence_from_images(gt_images)
        gt_kind = "image_sequence"
        gt_source = f"{len(gt_images)} frames under {gt_dir}"
    else:
        result["status"] = "no_official_gt"
        result["reason"] = f"no GT videos or images under {gt_dir}"
        return result

    result["gt"] = {
        "kind": gt_kind,
        "source": gt_source,
        "frame_count": len(gt_frames_all),
        "resolution": [int(gt_frames_all[0].shape[1]), int(gt_frames_all[0].shape[0])],
        "note": "official released reference for this trajectory; never used as our own output",
    }
    result["released_references"] = [str(v) for v in released_videos]

    # --- correspondence evidence (supporting, never a camera-truth claim) ---
    our_seq = grayscale_sequence_from_arrays(our_frames)
    checks: dict[str, Any] = {}
    if released_same_stage:
        ref_seq = grayscale_sequence_from_video(released_same_stage[0])
        checks["ours_vs_released_same_stage"] = verify_correspondence(ref_seq, our_seq)
        checks["ours_vs_released_same_stage"]["reference"] = str(released_same_stage[0])
        checks["gt_vs_released_same_stage"] = verify_correspondence(gt_seq, ref_seq)
        checks["gt_vs_released_same_stage"]["reference"] = str(released_same_stage[0])
    checks["gt_vs_ours"] = verify_correspondence(gt_seq, our_seq)
    checks["note"] = (
        "evidence class: appearance-correlation inference. It cannot prove camera pose or "
        "GT-to-camera frame indexing, because the released evaluation data publishes no such "
        "contract. Recorded so a reader can judge the pairing, not to assert physical truth. "
        "No released other-method video is ever used as our rendered output."
    )
    result["correspondence_evidence"] = checks
    result["camera_frame_contract_available"] = False
    result["camera_frame_contract_note"] = (
        "the released eval archive contains only mp4/jpg sequences with no per-frame camera "
        "index mapping; therefore GT pairing is not provable from camera truth"
    )
    appearance_agrees = bool((checks.get("gt_vs_ours") or {}).get("verified"))

    gt_idx = relative_sample_indices(len(gt_frames_all), frames_per_side)
    our_idx = relative_sample_indices(len(our_frames), frames_per_side)
    k = min(len(gt_idx), len(our_idx))
    result["sampling"] = {
        "frames_per_side": frames_per_side,
        "paired_indexing": "relative position: sample i of ours is compared with sample i of the GT",
        "controlled_sample_counts": {
            "gt_frames_available": len(gt_frames_all),
            "our_frames_available": len(our_frames),
            "gt_samples_used": k,
            "our_samples_used": k,
        },
        "gt_indices": gt_idx[:k],
        "our_indices": our_idx[:k],
    }
    gt_samples = [gt_frames_all[i] for i in gt_idx[:k]]
    our_samples = [our_frames[i] for i in our_idx[:k]]

    def paired_rows(ref_frames: Sequence[np.ndarray], ref_label: str) -> list[dict[str, Any]]:
        rows = []
        for i, (ourf, reff) in enumerate(zip(our_samples, ref_frames[:k])):
            t_ref = torch.tensor(np.asarray(reff).transpose(2, 0, 1)).float().to(device) / 255.0
            t_our = torch.tensor(ourf.transpose(2, 0, 1)).float().to(device) / 255.0
            if t_ref.shape != t_our.shape:
                import torch.nn.functional as F

                t_our = F.interpolate(
                    t_our.unsqueeze(0), size=t_ref.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze(0)
            values = metrics.reference(t_our.clamp(0, 1), t_ref.clamp(0, 1))
            rows.append({"sample_index": i, "our_index": our_idx[i], "reference": ref_label, **values})
            del t_ref, t_our
            free_cuda()
        return rows

    def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        agg: dict[str, Any] = {"num_pairs": len(rows)}
        for key in ("psnr", "ssim", "lpips", "l1"):
            vals = [r[key] for r in rows if r.get(key) is not None]
            agg[key] = float(np.mean(vals)) if vals else None
            agg[f"{key}_std"] = float(np.std(vals)) if vals else None
        return agg

    # --- paired vs the released render of the same stage (provable pairing) ---
    if not released_same_stage:
        result["paired_vs_released_render"] = {
            "status": "not_computed",
            "reason": f"the released archive has no ours_{stage} render for trajectory {our_path_name}",
        }
    else:
        ref_video_frames = read_video_frames(released_same_stage[0])
        ref_idx = relative_sample_indices(len(ref_video_frames), frames_per_side)
        refv = [ref_video_frames[i] for i in ref_idx[:k]]
        rows = paired_rows(refv, str(released_same_stage[0]))
        result["paired_vs_released_render"] = {
            "status": "computed",
            "evidence_class": "provable_by_construction",
            "reference": str(released_same_stage[0]),
            "reference_is_ground_truth": False,
            "pixel_domain": "released render raster; our render resized to it when rasters differ",
            "aggregate": aggregate(rows),
            "per_pair": rows,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / "official_paired_vs_released_render.csv", rows)

    # --- paired vs the released GT (appearance-inferred only) ---
    if not allow_inferred_paired:
        result["paired_vs_ground_truth"] = {
            "status": "not_reported",
            "reason": (
                "the released evaluation data provides no camera/frame contract mapping GT video "
                "frames to trajectory cameras, so paired GT metrics cannot be proven; only the "
                "distributional metrics are reported. Pass --official-inferred-paired to record the "
                "appearance-inferred numbers explicitly labelled as unverified."
            ),
            "appearance_evidence_agrees": appearance_agrees,
            "correspondence_evidence": "correspondence_evidence.gt_vs_ours",
        }
    else:
        rows = paired_rows(gt_samples, gt_source)
        result["paired_vs_ground_truth"] = {
            "status": "inferred_unverified",
            "evidence_class": "appearance_inferred",
            "verified_by_camera_contract": False,
            "warning": (
                "DO NOT cite as verified correspondence. Frame pairing rests on appearance "
                "correlation between our render and a merged reference video, not on camera truth."
            ),
            "pixel_domain": "released GT raster; our render resized to it when rasters differ",
            "aggregate": aggregate(rows),
            "per_pair": rows,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / "official_paired_vs_ground_truth_inferred.csv", rows)

    result["distributional"] = distributional_metrics(gt_samples, our_samples, work_dir)
    if result["distributional"].get("clip_fid") is None and result["distributional"].get("cmmd") is None:
        result["status"] = "distributional_unavailable"
    else:
        result["status"] = "reported"
    result["scope_summary"] = {
        "distributional_reported": result["distributional"].get("clip_fid") is not None
        or result["distributional"].get("cmmd") is not None,
        "paired_vs_released_render_reported": (result.get("paired_vs_released_render") or {}).get("status") == "computed",
        "paired_vs_ground_truth_reported": (result.get("paired_vs_ground_truth") or {}).get("status")
        == "inferred_unverified",
        "paired_vs_ground_truth_camera_verified": False,
    }
    write_json(out_dir / "external_metrics.json", _json_ready(result))
    return result


def cv2_imread_rgb(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# prerequisites
# --------------------------------------------------------------------------- #
def check_prerequisites(metrics: MetricStack, manifest: dict[str, Any], need_official: bool) -> list[dict[str, Any]]:
    prereqs: list[dict[str, Any]] = list(metrics.prerequisites)
    if need_official:
        try:
            import cleanfid  # noqa: F401

            try:
                from cleanfid import fid

                fid.get_folder_features  # noqa: B018
            except Exception:
                pass
        except Exception as exc:
            prereqs.append({"metric": "clip_fid", "status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"})
        try:
            import transformers  # noqa: F401

            from transformers import CLIPVisionModelWithProjection

            model_id = "openai/clip-vit-large-patch14-336"
            try:
                CLIPVisionModelWithProjection.from_pretrained(model_id)
            except Exception as exc:
                prereqs.append(
                    {
                        "metric": "cmmd",
                        "status": "needs_download",
                        "reason": f"{model_id} not loadable: {type(exc).__name__}: {exc}",
                    }
                )
        except Exception as exc:
            prereqs.append({"metric": "cmmd", "status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"})
    return prereqs


# --------------------------------------------------------------------------- #
# per (scene, stage) driver
# --------------------------------------------------------------------------- #
def build_status_payload(
    scene: str,
    stage: str,
    load_report: dict[str, Any],
    input_checks: dict[str, Any],
    heldout: dict[str, Any] | None,
    repro: dict[str, Any],
    novel_meta: Sequence[dict[str, Any]],
    official_status: str,
    official_file: str | None,
    fused: dict[str, Any] | None,
    prerequisites: Sequence[dict[str, Any]],
    timings: dict[str, float],
    status: str,
    eval_dir: Path,
    released_for_cleanup: bool,
) -> dict[str, Any]:
    """One status schema for both the early `running` and the final `completed` marker."""
    agg = (heldout or {}).get("aggregate") or {}
    cleared = bool(released_for_cleanup and input_checks["compatible"] and not repro.get("failed"))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "scene": scene,
        "stage": stage,
        "evaluator": "DatasetEvaluator",
        "updated_at": now_iso(),
        "model_load_verified": True,
        "input_data_compatible": bool(input_checks["compatible"]),
        "checkpoint_load_map_location": "cpu",
        "optimizer_state_loaded": False,
        "iteration": load_report["iteration"],
        "num_gaussians": load_report["checkpoint_contents"]["num_gaussians"],
        "checkpoint": load_report["checkpoint"]["path"],
        "checkpoint_sha256": load_report["checkpoint"].get("sha256"),
        "point_cloud": load_report["ply"]["path"],
        "point_cloud_sha256": load_report["ply"].get("sha256"),
        "ply_has_filter_3D": bool(load_report["ply_has_filter_3D"]),
        "appearance_state_loaded": bool(load_report["checkpoint_contents"]["has_appearance_embeddings"]),
        "heldout_cameras": load_report["heldout_cameras"],
        "heldout_aggregate": agg or None,
        "heldout_per_view_file": str(eval_dir / "heldout_metrics.json") if heldout else None,
        "heldout_per_view_csv": str(eval_dir / "heldout_metrics.csv") if heldout else None,
        "reproduction_check": repro,
        "novel_views": [m["path"] for m in novel_meta],
        "official_external_view_status": official_status,
        "official_external_view_file": official_file,
        "fused_ply": fused["path"] if fused else None,
        "prerequisites": list(prerequisites),
        "remote_deletion_cleared": cleared,
        "cleanup_gate": {
            "requires": "archive_status.json status=verified AND evaluation_status.json model_load_verified=true",
            "released": cleared,
        },
        "evidence": str(eval_dir / "evaluation.json"),
        "gpu_mem_mib": gpu_mem_mib(),
        "timings_seconds": timings,
    }
    return payload


def protocol_fingerprint(
    args: argparse.Namespace,
    checkpoint_id: dict[str, Any],
    ply_id: dict[str, Any],
    iteration: int | None,
    num_gaussians: int | None,
) -> dict[str, Any]:
    """Identity of what was actually requested and of the exact model assets.

    A cached `completed` marker only satisfies a new request when this fingerprint
    matches, so a satellite-only or earlier-protocol result can never stand in for the
    current full evaluation, and a re-archived/replaced checkpoint is re-evaluated.
    Asset identity comes from ArchiveCurator's verified sha256 digests in
    archive_status.json, so the check stays cheap (no re-hashing of multi-GB files).
    """

    def asset(entry: dict[str, Any] | None) -> dict[str, Any]:
        if not entry:
            return {"missing": True}
        return {
            "sha256": entry.get("sha256"),
            "size": entry.get("size"),
            "mtime_ns": entry.get("mtime_ns"),
        }

    requested = {
        "official": bool(args.official),
        "official_frames": int(args.official_frames),
        "official_inferred_paired": bool(args.official_inferred_paired),
        "novel": not bool(args.no_novel),
        "novel_frames": int(args.novel_frames),
        "max_paths": int(args.max_paths),
        "video": not bool(args.no_video),
        "renders": not bool(args.no_renders),
        "fused_ply": bool(args.fused_ply),
        "panels": not bool(args.no_panels),
        "mask_mode": str(args.mask_mode),
    }
    payload = {
        "checkpoint": asset(checkpoint_id),
        "point_cloud": asset(ply_id),
        "iteration": iteration,
        "num_gaussians": num_gaussians,
        "requested": requested,
        "schema": 1,
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    return payload


def archived_asset_ids(stage_dir: Path, stage: str) -> dict[str, dict[str, Any]]:
    """Verified model-asset digests from ArchiveCurator's marker (authoritative, cheap)."""
    marker = stage_dir / "archive_status.json"
    if not marker.is_file():
        return {}
    try:
        payload = read_json(marker)
    except Exception:
        return {}
    artifacts = payload.get("model_artifacts") or {}
    out: dict[str, dict[str, Any]] = {}
    for key in ("checkpoint", "point_cloud"):
        entry = artifacts.get(key)
        if isinstance(entry, dict):
            out[key] = {
                "sha256": entry.get("sha256"),
                "size": entry.get("bytes"),
                "path": entry.get("path"),
                "iteration": entry.get("iteration"),
            }
    return out


def fingerprint_assets(
    stage_dir: Path,
    stage: str,
    prior: dict[str, Any] | None = None,
    loaded: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Resolve the fingerprint identity of the archived checkpoint + PLY.

    One resolver is used both when storing and when checking a cached result, so the
    two fingerprints are always comparable. sha256 is taken from the verified archive
    marker (or an earlier evaluation) when known; size/mtime always describe the file
    currently on disk, which catches a re-archived checkpoint even without hashing.
    """
    checkpoint, ply_path, iteration = find_final_assets(stage_dir)
    ids = archived_asset_ids(stage_dir, stage)

    def ident(path: Path, verified: dict[str, Any], loaded_entry: dict[str, Any] | None) -> dict[str, Any]:
        known = verified.get("sha256")
        size = verified.get("size")
        if known is None and loaded_entry:
            known = loaded_entry.get("sha256")
        if size is None and loaded_entry:
            size = loaded_entry.get("size")
        stat = path.stat()
        return {
            "sha256": known,
            "size": size if size is not None else stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    ck_loaded = (loaded or {}).get("checkpoint")
    ply_loaded = (loaded or {}).get("ply")
    ck_id = ident(checkpoint, ids.get("checkpoint") or {}, ck_loaded)
    ply_id = ident(ply_path, ids.get("point_cloud") or {}, ply_loaded)
    if prior:
        ck_id["sha256"] = ck_id["sha256"] or prior.get("checkpoint_sha256")
        ply_id["sha256"] = ply_id["sha256"] or prior.get("point_cloud_sha256")
    return ck_id, ply_id, iteration


def classify_status(
    load_report: dict[str, Any],
    input_checks: dict[str, Any],
    heldout: dict[str, Any],
    official: dict[str, Any] | None,
    official_path_selected: bool,
    prerequisites: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    official_gt_exists: bool,
) -> dict[str, Any]:
    """`completed` only when every requested, reachable deliverable actually landed."""
    blockers: list[str] = []
    failures: list[str] = []
    unreachable: list[str] = []

    if not input_checks.get("compatible"):
        failures.append("input data compatibility check failed")
    agg = heldout.get("aggregate") or {}
    if agg.get("num_views", 0) < 1:
        failures.append("held-out split produced no scored views")
    for key in ("psnr", "ssim", "l1"):
        if agg.get(key) is None:
            failures.append(f"required held-out metric {key} is missing")
    if agg.get("lpips") is None:
        blockers.append("lpips unavailable (prerequisite missing); held-out metrics incomplete")

    if args.official:
        if not official_gt_exists:
            unreachable.append(
                "released official eval data has no ground truth for this scene (recorded explicitly); "
                "held-out satellite metrics and our own fixed-trajectory renders are the deliverables"
            )
        elif not official_path_selected:
            failures.append(
                "official ground truth exists for this scene but no checked-in camera path matches the "
                "released evaluation trajectory, so the official protocol could not run"
            )
        elif official is None:
            failures.append("official external-view protocol produced no result")
        else:
            distrib = official.get("distributional") or {}
            if distrib.get("clip_fid") is None and distrib.get("cmmd") is None:
                blockers.append("CLIP-FID and CMMD both unavailable; distributional scope incomplete")
            paired_render = (official.get("paired_vs_released_render") or {}).get("status")
            if paired_render == "not_computed":
                unreachable.append("released render of this stage absent; released-render pairing not computable")

    if failures:
        status = "failed"
    elif blockers:
        status = "partial"
    else:
        status = STATUS_COMPLETED
    return {
        "status": status,
        "failures": failures,
        "blockers": blockers,
        "unreachable_deliverables": unreachable,
        "note": (
            "completed requires every requested reachable deliverable to have succeeded; "
            "missing prerequisites or failed requested metrics yield partial/failed, never completed"
        ),
    }


def evaluate_stage(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    scene: str,
    stage: str,
    device: torch.device,
) -> dict[str, Any]:
    archive_root = Path(manifest["archive_root"])
    stage_dir = archive_root / scene / stage
    eval_dir = archive_root / scene / "evaluation" / stage
    work_dir = archive_root / ".eval_work" / scene / stage
    eval_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "evaluation_failure.json").unlink(missing_ok=True)

    arch = archive_state(stage_dir)
    if not (arch["present"] and arch["ready"]):
        return {"scene": scene, "stage": stage, "status": "skipped", "reason": "archive not verified", "archive": arch}

    if args.skip_completed:
        prior_path = eval_dir / "evaluation_status.json"
        prior = read_json(prior_path) if prior_path.is_file() else None
        if prior and prior.get("status") == STATUS_COMPLETED:
            ck_id, ply_id, _ = fingerprint_assets(stage_dir, stage, prior=prior)
            fresh_fp = protocol_fingerprint(
                args, ck_id, ply_id, prior.get("iteration"), prior.get("num_gaussians")
            )
            cached_fp = prior.get("protocol_fingerprint") or {}
            if cached_fp.get("digest") == fresh_fp.get("digest"):
                return {
                    "scene": scene,
                    "stage": stage,
                    "status": "already_completed",
                    "eval_dir": str(eval_dir),
                    "protocol_fingerprint": fresh_fp["digest"],
                }
            print(
                f"[eval] {scene}/{stage}: cached result is stale for this request/assets "
                f"(cached={cached_fp.get('digest')}, current={fresh_fp.get('digest')}); re-evaluating"
            )

    metrics = MetricStack(device)
    prerequisites = check_prerequisites(metrics, manifest, need_official=args.official)
    timings: dict[str, float] = {}
    started = time.perf_counter()
    repro_placeholder: dict[str, Any] = {"applicable": False, "reason": "not yet evaluated"}
    early_state: dict[str, Any] = {}

    def announce_model_load(first_view: dict[str, Any]) -> None:
        """Publish the cleanup gate as soon as the model renders correctly.

        ArchiveCurator's threshold is archive_status=verified +
        model_load_verified=true; the final `completed` marker is written only
        after every planned metric and missing-GT note is in place.
        """
        early_state["first_view"] = {
            k: first_view.get(k) for k in ("image_name", "psnr", "ssim", "lpips", "l1", "render_min", "render_max")
        }
        running = build_status_payload(
            scene, stage, load_report, input_checks, None, repro_placeholder, [], "not_started",
            None, None, prerequisites, timings, "running", eval_dir, released_for_cleanup=True,
        )
        running["first_rendered_view"] = early_state["first_view"]
        running["note"] = (
            "intermediate marker: model assets verified loadable on the local GPU and the first held-out "
            "view rendered; remote cleanup of the archived stage is released. Metrics are still running."
        )
        write_json(eval_dir / "evaluation_status.json", _json_ready(running))
        print(
            f"[eval] {scene}/{stage}: running marker published (model_load_verified=true, "
            f"first view psnr={early_state['first_view'].get('psnr')})",
            flush=True,
        )

    started_load = time.perf_counter()
    model, load_report = load_archive_model(manifest, scene, stage_dir, device, do_hash=not args.no_hash)
    timings["load_model"] = time.perf_counter() - started_load

    cfg = parse_cfg_args(stage_dir / "cfg_args")
    input_checks = check_input_data(manifest, scene, stage_dir, cfg, eval_dir)

    started = time.perf_counter()
    heldout = heldout_evaluate(
        model, eval_dir, metrics, device, args.mask_mode,
        save_renders=not args.no_renders, on_first_view=announce_model_load,
    )
    timings["heldout"] = time.perf_counter() - started
    repro = reproduction_check(scene, stage, heldout["aggregate"], args.psnr_tol, args.l1_tol)

    write_json(eval_dir / "heldout_metrics.json", _json_ready(heldout))
    write_csv(eval_dir / "heldout_metrics.csv", heldout["per_view"])

    novel_meta: list[dict[str, Any]] = []
    novel_frames_by_path: dict[str, list[np.ndarray]] = {}
    official_path: str | None = None
    if not args.no_novel:
        started = time.perf_counter()
        official_path = pick_official_path(manifest, scene) if args.official else None
        for path_json in camera_path_files(manifest, scene):
            if args.max_paths and len(novel_meta) >= args.max_paths:
                break
            keep = args.official and path_json.stem == official_path
            meta, frames = render_camera_path(
                model, path_json, eval_dir, device, args.novel_frames,
                save_video=not args.no_video, keep_frames=keep,
            )
            novel_meta.append(meta)
            if frames is not None:
                novel_frames_by_path[meta["path"]] = frames
        timings["novel_views"] = time.perf_counter() - started

    # Official external-view protocol, scored on the uncompressed renders of the
    # official trajectory (never on the lossy viewing video).
    official: dict[str, Any] | None = None
    official_status = "not_requested"
    official_file_path: Path | None = None
    if args.official:
        if official_path is None or official_path not in novel_frames_by_path:
            official = {
                "status": "trajectory_unavailable",
                "reason": (
                    "no checked-in camera path matches the official external-view trajectory for this "
                    "scene, so the official protocol cannot be run"
                ),
            }
            official_status = "trajectory_unavailable"
        else:
            frames = novel_frames_by_path[official_path]
            started = time.perf_counter()
            official = official_external_view(
                manifest, scene, stage, frames, official_path, metrics, device,
                eval_dir / "official", work_dir, args.official_frames,
                allow_inferred_paired=bool(args.official_inferred_paired),
            )
            timings["official_external_view"] = time.perf_counter() - started
            official_file_path = eval_dir / "official" / "external_metrics.json"
            scopes = official.get("scope_summary") or {}
            parts = []
            if scopes.get("distributional_reported"):
                parts.append("distributional")
            if scopes.get("paired_vs_released_render_reported"):
                parts.append("paired_vs_released_render")
            if scopes.get("paired_vs_ground_truth_reported"):
                parts.append("paired_vs_ground_truth(INFERRED,unverified)")
            official_status = "+".join(parts) if parts else official.get("status", "unknown")
        novel_frames_by_path.pop(official_path, None)
    novel_frames_by_path.clear()

    fused = None
    if args.fused_ply:
        started = time.perf_counter()
        fused_path = eval_dir / "fused" / f"{scene}_{stage}_fused.ply"
        fused_path.parent.mkdir(parents=True, exist_ok=True)
        model.gaussians.save_fused_ply(str(fused_path), bool(model.dataset.appearance_enabled))
        fused = {"path": str(fused_path), "bytes": fused_path.stat().st_size}
        timings["fused_ply"] = time.perf_counter() - started

    del model
    free_cuda()

    protocol = {
        "schema": "skyfall_gs_dataset_evaluation_protocol_v1",
        "written_at": now_iso(),
        "driver": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "repo_root": str(REPO_ROOT),
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "argv": sys.argv,
            "manifest": manifest["_manifest_path"],
        },
        "scene": scene,
        "stage": stage,
        "manifest_contract": {
            "archive_dir": str(stage_dir),
            "evaluation_dir": str(eval_dir),
            "evaluation_marker": str(eval_dir / "evaluation_status.json"),
        },
        "heldout_policy": {
            "cameras": "every camera of transforms_test.json loaded by Scene for this source_path",
            "resolution": "native (resolution=1 -> 2048x2048 for JAX, native for NYC)",
            "render_call": "gaussian_renderer.render(camera, gaussians, pipe, bg, kernel_size, testing=True)",
            "appearance": (
                "testing=True, appearance_embedding=None -> checkpoint appearance row min(6, n_train-1); "
                "identical for every view, matching train.py::training_report"
            ),
            "mask_mode": args.mask_mode,
            "clamping": "render and GT clamped to [0,1] float32 RGB",
            "metric_definitions": metrics.metric_definitions(),
        },
        "official_external_view_protocol": {
            "enabled": bool(args.official),
            "gt_root": str(Path(manifest["local_eval_archive_root"]) / split_of(scene) / scene / "GT"),
            "frames_per_side": OFFICIAL_FRAMES,
            "paired_requires_verified_correspondence": True,
            "distributional": "CLIP-FID (cleanfid clip_vit_b_32) + CMMD (openai/clip-vit-large-patch14-336) on 512 patches",
            "note": "missing external GT for a scene is reported explicitly, never silently skipped",
        },
        "retention": {
            "kept": [
                "heldout per-view + aggregate metrics",
                "heldout render/gt/absdiff rasters",
                "stage1/stage2 novel-view videos on fixed checked-in trajectories",
                "comparison panels",
                "official external-view metrics",
                "fused viewable PLY when --fused-ply",
            ],
            "removed_after_use": ["official patchified rasters", "per-frame render buffers"],
            "model_assets_touched": "none: the archived checkpoint and matching PLY are read-only",
        },
    }
    write_json(eval_dir / "protocol.json", _json_ready(protocol))

    evaluation = {
        "schema": EVAL_SCHEMA,
        "scene": scene,
        "stage": stage,
        "evaluated_at": now_iso(),
        "archive": arch,
        "model_load": load_report,
        "model_load_verified": True,
        "input_data_compatibility": input_checks,
        "heldout": heldout,
        "reproduction_check": repro,
        "novel_views": novel_meta,
        "fused_ply": fused,
        "official_external_view": official,
        "official_external_view_status": official_status,
        "prerequisites": prerequisites,
        "timings_seconds": timings,
        "gpu_mem_mib": gpu_mem_mib(),
    }
    write_json(eval_dir / "evaluation.json", _json_ready(evaluation))

    blocked = [p for p in prerequisites if p.get("metric") in {"lpips"}]
    official_gt_exists = bool(ensure_eval_data(manifest, scene) and (ensure_eval_data(manifest, scene) / "GT").is_dir())
    outcome = classify_status(
        load_report, input_checks, heldout, official, official_path is not None,
        prerequisites, args, official_gt_exists,
    )
    cleared = bool(
        load_report
        and input_checks["compatible"]
        and not blocked
        and heldout["aggregate"].get("psnr") is not None
        and outcome["status"] != "failed"
    )
    status = build_status_payload(
        scene, stage, load_report, input_checks, heldout, repro, novel_meta, official_status,
        str(official_file_path) if official_file_path is not None and official_file_path.is_file() else None,
        fused, prerequisites, timings, outcome["status"], eval_dir, released_for_cleanup=cleared,
    )
    status["protocol_fingerprint"] = protocol_fingerprint(
        args,
        *fingerprint_assets(stage_dir, stage, loaded=load_report)[:2],
        load_report["iteration"],
        load_report["checkpoint_contents"]["num_gaussians"],
    )
    status["status_outcome"] = outcome
    status["first_rendered_view"] = early_state.get("first_view")
    status["missing_external_gt"] = not official_gt_exists
    if status["missing_external_gt"]:
        status["missing_external_gt_reason"] = (
            "the released Skyfall-GS evaluation data has no ground truth for this scene; only the "
            "held-out satellite metrics and our own fixed-trajectory renders are available. This is "
            "reported explicitly and is not a substitute for external GT comparison."
        )
    official_payload = official or {}
    status["official_paired_vs_released_render"] = official_payload.get("paired_vs_released_render")
    status["official_paired_vs_ground_truth"] = official_payload.get("paired_vs_ground_truth")
    status["official_distributional_metrics"] = official_payload.get("distributional")
    status["official_scope_summary"] = official_payload.get("scope_summary")
    status["camera_frame_contract_available"] = official_payload.get("camera_frame_contract_available")
    write_json(eval_dir / "evaluation_status.json", _json_ready(status))
    return status


def pick_official_path(manifest: dict[str, Any], scene: str) -> str | None:
    """The official trajectory is the checked-in path whose stem matches the released renders."""
    scene_dir = ensure_eval_data(manifest, scene)
    if scene_dir is None or not (scene_dir / "GT").is_dir():
        return None
    stems: set[str] = set()
    for d in scene_dir.iterdir():
        if not d.is_dir() or d.name == "GT":
            continue
        stems.update(v.stem for v in d.glob("*.mp4"))
    available = {p.stem for p in camera_path_files(manifest, scene)}
    matches = sorted(stems & available)
    return matches[0] if matches else None


# --------------------------------------------------------------------------- #
# roll-ups
# --------------------------------------------------------------------------- #
def write_rollups(manifest: dict[str, Any], args: argparse.Namespace) -> None:
    archive_root = Path(manifest["archive_root"])
    rows: list[dict[str, Any]] = []
    scene_summaries: dict[str, Any] = {}
    for entry in manifest["scenes"]:
        scene = entry["scene"]
        scene_eval = archive_root / scene / "evaluation"
        per_stage: dict[str, Any] = {}
        for stage in args.stages:
            state = evaluation_state(scene_eval / stage)
            if not state["present"]:
                per_stage[stage] = {"status": "pending"}
                continue
            payload = state["payload"]
            if payload.get("status") != STATUS_COMPLETED:
                per_stage[stage] = {
                    "status": payload.get("status"),
                    "model_load_verified": payload.get("model_load_verified"),
                    "remote_deletion_cleared": payload.get("remote_deletion_cleared"),
                    "status_outcome": payload.get("status_outcome"),
                    "heldout": payload.get("heldout_aggregate"),
                }
                continue
            per_stage[stage] = {
                "status": payload["status"],
                "model_load_verified": payload.get("model_load_verified"),
                "remote_deletion_cleared": payload.get("remote_deletion_cleared"),
                "heldout": payload.get("heldout_aggregate"),
                "reproduction_passed": (payload.get("reproduction_check") or {}).get("passed"),
                "official_external_view_status": payload.get("official_external_view_status"),
                "official_scope_summary": payload.get("official_scope_summary"),
                "missing_external_gt": payload.get("missing_external_gt"),
            }
            agg = payload.get("heldout_aggregate") or {}
            scopes = payload.get("official_scope_summary") or {}
            distrib = payload.get("official_distributional_metrics") or {}
            rows.append(
                {
                    "scene": scene,
                    "stage": stage,
                    "num_views": agg.get("num_views"),
                    "psnr": agg.get("psnr"),
                    "ssim": agg.get("ssim"),
                    "lpips": agg.get("lpips"),
                    "l1": agg.get("l1"),
                    "reproduction_passed": (payload.get("reproduction_check") or {}).get("passed"),
                    "official_distributional_reported": scopes.get("distributional_reported"),
                    "clip_fid": distrib.get("clip_fid"),
                    "cmmd": distrib.get("cmmd"),
                    "paired_vs_released_render_reported": scopes.get("paired_vs_released_render_reported"),
                    "paired_vs_ground_truth_reported": scopes.get("paired_vs_ground_truth_reported"),
                    "paired_vs_ground_truth_camera_verified": scopes.get(
                        "paired_vs_ground_truth_camera_verified"
                    ),
                    "missing_external_gt": payload.get("missing_external_gt"),
                }
            )
        panelled = build_panels(archive_root, scene, args.stages) if not args.no_panels else []
        summary = {
            "scene": scene,
            "written_at": now_iso(),
            "stages": per_stage,
            "panels": panelled,
            "official_gt_available": (ensure_eval_data(manifest, scene) / "GT").is_dir()
            if ensure_eval_data(manifest, scene)
            else False,
        }
        scene_summaries[scene] = summary
        write_json(scene_eval / "scene_evaluation_status.json", _json_ready(summary))
    write_csv(archive_root / "evaluation_summary.csv", rows)
    write_json(
        archive_root / "evaluation_summary.json",
        _json_ready(
            {
                "schema": "skyfall_gs_evaluation_summary_v1",
                "written_at": now_iso(),
                "evaluator": "DatasetEvaluator",
                "manifest": manifest["_manifest_path"],
                "scenes": scene_summaries,
                "metrics_rows": rows,
            }
        ),
    )


# --------------------------------------------------------------------------- #
# queue
# --------------------------------------------------------------------------- #
def queue_lock(archive_root: Path):
    """Exclusive queue lock; the owner's PID/state record survives refused contenders.

    Opened without truncation so a losing instance cannot erase the running owner's
    PID record, then rewritten only after the lock is actually held.
    """
    lock_path = archive_root / ".eval_queue.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    fh.seek(0)
    fh.truncate()
    fh.write(f"{os.getpid()} {now_iso()}\n")
    fh.flush()
    os.fsync(fh.fileno())
    return fh


def queue_state_path(archive_root: Path) -> Path:
    return archive_root / ".eval_queue_state.json"


def write_queue_state(archive_root: Path, phase: str, scene: str | None = None, stage: str | None = None) -> None:
    """Heartbeat for supervised handover: shows whether the queue is idle or mid-stage."""
    write_json(
        queue_state_path(archive_root),
        {
            "pid": os.getpid(),
            "phase": phase,
            "scene": scene,
            "stage": stage,
            "updated_at": now_iso(),
            "lock_file": str(archive_root / ".eval_queue.lock"),
        },
    )


def pending_work(manifest: dict[str, Any], args: argparse.Namespace) -> list[tuple[str, str]]:
    """Scene/stage pairs whose archive is verified and whose result is not current.

    A cached result only counts when it is `completed` *and* was produced for this exact
    request and these exact archived assets (protocol fingerprint), so an old
    satellite-only or earlier-protocol result is re-evaluated rather than skipped.
    """
    archive_root = Path(manifest["archive_root"])
    work: list[tuple[str, str]] = []
    for entry in manifest["scenes"]:
        scene = entry["scene"]
        for stage in args.stages:
            stage_dir = archive_root / scene / stage
            arch = archive_state(stage_dir)
            if not (arch["present"] and arch["ready"]):
                continue
            eval_dir = archive_root / scene / "evaluation" / stage
            prior_path = eval_dir / "evaluation_status.json"
            if args.skip_completed and prior_path.is_file():
                prior = read_json(prior_path)
                if prior.get("status") == STATUS_COMPLETED:
                    ck_id, ply_id, _ = fingerprint_assets(stage_dir, stage, prior=prior)
                    fresh = protocol_fingerprint(
                        args, ck_id, ply_id, prior.get("iteration"), prior.get("num_gaussians")
                    )
                    if (prior.get("protocol_fingerprint") or {}).get("digest") == fresh["digest"]:
                        continue
            work.append((scene, stage))
    return work


def terminal_states(manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, str]:
    """Map each scene/stage to its evaluated state, treating partial/failed as unfinished work."""
    archive_root = Path(manifest["archive_root"])
    out: dict[str, str] = {}
    for entry in manifest["scenes"]:
        scene = entry["scene"]
        for stage in args.stages:
            state = evaluation_state(archive_root / scene / "evaluation" / stage)
            if not state["present"]:
                out[f"{scene}/{stage}"] = "pending"
            else:
                out[f"{scene}/{stage}"] = str(state["payload"].get("status") or "unknown")
    return out


def run_queue(args: argparse.Namespace, manifest: dict[str, Any], device: torch.device) -> int:
    lock = queue_lock(Path(manifest["archive_root"]))
    if lock is None:
        print("[queue] another evaluator queue holds the lock; exiting", flush=True)
        return 0
    print(f"[queue] started pid={os.getpid()} archive_root={manifest['archive_root']}", flush=True)
    idle_cycles = 0
    while True:
        work = pending_work(manifest, args)
        if work:
            idle_cycles = 0
            for scene, stage in work:
                print(f"[queue] evaluating {scene}/{stage}", flush=True)
                write_queue_state(Path(manifest["archive_root"]), "evaluating", scene, stage)
                try:
                    result = evaluate_stage(args, manifest, scene, stage, device)
                    print(f"[queue] {scene}/{stage} -> {result.get('status')}", flush=True)
                except Exception as exc:  # keep the queue alive; failure stays visible
                    import traceback

                    print(f"[queue] {scene}/{stage} FAILED: {type(exc).__name__}: {exc}", flush=True)
                    traceback.print_exc()
                    write_json(
                        Path(manifest["archive_root"]) / scene / "evaluation" / stage / "evaluation_failure.json",
                        {"scene": scene, "stage": stage, "error": f"{type(exc).__name__}: {exc}", "at": now_iso()},
                    )
                write_queue_state(Path(manifest["archive_root"]), "idle")
                write_rollups(manifest, args)
        else:
            idle_cycles += 1
            write_queue_state(Path(manifest["archive_root"]), "idle")
            write_rollups(manifest, args)
            states = terminal_states(manifest, args)
            if args.exit_when_complete and all(
                value in (STATUS_COMPLETED, "already_completed") for value in states.values()
            ):
                print("[queue] all scene/stage evaluations completed; exiting", flush=True)
                return 0
            unfinished = {k: v for k, v in states.items() if v not in (STATUS_COMPLETED, "already_completed")}
            print(f"[queue] no verified pending archive (idle cycle {idle_cycles}); states={unfinished}", flush=True)
        if args.queue_once:
            return 0
        time.sleep(args.poll_interval)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--scene", action="append", default=[], help="scene name (repeatable)")
    parser.add_argument("--all-scenes", action="store_true")
    parser.add_argument("--stages", nargs="+", default=["stage1", "stage2"], choices=["stage1", "stage2"])
    parser.add_argument("--queue", action="store_true", help="long-running consumer of newly verified archives")
    parser.add_argument("--queue-once", action="store_true", help="one queue pass then exit")
    parser.add_argument("--exit-when-complete", action="store_true")
    parser.add_argument("--poll-interval", type=int, default=120)
    parser.add_argument("--skip-completed", action="store_true", default=True)
    parser.add_argument("--no-skip-completed", dest="skip_completed", action="store_false")
    parser.add_argument("--official", action="store_true", help="run the official external-view protocol")
    parser.add_argument("--official-frames", type=int, default=OFFICIAL_FRAMES)
    parser.add_argument(
        "--official-inferred-paired",
        action="store_true",
        help=(
            "also record paired PSNR/SSIM/LPIPS against the released GT video, explicitly labelled "
            "appearance-inferred and camera-unverified (the released data publishes no GT-to-camera "
            "index contract)"
        ),
    )
    parser.add_argument("--fused-ply", action="store_true", help="export a viewable fused PLY per stage")
    parser.add_argument("--no-renders", action="store_true", help="skip held-out render PNGs")
    parser.add_argument("--no-novel", action="store_true", help="skip fixed camera-path renders")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-panels", action="store_true")
    parser.add_argument("--novel-frames", type=int, default=6)
    parser.add_argument("--max-paths", type=int, default=0)
    parser.add_argument("--mask-mode", choices=["none", "zero-fill"], default="none")
    parser.add_argument("--no-hash", action="store_true", help="skip checkpoint/PLY sha256 (repeat runs)")
    parser.add_argument("--psnr-tol", type=float, default=DEFAULT_PSNR_TOL)
    parser.add_argument("--l1-tol", type=float, default=DEFAULT_L1_TOL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    safe_state(args.quiet)
    manifest = load_manifest(args.manifest)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if args.queue or args.queue_once:
        return run_queue(args, manifest, device)

    scenes = args.scene or [e["scene"] for e in manifest["scenes"]]
    if args.all_scenes and not args.scene:
        scenes = [e["scene"] for e in manifest["scenes"]]
    results = []
    for scene in scenes:
        for stage in args.stages:
            try:
                result = evaluate_stage(args, manifest, scene, stage, device)
            except Exception as exc:
                import traceback

                traceback.print_exc()
                result = {"scene": scene, "stage": stage, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                out = Path(manifest["archive_root"]) / scene / "evaluation" / stage
                out.mkdir(parents=True, exist_ok=True)
                write_json(out / "evaluation_failure.json", {**result, "at": now_iso()})
            results.append(result)
            agg = (result.get("heldout_aggregate") or {}) if isinstance(result, dict) else {}
            print(f"[eval] {scene}/{stage}: {result.get('status')} psnr={agg.get('psnr')} l1={agg.get('l1')}", flush=True)
    write_rollups(manifest, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
