"""Checkpoint extras for LoD absorption: mix RNG, densify spec, and sidecar logs.

Segment restart can continue a saved training stage. It is not a claim that every
historical run recorded a complete source lineage.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


TRAIN_STATE_NAME = "train_state.pt"
DENSIFY_KEYS = (
    "densify_from",
    "densify_until",
    "densify_interval",
    "densify_grad_threshold",
    "max_points",
    "mix_ratio",
    "seed",
)


def densify_spec(
    *,
    densify_from: int,
    densify_until: int,
    densify_interval: int,
    densify_grad_threshold: float,
    max_points: int,
    mix_ratio: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "densify_from": int(densify_from),
        "densify_until": int(densify_until),
        "densify_interval": int(densify_interval),
        "densify_grad_threshold": float(densify_grad_threshold),
        "max_points": int(max_points),
        "mix_ratio": float(mix_ratio),
        "seed": int(seed),
    }


def spec_mismatches(saved: Mapping[str, Any] | None, current: Mapping[str, Any]) -> list[str]:
    if not saved:
        return []
    bad = []
    for key in DENSIFY_KEYS:
        if key not in saved:
            continue
        if saved[key] != current[key]:
            bad.append(key)
    return bad


def mix_rng(seed: int) -> random.Random:
    return random.Random(int(seed))


def advance_mix_rng(rng: random.Random, n_steps: int, *, mix_ratio: float, n_train: int) -> None:
    """Replay camera-mix draws so resume continues the same sequence."""

    dummy = list(range(max(int(n_train), 1)))
    for _ in range(int(n_steps)):
        if rng.random() < float(mix_ratio):
            rng.choice(dummy)


def capture_train_state(
    *,
    step: int,
    mix: random.Random,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "step": int(step),
        "mix_rng": mix.getstate(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state().cpu(),
        "spec": dict(spec),
    }
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state().cpu()
    return payload


def restore_train_state(payload: Mapping[str, Any], mix: random.Random) -> None:
    mix.setstate(payload["mix_rng"])
    if payload.get("python_rng") is not None:
        random.setstate(payload["python_rng"])
    if payload.get("numpy_rng") is not None:
        np.random.set_state(payload["numpy_rng"])
    if payload.get("torch_rng") is not None:
        torch.set_rng_state(payload["torch_rng"].cpu())
    if payload.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(payload["cuda_rng"].cpu())


def save_train_state(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), dest)


def load_train_state(path: str | os.PathLike[str] | None) -> dict[str, Any] | None:
    if not path:
        return None
    dest = Path(path)
    if not dest.is_file():
        return None
    return torch.load(dest, map_location="cpu", weights_only=False)


def sidecar_dir(output_dir: str | os.PathLike[str], resume_path: str | os.PathLike[str] | None) -> Path:
    """Prefer artifacts already in the output dir; otherwise use the checkpoint folder."""

    out = Path(output_dir)
    if (out / TRAIN_STATE_NAME).is_file() or (out / "densify_log.json").is_file():
        return out
    if resume_path:
        parent = Path(resume_path).resolve().parent
        if parent.is_dir():
            return parent
    return out


def load_json_list(path: Path, key: str | None = None) -> list:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if key is None:
        return list(payload) if isinstance(payload, list) else []
    return list(payload.get(key) or [])


def keep_upto(rows: list[Mapping[str, Any]], step: int) -> list[dict[str, Any]]:
    kept = []
    for row in rows:
        if int(row.get("step", -1)) <= int(step):
            kept.append(dict(row))
    return kept


def tensor_max_abs(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if tuple(left.shape) != tuple(right.shape):
        return None
    return float((left.detach().float() - right.detach().float()).abs().max().item())


def layer_param_delta(left, right) -> dict[str, Any]:
    report = {"n_left": int(left.xyz.shape[0]), "n_right": int(right.xyz.shape[0])}
    for name in ("xyz", "sh", "log_scales", "rotations", "opacity_logits", "filter_3d"):
        a = getattr(left, name)
        b = getattr(right, name)
        report[name] = {
            "shape_left": list(a.shape),
            "shape_right": list(b.shape),
            "max_abs": tensor_max_abs(a, b),
        }
    return report
