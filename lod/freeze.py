"""Fingerprints for frozen LoD layers and the attached Skyfall appearance."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import torch


def tensor_digest(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None or not torch.is_tensor(tensor):
        return None
    flat = tensor.detach().float().contiguous().reshape(-1).cpu()
    payload: dict[str, Any] = {
        "n": int(flat.numel()),
        "shape": list(tensor.shape),
        "sum": float(flat.sum().item()),
        "sumsq": float((flat * flat).sum().item()),
        "max_abs": float(flat.abs().max().item()) if int(flat.numel()) else 0.0,
    }
    payload["sha256"] = hashlib.sha256(flat.numpy().tobytes()).hexdigest()
    return payload


def _digests_equal(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if left is None and right is None:
        return True
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    if left.get("sha256") and right.get("sha256"):
        return left["sha256"] == right["sha256"]
    return left.get("n") == right.get("n") and left.get("sum") == right.get("sum") and left.get("sumsq") == right.get("sumsq")


def layer_digest(layer) -> dict[str, Any]:
    filter_3d = getattr(layer, "filter_3d", None)
    return {
        "n": int(layer.xyz.shape[0]),
        "frozen": bool(getattr(layer, "frozen", False)),
        "xyz": tensor_digest(layer.xyz),
        "sh": tensor_digest(layer.sh),
        "log_scales": tensor_digest(layer.log_scales),
        "rotations": tensor_digest(layer.rotations),
        "opacity_logits": tensor_digest(layer.opacity_logits),
        "filter_3d": tensor_digest(filter_3d),
    }


def appearance_freeze_report(appearance) -> dict[str, Any]:
    mlp_grad = False
    if appearance is not None and getattr(appearance, "mlp", None) is not None:
        mlp_grad = any(bool(param.requires_grad) for param in appearance.mlp.parameters())
    image_grad = False
    image = None if appearance is None else getattr(appearance, "image_embeddings", None)
    if torch.is_tensor(image):
        image_grad = bool(image.requires_grad)
    layer_grad = []
    embeddings = [] if appearance is None else list(getattr(appearance, "layer_embeddings", None) or [])
    for emb in embeddings:
        layer_grad.append(bool(torch.is_tensor(emb) and emb.requires_grad))
    gaussian = None if appearance is None else getattr(appearance, "gaussian_embeddings", None)
    gaussian_grad = bool(torch.is_tensor(gaussian) and gaussian.requires_grad)
    ok = not mlp_grad and not image_grad and not gaussian_grad and not any(layer_grad)
    return {
        "ok": ok,
        "mlp_requires_grad": mlp_grad,
        "image_embeddings_requires_grad": image_grad,
        "gaussian_embeddings_requires_grad": gaussian_grad,
        "layer_embeddings_requires_grad": layer_grad,
        "layer_embeddings": [tensor_digest(emb) if torch.is_tensor(emb) else None for emb in embeddings],
    }


def snapshot_levels(bundle, levels: Sequence[int]) -> dict[str, Any]:
    layers = {}
    for index in levels:
        layer = bundle.layer(index)
        if layer is None:
            layers[str(index)] = None
            continue
        layers[str(index)] = layer_digest(layer)
    return {
        "levels": layers,
        "appearance": appearance_freeze_report(getattr(bundle, "appearance", None)),
        "active_level": int(getattr(bundle.lod, "active_level", 0)),
    }


def completed_layers_frozen(bundle, *, up_to_level: int) -> dict[str, Any]:
    """Levels strictly below ``up_to_level`` must stay frozen."""

    report = {}
    ok = True
    for index in range(up_to_level):
        layer = bundle.layer(index)
        frozen = bool(layer is not None and getattr(layer, "frozen", False))
        report[str(index)] = frozen
        if not frozen:
            ok = False
    appearance = appearance_freeze_report(getattr(bundle, "appearance", None))
    return {"ok": ok and bool(appearance["ok"]), "layers": report, "appearance": appearance}


def snapshot_mismatches(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    before_levels = before.get("levels") or {}
    after_levels = after.get("levels") or {}
    for key, old in before_levels.items():
        new = after_levels.get(key)
        if old is None and new is None:
            continue
        if not isinstance(old, dict) or not isinstance(new, dict):
            mismatches.append(f"level_{key}")
            continue
        if bool(old.get("frozen")) and not bool(new.get("frozen")):
            mismatches.append(f"level_{key}_unfrozen")
        for field in ("xyz", "sh", "log_scales", "rotations", "opacity_logits", "filter_3d"):
            if not _digests_equal(old.get(field), new.get(field)):
                mismatches.append(f"level_{key}_{field}")
    before_app = (before.get("appearance") or {}).get("layer_embeddings") or []
    after_app = (after.get("appearance") or {}).get("layer_embeddings") or []
    frozen_indices = {int(key) for key in before_levels}
    for index, old in enumerate(before_app):
        if index not in frozen_indices:
            continue
        if index >= len(after_app):
            mismatches.append(f"appearance_layer_{index}_missing")
            continue
        if not _digests_equal(old, after_app[index]):
            mismatches.append(f"appearance_layer_{index}")
    if (after.get("appearance") or {}).get("ok") is False:
        mismatches.append("appearance_requires_grad")
    return mismatches


def assert_snapshot_unchanged(before: dict[str, Any], after: dict[str, Any]) -> None:
    bad = snapshot_mismatches(before, after)
    if bad:
        raise AssertionError("Frozen LoD state changed: " + ", ".join(bad))
