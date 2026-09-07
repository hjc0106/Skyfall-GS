"""Utilities for progressive zoom-refine MVP."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

from scene.gaussian_model import GaussianModel
from utils.loss_utils import l1_loss


def freeze_geometry_for_zoom(gaussians: GaussianModel) -> None:
    """Freeze geometry/opacity/appearance; keep SH features trainable."""
    for name in ("xyz", "scaling", "rotation", "opacity"):
        for group in gaussians.optimizer.param_groups:
            if group["name"] == name:
                group["params"][0].requires_grad_(False)

    if gaussians.appearance_enabled:
        gaussians.appearance_embeddings.requires_grad_(False)
        if gaussians._embeddings is not None:
            gaussians._embeddings.requires_grad_(False)
        if gaussians.appearance_mlp is not None:
            for param in gaussians.appearance_mlp.parameters():
                param.requires_grad_(False)


@dataclass
class GeometrySnapshot:
    xyz: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    opacity: torch.Tensor

    @classmethod
    def from_gaussians(cls, gaussians: GaussianModel) -> "GeometrySnapshot":
        return cls(
            xyz=gaussians._xyz.detach().clone(),
            scaling=gaussians._scaling.detach().clone(),
            rotation=gaussians._rotation.detach().clone(),
            opacity=gaussians._opacity.detach().clone(),
        )

    def assert_unchanged(self, gaussians: GaussianModel, rtol: float = 0.0, atol: float = 0.0) -> None:
        checks = {
            "_xyz": self.xyz,
            "_scaling": self.scaling,
            "_rotation": self.rotation,
            "_opacity": self.opacity,
        }
        for name, expected in checks.items():
            current = getattr(gaussians, name).detach()
            if not torch.equal(current, expected):
                max_diff = (current - expected).abs().max().item()
                raise AssertionError(
                    f"Geometry parameter {name} changed during appearance-only training (max diff={max_diff})."
                )


def select_appearance_embedding(
    gaussians: GaussianModel,
    base_cam_uid: int,
    is_train_view: bool,
) -> torch.Tensor:
    if not gaussians.appearance_enabled:
        return None

    if is_train_view:
        if base_cam_uid >= gaussians.appearance_embeddings.shape[0]:
            raise IndexError(
                f"Train view uid {base_cam_uid} exceeds appearance embedding count "
                f"{gaussians.appearance_embeddings.shape[0]}."
            )
        embedding = gaussians.appearance_embeddings[base_cam_uid]
    else:
        embedding = torch.mean(gaussians.appearance_embeddings, dim=0)
    return embedding.detach()


def embedding_for_train_camera(gaussians: GaussianModel, uid: int) -> Optional[torch.Tensor]:
    """A train camera's own frozen embedding.

    Supervising an original train view with a *different* view's embedding would ask the
    SH features to absorb an illumination mismatch, so each original view keeps its own.
    """
    if not gaussians.appearance_enabled:
        return None
    return gaussians.appearance_embeddings[uid].detach()


def save_tensor_image(image: torch.Tensor, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    arr = (image.detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(out_path)


def image_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(l1_loss(a, b).item())


def compute_post_train_metrics(
    render_before: torch.Tensor,
    render_after: torch.Tensor,
    refined: torch.Tensor,
) -> Dict[str, float]:
    dist_before = image_l1(render_before, refined)
    dist_after = image_l1(render_after, refined)
    return {
        "l1_before_to_refined": dist_before,
        "l1_after_to_refined": dist_after,
        "l1_moved_toward_refined": bool(dist_after < dist_before),
    }


def assert_post_train_checks(
    snapshot: GeometrySnapshot,
    gaussians: GaussianModel,
    metrics: Dict[str, float],
) -> None:
    """Run after artifacts are on disk, so a failure still leaves images to inspect."""
    snapshot.assert_unchanged(gaussians)
    if not metrics["l1_moved_toward_refined"]:
        raise AssertionError(
            f"render_after is not closer to refined than render_before "
            f"(before={metrics['l1_before_to_refined']:.6f}, "
            f"after={metrics['l1_after_to_refined']:.6f})."
        )


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
