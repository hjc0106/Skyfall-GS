"""Multi-view refinement scaffold with geometry-aligned RGB propagation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base import Refiner
from .types import MultiViewInput, RefinementRequest, RefinementResult


class FeaturePropagationAdapter(Protocol):
    """Optional adapter for a backbone-specific feature propagation module."""

    def __call__(
        self,
        target_image: torch.Tensor,
        neighbors: Sequence[MultiViewInput],
        request: RefinementRequest,
    ) -> torch.Tensor | Image.Image:
        ...


def _to_chw(image: Image.Image | torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)
    tensor = image if isinstance(image, torch.Tensor) else torch.as_tensor(image)
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"Image batch must have size 1, got {tuple(tensor.shape)}")
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Image must be CHW or HWC, got {tuple(tensor.shape)}")
    if tensor.shape[0] in (1, 3, 4):
        tensor = tensor
    elif tensor.shape[-1] in (1, 3, 4):
        tensor = tensor.permute(2, 0, 1)
    else:
        raise ValueError(f"Could not infer image channel dimension from {tuple(tensor.shape)}")
    tensor = tensor.float()
    if tensor.numel() and float(tensor.detach().amax().item()) > 1.0:
        tensor = tensor / 255.0
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    return tensor[:3]


def _to_mask(mask: Any, height: int, width: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.zeros((height, width), device=device, dtype=torch.float32)
    if isinstance(mask, Image.Image):
        array = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array)
    else:
        tensor = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask)
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3:
            tensor = tensor[0] if tensor.shape[0] == 1 else tensor[..., 0]
        if tensor.ndim != 2:
            raise ValueError(f"valid_mask must be a 2D plane, got {tuple(tensor.shape)}")
        tensor = tensor.float()
        if tensor.numel() and float(tensor.detach().amax().item()) > 1.0:
            tensor = tensor / 255.0
    tensor = tensor.to(device=device, dtype=torch.float32)
    if tensor.shape != (height, width):
        tensor = F.interpolate(tensor[None, None], size=(height, width), mode="nearest")[0, 0]
    return tensor.clamp(0.0, 1.0)


def _to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        image.detach().cpu().float().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0
    ).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


class MultiViewSRBackend(Refiner):
    """Fuse depth/occlusion-filtered neighbor RGB before single-view SR.

    This is a conservative first implementation of the multi-view boundary:
    it only consumes ``MultiViewInput.warped_image`` and its validity mask.  A
    feature-level adapter can be supplied later once a chosen SR backbone's
    feature scale and text-conditioning interface have been verified.
    """

    name = "multiview_sr"

    def __init__(
        self,
        single_view_backend: Refiner,
        *,
        feature_propagation: FeaturePropagationAdapter | None = None,
        min_coverage: float = 1e-4,
    ):
        self.single_view_backend = single_view_backend
        self.feature_propagation = feature_propagation
        self.min_coverage = float(min_coverage)

    def release_memory(self) -> None:
        """Release the wrapped SR model before an isolated VLM worker starts."""

        self.single_view_backend.release_memory()

    def _neighbors(self, request: RefinementRequest) -> list[MultiViewInput]:
        values = request.metadata.get("neighbor_views", ())
        result: list[MultiViewInput] = []
        for value in values:
            if isinstance(value, MultiViewInput):
                result.append(value)
            elif isinstance(value, dict):
                result.append(MultiViewInput(**value))
            else:
                raise TypeError(f"neighbor_views entries must be MultiViewInput/dict, got {type(value)!r}")
        return result

    def _fuse_rgb(
        self,
        target: torch.Tensor,
        neighbors: Sequence[MultiViewInput],
    ) -> tuple[torch.Tensor, float, int]:
        _, height, width = target.shape
        accum = target.clone()
        weights = torch.ones((height, width), device=target.device, dtype=target.dtype)
        used = 0
        for neighbor in neighbors:
            # Raw neighbor coordinates are intentionally not accepted here:
            # geometry_warp must produce warped_image and valid_mask first.
            if neighbor.warped_image is None or neighbor.valid_mask is None:
                continue
            warped = _to_chw(neighbor.warped_image).to(device=target.device, dtype=target.dtype)
            if warped.shape[-2:] != (height, width):
                warped = F.interpolate(warped[None], size=(height, width), mode="bilinear", align_corners=True)[0]
            mask = _to_mask(neighbor.valid_mask, height, width, target.device).to(dtype=target.dtype)
            weight = max(0.0, float(neighbor.weight)) * mask
            accum = accum + warped * weight[None]
            weights = weights + weight
            used += int(mask.gt(0).any().item())
        fused = accum / weights.clamp_min(1e-6)[None]
        coverage = float((weights > 1.0).float().mean().item())
        return fused.clamp(0.0, 1.0), coverage, used

    def refine(self, request: RefinementRequest) -> RefinementResult:
        neighbors = self._neighbors(request)
        target = _to_chw(request.image).float()
        fused, coverage, used = self._fuse_rgb(target, neighbors)

        if coverage < self.min_coverage:
            result = self.single_view_backend.refine(request)
            result.backend = self.name
            result.metadata.update(
                {
                    "fallback": "single_view_low_coverage",
                    "neighbor_count": len(neighbors),
                    "aligned_neighbor_count": used,
                    "coverage": coverage,
                    "feature_propagation": False,
                }
            )
            return result

        feature_used = False
        model_input: Image.Image = _to_pil(fused)
        if self.feature_propagation is not None:
            propagated = self.feature_propagation(fused, neighbors, request)
            if isinstance(propagated, Image.Image):
                model_input = propagated.convert("RGB")
            elif isinstance(propagated, torch.Tensor):
                model_input = _to_pil(_to_chw(propagated))
            else:
                raise TypeError(
                    "feature_propagation must return a PIL image or tensor; "
                    f"got {type(propagated)!r}"
                )
            feature_used = True

        inner_request = replace(
            request,
            image=model_input,
            model_config={
                **dict(request.model_config),
                "multiview_rgb_fusion": True,
                "feature_propagation": feature_used,
            },
        )
        result = self.single_view_backend.refine(inner_request)
        result.backend = self.name
        result.metadata.update(
            {
                "fallback": None,
                "neighbor_count": len(neighbors),
                "aligned_neighbor_count": used,
                "coverage": coverage,
                "feature_propagation": feature_used,
            }
        )
        return result


__all__ = ["FeaturePropagationAdapter", "MultiViewSRBackend"]
