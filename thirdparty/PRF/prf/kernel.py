from __future__ import annotations

import math

import numpy as np

from prf.config import PRFConfig
from prf.io_gs import covariance_world, weighted_quantile
from prf.types import GaussianScene, SurfacePatch


def compute_kernel_resolution(patch: SurfacePatch, scene: GaussianScene, cfg: PRFConfig) -> float:
    if patch.ids.size == 0:
        return math.inf
    cov = covariance_world(scene.rotations[patch.ids], scene.scales[patch.ids])
    tangent = patch.tangent
    surf = np.matmul(np.matmul(tangent.T, cov), tangent)
    eig = np.linalg.eigvalsh(surf)
    sigma_max = np.sqrt(np.maximum(eig.max(axis=1), 0.0))
    values = cfg.bandwidth_coeff * sigma_max
    det = np.maximum(np.linalg.det(surf), cfg.eps)
    weights = np.square(scene.opacity[patch.ids]) * np.sqrt(det)
    return weighted_quantile(values, weights, cfg.kernel_quantile)
