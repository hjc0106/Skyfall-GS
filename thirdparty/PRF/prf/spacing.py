from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from prf.config import PRFConfig
from prf.types import GaussianScene, SurfacePatch


def compute_spacing_resolution(patch: SurfacePatch, scene: GaussianScene, cfg: PRFConfig) -> float:
    if patch.ids.size < 2:
        return math.inf
    coords = (scene.mu[patch.ids] - patch.center) @ patch.tangent
    k = min(int(cfg.spacing_knn) + 1, coords.shape[0])
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=k)
    if k == 1:
        return math.inf
    neighbor = np.atleast_2d(distances)[:, 1:]
    mean_spacing = neighbor.mean(axis=1)
    return float(np.quantile(mean_spacing, cfg.spacing_quantile))
