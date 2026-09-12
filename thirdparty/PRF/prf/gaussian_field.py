from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from prf.types import GaussianScene


def sample_context_indices(
    scene: GaussianScene,
    count: int,
    *,
    min_opacity: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Opacity-weighted, voxel-thinned Gaussian indices for the 3D context layer."""
    rng = rng or np.random.default_rng(20260820)
    count = max(int(count), 0)
    keep = scene.opacity >= min_opacity
    if keep.sum() == 0 or count == 0:
        return np.zeros((0,), dtype=np.int64)
    source = np.nonzero(keep)[0]
    mu = scene.mu[source]
    opacity = scene.opacity[source]
    order = np.argsort(-opacity, kind="mergesort")
    spans = np.sort(np.maximum(np.ptp(mu, axis=0), 1e-6))
    surface_extent = float(np.sqrt(spans[-1] * spans[-2]))
    voxel = max(surface_extent / max(np.sqrt(count) * 1.5, 1.0), 0.25)
    candidates = np.zeros((0,), dtype=np.int64)
    for _ in range(6):
        keys = np.floor(mu / voxel).astype(np.int64)
        _, first = np.unique(keys[order], axis=0, return_index=True)
        candidates = source[order[first]]
        if candidates.size >= count or voxel <= 0.25:
            break
        voxel = max(voxel * 0.65, 0.25)
    if candidates.size <= count:
        return candidates.astype(np.int64)
    weights = np.asarray(scene.opacity[candidates], dtype=np.float64)
    weights = np.maximum(weights, 1e-6)
    weights /= weights.sum()
    chosen = rng.choice(candidates.size, size=count, replace=False, p=weights)
    return candidates[chosen].astype(np.int64)


def map_patch_values_to_points(
    points: np.ndarray,
    point_normals: np.ndarray | None,
    patch_centers: np.ndarray,
    patch_normals: np.ndarray,
    patch_t1: np.ndarray,
    patch_t2: np.ndarray,
    patch_radius: np.ndarray,
    patch_values: np.ndarray,
    *,
    k: int = 8,
    normal_angle_deg: float = 60.0,
    chunk_size: int = 100000,
) -> np.ndarray:
    """Surface-aware inverse-distance blend of patch values onto 3D points."""
    mapped, _ = map_patch_fields_to_points(
        points,
        point_normals,
        patch_centers,
        patch_normals,
        patch_t1,
        patch_t2,
        patch_radius,
        np.asarray(patch_values, dtype=np.float64).reshape(-1, 1),
        k=k,
        normal_angle_deg=normal_angle_deg,
        chunk_size=chunk_size,
    )
    return mapped[:, 0]


def map_patch_fields_to_points(
    points: np.ndarray,
    point_normals: np.ndarray | None,
    patch_centers: np.ndarray,
    patch_normals: np.ndarray,
    patch_t1: np.ndarray,
    patch_t2: np.ndarray,
    patch_radius: np.ndarray,
    patch_values: np.ndarray,
    *,
    k: int = 8,
    normal_angle_deg: float = 60.0,
    chunk_size: int = 100000,
) -> tuple[np.ndarray, np.ndarray]:
    """Map several patch fields to points while sharing one chunked KD query.

    Returns ``(mapped, nearest_patch_index)``. Non-finite patch values do not
    contribute to that field. Chunking keeps full multi-million Gaussian scenes
    within a bounded memory footprint.
    """
    n_points = int(points.shape[0])
    n_patches = int(patch_centers.shape[0])
    values_all = np.asarray(patch_values, dtype=np.float64)
    if values_all.ndim == 1:
        values_all = values_all[:, None]
    if values_all.shape[0] != n_patches:
        raise ValueError("patch_values first dimension must match patch_centers")
    out = np.full((n_points, values_all.shape[1]), np.nan, dtype=np.float32)
    nearest = np.full(n_points, -1, dtype=np.int32)
    if n_points == 0 or n_patches == 0:
        return out, nearest
    k = max(min(int(k), n_patches), 1)
    tree = cKDTree(patch_centers)
    normal_cosine_min = float(np.cos(np.deg2rad(normal_angle_deg)))
    chunk_size = max(int(chunk_size), 1)
    for start in range(0, n_points, chunk_size):
        stop = min(start + chunk_size, n_points)
        chunk = np.asarray(points[start:stop], dtype=np.float64)
        _, index = tree.query(chunk, k=k)
        index = np.asarray(index, dtype=np.int64)
        if index.ndim == 1:
            index = index[:, None]
        nearest[start:stop] = index[:, 0].astype(np.int32)

        delta = chunk[:, None, :] - patch_centers[index]
        d_normal = np.einsum("nki,nki->nk", delta, patch_normals[index])
        d_u = np.einsum("nki,nki->nk", delta, patch_t1[index])
        d_v = np.einsum("nki,nki->nk", delta, patch_t2[index])
        radius = np.maximum(patch_radius[index], 1e-3)
        sigma_n = np.maximum(0.35 * radius, 0.25)
        weight = np.exp(-(d_u * d_u + d_v * d_v) / (2.0 * radius * radius))
        weight *= np.exp(-(d_normal * d_normal) / (2.0 * sigma_n * sigma_n))
        if point_normals is not None:
            cosine = np.abs(
                np.einsum("ni,nki->nk", point_normals[start:stop], patch_normals[index])
            )
            weight *= np.where(cosine >= normal_cosine_min, cosine, 0.0)

        values = values_all[index]
        finite = np.isfinite(values)
        weighted = weight[:, :, None] * finite
        total = weighted.sum(axis=1)
        blended = np.sum(weighted * np.where(finite, values, 0.0), axis=1)
        valid = total > 1e-12
        chunk_out = out[start:stop]
        chunk_out[valid] = (blended[valid] / total[valid]).astype(np.float32)
    return out, nearest
